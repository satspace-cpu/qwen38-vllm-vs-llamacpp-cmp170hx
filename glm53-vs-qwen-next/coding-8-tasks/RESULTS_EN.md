# Eight coding tasks: GLM-5.3-Flash vs three Qwen configurations — correctness only
**14 October 2026 · [Русский](RESULTS_RU.md) · [Solution-free prompts](TASKS_ONLY_RU.md) · [Original Qwen benchmark](../../flash-next-awq-vs-fp8/README.md)**

**This report evaluates correctness only.** The GLM run shared the server with **two additional concurrent inference workloads**. Its elapsed agent times are **not comparable** with the prior Qwen runs and have been removed from this report, along with elapsed-time figures and CSV.

The previous Qwen study used **manual correctness scores from 0 to 10 per task**. GLM is now graded on the same **expert judgment scale: 76/80**. These scores measure solution correctness, **not execution speed or a uniform hidden-test pass percentage**. GLM marks are based on source review, repeated checks and executed counterexamples; some PostgreSQL and TypeScript verification remains less comprehensive, as noted below.

![Total correctness scores](figures/correctness_totals.svg)

![Per-task correctness scores](figures/correctness_by_task.svg)

| # | Task | Qwen Next FP8 | Qwen Next AWQ | Qwen 27B BF16 | **GLM-5.3** |
|---:|---|---:|---:|---:|---:|
| 1 | Python session grouping | 10 | 9.5 | 10 | **10** |
| 2 | AsyncCache | 10 | 10 | 10 | **10** |
| 3 | Covering range | 10 | 10 | 10 | **10** |
| 4 | PostgreSQL | 9.5 | 9.5 | 10 | **10** |
| 5 | TypeScript parsePort | 10 | 10 | 10 | **8** |
| 6 | Storage / email | 10 | 9.5 | 10 | **10** |
| 7 | LRU + TTL | 10 | 10 | 7.5 | **10** |
| 8 | Async run_jobs | 10 | 9.5 | 9.5 | **8** |
| **Total / 80** | **Expert correctness score** | **79.5** | **78** | **77** | **76** |

**Why GLM lost points:**

- **Task 5: 8/10.** GLM throws `RangeError` for `parsePort("8080abc", 3000)`, whereas the published reference returns fallback; the word “reject” in the abbreviated task is ambiguous on that particular behavior. More conclusively, `parsePort(8080, 0)` returns `8080` without checking an invalid fallback, while the original wording explicitly requires `RangeError` for invalid fallback. Both behaviors were reproduced.
- **Task 8: 8/10.** A job that itself raises `asyncio.CancelledError` aborts the runner and affects other workers, violating the stated isolation of single-job failures. Independently reproduced.

**Why the other tasks scored full points:** tasks 1–3 passed inspected behavior and rerun Python tests; task 4 passed PostgreSQL tests recorded by the agent, although independently recreating the database was out of scope. Task 6 meets the original brief's email and mutation-isolation requirements despite a different API from the reference example. The identified task-7 issue concerns **optional variable per-put TTL** not required by the original statement, so we do not penalize unrelated extra functionality.

## Key GLM findings

**Task 5 — parsePort (8/10).** The reference returns fallback for invalid supplied values, whereas GLM raises `RangeError` for `parsePort("8080abc", 3000)`; the wording “reject” is ambiguous here. More importantly, `parsePort(8080, 0)` does **not** raise for an invalid fallback. The original prompt explicitly requires a `RangeError` for invalid fallback. Both examples were independently reproduced.

**Task 6 — Storage.** The reference uses a Python `register(email,name)` / `login(email)` API, while GLM implemented a TypeScript password API. Email case folding and return-object mutation safeguards are present, but the interfaces are not drop-in equivalents. Production TS modules compile; the entire external Node test toolchain was not reproduced in the separate review.

**Task 7 — LRU+TTL.** A counterexample affects GLM's **optional per-put TTL override**: bounded expired-entry cleanup may evict a live entry while leaving an expired one. The original short brief did **not** require variable per-put TTL. Do not penalize GLM on the baseline task solely for a flaw in an extra feature.

**Task 8 — run_jobs (8/10).** When one job itself raises `asyncio.CancelledError`, GLM cancels the entire runner rather than isolating this job failure. The original brief explicitly says one job failure must not cancel others. This is a reproduced baseline defect, even though the solution's own tests passed.

## Validation and scoring limitations

- Reconstructed Python tests for tasks **1, 2, 3 and 8** passed when independently rerun, but an added task-8 case failed.
- PostgreSQL 16 was exercised in the original agent trace for task 4; it was not independently deployed here.
- Production TypeScript modules compiled; complete Node-based test suites were not independently replayed.
- GLM **76/80** and the Qwen totals are **manual expert ratings**. They use the same broad grading scale, but a stricter numeric comparison would require identical independent hidden tests across all four models.

The supplied agent archives are not mirrored publicly because they may include unnecessary local server/tool metadata. This follow-up makes **no elapsed-time or model-speed claim** about the eight coding tasks.
