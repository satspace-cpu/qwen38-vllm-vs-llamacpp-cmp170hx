# Eight coding tasks: GLM-5.3-Flash vs three Qwen configurations — correctness only
**14 October 2026 · [Русский](RESULTS_RU.md) · [Solution-free prompts](TASKS_ONLY_RU.md) · [Original Qwen benchmark](../../flash-next-awq-vs-fp8/README.md)**

**This report evaluates correctness only.** The GLM run shared the server with **two additional concurrent inference workloads**. Its elapsed agent times are **not comparable** with the prior Qwen runs and have been removed from this report, along with elapsed-time figures and CSV.

The Qwen numbers below are the **unchanged manual 0–10 scores** reported in the initial three-model study. For GLM, we report independently checked outcomes and limitations instead of inventing a directly comparable /80 score without a common hidden-test suite.

| Task | Qwen Next FP8 | Qwen Next AWQ | Qwen 27B BF16 | GLM-5.3-Flash assessment |
|---|---:|---:|---:|---|
| 1. Python session grouping | 10 | 9.5 | 10 | ✅ Passed inspected requirements; Python tests independently rerun |
| 2. AsyncCache concurrency | 10 | 10 | 10 | ✅ Shared in-flight work and cancellation isolation; tests rerun |
| 3. Smallest covering range | 10 | 10 | 10 | ✅ Heap-based algorithm, claimed complexity, tests rerun |
| 4. PostgreSQL aggregation | 9.5 | 9.5 | 10 | ◐ SQL tested within the agent, not redeployed in an independent DB |
| 5. TypeScript parsePort | 10 | 10 | 10 | ⚠️ Behavior differs from published reference on invalid supplied value |
| 6. Storage semantics | 10 | 9.5 | 10 | ⚠️ Email matching and object isolation work, but API differs from reference |
| 7. LRU + TTL | 10 | 10 | 7.5 | ✅ Core requirements appear met; a bug exists in optional per-put variable TTL extension outside the original brief |
| 8. Async job runner | 10 | 9.5 | 9.5 | ❌ Reproduced failure: job-raised `CancelledError` cancels other workers |
| **Original manual total** | **79.5 / 80** | **78 / 80** | **77 / 80** | **Not numerically graded yet** |

Legend: ✅ inspected requirements satisfied (not a proof of bug-free implementation); ◐ incomplete independent verification; ⚠️ contract mismatch or ambiguous original wording; ❌ reproduced specification violation.

## Key GLM findings

**Task 5 — parsePort.** The published reference returns the fallback for an invalid supplied value; only an invalid fallback raises `RangeError`. GLM raises `RangeError` for `parsePort("8080abc", 3000)`. This mismatch was reproduced independently. However, the short prompt's word “reject” does not strictly define the rejection mechanism. It is a **reference-contract difference**, not an unequivocal failure under every possible interpretation.

**Task 6 — Storage.** The reference uses a Python `register(email,name)` / `login(email)` API, while GLM implemented a TypeScript password API. Email case folding and return-object mutation safeguards are present, but the interfaces are not drop-in equivalents. Production TS modules compile; the entire external Node test toolchain was not reproduced in the separate review.

**Task 7 — LRU+TTL.** A counterexample affects GLM's **optional per-put TTL override**: bounded expired-entry cleanup may evict a live entry while leaving an expired one. The original short brief did **not** require variable per-put TTL. Do not penalize GLM on the baseline task solely for a flaw in an extra feature.

**Task 8 — run_jobs.** When one job itself raises `asyncio.CancelledError`, GLM cancels the entire runner rather than isolating this job failure. The original brief explicitly says one job failure must not cancel others. This is a reproduced baseline defect, even though the solution's own tests passed.

## Validation and scoring limitations

- Reconstructed Python tests for tasks **1, 2, 3 and 8** passed when independently rerun, but an added task-8 case failed.
- PostgreSQL 16 was exercised in the original agent trace for task 4; it was not independently deployed here.
- Production TypeScript modules compiled; complete Node-based test suites were not independently replayed.
- The previously published Qwen totals are **manual ratings**; a directly equivalent numeric GLM /80 score would require identical independent scenarios and interfaces across all four models.

The supplied agent archives are not mirrored publicly because they may include unnecessary local server/tool metadata. This follow-up makes **no elapsed-time or model-speed claim** about the eight coding tasks.
