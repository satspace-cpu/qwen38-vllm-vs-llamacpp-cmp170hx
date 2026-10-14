# GLM-5.3-Flash: eight coding-agent tasks — follow-up results
**14 October 2026 · provisional review · [Русский](RESULTS_RU.md) · [Solution-free tasks (RU)](TASKS_ONLY_RU.md) · [Main GLM report](../README.md)**

We repeated the eight coding tasks from the previously published [Qwen Next AWQ, Next FP8 and dense 27B BF16 comparison](../../flash-next-awq-vs-fp8/README.md) on **GLM-5.3-Flash AWQ**, running on **4× CMP 170HX / pipeline-parallel 4 / MTP4**. New timing numbers are based on actual `turn/start` and `turn/end` timestamps in the supplied DSH agent-session archive.

**Methodology warning:** The eight GLM tasks were sent as **eight consecutive turns in one cumulative agent session**, not eight independent fresh sessions. The context grew substantially; agent iteration and test depth differ between models. The prior Qwen values are sourced from the original published CSV. Dense Qwen 27B used **2 GPUs / TP2**, unlike the other configurations. Thus these are **observed agent elapsed times, not controlled inference-throughput ratios**.

## Per-task wall-clock time

![Each task elapsed time](figures/elapsed_per_task.svg)

| Task | GLM, s | Qwen AWQ, s | Qwen FP8, s | Qwen 27B BF16, s |
|---|---:|---:|---:|---:|
| 1. Python sessions | **255.9** | 15.3 | 33.8 | 36.7 |
| 2. AsyncCache | **446.3** | 46.9 | 59.1 | 71.4 |
| 3. Smallest covering range | **288.1** | 35.0 | 47.4 | 130.8 |
| 4. PostgreSQL | **376.4** | 34.6 | 43.1 | 69.8 |
| 5. TypeScript parsePort | **368.2** | 28.8 | 47.3 | 82.1 |
| 6. Storage | **175.1** | 22.8 | 30.1 | 59.4 |
| 7. LRU + TTL | **400.9** | 60.5 | 81.6 | 170.7 |
| 8. Async run_jobs | **205.1** | 71.4 | 82.7 | 437.2 |
| **Sum of active turns** | **2515.9** | **315.2** | **425.1** | **1058.1** |

![Total active elapsed time](figures/elapsed_totals.svg)

GLM's sum of **active turn time** was **41 min 56 s**. Elapsed duration from first turn start to last turn end was **~67 min 15 s**, including idle gaps. The session contains **103 agent steps** and **107 tool calls**, including file writes, Python execution, TypeScript compilation and real PostgreSQL verification.

[Raw curated CSV](data/elapsed_comparison.csv).

## Correctness review

| Task | Evidence and caveat |
|---|---|
| **1. Sessions** | Sorted copy, strict 30-minute boundary, `O(n log n)`; the reconstructed Python file's tests **passed when rerun independently**. |
| **2. AsyncCache** | Shared per-key in-flight task, concurrent distinct keys, retry on failure and waiter-cancellation isolation via `asyncio.shield`; original Python tests **reran successfully**. |
| **3. Covering range** | Heap of K entries, `O(N log K)`; Python tests including a 1M-value scaling case **reran successfully**. |
| **4. PostgreSQL** | Actual PostgreSQL 16 calls and verification against curated and 1M-row datasets in the agent trace. Not independently deployed to a new database here. |
| **5. parsePort** | **Reference-behavior mismatch:** the published baseline returns fallback for an invalid supplied value, whereas GLM throws `RangeError`, reproduced with `parsePort("8080abc", 3000)`. The abbreviated task wording “reject” is not perfectly explicit about *how* to reject; this is a mismatch with the published reference implementation rather than an unambiguous failure under every interpretation. |
| **6. Storage** | Email matching and mutation-safety look sound; **the interface changes** from the published Python `register(email,name)` / `login(email)` baseline to a TypeScript password API `register(email,password)` / `login(email,password)`. TS production modules compile separately; the agent's own checks passed, but API equivalence is absent. |
| **7. LRU + TTL** | Key LRU and TTL semantics are implemented. A separate counterexample in GLM's **additional per-put TTL API** evicts a live item while leaving an expired entry, because cleanup is capped. Per-put variable TTL was not an explicit requirement of the original short prompt, so this is not automatically a baseline task failure. |
| **8. run_jobs** | Reconstructed Python tests reran successfully. A newly executed counterexample shows that when **one job itself raises `asyncio.CancelledError`**, GLM aborts the runner and cancels remaining workers. This violates the “one job exception must not cancel others” interpretation followed by the published reference, which distinguishes job cancellation from external runner cancellation. |

The agent reported **57 tests for task 5, 37 for task 6 and 49 for task 7**. Those are **self-authored checks**, not a common independent hidden-test suite. Python tests for tasks 1, 2, 3 and 8 were independently rerun; production TS modules for tasks 5–7 were independently compiled. Full TS test infrastructure (including Node type packages) was not reinstalled in the separate review environment.

## Why no precise GLM /80 score yet?

The earlier study manually scored **Qwen FP8 79.5/80, AWQ 78/80 and dense 27B BF16 77/80**. A fair fourth score requires a **common independently graded test suite** across all solutions. We documented clear differences and counterexamples but deliberately **have not assigned GLM a comparable numeric /80 grade yet** rather than mixing distinct verification standards.

## Next reproducible experiment

Use the [individual solution-free prompts](README_RU.md) in fresh, isolated sessions; record actual checkpoint IDs, identical tool/token budgets and model configuration. Clarify the ambiguous invalid-input fallback behavior for parsePort and the expected storage API, and test cancellation/expiration edge cases on all four models. Finally, compare graded correctness with identical independent tests, then update the numeric quality chart.

Raw agent ZIP archives are intentionally not published because they include unrelated local tool and server history; curated evidence and timing CSV are provided instead.
