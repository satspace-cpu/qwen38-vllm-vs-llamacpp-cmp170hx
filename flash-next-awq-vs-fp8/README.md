# Qwen3.8: Flash-Next AWQ W4A16 vs Flash-Next FP8 vs Dense 27B on CMP 170HX — 520K Context + Coding Agent Test

[Русская версия](README_RU.md) · [Benchmark script](bench_vllm_512k_exact_mtp4.py) · [Tasks and reference solutions](TASKS_AND_SOLUTIONS.md) · [Long-context CSV](benchmark_results.csv) · [Coding-agent CSV](coding_agent_results.csv)

## Why this test

This started as a practical comparison of two quantizations of the **same Qwen3.8 Flash-Next architecture** on an unusual but very capable local inference server built around four unlocked NVIDIA CMP 170HX cards. A third run was then added with the **dense Qwen3.8-27B BF16 model on 2× CMP 170HX**, using the exact same eight coding-agent tasks.

The goal was not just to measure short-prompt tokens/s. We wanted to answer two questions that matter for real coding-agent use:

1. How do **Flash-Next AWQ W4A16** and **Flash-Next FP8** behave from 15K all the way to ~520K prompt tokens?
2. Does the much faster 4-bit AWQ model lose enough coding/reasoning quality to offset the speed gain?
3. How does the smaller **dense Qwen3.8-27B BF16** model compare on the same coding-agent workload?

The result on this machine was surprisingly clear: **AWQ W4A16 was about 21% faster on cold prefill and about 41% faster on decode, while manual review of eight coding tasks showed only a very small quality difference versus FP8.** The dense **27B BF16** run scored **77/80**, close to both Flash-Next variants. Its measured agent time was **1,058 s**, but that timing was obtained on **2× CMP 170HX / TP=2**, whereas both Flash-Next runs used **4× CMP 170HX**. Therefore the 27B wall-clock result is recorded as a useful two-GPU baseline, **not as an apples-to-apples speed comparison**.

This is not an official model benchmark and it should not be generalized to other GPUs without retesting. The result is especially hardware/backend dependent because the AWQ build used WNA16/Marlin on SM80 while the FP8 build used a different execution path.

---

## Test server

| Component | Configuration |
|---|---|
| GPUs | **4× NVIDIA CMP 170HX** (GA100 / SM80) |
| VRAM | **64 GB HBM2e per GPU, 256 GB total**, unlocked |
| GPU memory clock | 1728 MHz (NDIV 64 during this test period) |
| PCIe | Gen2 x16 per CMP 170HX |
| Topology | Single-NUMA EPYC platform, GPU peer access working between all pairs |
| CPU | AMD EPYC 7F52, 16 cores / 32 threads |
| System RAM | 128 GB DDR4 |
| OS | Debian Linux |
| NVIDIA driver | 610.57.04 |
| vLLM AWQ fork | `0.31.1.dev5+g61613eb01` with local Qwen3.8 Flash-Next / SM80 compatibility work |
| Parallelism | TP=4; AWQ additionally uses Expert Parallel |
| Speculative decoding | MTP, 4 speculative tokens |
| Max context | 524,288 tokens with YaRN override |
| KV cache | BF16 |

The CMP 170HX is an Ampere/GA100-derived card. These cards do **not** have native Hopper-style FP8 Tensor Cores, so FP8 performance depends heavily on software kernels. That matters when interpreting the comparison.

---

## Model configurations

### FP8 baseline

The FP8 setup was the established working baseline for Flash-Next on this server. It used:

- 4 GPUs with tensor parallelism;
- MTP depth 4;
- 524,288-token YaRN context extension;
- BF16 KV cache;
- PLE/Engram CPU offload using `VLLM_PLE_CPU_OFFLOAD=1`;
- the existing SM80 compatibility patches used for this Flash-Next FP8 checkpoint.

PLE/Engram CPU offload was important for the FP8 build because keeping the full PLE memory resident on GPU made the memory footprint much harder to manage on this setup.

### AWQ W4A16

The AWQ setup used the local checkpoint:

```text
Qwen3.8-Flash-Next-AWQ-W4A16
```

The working serving configuration used:

- TP=4;
- `--enable-expert-parallel`;
- 512 routed experts distributed as 128 local experts per GPU;
- CompressedTensors WNA16 MoE;
- **Marlin WNA16** backend;
- MTP depth 4;
- 524,288-token YaRN context extension;
- BF16 KV cache;
- `--engram-config.cpu_offload false`, so PLE/Engram remained on GPU.

A key implementation detail is that TP=4 without Expert Parallel was not valid for this checkpoint's static group scales: a TP shard would have an MoE intermediate size of 160, which is not divisible by the AWQ group size 128. Enabling Expert Parallel solved that layout problem.

The AWQ fork also needed a local compatibility fix so the model-wide `CompressedTensorsConfig` was not incorrectly applied to the PLE n-gram embedding path when those PLE weights were not stored in the same W4 format.

The exact working AWQ launch command was:

```bash
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 CUDA_VISIBLE_DEVICES=0,1,2,3 \
/mnt/llama-data/vllm-venv/vllm-0.31-qwen38-flash-next-awq-w4a16-sm80/bin/vllm serve \
/mnt/llama-data/modelsVLLM/Qwen3.8-Flash-Next-AWQ-W4A16/ \
--served-model-name Qwen3.8-Flash-Next-FP8 \
--host 0.0.0.0 \
--port 8000 \
--disable-access-log-for-endpoints /health,/-/health,/v1/models,/metrics \
--enable-auto-tool-choice \
--tool-call-parser qwen3_coder \
--reasoning-parser qwen3 \
--dtype bfloat16 \
--tokenizer-mode auto \
--gpu-memory-utilization 0.9 \
--tensor-parallel-size 4 \
--distributed-executor-backend mp \
--enable-expert-parallel \
--max-model-len 524288 \
--hf-overrides '{"text_config":{"rope_parameters":{"mrope_interleaved":true,"mrope_section":[11,11,10],"rope_type":"yarn","rope_theta":10000000,"partial_rotary_factor":0.25,"factor":2.0,"original_max_position_embeddings":262144}}}' \
--kv-cache-dtype bfloat16 \
--enable-prefix-caching \
--max-num-seqs 4 \
--max-num-batched-tokens 8192 \
--enable-chunked-prefill \
--scheduling-policy fcfs \
--async-scheduling \
--speculative-config '{"method":"mtp","num_speculative_tokens":4}' \
--attention-backend auto \
--load-format auto \
--engram-config.cpu_offload false
```

The served model name intentionally remained the same as the FP8 profile because the surrounding local software expected a fixed API model ID. It does not mean the AWQ checkpoint was FP8.

### Dense Qwen3.8-27B BF16

The third coding-agent run used the dense 27B checkpoint on **two CMP 170HX GPUs**:

```text
/run/media/server/nvme/vllm_model/Qwen3.8-27B/
```

Serving configuration:

- `CUDA_VISIBLE_DEVICES=0,3`;
- tensor parallel size 2;
- BF16 weights / activations;
- no quantization (`quantization=None`);
- MTP enabled with **5 speculative tokens**;
- native model context **262,144 tokens** for this coding test;
- BF16 KV cache;
- chunked prefill and prefix caching enabled;
- `max_num_batched_tokens=8192`.

The startup log resolved the model as `Qwen3_5ForConditionalGeneration` and the speculative head as `Qwen3_5MTP`. The 51.75 GiB checkpoint used about **26.16 GiB per GPU** under TP=2. vLLM reported about **32.12 GiB of KV-cache memory per GPU** and a total KV capacity of roughly **927k tokens**, so the 262K single-agent test had ample cache headroom.

This 27B run is included in the **coding-agent quality and wall-clock comparison only**. A 512K/520K long-context speed run for 27B had not yet been performed at the time of this article update.

---

## Benchmark methodology

The long-context benchmark was run through the OpenAI-compatible vLLM API using an **exact cold-prompt test**.

The exact benchmark script is included in this repository: [bench_vllm_512k_exact_mtp4.py](bench_vllm_512k_exact_mtp4.py).

uses `/tokenize` to build prompts at the requested real token count and gives every test a unique prefix so that vLLM's prefix cache cannot turn the run into an artificial hot-cache benchmark.

Each long-context point generated 512 output tokens. The tested prompt sizes were approximately:

```text
15K, 64K, 128K, 262K, 400K, 500K, 520K
```

A separate long-generation test used a short 119-token prompt and requested 10,000 output tokens.

`Effective prefill = prompt_tokens / TTFT`, so it includes API/scheduling/first-token overhead. It is therefore best used for **A/B comparison between these two runs**, not as a 1:1 replacement for an internal kernel-only prefill metric.

---

## Long-context speed results

| Prompt | FP8 prefill | AWQ W4A16 prefill | AWQ gain | FP8 decode | AWQ W4A16 decode | AWQ gain |
|---:|---:|---:|---:|---:|---:|---:|
| 15,004 | 2,796.7 | **3,469.0** | **+24.0%** | 184.7 | **260.1** | **+40.8%** |
| 64,003 | 2,853.5 | **3,452.6** | **+21.0%** | 187.3 | **263.8** | **+40.8%** |
| 128,001 | 2,830.8 | **3,431.9** | **+21.2%** | 187.5 | **263.8** | **+40.7%** |
| 262,006 | 2,789.1 | **3,377.6** | **+21.1%** | 191.4 | **271.5** | **+41.8%** |
| 400,004 | 2,751.2 | **3,318.9** | **+20.6%** | 194.0 | **273.3** | **+40.9%** |
| 500,008 | 2,729.0 | **3,293.2** | **+20.7%** | 194.9 | **276.5** | **+41.9%** |
| 520,005 | 2,725.7 | **3,284.1** | **+20.5%** | 195.9 | **276.8** | **+41.3%** |

Across these seven long-context points, the AWQ run was roughly **21.3% faster on effective cold prefill** and **41.2% faster on decode**.

### 520K result

The most striking point is the 520K cold run:

```text
Prompt:                520,005 tokens
AWQ TTFT:              158.341 s
AWQ effective prefill: 3,284.1 tok/s
AWQ generation:        276.8 tok/s
```

The corresponding FP8 result was:

```text
FP8 effective prefill: 2,725.7 tok/s
FP8 generation:        195.9 tok/s
```

### 10K-token generation test

| Model | Prompt | Output | Generation speed |
|---|---:|---:|---:|
| FP8 | 119 | 10,000 | 106.6 tok/s |
| AWQ W4A16 | 119 | 10,000 | **152.1 tok/s** |

That is a **+42.7%** decode advantage for AWQ in this long-generation test.

---

## Why the AWQ build is so fast here

The result should not be reduced to “4-bit is always faster than FP8.” Several things line up particularly well on this server:

1. **Routed expert weights are much smaller.** Weight-only INT4 reduces HBM traffic for the MoE expert matrices.
2. **Expert Parallel fits the model naturally.** The 512 experts are distributed across four GPUs instead of forcing an awkward TP split through the W4 group layout.
3. **Marlin WNA16 is a strong SM80 path.** The AWQ checkpoint landed on a backend that is very effective on Ampere-class hardware.
4. **PLE/Engram stays on GPU in the working AWQ configuration.** The smaller routed expert footprint leaves more room for this than the FP8 setup, which used CPU offload.
5. **The FP8 path on CMP 170HX is not native Hopper FP8.** GA100/SM80 does not have the same native FP8 execution hardware as newer architectures.

This means the observed speedup is a property of the full stack: **checkpoint format + vLLM kernels + EP layout + SM80 hardware + PLE placement**.

---

## Coding quality test

Speed alone is not enough for a coding model, so **all three configurations** were given the same eight tasks in separate fresh turns.

The task set covered:

1. Python session grouping / sorting / boundary correctness;
2. async cache race conditions, shared in-flight work and cancellation;
3. smallest covering range over many sorted lists;
4. PostgreSQL aggregation + latest-row selection + indexing;
5. strict TypeScript port parsing;
6. multi-file Python storage normalization and copy semantics;
7. O(1)-amortized LRU + TTL cache design;
8. bounded-concurrency async job runner with cancellation and generator support.

The complete task statements and reference solutions are in [TASKS_AND_SOLUTIONS.md](TASKS_AND_SOLUTIONS.md).

### Manual quality score

This was a manual review of correctness and requirement compliance, not an official benchmark score.

| Task | AWQ W4A16 | FP8 | Dense 27B BF16 | Main observation |
|---|---:|---:|---:|---|
| 1. Session grouping | 9.5 | **10.0** | **10.0** | All solved it correctly; 27B and FP8 were slightly cleaner |
| 2. AsyncCache | **10.0** | **10.0** | **10.0** | All handled shared work, retry and cancellation well |
| 3. Covering range | **10.0** | **10.0** | **10.0** | Correct heap solution, O(N log K) |
| 4. PostgreSQL | 9.5 | 9.5 | **10.0** | 27B produced a particularly clean SQL solution |
| 5. TypeScript parser | **10.0** | **10.0** | **10.0** | All met the strict parsing contract |
| 6. Storage semantics | 9.5 | **10.0** | **10.0** | AWQ additionally stripped email whitespace, which was outside the stated contract |
| 7. TTL/LRU | **10.0** | **10.0** | **7.5** | 27B tied expiry cleanup to LRU order and missed a stale-entry counterexample |
| 8. Bounded async runner | 9.5 | **10.0** | 9.5 | FP8 handled the subtle CancelledError distinction most cleanly |
| **Total** | **78.0 / 80** | **79.5 / 80** | **77.0 / 80** | All three were close, but 27B had one real logic miss in task 7 |

The important result is not the small score spread by itself. **AWQ did not fail any of the eight tasks outright**, while dense 27B also stayed close overall but exposed one meaningful reasoning error in task 7. Its LRU/TTL design used LRU order to drive lazy expiry cleanup, so an expired MRU entry could survive while a fresh LRU entry was evicted. That made task 7 a useful discriminator between an implementation that looked plausible and one that satisfied the full contract.

---

## Agent wall-clock time

The session logs contain turn timestamps, allowing the total solve time to be compared as well.

| Task | AWQ W4A16 | FP8 | Dense 27B BF16 |
|---|---:|---:|---:|
| 1 | **15.3 s** | 33.8 s | 36.7 s |
| 2 | **46.9 s** | 59.1 s | 71.4 s |
| 3 | **35.0 s** | 47.4 s | 130.8 s |
| 4 | **34.6 s** | 43.1 s | 69.8 s |
| 5 | **28.8 s** | 47.3 s | 82.1 s |
| 6 | **22.8 s** | 30.1 s | 59.4 s |
| 7 | **60.5 s** | 81.6 s | 170.7 s |
| 8 | **71.4 s** | 82.7 s | 437.2 s |
| **Total** | **315.2 s** | **425.1 s** | **1,058.1 s** |

AWQ completed the complete task set in about **5 min 15 s**, FP8 in about **7 min 05 s**, and the dense 27B TP2 baseline in about **17 min 38 s**. However, the 27B timing must be interpreted separately: it used only **2 GPUs / TP=2**, while AWQ and FP8 used **4 GPUs**. The ratios versus AWQ/FP8 are therefore descriptive only and must not be treated as a fair same-hardware speed ranking.

This is an agent-level result, so it includes not only raw decoding speed but also differences in how many reasoning/tool steps each run took. The 27B model was especially iterative on tasks 7 and 8.

---

## Practical conclusion

On this exact 4× CMP 170HX server, **Qwen3.8 Flash-Next AWQ W4A16 is currently the better day-to-day configuration**:

- about **+21% cold-prefill speed** across 15K–520K;
- about **+41% decode speed** across the same long-context range;
- about **+43%** on the 10K-output generation test;
- about **26% less wall time** on the eight-task coding-agent session;
- only a small manual quality gap versus FP8 in this test set: **78/80 vs 79.5/80**;
- the current dense 27B run provides a useful **2-GPU TP2 baseline** at **77/80 and 1,058 s**, but a **4-GPU TP4 rerun is still required** before comparing its wall-clock efficiency directly against the two Flash-Next runs.

FP8 still remains useful as a **maximum-confidence reference configuration** for difficult tasks where even a small quantization-induced reasoning difference may matter. Dense 27B remains interesting as a smaller two-GPU baseline. Its quality result is already useful, but its timing should stay provisional until the same eight tasks are rerun on **4× CMP 170HX / TP=4**.

The next useful test is not another synthetic speed run. It is a larger repo-level coding benchmark with multi-file edits, test execution, hidden regressions and long project context. That is where a small W4 quality loss, if present, is most likely to become visible.

---

## Reproducing the exact cold benchmark

On this server the benchmark was launched with:

```bash
cd /home/server/Обмен
python3 bench_vllm_512k_exact_mtp4.py
```

The long-context phase targets:

```text
15K  -> 512 output tokens
64K  -> 512
128K -> 512
262K -> 512
400K -> 512
500K -> 512
520K -> 512
```

plus a short-prompt 10,000-token generation test.

---

## Notes for reproducibility

- The API model ID was intentionally kept identical between profiles for compatibility with local software.
- Prefix caching was enabled on the server, but the exact-cold benchmark used a unique prefix per test so the long-context numbers above are cold-prompt measurements.
- The YaRN extension used `factor=2.0` with `original_max_position_embeddings=262144` and `max_model_len=524288`.
- Flash-Next AWQ/FP8 used MTP with four speculative tokens. The current dense 27B TP2 coding run used five speculative tokens; the planned TP4 rerun should preferably use the same MTP depth as the Flash-Next runs for cleaner timing comparison.
- These numbers are **single-request local inference results**, not multi-user serving throughput.
- The benchmark reflects this exact hardware and fork state. Kernel selection can change materially between vLLM versions.
