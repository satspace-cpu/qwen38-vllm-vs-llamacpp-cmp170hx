# Qwen3.8-27B: vLLM vs llama.cpp on 2× CMP 170HX 64 GB

[Русская версия](README_RU.md)

## Overview

This repository collects real-world long-context inference benchmarks of **Qwen3.8-27B BF16** on two unlocked NVIDIA CMP 170HX GPUs (64 GB each), comparing **vLLM** and **llama.cpp**.

The goal is practical agent/coding workload performance: prompt ingestion (prefill), generation speed at very long context, MTP/speculative decoding behavior, and multi-GPU split modes.

> Work in progress. Results below are measured runs from the same test system. More context sizes and configurations will be added.

## Test platform

| Component | Configuration |
|---|---|
| GPUs | 2× NVIDIA CMP 170HX, unlocked to 64 GB each |
| GPU power limit | up to 300 W per GPU |
| CPU | 2× Xeon E5-2697 v4 |
| Model | Qwen3.8-27B BF16 |
| Context target | 262,144 tokens |
| Parallel requests | 1 |
| vLLM | tensor parallel, MTP/speculative decoding enabled in measured long-context run |
| llama.cpp | CUDA build; layer/tensor split tested; BF16 KV cache |

## Headline result at ~260K context

| Engine / mode | Prompt tokens | Prefill | Generation | MTP acceptance |
|---|---:|---:|---:|---:|
| vLLM + MTP | 250,000 | ~1,200 tok/s effective from TTFT | ~20.7 tok/s from TPOT | 77.0% |
| llama.cpp layer + MTP | 260,052 | **835.6 tok/s** | **31.6 tok/s** | 36.4% |
| llama.cpp tensor + MTP | 260,052 | **705.5 tok/s** | **37.7 tok/s** | — |
| llama.cpp tensor, no MTP | 260,052 | **825.9 tok/s** | **26.1 tok/s** | n/a |

### Important vLLM metric note

The vLLM serving benchmark reported:

- 250,000 input tokens
- 2,048 generated tokens
- TTFT: 208.31 s
- TPOT: 48.20 ms/token
- overall output throughput: 6.67 tok/s
- speculative acceptance: 77.02%
- mean accepted length: 4.85

`1 / TPOT` gives roughly **20.75 generated tokens/s** during decode. The lower 6.67 tok/s figure includes the huge initial prefill in total request duration, so it should not be compared directly with llama.cpp's `eval time` generation rate.

## llama.cpp: layer split + MTP

Measured final timing:

```text
prompt eval time = 311200.27 ms / 260052 tokens
                 = 835.64 tokens/s

eval time        = 13087.51 ms / 414 tokens
                 = 31.56 tokens/s

draft acceptance = 0.36351
                 = 269 accepted / 740 generated
mean len          = 2.82
```

Generation accelerated during the run from about 27.3 tok/s around the first 100 generated tokens to ~34.4 tok/s over the latest 3-second window around 400 tokens.

## llama.cpp: tensor split + MTP

```text
prompt eval time = 368599.88 ms / 260052 tokens
                 = 705.51 tokens/s

eval time        = 7800.06 ms / 295 tokens
                 = 37.69 tokens/s
```

Tensor split produced **faster decode** than layer split in this run, but substantially slower long-context prefill.

The server also reported:

```text
backend sampling not supported with SPLIT_MODE_TENSOR; using CPU
spec ... backend offload failed ... using CPU sampler
```

This is an important caveat for interpreting tensor-mode behavior.

## llama.cpp: tensor split without MTP

With MTP disabled, the embedded MTP tensors in the model were correctly reported as unused (`blk.64...`). This is expected and is not a model loading error.

```text
prompt eval time = 314871.81 ms / 260052 tokens
                 = 825.90 tokens/s

eval time        = 10694.55 ms / 280 tokens
                 = 26.09 tokens/s

graphs reused    = 278
```

MTP therefore had a major positive effect on decode speed in tensor mode in this comparison: **26.1 → 37.7 tok/s**, while prefill dropped from **825.9 → 705.5 tok/s**.

## Long-context comparison

```text
Generation speed (~260K context)

llama.cpp tensor + MTP      37.69 tok/s  ██████████████████████████████████████
llama.cpp layer + MTP       31.56 tok/s  ████████████████████████████████
llama.cpp tensor no MTP     26.09 tok/s  ██████████████████████████
vLLM + MTP (TPOT-derived)   20.75 tok/s  █████████████████████
```

```text
Prefill speed (~250-260K context)

vLLM + MTP (250K/TTFT)     ~1200 tok/s   ████████████████████████████████████████
llama.cpp layer + MTP       835.64 tok/s  ████████████████████████████
llama.cpp tensor no MTP     825.90 tok/s  ████████████████████████████
llama.cpp tensor + MTP      705.51 tok/s  ████████████████████████
```

The vLLM prefill value above is an approximation (`250000 / TTFT`) because TTFT includes more than pure model prefill. Raw benchmark data is preserved so future runs can improve apples-to-apples comparison.

## What the tests show so far

1. **llama.cpp layer split is extremely strong at long-context prefill.** At ~260K it reaches 835.6 tok/s and beats llama.cpp tensor+MTP by ~18%.
2. **Tensor split helps generation.** At ~260K, tensor+MTP reaches 37.7 tok/s versus 31.6 tok/s for layer+MTP.
3. **MTP is very workload-dependent.** Tensor mode improves from 26.1 to 37.7 tok/s with MTP, but its prefill becomes slower.
4. **vLLM MTP acceptance is dramatically better in the captured run:** ~77% versus ~36% for llama.cpp layer+MTP.
5. **vLLM showed much smoother GPU load** during the observed runs, while llama.cpp power/load oscillated strongly between the two GPUs. This is an observation from the test system, not yet a quantified benchmark.
6. For an agent workload, the best engine cannot be selected from prefill alone. Decode speed at 100K–260K context matters heavily once a coding session grows.

## Reproducing the vLLM long-context test

```bash
vllm bench serve \
  --backend openai \
  --base-url http://127.0.0.1:8000 \
  --model /mnt/llama-data/modelsVLLM/Qwen3.8-27B \
  --dataset-name random \
  --input-len 250000 \
  --output-len 2048 \
  --num-prompts 1 \
  --max-concurrency 1 \
  --temperature 0
```

## Planned test matrix

We are expanding the dataset from small prompts up to the full 262K context window. Planned checkpoints include roughly 1K, 4K, 8K, 16K, 32K, 64K, 100K, 128K, 150K, 200K, 250K and near-262K.

For every checkpoint we want to record:

- exact prompt tokens;
- prompt/prefill tok/s;
- generated tokens;
- decode tok/s;
- TTFT;
- MTP acceptance rate and accepted length;
- GPU power/utilization behavior;
- layer vs tensor split;
- MTP on/off;
- vLLM vs llama.cpp.

## Why this repository exists

Most LLM benchmarks focus on short prompts or aggregate server throughput. That is not enough for long-running coding agents. Once an agent accumulates 100K–250K tokens of project context, the performance profile changes dramatically.

This project focuses specifically on that region.

## Status

Initial data uploaded. More measurements, CSV datasets and proper plotted charts will be added as the test matrix grows.


---

## New: Qwen3.8 Flash-Next AWQ W4A16 vs FP8 on 4× CMP 170HX

A new 520K-context comparison is now available, including exact cold-prefill/decode benchmarks and an eight-task coding-agent quality comparison.

- [English article](flash-next-awq-vs-fp8/README.md)
- [Русская версия](flash-next-awq-vs-fp8/README_RU.md)
- [Coding tasks and reference solutions](flash-next-awq-vs-fp8/TASKS_AND_SOLUTIONS.md)
- [Raw benchmark CSV](flash-next-awq-vs-fp8/benchmark_results.csv)

Headline result on this server: AWQ W4A16 was about **21% faster on cold prefill**, about **41% faster on decode**, and completed the coding-agent task set in about **26% less wall time**, while the manual quality score remained close to FP8 (**78/80 vs 79.5/80**).
