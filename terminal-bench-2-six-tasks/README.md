# Qwen3.8 vs cloud coding agents: six Terminal-Bench 2.0 tasks on 4× CMP 170HX

[Русская версия](README_RU.md) · [Complete rankings CSV](ranking_6_tasks.csv) · [Earlier three-model benchmark](../flash-next-awq-vs-fp8/README.md)

## Executive summary

On **six selected** Terminal-Bench 2.0 tasks, each of three locally hosted Qwen3.8 configurations running **DeepSeek Harness (DSH)** achieved **5/6 (83.3%)**. The public Terminal-Bench Analysis extract includes cloud systems with 6/6 equivalent averages (including GPT-5.5 + NexAU-AHE and GPT-5.3-Codex + LemonHarness), and many with lower averages. These figures are **not** full 89-task Terminal-Bench 2.0 scores, nor do they establish parity between the underlying model weights: the measured system is *model + agent + environment*, with different sampling protocols.

## Hardware and software

| Component | Configuration |
|---|---|
| GPU | **4× NVIDIA CMP 170HX**, GA100/Ampere (SM80) |
| VRAM | **64 GB unlocked HBM2e each, 256 GB total** |
| HBM clock | 1728 MHz / NDIV 64 during prior measurement period |
| PCIe/topology | Gen2 x16 per GPU, single-NUMA EPYC host, peer access across GPU pairs |
| CPU | **AMD EPYC 7F52**, 16 cores / 32 threads |
| RAM | 128 GB DDR4 |
| Operating system | Linux; package output from the Terminal-Bench setup identifies Ubuntu resolute, while the earlier article said Debian; exact benchmark-run OS release was not separately captured |
| NVIDIA driver | 610.57.04 in earlier server records, not revalidated during these runs |
| Docker | 29.1.3; Compose 2.40.3 |
| Harbor | 0.24.0, official task verifier |
| Agent | Web-based DeepSeek Harness with tools bridged into isolated Docker task environments |
| Agent context | Configured target 262,144 tokens; actual task context usage unmeasured |

The **Flash-Next AWQ W4A16** profile uses TP4, Expert Parallel, Marlin/WNA16 on SM80, BF16 KV and MTP4; **Flash-Next FP8** uses TP4, BF16 KV, MTP4 and PLE CPU offload; **dense Qwen3.8-27B BF16** uses TP2, BF16 KV and MTP5. These deployment configurations are documented in our [earlier article](../flash-next-awq-vs-fp8/README.md), rather than independently snapshotted for every Terminal-Bench session. The CMP170HX lacks Hopper-native FP8 Tensor Cores, making backend/kernel selection especially significant.

## Method

Six official tasks: `regex-log`, `db-wal-recovery`, `count-dataset-tokens`, `build-cython-ext`, `query-optimize`, `large-scale-text-editing`. Original instructions, task metadata, Docker environment and Harbor verifier were used. DSH accessed each task environment via tools, and the initial `regex-log` oracle verified the task setup (reward 1.0); **oracle results were never counted as model scores**.

Local binary 0/1 scores represent **selected final runs**, not controlled single-attempt pass@1. In particular, dense BF16's successful `build-cython-ext` trial followed earlier attempts and used an extended 1,800-second allowance versus the original 900 seconds. Cloud entries are **per-task mean rewards across published attempts**, taken from a user-supplied export of [Terminal-Bench Analysis](https://github.com/prime-radiant-inc/terminal-bench-analysis). Submission labels identify the agent; some rows combine multiple cloud models and cannot be attributed to any single model. Number of attempts and precise matched budgets are not specified in the extract. **Gemini CLI has no SQL value and is excluded from complete-ranking averages.**

## Full comparison: 3 local configurations and published cloud systems

**🟩 AWQ · 🟦 FP8 · 🟪 Dense BF16 — local systems are color-coded and bold, including every score.**

Each cell is a percentage (0–100); the six-task mean is the arithmetic average of all six cells. Cloud means and local selected-run outcomes are different statistics. Ties share a rank.

| System (model + agent) | Regex | WAL | Tokens | Cython | SQL | Text | 6-task mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| LemonHarness / GPT-5.3-Codex | 100 | 100 | 100 | 100 | 100 | 100 | 100.0% |
| NexAU-AHE / GPT-5.5 | 100 | 100 | 100 | 100 | 100 | 100 | 100.0% |
| OB-1 / GPT-5.4 + GPT-5.3 + Claude | 100 | 100 | 100 | 100 | 100 | 100 | 100.0% |
| LemonCode / GPT-5.3-Codex | 100 | 100 | 100 | 100 | 100 | 80 | 96.7% |
| Judy / Claude Opus 4.6 | 100 | 100 | 100 | 100 | 75 | 100 | 95.8% |
| Polaris / Claude 4.7 + GPT-5.5 + Gemini 3.1 | 100 | 100 | 100 | 100 | 80 | 80 | 93.3% |
| spoox-o-m / GPT-5.3-Codex | 100 | 80 | 100 | 100 | 75 | 100 | 92.5% |
| Capy / GPT-5.5 | 100 | 40 | 100 | 100 | 100 | 100 | 90.0% |
| Forge / Gemini 3.1 Pro | 100 | 100 | 100 | 20 | 100 | 100 | 86.7% |
| Mux / GPT-5.3-Codex | 100 | 60 | 100 | 100 | 60 | 100 | 86.7% |
| Judy / Gemini 3.1 Pro | 100 | 100 | 100 | 60 | 66.7 | 80 | 84.5% |
| Ante / Gemini 3.1 Pro | 100 | 20 | 80 | 100 | 100 | 100 | 83.3% |
| **🟪 LOCAL — Qwen3.8 27B BF16 / DSH** | **100** | **100** | **100** | **100** | **0** | **100** | **83.3%** |
| **🟩 LOCAL — Qwen3.8 Flash-Next AWQ W4A16 / DSH** | **100** | **100** | **0** | **100** | **100** | **100** | **83.3%** |
| **🟦 LOCAL — Qwen3.8 Flash-Next FP8 / DSH** | **100** | **100** | **100** | **100** | **0** | **100** | **83.3%** |
| WozCode / Claude Opus 4.6 | 100 | 100 | 60 | 80 | 60 | 100 | 83.3% |
| Droid / GPT-5.3-Codex | 100 | 40 | 100 | 100 | 40 | 100 | 80.0% |
| Junie CLI / multiple models | 100 | 20 | 100 | 100 | 60 | 100 | 80.0% |
| CodeBrain 1.5 / GPT-5.3-Codex | 100 | 100 | 60 | 100 | 0 | 100 | 76.7% |
| Copilot CLI / Claude Opus 4.6 | 100 | 0 | 100 | 100 | 60 | 100 | 76.7% |
| Meta-Harness / Claude Opus 4.6 | 100 | 100 | 40 | 100 | 20 | 100 | 76.7% |
| Simple-Codex / GPT-5.3-Codex | 100 | 0 | 100 | 100 | 60 | 100 | 76.7% |
| Capy / Claude Opus 4.6 | 100 | 20 | 100 | 100 | 20 | 100 | 73.3% |
| Terminus2 / GPT-5.3-Codex | 100 | 0 | 100 | 100 | 40 | 100 | 73.3% |
| Crux / Claude Opus 4.6 | 40 | 100 | 40 | 100 | 50 | 100 | 71.7% |
| OpenSage / GPT-5.3-Codex | 100 | 20 | 80 | 100 | 50 | 80 | 71.7% |
| clnkr / GPT-5.5 | 100 | 0 | 100 | 100 | 20 | 100 | 70.0% |
| Codelia / GPT-5.3-Codex | 100 | 0 | 100 | 100 | 20 | 100 | 70.0% |
| OB-1 / GPT-5.3 + Claude 4.5/4.6 | 100 | 0 | 100 | 100 | 20 | 100 | 70.0% |
| IndusAGI / GPT-5.3-Codex | 100 | 20 | 100 | 100 | 20 | 75 | 69.2% |
| CodeBrain 1 / GPT-5.3-Codex | 100 | 20 | 100 | 100 | 0 | 80 | 66.7% |
| Droid / Claude Opus 4.6 | 100 | 0 | 60 | 100 | 40 | 100 | 66.7% |
| Mux / Claude Opus 4.6 | 100 | 0 | 60 | 100 | 40 | 100 | 66.7% |
| Terminus-KIRA / Gemini 3.1 Pro | 100 | 20 | 100 | 80 | 0 | 100 | 66.7% |
| Terminus-KIRA / Claude Opus 4.6 | 100 | 0 | 80 | 100 | 0 | 100 | 63.3% |
| Terminus2 / Claude Opus 4.6 | 100 | 0 | 60 | 60 | 60 | 100 | 63.3% |
| Gemini CLI / Gemini 3.1 Pro | 100 | 20 | 100 | 50 | N/A | 20 | N/A |

[Download the machine-readable ranking](ranking_6_tasks.csv). Cloud entries transcribed from the supplied Datasette export; original per-trial submission IDs/logs were not part of that export.

## Findings

- **AWQ**, **FP8**, and **27B BF16** each achieved 5/6 but on different failure profiles. AWQ failed to create the required `count-dataset-tokens` output file; FP8 failed to deliver the required `query-optimize` output; dense BF16 produced SQL that passed correctness but missed the performance constraint. Missing artifacts are failures of *agent task completion*, not direct evidence of a wrong mathematical result.
- All three local agents passed **`db-wal-recovery`**, while the public averages range from 20% for Gemini 3.1 + Ante to 40% for GPT-5.5 + Capy and GPT-5.3-Codex + Droid, but **100%** for GPT-5.5 + NexAU-AHE. This illustrates how much an agent scaffold matters.
- AWQ alone among the local profiles passed **`query-optimize`**, whereas public top systems (GPT-5.5 + NexAU-AHE and GPT-5.3-Codex + LemonHarness) reached 100% for this task.
- The selected-run DSH agent wall times reported locally were **29m45s AWQ**, **36m28s FP8**, and **63m50s BF16**, with **148 / 172 / 226** tool calls respectively. Docker setup and verifier time are excluded. This is not a clean model inference-speed comparison: agent trajectories, TP2 vs TP4 and retries confound those totals.
- In the **separate 520K long-context speed benchmark**, AWQ achieved approximately **+21% effective cold prefill** and **+41% decode throughput** vs FP8 on this server. These speed gains cannot be assigned to cloud agents or inferred from this Terminal-Bench study.

## Conclusion

**Flash-Next AWQ W4A16 is the practical local favorite in this selected set:** a 5/6 score at the shortest measured agent time, and the only local profile passing the SQL optimization task. FP8 and dense 27B matched 5/6 through different successes and failures. The local systems are competitive with numerous published *cloud agent systems* on these six tasks, while the strongest cloud combinations score a perfect six-task mean. A rigorous model-level ranking needs identical attempt budgets, agent protocols, failure handling, and more tasks.

**Sources:** [Terminal-Bench 2.0](https://www.tbench.ai/?version=2.0) · [Terminal-Bench Analysis](https://github.com/prime-radiant-inc/terminal-bench-analysis) · [previous detailed server benchmarks](../flash-next-awq-vs-fp8/README.md).
