#!/usr/bin/env python3
"""
vLLM OpenAI-compatible API benchmark.

Designed to mirror the structure of bench_llama.py while using metrics that
can be measured correctly from a vLLM-compatible streaming endpoint.

Measures:
  - prompt tokens (from OpenAI usage, when available)
  - generated tokens (from OpenAI usage, when available)
  - TTFT: time to first streamed token
  - end-to-end wall time
  - generation throughput after first token
  - effective prefill throughput = prompt_tokens / TTFT

Important:
  vLLM's OpenAI API does not expose llama.cpp's internal prompt_ms /
  predicted_ms timing fields. Therefore "effective prefill tok/s" is an
  end-to-end estimate based on TTFT and includes HTTP/queue/first-token cost.
  Generation tok/s is measured from first token until request completion.

Usage examples:
  python3 bench_vllm.py
  python3 bench_vllm.py --url http://127.0.0.1:8000 --model Qwen3.8-Flash-Next-FP8
  python3 bench_vllm.py --only standard
  python3 bench_vllm.py --only long
  python3 bench_vllm.py --long-context 64000
"""

import argparse
import json
import sys
import time
import urllib.request
import urllib.error

DEFAULT_SERVER = "http://127.0.0.1:8000"
DEFAULT_MODEL = "Qwen3.8-Flash-Next-FP8"


# Standard tests: same general intent as bench_llama.py
TESTS = [
    ("tiny", "Привет, как дела?", 64),
    (
        "small",
        "Напиши подробное описание архитектуры transformers: attention, positional encoding, "
        "feed-forward, residual, layer norm. Объясни encoder-decoder.",
        256,
    ),
    (
        "medium",
        "Напиши статью о миграции птиц с разделами: 1) почему мигрируют, 2) навигация, "
        "3) маршруты Евразии, 4) влияние климата, 5) методы изучения, 6) заключение. "
        "Подробно, с примерами видов.",
        1024,
    ),
    (
        "large",
        "Напиши учебный курс по биоакустике для орнитологов: 1) физика звука, "
        "2) акустика птичьей вокализации, 3) полевые записи, 4) обработка аудио, "
        "5) автоматическое распознавание (BirdNET), 6) статистика, 7) примеры на Python. "
        "Каждый раздел развёрнуто с формулами и кодом.",
        2048,
    ),
]


def generate_long_prompt(target_tokens: int) -> str:
    """
    Generate a synthetic prompt approximately target_tokens long.

    This intentionally mirrors the old bench_llama.py generator so the two
    scripts exercise similar input text.
    """
    target_words = int(target_tokens / 1.3)
    paragraph = (
        "The quick brown fox jumps over the lazy dog near the riverbank where ancient oaks "
        "stand tall against the evening sky. Researchers have documented numerous species "
        "of migratory birds passing through this corridor each spring, using sophisticated "
        "GPS tracking devices to map their routes across continents. The acoustic monitoring "
        "equipment deployed along the flight paths captures vocalizations ranging from 1 to "
        "12 kilohertz, enabling automated identification through machine learning pipelines "
        "built on spectrogram analysis and neural network classifiers. Field observations "
        "indicate that environmental factors such as temperature gradients, atmospheric "
        "pressure changes, and magnetic field variations all play crucial roles in avian "
        "navigation systems that have evolved over millions of years of natural selection."
    )
    words = paragraph.split()
    text_words = (words * (target_words // len(words) + 1))[:target_words]
    # Unique first tokens prevent accidental prefix-cache reuse across benchmark runs.
    # The benchmark still works normally when prefix caching is disabled.
    return f"BENCH-COLD-{time.time_ns()}: " + " ".join(text_words)


LONG_GEN_PROMPT = (
    "Write an extremely long and detailed comprehensive encyclopedia-style article covering "
    "20 topics in exhaustive depth. Each section must be at least 500 words with detailed "
    "examples, historical context, technical details. Do not be brief. Write continuously "
    "without stopping until you reach the maximum length. Topics: computer architecture, "
    "transformer architecture, bird vocalization, aviation history, deep learning optimizers, "
    "ocean currents, writing systems, particle physics, cryptography, mathematics history, "
    "neuroscience, climate science, stellar evolution, molecular biology, AI history, "
    "quantum computing, oceanography, fungi classification, nuclear physics, music theory."
)


def parse_sse_stream(resp):
    """
    Yield parsed JSON objects from an OpenAI-compatible Server-Sent Events stream.
    """
    for raw_line in resp:
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line:
            continue
        if line.startswith("data:"):
            data = line[5:].strip()
        else:
            continue

        if data == "[DONE]":
            break

        try:
            yield json.loads(data)
        except json.JSONDecodeError:
            continue


def extract_text_from_chunk(chunk):
    """
    Support both /v1/completions style chunks and common compatibility variants.
    """
    choices = chunk.get("choices") or []
    if not choices:
        return ""

    choice = choices[0] or {}

    # /v1/completions
    text = choice.get("text")
    if isinstance(text, str):
        return text

    # Defensive fallback for chat-like chunks
    delta = choice.get("delta") or {}
    content = delta.get("content")
    if isinstance(content, str):
        return content

    return ""


def run_test(server, model, name, prompt, max_tokens, timeout=1200, temperature=0.0):
    """
    Run one streaming request and measure end-to-end serving metrics.
    """
    url = server.rstrip("/") + "/v1/completions"

    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        },
        method="POST",
    )

    t_start = time.perf_counter()
    t_first = None
    t_end = None
    usage = {}
    chars_received = 0
    chunks_with_text = 0

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for chunk in parse_sse_stream(resp):
                now = time.perf_counter()

                chunk_usage = chunk.get("usage")
                if isinstance(chunk_usage, dict) and chunk_usage:
                    usage = chunk_usage

                text = extract_text_from_chunk(chunk)
                if text:
                    if t_first is None:
                        t_first = now
                    chars_received += len(text)
                    chunks_with_text += 1

            t_end = time.perf_counter()

    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")[:1500]
        print(f"  ERROR HTTP {e.code}: {e.reason}")
        print(f"  {body}")
        return None
    except Exception as e:
        print(f"  ERROR: {e}")
        return None

    if t_end is None:
        t_end = time.perf_counter()

    wall_time = t_end - t_start
    ttft = (t_first - t_start) if t_first is not None else wall_time

    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    gen_tokens = int(usage.get("completion_tokens") or 0)
    total_tokens = int(usage.get("total_tokens") or (prompt_tokens + gen_tokens))

    # Fallback estimates only if server omitted usage.
    # Kept clearly separate from measured values.
    usage_estimated = False
    if prompt_tokens <= 0:
        prompt_tokens = max(1, int(len(prompt.split()) * 1.3))
        usage_estimated = True

    if gen_tokens <= 0:
        # Rough text-token fallback. English/Russian/code vary significantly.
        gen_tokens = max(1, int(chars_received / 4))
        usage_estimated = True

    decode_time = max(t_end - (t_first if t_first is not None else t_start), 1e-9)

    # One streamed token is already represented at t_first. For long outputs the
    # difference between N and N-1 is negligible, but N-1 is more rigorous.
    decode_tokens_after_first = max(gen_tokens - 1, 0)
    gen_tps = decode_tokens_after_first / decode_time if decode_tokens_after_first > 0 else 0.0

    effective_prefill_tps = prompt_tokens / ttft if ttft > 0 else 0.0
    e2e_output_tps = gen_tokens / wall_time if wall_time > 0 else 0.0
    total_tps = (prompt_tokens + gen_tokens) / wall_time if wall_time > 0 else 0.0

    return {
        "name": name,
        "prompt_tokens": prompt_tokens,
        "gen_tokens": gen_tokens,
        "total_tokens": total_tokens,
        "ttft_ms": ttft * 1000.0,
        "wall_time": wall_time,
        "decode_time": decode_time,
        "effective_prefill_tps": effective_prefill_tps,
        "gen_tps": gen_tps,
        "e2e_output_tps": e2e_output_tps,
        "total_tps": total_tps,
        "chars_received": chars_received,
        "chunks_with_text": chunks_with_text,
        "usage_estimated": usage_estimated,
    }


def format_result(r):
    suffix = " [usage estimated]" if r["usage_estimated"] else ""
    return "\n".join([
        f"  Prompt:          {r['prompt_tokens']} tokens{suffix}",
        f"  Generated:       {r['gen_tokens']} tokens",
        f"  TTFT:            {r['ttft_ms']:.0f} ms",
        f"  Wall time:       {r['wall_time']:.3f} s",
        f"  Effective prefill:{r['effective_prefill_tps']:10.1f} tok/s",
        f"  Generation:      {r['gen_tps']:10.1f} tok/s",
        f"  E2E output:      {r['e2e_output_tps']:10.1f} tok/s",
        f"  Total throughput:{r['total_tps']:10.1f} tok/s",
    ])


def run_standard_tests(args):
    results = []

    for name, prompt, max_tokens in TESTS:
        print("\n" + "=" * 64)
        print(f"  {name}: -> {max_tokens} generated tokens")
        print("  Running...")

        r = run_test(
            args.url,
            args.model,
            name,
            prompt,
            max_tokens,
            timeout=args.timeout,
            temperature=args.temperature,
        )

        if r:
            print(format_result(r))
            results.append(r)
        else:
            print("  FAILED")

    return results


def run_long_tests(args):
    results = []

    prompt_sizes = args.long_context
    for prompt_size in prompt_sizes:
        name = f"ctx_{prompt_size}"
        prompt = generate_long_prompt(prompt_size)

        print("\n" + "=" * 64)
        print(f"  {name}: ~{prompt_size} prompt tokens -> {args.long_output} generated")
        print("  Running...")

        r = run_test(
            args.url,
            args.model,
            name,
            prompt,
            args.long_output,
            timeout=args.long_timeout,
            temperature=args.temperature,
        )

        if r:
            print(format_result(r))
            results.append(r)
        else:
            print("  FAILED")

    if not args.skip_long_generation:
        print("\n" + "=" * 64)
        print(f"  long_gen: short prompt -> {args.long_gen_tokens} generated tokens")
        print("  Running...")

        r = run_test(
            args.url,
            args.model,
            "long_gen",
            LONG_GEN_PROMPT,
            args.long_gen_tokens,
            timeout=args.long_timeout,
            temperature=args.temperature,
        )

        if r:
            print(format_result(r))
            results.append(r)
        else:
            print("  FAILED")

    return results


def print_summary(results):
    if not results:
        print("\nNo successful benchmark results.")
        return

    print("\n" + "=" * 100)
    print("  SUMMARY")
    print("=" * 100)
    print(
        f"  {'Test':<14} {'Prompt':>9} {'Gen':>8} {'TTFT':>10} "
        f"{'EffPrefill':>12} {'Gen/s':>10} {'E2E out/s':>11}"
    )
    print(
        f"  {'-'*14} {'-'*9} {'-'*8} {'-'*10} "
        f"{'-'*12} {'-'*10} {'-'*11}"
    )

    for r in results:
        print(
            f"  {r['name']:<14} "
            f"{r['prompt_tokens']:>9} "
            f"{r['gen_tokens']:>8} "
            f"{r['ttft_ms']:>8.0f}ms "
            f"{r['effective_prefill_tps']:>12.1f} "
            f"{r['gen_tps']:>10.1f} "
            f"{r['e2e_output_tps']:>11.1f}"
        )

    good_gen = [r["gen_tps"] for r in results if r["gen_tps"] > 0]
    good_prefill = [r["effective_prefill_tps"] for r in results if r["effective_prefill_tps"] > 0]

    if good_prefill:
        print(f"\n  Avg effective prefill: {sum(good_prefill)/len(good_prefill):.1f} tok/s")
    if good_gen:
        print(f"  Avg generation:        {sum(good_gen)/len(good_gen):.1f} tok/s")

    print("\n  NOTE:")
    print("  - Gen/s is the most useful number for direct generation-speed comparison.")
    print("  - TTFT is the most useful number for comparing long-prompt responsiveness.")
    print("  - EffPrefill = prompt_tokens / TTFT, so it includes queue/HTTP/first-token overhead.")
    print("  - Do not compare EffPrefill 1:1 with llama.cpp internal prompt_per_second.")
    print()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark a running vLLM OpenAI-compatible server."
    )
    parser.add_argument("--url", default=DEFAULT_SERVER, help="vLLM base URL")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="served model name")
    parser.add_argument(
        "--only",
        choices=("all", "standard", "long"),
        default="all",
        help="which benchmark group to run",
    )
    parser.add_argument(
        "--long-context",
        type=int,
        nargs="+",
        default=[15000, 64000, 128000],
        help="synthetic long-context prompt sizes",
    )
    parser.add_argument(
        "--long-output",
        type=int,
        default=1024,
        help="generated tokens for each long-context test",
    )
    parser.add_argument(
        "--long-gen-tokens",
        type=int,
        default=10000,
        help="output length for the dedicated long-generation test",
    )
    parser.add_argument(
        "--skip-long-generation",
        action="store_true",
        help="skip the 10k-token generation test",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="generation temperature; 0 is best for repeatable benchmark runs",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1200,
        help="timeout for standard tests in seconds",
    )
    parser.add_argument(
        "--long-timeout",
        type=int,
        default=3600,
        help="timeout for long tests in seconds",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 100)
    print("  vLLM OpenAI API benchmark")
    print(f"  Server: {args.url}")
    print(f"  Model:  {args.model}")
    print("=" * 100)

    results = []

    if args.only in ("all", "standard"):
        print("\n  Phase 1: Standard tests")
        results.extend(run_standard_tests(args))

    if args.only in ("all", "long"):
        print("\n  Phase 2: Long-context tests")
        results.extend(run_long_tests(args))

    print_summary(results)


if __name__ == "__main__":
    main()