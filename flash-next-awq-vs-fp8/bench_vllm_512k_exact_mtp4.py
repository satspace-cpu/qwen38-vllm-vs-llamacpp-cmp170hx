#!/usr/bin/env python3
"""
vLLM OpenAI-compatible API benchmark — 512K EXACT COLD context / MTP=4 profile.

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
  python3 bench_vllm_512k_exact_mtp4.py
  python3 bench_vllm_512k_exact_mtp4.py --url http://127.0.0.1:8000 --model Qwen3.8-Flash-Next-FP8
  python3 bench_vllm_512k_exact_mtp4.py --only standard
  python3 bench_vllm_512k_exact_mtp4.py --only long
  python3 bench_vllm_512k_exact_mtp4.py --long-context 64000
"""

import argparse
import json
import sys
import time
import urllib.request
import urllib.error

DEFAULT_SERVER = "http://127.0.0.1:8000"
DEFAULT_MODEL = "Qwen3.8-Flash-Next-FP8"


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


def _tokenize_count(server: str, model: str, prompt: str, timeout: int = 3600) -> int:
    url = server.rstrip("/") + "/tokenize"
    payload = {"model": model, "prompt": prompt}
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    if isinstance(data.get("count"), int):
        return data["count"]
    if isinstance(data.get("tokens"), list):
        return len(data["tokens"])
    if isinstance(data.get("token_ids"), list):
        return len(data["token_ids"])
    raise RuntimeError(f"Unexpected /tokenize response: {str(data)[:500]}")


def _make_cold_prompt_words(word_count: int, test_id: str) -> str:
    marker = (
        f"COLD_CONTEXT_BENCHMARK_{test_id}_UNIQUE_PREFIX. "
        f"This prompt belongs only to test {test_id}. "
    )
    paragraph = (
        "The quick brown fox jumps over the lazy dog near the riverbank where ancient oaks "
        "stand tall against the evening sky. Researchers document migratory birds using GPS "
        "tracking, acoustic monitoring, spectrogram analysis, neural networks, atmospheric "
        "measurements, magnetic field observations, ecological surveys, and statistical models. "
        "Computer architecture, distributed inference, memory bandwidth, attention mechanisms, "
        "quantization, tensor parallelism, speculative decoding, and long context processing are "
        "described with technical detail for repeatable benchmark input. "
    )
    words = paragraph.split()
    body = (words * (word_count // len(words) + 1))[:word_count]
    return marker + " ".join(body)


def generate_long_prompt_exact(
    server: str,
    model: str,
    target_tokens: int,
    test_id: str,
    timeout: int = 3600,
    tolerance: int = 8,
) -> tuple[str, int]:
    word_count = max(1, int(target_tokens / 1.15))
    best_prompt = None
    best_count = None
    best_err = 10**18

    for _ in range(8):
        prompt = _make_cold_prompt_words(word_count, test_id)
        count = _tokenize_count(server, model, prompt, timeout=timeout)
        err = abs(count - target_tokens)
        if err < best_err:
            best_prompt, best_count, best_err = prompt, count, err
        if err <= tolerance:
            return prompt, count

        scale = target_tokens / max(count, 1)
        new_word_count = max(1, int(round(word_count * scale)))
        if new_word_count == word_count:
            new_word_count += 1 if count < target_tokens else -1
            new_word_count = max(1, new_word_count)
        word_count = new_word_count

    direction = 1 if best_count < target_tokens else -1
    wc = word_count
    for _ in range(24):
        wc = max(1, wc + direction)
        prompt = _make_cold_prompt_words(wc, test_id)
        count = _tokenize_count(server, model, prompt, timeout=timeout)
        err = abs(count - target_tokens)
        if err < best_err:
            best_prompt, best_count, best_err = prompt, count, err
        if err <= tolerance:
            return prompt, count
        if (direction == 1 and count > target_tokens) or (direction == -1 and count < target_tokens):
            direction *= -1

    return best_prompt, best_count


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
    choices = chunk.get("choices") or []
    if not choices:
        return ""

    choice = choices[0] or {}

    text = choice.get("text")
    if isinstance(text, str):
        return text

    delta = choice.get("delta") or {}
    content = delta.get("content")
    if isinstance(content, str):
        return content

    return ""


def run_test(server, model, name, prompt, max_tokens, timeout=1200, temperature=0.0):
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

    usage_estimated = False
    if prompt_tokens <= 0:
        prompt_tokens = max(1, int(len(prompt.split()) * 1.3))
        usage_estimated = True

    if gen_tokens <= 0:
        gen_tokens = max(1, int(chars_received / 4))
        usage_estimated = True

    decode_time = max(t_end - (t_first if t_first is not None else t_start), 1e-9)
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

        print("\n" + "=" * 64)
        print(f"  {name}: target {prompt_size} REAL prompt tokens -> {args.long_output} generated")
        print("  Building unique cold prompt and measuring it with /tokenize...")

        try:
            prompt, measured_prompt_tokens = generate_long_prompt_exact(
                args.url,
                args.model,
                prompt_size,
                test_id=name,
                timeout=args.long_timeout,
                tolerance=args.token_tolerance,
            )
        except Exception as e:
            print(f"  TOKENIZE ERROR: {e}")
            print("  FAILED")
            continue

        print(f"  /tokenize measured: {measured_prompt_tokens} tokens "
              f"(target {prompt_size}, delta {measured_prompt_tokens - prompt_size:+d})")
        print("  Running inference...")

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
    print("  - EXACT COLD profile measures prompt size through /tokenize and uses a unique prefix per test.")
    print("  - MTP is configured server-side; 520K + 512 output leaves headroom below max_model_len=524288.")
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
        default=[15000, 64000, 128000, 262000, 400000, 500000, 520000],
        help="synthetic long-context target sizes for the 512K profile",
    )
    parser.add_argument(
        "--long-output",
        type=int,
        default=512,
        help="generated tokens for each long-context test; 512 leaves safe headroom at ~520K prompt with MTP=4",
    )
    parser.add_argument(
        "--token-tolerance",
        type=int,
        default=8,
        help="allowed difference between requested and /tokenize-measured prompt tokens",
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
    print("  vLLM OpenAI API benchmark — 512K EXACT COLD / MTP=4")
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
