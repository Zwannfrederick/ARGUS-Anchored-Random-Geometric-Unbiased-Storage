"""Reproducible benchmark for the ARGUS native runtime after the codec refactor.

Measures only what this machine can actually measure, and labels each number by
what kind of claim it supports:

* ``reconstruction`` — synthetic fidelity of a tier's compress/decompress
  round-trip on random tensors. **Not** a model-quality result.
* ``runtime`` — wall-clock latency and memory of ARGUS operations.
* ``downstream`` — end-to-end model behavior. Only produced when a real model
  is available; otherwise reported as not measured rather than inferred.

Every run records the full environment (GPU, CUDA, torch/triton versions,
seeds, shapes) so a number can be reproduced or invalidated later.

Usage::

    python benchmarks/bench_native_runtime.py
    python benchmarks/bench_native_runtime.py --json results.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from typing import Any, Dict, List

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from argus_cache.backends.eviction import ImportanceSortPolicy
from argus_cache.core.memory_manager import PagedDynamicKVCache
from argus_cache.core.tier_registry import PipelineConfig, TierSpec

SEED = 1234
TIERS = ["fp8", "int8", "int4", "int2", "one_bit", "jl"]


def environment() -> Dict[str, Any]:
    env: Dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "seed": SEED,
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        env.update(
            {
                "gpu": props.name,
                "gpu_total_mib": props.total_memory // (1024 * 1024),
                "cuda": torch.version.cuda,
                "capability": f"{props.major}.{props.minor}",
            }
        )
    try:
        import triton

        env["triton"] = triton.__version__
    except ImportError:
        env["triton"] = None
    return env


def _cache_for(tier: str, page_size: int, max_pages: int = 64) -> PagedDynamicKVCache:
    return PagedDynamicKVCache(
        pipeline=PipelineConfig(
            tiers=[TierSpec(name=tier, backend=tier, max_pages=max_pages)],
            eviction_policy=ImportanceSortPolicy(),
            page_size=page_size,
            sink_tokens=0,
            max_active_pages=1,
        )
    )


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def bench_codecs(
    page_size: int, head_dim: int, heads: int, repeats: int
) -> List[Dict[str, Any]]:
    """Per-tier compression cost, storage cost, and reconstruction fidelity."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        return []

    rows: List[Dict[str, Any]] = []
    for tier in TIERS:
        torch.manual_seed(SEED)
        cache = _cache_for(tier, page_size)

        k = torch.randn(1, heads, page_size, head_dim, dtype=torch.float16, device=device)
        v = torch.randn_like(k)
        fp16_bytes = k.numel() * k.element_size()

        # Demote: one page in, one filler page to push it down a tier.
        _sync()
        t0 = time.perf_counter()
        cache.push_new_tokens(k.clone(), v.clone())
        cache.push_new_tokens(torch.randn_like(k), torch.randn_like(v))
        _sync()
        compress_ms = (time.perf_counter() - t0) * 1000

        pages = cache.pages_by_tier[tier]
        if not pages:
            continue
        page = pages[0]
        stored_bytes = (
            page["key_q"].numel() * page["key_q"].element_size()
            if page["key_q"] is not None
            else fp16_bytes
        )

        # Decompress latency, averaged over repeats after a warmup.
        cache._cpp_manager.peek_decompress_page(page, tier)
        _sync()
        samples = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            k_out, _ = cache._cpp_manager.peek_decompress_page(page, tier)
            _sync()
            samples.append((time.perf_counter() - t0) * 1000)

        rel_err = ((k_out.float() - k.float()).norm() / k.float().norm()).item()
        cos = torch.nn.functional.cosine_similarity(
            k_out.float().flatten(), k.float().flatten(), dim=0
        ).item()

        rows.append(
            {
                "metric_class": "reconstruction+runtime",
                "tier": tier,
                "fp16_bytes": fp16_bytes,
                "stored_bytes": stored_bytes,
                "compression_ratio": round(fp16_bytes / max(stored_bytes, 1), 2),
                "effective_bits": round(16 * stored_bytes / fp16_bytes, 3),
                "compress_ms": round(compress_ms, 3),
                "decompress_ms_median": round(statistics.median(samples), 4),
                "decompress_ms_p90": round(sorted(samples)[int(0.9 * len(samples))], 4),
                "relative_l2_error": round(rel_err, 4),
                "cosine_similarity": round(cos, 4),
            }
        )
    return rows


def bench_verbose_overhead(page_size: int, head_dim: int, steps: int) -> Dict[str, Any]:
    """Cost of the hot-path logging that used to be unconditional."""
    if not torch.cuda.is_available():
        return {}

    results = {}
    for label, verbose in (("verbose_off", False), ("verbose_on", True)):
        torch.manual_seed(SEED)
        cache = _cache_for("int4", page_size)
        cache._cpp_manager.set_verbose(verbose)

        k = torch.randn(1, 4, page_size, head_dim, dtype=torch.float16, device="cuda")
        for _ in range(4):
            cache.push_new_tokens(torch.randn_like(k), torch.randn_like(k))

        q = torch.randn(1, 4, 1, head_dim, dtype=torch.float16, device="cuda")
        cache.inplace_paged_attention(q)
        _sync()

        t0 = time.perf_counter()
        for _ in range(steps):
            cache.inplace_paged_attention(q)
        _sync()
        results[label] = round((time.perf_counter() - t0) / steps * 1000, 4)

    if results.get("verbose_off"):
        results["logging_overhead_pct"] = round(
            100 * (results["verbose_on"] / results["verbose_off"] - 1), 1
        )
    results["metric_class"] = "runtime"
    return results


def bench_attention_scaling(head_dim: int, heads: int, steps: int) -> List[Dict[str, Any]]:
    """Decode-step latency and peak VRAM across context lengths."""
    if not torch.cuda.is_available():
        return []

    rows = []
    for page_size in (128, 256, 512):
        for n_pages in (2, 8, 16):
            torch.manual_seed(SEED)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            cache = _cache_for("int4", page_size)
            k = torch.randn(
                1, heads, page_size, head_dim, dtype=torch.float16, device="cuda"
            )
            for _ in range(n_pages):
                cache.push_new_tokens(torch.randn_like(k), torch.randn_like(k))

            q = torch.randn(1, heads, 1, head_dim, dtype=torch.float16, device="cuda")
            cache.inplace_paged_attention(q)
            _sync()

            t0 = time.perf_counter()
            for _ in range(steps):
                cache.inplace_paged_attention(q)
            _sync()
            per_step_ms = (time.perf_counter() - t0) / steps * 1000

            rows.append(
                {
                    "metric_class": "runtime",
                    "context_tokens": page_size * n_pages,
                    "page_size": page_size,
                    "pages": n_pages,
                    "decode_step_ms": round(per_step_ms, 4),
                    "peak_vram_mib": round(
                        torch.cuda.max_memory_allocated() / (1024 * 1024), 2
                    ),
                }
            )
            del cache
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--page-size", type=int, default=128)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--json", type=str, default=None)
    args = parser.parse_args()

    env = environment()
    print("=" * 78)
    print("ARGUS native runtime benchmark")
    print("=" * 78)
    for key, value in env.items():
        print(f"  {key:16s} {value}")

    if not env["cuda_available"]:
        print("\nCUDA unavailable — the native data plane cannot be benchmarked here.")
        return 1

    results: Dict[str, Any] = {"environment": env, "config": vars(args)}

    print("\n-- Per-tier codec (synthetic reconstruction + runtime) " + "-" * 22)
    codecs = bench_codecs(args.page_size, args.head_dim, args.heads, args.repeats)
    results["codecs"] = codecs
    print(
        f"  {'tier':8s} {'ratio':>7s} {'eff.bits':>9s} {'comp ms':>9s} "
        f"{'decomp ms':>10s} {'rel L2':>8s} {'cos':>7s}"
    )
    for row in codecs:
        print(
            f"  {row['tier']:8s} {row['compression_ratio']:6.2f}x "
            f"{row['effective_bits']:9.3f} {row['compress_ms']:9.3f} "
            f"{row['decompress_ms_median']:10.4f} {row['relative_l2_error']:8.4f} "
            f"{row['cosine_similarity']:7.4f}"
        )

    print("\n-- Hot-path logging overhead (runtime) " + "-" * 38)
    verbose = bench_verbose_overhead(args.page_size, args.head_dim, args.steps)
    results["logging"] = verbose
    if verbose:
        print(f"  verbose off : {verbose['verbose_off']:.4f} ms/step")
        print(f"  verbose on  : {verbose['verbose_on']:.4f} ms/step")
        print(f"  overhead    : {verbose.get('logging_overhead_pct')}%")

    print("\n-- Decode latency and peak VRAM vs context (runtime) " + "-" * 24)
    scaling = bench_attention_scaling(args.head_dim, args.heads, args.steps)
    results["scaling"] = scaling
    print(f"  {'ctx tokens':>11s} {'page':>6s} {'pages':>6s} {'ms/step':>9s} {'peak MiB':>9s}")
    for row in scaling:
        print(
            f"  {row['context_tokens']:11d} {row['page_size']:6d} {row['pages']:6d} "
            f"{row['decode_step_ms']:9.4f} {row['peak_vram_mib']:9.2f}"
        )

    results["downstream_quality"] = {
        "measured": False,
        "reason": (
            "Downstream accuracy (perplexity, NIAH/RULER retrieval) is not "
            "produced by this script. The reconstruction numbers above are "
            "synthetic fidelity on random tensors and must not be reported as "
            "model quality."
        ),
    }
    print(
        "\nNOTE: downstream model quality is NOT measured here. The fidelity\n"
        "      columns are synthetic reconstruction error on random tensors."
    )

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)
        print(f"\nWrote {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
