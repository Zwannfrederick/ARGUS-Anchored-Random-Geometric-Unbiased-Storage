"""Isolated backend benchmarks and profiler gate for ARGUS Direct Paged Attention (Stage S7).

Measures:
- Latency per token (min, median, p95, max)
- VRAM footprint and peak transient allocation
- Throughput (tokens/sec) across context lengths (1K, 4K, 8K, 16K, 32K)
- Comparison across ACTIVE_FP16, GGML_Q8_0, GGML_Q4_0, and Full Reconstruction SDPA
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Any, Dict, List

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from argus_cache.core.page_table import CodecKind, PlacementLocation, StructureOfArraysPageTable
from argus_cache.core.backend_pool import ContiguousBlockPool
from argus_cache.core.direct_attention import DirectPagedAttentionEngine


import gc


@torch.inference_mode()
def benchmark_single_token_decode(
    codec: CodecKind,
    context_tokens: int,
    page_size: int = 128,
    q_heads: int = 24,
    kv_heads: int = 4,
    head_dim: int = 256,
    num_repeats: int = 100,
    warmup: int = 10,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Benchmarks single-token decode latency and memory under specified context size and tier."""
    gc.collect()
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    num_pages = math.ceil(context_tokens / page_size)
    batch = 1
    base_vram = torch.cuda.memory_allocated() if (device == "cuda" and torch.cuda.is_available()) else 0

    # Build page table
    pt = StructureOfArraysPageTable(capacity=max(num_pages, 8), device=device)
    active_pages = {}
    pools = {}

    if codec in (CodecKind.GGML_Q8_0, CodecKind.GGML_Q4_0):
        pool = ContiguousBlockPool(
            codec=codec,
            placement=PlacementLocation.GPU_DEVICE,
            max_slots=num_pages,
            page_size=page_size,
            num_heads=kv_heads,
            head_dim=head_dim,
            device=device,
        )
        pools[codec] = pool

        for p in range(num_pages):
            slot = pool.allocate_slot()
            tokens_in_page = min(page_size, context_tokens - p * page_size)
            # Fill dummy bytes
            pt.allocate_page(
                page_id=p,
                logical_pos=p * page_size,
                token_count=tokens_in_page,
                codec=codec,
                pool_slot=slot,
            )
    else:
        # ACTIVE FP16
        for p in range(num_pages):
            tokens_in_page = min(page_size, context_tokens - p * page_size)
            k = torch.randn(batch, kv_heads, tokens_in_page, head_dim, dtype=torch.float16, device=device)
            v = torch.randn(batch, kv_heads, tokens_in_page, head_dim, dtype=torch.float16, device=device)
            active_pages[p] = (k, v)
            pt.allocate_page(
                page_id=p,
                logical_pos=p * page_size,
                token_count=tokens_in_page,
                codec=CodecKind.ACTIVE_FP16,
            )

    query = torch.randn(batch, q_heads, 1, head_dim, dtype=torch.float16, device=device)

    # Warmup
    for _ in range(warmup):
        _ = DirectPagedAttentionEngine.decode_single_token(
            query=query,
            page_table=pt,
            pools=pools,
            active_pages=active_pages,
        )

    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()

    # Timing iterations
    latencies: List[float] = []
    for _ in range(num_repeats):
        start = time.perf_counter()
        _ = DirectPagedAttentionEngine.decode_single_token(
            query=query,
            page_table=pt,
            pools=pools,
            active_pages=active_pages,
        )
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - start) * 1000.0) # ms

    latencies.sort()
    median_ms = latencies[len(latencies) // 2]
    min_ms = latencies[0]
    max_ms = latencies[-1]
    p95_ms = latencies[int(len(latencies) * 0.95)]

    peak_vram = torch.cuda.max_memory_allocated() if (device == "cuda" and torch.cuda.is_available()) else 0
    live_vram = torch.cuda.memory_allocated() if (device == "cuda" and torch.cuda.is_available()) else 0

    return {
        "codec": codec.name,
        "context_tokens": context_tokens,
        "num_pages": num_pages,
        "latency_median_ms": round(median_ms, 4),
        "latency_p95_ms": round(p95_ms, 4),
        "latency_min_ms": round(min_ms, 4),
        "latency_max_ms": round(max_ms, 4),
        "throughput_tps": round(1000.0 / median_ms, 2) if median_ms > 0 else 0.0,
        "live_vram_mib": round(live_vram / (1024 * 1024), 2),
        "peak_vram_mib": round(peak_vram / (1024 * 1024), 2),
    }


def run_benchmarks(output_file: str = "docs/measurements/v040-fused-attention-benchmark.json") -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running ARGUS S7 Fused Attention Benchmarks on {device.upper()}...")
    contexts = [1024, 4096, 8192, 16384, 32768]
    codecs = [CodecKind.ACTIVE_FP16, CodecKind.GGML_Q8_0, CodecKind.GGML_Q4_0]

    results = []
    for ctx in contexts:
        for cod in codecs:
            print(f"Profiling {cod.name:14s} @ {ctx:5d} tokens...", end="", flush=True)
            res = benchmark_single_token_decode(
                codec=cod,
                context_tokens=ctx,
                page_size=128,
                q_heads=24,
                kv_heads=4,
                head_dim=256,
                num_repeats=30,
                warmup=5,
                device=device,
            )
            print(f" Median: {res['latency_median_ms']:7.3f} ms | VRAM: {res['live_vram_mib']:6.2f} MiB | {res['throughput_tps']:6.1f} tok/s")
            results.append(res)

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": {
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "cuda_available": torch.cuda.is_available(),
        },
        "model_geometry": {
            "q_heads": 24,
            "kv_heads": 4,
            "head_dim": 256,
            "page_size": 128,
        },
        "benchmarks": results,
    }

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nBenchmark suite finished successfully! Results saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="docs/measurements/v040-fused-attention-benchmark.json")
    args = parser.parse_args()
    run_benchmarks(args.output)
