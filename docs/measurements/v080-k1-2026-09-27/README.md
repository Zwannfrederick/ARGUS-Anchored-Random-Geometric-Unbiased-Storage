# K1: CPU decode attention that reads each KV head once (2026-09-27)

Experiment K1 of the [v0.8 plan](../../../plans/argus-v0.8.0.md). The
[census](../v080-census-2026-09-26/README.md) found that stock GGML CPU flash attention
reads every KV head once per query head: 8 times for Qwen3.6 and the Gemma 4 global layers.
K1 serves every query head of a KV head from one pass over its cells.

- Source: [`argus_cache/csrc/ggml_decode_attention.cpp`](../../../argus_cache/csrc/ggml_decode_attention.cpp),
  SIMD layer [`ggml_decode_simd.h`](../../../argus_cache/csrc/ggml_decode_simd.h).
- llama.cpp hook: [`integrations/llama.cpp/v08-kv.patch`](../../../integrations/llama.cpp/v08-kv.patch),
  `ARGUS_KV_CPU_ATTENTION=k1` with `-nkvo`.
- Reproduce: `bash run.sh <llama.cpp dir with build-cuda> <output dir> [repeats]` (≈ 15 minutes).
  - [`results/accuracy.txt`](results/accuracy.txt): accuracy against float64 and stock GGML.
  - `results/k1_bw_4t_run{1,2,3}.json`: timings, same binary and machine state, three runs.
  - [`results/machine.txt`](results/machine.txt): commit, CPU state, paused processes.

## How it works

- **A task is a chunk of cells across every KV head.** Each thread streams contiguous KV.
  Chunks depend only on the number of cells, so the result is bit-identical whatever the
  thread count (the test checks 1, 2, 3, 4 and 8 threads).
- **Per block of 32 cells and KV head**, the scores of all the rows sharing that head (8
  for these models) come from one pass over K, and their value sums from one pass over V.
  F16, Q8_0 and Q4_0 rows are decoded inside the FMA loops, so each element is decoded once
  per call. Other KV types are dequantized once per block.
- **AVX-512 or AVX2**, chosen at build time, with a scalar fallback. Everything is
  accumulated in F32; blocks merge by online softmax, chunks by log-sum-exp in a fixed
  order.
- **Output is a partial** (max, sum, unnormalized output) per row, the same format the
  GPU side of split attention (S1) produces.

## Accuracy

Maximum error relative to the largest |V|, against a float64 reference computed from the
same (dequantized) K/V ([`accuracy.txt`](results/accuracy.txt)):

| case | K1 | stock GGML |
|---|---:|---:|
| Qwen3.6 layer, f16, 20K cells | 1.6e-7 | 7.9e-4 |
| Qwen3.6 layer, q8_0 / q4_0 | 3.9e-8 / 5.0e-8 | 6.8e-4 / 6.8e-4 |
| Gemma 4 global layer, f16, head dim 512 | 7.1e-8 | 1.1e-3 |
| UI-Mate, 3 draft tokens, causal mask | 1.3e-7 | 1.6e-3 |

**K1 is about 5000 times closer to float64 than stock.** Stock rounds the query to F16 for
F16 K, and accumulates the value sum in F16 when V is F16. K1 keeps both in F32. The two
cases where stock matches (F32 KV, one cell) are the cases where it does neither.

## Speed

One full-attention layer, one query token, 4 threads pinned to the 4 physical cores,
median of 10 calls per run; the medians of three runs, with the range across runs.

| model (geometry) | KV | 16K cells | 64K | 131K | K1 at 131K | speed-up |
|---|---|---:|---:|---:|---:|---:|
| Qwen3.6 (16 q / 2 kv, d 256) | f16 | 10.71 → 1.39 ms | 48.70 → 5.17 | 98.40 → **10.28** | 26.1 GB/s | **9.6x** |
| | q8_0 | 8.05 → 1.34 | 32.30 → 5.22 | 64.43 → 10.36 | 13.8 GB/s | 6.2x |
| | q4_0 | 7.37 → 1.37 | 30.02 → 5.36 | 60.03 → 10.70 | 7.1 GB/s | 5.6x |
| Gemma 4 global (16 / 2, d 512) | f16 | 22.44 → 2.83 | 94.97 → 10.50 | 193.39 → **20.86** | 25.7 GB/s | **9.3x** |
| | q8_0 | 14.84 → 2.65 | 59.13 → 10.16 | 118.50 → 20.07 | 14.2 GB/s | 5.9x |
| | q4_0 | 12.93 → 2.63 | 51.50 → 10.17 | 102.99 → 20.22 | 7.5 GB/s | 5.1x |
| UI-Mate (16 / 4, d 256) | f16 | 18.34 → 2.02 | 77.51 → 7.56 | 154.58 → **14.80** | 36.3 GB/s | **10.4x** |
| | q8_0 | 11.24 → 1.87 | 43.50 → 7.16 | 88.66 → 14.10 | 20.2 GB/s | 6.3x |
| | q4_0 | 9.00 → 1.87 | 36.13 → 7.25 | 72.21 → 14.43 | 10.5 GB/s | 5.0x |

For Qwen3.6 at 131K cells this is the attention of 10 layers per decoded token falling
from ≈ 0.98 s to ≈ 0.10 s (arithmetic from the per-layer time).

End to end on a real model (Qwen2.5-0.5B, 5.3K-token prompt, `-nkvo`, greedy): decode
32.2 → 112.8 tok/s with the same 48 generated tokens.

## The gate, and why it is not met on this configuration

The plan's gate is ≥ 30 GB/s effective KV read at 131K cells on the census geometry
(Qwen3.6). K1 reaches **26.1 GB/s**.

- **K1 is compute-bound here, not memory-bound.** RAM streams 41.8 GB/s with 4 threads
  (re-measured today). One f16 cell of one KV head is 1 KiB and costs 4096 multiply-adds
  for 8 query heads (8 × 256 for the scores, 8 × 256 for the values). At 16 FMA lanes per
  cycle per core and 3.1 GHz, four cores peak at 49.6 GB/s; K1 runs at 53% of that.
- **Two consequences confirm it.**
  - With half the multiply-adds per byte (UI-Mate, 4 query heads per KV head), K1 reaches
    36.3 GB/s.
  - Q8_0 and Q4_0 KV take the same time per element as F16 (10.3–10.7 ms at 131K for
    Qwen3.6). On this CPU, quantized KV buys capacity, not decode speed.
- **Turbo was disabled** (`intel_pstate/no_turbo = 1`, 3.1 GHz cap) and cannot be changed
  without root. The stock kernel ran about 2x faster in the census (54 ms against 98 ms at
  131K), which points to different clock conditions then. A compute-bound kernel scales
  with the clock; the gate should be re-checked with turbo on.
- **What would reduce the instruction count further:** a cell-major score layout that
  removes the per-cell scatter of 8 scores and vectorizes the softmax across rows
  (estimated ~10%, not yet tried). Using fewer multiply-adds would require lower
  precision, which belongs to capacity mode.

## Findings on the way (each measured, then acted on)

| hypothesis | test | result |
|---|---|---|
| A block conversion pass and per-cell library calls cost more than the FMAs | rdtsc phase split, single thread | Conversion 34%, softmax 17% of 800 cycles per cell-head. Fused decode and a vectorized mask cut it to 397 |
| The L2 streamer does not cross 4 KiB pages (4 cells of Qwen3.6 KV) | software prefetch one block ahead | DRAM-resident 544 → 464 cycles; the gap to cache-resident fell from 150 to 57 |
| AVX-512 beats AVX2 on this core despite equal FMA width | same code built both ways | AVX2 is 25% slower: it needs one load per FMA, AVX-512 half as many |
| Two cells or two value chunks per step reuse loads | register-paired kernels | Rejected: 7–11% slower (register pressure); reverted |
| Unpinned threads share physical cores | `taskset -c 0-3` | Unpinned 4-thread runs were bimodal (10.2 or 18 ms); pinned runs repeat within 1–3%. llama.cpp runs use `--cpu-mask 0xF --cpu-strict 1` |

## Conditions

Intel i5-11300H (4 cores / 8 threads, AVX-512), turbo disabled, 31.9 GB RAM. The desktop
was quieted by [`benchmarks/quiet.sh`](../../../benchmarks/quiet.sh) for each run (IDE,
language servers, indexers, chat apps paused with SIGSTOP and resumed after; the list is
in `machine.txt`). `dirty=1` in `machine.txt` is the results directory being written.
