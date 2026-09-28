# M1: long filled contexts, stock against ARGUS tiers (2026-09-27)

Experiment M1 of the [v0.8 plan](../../../plans/argus-v0.8.0.md): the context actually
filled with prefilled tokens (filler text hiding one fact near the start), then 32 decoded
tokens that answer a question about it. Model: **Qwen3.6-35B-A3B** (UD-Q4_K_M, experts on
the CPU with `--cpu-moe`), f16 KV.

- Harness: [`benchmarks/bench_long_context.py`](../../../benchmarks/bench_long_context.py), one
  fresh server per run, one unmeasured 2048-token warmup request first (pages the model
  in), quiet desktop ([`benchmarks/quiet.sh`](../../../benchmarks/quiet.sh)).
- Configurations: [`configs.json`](configs.json). Common: `-ngl 99 --cpu-moe -fa on -np 1
  -b 2048 -ub 2048 --cache-ram 0`, 4 threads pinned to the 4 physical cores.
  - `stock-gpu-kv`: KV in VRAM (stock).
  - `stock-nkvo`: KV in RAM (stock `-nkvo`).
  - `k1-nkvo`: KV in RAM, K1 decode attention, prefill attention on the GPU (P2).
  - `tiers`: KV in pinned RAM, VRAM mirror sized automatically, split decode (S1), P2.
- Results: `results/qwen36-*.json` (every run with its command line, server log lines,
  memory peaks and coexistence record). Each run's path is proven by the server log
  (`ARGUS K1:` / `ARGUS KV tiers:` / `ARGUS S1:` lines); a run without them is an error.

## Decode (tokens per second, 32 tokens, context filled)

| context | stock GPU KV | stock `-nkvo` | K1 (`k1-nkvo`) | ARGUS tiers | tiers vs best stock |
|---|---:|---:|---:|---:|---:|
| 8K (probe) | 24.45 | 6.64 ¹ | 20.42 | 22.32 | 0.91x |
| 32K | **22.47** | 3.15 | 13.23 | 20.12 | 0.90x |
| 64K | **does not start** ² | 1.86 | 9.49 | **12.31** | **6.6x** |
| 128K | — | 1.02 ³ | 4.41 | **6.18** | **6.0x** |

¹ Earlier build (8K probe), before recurrent state stayed on the GPU under ARGUS.
² The context fails to allocate its prefill compute buffer (841 MB) next to 1.3 GB of KV.
³ Measured on 2026-09-28 on a freshly booted machine (`results/qwen36-128k-stock-nkvo.*`,
with a 1 s `vmstat` timeline): no swap traffic to speak of (33 pages in), iowait 0.04%. The
first attempt, on 2026-09-27, was stopped after 15 minutes because the machine was
thrashing (swap full, the model re-read from disk at ~300 MB/s). That was the machine's
accumulated state — it was the last run of the evening, with swap already filled by the
paused desktop apps — not a property of stock `-nkvo`: at 128K its host footprint (20.7 GB
mapped model, 2.56 GB KV, 592 MB compute) is within 0.1 GB of the ARGUS tiers run's.

Every run found the hidden fact (`needle_found`), including tiers at 128K.

- **Up to 32K the KV fits in VRAM** and stock GPU KV is the right choice; ARGUS tiers are
  within 10% of it with the whole context in the mirror (the gap is the ARGUS GPU decode
  kernel: 90–100 GB/s against stock flash attention's 178–185 GB/s, and CUDA graphs
  being disabled for ARGUS nodes).
- **From 64K stock cannot keep KV in VRAM.** Its only option is `-nkvo`, which is 6.6x
  slower than ARGUS tiers at 64K. The S1 gate (faster than both "all in RAM with K1" and
  stock `-nkvo` at 64K+ filled) is met: 12.31 against 9.49 and 1.86.
- Mirror sizes chosen by `ARGUS_KV_VRAM_BYTES=auto`: 32K fully mirrored; 64K 34,560 of
  65,536 cells (675 MiB); the rest ran on K1.

## Prefill

| context | stock GPU KV | stock `-nkvo` | ARGUS (P2) |
|---|---:|---:|---:|
| 32K | 111.6 s | 128.9 s | 113.8 s |
| 64K | — | 271.7 s | 242.2 s |
| 128K | — | 597.8 s | 540.3 s (K1 run: 544.8 s) |

**Prefill is dominated by expert weights, not KV.** With `--cpu-moe`, llama.cpp copies
every expert of every layer (≈ 21 GB) across PCIe (3.1 GB/s) for each prefill batch: ≈ 7 s
per 2048 tokens. 32K = 16 batches ≈ 112 s; 64K = 32 batches ≈ 224 s (+ 18 s attention);
128K = 64 batches ≈ 448 s (+ ~90 s attention). P2 removes stock `-nkvo`'s CPU attention
cost (29 s at 64K); it cannot touch the expert traffic. Next steps are in the handoff note.

## Coexistence (what the rest of the machine felt)

| run | swap in/out (pages) | major faults | iowait | probe late, p99 / max |
|---|---:|---:|---:|---:|
| tiers 64K | 0 / 256 | 390 | 0.08% | 0.60 / 5.2 ms |
| stock `-nkvo` 128K | 33 / 0 | 4179 | 0.04% | 0.49 / 8.5 ms |

(Every run's record is in its JSON: `coexistence`.) No sustained iowait in any run; the
probe thread asking to wake every 10 ms was late by under a millisecond at p99.

## Conditions and caveats

- Intel i5-11300H, turbo disabled (3.1 GHz cap), 31.9 GB RAM, RTX 3050 Ti Laptop 4 GB on
  PCIe Gen3 x4. Desktop paused with SIGSTOP during runs (list in the logs).
- One run per point (no repeats yet): these are single measurements.
- The 32K/64K `tiers` rows use the tiled GPU kernel (commit `8c18ad7`); the first 32K
  tiers run (19.65 tok/s, earlier kernel) is kept in `results/qwen36-32k.json`.
- DRAM traffic counters (`uncore_imc`) are not recorded: `perf_event_paranoid=2` and no
  root. Gemma 4 and UI-Mate are not measured yet.
