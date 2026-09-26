# ARGUS v0.8 — long context on a machine where the model fits and the context does not

## Goal

Run the longest possible context on one consumer machine by making GPU memory, RAM and
the model's own weights share the machine's bandwidth deliberately, with **light disk
use**: disk may hold KV, but never so much that it chokes the system. Speed matters,
capacity matters more.

**The machine stays usable while the model runs.** Most products are now built on LLMs,
and running one locally with a long context either fails for memory or makes the whole
computer crawl. ARGUS succeeds only if both hold: the long context runs, and the user's
other work (editor, browser, builds) does not feel the model running. That is a
measured property, not an impression: no swap activity, low iowait, CPU and RAM
headroom left to the rest of the system, and desktop responsiveness during the run.

The v0.7 question was "how close to stock at 4K"; at 4K the KV fits in VRAM and ARGUS
has nothing to win. The v0.8 question is:

> On this machine and this model, what is the longest **filled** context that decodes
> at a usable rate, and how fast is it compared with the best stock configuration?

Every result in this plan is recorded under `docs/measurements/v080-*` with its source
and the command that produced it. Numbers derived by arithmetic are labelled as
projections until a run replaces them.

## Machine and models

- i5-11300H (4 cores / 8 threads), 31.9 GB RAM, RTX 3050 Ti Laptop 4 GB on PCIe Gen3 x4.
- **Primary model: Qwen3.6-35B-A3B** (MoE, 22.65 GB; experts run on the CPU with
  `--cpu-moe`). It is the case where weights and KV compete for RAM bandwidth.
- **Secondary: UI-Mate-9B** (dense, 5.9 GB), the candidate for the neo agent.
- Gemma 4 (26B-A4B, E4B) are listed in the census; their global layers use head
  dim 512, which the ARGUS CUDA path does not support (≤ 256).

## What the census measured ([v080 census](../docs/measurements/v080-census-2026-09-26/README.md))

| link / kernel | rate |
|---|---:|
| GPU attention read (f16) | 184 GB/s |
| RAM streaming read (4 threads) | 42.2 GB/s |
| PCIe host → device | 3.14 GB/s |
| Stock CPU decode attention, Qwen3.6 layer (f16) | 4.9–5.2 GB/s of KV |

- **PCIe is 13x slower than RAM.** KV must not cross it; queries, partial results and
  page summaries may.
- **The stock CPU attention kernel reads each KV head once per query head** (8x for
  Qwen3.6), which saturates RAM with redundant reads. Quantized KV is no faster on the
  CPU for the same reason.
- **At 262K, KV bytes per token are 4.5–8.6x the active expert bytes.** Attention owns
  the RAM bandwidth at long context.

Projection for Qwen3.6 at 262K filled, attention per decoded token: stock `-nkvo`
≈ 1.09 s; split with the stock CPU kernel ≈ 0.88 s; split with a CPU kernel that reads
each KV head once ≈ 0.10 s (f16) or ≈ 0.05 s (q8_0 cold pages), plus ≈ 15 ms of experts.

## Principles

1. **Compute where the data is.** GPU pages are attended on the GPU, RAM pages on the
   CPU, in the same call; the two partial results are merged with log-sum-exp.
2. **Every byte read once per decoded token.** No per-query-head re-reads, no copies of
   KV across PCIe on the decode path.
3. **One memory plan for weights and KV.** VRAM and RAM bandwidth are budgets shared by
   expert weights and KV pages; placement follows measured marginal benefit.
4. **Disk-light.** Disk is a tier, never a heavy one:
   - no per-token disk reads on the decode path in the normal regime: KV that every
     decode step reads lives in VRAM or RAM, and capacity beyond RAM comes first from
     compression;
   - pages beyond RAM may live on disk only when attention rarely reads them (for
     example pages that E1 proves negligible);
   - writes are background, throttled and low priority (a bytes-per-second budget,
     idle I/O class);
   - KV of an idle session or a shared prefix is read once when the session resumes,
     and one read replaces a whole prefill. Evicting it demotes it (RAM, then the
     verified disk store of v0.5–v0.7) instead of dropping it.

   Every run records disk bytes read and written per decoded token and iowait; a run in
   which disk I/O stalls the system (sustained iowait) fails its gate.
   Production evidence for the split: the
   [Nebius/WEKA shared-KV benchmark](https://nebius.com/blog/posts/nebius-weka-shared-kv-cache-benchmark-hgx-b300)
   (8-hour agentic-coding replay, DeepSeek-V4-Pro, HGX B300) served 2.4x more requests
   with an NVMe tier under HBM, cache hits 40% → 93%, uncached input per request
   33,900 → 4,400 tokens. The gain came from not repeating prefill, not from faster
   decode.
5. **Two modes, never mixed silently.**
   - *Exact mode* (default): full softmax attention over every cell, KV stored at the
     precision llama.cpp was asked for. Each new path has a staged reference with the
     same block order and must be bit-exact with it, and within fp32 rounding of a
     float64 reference.
   - *Capacity mode*: cold pages stored at lower precision and/or pages skipped under a
     declared error bound. Quality is always measured and reported next to the speed.

## Experiments, in order

Each one: plan → measurement → accept or reject, recorded like v0.7's E-series.

**M1. Stock baseline with the context actually filled.**
Qwen3.6 and UI-Mate at 32K, 64K, 128K, 262K filled tokens: prefill time, decode tok/s,
VRAM and RAM peaks, for stock in-VRAM (while it fits), stock `-nkvo`, and stock with
quantized KV. `--cache-ram` is set explicitly in every run: its default (8 GiB of host
prompt cache) competes with 22.6 GB of Qwen3.6 weights for 31.9 GB of RAM. NVMe
sequential read (`O_DIRECT`) joins the census, for the session-store arithmetic. Every
run also records, in the same artifact: attention wall time per
decoded token, the number of populated KV cells, **physical DRAM bytes read** (memory
controller counters `uncore_imc_free_running_*/data_read`), CPU utilization, and the
**coexistence record**: swap-in/out and major faults (`vmstat`), iowait, free RAM
headroom, and the wake-up latency of a small periodic probe task running beside the
model (a stand-in for desktop responsiveness). Runs
only on a cleared machine (IDEs, Gradle, browsers, emulator closed): swap and reclaim
noise would contaminate the scaling curve. Also: free VRAM next to each model, and RAM bandwidth while experts are
read. Also records which v0.7 limits bite on these models (head dim 256, hybrid
linear-attention memory, sliding-window caches, F16-only CUDA path).
*This is the table every later experiment is judged against.*

**K1. CPU decode attention that reads each KV head once.**
One pass over a KV head serves all of its query heads (8 for Qwen3.6), block by block
with online softmax, f16 / q8_0 / q4_0. Gate: ≥ 30 GB/s effective KV read on the census
geometry at 131K cells (stock: 4.9–5.2), bit-exact with its staged reference, within fp32
rounding of float64. The evidence K1 must produce is not "the model got faster" but
**same KV → same attention result → far less DRAM traffic**: DRAM bytes read per
decoded token, measured with the same counters as M1, should fall toward one read of
the KV. Expected to matter on its own for stock-style `-nkvo` runs.

**Z1. RAM-bottomed store with a light disk tier.** The v0.7 store needs `ARGUS_KV_DIR`
and treats disk as the home of every page. v0.8 needs a store that works with RAM
(pinned or pageable) as its bottom tier and no backing file, keeping budgets, page
descriptors and verification of moves; when a disk directory is given, disk becomes an
optional lower tier under a write-rate budget and idle I/O priority.

**S1. Split attention.** GPU-resident pages on the GPU kernel, RAM pages on K1,
concurrently, merged by log-sum-exp; only q and the partial (output, max, sum) cross
PCIe. Gate: faster than both "all in RAM with K1" and stock `-nkvo` at 64K+ filled.

**B1. Bandwidth balancer.** Choose the GPU/RAM page split per step so both sides finish
together, measured under expert-weight traffic.

**P1. Weights–KV planner.** Decide per MB of VRAM whether it holds expert weights or
KV pages, from measured time saved per MB.

**C1. Capacity mode: compressed cold pages.** Recent pages at the requested precision,
older pages q8_0 → q4_0 (codecs already byte-compatible with llama.cpp). Quality gates:
needle-in-a-haystack at several depths and perplexity against the uncompressed run, on
the same filled contexts as M1.

**E1. Certified page skipping.** A small per-page key summary (per-dimension min/max)
lives on the GPU; from q it bounds the largest score any cell of a page can reach. Pages
whose combined softmax weight is provably below ε are not read, so the output error is
bounded by ε·max|v|. ε = 0 is exact mode. Related work to survey first: Quest,
InfiniGen, ShadowKV, MagicPIG, FastDecode.

**R1. Session and prefix reuse.** Stock llama.cpp at the pinned revision already has
a host-RAM prompt cache (`--cache-ram`, default 8 GiB), context checkpoints for hybrid
and sliding-window models (`--ctx-checkpoints`, 32 per slot; required for Qwen3.6's
linear-attention state) and manual slot save/restore to disk (`--slot-save-path`).
First measure those on a replayed agent session (neo, Claude Code): hit rate, prefilled
tokens per turn, time to first token, RAM they hold. Build only what is missing, most
likely a disk session tier that reuses the verified v0.5–v0.7 store, eviction that
demotes instead of drops, and one budget shared with the planner (P1).

## Success

- Qwen3.6 at 262K filled decodes several times faster than stock `-nkvo` (projection:
  ≈ 1 tok/s stock vs 5–10 tok/s), exact mode.
- The longest filled context that runs on this machine without heavy disk use, per
  model, in exact and capacity mode, with quality numbers for the latter and disk
  bytes and iowait for both.
- The machine stays usable during those runs: no swap activity, no sustained iowait,
  and probe wake-up latency close to the idle machine's.
- Every number reproducible from a script in `docs/measurements/v080-*`.

## Carried from the v0.7 backlog

- Plugin readiness: llama.cpp integration as its own package; a configuration API with
  environment variables as the default source; per-store budgets; devices other than
  CUDA device 0; gateway, dashboard and telemetry printing out of the core package.
- Runtime: multi-sequence KV (`-np > 1`); wait-free policy appends; a stream-ordered
  allocator; Q8/Q4 KV and head dim 512 on the CUDA path.
- Code health: `ggml_cuda_attention.cu` (1010), `manager.cpp` (1055) and
  `argus_cache/core/memory_manager.py` (1975) under the 1000-line ceiling; decide
  whether the HuggingFace store stays a research path.
