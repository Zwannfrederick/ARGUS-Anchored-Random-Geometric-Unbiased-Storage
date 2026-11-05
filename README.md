# ARGUS (`libargus`)

**An autonomous C++/CUDA KV-cache and MoE bandwidth orchestrator for LLM runtimes.
Instead of pinning the KV cache and expert weights to a single device, `libargus`
orchestrates them across a phase-shared hierarchy — GPU VRAM, pinned RAM, pageable
RAM, and verified `O_DIRECT` disk — so long contexts and large MoE models can run
within tight VRAM budgets while outperforming host-offloaded baselines in both
prefill and decode.**

ARGUS is not an inference server. It is the systems layer underneath one (`llama.cpp`
/ GGML), owning where each KV cell or page lives, how it is written, how prefill
streams expert weights across PCIe, and how decode attention reads across GPU and
CPU tiers at the same time.

**Status: v0.8.0.** Measured results on an RTX 3050 Ti Laptop (4 GB VRAM, PCIe Gen3 x4,
4-core Intel i5-11300H):
- **Heterogeneous CPU/GPU Execution Scheduler & Adaptive RAM/VRAM Memory Manager (`ARGUS_AUTO=1`):**
  On `Qwen3.6-35B-A3B` (`Q4_K_M`, 4 GB VRAM + 32 GB RAM), the heterogeneous scheduler (`ARGUS_HETERO_SCHED=1`, `ARGUS_HETERO_OVERLAP=1`) eliminates `150 MiB` of dead MoE compute-buffer reservations, bounds P3 graph reservation to a 2-layer envelope (`u_batch=512`, `n_outputs_max=16`), eliminates 480–960 redundant Gated DeltaNet graph rebuilds and `34.23–132.87 GiB` of unused `kq_mask` CPU writes, reclaims `1,521.9 MiB` of non-active GGUF host pages, and overlaps GPU `ffn_shexp` with CPU `RoutedExperts` via asynchronous batched PCIe transfers (`-48.9%` split sync latency). At **32K**, `CUDA0` compute buffer drops by **80.0%–85.0%** (`879.3 MiB` / `658.1 MiB` → **`132.03 MiB`**, saving **`524.0 MiB` VRAM**), prefill reaches **`431.93 tok/s`** (`73.49 s`, **`+49.1%`** over prior baseline), and decode doubles (`2.01x`, `p50` cut from `516.29 ms` to `259.85 ms`) under constrained RAM pressure while reaching **`14.50–21.93 tok/s`** (`45.60 ms/tok`) in warm RAM. At **64K (`65,536`)**, 100% of KV cells (`1,280.0 MiB`) fit in the `S1` GPU mirror on 4 GB VRAM (`376.87 tok/s` prefill, `5.29 tok/s` median decode, `3/3` multi-needle accuracy across 8K, 32K, and 64K) ([Hetero report](docs/measurements/v080-libargus-2026-10-27/hetero_architecture_report.md)).
- **35B MoE at 32K–128K filled context (`Qwen3.6-35B-A3B` Q4_K_M, `--cpu-moe`):**
  At 64K and 128K context, stock GPU KV cannot start (VRAM OOM). Against stock
  `-nkvo`, `libargus` (`S1` split decode + `K1` CPU attention + `P2` GPU prefill)
  delivers **6.6x faster decode at 64K** (`12.31` vs `1.86 tok/s`) and **6.0x faster
  decode at 128K** (`6.18` vs `1.02 tok/s`), while `P3` layer-major prefill cuts
  32K/64K prefill time by **1.93x / 1.86x** (`114.3 s → 59.1 s` at 32K; `242.2 s →
  130.4 s` at 64K) by streaming each layer's MoE experts across PCIe once per prompt
  instead of once per microbatch ([M1](docs/measurements/v080-m1-2026-09-27/README.md),
  [P3](docs/measurements/v080-prefill-2026-09-28/README.md)).
- **Mixed-Precision Tiers (`C1`), Multipole / V-Centroid Bounded-Mass Skipping (`E1`), & Sparse MoE `MADV_RANDOM` (`ARGUS_AUTO=1`):**
  On 8K filled context (`Qwen2.5-0.5B-Instruct` Q4_K_M), default `ARGUS_AUTO=1` with
  asymmetric `K=q8_0, V=q4_0` host KV (**-59.4% host KV RAM** + `f16` GPU mirror) cuts
  prefill time to **0.815 s (9,936 tok/s, 1.10x faster than stock)** and accelerates
  decode from **23.18 tok/s** (stock `-nkvo`) to **175.45 tok/s (7.57x)** while cutting
  peak RSS by **75.0 MiB**. Under a strict 16 MiB split mirror (1280 GPU + 6912 host
  cells), `ARGUS_AUTO=1` reaches **79.54 tok/s (3.43x)** on `f16` and **69.60 tok/s
  (3.00x)** on `K=q8_0, V=q4_0`, retrieving the hidden 6-digit needle in every mode.
  In standalone 16K–128K synthetic kernel benchmarks (without QK-Norm), `K1`
  exact is **9.9x–10.1x faster** than stock GGML CPU FlashAttention (`5.79 ms` vs
  `57.43 ms` at 64K; `11.35 ms` vs `114.72 ms` at 128K, with `q8_0` reaching `10.95 ms`),
  and `K1 + E1` bounded-mass block skipping (`ARGUS_KV_SKIP_EPS=1e-4`) reaches **0.91 ms
  (62.8x faster)** at 64K `f16` and **2.56 ms (44.9x faster)** at 128K `f16`/`q4_0`
  (**44.1x faster than stock `f16` with 47% less RAM** on `q8_0`) at `< 6.1e-7` relative
  error ([v0.8 levers](docs/measurements/v080-libargus-2026-10-27/README.md)).
- **Direct-disk paged KV (v0.7 path, 4K context, `Qwen2.5-0.5B`):**
  Prefill is **1.29x** stock llama.cpp's time with all KV on GPU and **1.36x** with the
  placement policy managing GPU/pinned/RAM/disk tiers; decode is **55.6** and
  **48.2 tok/s** vs stock's **37.5 tok/s** ([v0.7 baseline](docs/measurements/v070-e10-2026-09-26/)).

[![packaging](https://github.com/Zwannfrederick/ARGUS-Anchored-Random-Geometric-Unbiased-Storage/actions/workflows/packaging.yml/badge.svg)](https://github.com/Zwannfrederick/ARGUS-Anchored-Random-Geometric-Unbiased-Storage/actions/workflows/packaging.yml)

Türkçe belge: [README_TR.md](README_TR.md)

---

## Contents

- [Status at a glance](#status-at-a-glance)
- [What ARGUS is and is not](#what-argus-is-and-is-not)
- [Architecture: The Five Levers of `libargus`](#architecture-the-five-levers-of-libargus)
- [Using `libargus` with llama.cpp](#using-libargus-with-llamacpp) — build, autonomous mode, tiers, disk modes, every setting
- [Local stack: ARGUS Studio and the services](#local-stack-argus-studio-and-the-services)
- [v0.8 results: Long-context tiers, MoE prefill, C1 & E1](#v08-results-long-context-tiers-moe-prefill-c1--e1)
- [v0.7 results: Direct-disk paged KV](#v07-results-direct-disk-paged-kv)
- [Reading the stock comparison](#reading-the-stock-comparison)
- [Using ARGUS with HuggingFace (legacy research path)](#using-argus-with-huggingface-legacy-research-path)
- [Benchmark methodology](#benchmark-methodology)
- [Known limits](#known-limits)
- [Earlier evidence (v0.4–v0.6)](#earlier-evidence-v04v06)
- [Roadmap](#roadmap)

---

## Status at a glance

Nothing here is a projection. Every number is a recorded run you can open; every
behaviour is backed by a test that fails if it stops being true; anything that has
not been measured says **Not measured**.

| Claim | Status | Evidence |
|---|---|---|
| `ARGUS_AUTO=1` autonomously enables K1 CPU decode, S1 VRAM mirrors, P2 GPU prefill, P3 layer-major prefill, `MADV_RANDOM` on sparse MoE tensors, and the Heterogeneous CPU/GPU Execution Scheduler & Adaptive RAM/VRAM Manager (`ARGUS_HETERO_SCHED=1`, `ARGUS_HETERO_OVERLAP=1`), while keeping E1, Multipole, Hot-Expert Cache, and Top-$r$ opt-in | Proven | [`argus_config.h`](argus_cache/csrc/argus_config.h), [`argus_hetero.h`](argus_cache/csrc/argus_hetero.h), [`hetero_architecture_report.md`](docs/measurements/v080-libargus-2026-10-27/hetero_architecture_report.md) |
| K1 CPU decode attention (`f32`/`f16`/`q8_0`/`q4_0`) is within `2e-6` of `float64` reference and bit-identical across `{1,2,3,4,8}` threads | Proven (10 geometries, SIMD + scalar; `< 3e-7` isolated error across all 10 full-attention layers of `Qwen3.6-35B-A3B`) | [`test_ggml_decode_attention.cpp`](tests/cpp/test_ggml_decode_attention.cpp), [`v0.8 report`](docs/measurements/v080-libargus-2026-10-27/README.md) |
| S1 split decode attention (GPU `f16` mirror + CPU `K1` on `f16`/`q8_0`/`q4_0` host KV) matches `float64` reference within `2e-6` | Proven (7 `f16` + 2 `C1` mixed-precision cases) | [`test_ggml_split_attention.cpp`](tests/cpp/test_ggml_split_attention.cpp) |
| E1 + Multipole bounded-mass approximate skipping (`ARGUS_KV_SKIP_EPS=1e-4`, `ARGUS_MULTIPOLE=1`) prunes `12.0%–12.6%` K-blocks and `34.8%–60.2%` V-cells on real `Qwen3.6-35B-A3B` while reducing attention output error `2.4x–11.4x`, 8K mean KL divergence vs `E1 OFF` by `40%` (`2.50e-5`), and 32K max logit error by `2.79x` (`1.064` vs `2.969`) | Proven (8K & 32K 8-mode ablation on `Qwen3.6-35B-A3B`, `3/3` needles; synthetic non-QK-Norm kernel benchmark reaches `62.8x` at 64K `f16`) | [`real_a3b_validation.json`](docs/measurements/v080-libargus-2026-10-27/real_a3b_validation.json), [`README.md`](docs/measurements/v080-libargus-2026-10-27/README.md) |
| C1 mixed-precision tiers (`q8_0`, `q4_0`, or asymmetric `K=q8_0, V=q4_0` in host RAM + `f16` GPU mirror) cut host KV RAM by 47%–72% while accelerating decode 2.9x–7.6x over stock `-nkvo` | Proven at 8K on `Qwen2.5-0.5B` (`needle_found: true` on `f16`, `q8_0`, and `K=q8_0/V=q4_0`; `0.815 s` prefill and `175.45 tok/s` decode with `auto` mirror) | [`bench_e2e_result.json`](docs/measurements/v080-libargus-2026-10-27/README.md) |
| 64K & 128K decode on `Qwen3.6-35B-A3B` (`--cpu-moe`) where stock GPU KV OOMs | Proven: **6.6x faster** than stock `-nkvo` at 64K (`12.31` vs `1.86 tok/s`), **6.0x faster** at 128K (`6.18` vs `1.02 tok/s`), needle found | [M1 report](docs/measurements/v080-m1-2026-09-27/README.md) |
| P3 layer-major MoE prefill with phase-shared VRAM arena on `Qwen3.6-35B-A3B` | Proven at 5.3K–64K (**1.93x faster** at 32K, **1.86x faster** at 64K); 128K adaptive VRAM budget planner unit-tested, 128K live P3 run **Not measured** | [P3 report](docs/measurements/v080-prefill-2026-09-28/README.md), [`test_ggml_decode_attention.cpp`](tests/cpp/test_ggml_decode_attention.cpp) |
| Direct-disk KV pages migrate GPU ↔ pinned ↔ pageable ↔ disk, checksum-verified (`crc32c`), corrupt pages refused | Proven | [`test_ggml_cuda_mechanism.cpp`](tests/cpp/test_ggml_cuda_mechanism.cpp), [`test_ggml_disk_buffer.cpp`](tests/cpp/test_ggml_disk_buffer.cpp) |
| Unsupported attention features or `-np > 1` fall back cleanly to llama.cpp's own KV cache at load | Proven | [`test_unsupported_model_falls_back_to_stock_kv_at_load`](tests/test_native_llama_paged.py) |
| Long-context throughput at 262K | **Not measured** | — |
| End-to-end 32K–128K runs on `Gemma-4-26B-A4B` and `UI-Mate-9B` | **Not measured** (kernel geometries verified in `K1`/`S1`) | — |
| Quality under INT4, INT2, 1-bit or JL tiers on the HuggingFace research path | **Not measured**; only FP8 was reached | [Quality](#quality-huggingface-path) |

Local verification suites on the development machine (Python 3.14): Python **368 passed, 27 skipped**
(import/repo hygiene included); native `llama.cpp` K1 (`simd` + `scalar`) and S1/C1
CUDA split-attention suites **all passed**.

---

## What ARGUS is and is not

- **Not an inference server.** Sampling, batching, HTTP serving, and GGUF loading stay
  in `llama.cpp`. `libargus` hooks into `llama.cpp`'s KV cache, graph builder, and
  execution loop via three focused patches (`host-kv.patch`, `cuda-kv.patch`,
  `v08-kv.patch`).
- **Two complementary operating families in one library:**
  1. **v0.8 In-Memory & MoE Orchestrator (`ARGUS_AUTO=1`):** Operates directly on
     `llama.cpp`'s `-nkvo` host KV cache (`f16`, `q8_0`, `q4_0`). Mirrors the hot prefix
     in `f16` GPU VRAM (`S1`), computes the remaining host KV with grouped SIMD CPU
     decode (`K1`) and optional bounded-mass approximate skipping (`E1` + `Multipole` V-centroid compensation), runs prefill attention
     on the GPU (`P2`), shares the VRAM arena between prefill MoE expert caches and
     decode KV mirrors (`P3`), and applies `POSIX_MADV_RANDOM` to sparse host MoE expert tensors.
  2. **v0.7 Direct-Disk Paged Store (`ARGUS_KV_DIR`):** Places KV in an unlinked
     `O_DIRECT` backing file with 4 KiB checksummed pages promoted/demoted across GPU,
     pinned RAM, pageable RAM, and disk within strict byte budgets.
- **Honest scope.** Experimental side projects (`hermes`, `neo-mobile`, container
  experiments) are archived under [`experiments/`](experiments/README.md) and excluded
  from the package.

---

## Architecture: The Five Levers of `libargus`

```text
llama.cpp (-nkvo -fa on) + libargus (ARGUS_AUTO=1)
  |
  +-- [Prefill Phase] ----------------------------------------------------------+
  |   1. P3 Layer-Major MoE Prefill (ggml_layer_prefill.cpp):                   |
  |      Detaches KV mirrors, borrows the phase-shared VRAM arena to stage one  |
  |      layer's MoE experts on the GPU across all ubatches (1x PCIe transfer   |
  |      per layer instead of N_ubatches x), then refills KV mirrors from host. |
  |   2. P2 GPU Prefill Attention (ggml_kv_tiers.cpp):                          |
  |      Streams the host K/V view to the GPU once per batch for FlashAttention.|
  |                                                                             |
  +-- [Decode Phase] -----------------------------------------------------------+
      1. C1 Mixed-Precision Write-Through (ggml_split_attention.cu):
         A single CUDA kernel writes new KV rows in F16 to the GPU mirror and
         in F16, Q8_0, or Q4_0 to pinned host memory.
      2. S1 Split Decode Attention (ggml_split_attention.cu + ggml_kv_tiers.cpp):
         GPU computes softmax partials (max, sum, weighted V) over mirrored
         cells [0, G) in VRAM while CPU computes partials over [G, n_kv) in RAM;
         both merge by exact log-sum-exp reduction.
      3. K1 Grouped SIMD CPU Decode + E1 Bounded-Mass Skipping (ggml_decode_attention.cpp):
         Reads each KV head once for all GQA query heads with AVX2/AVX-512
         in-register decoding (F16, Q8_0, Q4_0). With ARGUS_KV_SKIP_EPS > 0,
         uses cached [mu_k, rad_k, mu_v] multipole boxes and sink watermarks to skip
         blocks/cells with bounded softmax mass and compensate skipped V cells via mu_v.
```

---

## Using `libargus` with llama.cpp

### 1. Build `llama.cpp` with `libargus`

`libargus` patches pinned `llama.cpp` revision `54315813269112dd0baed7112ec87ad93a8218ca`
(see [`integrations/llama.cpp/README.md`](integrations/llama.cpp/README.md)). Apply the
three patches in order (or the single combined `argus.patch`) and point CMake at `argus_cache/csrc`:

```bash
git -C /path/to/llama.cpp apply /path/to/ARGUS/integrations/llama.cpp/host-kv.patch
git -C /path/to/llama.cpp apply /path/to/ARGUS/integrations/llama.cpp/cuda-kv.patch
git -C /path/to/llama.cpp apply /path/to/ARGUS/integrations/llama.cpp/v08-kv.patch
cmake -S /path/to/llama.cpp -B /path/to/llama.cpp/build-cuda \
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DARGUS_CORE_DIR=/path/to/ARGUS/argus_cache/csrc
cmake --build /path/to/llama.cpp/build-cuda --target llama-server -j4
```

With no `ARGUS_*` environment variable set, the compiled binary behaves identically to
stock `llama.cpp`.

### 2. Recommended v0.8 Modes (`ARGUS_AUTO=1`)

```bash
# 1. Zero-config autonomous mode (F16 host KV + auto F16 VRAM mirror + K1 + P2 + P3 + MADV_RANDOM)
export ARGUS_AUTO=1
llama-server -m model.gguf -c 65536 -ngl 99 --cpu-moe -fa on -nkvo -b 65536 -ub 2048

# 2. C1 Asymmetric Mixed-Precision (~60% less host KV RAM, F16 GPU mirror) + Opt-in E1 Multipole Bounded-Mass Skip
export ARGUS_AUTO=1
export ARGUS_KV_SKIP_EPS=1e-4
export ARGUS_MULTIPOLE=1
llama-server -m model.gguf -c 65536 -ngl 99 -fa on -nkvo -ctk q8_0 -ctv q4_0
```

### 3. v0.8 Configuration Reference (`argus_config.h`)

| Variable | Values | Meaning |
|---|---|---|
| `ARGUS_AUTO` | `1` / `on` / `0` / `off` | Enables autonomous orchestration (`ARGUS_KV_CPU_ATTENTION=k1`, `ARGUS_KV_VRAM_BYTES=auto`, `ARGUS_KV_PREFILL=gpu`, `ARGUS_PREFILL=layers`, `ARGUS_MADV_RANDOM=1`; keeps `ARGUS_KV_SKIP_EPS=0.0`, `ARGUS_MULTIPOLE=0`, `ARGUS_MOE_CACHE=0`, `ARGUS_TOP_R=0` opt-in). Any explicit variable below overrides its respective knob. |
| `ARGUS_KV_CPU_ATTENTION` | `k1` / `stock` | Selects K1 grouped SIMD CPU decode attention over host KV. |
| `ARGUS_K1_MAX_TOKENS` | integer (default `16`) | Maximum batch token count handled by the K1 / S1 decode path (covers single-token decode and speculative / MTP draft batches). |
| `ARGUS_KV_SKIP_EPS` | float in `[0, 1)` (default `0.0`) | Opt-in E1 bounded-mass approximate skipping threshold (or `ARGUS_E1=1` for `1e-4`). `0.0` is exact; `1e-4` prunes cold 32-cell K-blocks and V-cells. |
| `ARGUS_MULTIPOLE` | `1` / `on` / `0` / `off` (default `0`) | Opt-in Multipole monopole/dipole K-box bounds and V-centroid compensation (`W_skip * mu_V`) when `ARGUS_KV_SKIP_EPS > 0`. |
| `ARGUS_MADV_RANDOM` | `1` / `on` / `0` / `off` (default `1` in `ARGUS_AUTO`) | Applies `POSIX_MADV_RANDOM` to host MoE expert tensors (`--cpu-moe`), disabling OS sequential readahead bloat. |
| `ARGUS_MOE_CACHE` | `1` / `on` / `0` / `off` (default `0`) | Opt-in frequency-tracked hot-expert cache (`mlock` / `MADV_COLD`, `ARGUS_HOT_EXPERTS=32`) for systems with raised `RLIMIT_MEMLOCK` (`ulimit -l`). |
| `ARGUS_TOP_R` | `1` / `on` / `0` / `off` (default `0`) | Opt-in Top-$r$ (`r=32`) Cauchy-Schwarz tail bound for K-boxes (disabled by default because QK-Norm models have flat channel spectra). |
| `ARGUS_KV_VRAM_BYTES` | `auto` or byte count | Enables S1 `F16` GPU mirrors of `-nkvo` host KV (`f16`, `q8_0`, `q4_0`). `auto` sizes the arena from free VRAM after compute buffers are reserved. |
| `ARGUS_KV_VRAM_MARGIN` | byte count (default `256 MiB`) | Free VRAM margin kept by `ARGUS_KV_VRAM_BYTES=auto` (adapts down to a `64 MiB` safety floor when needed to fit the P3 shared MoE expert cache). |
| `ARGUS_KV_PREFILL` | `gpu` / `cpu` | P2: runs prefill FlashAttention on the GPU when KV lives on the host (`-nkvo`). |
| `ARGUS_PREFILL` | `layers` / `batches` | P3: runs multi-ubatch prefill layer-major so `--cpu-moe` expert weights cross PCIe once per prompt call, sharing the VRAM arena with S1 KV mirrors. |

*(For the v0.7 direct-disk `ARGUS_KV_DIR` settings, see [`integrations/llama.cpp/README.md`](integrations/llama.cpp/README.md).)*

---

## Local stack: ARGUS Studio and the services

The repository also ships the stack it is developed on: an ARGUS-built
`llama-server` plus **ARGUS Studio**, a local chat UI with permissioned tools.
There is no Ollama in the path.

| Port | What | Notes |
|---|---|---|
| `8080` | ARGUS `llama-server`, OpenAI API only | llama.cpp's bundled web page is off (`"webui": false`); Studio replaces it |
| `8090` | ARGUS Studio | chat UI, documents, web search, Hermes desktop tools |

```bash
pip install --no-build-isolation "argus-cache[studio]"   # FastAPI/uvicorn for Studio
argus-studio --upstream http://127.0.0.1:8080/v1         # UI on :8090
```

`scripts/install_systemd_service.sh` installs two user services under
`argus.target`. They do **not** start at boot: loading an 18 GB MoE model alongside
the display manager held the login screen (measured: the engine started with SDDM
at 10 s, the desktop arrived 62 s later). The compositor starts the target after
login (`systemctl --user start argus.target`); the installed `ARGUS Studio`
launcher does the same and opens Studio as an app window.

With `--load-mode mmap` the experts stay on disk until first touched, so the first
request after boot paid the SSD read (a 14-token prefill took **9.7 s**). The
launcher now reads the GGUF into page cache at idle I/O and CPU priority
(`ionice -c3`, `nice 19`; honoured by BFQ), which only spends disk time nobody
else wants and leaves ordinary reclaimable cache. Same cold start afterwards:
**0.65 s**.

Hermes desktop tools (window list, OCR, clicks, the user's own Hyprland
shortcuts) load only when `experiments/hermes` is present; the PyPI install runs
Studio without them. Shortcuts are read live from `hyprctl binds -j` and pressed
as key combinations, so editing `hyprland.lua` never leaves them stale; anything
other than volume/media/brightness asks for approval first.

---

## v0.8 results: Long-context tiers, MoE prefill, C1 & E1

### 1. 35B MoE at 32K–128K Filled Context (`Qwen3.6-35B-A3B` UD-Q4_K_M, `--cpu-moe`, `f16` KV)

Measured via [`benchmarks/bench_long_context.py`](benchmarks/bench_long_context.py) on
RTX 3050 Ti Laptop (4 GB) + i5-11300H (4 pinned threads, turbo off). Each context is
completely filled with prefilled tokens hiding a 6-digit code near the beginning,
followed by 32 decoded tokens ([M1 data](docs/measurements/v080-m1-2026-09-27/README.md),
[P3 data](docs/measurements/v080-prefill-2026-09-28/README.md)):

| Context | Stock GPU KV Decode | Stock `-nkvo` Decode | `libargus` K1 (`-nkvo`) | `libargus` S1 Tiers (`auto`) | S1 vs Best Stock | Prefill: Batch-Major (`P2`) | Prefill: Layer-Major (`P3`) | `P3` Speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **32K** | **22.47 tok/s** | 3.15 tok/s | 13.23 tok/s | 20.12 tok/s | 0.90x | 114.3 s | **59.1 s** | **1.93x** |
| **64K** | *VRAM OOM* | 1.86 tok/s | 9.49 tok/s | **12.31 tok/s** | **6.6x** | 242.2 s | **130.4 s** | **1.86x** |
| **128K** | *VRAM OOM* | 1.02 tok/s | 4.41 tok/s | **6.18 tok/s** | **6.0x** | 540.3 s | *Not measured live* ¹ | — |

¹ *At 128K, the original static 256 MiB margin left 386.6 MiB of free VRAM (< 498 MiB needed for the largest layer's experts). `argus_plan_vram_budget` in `argus_config.h` resolves this by adapting the margin to a 64 MiB floor when `free_vram >= shared_expert_bytes + 64 MiB` (verified in unit tests; live 128K P3 rerun is not yet recorded).*

### 1.1 Real `Qwen3.6-35B-A3B` End-to-End Validation & SOTA-3 Ablation (`8192` & `32768`, 3 Warmup + 5 Measured Runs)

Measured via [`bench_real_a3b_validation.cpp`](docs/measurements/v080-libargus-2026-10-27/bench_real_a3b_validation.cpp) ([JSON artifact](docs/measurements/v080-libargus-2026-10-27/real_a3b_validation.json), [full report](docs/measurements/v080-libargus-2026-10-27/README.md)) on `Qwen3.6-35B-A3B-UD-Q4_K_M-MTP.gguf` (seed `2026`, deterministic greedy sampling, 32 decoded tokens, 3 needles at 10% / 50% / 85% depth: `657446`, `839204`, `412975`):

| Context | Mode | Prefill (`tok/s` / `s`) | TTFT (`s`) | Decode Median (`tok/s`) | Token Lat `p50 / p95` (`ms`) | K-Block Skip % | V-Cell Skip % | Logit `max_abs` (vs Stock / vs E1-OFF) | $\text{KL}_{\text{mean}}$ (vs Stock / vs E1-OFF) | Needles (`10%/50%/85%`) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| **8192** | `Stock llama.cpp` *(disk I/O bottleneck)* | 25.1 (`321.68 s`) | 321.85 | **5.73** | `168.0 / 237.6` | 0.00% | 0.00% | `0 / —` | `0 / —` | **3/3** |
| **8192** | `ARGUS AUTO (E1 OFF)` | **114.9 (`70.21 s`)** | **70.27** | **18.55** | **`48.3` / `65.0`** | 0.00% | 0.00% | `2.14 / 0` | `3.20e-5 / 0` | **3/3** |
| **8192** | `A: ARGUS AUTO (SOTA-3 OFF)` | 114.9 (`70.21 s`) | 70.27 | **18.31** | `48.9 / 66.5` | **12.49%** | **38.98%** | `2.27 / 1.60` | `3.65e-5 / 4.16e-5` | **3/3** |
| **8192** | `B: A + MADV_RANDOM` | 114.9 (`70.21 s`) | 70.27 | **18.04** | `49.7 / 68.7` | **12.49%** | **38.98%** | `2.27 / 1.60` | `3.65e-5 / 4.16e-5` | **3/3** |
| **8192** | `C: B + Hot-Expert Cache` *(68.1% hit)* | 114.9 (`70.21 s`) | 70.27 | **17.95** | `50.4 / 72.3` | **12.49%** | **38.98%** | `2.27 / 1.60` | `3.65e-5 / 4.16e-5` | **3/3** |
| **8192** | `D: C + Multipole / V-Centroid` | 114.9 (`70.21 s`) | 70.27 | **18.25** | `49.4` / **`64.4`** | **12.57%** | **39.01%** | `7.26 / 7.95` | `3.93e-5` / **`2.50e-5`** | **3/3** |
| **8192** | `E: D + Top-r` *(disables K-skip)* | 114.9 (`70.21 s`) | 70.27 | **17.49** | `50.0 / 70.0` | `0.00%` | `36.18%` | `3.78 / 3.94` | `5.18e-5 / 3.64e-5` | **3/3** |
| **32768** | `Stock llama.cpp` | 33.1 (`986.81 s`) | 987.18 | **2.69** | `361.7 / 457.5` | 0.00% | 0.00% | `0 / —` | `0 / —` | **3/3** |
| **32768** | `ARGUS AUTO (E1 OFF)` | **322.7 (`101.15 s`)** | **101.24** | **10.90** *(14.22¹)* | `85.7 / 149.6` | 0.00% | 0.00% | `1.51 / 0` | `2.68e-5 / 0` | **3/3** |
| **32768** | `A: ARGUS AUTO (SOTA-3 OFF)` | 322.7 (`101.15 s`) | 101.25 | **10.56** *(14.29¹)* | `84.4 / 131.0` | **12.17%** | **34.83%** | `2.38 / 2.46` | `2.10e-5 / 2.18e-5` | **3/3** |
| **32768** | `B: A + MADV_RANDOM` | 322.7 (`101.15 s`) | 101.24 | **10.79** | `84.7` / **`111.7`** | **12.17%** | **34.83%** | `2.64 / 2.72` | `4.63e-5 / 2.57e-5` | **3/3** |
| **32768** | `C: B + Hot-Expert Cache` *(68.7% hit)* | 322.7 (`101.15 s`) | 101.24 | **11.15** | **`81.7`** / `129.3` | **12.17%** | **34.83%** | `2.85 / 2.97` | `4.39e-5 / 3.68e-5` | **3/3** |
| **32768** | `D: C + Multipole / V-Centroid` | 322.7 (`101.15 s`) | 101.25 | **10.65** | `86.9 / 114.8` | **12.03%** | **34.81%** | **`1.94 / 1.06`** | **`2.85e-5 / 2.35e-5`** | **3/3** |
| **32768** | `E: D + Top-r` *(disables K-skip)* | 322.7 (`101.15 s`) | 101.25 | **10.49** | `91.1 / 147.5` | `0.00%` | `32.73%` | `2.52 / 2.44` | `1.74e-5 / 2.26e-5` | **3/3** |

¹ *In the earlier unloaded-host run (`MemAvailable > 23 GiB`, `RSS ~ 22.1 GiB`), `ARGUS AUTO` reached `14.22–14.29 tok/s` at 32K vs `Stock`'s `2.86 tok/s`. In the continuous 8-mode ablation above (`MemAvailable = 15.2 GiB`, `RSS = 10.6 GiB`), `ARGUS AUTO` delivered `10.90–11.15 tok/s` (`4.05x–4.15x` over `Stock`'s `2.69 tok/s`) and `322.7 tok/s` prefill (`9.76x` over `Stock`), while `MADV_RANDOM` cut 32K `p95` latency by `-25.3%` (`149.6 -> 111.7 ms`) and `Multipole / V-Centroid` cut 32K max logit error vs `E1 OFF` by `2.79x` (`2.969 -> 1.064`). See [full ablation & Layer 27 mathematical audit](docs/measurements/v080-libargus-2026-10-27/README.md).*

### 1.2 32K Runtime Bottleneck Isolation on `Qwen3.6-35B-A3B` (5-Question Systems Audit)

Measured via [`bench_32k_bottleneck_audit.cpp`](docs/measurements/v080-libargus-2026-10-27/bench_32k_bottleneck_audit.cpp) ([JSON telemetry](docs/measurements/v080-libargus-2026-10-27/bottleneck_32k_audit.json), [cold readahead JSON](docs/measurements/v080-libargus-2026-10-27/q3_cold_readahead.json), [Section 3.5 report](docs/measurements/v080-libargus-2026-10-27/README.md#35-32k-runtime-bottleneck-isolation-on-qwen36-35b-a3b)):

| Bottleneck Question | Measured 32K Finding | Key Metric / Evidence |
|---|---|---|
| **Q1. CPU MoE RAM Residency (`mincore(2)`)** | Across `18,662.00 MiB` (`4,777,592` 4 KiB pages) of CPU MoE weights (`466.55 MiB/layer × 40`), **`51.87%` (`9,680.17 MiB`)** is resident immediately after 32K `P3` prefill (`46.28%` / `8,637.75 MiB` under `~11–13 GiB` desktop RAM pressure; **`100%`** when `>= 19 GiB` free RAM is available). | `mincore(2)` per-layer scan across `ffn_{up,gate,down}_exps` |
| **Q2. Per-Layer Major Page Faults (`blk.0..39`)** | Because `P3` layer-major prefill streams `blk.0 -> blk.39` sequentially, **`blk.30..39` finish prefill warm at `61.0%–92.5%` residency** (`2,190` `majflt`, `8.55 MiB` read), pushing **`blk.0..29` down to `25.2%–50.3%`**. In decode, **`blk.18..29`** is the #1 fault hotspot (`30,590` `majflt`, `119.33 MiB` read over 8 tokens), followed by **`blk.0..7`** (`8,741` `majflt`, `34.85 MiB`). | Per-layer `ru_majflt` + `/proc/self/io` `read_bytes` hooks |
| **Q3. `MADV_RANDOM` vs `MADV_NORMAL` Readahead I/O** | Evicting `blk.0..7` (`3,712 MiB`) to `0.01%` residency and decoding 8 tokens (`335` routed experts = `607.19 MiB`): `MADV_NORMAL` pulls **`2,060.04 MiB`** into `blk.0..7` (**`+1,452.85 MiB` / `3.39x` readahead bloat**), reads `1,684.73 KiB/majflt`, triggers **`12,036.62 MiB` (`11.75 GiB`)** total disk reads across `blk.0..39`, and runs at `0.17 tok/s`. `MADV_RANDOM` pulls **`612.18 MiB`** (**`+5.00 MiB` overhead, `-99.66%` bloat**), reads **`4.07 KiB/majflt`**, cuts disk I/O **`19.29x` (`623.92 MiB`)**, and speeds up cold decode **`2.45x` (`0.42 tok/s`)**. | `madvise(MADV_DONTNEED)` + `posix_fadvise(POSIX_FADV_DONTNEED)` cold test |
| **Q4. `Multipole` K/V Bytes Read & Kernel Time** | Across 10 full-attention layers at 32K (`637.97 MiB/tok` exact KV): `E1 ON, Multipole OFF` reads **`317.11 MiB/tok` (GPU `S1`) / `313.95 MiB/tok` (CPU `K1`)** (`-50.3% / -50.8%`), cutting kernel time `11.522 -> 10.122 ms/tok` (`-12.15%`) on GPU `S1` and `37.977 -> 25.659 ms/tok` (`-32.44%`) on CPU `K1`. `Multipole ON` reads **`322.39 / 319.31 MiB/tok`** (`-49.5% / -49.9%`, `+6.56 MiB/tok` for $\mu_V$) and reaches **`9.780 ms/tok` (`-15.12%`)** on GPU `S1` (`26.368 ms/tok`, `-30.57%` on CPU `K1`). | `cudaEventElapsedTime` (`S1`) + `steady_clock` (`K1`) + exact byte counters |
| **Q5. Graph-Split & `P3` Decode Contribution** | At 32K warm RAM: `Stock (-nkvo)` runs at **`3.13 tok/s`** (`282` splits, `316.35 ms/tok`). `S1+K1` alone reaches **`9.00 tok/s` (`2.87x`, `-209.30 ms/tok`)**. Keeping Recurrent State on GPU (`ARGUS_RS_GPU_OPT=1`) eliminates **`180` graph splits (`282 -> 102`)** across 30 Gated DeltaNet layers and doubles decode to **`18.23 tok/s` (`+2.03x`, `-53.25 ms/tok`, `5.82x` over Stock)**. `MoE Split Fusion` prevents an `+80` split regression (`102 -> 182`) when `ARGUS_MOE_CACHE=1` is active. | `ARGUS_RS_GPU_OPT` & `ARGUS_MOE_SPLIT_OPT` ablation |

### 2. End-to-End `ARGUS_AUTO=1`, C1 Mixed-Precision & E1 Skipping (8K Filled Context, `Qwen2.5-0.5B`)

Measured on `qwen2.5-0.5b-instruct-q4_k_m.gguf` at 8192 context (8096 prompt tokens +
32 decoded tokens) across both a constrained 16 MiB split mirror (`1280` of `8192` cells
mirrored in `F16` on the GPU; `6912` cells on the host) and the default `ARGUS_AUTO=1`
`auto` mirror (`8192` cells mirrored in `F16` on the GPU with GPU mirror prefill bypass)
([data](docs/measurements/v080-libargus-2026-10-27/README.md)):

| Configuration | Host KV Codec | GPU Mirror | Host KV Memory Saved | Prefill | Decode Rate | vs Stock `-nkvo` | Peak RSS | Needle (`657446`) |
|---|---|---|---:|---:|---:|---:|---:|:---:|
| `stock-nkvo-f16` | `f16` | None | 0% | 0.898 s (9,012 tok/s) | **23.18 tok/s** | 1.00x | 713.5 MiB | Found |
| `argus-auto-f16` *(16 MiB split)* | `f16` | `f16` (1280) | 0% | 0.888 s (9,122 tok/s) | **79.54 tok/s** | **3.43x** | 699.3 MiB | Found |
| `argus-auto-c1-q8_0` *(16 MiB split)* | `q8_0` | `f16` (1280) | **-46.9% (1.88x)** | 0.891 s (9,084 tok/s) | **66.94 tok/s** | **2.89x** | 658.2 MiB (**-55.3 MiB**) | Found |
| `argus-auto-c1-q8_0-e1-skip` *(16 MiB split)* | `q8_0` | `f16` (1280) | **-46.9% (1.88x)** | 0.889 s (9,108 tok/s) | **67.67 tok/s** | **2.92x** | 665.2 MiB (**-48.3 MiB**) | Found |
| `argus-auto-c1-k_q8-v_q4-skip` *(16 MiB split)* | **K=`q8_0`, V=`q4_0`** | `f16` (1280) | **-59.4% (2.46x)** | 0.925 s (8,751 tok/s) | **69.60 tok/s** | **3.00x** | 659.7 MiB (**-53.8 MiB**) | Found |
| `argus-auto-full-mirror-k_q8-v_q4` *(`auto` mirror)* | **K=`q8_0`, V=`q4_0`** | `f16` (8192) | **-59.4% (2.46x)** | **0.815 s (9,936 tok/s)** | **175.45 tok/s** | **7.57x** | **638.5 MiB (-75.0 MiB)** | **Found** |

### 3. Standalone Synthetic CPU Decode Kernel: Stock GGML vs K1 Exact vs K1 + E1 Skip (`Qwen3.6-35B-A3B` Geometry Without QK-Norm)

Single-token CPU decode latency on synthetic non-QK-Norm tensors across 16K, 64K, and 128K cells (`d = 256`, `16/2`
heads, 4 threads, `-march=native`, [source & data](docs/measurements/v080-libargus-2026-10-27/README.md)):

| Context | KV Codec | KV Size / Layer | Stock GGML (ms) | `libargus` K1 Exact (ms) | `libargus` K1 + E1 Skip (ms) | Speedup vs Stock | Blocks Skipped (K / Total) | Max Rel Err |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| **16K** | `f16` | 32.0 MiB | 12.91 ms | 1.72 ms (7.5x) | **0.48 ms** | **26.7x** | 1002 / 1024 | `4.35e-07` |
| **16K** | `q8_0` | 17.0 MiB (-47%) | 10.12 ms | 2.41 ms (4.2x) | **0.33 ms** | **31.1x** *(39.1x vs Stock F16)* | 1002 / 1024 | `4.67e-07` |
| **16K** | `q4_0` | 9.0 MiB (-72%) | 7.84 ms | 2.38 ms (3.3x) | **0.45 ms** | **17.3x** *(28.7x vs Stock F16)* | 1002 / 1024 | `3.88e-07` |
| **64K** | `f16` | 128.0 MiB | 57.43 ms | 5.79 ms (9.9x) | **0.91 ms** | **62.8x** | 4074 / 4096 | `5.81e-07` |
| **64K** | `q8_0` | 68.0 MiB (-47%) | 34.32 ms | 9.60 ms (3.6x) | **1.33 ms** | **25.8x** *(43.2x vs Stock F16)* | 4074 / 4096 | `3.94e-07` |
| **64K** | `q4_0` | 36.0 MiB (-72%) | 37.43 ms | 9.40 ms (4.0x) | **1.34 ms** | **27.9x** *(42.9x vs Stock F16)* | 4074 / 4096 | `4.43e-07` |
| **128K** | `f16` | 256.0 MiB | 114.72 ms | 11.35 ms (10.1x) | **2.56 ms** | **44.9x** | 8170 / 8192 | `6.03e-07` |
| **128K** | `q8_0` | 136.0 MiB (-47%) | 81.50 ms | **10.95 ms (7.4x)** | **2.60 ms** | **31.4x** *(44.1x vs Stock F16)* | 8170 / 8192 | `3.28e-07` |
| **128K** | `q4_0` | 72.0 MiB (-72%) | 83.19 ms | 18.66 ms (4.5x) | **2.56 ms** | **32.5x** *(44.8x vs Stock F16)* | 8170 / 8192 | `4.10e-07` |

---

## v0.7 results: Direct-disk paged KV

`Qwen2.5-0.5B-Instruct` Q4_K_M, `F16` KV, 4096 context, 4016-token prompt, 16 generated
tokens, ubatch 64, RTX 3050 Ti Laptop (4 GB), `-nkvo`, GPU/pinned budgets 64 MiB
([data](docs/measurements/v070-e10-2026-09-26/)):

| Mode | Prefill | vs Stock | Decode | Output Hash |
|---|---:|---:|---:|---|
| stock llama.cpp, host KV (tensor-core FlashAttention) | 1.336 s (1.331–1.341) | 1.00x | 37.5 tok/s | `007ddc77` |
| ARGUS, GPU control (diagnostic, all KV on GPU) | 1.719 s (1.666–1.759) | 1.29x | 55.6 tok/s | `a152ed56` |
| ARGUS, policy on (GPU/pinned/RAM/disk) | 1.821 s (1.814–1.826) | 1.36x | 48.2 tok/s | `a152ed56` |

When KV exceeds the GPU budget (`2 MiB` GPU and `2 MiB` pinned tiers for `48 MiB` of
KV): `47.6 s` prefill and `1.0 tok/s` decode (`2.1 GB` read from disk per request).

---

## Reading the stock comparison

- **Arithmetic contracts.** In default exact mode (`ARGUS_KV_SKIP_EPS=0`), `K1` and `S1`
  match a `float64` reference within `2e-6` (`fp32` accumulation) and `K1` is
  bit-identical across thread counts. When `ARGUS_KV_SKIP_EPS > 0` (`E1`), `K1` and `S1`
  use bounded-mass approximate skipping with Multipole V-centroid compensation (`W_skip * mu_V`). Stock
  `llama.cpp` on the GPU uses tensor-core FlashAttention with its own accumulation
  order.
- **Measured hardware.** Every measurement in this repository was taken on one machine
  (Intel i5-11300H + RTX 3050 Ti Laptop 4 GB, PCIe Gen3 x4). Do not extrapolate ratios
  to multi-GPU server hardware without measuring there.

---

## Using ARGUS with HuggingFace (legacy research path)

The Python package (`pip install --no-build-isolation -e .`) provides the v0.4
HuggingFace `PagedDynamicKVCache` research path:

```python
import torch
from transformers import AutoModelForCausalLM
from argus_cache import AdaptiveCachePolicy, patch_model_with_argus

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct", dtype=torch.float16,
).to("cuda")

model = patch_model_with_argus(
    model,
    page_size=1024,
    max_active_pages=2,
    max_fp8_pages=2,
    sink_tokens=4,
    activation_policy=AdaptiveCachePolicy(mode="balanced", expected_tokens=16_448),
    pipeline_profile="balanced",
)
```

---

## Benchmark methodology

```bash
# 1. Python unit & hygiene suite
.venv/bin/python -m pytest -q --ignore=tests/test_native_llama_paged.py

# 2. Native llama.cpp C++/CUDA kernel & lifecycle suite
ARGUS_TEST_CUDA=1 \
ARGUS_LLAMA_CPP_DIR="$PWD/scratch/llama.cpp-v050" \
ARGUS_LLAMA_BUILD=build-cuda \
ARGUS_TEST_GGUF="$PWD/scratch/models/stories15M.gguf" \
ARGUS_TEST_QUANT_GGUF="$PWD/scratch/models/qwen2.5-0.5b-instruct-q4_k_m.gguf" \
  .venv/bin/python -m pytest -q tests/test_native_llama_paged.py

# 3. End-to-end v0.8 long-context & C1/E1 benchmark
.venv/bin/python benchmarks/bench_long_context.py \
  --server scratch/llama.cpp-v050/build-cuda/bin/llama-server \
  --model scratch/models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  --configs docs/measurements/v080-libargus-2026-10-27/bench_e2e_configs.json \
  --contexts 8192 --predict 32 --output result.json
```

---

## Known limits

- **Single KV stream (`-np 1`):** `libargus` manages single-sequence KV caches; multi-stream (`-np > 1` without a unified cache) or MLA/ALiBi/soft-capping models automatically fall back to `llama.cpp`'s native KV cache at load.
- **Supported head dimensions in S1 GPU kernels:** `d` in `{64, 128, 256, 512}`; other dimensions automatically fall back to K1 on the CPU.
- **Short-turn prefill is CPU-bound with experts in RAM:** prompts under the P3 threshold (128 tokens) run the MoE experts on the CPU at ~30–40 ms per token on the 4-core test machine, the same rate as decode; P3 takes over above it.
- **Pinned `llama.cpp` revision:** The patches target upstream commit `54315813269112dd0baed7112ec87ad93a8218ca`.

---

## Earlier evidence (v0.4–v0.6)

- **v0.5 KV ownership & quantized CPU parity:** Zero logit difference across 19 steps (including crop and state restore) for `f32`, `f16`, `q8_0`, and `q4_0` on `Qwen2.5-0.5B` ([artifact](docs/measurements/v050-llama-host-ownership-2026-09-15.json)).
- **v0.6 first operating point (4K):** Prefill `2.12x` (GPU control) and `7.42x` (policy on) vs stock; closed to `1.29x` and `1.36x` in v0.7 ([checkpoint](docs/measurements/v060-checkpoint-2026-09-18.md)).
- **v0.4 HuggingFace path (`Qwen2.5-0.5B`):** `-7.7%` VRAM at 16K for `4.2x` TPOT; perplexity delta `-0.0176` reaching only the `FP8` tier ([analysis](docs/findings-2026-08-14.md)).

### Quality (HuggingFace path)

On the HuggingFace research path, only the `FP8` tier was reached in perplexity evaluations; `INT4`, `INT2`, `1-bit`, and `JL` tiers are **Not measured** for end-to-end model quality.

---

## Roadmap

- **v0.8 (current):** `libargus` unified autonomous config (`ARGUS_AUTO=1`), `K1` grouped SIMD CPU decode, `E1` + `Multipole` bounded-mass approximate skipping, `S1` split GPU+CPU decode, `C1` mixed-precision (`Q8_0`/`Q4_0` host + `F16` GPU mirror) tiers, `P2` GPU prefill, `P3` phase-shared layer-major MoE prefill, and sparse MoE `MADV_RANDOM` + opt-in Hot-Expert Cache.
- **Next:** Record live 128K `P3` layer-major prefill timing on `Qwen3.6-35B-A3B` with the adaptive VRAM budget planner, and measure `Gemma-4-26B-A4B` and `UI-Mate-9B` end to end.

## License

ARGUS is licensed under the [Apache 2.0 License](LICENSE).
