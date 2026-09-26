# ARGUS

**A KV-cache memory manager for LLM runtimes. It treats the KV cache as a paged
memory hierarchy — GPU, pinned RAM, pageable RAM, disk — instead of one tensor
pinned to one device, so a context can outlive the VRAM that would normally hold
it.**

ARGUS is not an inference server. It is the layer underneath one, owning where
each KV page lives, how it is written and how attention reads it. The deepest
integration is llama.cpp, where ARGUS owns the KV allocation, the page writes and
the attention reads, and must stay bit-exact with its own reference path.

**Status: v0.7.0.** On the one workload measured end to end (4K context,
Qwen2.5-0.5B, RTX 3050 Ti Laptop), ARGUS prefill takes **1.29x** stock llama.cpp's
time with all KV on the GPU and **1.36x** with the placement policy managing
GPU/RAM/disk tiers. Decode is **faster than stock** in both modes (55.6 and
48.2 tok/s vs 37.5). At the start of v0.7 the same prefill was 2.09x stock and
the policy-on mode was 7.4x. These are measurements on one model, one GPU and one
context length, not a general speed claim. [Read the comparison
caveat](#reading-the-stock-comparison) before quoting them.

[![packaging](https://github.com/Zwannfrederick/ARGUS-Anchored-Random-Geometric-Unbiased-Storage/actions/workflows/packaging.yml/badge.svg)](https://github.com/Zwannfrederick/ARGUS-Anchored-Random-Geometric-Unbiased-Storage/actions/workflows/packaging.yml)

Türkçe belge: [README_TR.md](README_TR.md)

---

## Contents

- [Status at a glance](#status-at-a-glance)
- [What ARGUS is and is not](#what-argus-is-and-is-not)
- [Install](#install)
- [Using ARGUS with llama.cpp](#using-argus-with-llamacpp) — build, modes, every setting
- [Using ARGUS with HuggingFace](#using-argus-with-huggingface)
- [v0.7 results](#v07-results) and [how they moved](#how-v07-got-there)
- [Reading the stock comparison](#reading-the-stock-comparison)
- [How it works](#how-it-works)
- [Benchmark methodology](#benchmark-methodology)
- [Known limits](#known-limits)
- [Earlier evidence](#earlier-evidence-v04v06)
- [Roadmap](#roadmap)

---

## Status at a glance

Every "proven" row links to the artifact that proves it. Every "not measured" row
is open, not pending publication.

| Claim | Status | Evidence |
|---|---|---|
| ARGUS owns llama.cpp KV allocation, writes and attention reads | Proven | [host ownership](docs/measurements/v050-llama-host-ownership-2026-09-15.json) |
| Every ARGUS attention path is bit-exact with its staged reference | Proven (float-vector equality, adversarial masks, mutation-checked; 4K output hash `a152ed56` in every mode) | [v0.7 plan](plans/argus-v0.7.0.md) |
| Output is byte-identical to stock llama.cpp | **Only where the arithmetic matches** — proven on the v0.5 UI-Mate run; **not** on the v0.7 4K benchmark, where stock uses tensor-core FlashAttention | [UI-Mate parity](docs/measurements/v050-ui-mate-reference-parity-2026-09-16.json), [caveat](#reading-the-stock-comparison) |
| KV pages migrate GPU ↔ pinned ↔ pageable ↔ disk, source-preserving | Proven | [CUDA mechanism](docs/measurements/v050-cuda-mechanism-2026-09-16.json) |
| Placement policy keeps its decisions identical across attention paths | Proven (staged vs resident vs page-table paths, 4 MiB and 256 KiB budgets) | [E6](docs/measurements/v070-e6-policy-logic-2026-09-25.md), [E10](plans/argus-v0.7.0.md) |
| 4K prefill vs stock | **Slower**: 1.29x (GPU control), 1.36x (policy on) | [final baseline](docs/measurements/v070-e10-2026-09-26/) |
| 4K decode vs stock | **Faster**: 55.6 / 48.2 vs 37.5 tok/s | [final baseline](docs/measurements/v070-e10-2026-09-26/) |
| KV larger than the GPU budget | Works, disk-bound: 2 MiB GPU tier for 48 MiB of KV → 47.6 s prefill, 1.0 tok/s | [E6](docs/measurements/v070-e6-policy-logic-2026-09-25.md) |
| Unsupported models fall back to llama.cpp's own KV cache | Proven (`-np 2` starts and serves) | `tests/test_native_llama_paged.py` |
| Long-context throughput at 262K | **Not measured** | — |
| Other models, GPUs, context lengths | **Not measured** | — |
| Quality under INT4, INT2, 1-bit or JL tiers (HuggingFace path) | **Not measured**; only FP8 was reached | [Quality](#quality-huggingface-path) |

Local suites on the development machine for this release: Python **382 passed,
6 skipped**; native llama.cpp on the CUDA build **18 passed** (including the
quantized-KV lifecycle checks); native on the CPU build **11 passed, 7 skipped**
(the CUDA-only checks). The CI badge covers
packaging and repository hygiene only — hosted runners have no CUDA device, so a
green badge means the package ships the right files, not that ARGUS works.

## What ARGUS is and is not

- **Not an inference server.** Sampling, batching, API serving and model loading
  stay in the runtime. ARGUS manages KV memory.
- **Not a general speedup.** On the one measured workload ARGUS prefill is still
  slower than stock and decode is faster. Neither result is claimed beyond that
  workload.
- **Not a quantizer by default.** On the llama.cpp path KV keeps GGML's codec
  (F16 for the CUDA path). Per-page mixed precision exists only on the HuggingFace
  research path and has quality evidence only for FP8.
- **Not persistent storage.** Disk backing lives in an unlinked `O_DIRECT` file:
  nothing survives the process. "Verified backing" means the copy a page can be
  demoted to and read back from inside one process.

---

## Install

```bash
pip install torch                                        # must already be importable
pip install --no-build-isolation argus-cache             # Python runtime + native extension
pip install --no-build-isolation "argus-cache[gateway]"  # + Anthropic Messages gateway
```

ARGUS ships as a source distribution and compiles a CUDA extension on your
machine, so a CUDA-capable PyTorch, the CUDA toolkit and a C++17 compiler must
already be present. There is no prebuilt wheel: a binary compiled against one
PyTorch ABI and CUDA version would be wrong for most installs.

`--no-build-isolation` is required. The build reads your installed `torch` to
configure the extension; pip's isolated build would hide it and compile for an ABI
your runtime does not have.

The llama.cpp integration is **not** a pip feature: its sources
(`argus_cache/csrc/ggml_*`) and patches (`integrations/llama.cpp/`) ship in the
repository and in the source distribution, and are compiled into llama.cpp, as
described next. To work on ARGUS itself:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip && pip install -e . --no-build-isolation
python setup.py build_ext --inplace
```

---

## Using ARGUS with llama.cpp

### 1. Build llama.cpp with ARGUS

ARGUS patches one pinned llama.cpp revision (see
[`integrations/llama.cpp/README.md`](integrations/llama.cpp/README.md) for the
revision and the full contract). Verify the revision, then apply both patches in
order and point CMake at the ARGUS sources:

```bash
git -C /path/to/llama.cpp apply --check /path/to/ARGUS/integrations/llama.cpp/host-kv.patch
git -C /path/to/llama.cpp apply         /path/to/ARGUS/integrations/llama.cpp/host-kv.patch
git -C /path/to/llama.cpp apply --check /path/to/ARGUS/integrations/llama.cpp/cuda-kv.patch
git -C /path/to/llama.cpp apply         /path/to/ARGUS/integrations/llama.cpp/cuda-kv.patch
cmake -S /path/to/llama.cpp -B /path/to/llama.cpp/build \
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DARGUS_CORE_DIR=/path/to/ARGUS/argus_cache/csrc
cmake --build /path/to/llama.cpp/build --target llama-server -j
```

`CMAKE_CUDA_ARCHITECTURES=86` is the tested RTX 30-series value; use your GPU's.
A CPU-only build (`-DGGML_CUDA=OFF`) gets the host and direct-disk modes without
CUDA attention. With no `ARGUS_*` variable set, the patched binary behaves exactly
like stock llama.cpp.

### 2. Pick a mode

Everything is configured through environment variables read by `llama-server`
(or any program linking the patched `libllama`). Budgets are hard limits: going
over one fails with an explicit error instead of growing.

| Mode | What it is for | Required settings | llama-server flags |
|---|---|---|---|
| **Mapped host KV** | KV in file-backed host memory; llama.cpp's attention | `ARGUS_KV_DIR`, `ARGUS_KV_MAX_BYTES` | `-nkvo` |
| **CPU block attention** | ARGUS CPU attention over mapped KV | + `ARGUS_KV_RESIDENT_BYTES` | `-nkvo -fa on` |
| **Direct disk KV** | KV only on `O_DIRECT` disk, verified pages, CPU attention | + `ARGUS_KV_STAGING_BYTES` | `-nkvo -fa on` |
| **CUDA, policy off** | Reference disk path with CUDA attention (every read from disk) | + `ARGUS_KV_GPU_BYTES`, `ARGUS_KV_PINNED_BYTES` | `-nkvo -fa on -ctk f16 -ctv f16` |
| **CUDA, policy on** (the product mode) | Pages placed across GPU / pinned / RAM / disk within budgets | + `ARGUS_KV_POLICY=on` (optional `ARGUS_KV_RAM_BYTES`) | same |
| **GPU control** (diagnostic) | All KV GPU-authoritative, no disk at all | + `ARGUS_KV_GPU_CONTROL=1`, policy off | same |

A complete policy-on example, as used for the v0.7 measurements (4K context,
Qwen2.5-0.5B):

```bash
export ARGUS_KV_DIR=/path/on/a/physical/disk      # not tmpfs: pages must be evictable
export ARGUS_KV_MAX_BYTES=$((64 << 30))           # disk bytes for KV backing (2x KV size is reserved)
export ARGUS_KV_RESIDENT_BYTES=$((4 << 20))       # page-descriptor metadata budget
export ARGUS_KV_STAGING_BYTES=$((4 << 20))        # all ARGUS host scratch, bounce buffers, worker stacks
export ARGUS_KV_GPU_BYTES=$((64 << 20))           # GPU tier budget (KV pages + attention scratch)
export ARGUS_KV_PINNED_BYTES=$((64 << 20))        # pinned-RAM tier budget
export ARGUS_KV_POLICY=on
llama-server -m qwen2.5-0.5b-instruct-q4_k_m.gguf -c 4096 -np 1 -ngl 99 \
  -fa on -nkvo -ctk f16 -ctv f16 -ub 64
```

The GPU budget decides the regime. When all KV fits (48 MiB here) ARGUS runs its
fastest path; when it does not, pages live on pinned RAM, RAM or disk and reads
become disk-bound (see [results](#v07-results)).

### 3. Every setting

| Variable | Values | Meaning |
|---|---|---|
| `ARGUS_KV_DIR` | directory | Enables ARGUS. Backing files are created here (and unlinked). Use a physical filesystem. |
| `ARGUS_KV_MAX_BYTES` | bytes | Disk / mapping budget for KV backing. Direct disk mode reserves two slots per page (2x KV). |
| `ARGUS_KV_RESIDENT_BYTES` | bytes | Mapped mode: resident target that selects CPU block attention. Disk modes: budget for page descriptors and resident handles (metadata, not KV payload). |
| `ARGUS_KV_STAGING_BYTES` | bytes | Enables direct disk KV. Bounds every ARGUS host scratch buffer, bounce page, prefetch/flusher stack and attention scratch, process-wide. Must hold at least one encoded row plus a page. |
| `ARGUS_KV_BLOCK_CELLS` | cells (default 256) | Block size of CPU block attention. |
| `ARGUS_KV_GPU_BYTES` | bytes | Enables CUDA attention (disk modes). Hard budget for GPU KV pages and ARGUS GPU scratch — not a limit on llama.cpp's own VRAM. |
| `ARGUS_KV_PINNED_BYTES` | bytes | Required with the GPU tier. Pinned-RAM tier budget. |
| `ARGUS_KV_RAM_BYTES` | bytes (default unset = tier off) | Pageable-RAM tier budget for the policy. |
| `ARGUS_KV_POLICY` | `off` (default) / `on` | Placement policy. Unknown values fail. |
| `ARGUS_KV_GPU_CONTROL` | `1` / unset | Diagnostic GPU-authoritative store. No disk backing, no migration; requires policy off. |
| `ARGUS_KV_STATS_PATH` | file | JSON stats (budgets, peaks, bytes, policy counters, profile scopes), rewritten atomically after attention calls. |
| `ARGUS_KV_PROFILE` | `cpu` / `1` | CPU scopes and counters (`cpu`), plus CUDA event timing (`1`). Profiling changes timing; compare only against unprofiled runs of the same binary. |
| `ARGUS_KV_ATTENTION_PATH` | `cells-v2` (default), `cells-kc`, `cells-mlp`, `cells`, `batched`, `direct`, `staged` | Reference attention paths for A/B checks. All are bit-exact with `staged`. |
| `ARGUS_KV_CHECKSUM` | `crc32c` (default) / `fnv` | Page digest. Both detect accidental corruption; neither is an authenticity check. |
| `ARGUS_KV_PAGE_COMMIT` | `run` (default) / `page` | GPU-control host writes: whole-page runs or one page at a time (reference). |
| `ARGUS_KV_NO_OVERLAP` | set / unset | Staged attention: serial tiles instead of overlapped transfer and compute. |

### 4. What happens at runtime

- **Writes.** New K/V rows are written on the GPU, byte-identical to the CPU
  encoding (round-to-nearest-even F32→F16). In **policy-on** stores the page
  becomes *dirty* and a background flusher publishes it to its inactive disk slot
  (written, read back, checksum-verified, then published). In **GPU control**
  appends are resolved entirely on the device. When the GPU budget cannot hold
  the pages, the append falls back to the host path unchanged. Policy off keeps
  the host path: it is the reference.
- **Reads.** When every written page of a K/V view is on the GPU, attention reads
  a device page table and returns without waiting; otherwise the call borrows the
  pages (and copies cold pages into per-call scratch) and waits for the kernel.
  Both are bit-exact with `staged`.
- **Placement (policy on).** After each attention call, pages read at least twice
  are promoted to GPU, then pinned, then RAM, within budgets; colder pages are
  demoted to their verified disk copy when a tier is full. Attention scratch
  headroom is kept free so promotion never churns against it.
- **Unsupported models.** If a model uses MLA, attention sinks, a KQ bias,
  soft-capping, ALiBi or Grok attention, or the context has several KV streams
  (`-np > 1` without a unified cache), ARGUS leaves the KV cache to llama.cpp and
  logs `ARGUS KV disabled (<feature> is unsupported)`. Misconfiguration (missing
  `-nkvo`, disk KV without `-fa on`) still refuses to start.
- **Crash semantics.** The backing file is unlinked, so KV never outlives the
  process. Within the process, a dirty page's disk slot trails its GPU copy until
  the flusher (or a demotion, which flushes first) publishes it; every published
  slot is verified and the previous slot is kept until the new one verifies.

### 5. Observability

With `ARGUS_KV_STATS_PATH` set, the stats file reports budgets and peaks
(`peak_gpu_bytes`, `peak_pinned_bytes`, `peak_staging_bytes`, …), disk traffic
(`read_bytes`, `written_bytes`), `committed_pages`, policy counters
(`policy_promotions`, `policy_demotions`, `policy_rejected`,
`policy_nanoseconds`), and why each attention call did or did not take the
resident path (`resident_*_accepted`, `resident_*_reject_*`,
`resident_*_cold_pages`). With `ARGUS_KV_PROFILE` it adds inclusive and exclusive
CPU scopes per phase (prefill / decode) and, with `1`, CUDA kernel and copy time.

The benchmark harness `benchmarks/bench_llama_paged_context.py` drives all modes
(`stock-host-kv`, `argus-cuda-off`, `argus-cuda-on`, `argus-cuda-control`, …) with
a needle-in-context check, fresh servers per repeat and the stats above.

---

## Using ARGUS with HuggingFace

The research path from v0.4: ARGUS replaces a model's KV cache with a paged,
tiered cache that can quantize cold pages.

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
    activation_policy=AdaptiveCachePolicy(
        mode="balanced",
        expected_tokens=16_448,  # prompt + maximum requested output
    ),
    pipeline_profile="balanced",
)
```

`page_size=1024` was the best latency/memory compromise on the test machine, not
a universal recommendation. Page lifecycle events are kept in memory (bounded);
set `ARGUS_TRACE_PATH=/path/trace.jsonl` to also write them as JSON lines.
`ARGUS_LOG_LEVEL` sets the logger level.

---

## v0.7 results

Qwen2.5-0.5B-Instruct Q4_K_M, F16 KV, 4096 context, 4016-token prompt, 16
generated tokens, ubatch 64, RTX 3050 Ti Laptop (4 GB), KV on the host (`-nkvo`)
for every mode, GPU/pinned budgets 64 MiB. Profiler off, fresh server per repeat,
one warm-up; median (min–max). Final baseline at commit `95c74fa`
([data](docs/measurements/v070-e10-2026-09-26/)).

| mode | prefill | vs stock | decode | output hash |
|---|---:|---:|---:|---|
| stock llama.cpp, host KV (tensor-core FlashAttention) | 1.336 s (1.331–1.341) | 1.00x | 37.5 tok/s | `007ddc77` |
| ARGUS, GPU control (diagnostic, all KV on GPU) | 1.719 s (1.666–1.759) | 1.29x | 55.6 tok/s | `a152ed56` |
| ARGUS, policy on (GPU/pinned/RAM/disk) | 1.821 s (1.814–1.826) | 1.36x | 48.2 tok/s | `a152ed56` |

Placement counters in the measured request: policy on — 0 promotions, 0
demotions, 0 rejections, 12,768 committed pages, 0 cold pages; GPU control — no
policy activity and zero disk traffic.

**When the KV does not fit the GPU budget** (2 MiB GPU and 2 MiB pinned tiers for
48 MiB of KV, same workload): 47.6 s prefill and 1.0 tok/s decode, 36 promotions,
34 demotions and 2.1 GB read from disk per request. This is the regime ARGUS
exists for — the context survives — but it is disk-bound and far slower.

## How v0.7 got there

GPU-control prefill, same workload, each step a separate measured experiment
([plan](plans/argus-v0.7.0.md)):

| step | change | prefill | decode |
|---|---|---:|---:|
| v0.6.0 baseline | | 2.778 s (2.09x) | 8.0 tok/s |
| E1 | coalesced K-row loads in the exact kernel | 2.709 s | |
| E2 | hardware CRC32C digests, whole-page commits | 2.406 s | 8.1 |
| E4 | V loaded as `half2` | 2.328 s | |
| E5 | single-token decode on the resident path | | **39.3** |
| E7a | KV appends written on the GPU | ~1.98–2.03 s | 43 |
| E8 | appends resolved on the device, no host round trip | −3.5% | |
| E9 | attention reads the store page table without waiting | **1.68 s** | 55 |

Policy on moved from 9.2 s (v0.6/E2) to 2.0 s when appends went to the GPU with
background disk write-back (E7b), then to 1.82 s when attention read its page table
without waiting (E10). Two placement-policy faults found by audit and measurement
were fixed on the way (E6): rewrites no longer drop a page's promotion, and
promotion no longer fills the attention scratch headroom.

Most of the gain came from the data path around attention — waits, host round
trips, CPU encoding — not from the kernel. The exact kernel itself still takes
about 1.1 s of the prefill, against about 0.07 s for stock's FlashAttention.

## Reading the stock comparison

- **Different arithmetic contracts.** Stock llama.cpp here uses FlashAttention on
  tensor cores, with its own accumulation order and precision. ARGUS's kernels
  keep the FP32 accumulation order of ARGUS's staged reference, bit for bit
  (Category 1 exactness, no reassociation, no tensor cores). The two produce
  different bits: the 16-token outputs hash differently (`007ddc77` vs
  `a152ed56`), and both answer the needle question correctly. Speed ratios compare
  two different computations, not two implementations of the same one.
- **One workload.** One model, one GPU, one context length, KV on the host for
  every mode. Nothing here predicts 32K, 262K, another model or another GPU.
- **Diagnostic vs product.** GPU control keeps every page on the GPU with no disk
  backing; it bounds what the datapath can do. Policy on is the mode that
  actually manages tiers.

---

## How it works

```text
llama.cpp (patched)                       HuggingFace model
   |  KV allocation, set_rows, attention      |  cache replacement
   v                                          v
ARGUS disk store (C++/CUDA)              PagedDynamicKVCache (Python + C++)
   |- page descriptors: content / placement revisions, checksums
   |- tiers: GPU | pinned | pageable RAM | O_DIRECT disk (2 verified slots/page)
   |- placement policy (ggml_kv_policy.cpp), budgets owned by the store
   |- GPU appends + background write-back flusher
   '- attention: device page table (wait-free) | borrowed pages | staged reference
```

- **Placement and content are separate revision axes.** A byte-preserving,
  verified move does not invalidate reads; a write does.
- **The store owns budgets.** The policy proposes moves; the store verifies,
  refuses and counts.
- **Page lifetime with queued GPU work.** Page-table entries change only through
  stream-ordered publication; a replaced GPU page is freed with `cudaFree`, which
  on the tested driver waits for all queued device work; host in-place writes wait
  for the last table read. See the [E10 invariants](plans/argus-v0.7.0.md).
- **Source layout.** `argus_cache/csrc/ggml_disk_store.h` (shared store
  internals), `ggml_disk_buffer.cpp` (store lifetime, GGML buffer, host I/O),
  `ggml_disk_gpu.cpp` (GPU residency, appends, write-back, migration),
  `ggml_cuda_attention.cu` (kernels and dispatch), `ggml_kv_policy.cpp`,
  `ggml_paged_attention.cpp` (CPU attention), `ggml_host_buffer.cpp` (mapped mode).

The HuggingFace path's principle is that a logical KV page, its physical placement
and its physical precision are three separate things; compression tiers are
plugins selected by capability. Details: [`docs/architecture.md`](docs/architecture.md).
`AttentionAdapter` keeps model contracts out of the core, and
`argus_cache/models/hybrid_cache.py` states ownership for hybrid models: ARGUS owns
full-attention KV, never linear-attention state.

### Runtime status

| runtime | status | what ARGUS manages |
|---|---|---|
| llama.cpp | Deepest integration; CPU and CUDA attention; bit-exact with its reference | KV allocation, budgets, writes, placement and attention reads |
| HuggingFace Transformers | Research path, measured in v0.4 | The model's KV cache |
| Ollama | External adapter | Nothing inside Ollama; configuration and timing only |
| vLLM | Fails closed | Nothing; a real integration needs vLLM's KV connector ([notes](docs/vllm-verification.md)) |
| SGLang | Not implemented | Nothing |

---

## Benchmark methodology

- **Idle machine.** No other GPU compute process. Background re-indexers such as
  `codebase-memory-mcp` (one per editor/agent session, each respawning
  `--index-worker` processes) were paused with `SIGSTOP` for the duration of every
  final measurement and resumed afterwards: during one release-gate attempt they
  pushed load to 7 and a stock run to 3.1 s against 1.33 s. That attempt is kept,
  labelled, in [`v070-release-gate-2026-09-26/`](docs/measurements/v070-release-gate-2026-09-26/).
- **Same-binary A/B.** Variants are compared under one `llama-server` binary,
  switching `libllama.so` with `LD_LIBRARY_PATH` or an `ARGUS_KV_ATTENTION_PATH`
  reference, with repeats in alternating order.
- **Profiler off** for every timing that is reported; profiled runs are used for
  attribution only.
- **Exactness first.** A change is accepted only if every ARGUS path stays
  bit-exact with `staged` and the output hash is unchanged.

```bash
# Python suite (CUDA device optional for most tests; live Ollama tests need ARGUS_TEST_LIVE=1)
pytest tests/ -q

# Native llama.cpp + CUDA suite
ARGUS_LLAMA_CPP_DIR=/path/to/llama.cpp ARGUS_LLAMA_BUILD=build ARGUS_TEST_CUDA=1 \
ARGUS_TEST_GGUF=/path/to/stories15M.gguf pytest tests/test_native_llama_paged.py -q

# The 4K comparison
python benchmarks/bench_llama_paged_context.py --server /path/to/llama-server \
  --model qwen2.5-0.5b-instruct-q4_k_m.gguf --kv-dir /physical/disk/dir --contexts 4096 \
  --modes stock-host-kv argus-cuda-control argus-cuda-on --kv-type f16 --ubatch 64 \
  --predict 16 --resident-bytes 4194304 --gpu-bytes 67108864 --pinned-bytes 67108864 \
  --warmups 1 --repeats 5 --output result.json
```

## Known limits

- **CUDA path:** F16 KV only (Q8/Q4 KV use CPU attention), head dimension ≤ 256,
  fastest kernels need D = 64, one sequence (one KV stream), CUDA device 0.
- **Wait-free reads** need every written K/V page on the GPU; in policy mode the
  page table is only built when the whole store, the table and scratch headroom fit
  the GPU budget. Otherwise the waiting (borrowing) path is used.
- **Freeing a GPU page relies on `cudaFree` waiting for queued device work**, as it
  does on the tested driver. A stream-ordered allocator would remove that
  device-wide wait; it is not implemented.
- **Policy-on appends still synchronize** with the stream per call; only attention
  is wait-free there.
- **Configuration is environment variables only**; budgets are process-global.
- The llama.cpp patches target one pinned revision.
- The HuggingFace path's Python store and the llama.cpp store are separate
  implementations; per-page mixed precision is not available on llama.cpp.
- Everything above was measured on one GPU (RTX 3050 Ti Laptop), one model and 4K.

---

## Earlier evidence (v0.4–v0.6)

**llama.cpp KV ownership (v0.5).** Allocation, page IDs, write/read counters and a
trace of attention consuming those pages were required before ownership was
claimed. On a real CPU model run: maximum logit difference 0 across eight decode
steps after prefill and state restore
([artifact](docs/measurements/v050-llama-host-ownership-2026-09-15.json)). With
`q8_0` and `q4_0` KV, 4 MiB staging and 1 MiB resident budget: attention and logit
difference 0 over 19 steps including crop and state restore.

**v0.5 mechanism only.** Without a placement policy every read went to disk: one
UI-Mate request read 14.98 GB and took 752 s against 260 s for stock.

**v0.6 first operating point (4K).** Stock 1.332 s, GPU-resident 2.821 s (2.12x),
policy on 9.888 s (7.42x); decode 8.0 / 6.4 tok/s
([checkpoint](docs/measurements/v060-checkpoint-2026-09-18.md)).

**HuggingFace research path (v0.4).** Qwen2.5-0.5B, 1024-token pages: VRAM saving
grows with context (−7.7% at 16K) while TPOT rises to 4.2x baseline; no tested row
showed baseline OOM with ARGUS surviving
([artifact](docs/measurements/downstream-2026-08-14.json),
[analysis](docs/findings-2026-08-14.md)).

**Precision vs latency, engine in isolation.** At 32K, q4_0 holds the context in
44.16 MiB against FP16's 136.16 MiB — 3.1x less memory for 2.14x the latency. The
engine is not wired into the HuggingFace decode path
([artifact](docs/measurements/v040-fused-attention-benchmark.json)).

### Quality (HuggingFace path)

The measured perplexity delta is −0.0176, but only the near-lossless FP8 tier was
reached. INT4, INT2, 1-bit and JL quality are not validated.

### A negative result kept on purpose

An A/B sweep against a local llama-server appeared to show an ARGUS win; the audit
showed ARGUS was never loaded into that process. No claim is drawn from it
([artifact](docs/measurements/argus-ab-cache-comparison-2026-09-04.json)).

---

## Roadmap

- **v0.7 (this release):** the measured 4K gap closed from 2.09x to 1.29x (GPU
  control) and from 7.4x to 1.36x (policy on); decode faster than stock; exactness
  kept throughout. Record: [`plans/argus-v0.7.0.md`](plans/argus-v0.7.0.md).
- **v0.8 (backlog):** making ARGUS a plugin other runtimes can adopt — packaging,
  a configuration API, multi-instance budgets, multi-sequence support — and
  measurements beyond 4K. See [`plans/argus-v0.8.0.md`](plans/argus-v0.8.0.md).

## License

ARGUS is licensed under the [Apache 2.0 License](LICENSE).
