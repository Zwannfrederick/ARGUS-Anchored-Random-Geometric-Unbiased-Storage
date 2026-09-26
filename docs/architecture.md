# ARGUS Architecture (v0.7.0)

*Türkçe: [architecture_TR.md](architecture_TR.md)*

ARGUS is a KV-cache manager. It decides where every KV page lives (GPU, pinned
RAM, pageable RAM, disk), under hard budgets, and serves attention reads from
wherever the page is — without changing the model's arithmetic. It is a control
layer for cache memory, not a quantization algorithm.

It has two integration paths that share this principle but not code:

| path | store | status in v0.7 |
|---|---|---|
| **llama.cpp** (patched `libllama` / `ggml-cuda`) | C++/CUDA disk store in `argus_cache/csrc/ggml_*` | The product path. CPU and CUDA attention, bit-exact with its own reference |
| **HuggingFace Transformers** | `PagedDynamicKVCache` (Python + `argus_cpp_backend`) | Research path, last measured in v0.4 |

Every behavioural claim below ends with the test that fails if it stops being true,
and every number links to the recorded run it came from.

How to build, configure and run it: [README](../README.md). Why each design
choice was made, with measurements: [`plans/argus-v0.7.0.md`](../plans/argus-v0.7.0.md).

---

## 1. Map

```text
┌───────────────────────── llama.cpp path ─────────────────────────┐
│ llama.cpp (host-kv.patch + cuda-kv.patch)                        │
│   KV buffer type │ set_rows │ attention op │ state save/restore  │
│        │             │           │                               │
│        v             v           v                               │
│ ggml_disk_buffer.cpp   store lifetime, GGML buffer, host I/O     │
│ ggml_disk_gpu.cpp      GPU residency, appends, write-back, moves │
│ ggml_kv_policy.cpp     placement decisions within budgets        │
│ ggml_cuda_attention.cu CUDA kernels + dispatch                   │
│ ggml_paged_attention.cpp  CPU block attention                    │
│ ggml_host_buffer.cpp   mapped host KV (no disk store)            │
│        │                                                         │
│   tiers: GPU │ pinned │ pageable RAM │ O_DIRECT disk (2 slots)   │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────── HuggingFace path ────────────────────────┐
│ Public API       argus_cache/__init__.py                         │
│ Model contracts  argus_cache/models/  (AttentionAdapter, hybrid) │
│ Policy           argus_cache/core/, plugins/                     │
│ Native engine    csrc/manager.cpp, tier_codec, zero_copy_pool    │
│ Kernels          csrc/quantization_kernels.cu, Triton            │
└──────────────────────────────────────────────────────────────────┘
Runtime adapters (edge, no KV ownership): Ollama, vLLM (fails closed),
Anthropic-Messages ↔ OpenAI gateway.
```

---

## 2. The llama.cpp path

### 2.1 Where ARGUS hooks into llama.cpp

Two patches against one pinned llama.cpp revision
([`integrations/llama.cpp/`](../integrations/llama.cpp/README.md)):

| hook | patch | what ARGUS takes over |
|---|---|---|
| KV buffer type (`llama_kv_cache` constructor) | host-kv | With `ARGUS_KV_DIR` set, K/V tensors are allocated in ARGUS's buffer type: the disk store, or the mapped host buffer |
| `cpy_k` / `cpy_v` | host-kv | New rows go through `argus_ggml_disk_set_rows` instead of GGML's `set_rows` |
| attention graph | host-kv | `argus_ggml_paged_attention` replaces flash attention over ARGUS tensors |
| state `write_tensor` / `read_tensor` | host-kv | Save / restore stream disk tensors through a 4 KiB staging block, never a full-size copy |
| CUDA custom op | cuda-kv | `ggml_backend_cuda_register_custom_op`: one external op (marker + compute + buffer type) that the CUDA backend accepts on device 0 and never captures in a CUDA graph |
| load-time fallback | host-kv | Models ARGUS cannot serve keep llama.cpp's own KV (§2.9) |

All ARGUS CUDA operations (attention, set_rows) share one registered marker and
are told apart by a kind tag stored in the op parameters.

Evidence: [host ownership run](measurements/v050-llama-host-ownership-2026-09-15.json) (logit difference 0), end-to-end server tests in [`tests/test_native_llama_paged.py`](../tests/test_native_llama_paged.py).

### 2.2 The store

One store per K or V tensor buffer, carved into **4 KiB pages**.

- **Backing file.** Created in `ARGUS_KV_DIR` and unlinked immediately, opened
  `O_DIRECT`: KV never outlives the process and never goes through the page cache.
- **Two slots per page.** A write goes to the inactive slot, is read back and
  checksum-verified (CRC32C by default), and only then becomes active. The
  previous slot stays valid until the new one verifies. That is why direct disk
  mode reserves 2x the KV size.
- **Page descriptor.** `content_revision` (changes on writes), `placement_revision`
  (changes on moves), checksum, active slot, codec, access history, and — on CUDA
  builds — the resident copy plus `dirty` / `digest_pending` flags.
- **Two revision axes.** A verified, byte-preserving move does not invalidate
  readers; a write does.
- **Budgets are owned by the store.** Metadata (`ARGUS_KV_RESIDENT_BYTES`), host
  scratch (`ARGUS_KV_STAGING_BYTES`), GPU (`ARGUS_KV_GPU_BYTES`), pinned
  (`ARGUS_KV_PINNED_BYTES`), RAM (`ARGUS_KV_RAM_BYTES`). Exceeding one is an
  explicit error or a refused move — never silent growth.
- **Locks.** `mutex` guards descriptors and residency; `io_mutex` guards disk-slot
  I/O. The active slot changes only with both held (in that order), so the
  write-back flusher can write a slot without blocking attention.

Shared internals: `ggml_disk_store.h` (namespace `argus_disk`, not an API).

Evidence: slot publication (a failed write never changes the published slot) and corruption refusal in [`tests/cpp/test_ggml_disk_buffer.cpp`](../tests/cpp/test_ggml_disk_buffer.cpp); [`test_direct_disk_pages_preserve_data_and_enforce_budgets`](../tests/test_native_llama_paged.py).

### 2.3 Tiers and the placement policy

```text
GPU  ──  pinned RAM  ──  pageable RAM  ──  disk (verified slot, always the truth
 fastest                                   for published content)
```

With `ARGUS_KV_POLICY=on`, `ggml_kv_policy.cpp` runs around every attention call:

- **prepare** gathers the pages the call will read;
- **observe** records accesses (a 32-cell history per page) and moves pages:
  pages read at least twice are promoted toward the GPU, then pinned, then RAM;
  colder pages are demoted to their verified disk copy when a tier is full.
- Promotion keeps the attention scratch headroom free
  (`argus_kv_policy_headroom`), so it never churns against the call's own scratch.
- The policy proposes; the store verifies, refuses and counts
  (`policy_promotions`, `policy_demotions`, `policy_rejected`).

`move_page` copies, verifies, swaps the descriptor and only then frees the old
copy. A dirty page is flushed before it is demoted.

Evidence: [`test_cuda_policy_off_on_preserves_outputs_and_budgets`, `test_cuda_mixed_resident_path_preserves_policy_semantics`](../tests/test_native_llama_paged.py); both policy fixes measured in [E6](measurements/v070-e6-policy-logic-2026-09-25.md).

### 2.4 Write path (appends)

`set_rows_compute` tries three routes, in order:

1. **Device-resolved** (GPU control, every page already on the GPU): the encode
   kernel reads row indices on the device through the store's page-address table.
   An out-of-range row sets a pinned error flag that the next append refuses.
2. **Host-resolved GPU append** (policy on, CUDA, F16, rows inside one page): the
   touched pages get a GPU copy — updated in place, uploaded, or zeroed — and are
   marked **dirty**. Only taken when the GPU budget holds the pages, the row table
   and the policy's headroom.
3. **Host append** (policy off, or anything the above cannot hold): the reference
   path, written through the disk slots.

F32→F16 conversion on the GPU uses round-to-nearest-even, byte-identical to GGML's
CPU `from_float`.

Evidence: [`check_gpu_set_rows`](../tests/cpp/test_ggml_cuda_mechanism.cpp) compares every GPU-encoded row with GGML's `from_float` and checks out-of-range refusal; timings in [E7a](measurements/v070-e7a-gpu-set-rows-2026-09-25.md) and [E8](measurements/v070-e8-2026-09-26/).

### 2.5 Write-back

Each disk-backed store that received a GPU append owns a **Flusher** thread
(`ggml_disk_gpu.cpp`). For each dirty page it:

1. snapshots the GPU copy under `mutex`;
2. writes the inactive slot, reads it back and verifies it under `io_mutex` only;
3. publishes the slot under both locks — unless the page changed meanwhile, in
   which case it stays dirty for the next round.

Host readers of a GPU-written page compute its pending digest; a demotion flushes
the page synchronously first, and `argus_disk_flush` drains a store on demand.

Evidence: [`check_gpu_write_back`](../tests/cpp/test_ggml_cuda_mechanism.cpp); [E7b](measurements/v070-e7b-write-back-2026-09-25.md) records the regression that the separate I/O lock fixed (attention 1.53 → 5.94 s while the flusher held the store lock).

### 2.6 Read path (attention)

CUDA dispatch, first route that accepts wins:

| route | when | waits? |
|---|---|---|
| `table_attention` | GPU-control store with a page table | no |
| `policy_table_attention` | policy store whose table exists and every written K/V page is on the GPU | no |
| `try_resident` (borrow) | pages are borrowed where they live; cold pages copied into per-call scratch | yes, for the kernel |
| staged | the reference: tiles staged through host memory, transfer overlapped with compute | yes |

Kernels (`ARGUS_KV_ATTENTION_PATH`): `cells-v2` is the default (lane-per-cell
D = 64, coalesced K loads, `half2` V loads) and serves prefill and single-token
decode; `cells-kc`, `cells-mlp`, `cells`, `batched`, `direct` and `staged` are
kept as references. **Every path is bit-exact with `staged`.**

Evidence: [`tests/cpp/test_ggml_cuda_mechanism.cpp`](../tests/cpp/test_ggml_cuda_mechanism.cpp) runs `direct`, `batched`, `cells`, `cells-mlp`, `cells-kc` and `cells-v2` against `staged` for prefill, decode and mixed residency and requires equal float vectors; the 4K runs record hash `a152ed56` for every ARGUS mode ([final baseline](measurements/v070-e10-2026-09-26/final-baseline.json)).

The CPU path (`ggml_paged_attention.cpp`) computes exact causal attention one
cell block at a time, merging blocks with log-sum-exp; cold blocks go back to
storage after they are read.

### 2.7 Page lifetime with queued GPU work

Wait-free reads mean kernels can still be queued when the host moves a page.
The invariants that keep that safe (derivation: [E10](../plans/argus-v0.7.0.md)):

- **Stream-ordered publication.** A policy page-table entry changes only through
  `publish_entry`, a one-thread kernel on the attention stream, after every change
  of a page's GPU residency (promotion, demotion, append attach, failed host
  in-place write). Kernels queued earlier keep the entry they saw.
- **The table exists only if everything fits.** Every page, the table and the
  scratch headroom must fit the GPU budget together, so the table never displaces
  a page and placement decisions are those of the borrowing path.
- **Frees wait for queued work.** Replaced GPU pages are freed with `cudaFree`,
  which on the tested driver waits for all queued device work. A stream-ordered
  allocator (`cudaFreeAsync`) would remove that device-wide wait; not implemented.
- **Host writes wait for readers.** A host in-place write to a GPU page waits for
  the table's `read_event`; host reads of GPU-control pages wait for `write_event`.
- **No generations, epochs or refcounts.** `clear` and teardown free pages and the
  table the same way.

Evidence: [`check_policy_table`](../tests/cpp/test_ggml_cuda_mechanism.cpp) queues kernels behind a 512 MiB device copy, moves pages meanwhile and requires results equal to `staged`; [`cudafree-sync-probe.cu`](measurements/v070-e10-2026-09-26/cudafree-sync-probe.cu) measures `cudaFree` waiting (about 12 ms behind a 1 GiB copy).

### 2.8 Modes

| mode | store | attention |
|---|---|---|
| Mapped host KV | `ggml_host_buffer.cpp`, file-backed mapping | llama.cpp's own |
| CPU block attention | mapped | ARGUS CPU |
| Direct disk KV | disk store | ARGUS CPU |
| CUDA, policy off | disk store, every read from disk | ARGUS CUDA (reference) |
| **CUDA, policy on** | disk store + tiers | ARGUS CUDA — the product mode |
| GPU control | GPU-authoritative, no disk | ARGUS CUDA — diagnostic ceiling |

Settings for each: [README § Pick a mode](../README.md#2-pick-a-mode).

### 2.9 Failure semantics

- **Unsupported models fall back at load.** MLA, attention sinks, KQ bias,
  soft-capping, ALiBi, Grok attention or several KV streams (`-np > 1` without a
  unified cache): llama.cpp keeps its own KV and logs
  `ARGUS KV disabled (<feature> is unsupported)`.
- **Misconfiguration refuses to start** (missing `-nkvo`, disk KV without `-fa on`,
  unknown setting values).
- **Budgets refuse, they do not grow.** A move that does not fit is rejected and
  counted; an append that does not fit on the GPU takes the host path.
- **Corruption is detected, not tolerated.** A slot that fails its checksum is
  never published; reads of an unverifiable page fail loudly.
- **Crash.** The backing file is unlinked; nothing persists. Within the process a
  dirty page's disk slot trails its GPU copy until flushed.

Evidence: [`test_unsupported_model_falls_back_to_stock_kv_at_load`](../tests/test_native_llama_paged.py); corruption refusal in [`tests/cpp/test_ggml_disk_buffer.cpp`](../tests/cpp/test_ggml_disk_buffer.cpp).

### 2.10 Source map

| file | lines | responsibility |
|---|---:|---|
| `ggml_disk_store.h` | 107 | Store and page structs, shared internals |
| `ggml_disk_buffer.{h,cpp}` | 103 + 783 | Store lifetime, GGML buffer interface, host reads/writes, slots, stats |
| `ggml_disk_gpu.cpp` | 707 | GPU residency, appends, flusher, `move_page`, policy page table |
| `ggml_cuda_attention.{h,cu}` | 115 + 1010 | Kernels, set_rows/attention dispatch, events |
| `ggml_kv_policy.{h,cpp}` | 14 + 131 | Placement policy |
| `ggml_paged_attention.cpp` | 427 | CPU block attention |
| `ggml_host_buffer.{h,cpp}` | 26 + 297 | Mapped host KV |
| `ggml_profile.h` | 152 | CPU scopes and CUDA event profiling |
| `integrations/llama.cpp/*.patch` | 259 + 144 | llama.cpp hooks |

Observability (`ARGUS_KV_STATS_PATH`, `ARGUS_KV_PROFILE`):
[README § Observability](../README.md#5-observability).

---

## 3. The HuggingFace path

### 3.1 Layering and ownership

**Python owns decisions** (which page to evict, which tier is next, when to
spill). **C++ owns mechanics** (where bytes live, how they are packed, when
kernels launch). Calls cross the boundary synchronously with the GIL held; the
background prefetch worker is forbidden from calling into Python.

The principle: a logical KV page, its physical placement and its physical
precision are three separate things.

| module (`argus_cache/core/`) | responsibility | lines |
|---|---|---:|
| `memory_manager.py` | Coordinator: page lifecycle, tier cascade, attention assembly | 1975 |
| `telemetry.py` | Compression / bandwidth accounting, VRAM and fragmentation reports | 397 |
| `granularity.py` | Experimental page split / merge | 334 |
| `activation.py` | Pressure-aware routing: exact-cache bypass until ARGUS pays off | 186 |
| `host_spill.py` | Lossless spill to pinned host memory | 185 |
| `pool_allocator.py` | Per-tier compressed pools shaped from the tier's codec | 131 |
| `jl_operators.py` | Cached JL projection / reconstruction operators | 128 |
| `outliers.py` | Outlier isolation and restoration | 85 |
| `page_table.py`, `backend_pool.py`, `direct_attention.py` | SoA descriptor table, contiguous q8_0/q4_0 pools, exact online-softmax attention over them (measured in isolation, not on the HF decode path) | 220 / 130 / 197 |
| `page_store.py`, `disk_pool.py` | Transactional placement across bounded pools; compressed disk pages with atomic replacement | 146 / 158 |

### 3.2 The tier codec

A tier's storage format is described numerically by `argus::TierCodec`
(`csrc/tier_codec.h`), never by its name:

| field | meaning |
|---|---|
| `kind` | `SignedLinear` · `UnsignedAffine` · `SignPacked` · `Projection` · `Passthrough` |
| `bits` | bits per stored element |
| `pack_factor` | derived: `8/bits` for sub-byte codecs |
| `levels` | derived from kind and bits |
| `compression_ratio` | storage cost relative to fp16 |
| `lossy` | capability metadata |

Every demotion, resurrection, peek and prefetch goes through `compress_page()` /
`decompress_page()` and one parameterized `dequantize_generic_kernel`. **Adding a
tier is a registry entry, not a new branch.**

### 3.3 Plugins

A quantization backend is four methods (`compress`, `decompress`,
`decompress_batch`, `memory_bytes`) plus a `BackendCapabilities` declaration.
Declaring a `native_codec` lets the native engine compress the tier itself; a
plugin without one still works and spills losslessly.

```python
from argus_cache import (
    PagedDynamicKVCache, PipelineConfig, TierSpec,
    BackendCapabilities, NativeCodecSpec,
    register_quantizer, unregister_quantizer,
)

class TernaryBackend:
    def compress(self, tensor, **kw):
        scale = tensor.abs().amax().clamp_min(1e-8)
        return {"q": torch.round(tensor / scale).clamp(-1, 1).to(torch.int8),
                "scales": scale}
    def decompress(self, c, **kw):
        return (c["q"].to(torch.float32) * c["scales"]).to(torch.float16)
    def decompress_batch(self, cs, **kw):
        return [self.decompress(c, **kw) for c in cs]
    def memory_bytes(self, c):
        return c["q"].nelement() * c["q"].element_size()

unregister_quantizer("one_bit")
register_quantizer("ternary", TernaryBackend,
    BackendCapabilities(name="ternary", effective_bits=2.0,
        native_codec=NativeCodecSpec(kind="unsigned_affine", bits=2)))

cache = PagedDynamicKVCache(pipeline=PipelineConfig(tiers=[
    TierSpec(name="fp8",     backend="fp8",     max_pages=1),
    TierSpec(name="ternary", backend="ternary", max_pages=8),
]))
```

Policy asks what a backend *costs*, never what it is *called*:
`available_quantizers(device="cuda", dtype=torch.float16, max_effective_bits=4.0)`.

> **Packing-axis warning.** The Python backends in
> `argus_cache/backends/quantization.py` pack along the sequence axis; the native
> kernels pack along `head_dim`. A page compressed by one must never be decoded by
> the other. Read native-tier pages back with `peek_decompress_page()`.

### 3.4 Model contracts

- `models/hf_attention.py` registers `argus` through Transformers'
  `AttentionInterface`. A model-aware `AttentionAdapter` registry decides query
  layout, eligibility and output contract. Qwen2 full-attention, eval-mode,
  unmasked single-token decode is the validated native contract; everything else
  reconstructs K/V or keeps the model's own attention (fail closed).
- `models/hybrid_cache.py` states ownership for hybrid models: ARGUS owns
  full-attention KV, never linear-attention recurrent state.

### 3.5 Runtime adapters

Adapters live at the edge; the core never imports one. Lifecycle is explicit and
idempotent (`initialize → activate → deactivate → shutdown`), teardown never raises.

| adapter | kind | manages KV? |
|---|---|---|
| Ollama | `EXTERNAL` | No. Configuration pass-through and timings only; verified against a live server (2026-08-14) |
| vLLM | `IN_PROCESS` | No — fails closed. A real integration needs `KVConnectorBase_V1` and/or a custom `AttentionBackend` ([notes](vllm-verification.md)) |
| Messages gateway | HTTP | No. Translates Anthropic Messages ↔ OpenAI chat completions for a local `llama-server` |
| SGLang | — | Not implemented; a subclass of `RuntimeAdapter` plus one registry line |

---

## 4. Measurements

- **v0.7 (llama.cpp, 4K, Qwen2.5-0.5B, RTX 3050 Ti Laptop):** stock 1.336 s;
  GPU control 1.719 s (1.29x), decode 55.6 tok/s; policy on 1.821 s (1.36x),
  decode 48.2 tok/s; stock decode 37.5 tok/s. Methodology and caveats:
  [README § v0.7 results](../README.md#v07-results).
- **Per-experiment records** (E1–E10): [`plans/argus-v0.7.0.md`](../plans/argus-v0.7.0.md)
  and [`docs/measurements/`](measurements/README.md).

### 4.1 Historical: the HuggingFace path (v0.4)

Recorded 2026-08-14, RTX 3050 Ti Laptop, CUDA 13.0, torch 2.12.0. Kept because
the HF path has not been re-measured since.

**Codec runtime and synthetic fidelity** (random Gaussian input — measures the
codec, not the model; [artifact](measurements/native-2026-08-14.json)):

| tier | ratio | decompress (ms) | rel. L2 | cosine |
|---|---:|---:|---:|---:|
| fp8 | 2.00× | 0.0729 | 0.0097 | 1.0000 |
| int8 | 2.00× | 0.0752 | 0.0097 | 1.0000 |
| int4 | 4.00× | 0.0523 | 0.1593 | 0.9876 |
| int2 | 8.00× | 0.0403 | 0.8332 | 0.8135 |
| one_bit | 16.00× | 0.0359 | 0.6023 | 0.7983 |
| jl | 4.00× | 0.1013 | 1.1733 | 0.2666 |

JL on white noise is meaningless by construction. On real Qwen2.5-0.5B keys, JL
(a smoothness prior along the sequence axis, not a low-rank method) reached median
rel. L2 0.413 against 0.565 for the shipped int2, winning 20 of 24 layers
([artifact](measurements/jl-2026-08-14.json)).

**End to end** (Qwen2.5-0.5B-Instruct, FP16, page 1024;
[artifact](measurements/downstream-2026-08-14.json)):

| context | baseline TTFT | ARGUS TTFT | baseline TPOT | ARGUS TPOT | baseline VRAM | ARGUS VRAM |
|---:|---:|---:|---:|---:|---:|---:|
| 4,096 | 0.316 s | 0.385 s | 19.14 ms | 40.21 ms | 1149.4 MiB | 1137.4 MiB |
| 16,384 | 1.819 s | 3.992 s | 18.84 ms | 79.78 ms | 1722.5 MiB | 1590.6 MiB |

VRAM saving grows with context (−7.7% at 16K) while TPOT rises to 4.2x: the HF
interface needs a contiguous K/V tensor, so stored pages are decompressed every
step. Perplexity 33.9169 baseline vs 33.8993 ARGUS (only ACTIVE + FP8 were
occupied, so this validates no lossy archival tier).

**Direct paged attention in isolation** (24 query heads, 4 KV heads, head_dim 256;
[artifact](measurements/v040-fused-attention-benchmark.json)): at 32K, q4_0 costs
2.14x the latency of FP16 and holds the context in 3.1x less memory. Runtime
class only, not an end-to-end result.

**A negative result, kept.** The 2026-09-04 llama.cpp A/B
([artifact](measurements/argus-ab-cache-comparison-2026-09-04.json)) never loaded
ARGUS into the server (`argus_maps_count: 0`); the gap it showed came from a
prompt-prefix cache. It predates the llama.cpp integration (v0.5+) and supports
no claim.

---

## 5. Testing

```bash
# Python suite (live Ollama tests need ARGUS_TEST_LIVE=1)
pytest tests/ -q

# Native llama.cpp + CUDA suite
ARGUS_LLAMA_CPP_DIR=/path/to/llama.cpp ARGUS_LLAMA_BUILD=build ARGUS_TEST_CUDA=1 \
ARGUS_TEST_GGUF=/path/to/stories15M.gguf pytest tests/test_native_llama_paged.py -q
```

- `tests/cpp/test_ggml_cuda_mechanism.cpp` — GPU set_rows parity and refusal,
  write-back, scratch headroom, and an adversarial policy-table test that queues
  kernels behind a 512 MiB copy while pages move.
- `tests/cpp/test_llama_paged_attention.cpp` — ARGUS against stock llama.cpp on a
  real model, including crop and state save / restore.
- `tests/test_native_llama_paged.py` — end-to-end `llama-server` runs per mode,
  including the load-time fallback.
- `tests/test_plugin_system.py` — registration, removal, capability filtering,
  tier replacement and native-codec propagation.
