# ARGUS Architecture

ARGUS is a **configurable heterogeneous KV-cache management runtime**. It is a
control layer for cache memory, not a quantization algorithm. The six
compression tiers it ships with are plugins; the inference runtimes it supports
are adapters. Neither is baked into the engine.

---

## 1. Layering

```
┌──────────────────────────────────────────────────────────────┐
│  Public Python API            argus_cache/__init__.py        │
│  PagedDynamicKVCache, PagedDynamicQuantizedCache, TierSpec    │
├──────────────────────────────────────────────────────────────┤
│  Runtime adapters (edge)      argus_cache/adapters/          │
│  vLLM · Ollama · (SGLang: additive)                          │
├──────────────────────────────────────────────────────────────┤
│  Policy & configuration       argus_cache/core/, plugins/    │
│  tier pipeline · eviction · importance · telemetry           │
├──────────────────────────────────────────────────────────────┤
│  Native engine (C++)          argus_cache/csrc/manager.cpp   │
│  page pool · tier cascade · zero-copy host spill · SDPA      │
├──────────────────────────────────────────────────────────────┤
│  Data plane (CUDA / Triton)   csrc/quantization_kernels.cu   │
│  one generic pack/unpack kernel · Triton fused attention     │
└──────────────────────────────────────────────────────────────┘
```

### 1.1 Inside `argus_cache/core/`

`memory_manager.py` was a 2662-line god object holding every responsibility at
once. It is now a coordinator that owns page lifecycle and tier policy, with
five collaborators it delegates to. Each is independently testable, and each
was moved only after characterization tests pinned its existing behavior.

| module | responsibility | lines |
|---|---|---:|
| `memory_manager.py` | page lifecycle, tier cascade policy, attention assembly | 1785 |
| `telemetry.py` | compression/bandwidth accounting, VRAM and fragmentation reports | 397 |
| `granularity.py` | experimental page split/merge (ACTIVE pages only) | 317 |
| `host_spill.py` | lossless spill to pinned host memory, both directions idempotent | 185 |
| `jl_operators.py` | cached JL projection/reconstruction operators | 128 |
| `pool_allocator.py` | per-tier compressed page pools, shaped from capabilities | 131 |
| `outliers.py` | outlier isolation and restoration | 85 |

The split target was five separated responsibilities, not a line count;
`memory_manager.py` landed at 1785 rather than the 1400–1600 originally
estimated, because attention assembly and the cascade policy genuinely belong
to the coordinator and were not split further to hit a number.

`pool_allocator.py` is where the **last name-based dispatch in Python** was
removed. Pool shapes now derive from a tier's declared codec — sub-byte kinds
pack `8 // bits` values per byte, affine kinds get a zero-point pool, signed
kinds store `int8` — so a third-party plugin tier gets a correctly shaped pool.
Previously it matched none of the five hardcoded name branches, received no
pool, and silently fell back to uncompressed spill with no error raised.

**Ownership.** Python owns *decisions* (which page to evict, which tier is
next, when to spill). C++ owns *mechanics* (where bytes live, how they are
packed, when kernels launch). The boundary is crossed synchronously from Python
calls, so the GIL is held for every callback into Python — with one deliberate
exception, the background prefetch worker, which is forbidden from invoking
Python callbacks at all.

---

## 2. The tier codec — how tiers stopped being hardcoded

Previously, `manager.cpp` decided how to compress and decompress a page by
comparing the tier's **name**:

```cpp
if (tier == "fp8")        { launch_dequantize_fp8(...); }
else if (tier == "int8")  { launch_dequantize_int8(...); }
else if (tier == "int4")  { launch_dequantize_int4_flat(...); }
// ... and so on, repeated in four separate call sites
```

That chain existed in `demote_to_next_tier`, `resurrect_page`,
`peek_decompress_page`, and the prefetch worker. Adding a tier meant editing
all four, and a plugin tier could never be more than an uncompressed spill.

A tier's storage format is now described **numerically**, by
`argus::TierCodec` (`csrc/tier_codec.h`):

| field | meaning |
|---|---|
| `kind` | `SignedLinear` · `UnsignedAffine` · `SignPacked` · `Projection` · `Passthrough` |
| `bits` | bits per stored element |
| `pack_factor` | derived: `8/bits` for sub-byte codecs |
| `levels` | derived from kind and bits |
| `compression_ratio` | storage cost relative to fp16 |
| `lossy` | capability metadata |

All four call sites now share exactly two methods — `compress_page()` and
`decompress_page()` — and the five bespoke CUDA kernels collapsed into one
parameterized `dequantize_generic_kernel`. **Adding a tier is a registry entry,
not a new branch.**

The only remaining name comparison in the engine is `tier_name == "active"`,
which is a *state*, not a format.

| file | before | after |
|---|---:|---:|
| `csrc/manager.cpp` | 1207 | 763 |
| `csrc/quantization_kernels.cu` | 183 | 81 |

---

## 3. Plugin architecture

A quantization backend is four methods (`compress`, `decompress`,
`decompress_batch`, `memory_bytes`) plus a `BackendCapabilities` declaration:

```python
BackendCapabilities(
    name="int4",
    effective_bits=4.0,          # cost, used instead of name checks
    lossy=True,
    supported_devices=frozenset({"cuda", "cpu"}),
    supported_dtypes=frozenset({torch.float16, torch.bfloat16, torch.float32}),
    requires_calibration=False,
    supports_reconstruction=True,
    native_codec=NativeCodecSpec(kind="unsigned_affine", bits=4),
)
```

Declaring a `native_codec` is what promotes a plugin from Python-side-only to a
tier the **native engine compresses itself**, using the same generic kernel as
the built-ins. A plugin that declares none still works — its pages spill
losslessly rather than being decoded with a guessed bit layout.

### Capability-based policy

Policy code asks what a backend *costs*, never what it is *called*:

```python
available_quantizers(device="cuda", dtype=torch.float16, max_effective_bits=4.0)
# ['jl', 'int4', 'int2', 'one_bit']  — ordered most-expensive-first
```

Former name checks such as `spec.name == 'jl'` are now `spec.is_projection`,
which reads the backend's declared codec kind. A projection backend registered
under any name is handled correctly.

### Replacing a tier — removing 1-bit

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

unregister_quantizer("one_bit")                       # 1-bit is gone
register_quantizer("ternary", TernaryBackend,
    BackendCapabilities(name="ternary", effective_bits=2.0,
        native_codec=NativeCodecSpec(kind="unsigned_affine", bits=2)))

cache = PagedDynamicKVCache(pipeline=PipelineConfig(tiers=[
    TierSpec(name="fp8",     backend="fp8",     max_pages=1),
    TierSpec(name="ternary", backend="ternary", max_pages=8),
]))
```

The memory manager is not modified. Covered by
`tests/test_plugin_system.py::test_disable_one_bit_and_install_custom_quantizer`.

> **Packing-axis warning.** A backend's `native_codec` describes how the **C++
> engine** stores the tier. The Python backend classes in
> `argus_cache/backends/quantization.py` pack along a *different axis* than the
> native kernels. A page compressed by one must never be decoded by the other.
> Read back native-tier pages with `peek_decompress_page()`.

---

## 4. Runtime adapters

Adapters live at the edge; the core never imports one. Each declares what it
can actually do, and adapters distinguish two fundamentally different cases:

| kind | meaning | example |
|---|---|---|
| `IN_PROCESS` | runtime runs here; ARGUS can own its KV cache | vLLM |
| `EXTERNAL` | separate process with its own cache | Ollama |

Lifecycle is explicit and idempotent — `initialize → activate → deactivate →
shutdown` — with teardown guaranteed non-raising, so an adapter failure can
never leave a runtime half-patched or disturb ARGUS state.

### Ollama (`EXTERNAL`, `manages_kv_cache=False`)

Ollama wraps llama.cpp behind an HTTP server owning its own KV cache in its own
address space. **ARGUS cannot manage it**, and the adapter says so in its
capabilities and in every telemetry payload. What it legitimately provides:
lifecycle-managed connection, model-presence validation with an actionable
error, pass-through of Ollama's own `num_ctx` / `cache_type_k|v` settings, and
per-request timings for use as an external baseline. Transport is injected, so
it is testable without a running server.

**Verification status: verified against a live server, 2026-08-14.**
Ollama `0.32.11` (`ollama-cuda 0.32.11-1`), model `qwen2.5:0.5b`, on the
RTX 3050 Ti Laptop host. Five live tests pass against the real HTTP API —
version probe, real decode timings, actionable rejection of a missing model,
telemetry reporting the loaded model, and three activate/generate/deactivate
cycles without drift. The adapter needed no changes: its assumptions about
field names and endpoint shapes matched the server as written.

Recorded baseline: `docs/measurements/ollama-2026-08-14.json`, regenerate with
`python scripts/measure_ollama.py --output <path> --date <date>`.

| metric | value |
|---|---|
| median decode throughput | 264.0 tok/s |
| range across 5 runs | 242.7 – 266.1 tok/s |
| prompt eval | 0.004 – 0.031 s |
| tokens generated per run | 64 |

**Claim class: `runtime`, and external.** These describe an unmodified Ollama
server. ARGUS does not manage this KV cache, so these numbers are not an ARGUS
result and no speedup may be derived from them. They exist as a reference point
for the adapter, nothing more.

### vLLM (`IN_PROCESS`, experimental)

The previous integration divided vLLM's `block_tables` by 4 or 16. Those
entries are physical block **indices**, not byte offsets — dividing them aliases
unrelated sequences onto shared blocks. It compressed nothing and corrupted
attention. It has been removed; `inject_argus_to_vllm()` now raises with that
explanation.

`VLLMAdapter` instead: guards on a supported vLLM version range rather than
guessing at an unknown seam; patches one recorded attribute per target and
restores it exactly (`is_fully_restored()`); and rolls back **every** patch if
any target fails, because a half-patched vLLM mixing ARGUS and native attention
is worse than no patch. It does **not** replace vLLM's block allocator.

### Adding SGLang

Subclass `RuntimeAdapter`, implement `_do_initialize/_do_activate/_do_deactivate`,
add one line to `_ADAPTERS` in `adapters/__init__.py`. No engine change.

---

## 5. Benchmarks

Numbers are separated by the kind of claim they support. **Synthetic
reconstruction fidelity is not downstream model accuracy** and is never
presented as such.

Reproduce with:

```bash
python benchmarks/bench_native_runtime.py --json results.json
```

Every number in §5.1, §5.2 and §5.5 comes from
`docs/measurements/native-2026-08-14.json`, recorded against commit
`2951715` on a clean tree with the extension rebuilt from scratch. Provenance
for that tree is in `docs/measurements/baseline-2026-08-14-post.json`
(`git_dirty: false`, 191 passed / 1 skipped).

**Environment:** RTX 3050 Ti Laptop (4 GB, SM 8.6) · CUDA 13.0 · torch
2.12.0+cu130 · Triton 3.7.0 · Python 3.14.7 · seed 1234 · fp16 ·
batch 1 · 8 heads · head_dim 64 · page_size 128.

### 5.1 Synthetic reconstruction + codec runtime

Random Gaussian tensors. Fidelity here measures the codec, **not** the model.

| tier | ratio | compress (ms) | decompress (ms, median) | rel. L2 | cosine |
|---|---:|---:|---:|---:|---:|
| fp8 | 2.00× | 57.52 | 0.0729 | 0.0097 | 1.0000 |
| int8 | 2.00× | 1.89 | 0.0752 | 0.0097 | 1.0000 |
| int4 | 4.00× | 11.48 | 0.0523 | 0.1593 | 0.9876 |
| int2 | 8.00× | 1.87 | 0.0403 | 0.8332 | 0.8135 |
| one_bit | 16.00× | 13.01 | 0.0359 | 0.6023 | 0.7983 |
| jl | 4.00× | 225.28 | 0.1013 | 1.1733 | 0.2666 |

Notes, stated rather than glossed:

* **fp8/int8 compress timings include one-time CUDA context and pool warmup**
  on the first tier measured; int8 (1.89 ms) is the steady-state cost of the
  same operation.
* **JL fidelity on random input is not meaningful.** Reconstructing a 4×
  projection of white noise is information-theoretically impossible, so
  `cos = 0.27` here is a property of the input, not of the tier. It is
  evaluated on real activations in §5.4 instead.
* **int2's error matches theory.** Round-to-nearest over 4 levels spanning
  ~6.6σ gives RMS ≈ step/√12 ≈ 0.64 relative; the measured 0.83 is within
  seed variance for a single page.

### 5.2 Decode latency and peak VRAM vs context

int4 tier, single decode step, median over 30 steps.

| context tokens | page size | pages | ms/step | peak VRAM (MiB) |
|---:|---:|---:|---:|---:|
| 256 | 128 | 2 | 0.72 | 51.6 |
| 1024 | 128 | 8 | 4.74 | 61.6 |
| 2048 | 128 | 16 | 10.61 | 73.1 |
| 512 | 256 | 2 | 0.78 | 75.3 |
| 2048 | 256 | 8 | 5.24 | 95.3 |
| 4096 | 256 | 16 | 11.35 | 118.3 |
| 1024 | 512 | 2 | 0.94 | 122.6 |
| 4096 | 512 | 8 | 7.97 | 162.7 |
| 8192 | 512 | 16 | 14.78 | 208.7 |

Decode cost grows roughly linearly in resident pages, because every compressed
page is decompressed and concatenated each step. Larger pages amortize better
at equal context (4096 tokens: 11.35 ms at page 256 vs 7.97 ms at page 512).

### 5.3 Not measured

The following are **not** produced by this run and no claim is made about them:
TTFT and TPOT on a real model, perplexity delta, NIAH/RULER retrieval, vLLM
throughput, CPU-spill overhead under real memory pressure, multi-GPU. Any
figure for these in older documentation predates this refactor and has not
been revalidated.

### 5.4 The JL tier on real activations — verdict

`docs/measurements/jl-2026-08-14.json`, regenerate with:

```bash
python benchmarks/bench_jl_fidelity.py --json <path> --date <date> --tokens 512
```

Real K activations from Qwen2.5-0.5B-Instruct, all 24 layers, measured
**per 64-token page** (the unit the tier actually compresses), against two
equal-budget 4× controls.

**First, a correction to how this tier was described.** JL was justified above
by "the low-rank structure of real KV tensors". That reasoning was wrong. The
operator is a *smoothness*-regularized least-squares inverse — a 1-D Laplacian
prior along the sequence axis — and it recovers a rank-4 random signal no
better than white noise (rel. error 1.106 vs 1.111, measured). Low effective
rank is not what it exploits, and `tests/test_jl_operators.py` now asserts
this so the framing cannot quietly drift back.

What it does exploit is **token-to-token smoothness**, and the measurement is
unambiguous: relative reconstruction error tracks page roughness with
**r = 0.988** across all 24 layers.

| result | value |
|---|---|
| JL median rel. L2 | **0.413** |
| int2 single-scale control | 0.630 |
| int2 as shipped (per-group scales) | 0.565 |
| JL cosine range | 0.841 – 0.9996 |
| JL beats shipped int2 | **20 / 24 layers** |
| median page roughness | 0.427 |

**Verdict: the tier is justified, and it is kept.** It beats the strictly
harder control — the int2 backend ARGUS actually ships, not a strawman — on
20 of 24 layers at the same storage cost. The four losses (layers 6, 13, 16,
18) are precisely the four roughest pages, which is the failure mode the
prior predicts rather than a surprise.

Two honest limits on this result. It is one model, one prompt, keys only —
values were not measured. And it is claim class `reconstruction`: JL wins on
tensor fidelity, which is not the same as winning on perplexity. §5.3 still
applies.

---

### 5.5 Verbose logging overhead

`ARGUS_VERBOSE` gates every hot-path `std::cout` in the native engine. It is
off by default, and this is why:

| logging | ms/step | overhead |
|---|---:|---:|
| off (default) | 1.612 | — |
| on | 1.783 | **+10.6 %** |

Stream insertion on the demotion path is not free; leaving it always-on cost
over a tenth of decode time.

## 6. Correctness fixes found during the refactor

Both were pre-existing and are covered by tests.

1. **Truncation instead of rounding.** Quantization cast float→int directly,
   which truncates toward zero and biases every value by up to half a
   quantization step, roughly doubling reconstruction error on the low-bit
   affine tiers. Now rounds to nearest. `int4` relative L2 went from ≈0.27 to
   0.159.

2. **1-bit used the wrong magnitude.** `SignPacked` reconstructs as ±`scale`,
   and used the per-page **maximum** as that magnitude. The L2-optimal scalar
   is the **mean absolute value** (`argmin_s E[(|x|−s)²] = E[|x|]`); using the
   max inflated the reconstructed norm ~3.5× for Gaussian input. Relative L2
   dropped from **3.51 to 0.60** at identical storage cost and identical
   cosine similarity.

Additionally, the prefetch worker previously cached **undefined tensors** when
a JL page had no cached reconstruction operator (it cannot build one — no GIL
on that thread), which surfaced later as an empty page during attention. It now
skips the speculation.

---

## 7. Testing

```bash
pytest tests/ -q     # 132 passed, 2 skipped
```

`tests/test_plugin_system.py` (32 tests) covers registration, removal, invalid
configurations, capability filtering, tier replacement, single-tier pipelines,
native-codec propagation to C++, and per-tier round-trip fidelity with error
budgets derived from quantization theory.

`tests/test_adapters.py` (32 tests, 1 skipped) covers both adapters' lifecycle
idempotency, failure isolation, all-or-nothing rollback, exact restoration
across repeated activate/deactivate cycles, and telemetry that never overclaims.
The single skip requires an installed vLLM; the Ollama live tests run.
