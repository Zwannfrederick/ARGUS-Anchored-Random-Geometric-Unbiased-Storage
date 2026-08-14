# Changelog

## Unreleased — downstream revalidation and memory-path repair

### Fixed
- Removed persistent decompressed FP16 KV mirrors. Attention assembly now
  fills one transient output tensor directly from native pages.
- Streamed complete prefill pages into the native manager instead of retaining
  a full-prompt Python staging buffer per layer.
- Stopped allocating unused Python tier pools beside native-owned storage;
  legacy pools now allocate only when the legacy path is used.
- Native callbacks now hold weak cache references, and explicit teardown clears
  them immediately. Completed caches can no longer remain alive through a
  C++→Python→C++ reference cycle or contaminate later benchmark arms.
- Made JL operators lazy, avoiding per-layer projection setup until a page
  actually reaches the archival tier.
- Fixed the downstream harness to retain only final-position prefill logits,
  warm up both arms, release caches between arms, and record exact provenance.

### Changed
- `VLLMAdapter` now fails closed. A probe against vLLM 0.27.1 confirmed the
  model-class forward wrapper did not own or modify vLLM KV blocks. The adapter
  reports `manages_kv_cache=False` until a `KVConnectorBase_V1` and/or custom
  `AttentionBackend` implementation exists.
- The package root uses lazy exports so adapter inspection works in an isolated
  environment with a different torch ABI from the compiled ARGUS extension.
- Withdrawn vLLM/OOM/throughput claims were removed from both READMEs and
  replaced with the revalidated downstream result.

### Measured
- Qwen2.5-0.5B-Instruct on RTX 3050 Ti Laptop: at 16K, peak VRAM falls from
  1722.5 to 1590.6 MiB (7.7%), while TPOT rises from 18.84 to 79.78 ms (4.2x).
  This proves a memory reduction, not baseline-OOM survival.
- Token-by-token perplexity delta is -0.0176 with only ACTIVE and FP8 tiers
  occupied; no downstream claim is made for deeper lossy tiers.

## v0.3.0 — Native tier-codec engine, plugin system, runtime adapters

### Added
- Generic tier codec in C++ (`csrc/tier_codec.{h,cpp}`): a tier's storage
  format is described numerically (kind, bits, pack factor) instead of
  dispatched on its name. Registering a tier no longer requires editing the
  engine.
- Plugin system (`argus_cache.plugins`): quantization backends register with
  capability metadata (effective bits, devices, dtypes, calibration,
  reconstruction, native codec). Any built-in tier — including 1-bit — can be
  removed and replaced without touching the memory manager.
- Runtime adapters (`argus_cache.adapters`): explicit, idempotent
  `initialize/activate/deactivate/shutdown` lifecycle with guaranteed
  non-raising teardown. `OllamaAdapter` (external runtime) and `VLLMAdapter`
  (in-process, experimental).
- `benchmarks/bench_native_runtime.py`, `benchmarks/bench_jl_fidelity.py`,
  `scripts/record_baseline.py`, `scripts/measure_ollama.py`, and
  `docs/measurements/` — every documented number is now traceable to a
  committed artifact and a command that regenerates it.
- Stress coverage (`tests/test_stability.py`): cache-lifetime leaks, a
  200-page cascade, prefetch/attention concurrency, repeated spill cycles, and
  finite attention across a 60-step decode run.

### Fixed
- **Quantization truncated instead of rounding.** Casting float to int
  truncates toward zero, biasing every value by up to half a step. int4
  relative L2 error: ~0.27 → 0.159.
- **1-bit used the wrong magnitude.** `SignPacked` reconstructs as ±scale and
  used the per-page maximum; the L2-optimal scalar is the mean absolute value.
  Relative L2: **3.51 → 0.60** at identical storage cost.
- **Prefetch worker cached undefined tensors** for projection pages with no
  cached reconstruction operator, surfacing later as an empty page during
  attention. It now skips the speculation.
- **Unbounded tiers could not allocate.** A tier declared `max_pages=-1` (the
  archival floor) passed the sentinel to `torch.zeros`, raising "Dimension
  size must be non-negative" and taking down any pipeline using one. Unbounded
  tiers now allocate per page instead of from a fixed pool.
- **The root `models/attention_wrapper.py` was a divergent copy**, 121 lines
  behind the canonical class and silently missing `pipeline=` and
  `balloon_driver=` support. It is now a re-export shim, with a test that
  fails if any root shim re-implements rather than re-exports.

### Changed
- `manager.cpp` 1207 → 774 lines; `quantization_kernels.cu` 183 → 81 (five
  bespoke kernels became one parameterized kernel).
- `memory_manager.py` 2662 → 1785 lines, split by responsibility into
  `telemetry.py`, `jl_operators.py`, `pool_allocator.py`, `granularity.py`,
  `host_spill.py`, and `outliers.py`.
- Static pool shapes derive from a tier's declared bit width, not its name.
  This was the last name-based dispatch on the Python side; a third-party
  plugin tier previously matched no branch, got no pool, and silently fell
  back to uncompressed spill.
- `ARGUS_VERBOSE` now gates every hot-path log in the native engine. Measured
  cost when on: +10.6 % decode time.

### Removed
- **`inject_argus_to_vllm()`.** It divided vLLM's `block_tables` by a
  "reduction factor", but those entries are physical block *indices*, not byte
  offsets — it aliased unrelated sequences onto shared blocks, compressed
  nothing, and corrupted attention under load. Use `VLLMAdapter`.
  `argus_vllm_models` remains as a deprecation shim.

### Verified
- **Ollama adapter, against a live server** (0.32.11, `qwen2.5:0.5b`): five
  live tests covering version probe, real decode timings, actionable rejection
  of a missing model, telemetry, and repeated lifecycle cycles. The adapter
  needed no changes. Baseline in `docs/measurements/ollama-2026-08-14.json`,
  labelled external — ARGUS does not manage that KV cache.
- **The JL tier, on real activations** (Qwen2.5-0.5B-Instruct, 24 layers, per
  64-token page): it beats the shipped int2 backend at the same 4× storage
  budget on 20 of 24 layers (median rel. L2 0.413 vs 0.565). The tier is
  justified and kept. Its prior is *smoothness* along the sequence axis, not
  low rank — reconstruction error tracks page roughness at r = 0.988 — and the
  previous "low-rank structure" justification was wrong.

### Benchmark claims withdrawn
All previously published "ARGUS-vLLM" VRAM figures were measured through the
removed injection path and are **not supported**. `docs/architecture.md §5`
carries the revalidated measurements and an explicit list of what has not been
measured.

### Known limitations
- `VLLMAdapter` is experimental and unverified against a live vLLM.
- ARGUS does not manage Ollama's KV cache; the adapter configures and measures
  an external process.
- Predictive paging remains experimental and off by default.
- No downstream quality measurement (perplexity, retrieval) is published. Every
  fidelity figure is synthetic reconstruction error, which is not accuracy.
