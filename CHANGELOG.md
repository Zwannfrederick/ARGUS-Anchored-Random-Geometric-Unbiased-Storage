# Changelog

## v0.4.0 — Direct paged attention, hybrid ownership, and an honest llama.cpp negative

### Added
- **Structure-of-Arrays page table** (`core/page_table.py`): precision
  (`ACTIVE_FP16`, `GGML_Q8_0`, `GGML_Q4_0`) is separated from placement
  (`GPU_DEVICE`, `HOST_PINNED`, `HOST_PAGEABLE`) in one contiguous descriptor
  table. Generation counters make a stale async eviction detectable instead of
  silently reading a recycled slot. The decode hot path no longer walks Python
  objects or raw pointers.
- **Contiguous block pool** (`core/backend_pool.py`): quantized pages are
  allocated from unified fixed-size chunks rather than per-page, which removes
  allocator fragmentation and makes batched PCIe streaming possible.
- **Direct paged attention engine** (`core/direct_attention.py`): exact
  tile-by-tile online-softmax recurrence run straight over the descriptor table
  and quantized block pools. No context-sized FP16 KV tensor is ever
  materialized; FP16 reconstruction is bounded to one page tile.
- **Hybrid cache ownership contract** (`models/hybrid_cache.py`): for
  architectures mixing full and linear attention (e.g. Qwen3.8 Gated DeltaNet),
  full-attention layers are owned by ARGUS while recurrent/conv state is
  explicitly not. Deterministic state digests prove ARGUS operations never
  mutate recurrent state; rollback and sequence operations fail closed on an
  unsupported layer role.
- **Anthropic Messages gateway** (`adapters/claude_gateway.py`): translates
  `/v1/messages` to OpenAI `/v1/chat/completions`, including tool-use blocks and
  bidirectional SSE streaming, so an Anthropic-protocol client can drive a local
  llama-server. Stdlib `http.server` plus `httpx`; no server framework added.
- **Service manager** (`scripts/argus_service.py`): OFF / STARTING / READY /
  ERROR lifecycle with health probes, plus a telemetry UI and a Waybar module.

### Measured
- **Direct paged attention, precision vs latency.** Qwen-like geometry
  (24 q-heads, 4 kv-heads, head_dim 256, page 128) on the RTX 3050 Ti Laptop.
  The trade is monotone and steep: q4_0 holds a 32K context in 44.16 MiB
  against FP16's 136.16 MiB (3.1x less) while costing 2.14x the latency.

  | context | ACTIVE_FP16 | GGML_Q8_0 | GGML_Q4_0 |
  |---:|---|---|---|
  | 1,024 | 1.98 ms / 12.15 MiB | 3.15 ms / 10.28 MiB | 4.24 ms / 9.28 MiB |
  | 4,096 | 7.51 ms / 24.15 MiB | 11.43 ms / 16.65 MiB | 15.86 ms / 12.65 MiB |
  | 8,192 | 14.71 ms / 40.15 MiB | 22.55 ms / 25.15 MiB | 31.61 ms / 17.15 MiB |
  | 16,384 | 29.26 ms / 72.15 MiB | 44.97 ms / 44.15 MiB | 62.77 ms / 26.15 MiB |
  | 32,768 | 58.48 ms / 136.16 MiB | 90.15 ms / 76.16 MiB | 125.25 ms / 44.16 MiB |

  Artifact: `docs/measurements/v040-fused-attention-benchmark.json`.
  This is a **runtime** measurement of the engine in isolation, not an
  end-to-end serving result.
- **q4_0 KV quantization cost no measurable retrieval accuracy** at a 31k-token
  prompt against 8 confusable distractors: f16, q8_0 and q4_0 each scored 4/4
  and produced byte-identical answers. This does **not** establish general
  output quality — retrieving a literal string already in context is a
  forgiving task, and identical outputs across all three widths suggest the
  probe never reached the model's margin.
  Artifact: `docs/measurements/kv-quantization-retrieval-2026-08-16.json`.

### Packaging
- `matplotlib` and `pytest` were declared runtime dependencies through 0.3.0,
  but nothing under `argus_cache/` imports either — they served the benchmarks
  and the test suite. Both moved to a `dev` extra.
- `httpx` is genuinely imported, by the optional Anthropic gateway adapter
  only. It is declared as a `gateway` extra rather than pulled into every
  install.
- Added project URLs, keywords, and CUDA/Linux classifiers so the PyPI page
  points at the repository and the changelog.
- Published as an sdist only. A `linux_x86_64` wheel built against one local
  PyTorch ABI and CUDA version would be wrong for nearly every installer, and
  PyPI rejects that platform tag regardless.
- Documented that `pip install` needs `--no-build-isolation`. `setup.py` reads
  the installed `torch` to configure the CUDA extension, and pip's default
  isolated build hides it, so the install fails with
  `ModuleNotFoundError: No module named 'torch'`. Adding `torch` to
  `build-system.requires` would be worse: pip would fetch some torch into a
  throwaway environment and the extension would be compiled for an ABI the
  runtime does not have.

### Fixed (tests)
- `test_gateway_server_health_and_models_endpoints` asserted that `/v1/models`
  contained `qwen3.8-27b`. That id comes from whatever a local Ollama reports,
  not from the gateway, so the test failed on every machine without that exact
  model in its library — including this one once the library moved on to
  `qwen3.6-35b-a3b`. It now asserts only the static Anthropic ids the handler
  actually guarantees.

### Negative result — ARGUS is not integrated with llama.cpp
An A/B sweep was run at 4K/16K/32K/64K against a local llama-server
(Qwen3.6-35B-A3B, q4_0 KV) intending to compare an ARGUS-backed cache with a
vanilla one. **The audit in the artifact shows ARGUS was never loaded:**
`argus_in_llama_server_maps: false`, `argus_maps_count: 0`. Peak VRAM is
byte-identical between the two arms at every context (3594 / 3596 / 3598 MiB),
which confirms both arms were the same process.

The decode-rate difference that was observed (e.g. 17.66 vs 11.88 tok/s at 16K)
is therefore attributable to the gateway's prompt-prefix cache — TTFT drops to
99.88 ms where the vanilla arm reprocesses the prompt — and **not** to ARGUS.
No ARGUS claim is made from this run. It is published as-is because a
misattributed win is exactly the kind of result that survives unchallenged.
Artifact: `docs/measurements/argus-ab-cache-comparison-2026-09-04.json`.

The llama.cpp sweeps shipped alongside it (`load-mode-comparison`,
`pmin-sweep`, `speculative-sweep-n2-n3-n4`) are **llama.cpp runtime tuning**,
not ARGUS measurements, and are labelled as such.

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
