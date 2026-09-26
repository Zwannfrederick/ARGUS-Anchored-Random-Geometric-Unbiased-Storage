# ARGUS v0.8 — backlog

Short list carried out of the v0.7 audits and release. Nothing here is started;
each item needs its own plan, measurement and acceptance before implementation.

**Plugin readiness** (from `docs/plans/control-audit.md`: PL1/PL2/PL3/PL6)
- Package the llama.cpp integration as its own unit, separate from the
  PyTorch/HuggingFace package (which pulls torch, triton and transformers).
- Add a configuration API (`ArgusConfig` / C API) and keep environment variables
  only as the default source.
- Scope budgets per store or context instead of process-wide, and support devices
  other than CUDA device 0.
- Move the Anthropic gateway, dashboard and telemetry printing out of the core
  package.

**Runtime coverage**
- Multi-sequence KV: several streams, server `-np > 1`.
- Wait-free (async) appends for policy stores; E7b appends still synchronize.
- A stream-ordered allocator (`cudaFreeAsync`) to remove the reliance on
  `cudaFree` waiting for all queued device work.
- Q8/Q4 KV on the CUDA path.

**Measurements**
- Context lengths beyond 4K, up to the deferred 262K.
- Other models and GPUs.
- KV larger than VRAM as a real workload: the tight-budget regime is disk-bound at
  47.6 s / 1.0 tok/s at 4K.

**Code health**
- Bring `argus_cache/core/memory_manager.py` (1972 lines) and `manager.cpp`
  (1055 lines) under the 1000-line ceiling.
- Decide which store is the product — the Python HuggingFace store or the llama.cpp
  store — and whether per-page mixed precision moves to llama.cpp.
