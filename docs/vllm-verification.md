# vLLM compatibility verification — 2026-08-14

## Environment

The probe was isolated so installing vLLM could not replace the primary
benchmark environment's torch build.

| component | value |
|---|---|
| vLLM | 0.27.1 |
| torch | 2.13.0+cu130 |
| Python | 3.14.7 |
| primary ARGUS torch | 2.12.0+cu130, unchanged |

The vLLM wheel/package in this environment does not provide a loadable
`vllm._C`, so an engine-level generation run was not possible. Python API
inspection and adapter tests were still possible.

## Observed seam

The installed model class exposes:

```text
vllm.model_executor.models.llama.LlamaAttention.forward(
    self, positions, hidden_states
)
```

This is not a KV-cache ownership boundary. vLLM's attention layer delegates to
its own attention implementation and block pool. Attaching `_argus_cache` to
the Python model object does not cause the runtime to store, read, compress, or
free blocks through ARGUS.

The former adapter therefore reported an integration that did no cache
management. The older helper was worse: dividing `block_tables` entries by a
compression ratio modified physical block indices, not byte offsets, and
could alias unrelated blocks without compressing them.

## Correct integration boundary

vLLM 0.27.1 exposes two mechanisms relevant to a future implementation:

- [`KVConnectorBase_V1`](https://docs.vllm.ai/en/latest/api/vllm/distributed/kv_transfer/kv_connector/v1/)
  for moving or offloading runtime-owned KV blocks;
- a registered custom
  [`AttentionBackend`](https://docs.vllm.ai/en/latest/api/vllm/v1/attention/backends/registry/)
  for cache layout and attention over that layout.

ARGUS needs an implementation at one or both boundaries. A connector alone can
provide capacity/offload behavior, while a custom backend is needed to avoid
reconstructing the entire cache before attention.

## Current behavior

`VLLMAdapter` intentionally has:

```text
SUPPORTED_VLLM_RANGE = ()
manages_kv_cache = False
integration_available = False
```

Initialization reports the installed vLLM version and raises an actionable
error. It performs no monkey patch, so `is_fully_restored()` is always true.
The source-compatible constructor parameters remain accepted, but none can
re-enable the ineffective path.

Verification result in the isolated environment:

```text
10 passed, 23 deselected
```

This is a negative but useful result: vLLM support is unavailable rather than
silently false.
