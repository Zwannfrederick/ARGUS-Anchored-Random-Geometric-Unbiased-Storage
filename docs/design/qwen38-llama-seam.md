# Qwen3.8 llama.cpp ownership seam

This note pins the first production seam investigation for ARGUS v0.4. It is
not an implementation claim. Source anchors refer to llama.cpp revision
`6b4344ecc7e6a493daba9681991852eb06bb3b33`.

## Architecture identity

The Qwen3.8 checkpoint name does not introduce a new runtime architecture ID.
Its official configuration declares `model_type = qwen3_5`, and the pinned
GGUF is therefore expected to load through llama.cpp's `LLM_ARCH_QWEN35` path.
This is the intended mapping, not a compatibility alias inferred by ARGUS.

The official text configuration declares 64 layers and
`full_attention_interval = 4`: 48 linear-attention layers and 16 growing-KV
full-attention layers. Full attention uses 24 query heads, 4 KV heads, and a
256-element head dimension. v0.4 is text-only and does not load the separate
vision projector or MTP artifact.

The downloaded Q4_K_M artifact independently confirms this contract in its
GGUF v3 header: architecture `qwen35`, 64 blocks, interval 4, 24/4 heads,
256-element K/V, and a 262144-token context. Its exact byte size and SHA-256
match the pinned repository metadata. A stock CUDA llama.cpp text-only smoke
also loaded the artifact and generated tokens on the target RTX 3050 Ti; this
proves artifact/runtime compatibility, not the pending 32K placement gate.

## Narrow ownership boundary

The current factory is in `src/llama-model.cpp`, inside the hybrid branch of
`llama_model::create_memory()`. For Qwen35 it supplies two layer filters:

- `filter_attn`: main-stack layers where `!hparams.is_recr(il)`;
- `filter_recr`: main-stack layers where `hparams.is_recr(il)`.

It constructs `llama_memory_hybrid`, whose two independent members are:

- `mem_attn`: `std::unique_ptr<llama_kv_cache>`;
- `mem_recr`: `std::unique_ptr<llama_memory_recurrent>`.

The first ARGUS fork should replace only the attention-cache implementation
behind `mem_attn`. It must preserve `mem_recr` construction, storage type,
batch preparation, rollback snapshots, and serialization unchanged.

The graph boundary is equally explicit. `src/models/qwen35.cpp` calls
`build_inp_mem_hybrid()`, then dispatches recurrent layers through
`build_layer_attn_linear(inp->get_recr(), ...)` and full-attention layers
through `build_layer_attn(inp->get_attn(), ...)`. The full-attention builder
applies Q/K normalization and MRoPE before handing K/V to `build_attn`, and
applies the attention-output gate after the attention result. ARGUS must own
the stored post-RoPE K and V without moving either normalization or output
gating across the seam.

## Lifecycle contract

`llama_memory_hybrid` forwards the following operations to both child memories:

- clear;
- sequence copy, keep, add, and divide;
- sequence position queries;
- state read/write.

Sequence removal is ordered deliberately: recurrent removal runs first because
it can fail without mutation, then attention removal runs. The ARGUS attention
child must preserve that transaction order and return semantics.

Partial state writes omit attention memory but retain recurrent state. The
ARGUS child must not reinterpret this flag. Context shift, prefix mutation,
prompt-cache restore, cancellation rollback, and partial sequence removal stay
fail-closed until the pinned hybrid lifecycle matrix passes.

## Integration shape

The public `llama.h` API does not inject a cache constructor, so a wrapper
cannot prove ownership. The smallest viable production experiment is a pinned
llama.cpp fork that:

1. adds an ARGUS attention-memory implementation conforming to
   `llama_memory_i`/the graph input contract;
2. selects it only in the Qwen35 hybrid factory when an explicit `--argus`
   option is enabled;
3. keeps `--argus off` in the same binary as a stock fallback;
4. reports attention and recurrent bytes independently through
   `memory_breakdown()`;
5. uses ggml backend buffers and scheduler-owned events rather than launching
   an unrelated PyTorch CUDA stream.

The first proof replaces one full-attention layer's storage/read path and
demonstrates changed allocated bytes, unchanged recurrent-state digest during
cache-only lifecycle operations, and stock output parity in ACTIVE/exact mode.

## Sources

- https://huggingface.co/Qwen/Qwen3.8-27B
- https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF
- https://github.com/ggml-org/llama.cpp/blob/6b4344ecc7e6a493daba9681991852eb06bb3b33/src/llama-model.cpp
- https://github.com/ggml-org/llama.cpp/blob/6b4344ecc7e6a493daba9681991852eb06bb3b33/src/llama-memory-hybrid.cpp
- https://github.com/ggml-org/llama.cpp/blob/6b4344ecc7e6a493daba9681991852eb06bb3b33/src/models/qwen35.cpp
