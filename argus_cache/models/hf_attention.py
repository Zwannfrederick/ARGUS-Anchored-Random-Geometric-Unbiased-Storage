"""Model-aware bridge between Transformers attention and ARGUS pages.

Transformers normalizes most decoder attention inputs to ``[B, H, T, D]``,
but models still differ in masking, local attention, head layout, and output
contracts. This registry keeps those decisions outside the cache data plane.
The model patch selects this bridge only for explicitly registered contracts.
"""

from __future__ import annotations

from typing import Any

import torch


class AttentionAdapter:
    """Extension point for one model family's native attention contract."""

    def can_use_native(
        self,
        module: torch.nn.Module,
        query: torch.Tensor,
        attention_mask: torch.Tensor | None,
        *,
        dropout: float = 0.0,
        **kwargs: Any,
    ) -> bool:
        return False

    def prepare_query(self, module: torch.nn.Module, query: torch.Tensor) -> torch.Tensor:
        return query

    def format_output(self, module: torch.nn.Module, output: torch.Tensor) -> torch.Tensor:
        # Native ARGUS returns [batch, heads, query, dim]; the functional
        # Transformers attention contract returns [batch, query, heads, dim].
        return output.transpose(1, 2).contiguous()


class FullAttentionGQADecodeAdapter(AttentionAdapter):
    """Narrow native contract shared by validated standard GQA families."""

    def can_use_native(
        self,
        module: torch.nn.Module,
        query: torch.Tensor,
        attention_mask: torch.Tensor | None,
        *,
        dropout: float = 0.0,
        **kwargs: Any,
    ) -> bool:
        return (
            query.shape[-2] == 1
            and not getattr(module, "training", False)
            and dropout == 0.0
            and attention_mask is None
            and kwargs.get("sliding_window") is None
            and not kwargs.get("output_attentions", False)
        )


class Qwen2AttentionAdapter(FullAttentionGQADecodeAdapter):
    """Validated full-attention decode contract for Qwen2."""


class LlamaAttentionAdapter(FullAttentionGQADecodeAdapter):
    """Validated full-attention decode contract for the Llama family."""


_ATTENTION_ADAPTERS: dict[str, AttentionAdapter] = {
    "qwen2": Qwen2AttentionAdapter(),
    "llama": LlamaAttentionAdapter(),
}


def register_attention_adapter(model_type: str, adapter: AttentionAdapter) -> None:
    """Register or replace the native feeding contract for ``model_type``."""
    if not model_type:
        raise ValueError("model_type must be non-empty")
    if not isinstance(adapter, AttentionAdapter):
        raise TypeError("adapter must inherit AttentionAdapter")
    _ATTENTION_ADAPTERS[model_type] = adapter


def unregister_attention_adapter(model_type: str) -> None:
    """Remove a custom model contract; unknown models then fail closed."""
    _ATTENTION_ADAPTERS.pop(model_type, None)


def has_attention_adapter(model_type: str | None) -> bool:
    """Return whether native feeding was explicitly validated for a model."""
    return model_type in _ATTENTION_ADAPTERS


def _model_type(module: torch.nn.Module) -> str | None:
    config = getattr(module, "config", None)
    return getattr(config, "model_type", None)


def _repeat_kv(states: torch.Tensor, groups: int) -> torch.Tensor:
    if groups == 1:
        return states
    batch, heads, tokens, dim = states.shape
    states = states[:, :, None, :, :].expand(batch, heads, groups, tokens, dim)
    return states.reshape(batch, heads * groups, tokens, dim)


def _sdpa_fallback(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    *,
    dropout: float,
    scaling: float | None,
) -> tuple[torch.Tensor, None]:
    groups = int(getattr(module, "num_key_value_groups", 1))
    key = _repeat_kv(key, groups)
    value = _repeat_kv(value, groups)
    is_causal = bool(
        query.shape[-2] > 1
        and attention_mask is None
        and getattr(module, "is_causal", True)
    )
    output = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=dropout,
        scale=scaling,
        is_causal=is_causal,
    )
    return output.transpose(1, 2).contiguous(), None


def argus_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    """Dispatch native page attention only through a validated model adapter."""
    manager = getattr(key, "_argus_cache_manager", None)
    adapter = _ATTENTION_ADAPTERS.get(_model_type(module))
    if manager is not None and adapter is not None and adapter.can_use_native(
        module,
        query,
        attention_mask,
        dropout=dropout,
        **kwargs,
    ):
        native_query = adapter.prepare_query(module, query)
        output = manager.inplace_paged_attention(native_query, scale=scaling)
        return adapter.format_output(module, output), None

    # A tagged tensor contains only the newest chunk. Reconstruct solely for
    # unsupported/prefill/masked cases so the fallback remains exact.
    if manager is not None:
        key, value = manager.get_all_keys_values()
    return _sdpa_fallback(
        module,
        query,
        key,
        value,
        attention_mask,
        dropout=dropout,
        scaling=scaling,
    )


def install_transformers_attention() -> None:
    """Register ARGUS attention and SDPA-compatible mask construction."""
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, sdpa_mask
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    ALL_ATTENTION_FUNCTIONS.register("argus", argus_attention_forward)
    ALL_MASK_ATTENTION_FUNCTIONS.register("argus", sdpa_mask)
