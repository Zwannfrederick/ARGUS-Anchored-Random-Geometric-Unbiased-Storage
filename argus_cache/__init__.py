"""ARGUS — a configurable heterogeneous KV-cache management runtime.

ARGUS is a control layer for KV-cache memory, not a single quantization
algorithm. Python owns configuration, policy, and the public API; a native
C++/CUDA engine owns the data plane (page storage, compression kernels, fused
attention). Quantization tiers are plugins (:mod:`argus_cache.plugins`) and
inference runtimes are adapters (:mod:`argus_cache.adapters`), so neither is
baked into the cache manager.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


# Keep the package root light.  In particular, ``import argus_cache.adapters``
# must work in an isolated runtime environment whose torch ABI differs from the
# locally-built ARGUS extension.  Eagerly importing memory_manager here made
# that impossible: Python loads the package root before any submodule.
_LAZY_EXPORTS = {
    "PagedDynamicQuantizedCache": (
        "argus_cache.models.attention_wrapper",
        "PagedDynamicQuantizedCache",
    ),
    "PagedDynamicKVCache": ("argus_cache.core.memory_manager", "PagedDynamicKVCache"),
    "ElasticCacheBalloonDriver": (
        "argus_cache.core.balloon_driver",
        "ElasticCacheBalloonDriver",
    ),
    "PipelineConfig": ("argus_cache.core.tier_registry", "PipelineConfig"),
    "TierSpec": ("argus_cache.core.tier_registry", "TierSpec"),
    "AdaptiveCachePolicy": ("argus_cache.core.activation", "AdaptiveCachePolicy"),
    "AttentionAdapter": ("argus_cache.models.hf_attention", "AttentionAdapter"),
    "register_attention_adapter": (
        "argus_cache.models.hf_attention",
        "register_attention_adapter",
    ),
    "unregister_attention_adapter": (
        "argus_cache.models.hf_attention",
        "unregister_attention_adapter",
    ),
    "BackendCapabilities": ("argus_cache.plugins", "BackendCapabilities"),
    "NativeCodecSpec": ("argus_cache.plugins", "NativeCodecSpec"),
    "register_quantizer": ("argus_cache.plugins", "register_quantizer"),
    "unregister_quantizer": ("argus_cache.plugins", "unregister_quantizer"),
    "list_quantizers": ("argus_cache.plugins", "list_quantizers"),
    "available_quantizers": ("argus_cache.plugins", "available_quantizers"),
    "get_capabilities": ("argus_cache.plugins", "get_capabilities"),
    "LayerRole": ("argus_cache.models.hybrid_cache", "LayerRole"),
    "HybridTopology": ("argus_cache.models.hybrid_cache", "HybridTopology"),
    "HybridQwenCache": ("argus_cache.models.hybrid_cache", "HybridQwenCache"),
    "CodecKind": ("argus_cache.core.page_table", "CodecKind"),
    "PlacementLocation": ("argus_cache.core.page_table", "PlacementLocation"),
    "StructureOfArraysPageTable": ("argus_cache.core.page_table", "StructureOfArraysPageTable"),
    "ContiguousBlockPool": ("argus_cache.core.backend_pool", "ContiguousBlockPool"),
    "DirectPagedAttentionEngine": ("argus_cache.core.direct_attention", "DirectPagedAttentionEngine"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _LAZY_EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value

def patch_model_with_argus(
    model, 
    page_size=4096, 
    max_active_pages=2, 
    max_fp8_pages=2, 
    max_int8_pages=2, 
    max_int4_pages=2, 
    max_int2_pages=2,
    max_one_bit_pages=2,
    sink_tokens=4,
    pipeline=None,
    activation_policy=None,
    pipeline_profile=None,
):
    """
    Patches a HuggingFace causal language model to automatically use
    the ARGUS (PagedDynamicQuantizedCache) KV Cache manager.
    """
    from argus_cache.models.hf_attention import (
        has_attention_adapter,
        install_transformers_attention,
    )

    install_transformers_attention()
    set_attention = getattr(model, "set_attn_implementation", None)
    model_config = getattr(model, "config", None)
    text_config = (
        model_config.get_text_config(decoder=True)
        if model_config is not None and hasattr(model_config, "get_text_config")
        else model_config
    )
    direct_attention = callable(set_attention) and has_attention_adapter(
        getattr(text_config, "model_type", None)
    )
    if direct_attention:
        try:
            set_attention("argus")
        except ValueError:
            # Older/non-functional model implementations cannot select a
            # registered attention function. They retain reconstruction mode.
            direct_attention = False

    cache_class = __getattr__("PagedDynamicQuantizedCache")
    original_prep = model.prepare_inputs_for_generation

    def new_cache():
        return cache_class(
            page_size=page_size,
            max_active_pages=max_active_pages,
            max_fp8_pages=max_fp8_pages,
            max_int8_pages=max_int8_pages,
            max_int4_pages=max_int4_pages,
            max_int2_pages=max_int2_pages,
            max_one_bit_pages=max_one_bit_pages,
            sink_tokens=sink_tokens,
            pipeline=pipeline,
            model_config=text_config,
            activation_policy=activation_policy,
            pipeline_profile=pipeline_profile,
            direct_attention=direct_attention,
        )

    def prepare_inputs_for_generation_argus(*args, **kwargs):
        past_key_values = kwargs.get("past_key_values", None)
        if past_key_values is None:
            kwargs["past_key_values"] = new_cache()
        return original_prep(*args, **kwargs)

    model.prepare_inputs_for_generation = prepare_inputs_for_generation_argus

    # Transformers 5 prepares DynamicCache before prepare_inputs_for_generation,
    # so inject ARGUS at the earlier GenerationMixin lifecycle seam as well.
    original_cache_prep = getattr(model, "_prepare_cache_for_generation", None)
    if callable(original_cache_prep):

        def prepare_cache_for_generation_argus(
            generation_config,
            model_kwargs,
            generation_mode,
            batch_size,
            max_cache_length,
        ):
            if (
                generation_config.use_cache is not False
                and model_kwargs.get("past_key_values") is None
            ):
                mode = getattr(generation_mode, "value", str(generation_mode))
                if mode not in {"greedy_search", "sample"}:
                    raise NotImplementedError(
                        "ARGUS generation currently supports greedy search and sampling; "
                        f"{mode} requires cache reorder/crop semantics."
                    )
                if generation_config.cache_implementation is not None:
                    raise ValueError(
                        "ARGUS supplies past_key_values; do not also set "
                        "cache_implementation in generate()."
                    )
                model_kwargs["past_key_values"] = new_cache()
            return original_cache_prep(
                generation_config,
                model_kwargs,
                generation_mode,
                batch_size,
                max_cache_length,
            )

        model._prepare_cache_for_generation = prepare_cache_for_generation_argus
    return model

__all__ = [
    # Cache runtime
    "PagedDynamicQuantizedCache",
    "PagedDynamicKVCache",
    "ElasticCacheBalloonDriver",
    "patch_model_with_argus",
    # Tier configuration
    "PipelineConfig",
    "TierSpec",
    "AdaptiveCachePolicy",
    "AttentionAdapter",
    "register_attention_adapter",
    "unregister_attention_adapter",
    # Plugin system
    "BackendCapabilities",
    "NativeCodecSpec",
    "register_quantizer",
    "unregister_quantizer",
    "list_quantizers",
    "available_quantizers",
    "get_capabilities",
    # Hybrid cache contract
    "LayerRole",
    "HybridTopology",
    "HybridQwenCache",
    # Page table & backend pool ABI
    "CodecKind",
    "PlacementLocation",
    "StructureOfArraysPageTable",
    "ContiguousBlockPool",
    "DirectPagedAttentionEngine",
]
