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
    "BackendCapabilities": ("argus_cache.plugins", "BackendCapabilities"),
    "NativeCodecSpec": ("argus_cache.plugins", "NativeCodecSpec"),
    "register_quantizer": ("argus_cache.plugins", "register_quantizer"),
    "unregister_quantizer": ("argus_cache.plugins", "unregister_quantizer"),
    "list_quantizers": ("argus_cache.plugins", "list_quantizers"),
    "available_quantizers": ("argus_cache.plugins", "available_quantizers"),
    "get_capabilities": ("argus_cache.plugins", "get_capabilities"),
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
    pipeline=None
):
    """
    Patches a HuggingFace causal language model to automatically use
    the ARGUS (PagedDynamicQuantizedCache) KV Cache manager.
    """
    cache_class = __getattr__("PagedDynamicQuantizedCache")
    original_prep = model.prepare_inputs_for_generation

    def prepare_inputs_for_generation_argus(*args, **kwargs):
        past_key_values = kwargs.get("past_key_values", None)
        if past_key_values is None:
            kwargs["past_key_values"] = cache_class(
                page_size=page_size,
                max_active_pages=max_active_pages,
                max_fp8_pages=max_fp8_pages,
                max_int8_pages=max_int8_pages,
                max_int4_pages=max_int4_pages,
                max_int2_pages=max_int2_pages,
                max_one_bit_pages=max_one_bit_pages,
                sink_tokens=sink_tokens,
                pipeline=pipeline
            )
        return original_prep(*args, **kwargs)

    model.prepare_inputs_for_generation = prepare_inputs_for_generation_argus
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
    # Plugin system
    "BackendCapabilities",
    "NativeCodecSpec",
    "register_quantizer",
    "unregister_quantizer",
    "list_quantizers",
    "available_quantizers",
    "get_capabilities",
]
