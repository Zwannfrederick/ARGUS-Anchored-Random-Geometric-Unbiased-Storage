"""ARGUS — a configurable heterogeneous KV-cache management runtime.

ARGUS is a control layer for KV-cache memory, not a single quantization
algorithm. Python owns configuration, policy, and the public API; a native
C++/CUDA engine owns the data plane (page storage, compression kernels, fused
attention). Quantization tiers are plugins (:mod:`argus_cache.plugins`) and
inference runtimes are adapters (:mod:`argus_cache.adapters`), so neither is
baked into the cache manager.
"""

from .models.attention_wrapper import PagedDynamicQuantizedCache
from .core.memory_manager import PagedDynamicKVCache
from .core.balloon_driver import ElasticCacheBalloonDriver
from .core.tier_registry import PipelineConfig, TierSpec
from .plugins import (
    BackendCapabilities,
    NativeCodecSpec,
    available_quantizers,
    get_capabilities,
    list_quantizers,
    register_quantizer,
    unregister_quantizer,
)

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
    original_prep = model.prepare_inputs_for_generation

    def prepare_inputs_for_generation_argus(*args, **kwargs):
        past_key_values = kwargs.get("past_key_values", None)
        if past_key_values is None:
            kwargs["past_key_values"] = PagedDynamicQuantizedCache(
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
