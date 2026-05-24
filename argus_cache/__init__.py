from .models.attention_wrapper import PagedDynamicQuantizedCache
from .core.memory_manager import PagedDynamicKVCache

def patch_model_with_argus(
    model, 
    page_size=4096, 
    max_active_pages=2, 
    max_fp8_pages=2, 
    max_int8_pages=2, 
    max_int4_pages=2, 
    max_int2_pages=2,
    max_one_bit_pages=2,
    sink_tokens=4
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
                sink_tokens=sink_tokens
            )
        return original_prep(*args, **kwargs)

    model.prepare_inputs_for_generation = prepare_inputs_for_generation_argus
    return model

__all__ = [
    "PagedDynamicQuantizedCache",
    "PagedDynamicKVCache",
    "patch_model_with_argus"
]
