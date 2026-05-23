import torch
from core.memory_manager import PagedDynamicKVCache

# Attempt to import HuggingFace Cache base class for seamless integration
try:
    from transformers.cache_utils import Cache
except ImportError:
    class Cache:
        """Fallback base class if transformers is not installed."""
        pass

class PagedDynamicQuantizedCache(Cache):
    def __init__(self, page_size=4096, max_active_pages=2, max_fp8_pages=2, max_int8_pages=2, max_int4_pages=2, sink_tokens=4):
        """
        HuggingFace-compatible Caching Layer: PagedDynamicQuantizedCache.
        Supports 5-Tier dynamic memory lifecycle with permanent Outlier-Aware Attention Sink protection.
        """
        super().__init__(layers=[])
        self.page_size = page_size
        self.max_active_pages = max_active_pages
        self.max_fp8_pages = max_fp8_pages
        self.max_int8_pages = max_int8_pages
        self.max_int4_pages = max_int4_pages
        self.sink_tokens = sink_tokens
        
        # Maps layer index to its corresponding PagedDynamicKVCache instance
        self.layer_caches = {}

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int, cache_kwargs=None):
        """
        Updates the cache with the new key and value states for a specific layer.
        """
        if layer_idx not in self.layer_caches:
            self.layer_caches[layer_idx] = PagedDynamicKVCache(
                page_size=self.page_size,
                max_active_pages=self.max_active_pages,
                max_fp8_pages=self.max_fp8_pages,
                max_int8_pages=self.max_int8_pages,
                max_int4_pages=self.max_int4_pages,
                sink_tokens=self.sink_tokens
            )
            
        layer_cache = self.layer_caches[layer_idx]
        
        is_anchor = None
        if cache_kwargs is not None and "is_anchor" in cache_kwargs:
            is_anchor = cache_kwargs["is_anchor"]
            
        layer_cache.push_new_tokens(key_states, value_states, is_anchor=is_anchor)
        return layer_cache.get_all_keys_values()

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """
        Returns the current total sequence length stored in the cache for a given layer.
        """
        if layer_idx not in self.layer_caches:
            return 0
            
        cache = self.layer_caches[layer_idx]
        length = 0
        
        # Add Attention Sinks count
        if cache.sink_k is not None:
            length += cache.sink_k.shape[-2]
            
        # Calculate tokens from all page levels
        num_pages = (
            len(cache.active_pages) + 
            len(cache.fp8_pages) + 
            len(cache.int8_pages) + 
            len(cache.int4_pages) + 
            len(cache.int2_pages)
        )
        length += num_pages * cache.page_size
        
        # Add remaining tokens in the temporary buffer
        if cache.k_buffer is not None:
            length += cache.k_buffer.shape[-2]
            
        return length

    def get_usable_length(self, seq_len: int, layer_idx: int = 0) -> int:
        """
        Returns the sequence length that can be used for attention.
        """
        return self.get_seq_length(layer_idx)

    def get_vram_usage(self) -> int:
        """
        Retrieves the total VRAM usage (in bytes) of all layer caches.
        """
        return sum(cache.get_vram_usage() for cache in self.layer_caches.values())

    def reset(self):
        """
        Clears the cache contents.
        """
        self.layer_caches.clear()
