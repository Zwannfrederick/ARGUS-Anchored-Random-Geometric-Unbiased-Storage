import torch
from ..core.memory_manager import PagedDynamicKVCache

# Attempt to import HuggingFace Cache base class for seamless integration
try:
    from transformers.cache_utils import Cache
except ImportError:
    class Cache:
        """Fallback base class if transformers is not installed."""
        pass

class PagedDynamicQuantizedCache(Cache):
    def __init__(
        self, 
        page_size=4096, 
        max_active_pages=2, 
        max_fp8_pages=2, 
        max_int8_pages=2, 
        max_int4_pages=2, 
        max_int2_pages=2,
        max_one_bit_pages=2,
        sink_tokens=4,
        threshold_sigma=3.0
    ):
        """
        HuggingFace-compatible Caching Layer: PagedDynamicQuantizedCache.
        Supports 7-Tier dynamic memory lifecycle with permanent Outlier-Aware Attention Sink protection
        and dynamic standard deviation outlier key-value isolation.
        """
        super().__init__(layers=[])
        self.page_size = page_size
        self.max_active_pages = max_active_pages
        self.max_fp8_pages = max_fp8_pages
        self.max_int8_pages = max_int8_pages
        self.max_int4_pages = max_int4_pages
        self.max_int2_pages = max_int2_pages
        self.max_one_bit_pages = max_one_bit_pages
        self.sink_tokens = sink_tokens
        self.threshold_sigma = threshold_sigma
        
        # Maps layer index to its corresponding PagedDynamicKVCache instance
        self.layer_caches = {}

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int, cache_kwargs=None):
        """
        Updates the cache with the new key and value states for a specific layer.
        Performs Auto-Adaptive Tiering based on key_states dtype (FP16/BF16 vs FP8/INT8 fine-tuned models).
        """
        if layer_idx not in self.layer_caches:
            # Auto-Adaptive Tiering logic:
            if key_states.dtype in (torch.float32, torch.float16, torch.bfloat16):
                # Standard high precision flow
                active_p = self.max_active_pages
                fp8_p = self.max_fp8_pages
                int8_p = self.max_int8_pages
                int4_p = self.max_int4_pages
                int2_p = self.max_int2_pages
                one_bit_p = self.max_one_bit_pages
            else:
                # Low-precision inputs (FP8/INT8 models).
                # Shorten FP16/FP8 active stages and increase heavy compression capacity.
                active_p = max(1, self.max_active_pages // 2)
                fp8_p = 1
                int8_p = self.max_int8_pages
                int4_p = max(1, self.max_int4_pages * 2)
                int2_p = max(1, self.max_int2_pages * 2)
                one_bit_p = max(1, self.max_one_bit_pages * 2)

            self.layer_caches[layer_idx] = PagedDynamicKVCache(
                page_size=self.page_size,
                max_active_pages=active_p,
                max_fp8_pages=fp8_p,
                max_int8_pages=int8_p,
                max_int4_pages=int4_p,
                max_int2_pages=int2_p,
                max_one_bit_pages=one_bit_p,
                sink_tokens=self.sink_tokens,
                threshold_sigma=self.threshold_sigma
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
            
        # VIP Anchors count
        if cache.anchor_k is not None:
            length += cache.anchor_k.shape[-2]
            
        # Calculate tokens from all page levels (7 Tiers)
        num_pages = (
            len(cache.active_pages) + 
            len(cache.fp8_pages) + 
            len(cache.int8_pages) + 
            len(cache.int4_pages) + 
            len(cache.int2_pages) +
            len(cache.one_bit_pages) + 
            len(cache.jl_pages)
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

    def speculate_and_prefetch(self, attn_weights=None):
        """
        Delegates speculative prefetching across all active layer caches.
        """
        for cache in self.layer_caches.values():
            cache.speculate_and_prefetch(attn_weights)

    def swap_out_to_host(self):
        """
        Delegates swapping to CPU host RAM across all active layer caches to prevent GPU OOMs.
        """
        for cache in self.layer_caches.values():
            cache.swap_out_to_host()

    def swap_in_to_device(self, device="cuda"):
        """
        Delegates swapping back to active GPU VRAM across all layers.
        """
        for cache in self.layer_caches.values():
            cache.swap_in_to_device(device)

    @property
    def is_swapped_out(self) -> bool:
        """
        Returns True if the cache layer states are swapped out to host memory.
        """
        if not self.layer_caches:
            return False
        # If any layer is swapped out, return True
        return any(cache.is_swapped_out for cache in self.layer_caches.values())

    def reset(self):
        """
        Clears the cache contents.
        """
        self.layer_caches.clear()
