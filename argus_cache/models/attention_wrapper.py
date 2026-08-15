import torch
from argus_cache.core.memory_manager import PagedDynamicKVCache
from argus_cache.core.activation import (
    ActivationDecision,
    AdaptiveCachePolicy,
    CacheGeometry,
    MemorySnapshot,
)
from argus_cache.core.tier_registry import PipelineConfig, TierSpec

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
        threshold_sigma=3.0,
        pipeline=None,
        balloon_driver=None,
        model_config=None,
        activation_policy=None,
        memory_probe=None,
        pipeline_profile=None,
        direct_attention=False,
    ):
        """
        HuggingFace-compatible adaptive cache. Balanced mode bypasses ARGUS
        while exact K/V fits, then uses an ACTIVE-to-FP8 serving profile under
        pressure. ``pipeline_profile="research"`` retains the deep cascade.
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
        self.pipeline = pipeline
        self.balloon_driver = balloon_driver
        self.model_config = model_config
        self.activation_policy = activation_policy or AdaptiveCachePolicy()
        self._memory_probe = memory_probe or MemorySnapshot.current_cuda
        self.pipeline_profile = pipeline_profile or (
            "balanced" if model_config is not None else "research"
        )
        self.direct_attention = direct_attention
        if self.pipeline_profile not in ("balanced", "research"):
            raise ValueError("pipeline_profile must be 'balanced' or 'research'")
        
        if pipeline is not None:
            self.page_size = getattr(pipeline, 'page_size', page_size)
            self.max_active_pages = getattr(pipeline, 'max_active_pages', max_active_pages)
            tier_dict = {t.name: t.max_pages for t in getattr(pipeline, 'tiers', [])}
            self.max_fp8_pages = tier_dict.get('fp8', max_fp8_pages)
            self.max_int8_pages = tier_dict.get('int8', max_int8_pages)
            self.max_int4_pages = tier_dict.get('int4', max_int4_pages)
            self.max_int2_pages = tier_dict.get('int2', max_int2_pages)
            self.max_one_bit_pages = tier_dict.get('one_bit', max_one_bit_pages)
            self.sink_tokens = getattr(pipeline, 'sink_tokens', sink_tokens)
            self.threshold_sigma = getattr(pipeline, 'threshold_sigma', threshold_sigma)
            
        # Maps layer index to its corresponding PagedDynamicKVCache instance
        self.layer_caches = {}
        # The true bypass path intentionally never creates a native manager.
        # It behaves like an exact DynamicCache until the activation policy
        # predicts that the full K/V allocation would exceed the VRAM budget.
        self._exact_layers = {}
        self._request_started = False
        self._argus_active = False
        self._last_mode_active = False
        self._activation_decision = None

    @property
    def activation_state(self) -> str:
        return "argus" if self._argus_active else "exact"

    @property
    def activation_decision(self):
        return self._activation_decision

    def _simple_pipeline(self) -> PipelineConfig:
        """Serving profile: keep hot pages exact and demote cold pages to FP8."""
        return PipelineConfig(
            tiers=[
                TierSpec(
                    "fp8",
                    "fp8",
                    max_pages=-1,
                    priority=1,
                    use_outlier_isolation=False,
                )
            ],
            page_size=self.page_size,
            sink_tokens=self.sink_tokens,
            threshold_sigma=self.threshold_sigma,
            max_active_pages=self.max_active_pages,
            balloon_driver=self.balloon_driver,
            streaming_attention=True,
        )

    def _research_pipeline(self) -> PipelineConfig:
        """Compatibility profile retaining the configurable deep cascade."""
        from argus_cache.backends.eviction import ImportanceSortPolicy

        return PipelineConfig(
            tiers=[
                TierSpec("fp8", "fp8", max_pages=self.max_fp8_pages, priority=1),
                TierSpec("int8", "int8", max_pages=self.max_int8_pages, priority=2),
                TierSpec("int4", "int4", max_pages=self.max_int4_pages, priority=3),
                TierSpec("int2", "int2", max_pages=self.max_int2_pages, priority=4),
                TierSpec(
                    "one_bit", "one_bit", max_pages=self.max_one_bit_pages, priority=5
                ),
                TierSpec("jl", "jl", max_pages=-1, priority=6),
            ],
            eviction_policy=ImportanceSortPolicy(),
            page_size=self.page_size,
            sink_tokens=self.sink_tokens,
            threshold_sigma=self.threshold_sigma,
            max_active_pages=self.max_active_pages,
            balloon_driver=self.balloon_driver,
        )

    def _layer_pipeline(self, key_states: torch.Tensor):
        if self.pipeline is None:
            if self.pipeline_profile == "balanced":
                return self._simple_pipeline()
            return self._research_pipeline()
        if key_states.dtype in (torch.float32, torch.float16, torch.bfloat16):
            return self.pipeline

        # Retain the legacy low-precision adaptation for explicitly supplied
        # research pipelines.  The default serving pipeline is already FP8-only.
        import copy

        layer_pipeline = copy.deepcopy(self.pipeline)
        layer_pipeline.max_active_pages = max(1, layer_pipeline.max_active_pages // 2)
        for spec in layer_pipeline.tiers:
            if spec.name == "fp8":
                spec.max_pages = 1
            elif spec.name in ("int4", "int2", "one_bit"):
                spec.max_pages = max(1, spec.max_pages * 2)
        return layer_pipeline

    def _create_layer_cache(self, layer_idx: int, sample: torch.Tensor):
        if layer_idx in self.layer_caches:
            return self.layer_caches[layer_idx]
        manager = PagedDynamicKVCache(pipeline=self._layer_pipeline(sample))
        self.layer_caches[layer_idx] = manager
        return manager

    def _decide_activation(self, sample: torch.Tensor, tokens: int):
        policy = self.activation_policy
        if self.model_config is None:
            # Geometry cannot be guessed safely. Preserve the historical ARGUS
            # behavior unless the caller explicitly requested latency mode.
            activate = policy.mode != "latency"
            reason = "missing-model-geometry" if activate else "latency-mode"
            return ActivationDecision(activate, reason, 0, 0, None)

        geometry = CacheGeometry.from_tensor(sample, self.model_config)
        tier_bits = (
            self.pipeline.tiers[0].effective_bits
            if self.pipeline is not None and self.pipeline.tiers
            else 8.0
        )
        max_active_pages = (
            self.pipeline.max_active_pages
            if self.pipeline is not None
            else self.max_active_pages
        )
        return policy.decide(
            geometry=geometry,
            tokens=tokens,
            page_size=self.page_size,
            max_active_pages=max_active_pages,
            tier_effective_bits=tier_bits,
            memory=self._memory_probe(),
            was_active=self._last_mode_active,
        )

    def _activate_argus(self):
        if self._argus_active:
            return
        self._argus_active = True
        self._last_mode_active = True
        for layer_idx, (keys, values) in list(self._exact_layers.items()):
            manager = self._create_layer_cache(layer_idx, keys)
            manager.push_new_tokens(keys, values)
        self._exact_layers.clear()

    def _route_request(self, key_states: torch.Tensor, layer_idx: int):
        existing = self.get_seq_length(layer_idx)
        tokens = existing + int(key_states.shape[-2])
        if not self._request_started:
            self._request_started = True
            self._activation_decision = self._decide_activation(key_states, tokens)
            if self._activation_decision.activate:
                self._activate_argus()
            return

        # A missing/underestimated expected length may cross the pressure gate
        # later. Escalation is one-way for the lifetime of this request so
        # decode never oscillates between storage formats.
        if (
            not self._argus_active
            and self.activation_policy.mode == "balanced"
            and layer_idx == 0
        ):
            decision = self._decide_activation(key_states, tokens)
            self._activation_decision = decision
            if decision.activate:
                self._activate_argus()

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int, cache_kwargs=None):
        """
        Updates the cache with the new key and value states for a specific layer.
        Performs Auto-Adaptive Tiering based on key_states dtype (FP16/BF16 vs FP8/INT8 fine-tuned models).
        """
        self._route_request(key_states, layer_idx)

        if not self._argus_active:
            previous = self._exact_layers.get(layer_idx)
            if previous is None:
                combined = (key_states, value_states)
            else:
                combined = (
                    torch.cat((previous[0], key_states), dim=-2),
                    torch.cat((previous[1], value_states), dim=-2),
                )
            self._exact_layers[layer_idx] = combined
            return combined

        layer_cache = self._create_layer_cache(layer_idx, key_states)
        
        is_anchor = None
        if cache_kwargs is not None and "is_anchor" in cache_kwargs:
            is_anchor = cache_kwargs["is_anchor"]
            
        layer_cache.push_new_tokens(key_states, value_states, is_anchor=is_anchor)
        if self.direct_attention:
            # Transformers passes these tensors directly to its selected
            # functional attention implementation. The model-aware ARGUS
            # bridge consumes the manager marker without reconstructing K/V.
            # Use lightweight views so tensors retained by the native manager
            # do not themselves own a Python reference back to that manager.
            tagged_keys = key_states.view_as(key_states)
            tagged_values = value_states.view_as(value_states)
            tagged_keys._argus_cache_manager = layer_cache
            tagged_values._argus_cache_manager = layer_cache
            return tagged_keys, tagged_values
        return layer_cache.get_all_keys_values()

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """
        Returns the current total sequence length stored in the cache for a given layer.
        """
        if layer_idx in self._exact_layers:
            return int(self._exact_layers[layer_idx][0].shape[-2])
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
            
        # Calculate tokens from all page levels (pluggable tiers + active pool)
        num_pages = len(cache.active_pages) + sum(len(pages) for pages in cache.pages_by_tier.values())
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

    def get_mask_sizes(self, query_length: int, layer_idx: int) -> tuple[int, int]:
        """Expose dynamic-cache geometry to Transformers mask builders."""
        return self.get_seq_length(layer_idx) + query_length, 0

    def get_max_cache_shape(self) -> int:
        """ARGUS dynamic caches have no fixed logical sequence limit."""
        return -1

    def get_vram_usage(self) -> int:
        """
        Retrieves the total VRAM usage (in bytes) of all layer caches.
        """
        exact = sum(
            tensor.numel() * tensor.element_size()
            for pair in self._exact_layers.values()
            for tensor in pair
            if tensor.device.type == "cuda"
        )
        return exact + sum(cache.get_vram_usage() for cache in self.layer_caches.values())

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
        for cache in self.layer_caches.values():
            cache.close()
        self.layer_caches.clear()
        self._exact_layers.clear()
        self._request_started = False
        self._argus_active = False
