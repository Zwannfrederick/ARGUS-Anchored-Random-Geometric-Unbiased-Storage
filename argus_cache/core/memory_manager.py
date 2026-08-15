import torch
import argus_cpp_backend
import weakref
import threading
from .quantization import (
    quantize_to_int8,
    dequantize_from_int8,
    quantize_to_int4_packed,
    dequantize_from_int4_packed,
    quantize_to_int2_packed,
    dequantize_from_int2_packed,
    quantize_to_1bit_packed,
    dequantize_from_1bit_packed,
    quantize_to_fp8_simulated,
    dequantize_from_fp8_simulated,
    quantize_to_jl_projection,
    dequantize_from_jl_projection
)
from .tier_registry import PipelineConfig, TierSpec, QuantizationBackend, EvictionPolicy, ScoringFunction
from .zero_copy_pool import ZeroCopyHostPool
from .jl_operators import JLOperatorCache
from .pool_allocator import StaticPoolAllocator
from .granularity import GranularityManager
from .host_spill import HostSpillManager
from .triton_kernels import triton_fused_paged_attention

# ─── Compiled Attention Core ────────────────────────────────────────────────
# torch.compile fuses cat + matmul + softmax + matmul into a single CUDA graph.
# dynamic=True handles the varying KV sequence lengths across steps.
# reduce-overhead mode avoids recompilation storms from shape changes.
#
# On hardware where cuMemHostAlloc+DEVICEMAP is unavailable (Optimus/MUXless
# mobile GPUs), this compilation is the primary speed lever — it eliminates
# the per-op Python dispatch overhead that was masking the PCIe bandwidth saving.
def _sdp_attention_core(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                        scale: float) -> torch.Tensor:
    """Scaled dot-product attention with explicit scale. Compiled below."""
    return torch.nn.functional.scaled_dot_product_attention(
        q, k, v,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=scale,
    )

def _qos_weights_core(q: torch.Tensor, k: torch.Tensor, scale: float) -> torch.Tensor:
    _qk = torch.matmul(q, k.transpose(-1, -2)) * scale
    weights = torch.softmax(_qk, dim=-1).mean(dim=(0, 1))
    if weights.dim() > 1:
        weights = weights.mean(dim=0)
    return weights

class SafeCompileWrapper:
    def __init__(self, compiled_fn, fallback_fn):
        self.compiled_fn = compiled_fn
        self.fallback_fn = fallback_fn
        self.use_fallback = False

    def __call__(self, *args, **kwargs):
        if self.use_fallback:
            return self.fallback_fn(*args, **kwargs)
        try:
            return self.compiled_fn(*args, **kwargs)
        except Exception as e:
            # Catch compilation/linker errors at runtime and fall back permanently to eager mode
            argus_log("WARNING", f"torch.compile failed at runtime: {e}. Falling back to eager execution mode.", line_no=42)
            self.use_fallback = True
            return self.fallback_fn(*args, **kwargs)

# Lazy compile: only triggered on first call so import time stays fast.
# Falls back to eager execution if torch.compile is not available.
try:
    _raw_compiled_sdp_attention = torch.compile(
        _sdp_attention_core,
        mode="reduce-overhead",   # CUDA Graphs — best for dynamic seq lengths
        dynamic=True,             # avoids recompilation for each new seq length
        fullgraph=False,          # allow graph breaks (safer for first integration)
    )
    _raw_compiled_qos_weights = torch.compile(
        _qos_weights_core,
        mode="reduce-overhead",
        dynamic=True,
        fullgraph=False,
    )
    _compiled_sdp_attention = SafeCompileWrapper(_raw_compiled_sdp_attention, _sdp_attention_core)
    _compiled_qos_weights = SafeCompileWrapper(_raw_compiled_qos_weights, _qos_weights_core)
    _COMPILE_ENABLED = True
except Exception:
    # torch.compile not available (older PyTorch, no CUDA, CI environments)
    _compiled_sdp_attention = _sdp_attention_core
    _compiled_qos_weights = _qos_weights_core
    _COMPILE_ENABLED = False

class PageDict(dict):
    """A dictionary subclass that redirects legacy key lookups to the pluggable backend fields."""
    def __init__(self, *args, **kwargs):
        if len(args) == 1 and not isinstance(args[0], dict) and hasattr(args[0], 'page_id'):
            obj = args[0]
            d = {}
            for k in [
                'page_id', 'pool_slot', 'pool_idx', 'page_size', 'importance_score',
                'attention_sum', 'last_step_accessed', 'tier_name', 'key', 'value',
                'key_compressed', 'value_compressed', 'key_scale', 'value_scale',
                'key_min', 'value_min'
            ]:
                try:
                    val = obj[k]
                    if val is not None:
                        d[k] = val
                except (KeyError, TypeError, IndexError):
                    if hasattr(obj, k):
                        val = getattr(obj, k)
                        if val is not None:
                            d[k] = val
            super().__init__(d, **kwargs)
        else:
            super().__init__(*args, **kwargs)

    @property
    def key_q(self):
        return self.get('key_compressed', {}).get('q')
    @property
    def value_q(self):
        return self.get('value_compressed', {}).get('q')
    @property
    def key_scales(self):
        return self.get('key_compressed', {}).get('scales')
    @property
    def value_scales(self):
        return self.get('value_compressed', {}).get('scales')
    @property
    def key_packed(self):
        return self.get('key_compressed', {}).get('q')
    @property
    def value_packed(self):
        return self.get('value_compressed', {}).get('q')
    @property
    def key_min(self):
        return self.get('key_compressed', {}).get('min_vals')
    @property
    def value_min(self):
        return self.get('value_compressed', {}).get('min_vals')
    @property
    def key_proj(self):
        return self.get('key_compressed', {}).get('q')
    @property
    def value_proj(self):
        return self.get('value_compressed', {}).get('q')

    def __getitem__(self, key):
        if key == 'key_compressed' and 'key_compressed' not in self:
            if 'key_q' in self:
                return {'q': self['key_q'], 'scales': self.get('key_scales')}
            if 'key_packed' in self:
                return {'q': self['key_packed'], 'scales': self.get('key_scales'), 'min_vals': self.get('key_min')}
            if 'key_proj' in self:
                return {'q': self['key_proj'], 'w_proj': self.get('w_proj')}
        if key == 'value_compressed' and 'value_compressed' not in self:
            if 'value_q' in self:
                return {'q': self['value_q'], 'scales': self.get('value_scales')}
            if 'value_packed' in self:
                return {'q': self['value_packed'], 'scales': self.get('value_scales'), 'min_vals': self.get('value_min')}
            if 'value_proj' in self:
                return {'q': self['value_proj'], 'w_proj': self.get('w_proj')}

        if key in ('key_q', 'key_packed', 'key_proj'):
            return self.get('key_compressed', {}).get('q')
        if key in ('value_q', 'value_packed', 'value_proj'):
            return self.get('value_compressed', {}).get('q')
        if key == 'key_scales':
            return self.get('key_compressed', {}).get('scales')
        if key == 'value_scales':
            return self.get('value_compressed', {}).get('scales')
        if key == 'key_min':
            return self.get('key_compressed', {}).get('min_vals')
        if key == 'value_min':
            return self.get('value_compressed', {}).get('min_vals')
        return super().__getitem__(key)

    def __contains__(self, key):
        if key == 'key_compressed':
            return super().__contains__('key_compressed') or super().__contains__('key_q') or super().__contains__('key_packed') or super().__contains__('key_proj')
        if key == 'value_compressed':
            return super().__contains__('value_compressed') or super().__contains__('value_q') or super().__contains__('value_packed') or super().__contains__('value_proj')

        if key in ('key_q', 'key_packed', 'key_proj'):
            return (super().__contains__('key_compressed') and 'q' in super().__getitem__('key_compressed')) or super().__contains__(key)
        if key in ('value_q', 'value_packed', 'value_proj'):
            return (super().__contains__('value_compressed') and 'q' in super().__getitem__('value_compressed')) or super().__contains__(key)
        if key == 'key_scales':
            return (super().__contains__('key_compressed') and 'scales' in super().__getitem__('key_compressed')) or super().__contains__(key)
        if key == 'value_scales':
            return (super().__contains__('value_compressed') and 'scales' in super().__getitem__('value_compressed')) or super().__contains__(key)
        if key == 'key_min':
            return (super().__contains__('key_compressed') and 'min_vals' in super().__getitem__('key_compressed')) or super().__contains__(key)
        if key == 'value_min':
            return (super().__contains__('value_compressed') and 'min_vals' in super().__getitem__('value_compressed')) or super().__contains__(key)
        return super().__contains__(key)

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

class ArgusConfig:
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
        vram_oom_threshold_ratio=0.85, # 85% VRAM trigger
        importance_alpha=0.5,           # frequency weight
        importance_beta=0.3,            # recency weight
        importance_gamma=0.2,           # entropy weight
        resurrection_threshold=0.01,     # attention threshold for reheating
        micro_page_size=None,
        strict_alignment=False,
        balloon_driver=None,
        force_qos=False
    ):
        self.page_size = page_size
        self.max_active_pages = max_active_pages
        self.max_fp8_pages = max_fp8_pages
        self.max_int8_pages = max_int8_pages
        self.max_int4_pages = max_int4_pages
        self.max_int2_pages = max_int2_pages
        self.max_one_bit_pages = max_one_bit_pages
        self.sink_tokens = sink_tokens
        self.threshold_sigma = threshold_sigma
        self.vram_oom_threshold_ratio = vram_oom_threshold_ratio
        self.importance_alpha = importance_alpha
        self.importance_beta = importance_beta
        self.importance_gamma = importance_gamma
        self.resurrection_threshold = resurrection_threshold
        self.micro_page_size = micro_page_size
        self.strict_alignment = strict_alignment
        self.balloon_driver = balloon_driver
        self.force_qos = force_qos

from .logger import argus_log

def calculate_tensor_entropy(tensor):
    """
    Computes a fast, numerically stable entropy proxy of the tensor's activation magnitudes.
    """
    if tensor is None or tensor.numel() == 0:
        return 0.0
    abs_t = torch.abs(tensor).to(torch.float32)
    sum_abs = abs_t.sum()
    if sum_abs == 0:
        return 0.0
    p = abs_t / sum_abs
    p = torch.clamp(p, min=1e-9)
    entropy = -torch.sum(p * torch.log(p)).item()
    return entropy

from .outliers import _apply_outlier_restoration, isolate_outliers  # noqa: F401  (re-exported)


class PagedDynamicKVCache:
    @property
    def active_pages(self):
        return self._cpp_manager.active_pages

    @active_pages.setter
    def active_pages(self, val):
        self._cpp_manager.active_pages = val

    @property
    def pages_by_tier(self):
        return self._cpp_manager.pages_by_tier

    @pages_by_tier.setter
    def pages_by_tier(self, val):
        self._cpp_manager.pages_by_tier = val

    @property
    def max_active_pages(self):
        return self._cpp_manager.max_active_pages

    @max_active_pages.setter
    def max_active_pages(self, val):
        self._cpp_manager.max_active_pages = val

    @property
    def generation_step(self):
        return self._cpp_manager.generation_step

    @generation_step.setter
    def generation_step(self, val):
        self._cpp_manager.generation_step = val

    @property
    def prefetch_cache(self):
        """Page IDs currently held in the C++ manager's live prefetch cache."""
        return {pid: True for pid in self._cpp_manager.get_prefetched_page_ids()}

    @property
    def prefetch_hits(self):
        """Count of resurrections/reads served from the prefetch cache instead of a fresh dequant."""
        return self._cpp_manager.get_prefetch_hit_count()

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
        config=None,
        pipeline=None,
        strict_alignment=False,
        balloon_driver=None,
        force_qos=False
    ):
        """
        Generic Outlier-Aware 7-Tier Token-Based Paged Dynamic KV Cache Manager.
        Memory Lifecycle:
            FP16 (Sinks) + FP16 (Active) -> FP8 (Light) -> INT8 (Medium) -> INT4 (Heavy) -> INT2 (Super Heavy) -> 1-Bit (Sign) -> JL Projection (Archive)
        """
        if pipeline is None:
            if config is not None and hasattr(config, 'tiers'):
                pipeline = config
                
        if pipeline is None:
            if config is None:
                self.config = ArgusConfig(
                    page_size=page_size,
                    max_active_pages=max_active_pages,
                    max_fp8_pages=max_fp8_pages,
                    max_int8_pages=max_int8_pages,
                    max_int4_pages=max_int4_pages,
                    max_int2_pages=max_int2_pages,
                    max_one_bit_pages=max_one_bit_pages,
                    sink_tokens=sink_tokens,
                    threshold_sigma=threshold_sigma,
                    strict_alignment=strict_alignment,
                    balloon_driver=balloon_driver,
                    force_qos=force_qos
                )
            else:
                self.config = config
                
            from argus_cache.backends.quantization import (
                FP8Backend, INT8Backend, INT4Backend, INT2Backend, OneBitBackend, JLProjectionBackend
            )
            from argus_cache.backends.eviction import ImportanceSortPolicy
            pipeline = PipelineConfig(
                tiers=[
                    TierSpec("fp8", FP8Backend(), max_pages=self.config.max_fp8_pages, priority=1),
                    TierSpec("int8", INT8Backend(), max_pages=self.config.max_int8_pages, priority=2),
                    TierSpec("int4", INT4Backend(), max_pages=self.config.max_int4_pages, priority=3),
                    TierSpec("int2", INT2Backend(), max_pages=self.config.max_int2_pages, priority=4),
                    TierSpec("one_bit", OneBitBackend(), max_pages=self.config.max_one_bit_pages, priority=5),
                    TierSpec("jl", JLProjectionBackend(), max_pages=-1, priority=6),
                ],
                eviction_policy=ImportanceSortPolicy(),
                page_size=self.config.page_size,
                sink_tokens=self.config.sink_tokens,
                threshold_sigma=self.config.threshold_sigma
            )
        else:
            self.config = pipeline
            
        self.pipeline_config = pipeline
        self.tier_specs = pipeline.tiers
        self.tier_name_to_spec = {spec.name: spec for spec in self.tier_specs}
        self.eviction_policy = pipeline.eviction_policy
        
        self.page_size = self.pipeline_config.page_size
        if self.page_size % 128 != 0:
            if getattr(self.config, 'strict_alignment', False):
                raise ValueError(f"page_size ({self.page_size}) must be a multiple of 128 for FlashAttention alignment when strict_alignment is enabled.")
            else:
                argus_log("WARNING", f"page_size ({self.page_size}) is not a multiple of 128. FlashAttention block alignment is bypassed.", line_no=245)

        import argus_cpp_backend
        # 10. Load C++ Extension Backend
        device_id = torch.cuda.current_device() if torch.cuda.is_available() else 0
        self._cpp_manager = argus_cpp_backend.ArgusCppManager(self.page_size, self.config.max_active_pages, device_id)
        
        # Configure C++ parameters
        if hasattr(self.config, "w_proj"):
            self._cpp_manager.set_jl_projection_matrix(self.config.w_proj)
        if hasattr(self.config, "recon_operator"):
            self._cpp_manager.set_jl_recon_operator(self.config.recon_operator)

        # JL matrices depend on the device/dtype of the first tensor that
        # actually reaches the tier, which isn't known at construction time.
        # Register lazy providers so C++ can request them on first use instead
        # of relying on an eagerly-built (and possibly wrongly-shaped) default.
        cache_ref = weakref.ref(self)

        def _jl_projection_provider(sample):
            cache = cache_ref()
            if cache is None:
                raise RuntimeError("ARGUS cache was released before JL projection")
            return cache.get_jl_projection_matrix(
                sample.device, sample.dtype, sample.shape[-2]
            )

        def _jl_recon_provider(compressed_sample, seq_len):
            cache = cache_ref()
            if cache is None:
                raise RuntimeError("ARGUS cache was released before JL reconstruction")
            return cache.get_jl_reconstruction_operator(
                compressed_sample.device, compressed_sample.dtype, seq_len=seq_len
            )

        self._cpp_manager.set_jl_projection_provider(_jl_projection_provider)
        self._cpp_manager.set_jl_recon_provider(_jl_recon_provider)

        # Restore the pluggable eviction-policy hooks into the C++ hot path:
        # which page leaves the active pool, and per-page access bookkeeping
        # (reference bits / heat registers / importance recalculation).
        self._protected_page_id = None
        # A cache's page lifecycle is mutable and must be linearizable. CUDA
        # kernels may run asynchronously, but two Python threads cannot safely
        # select victims / resurrect / split the same page lists concurrently.
        self._attention_lock = threading.RLock()

        def _select_active_victim(pages):
            cache = cache_ref()
            all_pages = list(pages)
            if cache is None:
                return 0
            candidates = [
                page
                for page in all_pages
                if page.page_id != cache._protected_page_id
            ]
            if not candidates:
                candidates = all_pages
            if cache.eviction_policy is not None:
                victim = cache.eviction_policy.select_victim(candidates, context={})
            else:
                victim = min(candidates, key=lambda page: page.importance_score)
            return all_pages.index(victim)

        self._cpp_manager.set_active_pool_victim_selector(_select_active_victim)

        def _on_page_access(page, block_w, step):
            cache = cache_ref()
            if cache is None:
                return
            if cache.eviction_policy is not None and hasattr(cache.eviction_policy, 'on_access'):
                cache.eviction_policy.on_access(page, block_w, step)
            cache._calculate_importance(page)
        self._cpp_manager.set_on_page_access_callback(_on_page_access)
        self._cpp_manager.set_force_qos(getattr(self.config, 'force_qos', False))
        self._cpp_manager.set_streaming_attention(
            getattr(self.config, "streaming_attention", False)
        )


        for spec in self.tier_specs:
            self._cpp_manager.add_tier_cpp(spec.name)
            self._cpp_manager.set_tier_max_pages(spec.name, spec.max_pages)
            # Give the native engine this tier's storage format. Tiers whose
            # backend declares no native codec stay unregistered and spill
            # uncompressed rather than being decoded with the wrong layout.
            self._sync_tier_codec(spec)

        self._cpp_manager.set_tier_pipeline_cpp([s.name for s in self.tier_specs])
        self.zero_copy_pool = ZeroCopyHostPool()
        self.zero_copy_pool._backend = self._cpp_manager.get_host_pool()
        self._zero_copy_tensor_registry = {}

        # Synchronize tier specs with C++
        for tier_name, spec in self.tier_name_to_spec.items():
            self._cpp_manager.set_tier_max_pages(tier_name, spec.max_pages)

        self.max_active_pages = self.config.max_active_pages
        self.threshold_sigma = self.pipeline_config.threshold_sigma
        self.micro_page_size = getattr(self.pipeline_config, 'micro_page_size', None)
        if self.micro_page_size is None:
            self.micro_page_size = getattr(self.config, 'micro_page_size', None)
        if self.micro_page_size is None:
            # Default micro-page size must stay a multiple of 8: the bit-packed
            # tiers require it (one_bit packs 8-wide, int2 packs 4-wide, int4
            # packs 2-wide) — anything not divisible by 8 crashes the first
            # time a split micro-page cascades into one of those tiers. Callers
            # that explicitly pass micro_page_size opt out of this guard.
            self.micro_page_size = max(8, (self.page_size // 4 // 8) * 8)
        self._granularity = GranularityManager(self, self.micro_page_size)
        self.balloon_driver = getattr(pipeline, 'balloon_driver', None)
        if self.balloon_driver is None:
            self.balloon_driver = getattr(self.config, 'balloon_driver', None)
        
        # Save legacy backing values for backward compatibility properties
        self._max_fp8_pages = self.tier_name_to_spec["fp8"].max_pages if "fp8" in self.tier_name_to_spec else 2
        self._max_int8_pages = self.tier_name_to_spec["int8"].max_pages if "int8" in self.tier_name_to_spec else 2
        self._max_int4_pages = self.tier_name_to_spec["int4"].max_pages if "int4" in self.tier_name_to_spec else 2
        self._max_int2_pages = self.tier_name_to_spec["int2"].max_pages if "int2" in self.tier_name_to_spec else 2
        self._max_one_bit_pages = self.tier_name_to_spec["one_bit"].max_pages if "one_bit" in self.tier_name_to_spec else 2

        assert self.page_size % 8 == 0, "Page size must be a multiple of 8 for 1-bit packing."
        
        # Generation step tracking for recency scoring
        self.generation_step = 0
        self.event_log = []
        
        # Outlier-Aware: Attention Sinks (First N tokens kept in FP16 permanently)
        self.sink_tokens = self.pipeline_config.sink_tokens
        self.sink_k = None
        self.sink_v = None
        
        # VIP Outliers: Newline & Rhyme Anchors kept in FP16 permanently
        self.anchor_k = None
        self.anchor_v = None
        
        # (pages_by_tier and active_pages are now dynamic descriptors delegating to C++)
        self.active_pool_k = None
        self.active_pool_v = None
        
        # Shared static JL operators (page_size-length, back-compat attributes)
        self.w_proj = None
        self._jl_recon_operator = None
        # Per-(device, dtype, seq_len) cache lives in JLOperatorCache —
        # variable-granularity micro-pages need differently-shaped operators
        # than full-size pages, since JL projects along the sequence axis.
        self._jl_operators = JLOperatorCache(
            page_size=self.page_size,
            on_full_page_cuda=self._publish_jl_operator,
        )
        
        # Per-tier compressed page pools. Shapes come from each tier's declared
        # codec, not from its name, so plugin tiers get pools too.
        self._pool_allocator = StaticPoolAllocator(page_size=self.page_size)

        # Static buffer state (allocated on demand during ensure_pools_allocated)
        self._pools_allocated = False
        
        # Speculative prefetching (prefetch_cache/prefetch_hits are properties
        # backed by the C++ manager's real prefetch state; see above)
        self.prefetch_misses = 0
        
        # CUDA Stream for async prefetching
        self.prefetch_stream = None
        
        # Context swapping / Zero-OOM multi-tenant guard state
        self.is_swapped_out = False
        self._host_spill = HostSpillManager(self)

        # Telemetry metrics
        self.num_resurrections = 0
        self.num_cpu_spills = 0
        self.total_dequant_time = 0.0
        self.num_dequants = 0
        self.total_demotions = 0
        
        # Advanced telemetry metrics
        self.dequant_latencies = []
        self.pending_cuda_events = []
        self.total_attention_calls = 0
        self.page_lifetimes = {}        # page_id -> birth_step
        self.total_page_lifetimes = 0
        self.completed_page_lifetimes_count = 0
        self.resurrection_depths = []   # list of depths
        self.cascade_counts = {
            'fp16_to_fp8': 0,
            'fp8_to_int8': 0,
            'int8_to_int4': 0,
            'int4_to_int2': 0,
            'int2_to_one_bit': 0,
            'one_bit_to_jl': 0
        }
        
        # PCIe streaming metrics (separate from dequant latency)
        self.pcie_swap_latencies = []
        self.num_pcie_swaps = 0
        self.pcie_bytes_swapped = 0
        
        # Locality Predictor State
        self.page_access_ema = {}
        self.page_access_history = {}

        # Observability lives in its own module; the manager keeps thin
        # delegating methods so the public API is unchanged.
        from argus_cache.core.telemetry import CacheTelemetry
        self._telemetry = CacheTelemetry(self)

        # Compatibility placeholders.  Older builds kept every compressed
        # tier fully decompressed here between decode steps.  That persistent
        # FP16 mirror cancelled the cache's memory saving and was the main
        # reason the HuggingFace path used more VRAM than an exact cache.
        self._cache_version = 0
        self._tiers_version = -1
        self._decompressed_tiers_k = None
        self._decompressed_tiers_v = None

    def _invalidate_decompressed_cache(self):
        self._decompressed_tiers_k = None
        self._decompressed_tiers_v = None
        self._cache_version += 1

    def close(self):
        """Release Python callbacks held by the native manager.

        ``ArgusCppManager`` owns callback objects that close over this Python
        cache (eviction, access accounting, and lazy JL operators).  Without
        explicitly clearing them, dropping a HuggingFace cache wrapper leaves
        a C++ -> Python -> C++ ownership cycle that Python's GC cannot see;
        successive benchmark arms then inherit every earlier cache allocation.
        """
        cpp = getattr(self, "_cpp_manager", None)
        if cpp is None:
            return
        cpp.set_active_pool_victim_selector(None)
        cpp.set_on_page_access_callback(None)
        cpp.set_jl_projection_provider(None)
        cpp.set_jl_recon_provider(None)

    @property
    def fp8_pages(self):
        return self.pages_by_tier.get("fp8", [])
    @fp8_pages.setter
    def fp8_pages(self, val):
        self.pages_by_tier["fp8"] = val

    @property
    def int8_pages(self):
        return self.pages_by_tier.get("int8", [])
    @int8_pages.setter
    def int8_pages(self, val):
        self.pages_by_tier["int8"] = val

    @property
    def int4_pages(self):
        return self.pages_by_tier.get("int4", [])
    @int4_pages.setter
    def int4_pages(self, val):
        self.pages_by_tier["int4"] = val

    @property
    def int2_pages(self):
        return self.pages_by_tier.get("int2", [])
    @int2_pages.setter
    def int2_pages(self, val):
        self.pages_by_tier["int2"] = val

    @property
    def one_bit_pages(self):
        return self.pages_by_tier.get("one_bit", [])
    @one_bit_pages.setter
    def one_bit_pages(self, val):
        self.pages_by_tier["one_bit"] = val

    @property
    def jl_pages(self):
        return self.pages_by_tier.get("jl", [])
    @jl_pages.setter
    def jl_pages(self, val):
        self.pages_by_tier["jl"] = val

    @property
    def max_fp8_pages(self):
        spec = self.tier_name_to_spec.get("fp8")
        return spec.max_pages if spec else getattr(self, "_max_fp8_pages", 2)
    @max_fp8_pages.setter
    def max_fp8_pages(self, val):
        self._max_fp8_pages = val
        spec = self.tier_name_to_spec.get("fp8")
        if spec:
            spec.max_pages = val

    @property
    def max_int8_pages(self):
        spec = self.tier_name_to_spec.get("int8")
        return spec.max_pages if spec else getattr(self, "_max_int8_pages", 2)
    @max_int8_pages.setter
    def max_int8_pages(self, val):
        self._max_int8_pages = val
        spec = self.tier_name_to_spec.get("int8")
        if spec:
            spec.max_pages = val

    @property
    def max_int4_pages(self):
        spec = self.tier_name_to_spec.get("int4")
        return spec.max_pages if spec else getattr(self, "_max_int4_pages", 2)
    @max_int4_pages.setter
    def max_int4_pages(self, val):
        self._max_int4_pages = val
        spec = self.tier_name_to_spec.get("int4")
        if spec:
            spec.max_pages = val

    @property
    def max_int2_pages(self):
        spec = self.tier_name_to_spec.get("int2")
        return spec.max_pages if spec else getattr(self, "_max_int2_pages", 2)
    @max_int2_pages.setter
    def max_int2_pages(self, val):
        self._max_int2_pages = val
        spec = self.tier_name_to_spec.get("int2")
        if spec:
            spec.max_pages = val

    @property
    def max_one_bit_pages(self):
        spec = self.tier_name_to_spec.get("one_bit")
        return spec.max_pages if spec else getattr(self, "_max_one_bit_pages", 2)
    @max_one_bit_pages.setter
    def max_one_bit_pages(self, val):
        self._max_one_bit_pages = val
        spec = self.tier_name_to_spec.get("one_bit")
        if spec:
            spec.max_pages = val

    @property
    def page_offsets(self):
        """
        Dynamically constructs the page offsets tracking tensor (mostly for tests and debugging).
        """
        offsets = []
        current_offset = 0

        # Sinks
        if self.sink_k is not None:
            offsets.append(current_offset)
            current_offset += self.sink_k.shape[-2]

        # Anchors
        if self.anchor_k is not None:
            offsets.append(current_offset)
            current_offset += self.anchor_k.shape[-2]

        # Active Pages
        for page in self.active_pages:
            offsets.append(current_offset)
            current_offset += page.get('page_size', self.page_size)

        # Buffer
        if self.k_buffer is not None and self.k_buffer.shape[-2] > 0:
            offsets.append(current_offset)
            current_offset += self.k_buffer.shape[-2]

        # Compressed Tiers (Coldest to Hottest)
        for spec in reversed(self.tier_specs):
            for page in self.pages_by_tier.get(spec.name, []):
                offsets.append(current_offset)
                current_offset += page.get('page_size', self.page_size)

        # Determine target device
        device = "cpu"
        if self.sink_k is not None:
            device = self.sink_k.device
        elif self.active_pool_k is not None:
            device = self.active_pool_k.device
            
        return torch.tensor(offsets, dtype=torch.long, device=device)

    @page_offsets.setter
    def page_offsets(self, val):
        pass

    @property
    def pools_by_tier(self):
        """The live ``{f"{tier}_{field}": tensor}`` mapping owned by the allocator.

        Read-only by design: pools are created through the allocator so their
        shapes stay derived from tier capabilities. Mutating entries in place
        (which the tier-removal path does) is still fine.
        """
        return self._pool_allocator.pools

    @property
    def fp8_pool_k_q(self):
        return self.pools_by_tier.get("fp8_key_q")
    @property
    def fp8_pool_v_q(self):
        return self.pools_by_tier.get("fp8_value_q")
    @property
    def fp8_pool_k_scales(self):
        return self.pools_by_tier.get("fp8_key_scales")
    @property
    def fp8_pool_v_scales(self):
        return self.pools_by_tier.get("fp8_value_scales")

    @property
    def int8_pool_k_q(self):
        return self.pools_by_tier.get("int8_key_q")
    @property
    def int8_pool_v_q(self):
        return self.pools_by_tier.get("int8_value_q")
    @property
    def int8_pool_k_scales(self):
        return self.pools_by_tier.get("int8_key_scales")
    @property
    def int8_pool_v_scales(self):
        return self.pools_by_tier.get("int8_value_scales")

    @property
    def int4_pool_k_packed(self):
        return self.pools_by_tier.get("int4_key_q")
    @property
    def int4_pool_v_packed(self):
        return self.pools_by_tier.get("int4_value_q")
    @property
    def int4_pool_k_scales(self):
        return self.pools_by_tier.get("int4_key_scales")
    @property
    def int4_pool_v_scales(self):
        return self.pools_by_tier.get("int4_value_scales")
    @property
    def int4_pool_k_min(self):
        return self.pools_by_tier.get("int4_key_min_vals")
    @property
    def int4_pool_v_min(self):
        return self.pools_by_tier.get("int4_value_min_vals")

    @property
    def int2_pool_k_packed(self):
        return self.pools_by_tier.get("int2_key_q")
    @property
    def int2_pool_v_packed(self):
        return self.pools_by_tier.get("int2_value_q")
    @property
    def int2_pool_k_scales(self):
        return self.pools_by_tier.get("int2_key_scales")
    @property
    def int2_pool_v_scales(self):
        return self.pools_by_tier.get("int2_value_scales")
    @property
    def int2_pool_k_min(self):
        return self.pools_by_tier.get("int2_key_min_vals")
    @property
    def int2_pool_v_min(self):
        return self.pools_by_tier.get("int2_value_min_vals")

    @property
    def one_bit_pool_k_packed(self):
        return self.pools_by_tier.get("one_bit_key_q")
    @property
    def one_bit_pool_v_packed(self):
        return self.pools_by_tier.get("one_bit_value_q")
    @property
    def one_bit_pool_k_scales(self):
        return self.pools_by_tier.get("one_bit_key_scales")
    @property
    def one_bit_pool_v_scales(self):
        return self.pools_by_tier.get("one_bit_value_scales")

    def _publish_jl_operator(self, kind, tensor):
        """Mirror a full-page CUDA operator into the native manager.

        Only CUDA, full-page operators reach here: the C++ auto-cascade holds
        exactly one of each and always works on CUDA tensors, so a CPU-built
        or micro-page operator (e.g. from split_page on a swapped-out page)
        would silently poison the next demotion.
        """
        if kind == "projection":
            self.w_proj = tensor
            if hasattr(self, '_cpp_manager'):
                self._cpp_manager.set_jl_projection_matrix(tensor)
        else:
            self._jl_recon_operator = tensor
            if hasattr(self, '_cpp_manager'):
                self._cpp_manager.set_jl_recon_operator(tensor)

    def get_jl_projection_matrix(self, device, dtype, seq_len=None):
        """Projection matrix [seq_len // 4, seq_len]. See JLOperatorCache."""
        return self._jl_operators.projection(device, dtype, seq_len)

    def get_jl_reconstruction_operator(self, device, dtype, alpha=1e-3, seq_len=None):
        """Reconstruction operator [seq_len, seq_len // 4]. See JLOperatorCache."""
        return self._jl_operators.reconstruction(device, dtype, alpha, seq_len)


    def log_event(self, event_type, page_id, **kwargs):
        import json
        event = {
            'event': event_type,
            'page_id': page_id,
            'step': self.generation_step,
            'timestamp': getattr(self, 'generation_step', 0)
        }
        event.update(kwargs)
        if not hasattr(self, 'event_log'):
            self.event_log = []
        self.event_log.append(event)
        
        # Real-time structured lifecycle tracing (ignored by git via tests/*.jsonl)
        try:
            import os
            os.makedirs("tests", exist_ok=True)
            with open("tests/argus_attention_trace.jsonl", "a") as f:
                f.write(json.dumps(event) + "\n")
        except Exception:
            pass

    @property
    def k_buffer(self):
        if not hasattr(self, 'buffer_length') or self.buffer_length == 0:
            return None
        return self.static_k_buffer[..., :self.buffer_length, :]

    @property
    def v_buffer(self):
        if not hasattr(self, 'buffer_length') or self.buffer_length == 0:
            return None
        return self.static_v_buffer[..., :self.buffer_length, :]

    def _get_pool_tensor(self, tier_name, key, shape, dtype, device):
        return self._pool_allocator.ensure(
            tier_name, key, self.tier_name_to_spec[tier_name].max_pages,
            shape, dtype, device,
        )

    def _ensure_pools_allocated(self, keys: torch.Tensor, values: torch.Tensor):
        batch, num_heads, _, head_dim = keys.shape
        device = keys.device
        dtype = keys.dtype
        
        if (hasattr(self, "_pools_allocated") and self._pools_allocated and
            self.static_k_buffer.shape[0] == batch and
            self.static_k_buffer.shape[1] == num_heads and
            self.static_k_buffer.shape[3] == head_dim and
            self.static_k_buffer.device == device and
            self.static_k_buffer.dtype == dtype):
            return
            
        # Only an incomplete page belongs in the staging buffer.  Prefill can
        # arrive as tens of thousands of tokens; sizing this buffer to the
        # whole input retained an exact-cache-sized tensor on every layer even
        # after all full pages had moved into the native manager.
        self.static_k_buffer = torch.zeros(batch, num_heads, self.page_size, head_dim, device=device, dtype=dtype)
        self.static_v_buffer = torch.zeros(batch, num_heads, self.page_size, head_dim, device=device, dtype=dtype)
        self.buffer_length = 0

        # Active and compressed page storage is owned by ArgusCppManager.  The
        # old Python mirrors were never passed to C++ and consumed ~23 MiB on
        # the 24-layer reference model before holding a single useful byte.
        self.active_pool_k = None
        self.active_pool_v = None
        self.active_pool_idx = 0

        self._pool_allocator.reset()
        
        # JL operators are intentionally lazy.  The native manager invokes the
        # providers registered in __init__ only when a page actually reaches a
        # projection tier; building a page_size x page_size/4 inverse on every
        # layer made even sub-page requests pay the deepest archive's startup
        # and memory cost.
        self._pools_allocated = True

    def _allocate_pool_for_tier(self, name, max_pages, device=None, dtype=None, batch=None, num_heads=None, head_dim=None):
        """Pre-allocates the static pools for a specific tier if pools are already active."""
        if device is None or dtype is None or batch is None or num_heads is None or head_dim is None:
            if not hasattr(self, 'static_k_buffer') or self.static_k_buffer is None:
                return # Pools not yet allocated
            batch, num_heads, _, head_dim = self.static_k_buffer.shape
            device = self.static_k_buffer.device
            dtype = self.static_k_buffer.dtype

        spec = self.tier_name_to_spec.get(name)
        if spec is None:
            return
        self._pool_allocator.allocate_for_tier(
            spec, max_pages, device, dtype, batch, num_heads, head_dim
        )

    def _is_projection_tier(self, tier_name: str) -> bool:
        """Whether a tier's backend is a linear projection rather than a quantizer.

        Capability lookup, not a name check: a projection tier needs a
        ``w_proj``/``recon_operator`` threaded through compress/decompress,
        and that requirement belongs to the backend, not to the string "jl".
        """
        spec = self.tier_name_to_spec.get(tier_name)
        return spec is not None and spec.is_projection

    def _sync_tier_codec(self, spec: TierSpec):
        """Publish a tier's native storage format to the C++ engine.

        Without this, a plugin tier reaches C++ as an unknown name and its
        pages spill uncompressed. With it, the native engine compresses the
        tier using the same generic kernel path as the built-ins.
        """
        cpp = getattr(self, '_cpp_manager', None)
        if cpp is None:
            return

        caps = spec.capabilities
        native = caps.native_codec if caps is not None else None
        if native is None:
            # No declared native format: leave the tier unregistered so C++
            # falls back to a lossless host spill rather than guessing a
            # bit layout and corrupting the page.
            if cpp.has_codec(spec.name):
                cpp.unregister_codec(spec.name)
            return

        import argus_cpp_backend
        codec = argus_cpp_backend.TierCodec()
        codec.name = spec.name
        codec.kind = {
            "signed_linear": argus_cpp_backend.CodecKind.SIGNED_LINEAR,
            "unsigned_affine": argus_cpp_backend.CodecKind.UNSIGNED_AFFINE,
            "sign_packed": argus_cpp_backend.CodecKind.SIGN_PACKED,
            "projection": argus_cpp_backend.CodecKind.PROJECTION,
            "passthrough": argus_cpp_backend.CodecKind.PASSTHROUGH,
            "ggml_q8_0": argus_cpp_backend.CodecKind.GGML_Q8_0,
            "ggml_q4_0": argus_cpp_backend.CodecKind.GGML_Q4_0,
        }[native.kind]
        codec.bits = native.bits
        codec.lossy = caps.lossy
        codec.compression_ratio = native.compression_ratio
        cpp.register_codec(codec)

    def add_tier(self, spec: TierSpec, index: int = -1):
        """Adds a new cache tier spec dynamically to the pipeline."""
        if spec.name in self.tier_name_to_spec:
            raise ValueError(f"Tier with name {spec.name} already exists.")

        if index == -1:
            self.tier_specs.append(spec)
        else:
            self.tier_specs.insert(index, spec)

        self.tier_name_to_spec[spec.name] = spec
        if hasattr(self, '_cpp_manager'):
            self._cpp_manager.add_tier_cpp(spec.name)
            self._cpp_manager.set_tier_pipeline_cpp([s.name for s in self.tier_specs])
            self._cpp_manager.set_tier_max_pages(spec.name, spec.max_pages)
            self._sync_tier_codec(spec)

        if spec.use_static_pool:
            self._allocate_pool_for_tier(spec.name, spec.max_pages)
            
        argus_log("INFO", f"Dynamically added tier: {spec.name.upper()} (priority={spec.priority}, max_pages={spec.max_pages})", line_no=683)
        self._invalidate_decompressed_cache()

    def remove_tier(self, name: str):
        """Removes a cache tier by name dynamically. Re-evaluates and resurrects any pages currently in it first."""
        spec = self.tier_name_to_spec.pop(name, None)
        if spec is None:
            raise ValueError(f"Tier with name {name} does not exist.")
            
        # Resurrect pages currently in this tier first to avoid losing data
        pages = list(self.pages_by_tier.get(name, []))
        for p in pages:
            self._resurrect_page(p, name)
            
        if hasattr(self, '_cpp_manager'):
            self._cpp_manager.remove_tier_cpp(name)
        
        # Remove spec from list
        self.tier_specs = [s for s in self.tier_specs if s.name != name]
        
        if hasattr(self, '_cpp_manager'):
            self._cpp_manager.set_tier_pipeline_cpp([s.name for s in self.tier_specs])

        
        # Clean up pools
        keys_to_remove = [k for k in self.pools_by_tier.keys() if k.startswith(f"{name}_")]
        for k in keys_to_remove:
            self.pools_by_tier.pop(k, None)
            
        argus_log("INFO", f"Dynamically removed tier: {name.upper()}", line_no=684)
        self._invalidate_decompressed_cache()

    def push_new_tokens(self, keys: torch.Tensor, values: torch.Tensor, is_anchor: torch.Tensor = None):
        """Append tokens as one linearizable cache-lifecycle transition."""
        with self._attention_lock:
            return self._push_new_tokens_unlocked(keys, values, is_anchor=is_anchor)

    def _push_new_tokens_unlocked(self, keys: torch.Tensor, values: torch.Tensor, is_anchor: torch.Tensor = None):
        """
        Pushes new key and value tensors into the cache. Segments pages automatically,
        while isolating the initial Attention Sinks and VIP anchors.
        """
        # Anchors/sinks are extracted below from whatever device the caller
        # passed in — if that's CPU, sink_k/sink_v end up CPU (and pinned) by
        # design, since they're meant to sit in zero-copy host memory rather
        # than consume GPU VRAM. Only the remainder of the sequence (the part
        # that actually enters the paged cache) is moved to CUDA, and only
        # after extraction so this ordering is preserved.
        self._check_and_prevent_oom()
        # 0. Extract Newline / Rhyme Anchors if is_anchor mask is provided
        if is_anchor is not None:
            seq_len = keys.shape[-2]
            if is_anchor.dim() > 1:
                is_anchor = is_anchor.view(-1)
            
            # Match is_anchor length to keys sequence dimension
            if is_anchor.shape[0] != seq_len:
                is_anchor = is_anchor[:seq_len]
                if is_anchor.shape[0] < seq_len:
                    padding = torch.zeros(seq_len - is_anchor.shape[0], dtype=torch.bool, device=keys.device)
                    is_anchor = torch.cat([is_anchor, padding])
            
            anchor_indices = torch.where(is_anchor)[0]
            if len(anchor_indices) > 0:
                anchors_k = keys[..., anchor_indices, :]
                anchors_v = values[..., anchor_indices, :]
                
                if self.anchor_k is None:
                    self.anchor_k = anchors_k.clone()
                    self.anchor_v = anchors_v.clone()
                else:
                    self.anchor_k = torch.cat([self.anchor_k, anchors_k], dim=-2)
                    self.anchor_v = torch.cat([self.anchor_v, anchors_v], dim=-2)
                
                # Filter out anchors from the main sequence to bypass normal quantization
                non_anchor_indices = torch.where(~is_anchor)[0]
                keys = keys[..., non_anchor_indices, :]
                values = values[..., non_anchor_indices, :]
                
                if keys.shape[-2] == 0:
                    return

        # 1. Extract Attention Sinks if not already done
        if self.sink_k is None and self.sink_tokens > 0:
            seq_len = keys.shape[-2]
            if seq_len >= self.sink_tokens:
                self.sink_k = keys[..., :self.sink_tokens, :].clone()
                self.sink_v = values[..., :self.sink_tokens, :].clone()
                if self.sink_k.device.type == "cpu":
                    self.sink_k = self.sink_k.pin_memory()
                    self.sink_v = self.sink_v.pin_memory()
                
                # Rest of the sequence goes to normal cache
                keys = keys[..., self.sink_tokens:, :]
                values = values[..., self.sink_tokens:, :]
            else:
                # If prompt is very short, keep it in sinks until we accumulate enough
                self.sink_k = keys.clone()
                self.sink_v = values.clone()
                if self.sink_k.device.type == "cpu":
                    self.sink_k = self.sink_k.pin_memory()
                    self.sink_v = self.sink_v.pin_memory()
                return

        # Now that sinks/anchors have been carved off, move the rest of the
        # sequence (the part that actually enters the paged cache) to CUDA.
        if torch.cuda.is_available():
            if keys.device.type == "cpu":
                keys = keys.cuda()
            if values.device.type == "cpu":
                values = values.cuda()

        # Ensure pools are allocated with correct batch and head size
        self._ensure_pools_allocated(keys, values)
        
        def push_page(page_k, page_v):
            # Record page list before push to detect demotions
            before_active_ids = {p.page_id for p in self._cpp_manager.active_pages}

            # Delegate page creation to C++ manager
            self._cpp_manager.push_new_tokens(page_k, page_v)

            # Detect demoted pages
            after_active = self._cpp_manager.active_pages
            after_active_ids = {p.page_id for p in after_active}
            demoted_ids = before_active_ids - after_active_ids
            for pid in demoted_ids:
                tier_to = "fp8"
                for spec in self.tier_specs:
                    if any(p.page_id == pid for p in self.pages_by_tier.get(spec.name, [])):
                        tier_to = spec.name
                        break
                self.log_event("demote", pid, tier_from="active", tier_to=tier_to)

            # Find the new page that was created
            new_page = [p for p in after_active if p.page_id not in before_active_ids]
            if new_page:
                self.log_event("create", new_page[0].page_id)

        num_new = int(keys.shape[-2])
        cursor = 0

        # Complete a partial page left by the previous decode step.
        if self.buffer_length:
            take = min(self.page_size - self.buffer_length, num_new)
            end = self.buffer_length + take
            self.static_k_buffer[..., self.buffer_length:end, :].copy_(
                keys[..., :take, :]
            )
            self.static_v_buffer[..., self.buffer_length:end, :].copy_(
                values[..., :take, :]
            )
            self.buffer_length = end
            cursor = take
            if self.buffer_length == self.page_size:
                push_page(self.static_k_buffer.clone(), self.static_v_buffer.clone())
                self.buffer_length = 0

        # Stream full prefill pages directly.  Clone each slice because active
        # pages must not retain a view (and therefore the storage) of the whole
        # model-produced prefill tensor.
        while cursor + self.page_size <= num_new:
            end = cursor + self.page_size
            push_page(
                keys[..., cursor:end, :].clone(),
                values[..., cursor:end, :].clone(),
            )
            cursor = end

        # Keep only the final incomplete page for the next update.
        remaining = num_new - cursor
        if remaining:
            self.static_k_buffer[..., :remaining, :].copy_(keys[..., cursor:, :])
            self.static_v_buffer[..., :remaining, :].copy_(values[..., cursor:, :])
            self.buffer_length = remaining

    def _calculate_importance(self, page):
        alpha = self.config.importance_alpha
        beta = self.config.importance_beta
        gamma = self.config.importance_gamma
        
        attention_sum = page.get('attention_sum', 0.0)
        last_step = page.get('last_step_accessed', 0)
        recency = 1.0 / (1.0 + float(self.generation_step - last_step))
        entropy = page.get('entropy', 0.0)
        
        # --- Locality Predictor ---
        page_id = page.get('page_id')
        accessed_now = 1.0 if (self.generation_step == last_step) else 0.0
        prev_ema = self.page_access_ema.get(page_id, 0.0)
        
        # EMA alpha coefficient = 0.15 for temporal access decay tracking
        ema = 0.15 * accessed_now + 0.85 * prev_ema
        self.page_access_ema[page_id] = ema
        
        # Stride prediction: check if page accesses happen in uniform step strides
        if page_id not in self.page_access_history:
            self.page_access_history[page_id] = []
        if accessed_now > 0.0 and (not self.page_access_history[page_id] or self.page_access_history[page_id][-1] != self.generation_step):
            self.page_access_history[page_id].append(self.generation_step)
            if len(self.page_access_history[page_id]) > 5:
                self.page_access_history[page_id].pop(0)
                
        stride_bonus = 0.0
        history = self.page_access_history.get(page_id, [])
        if len(history) >= 3:
            strides = [history[i] - history[i-1] for i in range(1, len(history))]
            # If uniform sequence access strides exist, predict next access step
            if len(set(strides)) == 1:
                next_predicted = history[-1] + strides[0]
                if abs(next_predicted - self.generation_step) <= 2:
                    stride_bonus = 0.8  # Strong future reuse prediction bonus
                    
        # --- Adaptive Entropy-Aware Policy ---
        # Highly cognitive, outlier-heavy pages get a dynamic scaling factor to stay warm longer
        # Boilerplate/repetitive low-entropy pages get scaled down for rapid cascading compression
        entropy_factor = 1.0
        if entropy > 10.0:
            entropy_factor = 1.3
        elif entropy < 5.0:
            entropy_factor = 0.6
            
        score = (alpha * attention_sum + beta * recency + gamma * (entropy * entropy_factor)) + (0.5 * ema) + stride_bonus
        page['importance_score'] = score
        return score

    def _get_free_pool_idx(self, pages_list, max_pages):
        used_indices = {p['pool_idx'] for p in pages_list if 'pool_idx' in p}
        for idx in range(max_pages):
            if idx not in used_indices:
                return idx
        return 0

    def _check_and_prevent_oom(self):
        # Calculate current context length
        total_tokens = (len(self.active_pages) + sum(len(pages) for pages in self.pages_by_tier.values())) * self.page_size
        # Soft-Eviction is bypassed in unit tests (bypassed if page_size < 128 or threshold_ratio <= 0.0)
        is_short_context = (total_tokens < 4096) and (self.page_size >= 128) and (self.config.vram_oom_threshold_ratio > 0.0)

        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            total_mem = torch.cuda.get_device_properties(device).total_memory
            allocated_mem = torch.cuda.memory_allocated(device)
            # Include invisible cuMemHostAlloc-locked bytes in pressure calculation
            invisible_locked = self.zero_copy_pool._total_allocated_bytes
            effective_allocated = allocated_mem + invisible_locked
            ratio = float(effective_allocated) / float(total_mem)
            
            # Soft-Eviction/Hysteresis: In short context windows, tolerate higher VRAM usage (92% instead of 85%)
            # to prevent premature and aggressive page demotions.
            target_threshold = self.config.vram_oom_threshold_ratio
            if is_short_context:
                target_threshold = max(target_threshold, 0.92)

            if ratio >= target_threshold:
                if self.balloon_driver is not None:
                    self.balloon_driver.inflate(self)
                    allocated_mem = torch.cuda.memory_allocated(device)
                    effective_allocated = allocated_mem + self.zero_copy_pool._total_allocated_bytes
                    ratio = float(effective_allocated) / float(total_mem)
                
                if ratio >= target_threshold:
                    self.num_cpu_spills += 1
                    frag_risk = self.zero_copy_pool.get_fragmentation_report().get('fragmentation_risk', 'N/A')
                    argus_log("WARNING", f"VRAM pressure detected (pytorch: {allocated_mem/(1024**3):.2f}GB + driver-locked: {invisible_locked/(1024**2):.1f}MB / {total_mem/(1024**3):.2f}GB, ratio: {ratio*100:.1f}%, frag_risk: {frag_risk}) → spilling cold pages to CPU Host RAM", line_no=431)
                    # Proactive OOM Protection: Spill all compressed pages to CPU Host RAM
                    self.swap_out_to_host()
            elif ratio < target_threshold - 0.15:
                if self.balloon_driver is not None:
                    self.balloon_driver.deflate(self)
        else:
            # Fallback for CPU unit testing: if vram_oom_threshold_ratio <= 0, trigger swap out!
            if self.config.vram_oom_threshold_ratio <= 0.0:
                if self.balloon_driver is not None:
                    self.balloon_driver.inflate(self)
                else:
                    self.num_cpu_spills += 1
                    argus_log("WARNING", f"VRAM pressure detected (CPU Fallback) → spilling cold pages to CPU Host RAM | vram_oom_threshold_ratio={self.config.vram_oom_threshold_ratio}", line_no=436)
                    self.swap_out_to_host()
            else:
                if self.balloon_driver is not None:
                    self.balloon_driver.deflate(self)

    def get_cache_telemetry(self):
        """(compression_ratio, bandwidth_saved_pct, total_pages, compressed_bytes)."""
        return self._telemetry.snapshot()

    def _sync_pending_cuda_events(self):
        return self._telemetry.sync_pending_cuda_events()

    def print_telemetry_summary(self):
        return self._telemetry.print_summary()
    def _resurrect_page(self, page, tier):
        import time
        use_cuda_event = torch.cuda.is_available()
        if use_cuda_event:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        else:
            start_time = time.perf_counter()

        # Record page list before to detect demotions
        before_active_ids = {p.page_id for p in self._cpp_manager.active_pages}

        # Delegate actual resurrection and memory movement to C++
        self._cpp_manager.resurrect_page(page, tier)

        # C++ only updates last_step_accessed on resurrection — recalculate
        # importance_score now so the freshly-resurrected (hot) page doesn't
        # look cold and get immediately re-evicted by manage_memory_lifecycle
        # below, which runs the pluggable eviction policy on the active pool.
        self._calculate_importance(page)

        if use_cuda_event:
            end_event.record()
            self.pending_cuda_events.append((start_event, end_event))
            self.num_resurrections += 1
        else:
            dequant_time = (time.perf_counter() - start_time) * 1000  # in ms
            self.total_dequant_time += dequant_time
            self.num_dequants += 1
            self.num_resurrections += 1
            self.dequant_latencies.append(dequant_time)
        try:
            spec = self.tier_name_to_spec.get(tier)
            depth = self.tier_specs.index(spec) + 1
        except ValueError:
            depth = 1
        self.resurrection_depths.append(depth)
        self.page_lifetimes[page.get('page_id')] = self.generation_step

        # Determine resurrection reason & log it
        rec = 1.0 / (1.0 + float(self.generation_step - page.get('last_step_accessed', 0)))
        att = page.get('attention_sum', 0.0)
        reason = "high attention recurrence" if att >= rec else "high recency bias"
        latency_str = "deferred (async)" if use_cuda_event else f"{dequant_time:.3f}ms"
        argus_log("INFO", f"Page {page.get('page_id')} resurrected (importance={page.get('importance_score', 0.0):.2f}) | Tier: {tier.upper()} -> FP16 Transient | Reason: {reason} | Dequant Latency: {latency_str}", line_no=700)
        
        if tier in ['int2', 'one_bit', 'jl']:
            argus_log("INFO", f"Restored {tier.upper()} archive page to FP16 transient buffer", line_no=701)
        if tier in ['int4', 'int2', 'one_bit', 'jl']:
            argus_log("INFO", f"Attention spike detected on archived memory", line_no=702)

        # 2. Let C++ manager handle memory lifecycle / eviction for the active pool
        self._protected_page_id = page.get('page_id')
        try:
            self._cpp_manager.manage_memory_lifecycle()
        finally:
            self._protected_page_id = None
        self.log_event("resurrect", page.get('page_id'), tier=tier)

        # Detect demoted pages from manage_memory_lifecycle
        after_active_ids = {p.page_id for p in self._cpp_manager.active_pages}
        res_pid = page.get('page_id')
        demoted_ids = (before_active_ids | {res_pid}) - after_active_ids
        for pid in demoted_ids:
            tier_to = "fp8"
            for spec in self.tier_specs:
                if any(p.page_id == pid for p in self.pages_by_tier.get(spec.name, [])):
                    tier_to = spec.name
                    break
            self.log_event("demote", pid, tier_from="active", tier_to=tier_to)

        self._invalidate_decompressed_cache()

    def manage_memory_lifecycle(self):
        """
        Transitions pages down the memory lifecycle when limits are exceeded.
        """
        while len(self.active_pages) >= self.max_active_pages:
            if self.eviction_policy is not None:
                page = self.eviction_policy.select_victim(self.active_pages, context={})
                self.active_pages.remove(page)
            else:
                self.active_pages.sort(key=lambda p: p.get('importance_score', 0.0))
                page = self.active_pages.pop(0)
            self._demote_to_next_tier(page, -1)
            
            # The most important active page that is preserved:
            if self.active_pages:
                preserved_page = self.active_pages[-1]
                att = preserved_page.get('attention_sum', 0.0)
                entropy = preserved_page.get('entropy', 0.0)
                rec = 1.0 / (1.0 + float(self.generation_step - preserved_page.get('last_step_accessed', 0)))
                
                if att >= rec and att >= entropy:
                    reason = "high attention recurrence"
                elif rec >= att and rec >= entropy:
                    reason = "high recency bias"
                else:
                    reason = "high outlier magnitude"
                    
                argus_log("INFO", f"Page {preserved_page.get('page_id')} preserved in FP16 | Reason: {reason} (importance={preserved_page.get('importance_score', 0.0):.2f})", line_no=734)

    def _demote_to_next_tier(self, page, current_tier_idx: int):
        """Genel demote: hangi tier olursa olsun, backend'in compress() metodunu çağır."""
        next_tier_idx = current_tier_idx + 1
        if next_tier_idx >= len(self.tier_specs):
            return  # Son katman, daha aşağı gidemez
            
        current_spec = self.tier_specs[current_tier_idx] if current_tier_idx >= 0 else None
        next_spec = self.tier_specs[next_tier_idx]
        
        # 1. Mevcut katmandaki veriyi decompress et (veya FP16 active page ise doğrudan al)
        has_outliers = 'key_out_indices' in page and page['key_out_indices'] is not None

        if current_spec is None:
            # Demoting from FP16 active page
            raw_k = page['key']
            raw_v = page['value']
            if next_spec.use_outlier_isolation:
                k_norm, k_out, k_mask = isolate_outliers(raw_k, self.threshold_sigma)
                v_norm, v_out, v_mask = isolate_outliers(raw_v, self.threshold_sigma)
                k_out_indices = torch.nonzero(k_mask).to(torch.int16)
                k_out_values = k_out[k_mask]
                v_out_indices = torch.nonzero(v_mask).to(torch.int16)
                v_out_values = v_out[v_mask]
            else:
                k_norm, v_norm = raw_k, raw_v
                k_out_indices, k_out_values = None, None
                v_out_indices, v_out_values = None, None
        else:
            # Demoting from a compressed tier
            decomp_args = {"seq_dim": -1}
            if current_spec.is_projection:
                recon_op = self.get_jl_reconstruction_operator(page['key_proj'].device, page['key_proj'].dtype)
                decomp_args['recon_operator'] = recon_op
            
            if has_outliers:
                # Outliers already isolated previously. Decompress the normalized part only,
                # and directly propagate the outliers without running isolate_outliers again.
                k_norm = current_spec.backend.decompress(page['key_compressed'], **decomp_args)
                v_norm = current_spec.backend.decompress(page['value_compressed'], **decomp_args)
                k_out_indices = page.get('key_out_indices')
                k_out_values = page.get('key_out_values')
                v_out_indices = page.get('value_out_indices')
                v_out_values = page.get('value_out_values')
            else:
                # No outliers isolated yet. Decompress the whole thing.
                raw_k = current_spec.backend.decompress(page['key_compressed'], **decomp_args)
                raw_v = current_spec.backend.decompress(page['value_compressed'], **decomp_args)
                
                # Check if we should isolate outliers now
                if next_spec.use_outlier_isolation:
                    k_norm, k_out, k_mask = isolate_outliers(raw_k, self.threshold_sigma)
                    v_norm, v_out, v_mask = isolate_outliers(raw_v, self.threshold_sigma)
                    k_out_indices = torch.nonzero(k_mask).to(torch.int16)
                    k_out_values = k_out[k_mask]
                    v_out_indices = torch.nonzero(v_mask).to(torch.int16)
                    v_out_values = v_out[v_mask]
                else:
                    k_norm, v_norm = raw_k, raw_v
                    k_out_indices, k_out_values = None, None
                    v_out_indices, v_out_values = None, None
                    
        # 3. Sonraki katmanın backend'i ile compress et
        comp_args = {"seq_dim": -1}
        if next_spec.is_projection:
            w_proj = self.get_jl_projection_matrix(k_norm.device, k_norm.dtype)
            comp_args['w_proj'] = w_proj
            
        key_comp = next_spec.backend.compress(k_norm, **comp_args)
        value_comp = next_spec.backend.compress(v_norm, **comp_args)

        # This legacy Python demotion path is not used by the native lifecycle,
        # so allocate its optional pool only if a caller explicitly invokes it.
        if next_spec.use_static_pool and not any(
            key.startswith(f"{next_spec.name}_") for key in self.pools_by_tier
        ):
            batch, num_heads, _, head_dim = k_norm.shape
            self._allocate_pool_for_tier(
                next_spec.name,
                next_spec.max_pages,
                k_norm.device,
                k_norm.dtype,
                batch,
                num_heads,
                head_dim,
            )
        
        # 4. Limit aşımı kontrolü (Eviction)
        pages_list = self.pages_by_tier[next_spec.name]
        if next_spec.max_pages > 0 and len(pages_list) >= next_spec.max_pages:
            if self.eviction_policy is not None:
                victim = self.eviction_policy.select_victim(pages_list, context={})
                pages_list.remove(victim)
            else:
                pages_list.sort(key=lambda p: p.get('importance_score', 0.0))
                victim = pages_list.pop(0)
            self._demote_to_next_tier(victim, next_tier_idx)
            
        # 5. Sıkıştırılmış veriyi havuz veya dinamik yapıya yerleştir
        use_pool = next_spec.use_static_pool and page.get('page_size', self.page_size) == self.page_size and any(f"{next_spec.name}_key_{k}" in self.pools_by_tier for k in key_comp.keys())
        if use_pool:
            idx = self._get_free_pool_idx(pages_list, next_spec.max_pages)
            key_comp_ref = {}
            value_comp_ref = {}
            for k in key_comp.keys():
                pool_tensor = self.pools_by_tier[f"{next_spec.name}_key_{k}"][idx]
                pool_tensor.copy_(key_comp[k])
                key_comp_ref[k] = pool_tensor
            for k in value_comp.keys():
                pool_tensor = self.pools_by_tier[f"{next_spec.name}_value_{k}"][idx]
                pool_tensor.copy_(value_comp[k])
                value_comp_ref[k] = pool_tensor
        else:
            idx = None
            key_comp_ref = key_comp
            value_comp_ref = value_comp
            
        # 6. Page modelini güncelle ve listeye ekle
        new_page = PageDict(page)
        new_page['key_compressed'] = key_comp_ref
        new_page['value_compressed'] = value_comp_ref
        new_page['key_out_indices'] = k_out_indices
        new_page['key_out_values'] = k_out_values
        new_page['value_out_indices'] = v_out_indices
        new_page['value_out_values'] = v_out_values
        if idx is not None:
            new_page['pool_idx'] = idx
        else:
            new_page.pop('pool_idx', None)
            
        new_page.pop('key', None)
        new_page.pop('value', None)
        
        pages_list.append(new_page)
        
        # 7. Telemetri
        tier_from = "active" if current_tier_idx == -1 else current_spec.name
        self.log_event("demote", page.get('page_id'), tier_from=tier_from, tier_to=next_spec.name)
        self.total_demotions += 1
        
        cascade_from = "fp16" if tier_from == "active" else tier_from
        cascade_key = f"{cascade_from}_to_{next_spec.name}"
        self.cascade_counts[cascade_key] = self.cascade_counts.get(cascade_key, 0) + 1
        
        if current_tier_idx == -1:
            pid = page.get('page_id')
            if pid in self.page_lifetimes:
                lifetime = self.generation_step - self.page_lifetimes[pid]
                self.total_page_lifetimes += lifetime
                self.completed_page_lifetimes_count += 1
                del self.page_lifetimes[pid]
                
        argus_log("INFO", f"Demoting Page {page.get('page_id')} ({tier_from.upper()} -> {next_spec.name.upper()}) | Reason: low attention prominence (importance={page.get('importance_score', 0.0):.2f})", line_no=800)
        self._invalidate_decompressed_cache()

    def _decompress_tier_pages_batched(self, spec, pages):
        """
        Decompresses all pages in a tier, one per page.

        Pages already in the prefetch cache are returned as-is. Everything
        else goes through the C++ manager's native dequant kernels/JL matmul
        (peek_decompress_page) rather than the pluggable Python
        QuantizationBackend — the Python backends pack bits along the
        sequence axis while these tiers are always compressed by C++ (which
        packs along head_dim), so decoding via the Python side silently
        produces wrong values for anything actually written by C++.
        """
        results = [None] * len(pages)

        for i, page in enumerate(pages):
            cached = self._cpp_manager.get_prefetched_tensors(page.get('page_id'))
            if cached is not None:
                k, v = cached
            else:
                self.prefetch_misses += 1
                k, v = self._cpp_manager.peek_decompress_page(page, spec.name)

            k = _apply_outlier_restoration(k, page, key='key')
            v = _apply_outlier_restoration(v, page, key='value')
            results[i] = (k, v)

        return results

    def get_all_keys_values(self):
        """
        Reconstructs all cache levels back to single FP16 tensors, prepending VIP anchors & Attention Sinks.
        Automatically handles host-to-device swapping if the cache is swapped out.
        """
        # Automatic Swap-In Safeguard
        if self.is_swapped_out:
            target_dev = self.active_pool_k.device if self.active_pool_k is not None else "cuda"
            self.swap_in_to_device(device=target_dev)
            
        # The HuggingFace Cache API requires one contiguous K/V pair.  Build
        # that pair directly and copy each decompressed page into its final
        # slice.  The old implementation first concatenated all cold pages,
        # retained that full FP16 tensor on every layer, and then concatenated
        # it again with the hot pages.  At long context this produced a
        # persistent exact-cache-sized mirror plus a second transient copy.
        tier_entries = [
            (spec, page)
            for spec in reversed(self.tier_specs)
            for page in self.pages_by_tier.get(spec.name, [])
        ]
        resident_pairs = []
        if self.sink_k is not None:
            resident_pairs.append((self.sink_k, self.sink_v))
        if self.anchor_k is not None:
            resident_pairs.append((self.anchor_k, self.anchor_v))
        resident_pairs.extend((page["key"], page["value"]) for page in self.active_pages)
        if self.k_buffer is not None and self.k_buffer.shape[-2] > 0:
            resident_pairs.append((self.k_buffer, self.v_buffer))

        if not tier_entries and not resident_pairs:
            return None, None

        target_dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        total_tokens = sum(int(k.shape[-2]) for k, _ in resident_pairs)
        total_tokens += sum(int(page.get("page_size", self.page_size)) for _, page in tier_entries)

        first_pair = resident_pairs[0] if resident_pairs else None
        first_tier = 0
        if first_pair is None:
            spec, page = tier_entries[0]
            first_pair = self._decompress_tier_pages_batched(spec, [page])[0]
            first_tier = 1

        first_k, first_v = first_pair
        output_shape = (*first_k.shape[:-2], total_tokens, first_k.shape[-1])
        out_k = torch.empty(output_shape, device=target_dev, dtype=torch.float16)
        out_v = torch.empty(output_shape, device=target_dev, dtype=torch.float16)
        offset = 0

        def append_pair(k, v):
            nonlocal offset
            length = int(k.shape[-2])
            out_k[..., offset : offset + length, :].copy_(
                k.to(device=target_dev, dtype=torch.float16)
            )
            out_v[..., offset : offset + length, :].copy_(
                v.to(device=target_dev, dtype=torch.float16)
            )
            offset += length

        # Preserve the historical logical order: sinks, anchors, coldest to
        # hottest compressed tiers, active pages, then the partial buffer.
        resident_prefix = int(self.sink_k is not None) + int(self.anchor_k is not None)
        for pair in resident_pairs[:resident_prefix]:
            append_pair(*pair)
        if first_tier:
            append_pair(first_k, first_v)
        for spec, page in tier_entries[first_tier:]:
            append_pair(*self._decompress_tier_pages_batched(spec, [page])[0])
        for pair in resident_pairs[resident_prefix:]:
            append_pair(*pair)

        assert offset == total_tokens
        # These remain None intentionally: compressed storage must never gain
        # a persistent FP16 mirror merely to accelerate the next decode step.
        self._decompressed_tiers_k = None
        self._decompressed_tiers_v = None
        self._tiers_version = -1
        return out_k, out_v

    def inplace_paged_attention(self, q: torch.Tensor, scale: float = None) -> torch.Tensor:
        """
        Computes scaled dot-product attention block-by-block/page-by-page.
        Delegates to the C++ ArgusCppManager which triggers the optimal FlashAttention/SDPA paths.
        """
        import math
        batch, num_heads, q_len, head_dim = q.shape
        orig_device = q.device
        
        if torch.cuda.is_available():
            if q.device.type == "cpu":
                q = q.cuda()

        device = q.device
        dtype = q.dtype
        
        if scale is None:
            scale = 1.0 / math.sqrt(head_dim)
            
        self.total_attention_calls += 1

        # Automatic Swap-In Safeguard
        if self.is_swapped_out:
            self.swap_in_to_device(device=device)

        # Get C++ signature values:
        # Default value for optional tensors: None or empty torch.Tensor()
        # In python, we pass None to pybind11 which converts it to undefined torch::Tensor()
        sink_k = self.sink_k if self.sink_k is not None else torch.Tensor()
        sink_v = self.sink_v if self.sink_v is not None else torch.Tensor()
        anchor_k = self.anchor_k if self.anchor_k is not None else torch.Tensor()
        anchor_v = self.anchor_v if self.anchor_v is not None else torch.Tensor()
        k_buffer = self.k_buffer if self.k_buffer is not None else torch.Tensor()
        v_buffer = self.v_buffer if self.v_buffer is not None else torch.Tensor()
        
        if torch.cuda.is_available():
            if sink_k.numel() > 0 and sink_k.device.type == "cpu":
                sink_k = sink_k.cuda()
            if sink_v.numel() > 0 and sink_v.device.type == "cpu":
                sink_v = sink_v.cuda()
            if anchor_k.numel() > 0 and anchor_k.device.type == "cpu":
                anchor_k = anchor_k.cuda()
            if anchor_v.numel() > 0 and anchor_v.device.type == "cpu":
                anchor_v = anchor_v.cuda()
            if k_buffer.numel() > 0 and k_buffer.device.type == "cpu":
                k_buffer = k_buffer.cuda()
            if v_buffer.numel() > 0 and v_buffer.device.type == "cpu":
                v_buffer = v_buffer.cuda()

        resurrection_threshold = getattr(self.config, 'resurrection_threshold', 0.15)

        # Delegate entirely to C++ manager. Page/QoS bookkeeping and the
        # following granularity pass form one state transition, so serialize
        # them for callers sharing a cache across threads.
        with self._attention_lock:
            attn_output = self._cpp_manager.inplace_paged_attention(
                q,
                scale,
                sink_k,
                sink_v,
                anchor_k,
                anchor_v,
                k_buffer,
                v_buffer,
                resurrection_threshold
            )

            self.manage_variable_granularity()

        if orig_device.type == "cpu":
            attn_output = attn_output.cpu()

        return attn_output

    def _write_compressed_field(self, page, prefix, comp):
        return self._granularity._write_compressed_field(page, prefix, comp)

    def split_page(self, page, tier_name=None):
        """Split a mega-page into micro-pages. See GranularityManager."""
        return self._granularity.split(page, tier_name)

    def merge_pages(self, pages, tier_name=None):
        """Merge micro-pages back into a mega-page. See GranularityManager."""
        return self._granularity.merge(pages, tier_name)

    def manage_variable_granularity(self):
        """Split cold pages and merge hot runs. See GranularityManager."""
        return self._granularity.rebalance()

    def speculate_and_prefetch(self, attn_weights=None):
        """
        Predicts which pages will be heavily attended to next and asks the
        C++ manager to pre-dequantize them on its background prefetch stream,
        so a later resurrect_page()/inplace_paged_attention() call can serve
        them straight from prefetch_cache_ instead of paying dequant latency.

        Page *selection* stays here (it needs attn_weights / tier_specs
        heuristics); the actual dequantization is delegated to C++ so the
        result lands in the same prefetch_cache_ the live hot path reads —
        unlike the old Python-side simulation, which decompressed pages into
        a separate dict that inplace_paged_attention never looked at.
        """
        if self.is_swapped_out:
            return
            
        pages_to_prefetch = []
        
        if attn_weights is not None:
            try:
                # attn_weights shape: (batch, num_heads, q_len, kv_len)
                weights = attn_weights.mean(dim=(0, 1))[-1] # Average heads, last query token
                
                # Sinks and anchors offsets
                sink_len = self.sink_k.shape[-2] if self.sink_k is not None else 0
                anchor_len = self.anchor_k.shape[-2] if self.anchor_k is not None else 0
                offset = sink_len + anchor_len
                
                page_weights = []
                all_p = []
                for spec in reversed(self.tier_specs):
                    all_p.extend((spec.name, p) for p in self.pages_by_tier.get(spec.name, []))
                
                current_offset = offset
                sums = []
                for idx, (tier, page) in enumerate(all_p):
                    p_size = page.get('page_size', self.page_size)
                    start = current_offset
                    end = start + p_size
                    if end <= len(weights):
                        sums.append(weights[start:end].mean())
                    else:
                        sums.append(None)
                    current_offset += p_size
                
                # Stack and transfer in ONE batch to avoid CPU-GPU synchronization loop
                sums_tensor_list = [s for s in sums if s is not None]
                if sums_tensor_list:
                    stacked_sums = torch.stack(sums_tensor_list).cpu()
                    sums_val = stacked_sums.tolist()
                else:
                    sums_val = []
                
                sums_idx = 0
                for idx, (tier, page) in enumerate(all_p):
                    if sums[idx] is not None:
                        w = sums_val[sums_idx]
                        sums_idx += 1
                    else:
                        w = 0.0
                    page_weights.append((w, page, tier))
                    
                page_weights.sort(key=lambda x: x[0], reverse=True)
                for w, page, tier in page_weights[:2]:
                    pages_to_prefetch.append((page, tier))
            except Exception:
                pass
                
        if not pages_to_prefetch:
            for spec in self.tier_specs:
                pages = self.pages_by_tier.get(spec.name, [])
                if pages:
                    pages_to_prefetch.append((pages[-1], spec.name))
                    if len(pages_to_prefetch) >= 2:
                        break

        page_ids = [page.get('page_id') for page, _tier in pages_to_prefetch]
        if page_ids:
            self._cpp_manager.speculate_and_prefetch(page_ids)
            self._cpp_manager.wait_for_prefetch_idle()

    def swap_out_to_host(self):
        """Spill the whole cache to pinned host memory. See HostSpillManager."""
        return self._host_spill.spill_out()

    def swap_in_to_device(self, device="cuda"):
        """Restore a spilled cache to ``device``. See HostSpillManager."""
        return self._host_spill.spill_in(device)

    def get_allocator_fragmentation_report(self) -> dict:
        return self._telemetry.allocator_fragmentation()

    def get_vram_usage(self):
        """Total bytes resident on the active device across all tiers."""
        return self._telemetry.vram_usage()
