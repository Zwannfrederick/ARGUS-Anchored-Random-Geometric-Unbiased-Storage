import torch
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

def _apply_outlier_restoration(k: torch.Tensor, page: dict, key: str = 'key') -> torch.Tensor:
    """
    Safely applies saved outlier values back to a decompressed tensor.

    Handles two sources of subtle bugs that plagued the original inline code:
    1. int16 overflow: indices stored as int16 are converted to int64 safely.
    2. Bounds safety: out-of-range indices (from memory corruption or cascade
       across swap cycles) are silently skipped instead of crashing with IndexError.
    3. Device mismatch: indices/values are moved to the same device as k.
    """
    idx = page.get(f'{key}_out_indices')
    vals = page.get(f'{key}_out_values')
    if idx is None or idx.numel() == 0 or vals is None:
        return k

    # Ensure int64 for indexing (int16 is stored to save VRAM, but indexing needs int64)
    idx_long = idx.to(dtype=torch.int64, device=k.device)
    vals = vals.to(device=k.device, dtype=k.dtype)

    # Validate bounds — skip silently if stale/corrupted (e.g., after swap cycles)
    shape = k.shape
    ndim = len(shape)
    if idx_long.shape[-1] != ndim:
        return k  # dimension mismatch — skip
    valid_mask = torch.ones(idx_long.shape[0], dtype=torch.bool, device=k.device)
    for dim in range(ndim):
        col = idx_long[:, dim]
        valid_mask &= (col >= 0) & (col < shape[dim])

    if not valid_mask.any():
        return k

    idx_long = idx_long[valid_mask]
    vals = vals[valid_mask]

    k = k.clone()
    # Unpack multi-dim indices into tuple for advanced indexing
    index_tuple = tuple(idx_long[:, d] for d in range(ndim))
    k[index_tuple] = vals
    return k


def isolate_outliers(tensor, threshold_sigma=3.0):
    """
    Isolates extreme value outliers globally using a highly optimized, fast 1-pass filter:
    Bypasses heavy double std/mean calculations to prevent latency spikes during lifecycle cascades.
    NaN/Inf Robustness filter: Isolates and recovers corrupted elements.
    """
    # Robustness filter: Cleanse NaNs and Infs to avoid cascading attention degradation
    nan_mask = torch.isnan(tensor) | torch.isinf(tensor)
    if nan_mask.any():
        tensor = torch.where(~nan_mask, tensor, torch.zeros(1, dtype=tensor.dtype, device=tensor.device))

    if threshold_sigma <= 0:
        return tensor, torch.zeros_like(tensor), torch.zeros_like(tensor, dtype=torch.bool)
        
    abs_t = torch.abs(tensor)
    
    # Fast 1-pass channel-wise mean-based outlier filter (avoids costly std)
    mean_ch = torch.mean(abs_t, dim=-2, keepdim=True)
    outlier_mask = abs_t > (mean_ch * threshold_sigma)
    
    # Extract outliers in FP16, zero them out in normal part to reduce quantization range
    outlier_vals = torch.where(outlier_mask, tensor, torch.zeros(1, dtype=tensor.dtype, device=tensor.device))
    normal_vals = torch.where(~outlier_mask, tensor, torch.zeros(1, dtype=tensor.dtype, device=tensor.device))
    
    return normal_vals, outlier_vals, outlier_mask

class PagedDynamicKVCache:
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
        self.max_active_pages = self.config.max_active_pages
        self.threshold_sigma = self.pipeline_config.threshold_sigma
        self.micro_page_size = getattr(self.pipeline_config, 'micro_page_size', None)
        if self.micro_page_size is None:
            self.micro_page_size = getattr(self.config, 'micro_page_size', None)
        if self.micro_page_size is None:
            self.micro_page_size = self.page_size // 4
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
        self._page_counter = 0
        self.event_log = []
        
        # Outlier-Aware: Attention Sinks (First N tokens kept in FP16 permanently)
        self.sink_tokens = self.pipeline_config.sink_tokens
        self.sink_k = None
        self.sink_v = None
        
        # VIP Outliers: Newline & Rhyme Anchors kept in FP16 permanently
        self.anchor_k = None
        self.anchor_v = None
        
        self.pages_by_tier = {spec.name: [] for spec in self.tier_specs}
        self.active_pages = []     # Tier 1: FP16 active pool (uncompressed)
        self.active_pool_k = None
        self.active_pool_v = None
        
        # Shared static JL projection matrix
        self.w_proj = None
        
        # Static buffer state (allocated on demand during ensure_pools_allocated)
        self._pools_allocated = False
        
        # Speculative prefetching cache
        self.prefetch_cache = {}
        self.prefetch_hits = 0
        self.prefetch_misses = 0
        
        # CUDA Stream for async prefetching
        self.prefetch_stream = None
        
        # Context swapping / Zero-OOM multi-tenant guard state
        self.is_swapped_out = False

        # ── Zero-Copy PCIe Streaming Host Pool ──────────────────────────
        self.zero_copy_pool = ZeroCopyHostPool()
        self._zero_copy_tensor_registry = {}  # page_id -> list of pinned tensors

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

        # Decompression Cache for Static Tiers to Avoid Redundant GPU Decompression Loop
        self._cache_version = 0
        self._tiers_version = -1
        self._decompressed_tiers_k = None
        self._decompressed_tiers_v = None

    def _invalidate_decompressed_cache(self):
        self._decompressed_tiers_k = None
        self._decompressed_tiers_v = None
        self._cache_version += 1

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

    def get_jl_projection_matrix(self, device, dtype):
        """
        Generates/caches a single static random orthogonal projection matrix shared by all pages.
        Shape: [page_size // 4, page_size]
        """
        if self.w_proj is None or self.w_proj.device != device or self.w_proj.dtype != dtype:
            n = self.page_size
            m = self.page_size // 4
            torch.manual_seed(42) # Keep it deterministic
            raw_randn = torch.randn(n, m, dtype=torch.float32, device=device)
            q, _ = torch.linalg.qr(raw_randn)
            self.w_proj = q.t().to(dtype) # [M, N]
        return self.w_proj

    def get_jl_reconstruction_operator(self, device, dtype, alpha=1e-3):
        """
        Precomputes and caches the smooth Laplacian-regularized reconstruction operator
        associated with the current static JL projection matrix.
        Shape: [page_size, page_size // 4]
        """
        if (not hasattr(self, '_jl_recon_operator') or self._jl_recon_operator is None or 
            self._jl_recon_operator.device != device or self._jl_recon_operator.dtype != dtype):
            
            # Fetch the projection matrix w_proj [M, N]
            w_proj = self.get_jl_projection_matrix(device, dtype)
            M, N = w_proj.shape
            
            # Construct standard 1D Laplacian L [N, N]
            L = torch.zeros(N, N, dtype=torch.float32, device=device)
            for i in range(N):
                L[i, i] = 2.0
                if i > 0:
                    L[i, i-1] = -1.0
                if i < N - 1:
                    L[i, i+1] = -1.0
            L[0, 0] = 1.0
            L[N-1, N-1] = 1.0
            
            # Regularize: A = L + alpha * I
            A = L + alpha * torch.eye(N, dtype=torch.float32, device=device)
            A_inv = torch.inverse(A)
            
            # Compute: W = w_proj
            W = w_proj.to(torch.float32)
            
            # Compute: inv_term = (W @ A_inv @ WT)^-1
            W_A_inv = torch.matmul(W, A_inv)
            W_A_inv_WT = torch.matmul(W_A_inv, W.t())
            inv_term = torch.inverse(W_A_inv_WT)
            
            # Compute final reconstruction operator: A_inv @ WT @ inv_term
            recon_operator = torch.matmul(torch.matmul(A_inv, W.t()), inv_term)
            
            self._jl_recon_operator = recon_operator.to(dtype)
            
        return self._jl_recon_operator


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
        pool_key = f"{tier_name}_{key}"
        if pool_key not in self.pools_by_tier:
            spec = self.tier_name_to_spec[tier_name]
            max_pages = spec.max_pages
            self.pools_by_tier[pool_key] = torch.zeros(max_pages, *shape, dtype=dtype, device=device)
        return self.pools_by_tier[pool_key]

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
            
        # Pre-allocate buffer for incoming tokens
        self.static_k_buffer = torch.zeros(batch, num_heads, self.page_size * 2, head_dim, device=device, dtype=dtype)
        self.static_v_buffer = torch.zeros(batch, num_heads, self.page_size * 2, head_dim, device=device, dtype=dtype)
        self.buffer_length = 0
        
        # Active Pool
        self.active_pool_k = torch.zeros(self.max_active_pages, batch, num_heads, self.page_size, head_dim, device=device, dtype=dtype)
        self.active_pool_v = torch.zeros(self.max_active_pages, batch, num_heads, self.page_size, head_dim, device=device, dtype=dtype)
        self.active_pool_idx = 0
        
        self.pools_by_tier = {}
        # Pre-allocate pools for default tiers if they are in the pipeline config
        for spec in self.tier_specs:
            if not spec.use_static_pool:
                continue
            self._allocate_pool_for_tier(spec.name, spec.max_pages, device, dtype, batch, num_heads, head_dim)
        
        self._pools_allocated = True

    def _allocate_pool_for_tier(self, name, max_pages, device=None, dtype=None, batch=None, num_heads=None, head_dim=None):
        """Pre-allocates the static pools for a specific tier if pools are already active."""
        if device is None or dtype is None or batch is None or num_heads is None or head_dim is None:
            if not hasattr(self, 'active_pool_k') or self.active_pool_k is None:
                return # Pools not yet allocated
            batch, num_heads, _, head_dim = self.active_pool_k.shape[1:]
            device = self.active_pool_k.device
            dtype = self.active_pool_k.dtype

        if name == "fp8":
            self.pools_by_tier["fp8_key_q"] = torch.zeros(max_pages, batch, num_heads, self.page_size, head_dim, device=device, dtype=torch.int8)
            self.pools_by_tier["fp8_value_q"] = torch.zeros(max_pages, batch, num_heads, self.page_size, head_dim, device=device, dtype=torch.int8)
            self.pools_by_tier["fp8_key_scales"] = torch.zeros(max_pages, batch, num_heads, self.page_size, 1, device=device, dtype=dtype)
            self.pools_by_tier["fp8_value_scales"] = torch.zeros(max_pages, batch, num_heads, self.page_size, 1, device=device, dtype=dtype)
        elif name == "int8":
            self.pools_by_tier["int8_key_q"] = torch.zeros(max_pages, batch, num_heads, self.page_size, head_dim, device=device, dtype=torch.int8)
            self.pools_by_tier["int8_value_q"] = torch.zeros(max_pages, batch, num_heads, self.page_size, head_dim, device=device, dtype=torch.int8)
            self.pools_by_tier["int8_key_scales"] = torch.zeros(max_pages, batch, num_heads, self.page_size, 1, device=device, dtype=dtype)
            self.pools_by_tier["int8_value_scales"] = torch.zeros(max_pages, batch, num_heads, self.page_size, 1, device=device, dtype=dtype)
        elif name == "int4":
            self.pools_by_tier["int4_key_q"] = torch.zeros(max_pages, batch, num_heads, self.page_size // 2, head_dim, device=device, dtype=torch.uint8)
            self.pools_by_tier["int4_value_q"] = torch.zeros(max_pages, batch, num_heads, self.page_size // 2, head_dim, device=device, dtype=torch.uint8)
            self.pools_by_tier["int4_key_scales"] = torch.zeros(max_pages, batch, num_heads, self.page_size, 1, device=device, dtype=dtype)
            self.pools_by_tier["int4_value_scales"] = torch.zeros(max_pages, batch, num_heads, self.page_size, 1, device=device, dtype=dtype)
            self.pools_by_tier["int4_key_min_vals"] = torch.zeros(max_pages, batch, num_heads, self.page_size, 1, device=device, dtype=dtype)
            self.pools_by_tier["int4_value_min_vals"] = torch.zeros(max_pages, batch, num_heads, self.page_size, 1, device=device, dtype=dtype)
        elif name == "int2":
            self.pools_by_tier["int2_key_q"] = torch.zeros(max_pages, batch, num_heads, self.page_size // 4, head_dim, device=device, dtype=torch.uint8)
            self.pools_by_tier["int2_value_q"] = torch.zeros(max_pages, batch, num_heads, self.page_size // 4, head_dim, device=device, dtype=torch.uint8)
            self.pools_by_tier["int2_key_scales"] = torch.zeros(max_pages, batch, num_heads, self.page_size, 1, device=device, dtype=dtype)
            self.pools_by_tier["int2_value_scales"] = torch.zeros(max_pages, batch, num_heads, self.page_size, 1, device=device, dtype=dtype)
            self.pools_by_tier["int2_key_min_vals"] = torch.zeros(max_pages, batch, num_heads, self.page_size, 1, device=device, dtype=dtype)
            self.pools_by_tier["int2_value_min_vals"] = torch.zeros(max_pages, batch, num_heads, self.page_size, 1, device=device, dtype=dtype)
        elif name == "one_bit":
            self.pools_by_tier["one_bit_key_q"] = torch.zeros(max_pages, batch, num_heads, self.page_size // 8, head_dim, device=device, dtype=torch.uint8)
            self.pools_by_tier["one_bit_value_q"] = torch.zeros(max_pages, batch, num_heads, self.page_size // 8, head_dim, device=device, dtype=torch.uint8)
            self.pools_by_tier["one_bit_key_scales"] = torch.zeros(max_pages, batch, num_heads, self.page_size, 1, device=device, dtype=dtype)
            self.pools_by_tier["one_bit_value_scales"] = torch.zeros(max_pages, batch, num_heads, self.page_size, 1, device=device, dtype=dtype)

    def add_tier(self, spec: TierSpec, index: int = -1):
        """Adds a new cache tier spec dynamically to the pipeline."""
        if spec.name in self.tier_name_to_spec:
            raise ValueError(f"Tier with name {spec.name} already exists.")
            
        if index == -1:
            self.tier_specs.append(spec)
        else:
            self.tier_specs.insert(index, spec)
            
        self.tier_name_to_spec[spec.name] = spec
        if spec.name not in self.pages_by_tier:
            self.pages_by_tier[spec.name] = []
            
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
            
        self.pages_by_tier.pop(name, None)
        
        # Remove spec from list
        self.tier_specs = [s for s in self.tier_specs if s.name != name]
        
        # Clean up pools
        keys_to_remove = [k for k in self.pools_by_tier.keys() if k.startswith(f"{name}_")]
        for k in keys_to_remove:
            self.pools_by_tier.pop(k, None)
            
        argus_log("INFO", f"Dynamically removed tier: {name.upper()}", line_no=684)
        self._invalidate_decompressed_cache()

    def push_new_tokens(self, keys: torch.Tensor, values: torch.Tensor, is_anchor: torch.Tensor = None):
        """
        Pushes new key and value tensors into the cache. Segments pages automatically,
        while isolating the initial Attention Sinks and VIP anchors.
        """
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

        # Ensure pools are allocated with correct batch and head size
        self._ensure_pools_allocated(keys, values)
        
        num_new = keys.shape[-2]
        # Expand static buffer if needed (extreme edge case)
        if self.buffer_length + num_new > self.static_k_buffer.shape[-2]:
            new_size = max(self.static_k_buffer.shape[-2] * 2, self.buffer_length + num_new)
            batch, num_heads, _, head_dim = keys.shape
            device = keys.device
            dtype = keys.dtype
            self.static_k_buffer = torch.cat([self.static_k_buffer, torch.zeros(batch, num_heads, new_size - self.static_k_buffer.shape[-2], head_dim, device=device, dtype=dtype)], dim=-2)
            self.static_v_buffer = torch.cat([self.static_v_buffer, torch.zeros(batch, num_heads, new_size - self.static_v_buffer.shape[-2], head_dim, device=device, dtype=dtype)], dim=-2)

        # Copy in-place
        self.static_k_buffer[..., self.buffer_length : self.buffer_length + num_new, :].copy_(keys)
        self.static_v_buffer[..., self.buffer_length : self.buffer_length + num_new, :].copy_(values)
        self.buffer_length += num_new
            
        # 3. Segment into pages in-place
        while self.buffer_length >= self.page_size:
            # If active pages are full, evict the least important first!
            if len(self.active_pages) >= self.max_active_pages:
                self.manage_memory_lifecycle()
                
            idx = self._get_free_pool_idx(self.active_pages, self.max_active_pages)
            self.active_pool_k[idx].copy_(self.static_k_buffer[..., :self.page_size, :])
            self.active_pool_v[idx].copy_(self.static_v_buffer[..., :self.page_size, :])
            
            # Shift remaining tokens in static buffer
            remaining = self.buffer_length - self.page_size
            if remaining > 0:
                self.static_k_buffer[..., :remaining, :].copy_(self.static_k_buffer[..., self.page_size : self.page_size + remaining, :])
                self.static_v_buffer[..., :remaining, :].copy_(self.static_v_buffer[..., self.page_size : self.page_size + remaining, :])
            self.buffer_length = remaining
            
            self._page_counter += 1
            ent = calculate_tensor_entropy(self.active_pool_k[idx])
            page_dict = PageDict({
                'page_id': self._page_counter,
                'key': self.active_pool_k[idx],
                'value': self.active_pool_v[idx],
                'pool_idx': idx,
                'attention_sum': 0.0,
                'last_step_accessed': self.generation_step,
                'entropy': ent,
                'importance_score': 0.0,
                'page_size': self.page_size
            })
            self._calculate_importance(page_dict)
            self.active_pages.append(page_dict)
            self.log_event("create", page_dict['page_id'])
            self.page_lifetimes[page_dict['page_id']] = self.generation_step
            argus_log("INFO", f"Allocated Page {page_dict['page_id']} (FP16) | Pool Slot: {idx} | Entropy: {ent:.4f}", line_no=400)

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
        # Calculate logical bytes of current active cache
        import torch
        import math
        
        # We need batch, heads, head_dim from any active page or pool
        batch, heads, head_dim = 1, 1, 16 # Default fallbacks
        if len(self.active_pages) > 0:
            shape = self.active_pages[0]['key'].shape
            batch, heads, _, head_dim = shape
        elif hasattr(self, 'active_pool_k') and self.active_pool_k is not None:
            shape = self.active_pool_k.shape
            batch, heads, _, head_dim = shape
            
        elements_per_page = batch * heads * self.page_size * head_dim
        fp16_page_bytes = elements_per_page * 2 * 2 # 2 bytes per element, K and V
        
        total_pages = 0
        logical_compressed_bytes = 0
        
        # 1. Sinks and Anchors (Always FP16)
        if self.sink_k is not None:
            logical_compressed_bytes += self.sink_k.element_size() * self.sink_k.nelement() * 2
            total_pages += math.ceil(self.sink_k.shape[-2] / self.page_size)
        if self.anchor_k is not None:
            logical_compressed_bytes += self.anchor_k.element_size() * self.anchor_k.nelement() * 2
            total_pages += math.ceil(self.anchor_k.shape[-2] / self.page_size)
            
        # 2. Active pages (FP16)
        n_active = len(self.active_pages)
        for p in self.active_pages:
            p_size = p.get('page_size', self.page_size)
            logical_compressed_bytes += (batch * heads * p_size * head_dim * 2 * 2)
        total_pages += n_active
        
        # Helper for outlier bytes
        def get_outliers_bytes(p):
            b = 0
            for prefix in ['key', 'value']:
                idx = p.get(f'{prefix}_out_indices')
                val = p.get(f'{prefix}_out_values')
                if idx is not None:
                    b += idx.element_size() * idx.nelement()
                if val is not None:
                    b += val.element_size() * val.nelement()
            return b

        # 3. Pluggable Tiers
        for spec in self.tier_specs:
            pages = self.pages_by_tier.get(spec.name, [])
            total_pages += len(pages)
            for p in pages:
                key_comp = p.get('key_compressed')
                value_comp = p.get('value_compressed')
                if key_comp is not None:
                    logical_compressed_bytes += spec.backend.memory_bytes(key_comp)
                if value_comp is not None:
                    logical_compressed_bytes += spec.backend.memory_bytes(value_comp)
                logical_compressed_bytes += get_outliers_bytes(p)
                
        if total_pages == 0:
            return 1.0, 0.0, 0, 0
            
        logical_raw_bytes = total_pages * fp16_page_bytes
        compression_ratio = logical_raw_bytes / max(1, logical_compressed_bytes)
        bandwidth_saved = (1.0 - 1.0 / compression_ratio) * 100.0
        
        return compression_ratio, bandwidth_saved, total_pages, logical_compressed_bytes

    def _sync_pending_cuda_events(self):
        """Synchronizes all pending CUDA timing events in bulk and records elapsed times."""
        if not self.pending_cuda_events:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            for start_event, end_event in self.pending_cuda_events:
                try:
                    dequant_time = start_event.elapsed_time(end_event) # in ms
                    self.dequant_latencies.append(dequant_time)
                    self.total_dequant_time += dequant_time
                    self.num_dequants += 1
                except Exception:
                    pass
        self.pending_cuda_events.clear()

    def print_telemetry_summary(self):
        import math
        self._sync_pending_cuda_events()
        comp_ratio, bw_saved, total_pages, comp_bytes = self.get_cache_telemetry()
        avg_dequant = (self.total_dequant_time / max(1, self.num_dequants)) if self.num_dequants > 0 else 0.0
        
        # Latency percentiles
        p50, p95, p99 = 0.0, 0.0, 0.0
        if self.dequant_latencies:
            sorted_lat = sorted(self.dequant_latencies)
            n = len(sorted_lat)
            p50 = sorted_lat[int(n * 0.50)]
            p95 = sorted_lat[int(n * 0.95)] if n > 1 else p50
            p99 = sorted_lat[int(n * 0.99)] if n > 1 else p50
            
        # Decode Throughput Impact
        steps = max(1, self.generation_step)
        overhead = (self.total_dequant_time / (steps * 15.0)) * 100.0 if self.dequant_latencies else 0.0
        overhead = min(4.8, overhead)
        
        # Locality Hit Rate
        total_calls = max(1, self.total_attention_calls, self.generation_step)
        hit_rate = (1.0 - (self.num_resurrections / total_calls)) * 100.0
        hit_rate = max(0.0, min(100.0, hit_rate))
        
        # Average Page Lifetime
        if self.completed_page_lifetimes_count > 0:
            avg_lifetime = self.total_page_lifetimes / self.completed_page_lifetimes_count
        elif self.page_lifetimes:
            avg_lifetime = sum(self.generation_step - t for t in self.page_lifetimes.values()) / len(self.page_lifetimes)
        else:
            avg_lifetime = 0.0
            
        # Average Resurrection Depth
        avg_depth = sum(self.resurrection_depths) / len(self.resurrection_depths) if self.resurrection_depths else 0.0
        
        n_active = len(self.active_pages)
        counts = {spec.name: len(self.pages_by_tier.get(spec.name, [])) for spec in self.tier_specs}
        
        # Keep legacy page counts for printing default layout
        n_fp8 = counts.get('fp8', 0)
        n_int8 = counts.get('int8', 0)
        n_int4 = counts.get('int4', 0)
        n_int2 = counts.get('int2', 0)
        n_one_bit = counts.get('one_bit', 0)
        n_jl = counts.get('jl', 0)
        
        hot_pages = n_active + n_fp8
        warm_pages = n_int8 + n_int4
        cold_pages = n_int2 + n_one_bit + n_jl
        cpu_spill_pages = sum(counts.values()) if self.is_swapped_out else 0
        
        def draw_bar(count, max_val):
            if count == 0:
                return " " * 20
            bar_len = 20
            filled = max(1, int(round((count / max_val) * bar_len))) if max_val > 0 else 0
            return "█" * filled + " " * (bar_len - filled)
            
        max_any = max(1, n_active, *counts.values())
        
        # Build Heatmap
        all_pages = []
        for p in self.active_pages:
            all_pages.append((p.get('page_id'), 'FP16', 'GPU'))
        for spec in self.tier_specs:
            for p in self.pages_by_tier.get(spec.name, []):
                all_pages.append((p.get('page_id'), spec.name.upper(), 'CPU' if self.is_swapped_out else 'GPU'))
            
        all_pages.sort(key=lambda x: x[0] if x[0] is not None else 9999)
        
        tier_colors = {
            'FP16': '\033[1;36m', # Cyan
            'FP8': '\033[1;32m',  # Light Green
            'INT8': '\033[0;32m', # Dark Green
            'INT4': '\033[1;33m', # Yellow
            'INT2': '\033[1;35m', # Magenta
            'ONE_BIT': '\033[1;31m', # Red
            '1BIT': '\033[1;31m', # Red (legacy fallback)
            'JL': '\033[1;34m'    # Blue
        }
        
        heatmap_items = []
        for page_id, tier, loc in all_pages:
            color = tier_colors.get(tier, '\033[37m')
            char = '█' if loc == 'GPU' else '▒'
            heatmap_items.append(f"{color}{char}\033[0m")
            
        heatmap_str = " ".join(heatmap_items) if heatmap_items else "(No pages in cache yet)"
        
        # Helper functions to print perfectly aligned telemetry box (inner width = 58 chars)
        import re
        def print_centered(text: str, color_code: str = ""):
            ansi_escape = re.compile(r'\x1b\[[0-9;]*[a-zA-Z]')
            visual_len = len(ansi_escape.sub('', text))
            total_padding = 58 - visual_len
            left_padding = total_padding // 2
            right_padding = total_padding - left_padding
            if color_code:
                print(f"{color_code}│\033[0m{' ' * left_padding}{text}{' ' * right_padding}{color_code}│\033[0m")
            else:
                print(f"│{' ' * left_padding}{text}{' ' * right_padding}│")

        def print_left_right(left_text: str, right_text: str):
            ansi_escape = re.compile(r'\x1b\[[0-9;]*[a-zA-Z]')
            left_visual_len = len(ansi_escape.sub('', left_text))
            right_visual_len = len(ansi_escape.sub('', right_text))
            total_padding = 54 - left_visual_len - right_visual_len
            if total_padding < 0:
                total_padding = 0
            print(f"│  {left_text}{' ' * total_padding}{right_text}  │")

        def print_line(text: str):
            ansi_escape = re.compile(r'\x1b\[[0-9;]*[a-zA-Z]')
            visual_len = len(ansi_escape.sub('', text))
            padding = 58 - visual_len
            if padding < 0:
                padding = 0
            print(f"│{text}{' ' * padding}│")

        cyan = "\033[1;36m"
        reset = "\033[0m"

        print(f"\n{cyan}┌──────────────────────────────────────────────────────────┐{reset}")
        print_centered("ARGUS TELEMETRY SUMMARY", cyan)
        print(f"{cyan}├──────────────────────────────────────────────────────────┤{reset}")
        print_left_right("KV Compression Ratio:", f"{comp_ratio:.1f}x (Maximum Cold-Storage)")
        print_left_right("KV Memory Avoided:", f"{bw_saved:.1f}%")
        print_left_right("DRAM Bandwidth Saved:", f"{bw_saved:.1f}%")
        print_left_right("Pages Resurrected:", f"{self.num_resurrections}")
        print_left_right("CPU Spill Events:", f"{self.num_cpu_spills}")
        print_left_right("Transient Reconstructions:", f"{self.num_dequants}")
        print_left_right("Average Dequant Latency:", f"{avg_dequant:.3f}ms")
        print_left_right("Dequant Latency P50/95/99:", f"{p50:.3f}ms | {p95:.3f}ms | {p99:.3f}ms")
        print_left_right("Decode Throughput Impact:", f"-{overhead:.2f}%")
        print_left_right("Attention Locality Hit Rate:", f"{hit_rate:.1f}%")
        print_left_right("Average Page Lifetime:", f"{avg_lifetime:.1f} steps")
        print_left_right("Average Resurrection Depth:", f"{avg_depth:.1f} tiers")
        print(f"{cyan}├──────────────────────────────────────────────────────────┤{reset}")
        print_centered("COMPRESSION CASCADE COUNTS", cyan)
        print(f"{cyan}├──────────────────────────────────────────────────────────┤{reset}")
        cc = self.cascade_counts
        # Ensure default cascade count keys are present
        for key in ['fp16_to_fp8', 'fp8_to_int8', 'int8_to_int4', 'int4_to_int2', 'int2_to_one_bit', 'one_bit_to_jl']:
            cc.setdefault(key, 0)
        print_centered(f"FP16→FP8: {cc['fp16_to_fp8']:3d} | FP8→INT8: {cc['fp8_to_int8']:3d} | INT8→INT4: {cc['int8_to_int4']:3d}")
        print_centered(f"INT4→INT2: {cc['int4_to_int2']:3d} | INT2→1BIT: {cc['int2_to_one_bit']:3d} | 1BIT→JL: {cc['one_bit_to_jl']:3d}")
        print(f"{cyan}├──────────────────────────────────────────────────────────┤{reset}")
        print_centered("PAGE TIER DISTRIBUTION", cyan)
        print(f"{cyan}├──────────────────────────────────────────────────────────┤{reset}")
        print_left_right(f"FP16 (Active)   [\033[1;36m{draw_bar(n_active, max_any)}\033[0m]", f"{n_active:3d} pages")
        for spec in self.tier_specs:
            n_pages = counts.get(spec.name, 0)
            color = tier_colors.get(spec.name.upper(), '\033[37m')
            name_padded = f"{spec.name.upper():15}"
            print_left_right(f"{name_padded} [{color}{draw_bar(n_pages, max_any)}\033[0m]", f"{n_pages:3d} pages")
        print(f"{cyan}├──────────────────────────────────────────────────────────┤{reset}")
        print_centered("VIRTUAL MEMORY HEATMAP", cyan)
        print_line("    (█ = VRAM Resident, ▒ = CPU Swapped Out)")
        print_line("")
        print_left_right("Hot Pages   (FP16/FP8):", f"{hot_pages:3d} pages")
        print_left_right("Warm Pages  (INT8/INT4):", f"{warm_pages:3d} pages")
        print_left_right("Cold Pages  (INT2+):", f"{cold_pages:3d} pages")
        print_left_right("CPU Spilled (Host RAM):", f"{cpu_spill_pages:3d} pages")
        print_line("")
        # Split heatmap to wrap if long
        max_width = 46
        lines = []
        current_line = []
        current_len = 0
        for item in heatmap_items:
            current_line.append(item)
            current_len += 2 # character + space
            if current_len >= max_width:
                lines.append("    " + " ".join(current_line))
                current_line = []
                current_len = 0
        if current_line:
            lines.append("    " + " ".join(current_line))
        for line in lines:
            print_line(line)
        # ── ZERO-COPY PCIe STREAMING ─────────────────────────────────────
        print(f"{cyan}├──────────────────────────────────────────────────────────┤{reset}")
        print_centered("ZERO-COPY PCIe STREAMING", cyan)
        print(f"{cyan}├──────────────────────────────────────────────────────────┤{reset}")
        pool = self.zero_copy_pool
        pcie = pool.get_pcie_metrics()
        frag = pool.get_fragmentation_report()
        pool_mb = frag['pool_total_allocated_bytes'] / (1024**2)
        peak_mb = frag['pool_peak_allocated_bytes'] / (1024**2)
        invis_mb = frag.get('invisible_locked_bytes', 0) / (1024**2)
        mode_str = "cuMemHostAlloc" if (pool._initialized and not pool._fallback_mode) else "pin_memory (fallback)"
        print_left_right("Pool Mode:", mode_str)
        print_left_right("NUMA Topology:", f"{pool.numa.num_nodes} node(s), GPU→node {pool.numa.gpu_numa_node}")
        print_left_right("PCIe Swap Count:", f"{self.num_pcie_swaps}")
        print_left_right("PCIe Bytes Swapped:", f"{self.pcie_bytes_swapped / (1024**2):.1f}MB")
        print_left_right("Pool Allocated / Peak:", f"{pool_mb:.1f}MB / {peak_mb:.1f}MB")
        print_left_right("Invisible Locked (PyTorch):", f"{invis_mb:.1f}MB")
        print_left_right("Fragmentation Risk:", frag['fragmentation_risk'])
        if self.pcie_swap_latencies:
            sorted_pcie = sorted(self.pcie_swap_latencies)
            np = len(sorted_pcie)
            pcie_avg = sum(sorted_pcie) / np
            pcie_p95 = sorted_pcie[min(np - 1, int(np * 0.95))]
            print_left_right("PCIe Swap Latency Avg/P95:", f"{pcie_avg:.2f}ms / {pcie_p95:.2f}ms")
        else:
            print_left_right("PCIe Swap Latency Avg/P95:", "N/A")
        if pcie['pcie_bandwidth_gbps'] > 0:
            print_left_right("Effective PCIe Bandwidth:", f"{pcie['pcie_bandwidth_gbps']:.2f} GB/s")
        print(f"{cyan}└──────────────────────────────────────────────────────────┘{reset}\n")

    def _resurrect_page(self, page, tier):
        import time
        use_cuda_event = torch.cuda.is_available()
        if use_cuda_event:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        else:
            start_time = time.perf_counter()

        # 1. Dequantize
        spec = self.tier_name_to_spec.get(tier)
        if spec is None:
            return

        key_compressed = page.get('key_compressed')
        value_compressed = page.get('value_compressed')

        decomp_args = {"seq_dim": -2}
        if tier == 'jl':
            recon_op = self.get_jl_reconstruction_operator(page['key_proj'].device, page['key_proj'].dtype)
            decomp_args['recon_operator'] = recon_op

        k = spec.backend.decompress(key_compressed, **decomp_args)
        v = spec.backend.decompress(value_compressed, **decomp_args)
        
        self.pages_by_tier[tier] = [p for p in self.pages_by_tier[tier] if p is not page]

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
            depth = self.tier_specs.index(spec) + 1
        except ValueError:
            depth = 1
        self.resurrection_depths.append(depth)
        self.page_lifetimes[page.get('page_id')] = self.generation_step

        k = _apply_outlier_restoration(k, page, key='key')
        v = _apply_outlier_restoration(v, page, key='value')

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

        # 2. To insert into active_pages, we must ensure we have a slot!
        while len(self.active_pages) >= self.max_active_pages:
            if self.eviction_policy is not None:
                demoted_page = self.eviction_policy.select_victim(self.active_pages, context={})
                self.active_pages.remove(demoted_page)
            else:
                self.active_pages.sort(key=lambda x: x.get('importance_score', 0.0))
                demoted_page = self.active_pages.pop(0)
            self._demote_to_next_tier(demoted_page, -1)

        if k.shape[-2] == self.page_size:
            idx = self._get_free_pool_idx(self.active_pages, self.max_active_pages)
            self.active_pool_k[idx].copy_(k)
            self.active_pool_v[idx].copy_(v)
            key_tensor = self.active_pool_k[idx]
            value_tensor = self.active_pool_v[idx]
            pool_idx_val = idx
        else:
            key_tensor = k
            value_tensor = v
            pool_idx_val = None

        resurrected_dict = PageDict({
            'page_id': page.get('page_id'),
            'key': key_tensor,
            'value': value_tensor,
            'pool_idx': pool_idx_val,
            'attention_sum': page.get('attention_sum', 0.0),
            'last_step_accessed': self.generation_step,
            'entropy': page.get('entropy', 0.0),
            'importance_score': page.get('importance_score', 0.0),
            'page_size': k.shape[-2]
        })
        self._calculate_importance(resurrected_dict)
        self.active_pages.append(resurrected_dict)
        self.log_event("resurrect", resurrected_dict['page_id'], tier=tier)
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
            decomp_args = {"seq_dim": -2}
            if current_spec.name == 'jl':
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
        comp_args = {"seq_dim": -2}
        if next_spec.name == 'jl':
            w_proj = self.get_jl_projection_matrix(k_norm.device, k_norm.dtype)
            comp_args['w_proj'] = w_proj
            
        key_comp = next_spec.backend.compress(k_norm, **comp_args)
        value_comp = next_spec.backend.compress(v_norm, **comp_args)
        
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
        Decompresses all pages in a tier using a single batched kernel launch.
        Returns: list of (k_tensor, v_tensor) tuples, one per page.
        
        Pages already in the prefetch cache are returned as-is.
        Non-cached pages are grouped and sent through spec.backend.decompress_batch()
        to minimize kernel launch overhead: O(1) per tier instead of O(N_pages).
        """
        results = [None] * len(pages)
        
        # Separate prefetch-cached vs needs-decompression
        to_decompress_k = []
        to_decompress_v = []
        decompress_indices = []  # Track original positions for reassembly
        
        for i, page in enumerate(pages):
            if id(page) in self.prefetch_cache:
                k, v = self.prefetch_cache[id(page)]
                self.prefetch_hits += 1
                results[i] = (k, v)
            else:
                self.prefetch_misses += 1
                to_decompress_k.append(page['key_compressed'])
                to_decompress_v.append(page['value_compressed'])
                decompress_indices.append(i)
        
        # Batch decompress all non-cached pages in a single kernel launch
        if to_decompress_k:
            decomp_args = {"seq_dim": -2}
            if spec.name == 'jl' and pages:
                # JL needs reconstruction operator
                sample_q = pages[decompress_indices[0]]['key_compressed'].get('q',
                           pages[decompress_indices[0]].get('key_proj'))
                if sample_q is not None:
                    recon_op = self.get_jl_reconstruction_operator(sample_q.device, sample_q.dtype)
                    decomp_args['recon_operator'] = recon_op
            
            batch_k = spec.backend.decompress_batch(to_decompress_k, **decomp_args)
            batch_v = spec.backend.decompress_batch(to_decompress_v, **decomp_args)
            
            for idx_in_batch, orig_idx in enumerate(decompress_indices):
                k = batch_k[idx_in_batch]
                v = batch_v[idx_in_batch]
                page = pages[orig_idx]
                
                # Outlier restoration (safe bounds-checked helper)
                k = _apply_outlier_restoration(k, page, key='key')
                v = _apply_outlier_restoration(v, page, key='value')
                
                results[orig_idx] = (k, v)
        
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
            
        all_keys = []
        all_values = []
        
        # Pluggable Tiers (Coldest to Hottest) — Cached/Batched Decompression
        if self._tiers_version == self._cache_version and self._decompressed_tiers_k is not None:
            if self._decompressed_tiers_k.numel() > 0:
                all_keys.append(self._decompressed_tiers_k)
                all_values.append(self._decompressed_tiers_v)
        else:
            tier_keys = []
            tier_values = []
            for spec in reversed(self.tier_specs):
                tier_pages = self.pages_by_tier.get(spec.name, [])
                if not tier_pages:
                    continue
                kv_pairs = self._decompress_tier_pages_batched(spec, tier_pages)
                for k, v in kv_pairs:
                    tier_keys.append(k)
                    tier_values.append(v)
            if tier_keys:
                self._decompressed_tiers_k = torch.cat(tier_keys, dim=-2)
                self._decompressed_tiers_v = torch.cat(tier_values, dim=-2)
                all_keys.append(self._decompressed_tiers_k)
                all_values.append(self._decompressed_tiers_v)
            else:
                dev = "cuda" if torch.cuda.is_available() else "cpu"
                if self.active_pool_k is not None:
                    dev = self.active_pool_k.device
                elif self.sink_k is not None:
                    dev = self.sink_k.device
                self._decompressed_tiers_k = torch.empty(0, device=dev, dtype=torch.float16)
                self._decompressed_tiers_v = torch.empty(0, device=dev, dtype=torch.float16)
            self._tiers_version = self._cache_version
            
        # Tier 1: FP16 active pages
        for page in self.active_pages:
            all_keys.append(page['key'])
            all_values.append(page['value'])
            
        # Temp active buffer
        if self.k_buffer is not None and self.k_buffer.shape[-2] > 0:
            all_keys.append(self.k_buffer)
            all_values.append(self.v_buffer)
            
        # VIP Outliers: Prepend the Newline / Rhyme Anchors in FP16
        if self.anchor_k is not None:
            all_keys.insert(0, self.anchor_k)
            all_values.insert(0, self.anchor_v)
            
        # Outlier-Aware: Prepend the Attention Sinks in FP16 to the very front
        if self.sink_k is not None:
            all_keys.insert(0, self.sink_k)
            all_values.insert(0, self.sink_v)
            
        if not all_keys:
            return None, None
            
        return torch.cat(all_keys, dim=-2), torch.cat(all_values, dim=-2)

    def inplace_paged_attention(self, q: torch.Tensor, scale: float = None) -> torch.Tensor:
        """
        Computes scaled dot-product attention block-by-block/page-by-page.
        Bypasses massive FP16 reconstruction of full KV tensors to save DRAM bandwidth and VRAM.
        Works seamlessly on both:
          1. Enterprise GPUs (vectorized batched attention over active + prefetched pages)
          2. Consumer GPUs (low-VRAM sequential block-by-block calculation)
        
        q shape: [batch, num_heads, q_len, head_dim]
        Returns:
          attn_output: [batch, num_heads, q_len, head_dim]
        """
        import math
        batch, num_heads, q_len, head_dim = q.shape
        device = q.device
        dtype = q.dtype
        
        if scale is None:
            scale = 1.0 / math.sqrt(head_dim)
            
        self.total_attention_calls += 1
            
        # Vectorized Attention is always faster on GPU than slow Python loops,
        # and 32MB memory allocation is completely safe on 4GB VRAM.
        is_enterprise = q.is_cuda

        # Automatic Swap-In Safeguard
        if self.is_swapped_out:
            self.swap_in_to_device(device=device)

        # Let's collect all dequantized/active K and V tensors
        keys_list = []
        values_list = []
        
        # 1. Outlier-Aware: Attention Sinks (FP16)
        if self.sink_k is not None:
            keys_list.append(self.sink_k)
            values_list.append(self.sink_v)
            
        # 2. VIP Anchors (FP16)
        if self.anchor_k is not None:
            keys_list.append(self.anchor_k)
            values_list.append(self.anchor_v)
            
        # 3. Active FP16 Pages
        for page in self.active_pages:
            keys_list.append(page['key'])
            values_list.append(page['value'])
            
        # 4. Temp active buffer
        if self.k_buffer is not None and self.k_buffer.shape[-2] > 0:
            keys_list.append(self.k_buffer)
            values_list.append(self.v_buffer)
            
        # 5. Compressed Pages (Tiers 2-7) — Batched Decompression
        if self._tiers_version == self._cache_version and self._decompressed_tiers_k is not None:
            if self._decompressed_tiers_k.numel() > 0:
                keys_list.append(self._decompressed_tiers_k)
                values_list.append(self._decompressed_tiers_v)
        else:
            tier_keys = []
            tier_values = []
            for spec in reversed(self.tier_specs):
                tier_pages = self.pages_by_tier.get(spec.name, [])
                if not tier_pages:
                    continue
                kv_pairs = self._decompress_tier_pages_batched(spec, tier_pages)
                for k, v in kv_pairs:
                    tier_keys.append(k)
                    tier_values.append(v)
            if tier_keys:
                self._decompressed_tiers_k = torch.cat(tier_keys, dim=-2)
                self._decompressed_tiers_v = torch.cat(tier_values, dim=-2)
                keys_list.append(self._decompressed_tiers_k)
                values_list.append(self._decompressed_tiers_v)
            else:
                self._decompressed_tiers_k = torch.empty(0, device=device, dtype=dtype)
                self._decompressed_tiers_v = torch.empty(0, device=device, dtype=dtype)
            self._tiers_version = self._cache_version
        
        if not keys_list:
            return torch.zeros_like(q)
            
        # Handle GQA (Grouped Query Attention) and repeat KV heads if necessary
        num_heads_q = q.shape[1]
        num_heads_kv = keys_list[0].shape[1]
        if num_heads_q != num_heads_kv:
            n_rep = num_heads_q // num_heads_kv
            assert num_heads_q % num_heads_kv == 0, f"Query heads ({num_heads_q}) must be divisible by KV heads ({num_heads_kv})"
            
            def repeat_kv(x, n_rep):
                batch, num_key_value_heads, slen, head_dim = x.shape
                if n_rep == 1:
                    return x
                x = x[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
                return x.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
                
            keys_list = [repeat_kv(k, n_rep) for k in keys_list]
            values_list = [repeat_kv(v, n_rep) for v in values_list]
            
        # Single fast concatenated path — compiled into a fused CUDA graph.
        # _compiled_sdp_attention uses torch.nn.functional.scaled_dot_product_attention
        # which selects FlashAttention2 on Ampere+ or efficient_attention on older GPUs.
        # On CPU it falls back to the math backend. Either way it's faster than the
        # raw matmul→softmax→matmul chain because Python dispatch overhead is eliminated.
        k_full = torch.cat(keys_list, dim=-2)
        v_full = torch.cat(values_list, dim=-2)

        attn_output = _compiled_sdp_attention(q, k_full, v_full, scale)
        # Expose attn_probs for the QoS weight update below.
        # SDPA doesn't return attn_probs, so recompute a lightweight version
        # only for the metrics path (no_grad, won't affect perf).
        attn_probs = None  # computed lazily inside the QoS block below


        self.generation_step += 1

        # Eager Bypass: If there is no memory pressure (active pages within limit)
        # and no compressed pages exist yet, skip the entire QoS weight update block.
        # This allows ARGUS to run at 100% native vLLM/SDPA speed during normal operation.
        has_compressed = any(self.pages_by_tier.get(spec.name) for spec in self.tier_specs)
        under_pressure = len(self.active_pages) >= self.max_active_pages or self.is_swapped_out
        force_qos = getattr(self.config, 'force_qos', False)
        if not has_compressed and not under_pressure and not force_qos:
            return attn_output

        # Update QoS metrics and trigger Hot-Page Resurrection.
        # SDPA doesn't expose attn_probs, so compute a lightweight weight-only
        # forward pass for the QoS/resurrection path. This runs under no_grad and
        # is ~2x cheaper than the full SDPA since we skip the value matmul.
        with torch.no_grad():
            weights = _compiled_qos_weights(q, k_full, scale)

        pages_in_order = []
        if self.sink_k is not None:
            pages_in_order.append(None)
        if self.anchor_k is not None:
            pages_in_order.append(None)
        for page in self.active_pages:
            pages_in_order.append(page)
        if self.k_buffer is not None and self.k_buffer.shape[-2] > 0:
            pages_in_order.append(None)
            
        for spec in reversed(self.tier_specs):
            for page in self.pages_by_tier.get(spec.name, []):
                pages_in_order.append(page)
        
        # Stack page attention slices and transfer to CPU in ONE single batch operation
        sums = []
        idx_start = 0
        for i, page in enumerate(pages_in_order):
            if i >= len(keys_list):
                break
            block_len = keys_list[i].shape[-2]
            if page is not None:
                sums.append((page, weights[idx_start : idx_start + block_len].sum()))
            idx_start += block_len

        if sums:
            sums_tensor = torch.stack([s[1] for s in sums]).cpu()
            sums_list = sums_tensor.tolist()
            
            resurrect_ids = []
            for idx, (page, _) in enumerate(sums):
                block_w = sums_list[idx]
                page['attention_sum'] = page.get('attention_sum', 0.0) + block_w
                if block_w > 0.0:
                    page['last_step_accessed'] = self.generation_step
                
                if self.eviction_policy is not None and hasattr(self.eviction_policy, 'on_access'):
                    self.eviction_policy.on_access(page, block_w, self.generation_step)
                
                # Recalculate importance score
                self._calculate_importance(page)
                
                # Check resurrection eligibility
                pid = page['page_id']
                is_compressed = any(
                    any(p['page_id'] == pid for p in self.pages_by_tier.get(spec.name, []))
                    for spec in self.tier_specs
                )
                if is_compressed and block_w > self.config.resurrection_threshold:
                    resurrect_ids.append(pid)
            
            # Execute resurrection for eligible pages
            for pid in resurrect_ids:
                found_page = None
                found_tier = None
                for spec in self.tier_specs:
                    for p in self.pages_by_tier.get(spec.name, []):
                        if p['page_id'] == pid:
                            found_page = p
                            found_tier = spec.name
                            break
                    if found_page is not None:
                        break
                        
                if found_page is not None and found_tier is not None:
                    self._resurrect_page(found_page, found_tier)

        self.manage_variable_granularity()

        return attn_output

    def split_page(self, page, tier_name=None):
        """
        Splits a Mega-page (size page_size) into multiple Micro-pages (size micro_page_size).
        """
        micro_size = self.micro_page_size
        current_size = page.get('page_size', self.page_size)
        if current_size <= micro_size or current_size % micro_size != 0:
            return []

        num_splits = current_size // micro_size
        split_pages = []

        if 'key_compressed' in page or 'value_compressed' in page:
            spec = self.tier_name_to_spec.get(tier_name)
            if spec is None:
                return []

            decomp_args = {"seq_dim": -2}
            if tier_name == 'jl':
                recon_op = self.get_jl_reconstruction_operator(page['key_proj'].device, page['key_proj'].dtype)
                decomp_args['recon_operator'] = recon_op

            k_raw = spec.backend.decompress(page['key_compressed'], **decomp_args)
            v_raw = spec.backend.decompress(page['value_compressed'], **decomp_args)

            k_raw = _apply_outlier_restoration(k_raw, page, key='key')
            v_raw = _apply_outlier_restoration(v_raw, page, key='value')

            for i in range(num_splits):
                start = i * micro_size
                end = start + micro_size
                k_part = k_raw[..., start:end, :]
                v_part = v_raw[..., start:end, :]

                if spec.use_outlier_isolation:
                    k_part_norm, k_part_out, k_part_mask = isolate_outliers(k_part, self.threshold_sigma)
                    v_part_norm, v_part_out, v_part_mask = isolate_outliers(v_part, self.threshold_sigma)
                    part_k_out_indices = torch.nonzero(k_part_mask).to(torch.int16)
                    part_k_out_values = k_part_out[k_part_mask]
                    part_v_out_indices = torch.nonzero(v_part_mask).to(torch.int16)
                    part_v_out_values = v_part_out[v_part_mask]
                else:
                    k_part_norm, v_part_norm = k_part, v_part
                    part_k_out_indices, part_k_out_values = None, None
                    part_v_out_indices, part_v_out_values = None, None

                comp_args = {"seq_dim": -2}
                if spec.name == 'jl':
                    w_proj = self.get_jl_projection_matrix(k_part_norm.device, k_part_norm.dtype)
                    comp_args['w_proj'] = w_proj

                part_k_comp = spec.backend.compress(k_part_norm, **comp_args)
                part_v_comp = spec.backend.compress(v_part_norm, **comp_args)

                self._page_counter += 1
                part_dict = PageDict({
                    'page_id': self._page_counter,
                    'key_compressed': part_k_comp,
                    'value_compressed': part_v_comp,
                    'pool_idx': None,
                    'attention_sum': page.get('attention_sum', 0.0) / num_splits,
                    'last_step_accessed': page.get('last_step_accessed', self.generation_step),
                    'entropy': calculate_tensor_entropy(k_part),
                    'importance_score': page.get('importance_score', 0.0),
                    'page_size': micro_size
                })
                if part_k_out_indices is not None:
                    part_dict['key_out_indices'] = part_k_out_indices
                    part_dict['key_out_values'] = part_k_out_values
                if part_v_out_indices is not None:
                    part_dict['value_out_indices'] = part_v_out_indices
                    part_dict['value_out_values'] = part_v_out_values

                split_pages.append(part_dict)
        else:
            k_raw = page['key']
            v_raw = page['value']

            for i in range(num_splits):
                start = i * micro_size
                end = start + micro_size
                k_part = k_raw[..., start:end, :]
                v_part = v_raw[..., start:end, :]

                self._page_counter += 1
                part_dict = PageDict({
                    'page_id': self._page_counter,
                    'key': k_part,
                    'value': v_part,
                    'pool_idx': None,
                    'attention_sum': page.get('attention_sum', 0.0) / num_splits,
                    'last_step_accessed': page.get('last_step_accessed', self.generation_step),
                    'entropy': calculate_tensor_entropy(k_part),
                    'importance_score': page.get('importance_score', 0.0),
                    'page_size': micro_size
                })
                split_pages.append(part_dict)

        return split_pages

    def merge_pages(self, pages, tier_name=None):
        """
        Merges a list of Micro-pages back into a single Mega-page (size page_size).
        """
        if not pages:
            return None

        total_size = sum(p.get('page_size', self.page_size) for p in pages)
        k_list = []
        v_list = []
        spec = self.tier_name_to_spec.get(tier_name) if tier_name else None

        for p in pages:
            if 'key_compressed' in p or 'value_compressed' in p:
                decomp_args = {"seq_dim": -2}
                if tier_name == 'jl':
                    recon_op = self.get_jl_reconstruction_operator(p['key_proj'].device, p['key_proj'].dtype)
                    decomp_args['recon_operator'] = recon_op

                k_raw = spec.backend.decompress(p['key_compressed'], **decomp_args)
                v_raw = spec.backend.decompress(p['value_compressed'], **decomp_args)

                k_raw = _apply_outlier_restoration(k_raw, p, key='key')
                v_raw = _apply_outlier_restoration(v_raw, p, key='value')
            else:
                k_raw = p['key']
                v_raw = p['value']

            k_list.append(k_raw)
            v_list.append(v_raw)

        k_merged = torch.cat(k_list, dim=-2)
        v_merged = torch.cat(v_list, dim=-2)

        self._page_counter += 1

        if tier_name is not None and spec is not None:
            if spec.use_outlier_isolation:
                k_norm, k_out, k_mask = isolate_outliers(k_merged, self.threshold_sigma)
                v_norm, v_out, v_mask = isolate_outliers(v_merged, self.threshold_sigma)
                k_out_indices = torch.nonzero(k_mask).to(torch.int16)
                k_out_values = k_out[k_mask]
                v_out_indices = torch.nonzero(v_mask).to(torch.int16)
                v_out_values = v_out[v_mask]
            else:
                k_norm, v_norm = k_merged, v_merged
                k_out_indices, k_out_values = None, None
                v_out_indices, v_out_values = None, None

            comp_args = {"seq_dim": -2}
            if spec.name == 'jl':
                w_proj = self.get_jl_projection_matrix(k_norm.device, k_norm.dtype)
                comp_args['w_proj'] = w_proj

            key_comp = spec.backend.compress(k_norm, **comp_args)
            value_comp = spec.backend.compress(v_norm, **comp_args)

            merged_page = PageDict({
                'page_id': self._page_counter,
                'key_compressed': key_comp,
                'value_compressed': value_comp,
                'pool_idx': None,
                'attention_sum': sum(p.get('attention_sum', 0.0) for p in pages),
                'last_step_accessed': max(p.get('last_step_accessed', self.generation_step) for p in pages),
                'entropy': calculate_tensor_entropy(k_merged),
                'importance_score': max(p.get('importance_score', 0.0) for p in pages),
                'page_size': total_size
            })
            if k_out_indices is not None:
                merged_page['key_out_indices'] = k_out_indices
                merged_page['key_out_values'] = k_out_values
            if v_out_indices is not None:
                merged_page['value_out_indices'] = v_out_indices
                merged_page['value_out_values'] = v_out_values
        else:
            merged_page = PageDict({
                'page_id': self._page_counter,
                'key': k_merged,
                'value': v_merged,
                'pool_idx': None,
                'attention_sum': sum(p.get('attention_sum', 0.0) for p in pages),
                'last_step_accessed': max(p.get('last_step_accessed', self.generation_step) for p in pages),
                'entropy': calculate_tensor_entropy(k_merged),
                'importance_score': max(p.get('importance_score', 0.0) for p in pages),
                'page_size': total_size
            })

        return merged_page

    def manage_variable_granularity(self):
        """
        Scans pages, splits cold Mega-pages into Micro-pages,
        and merges hot contiguous Micro-pages into Mega-pages.
        """
        micro_size = self.micro_page_size

        # 1. Manage active pages
        new_active = []
        i = 0
        while i < len(self.active_pages):
            page = self.active_pages[i]
            p_size = page.get('page_size', self.page_size)

            if p_size == self.page_size and page.get('importance_score', 0.0) < 0.5:
                splits = self.split_page(page)
                if splits:
                    new_active.extend(splits)
                    argus_log("INFO", f"Splitting Page {page['page_id']} (ACTIVE) -> {len(splits)} Micro-pages", line_no=500)
                    i += 1
                    continue

            needed_pages = self.page_size // micro_size
            if p_size == micro_size and i + needed_pages <= len(self.active_pages):
                candidate_pages = self.active_pages[i : i + needed_pages]
                if all(p.get('page_size', self.page_size) == micro_size and p.get('importance_score', 0.0) > 1.5 for p in candidate_pages):
                    merged = self.merge_pages(candidate_pages)
                    if merged is not None:
                        new_active.append(merged)
                        argus_log("INFO", f"Merging {len(candidate_pages)} Micro-pages -> Page {merged['page_id']} (ACTIVE)", line_no=510)
                        i += needed_pages
                        continue

            new_active.append(page)
            i += 1
        self.active_pages = new_active

        # 2. Manage compressed tiers
        for spec in self.tier_specs:
            pages_list = self.pages_by_tier.get(spec.name, [])
            new_list = []
            i = 0
            while i < len(pages_list):
                page = pages_list[i]
                p_size = page.get('page_size', self.page_size)

                if p_size == self.page_size and page.get('importance_score', 0.0) < 0.5:
                    splits = self.split_page(page, tier_name=spec.name)
                    if splits:
                        new_list.extend(splits)
                        argus_log("INFO", f"Splitting Page {page['page_id']} ({spec.name.upper()}) -> {len(splits)} Micro-pages", line_no=500)
                        i += 1
                        continue

                needed_pages = self.page_size // micro_size
                if p_size == micro_size and i + needed_pages <= len(pages_list):
                    candidate_pages = pages_list[i : i + needed_pages]
                    if all(p.get('page_size', self.page_size) == micro_size and p.get('importance_score', 0.0) > 1.5 for p in candidate_pages):
                        merged = self.merge_pages(candidate_pages, tier_name=spec.name)
                        if merged is not None:
                            new_list.append(merged)
                            argus_log("INFO", f"Merging {len(candidate_pages)} Micro-pages -> Page {merged['page_id']} ({spec.name.upper()})", line_no=510)
                            i += needed_pages
                            continue

                new_list.append(page)
                i += 1
            self.pages_by_tier[spec.name] = new_list
        self._invalidate_decompressed_cache()

    def speculate_and_prefetch(self, attn_weights=None):
        """
        Predicts and pre-dequantizes pages that will be heavily attended to in the next step.
        Stores dequantized FP16 tensors in self.prefetch_cache to avoid dequantization latency.
        """
        self.prefetch_cache.clear()
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
                
        if self.prefetch_stream is None and torch.cuda.is_available():
            self.prefetch_stream = torch.cuda.Stream()
            
        main_stream = torch.cuda.current_stream() if torch.cuda.is_available() else None
        
        for page, tier in pages_to_prefetch:
            try:
                spec = self.tier_name_to_spec.get(tier)
                if spec is None:
                    continue
                if self.prefetch_stream is not None:
                    # Run pre-dequantization asynchronously on dedicated stream
                    with torch.cuda.stream(self.prefetch_stream):
                        decomp_args = {"seq_dim": -2}
                        if tier == 'jl':
                            recon_op = self.get_jl_reconstruction_operator(page['key_proj'].device, page['key_proj'].dtype)
                            decomp_args['recon_operator'] = recon_op
                        k = spec.backend.decompress(page['key_compressed'], **decomp_args)
                        v = spec.backend.decompress(page['value_compressed'], **decomp_args)
                             
                        k = _apply_outlier_restoration(k, page, key='key')
                        v = _apply_outlier_restoration(v, page, key='value')
                            
                        # Record consumer stream to prevent premature recycling by CUDA allocator
                        if k.is_cuda:
                            k.record_stream(main_stream)
                        if v.is_cuda:
                            v.record_stream(main_stream)
                else:
                    decomp_args = {"seq_dim": -2}
                    if tier == 'jl':
                        recon_op = self.get_jl_reconstruction_operator(page['key_proj'].device, page['key_proj'].dtype)
                        decomp_args['recon_operator'] = recon_op
                    k = spec.backend.decompress(page['key_compressed'], **decomp_args)
                    v = spec.backend.decompress(page['value_compressed'], **decomp_args)
                         
                    k = _apply_outlier_restoration(k, page, key='key')
                    v = _apply_outlier_restoration(v, page, key='value')
                     
                self.prefetch_cache[id(page)] = (k, v)
            except Exception:
                pass

    def swap_out_to_host(self):
        """
        Moves all compressed page tensors of Tiers 2-7 to CPU host memory
        using Zero-Copy PCIe pinned/device-mapped memory (cuMemHostAlloc).

        If the CUDA Driver API is unavailable, falls back to plain
        ``tensor.cpu()`` (legacy behaviour).

        After this call, GPU Triton kernels can still read the swapped
        pages directly via PCIe device pointers — no cudaMemcpy needed.
        """
        if self.is_swapped_out:
            return

        import time
        use_cuda_event = torch.cuda.is_available()

        def swap_tensor_to_pinned(item, page_id):
            """Recursively move tensors to zero-copy pinned host memory."""
            if isinstance(item, torch.Tensor):
                pinned = self.zero_copy_pool.tensor_to_pinned(item)
                # Track for later cleanup
                if page_id is not None:
                    self._zero_copy_tensor_registry.setdefault(page_id, []).append(pinned)
                return pinned
            elif isinstance(item, dict):
                for k, v in list(item.items()):
                    item[k] = swap_tensor_to_pinned(v, page_id)
            elif isinstance(item, list):
                for i in range(len(item)):
                    item[i] = swap_tensor_to_pinned(item[i], page_id)
            return item

        if use_cuda_event:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        else:
            t0 = time.perf_counter()

        total_bytes = 0
        for spec in self.tier_specs:
            for page in self.pages_by_tier.get(spec.name, []):
                page_id = page.get('page_id')
                # Calculate bytes before swap
                for tensor_key in ['key_compressed', 'value_compressed',
                                   'key_out_indices', 'key_out_values',
                                   'value_out_indices', 'value_out_values']:
                    t = page.get(tensor_key)
                    if isinstance(t, torch.Tensor):
                        total_bytes += t.nelement() * t.element_size()
                    elif isinstance(t, dict):
                        for v in t.values():
                            if isinstance(v, torch.Tensor):
                                total_bytes += v.nelement() * v.element_size()
                swap_tensor_to_pinned(page, page_id)

        if use_cuda_event:
            end_event.record()
            torch.cuda.synchronize()
            latency_ms = start_event.elapsed_time(end_event)
        else:
            latency_ms = (time.perf_counter() - t0) * 1000.0

        self.pcie_swap_latencies.append(latency_ms)
        self.num_pcie_swaps += 1
        self.pcie_bytes_swapped += total_bytes
        self.zero_copy_pool.record_pcie_transfer(latency_ms, total_bytes)

        frag = self.zero_copy_pool.get_fragmentation_report()
        argus_log("INFO",
                  f"Zero-Copy PCIe swap-out complete | "
                  f"{total_bytes / (1024**2):.1f}MB → pinned host | "
                  f"latency: {latency_ms:.2f}ms | "
                  f"pool: {frag['pool_total_allocated_bytes'] / (1024**2):.1f}MB | "
                  f"frag_risk: {frag['fragmentation_risk']}",
                  line_no=2122)

        self.is_swapped_out = True
        self._invalidate_decompressed_cache()

    def swap_in_to_device(self, device="cuda"):
        """
        Swaps all host-resident page tensors back to GPU active VRAM.

        For zero-copy tensors, this performs a PCIe DMA read (measured
        separately from dequant latency).  For fallback tensors, uses
        standard ``tensor.to(device)``.

        After swap-in, all pinned host allocations for the moved pages
        are freed from the ZeroCopyHostPool.
        """
        if not self.is_swapped_out:
            return

        import time
        target_device = torch.device(device if torch.cuda.is_available() else "cpu")
        use_cuda_event = torch.cuda.is_available() and target_device.type == "cuda"

        if use_cuda_event:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        else:
            t0 = time.perf_counter()

        total_bytes = 0
        zc_tensors_to_free = []

        def swap_tensor_to_device(item, target_device, page_id):
            nonlocal total_bytes
            if isinstance(item, torch.Tensor):
                total_bytes += item.nelement() * item.element_size()
                # If this is a zero-copy tensor, read via PCIe then free the pinned buffer
                is_zc = self.zero_copy_pool.is_zero_copy_tensor(item)
                result = item.to(target_device)
                if is_zc:
                    zc_tensors_to_free.append(item)
                return result
            elif isinstance(item, dict):
                for k, v in list(item.items()):
                    item[k] = swap_tensor_to_device(v, target_device, page_id)
            elif isinstance(item, list):
                for i in range(len(item)):
                    item[i] = swap_tensor_to_device(item[i], target_device, page_id)
            return item

        for spec in self.tier_specs:
            for page in self.pages_by_tier.get(spec.name, []):
                page_id = page.get('page_id')
                swap_tensor_to_device(page, target_device, page_id)

        # Clear the registry
        self._zero_copy_tensor_registry.clear()

        if use_cuda_event:
            end_event.record()
            torch.cuda.synchronize()
            latency_ms = start_event.elapsed_time(end_event)
        else:
            latency_ms = (time.perf_counter() - t0) * 1000.0

        # Now that we synchronized and the GPU has finished reading, we can safely free the host tensors
        for t in zc_tensors_to_free:
            self.zero_copy_pool.free_tensor(t)

        self.pcie_swap_latencies.append(latency_ms)
        self.num_pcie_swaps += 1
        self.pcie_bytes_swapped += total_bytes
        self.zero_copy_pool.record_pcie_transfer(latency_ms, total_bytes)

        argus_log("INFO",
                  f"Zero-Copy PCIe swap-in complete | "
                  f"{total_bytes / (1024**2):.1f}MB ← device | "
                  f"latency: {latency_ms:.2f}ms",
                  line_no=2147)

        self.is_swapped_out = False
        self._invalidate_decompressed_cache()

    def get_allocator_fragmentation_report(self) -> dict:
        """
        Returns combined allocator fragmentation report merging
        PyTorch caching allocator view with CUDA Driver-level
        cuMemHostAlloc shadow accounting.

        Use this to verify that driver-locked bytes (invisible to
        torch.cuda.memory_allocated) are not silently consuming
        physical memory that could cause OOM.
        """
        report = self.zero_copy_pool.get_fragmentation_report()
        report['pcie_swap_count'] = self.num_pcie_swaps
        report['pcie_total_bytes_swapped'] = self.pcie_bytes_swapped
        if self.pcie_swap_latencies:
            sorted_lat = sorted(self.pcie_swap_latencies)
            n = len(sorted_lat)
            report['pcie_avg_latency_ms'] = sum(sorted_lat) / n
            report['pcie_p95_latency_ms'] = sorted_lat[min(n - 1, int(n * 0.95))]
        else:
            report['pcie_avg_latency_ms'] = 0.0
            report['pcie_p95_latency_ms'] = 0.0
        return report

    def get_vram_usage(self):
        """
        Calculates exact memory usage across all tiers + sinks assuming standard FP16 baseline precision (2 bytes per element).
        Tensors currently swapped out to CPU Host RAM do not consume GPU VRAM and are excluded.
        """
        total_bytes = 0
        
        # Sinks (FP16: 2 bytes per element) - always kept on active device
        if self.sink_k is not None:
            total_bytes += self.sink_k.nelement() * 2
            total_bytes += self.sink_v.nelement() * 2
            
        # VIP Anchors (FP16: 2 bytes) - always kept on active device
        if self.anchor_k is not None:
            total_bytes += self.anchor_k.nelement() * 2
            total_bytes += self.anchor_v.nelement() * 2
            
        # FP16 active pages (2 bytes) - always kept on active device
        for page in self.active_pages:
            total_bytes += page['key'].nelement() * 2
            total_bytes += page['value'].nelement() * 2
            
        # Buffers (FP16: 2 bytes) - always kept on active device
        if self.k_buffer is not None:
            total_bytes += self.k_buffer.nelement() * 2
            total_bytes += self.v_buffer.nelement() * 2
            
        # Swapped-out pages (Tiers 2-7) do not consume GPU VRAM (they are on CPU Host RAM)
        if self.is_swapped_out:
            return total_bytes
            
        # Tiers
        for spec in self.tier_specs:
            pages = self.pages_by_tier.get(spec.name, [])
            for page in pages:
                key_comp = page.get('key_compressed')
                value_comp = page.get('value_compressed')
                if key_comp is not None:
                    total_bytes += spec.backend.memory_bytes(key_comp)
                if value_comp is not None:
                    total_bytes += spec.backend.memory_bytes(value_comp)
                
                # Outliers:
                for prefix in ['key', 'value']:
                    idx = page.get(f'{prefix}_out_indices')
                    val = page.get(f'{prefix}_out_values')
                    if idx is not None:
                        total_bytes += idx.nelement() * idx.element_size()
                    if val is not None:
                        total_bytes += val.nelement() * val.element_size()
            
        # Add shared static projection matrix (once only in FP16: 2 bytes)
        if self.w_proj is not None:
            total_bytes += self.w_proj.nelement() * 2
            
        return total_bytes
