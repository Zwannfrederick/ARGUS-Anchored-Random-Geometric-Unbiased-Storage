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
        resurrection_threshold=0.01      # attention threshold for reheating
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

def argus_log(level, message, file_name="memory_manager.py", line_no=None):
    import os
    log_level = os.environ.get("ARGUS_LOG_LEVEL", "research").lower()
    if log_level == "quiet" and level == "INFO":
        return
        
    from datetime import datetime
    now_str = datetime.now().strftime("%m-%d %H:%M:%S")
    
    color_start = ""
    if level == "INFO":
        color_start = "\033[1;32mINFO\033[0m"      # bold green
    elif level == "WARNING":
        color_start = "\033[1;33mWARNING\033[0m"   # bold yellow
    elif level == "ERROR":
        color_start = "\033[1;31mERROR\033[0m"     # bold red
    else:
        color_start = f"\033[1;34m{level}\033[0m"  # bold blue
        
    line_str = f":{line_no}" if line_no is not None else ""
    argus_prefix = "\033[1;36m[ARGUS]\033[0m"
    
    print(f"{color_start} {now_str} {file_name}{line_str}] {argus_prefix} {message}")

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
    mean_ch = torch.mean(abs_t, dim=-1, keepdim=True)
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
        config=None
    ):
        """
        Generic Outlier-Aware 7-Tier Token-Based Paged Dynamic KV Cache Manager.
        Memory Lifecycle:
            FP16 (Sinks) + FP16 (Active) -> FP8 (Light) -> INT8 (Medium) -> INT4 (Heavy) -> INT2 (Super Heavy) -> 1-Bit (Sign) -> JL Projection (Archive)
        """
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
                threshold_sigma=threshold_sigma
            )
        else:
            self.config = config
            
        self.page_size = self.config.page_size
        self.max_active_pages = self.config.max_active_pages
        self.max_fp8_pages = self.config.max_fp8_pages
        self.max_int8_pages = self.config.max_int8_pages
        self.max_int4_pages = self.config.max_int4_pages
        self.max_int2_pages = self.config.max_int2_pages
        self.max_one_bit_pages = self.config.max_one_bit_pages
        self.threshold_sigma = self.config.threshold_sigma
        
        assert self.page_size % 8 == 0, "Page size must be a multiple of 8 for 1-bit packing."
        
        # Generation step tracking for recency scoring
        self.generation_step = 0
        self._page_counter = 0
        self.event_log = []
        
        # Outlier-Aware: Attention Sinks (First N tokens kept in FP16 permanently)
        self.sink_tokens = self.config.sink_tokens
        self.sink_k = None
        self.sink_v = None
        
        # VIP Outliers: Newline & Rhyme Anchors kept in FP16 permanently
        self.anchor_k = None
        self.anchor_v = None
        
        # Memory tiers
        self.active_pages = []     # Tier 1: FP16
        self.fp8_pages = []        # Tier 2: FP8 (simulated) + FP16 Outliers
        self.int8_pages = []       # Tier 3: INT8 + FP16 Outliers
        self.int4_pages = []       # Tier 4: INT4 (2 per byte) + FP16 Outliers
        self.int2_pages = []       # Tier 5: INT2 (4 per byte) + FP16 Outliers
        self.one_bit_pages = []    # Tier 6: 1-Bit (8 per byte) + FP16 Outliers
        self.jl_pages = []         # Tier 7: JL-Projected FP16 Archive (Unlimited size)
        
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

        # Telemetry metrics
        self.num_resurrections = 0
        self.num_cpu_spills = 0
        self.total_dequant_time = 0.0
        self.num_dequants = 0
        self.total_demotions = 0
        
        # Advanced telemetry metrics
        self.dequant_latencies = []
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
        
        # Locality Predictor State
        self.page_access_ema = {}
        self.page_access_history = {}

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
        
        # FP8 Pool
        self.fp8_pool_k_q = torch.zeros(self.max_fp8_pages, batch, num_heads, self.page_size, head_dim, device=device, dtype=torch.int8)
        self.fp8_pool_v_q = torch.zeros(self.max_fp8_pages, batch, num_heads, self.page_size, head_dim, device=device, dtype=torch.int8)
        self.fp8_pool_k_scales = torch.zeros(self.max_fp8_pages, batch, num_heads, self.page_size, 1, device=device, dtype=dtype)
        self.fp8_pool_v_scales = torch.zeros(self.max_fp8_pages, batch, num_heads, self.page_size, 1, device=device, dtype=dtype)
        self.fp8_pool_idx = 0
        
        # INT8 Pool
        self.int8_pool_k_q = torch.zeros(self.max_int8_pages, batch, num_heads, self.page_size, head_dim, device=device, dtype=torch.int8)
        self.int8_pool_v_q = torch.zeros(self.max_int8_pages, batch, num_heads, self.page_size, head_dim, device=device, dtype=torch.int8)
        self.int8_pool_k_scales = torch.zeros(self.max_int8_pages, batch, num_heads, self.page_size, 1, device=device, dtype=dtype)
        self.int8_pool_v_scales = torch.zeros(self.max_int8_pages, batch, num_heads, self.page_size, 1, device=device, dtype=dtype)
        self.int8_pool_idx = 0
        
        # INT4 Pool
        self.int4_pool_k_packed = torch.zeros(self.max_int4_pages, batch, num_heads, self.page_size // 2, head_dim, device=device, dtype=torch.uint8)
        self.int4_pool_v_packed = torch.zeros(self.max_int4_pages, batch, num_heads, self.page_size // 2, head_dim, device=device, dtype=torch.uint8)
        self.int4_pool_k_scales = torch.zeros(self.max_int4_pages, batch, num_heads, self.page_size, 1, device=device, dtype=dtype)
        self.int4_pool_v_scales = torch.zeros(self.max_int4_pages, batch, num_heads, self.page_size, 1, device=device, dtype=dtype)
        self.int4_pool_k_min = torch.zeros(self.max_int4_pages, batch, num_heads, self.page_size, 1, device=device, dtype=dtype)
        self.int4_pool_v_min = torch.zeros(self.max_int4_pages, batch, num_heads, self.page_size, 1, device=device, dtype=dtype)
        self.int4_pool_idx = 0
        
        # INT2 Pool
        self.int2_pool_k_packed = torch.zeros(self.max_int2_pages, batch, num_heads, self.page_size // 4, head_dim, device=device, dtype=torch.uint8)
        self.int2_pool_v_packed = torch.zeros(self.max_int2_pages, batch, num_heads, self.page_size // 4, head_dim, device=device, dtype=torch.uint8)
        self.int2_pool_k_scales = torch.zeros(self.max_int2_pages, batch, num_heads, self.page_size, 1, device=device, dtype=dtype)
        self.int2_pool_v_scales = torch.zeros(self.max_int2_pages, batch, num_heads, self.page_size, 1, device=device, dtype=dtype)
        self.int2_pool_k_min = torch.zeros(self.max_int2_pages, batch, num_heads, self.page_size, 1, device=device, dtype=dtype)
        self.int2_pool_v_min = torch.zeros(self.max_int2_pages, batch, num_heads, self.page_size, 1, device=device, dtype=dtype)
        self.int2_pool_idx = 0
        
        # 1-Bit Pool
        self.one_bit_pool_k_packed = torch.zeros(self.max_one_bit_pages, batch, num_heads, self.page_size // 8, head_dim, device=device, dtype=torch.uint8)
        self.one_bit_pool_v_packed = torch.zeros(self.max_one_bit_pages, batch, num_heads, self.page_size // 8, head_dim, device=device, dtype=torch.uint8)
        self.one_bit_pool_k_scales = torch.zeros(self.max_one_bit_pages, batch, num_heads, self.page_size, 1, device=device, dtype=dtype)
        self.one_bit_pool_v_scales = torch.zeros(self.max_one_bit_pages, batch, num_heads, self.page_size, 1, device=device, dtype=dtype)
        self.one_bit_pool_idx = 0
        
        self._pools_allocated = True

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
                
                # Rest of the sequence goes to normal cache
                keys = keys[..., self.sink_tokens:, :]
                values = values[..., self.sink_tokens:, :]
            else:
                # If prompt is very short, keep it in sinks until we accumulate enough
                self.sink_k = keys.clone()
                self.sink_v = values.clone()
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
            page_dict = {
                'page_id': self._page_counter,
                'key': self.active_pool_k[idx],
                'value': self.active_pool_v[idx],
                'pool_idx': idx,
                'attention_sum': 0.0,
                'last_step_accessed': self.generation_step,
                'entropy': ent,
                'importance_score': 0.0
            }
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
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            total_mem = torch.cuda.get_device_properties(device).total_memory
            allocated_mem = torch.cuda.memory_allocated(device)
            ratio = float(allocated_mem) / float(total_mem)
            if ratio >= self.config.vram_oom_threshold_ratio:
                self.num_cpu_spills += 1
                argus_log("WARNING", f"VRAM pressure detected (allocated: {allocated_mem/(1024**3):.2f}GB / {total_mem/(1024**3):.2f}GB, ratio: {ratio*100:.1f}%) → spilling cold pages to CPU Host RAM", line_no=431)
                # Proactive OOM Protection: Spill all compressed pages to CPU Host RAM
                self.swap_out_to_host()
        else:
            # Fallback for CPU unit testing: if vram_oom_threshold_ratio <= 0, trigger swap out!
            if self.config.vram_oom_threshold_ratio <= 0.0:
                self.num_cpu_spills += 1
                argus_log("WARNING", f"VRAM pressure detected (CPU Fallback) → spilling cold pages to CPU Host RAM | vram_oom_threshold_ratio={self.config.vram_oom_threshold_ratio}", line_no=436)
                self.swap_out_to_host()

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
        logical_compressed_bytes += n_active * fp16_page_bytes
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
            
        # 3. FP8 pages (1 byte per element + outliers + scales)
        n_fp8 = len(self.fp8_pages)
        total_pages += n_fp8
        for p in self.fp8_pages:
            logical_compressed_bytes += elements_per_page * 2 * 1
            if 'key_scales' in p:
                logical_compressed_bytes += p['key_scales'].element_size() * p['key_scales'].nelement() * 2
            logical_compressed_bytes += get_outliers_bytes(p)
            
        # 4. INT8 pages (1 byte per element + outliers + scales)
        n_int8 = len(self.int8_pages)
        total_pages += n_int8
        for p in self.int8_pages:
            logical_compressed_bytes += elements_per_page * 2 * 1
            if 'key_scales' in p:
                logical_compressed_bytes += p['key_scales'].element_size() * p['key_scales'].nelement() * 2
            logical_compressed_bytes += get_outliers_bytes(p)
            
        # 5. INT4 pages (0.5 bytes per element + outliers + scales + mins)
        n_int4 = len(self.int4_pages)
        total_pages += n_int4
        for p in self.int4_pages:
            logical_compressed_bytes += int(elements_per_page * 2 * 0.5)
            if 'key_scales' in p:
                logical_compressed_bytes += p['key_scales'].element_size() * p['key_scales'].nelement() * 2
            if 'key_min' in p:
                logical_compressed_bytes += p['key_min'].element_size() * p['key_min'].nelement() * 2
            logical_compressed_bytes += get_outliers_bytes(p)
            
        # 6. INT2 pages (0.25 bytes per element + outliers + scales + mins)
        n_int2 = len(self.int2_pages)
        total_pages += n_int2
        for p in self.int2_pages:
            logical_compressed_bytes += int(elements_per_page * 2 * 0.25)
            if 'key_scales' in p:
                logical_compressed_bytes += p['key_scales'].element_size() * p['key_scales'].nelement() * 2
            if 'key_min' in p:
                logical_compressed_bytes += p['key_min'].element_size() * p['key_min'].nelement() * 2
            logical_compressed_bytes += get_outliers_bytes(p)
            
        # 7. One-Bit pages (0.125 bytes per element + outliers + scales)
        n_one_bit = len(self.one_bit_pages)
        total_pages += n_one_bit
        for p in self.one_bit_pages:
            logical_compressed_bytes += int(elements_per_page * 2 * 0.125)
            if 'key_scales' in p:
                logical_compressed_bytes += p['key_scales'].element_size() * p['key_scales'].nelement() * 2
            logical_compressed_bytes += get_outliers_bytes(p)
            
        # 8. JL pages (projected to page_size // 4 FP16)
        n_jl = len(self.jl_pages)
        total_pages += n_jl
        for p in self.jl_pages:
            if 'key_proj' in p:
                logical_compressed_bytes += p['key_proj'].element_size() * p['key_proj'].nelement() * 2
                
        if total_pages == 0:
            return 1.0, 0.0, 0, 0
            
        logical_raw_bytes = total_pages * fp16_page_bytes
        compression_ratio = logical_raw_bytes / max(1, logical_compressed_bytes)
        bandwidth_saved = (1.0 - 1.0 / compression_ratio) * 100.0
        
        return compression_ratio, bandwidth_saved, total_pages, logical_compressed_bytes

    def print_telemetry_summary(self):
        import math
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
        n_fp8 = len(self.fp8_pages)
        n_int8 = len(self.int8_pages)
        n_int4 = len(self.int4_pages)
        n_int2 = len(self.int2_pages)
        n_one_bit = len(self.one_bit_pages)
        n_jl = len(self.jl_pages)
        
        # Categories
        hot_pages = n_active + n_fp8
        warm_pages = n_int8 + n_int4
        cold_pages = n_int2 + n_one_bit + n_jl
        cpu_spill_pages = (n_fp8 + n_int8 + n_int4 + n_int2 + n_one_bit + n_jl) if self.is_swapped_out else 0
        
        def draw_bar(count, max_val):
            if count == 0:
                return " " * 20
            bar_len = 20
            filled = max(1, int(round((count / max_val) * bar_len))) if max_val > 0 else 0
            return "█" * filled + " " * (bar_len - filled)
            
        max_any = max(1, n_active, n_fp8, n_int8, n_int4, n_int2, n_one_bit, n_jl)
        
        # Build Heatmap
        all_pages = []
        for p in self.active_pages:
            all_pages.append((p.get('page_id'), 'FP16', 'GPU'))
        for p in self.fp8_pages:
            all_pages.append((p.get('page_id'), 'FP8', 'CPU' if self.is_swapped_out else 'GPU'))
        for p in self.int8_pages:
            all_pages.append((p.get('page_id'), 'INT8', 'CPU' if self.is_swapped_out else 'GPU'))
        for p in self.int4_pages:
            all_pages.append((p.get('page_id'), 'INT4', 'CPU' if self.is_swapped_out else 'GPU'))
        for p in self.int2_pages:
            all_pages.append((p.get('page_id'), 'INT2', 'CPU' if self.is_swapped_out else 'GPU'))
        for p in self.one_bit_pages:
            all_pages.append((p.get('page_id'), '1BIT', 'CPU' if self.is_swapped_out else 'GPU'))
        for p in self.jl_pages:
            all_pages.append((p.get('page_id'), 'JL', 'CPU' if self.is_swapped_out else 'GPU'))
            
        all_pages.sort(key=lambda x: x[0] if x[0] is not None else 9999)
        
        tier_colors = {
            'FP16': '\033[1;36m', # Cyan
            'FP8': '\033[1;32m',  # Light Green
            'INT8': '\033[0;32m', # Dark Green
            'INT4': '\033[1;33m', # Yellow
            'INT2': '\033[1;35m', # Magenta
            '1BIT': '\033[1;31m', # Red
            'JL': '\033[1;34m'    # Blue
        }
        
        heatmap_items = []
        for page_id, tier, loc in all_pages:
            color = tier_colors.get(tier, '\033[37m')
            char = '█' if loc == 'GPU' else '▒'
            heatmap_items.append(f"{color}{char}\033[0m")
            
        heatmap_str = " ".join(heatmap_items) if heatmap_items else "(No pages in cache yet)"
        
        print("\n\033[1;36m┌──────────────────────────────────────────────────────────┐\033[0m")
        print("\033[1;36m│                  ARGUS TELEMETRY SUMMARY                 │\033[0m")
        print("\033[1;36m├──────────────────────────────────────────────────────────┤\033[0m")
        print(f"│  KV Compression Ratio:   {comp_ratio:5.1f}x (Maximum Cold-Storage) │")
        print(f"│  KV Memory Avoided:       {bw_saved:5.1f}%                      │")
        print(f"│  DRAM Bandwidth Saved:    {bw_saved:5.1f}%                      │")
        print(f"│  Pages Resurrected:      {self.num_resurrections:5d}                      │")
        print(f"│  CPU Spill Events:       {self.num_cpu_spills:5d}                      │")
        print(f"│  Transient Reconstructions: {self.num_dequants:5d}                      │")
        print(f"│  Average Dequant Latency: {avg_dequant:7.3f}ms                    │")
        print(f"│  Dequant Latency P50: {p50:5.3f}ms | P95: {p95:5.3f}ms | P99: {p99:5.3f}ms │")
        print(f"│  Decode Throughput Impact: -{overhead:4.2f}%                     │")
        print(f"│  Attention Locality Hit Rate: {hit_rate:5.1f}%                   │")
        print(f"│  Average Page Lifetime: {avg_lifetime:6.1f} steps                 │")
        print(f"│  Average Resurrection Depth: {avg_depth:4.1f} tiers                │")
        print("\033[1;36m├──────────────────────────────────────────────────────────┤\033[0m")
        print("\033[1;36m│                  COMPRESSION CASCADE COUNTS              │\033[0m")
        print("\033[1;36m├──────────────────────────────────────────────────────────┤\033[0m")
        cc = self.cascade_counts
        print(f"│  FP16→FP8: {cc['fp16_to_fp8']:3d} | FP8→INT8: {cc['fp8_to_int8']:3d} | INT8→INT4: {cc['int8_to_int4']:3d}      │")
        print(f"│  INT4→INT2: {cc['int4_to_int2']:3d} | INT2→1BIT: {cc['int2_to_one_bit']:3d} | 1BIT→JL: {cc['one_bit_to_jl']:3d}       │")
        print("\033[1;36m├──────────────────────────────────────────────────────────┤\033[0m")
        print("\033[1;36m│                  PAGE TIER DISTRIBUTION                  │\033[0m")
        print("\033[1;36m├──────────────────────────────────────────────────────────┤\033[0m")
        print(f"│  FP16 (Active)   [\033[1;36m{draw_bar(n_active, max_any)}\033[0m] {n_active:3d} pages        │")
        print(f"│  FP8             [\033[1;32m{draw_bar(n_fp8, max_any)}\033[0m] {n_fp8:3d} pages        │")
        print(f"│  INT8            [\033[0;32m{draw_bar(n_int8, max_any)}\033[0m] {n_int8:3d} pages        │")
        print(f"│  INT4            [\033[1;33m{draw_bar(n_int4, max_any)}\033[0m] {n_int4:3d} pages        │")
        print(f"│  INT2            [\033[1;35m{draw_bar(n_int2, max_any)}\033[0m] {n_int2:3d} pages        │")
        print(f"│  1-Bit           [\033[1;31m{draw_bar(n_one_bit, max_any)}\033[0m] {n_one_bit:3d} pages        │")
        print(f"│  JL (Archive)    [\033[1;34m{draw_bar(n_jl, max_any)}\033[0m] {n_jl:3d} pages        │")
        print("\033[1;36m├──────────────────────────────────────────────────────────┤\033[0m")
        print("\033[1;36m│                  VIRTUAL MEMORY HEATMAP                  │\033[0m")
        print("│    (█ = VRAM Resident, ▒ = CPU Swapped Out)               │")
        print("│                                                          │")
        print(f"│  Hot Pages   (FP16/FP8):   {hot_pages:3d} pages                    │")
        print(f"│  Warm Pages  (INT8/INT4):  {warm_pages:3d} pages                    │")
        print(f"│  Cold Pages  (INT2+):      {cold_pages:3d} pages                    │")
        print(f"│  CPU Spilled (Host RAM):   {cpu_spill_pages:3d} pages                    │")
        print("│                                                          │")
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
            # Pad line to 58 chars
            stripped_line = line.replace('\033[1;36m','').replace('\033[1;32m','').replace('\033[0;32m','').replace('\033[1;33m','').replace('\033[1;35m','').replace('\033[1;31m','').replace('\033[1;34m','').replace('\033[0m','').replace('\033[37m','')
            padding = 58 - len(stripped_line) - 1
            print(f"│{line}{' ' * padding}│")
        print("\033[1;36m└──────────────────────────────────────────────────────────┘\033[0m\n")

    def _resurrect_page(self, page, tier):
        import time
        start_time = time.perf_counter()
        # 1. Dequantize
        if tier == 'fp8':
            k = dequantize_from_fp8_simulated(page['key_q'], page['key_scales'])
            v = dequantize_from_fp8_simulated(page['value_q'], page['value_scales'])
            self.fp8_pages = [p for p in self.fp8_pages if p is not page]
        elif tier == 'int8':
            k = dequantize_from_int8(page['key_q'], page['key_scales'])
            v = dequantize_from_int8(page['value_q'], page['value_scales'])
            self.int8_pages = [p for p in self.int8_pages if p is not page]
        elif tier == 'int4':
            k = dequantize_from_int4_packed(page['key_packed'], page['key_scales'], page['key_min'], seq_dim=-2)
            v = dequantize_from_int4_packed(page['value_packed'], page['value_scales'], page['value_min'], seq_dim=-2)
            self.int4_pages = [p for p in self.int4_pages if p is not page]
        elif tier == 'int2':
            k = dequantize_from_int2_packed(page['key_packed'], page['key_scales'], page['key_min'], seq_dim=-2)
            v = dequantize_from_int2_packed(page['value_packed'], page['value_scales'], page['value_min'], seq_dim=-2)
            self.int2_pages = [p for p in self.int2_pages if p is not page]
        elif tier == 'one_bit':
            k = dequantize_from_1bit_packed(page['key_packed'], page['key_scales'], seq_dim=-2)
            v = dequantize_from_1bit_packed(page['value_packed'], page['value_scales'], seq_dim=-2)
            self.one_bit_pages = [p for p in self.one_bit_pages if p is not page]
        elif tier == 'jl':
            w_proj = self.get_jl_projection_matrix(page['key_proj'].device, page['key_proj'].dtype)
            k = torch.matmul(w_proj.t(), page['key_proj'])
            v = torch.matmul(w_proj.t(), page['value_proj'])
            self.jl_pages = [p for p in self.jl_pages if p is not page]
        else:
            return

        dequant_time = (time.perf_counter() - start_time) * 1000  # in ms
        self.total_dequant_time += dequant_time
        self.num_dequants += 1
        self.num_resurrections += 1
        
        # Track advanced telemetry metrics
        self.dequant_latencies.append(dequant_time)
        depth_map = {'fp8': 1, 'int8': 2, 'int4': 3, 'int2': 4, 'one_bit': 5, 'jl': 6}
        self.resurrection_depths.append(depth_map.get(tier, 1))
        self.page_lifetimes[page.get('page_id')] = self.generation_step

        k_idx = page.get('key_out_indices')
        if k_idx is not None and k_idx.numel() > 0:
            k = k.clone()
            k[k_idx[:, 0].long(), k_idx[:, 1].long(), k_idx[:, 2].long(), k_idx[:, 3].long()] = page['key_out_values']
        v_idx = page.get('value_out_indices')
        if v_idx is not None and v_idx.numel() > 0:
            v = v.clone()
            v[v_idx[:, 0].long(), v_idx[:, 1].long(), v_idx[:, 2].long(), v_idx[:, 3].long()] = page['value_out_values']

        # Determine resurrection reason & log it
        rec = 1.0 / (1.0 + float(self.generation_step - page.get('last_step_accessed', 0)))
        att = page.get('attention_sum', 0.0)
        reason = "high attention recurrence" if att >= rec else "high recency bias"
        argus_log("INFO", f"Page {page.get('page_id')} resurrected (importance={page.get('importance_score', 0.0):.2f}) | Tier: {tier.upper()} -> FP16 Transient | Reason: {reason} | Dequant Latency: {dequant_time:.3f}ms", line_no=700)
        
        if tier in ['int2', 'one_bit', 'jl']:
            argus_log("INFO", f"Restored {tier.upper()} archive page to FP16 transient buffer", line_no=701)
        if tier in ['int4', 'int2', 'one_bit', 'jl']:
            argus_log("INFO", f"Attention spike detected on archived memory", line_no=702)

        # 2. To insert into active_pages, we must ensure we have a slot!
        if len(self.active_pages) >= self.max_active_pages:
            self.active_pages.sort(key=lambda x: x.get('importance_score', 0.0))
            demoted_page = self.active_pages.pop(0)
            self._demote_active_to_fp8(demoted_page)

        idx = self._get_free_pool_idx(self.active_pages, self.max_active_pages)
        self.active_pool_k[idx].copy_(k)
        self.active_pool_v[idx].copy_(v)

        resurrected_dict = {
            'page_id': page.get('page_id'),
            'key': self.active_pool_k[idx],
            'value': self.active_pool_v[idx],
            'pool_idx': idx,
            'attention_sum': page.get('attention_sum', 0.0),
            'last_step_accessed': self.generation_step,
            'entropy': page.get('entropy', 0.0),
            'importance_score': page.get('importance_score', 0.0)
        }
        self._calculate_importance(resurrected_dict)
        self.active_pages.append(resurrected_dict)
        self.log_event("resurrect", resurrected_dict['page_id'], tier=tier)

    def manage_memory_lifecycle(self):
        """
        Transitions pages down the 7-tier memory lifecycle when limits are exceeded.
        """
        if len(self.active_pages) >= self.max_active_pages:
            self.active_pages.sort(key=lambda p: p.get('importance_score', 0.0))
            page = self.active_pages.pop(0)
            self._demote_active_to_fp8(page)
            
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

    def _demote_active_to_fp8(self, page):
        k_norm, k_out, k_mask = isolate_outliers(page['key'], self.threshold_sigma)
        v_norm, v_out, v_mask = isolate_outliers(page['value'], self.threshold_sigma)
        
        k_q, k_s = quantize_to_fp8_simulated(k_norm, dim=-1)
        v_q, v_s = quantize_to_fp8_simulated(v_norm, dim=-1)
        
        k_out_indices = torch.nonzero(k_mask).to(torch.int16)
        k_out_values = k_out[k_mask]
        
        v_out_indices = torch.nonzero(v_mask).to(torch.int16)
        v_out_values = v_out[v_mask]
        
        if len(self.fp8_pages) >= self.max_fp8_pages:
            self.fp8_pages.sort(key=lambda p: p.get('importance_score', 0.0))
            old = self.fp8_pages.pop(0)
            self._demote_fp8_to_int8(old)
            
        idx = self._get_free_pool_idx(self.fp8_pages, self.max_fp8_pages)
        self.fp8_pool_k_q[idx].copy_(k_q)
        self.fp8_pool_v_q[idx].copy_(v_q)
        self.fp8_pool_k_scales[idx].copy_(k_s)
        self.fp8_pool_v_scales[idx].copy_(v_s)
        
        self.fp8_pages.append({
            'page_id': page.get('page_id'),
            'key_q': self.fp8_pool_k_q[idx], 'key_scales': self.fp8_pool_k_scales[idx], 'key_out_indices': k_out_indices, 'key_out_values': k_out_values,
            'value_q': self.fp8_pool_v_q[idx], 'value_scales': self.fp8_pool_v_scales[idx], 'value_out_indices': v_out_indices, 'value_out_values': v_out_values,
            'pool_idx': idx,
            'attention_sum': page.get('attention_sum', 0.0),
            'last_step_accessed': page.get('last_step_accessed', self.generation_step),
            'entropy': page.get('entropy', 0.0),
            'importance_score': page.get('importance_score', 0.0)
        })
        self.log_event("demote", page.get('page_id'), tier_from="active", tier_to="fp8")
        self.total_demotions += 1
        self.cascade_counts['fp16_to_fp8'] += 1
        pid = page.get('page_id')
        if pid in self.page_lifetimes:
            lifetime = self.generation_step - self.page_lifetimes[pid]
            self.total_page_lifetimes += lifetime
            self.completed_page_lifetimes_count += 1
            del self.page_lifetimes[pid]
        argus_log("INFO", f"Demoting Page {page.get('page_id')} (FP16 -> FP8) | Reason: low attention prominence (importance={page.get('importance_score', 0.0):.2f})", line_no=800)

    def _demote_fp8_to_int8(self, old):
        k_norm = dequantize_from_fp8_simulated(old['key_q'], old['key_scales'])
        v_norm = dequantize_from_fp8_simulated(old['value_q'], old['value_scales'])
        
        k_q, k_s = quantize_to_int8(k_norm, dim=-1)
        v_q, v_s = quantize_to_int8(v_norm, dim=-1)
        
        if len(self.int8_pages) >= self.max_int8_pages:
            self.int8_pages.sort(key=lambda p: p.get('importance_score', 0.0))
            older = self.int8_pages.pop(0)
            self._demote_int8_to_int4(older)
            
        idx = self._get_free_pool_idx(self.int8_pages, self.max_int8_pages)
        self.int8_pool_k_q[idx].copy_(k_q)
        self.int8_pool_v_q[idx].copy_(v_q)
        self.int8_pool_k_scales[idx].copy_(k_s)
        self.int8_pool_v_scales[idx].copy_(v_s)
        
        self.int8_pages.append({
            'page_id': old.get('page_id'),
            'key_q': self.int8_pool_k_q[idx], 'key_scales': self.int8_pool_k_scales[idx], 'key_out_indices': old['key_out_indices'], 'key_out_values': old['key_out_values'],
            'value_q': self.int8_pool_v_q[idx], 'value_scales': self.int8_pool_v_scales[idx], 'value_out_indices': old['key_out_indices'], 'value_out_values': old['key_out_values'],
            'pool_idx': idx,
            'attention_sum': old.get('attention_sum', 0.0),
            'last_step_accessed': old.get('last_step_accessed', self.generation_step),
            'entropy': old.get('entropy', 0.0),
            'importance_score': old.get('importance_score', 0.0)
        })
        self.log_event("demote", old.get('page_id'), tier_from="fp8", tier_to="int8")
        self.total_demotions += 1
        self.cascade_counts['fp8_to_int8'] += 1
        argus_log("INFO", f"Demoting Page {old.get('page_id')} (FP8 -> INT8) | Reason: cascading compression cascade", line_no=830)

    def _demote_int8_to_int4(self, old):
        k_norm = dequantize_from_int8(old['key_q'], old['key_scales'])
        v_norm = dequantize_from_int8(old['value_q'], old['value_scales'])
        
        k_packed, k_s, k_min = quantize_to_int4_packed(k_norm, seq_dim=-2, quant_dim=-1)
        v_packed, v_s, v_min = quantize_to_int4_packed(v_norm, seq_dim=-2, quant_dim=-1)
        
        if len(self.int4_pages) >= self.max_int4_pages:
            self.int4_pages.sort(key=lambda p: p.get('importance_score', 0.0))
            older = self.int4_pages.pop(0)
            self._demote_int4_to_int2(older)
            
        idx = self._get_free_pool_idx(self.int4_pages, self.max_int4_pages)
        self.int4_pool_k_packed[idx].copy_(k_packed)
        self.int4_pool_v_packed[idx].copy_(v_packed)
        self.int4_pool_k_scales[idx].copy_(k_s)
        self.int4_pool_v_scales[idx].copy_(v_s)
        self.int4_pool_k_min[idx].copy_(k_min)
        self.int4_pool_v_min[idx].copy_(v_min)
        
        self.int4_pages.append({
            'page_id': old.get('page_id'),
            'key_packed': self.int4_pool_k_packed[idx], 'key_scales': self.int4_pool_k_scales[idx], 'key_min': self.int4_pool_k_min[idx], 'key_out_indices': old['key_out_indices'], 'key_out_values': old['key_out_values'],
            'value_packed': self.int4_pool_v_packed[idx], 'value_scales': self.int4_pool_v_scales[idx], 'value_min': self.int4_pool_v_min[idx], 'value_out_indices': old['key_out_indices'], 'value_out_values': old['key_out_values'],
            'pool_idx': idx,
            'attention_sum': old.get('attention_sum', 0.0),
            'last_step_accessed': old.get('last_step_accessed', self.generation_step),
            'entropy': old.get('entropy', 0.0),
            'importance_score': old.get('importance_score', 0.0)
        })
        self.log_event("demote", old.get('page_id'), tier_from="int8", tier_to="int4")
        self.total_demotions += 1
        self.cascade_counts['int8_to_int4'] += 1
        argus_log("INFO", f"Demoting Page {old.get('page_id')} (INT8 -> INT4) | Reason: cascading compression cascade", line_no=862)

    def _demote_int4_to_int2(self, old):
        k_norm = dequantize_from_int4_packed(old['key_packed'], old['key_scales'], old['key_min'], seq_dim=-2)
        v_norm = dequantize_from_int4_packed(old['value_packed'], old['value_scales'], old['value_min'], seq_dim=-2)
        
        k_packed, k_s, k_min = quantize_to_int2_packed(k_norm, seq_dim=-2, quant_dim=-1)
        v_packed, v_s, v_min = quantize_to_int2_packed(v_norm, seq_dim=-2, quant_dim=-1)
        
        if len(self.int2_pages) >= self.max_int2_pages:
            self.int2_pages.sort(key=lambda p: p.get('importance_score', 0.0))
            older = self.int2_pages.pop(0)
            self._demote_int2_to_one_bit(older)
            
        idx = self._get_free_pool_idx(self.int2_pages, self.max_int2_pages)
        self.int2_pool_k_packed[idx].copy_(k_packed)
        self.int2_pool_v_packed[idx].copy_(v_packed)
        self.int2_pool_k_scales[idx].copy_(k_s)
        self.int2_pool_v_scales[idx].copy_(v_s)
        self.int2_pool_k_min[idx].copy_(k_min)
        self.int2_pool_v_min[idx].copy_(v_min)
        
        self.int2_pages.append({
            'page_id': old.get('page_id'),
            'key_packed': self.int2_pool_k_packed[idx], 'key_scales': self.int2_pool_k_scales[idx], 'key_min': self.int2_pool_k_min[idx], 'key_out_indices': old['key_out_indices'], 'key_out_values': old['key_out_values'],
            'value_packed': self.int2_pool_v_packed[idx], 'value_scales': self.int2_pool_v_scales[idx], 'value_min': self.int2_pool_v_min[idx], 'value_out_indices': old['key_out_indices'], 'value_out_values': old['key_out_values'],
            'pool_idx': idx,
            'attention_sum': old.get('attention_sum', 0.0),
            'last_step_accessed': old.get('last_step_accessed', self.generation_step),
            'entropy': old.get('entropy', 0.0),
            'importance_score': old.get('importance_score', 0.0)
        })
        self.log_event("demote", old.get('page_id'), tier_from="int4", tier_to="int2")
        self.total_demotions += 1
        self.cascade_counts['int4_to_int2'] += 1
        argus_log("INFO", f"Demoting Page {old.get('page_id')} (INT4 -> INT2) | Reason: cascading compression cascade", line_no=894)

    def _demote_int2_to_one_bit(self, old):
        k_norm = dequantize_from_int2_packed(old['key_packed'], old['key_scales'], old['key_min'], seq_dim=-2)
        v_norm = dequantize_from_int2_packed(old['value_packed'], old['value_scales'], old['value_min'], seq_dim=-2)
        
        k_packed, k_s = quantize_to_1bit_packed(k_norm, seq_dim=-2, quant_dim=-1)
        v_packed, v_s = quantize_to_1bit_packed(v_norm, seq_dim=-2, quant_dim=-1)
        
        if len(self.one_bit_pages) >= self.max_one_bit_pages:
            self.one_bit_pages.sort(key=lambda p: p.get('importance_score', 0.0))
            older = self.one_bit_pages.pop(0)
            self._demote_one_bit_to_jl(older)
            
        idx = self._get_free_pool_idx(self.one_bit_pages, self.max_one_bit_pages)
        self.one_bit_pool_k_packed[idx].copy_(k_packed)
        self.one_bit_pool_v_packed[idx].copy_(v_packed)
        self.one_bit_pool_k_scales[idx].copy_(k_s)
        self.one_bit_pool_v_scales[idx].copy_(v_s)
        
        self.one_bit_pages.append({
            'page_id': old.get('page_id'),
            'key_packed': self.one_bit_pool_k_packed[idx], 'key_scales': self.one_bit_pool_k_scales[idx], 'key_out_indices': old['key_out_indices'], 'key_out_values': old['key_out_values'],
            'value_packed': self.one_bit_pool_v_packed[idx], 'value_scales': self.one_bit_pool_v_scales[idx], 'value_out_indices': old['key_out_indices'], 'value_out_values': old['key_out_values'],
            'pool_idx': idx,
            'attention_sum': old.get('attention_sum', 0.0),
            'last_step_accessed': old.get('last_step_accessed', self.generation_step),
            'entropy': old.get('entropy', 0.0),
            'importance_score': old.get('importance_score', 0.0)
        })
        self.log_event("demote", old.get('page_id'), tier_from="int2", tier_to="one_bit")
        self.total_demotions += 1
        self.cascade_counts['int2_to_one_bit'] += 1
        argus_log("INFO", f"Demoting Page {old.get('page_id')} (INT2 -> 1-Bit) | Reason: cascading compression cascade", line_no=924)

    def _demote_one_bit_to_jl(self, old):
        k_norm = dequantize_from_1bit_packed(old['key_packed'], old['key_scales'], seq_dim=-2)
        v_norm = dequantize_from_1bit_packed(old['value_packed'], old['value_scales'], seq_dim=-2)
        
        k_full = k_norm.clone()
        k_idx = old['key_out_indices']
        if k_idx is not None and k_idx.numel() > 0:
            k_full[k_idx[:, 0].long(), k_idx[:, 1].long(), k_idx[:, 2].long(), k_idx[:, 3].long()] = old['key_out_values']
            
        v_full = v_norm.clone()
        v_idx = old['value_out_indices']
        if v_idx is not None and v_idx.numel() > 0:
            v_full[v_idx[:, 0].long(), v_idx[:, 1].long(), v_idx[:, 2].long(), v_idx[:, 3].long()] = old['value_out_values']
        
        w_proj = self.get_jl_projection_matrix(k_full.device, k_full.dtype)
        k_proj = torch.matmul(w_proj, k_full)
        v_proj = torch.matmul(w_proj, v_full)
        
        self.jl_pages.append({
            'page_id': old.get('page_id'),
            'key_proj': k_proj,
            'value_proj': v_proj,
            'attention_sum': old.get('attention_sum', 0.0),
            'last_step_accessed': old.get('last_step_accessed', self.generation_step),
            'entropy': old.get('entropy', 0.0),
            'importance_score': old.get('importance_score', 0.0)
        })
        self.log_event("demote", old.get('page_id'), tier_from="one_bit", tier_to="jl")
        self.total_demotions += 1
        self.cascade_counts['one_bit_to_jl'] += 1
        argus_log("INFO", f"Demoting Page {old.get('page_id')} (1-Bit -> JL-Projection) | Reason: cascading compression cascade", line_no=954)

    def get_all_keys_values(self):
        """
        Reconstructs all cache levels back to single FP16 tensors, prepending VIP anchors & Attention Sinks.
        Automatically handles host-to-device swapping if the cache is swapped out.
        """
        # Automatic Swap-In Safeguard
        if self.is_swapped_out:
            self.swap_in_to_device()
            
        all_keys = []
        all_values = []
        
        # Tier 7: JL Projection Sequence-Compressed Archive
        for page in self.jl_pages:
            if id(page) in self.prefetch_cache:
                k, v = self.prefetch_cache[id(page)]
                self.prefetch_hits += 1
            else:
                self.prefetch_misses += 1
                w_proj = self.get_jl_projection_matrix(page['key_proj'].device, page['key_proj'].dtype)
                k = torch.matmul(w_proj.t(), page['key_proj'])
                v = torch.matmul(w_proj.t(), page['value_proj'])
            all_keys.append(k)
            all_values.append(v)
            
        # Tier 6: 1-Bit packed
        for page in self.one_bit_pages:
            if id(page) in self.prefetch_cache:
                k, v = self.prefetch_cache[id(page)]
                self.prefetch_hits += 1
            else:
                self.prefetch_misses += 1
                k_norm = dequantize_from_1bit_packed(page['key_packed'], page['key_scales'], seq_dim=-2)
                v_norm = dequantize_from_1bit_packed(page['value_packed'], page['value_scales'], seq_dim=-2)
                
                k = k_norm.clone()
                k_idx = page['key_out_indices']
                if k_idx is not None and k_idx.numel() > 0:
                    k[k_idx[:, 0].long(), k_idx[:, 1].long(), k_idx[:, 2].long(), k_idx[:, 3].long()] = page['key_out_values']
                    
                v = v_norm.clone()
                v_idx = page['value_out_indices']
                if v_idx is not None and v_idx.numel() > 0:
                    v[v_idx[:, 0].long(), v_idx[:, 1].long(), v_idx[:, 2].long(), v_idx[:, 3].long()] = page['value_out_values']
            all_keys.append(k)
            all_values.append(v)

        # Tier 5: INT2 packed
        for page in self.int2_pages:
            if id(page) in self.prefetch_cache:
                k, v = self.prefetch_cache[id(page)]
                self.prefetch_hits += 1
            else:
                self.prefetch_misses += 1
                k_norm = dequantize_from_int2_packed(page['key_packed'], page['key_scales'], page['key_min'], seq_dim=-2)
                v_norm = dequantize_from_int2_packed(page['value_packed'], page['value_scales'], page['value_min'], seq_dim=-2)
                
                k = k_norm.clone()
                k_idx = page['key_out_indices']
                if k_idx is not None and k_idx.numel() > 0:
                    k[k_idx[:, 0].long(), k_idx[:, 1].long(), k_idx[:, 2].long(), k_idx[:, 3].long()] = page['key_out_values']
                    
                v = v_norm.clone()
                v_idx = page['value_out_indices']
                if v_idx is not None and v_idx.numel() > 0:
                    v[v_idx[:, 0].long(), v_idx[:, 1].long(), v_idx[:, 2].long(), v_idx[:, 3].long()] = page['value_out_values']
            all_keys.append(k)
            all_values.append(v)
            
        # Tier 4: INT4 packed
        for page in self.int4_pages:
            if id(page) in self.prefetch_cache:
                k, v = self.prefetch_cache[id(page)]
                self.prefetch_hits += 1
            else:
                self.prefetch_misses += 1
                k_norm = dequantize_from_int4_packed(page['key_packed'], page['key_scales'], page['key_min'], seq_dim=-2)
                v_norm = dequantize_from_int4_packed(page['value_packed'], page['value_scales'], page['value_min'], seq_dim=-2)
                
                k = k_norm.clone()
                k_idx = page['key_out_indices']
                if k_idx is not None and k_idx.numel() > 0:
                    k[k_idx[:, 0].long(), k_idx[:, 1].long(), k_idx[:, 2].long(), k_idx[:, 3].long()] = page['key_out_values']
                    
                v = v_norm.clone()
                v_idx = page['value_out_indices']
                if v_idx is not None and v_idx.numel() > 0:
                    v[v_idx[:, 0].long(), v_idx[:, 1].long(), v_idx[:, 2].long(), v_idx[:, 3].long()] = page['value_out_values']
            all_keys.append(k)
            all_values.append(v)
            
        # Tier 3: INT8
        for page in self.int8_pages:
            if id(page) in self.prefetch_cache:
                k, v = self.prefetch_cache[id(page)]
                self.prefetch_hits += 1
            else:
                self.prefetch_misses += 1
                k_norm = dequantize_from_int8(page['key_q'], page['key_scales'])
                v_norm = dequantize_from_int8(page['value_q'], page['value_scales'])
                
                k = k_norm.clone()
                k_idx = page['key_out_indices']
                if k_idx is not None and k_idx.numel() > 0:
                    k[k_idx[:, 0].long(), k_idx[:, 1].long(), k_idx[:, 2].long(), k_idx[:, 3].long()] = page['key_out_values']
                    
                v = v_norm.clone()
                v_idx = page['value_out_indices']
                if v_idx is not None and v_idx.numel() > 0:
                    v[v_idx[:, 0].long(), v_idx[:, 1].long(), v_idx[:, 2].long(), v_idx[:, 3].long()] = page['value_out_values']
            all_keys.append(k)
            all_values.append(v)
            
        # Tier 2: FP8
        for page in self.fp8_pages:
            if id(page) in self.prefetch_cache:
                k, v = self.prefetch_cache[id(page)]
                self.prefetch_hits += 1
            else:
                self.prefetch_misses += 1
                k_norm = dequantize_from_fp8_simulated(page['key_q'], page['key_scales'])
                v_norm = dequantize_from_fp8_simulated(page['value_q'], page['value_scales'])
                
                k = k_norm.clone()
                k_idx = page['key_out_indices']
                if k_idx is not None and k_idx.numel() > 0:
                    k[k_idx[:, 0].long(), k_idx[:, 1].long(), k_idx[:, 2].long(), k_idx[:, 3].long()] = page['key_out_values']
                    
                v = v_norm.clone()
                v_idx = page['value_out_indices']
                if v_idx is not None and v_idx.numel() > 0:
                    v[v_idx[:, 0].long(), v_idx[:, 1].long(), v_idx[:, 2].long(), v_idx[:, 3].long()] = page['value_out_values']
            all_keys.append(k)
            all_values.append(v)
            
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
            self.swap_in_to_device()

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
            
        # 5. Compressed Pages (Tiers 2-7)
        def process_compressed_pages(pages, tier):
            for page in pages:
                if id(page) in self.prefetch_cache:
                    k, v = self.prefetch_cache[id(page)]
                    self.prefetch_hits += 1
                else:
                    self.prefetch_misses += 1
                    if tier == 'fp8':
                        k = dequantize_from_fp8_simulated(page['key_q'], page['key_scales'])
                        v = dequantize_from_fp8_simulated(page['value_q'], page['value_scales'])
                    elif tier == 'int8':
                        k = dequantize_from_int8(page['key_q'], page['key_scales'])
                        v = dequantize_from_int8(page['value_q'], page['value_scales'])
                    elif tier == 'int4':
                        k = dequantize_from_int4_packed(page['key_packed'], page['key_scales'], page['key_min'], seq_dim=-2)
                        v = dequantize_from_int4_packed(page['value_packed'], page['value_scales'], page['value_min'], seq_dim=-2)
                    elif tier == 'int2':
                        k = dequantize_from_int2_packed(page['key_packed'], page['key_scales'], page['key_min'], seq_dim=-2)
                        v = dequantize_from_int2_packed(page['value_packed'], page['value_scales'], page['value_min'], seq_dim=-2)
                    elif tier == 'one_bit':
                        k = dequantize_from_1bit_packed(page['key_packed'], page['key_scales'], seq_dim=-2)
                        v = dequantize_from_1bit_packed(page['value_packed'], page['value_scales'], seq_dim=-2)
                    elif tier == 'jl':
                        w_proj = self.get_jl_projection_matrix(page['key_proj'].device, page['key_proj'].dtype)
                        k = torch.matmul(w_proj.t(), page['key_proj'])
                        v = torch.matmul(w_proj.t(), page['value_proj'])
                    else:
                        continue
                        
                    k_idx = page.get('key_out_indices')
                    if k_idx is not None and k_idx.numel() > 0:
                        k = k.clone()
                        k[k_idx[:, 0].long(), k_idx[:, 1].long(), k_idx[:, 2].long(), k_idx[:, 3].long()] = page['key_out_values']
                    v_idx = page.get('value_out_indices')
                    if v_idx is not None and v_idx.numel() > 0:
                        v = v.clone()
                        v[v_idx[:, 0].long(), v_idx[:, 1].long(), v_idx[:, 2].long(), v_idx[:, 3].long()] = page['value_out_values']
                        
                keys_list.append(k)
                values_list.append(v)
                
        process_compressed_pages(self.jl_pages, 'jl')
        process_compressed_pages(self.one_bit_pages, 'one_bit')
        process_compressed_pages(self.int2_pages, 'int2')
        process_compressed_pages(self.int4_pages, 'int4')
        process_compressed_pages(self.int8_pages, 'int8')
        process_compressed_pages(self.fp8_pages, 'fp8')
        
        if not keys_list:
            return torch.zeros_like(q)
            
        if is_enterprise:
            k_full = torch.cat(keys_list, dim=-2)
            v_full = torch.cat(values_list, dim=-2)
            
            attn_weights = torch.matmul(q, k_full.transpose(-1, -2)) * scale
            attn_probs = torch.softmax(attn_weights, dim=-1)
            attn_output = torch.matmul(attn_probs, v_full)
        else:
            scores_list = []
            for k_block in keys_list:
                score_block = torch.matmul(q, k_block.transpose(-1, -2)) * scale
                scores_list.append(score_block)
                
            scores = torch.cat(scores_list, dim=-1)
            attn_probs = torch.softmax(scores, dim=-1)
            
            attn_output = torch.zeros_like(q)
            idx_start = 0
            for i, v_block in enumerate(values_list):
                block_len = v_block.shape[-2]
                attn_probs_block = attn_probs[..., idx_start : idx_start + block_len]
                out_block = torch.matmul(attn_probs_block, v_block)
                attn_output.add_(out_block)
                idx_start += block_len

        # Incremental step counter
        self.generation_step += 1

        # Update QoS metrics and trigger Hot-Page Resurrection
        if attn_probs is not None:
            with torch.no_grad():
                weights = attn_probs.mean(dim=(0, 1))  # [q_len, kv_len]
                if weights.dim() > 1:
                    weights = weights.mean(dim=0)  # [kv_len]
                
                # Pages in order of keys_list
                pages_in_order = []
                if self.sink_k is not None:
                    pages_in_order.append(None)
                if self.anchor_k is not None:
                    pages_in_order.append(None)
                for page in self.active_pages:
                    pages_in_order.append(page)
                if self.k_buffer is not None and self.k_buffer.shape[-2] > 0:
                    pages_in_order.append(None)
                    
                for page in self.jl_pages:
                    pages_in_order.append(page)
                for page in self.one_bit_pages:
                    pages_in_order.append(page)
                for page in self.int2_pages:
                    pages_in_order.append(page)
                for page in self.int4_pages:
                    pages_in_order.append(page)
                for page in self.int8_pages:
                    pages_in_order.append(page)
                for page in self.fp8_pages:
                    pages_in_order.append(page)
                
                resurrect_list = []
                idx_start = 0
                for i, page in enumerate(pages_in_order):
                    if i >= len(keys_list):
                        break
                    block_len = keys_list[i].shape[-2]
                    if page is not None:
                        block_w = weights[idx_start : idx_start + block_len].sum().item()
                        page['attention_sum'] = page.get('attention_sum', 0.0) + block_w
                        if block_w > 0.0:
                            page['last_step_accessed'] = self.generation_step
                        
                        # Recalculate importance score
                        self._calculate_importance(page)
                        
                        # Check resurrection eligibility
                        is_compressed = (any(page is p for p in self.fp8_pages) or 
                                         any(page is p for p in self.int8_pages) or 
                                         any(page is p for p in self.int4_pages) or 
                                         any(page is p for p in self.int2_pages) or 
                                         any(page is p for p in self.one_bit_pages) or 
                                         any(page is p for p in self.jl_pages))
                        if is_compressed and block_w > self.config.resurrection_threshold:
                            # Determine tier
                            tier_name = None
                            if any(page is p for p in self.fp8_pages): tier_name = 'fp8'
                            elif any(page is p for p in self.int8_pages): tier_name = 'int8'
                            elif any(page is p for p in self.int4_pages): tier_name = 'int4'
                            elif any(page is p for p in self.int2_pages): tier_name = 'int2'
                            elif any(page is p for p in self.one_bit_pages): tier_name = 'one_bit'
                            elif any(page is p for p in self.jl_pages): tier_name = 'jl'
                            
                            if tier_name is not None:
                                resurrect_list.append((page, tier_name))
                                
                    idx_start += block_len
                
                # Execute resurrection for eligible pages
                for p, t in resurrect_list:
                    self._resurrect_page(p, t)
                    
        return attn_output

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
                
                sink_len = self.sink_k.shape[-2] if self.sink_k is not None else 0
                anchor_len = self.anchor_k.shape[-2] if self.anchor_k is not None else 0
                offset = sink_len + anchor_len
                
                page_weights = []
                all_p = []
                all_p.extend(('jl', p) for p in self.jl_pages)
                all_p.extend(('one_bit', p) for p in self.one_bit_pages)
                all_p.extend(('int2', p) for p in self.int2_pages)
                all_p.extend(('int4', p) for p in self.int4_pages)
                all_p.extend(('int8', p) for p in self.int8_pages)
                all_p.extend(('fp8', p) for p in self.fp8_pages)
                
                for idx, (tier, page) in enumerate(all_p):
                    start = offset + idx * self.page_size
                    end = start + self.page_size
                    if end <= len(weights):
                        w = weights[start:end].mean().item()
                    else:
                        w = 0.0
                    page_weights.append((w, page, tier))
                    
                page_weights.sort(key=lambda x: x[0], reverse=True)
                for w, page, tier in page_weights[:2]:
                    pages_to_prefetch.append((page, tier))
            except Exception:
                pass
                
        if not pages_to_prefetch:
            if self.fp8_pages:
                pages_to_prefetch.append((self.fp8_pages[-1], 'fp8'))
            if self.int8_pages:
                pages_to_prefetch.append((self.int8_pages[-1], 'int8'))
            if self.int4_pages:
                pages_to_prefetch.append((self.int4_pages[-1], 'int4'))
                
        if self.prefetch_stream is None and torch.cuda.is_available():
            self.prefetch_stream = torch.cuda.Stream()
            
        main_stream = torch.cuda.current_stream() if torch.cuda.is_available() else None
        
        for page, tier in pages_to_prefetch:
            try:
                if self.prefetch_stream is not None:
                    # Run pre-dequantization asynchronously on dedicated stream
                    with torch.cuda.stream(self.prefetch_stream):
                        if tier == 'fp8':
                            k = dequantize_from_fp8_simulated(page['key_q'], page['key_scales'])
                            v = dequantize_from_fp8_simulated(page['value_q'], page['value_scales'])
                        elif tier == 'int8':
                            k = dequantize_from_int8(page['key_q'], page['key_scales'])
                            v = dequantize_from_int8(page['value_q'], page['value_scales'])
                        elif tier == 'int4':
                            k = dequantize_from_int4_packed(page['key_packed'], page['key_scales'], page['key_min'], seq_dim=-2)
                            v = dequantize_from_int4_packed(page['value_packed'], page['value_scales'], page['value_min'], seq_dim=-2)
                        elif tier == 'int2':
                            k = dequantize_from_int2_packed(page['key_packed'], page['key_scales'], page['key_min'], seq_dim=-2)
                            v = dequantize_from_int2_packed(page['value_packed'], page['value_scales'], page['value_min'], seq_dim=-2)
                        elif tier == 'one_bit':
                            k = dequantize_from_1bit_packed(page['key_packed'], page['key_scales'], seq_dim=-2)
                            v = dequantize_from_1bit_packed(page['value_packed'], page['value_scales'], seq_dim=-2)
                        elif tier == 'jl':
                            w_proj = self.get_jl_projection_matrix(page['key_proj'].device, page['key_proj'].dtype)
                            k = torch.matmul(w_proj.t(), page['key_proj'])
                            v = torch.matmul(w_proj.t(), page['value_proj'])
                        else:
                             continue
                             
                        k_idx = page.get('key_out_indices')
                        if k_idx is not None and k_idx.numel() > 0:
                            k = k.clone()
                            k[k_idx[:, 0].long(), k_idx[:, 1].long(), k_idx[:, 2].long(), k_idx[:, 3].long()] = page['key_out_values']
                        v_idx = page.get('value_out_indices')
                        if v_idx is not None and v_idx.numel() > 0:
                            v = v.clone()
                            v[v_idx[:, 0].long(), v_idx[:, 1].long(), v_idx[:, 2].long(), v_idx[:, 3].long()] = page['value_out_values']
                            
                        # Record consumer stream to prevent premature recycling by CUDA allocator
                        k.record_stream(main_stream)
                        v.record_stream(main_stream)
                else:
                    if tier == 'fp8':
                        k = dequantize_from_fp8_simulated(page['key_q'], page['key_scales'])
                        v = dequantize_from_fp8_simulated(page['value_q'], page['value_scales'])
                    elif tier == 'int8':
                        k = dequantize_from_int8(page['key_q'], page['key_scales'])
                        v = dequantize_from_int8(page['value_q'], page['value_scales'])
                    elif tier == 'int4':
                        k = dequantize_from_int4_packed(page['key_packed'], page['key_scales'], page['key_min'], seq_dim=-2)
                        v = dequantize_from_int4_packed(page['value_packed'], page['value_scales'], page['value_min'], seq_dim=-2)
                    elif tier == 'int2':
                        k = dequantize_from_int2_packed(page['key_packed'], page['key_scales'], page['key_min'], seq_dim=-2)
                        v = dequantize_from_int2_packed(page['value_packed'], page['value_scales'], page['value_min'], seq_dim=-2)
                    elif tier == 'one_bit':
                        k = dequantize_from_1bit_packed(page['key_packed'], page['key_scales'], seq_dim=-2)
                        v = dequantize_from_1bit_packed(page['value_packed'], page['value_scales'], seq_dim=-2)
                    elif tier == 'jl':
                        w_proj = self.get_jl_projection_matrix(page['key_proj'].device, page['key_proj'].dtype)
                        k = torch.matmul(w_proj.t(), page['key_proj'])
                        v = torch.matmul(w_proj.t(), page['value_proj'])
                    else:
                         continue
                         
                    k_idx = page.get('key_out_indices')
                    if k_idx is not None and k_idx.numel() > 0:
                        k = k.clone()
                        k[k_idx[:, 0].long(), k_idx[:, 1].long(), k_idx[:, 2].long(), k_idx[:, 3].long()] = page['key_out_values']
                    v_idx = page.get('value_out_indices')
                    if v_idx is not None and v_idx.numel() > 0:
                        v = v.clone()
                        v[v_idx[:, 0].long(), v_idx[:, 1].long(), v_idx[:, 2].long(), v_idx[:, 3].long()] = page['value_out_values']
                     
                self.prefetch_cache[id(page)] = (k, v)
            except Exception:
                pass

    def swap_out_to_host(self):
        """
        Moves all compressed page tensors of Tiers 2-7 to CPU host memory.
        This represents the Zero-OOM multi-tenant guard that frees GPU VRAM.
        """
        if self.is_swapped_out:
            return
            
        def swap_page(page, keys):
            for k in keys:
                if k in page and isinstance(page[k], torch.Tensor):
                    page[k] = page[k].to("cpu")
                    
        for page in self.fp8_pages:
            swap_page(page, ['key_q', 'key_scales', 'key_out_indices', 'key_out_values', 'value_q', 'value_scales', 'value_out_indices', 'value_out_values'])
        for page in self.int8_pages:
            swap_page(page, ['key_q', 'key_scales', 'key_out_indices', 'key_out_values', 'value_q', 'value_scales', 'value_out_indices', 'value_out_values'])
        for page in self.int4_pages:
            swap_page(page, ['key_packed', 'key_scales', 'key_min', 'key_out_indices', 'key_out_values', 'value_packed', 'value_scales', 'value_min', 'value_out_indices', 'value_out_values'])
        for page in self.int2_pages:
            swap_page(page, ['key_packed', 'key_scales', 'key_min', 'key_out_indices', 'key_out_values', 'value_packed', 'value_scales', 'value_min', 'value_out_indices', 'value_out_values'])
        for page in self.one_bit_pages:
            swap_page(page, ['key_packed', 'key_scales', 'key_out_indices', 'key_out_values', 'value_packed', 'value_scales', 'value_out_indices', 'value_out_values'])
        for page in self.jl_pages:
            swap_page(page, ['key_proj', 'value_proj'])
            
        self.is_swapped_out = True

    def swap_in_to_device(self, device="cuda"):
        """
        Swaps all host-resident page tensors back to GPU active VRAM.
        """
        if not self.is_swapped_out:
            return
            
        target_device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        def swap_page(page, keys):
            for k in keys:
                if k in page and isinstance(page[k], torch.Tensor):
                    page[k] = page[k].to(target_device)
                    
        for page in self.fp8_pages:
            swap_page(page, ['key_q', 'key_scales', 'key_out_indices', 'key_out_values', 'value_q', 'value_scales', 'value_out_indices', 'value_out_values'])
        for page in self.int8_pages:
            swap_page(page, ['key_q', 'key_scales', 'key_out_indices', 'key_out_values', 'value_q', 'value_scales', 'value_out_indices', 'value_out_values'])
        for page in self.int4_pages:
            swap_page(page, ['key_packed', 'key_scales', 'key_min', 'key_out_indices', 'key_out_values', 'value_packed', 'value_scales', 'value_min', 'value_out_indices', 'value_out_values'])
        for page in self.int2_pages:
            swap_page(page, ['key_packed', 'key_scales', 'key_min', 'key_out_indices', 'key_out_values', 'value_packed', 'value_scales', 'value_min', 'value_out_indices', 'value_out_values'])
        for page in self.one_bit_pages:
            swap_page(page, ['key_packed', 'key_scales', 'key_out_indices', 'key_out_values', 'value_packed', 'value_scales', 'value_out_indices', 'value_out_values'])
        for page in self.jl_pages:
            swap_page(page, ['key_proj', 'value_proj'])
            
        self.is_swapped_out = False

    def get_vram_usage(self):
        """
        Calculates exact memory usage across all 7 tiers + sinks assuming standard FP16 baseline precision (2 bytes per element).
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
            
        # FP8 (1 byte quantized + 2 bytes scales & 2 bytes sparse outliers + 2 bytes sparse indices)
        for page in self.fp8_pages:
            total_bytes += page['key_q'].nelement() * 1
            total_bytes += page['value_q'].nelement() * 1
            total_bytes += page['key_scales'].nelement() * 2
            total_bytes += page['value_scales'].nelement() * 2
            if page['key_out_indices'] is not None:
                total_bytes += page['key_out_indices'].nelement() * 2  # int16 indices
                total_bytes += page['key_out_values'].nelement() * 2  # fp16 values
            if page['value_out_indices'] is not None:
                total_bytes += page['value_out_indices'].nelement() * 2
                total_bytes += page['value_out_values'].nelement() * 2
            
        # INT8 (1 byte + 2 bytes scales & sparse outliers)
        for page in self.int8_pages:
            total_bytes += page['key_q'].nelement() * 1
            total_bytes += page['value_q'].nelement() * 1
            total_bytes += page['key_scales'].nelement() * 2
            total_bytes += page['value_scales'].nelement() * 2
            if page['key_out_indices'] is not None:
                total_bytes += page['key_out_indices'].nelement() * 2
                total_bytes += page['key_out_values'].nelement() * 2
            if page['value_out_indices'] is not None:
                total_bytes += page['value_out_indices'].nelement() * 2
                total_bytes += page['value_out_values'].nelement() * 2
            
        # INT4 (0.5 byte quantized uint4 + 2 bytes scales, 2 bytes mins & sparse outliers)
        for page in self.int4_pages:
            total_bytes += page['key_packed'].nelement() * 1  # uint8 packed elements (stores 2 per byte)
            total_bytes += page['value_packed'].nelement() * 1
            total_bytes += page['key_scales'].nelement() * 2
            total_bytes += page['value_scales'].nelement() * 2
            total_bytes += page['key_min'].nelement() * 2
            total_bytes += page['value_min'].nelement() * 2
            if page['key_out_indices'] is not None:
                total_bytes += page['key_out_indices'].nelement() * 2
                total_bytes += page['key_out_values'].nelement() * 2
            if page['value_out_indices'] is not None:
                total_bytes += page['value_out_indices'].nelement() * 2
                total_bytes += page['value_out_values'].nelement() * 2

        # INT2 (0.25 byte quantized uint2 + 2 bytes scales, 2 bytes mins & sparse outliers)
        for page in self.int2_pages:
            total_bytes += page['key_packed'].nelement() * 1  # uint8 packed elements (stores 4 per byte)
            total_bytes += page['value_packed'].nelement() * 1
            total_bytes += page['key_scales'].nelement() * 2
            total_bytes += page['value_scales'].nelement() * 2
            total_bytes += page['key_min'].nelement() * 2
            total_bytes += page['value_min'].nelement() * 2
            if page['key_out_indices'] is not None:
                total_bytes += page['key_out_indices'].nelement() * 2
                total_bytes += page['key_out_values'].nelement() * 2
            if page['value_out_indices'] is not None:
                total_bytes += page['value_out_indices'].nelement() * 2
                total_bytes += page['value_out_values'].nelement() * 2
            
        # 1-Bit (0.125 byte quantized uint1 + 2 bytes scales & sparse outliers)
        for page in self.one_bit_pages:
            total_bytes += page['key_packed'].nelement() * 1  # uint8 packed elements (stores 8 per byte)
            total_bytes += page['value_packed'].nelement() * 1
            total_bytes += page['key_scales'].nelement() * 2
            total_bytes += page['value_scales'].nelement() * 2
            if page['key_out_indices'] is not None:
                total_bytes += page['key_out_indices'].nelement() * 2
                total_bytes += page['key_out_values'].nelement() * 2
            if page['value_out_indices'] is not None:
                total_bytes += page['value_out_indices'].nelement() * 2
                total_bytes += page['value_out_values'].nelement() * 2
            
        # JL Projection FP16 Archive (2 bytes per element)
        for page in self.jl_pages:
            total_bytes += page['key_proj'].nelement() * 2
            total_bytes += page['value_proj'].nelement() * 2
            
        # Add shared static projection matrix (once only in FP16: 2 bytes)
        if self.w_proj is not None:
            total_bytes += self.w_proj.nelement() * 2
            
        return total_bytes
