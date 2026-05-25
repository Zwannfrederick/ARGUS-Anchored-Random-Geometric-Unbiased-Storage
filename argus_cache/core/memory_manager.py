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

def isolate_outliers(tensor, threshold_sigma=3.0):
    """
    Isolates extreme value outliers globally using a hybrid 2-way statistical filter:
    1. Channel-wise (dim=-1) - protects activation channel spikes (LLM.int8 style).
    2. Sequence-wise (dim=-2) - protects token-level semantic anchors (Passkey Retrieval).
    Outliers are kept in FP16 permanently, while normal values are returned separately.
    """
    if threshold_sigma <= 0:
        # Outlier isolation disabled
        return tensor, torch.zeros_like(tensor), torch.zeros_like(tensor, dtype=torch.bool)
        
    abs_t = torch.abs(tensor)
    
    # 1. Channel-wise outliers (activation spikes)
    mean_ch = torch.mean(abs_t, dim=-1, keepdim=True)
    std_ch = torch.std(abs_t, dim=-1, keepdim=True).nan_to_num(0.0)
    mask_ch = abs_t > (mean_ch + threshold_sigma * std_ch)
    
    # 2. Sequence-wise outliers (token anchors/passkey)
    mean_seq = torch.mean(abs_t, dim=-2, keepdim=True)
    std_seq = torch.std(abs_t, dim=-2, keepdim=True).nan_to_num(0.0)
    mask_seq = abs_t > (mean_seq + threshold_sigma * std_seq)
    
    # Combine masks
    outlier_mask = mask_ch | mask_seq
    
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
        threshold_sigma=3.0
    ):
        """
        Generic Outlier-Aware 7-Tier Token-Based Paged Dynamic KV Cache Manager.
        Memory Lifecycle:
            FP16 (Sinks) + FP16 (Active) -> FP8 (Light) -> INT8 (Medium) -> INT4 (Heavy) -> INT2 (Super Heavy) -> 1-Bit (Sign) -> JL Projection (Archive)
        
        Args:
            page_size (int): Number of tokens per page. Must be a multiple of 8.
            max_active_pages (int): Max active FP16 pages.
            max_fp8_pages (int): Max FP8 pages.
            max_int8_pages (int): Max INT8 pages.
            max_int4_pages (int): Max INT4 packed pages.
            max_int2_pages (int): Max INT2 packed pages.
            max_one_bit_pages (int): Max 1-Bit packed pages.
            sink_tokens (int): Number of initial tokens (Attention Sinks) kept in FP16 permanently.
            threshold_sigma (float): Std dev multiplier for dynamic outlier isolation.
        """
        assert page_size % 8 == 0, "Page size must be a multiple of 8 for 1-bit packing."
        self.page_size = page_size
        self.max_active_pages = max_active_pages
        self.max_fp8_pages = max_fp8_pages
        self.max_int8_pages = max_int8_pages
        self.max_int4_pages = max_int4_pages
        self.max_int2_pages = max_int2_pages
        self.max_one_bit_pages = max_one_bit_pages
        
        self.threshold_sigma = threshold_sigma
        
        # Outlier-Aware: Attention Sinks (First N tokens kept in FP16 permanently)
        self.sink_tokens = sink_tokens
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
            idx = self.active_pool_idx
            self.active_pool_k[idx].copy_(self.static_k_buffer[..., :self.page_size, :])
            self.active_pool_v[idx].copy_(self.static_v_buffer[..., :self.page_size, :])
            
            # Shift remaining tokens in static buffer
            remaining = self.buffer_length - self.page_size
            if remaining > 0:
                self.static_k_buffer[..., :remaining, :].copy_(self.static_k_buffer[..., self.page_size : self.page_size + remaining, :])
                self.static_v_buffer[..., :remaining, :].copy_(self.static_v_buffer[..., self.page_size : self.page_size + remaining, :])
            self.buffer_length = remaining
            
            self.active_pages.append({
                'key': self.active_pool_k[idx],
                'value': self.active_pool_v[idx],
                'pool_idx': idx
            })
            
            self.active_pool_idx = (self.active_pool_idx + 1) % self.max_active_pages
            
            self.manage_memory_lifecycle()

    def manage_memory_lifecycle(self):
        """
        Transitions pages down the 7-tier memory lifecycle when limits are exceeded.
        """
        # 1. FP16 -> FP8 (simulated) with dynamic outlier isolation
        if len(self.active_pages) > self.max_active_pages:
            page = self.active_pages.pop(0)
            
            k_norm, k_out, k_mask = isolate_outliers(page['key'], self.threshold_sigma)
            v_norm, v_out, v_mask = isolate_outliers(page['value'], self.threshold_sigma)
            
            k_q, k_s = quantize_to_fp8_simulated(k_norm, dim=-1)
            v_q, v_s = quantize_to_fp8_simulated(v_norm, dim=-1)
            
            # Store outliers in highly memory-efficient sparse coordinate formats (int16 indices + fp16 values)
            k_out_indices = torch.nonzero(k_mask).to(torch.int16)
            k_out_values = k_out[k_mask]
            
            v_out_indices = torch.nonzero(v_mask).to(torch.int16)
            v_out_values = v_out[v_mask]
            
            idx = self.fp8_pool_idx
            self.fp8_pool_k_q[idx].copy_(k_q)
            self.fp8_pool_v_q[idx].copy_(v_q)
            self.fp8_pool_k_scales[idx].copy_(k_s)
            self.fp8_pool_v_scales[idx].copy_(v_s)
            
            self.fp8_pages.append({
                'key_q': self.fp8_pool_k_q[idx], 'key_scales': self.fp8_pool_k_scales[idx], 'key_out_indices': k_out_indices, 'key_out_values': k_out_values,
                'value_q': self.fp8_pool_v_q[idx], 'value_scales': self.fp8_pool_v_scales[idx], 'value_out_indices': v_out_indices, 'value_out_values': v_out_values,
                'pool_idx': idx
            })
            self.fp8_pool_idx = (self.fp8_pool_idx + 1) % self.max_fp8_pages
            
        # 2. FP8 -> INT8
        if len(self.fp8_pages) > self.max_fp8_pages:
            old = self.fp8_pages.pop(0)
            k_norm = dequantize_from_fp8_simulated(old['key_q'], old['key_scales'])
            v_norm = dequantize_from_fp8_simulated(old['value_q'], old['value_scales'])
            
            k_q, k_s = quantize_to_int8(k_norm, dim=-1)
            v_q, v_s = quantize_to_int8(v_norm, dim=-1)
            
            idx = self.int8_pool_idx
            self.int8_pool_k_q[idx].copy_(k_q)
            self.int8_pool_v_q[idx].copy_(v_q)
            self.int8_pool_k_scales[idx].copy_(k_s)
            self.int8_pool_v_scales[idx].copy_(v_s)
            
            self.int8_pages.append({
                'key_q': self.int8_pool_k_q[idx], 'key_scales': self.int8_pool_k_scales[idx], 'key_out_indices': old['key_out_indices'], 'key_out_values': old['key_out_values'],
                'value_q': self.int8_pool_v_q[idx], 'value_scales': self.int8_pool_v_scales[idx], 'value_out_indices': old['value_out_indices'], 'value_out_values': old['value_out_values'],
                'pool_idx': idx
            })
            self.int8_pool_idx = (self.int8_pool_idx + 1) % self.max_int8_pages
            
        # 3. INT8 -> INT4 (Packed 2 per byte)
        if len(self.int8_pages) > self.max_int8_pages:
            old = self.int8_pages.pop(0)
            k_norm = dequantize_from_int8(old['key_q'], old['key_scales'])
            v_norm = dequantize_from_int8(old['value_q'], old['value_scales'])
            
            k_packed, k_s, k_min = quantize_to_int4_packed(k_norm, seq_dim=-2, quant_dim=-1)
            v_packed, v_s, v_min = quantize_to_int4_packed(v_norm, seq_dim=-2, quant_dim=-1)
            
            idx = self.int4_pool_idx
            self.int4_pool_k_packed[idx].copy_(k_packed)
            self.int4_pool_v_packed[idx].copy_(v_packed)
            self.int4_pool_k_scales[idx].copy_(k_s)
            self.int4_pool_v_scales[idx].copy_(v_s)
            self.int4_pool_k_min[idx].copy_(k_min)
            self.int4_pool_v_min[idx].copy_(v_min)
            
            self.int4_pages.append({
                'key_packed': self.int4_pool_k_packed[idx], 'key_scales': self.int4_pool_k_scales[idx], 'key_min': self.int4_pool_k_min[idx], 'key_out_indices': old['key_out_indices'], 'key_out_values': old['key_out_values'],
                'value_packed': self.int4_pool_v_packed[idx], 'value_scales': self.int4_pool_v_scales[idx], 'value_min': self.int4_pool_v_min[idx], 'value_out_indices': old['value_out_indices'], 'value_out_values': old['value_out_values'],
                'pool_idx': idx
            })
            self.int4_pool_idx = (self.int4_pool_idx + 1) % self.max_int4_pages
            
        # 4. INT4 -> INT2 (Packed 4 per byte)
        if len(self.int4_pages) > self.max_int4_pages:
            old = self.int4_pages.pop(0)
            k_norm = dequantize_from_int4_packed(old['key_packed'], old['key_scales'], old['key_min'], seq_dim=-2)
            v_norm = dequantize_from_int4_packed(old['value_packed'], old['value_scales'], old['value_min'], seq_dim=-2)
            
            k_packed, k_s, k_min = quantize_to_int2_packed(k_norm, seq_dim=-2, quant_dim=-1)
            v_packed, v_s, v_min = quantize_to_int2_packed(v_norm, seq_dim=-2, quant_dim=-1)
            
            idx = self.int2_pool_idx
            self.int2_pool_k_packed[idx].copy_(k_packed)
            self.int2_pool_v_packed[idx].copy_(v_packed)
            self.int2_pool_k_scales[idx].copy_(k_s)
            self.int2_pool_v_scales[idx].copy_(v_s)
            self.int2_pool_k_min[idx].copy_(k_min)
            self.int2_pool_v_min[idx].copy_(v_min)
            
            self.int2_pages.append({
                'key_packed': self.int2_pool_k_packed[idx], 'key_scales': self.int2_pool_k_scales[idx], 'key_min': self.int2_pool_k_min[idx], 'key_out_indices': old['key_out_indices'], 'key_out_values': old['key_out_values'],
                'value_packed': self.int2_pool_v_packed[idx], 'value_scales': self.int2_pool_v_scales[idx], 'value_min': self.int2_pool_v_min[idx], 'value_out_indices': old['value_out_indices'], 'value_out_values': old['value_out_values'],
                'pool_idx': idx
            })
            self.int2_pool_idx = (self.int2_pool_idx + 1) % self.max_int2_pages
 
        # 5. INT2 -> 1-Bit (Packed 8 per byte)
        if len(self.int2_pages) > self.max_int2_pages:
            old = self.int2_pages.pop(0)
            k_norm = dequantize_from_int2_packed(old['key_packed'], old['key_scales'], old['key_min'], seq_dim=-2)
            v_norm = dequantize_from_int2_packed(old['value_packed'], old['value_scales'], old['value_min'], seq_dim=-2)
            
            k_packed, k_s = quantize_to_1bit_packed(k_norm, seq_dim=-2, quant_dim=-1)
            v_packed, v_s = quantize_to_1bit_packed(v_norm, seq_dim=-2, quant_dim=-1)
            
            idx = self.one_bit_pool_idx
            self.one_bit_pool_k_packed[idx].copy_(k_packed)
            self.one_bit_pool_v_packed[idx].copy_(v_packed)
            self.one_bit_pool_k_scales[idx].copy_(k_s)
            self.one_bit_pool_v_scales[idx].copy_(v_s)
            
            self.one_bit_pages.append({
                'key_packed': self.one_bit_pool_k_packed[idx], 'key_scales': self.one_bit_pool_k_scales[idx], 'key_out_indices': old['key_out_indices'], 'key_out_values': old['key_out_values'],
                'value_packed': self.one_bit_pool_v_packed[idx], 'value_scales': self.one_bit_pool_v_scales[idx], 'value_out_indices': old['value_out_indices'], 'value_out_values': old['value_out_values'],
                'pool_idx': idx
            })
            self.one_bit_pool_idx = (self.one_bit_pool_idx + 1) % self.max_one_bit_pages
            
        # 6. 1-Bit -> JL Projection FP16 Archive
        if len(self.one_bit_pages) > self.max_one_bit_pages:
            old = self.one_bit_pages.pop(0)
            k_norm = dequantize_from_1bit_packed(old['key_packed'], old['key_scales'], seq_dim=-2)
            v_norm = dequantize_from_1bit_packed(old['value_packed'], old['value_scales'], seq_dim=-2)
            
            # Reconstruct FP16 full tensors from 1-bit and sparse outliers before projection
            k_full = k_norm.clone()
            k_idx = old['key_out_indices']
            if k_idx is not None and k_idx.numel() > 0:
                k_full[k_idx[:, 0].long(), k_idx[:, 1].long(), k_idx[:, 2].long(), k_idx[:, 3].long()] = old['key_out_values']
                
            v_full = v_norm.clone()
            v_idx = old['value_out_indices']
            if v_idx is not None and v_idx.numel() > 0:
                v_full[v_idx[:, 0].long(), v_idx[:, 1].long(), v_idx[:, 2].long(), v_idx[:, 3].long()] = old['value_out_values']
            
            # Project using shared static projection matrix (no individual matrix stored in page)
            w_proj = self.get_jl_projection_matrix(k_full.device, k_full.dtype)
            k_proj = torch.matmul(w_proj, k_full)
            v_proj = torch.matmul(w_proj, v_full)
            
            self.jl_pages.append({
                'key_proj': k_proj,
                'value_proj': v_proj
            })

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
