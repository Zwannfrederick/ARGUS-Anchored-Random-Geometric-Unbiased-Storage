import torch
from core.quantization import (
    quantize_to_int8,
    dequantize_from_int8,
    quantize_to_int4_packed,
    dequantize_from_int4_packed,
    quantize_to_fp8_simulated,
    dequantize_from_fp8_simulated,
    quantize_to_jl_projection,
    dequantize_from_jl_projection
)

class PagedDynamicKVCache:
    def __init__(self, page_size=4096, max_active_pages=2, max_fp8_pages=2, max_int8_pages=2, max_int4_pages=2, sink_tokens=4):
        """
        Generic Outlier-Aware 5-Tier Token-Based Paged Dynamic KV Cache Manager.
        Memory Lifecycle:
            FP16 (Sinks) + FP16 (Active) -> FP8 (Light) -> INT8 (Medium) -> INT4 (Heavy) -> JL Projection (FP16 Sequence-Compressed Archive)
        
        Args:
            page_size (int): Number of tokens per page. Must be a multiple of 4.
            max_active_pages (int): Max active FP16 pages.
            max_fp8_pages (int): Max FP8 pages.
            max_int8_pages (int): Max INT8 pages.
            max_int4_pages (int): Max INT4 packed pages.
            sink_tokens (int): Number of initial tokens (Attention Sinks) kept in FP16 permanently.
        """
        assert page_size % 4 == 0, "Page size must be a multiple of 4 for 2-bit super packing."
        self.page_size = page_size
        self.max_active_pages = max_active_pages
        self.max_fp8_pages = max_fp8_pages
        self.max_int8_pages = max_int8_pages
        self.max_int4_pages = max_int4_pages
        
        # Outlier-Aware: Attention Sinks (First N tokens kept in FP16 permanently)
        self.sink_tokens = sink_tokens
        self.sink_k = None
        self.sink_v = None
        
        # VIP Outliers: Newline & Rhyme Anchors kept in FP16 permanently
        self.anchor_k = None
        self.anchor_v = None
        
        # Memory tiers
        self.active_pages = []   # Tier 1: FP16
        self.fp8_pages = []      # Tier 2: FP8 (stored as int8 + scale)
        self.int8_pages = []     # Tier 3: INT8
        self.int4_pages = []     # Tier 4: INT4 (packed 2 per byte)
        self.int2_pages = []     # Tier 5: JL-Projected FP16 Sequence-Compressed Archive (Unlimited size)
        
        # Temp active buffers for incoming tokens
        self.k_buffer = None
        self.v_buffer = None

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

        # 2. Append to normal active buffers
        if self.k_buffer is None:
            self.k_buffer = keys
            self.v_buffer = values
        else:
            self.k_buffer = torch.cat([self.k_buffer, keys], dim=-2)
            self.v_buffer = torch.cat([self.v_buffer, values], dim=-2)
            
        # 3. Segment into pages
        while self.k_buffer.shape[-2] >= self.page_size:
            page_k = self.k_buffer[..., :self.page_size, :]
            page_v = self.v_buffer[..., :self.page_size, :]
            
            self.k_buffer = self.k_buffer[..., self.page_size:, :]
            self.v_buffer = self.v_buffer[..., self.page_size:, :]
            
            self.active_pages.append({
                'key': page_k,
                'value': page_v
            })
            
            self.manage_memory_lifecycle()

    def manage_memory_lifecycle(self):
        """
        Transitions pages down the 5-tier memory lifecycle when limits are exceeded.
        """
        # 1. FP16 -> FP8 (e4m3fn)
        if len(self.active_pages) > self.max_active_pages:
            page = self.active_pages.pop(0)
            k_q, k_s = quantize_to_fp8_simulated(page['key'], dim=-1)
            v_q, v_s = quantize_to_fp8_simulated(page['value'], dim=-1)
            self.fp8_pages.append({
                'key_q': k_q, 'key_scales': k_s,
                'value_q': v_q, 'value_scales': v_s
            })
            
        # 2. FP8 -> INT8
        if len(self.fp8_pages) > self.max_fp8_pages:
            old = self.fp8_pages.pop(0)
            k_fp16 = dequantize_from_fp8_simulated(old['key_q'], old['key_scales'])
            v_fp16 = dequantize_from_fp8_simulated(old['value_q'], old['value_scales'])
            
            k_q, k_s = quantize_to_int8(k_fp16, dim=-1)
            v_q, v_s = quantize_to_int8(v_fp16, dim=-1)
            self.int8_pages.append({
                'key_q': k_q, 'key_scales': k_s,
                'value_q': v_q, 'value_scales': v_s
            })
            
        # 3. INT8 -> INT4 (Packed)
        if len(self.int8_pages) > self.max_int8_pages:
            old = self.int8_pages.pop(0)
            k_fp16 = dequantize_from_int8(old['key_q'], old['key_scales'])
            v_fp16 = dequantize_from_int8(old['value_q'], old['value_scales'])
            
            k_packed, k_s, k_min = quantize_to_int4_packed(k_fp16, seq_dim=-2, quant_dim=-1)
            v_packed, v_s, v_min = quantize_to_int4_packed(v_fp16, seq_dim=-2, quant_dim=-1)
            self.int4_pages.append({
                'key_packed': k_packed, 'key_scales': k_s, 'key_min': k_min,
                'value_packed': v_packed, 'value_scales': v_s, 'value_min': v_min
            })
            
        # 4. INT4 -> JL Projection (FP16 Sequence-Compressed Archive)
        if len(self.int4_pages) > self.max_int4_pages:
            old = self.int4_pages.pop(0)
            k_fp16 = dequantize_from_int4_packed(old['key_packed'], old['key_scales'], old['key_min'], seq_dim=-2)
            v_fp16 = dequantize_from_int4_packed(old['value_packed'], old['value_scales'], old['value_min'], seq_dim=-2)
            
            # Reduce sequence length 4x keeping perfect FP16 precision
            k_proj, w_k = quantize_to_jl_projection(k_fp16, ratio=4)
            v_proj, w_v = quantize_to_jl_projection(v_fp16, ratio=4)
            self.int2_pages.append({
                'key_proj': k_proj, 'w_k': w_k,
                'value_proj': v_proj, 'w_v': w_v
            })

    def get_all_keys_values(self):
        """
        Reconstructs all cache levels back to single FP16 tensors, prepending the Attention Sinks.
        """
        all_keys = []
        all_values = []
        
        # Tier 5: JL Projection Sequence-Compressed Archive
        for page in self.int2_pages:
            k = dequantize_from_jl_projection(page['key_proj'], page['w_k'])
            v = dequantize_from_jl_projection(page['value_proj'], page['w_v'])
            all_keys.append(k)
            all_values.append(v)
            
        # Tier 4: INT4 packed
        for page in self.int4_pages:
            k = dequantize_from_int4_packed(page['key_packed'], page['key_scales'], page['key_min'], seq_dim=-2)
            v = dequantize_from_int4_packed(page['value_packed'], page['value_scales'], page['value_min'], seq_dim=-2)
            all_keys.append(k)
            all_values.append(v)
            
        # Tier 3: INT8
        for page in self.int8_pages:
            k = dequantize_from_int8(page['key_q'], page['key_scales'])
            v = dequantize_from_int8(page['value_q'], page['value_scales'])
            all_keys.append(k)
            all_values.append(v)
            
        # Tier 2: FP8
        for page in self.fp8_pages:
            k = dequantize_from_fp8_simulated(page['key_q'], page['key_scales'])
            v = dequantize_from_fp8_simulated(page['value_q'], page['value_scales'])
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

    def get_vram_usage(self):
        """
        Calculates theoretical memory usage across all 5 tiers + sinks.
        """
        total_bytes = 0
        
        # Sinks (FP16: 2 bytes)
        if self.sink_k is not None:
            total_bytes += self.sink_k.element_size() * self.sink_k.nelement() * 2
            
        # VIP Anchors (FP16: 2 bytes)
        if self.anchor_k is not None:
            total_bytes += self.anchor_k.element_size() * self.anchor_k.nelement() * 2
            
        # FP16 (2 bytes)
        for page in self.active_pages:
            total_bytes += page['key'].element_size() * page['key'].nelement() * 2
            
        # FP8 (1 byte + scales)
        for page in self.fp8_pages:
            total_bytes += page['key_q'].element_size() * page['key_q'].nelement() * 2
            total_bytes += page['key_scales'].element_size() * page['key_scales'].nelement() * 2
            
        # INT8 (1 byte + scales)
        for page in self.int8_pages:
            total_bytes += page['key_q'].element_size() * page['key_q'].nelement() * 2
            total_bytes += page['key_scales'].element_size() * page['key_scales'].nelement() * 2
            
        # INT4 (0.5 byte + scales & mins)
        for page in self.int4_pages:
            total_bytes += page['key_packed'].element_size() * page['key_packed'].nelement() * 2
            total_bytes += page['key_scales'].element_size() * page['key_scales'].nelement() * 2
            total_bytes += page['key_min'].element_size() * page['key_min'].nelement() * 2
            
        # JL Projection Sequence-Compressed FP16 (1/4 size in FP16 = 0.5 byte equivalent + projection matrices)
        for page in self.int2_pages:
            total_bytes += page['key_proj'].element_size() * page['key_proj'].nelement() * 2
            total_bytes += page['value_proj'].element_size() * page['value_proj'].nelement() * 2
            total_bytes += page['w_k'].element_size() * page['w_k'].nelement() * 2
            total_bytes += page['w_v'].element_size() * page['w_v'].nelement() * 2
            
        # Buffers
        if self.k_buffer is not None:
            total_bytes += self.k_buffer.element_size() * self.k_buffer.nelement() * 2
            
        return total_bytes
