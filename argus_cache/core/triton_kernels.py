import torch

# Try importing Triton. If not available, we use the PyTorch vector fallback.
TRITON_AVAILABLE = False
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    pass

# =====================================================================
# 1. TRITON GPU JIT KERNELS
# =====================================================================

if TRITON_AVAILABLE:
    @triton.jit
    def triton_pack_int4_kernel(
        x_ptr, packed_ptr, num_elements,
        BLOCK_SIZE: tl.constexpr
    ):
        """Triton Kernel: Packs two 4-bit values into one uint8."""
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        
        # We pack 2 values per iteration, so offsets cover even/odd indices
        offsets_even = block_start * 2 + tl.arange(0, BLOCK_SIZE) * 2
        offsets_odd = offsets_even + 1
        
        mask_even = offsets_even < num_elements
        mask_odd = offsets_odd < num_elements
        
        even_vals = tl.load(x_ptr + offsets_even, mask=mask_even, other=0).to(tl.uint8)
        odd_vals = tl.load(x_ptr + offsets_odd, mask=mask_odd, other=0).to(tl.uint8)
        
        # Pack lower 4 bits of even and odd
        packed = (even_vals & 0x0F) | ((odd_vals & 0x0F) << 4)
        
        offsets_packed = block_start + tl.arange(0, BLOCK_SIZE)
        mask_packed = offsets_packed < (num_elements // 2)
        tl.store(packed_ptr + offsets_packed, packed, mask=mask_packed)

    @triton.jit
    def triton_unpack_int4_kernel(
        packed_ptr, unpacked_ptr, scales_ptr, min_vals_ptr,
        num_elements,
        BLOCK_SIZE: tl.constexpr
    ):
        """Triton Kernel: Unpacks one uint8 into two 4-bit values and performs fused dequantization."""
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        
        offsets_packed = block_start + tl.arange(0, BLOCK_SIZE)
        mask_packed = offsets_packed < (num_elements // 2)
        
        packed = tl.load(packed_ptr + offsets_packed, mask=mask_packed, other=0)
        
        even_vals = packed & 0x0F
        odd_vals = (packed >> 4) & 0x0F
        
        offsets_even = block_start * 2 + tl.arange(0, BLOCK_SIZE) * 2
        offsets_odd = offsets_even + 1
        
        mask_even = offsets_even < num_elements
        mask_odd = offsets_odd < num_elements
        
        # Extremely fast: Load single scalar scale and min per token (since BLOCK_SIZE == head_dim)
        scale_even = tl.load(scales_ptr + (2 * pid))
        min_even = tl.load(min_vals_ptr + (2 * pid))
        
        scale_odd = tl.load(scales_ptr + (2 * pid + 1))
        min_odd = tl.load(min_vals_ptr + (2 * pid + 1))
        
        even_dequant = even_vals.to(tl.float32) * scale_even + min_even
        odd_dequant = odd_vals.to(tl.float32) * scale_odd + min_odd
        
        tl.store(unpacked_ptr + offsets_even, even_dequant, mask=mask_even)
        tl.store(unpacked_ptr + offsets_odd, odd_dequant, mask=mask_odd)

    @triton.jit
    def triton_pack_1bit_kernel(
        x_ptr, packed_ptr, num_elements,
        BLOCK_SIZE: tl.constexpr
    ):
        """Triton Kernel: Packs eight 1-bit sign values into one uint8."""
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        
        offsets_0 = block_start * 8 + tl.arange(0, BLOCK_SIZE) * 8
        
        val0 = tl.load(x_ptr + offsets_0, mask=offsets_0 < num_elements, other=0)
        val1 = tl.load(x_ptr + (offsets_0 + 1), mask=(offsets_0 + 1) < num_elements, other=0)
        val2 = tl.load(x_ptr + (offsets_0 + 2), mask=(offsets_0 + 2) < num_elements, other=0)
        val3 = tl.load(x_ptr + (offsets_0 + 3), mask=(offsets_0 + 3) < num_elements, other=0)
        val4 = tl.load(x_ptr + (offsets_0 + 4), mask=(offsets_0 + 4) < num_elements, other=0)
        val5 = tl.load(x_ptr + (offsets_0 + 5), mask=(offsets_0 + 5) < num_elements, other=0)
        val6 = tl.load(x_ptr + (offsets_0 + 6), mask=(offsets_0 + 6) < num_elements, other=0)
        val7 = tl.load(x_ptr + (offsets_0 + 7), mask=(offsets_0 + 7) < num_elements, other=0)
        
        b0 = (val0 >= 0).to(tl.uint8)
        b1 = (val1 >= 0).to(tl.uint8)
        b2 = (val2 >= 0).to(tl.uint8)
        b3 = (val3 >= 0).to(tl.uint8)
        b4 = (val4 >= 0).to(tl.uint8)
        b5 = (val5 >= 0).to(tl.uint8)
        b6 = (val6 >= 0).to(tl.uint8)
        b7 = (val7 >= 0).to(tl.uint8)
        
        packed = b0 | (b1 << 1) | (b2 << 2) | (b3 << 3) | (b4 << 4) | (b5 << 5) | (b6 << 6) | (b7 << 7)
        
        offsets_packed = block_start + tl.arange(0, BLOCK_SIZE)
        mask_packed = offsets_packed < (num_elements // 8)
        tl.store(packed_ptr + offsets_packed, packed, mask=mask_packed)

    @triton.jit
    def triton_unpack_1bit_kernel(
        packed_ptr, unpacked_ptr, scales_ptr,
        num_elements,
        BLOCK_SIZE: tl.constexpr
    ):
        """Triton Kernel: Unpacks one uint8 into eight 1-bit values and performs fused dequantization."""
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        
        offsets_packed = block_start + tl.arange(0, BLOCK_SIZE)
        mask_packed = offsets_packed < (num_elements // 8)
        
        packed = tl.load(packed_ptr + offsets_packed, mask=mask_packed, other=0)
        
        b0 = (packed & 0x01).to(tl.float32)
        b1 = ((packed >> 1) & 0x01).to(tl.float32)
        b2 = ((packed >> 2) & 0x01).to(tl.float32)
        b3 = ((packed >> 3) & 0x01).to(tl.float32)
        b4 = ((packed >> 4) & 0x01).to(tl.float32)
        b5 = ((packed >> 5) & 0x01).to(tl.float32)
        b6 = ((packed >> 6) & 0x01).to(tl.float32)
        b7 = ((packed >> 7) & 0x01).to(tl.float32)
        
        offsets_0 = block_start * 8 + tl.arange(0, BLOCK_SIZE) * 8
        
        # Extremely fast: Load single scalar scales for the 8 tokens processed by this block (BLOCK_SIZE == head_dim)
        s0 = tl.load(scales_ptr + (8 * pid + 0))
        s1 = tl.load(scales_ptr + (8 * pid + 1))
        s2 = tl.load(scales_ptr + (8 * pid + 2))
        s3 = tl.load(scales_ptr + (8 * pid + 3))
        s4 = tl.load(scales_ptr + (8 * pid + 4))
        s5 = tl.load(scales_ptr + (8 * pid + 5))
        s6 = tl.load(scales_ptr + (8 * pid + 6))
        s7 = tl.load(scales_ptr + (8 * pid + 7))
        
        v0 = (b0 * 2.0 - 1.0) * s0
        v1 = (b1 * 2.0 - 1.0) * s1
        v2 = (b2 * 2.0 - 1.0) * s2
        v3 = (b3 * 2.0 - 1.0) * s3
        v4 = (b4 * 2.0 - 1.0) * s4
        v5 = (b5 * 2.0 - 1.0) * s5
        v6 = (b6 * 2.0 - 1.0) * s6
        v7 = (b7 * 2.0 - 1.0) * s7
        
        tl.store(unpacked_ptr + offsets_0, v0, mask=offsets_0 < num_elements)
        tl.store(unpacked_ptr + (offsets_0 + 1), v1, mask=(offsets_0 + 1) < num_elements)
        tl.store(unpacked_ptr + (offsets_0 + 2), v2, mask=(offsets_0 + 2) < num_elements)
        tl.store(unpacked_ptr + (offsets_0 + 3), v3, mask=(offsets_0 + 3) < num_elements)
        tl.store(unpacked_ptr + (offsets_0 + 4), v4, mask=(offsets_0 + 4) < num_elements)
        tl.store(unpacked_ptr + (offsets_0 + 5), v5, mask=(offsets_0 + 5) < num_elements)
        tl.store(unpacked_ptr + (offsets_0 + 6), v6, mask=(offsets_0 + 6) < num_elements)
        tl.store(unpacked_ptr + (offsets_0 + 7), v7, mask=(offsets_0 + 7) < num_elements)

# =====================================================================
# 2. PYTHON API WRAPPERS WITH AUTOMATIC FALLBACK
# =====================================================================

def triton_pack_int4(tensor: torch.Tensor, seq_dim: int = -2):
    """
    Interface for 4-bit packing on GPU using Triton, with PyTorch vector fallback.
    """
    if not TRITON_AVAILABLE or not tensor.is_cuda:
        if seq_dim == -2 or seq_dim == len(tensor.shape) - 2:
            reshaped = tensor.contiguous().view(*tensor.shape[:-2], tensor.shape[-2] // 2, 2, tensor.shape[-1])
            return reshaped[..., 0, :] | (reshaped[..., 1, :] << 4)
        else:
            num_dims = len(tensor.shape)
            slice_even = [slice(None)] * num_dims
            slice_even[seq_dim] = slice(0, None, 2)
            slice_odd = [slice(None)] * num_dims
            slice_odd[seq_dim] = slice(1, None, 2)
            even_vals = tensor[tuple(slice_even)]
            odd_vals = tensor[tuple(slice_odd)]
            return even_vals | (odd_vals << 4)

    orig_shape = list(tensor.shape)
    seq_len = orig_shape[seq_dim]
    
    assert seq_len % 2 == 0, "Sequence length must be even for 4-bit packing."
    
    x = tensor.contiguous()
    num_elements = x.numel()
    
    packed_shape = list(orig_shape)
    packed_shape[seq_dim] = seq_len // 2
    packed = torch.empty(packed_shape, dtype=torch.uint8, device=tensor.device)
    
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(num_elements // 2, meta['BLOCK_SIZE']),)
    
    triton_pack_int4_kernel[grid](
        x, packed, num_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return packed

def triton_unpack_int4(packed: torch.Tensor, scales: torch.Tensor, min_vals: torch.Tensor, seq_dim: int = -2):
    """
    Interface for 4-bit unpacking on GPU using Triton, with PyTorch vector fallback.
    """
    if not TRITON_AVAILABLE or not packed.is_cuda:
        even_vals = packed & 0x0F
        odd_vals = (packed >> 4) & 0x0F
        
        unpacked_shape = list(packed.shape)
        unpacked_shape[seq_dim] = packed.shape[seq_dim] * 2
        unpacked = torch.empty(unpacked_shape, dtype=scales.dtype, device=packed.device)
        
        if seq_dim == -2 or seq_dim == len(unpacked_shape) - 2:
            reshaped = unpacked.view(*unpacked_shape[:-2], packed.shape[-2], 2, unpacked_shape[-1])
            reshaped[..., 0, :] = even_vals
            reshaped[..., 1, :] = odd_vals
        else:
            num_dims = len(unpacked_shape)
            slice_even = [slice(None)] * num_dims
            slice_even[seq_dim] = slice(0, None, 2)
            slice_odd = [slice(None)] * num_dims
            slice_odd[seq_dim] = slice(1, None, 2)
            
            unpacked[tuple(slice_even)] = even_vals
            unpacked[tuple(slice_odd)] = odd_vals
            
        return unpacked.to(scales.dtype) * scales + min_vals

    packed_shape = list(packed.shape)
    seq_len_packed = packed_shape[seq_dim]
    num_elements = packed.numel() * 2
    
    unpacked_shape = list(packed_shape)
    unpacked_shape[seq_dim] = seq_len_packed * 2
    unpacked = torch.empty(unpacked_shape, dtype=scales.dtype, device=packed.device)
    
    # We set BLOCK_SIZE exactly to head_dim for uniform scalar loads
    BLOCK_SIZE = packed.shape[-1]
    grid = lambda meta: (triton.cdiv(num_elements // 2, meta['BLOCK_SIZE']),)
    
    triton_unpack_int4_kernel[grid](
        packed, unpacked, scales, min_vals, num_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return unpacked

def triton_pack_1bit(tensor: torch.Tensor, seq_dim: int = -2):
    """
    Packs a float/half tensor into 1-bit packed uint8 along seq_dim using Triton or PyTorch fallback.
    """
    if not TRITON_AVAILABLE or not tensor.is_cuda:
        binarized = (tensor >= 0).to(torch.uint8)
        if seq_dim == -2 or seq_dim == len(tensor.shape) - 2:
            reshaped = binarized.contiguous().view(*binarized.shape[:-2], binarized.shape[-2] // 8, 8, binarized.shape[-1])
            packed = (
                reshaped[..., 0, :] |
                (reshaped[..., 1, :] << 1) |
                (reshaped[..., 2, :] << 2) |
                (reshaped[..., 3, :] << 3) |
                (reshaped[..., 4, :] << 4) |
                (reshaped[..., 5, :] << 5) |
                (reshaped[..., 6, :] << 6) |
                (reshaped[..., 7, :] << 7)
            )
            return packed
        else:
            num_dims = len(tensor.shape)
            slice_0 = [slice(None)] * num_dims
            slice_0[seq_dim] = slice(0, None, 8)
            slice_1 = [slice(None)] * num_dims
            slice_1[seq_dim] = slice(1, None, 8)
            slice_2 = [slice(None)] * num_dims
            slice_2[seq_dim] = slice(2, None, 8)
            slice_3 = [slice(None)] * num_dims
            slice_3[seq_dim] = slice(3, None, 8)
            slice_4 = [slice(None)] * num_dims
            slice_4[seq_dim] = slice(4, None, 8)
            slice_5 = [slice(None)] * num_dims
            slice_5[seq_dim] = slice(5, None, 8)
            slice_6 = [slice(None)] * num_dims
            slice_6[seq_dim] = slice(6, None, 8)
            slice_7 = [slice(None)] * num_dims
            slice_7[seq_dim] = slice(7, None, 8)
            
            packed = (
                binarized[tuple(slice_0)] |
                (binarized[tuple(slice_1)] << 1) |
                (binarized[tuple(slice_2)] << 2) |
                (binarized[tuple(slice_3)] << 3) |
                (binarized[tuple(slice_4)] << 4) |
                (binarized[tuple(slice_5)] << 5) |
                (binarized[tuple(slice_6)] << 6) |
                (binarized[tuple(slice_7)] << 7)
            )
            return packed

    orig_shape = list(tensor.shape)
    seq_len = orig_shape[seq_dim]
    assert seq_len % 8 == 0, "Sequence length must be a multiple of 8 for 1-bit packing."
    
    x = tensor.contiguous()
    num_elements = x.numel()
    
    packed_shape = list(orig_shape)
    packed_shape[seq_dim] = seq_len // 8
    packed = torch.empty(packed_shape, dtype=torch.uint8, device=tensor.device)
    
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(num_elements // 8, meta['BLOCK_SIZE']),)
    
    triton_pack_1bit_kernel[grid](
        x, packed, num_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return packed

def triton_unpack_1bit(packed: torch.Tensor, scales: torch.Tensor, seq_dim: int = -2):
    """
    Unpacks a 1-bit packed uint8 tensor along seq_dim.
    """
    if not TRITON_AVAILABLE or not packed.is_cuda:
        b0 = packed & 0x01
        b1 = (packed >> 1) & 0x01
        b2 = (packed >> 2) & 0x01
        b3 = (packed >> 3) & 0x01
        b4 = (packed >> 4) & 0x01
        b5 = (packed >> 5) & 0x01
        b6 = (packed >> 6) & 0x01
        b7 = (packed >> 7) & 0x01
        
        unpacked_shape = list(packed.shape)
        unpacked_shape[seq_dim] = packed.shape[seq_dim] * 8
        unpacked = torch.empty(unpacked_shape, dtype=torch.uint8, device=packed.device)
        
        if seq_dim == -2 or seq_dim == len(unpacked_shape) - 2:
            reshaped = unpacked.view(*unpacked_shape[:-2], packed.shape[-2], 8, unpacked_shape[-1])
            reshaped[..., 0, :] = b0
            reshaped[..., 1, :] = b1
            reshaped[..., 2, :] = b2
            reshaped[..., 3, :] = b3
            reshaped[..., 4, :] = b4
            reshaped[..., 5, :] = b5
            reshaped[..., 6, :] = b6
            reshaped[..., 7, :] = b7
        else:
            num_dims = len(unpacked_shape)
            slice_0 = [slice(None)] * num_dims
            slice_0[seq_dim] = slice(0, None, 8)
            slice_1 = [slice(None)] * num_dims
            slice_1[seq_dim] = slice(1, None, 8)
            slice_2 = [slice(None)] * num_dims
            slice_2[seq_dim] = slice(2, None, 8)
            slice_3 = [slice(None)] * num_dims
            slice_3[seq_dim] = slice(3, None, 8)
            slice_4 = [slice(None)] * num_dims
            slice_4[seq_dim] = slice(4, None, 8)
            slice_5 = [slice(None)] * num_dims
            slice_5[seq_dim] = slice(5, None, 8)
            slice_6 = [slice(None)] * num_dims
            slice_6[seq_dim] = slice(6, None, 8)
            slice_7 = [slice(None)] * num_dims
            slice_7[seq_dim] = slice(7, None, 8)
            
            unpacked[tuple(slice_0)] = b0
            unpacked[tuple(slice_1)] = b1
            unpacked[tuple(slice_2)] = b2
            unpacked[tuple(slice_3)] = b3
            unpacked[tuple(slice_4)] = b4
            unpacked[tuple(slice_5)] = b5
            unpacked[tuple(slice_6)] = b6
            unpacked[tuple(slice_7)] = b7
            
        return (unpacked.to(scales.dtype) * 2.0 - 1.0) * scales

    packed_shape = list(packed.shape)
    seq_len_packed = packed_shape[seq_dim]
    num_elements = packed.numel() * 8
    
    unpacked_shape = list(packed_shape)
    unpacked_shape[seq_dim] = seq_len_packed * 8
    unpacked = torch.empty(unpacked_shape, dtype=scales.dtype, device=packed.device)
    
    # We set BLOCK_SIZE exactly to head_dim for uniform scalar loads
    BLOCK_SIZE = packed.shape[-1]
    grid = lambda meta: (triton.cdiv(num_elements // 8, meta['BLOCK_SIZE']),)
    
    triton_unpack_1bit_kernel[grid](
        packed, unpacked, scales, num_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return unpacked
