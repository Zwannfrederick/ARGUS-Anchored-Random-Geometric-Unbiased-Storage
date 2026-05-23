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
        packed_ptr, unpacked_ptr, num_elements,
        BLOCK_SIZE: tl.constexpr
    ):
        """Triton Kernel: Unpacks one uint8 into two 4-bit values."""
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
        
        tl.store(unpacked_ptr + offsets_even, even_vals, mask=mask_even)
        tl.store(unpacked_ptr + offsets_odd, odd_vals, mask=mask_odd)

# =====================================================================
# 2. PYTHON API WRAPPERS WITH AUTOMATIC FALLBACK
# =====================================================================

def triton_pack_int4(tensor: torch.Tensor, seq_dim: int = -2):
    """
    Interface for 4-bit packing on GPU using Triton, with PyTorch vector fallback.
    """
    if not TRITON_AVAILABLE or not tensor.is_cuda:
        # Fallback to PyTorch vector implementation (which is also highly optimized)
        num_dims = len(tensor.shape)
        slice_even = [slice(None)] * num_dims
        slice_even[seq_dim] = slice(0, None, 2)
        slice_odd = [slice(None)] * num_dims
        slice_odd[seq_dim] = slice(1, None, 2)
        
        even_vals = tensor[tuple(slice_even)]
        odd_vals = tensor[tuple(slice_odd)]
        return even_vals | (odd_vals << 4)

    # Triton implementation
    # Flatten tensor to treat it as a 1D contiguous block on GPU
    orig_shape = list(tensor.shape)
    seq_len = orig_shape[seq_dim]
    
    assert seq_len % 2 == 0, "Sequence length must be even for 4-bit packing."
    
    # Rearrange sequence dimension to be contiguous for the kernel if needed
    # (For caching, seq_len is usually contiguous)
    x = tensor.contiguous()
    num_elements = x.numel()
    
    packed_shape = list(orig_shape)
    packed_shape[seq_dim] = seq_len // 2
    packed = torch.empty(packed_shape, dtype=torch.uint8, device=tensor.device)
    
    # Configure grid and block size
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(num_elements // 2, meta['BLOCK_SIZE']),)
    
    triton_pack_int4_kernel[grid](
        x, packed, num_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return packed

def triton_unpack_int4(packed: torch.Tensor, original_dtype: torch.dtype, seq_dim: int = -2):
    """
    Interface for 4-bit unpacking on GPU using Triton, with PyTorch vector fallback.
    """
    if not TRITON_AVAILABLE or not packed.is_cuda:
        # PyTorch fallback
        even_vals = packed & 0x0F
        odd_vals = (packed >> 4) & 0x0F
        
        unpacked_shape = list(packed.shape)
        unpacked_shape[seq_dim] = packed.shape[seq_dim] * 2
        
        unpacked = torch.empty(unpacked_shape, dtype=torch.uint8, device=packed.device)
        
        num_dims = len(unpacked_shape)
        slice_even = [slice(None)] * num_dims
        slice_even[seq_dim] = slice(0, None, 2)
        slice_odd = [slice(None)] * num_dims
        slice_odd[seq_dim] = slice(1, None, 2)
        
        unpacked[tuple(slice_even)] = even_vals
        unpacked[tuple(slice_odd)] = odd_vals
        return unpacked

    # Triton implementation
    packed_shape = list(packed.shape)
    seq_len_packed = packed_shape[seq_dim]
    num_elements = packed.numel() * 2
    
    unpacked_shape = list(packed_shape)
    unpacked_shape[seq_dim] = seq_len_packed * 2
    unpacked = torch.empty(unpacked_shape, dtype=torch.uint8, device=packed.device)
    
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(num_elements // 2, meta['BLOCK_SIZE']),)
    
    triton_unpack_int4_kernel[grid](
        packed, unpacked, num_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return unpacked
