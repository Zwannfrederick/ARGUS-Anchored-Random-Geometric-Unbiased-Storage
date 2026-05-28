import torch

def quantize_to_int8(tensor: torch.Tensor, dim: int = -1):
    """
    Symmetric per-channel/per-token INT8 quantization.
    Returns:
        quantized_tensor (torch.int8)
        scale (torch.float16 or same as input)
    """
    # Keep original dtype for scales
    dtype = tensor.dtype
    
    # Calculate absolute max along the quantization dimension
    # Add a small epsilon to avoid division by zero
    max_vals = torch.amax(torch.abs(tensor), dim=dim, keepdim=True)
    scales = max_vals / 127.0
    scales = torch.clamp(scales, min=1e-8)
    
    # Quantize and clamp to signed int8 range [-128, 127]
    quantized = torch.round(tensor / scales).clamp(-128, 127).to(torch.int8)
    
    return quantized, scales

def dequantize_from_int8(quantized: torch.Tensor, scales: torch.Tensor):
    """
    Dequantize INT8 back to the scale's dtype.
    """
    return quantized.to(scales.dtype) * scales

def quantize_to_int4_packed(tensor: torch.Tensor, seq_dim: int = -2, quant_dim: int = -1):
    """
    Asymmetric INT4 quantization with bit-packing.
    Packs two 4-bit values along seq_dim (which must have an even size) into one uint8.
    Quantization scale/min are calculated per-channel along quant_dim.
    
    Returns:
        packed_tensor (torch.uint8) - half the size of seq_dim
        scale (torch.Tensor)
        min_val (torch.Tensor)
    """
    assert tensor.shape[seq_dim] % 2 == 0, f"Sequence dimension size ({tensor.shape[seq_dim]}) must be even for 4-bit packing."
    dtype = tensor.dtype
    
    # Compute min and max along quant_dim
    min_vals = torch.amin(tensor, dim=quant_dim, keepdim=True)
    max_vals = torch.amax(tensor, dim=quant_dim, keepdim=True)
    
    # Calculate scales for asymmetric 4-bit [0, 15]
    scales = (max_vals - min_vals) / 15.0
    scales = torch.clamp(scales, min=1e-8)
    
    # Quantize to [0, 15]
    quantized = torch.round((tensor - min_vals) / scales).clamp(0, 15).to(torch.uint8)
    
    # Pack using Triton Custom GPU Kernel with PyTorch Fallback
    from .triton_kernels import triton_pack_int4
    packed = triton_pack_int4(quantized, seq_dim=seq_dim)
    
    return packed, scales, min_vals

def dequantize_from_int4_packed(packed: torch.Tensor, scales: torch.Tensor, min_vals: torch.Tensor, seq_dim: int = -2):
    """
    Unpack packed uint8 tensor and dequantize to FP16/original precision.
    """
    # Unpack using Triton Custom GPU Kernel with PyTorch Fallback
    from .triton_kernels import triton_unpack_int4
    return triton_unpack_int4(packed, scales, min_vals, seq_dim=seq_dim)

def quantize_to_int2_packed(tensor: torch.Tensor, seq_dim: int = -2, quant_dim: int = -1):
    """
    Asymmetric INT2 super-quantization with 4-way bit-packing.
    Packs four 2-bit values along seq_dim (must be a multiple of 4) into one uint8.
    
    Returns:
        packed_tensor (torch.uint8) - 1/4 the size of seq_dim
        scale (torch.Tensor)
        min_val (torch.Tensor)
    """
    assert tensor.shape[seq_dim] % 4 == 0, f"Sequence dimension size ({tensor.shape[seq_dim]}) must be a multiple of 4 for 2-bit packing."
    dtype = tensor.dtype
    
    # Compute min and max along quant_dim
    min_vals = torch.amin(tensor, dim=quant_dim, keepdim=True)
    max_vals = torch.amax(tensor, dim=quant_dim, keepdim=True)
    
    # Calculate scales for asymmetric 2-bit [0, 3]
    scales = (max_vals - min_vals) / 3.0
    scales = torch.clamp(scales, min=1e-8)
    
    # Quantize to [0, 3]
    quantized = torch.round((tensor - min_vals) / scales).clamp(0, 3).to(torch.uint8)
    
    # Extract slices for packing using contiguous view if seq_dim is second-to-last
    if seq_dim == -2 or seq_dim == len(tensor.shape) - 2:
        reshaped = quantized.contiguous().view(*quantized.shape[:-2], quantized.shape[-2] // 4, 4, quantized.shape[-1])
        packed = (
            reshaped[..., 0, :] |
            (reshaped[..., 1, :] << 2) |
            (reshaped[..., 2, :] << 4) |
            (reshaped[..., 3, :] << 6)
        )
    else:
        num_dims = len(tensor.shape)
        
        slice_0 = [slice(None)] * num_dims
        slice_0[seq_dim] = slice(0, None, 4)
        
        slice_1 = [slice(None)] * num_dims
        slice_1[seq_dim] = slice(1, None, 4)
        
        slice_2 = [slice(None)] * num_dims
        slice_2[seq_dim] = slice(2, None, 4)
        
        slice_3 = [slice(None)] * num_dims
        slice_3[seq_dim] = slice(3, None, 4)
        
        v0 = quantized[tuple(slice_0)]
        v1 = quantized[tuple(slice_1)]
        v2 = quantized[tuple(slice_2)]
        v3 = quantized[tuple(slice_3)]
        
        packed = v0 | (v1 << 2) | (v2 << 4) | (v3 << 6)
    
    return packed, scales, min_vals

def dequantize_from_int2_packed(packed: torch.Tensor, scales: torch.Tensor, min_vals: torch.Tensor, seq_dim: int = -2):
    """
    Unpack packed uint8 tensor and dequantize 2-bit values back to FP16/original precision.
    """
    # Unpack four 2-bit values
    v0 = packed & 0x03
    v1 = (packed >> 2) & 0x03
    v2 = (packed >> 4) & 0x03
    v3 = (packed >> 6) & 0x03
    
    # Reconstruct shape along seq_dim (4x size of packed seq_dim)
    packed_shape = list(packed.shape)
    unpacked_shape = list(packed.shape)
    unpacked_shape[seq_dim] = packed_shape[seq_dim] * 4
    
    # Pre-allocate unpacked tensor
    unpacked = torch.empty(unpacked_shape, dtype=torch.uint8, device=packed.device)
    
    if seq_dim == -2 or seq_dim == len(unpacked_shape) - 2:
        reshaped = unpacked.view(*unpacked_shape[:-2], packed_shape[-2], 4, unpacked_shape[-1])
        reshaped[..., 0, :] = v0
        reshaped[..., 1, :] = v1
        reshaped[..., 2, :] = v2
        reshaped[..., 3, :] = v3
    else:
        num_dims = len(unpacked_shape)
        slice_0 = [slice(None)] * num_dims
        slice_0[seq_dim] = slice(0, None, 4)
        
        slice_1 = [slice(None)] * num_dims
        slice_1[seq_dim] = slice(1, None, 4)
        
        slice_2 = [slice(None)] * num_dims
        slice_2[seq_dim] = slice(2, None, 4)
        
        slice_3 = [slice(None)] * num_dims
        slice_3[seq_dim] = slice(3, None, 4)
        
        unpacked[tuple(slice_0)] = v0
        unpacked[tuple(slice_1)] = v1
        unpacked[tuple(slice_2)] = v2
        unpacked[tuple(slice_3)] = v3
    
    # Dequantize
    dequantized = unpacked.to(scales.dtype) * scales + min_vals
    return dequantized

def quantize_to_1bit_packed(tensor: torch.Tensor, seq_dim: int = -2, quant_dim: int = -1):
    """
    Symmetric 1-bit quantization with bit-packing along seq_dim.
    Packs 8 sign bits (from seq_dim, which must be a multiple of 8) into one uint8.
    The scale factor is calculated as the mean of absolute values along quant_dim.
    
    Returns:
        packed_tensor (torch.uint8)
        scale (torch.Tensor)
    """
    assert tensor.shape[seq_dim] % 8 == 0, f"Sequence dimension size ({tensor.shape[seq_dim]}) must be a multiple of 8 for 1-bit packing."
    
    # Calculate scale factor per-channel/per-token along quant_dim
    # We use mean of absolute values
    scales = torch.mean(torch.abs(tensor), dim=quant_dim, keepdim=True)
    scales = torch.clamp(scales, min=1e-8)
    
    # Binarize and pack
    from .triton_kernels import triton_pack_1bit
    packed = triton_pack_1bit(tensor, seq_dim=seq_dim)
    
    return packed, scales

def dequantize_from_1bit_packed(packed: torch.Tensor, scales: torch.Tensor, seq_dim: int = -2):
    """
    Unpack packed uint8 1-bit tensor and scale back to original precision.
    """
    from .triton_kernels import triton_unpack_1bit
    return triton_unpack_1bit(packed, scales, seq_dim=seq_dim)

def quantize_to_fp8_simulated(tensor: torch.Tensor, dim: int = -1):
    """
    Simulated symmetric FP8 (e4m3fn style) quantization.
    Uses float8 range max=240.0, storing in int8 to replicate 8-bit memory storage (1 byte).
    """
    # Calculate absolute max along dim
    max_vals = torch.amax(torch.abs(tensor), dim=dim, keepdim=True)
    scales = max_vals / 240.0
    scales = torch.clamp(scales, min=1e-8)
    
    # Scale and clamp to e4m3 signed range [-240, 240]
    quantized = torch.round(tensor / scales).clamp(-240, 240).to(torch.int8)
    
    return quantized, scales

def dequantize_from_fp8_simulated(quantized: torch.Tensor, scales: torch.Tensor):
    """
    Dequantize simulated FP8 back to original dtype.
    """
    return quantized.to(scales.dtype) * scales

def quantize_to_jl_projection(tensor: torch.Tensor, ratio: int = 4):
    """
    Johnson-Lindenstrauss Random Orthogonal Matrix Projection sequence compression.
    Projects sequence dimension N to M (where M = N // ratio) keeping FP16 precision.
    
    Returns:
        compressed_tensor (torch.Tensor) - shape: [..., M, D]
        projection_matrix (torch.Tensor) - shape: [M, N]
    """
    orig_shape = list(tensor.shape)
    n = orig_shape[-2]
    m = n // ratio
    
    # Run QR decomposition on CPU to avoid CUDA OOM for large context lengths
    cpu_device = torch.device("cpu")
    raw_randn = torch.randn(n, m, dtype=torch.float32, device=cpu_device)
    q, _ = torch.linalg.qr(raw_randn)
    w_proj = q.t().to(tensor.dtype).to(tensor.device) # [M, N]
    
    compressed = torch.matmul(w_proj, tensor)
    
    return compressed, w_proj

def dequantize_from_jl_projection(compressed: torch.Tensor, w_proj: torch.Tensor, recon_operator: torch.Tensor = None, alpha: float = 1e-3):
    """
    De-project compressed memory back to original sequence length.
    If recon_operator is provided, it uses it directly (highly recommended for performance).
    Otherwise, it computes it dynamically.
    Reconstructs shape back to [..., N, D].
    """
    if recon_operator is not None:
        return torch.matmul(recon_operator, compressed)
        
    device = compressed.device
    dtype = compressed.dtype
    M, N = w_proj.shape
    
    # Run the operator precomputation steps on CPU to avoid CUDA OOM / fragmentation
    cpu_device = torch.device("cpu")
    
    # Vectorized tridiagonal Laplacian formulation
    L = torch.zeros(N, N, dtype=torch.float32, device=cpu_device)
    L.diagonal().fill_(2.0)
    L.diagonal(-1).fill_(-1.0)
    L.diagonal(1).fill_(-1.0)
    L[0, 0] = 1.0
    L[N-1, N-1] = 1.0
    
    # Regularize: A = L + alpha * I
    A = L + alpha * torch.eye(N, dtype=torch.float32, device=cpu_device)
    
    W = w_proj.to(torch.float32).to(cpu_device)
    
    # Solves A * Y = W^T for Y (which is equivalent to Y = A_inv * W^T).
    # Since A is tridiagonal, this solve is extremely fast and avoids computing N x N A_inv.
    # W^T shape: [N, M], A shape: [N, N] -> Y shape: [N, M]
    Y = torch.linalg.solve(A, W.t())
    
    # Compute W * Y -> shape: [M, M]
    WY = torch.matmul(W, Y)
    
    # Compute inverse of WY -> shape: [M, M]. This is much smaller and faster (M = N // ratio).
    inv_term = torch.inverse(WY)
    
    # recon = Y * inv_term -> shape: [N, M]
    recon = torch.matmul(Y, inv_term).to(device).to(dtype)
    return torch.matmul(recon, compressed)
