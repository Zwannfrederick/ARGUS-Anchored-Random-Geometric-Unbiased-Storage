"""Outlier isolation and restoration for quantized KV pages.

A handful of extreme activations dominate a page's dynamic range, so
quantizing them alongside everything else wastes most of the available levels
on values that never occur. Isolating them first lets the quantizer use its
range on the bulk of the distribution, at the cost of storing the outliers
separately in full precision.

These live apart from the memory manager because both the manager and the
variable-granularity subsystem need them, and importing one from the other
would be circular.
"""

from __future__ import annotations

import torch


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
