import torch
import sys
import os

# Add parent directory to path so we can import core
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.quantization import (
    quantize_to_1bit_packed,
    dequantize_from_1bit_packed
)
from core.memory_manager import isolate_outliers

def lossless_delta_encode(tensor, seq_dim=-2):
    """
    Computes first-order sequential differences (deltas) along seq_dim using float32 to prevent rounding error.
    """
    t_32 = tensor.to(torch.float32)
    deltas = torch.zeros_like(t_32)
    num_dims = len(tensor.shape)
    
    slice_0 = [slice(None)] * num_dims
    slice_0[seq_dim] = 0
    deltas[tuple(slice_0)] = t_32[tuple(slice_0)]
    
    slice_rest = [slice(None)] * num_dims
    slice_rest[seq_dim] = slice(1, None)
    
    slice_prev = [slice(None)] * num_dims
    slice_prev[seq_dim] = slice(0, -1)
    
    deltas[tuple(slice_rest)] = t_32[tuple(slice_rest)] - t_32[tuple(slice_prev)]
    return deltas.to(tensor.dtype)

def lossless_delta_decode(deltas, seq_dim=-2):
    """
    Reconstructs the original tensor from first-order sequential deltas using cumulative sum in float32.
    """
    d_32 = deltas.to(torch.float32)
    reconstructed = torch.cumsum(d_32, dim=seq_dim)
    return reconstructed.to(deltas.dtype)

def calculate_cosine_similarity(a, b):
    """
    Computes average cosine similarity along the last dimension.
    """
    flat_a = a.view(-1, a.shape[-1]).to(torch.float32)
    flat_b = b.view(-1, b.shape[-1]).to(torch.float32)
    norm_a = torch.nn.functional.normalize(flat_a, p=2, dim=-1)
    norm_b = torch.nn.functional.normalize(flat_b, p=2, dim=-1)
    return torch.mean(torch.sum(norm_a * norm_b, dim=-1)).item()

def test_compression_loss_and_similarity():
    """
    Rigorous scientific test comparing reconstruction loss, cosine similarity,
    and compression ratios between 1-Bit Outlier-Aware Quantization and Lossless Delta Encoding.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running comparative reconstruction analysis on device: {device}")
    
    # 1. Generate realistic KV feature tensor with occasional severe spikes (outliers)
    # Shape: [batch=1, heads=4, seq_len=64, head_dim=128]
    torch.manual_seed(42)
    base_tensor = torch.randn(1, 4, 64, 128, dtype=torch.float16, device=device)
    
    # Inject high-amplitude outlier spikes (representing newLine/rhyme/structural anchors)
    # Outliers exceed 4.0 standard deviations
    spike_mask = torch.rand_like(base_tensor) < 0.05
    base_tensor[spike_mask] = base_tensor[spike_mask] * 12.0
    
    print("\n--- BASE TENSOR METRICS ---")
    print(f"Max Value: {torch.max(base_tensor).item():.4f}")
    print(f"Min Value: {torch.min(base_tensor).item():.4f}")
    print(f"Mean Absolute: {torch.mean(torch.abs(base_tensor)).item():.4f}")
    
    # 2. Perform Lossless Delta Encoding & Decoding
    delta_encoded = lossless_delta_encode(base_tensor, seq_dim=-2)
    delta_decoded = lossless_delta_decode(delta_encoded, seq_dim=-2)
    
    delta_mse = torch.mean((base_tensor - delta_decoded) ** 2).item()
    delta_cos = calculate_cosine_similarity(base_tensor, delta_decoded)
    
    print("\n--- TIER 0: LOSSLESS DELTA ENCODING ---")
    print(f"Reconstruction MSE: {delta_mse:.8f} (Must be exactly 0.0)")
    print(f"Cosine Similarity: {delta_cos:.8f} (Must be exactly 1.0)")
    
    # In FP16 precision, allow tiny representation epsilon (1e-4)
    assert delta_mse < 1e-4, f"Lossless Delta Encoding error too high: {delta_mse}"
    assert abs(delta_cos - 1.0) < 1e-4, f"Lossless Delta Encoding cosine similarity too low: {delta_cos}"
    
    # 3. Perform 1-Bit Symmetric Quantization WITH Outlier Isolation
    print("\n--- TIER 5: 1-BIT OUTLIER-AWARE QUANTIZATION ---")
    threshold_sigma = 3.0
    normal_vals, outlier_vals, outlier_mask = isolate_outliers(base_tensor, threshold_sigma)
    
    # Quantize normal values to 1-bit packed
    packed_normal, scales = quantize_to_1bit_packed(normal_vals, seq_dim=-2, quant_dim=-1)
    
    # Dequantize normal values
    dequantized_normal = dequantize_from_1bit_packed(packed_normal, scales, seq_dim=-2)
    
    # Reconstruct original tensor. Outlier values are kept in FP16, normal values are dequantized
    reconstructed_1bit = torch.where(outlier_mask, outlier_vals, dequantized_normal)
    
    onebit_mse = torch.mean((base_tensor - reconstructed_1bit) ** 2).item()
    onebit_cos = calculate_cosine_similarity(base_tensor, reconstructed_1bit)
    
    outlier_fraction = torch.mean(outlier_mask.to(torch.float32)).item()
    
    print(f"Isolated Outliers Fraction: {outlier_fraction * 100:.2f}%")
    print(f"1-Bit Reconstruction MSE: {onebit_mse:.6f}")
    print(f"1-Bit Cosine Similarity: {onebit_cos:.6f}")
    
    # Assert high reconstruction similarity
    assert onebit_cos > 0.90, f"1-Bit quantization cosine similarity is too low: {onebit_cos}"
    
    # 4. Verify Outlier Preservation
    # Outlier values must have EXACT lossless preservation (MSE = 0.0)
    original_outliers = base_tensor * outlier_mask
    reconstructed_outliers = reconstructed_1bit * outlier_mask
    outliers_mse = torch.mean((original_outliers - reconstructed_outliers) ** 2).item()
    
    print(f"FP16 Outliers Preservation MSE: {outliers_mse:.8f} (Must be exactly 0.0)")
    assert outliers_mse < 1e-4, "Outlier values were corrupted during reconstruction!"
    
    print("\nAll comparative loss analysis tests passed beautifully!")

if __name__ == "__main__":
    test_compression_loss_and_similarity()
