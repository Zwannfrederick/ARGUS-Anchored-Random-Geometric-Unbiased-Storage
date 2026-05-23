import torch
import sys
import os

# Add parent directory to path so we can import core
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.quantization import (
    quantize_to_int8,
    dequantize_from_int8,
    quantize_to_int4_packed,
    dequantize_from_int4_packed
)

def test_int8_quantization():
    print("Testing INT8 Quantization...")
    # Shape: (batch=1, heads=2, seq_len=10, head_dim=64)
    x = torch.randn(1, 2, 10, 64, dtype=torch.float16, device="cuda" if torch.cuda.is_available() else "cpu")
    
    quantized, scales = quantize_to_int8(x)
    dequantized = dequantize_from_int8(quantized, scales)
    
    # Calculate error
    max_error = torch.max(torch.abs(x - dequantized)).item()
    mean_error = torch.mean(torch.abs(x - dequantized)).item()
    
    print(f"INT8 Max Error: {max_error:.4f}, Mean Error: {mean_error:.4f}")
    assert mean_error < 0.05, f"INT8 quantization error too high: {mean_error}"
    print("INT8 Quantization Test PASSED!")

def test_int4_quantization():
    print("Testing INT4 Packed Quantization...")
    # Shape: (batch=1, heads=2, seq_len=8, head_dim=64) - seq_len must be even
    x = torch.randn(1, 2, 8, 64, dtype=torch.float16, device="cuda" if torch.cuda.is_available() else "cpu")
    
    packed, scales, min_vals = quantize_to_int4_packed(x, seq_dim=-2, quant_dim=-1)
    
    # Assert packed sequence length is half of original
    assert packed.shape[-2] == x.shape[-2] // 2, f"Packed shape mismatch: {packed.shape}"
    
    dequantized = dequantize_from_int4_packed(packed, scales, min_vals, seq_dim=-2)
    
    assert dequantized.shape == x.shape, f"Dequantized shape mismatch: {dequantized.shape} vs {x.shape}"
    
    # Calculate error
    max_error = torch.max(torch.abs(x - dequantized)).item()
    mean_error = torch.mean(torch.abs(x - dequantized)).item()
    
    print(f"INT4 Max Error: {max_error:.4f}, Mean Error: {mean_error:.4f}")
    assert mean_error < 0.25, f"INT4 quantization error too high: {mean_error}"
    print("INT4 Quantization Test PASSED!")

if __name__ == "__main__":
    # Ensure CUDA is used if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running tests on device: {device}")
    test_int8_quantization()
    test_int4_quantization()
    print("All quantization tests successfully passed!")
