import torch
import sys
import os

# Add parent directory to path so we can import core
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.memory_manager import PagedDynamicKVCache

def test_cache_transitions():
    print("Testing 7-Tier PagedDynamicKVCache transitions...")
    
    # Configuration - using page_size = 8 because 1-bit requires multiples of 8
    page_size = 8  
    max_active = 1
    max_fp8 = 1
    max_int8 = 1
    max_int4 = 1
    max_int2 = 1
    max_one_bit = 1
    sink_tokens = 8
    
    # Head dim = 16, Batch = 1, Heads = 1
    cache = PagedDynamicKVCache(
        page_size=page_size,
        max_active_pages=max_active,
        max_fp8_pages=max_fp8,
        max_int8_pages=max_int8,
        max_int4_pages=max_int4,
        max_int2_pages=max_int2,
        max_one_bit_pages=max_one_bit,
        sink_tokens=sink_tokens,
        threshold_sigma=3.0
    )
    
    # 1. Push first 8 tokens (isolates attention sinks)
    print("Pushing 8 tokens (sinks)...")
    k1 = torch.randn(1, 1, 8, 16, dtype=torch.float16)
    v1 = torch.randn(1, 1, 8, 16, dtype=torch.float16)
    cache.push_new_tokens(k1, v1)
    
    assert cache.sink_k is not None
    assert cache.sink_k.shape[-2] == 8
    assert len(cache.active_pages) == 0
    print("Attention Sinks test passed!")
    
    # 2. Push 4 tokens (active buffer)
    print("Pushing 4 tokens (active buffer)...")
    k2 = torch.randn(1, 1, 4, 16, dtype=torch.float16)
    v2 = torch.randn(1, 1, 4, 16, dtype=torch.float16)
    cache.push_new_tokens(k2, v2)
    
    assert cache.k_buffer.shape[-2] == 4
    assert len(cache.active_pages) == 0
    print("Buffer check passed!")
    
    # 3. Push 4 tokens (triggers first FP16 page)
    # Tokens after sinks = 8. Page size = 8.
    print("Pushing 4 tokens to form first FP16 active page...")
    k3 = torch.randn(1, 1, 4, 16, dtype=torch.float16)
    v3 = torch.randn(1, 1, 4, 16, dtype=torch.float16)
    cache.push_new_tokens(k3, v3)
    
    assert len(cache.active_pages) == 1
    assert len(cache.fp8_pages) == 0
    print("First FP16 active page formed!")
    
    # 4. Push 8 more tokens (triggers FP16 -> FP8 transition)
    # Total normal pages = 2. max_active = 1.
    print("Pushing 8 tokens to trigger FP16 -> FP8 transition...")
    k4 = torch.randn(1, 1, 8, 16, dtype=torch.float16)
    v4 = torch.randn(1, 1, 8, 16, dtype=torch.float16)
    cache.push_new_tokens(k4, v4)
    
    assert len(cache.active_pages) == 1
    assert len(cache.fp8_pages) == 1
    assert len(cache.int8_pages) == 0
    print("FP16 -> FP8 transition passed!")
    
    # 5. Push 8 more tokens (triggers FP8 -> INT8 transition)
    print("Pushing 8 tokens to trigger FP8 -> INT8 transition...")
    k5 = torch.randn(1, 1, 8, 16, dtype=torch.float16)
    v5 = torch.randn(1, 1, 8, 16, dtype=torch.float16)
    cache.push_new_tokens(k5, v5)
    
    assert len(cache.active_pages) == 1
    assert len(cache.fp8_pages) == 1
    assert len(cache.int8_pages) == 1
    assert len(cache.int4_pages) == 0
    print("FP8 -> INT8 transition passed!")
    
    # 6. Push 8 more tokens (triggers INT8 -> INT4 transition)
    print("Pushing 8 tokens to trigger INT8 -> INT4 transition...")
    k6 = torch.randn(1, 1, 8, 16, dtype=torch.float16)
    v6 = torch.randn(1, 1, 8, 16, dtype=torch.float16)
    cache.push_new_tokens(k6, v6)
    
    assert len(cache.active_pages) == 1
    assert len(cache.fp8_pages) == 1
    assert len(cache.int8_pages) == 1
    assert len(cache.int4_pages) == 1
    assert len(cache.int2_pages) == 0
    print("INT8 -> INT4 transition passed!")
    
    # 7. Push 8 more tokens (triggers INT4 -> INT2 transition)
    print("Pushing 8 tokens to trigger INT4 -> INT2 transition...")
    k7 = torch.randn(1, 1, 8, 16, dtype=torch.float16)
    v7 = torch.randn(1, 1, 8, 16, dtype=torch.float16)
    cache.push_new_tokens(k7, v7)
    
    assert len(cache.active_pages) == 1
    assert len(cache.fp8_pages) == 1
    assert len(cache.int8_pages) == 1
    assert len(cache.int4_pages) == 1
    assert len(cache.int2_pages) == 1
    assert len(cache.one_bit_pages) == 0
    print("INT4 -> INT2 transition passed!")

    # 8. Push 8 more tokens (triggers INT2 -> 1-Bit transition)
    print("Pushing 8 tokens to trigger INT2 -> 1-Bit transition...")
    k8 = torch.randn(1, 1, 8, 16, dtype=torch.float16)
    v8 = torch.randn(1, 1, 8, 16, dtype=torch.float16)
    cache.push_new_tokens(k8, v8)
    
    assert len(cache.active_pages) == 1
    assert len(cache.fp8_pages) == 1
    assert len(cache.int8_pages) == 1
    assert len(cache.int4_pages) == 1
    assert len(cache.int2_pages) == 1
    assert len(cache.one_bit_pages) == 1
    assert len(cache.jl_pages) == 0
    print("INT2 -> 1-Bit transition passed!")

    # 9. Push 8 more tokens (triggers 1-Bit -> JL-Projection transition)
    print("Pushing 8 tokens to trigger 1-Bit -> JL-Projection transition...")
    k9 = torch.randn(1, 1, 8, 16, dtype=torch.float16)
    v9 = torch.randn(1, 1, 8, 16, dtype=torch.float16)
    cache.push_new_tokens(k9, v9)
    
    assert len(cache.active_pages) == 1
    assert len(cache.fp8_pages) == 1
    assert len(cache.int8_pages) == 1
    assert len(cache.int4_pages) == 1
    assert len(cache.int2_pages) == 1
    assert len(cache.one_bit_pages) == 1
    assert len(cache.jl_pages) == 1
    print("1-Bit -> JL-Projection transition passed!")
    
    # 10. Retrieve and reconstruct all keys/values
    print("Retrieving and reconstructing all keys/values...")
    all_k, all_v = cache.get_all_keys_values()
    
    print(f"Reconstructed K actual shape: {all_k.shape}")
    assert all_k.shape == (1, 1, 64, 16)
    assert all_v.shape == (1, 1, 64, 16)
    
    # Compare with the original concat of keys
    original_k = torch.cat([k1, k2, k3, k4, k5, k6, k7, k8, k9], dim=-2)
    reconstruction_error = torch.mean(torch.abs(original_k - all_k)).item()
    print(f"Total 7-Tier Reconstruction Error: {reconstruction_error:.4f}")
    assert reconstruction_error < 0.8, "Error too high!"
    print("7-Tier Reconstruction check passed!")
    
    # 11. VRAM calculation check
    vram_bytes = cache.get_vram_usage()
    print(f"Calculated VRAM usage: {vram_bytes} bytes")
    assert vram_bytes > 0
    print("VRAM check passed!")

if __name__ == "__main__":
    test_cache_transitions()
    print("All 7-Tier KV Cache tests successfully passed!")
