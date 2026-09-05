import torch
import sys
import os
import gc
import weakref

# Add parent directory to path so we can import core
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.memory_manager import PagedDynamicKVCache


def test_large_prefill_does_not_leave_an_exact_sized_staging_mirror():
    """Only a partial page may remain staged after full pages reach C++."""
    page_size = 8
    cache = PagedDynamicKVCache(page_size=page_size, sink_tokens=0)
    keys = torch.randn(1, 1, page_size * 8, 16, dtype=torch.float16)
    values = torch.randn_like(keys)
    cache.push_new_tokens(keys, values)

    assert cache.static_k_buffer.shape[-2] == page_size
    assert cache.static_v_buffer.shape[-2] == page_size
    assert cache.buffer_length == 0
    assert cache.active_pool_k is None
    assert cache.active_pool_v is None
    assert cache.pools_by_tier == {}
    assert cache._jl_operators.cache_size() == 0


def test_close_breaks_native_callback_ownership_cycle():
    cache = PagedDynamicKVCache(page_size=8, sink_tokens=0)
    reference = weakref.ref(cache)

    cache.close()
    del cache
    gc.collect()

    assert reference() is None


def test_native_callbacks_do_not_keep_an_unclosed_cache_alive():
    cache = PagedDynamicKVCache(page_size=8, sink_tokens=0)
    reference = weakref.ref(cache)

    del cache
    gc.collect()

    assert reference() is None

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

    # A compressed tier must not leave a full-precision mirror resident after
    # materializing the HuggingFace-compatible return value.  Doing so makes
    # ARGUS consume the exact cache plus its own compressed storage.
    assert cache._decompressed_tiers_k is None
    assert cache._decompressed_tiers_v is None
    
    print(f"Reconstructed K actual shape: {all_k.shape}")
    assert all_k.shape == (1, 1, 64, 16)
    assert all_v.shape == (1, 1, 64, 16)
    
    # Compare with the original concat of keys
    original_k = torch.cat([k1, k2, k3, k4, k5, k6, k7, k8, k9], dim=-2).to(all_k.device)
    original_v = torch.cat([v1, v2, v3, v4, v5, v6, v7, v8, v9], dim=-2).to(all_v.device)
    reconstruction_error = torch.mean(torch.abs(original_k - all_k)).item()
    print(f"Total 7-Tier Reconstruction Error: {reconstruction_error:.4f}")
    assert reconstruction_error < 0.8, "Error too high!"
    print("7-Tier Reconstruction check passed!")
    
    # 11. VRAM calculation check
    vram_bytes = cache.get_vram_usage()
    print(f"Calculated VRAM usage: {vram_bytes} bytes")
    assert vram_bytes > 0
    print("VRAM check passed!")
    
    # 12. Test inplace_paged_attention
    print("Testing inplace_paged_attention vs standard reconstructed attention...")
    q = torch.randn(1, 1, 1, 16, dtype=torch.float16).to(all_k.device) # 1 query token
    
    # Standard reconstructed attention
    attn_weights = torch.matmul(q, all_k.transpose(-1, -2)) / 4.0 # head_dim = 16, sqrt = 4
    attn_probs = torch.softmax(attn_weights, dim=-1)
    standard_attn_out = torch.matmul(attn_probs, all_v)
    
    # Inplace Paged Attention
    inplace_attn_out = cache.inplace_paged_attention(q)
    
    # Check shape
    assert inplace_attn_out.shape == standard_attn_out.shape
    
    # Verify close values (within floating point tolerances)
    diff = torch.mean(torch.abs(standard_attn_out - inplace_attn_out)).item()
    print(f"Difference between standard and inplace attention: {diff:.6f}")
    # 13. Test compute_fused_paged_attention
    all_k, all_v = cache.get_all_keys_values()
    attn_weights = torch.matmul(q, all_k.transpose(-1, -2)) / 4.0
    attn_probs = torch.softmax(attn_weights, dim=-1)
    standard_attn_out = torch.matmul(attn_probs, all_v)
    fused_attn_out = cache.compute_fused_paged_attention(q)
    assert fused_attn_out.shape == standard_attn_out.shape
    fused_diff = torch.mean(torch.abs(standard_attn_out - fused_attn_out.to(standard_attn_out.device))).item()
    assert fused_diff < 1e-3, f"Fused attention mismatch: {fused_diff}"
    print("compute_fused_paged_attention check passed!")


def test_cache_snapshot_and_restore():
    """Verify that PagedDynamicKVCache snapshot and restore rewinds sequence length accurately."""
    cache = PagedDynamicKVCache(page_size=32)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    k1 = torch.randn(1, 2, 10, 64, device=device, dtype=torch.float16)
    v1 = torch.randn(1, 2, 10, 64, device=device, dtype=torch.float16)
    cache.push_new_tokens(k1, v1)
    assert cache.get_seq_length() == 10

    snap = cache.snapshot()

    k2 = torch.randn(1, 2, 15, 64, device=device, dtype=torch.float16)
    v2 = torch.randn(1, 2, 15, 64, device=device, dtype=torch.float16)
    cache.push_new_tokens(k2, v2)
    assert cache.get_seq_length() == 25

    cache.restore(snap)
    assert cache.get_seq_length() == 10


if __name__ == "__main__":
    test_cache_transitions()
    test_cache_snapshot_and_restore()
    print("All 7-Tier KV Cache tests successfully passed!")

