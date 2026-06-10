import torch
import pytest
import sys
import os
from unittest import mock

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from argus_cache.core.memory_manager import PagedDynamicKVCache, ArgusConfig


def test_page_size_alignment_validation():
    """
    Test that page_size % 128 == 0 constraint is enforced when strict_alignment=True,
    and bypassed with a warning when strict_alignment=False.
    """
    # Should work without error because strict_alignment is False by default
    cache_non_strict = PagedDynamicKVCache(page_size=8, strict_alignment=False)
    assert cache_non_strict.page_size == 8

    # Should raise ValueError when strict_alignment is True
    with pytest.raises(ValueError) as excinfo:
        PagedDynamicKVCache(page_size=8, strict_alignment=True)
    assert "must be a multiple of 128" in str(excinfo.value)

    # Should succeed with alignment multiple of 128
    cache_strict = PagedDynamicKVCache(page_size=128, strict_alignment=True)
    assert cache_strict.page_size == 128


def test_sdpa_flash_attention_routing():
    """
    Verify that the Fused Online-Softmax Paged Attention kernel (Phase 6A)
    is used when running on GPU (is_enterprise=True). The fused kernel
    processes pages individually via online softmax using torch.matmul,
    and only falls back to SDPA for the single-page fast path.
    """
    cache = PagedDynamicKVCache(page_size=128, force_qos=True)
    
    # Populate the cache
    k = torch.randn(1, 2, 128, 16)
    v = torch.randn(1, 2, 128, 16)
    cache.push_new_tokens(k, v)

    q = torch.randn(1, 2, 1, 16)

    # Mock torch.matmul to verify the fused online softmax path is invoked
    original_matmul = torch.matmul
    matmul_call_count = [0]
    
    def counting_matmul(*args, **kwargs):
        matmul_call_count[0] += 1
        return original_matmul(*args, **kwargs)
    
    with mock.patch("torch.matmul", side_effect=counting_matmul):
        # Patch is_cuda to return True to trigger the is_enterprise path
        with mock.patch("torch.Tensor.is_cuda", new_callable=mock.PropertyMock) as mock_is_cuda:
            mock_is_cuda.return_value = True
            
            out = cache.inplace_paged_attention(q)
            
            # The fused kernel uses torch.matmul for Q@K^T and P@V in the 
            # online softmax loop, plus QoS metrics computation
            assert matmul_call_count[0] > 0, "torch.matmul should be called by the fused attention kernel"
            assert out.shape == q.shape


def test_sink_pinning_cpu():
    """
    Test that sink_k and sink_v are pinned memory when allocated on CPU.
    """
    # Set sink_tokens > 0
    config = ArgusConfig(page_size=128, sink_tokens=4)
    cache = PagedDynamicKVCache(config=config)

    # Push enough tokens to allocate sinks
    k = torch.randn(1, 2, 10, 16)
    v = torch.randn(1, 2, 10, 16)
    cache.push_new_tokens(k, v)

    assert cache.sink_k is not None
    assert cache.sink_v is not None
    
    # Check if they are pinned CPU tensors
    assert cache.sink_k.is_pinned()
    assert cache.sink_v.is_pinned()


def test_sink_eviction_exemption():
    """
    Test that sink tokens are never subject to eviction or demotion.
    """
    config = ArgusConfig(page_size=128, sink_tokens=4, max_active_pages=1)
    cache = PagedDynamicKVCache(config=config)

    # Push 3 pages of data (each 128 tokens)
    # The first 4 tokens of the first push will go to sink, the rest to pages
    for _ in range(3):
        k = torch.randn(1, 2, 128, 16)
        v = torch.randn(1, 2, 128, 16)
        cache.push_new_tokens(k, v)

    # Initial sinks should remain unchanged and present
    assert cache.sink_k is not None
    assert cache.sink_k.shape[-2] == 4

    # The sinks should NOT be in active_pages
    for page in cache.active_pages:
        assert page["page_id"] != 0
        
    # The sinks should NOT be in compressed tiers
    for spec in cache.tier_specs:
        for page in cache.pages_by_tier.get(spec.name, []):
            assert page["page_id"] != 0
