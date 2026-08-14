import torch
import pytest
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from argus_cache.core.memory_manager import PagedDynamicKVCache, ArgusConfig
from argus_cache.core.tier_registry import PipelineConfig, TierSpec
from argus_cache.backends.quantization import INT8Backend, FP8Backend

def test_prefetch_without_weights():
    """
    Tests prefetching fallback when no attention weights are provided.
    It should prefetch the coldest/most recent page from the tiers.
    """
    page_size = 8
    config = ArgusConfig(
        page_size=page_size,
        max_active_pages=1,
        max_fp8_pages=1,
        sink_tokens=0,
        threshold_sigma=3.0
    )
    cache = PagedDynamicKVCache(config=config)
    
    # Push 3 pages of tokens. This will demote pages to FP8/INT8.
    for _ in range(3):
        k = torch.randn(1, 1, page_size, 16, dtype=torch.float16)
        v = torch.randn(1, 1, page_size, 16, dtype=torch.float16)
        cache.push_new_tokens(k, v)
        
    assert len(cache.active_pages) == 1
    assert len(cache.pages_by_tier['fp8']) == 1
    assert len(cache.pages_by_tier['int8']) == 1
    
    # Run prefetch without weights
    cache.speculate_and_prefetch()
    
    # Check that prefetch cache is populated
    assert len(cache.prefetch_cache) > 0
    
    # Retrieve the page IDs from prefetch cache
    prefetched_page_ids = list(cache.prefetch_cache.keys())
    
    # Ensure the pages in FP8/INT8 are prefetched
    fp8_page = cache.pages_by_tier['fp8'][0]
    int8_page = cache.pages_by_tier['int8'][0]
    assert fp8_page.get('page_id') in prefetched_page_ids or int8_page.get('page_id') in prefetched_page_ids
    
    # Run attention and verify prefetch_hits
    q = torch.randn(1, 1, 1, 16, dtype=torch.float16)
    initial_hits = cache.prefetch_hits
    
    out = cache.inplace_paged_attention(q)
    assert out is not None
    assert cache.prefetch_hits > initial_hits

def test_prefetch_with_weights():
    """
    Tests prefetching guided by attention weights.
    We pass attention weights favoring a specific page, and verify it gets prefetched.
    """
    page_size = 8
    config = ArgusConfig(
        page_size=page_size,
        max_active_pages=1,
        max_fp8_pages=1,
        sink_tokens=0,
        threshold_sigma=3.0
    )
    cache = PagedDynamicKVCache(config=config)
    
    # Push 3 pages of tokens.
    for _ in range(3):
        k = torch.randn(1, 1, page_size, 16, dtype=torch.float16)
        v = torch.randn(1, 1, page_size, 16, dtype=torch.float16)
        cache.push_new_tokens(k, v)
        
    # Get the pages
    # reverse(tier_specs) order: int8, then fp8.
    # index 0 of all_p is the int8 page.
    # index 1 of all_p is the fp8 page.
    int8_page = cache.pages_by_tier['int8'][0]
    fp8_page = cache.pages_by_tier['fp8'][0]
    
    # Construct attention weights
    # Total tokens in cache: 3 pages * 8 tokens = 24 tokens.
    # Shape should be (batch, num_heads, q_len, kv_len) -> (1, 1, 1, 24)
    # We want to give high weights to the int8 page (idx 0 in all_p, which corresponds to tokens 0 to 8)
    attn_weights = torch.zeros(1, 1, 1, 24, dtype=torch.float16)
    attn_weights[0, 0, 0, 0:8] = 1.0  # High attention to INT8 page
    attn_weights[0, 0, 0, 8:16] = 0.0 # Low attention to FP8 page
    
    cache.speculate_and_prefetch(attn_weights)
    
    # Check that INT8 page is prefetched
    assert int8_page.get('page_id') in cache.prefetch_cache
    
    # Retrieve prefetch hit count
    initial_hits = cache.prefetch_hits
    q = torch.randn(1, 1, 1, 16, dtype=torch.float16)
    out = cache.inplace_paged_attention(q)
    assert out is not None
    assert cache.prefetch_hits > initial_hits
