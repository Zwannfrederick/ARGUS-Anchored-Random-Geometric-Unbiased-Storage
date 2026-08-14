import torch
import pytest
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from argus_cache.core.memory_manager import PagedDynamicKVCache, ArgusConfig
from argus_cache.core.tier_registry import PipelineConfig, TierSpec
from argus_cache.backends.quantization import INT8Backend
import argus_cpp_backend

def test_variable_granularity_splitting():
    """
    Tests that a Mega-page of size page_size is correctly split into micro-pages
    when its importance score drops below the split threshold (< 0.5).
    """
    page_size = 8
    micro_size = 2
    
    config = ArgusConfig(
        page_size=page_size,
        max_active_pages=3,
        sink_tokens=0,
        threshold_sigma=3.0,
        micro_page_size=micro_size
    )
    cache = PagedDynamicKVCache(config=config)
    
    # Push 2 pages worth of data (16 tokens)
    for _ in range(2):
        k = torch.randn(1, 1, page_size, 16, dtype=torch.float16)
        v = torch.randn(1, 1, page_size, 16, dtype=torch.float16)
        cache.push_new_tokens(k, v)
        
    assert len(cache.active_pages) == 2
    assert cache.active_pages[0]['page_size'] == page_size
    
    # Force first page to be cold (importance < 0.5)
    cache.active_pages[0]['importance_score'] = 0.1
    cache.active_pages[1]['importance_score'] = 1.0  # Warm, shouldn't split
    
    # Run granularity management
    cache.manage_variable_granularity()
    
    # First page (size 8) splits into 4 micro-pages (size 2).
    # Second page (size 8) remains unsplit.
    # Total active pages: 4 (splits) + 1 (original) = 5
    assert len(cache.active_pages) == 5
    
    # Check that split micro-pages have correct page_size and shape
    for i in range(4):
        p = cache.active_pages[i]
        assert p['page_size'] == micro_size
        assert p['key'].shape[-2] == micro_size
        assert p['value'].shape[-2] == micro_size
        
    # Unsplit page is at index 4
    assert cache.active_pages[4]['page_size'] == page_size
    assert cache.active_pages[4]['key'].shape[-2] == page_size

def test_variable_granularity_merging():
    """
    Tests that contiguous micro-pages of size micro_page_size are merged back into
    a single Mega-page when they become hot (importance > 1.5).
    """
    page_size = 8
    micro_size = 2
    
    config = ArgusConfig(
        page_size=page_size,
        max_active_pages=5,
        sink_tokens=0,
        threshold_sigma=3.0,
        micro_page_size=micro_size
    )
    cache = PagedDynamicKVCache(config=config)
    
    # Set up active pages with 4 micro-pages (totaling 8 tokens)
    # To bypass push_new_tokens allocating standard mega-pages, we manually push them.
    for i in range(4):
        k = torch.randn(1, 1, micro_size, 16, dtype=torch.float16)
        v = torch.randn(1, 1, micro_size, 16, dtype=torch.float16)
        page = argus_cpp_backend.create_page()
        page['page_id'] = i + 1
        page['key'] = k
        page['value'] = v
        page['pool_idx'] = -1
        page['importance_score'] = 2.0
        page['page_size'] = micro_size
        cache.active_pages = cache.active_pages + [page]
        
    assert len(cache.active_pages) == 4
    
    # Run granularity management
    cache.manage_variable_granularity()
    
    # 4 micro-pages of size 2 merge into 1 Mega-page of size 8
    assert len(cache.active_pages) == 1
    merged = cache.active_pages[0]
    assert merged['page_size'] == page_size
    assert merged['key'].shape[-2] == page_size
    assert merged['value'].shape[-2] == page_size

def test_mixed_granularity_attention():
    """
    Verifies that attention works over mixed page sizes (mega and micro pages)
    and dynamically constructs page_offsets properly.
    """
    page_size = 8
    micro_size = 2
    
    config = ArgusConfig(
        page_size=page_size,
        max_active_pages=10,
        sink_tokens=2, # Include sinks
        threshold_sigma=3.0,
        micro_page_size=micro_size
    )
    cache = PagedDynamicKVCache(config=config)
    
    # Add sink tokens
    cache.sink_k = torch.randn(1, 1, 2, 16, dtype=torch.float16, device='cuda')
    cache.sink_v = torch.randn(1, 1, 2, 16, dtype=torch.float16, device='cuda')
    
    page1 = argus_cpp_backend.create_page()
    page1['page_id'] = 1
    page1['key'] = torch.randn(1, 1, page_size, 16, dtype=torch.float16, device='cuda')
    page1['value'] = torch.randn(1, 1, page_size, 16, dtype=torch.float16, device='cuda')
    page1['pool_idx'] = -1
    page1['importance_score'] = 1.0
    page1['page_size'] = page_size
    cache.active_pages = cache.active_pages + [page1]
    
    page2 = argus_cpp_backend.create_page()
    page2['page_id'] = 2
    page2['key'] = torch.randn(1, 1, micro_size, 16, dtype=torch.float16, device='cuda')
    page2['value'] = torch.randn(1, 1, micro_size, 16, dtype=torch.float16, device='cuda')
    page2['pool_idx'] = -1
    page2['importance_score'] = 1.0
    page2['page_size'] = micro_size
    cache.active_pages = cache.active_pages + [page2]
    
    # Run attention
    q = torch.randn(1, 1, 1, 16, dtype=torch.float16, device='cuda')
    out = cache.inplace_paged_attention(q)
    
    assert out is not None
    assert out.shape == (1, 1, 1, 16)
    
    # Check page offsets tensor
    # Order:
    # 1. Sinks (length 2) -> start offset 0
    # 2. Active Mega-page (length 8) -> start offset 2
    # 3. Active Micro-page (length 2) -> start offset 10
    # Expected page_offsets: [0, 2, 10]
    expected_offsets = torch.tensor([0, 2, 10], dtype=torch.long)
    assert torch.equal(cache.page_offsets.cpu(), expected_offsets)
