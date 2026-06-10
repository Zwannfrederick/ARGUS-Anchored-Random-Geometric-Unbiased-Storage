import torch
import pytest
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from argus_cache.core.memory_manager import PagedDynamicKVCache, ArgusConfig
from argus_cache.core.tier_registry import TierSpec
from argus_cache.backends.quantization import INT8Backend

def test_dynamic_add_remove_tier():
    page_size = 16
    # Create a cache with small limits to trigger cascades easily
    config = ArgusConfig(
        page_size=page_size,
        max_active_pages=1,
        max_fp8_pages=1,
        sink_tokens=0,
        threshold_sigma=3.0
    )
    cache = PagedDynamicKVCache(config=config)
    
    # Verify default tiers
    assert any(s.name == 'fp8' for s in cache.tier_specs)
    assert any(s.name == 'int8' for s in cache.tier_specs)
    
    # Push 3 pages of data
    for i in range(3):
        k = torch.randn(1, 1, page_size, 16, dtype=torch.float16)
        v = torch.randn(1, 1, page_size, 16, dtype=torch.float16)
        cache.push_new_tokens(k, v)
        
    # Check page distribution
    assert len(cache.active_pages) == 1
    assert len(cache.pages_by_tier['fp8']) == 1
    assert len(cache.pages_by_tier['int8']) == 1
    
    # 1. Dynamically add a custom tier
    custom_spec = TierSpec(
        name="custom_int8",
        backend=INT8Backend(),
        max_pages=2,
        priority=10,
        use_static_pool=False, # Use dynamic tensors for this custom tier
        use_outlier_isolation=True
    )
    
    # Insert custom_int8 as tier 2 (after fp8 and int8)
    cache.add_tier(custom_spec, index=2)
    
    assert cache.tier_specs[2].name == "custom_int8"
    assert "custom_int8" in cache.pages_by_tier
    
    # Push 3 more pages to trigger demotion down the new pipeline
    for i in range(3):
        k = torch.randn(1, 1, page_size, 16, dtype=torch.float16)
        v = torch.randn(1, 1, page_size, 16, dtype=torch.float16)
        cache.push_new_tokens(k, v)
        
    # Verify the custom tier now contains pages
    assert len(cache.pages_by_tier['custom_int8']) > 0
    
    # 2. Dynamically remove the custom tier
    cache.remove_tier("custom_int8")
    
    assert "custom_int8" not in cache.tier_name_to_spec
    assert "custom_int8" not in cache.pages_by_tier
    assert not any(s.name == "custom_int8" for s in cache.tier_specs)
    
    # Verify the cache functions normally and attention runs without issues
    q = torch.randn(1, 1, 1, 16, dtype=torch.float16)
    out = cache.inplace_paged_attention(q)
    assert out is not None
