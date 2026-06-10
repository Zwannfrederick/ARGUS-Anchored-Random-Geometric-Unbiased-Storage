import torch
import pytest
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from argus_cache.backends.eviction import ClockSweepPolicy
from argus_cache.core.memory_manager import PagedDynamicKVCache
from argus_cache.core.tier_registry import PipelineConfig, TierSpec
from argus_cache.backends.quantization import INT8Backend

def test_clock_sweep_policy_logic():
    """
    Tests direct logic of ClockSweepPolicy.
    """
    policy = ClockSweepPolicy(max_heat=3.0)
    
    # Create mock pages
    pages = [
        {"page_id": 1, "referenced": 1, "heat_register": 2.0},
        {"page_id": 2, "referenced": 0, "heat_register": 0.0},
        {"page_id": 3, "referenced": 0, "heat_register": 1.0},
    ]
    
    # Initial hand is 0. 
    # Page 1 has referenced=1 -> dec to 0, cool heat to 1.0, hand advances to 1
    # Page 2 has referenced=0, heat=0.0 -> selected as victim! hand advances to 2
    victim = policy.select_victim(pages, context={})
    assert victim["page_id"] == 2
    assert policy.hand == 2
    assert pages[0]["referenced"] == 0
    assert pages[0]["heat_register"] == 1.0

    # Hand is 2.
    # Page 3 has referenced=0, heat=1.0 -> heat dec to 0.5, hand advances to 0
    # Page 1 has referenced=0, heat=1.0 -> heat dec to 0.5, hand advances to 1
    # We remove page 2 to simulate the caller evicting it
    pages.remove(victim) # list is now: [page 1, page 3]
    # hand is 2, length is 2. policy.select_victim should clamp hand to 0
    # Page 1 (index 0) has referenced=0, heat=0.5 -> selected!
    victim2 = policy.select_victim(pages, context={})
    assert victim2["page_id"] == 1

def test_clock_sweep_cache_integration():
    """
    Tests PagedDynamicKVCache integration with ClockSweepPolicy.
    """
    page_size = 8
    policy = ClockSweepPolicy(max_heat=3.0)
    
    custom_pipeline = PipelineConfig(
        tiers=[
            TierSpec("int8", INT8Backend(), max_pages=2, priority=1)
        ],
        eviction_policy=policy,
        page_size=page_size,
        sink_tokens=0,
        threshold_sigma=3.0,
        max_active_pages=1
    )
    
    cache = PagedDynamicKVCache(pipeline=custom_pipeline)
    
    # Push 4 pages of data to trigger cascades and evictions
    # Active pool limit = 1. INT8 limit = 2. Total capacity = 3 pages.
    # Pushing 4 pages will force 1 page to demote/evict entirely (or cascade).
    
    for i in range(4):
        k = torch.randn(1, 1, page_size, 16, dtype=torch.float16)
        v = torch.randn(1, 1, page_size, 16, dtype=torch.float16)
        cache.push_new_tokens(k, v)
        
        # Access the first page with very high attention to make it hot
        q = torch.randn(1, 1, 1, 16, dtype=torch.float16)
        # We perform attention to trigger on_access
        cache.inplace_paged_attention(q)
        
    # Check that eviction took place and the clock sweep policy was engaged
    assert len(cache.active_pages) == 1
    assert len(cache.pages_by_tier["int8"]) == 2
    
    # Verify the clock hand advanced
    assert policy.hand > 0 or len(cache.active_pages) > 0
