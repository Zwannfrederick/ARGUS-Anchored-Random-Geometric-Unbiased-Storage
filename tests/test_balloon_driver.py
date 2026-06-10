import torch
import pytest
from argus_cache.core.memory_manager import PagedDynamicKVCache, ArgusConfig
from argus_cache.core.balloon_driver import ElasticCacheBalloonDriver
from argus_cache.backends.eviction import ClockSweepPolicy
from argus_cache.core.tier_registry import PipelineConfig, TierSpec
from argus_cache.backends.quantization import FP8Backend, INT8Backend

def test_balloon_inflate_deflate_lifecycle():
    driver = ElasticCacheBalloonDriver()
    
    # Custom pipeline configuration with 2 tiers
    pipeline = PipelineConfig(
        tiers=[
            TierSpec("fp8", FP8Backend(), max_pages=3, priority=1),
            TierSpec("int8", INT8Backend(), max_pages=4, priority=2)
        ],
        page_size=128,
        sink_tokens=0,
        max_active_pages=3,
        balloon_driver=driver
    )
    
    cache = PagedDynamicKVCache(pipeline=pipeline)
    
    # Register check
    driver._register_cache(cache)
    assert cache.max_active_pages == 3
    assert cache.tier_name_to_spec["fp8"].max_pages == 3
    assert cache.tier_name_to_spec["int8"].max_pages == 4
    
    # Inflate 1
    driver.inflate(cache)
    assert cache.max_active_pages == 2
    assert cache.tier_name_to_spec["fp8"].max_pages == 2
    assert cache.tier_name_to_spec["int8"].max_pages == 3
    
    # Inflate 2
    driver.inflate(cache)
    assert cache.max_active_pages == 1
    assert cache.tier_name_to_spec["fp8"].max_pages == 1
    assert cache.tier_name_to_spec["int8"].max_pages == 2
    
    # Try to inflate past limits (active pages minimum 1)
    driver.inflate(cache)
    assert cache.max_active_pages == 1
    
    # Deflate
    driver.deflate(cache)
    assert cache.max_active_pages == 3
    assert cache.tier_name_to_spec["fp8"].max_pages == 3
    assert cache.tier_name_to_spec["int8"].max_pages == 4

def test_balloon_clock_sweep_threshold_reduction():
    driver = ElasticCacheBalloonDriver()
    policy = ClockSweepPolicy(max_heat=3.0)
    
    pipeline = PipelineConfig(
        tiers=[
            TierSpec("fp8", FP8Backend(), max_pages=2, priority=1)
        ],
        eviction_policy=policy,
        page_size=128,
        sink_tokens=0,
        max_active_pages=3,
        balloon_driver=driver
    )
    
    cache = PagedDynamicKVCache(pipeline=pipeline)
    
    # Push page and simulate accesses to increase heat and reference bits
    k = torch.randn(1, 1, 128, 16)
    v = torch.randn(1, 1, 128, 16)
    cache.push_new_tokens(k, v)
    
    assert len(cache.active_pages) == 1
    page = cache.active_pages[0]
    policy.on_access(page, attention_score=2.0, step=0)
    
    assert page["referenced"] == 1
    assert page["heat_register"] == 2.0
    assert policy.max_heat == 3.0
    
    # Inflate the balloon
    driver.inflate(cache)
    
    # Clock sweep max_heat should be reduced
    assert policy.max_heat == 1.5
    # Referenced bit should be reset
    assert page["referenced"] == 0
    # Heat register should be halved
    assert page["heat_register"] == 1.0
    
    # Deflate
    driver.deflate(cache)
    assert policy.max_heat == 3.0

def test_balloon_oom_integration():
    driver = ElasticCacheBalloonDriver()
    
    # Setup cache with OOM trigger ratio <= 0.0 to force pressure detection on CPU fallback
    cache = PagedDynamicKVCache(
        page_size=128,
        sink_tokens=0,
        max_active_pages=3,
        balloon_driver=driver
    )
    # Force negative threshold ratio to trigger pressure check fallback
    cache.config.vram_oom_threshold_ratio = -1.0
    
    # Initially active pages limit is 3
    assert cache.max_active_pages == 3
    
    # Pushing tokens triggers OOM check
    k = torch.randn(1, 1, 128, 16)
    v = torch.randn(1, 1, 128, 16)
    cache.push_new_tokens(k, v)
    
    # Balloon should have been inflated to reclaim memory
    assert cache.max_active_pages < 3
    
    # Relieve pressure
    cache.config.vram_oom_threshold_ratio = 0.95
    # Force a check again
    cache._check_and_prevent_oom()
    
    # Balloon should deflate
    assert cache.max_active_pages == 3

def test_balloon_multi_tenant_rebalance():
    driver = ElasticCacheBalloonDriver()
    
    cache_active = PagedDynamicKVCache(page_size=128, sink_tokens=0, max_active_pages=3, balloon_driver=driver)
    cache_idle = PagedDynamicKVCache(page_size=128, sink_tokens=0, max_active_pages=3, balloon_driver=driver)
    
    # Put page in both caches
    k = torch.randn(1, 1, 128, 16)
    v = torch.randn(1, 1, 128, 16)
    cache_active.push_new_tokens(k, v)
    cache_idle.push_new_tokens(k, v)
    
    assert len(cache_active.active_pages) == 1
    assert len(cache_idle.active_pages) == 1
    
    # Simulate high activity on active cache by increasing importance
    for page in cache_active.active_pages:
        page["importance_score"] = 10.0
    for page in cache_idle.active_pages:
        page["importance_score"] = 0.1
        
    # Rebalance
    driver.rebalance([cache_active, cache_idle])
    
    # Idle cache balloon inflated
    assert cache_idle.max_active_pages < 3
    # Active cache balloon deflated / normal
    assert cache_active.max_active_pages == 3
