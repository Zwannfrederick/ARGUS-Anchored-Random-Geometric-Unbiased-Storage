"""CPU spill must round-trip page data and be idempotent.

Spilling twice, or restoring without spilling, must not corrupt state -- this
runs under memory pressure, which is exactly when retry logic fires.
"""

import torch

from argus_cache.backends.eviction import ImportanceSortPolicy
from argus_cache.core.memory_manager import PagedDynamicKVCache
from argus_cache.core.tier_registry import PipelineConfig, TierSpec

PAGE = 16


def _cache():
    return PagedDynamicKVCache(
        pipeline=PipelineConfig(
            tiers=[TierSpec(name="int4", backend="int4", max_pages=8)],
            eviction_policy=ImportanceSortPolicy(),
            page_size=PAGE,
            sink_tokens=0,
            max_active_pages=1,
        )
    )


def _fill(cache, n=4):
    for _ in range(n):
        k = torch.randn(1, 1, PAGE, 16, dtype=torch.float16)
        cache.push_new_tokens(k, torch.randn_like(k))


def test_spill_out_then_in_preserves_page_count():
    cache = _cache()
    _fill(cache)
    before = cache._cpp_manager.get_page_count()

    cache.swap_out_to_host()
    cache.swap_in_to_device("cpu")

    assert cache._cpp_manager.get_page_count() == before


def test_double_spill_is_idempotent():
    cache = _cache()
    _fill(cache)

    cache.swap_out_to_host()
    cache.swap_out_to_host()

    assert cache.is_swapped_out


def test_restore_without_spill_is_a_noop():
    cache = _cache()
    _fill(cache)
    before = cache._cpp_manager.get_page_count()

    cache.swap_in_to_device("cpu")

    assert cache._cpp_manager.get_page_count() == before


def test_spill_clears_the_swapped_out_flag_on_restore():
    cache = _cache()
    _fill(cache)

    cache.swap_out_to_host()
    assert cache.is_swapped_out
    cache.swap_in_to_device("cpu")

    assert not cache.is_swapped_out


def test_spill_round_trip_preserves_active_page_content():
    """The whole point of spill is that it is lossless -- it moves bytes to
    host memory, it does not compress. If content changes, a spilled page
    silently degrades on restore."""
    cache = _cache()
    _fill(cache)
    before = [p["key"].clone() for p in cache.active_pages]

    cache.swap_out_to_host()
    cache.swap_in_to_device("cpu")

    after = [p["key"] for p in cache.active_pages]
    assert len(after) == len(before)
    for a, b in zip(after, before):
        assert torch.equal(a, b), "spill round trip altered an active page"
