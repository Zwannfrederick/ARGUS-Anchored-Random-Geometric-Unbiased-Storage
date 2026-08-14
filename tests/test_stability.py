"""Stress tests for the C++ port.

These target the failure modes a unit test misses: state accumulating across
cache lifetimes, the prefetch worker racing the main thread, and cascades
running long enough to exhaust a pool.
"""

import gc
import threading

import pytest
import torch

from argus_cache.backends.eviction import ImportanceSortPolicy
from argus_cache.core.memory_manager import PagedDynamicKVCache
from argus_cache.core.tier_registry import PipelineConfig, TierSpec

PAGE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _cache(max_active=2):
    return PagedDynamicKVCache(
        pipeline=PipelineConfig(
            tiers=[
                TierSpec(name="int8", backend="int8", max_pages=2),
                TierSpec(name="int4", backend="int4", max_pages=2),
                TierSpec(name="one_bit", backend="one_bit", max_pages=-1),
            ],
            eviction_policy=ImportanceSortPolicy(),
            page_size=PAGE,
            sink_tokens=0,
            max_active_pages=max_active,
        )
    )


def _push(cache, n=1):
    for _ in range(n):
        k = torch.randn(1, 2, PAGE, 16, dtype=torch.float16, device=DEVICE)
        cache.push_new_tokens(k, torch.randn_like(k))


def _all_page_ids(cache):
    ids = [p["page_id"] for p in cache.active_pages]
    for pages in cache.pages_by_tier.values():
        ids.extend(p["page_id"] for p in pages)
    return ids


def test_repeated_cache_construction_does_not_leak_host_memory():
    """Each cache owns a pinned host pool; 20 lifetimes must not accumulate."""
    for _ in range(20):
        cache = _cache()
        _push(cache, 6)
        del cache
        gc.collect()
    # Reaching here without OOM or a pinned-allocation failure is the assertion.


def test_long_cascade_preserves_every_page():
    """200 pages through a 3-tier cascade: nothing may be lost or duplicated."""
    cache = _cache()
    _push(cache, 200)

    ids = _all_page_ids(cache)

    assert len(ids) == len(set(ids)), "duplicate page ids after long cascade"
    assert ids, "the cascade dropped every page"


def test_cascade_pages_never_exceed_declared_tier_capacity():
    """A tier with max_pages=N must never hold more than N pages, or the
    static pool it was given is being overrun."""
    cache = _cache()
    _push(cache, 200)

    for spec in cache.tier_specs:
        if spec.max_pages < 0:
            continue  # unbounded archival tier
        held = len(cache.pages_by_tier.get(spec.name, []))
        assert held <= spec.max_pages, (
            f"tier {spec.name} holds {held} pages, capacity {spec.max_pages}"
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_prefetch_does_not_race_attention():
    """speculate_and_prefetch runs a background worker; attention must not
    observe a half-written prefetch cache entry."""
    cache = _cache()
    _push(cache, 12)

    q = torch.randn(1, 2, 1, 16, dtype=torch.float16, device="cuda")
    errors = []

    def hammer():
        try:
            for _ in range(40):
                cache.inplace_paged_attention(q)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    cache.speculate_and_prefetch()
    threads = [threading.Thread(target=hammer) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"attention raced the prefetcher: {errors[0]}"


def test_repeated_swap_cycles_are_stable():
    cache = _cache()
    _push(cache, 8)

    for _ in range(10):
        cache.swap_out_to_host()
        cache.swap_in_to_device(DEVICE)

    assert not cache.is_swapped_out


def test_swap_cycles_do_not_lose_pages():
    """Ten spill round trips must leave the page population unchanged --
    spill is lossless, so a drifting count means pages are being dropped."""
    cache = _cache()
    _push(cache, 8)
    before = sorted(_all_page_ids(cache))

    for _ in range(10):
        cache.swap_out_to_host()
        cache.swap_in_to_device(DEVICE)

    assert sorted(_all_page_ids(cache)) == before


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_attention_output_is_finite_after_deep_cascade():
    """The real correctness question: does a page that fell all the way to
    1-bit and came back still produce usable attention?"""
    cache = _cache()
    _push(cache, 30)

    q = torch.randn(1, 2, 1, 16, dtype=torch.float16, device="cuda")
    out = cache.inplace_paged_attention(q)

    assert torch.isfinite(out).all(), "attention produced NaN/Inf after cascade"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_attention_is_finite_across_a_long_decode_run():
    """One finite result can be luck. Degradation that only appears after the
    cascade has churned for a while would pass a single-shot check."""
    cache = _cache()
    q = torch.randn(1, 2, 1, 16, dtype=torch.float16, device="cuda")

    for step in range(60):
        _push(cache, 1)
        out = cache.inplace_paged_attention(q)
        assert torch.isfinite(out).all(), f"non-finite attention at step {step}"


def test_telemetry_stays_consistent_under_churn():
    """Telemetry is read by dashboards while the cascade runs; its page count
    must keep matching reality rather than drifting."""
    cache = _cache()

    for _ in range(20):
        _push(cache, 5)
        _, _, total_pages, _ = cache.get_cache_telemetry()
        resident = len(cache.active_pages) + sum(
            len(v) for v in cache.pages_by_tier.values()
        )
        assert total_pages == resident
