"""Invariants for the experimental variable-granularity subsystem.

Splitting and merging move token data between page objects. The invariants
that must hold regardless of tier: no token is lost, no token is duplicated,
and micro-page sizes stay compatible with sub-byte packing.
"""

import pytest
import torch

from argus_cache.backends.eviction import ImportanceSortPolicy
from argus_cache.core.memory_manager import PagedDynamicKVCache
from argus_cache.core.tier_registry import PipelineConfig, TierSpec

PAGE = 64


def _cache(micro=16):
    return PagedDynamicKVCache(
        pipeline=PipelineConfig(
            tiers=[TierSpec(name="int4", backend="int4", max_pages=8)],
            eviction_policy=ImportanceSortPolicy(),
            page_size=PAGE,
            sink_tokens=0,
            max_active_pages=4,
            micro_page_size=micro,
        )
    )


def _one_page(cache):
    k = torch.randn(1, 1, PAGE, 16, dtype=torch.float16)
    cache.push_new_tokens(k, torch.randn_like(k))
    return cache.active_pages[0]


def test_split_preserves_total_token_count():
    cache = _cache()
    page = _one_page(cache)

    parts = cache.split_page(page)

    assert sum(p["page_size"] for p in parts) == PAGE


def test_split_assigns_unique_page_ids():
    """Split pages draw IDs from the C++ counter; colliding IDs would make the
    prefetch cache return another page's data."""
    cache = _cache()
    page = _one_page(cache)

    parts = cache.split_page(page)

    ids = [p["page_id"] for p in parts]
    assert len(ids) == len(set(ids)), f"duplicate page ids after split: {ids}"


def test_split_page_ids_do_not_collide_with_live_pages():
    """A split part reusing a resident page's id would alias it in every
    id-keyed structure (prefetch cache, attention gather, telemetry)."""
    cache = _cache()
    page = _one_page(cache)
    live = {p["page_id"] for p in cache.active_pages}

    parts = cache.split_page(page)

    fresh = {p["page_id"] for p in parts} - {page["page_id"]}
    assert not (fresh & live), "split reused a live page id"


def test_merge_restores_the_original_token_count():
    cache = _cache()
    page = _one_page(cache)
    parts = cache.split_page(page)

    merged = cache.merge_pages(parts)

    assert merged["page_size"] == PAGE


def test_merge_recovers_the_original_content():
    """Split then merge is the identity on an uncompressed active page. If it
    is not, variable granularity is silently corrupting the cache."""
    cache = _cache()
    page = _one_page(cache)
    original_k = page["key"].clone()

    merged = cache.merge_pages(cache.split_page(page))

    assert torch.allclose(merged["key"], original_k, atol=1e-3)


@pytest.mark.parametrize("micro", [8, 16, 32])
def test_micro_page_size_stays_packing_compatible(micro):
    """Sub-byte tiers pack 8/4/2 elements per byte; a micro-page size that is
    not a multiple of 8 crashes the first time one cascades into one_bit."""
    cache = _cache(micro=micro)
    assert cache.micro_page_size % 8 == 0
