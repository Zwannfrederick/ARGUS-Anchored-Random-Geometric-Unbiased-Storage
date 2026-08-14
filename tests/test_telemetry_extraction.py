"""Telemetry must survive extraction unchanged.

This is a refactor, so the contract is that nothing observable changes. These
tests pin the current output shape before the move so the move can be verified
rather than assumed.
"""

import torch

from argus_cache.backends.eviction import ImportanceSortPolicy
from argus_cache.core.memory_manager import PagedDynamicKVCache
from argus_cache.core.tier_registry import PipelineConfig, TierSpec

PAGE = 16


def _cache():
    return PagedDynamicKVCache(
        pipeline=PipelineConfig(
            tiers=[
                TierSpec(name="fp8", backend="fp8", max_pages=1),
                TierSpec(name="int4", backend="int4", max_pages=4),
            ],
            eviction_policy=ImportanceSortPolicy(),
            page_size=PAGE,
            sink_tokens=0,
            max_active_pages=1,
        )
    )


def _fill(cache, pages=4):
    for _ in range(pages):
        k = torch.randn(1, 1, PAGE, 16, dtype=torch.float16)
        cache.push_new_tokens(k, torch.randn_like(k))


def test_telemetry_returns_the_documented_four_tuple():
    """get_cache_telemetry returns a positional tuple, not a dict:
    (compression_ratio, bandwidth_saved_pct, total_pages, compressed_bytes).
    Callers unpack it positionally, so the arity and order are the contract."""
    cache = _cache()
    _fill(cache)

    ratio, saved_pct, total_pages, compressed_bytes = cache.get_cache_telemetry()

    assert ratio > 1.0, "a cache with compressed tiers must beat fp16"
    assert 0.0 <= saved_pct < 100.0
    assert total_pages > 0
    assert compressed_bytes > 0


def test_telemetry_page_count_matches_manager_state():
    cache = _cache()
    _fill(cache, pages=4)

    _, _, total_pages, _ = cache.get_cache_telemetry()

    resident = len(cache.active_pages) + sum(
        len(pages) for pages in cache.pages_by_tier.values()
    )
    assert total_pages == resident


def test_telemetry_bandwidth_saved_follows_from_ratio():
    """bandwidth_saved is derived as (1 - 1/ratio) * 100; the extraction must
    not desynchronize the two."""
    cache = _cache()
    _fill(cache)

    ratio, saved_pct, _, _ = cache.get_cache_telemetry()

    assert saved_pct == (1.0 - 1.0 / ratio) * 100.0


def test_print_telemetry_summary_produces_output(capsys):
    cache = _cache()
    _fill(cache)

    cache.print_telemetry_summary()

    assert capsys.readouterr().out.strip(), "summary produced no output"


def test_vram_usage_returns_bytes_as_an_int():
    """get_vram_usage returns a single byte count, not a report dict."""
    cache = _cache()
    _fill(cache)

    usage = cache.get_vram_usage()

    assert isinstance(usage, int)
    assert usage > 0


def test_vram_usage_agrees_with_telemetry_byte_count():
    cache = _cache()
    _fill(cache)

    _, _, _, compressed_bytes = cache.get_cache_telemetry()

    assert cache.get_vram_usage() == compressed_bytes


def test_allocator_fragmentation_report_has_pcie_latency_fields():
    cache = _cache()
    _fill(cache)

    report = cache.get_allocator_fragmentation_report()

    assert "pcie_avg_latency_ms" in report
    assert "pcie_p95_latency_ms" in report


def test_telemetry_survives_with_no_pages():
    """Telemetry is polled by dashboards before any token is pushed."""
    cache = _cache()

    assert cache.get_cache_telemetry() is not None
    assert cache.get_vram_usage() is not None
