"""The streaming decode prototype must remain exact relative to SDPA."""

import pytest
import torch

from argus_cache.core.memory_manager import PagedDynamicKVCache
from argus_cache.core.tier_registry import PipelineConfig, TierSpec


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _cache(streaming: bool):
    return PagedDynamicKVCache(
        pipeline=PipelineConfig(
            tiers=[TierSpec("fp8", "fp8", max_pages=-1)],
            page_size=128,
            sink_tokens=0,
            max_active_pages=1,
            streaming_attention=streaming,
        )
    )


def test_streaming_attention_matches_sdpa_with_compressed_pages():
    torch.manual_seed(7)
    reference = _cache(False)
    streaming = _cache(True)
    for _ in range(3):
        keys = torch.randn(1, 2, 128, 32, device="cuda", dtype=torch.float16)
        values = torch.randn_like(keys)
        reference.push_new_tokens(keys.clone(), values.clone())
        streaming.push_new_tokens(keys.clone(), values.clone())

    query = torch.randn(1, 2, 1, 32, device="cuda", dtype=torch.float16)
    expected = reference.inplace_paged_attention(query)
    actual = streaming.inplace_paged_attention(query)

    torch.testing.assert_close(actual, expected, rtol=3e-3, atol=3e-3)
    assert len(streaming.pages_by_tier["fp8"]) == 2


def test_streaming_attention_supports_grouped_query_attention():
    cache = _cache(True)
    keys = torch.randn(1, 1, 128, 32, device="cuda", dtype=torch.float16)
    values = torch.randn_like(keys)
    cache.push_new_tokens(keys, values)
    query = torch.randn(1, 4, 1, 32, device="cuda", dtype=torch.float16)

    actual = cache.inplace_paged_attention(query)
    expected = torch.nn.functional.scaled_dot_product_attention(
        query, keys, values, enable_gqa=True
    )

    torch.testing.assert_close(actual, expected, rtol=3e-3, atol=3e-3)
