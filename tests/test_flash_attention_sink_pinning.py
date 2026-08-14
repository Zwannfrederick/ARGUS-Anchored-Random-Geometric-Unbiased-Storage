import torch
import pytest
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from argus_cache.core.memory_manager import PagedDynamicKVCache, ArgusConfig


def test_page_size_alignment_validation():
    """
    Test that page_size % 128 == 0 constraint is enforced when strict_alignment=True,
    and bypassed with a warning when strict_alignment=False.
    """
    # Should work without error because strict_alignment is False by default
    cache_non_strict = PagedDynamicKVCache(page_size=8, strict_alignment=False)
    assert cache_non_strict.page_size == 8

    # Should raise ValueError when strict_alignment is True
    with pytest.raises(ValueError) as excinfo:
        PagedDynamicKVCache(page_size=8, strict_alignment=True)
    assert "must be a multiple of 128" in str(excinfo.value)

    # Should succeed with alignment multiple of 128
    cache_strict = PagedDynamicKVCache(page_size=128, strict_alignment=True)
    assert cache_strict.page_size == 128


def test_sdpa_flash_attention_routing():
    """
    Verify that QoS/importance bookkeeping (attention_sum, importance_score)
    actually runs after inplace_paged_attention when force_qos=True, and is
    skipped by the Eager Bypass optimization when force_qos=False and there
    is no memory pressure.

    The attention math itself (SDPA + the online-softmax QoS weight pass)
    runs natively in C++ via ATen, so it can't be observed by mocking
    torch.matmul from Python — instead this checks the C++-side effect that
    QoS routing is actually gated on: does the page's bookkeeping update.
    """
    q = torch.randn(1, 2, 1, 16)

    # force_qos=True must run the QoS pass even with a single page and no
    # memory pressure — i.e. never take the Eager Bypass shortcut.
    # sink_tokens=0 so the full 128 tokens pushed form exactly one page.
    cache_forced = PagedDynamicKVCache(page_size=128, sink_tokens=0, force_qos=True)
    k = torch.randn(1, 2, 128, 16)
    v = torch.randn(1, 2, 128, 16)
    cache_forced.push_new_tokens(k, v)
    page_forced = cache_forced.active_pages[0]
    attn_sum_before = page_forced['attention_sum']
    out = cache_forced.inplace_paged_attention(q)
    assert out.shape == q.shape
    assert cache_forced.active_pages[0]['attention_sum'] != attn_sum_before, \
        "force_qos=True should run the QoS pass and update attention_sum even under no memory pressure"

    # force_qos=False with no memory pressure should take the Eager Bypass —
    # SDPA output is still correct, but QoS bookkeeping is skipped for speed.
    cache_bypassed = PagedDynamicKVCache(page_size=128, sink_tokens=0, force_qos=False)
    cache_bypassed.push_new_tokens(k.clone(), v.clone())
    page_bypassed = cache_bypassed.active_pages[0]
    attn_sum_before = page_bypassed['attention_sum']
    out2 = cache_bypassed.inplace_paged_attention(q)
    assert out2.shape == q.shape
    assert cache_bypassed.active_pages[0]['attention_sum'] == attn_sum_before, \
        "force_qos=False with no memory pressure should take the Eager Bypass and skip QoS bookkeeping"


def test_sink_pinning_cpu():
    """
    Test that sink_k and sink_v are pinned memory when allocated on CPU.
    """
    # Set sink_tokens > 0
    config = ArgusConfig(page_size=128, sink_tokens=4)
    cache = PagedDynamicKVCache(config=config)

    # Push enough tokens to allocate sinks
    k = torch.randn(1, 2, 10, 16)
    v = torch.randn(1, 2, 10, 16)
    cache.push_new_tokens(k, v)

    assert cache.sink_k is not None
    assert cache.sink_v is not None
    
    # Check if they are pinned CPU tensors
    assert cache.sink_k.is_pinned()
    assert cache.sink_v.is_pinned()


def test_sink_eviction_exemption():
    """
    Test that sink tokens are never subject to eviction or demotion.

    Sink tokens are carved off into cache.sink_k/sink_v before the remaining
    sequence is ever segmented into Page objects, so they structurally cannot
    appear in active_pages or any compressed tier — there is no "sink page"
    to accidentally evict. This test verifies that guarantee two ways: the
    sink buffer still holds exactly what was first pushed (proof nothing later
    touched it), and every token pushed is accounted for across sinks + pages
    (proof the sink tokens were separated once, not duplicated or dropped).
    """
    config = ArgusConfig(page_size=128, sink_tokens=4, max_active_pages=1)
    cache = PagedDynamicKVCache(config=config)

    # Push 3 pages of data (each 128 tokens).
    # The first 4 tokens of the first push go to the sink, the rest to pages.
    k1 = torch.randn(1, 2, 128, 16)
    v1 = torch.randn(1, 2, 128, 16)
    expected_sink_k = k1[..., :4, :].clone()
    expected_sink_v = v1[..., :4, :].clone()
    cache.push_new_tokens(k1, v1)

    for _ in range(2):
        k = torch.randn(1, 2, 128, 16)
        v = torch.randn(1, 2, 128, 16)
        cache.push_new_tokens(k, v)

    # Sinks must still hold exactly the first push's first 4 tokens, verbatim —
    # proof they were never touched by any later demotion/eviction cascade.
    assert cache.sink_k is not None
    assert cache.sink_k.shape[-2] == 4
    assert torch.allclose(cache.sink_k.cpu().float(), expected_sink_k.float(), atol=1e-3)
    assert torch.allclose(cache.sink_v.cpu().float(), expected_sink_v.float(), atol=1e-3)

    # Sink tokens are never represented as Page objects at all, so they
    # structurally cannot appear in active_pages or any compressed tier.
    all_pages = list(cache.active_pages)
    for spec in cache.tier_specs:
        all_pages.extend(cache.pages_by_tier.get(spec.name, []))
    total_page_tokens = sum(p.get('page_size', cache.page_size) for p in all_pages)
    # Tokens not yet forming a full page sit in the static staging buffer.
    # 3 pushes * 128 tokens == sink tokens + paged tokens + still-buffered tokens.
    assert 4 + total_page_tokens + cache.buffer_length == 3 * 128
