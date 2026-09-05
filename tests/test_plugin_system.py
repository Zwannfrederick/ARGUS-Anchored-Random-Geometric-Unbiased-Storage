"""Tests for the ARGUS plugin contract.

The claim under test is that ARGUS is a cache-management runtime rather than a
fixed quantization algorithm: a user must be able to disable a shipped tier
(1-bit) and install their own quantizer without touching the memory manager.
"""

import os
import sys

import pytest
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argus_cpp_backend
from argus_cache.core.memory_manager import ArgusConfig, PagedDynamicKVCache
from argus_cache.core.tier_registry import PipelineConfig, TierSpec
from argus_cache.plugins import (
    BackendCapabilities,
    NativeCodecSpec,
    PluginError,
    available_quantizers,
    get_capabilities,
    get_quantizer,
    list_quantizers,
    register_builtin_quantizers,
    register_quantizer,
    unregister_quantizer,
)
from argus_cache.backends.eviction import ImportanceSortPolicy


@pytest.fixture(autouse=True)
def restore_registry():
    """Every test starts and ends with the built-ins registered."""
    yield
    from argus_cache.plugins import REGISTRY

    REGISTRY.clear()
    register_builtin_quantizers()


class TernaryBackend:
    """A minimal third-party quantizer: symmetric ternary {-s, 0, +s}.

    Deliberately not derived from any ARGUS class — a plugin only has to
    satisfy the four-method contract.
    """

    def compress(self, tensor, **kwargs):
        scale = tensor.abs().amax().clamp_min(1e-8)
        q = torch.round(tensor / scale * 1.0).clamp(-1, 1).to(torch.int8)
        return {"q": q, "scales": scale}

    def decompress(self, compressed, **kwargs):
        return (compressed["q"].to(torch.float32) * compressed["scales"]).to(
            torch.float16
        )

    def decompress_batch(self, compressed_list, **kwargs):
        return [self.decompress(c, **kwargs) for c in compressed_list]

    def memory_bytes(self, compressed):
        return compressed["q"].nelement() * compressed["q"].element_size()


TERNARY_CAPS = BackendCapabilities(
    name="ternary",
    effective_bits=2.0,
    native_codec=NativeCodecSpec(kind="unsigned_affine", bits=2),
    description="Third-party symmetric ternary quantizer.",
)


# ── registration / removal ──────────────────────────────────────────────────


def test_register_and_lookup_plugin():
    register_quantizer("ternary", TernaryBackend, TERNARY_CAPS)

    assert "ternary" in list_quantizers()
    assert isinstance(get_quantizer("ternary"), TernaryBackend)
    assert get_capabilities("ternary").effective_bits == 2.0


def test_registration_is_idempotent_only_with_replace():
    register_quantizer("ternary", TernaryBackend, TERNARY_CAPS)

    with pytest.raises(PluginError, match="already registered"):
        register_quantizer("ternary", TernaryBackend, TERNARY_CAPS)

    # Explicit override is allowed.
    register_quantizer("ternary", TernaryBackend, TERNARY_CAPS, replace=True)


def test_unregister_removes_plugin():
    register_quantizer("ternary", TernaryBackend, TERNARY_CAPS)
    unregister_quantizer("ternary")

    assert "ternary" not in list_quantizers()
    with pytest.raises(PluginError, match="not registered"):
        get_quantizer("ternary")


def test_unregister_unknown_plugin_raises():
    with pytest.raises(PluginError, match="not registered"):
        unregister_quantizer("does_not_exist")


def test_plugin_instances_are_shared():
    """One instance per registration, so backends can cache derived state."""
    register_quantizer("ternary", TernaryBackend, TERNARY_CAPS)
    assert get_quantizer("ternary") is get_quantizer("ternary")


# ── invalid plugin configurations ───────────────────────────────────────────


def test_non_callable_factory_rejected():
    with pytest.raises(PluginError, match="must be callable"):
        register_quantizer("bad", TernaryBackend(), TERNARY_CAPS)


def test_capabilities_name_mismatch_rejected():
    with pytest.raises(PluginError, match="does not match"):
        register_quantizer("other_name", TernaryBackend, TERNARY_CAPS)


def test_backend_missing_required_method_rejected_on_use():
    class Incomplete:
        def compress(self, tensor, **kwargs):
            return {}

    caps = BackendCapabilities(name="incomplete", effective_bits=4.0)
    register_quantizer("incomplete", Incomplete, caps)

    # Registration is cheap and lazy; the contract violation surfaces when the
    # backend is actually constructed, with the missing methods named.
    with pytest.raises(PluginError, match="decompress"):
        get_quantizer("incomplete")


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(name="x", effective_bits=0.0), "effective_bits"),
        (dict(name="x", effective_bits=17.0), "effective_bits"),
        (dict(name="", effective_bits=4.0), "non-empty"),
        (dict(name="x", effective_bits=4.0, supported_devices=frozenset()), "device"),
        (dict(name="x", effective_bits=4.0, supported_dtypes=frozenset()), "dtype"),
    ],
)
def test_invalid_capabilities_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        BackendCapabilities(**kwargs)


@pytest.mark.parametrize(
    "kind, bits",
    [
        ("unsigned_affine", 3),  # 3 bits does not pack byte-aligned
        ("sign_packed", 2),
        ("signed_linear", 4),
        ("nonsense", 8),
    ],
)
def test_invalid_native_codec_rejected(kind, bits):
    with pytest.raises(ValueError):
        NativeCodecSpec(kind=kind, bits=bits)


@pytest.mark.parametrize(
    "kind,bits,effective_bits",
    [
        ("ggml_q8_0", 8, 8.5),
        ("ggml_q4_0", 4, 4.5),
    ],
)
def test_ggml_block_codecs_include_scale_overhead(kind, bits, effective_bits):
    codec = NativeCodecSpec(kind=kind, bits=bits)

    assert codec.effective_bits == effective_bits


def test_lossless_backend_cannot_claim_lossy_codec():
    with pytest.raises(ValueError, match="lossless"):
        BackendCapabilities(
            name="x",
            effective_bits=8.0,
            lossy=False,
            native_codec=NativeCodecSpec(kind="signed_linear", bits=8),
        )


# ── capability-based selection (no name checks) ─────────────────────────────


def test_available_quantizers_filters_by_budget():
    cheap = available_quantizers(max_effective_bits=2.0)

    assert "one_bit" in cheap and "int2" in cheap
    assert "fp8" not in cheap and "int8" not in cheap
    # Ordered most-expensive-first, which is cascade order.
    bits = [get_capabilities(n).effective_bits for n in cheap]
    assert bits == sorted(bits, reverse=True)


def test_available_quantizers_filters_by_dtype():
    register_quantizer(
        "fp32_only",
        TernaryBackend,
        BackendCapabilities(
            name="fp32_only",
            effective_bits=2.0,
            supported_dtypes=frozenset({torch.float32}),
        ),
    )

    assert "fp32_only" in available_quantizers(dtype=torch.float32)
    assert "fp32_only" not in available_quantizers(dtype=torch.float16)


def test_projection_is_detected_by_capability_not_name():
    """A projection tier registered under an arbitrary name is still handled
    as a projection — this is what removes the hardcoded 'jl' checks."""
    spec = TierSpec(name="archive", backend="jl", max_pages=2)

    assert spec.is_projection
    assert not TierSpec(name="archive2", backend="int2").is_projection


# ── the headline scenario: replace 1-bit with a custom quantizer ────────────


def _build_cache(tiers, page_size=16):
    return PagedDynamicKVCache(
        pipeline=PipelineConfig(
            tiers=tiers,
            eviction_policy=ImportanceSortPolicy(),
            page_size=page_size,
            sink_tokens=0,
            max_active_pages=1,
        )
    )


def test_disable_one_bit_and_install_custom_quantizer():
    unregister_quantizer("one_bit")
    assert "one_bit" not in list_quantizers()

    register_quantizer("ternary", TernaryBackend, TERNARY_CAPS)

    page_size = 16
    cache = _build_cache(
        [
            TierSpec(name="fp8", backend="fp8", max_pages=1, priority=1),
            TierSpec(name="ternary", backend="ternary", max_pages=4, priority=2),
        ],
        page_size=page_size,
    )

    assert [s.name for s in cache.tier_specs] == ["fp8", "ternary"]
    # The manager never learned the tier's name — it read its cost.
    assert cache.tier_name_to_spec["ternary"].effective_bits == 2.0

    for _ in range(4):
        k = torch.randn(1, 1, page_size, 16, dtype=torch.float16)
        v = torch.randn(1, 1, page_size, 16, dtype=torch.float16)
        cache.push_new_tokens(k, v)

    assert len(cache.pages_by_tier["ternary"]) > 0, "custom tier never received pages"

    q = torch.randn(1, 1, 1, 16, dtype=torch.float16)
    assert cache.inplace_paged_attention(q) is not None


def test_pipeline_without_one_bit_still_cascades():
    """Removing a shipped tier must not break the cascade."""
    page_size = 16
    cache = _build_cache(
        [
            TierSpec(name="int8", backend="int8", max_pages=1, priority=1),
            TierSpec(name="int4", backend="int4", max_pages=1, priority=2),
            TierSpec(name="jl", backend="jl", max_pages=-1, priority=3),
        ],
        page_size=page_size,
    )

    assert "one_bit" not in cache.tier_name_to_spec

    for _ in range(5):
        k = torch.randn(1, 1, page_size, 16, dtype=torch.float16)
        v = torch.randn(1, 1, page_size, 16, dtype=torch.float16)
        cache.push_new_tokens(k, v)

    total = len(cache.active_pages) + sum(
        len(cache.pages_by_tier[t]) for t in cache.pages_by_tier
    )
    assert total == 5, "pages were lost while cascading through a custom pipeline"


def test_single_tier_pipeline_demotes_into_its_own_first_tier():
    """Regression: demotion from the active pool must target the configured
    first tier, not a hardcoded fp8."""
    page_size = 16
    cache = _build_cache(
        [TierSpec(name="int8", backend="int8", max_pages=8, priority=1)],
        page_size=page_size,
    )

    for _ in range(3):
        k = torch.randn(1, 1, page_size, 16, dtype=torch.float16)
        v = torch.randn(1, 1, page_size, 16, dtype=torch.float16)
        cache.push_new_tokens(k, v)

    assert len(cache.pages_by_tier["int8"]) == 2
    assert "fp8" not in cache.pages_by_tier or not cache.pages_by_tier["fp8"]


# ── native codec propagation ────────────────────────────────────────────────


def test_builtin_tiers_publish_native_codecs_to_cpp():
    cache = _build_cache(
        [
            TierSpec(name="fp8", backend="fp8", max_pages=1),
            TierSpec(name="one_bit", backend="one_bit", max_pages=2),
        ]
    )
    cpp = cache._cpp_manager

    assert cpp.has_codec("fp8")
    assert cpp.get_codec("one_bit").pack_factor == 8
    assert cpp.get_codec("one_bit").bits == 1


def test_ggml_block_tiers_publish_exact_layout_to_cpp():
    cache = _build_cache(
        [
            TierSpec(name="q8_0", backend="q8_0", max_pages=1),
            TierSpec(name="q4_0", backend="q4_0", max_pages=2),
        ]
    )
    cpp = cache._cpp_manager

    q8 = cpp.get_codec("q8_0")
    q4 = cpp.get_codec("q4_0")
    assert q8.kind == argus_cpp_backend.CodecKind.GGML_Q8_0
    assert q8.block_size == 32
    assert q8.block_bytes == 34
    assert q8.effective_bits == 8.5
    assert q4.kind == argus_cpp_backend.CodecKind.GGML_Q4_0
    assert q4.block_size == 32
    assert q4.block_bytes == 18
    assert q4.effective_bits == 4.5


def test_custom_tier_native_codec_reaches_cpp():
    register_quantizer("ternary", TernaryBackend, TERNARY_CAPS)
    cache = _build_cache(
        [
            TierSpec(name="fp8", backend="fp8", max_pages=1),
            TierSpec(name="ternary", backend="ternary", max_pages=2),
        ]
    )

    codec = cache._cpp_manager.get_codec("ternary")
    assert codec.kind == argus_cpp_backend.CodecKind.UNSIGNED_AFFINE
    assert codec.bits == 2
    assert codec.pack_factor == 4


def test_tier_without_native_codec_falls_back_to_passthrough():
    """A backend that declares no native format must spill losslessly rather
    than being decoded with a guessed bit layout."""
    register_quantizer(
        "python_only",
        TernaryBackend,
        BackendCapabilities(name="python_only", effective_bits=2.0),
    )
    cache = _build_cache(
        [
            TierSpec(name="fp8", backend="fp8", max_pages=1),
            TierSpec(name="python_only", backend="python_only", max_pages=2),
        ]
    )

    codec = cache._cpp_manager.get_codec("python_only")
    assert codec.kind == argus_cpp_backend.CodecKind.PASSTHROUGH


# ── native codec round-trip fidelity ────────────────────────────────────────


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "tier, tolerance",
    [
        # Budgets are the theoretical round-to-nearest error for standard
        # normal input, relative L2: with L levels spanning the observed range
        # R, step = R/(L-1) and RMS error = step/sqrt(12). For 512 Gaussian
        # samples R is about 6.6 sigma, giving ~0.02 (8-bit), ~0.13 (int4) and
        # ~0.64 (int2), each allowed a margin for seed variance. Truncating
        # instead of rounding roughly doubles these, which is what these
        # numbers are here to catch.
        ("fp8", 0.02),
        ("int8", 0.02),
        ("int4", 0.15),
        ("int2", 0.75),
    ],
)
def test_native_codec_roundtrip_fidelity(tier, tolerance):
    """The generic pack/unpack path must reconstruct within the tier's budget.

    Guards the refactor that replaced five bespoke CUDA kernels with one
    parameterized kernel: a wrong shift or pack order shows up here as
    reconstruction error far above the tier's quantization step.
    """
    page_size = 32
    head_dim = 16
    cache = _build_cache(
        [TierSpec(name=tier, backend=tier, max_pages=8)], page_size=page_size
    )

    torch.manual_seed(0)
    k = torch.randn(1, 1, page_size, head_dim, dtype=torch.float16, device="cuda")
    v = torch.randn(1, 1, page_size, head_dim, dtype=torch.float16, device="cuda")
    cache.push_new_tokens(k.clone(), v.clone())
    cache.push_new_tokens(
        torch.randn_like(k), torch.randn_like(v)
    )  # forces the first page down a tier

    page = cache.pages_by_tier[tier][0]
    k_out, _ = cache._cpp_manager.peek_decompress_page(page, tier)

    rel_err = ((k_out.float() - k.float()).norm() / k.float().norm()).item()
    assert rel_err < tolerance, f"{tier} round-trip error {rel_err:.4f} > {tolerance}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "tier,block_bytes,tolerance",
    [("q8_0", 34, 0.02), ("q4_0", 18, 0.16)],
)
def test_ggml_block_codec_roundtrip_and_storage(tier, block_bytes, tolerance):
    page_size = 32
    head_dim = 32
    cache = _build_cache(
        [TierSpec(name=tier, backend=tier, max_pages=8)], page_size=page_size
    )

    torch.manual_seed(0)
    k = torch.randn(1, 1, page_size, head_dim, dtype=torch.float16, device="cuda")
    v = torch.randn_like(k)
    cache.push_new_tokens(k.clone(), v.clone())
    cache.push_new_tokens(torch.randn_like(k), torch.randn_like(v))

    page = cache.pages_by_tier[tier][0]
    assert page.key_compressed.dtype == torch.uint8
    assert page.key_compressed.numel() == k.numel() // 32 * block_bytes

    k_out, _ = cache._cpp_manager.peek_decompress_page(page, tier)
    rel_err = ((k_out.float() - k.float()).norm() / k.float().norm()).item()
    assert rel_err < tolerance, f"{tier} round-trip error {rel_err:.4f} > {tolerance}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_ggml_q4_0_uses_llama_nibble_order():
    cache = _build_cache(
        [TierSpec(name="q4_0", backend="q4_0", max_pages=8)], page_size=32
    )
    row = torch.arange(-16, 16, dtype=torch.float16, device="cuda")
    k = row.view(1, 1, 1, 32).expand(1, 1, 32, 32).contiguous()
    cache.push_new_tokens(k, k)
    cache.push_new_tokens(torch.zeros_like(k), torch.zeros_like(k))

    block = cache.pages_by_tier["q4_0"][0].key_compressed.flatten()[:18]
    scale = block[:2].view(torch.float16).item()
    packed = block[2:]
    expected_scale = 16.0 / 8.0
    expected_low = torch.trunc(row[:16].cpu() / expected_scale + 8.5).clamp(0, 15)
    expected_high = torch.trunc(row[16:].cpu() / expected_scale + 8.5).clamp(0, 15)
    expected = (expected_low.to(torch.uint8) | (expected_high.to(torch.uint8) << 4))

    assert scale == expected_scale
    assert torch.equal(packed, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_ggml_q8_0_uses_llama_block_layout():
    cache = _build_cache(
        [TierSpec(name="q8_0", backend="q8_0", max_pages=8)], page_size=32
    )
    row = torch.arange(-16, 16, dtype=torch.float16, device="cuda")
    k = row.view(1, 1, 1, 32).expand(1, 1, 32, 32).contiguous()
    cache.push_new_tokens(k, k)
    cache.push_new_tokens(torch.zeros_like(k), torch.zeros_like(k))

    block = cache.pages_by_tier["q8_0"][0].key_compressed.flatten()[:34]
    stored_scale = block[:2].view(torch.float16).item()
    stored_quants = block[2:].view(torch.int8)
    quant_scale = 16.0 / 127.0
    expected_scale = torch.tensor(quant_scale, dtype=torch.float16).item()
    expected_quants = torch.round(row.cpu() / quant_scale).to(torch.int8)

    assert stored_scale == expected_scale
    assert torch.equal(stored_quants, expected_quants)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_ggml_q8_0_rounds_halfway_away_from_zero_like_llama():
    cache = _build_cache(
        [TierSpec(name="q8_0", backend="q8_0", max_pages=8)], page_size=32
    )
    row = torch.zeros(32, dtype=torch.float32, device="cuda")
    row[:5] = torch.tensor([127.0, 0.5, -0.5, 1.5, -1.5], device="cuda")
    k = row.view(1, 1, 1, 32).expand(1, 1, 32, 32).contiguous()
    cache.push_new_tokens(k, k)
    cache.push_new_tokens(torch.zeros_like(k), torch.zeros_like(k))

    block = cache.pages_by_tier["q8_0"][0].key_compressed.flatten()[:34]

    assert block[:2].view(torch.float16).item() == 1.0
    assert torch.equal(
        block[2:7].view(torch.int8),
        torch.tensor([127, 1, -1, 2, -2], dtype=torch.int8),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "tier,magnitude", [("q8_0", 1.0e6), ("q4_0", 5.0e5)]
)
def test_ggml_block_codecs_preserve_finite_fp32_range(tier, magnitude):
    cache = _build_cache(
        [TierSpec(name=tier, backend=tier, max_pages=8)], page_size=32
    )
    row = torch.linspace(-magnitude, magnitude, 32, dtype=torch.float32, device="cuda")
    k = row.view(1, 1, 1, 32).expand(1, 1, 32, 32).contiguous()
    cache.push_new_tokens(k, k)
    cache.push_new_tokens(torch.zeros_like(k), torch.zeros_like(k))

    page = cache.pages_by_tier[tier][0]
    restored, _ = cache._cpp_manager.peek_decompress_page(page, tier)

    assert restored.dtype == torch.float32
    assert torch.isfinite(restored).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "tier,magnitude", [("q8_0", 9.0e6), ("q4_0", 1.0e6)]
)
def test_ggml_block_codecs_reject_unrepresentable_fp16_scale(tier, magnitude):
    cache = _build_cache(
        [TierSpec(name=tier, backend=tier, max_pages=8)], page_size=32
    )
    row = torch.linspace(-magnitude, magnitude, 32, dtype=torch.float32, device="cuda")
    k = row.view(1, 1, 1, 32).expand(1, 1, 32, 32).contiguous()
    cache.push_new_tokens(k, k)

    with pytest.raises(RuntimeError, match="cannot represent this block range"):
        cache.push_new_tokens(torch.zeros_like(k), torch.zeros_like(k))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_one_bit_roundtrip_preserves_sign():
    """1-bit is sign-only, so magnitude error is expected but sign must hold."""
    page_size = 32
    head_dim = 16
    cache = _build_cache(
        [TierSpec(name="one_bit", backend="one_bit", max_pages=8)],
        page_size=page_size,
    )

    torch.manual_seed(0)
    k = torch.randn(1, 1, page_size, head_dim, dtype=torch.float16, device="cuda")
    v = torch.randn_like(k)
    cache.push_new_tokens(k.clone(), v.clone())
    cache.push_new_tokens(torch.randn_like(k), torch.randn_like(v))

    page = cache.pages_by_tier["one_bit"][0]
    k_out, _ = cache._cpp_manager.peek_decompress_page(page, "one_bit")

    agreement = ((k_out >= 0) == (k >= 0)).float().mean().item()
    assert agreement > 0.99, f"sign agreement only {agreement:.3f}"


# ── ggml cascade, end to end ────────────────────────────────────────────────


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_ggml_cascade_actually_demotes_through_both_tiers():
    """Under real pressure the cascade must hold two precisions at once.

    Configuring a cascade proves nothing on its own -- if every page ends up in
    one tier, the result is what llama.cpp already does with a single
    ``-ctk`` setting, and the reason for the profile disappears. The claim
    being tested is that recent pages sit at q8_0 while older ones fall to
    q4_0 *simultaneously*.
    """
    from argus_cache.models.attention_wrapper import PagedDynamicQuantizedCache

    pipeline = PagedDynamicQuantizedCache(pipeline_profile="ggml")._ggml_pipeline()
    pipeline.page_size = 16
    pipeline.sink_tokens = 0
    pipeline.max_active_pages = 1
    pipeline.tiers[0].max_pages = 2
    cache = PagedDynamicKVCache(pipeline=pipeline)

    torch.manual_seed(0)
    for _ in range(12):
        k = torch.randn(1, 1, 16, 32, dtype=torch.float16, device="cuda")
        cache.push_new_tokens(k, torch.randn_like(k))

    occupied = {t: len(p) for t, p in cache.pages_by_tier.items() if p}

    assert "q8_0" in occupied, f"nothing reached q8_0: {occupied}"
    assert "q4_0" in occupied, f"nothing reached q4_0: {occupied}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_ggml_cascade_pages_survive_a_decompression_round_trip():
    """A demoted page must still decode to something close to what went in.

    Heterogeneous storage is only useful if the cheap tier remains usable;
    a cascade that quietly corrupts old pages would trade quality for capacity
    without saying so.
    """
    from argus_cache.models.attention_wrapper import PagedDynamicQuantizedCache

    pipeline = PagedDynamicQuantizedCache(pipeline_profile="ggml")._ggml_pipeline()
    pipeline.page_size = 32
    pipeline.sink_tokens = 0
    pipeline.max_active_pages = 1
    cache = PagedDynamicKVCache(pipeline=pipeline)

    torch.manual_seed(0)
    first = torch.randn(1, 1, 32, 32, dtype=torch.float16, device="cuda")
    cache.push_new_tokens(first.clone(), first.clone())
    for _ in range(4):
        k = torch.randn(1, 1, 32, 32, dtype=torch.float16, device="cuda")
        cache.push_new_tokens(k, torch.randn_like(k))

    tier = next(t for t, pages in cache.pages_by_tier.items() if pages)
    page = cache.pages_by_tier[tier][0]
    restored, _ = cache._cpp_manager.peek_decompress_page(page, tier)

    assert torch.isfinite(restored).all(), "decompressed page contains non-finite values"
    rel_err = (restored.float() - first.float()).norm() / first.float().norm()
    assert rel_err < 0.25, f"{tier} round-trip error {rel_err:.4f}"
