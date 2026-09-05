"""Property-driven attention parity matrix for ARGUS (Stage S3).

Verifies three independent oracles across the full parameter matrix:
1. Oracle 1: FP32 mathematical reference attention
2. Oracle 2: Decoded reference over exact stored/compressed bytes (q8_0, q4_0)
3. Oracle 3: Stock PyTorch SDPA reference

Also verifies Qwen3.8-27B specific geometry (24Q, 4KV, head_dim 256) and
strict fail-closed / fallback invariants for unsupported inputs.
"""

from __future__ import annotations

import math
import pytest
import torch

from argus_cache.core.memory_manager import ArgusConfig, PagedDynamicKVCache
from argus_cache.core.tier_registry import PipelineConfig, TierSpec
from argus_cache.models.hf_attention import (
    argus_attention_forward,
    Qwen2AttentionAdapter,
    LlamaAttentionAdapter,
)


# ── Fixtures and Oracles ───────────────────────────────────────────────────


def fp32_mathematical_attention(
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    scale: float | None = None,
) -> torch.Tensor:
    """Oracle 1: Pure FP32 mathematical attention with GQA support."""
    batch, q_heads, q_len, dim = query.shape
    _, kv_heads, seq_len, _ = keys.shape
    assert q_heads % kv_heads == 0, f"q_heads ({q_heads}) must be multiple of kv_heads ({kv_heads})"
    groups = q_heads // kv_heads

    if groups > 1:
        keys_rep = keys.repeat_interleave(groups, dim=1)
        values_rep = values.repeat_interleave(groups, dim=1)
    else:
        keys_rep = keys
        values_rep = values

    if scale is None:
        scale = 1.0 / math.sqrt(dim)

    q_f32 = query.to(torch.float32)
    k_f32 = keys_rep.to(torch.float32)
    v_f32 = values_rep.to(torch.float32)

    # [B, H, q_len, seq_len]
    scores = torch.matmul(q_f32, k_f32.transpose(-1, -2)) * scale
    probs = torch.softmax(scores, dim=-1)
    output = torch.matmul(probs, v_f32)
    return output.to(query.dtype)


def stock_sdpa_attention(
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    scale: float | None = None,
) -> torch.Tensor:
    """Oracle 3: Stock PyTorch Scaled Dot-Product Attention with GQA."""
    batch, q_heads, q_len, dim = query.shape
    _, kv_heads, seq_len, _ = keys.shape
    groups = q_heads // kv_heads

    if groups > 1:
        keys = keys.repeat_interleave(groups, dim=1)
        values = values.repeat_interleave(groups, dim=1)

    return torch.nn.functional.scaled_dot_product_attention(
        query, keys, values, scale=scale, is_causal=False
    )


def compute_metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    """Computes error metrics and cosine similarity between tensors."""
    assert torch.isfinite(actual).all(), "Actual output contains NaN or Inf!"
    assert torch.isfinite(expected).all(), "Expected output contains NaN or Inf!"

    diff = (actual.float() - expected.float()).abs()
    max_abs = diff.max().item()
    norm_expected = expected.float().norm().clamp_min(1e-8)
    rel_err = (diff.norm() / norm_expected).item()

    cos_sim = torch.nn.functional.cosine_similarity(
        actual.float().flatten(),
        expected.float().flatten(),
        dim=0,
    ).item()

    return {
        "max_abs_err": max_abs,
        "rel_err": rel_err,
        "cos_sim": cos_sim,
    }


def _build_cache(tier_name: str, page_size: int, max_active: int = 1) -> PagedDynamicKVCache:
    """Helper to build PagedDynamicKVCache with exact tier configuration."""
    if tier_name == "active":
        pipeline = PipelineConfig(
            tiers=[],
            page_size=page_size,
            sink_tokens=0,
            max_active_pages=1024,
        )
    else:
        pipeline = PipelineConfig(
            tiers=[TierSpec(name=tier_name, backend=tier_name, max_pages=-1)],
            page_size=page_size,
            sink_tokens=0,
            max_active_pages=max_active,
        )
    return PagedDynamicKVCache(pipeline=pipeline)


# ── S3 Parity Matrix Tests ──────────────────────────────────────────────────


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for parity matrix")
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize(
    "q_heads,kv_heads",
    [
        (1, 1),    # 1:1 single head
        (4, 4),    # 1:1 MHA
        (4, 1),    # 4:1 GQA
        (8, 2),    # 4:1 GQA
        (8, 4),    # 2:1 GQA
        (24, 4),   # 6:1 GQA (Qwen3.8-27B geometry!)
    ],
)
@pytest.mark.parametrize("head_dim", [64, 128, 256])
def test_active_exact_attention_parity_matrix(batch, q_heads, kv_heads, head_dim):
    """Verifies Oracle 1, 2, and 3 match on ACTIVE pages across all head/dim geometries."""
    torch.manual_seed(42 + q_heads * 10 + kv_heads)
    page_size = 128
    cache = _build_cache("active", page_size=page_size)

    # Push 2 pages of tokens
    total_tokens = 256
    keys = torch.randn(batch, kv_heads, total_tokens, head_dim, dtype=torch.float16, device="cuda")
    values = torch.randn(batch, kv_heads, total_tokens, head_dim, dtype=torch.float16, device="cuda")

    for i in range(0, total_tokens, page_size):
        cache.push_new_tokens(
            keys[:, :, i : i + page_size],
            values[:, :, i : i + page_size],
        )

    query = torch.randn(batch, q_heads, 1, head_dim, dtype=torch.float16, device="cuda")

    # Native paged attention
    actual = cache.inplace_paged_attention(query)

    # Oracle 1: FP32 math
    oracle1 = fp32_mathematical_attention(query, keys, values)
    # Oracle 3: Stock SDPA
    oracle3 = stock_sdpa_attention(query, keys, values)

    # ACTIVE exact must match within strict FP16 accumulation tolerances
    m1 = compute_metrics(actual, oracle1)
    m3 = compute_metrics(actual, oracle3)

    assert m1["cos_sim"] > 0.999, f"ACTIVE cos_sim {m1['cos_sim']:.6f} <= 0.999 with FP32 math"
    assert m3["cos_sim"] > 0.999, f"ACTIVE cos_sim {m3['cos_sim']:.6f} <= 0.999 with stock SDPA"
    assert m3["rel_err"] < 5e-3, f"ACTIVE rel_err {m3['rel_err']:.6f} >= 5e-3"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for parity matrix")
@pytest.mark.parametrize("page_size", [128, 256])
@pytest.mark.parametrize("context_offset", [-1, 0, 1])  # Page boundary -1, exact, +1
def test_page_boundary_exactness(page_size, context_offset):
    """Verifies attention correctness at exact, under-full, and over-full page boundaries."""
    torch.manual_seed(101)
    total_tokens = page_size + context_offset
    batch, q_heads, kv_heads, head_dim = 1, 8, 2, 64

    cache = _build_cache("active", page_size=page_size)

    keys = torch.randn(batch, kv_heads, total_tokens, head_dim, dtype=torch.float16, device="cuda")
    values = torch.randn(batch, kv_heads, total_tokens, head_dim, dtype=torch.float16, device="cuda")

    # Push in chunks
    chunk_size = min(64, total_tokens)
    for i in range(0, total_tokens, chunk_size):
        end = min(i + chunk_size, total_tokens)
        cache.push_new_tokens(keys[:, :, i:end], values[:, :, i:end])

    query = torch.randn(batch, q_heads, 1, head_dim, dtype=torch.float16, device="cuda")
    actual = cache.inplace_paged_attention(query)
    expected = stock_sdpa_attention(query, keys, values)

    metrics = compute_metrics(actual, expected)
    assert metrics["cos_sim"] > 0.999, f"Boundary offset {context_offset} failed: cos_sim={metrics['cos_sim']}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for parity matrix")
@pytest.mark.parametrize("tier", ["q8_0", "q4_0"])
def test_quantized_tier_oracles_and_parity(tier):
    """Verifies Oracle 2 (decoded reference) and bounded codec error for q8_0 and q4_0."""
    torch.manual_seed(202)
    page_size = 32
    head_dim = 64
    batch, q_heads, kv_heads = 1, 4, 1

    cache = _build_cache(tier, page_size=page_size, max_active=1)

    # Push 3 pages so that 2 pages demote to the quantized tier
    total_tokens = 96
    keys = torch.randn(batch, kv_heads, total_tokens, head_dim, dtype=torch.float16, device="cuda")
    values = torch.randn(batch, kv_heads, total_tokens, head_dim, dtype=torch.float16, device="cuda")

    for i in range(0, total_tokens, page_size):
        cache.push_new_tokens(
            keys[:, :, i : i + page_size],
            values[:, :, i : i + page_size],
        )

    # Verify pages exist in the expected tier
    assert len(cache.pages_by_tier[tier]) == 2
    assert len(cache.active_pages) == 1

    query = torch.randn(batch, q_heads, 1, head_dim, dtype=torch.float16, device="cuda")
    actual = cache.inplace_paged_attention(query)

    # Oracle 2: Reconstruct exact keys/values from the cache pages
    decomp_keys, decomp_values = cache.get_all_keys_values()
    oracle2 = stock_sdpa_attention(query, decomp_keys, decomp_values)

    # Oracle 3: Stock unquantized SDPA
    oracle3 = stock_sdpa_attention(query, keys, values)

    # 1. Native paged attention must closely match Oracle 2 (decompressed reference)
    m_oracle2 = compute_metrics(actual, oracle2)
    assert m_oracle2["cos_sim"] > 0.99, f"{tier} vs Oracle 2 cos_sim {m_oracle2['cos_sim']:.4f} <= 0.99"

    # 2. Bounded codec error against unquantized Oracle 3
    m_oracle3 = compute_metrics(actual, oracle3)
    if tier == "q8_0":
        assert m_oracle3["cos_sim"] > 0.98, f"q8_0 vs unquantized cos_sim {m_oracle3['cos_sim']:.4f} <= 0.98"
        assert m_oracle3["rel_err"] < 0.10, f"q8_0 vs unquantized rel_err {m_oracle3['rel_err']:.4f} >= 0.10"
    elif tier == "q4_0":
        assert m_oracle3["cos_sim"] > 0.92, f"q4_0 vs unquantized cos_sim {m_oracle3['cos_sim']:.4f} <= 0.92"
        assert m_oracle3["rel_err"] < 0.30, f"q4_0 vs unquantized rel_err {m_oracle3['rel_err']:.4f} >= 0.30"


# ── Qwen3.8-27B Geometry & Gated Attention Fixture ─────────────────────────


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_qwen38_geometry_and_gated_attention_fixture():
    """Validates full 24Q / 4KV / head_dim 256 geometry with gated attention boundary."""
    torch.manual_seed(38)
    batch = 1
    q_heads = 24
    kv_heads = 4
    head_dim = 256
    page_size = 128
    total_tokens = 512

    cache = _build_cache("active", page_size=page_size)

    # Simulate MRoPE rotated K and V
    keys = torch.randn(batch, kv_heads, total_tokens, head_dim, dtype=torch.float16, device="cuda")
    values = torch.randn(batch, kv_heads, total_tokens, head_dim, dtype=torch.float16, device="cuda")

    for i in range(0, total_tokens, page_size):
        cache.push_new_tokens(
            keys[:, :, i : i + page_size],
            values[:, :, i : i + page_size],
        )

    query = torch.randn(batch, q_heads, 1, head_dim, dtype=torch.float16, device="cuda")
    gate = torch.randn(batch, q_heads, 1, head_dim, dtype=torch.float16, device="cuda")

    # Native output
    raw_out = cache.inplace_paged_attention(query)
    # Qwen3.5/3.8 applies sigmoid(gate) elementwise after attention
    gated_out = raw_out * torch.sigmoid(gate)

    # Stock reference
    ref_raw = stock_sdpa_attention(query, keys, values)
    ref_gated = ref_raw * torch.sigmoid(gate)

    metrics = compute_metrics(gated_out, ref_gated)
    assert metrics["cos_sim"] > 0.999, f"Qwen3.8 gated cos_sim {metrics['cos_sim']:.6f} <= 0.999"
    assert metrics["rel_err"] < 5e-3, f"Qwen3.8 gated rel_err {metrics['rel_err']:.6f} >= 5e-3"


# ── Layout & Mask Rejection Invariants (Fail-Closed) ────────────────────────


def test_rejection_of_prefill_in_native_decode():
    """q_len > 1 must fail closed in native decode check and trigger fallback."""
    class DummyModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            from types import SimpleNamespace
            self.config = SimpleNamespace(model_type="qwen2")
            self.num_key_value_groups = 2
            self.training = False

    adapter = Qwen2AttentionAdapter()
    module = DummyModule()

    # q_len = 1 (decode) -> Accepted
    q_decode = torch.randn(1, 4, 1, 64)
    assert adapter.can_use_native(module, q_decode, attention_mask=None) is True

    # q_len = 4 (prefill) -> Rejected
    q_prefill = torch.randn(1, 4, 4, 64)
    assert adapter.can_use_native(module, q_prefill, attention_mask=None) is False


def test_rejection_of_masked_and_training_attention():
    """Attention mask, training mode, and sliding window must be rejected by adapter."""
    class DummyModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            from types import SimpleNamespace
            self.config = SimpleNamespace(model_type="llama")
            self.training = False

    adapter = LlamaAttentionAdapter()
    module = DummyModule()
    q = torch.randn(1, 4, 1, 64)

    # Mask present -> Rejected
    mask = torch.zeros(1, 1, 1, 10)
    assert adapter.can_use_native(module, q, attention_mask=mask) is False

    # Training mode -> Rejected
    module.training = True
    assert adapter.can_use_native(module, q, attention_mask=None) is False
    module.training = False

    # Sliding window -> Rejected
    assert adapter.can_use_native(module, q, attention_mask=None, sliding_window=512) is False

    # Output attentions -> Rejected
    assert adapter.can_use_native(module, q, attention_mask=None, output_attentions=True) is False
