"""Tests for S5 (Page Table ABI & Backend Block Pools) and S6 (Direct Paged Attention).

Validates:
1. StructureOfArraysPageTable monotonic generation counters and compact layout.
2. ContiguousBlockPool exact byte sizing, slot recycling, and live tracking.
3. DirectPagedAttentionEngine online-softmax single-token execution over mixed precision tiers.
4. GQA support (including Qwen3.8 24Q:4KV) and parity with mathematical reference attention.
"""

import math
import pytest
import torch

from argus_cache.core.page_table import CodecKind, PlacementLocation, StructureOfArraysPageTable
from argus_cache.core.backend_pool import ContiguousBlockPool
from argus_cache.core.direct_attention import DirectPagedAttentionEngine


import numpy as np


def _quantize_q8_0_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Helper to quantize FP16/FP32 tensor into standard GGML q8_0 bytes."""
    flat = tensor.float().flatten()
    num_blocks = flat.numel() // 32
    raw = bytearray()
    for b in range(num_blocks):
        block_vals = flat[b * 32 : (b + 1) * 32]
        d = block_vals.abs().max().item() / 127.0 if block_vals.abs().max() > 0 else 1.0
        scale_fp16 = torch.tensor(d, dtype=torch.float16)
        quants = torch.round(block_vals / d).clamp(-128, 127).to(torch.int8)
        raw.extend(scale_fp16.numpy().tobytes())
        raw.extend(quants.numpy().tobytes())
    return torch.from_numpy(np.frombuffer(raw, dtype=np.uint8).copy()).to(torch.uint8)


def _quantize_q4_0_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Helper to quantize FP16/FP32 tensor into standard GGML q4_0 bytes."""
    flat = tensor.float().flatten()
    num_blocks = flat.numel() // 32
    raw = bytearray()
    for b in range(num_blocks):
        block_vals = flat[b * 32 : (b + 1) * 32]
        d = block_vals.abs().max().item() / -8.0 if block_vals.abs().max() > 0 else 1.0
        scale_fp16 = torch.tensor(-block_vals.abs().max().item() / 8.0, dtype=torch.float16)
        scale_val = scale_fp16.item() if scale_fp16.item() != 0 else 1.0
        # GGML q4_0 packing: low 16 nibbles, high 16 nibbles
        qs = torch.trunc(block_vals / scale_val + 8.5).clamp(0, 15).to(torch.uint8)
        low = qs[:16]
        high = qs[16:]
        packed = low | (high << 4)
        raw.extend(scale_fp16.numpy().tobytes())
        raw.extend(packed.numpy().tobytes())
    return torch.from_numpy(np.frombuffer(raw, dtype=np.uint8).copy()).to(torch.uint8)


# ── S5: Page Table & Pool Invariants ───────────────────────────────────────


def test_soa_page_table_operations_and_generations():
    """Validates SoA page table allocation, update, compaction, and monotonic generation."""
    pt = StructureOfArraysPageTable(capacity=4)

    # 1. Allocate initial pages
    idx0 = pt.allocate_page(page_id=10, logical_pos=0, token_count=64, codec=CodecKind.ACTIVE_FP16)
    idx1 = pt.allocate_page(page_id=20, logical_pos=64, token_count=64, codec=CodecKind.ACTIVE_FP16)
    assert pt.num_pages == 2
    assert idx0 == 0 and idx1 == 1

    desc0 = pt.get_descriptor(10)
    assert desc0.page_id == 10
    assert desc0.logical_pos == 0
    assert desc0.codec == CodecKind.ACTIVE_FP16
    initial_gen = desc0.generation

    # 2. Update placement / precision
    pt.update_placement(10, new_placement=PlacementLocation.HOST_PINNED, new_codec=CodecKind.GGML_Q8_0, new_pool_slot=3)
    desc0_updated = pt.get_descriptor(10)
    assert desc0_updated.codec == CodecKind.GGML_Q8_0
    assert desc0_updated.placement == PlacementLocation.HOST_PINNED
    assert desc0_updated.pool_slot == 3
    assert desc0_updated.generation > initial_gen
    assert pt.is_valid_generation(10, desc0_updated.generation) is True
    assert pt.is_valid_generation(10, initial_gen) is False  # Stale generation rejected!

    # 3. Dynamic growth when exceeding initial capacity
    pt.allocate_page(page_id=30, logical_pos=128, token_count=64)
    pt.allocate_page(page_id=40, logical_pos=192, token_count=64)
    pt.allocate_page(page_id=50, logical_pos=256, token_count=64)  # Triggers growth
    assert pt.capacity >= 8
    assert pt.num_pages == 5

    # 4. Remove page and check compaction
    pt.remove_page(20)
    assert pt.num_pages == 4
    with pytest.raises(KeyError):
        pt.get_descriptor(20)


def test_contiguous_block_pool_allocation_and_recycling():
    """Validates exact byte sizing and slot recycling in ContiguousBlockPool."""
    page_size = 32
    num_heads = 4
    head_dim = 64
    max_slots = 4

    pool = ContiguousBlockPool(
        codec=CodecKind.GGML_Q8_0,
        placement=PlacementLocation.GPU_DEVICE,
        max_slots=max_slots,
        page_size=page_size,
        num_heads=num_heads,
        head_dim=head_dim,
    )

    # Elements per page = 4 * 32 * 64 = 8192 elements = 256 blocks of 32
    # Bytes per page = 256 * 34 = 8704 bytes
    assert pool.bytes_per_page == 8704
    assert pool.capacity_bytes() == max_slots * 8704 * 2

    # Allocate all slots
    slots = [pool.allocate_slot() for _ in range(max_slots)]
    assert len(slots) == max_slots
    assert pool.live_bytes() == pool.capacity_bytes()

    with pytest.raises(RuntimeError, match="out of memory"):
        pool.allocate_slot()

    # Free a slot and reallocate
    freed = slots[1]
    pool.free_slot(freed)
    assert pool.live_bytes() == (max_slots - 1) * 8704 * 2
    new_slot = pool.allocate_slot()
    assert new_slot == freed


# ── S6: Direct Attention Single-Token Decode ───────────────────────────────


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_direct_paged_attention_mixed_active_and_q8_parity():
    """Validates DirectPagedAttentionEngine over mixed ACTIVE and Q8 pages matches FP32 math."""
    torch.manual_seed(42)
    batch = 1
    q_heads = 8
    kv_heads = 2  # 4:1 GQA
    head_dim = 64
    page_size = 32

    # Construct page table with 1 ACTIVE page + 2 Q8 pages
    pt = StructureOfArraysPageTable(capacity=8, device="cuda")
    q8_pool = ContiguousBlockPool(
        codec=CodecKind.GGML_Q8_0,
        placement=PlacementLocation.GPU_DEVICE,
        max_slots=4,
        page_size=page_size,
        num_heads=kv_heads,
        head_dim=head_dim,
    )

    # Page 0 (Q8_0)
    k0 = torch.randn(batch, kv_heads, page_size, head_dim, dtype=torch.float16)
    v0 = torch.randn(batch, kv_heads, page_size, head_dim, dtype=torch.float16)
    slot0 = q8_pool.allocate_slot()
    q8_pool.write_page_bytes(slot0, _quantize_q8_0_tensor(k0), _quantize_q8_0_tensor(v0))
    pt.allocate_page(page_id=0, logical_pos=0, token_count=page_size, codec=CodecKind.GGML_Q8_0, pool_slot=slot0)

    # Page 1 (Q8_0)
    k1 = torch.randn(batch, kv_heads, page_size, head_dim, dtype=torch.float16)
    v1 = torch.randn(batch, kv_heads, page_size, head_dim, dtype=torch.float16)
    slot1 = q8_pool.allocate_slot()
    q8_pool.write_page_bytes(slot1, _quantize_q8_0_tensor(k1), _quantize_q8_0_tensor(v1))
    pt.allocate_page(page_id=1, logical_pos=32, token_count=page_size, codec=CodecKind.GGML_Q8_0, pool_slot=slot1)

    # Page 2 (ACTIVE FP16)
    k2 = torch.randn(batch, kv_heads, page_size, head_dim, dtype=torch.float16, device="cuda")
    v2 = torch.randn(batch, kv_heads, page_size, head_dim, dtype=torch.float16, device="cuda")
    pt.allocate_page(page_id=2, logical_pos=64, token_count=page_size, codec=CodecKind.ACTIVE_FP16)

    active_dict = {2: (k2, v2)}
    pools = {CodecKind.GGML_Q8_0: q8_pool}

    # Query token
    query = torch.randn(batch, q_heads, 1, head_dim, dtype=torch.float16, device="cuda")

    # Run direct single-token paged attention
    actual = DirectPagedAttentionEngine.decode_single_token(
        query=query,
        page_table=pt,
        pools=pools,
        active_pages=active_dict,
    )

    # Reference attention over dequantized full context
    k0_deq = DirectPagedAttentionEngine.dequantize_q8_0_page(q8_pool.k_storage[slot0], batch, kv_heads, page_size, head_dim).cuda()
    v0_deq = DirectPagedAttentionEngine.dequantize_q8_0_page(q8_pool.v_storage[slot0], batch, kv_heads, page_size, head_dim).cuda()
    k1_deq = DirectPagedAttentionEngine.dequantize_q8_0_page(q8_pool.k_storage[slot1], batch, kv_heads, page_size, head_dim).cuda()
    v1_deq = DirectPagedAttentionEngine.dequantize_q8_0_page(q8_pool.v_storage[slot1], batch, kv_heads, page_size, head_dim).cuda()

    k_full = torch.cat([k0_deq, k1_deq, k2.float()], dim=2)
    v_full = torch.cat([v0_deq, v1_deq, v2.float()], dim=2)

    # Repeat KV for GQA
    k_full_rep = k_full.repeat_interleave(q_heads // kv_heads, dim=1)
    v_full_rep = v_full.repeat_interleave(q_heads // kv_heads, dim=1)

    scale = 1.0 / math.sqrt(head_dim)
    scores = torch.matmul(query.float(), k_full_rep.transpose(-1, -2)) * scale
    probs = torch.softmax(scores, dim=-1)
    expected = torch.matmul(probs, v_full_rep).half()

    cos_sim = torch.nn.functional.cosine_similarity(actual.flatten().float(), expected.flatten().float(), dim=0).item()
    assert cos_sim > 0.999, f"Direct attention cos_sim {cos_sim:.6f} <= 0.999"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_direct_paged_attention_qwen38_geometry():
    """Validates DirectPagedAttentionEngine on Qwen3.8 geometry (24Q:4KV, head_dim 256)."""
    torch.manual_seed(38)
    batch = 1
    q_heads = 24
    kv_heads = 4
    head_dim = 256
    page_size = 64

    pt = StructureOfArraysPageTable(capacity=4, device="cuda")
    q8_pool = ContiguousBlockPool(
        codec=CodecKind.GGML_Q8_0,
        placement=PlacementLocation.GPU_DEVICE,
        max_slots=2,
        page_size=page_size,
        num_heads=kv_heads,
        head_dim=head_dim,
    )

    k = torch.randn(batch, kv_heads, page_size, head_dim, dtype=torch.float16)
    v = torch.randn(batch, kv_heads, page_size, head_dim, dtype=torch.float16)
    slot = q8_pool.allocate_slot()
    q8_pool.write_page_bytes(slot, _quantize_q8_0_tensor(k), _quantize_q8_0_tensor(v))
    pt.allocate_page(page_id=0, logical_pos=0, token_count=page_size, codec=CodecKind.GGML_Q8_0, pool_slot=slot)

    query = torch.randn(batch, q_heads, 1, head_dim, dtype=torch.float16, device="cuda")
    actual = DirectPagedAttentionEngine.decode_single_token(
        query=query,
        page_table=pt,
        pools={CodecKind.GGML_Q8_0: q8_pool},
    )

    assert actual.shape == (batch, q_heads, 1, head_dim)
    assert torch.isfinite(actual).all()
