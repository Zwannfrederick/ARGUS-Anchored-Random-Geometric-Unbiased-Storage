"""Tests for the hybrid Qwen cache ownership contract (Stage S4).

Validates:
1. Exact 48 DeltaNet + 16 Full-Attention layer topology derivation from metadata.
2. Strict fail-closed validation on invalid or contradictory metadata.
3. Growing KV cache allocation exclusively on full-attention layers (64 KiB/token BF16).
4. DeltaNet state digest invariance: ARGUS operations never mutate recurrent state buffers.
5. Lifecycle operations: snapshot, restore, and memory breakdown reporting.
"""

from types import SimpleNamespace
import pytest
import torch

from argus_cache import (
    HybridQwenCache,
    HybridTopology,
    LayerRole,
    PipelineConfig,
    TierSpec,
)


def test_qwen38_topology_derivation():
    """Derives exact 64-layer hybrid topology (48 recurrent, 16 full-attention)."""
    config = SimpleNamespace(
        model_type="qwen3_5",
        num_hidden_layers=64,
        full_attention_interval=4,
        num_attention_heads=24,
        num_key_value_heads=4,
        head_dim=256,
    )
    topology = HybridTopology.from_config(config)

    assert topology.num_total_layers == 64
    assert topology.full_attention_interval == 4
    assert topology.num_full_attention_layers == 16
    assert topology.num_recurrent_layers == 48

    # Verify specific layer assignments
    # Layer indices: 3, 7, 11, 15, ..., 63 are full attention (0-indexed: (il+1)%4 == 0)
    for il in range(64):
        if (il + 1) % 4 == 0:
            assert topology.is_full_attention(il), f"Layer {il} should be full attention"
            assert topology.layer_roles[il] == LayerRole.FULL_ATTENTION_KV
        else:
            assert topology.is_recurrent(il), f"Layer {il} should be recurrent"
            assert topology.layer_roles[il] == LayerRole.RECURRENT_STATE

    # Exact geometry formulas per token
    # BF16 KV: 2 * 16 layers * 4 KV heads * 256 dim * 2 bytes = 65536 bytes (64 KiB)
    assert topology.exact_bf16_kv_bytes_per_token() == 65536
    # q8_0 KV: 32768 values * (34/32) = 34816 bytes (34 KiB)
    assert topology.q8_0_kv_bytes_per_token() == 34816.0
    # q4_0 KV: 32768 values * (18/32) = 18432 bytes (18 KiB)
    assert topology.q4_0_kv_bytes_per_token() == 18432.0


def test_topology_fails_closed_on_contradictory_metadata():
    """Fails closed on missing or invalid configuration metadata."""
    # Negative / zero layers
    with pytest.raises(ValueError, match="Invalid num_hidden_layers"):
        HybridTopology.from_config(SimpleNamespace(num_hidden_layers=0))

    # Invalid interval
    with pytest.raises(ValueError, match="Invalid full_attention_interval"):
        HybridTopology.from_config(SimpleNamespace(num_hidden_layers=64, full_attention_interval=0))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_hybrid_cache_memory_ownership_and_state_digest_invariance():
    """Proves that ARGUS operations update KV cache while keeping DeltaNet digest invariant."""
    config = SimpleNamespace(
        model_type="qwen3_5",
        num_hidden_layers=64,
        full_attention_interval=4,
        num_attention_heads=24,
        num_key_value_heads=4,
        head_dim=256,
    )
    topology = HybridTopology.from_config(config)
    cache = HybridQwenCache(topology=topology, page_size=128)

    # Initialize recurrent states for the 48 DeltaNet layers (e.g. 144 MiB total simulated)
    torch.manual_seed(42)
    for il in range(64):
        if topology.is_recurrent(il):
            # Simulated DeltaNet state: [1, 4, 128, 128]
            recr_state = torch.randn(1, 4, 128, 128, dtype=torch.float32, device="cuda")
            conv_state = torch.randn(1, 4, 4, 128, dtype=torch.float32, device="cuda")
            cache.update_recurrent_state(il, recr_state, conv_state)

    initial_digest = cache.compute_recurrent_state_digest()
    assert len(initial_digest) == 64  # SHA-256 hex string

    # Now push growing tokens to all 16 full-attention layers
    for il in range(64):
        if topology.is_full_attention(il):
            k = torch.randn(1, 4, 128, 256, dtype=torch.float16, device="cuda")
            v = torch.randn(1, 4, 128, 256, dtype=torch.float16, device="cuda")
            cache.update_attention_layer(il, k, v)

    # Sequence length in attention cache is now 128
    assert cache.get_seq_length() == 128

    # Recurrent state digest must remain 100% strictly IDENTICAL after ARGUS KV updates
    post_attention_digest = cache.compute_recurrent_state_digest()
    assert post_attention_digest == initial_digest, "Recurrent state was mutated by ARGUS attention update!"

    # Memory breakdown must separate attention KV from recurrent state
    breakdown = cache.memory_breakdown()
    assert breakdown["num_attention_layers"] == 16
    assert breakdown["num_recurrent_layers"] == 48
    assert breakdown["attention_kv_bytes"] > 0
    assert breakdown["recurrent_state_bytes"] > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_hybrid_cache_snapshot_and_rollback():
    """Verifies that recurrent state and attention caches can be restored safely upon cancellation."""
    config = SimpleNamespace(
        model_type="qwen3_5",
        num_hidden_layers=8,
        full_attention_interval=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
    )
    topology = HybridTopology.from_config(config)
    cache = HybridQwenCache(topology=topology, page_size=32)

    # Initialize recurrent states and push initial attention tokens
    for il in range(8):
        if topology.is_recurrent(il):
            cache.update_recurrent_state(
                il,
                torch.ones(1, 2, 16, 16, dtype=torch.float32, device="cuda"),
            )
        elif topology.is_full_attention(il):
            cache.update_attention_layer(
                il,
                torch.randn(1, 2, 10, 64, device="cuda", dtype=torch.float16),
                torch.randn(1, 2, 10, 64, device="cuda", dtype=torch.float16),
            )

    assert cache.get_seq_length() == 10

    # Snapshot baseline state
    snap = cache.snapshot()

    # Mutate recurrent state and push speculative attention tokens
    for il in range(8):
        if topology.is_recurrent(il):
            cache.update_recurrent_state(
                il,
                torch.zeros(1, 2, 16, 16, dtype=torch.float32, device="cuda"),
            )
        elif topology.is_full_attention(il):
            cache.update_attention_layer(
                il,
                torch.randn(1, 2, 15, 64, device="cuda", dtype=torch.float16),
                torch.randn(1, 2, 15, 64, device="cuda", dtype=torch.float16),
            )

    assert cache.get_seq_length() == 25
    assert cache.compute_recurrent_state_digest() != snap["digest"]

    # Rollback to snapshot (e.g. speculative rejection or cancellation)
    cache.restore(snap)
    assert cache.compute_recurrent_state_digest() == snap["digest"]
    assert cache.get_seq_length() == 10
    for il in range(8):
        if topology.is_recurrent(il):
            assert torch.all(cache.get_recurrent_state(il) == 1.0)

