"""Adaptive cache activation must buy memory before it costs latency."""

from __future__ import annotations

import torch

from argus_cache.core.activation import (
    AdaptiveCachePolicy,
    CacheGeometry,
    MemorySnapshot,
)
from argus_cache.models import attention_wrapper
from argus_cache import patch_model_with_argus


MIB = 1024 * 1024


def _geometry() -> CacheGeometry:
    return CacheGeometry(
        layers=24,
        batch_size=1,
        kv_heads=2,
        head_dim=64,
        element_size=2,
    )


def test_capacity_estimate_matches_exact_kv_formula():
    geometry = _geometry()

    assert geometry.exact_bytes(16_384) == 192 * MIB


def test_balanced_mode_bypasses_when_projected_cache_fits():
    policy = AdaptiveCachePolicy(
        mode="balanced",
        min_savings_bytes=32 * MIB,
        activate_ratio=0.90,
        deactivate_ratio=0.75,
    )
    roomy_gpu = MemorySnapshot(free_bytes=3 * 1024 * MIB, total_bytes=4 * 1024 * MIB)

    decision = policy.decide(
        geometry=_geometry(),
        tokens=16_384,
        page_size=1024,
        max_active_pages=2,
        tier_effective_bits=8.0,
        memory=roomy_gpu,
        was_active=False,
    )

    assert not decision.activate
    assert decision.reason == "exact-cache-fits"


def test_balanced_mode_activates_only_when_pressure_and_gain_gates_pass():
    policy = AdaptiveCachePolicy(
        mode="balanced",
        min_savings_bytes=32 * MIB,
        activate_ratio=0.90,
        deactivate_ratio=0.75,
    )
    pressured_gpu = MemorySnapshot(free_bytes=220 * MIB, total_bytes=4 * 1024 * MIB)

    decision = policy.decide(
        geometry=_geometry(),
        tokens=16_384,
        page_size=1024,
        max_active_pages=2,
        tier_effective_bits=8.0,
        memory=pressured_gpu,
        was_active=False,
    )

    assert decision.activate
    assert decision.reason == "memory-pressure"
    assert decision.estimated_savings_bytes >= 32 * MIB


def test_hysteresis_keeps_an_active_policy_on_between_thresholds():
    policy = AdaptiveCachePolicy(
        mode="balanced",
        min_savings_bytes=0,
        activate_ratio=0.90,
        deactivate_ratio=0.75,
    )
    between_thresholds = MemorySnapshot(
        free_bytes=int(0.18 * 4 * 1024 * MIB),
        total_bytes=4 * 1024 * MIB,
    )

    decision = policy.decide(
        geometry=_geometry(),
        tokens=1024,
        page_size=1024,
        max_active_pages=2,
        tier_effective_bits=8.0,
        memory=between_thresholds,
        was_active=True,
    )

    assert decision.activate
    assert decision.reason == "hysteresis-hold"


def test_latency_and_capacity_modes_are_explicit():
    common = dict(
        geometry=_geometry(),
        tokens=16_384,
        page_size=1024,
        max_active_pages=2,
        tier_effective_bits=8.0,
        memory=None,
        was_active=False,
    )

    assert not AdaptiveCachePolicy(mode="latency").decide(**common).activate
    assert AdaptiveCachePolicy(mode="capacity").decide(**common).activate


def test_invalid_hysteresis_thresholds_are_rejected():
    try:
        AdaptiveCachePolicy(activate_ratio=0.70, deactivate_ratio=0.80)
    except ValueError as exc:
        assert "deactivate_ratio" in str(exc)
    else:
        raise AssertionError("invalid hysteresis thresholds were accepted")


def test_geometry_can_be_inferred_from_model_config_and_tensor():
    class Config:
        num_hidden_layers = 24

    sample = torch.empty((2, 4, 3, 128), dtype=torch.float16)

    geometry = CacheGeometry.from_tensor(sample, Config())

    assert geometry == CacheGeometry(24, 2, 4, 128, 2)


class _ModelConfig:
    num_hidden_layers = 2
    model_type = "qwen2"


class _FakeLayerCache:
    instances = []

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.chunks = []
        self.closed = False
        self.__class__.instances.append(self)

    def push_new_tokens(self, keys, values, is_anchor=None):
        self.chunks.append((keys.clone(), values.clone()))

    def get_all_keys_values(self):
        return tuple(torch.cat(parts, dim=-2) for parts in zip(*self.chunks))

    def get_vram_usage(self):
        return sum(t.numel() * t.element_size() for pair in self.chunks for t in pair)

    def close(self):
        self.closed = True


def _wrapper(monkeypatch, *, policy, memory):
    _FakeLayerCache.instances.clear()
    monkeypatch.setattr(attention_wrapper, "PagedDynamicKVCache", _FakeLayerCache)
    return attention_wrapper.PagedDynamicQuantizedCache(
        page_size=4,
        max_active_pages=1,
        model_config=_ModelConfig(),
        activation_policy=policy,
        memory_probe=lambda: memory,
    )


def test_wrapper_uses_true_exact_bypass_when_balanced_policy_fits(monkeypatch):
    cache = _wrapper(
        monkeypatch,
        policy=AdaptiveCachePolicy(mode="balanced", min_savings_bytes=0),
        memory=MemorySnapshot(free_bytes=9000, total_bytes=10_000),
    )
    keys = torch.randn(1, 1, 2, 4)
    values = torch.randn_like(keys)

    out_k, out_v = cache.update(keys, values, layer_idx=0)

    assert torch.equal(out_k, keys)
    assert torch.equal(out_v, values)
    assert cache.layer_caches == {}
    assert cache.activation_state == "exact"
    assert cache.get_seq_length(0) == 2


def test_wrapper_migrates_exact_layers_once_when_pressure_arrives(monkeypatch):
    snapshots = iter(
        [
            MemorySnapshot(free_bytes=9000, total_bytes=10_000),
            MemorySnapshot(free_bytes=500, total_bytes=10_000),
        ]
    )
    _FakeLayerCache.instances.clear()
    monkeypatch.setattr(attention_wrapper, "PagedDynamicKVCache", _FakeLayerCache)
    cache = attention_wrapper.PagedDynamicQuantizedCache(
        page_size=4,
        max_active_pages=1,
        model_config=_ModelConfig(),
        activation_policy=AdaptiveCachePolicy(mode="balanced", min_savings_bytes=0),
        memory_probe=lambda: next(snapshots),
    )
    first_k = torch.randn(1, 1, 2, 4)
    first_v = torch.randn_like(first_k)
    next_k = torch.randn(1, 1, 300, 4)
    next_v = torch.randn_like(next_k)

    cache.update(first_k, first_v, layer_idx=0)
    out_k, out_v = cache.update(next_k, next_v, layer_idx=0)

    assert cache.activation_state == "argus"
    assert cache._exact_layers == {}
    assert len(_FakeLayerCache.instances) == 1
    assert len(_FakeLayerCache.instances[0].chunks) == 2
    assert torch.equal(out_k, torch.cat((first_k, next_k), dim=-2))
    assert torch.equal(out_v, torch.cat((first_v, next_v), dim=-2))


def test_balanced_activation_uses_the_simple_fp8_pipeline(monkeypatch):
    cache = _wrapper(
        monkeypatch,
        policy=AdaptiveCachePolicy(
            mode="balanced",
            expected_tokens=300,
            min_savings_bytes=0,
        ),
        memory=MemorySnapshot(free_bytes=500, total_bytes=10_000),
    )
    keys = torch.randn(1, 1, 2, 4)

    cache.update(keys, keys, layer_idx=0)

    pipeline = _FakeLayerCache.instances[0].kwargs["pipeline"]
    assert [tier.name for tier in pipeline.tiers] == ["fp8"]
    assert pipeline.tiers[0].max_pages == -1


def test_direct_attention_returns_a_manager_tag_instead_of_reconstructing(monkeypatch):
    _FakeLayerCache.instances.clear()
    monkeypatch.setattr(attention_wrapper, "PagedDynamicKVCache", _FakeLayerCache)
    cache = attention_wrapper.PagedDynamicQuantizedCache(
        page_size=4,
        max_active_pages=1,
        model_config=_ModelConfig(),
        activation_policy=AdaptiveCachePolicy(mode="capacity"),
        direct_attention=True,
    )
    keys = torch.randn(1, 1, 2, 4)
    values = torch.randn_like(keys)

    out_k, out_v = cache.update(keys, values, layer_idx=0)

    assert out_k.data_ptr() == keys.data_ptr()
    assert out_v.data_ptr() == values.data_ptr()
    assert out_k._argus_cache_manager is _FakeLayerCache.instances[0]
    assert out_v._argus_cache_manager is _FakeLayerCache.instances[0]


def test_direct_legacy_constructor_keeps_the_research_pipeline(monkeypatch):
    _FakeLayerCache.instances.clear()
    monkeypatch.setattr(attention_wrapper, "PagedDynamicKVCache", _FakeLayerCache)
    cache = attention_wrapper.PagedDynamicQuantizedCache(
        page_size=4,
        max_active_pages=1,
        activation_policy=AdaptiveCachePolicy(mode="capacity"),
    )
    keys = torch.randn(1, 1, 2, 4)

    cache.update(keys, keys, layer_idx=0)

    pipeline = _FakeLayerCache.instances[0].kwargs["pipeline"]
    assert [tier.name for tier in pipeline.tiers] == [
        "fp8",
        "int8",
        "int4",
        "int2",
        "one_bit",
        "jl",
    ]


def test_reset_clears_both_exact_and_argus_storage(monkeypatch):
    exact = _wrapper(
        monkeypatch,
        policy=AdaptiveCachePolicy(mode="latency", min_savings_bytes=0),
        memory=None,
    )
    keys = torch.randn(1, 1, 2, 4)
    exact.update(keys, keys, layer_idx=0)
    exact.reset()
    assert exact.get_seq_length(0) == 0

    active = _wrapper(
        monkeypatch,
        policy=AdaptiveCachePolicy(mode="capacity", min_savings_bytes=0),
        memory=None,
    )
    active.update(keys, keys, layer_idx=0)
    manager = _FakeLayerCache.instances[-1]
    active.reset()
    assert manager.closed
    assert active.get_seq_length(0) == 0


def test_model_patch_supplies_geometry_and_policy_to_the_cache():
    policy = AdaptiveCachePolicy(mode="latency")

    class Model:
        config = _ModelConfig()

        @staticmethod
        def prepare_inputs_for_generation(*args, **kwargs):
            return kwargs

    model = patch_model_with_argus(Model(), activation_policy=policy)

    prepared = model.prepare_inputs_for_generation(torch.ones((1, 2), dtype=torch.long))
    cache = prepared["past_key_values"]
    assert cache.model_config is model.config
    assert cache.activation_policy is policy


def test_model_patch_enables_registered_direct_attention():
    class Model:
        config = _ModelConfig()

        def set_attn_implementation(self, name):
            self.attention_name = name

        @staticmethod
        def prepare_inputs_for_generation(*args, **kwargs):
            return kwargs

    model = patch_model_with_argus(Model())
    prepared = model.prepare_inputs_for_generation(torch.ones((1, 2), dtype=torch.long))

    assert model.attention_name == "argus"
    assert prepared["past_key_values"].direct_attention


def test_model_patch_preserves_unknown_model_attention():
    class UnknownConfig:
        num_hidden_layers = 2
        model_type = "custom_unregistered_model"

    class Model:
        config = UnknownConfig()

        def set_attn_implementation(self, name):
            raise AssertionError("unregistered model attention was replaced")

        @staticmethod
        def prepare_inputs_for_generation(*args, **kwargs):
            return kwargs

    model = patch_model_with_argus(Model())
    prepared = model.prepare_inputs_for_generation(torch.ones((1, 2), dtype=torch.long))

    assert not prepared["past_key_values"].direct_attention
