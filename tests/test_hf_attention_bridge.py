"""Model-aware Hugging Face attention bridge tests."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from transformers import LlamaConfig, LlamaForCausalLM, Qwen2Config, Qwen2ForCausalLM

from argus_cache import AdaptiveCachePolicy, patch_model_with_argus
from argus_cache.models.attention_wrapper import PagedDynamicQuantizedCache
from argus_cache.models.hf_attention import (
    AttentionAdapter,
    argus_attention_forward,
    register_attention_adapter,
    unregister_attention_adapter,
)


class _Manager:
    def __init__(self, keys: torch.Tensor, values: torch.Tensor):
        self.keys = keys
        self.values = values
        self.native_calls = 0
        self.last_query = None

    def get_seq_length(self):
        return self.keys.shape[-2]

    def inplace_paged_attention(self, query, scale=None):
        self.native_calls += 1
        self.last_query = query
        groups = query.shape[1] // self.keys.shape[1]
        keys = self.keys.repeat_interleave(groups, dim=1)
        values = self.values.repeat_interleave(groups, dim=1)
        return torch.nn.functional.scaled_dot_product_attention(
            query,
            keys,
            values,
            scale=scale,
            is_causal=False,
        )

    def get_all_keys_values(self):
        return self.keys, self.values


def _tag(tensor: torch.Tensor, manager: _Manager) -> torch.Tensor:
    tensor._argus_cache_manager = manager
    return tensor


def _module(
    model_type="qwen2", *, training=False, sliding_window=None, num_key_value_groups=1
):
    return SimpleNamespace(
        config=SimpleNamespace(model_type=model_type),
        training=training,
        num_key_value_groups=num_key_value_groups,
        sliding_window=sliding_window,
    )


def test_qwen2_decode_uses_native_manager_and_formats_output():
    keys = torch.randn(1, 2, 5, 8)
    values = torch.randn_like(keys)
    query = torch.randn(1, 2, 1, 8)
    manager = _Manager(keys, values)

    output, weights = argus_attention_forward(
        _module(),
        query,
        _tag(keys[:, :, -1:], manager),
        values[:, :, -1:],
        attention_mask=None,
        scaling=8**-0.5,
    )

    expected = manager.inplace_paged_attention(query, scale=8**-0.5)
    assert manager.native_calls == 2
    assert weights is None
    assert output.shape == (1, 1, 2, 8)
    assert torch.allclose(output, expected.transpose(1, 2), atol=1e-5)


def test_llama_decode_uses_the_same_registry_contract():
    """A second GQA family must not require core model-name branching."""
    keys = torch.randn(1, 1, 5, 8)
    values = torch.randn_like(keys)
    query = torch.randn(1, 4, 1, 8)
    manager = _Manager(keys, values)

    output, weights = argus_attention_forward(
        _module("llama", num_key_value_groups=4),
        query,
        _tag(keys[:, :, -1:], manager),
        values[:, :, -1:],
        attention_mask=None,
        scaling=8**-0.5,
    )

    assert manager.native_calls == 1
    assert weights is None
    expected = torch.nn.functional.scaled_dot_product_attention(
        query,
        keys,
        values,
        scale=8**-0.5,
        enable_gqa=True,
    ).transpose(1, 2)
    assert torch.allclose(output, expected, atol=1e-5)


def test_qwen2_prefill_reconstructs_and_falls_back_to_sdpa():
    keys = torch.randn(1, 2, 4, 8)
    values = torch.randn_like(keys)
    query = torch.randn(1, 2, 2, 8)
    manager = _Manager(keys, values)

    output, _ = argus_attention_forward(
        _module(),
        query,
        _tag(keys[:, :, -2:], manager),
        values[:, :, -2:],
        attention_mask=None,
        scaling=8**-0.5,
    )

    assert manager.native_calls == 0
    assert output.shape == (1, 2, 2, 8)


def test_unknown_model_fails_closed_to_reconstructed_sdpa():
    keys = torch.randn(1, 2, 5, 8)
    values = torch.randn_like(keys)
    query = torch.randn(1, 2, 1, 8)
    manager = _Manager(keys, values)

    output, _ = argus_attention_forward(
        _module("unknown_architecture"),
        query,
        _tag(keys[:, :, -1:], manager),
        values[:, :, -1:],
        attention_mask=None,
        scaling=8**-0.5,
    )

    assert manager.native_calls == 0
    assert output.shape == (1, 1, 2, 8)


def test_qwen2_masked_decode_falls_back_instead_of_ignoring_mask():
    keys = torch.randn(1, 2, 5, 8)
    values = torch.randn_like(keys)
    query = torch.randn(1, 2, 1, 8)
    manager = _Manager(keys, values)
    mask = torch.zeros(1, 1, 1, 5)
    mask[..., 0] = torch.finfo(mask.dtype).min

    output, _ = argus_attention_forward(
        _module(),
        query,
        _tag(keys[:, :, -1:], manager),
        values[:, :, -1:],
        attention_mask=mask,
        scaling=8**-0.5,
    )

    expected = torch.nn.functional.scaled_dot_product_attention(
        query,
        keys,
        values,
        attn_mask=mask,
        scale=8**-0.5,
        is_causal=False,
    ).transpose(1, 2)
    assert manager.native_calls == 0
    assert torch.allclose(output, expected)


def test_model_specific_adapter_can_override_native_feeding():
    class CustomAdapter(AttentionAdapter):
        def can_use_native(self, module, query, attention_mask, **kwargs):
            return query.shape[-2] == 1

        def prepare_query(self, module, query):
            return query * 2

    register_attention_adapter("custom_arch", CustomAdapter())
    try:
        keys = torch.randn(1, 2, 3, 8)
        values = torch.randn_like(keys)
        query = torch.randn(1, 2, 1, 8)
        manager = _Manager(keys, values)

        argus_attention_forward(
            _module("custom_arch"),
            query,
            _tag(keys[:, :, -1:], manager),
            values[:, :, -1:],
            attention_mask=None,
            scaling=8**-0.5,
        )

        expected = torch.nn.functional.scaled_dot_product_attention(
            query * 2, keys, values, scale=8**-0.5, is_causal=False
        )
        assert manager.native_calls == 1
        assert torch.equal(manager.last_query, query * 2)
        assert expected.shape == query.shape
    finally:
        unregister_attention_adapter("custom_arch")


def test_transformers_generation_injects_argus_before_dynamic_cache():
    config = Qwen2Config(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    model = patch_model_with_argus(
        Qwen2ForCausalLM(config).eval(),
        page_size=8,
        activation_policy=AdaptiveCachePolicy(mode="latency"),
    )

    result = model.generate(
        torch.tensor([[1, 2, 3]]),
        max_new_tokens=2,
        return_dict_in_generate=True,
    )

    assert isinstance(result.past_key_values, PagedDynamicQuantizedCache)
    assert result.past_key_values.get_seq_length() == 4


def test_llama_generation_uses_the_registered_adapter_without_core_changes():
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    model = patch_model_with_argus(
        LlamaForCausalLM(config).eval(),
        page_size=8,
        activation_policy=AdaptiveCachePolicy(mode="latency"),
    )
    assert model.config._attn_implementation == "argus"

    result = model.generate(
        torch.tensor([[1, 2, 3]]),
        max_new_tokens=2,
        min_new_tokens=2,
        return_dict_in_generate=True,
    )

    assert isinstance(result.past_key_values, PagedDynamicQuantizedCache)
    assert result.past_key_values.get_seq_length() == 4
