import sys
import os
import torch
import pytest
from transformers import LlamaConfig, LlamaForCausalLM

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from argus_cache import patch_model_with_argus, PagedDynamicQuantizedCache
from argus_cache.core.tier_registry import PipelineConfig, TierSpec
from argus_cache.backends.quantization import FP8Backend, INT4Backend, JLProjectionBackend

def test_llama_gqa_integration_and_generation():
    """
    Verifies that a Llama model using Grouped-Query Attention (GQA) can be successfully 
    patched, run forward passes, and generate text using the ARGUS cache backend.
    """
    config = LlamaConfig(
        vocab_size=100,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=1,  # GQA: 4 query heads vs 1 KV head (1:4 ratio)
        max_position_embeddings=512,
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LlamaForCausalLM(config).half().to(device)
    
    # Patch using small limits to trigger page allocation & compression cascade quickly
    patch_model_with_argus(
        model,
        page_size=4,
        max_active_pages=1,
        max_fp8_pages=1,
        max_int8_pages=1,
        max_int4_pages=1,
        max_int2_pages=1,
        max_one_bit_pages=1,
        sink_tokens=2
    )
    
    # Generate random tokens
    input_ids = torch.randint(0, 100, (1, 8), device=device)
    
    # Run a simple forward pass to check dimensions and outputs
    with torch.no_grad():
        outputs = model(input_ids)
    assert outputs.logits.shape == (1, 8, 100)
    
    # Test generation (auto-regressive execution).
    # min_new_tokens forces exactly max_new_tokens to be produced — without it
    # this randomly-initialized (untrained) model can emit the EOS token at
    # any step by pure chance and stop generation early, making the expected
    # output shape flaky.
    with torch.no_grad():
        gen_out = model.generate(input_ids, max_new_tokens=10, min_new_tokens=10)
    assert gen_out.shape == (1, 18)

def test_llama_with_custom_pipeline_config():
    """
    Verifies that a patched Llama model can function normally when configured 
    with a custom PipelineConfig containing custom TierSpecs.
    """
    config = LlamaConfig(
        vocab_size=100,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,  # GQA (1:2 ratio)
        max_position_embeddings=512,
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LlamaForCausalLM(config).half().to(device)
    
    # Custom 3-tier pipeline config
    custom_pipeline = PipelineConfig(
        tiers=[
            TierSpec("fp8", FP8Backend(), max_pages=1, priority=1),
            TierSpec("int4", INT4Backend(), max_pages=1, priority=2),
            TierSpec("jl", JLProjectionBackend(), max_pages=-1, priority=3)
        ],
        page_size=8,
        sink_tokens=4,
        threshold_sigma=2.0
    )
    
    patch_model_with_argus(model, pipeline=custom_pipeline)
    
    input_ids = torch.randint(0, 100, (1, 12), device=device)
    
    # Verify forward and generate works under a custom pipeline configuration
    with torch.no_grad():
        outputs = model(input_ids)
    assert outputs.logits.shape == (1, 12, 100)
    
    with torch.no_grad():
        gen_out = model.generate(input_ids, max_new_tokens=8, min_new_tokens=8)
    assert gen_out.shape == (1, 20)
