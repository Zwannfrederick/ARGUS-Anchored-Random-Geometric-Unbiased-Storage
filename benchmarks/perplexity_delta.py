#!/usr/bin/env python3
"""
ARGUS Perplexity Delta (ΔPPL) Evaluation Benchmark
Measures language modeling perplexity difference between vanilla PyTorch KV cache 
and the ARGUS multi-tier KV cache.
"""

import sys
import os
import argparse
import json
import torch
from transformers import LlamaConfig, LlamaForCausalLM

# Ensure the workspace is in the python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from argus_cache import patch_model_with_argus

def run_ppl_benchmark(device, num_tokens=256, seed=42):
    print("=" * 90)
    print(f"  ARGUS PERPLEXITY DELTA (ΔPPL) BENCHMARK (Seed: {seed}, Tokens: {num_tokens})")
    print("=" * 90)
    
    # 1. Initialize a deterministic config and model (small size for fast, local evaluation)
    torch.manual_seed(seed)
    config = LlamaConfig(
        vocab_size=1000,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
    )
    
    # Generate deterministic evaluation text (token IDs)
    input_ids = torch.randint(0, config.vocab_size, (1, num_tokens), device=device)
    
    # Create models
    model_vanilla = LlamaForCausalLM(config).half().to(device)
    model_vanilla.eval()
    
    model_argus = LlamaForCausalLM(config).half().to(device)
    # Load same weights
    model_argus.load_state_dict(model_vanilla.state_dict())
    model_argus.eval()
    
    # Patch ARGUS model
    patch_model_with_argus(
        model_argus,
        page_size=16,
        max_active_pages=2,
        max_fp8_pages=2,
        max_int8_pages=2,
        max_int4_pages=2,
        max_int2_pages=2,
        max_one_bit_pages=2,
        sink_tokens=4
    )
    
    # 2. Compute Vanilla Perplexity
    print("Evaluating Vanilla KV Cache Perplexity...")
    with torch.no_grad():
        outputs_vanilla = model_vanilla(input_ids)
        logits_vanilla = outputs_vanilla.logits
        
        shift_logits_v = logits_vanilla[..., :-1, :].contiguous()
        shift_labels_v = input_ids[..., 1:].contiguous()
        
        loss_fct = torch.nn.CrossEntropyLoss()
        loss_v = loss_fct(shift_logits_v.view(-1, config.vocab_size), shift_labels_v.view(-1))
        ppl_vanilla = torch.exp(loss_v).item()
        
    print(f"Vanilla PPL: {ppl_vanilla:.4f}")
    
    # 3. Compute ARGUS Perplexity
    print("Evaluating ARGUS KV Cache Perplexity...")
    # Reset generation steps or states if necessary, but forward pass handles it
    with torch.no_grad():
        outputs_argus = model_argus(input_ids)
        logits_argus = outputs_argus.logits
        
        shift_logits_a = logits_argus[..., :-1, :].contiguous()
        shift_labels_a = input_ids[..., 1:].contiguous()
        
        loss_a = loss_fct(shift_logits_a.view(-1, config.vocab_size), shift_labels_a.view(-1))
        ppl_argus = torch.exp(loss_a).item()
        
    print(f"ARGUS PPL: {ppl_argus:.4f}")
    
    # 4. Compute Delta
    delta_ppl = ppl_argus - ppl_vanilla
    print(f"ΔPPL: {delta_ppl:+.6f}")
    print("=" * 90)
    
    return {
        "ppl_vanilla": ppl_vanilla,
        "ppl_argus": ppl_argus,
        "delta_ppl": delta_ppl
    }

def main():
    parser = argparse.ArgumentParser(description="Deterministic Perplexity Delta Benchmark")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--tokens", type=int, default=256, help="Number of tokens to evaluate")
    parser.add_argument("--export", type=str, default="benchmarks/perplexity_results.json", help="Path to export JSON results")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run on")
    
    args = parser.parse_args()
    
    results = run_ppl_benchmark(device=args.device, num_tokens=args.tokens, seed=args.seed)
    
    if args.export:
        os.makedirs(os.path.dirname(os.path.abspath(args.export)), exist_ok=True)
        with open(args.export, "w", encoding="utf-8") as f:
            json.dump({
                "benchmark": "Perplexity Delta (ΔPPL)",
                "parameters": {
                    "seed": args.seed,
                    "tokens": args.tokens,
                    "device": args.device
                },
                "results": results
            }, f, indent=4)
        print(f"Results successfully exported to: {args.export}\n")

if __name__ == "__main__":
    main()
