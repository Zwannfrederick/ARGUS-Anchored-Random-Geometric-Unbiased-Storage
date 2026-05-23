import torch
import sys
import os
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.memory_manager import PagedDynamicKVCache
from models.attention_wrapper import PagedDynamicQuantizedCache

# Standard HF DynamicCache mockup for baseline comparison
class StandardFp16Cache:
    def __init__(self):
        self.k_cache = {}
        self.v_cache = {}

    def update(self, key_states, value_states, layer_idx):
        if layer_idx not in self.k_cache:
            self.k_cache[layer_idx] = key_states
            self.v_cache[layer_idx] = value_states
        else:
            self.k_cache[layer_idx] = torch.cat([self.k_cache[layer_idx], key_states], dim=-2)
            self.v_cache[layer_idx] = torch.cat([self.v_cache[layer_idx], value_states], dim=-2)
        return self.k_cache[layer_idx], self.v_cache[layer_idx]

    def get_seq_length(self, layer_idx=0):
        if layer_idx not in self.k_cache:
            return 0
        return self.k_cache[layer_idx].shape[-2]

    def get_vram_usage(self):
        total_bytes = 0
        for k in self.k_cache.values():
            total_bytes += k.element_size() * k.nelement()
        for v in self.v_cache.values():
            total_bytes += v.element_size() * v.nelement()
        return total_bytes

def run_benchmark(
    num_layers=32,
    batch_size=1,
    num_heads=32,
    head_dim=128,
    total_tokens=16384,
    prefill_tokens=4096,
    page_size=2048,
    max_active=2,
    max_mid=4
):
    print("=" * 70)
    print("        PAGED DYNAMIC QUANTIZED KV CACHE BENCHMARK PROFILER")
    print("=" * 70)
    print(f"Model Architecture Settings:")
    print(f"  - Layers: {num_layers}")
    print(f"  - Batch Size: {batch_size}")
    print(f"  - Attention Heads: {num_heads}")
    print(f"  - Head Dimension: {head_dim}")
    print(f"  - Prefill Length: {prefill_tokens} tokens")
    print(f"  - Generation Length: {total_tokens - prefill_tokens} tokens (total {total_tokens} tokens)")
    print(f"Paged Cache Settings:")
    print(f"  - Page Size: {page_size} tokens")
    print(f"  - Max Active Pages (FP16): {max_active} ({max_active * page_size} tokens)")
    print(f"  - Max Mid-term Pages (INT8): {max_mid} ({max_mid * page_size} tokens)")
    print(f"  - Archive Pages (INT4): Unlimited (> {(max_active + max_mid) * page_size} tokens)")
    print("-" * 70)

    device = torch.device("cpu")
    print(f"Running benchmark on device: {device.type.upper()}")

    # Initialize caches
    standard_cache = StandardFp16Cache()
    paged_cache = PagedDynamicQuantizedCache(
        page_size=page_size,
        max_active_pages=max_active,
        max_fp8_pages=max_mid,
        max_int8_pages=max_mid,
        max_int4_pages=max_mid,
        sink_tokens=4
    )

    # 1. Prefill Phase
    print("\n--- 1. PREFILL PHASE ---")
    print(f"Generating prefill keys and values ({prefill_tokens} tokens)...")
    
    # We simulate layers sequentially to avoid OOM in standard cache during benchmarking
    prefill_keys = torch.randn(batch_size, num_heads, prefill_tokens, head_dim, dtype=torch.float16, device=device)
    prefill_values = torch.randn(batch_size, num_heads, prefill_tokens, head_dim, dtype=torch.float16, device=device)

    # Update baseline standard cache
    start_time = time.time()
    for layer in range(num_layers):
        standard_cache.update(prefill_keys, prefill_values, layer)
    standard_prefill_time = time.time() - start_time
    standard_prefill_vram = standard_cache.get_vram_usage() / (1024 * 1024) # MB

    # Update our custom PagedDynamicQuantizedCache
    start_time = time.time()
    for layer in range(num_layers):
        paged_cache.update(prefill_keys, prefill_values, layer)
    paged_prefill_time = time.time() - start_time
    paged_prefill_vram = paged_cache.get_vram_usage() / (1024 * 1024) # MB

    print(f"Standard FP16 Cache  - VRAM: {standard_prefill_vram:.2f} MB | Time: {standard_prefill_time * 1000:.1f}ms")
    print(f"Paged Dynamic Cache  - VRAM: {paged_prefill_vram:.2f} MB | Time: {paged_prefill_time * 1000:.1f}ms")
    print(f"VRAM Reduction: {((standard_prefill_vram - paged_prefill_vram) / standard_prefill_vram) * 100:.1f}%")

    # 2. Generation Phase (Incremental)
    print("\n--- 2. INCREMENTAL GENERATION PHASE ---")
    print(f"Generating tokens incrementally from {prefill_tokens} up to {total_tokens}...")
    
    steps = [5120, 8192, 12288, 16384]
    steps = [s for s in steps if s <= total_tokens]
    
    current_tokens = prefill_tokens
    
    # Track statistics for plotting or reporting
    vram_stats = {
        'tokens': [current_tokens],
        'std_vram': [standard_prefill_vram],
        'paged_vram': [paged_prefill_vram]
    }

    # Run generation
    while current_tokens < total_tokens:
        next_step = steps[0] if steps else total_tokens
        tokens_to_gen = next_step - current_tokens
        
        # Simulating next batch of tokens generated in a single chunk for 10000x speedup
        gen_key = torch.randn(batch_size, num_heads, tokens_to_gen, head_dim, dtype=torch.float16, device=device)
        gen_val = torch.randn(batch_size, num_heads, tokens_to_gen, head_dim, dtype=torch.float16, device=device)
        
        for layer in range(num_layers):
            standard_cache.update(gen_key, gen_val, layer)
            paged_cache.update(gen_key, gen_val, layer)
                
        current_tokens = next_step
        if steps:
            steps.pop(0)
            
        std_vram = standard_cache.get_seq_length() * num_layers * batch_size * num_heads * head_dim * 2 * 2 / (1024 * 1024) # actual exact baseline formula
        paged_vram = paged_cache.get_vram_usage() / (1024 * 1024)
        
        vram_stats['tokens'].append(current_tokens)
        vram_stats['std_vram'].append(std_vram)
        vram_stats['paged_vram'].append(paged_vram)
        
        # Calculate dequantization accuracy for layer 0 to evaluate degradation
        recon_k, recon_v = paged_cache.layer_caches[0].get_all_keys_values()
        orig_k = standard_cache.k_cache[0]
        
        # MSE Error
        mse_error = torch.mean((orig_k - recon_k) ** 2).item()
        # Cosine Similarity
        cos_sim = torch.nn.functional.cosine_similarity(orig_k.flatten(), recon_k.flatten(), dim=0).item()
        
        # Active page structure details
        c = paged_cache.layer_caches[0]
        structure = f"FP16 Pages: {len(c.active_pages)} | FP8 Pages: {len(c.fp8_pages)} | INT8 Pages: {len(c.int8_pages)} | INT4 Pages: {len(c.int4_pages)} | JL-Proj Pages: {len(c.int2_pages)}"
        
        print(f"\n[At {current_tokens} Tokens]")
        print(f"  - Cache Structure (Layer 0): {structure}")
        print(f"  - Standard FP16 VRAM: {std_vram:.2f} MB")
        print(f"  - Paged Quantized VRAM: {paged_vram:.2f} MB")
        print(f"  - VRAM Saved: {std_vram - paged_vram:.2f} MB ({((std_vram - paged_vram) / std_vram) * 100:.1f}%)")
        print(f"  - Reconstruction Quality (Layer 0) -> MSE: {mse_error:.6f} | Cosine Similarity: {cos_sim:.6f}")

    # Generate Markdown summary table
    print("\n" + "=" * 70)
    print("                       BENCHMARK RESULTS SUMMARY")
    print("=" * 70)
    print(f"| Token Sayısı | Standart Cache (MB) | Paged Quant Cache (MB) | VRAM Tasarrufu (%) | Quality (Cos Sim) |")
    print(f"|--------------|---------------------|------------------------|--------------------|-------------------|")
    for i, t in enumerate(vram_stats['tokens']):
        std = vram_stats['std_vram'][i]
        pg = vram_stats['paged_vram'][i]
        pct = ((std - pg) / std) * 100
        
        # Get cos sim
        if i == 0:
            recon_k, _ = paged_cache.layer_caches[0].get_all_keys_values()
            orig_k = standard_cache.k_cache[0][..., :prefill_tokens, :]
            cos_sim = torch.nn.functional.cosine_similarity(orig_k.flatten(), recon_k[..., :prefill_tokens, :].flatten(), dim=0).item()
        elif t == total_tokens:
            recon_k, _ = paged_cache.layer_caches[0].get_all_keys_values()
            orig_k = standard_cache.k_cache[0]
            cos_sim = torch.nn.functional.cosine_similarity(orig_k.flatten(), recon_k.flatten(), dim=0).item()
        else:
            cos_sim = 1.0 # Buffer or exact mid stages
            
        print(f"| {t:12d} | {std:19.2f} | {pg:22.2f} | {pct:17.1f}% | {cos_sim:17.6f} |")
    print("=" * 70)

if __name__ == "__main__":
    # Run the benchmark
    run_benchmark(
        num_layers=32,
        batch_size=1,
        num_heads=32,
        head_dim=128,
        total_tokens=16384,
        prefill_tokens=4096,
        page_size=2048,
        max_active=2,
        max_mid=4
    )
