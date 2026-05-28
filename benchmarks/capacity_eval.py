#!/usr/bin/env python3
"""
ARGUS: Context Capacity stress benchmark
Measures maximum stable context window sizes, tracks OOM ceilings,
and calculates Usable Context Expansion Ratios compared to standard caching.
"""

import sys
import os
import time
import torch

# Add parent directory to path so we can import core
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.memory_manager import PagedDynamicKVCache, ArgusConfig

def format_bytes(b):
    """Formats bytes to a readable string."""
    if b >= 1024**3:
        return f"{b / 1024**3:.2f} GB"
    elif b >= 1024**2:
        return f"{b / 1024**2:.2f} MB"
    else:
        return f"{b / 1024:.2f} KB"

def run_standard_capacity_test(max_length=65536, step_size=4096):
    """Stress tests standard FP16 cache append to find OOM threshold."""
    print("\n\033[1;31m[STRESS] Running Capacity Test on Standard FP16 Caching...\033[0m")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if device == "cpu":
        print("[WARNING] CUDA is not available. OOM ceilings will be simulated using RAM thresholds.")
        
    k_cache = None
    v_cache = None
    
    stable_length = 0
    oom_triggered = False
    
    try:
        for length in range(step_size, max_length + 1, step_size):
            # Generate new block of tokens
            k = torch.randn(1, 4, step_size, 16, dtype=torch.float16, device=device)
            v = torch.randn(1, 4, step_size, 16, dtype=torch.float16, device=device)
            
            if k_cache is None:
                k_cache = k
                v_cache = v
            else:
                k_cache = torch.cat([k_cache, k], dim=-2)
                v_cache = torch.cat([v_cache, v], dim=-2)
                
            # Perform dummy attention lookup to allocate workspace memory
            q = torch.randn(1, 4, 1, 16, dtype=torch.float16, device=device)
            _ = torch.matmul(q, k_cache.transpose(-2, -1))
            
            if device == "cuda":
                torch.cuda.synchronize()
                mem = torch.cuda.max_memory_allocated()
            else:
                # Simulated allocation
                mem = (k_cache.nelement() + v_cache.nelement()) * 2
                
            stable_length = length
            print(f" -> Context: {length:6d} tokens | VRAM Allocated: {format_bytes(mem)}")
            
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "oom" in str(e).lower():
            oom_triggered = True
            print(f"\033[1;31m[OOM ❌] Standard cache crashed at context length: {stable_length + step_size} tokens!\033[0m")
        else:
            raise e
            
    return stable_length, oom_triggered

def run_argus_capacity_test(max_length=65536, step_size=4096):
    """Stress tests ARGUS Virtual Memory Cache to find its capacity ceiling."""
    print("\n\033[1;32m[STRESS] Running Capacity Test on ARGUS Virtual Memory Caching...\033[0m")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Configure tight cache limits to force aggressive hierarchical demotions and CPU spilling
    config = ArgusConfig(
        page_size=1024,
        max_active_pages=2,
        max_fp8_pages=2,
        max_int8_pages=2,
        max_int4_pages=2,
        sink_tokens=4,
        vram_oom_threshold_ratio=0.85
    )
    cache = PagedDynamicKVCache(config=config)
    
    stable_length = 0
    oom_triggered = False
    
    try:
        for length in range(step_size, max_length + 1, step_size):
            # Generate new block of tokens
            k = torch.randn(1, 4, step_size, 16, dtype=torch.float16, device=device)
            v = torch.randn(1, 4, step_size, 16, dtype=torch.float16, device=device)
            
            # Push tokens into manager
            cache.push_new_tokens(k, v)
            
            # Retrieve reconstructed states for a dummy attention check
            recon_k, _ = cache.get_all_keys_values()
            q = torch.randn(1, 4, 1, 16, dtype=torch.float16, device=device)
            _ = torch.matmul(q, recon_k.transpose(-2, -1))
            
            if device == "cuda":
                torch.cuda.synchronize()
                mem = torch.cuda.max_memory_allocated()
            else:
                # Simulated allocation
                mem = cache.get_vram_usage()
                
            stable_length = length
            
            # Show paging info
            active_count = len(cache.active_pages)
            archived_count = len(cache.jl_pages) + len(cache.one_bit_pages)
            cpu_spill_info = " (CPU Swapped)" if cache.is_swapped_out else ""
            
            print(f" -> Context: {length:6d} tokens | VRAM Allocated: {format_bytes(mem)} | Cache State: active={active_count} archived={archived_count}{cpu_spill_info}")
            
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "oom" in str(e).lower():
            oom_triggered = True
            print(f"\033[1;31m[OOM ❌] ARGUS cache crashed at context length: {stable_length + step_size} tokens!\033[0m")
        else:
            raise e
            
    return stable_length, oom_triggered

def run_comparative_benchmark(max_length=65536):
    """Executes both capacity tests and calculates the Usable Context Expansion Ratio."""
    print("=" * 80)
    print(f"        ARGUS CONTEXT CAPACITY BENCHMARK & OOM THRESHOLD ANALYSIS")
    print("=" * 80)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
    # 1. Run Standard
    std_stable, std_oom = run_standard_capacity_test(max_length=max_length)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
    # 2. Run ARGUS
    argus_stable, argus_oom = run_argus_capacity_test(max_length=max_length)
    
    # 3. Calculate Expansion Ratio
    expansion_ratio = argus_stable / max(1, std_stable)
    
    print("\n" + "=" * 80)
    print("                       FINAL STRESS BENCHMARK RESULTS")
    print("=" * 80)
    print(f" - Standard Caching Max Stable Context : {std_stable:6d} tokens" + (" (OOM triggered ❌)" if std_oom else " (Complete ✅)"))
    print(f" - ARGUS Caching Max Stable Context    : {argus_stable:6d} tokens" + (" (OOM triggered ❌)" if argus_oom else " (Complete ✅)"))
    print(f" - Usable Context Expansion Ratio      : \033[1;32m{expansion_ratio:.2f}x expansion\033[0m")
    print("=" * 80)

if __name__ == "__main__":
    # Run benchmark up to 64K context tokens
    run_comparative_benchmark(max_length=65536)
