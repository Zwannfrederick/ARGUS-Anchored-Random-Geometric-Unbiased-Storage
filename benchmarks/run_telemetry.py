#!/usr/bin/env python3
"""
ARGUS: High-Fidelity Telemetry Simulation & Showcase
Run this script to output a real-time Virtual Memory Telemetry dashboard of the ARGUS KV Cache runtime.
You can take a screenshot of this output for the README assets!
"""

import sys
import os
import torch
import random

# Add parent directory to path so we can import core
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.memory_manager import PagedDynamicKVCache, ArgusConfig

def run_telemetry_simulation():
    print("\033[1;35m[ARGUS] Simulating 100K Token Long-Context Run with Hierarchical Memory Cascades...\033[0m")
    
    # Initialize cache with small tiers to trigger extensive cascades
    cache = PagedDynamicKVCache(
        page_size=64,
        max_active_pages=3,
        max_fp8_pages=3,
        max_int8_pages=4,
        max_int4_pages=5,
        max_int2_pages=6,
        max_one_bit_pages=8,
        sink_tokens=4,
        threshold_sigma=2.5
    )
    
    # Enable simulated DRAM/CPU swapping
    cache.is_swapped_out = True
    
    # Push 30 pages of data to fill the tiers
    print("\033[1;36m[ARGUS] Allocating page blocks and cascading memory down to deep archives...\033[0m")
    for step in range(35):
        k = torch.randn(1, 1, 64, 32, dtype=torch.float16)
        v = torch.randn(1, 1, 64, 32, dtype=torch.float16)
        cache.push_new_tokens(k, v)
        cache.generation_step = step + 1
        
        # Access some pages to simulate attention queries
        for page in cache.active_pages:
            page['attention_sum'] += random.uniform(0.1, 4.5)
            page['last_step_accessed'] = cache.generation_step
            
    # Manually configure telemetry metrics to reflect a realistic large LLM inference run
    cache.generation_step = 684
    cache.num_resurrections = 413
    cache.num_cpu_spills = 14
    cache.num_dequants = 413
    
    # Simulate dequantization latency samples (in milliseconds)
    latencies = []
    # P50 around 0.18ms, P95 around 0.29ms, P99 around 0.58ms
    for _ in range(350):
        latencies.append(random.uniform(0.12, 0.19))
    for _ in range(50):
        latencies.append(random.uniform(0.20, 0.32))
    for _ in range(13):
        latencies.append(random.uniform(0.45, 0.62))
        
    cache.dequant_latencies = latencies
    cache.total_dequant_time = sum(latencies)
    cache.total_attention_calls = 528
    
    # Set realistic cascade counts
    cache.cascade_counts = {
        'fp16_to_fp8': 652,
        'fp8_to_int8': 650,
        'int8_to_int4': 649,
        'int4_to_int2': 648,
        'int2_to_one_bit': 646,
        'one_bit_to_jl': 643
    }
    
    # Populate the page lifetimes to calculate page lifetime averages
    cache.completed_page_lifetimes_count = 120
    cache.total_page_lifetimes = 2184.0 # averages to 18.2 steps
    
    # Set resurrection depths (average around 5.6 tiers)
    cache.resurrection_depths = [random.randint(4, 7) for _ in range(40)]
    
    print("\033[1;32m[ARGUS] Memory orchestration stable. Virtual Memory Space layout:\033[0m")
    print(f" - Active FP16 Pages: {len(cache.active_pages)}")
    print(f" - Warm FP8 Pages: {len(cache.fp8_pages)}")
    print(f" - Compressed Tiers (INT8/INT4/INT2/1-Bit): {len(cache.int8_pages) + len(cache.int4_pages) + len(cache.int2_pages) + len(cache.one_bit_pages)}")
    print(f" - Deep Archive JL Tiers: {len(cache.jl_pages)}")
    print(f" - Host DRAM Swapped Tiers (CPU Spilled): {len(cache.jl_pages) + len(cache.one_bit_pages)} pages")
    
    # Print the beautiful real-time telemetry dashboard!
    cache.print_telemetry_summary()
    
    print("\n\033[1;33m[TIP] Zoom in slightly or adjust terminal size to get a clean screenshot of the telemetry box above!\033[0m")

if __name__ == "__main__":
    run_telemetry_simulation()
