#!/usr/bin/env python3
"""
ARGUS Memory Pressure Time-Series Evaluation Benchmark
Simulates auto-regressive decoding to track and plot VRAM allocation, 
compression tier page distribution, and balloon driver actions over generation steps.
"""

import sys
import os
import argparse
import json
import torch

# Ensure workspace is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.memory_manager import PagedDynamicKVCache, ArgusConfig
from core.balloon_driver import ElasticCacheBalloonDriver

def run_simulation(steps, device):
    print("=" * 90)
    print(f"  ARGUS MEMORY PRESSURE TIME-SERIES SIMULATION ({steps} Steps on {device})")
    print("=" * 90)
    
    driver = ElasticCacheBalloonDriver()
    
    # Configure cache with tight limits to trigger active compression cascades
    config = ArgusConfig(
        page_size=128,
        max_active_pages=4,
        max_fp8_pages=3,
        max_int8_pages=3,
        max_int4_pages=3,
        max_int2_pages=3,
        max_one_bit_pages=3,
        sink_tokens=4,
        vram_oom_threshold_ratio=0.85, # 85% limit
        balloon_driver=driver
    )
    
    cache = PagedDynamicKVCache(config=config)
    
    history = {
        "step": [],
        "active_pages": [],
        "fp8": [],
        "int8": [],
        "int4": [],
        "int2": [],
        "one_bit": [],
        "jl": [],
        "cpu_spilled": [],
        "vram_ratio": [],
        "balloon_inflated": []
    }
    
    # Simulate step-by-step decoding
    for step in range(steps):
        # Push 128 new tokens at each step
        k = torch.randn(1, 1, 128, 16, dtype=torch.float16, device=device)
        v = torch.randn(1, 1, 128, 16, dtype=torch.float16, device=device)
        cache.push_new_tokens(k, v)
        
        # Simulate active page access and scoring
        for page in cache.active_pages:
            page["importance_score"] = float(torch.rand(1).item() * 3.0)
            
        # Collect page tier counts
        history["step"].append(step)
        history["active_pages"].append(len(cache.active_pages))
        history["fp8"].append(len(cache.pages_by_tier.get("fp8", [])))
        history["int8"].append(len(cache.pages_by_tier.get("int8", [])))
        history["int4"].append(len(cache.pages_by_tier.get("int4", [])))
        history["int2"].append(len(cache.pages_by_tier.get("int2", [])))
        history["one_bit"].append(len(cache.pages_by_tier.get("one_bit", [])))
        history["jl"].append(len(cache.pages_by_tier.get("jl", [])))
        
        # Calculate CPU Swapped pages
        spilled_count = sum(1 for p in cache.active_pages if p.get('key').device.type == 'cpu')
        for spec in cache.tier_specs:
            pages = cache.pages_by_tier.get(spec.name, [])
            for p in pages:
                if 'key_compressed' in p:
                    for k_part in p['key_compressed'].values():
                        if k_part.device.type == 'cpu':
                            spilled_count += 1
                            break
                            
        history["cpu_spilled"].append(spilled_count)
        
        # Estimate ratio
        if torch.cuda.is_available():
            device_idx = torch.cuda.current_device()
            allocated = torch.cuda.memory_allocated(device_idx)
            total = torch.cuda.get_device_properties(device_idx).total_memory
            history["vram_ratio"].append(float(allocated) / float(total))
        else:
            # CPU simulation mock ratio based on page load
            history["vram_ratio"].append(min(1.0, 0.1 + 0.02 * step))
            
        # Track balloon inflation status
        history["balloon_inflated"].append(driver.inflated_active_pages.get(cache, 0))
        
        # Print status summary
        if step % 10 == 0:
            print(f"Step {step:2d} | Active: {len(cache.active_pages)} | FP8: {len(cache.pages_by_tier.get('fp8', []))} | INT4: {len(cache.pages_by_tier.get('int4', []))} | 1-Bit: {len(cache.pages_by_tier.get('one_bit', []))} | Spilled: {spilled_count}")
            
    print("=" * 90)
    return history

def plot_memory_pressure_timeseries(history, output_path):
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        steps = history["step"]
        
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, dpi=150)
        
        # Stacked area chart of the page tier distribution
        labels = ["Active (FP16)", "FP8", "INT8", "INT4", "INT2", "1-Bit", "JL (Archive)"]
        colors = ["#4a90e2", "#7cb5ec", "#90ed7d", "#f7a35c", "#8085e9", "#f15c80", "#e4d354"]
        
        y = np.vstack([
            history["active_pages"],
            history["fp8"],
            history["int8"],
            history["int4"],
            history["int2"],
            history["one_bit"],
            history["jl"]
        ])
        
        ax1.stackplot(steps, y, labels=labels, colors=colors, alpha=0.85)
        ax1.set_ylabel("Cached Pages Count", fontweight="bold")
        ax1.set_title("ARGUS Memory Compression Tier Page Distribution Over Decoding Steps", fontsize=12, fontweight="bold", pad=10)
        ax1.legend(loc="upper left", frameon=True)
        
        # Second axis: VRAM ratio + Balloon Driver action
        ax2.plot(steps, [r * 100 for r in history["vram_ratio"]], color="#e06666", linewidth=2, label="VRAM Utilization (%)")
        ax2.set_ylabel("VRAM Utilization (%)", color="#e06666", fontweight="bold")
        ax2.tick_params(axis="y", labelcolor="#e06666")
        
        ax2_twin = ax2.twinx()
        ax2_twin.plot(steps, history["balloon_inflated"], color="#674ea7", linewidth=2.5, linestyle="-.", label="Balloon Inflation Level")
        ax2_twin.set_ylabel("Balloon Inflation Level", color="#674ea7", fontweight="bold")
        ax2_twin.tick_params(axis="y", labelcolor="#674ea7")
        
        # Custom ticks
        ax2_twin.set_yticks(np.arange(0, max(history["balloon_inflated"]) + 2, 1))
        
        ax2.set_xlabel("Decoding Step", fontweight="bold")
        ax2.set_title("VRAM Utilization & Ballooning Action", fontsize=11, fontweight="bold", pad=10)
        
        fig.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"Time-series memory plot saved successfully to: {output_path}")
    except Exception as e:
        print(f"Could not generate plot due to: {e}")

def main():
    parser = argparse.ArgumentParser(description="ARGUS Memory Pressure Time-Series Simulation")
    parser.add_argument("--steps", type=int, default=50, help="Number of decoding steps to simulate")
    parser.add_argument("--export", type=str, default="benchmarks/memory_pressure_results.json", help="Path to export JSON results")
    parser.add_argument("--plot", type=str, default="benchmarks/memory_pressure_timeseries.png", help="Path to export PNG plot")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run on")
    
    args = parser.parse_args()
    
    history = run_simulation(steps=args.steps, device=args.device)
    
    if args.export:
        os.makedirs(os.path.dirname(os.path.abspath(args.export)), exist_ok=True)
        with open(args.export, "w", encoding="utf-8") as f:
            json.dump({
                "benchmark": "Memory Pressure Time-Series",
                "history": history
            }, f, indent=4)
        print(f"Results successfully exported to: {args.export}\n")
        
    if args.plot:
        plot_memory_pressure_timeseries(history, args.plot)

if __name__ == "__main__":
    main()
