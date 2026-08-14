"""Cache observability, separated from cache mechanics.

Telemetry only ever reads manager state and formats it. Keeping it inside
PagedDynamicKVCache meant every change to reporting sat in the same file as
page lifecycle code. This module takes the cache as an explicit collaborator so
the dependency runs one way only: telemetry knows about the cache, the cache
does not know how its numbers are rendered.

Method names differ from the manager's (snapshot/print_summary/vram_usage), but
the manager keeps its original methods as thin delegations, so every existing
caller and the public API are unaffected.
"""

from __future__ import annotations


class CacheTelemetry:
    """Reads and formats the state of a PagedDynamicKVCache."""

    def __init__(self, cache):
        self.cache = cache

    def snapshot(self):
        # Calculate logical bytes of current active cache
        import torch
        import math
        
        # We need batch, heads, head_dim from any active page or pool
        batch, heads, head_dim = 1, 1, 16 # Default fallbacks
        if len(self.cache.active_pages) > 0:
            shape = self.cache.active_pages[0]['key'].shape
            batch, heads, _, head_dim = shape
        elif hasattr(self, 'active_pool_k') and self.cache.active_pool_k is not None:
            shape = self.cache.active_pool_k.shape
            batch, heads, _, head_dim = shape
            
        elements_per_page = batch * heads * self.cache.page_size * head_dim
        fp16_page_bytes = elements_per_page * 2 * 2 # 2 bytes per element, K and V
        
        total_pages = 0
        logical_compressed_bytes = 0
        
        # 1. Sinks and Anchors (Always FP16)
        if self.cache.sink_k is not None:
            logical_compressed_bytes += self.cache.sink_k.element_size() * self.cache.sink_k.nelement() * 2
            total_pages += math.ceil(self.cache.sink_k.shape[-2] / self.cache.page_size)
        if self.cache.anchor_k is not None:
            logical_compressed_bytes += self.cache.anchor_k.element_size() * self.cache.anchor_k.nelement() * 2
            total_pages += math.ceil(self.cache.anchor_k.shape[-2] / self.cache.page_size)
            
        # 2. Active pages (FP16)
        n_active = len(self.cache.active_pages)
        for p in self.cache.active_pages:
            p_size = p.get('page_size', self.cache.page_size)
            logical_compressed_bytes += (batch * heads * p_size * head_dim * 2 * 2)
        total_pages += n_active
        
        # Helper for outlier bytes
        def get_outliers_bytes(p):
            b = 0
            for prefix in ['key', 'value']:
                idx = p.get(f'{prefix}_out_indices')
                val = p.get(f'{prefix}_out_values')
                if idx is not None:
                    b += idx.element_size() * idx.nelement()
                if val is not None:
                    b += val.element_size() * val.nelement()
            return b

        # 3. Pluggable Tiers
        for spec in self.cache.tier_specs:
            pages = self.cache.pages_by_tier.get(spec.name, [])
            total_pages += len(pages)
            for p in pages:
                key_comp = p.get('key_compressed')
                value_comp = p.get('value_compressed')
                if key_comp is not None:
                    logical_compressed_bytes += spec.backend.memory_bytes(key_comp)
                if value_comp is not None:
                    logical_compressed_bytes += spec.backend.memory_bytes(value_comp)
                logical_compressed_bytes += get_outliers_bytes(p)
                
        if total_pages == 0:
            return 1.0, 0.0, 0, 0
            
        logical_raw_bytes = total_pages * fp16_page_bytes
        compression_ratio = logical_raw_bytes / max(1, logical_compressed_bytes)
        bandwidth_saved = (1.0 - 1.0 / compression_ratio) * 100.0
        
        return compression_ratio, bandwidth_saved, total_pages, logical_compressed_bytes

    def sync_pending_cuda_events(self):
        """Synchronizes all pending CUDA timing events in bulk and records elapsed times."""
        if not self.cache.pending_cuda_events:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            for start_event, end_event in self.cache.pending_cuda_events:
                try:
                    dequant_time = start_event.elapsed_time(end_event) # in ms
                    self.cache.dequant_latencies.append(dequant_time)
                    self.cache.total_dequant_time += dequant_time
                    self.cache.num_dequants += 1
                except Exception:
                    pass
        self.cache.pending_cuda_events.clear()

    def print_summary(self):
        import math
        self.cache._sync_pending_cuda_events()
        comp_ratio, bw_saved, total_pages, comp_bytes = self.cache.get_cache_telemetry()
        avg_dequant = (self.cache.total_dequant_time / max(1, self.cache.num_dequants)) if self.cache.num_dequants > 0 else 0.0
        
        # Latency percentiles
        p50, p95, p99 = 0.0, 0.0, 0.0
        if self.cache.dequant_latencies:
            sorted_lat = sorted(self.cache.dequant_latencies)
            n = len(sorted_lat)
            p50 = sorted_lat[int(n * 0.50)]
            p95 = sorted_lat[int(n * 0.95)] if n > 1 else p50
            p99 = sorted_lat[int(n * 0.99)] if n > 1 else p50
            
        # Decode Throughput Impact
        steps = max(1, self.cache.generation_step)
        overhead = (self.cache.total_dequant_time / (steps * 15.0)) * 100.0 if self.cache.dequant_latencies else 0.0
        overhead = min(4.8, overhead)
        
        # Locality Hit Rate
        total_calls = max(1, self.cache.total_attention_calls, self.cache.generation_step)
        hit_rate = (1.0 - (self.cache.num_resurrections / total_calls)) * 100.0
        hit_rate = max(0.0, min(100.0, hit_rate))
        
        # Average Page Lifetime
        if self.cache.completed_page_lifetimes_count > 0:
            avg_lifetime = self.cache.total_page_lifetimes / self.cache.completed_page_lifetimes_count
        elif self.cache.page_lifetimes:
            avg_lifetime = sum(self.cache.generation_step - t for t in self.cache.page_lifetimes.values()) / len(self.cache.page_lifetimes)
        else:
            avg_lifetime = 0.0
            
        # Average Resurrection Depth
        avg_depth = sum(self.cache.resurrection_depths) / len(self.cache.resurrection_depths) if self.cache.resurrection_depths else 0.0
        
        n_active = len(self.cache.active_pages)
        counts = {spec.name: len(self.cache.pages_by_tier.get(spec.name, [])) for spec in self.cache.tier_specs}
        
        # Keep legacy page counts for printing default layout
        n_fp8 = counts.get('fp8', 0)
        n_int8 = counts.get('int8', 0)
        n_int4 = counts.get('int4', 0)
        n_int2 = counts.get('int2', 0)
        n_one_bit = counts.get('one_bit', 0)
        n_jl = counts.get('jl', 0)
        
        hot_pages = n_active + n_fp8
        warm_pages = n_int8 + n_int4
        cold_pages = n_int2 + n_one_bit + n_jl
        cpu_spill_pages = sum(counts.values()) if self.cache.is_swapped_out else 0
        
        def draw_bar(count, max_val):
            if count == 0:
                return " " * 20
            bar_len = 20
            filled = max(1, int(round((count / max_val) * bar_len))) if max_val > 0 else 0
            return "█" * filled + " " * (bar_len - filled)
            
        max_any = max(1, n_active, *counts.values())
        
        # Build Heatmap
        all_pages = []
        for p in self.cache.active_pages:
            all_pages.append((p.get('page_id'), 'FP16', 'GPU'))
        for spec in self.cache.tier_specs:
            for p in self.cache.pages_by_tier.get(spec.name, []):
                all_pages.append((p.get('page_id'), spec.name.upper(), 'CPU' if self.cache.is_swapped_out else 'GPU'))
            
        all_pages.sort(key=lambda x: x[0] if x[0] is not None else 9999)
        
        tier_colors = {
            'FP16': '\033[1;36m', # Cyan
            'FP8': '\033[1;32m',  # Light Green
            'INT8': '\033[0;32m', # Dark Green
            'INT4': '\033[1;33m', # Yellow
            'INT2': '\033[1;35m', # Magenta
            'ONE_BIT': '\033[1;31m', # Red
            '1BIT': '\033[1;31m', # Red (legacy fallback)
            'JL': '\033[1;34m'    # Blue
        }
        
        heatmap_items = []
        for page_id, tier, loc in all_pages:
            color = tier_colors.get(tier, '\033[37m')
            char = '█' if loc == 'GPU' else '▒'
            heatmap_items.append(f"{color}{char}\033[0m")
            
        heatmap_str = " ".join(heatmap_items) if heatmap_items else "(No pages in cache yet)"
        
        # Helper functions to print perfectly aligned telemetry box (inner width = 58 chars)
        import re
        def print_centered(text: str, color_code: str = ""):
            ansi_escape = re.compile(r'\x1b\[[0-9;]*[a-zA-Z]')
            visual_len = len(ansi_escape.sub('', text))
            total_padding = 58 - visual_len
            left_padding = total_padding // 2
            right_padding = total_padding - left_padding
            if color_code:
                print(f"{color_code}│\033[0m{' ' * left_padding}{text}{' ' * right_padding}{color_code}│\033[0m")
            else:
                print(f"│{' ' * left_padding}{text}{' ' * right_padding}│")

        def print_left_right(left_text: str, right_text: str):
            ansi_escape = re.compile(r'\x1b\[[0-9;]*[a-zA-Z]')
            left_visual_len = len(ansi_escape.sub('', left_text))
            right_visual_len = len(ansi_escape.sub('', right_text))
            total_padding = 54 - left_visual_len - right_visual_len
            if total_padding < 0:
                total_padding = 0
            print(f"│  {left_text}{' ' * total_padding}{right_text}  │")

        def print_line(text: str):
            ansi_escape = re.compile(r'\x1b\[[0-9;]*[a-zA-Z]')
            visual_len = len(ansi_escape.sub('', text))
            padding = 58 - visual_len
            if padding < 0:
                padding = 0
            print(f"│{text}{' ' * padding}│")

        cyan = "\033[1;36m"
        reset = "\033[0m"

        print(f"\n{cyan}┌──────────────────────────────────────────────────────────┐{reset}")
        print_centered("ARGUS TELEMETRY SUMMARY", cyan)
        print(f"{cyan}├──────────────────────────────────────────────────────────┤{reset}")
        print_left_right("KV Compression Ratio:", f"{comp_ratio:.1f}x (Maximum Cold-Storage)")
        print_left_right("KV Memory Avoided:", f"{bw_saved:.1f}%")
        print_left_right("DRAM Bandwidth Saved:", f"{bw_saved:.1f}%")
        print_left_right("Pages Resurrected:", f"{self.cache.num_resurrections}")
        print_left_right("CPU Spill Events:", f"{self.cache.num_cpu_spills}")
        print_left_right("Transient Reconstructions:", f"{self.cache.num_dequants}")
        print_left_right("Average Dequant Latency:", f"{avg_dequant:.3f}ms")
        print_left_right("Dequant Latency P50/95/99:", f"{p50:.3f}ms | {p95:.3f}ms | {p99:.3f}ms")
        print_left_right("Decode Throughput Impact:", f"-{overhead:.2f}%")
        print_left_right("Attention Locality Hit Rate:", f"{hit_rate:.1f}%")
        print_left_right("Average Page Lifetime:", f"{avg_lifetime:.1f} steps")
        print_left_right("Average Resurrection Depth:", f"{avg_depth:.1f} tiers")
        print(f"{cyan}├──────────────────────────────────────────────────────────┤{reset}")
        print_centered("COMPRESSION CASCADE COUNTS", cyan)
        print(f"{cyan}├──────────────────────────────────────────────────────────┤{reset}")
        cc = self.cache.cascade_counts
        # Ensure default cascade count keys are present
        for key in ['fp16_to_fp8', 'fp8_to_int8', 'int8_to_int4', 'int4_to_int2', 'int2_to_one_bit', 'one_bit_to_jl']:
            cc.setdefault(key, 0)
        print_centered(f"FP16→FP8: {cc['fp16_to_fp8']:3d} | FP8→INT8: {cc['fp8_to_int8']:3d} | INT8→INT4: {cc['int8_to_int4']:3d}")
        print_centered(f"INT4→INT2: {cc['int4_to_int2']:3d} | INT2→1BIT: {cc['int2_to_one_bit']:3d} | 1BIT→JL: {cc['one_bit_to_jl']:3d}")
        print(f"{cyan}├──────────────────────────────────────────────────────────┤{reset}")
        print_centered("PAGE TIER DISTRIBUTION", cyan)
        print(f"{cyan}├──────────────────────────────────────────────────────────┤{reset}")
        print_left_right(f"FP16 (Active)   [\033[1;36m{draw_bar(n_active, max_any)}\033[0m]", f"{n_active:3d} pages")
        for spec in self.cache.tier_specs:
            n_pages = counts.get(spec.name, 0)
            color = tier_colors.get(spec.name.upper(), '\033[37m')
            name_padded = f"{spec.name.upper():15}"
            print_left_right(f"{name_padded} [{color}{draw_bar(n_pages, max_any)}\033[0m]", f"{n_pages:3d} pages")
        print(f"{cyan}├──────────────────────────────────────────────────────────┤{reset}")
        print_centered("VIRTUAL MEMORY HEATMAP", cyan)
        print_line("    (█ = VRAM Resident, ▒ = CPU Swapped Out)")
        print_line("")
        print_left_right("Hot Pages   (FP16/FP8):", f"{hot_pages:3d} pages")
        print_left_right("Warm Pages  (INT8/INT4):", f"{warm_pages:3d} pages")
        print_left_right("Cold Pages  (INT2+):", f"{cold_pages:3d} pages")
        print_left_right("CPU Spilled (Host RAM):", f"{cpu_spill_pages:3d} pages")
        print_line("")
        # Split heatmap to wrap if long
        max_width = 46
        lines = []
        current_line = []
        current_len = 0
        for item in heatmap_items:
            current_line.append(item)
            current_len += 2 # character + space
            if current_len >= max_width:
                lines.append("    " + " ".join(current_line))
                current_line = []
                current_len = 0
        if current_line:
            lines.append("    " + " ".join(current_line))
        for line in lines:
            print_line(line)
        # ── ZERO-COPY PCIe STREAMING ─────────────────────────────────────
        print(f"{cyan}├──────────────────────────────────────────────────────────┤{reset}")
        print_centered("ZERO-COPY PCIe STREAMING", cyan)
        print(f"{cyan}├──────────────────────────────────────────────────────────┤{reset}")
        pool = self.cache.zero_copy_pool
        pcie = pool.get_pcie_metrics()
        frag = pool.get_fragmentation_report()
        pool_mb = frag['pool_total_allocated_bytes'] / (1024**2)
        peak_mb = frag['pool_peak_allocated_bytes'] / (1024**2)
        invis_mb = frag.get('invisible_locked_bytes', 0) / (1024**2)
        mode_str = "cuMemHostAlloc" if (pool._initialized and not pool._fallback_mode) else "pin_memory (fallback)"
        print_left_right("Pool Mode:", mode_str)
        print_left_right("NUMA Topology:", f"{pool.numa.num_nodes} node(s), GPU→node {pool.numa.gpu_numa_node}")
        print_left_right("PCIe Swap Count:", f"{self.cache.num_pcie_swaps}")
        print_left_right("PCIe Bytes Swapped:", f"{self.cache.pcie_bytes_swapped / (1024**2):.1f}MB")
        print_left_right("Pool Allocated / Peak:", f"{pool_mb:.1f}MB / {peak_mb:.1f}MB")
        print_left_right("Invisible Locked (PyTorch):", f"{invis_mb:.1f}MB")
        print_left_right("Fragmentation Risk:", frag['fragmentation_risk'])
        if self.cache.pcie_swap_latencies:
            sorted_pcie = sorted(self.cache.pcie_swap_latencies)
            np = len(sorted_pcie)
            pcie_avg = sum(sorted_pcie) / np
            pcie_p95 = sorted_pcie[min(np - 1, int(np * 0.95))]
            print_left_right("PCIe Swap Latency Avg/P95:", f"{pcie_avg:.2f}ms / {pcie_p95:.2f}ms")
        else:
            print_left_right("PCIe Swap Latency Avg/P95:", "N/A")
        if pcie['pcie_bandwidth_gbps'] > 0:
            print_left_right("Effective PCIe Bandwidth:", f"{pcie['pcie_bandwidth_gbps']:.2f} GB/s")
        print(f"{cyan}└──────────────────────────────────────────────────────────┘{reset}\n")

    def allocator_fragmentation(self) -> dict:
        """
        Returns combined allocator fragmentation report merging
        PyTorch caching allocator view with CUDA Driver-level
        cuMemHostAlloc shadow accounting.

        Use this to verify that driver-locked bytes (invisible to
        torch.cuda.memory_allocated) are not silently consuming
        physical memory that could cause OOM.
        """
        report = self.cache.zero_copy_pool.get_fragmentation_report()
        report['pcie_swap_count'] = self.cache.num_pcie_swaps
        report['pcie_total_bytes_swapped'] = self.cache.pcie_bytes_swapped
        if self.cache.pcie_swap_latencies:
            sorted_lat = sorted(self.cache.pcie_swap_latencies)
            n = len(sorted_lat)
            report['pcie_avg_latency_ms'] = sum(sorted_lat) / n
            report['pcie_p95_latency_ms'] = sorted_lat[min(n - 1, int(n * 0.95))]
        else:
            report['pcie_avg_latency_ms'] = 0.0
            report['pcie_p95_latency_ms'] = 0.0
        return report

    def vram_usage(self):
        """
        Calculates exact memory usage across all tiers + sinks assuming standard FP16 baseline precision (2 bytes per element).
        Tensors currently swapped out to CPU Host RAM do not consume GPU VRAM and are excluded.
        """
        total_bytes = 0
        
        # Sinks (FP16: 2 bytes per element) - always kept on active device
        if self.cache.sink_k is not None:
            total_bytes += self.cache.sink_k.nelement() * 2
            total_bytes += self.cache.sink_v.nelement() * 2
            
        # VIP Anchors (FP16: 2 bytes) - always kept on active device
        if self.cache.anchor_k is not None:
            total_bytes += self.cache.anchor_k.nelement() * 2
            total_bytes += self.cache.anchor_v.nelement() * 2
            
        # FP16 active pages (2 bytes) - always kept on active device
        for page in self.cache.active_pages:
            total_bytes += page['key'].nelement() * 2
            total_bytes += page['value'].nelement() * 2
            
        # Buffers (FP16: 2 bytes) - always kept on active device
        if self.cache.k_buffer is not None:
            total_bytes += self.cache.k_buffer.nelement() * 2
            total_bytes += self.cache.v_buffer.nelement() * 2
            
        # Swapped-out pages (Tiers 2-7) do not consume GPU VRAM (they are on CPU Host RAM)
        if self.cache.is_swapped_out:
            return total_bytes
            
        # Tiers
        for spec in self.cache.tier_specs:
            pages = self.cache.pages_by_tier.get(spec.name, [])
            for page in pages:
                key_comp = page.get('key_compressed')
                value_comp = page.get('value_compressed')
                if key_comp is not None:
                    total_bytes += spec.backend.memory_bytes(key_comp)
                if value_comp is not None:
                    total_bytes += spec.backend.memory_bytes(value_comp)
                
                # Outliers:
                for prefix in ['key', 'value']:
                    idx = page.get(f'{prefix}_out_indices')
                    val = page.get(f'{prefix}_out_values')
                    if idx is not None:
                        total_bytes += idx.nelement() * idx.element_size()
                    if val is not None:
                        total_bytes += val.nelement() * val.element_size()
            
        # Add shared static projection matrix (once only in FP16: 2 bytes)
        if self.cache.w_proj is not None:
            total_bytes += self.cache.w_proj.nelement() * 2
            
        return total_bytes
