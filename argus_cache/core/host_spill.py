"""Spilling the whole cache to host memory and restoring it.

This is the bottom of the tier cascade and the only *lossless* step in it:
every other demotion trades precision for space, while spill trades latency
for space by moving bytes over PCIe into pinned host memory. It exists so
that a process under VRAM pressure can survive rather than degrade.

Because it runs precisely when memory is scarce -- when retry logic and
out-of-memory handlers are firing -- both directions are idempotent. Spilling
an already-spilled cache and restoring one that was never spilled are both
no-ops, not errors.
"""

from __future__ import annotations

import torch

from .logger import argus_log


class HostSpillManager:
    """Moves a :class:`PagedDynamicKVCache`'s pages between device and host."""

    def __init__(self, cache) -> None:
        self.cache = cache

    def spill_out(self):
        """
        Moves all compressed page tensors of Tiers 2-7 to CPU host memory
        using Zero-Copy PCIe pinned/device-mapped memory (cuMemHostAlloc).

        If the CUDA Driver API is unavailable, falls back to plain
        ``tensor.cpu()`` (legacy behaviour).

        After this call, GPU Triton kernels can still read the swapped
        pages directly via PCIe device pointers — no cudaMemcpy needed.
        """
        if self.cache.is_swapped_out:
            return

        import time
        use_cuda_event = torch.cuda.is_available()

        def swap_tensor_to_pinned(item, page_id):
            """Recursively move tensors to zero-copy pinned host memory."""
            if isinstance(item, torch.Tensor):
                pinned = self.cache.zero_copy_pool.tensor_to_pinned(item)
                # Track for later cleanup
                if page_id is not None:
                    self.cache._zero_copy_tensor_registry.setdefault(page_id, []).append(pinned)
                return pinned
            elif isinstance(item, dict):
                for k, v in list(item.items()):
                    item[k] = swap_tensor_to_pinned(v, page_id)
            elif isinstance(item, list):
                for i in range(len(item)):
                    item[i] = swap_tensor_to_pinned(item[i], page_id)
            return item

        if use_cuda_event:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        else:
            t0 = time.perf_counter()

        total_bytes = 0
        for spec in self.cache.tier_specs:
            for page in self.cache.pages_by_tier.get(spec.name, []):
                page_id = page.get('page_id')
                # Calculate bytes before swap
                for tensor_key in ['key_compressed', 'value_compressed',
                                   'key_out_indices', 'key_out_values',
                                   'value_out_indices', 'value_out_values']:
                    t = page.get(tensor_key)
                    if isinstance(t, torch.Tensor):
                        total_bytes += t.nelement() * t.element_size()
                    elif isinstance(t, dict):
                        for v in t.values():
                            if isinstance(v, torch.Tensor):
                                total_bytes += v.nelement() * v.element_size()
                swap_tensor_to_pinned(page, page_id)

        if use_cuda_event:
            end_event.record()
            torch.cuda.synchronize()
            latency_ms = start_event.elapsed_time(end_event)
        else:
            latency_ms = (time.perf_counter() - t0) * 1000.0

        self.cache.pcie_swap_latencies.append(latency_ms)
        self.cache.num_pcie_swaps += 1
        self.cache.pcie_bytes_swapped += total_bytes
        self.cache.zero_copy_pool.record_pcie_transfer(latency_ms, total_bytes)

        frag = self.cache.zero_copy_pool.get_fragmentation_report()
        argus_log("INFO",
                  f"Zero-Copy PCIe swap-out complete | "
                  f"{total_bytes / (1024**2):.1f}MB → pinned host | "
                  f"latency: {latency_ms:.2f}ms | "
                  f"pool: {frag['pool_total_allocated_bytes'] / (1024**2):.1f}MB | "
                  f"frag_risk: {frag['fragmentation_risk']}",
                  line_no=2122)

        self.cache.is_swapped_out = True
        self.cache._invalidate_decompressed_cache()

    def spill_in(self, device="cuda"):
        """
        Swaps all host-resident page tensors back to GPU active VRAM.

        For zero-copy tensors, this performs a PCIe DMA read (measured
        separately from dequant latency).  For fallback tensors, uses
        standard ``tensor.to(device)``.

        After swap-in, all pinned host allocations for the moved pages
        are freed from the ZeroCopyHostPool.
        """
        if not self.cache.is_swapped_out:
            return

        import time
        target_device = torch.device(device if torch.cuda.is_available() else "cpu")
        use_cuda_event = torch.cuda.is_available() and target_device.type == "cuda"

        if use_cuda_event:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        else:
            t0 = time.perf_counter()

        total_bytes = 0
        zc_tensors_to_free = []

        def swap_tensor_to_device(item, target_device, page_id):
            nonlocal total_bytes
            if isinstance(item, torch.Tensor):
                total_bytes += item.nelement() * item.element_size()
                # If this is a zero-copy tensor, read via PCIe then free the pinned buffer
                is_zc = self.cache.zero_copy_pool.is_zero_copy_tensor(item)
                result = item.to(target_device)
                if is_zc:
                    zc_tensors_to_free.append(item)
                return result
            elif isinstance(item, dict):
                for k, v in list(item.items()):
                    item[k] = swap_tensor_to_device(v, target_device, page_id)
            elif isinstance(item, list):
                for i in range(len(item)):
                    item[i] = swap_tensor_to_device(item[i], target_device, page_id)
            return item

        for spec in self.cache.tier_specs:
            for page in self.cache.pages_by_tier.get(spec.name, []):
                page_id = page.get('page_id')
                swap_tensor_to_device(page, target_device, page_id)

        # Clear the registry
        self.cache._zero_copy_tensor_registry.clear()

        if use_cuda_event:
            end_event.record()
            torch.cuda.synchronize()
            latency_ms = start_event.elapsed_time(end_event)
        else:
            latency_ms = (time.perf_counter() - t0) * 1000.0

        # Now that we synchronized and the GPU has finished reading, we can safely free the host tensors
        for t in zc_tensors_to_free:
            self.cache.zero_copy_pool.free_tensor(t)

        self.cache.pcie_swap_latencies.append(latency_ms)
        self.cache.num_pcie_swaps += 1
        self.cache.pcie_bytes_swapped += total_bytes
        self.cache.zero_copy_pool.record_pcie_transfer(latency_ms, total_bytes)

        argus_log("INFO",
                  f"Zero-Copy PCIe swap-in complete | "
                  f"{total_bytes / (1024**2):.1f}MB ← device | "
                  f"latency: {latency_ms:.2f}ms",
                  line_no=2147)

        self.cache.is_swapped_out = False
        self.cache._invalidate_decompressed_cache()
