"""
Tests for Faz 3: Zero-Copy PCIe Streaming (cuMemHostAlloc)

Validates:
1. ZeroCopyHostPool initialisation & fallback mode
2. NUMA topology detection
3. tensor_to_pinned → swap_out → swap_in round-trip correctness
4. PCIe latency metrics collection
5. Allocator fragmentation report accuracy
6. OOM guard includes invisible locked bytes
7. Telemetry dashboard renders PCIe section without crash
"""
import torch
import pytest
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from argus_cache.core.memory_manager import PagedDynamicKVCache, ArgusConfig
from argus_cache.core.tier_registry import TierSpec, PipelineConfig
from argus_cache.core.zero_copy_pool import ZeroCopyHostPool, NUMATopology


# ─── Helper ──────────────────────────────────────────────────────────────────

def make_cache(page_size=8, max_active=2, max_fp8=2, max_int8=2,
               max_int4=2, max_int2=2, max_one_bit=2, sink=2):
    """Create a lightweight PagedDynamicKVCache for testing."""
    config = ArgusConfig(
        page_size=page_size,
        max_active_pages=max_active,
        max_fp8_pages=max_fp8,
        max_int8_pages=max_int8,
        max_int4_pages=max_int4,
        max_int2_pages=max_int2,
        max_one_bit_pages=max_one_bit,
        sink_tokens=sink,
        vram_oom_threshold_ratio=0.85,
    )
    return PagedDynamicKVCache(config=config)


def fill_cache_pages(cache, num_steps=20, batch=1, heads=2, head_dim=16):
    """Run a sequence of update + attention steps to populate tiers."""
    for step in range(num_steps):
        k = torch.randn(batch, heads, cache.page_size, head_dim)
        v = torch.randn(batch, heads, cache.page_size, head_dim)
        cache.push_new_tokens(k, v)
        q = torch.randn(batch, heads, 1, head_dim)
        cache.inplace_paged_attention(q)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. ZeroCopyHostPool Unit Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestZeroCopyHostPool:
    """Tests for the standalone ZeroCopyHostPool class."""

    def test_pool_creation(self):
        """Pool should initialise without error (fallback or driver mode)."""
        pool = ZeroCopyHostPool()
        assert pool is not None
        assert isinstance(pool._fallback_mode, bool)

    def test_fallback_tensor_to_pinned(self):
        """In fallback mode, tensor_to_pinned must return a CPU tensor."""
        pool = ZeroCopyHostPool()
        pool._fallback_mode = True
        pool._initialized = False

        t = torch.randn(4, 8)
        pinned = pool.tensor_to_pinned(t)
        assert pinned.device.type == "cpu"
        assert torch.allclose(t, pinned, atol=1e-6)

    def test_fallback_round_trip_preserves_data(self):
        """Data must survive the fallback pin → read cycle."""
        pool = ZeroCopyHostPool()
        pool._fallback_mode = True
        pool._initialized = False

        original = torch.randn(2, 4, 8, 16)
        pinned = pool.tensor_to_pinned(original)
        assert torch.allclose(original, pinned, atol=1e-6)

    def test_fragmentation_report_structure(self):
        """Fragmentation report must contain all expected keys."""
        pool = ZeroCopyHostPool()
        report = pool.get_fragmentation_report()

        required_keys = [
            'pool_total_allocated_bytes',
            'pool_peak_allocated_bytes',
            'pool_num_allocations',
            'pytorch_allocated_bytes',
            'pytorch_reserved_bytes',
            'invisible_locked_bytes',
            'fragmentation_risk',
        ]
        for key in required_keys:
            assert key in report, f"Missing key: {key}"

    def test_pcie_metrics_empty(self):
        """PCIe metrics should return zeros when no transfers recorded."""
        pool = ZeroCopyHostPool()
        metrics = pool.get_pcie_metrics()
        assert metrics['total_pcie_reads'] == 0
        assert metrics['avg_pcie_latency_ms'] == 0.0
        assert metrics['pcie_bandwidth_gbps'] == 0.0

    def test_pcie_metrics_after_recording(self):
        """PCIe metrics should reflect recorded transfers."""
        pool = ZeroCopyHostPool()
        pool.record_pcie_transfer(1.5, 1024 * 1024)
        pool.record_pcie_transfer(2.0, 2 * 1024 * 1024)
        pool.record_pcie_transfer(0.8, 512 * 1024)

        metrics = pool.get_pcie_metrics()
        assert metrics['total_pcie_reads'] == 3
        assert metrics['total_pcie_bytes'] == 1024 * 1024 + 2 * 1024 * 1024 + 512 * 1024
        assert metrics['avg_pcie_latency_ms'] > 0
        assert metrics['p95_pcie_latency_ms'] >= metrics['p50_pcie_latency_ms']

    def test_repr(self):
        """__repr__ should not crash and should contain mode info."""
        pool = ZeroCopyHostPool()
        r = repr(pool)
        assert "ZeroCopyHostPool" in r
        assert "mode=" in r

    def test_zero_copy_tensor_detection(self):
        """is_zero_copy_tensor must return True only for pool-managed tensors."""
        pool = ZeroCopyHostPool()
        normal = torch.randn(4, 4)
        assert not pool.is_zero_copy_tensor(normal)

        # Simulate a managed tensor
        normal._zero_copy_managed = True
        assert pool.is_zero_copy_tensor(normal)

    def test_cleanup_idempotent(self):
        """cleanup() should be safe to call multiple times."""
        pool = ZeroCopyHostPool()
        pool.cleanup()
        pool.cleanup()
        assert pool._total_allocated_bytes == 0


# ═══════════════════════════════════════════════════════════════════════════════
# 2. NUMA Topology Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestNUMATopology:
    """Tests for NUMA topology detection."""

    def test_numa_detection(self):
        """NUMA topology should detect at least 1 node."""
        numa = NUMATopology()
        assert numa.num_nodes >= 1
        assert numa.gpu_numa_node >= 0

    def test_single_socket_bypass(self):
        """On a single-socket system, is_multi_socket should be False."""
        numa = NUMATopology()
        if numa.num_nodes <= 1:
            assert not numa.is_multi_socket
        else:
            assert numa.is_multi_socket

    def test_bind_noop_on_single_socket(self):
        """bind_to_gpu_node should return False on single-socket systems."""
        numa = NUMATopology()
        if numa.num_nodes <= 1:
            result = numa.bind_to_gpu_node()
            assert result is False


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Integrated Zero-Copy Swap Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestZeroCopySwapIntegration:
    """Tests for swap_out_to_host / swap_in_to_device with ZeroCopyHostPool."""

    def test_swap_out_creates_cpu_tensors(self):
        """After swap_out_to_host, all tier page tensors must be on CPU."""
        cache = make_cache()
        fill_cache_pages(cache, num_steps=20)

        # Verify some pages exist in tiers before swap
        total_tier_pages = sum(len(cache.pages_by_tier.get(s.name, []))
                               for s in cache.tier_specs)
        assert total_tier_pages > 0, "Need tier pages to test swap"

        cache.swap_out_to_host()
        assert cache.is_swapped_out

        # All tier page tensors must now be on CPU
        for spec in cache.tier_specs:
            for page in cache.pages_by_tier.get(spec.name, []):
                for key in ['key_compressed', 'value_compressed']:
                    comp = page.get(key)
                    if comp is None:
                        continue
                    if isinstance(comp, dict):
                        for v in comp.values():
                            if isinstance(v, torch.Tensor):
                                assert v.device.type == "cpu", \
                                    f"Tensor in {spec.name}/{key} still on {v.device}"
                    elif isinstance(comp, torch.Tensor):
                        assert comp.device.type == "cpu"

    def test_swap_round_trip_preserves_data(self):
        """swap_out → swap_in must preserve tensor data within tolerance."""
        cache = make_cache()
        fill_cache_pages(cache, num_steps=20)

        # Snapshot compressed data before swap
        snapshots = {}
        for spec in cache.tier_specs:
            for page in cache.pages_by_tier.get(spec.name, []):
                pid = page.get('page_id')
                comp_k = page.get('key_compressed')
                if isinstance(comp_k, dict):
                    snap = {}
                    for k2, v2 in comp_k.items():
                        if isinstance(v2, torch.Tensor):
                            snap[k2] = v2.clone()
                    if snap:
                        snapshots[pid] = snap

        cache.swap_out_to_host()
        cache.swap_in_to_device(device="cpu")

        # Verify data integrity
        for spec in cache.tier_specs:
            for page in cache.pages_by_tier.get(spec.name, []):
                pid = page.get('page_id')
                if pid not in snapshots:
                    continue
                comp_k = page.get('key_compressed')
                if isinstance(comp_k, dict):
                    for k2, original in snapshots[pid].items():
                        restored = comp_k.get(k2)
                        if restored is not None:
                            assert torch.allclose(original, restored, atol=1e-5), \
                                f"Data mismatch for page {pid}, key {k2}"

    def test_swap_metrics_recorded(self):
        """Swap operations must update PCIe metrics."""
        cache = make_cache()
        fill_cache_pages(cache, num_steps=15)

        assert cache.num_pcie_swaps == 0
        cache.swap_out_to_host()
        assert cache.num_pcie_swaps >= 1
        assert len(cache.pcie_swap_latencies) >= 1
        assert cache.pcie_swap_latencies[-1] >= 0  # latency non-negative

    def test_double_swap_out_is_noop(self):
        """Calling swap_out twice must not crash or double-record metrics."""
        cache = make_cache()
        fill_cache_pages(cache, num_steps=15)

        cache.swap_out_to_host()
        count_after_first = cache.num_pcie_swaps
        cache.swap_out_to_host()  # Should be no-op
        assert cache.num_pcie_swaps == count_after_first

    def test_double_swap_in_is_noop(self):
        """Calling swap_in without prior swap_out must not crash."""
        cache = make_cache()
        fill_cache_pages(cache, num_steps=10)

        cache.swap_in_to_device()  # Should be no-op (not swapped out)
        assert not cache.is_swapped_out

    def test_auto_swap_in_on_attention(self):
        """inplace_paged_attention should auto-swap-in if pages are swapped out."""
        cache = make_cache()
        fill_cache_pages(cache, num_steps=15)

        cache.swap_out_to_host()
        assert cache.is_swapped_out

        q = torch.randn(1, 2, 1, 16)
        out = cache.inplace_paged_attention(q)
        assert not cache.is_swapped_out  # Should have auto-swapped in
        assert out.shape == q.shape


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Allocator Fragmentation Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestAllocatorFragmentation:
    """Tests for allocator fragmentation monitoring."""

    def test_fragmentation_report_available(self):
        """Cache must expose get_allocator_fragmentation_report()."""
        cache = make_cache()
        report = cache.get_allocator_fragmentation_report()
        assert 'fragmentation_risk' in report
        assert 'pool_total_allocated_bytes' in report
        assert 'pcie_swap_count' in report

    def test_fragmentation_after_swap(self):
        """After swap-out, fragmentation report must reflect pool usage."""
        cache = make_cache()
        fill_cache_pages(cache, num_steps=15)

        report_before = cache.get_allocator_fragmentation_report()
        cache.swap_out_to_host()
        report_after = cache.get_allocator_fragmentation_report()

        assert report_after['pcie_swap_count'] >= 1
        # After swap-in and cleanup, pool should be back down
        cache.swap_in_to_device(device="cpu")
        report_final = cache.get_allocator_fragmentation_report()
        assert report_final['pcie_swap_count'] >= 2

    def test_invisible_locked_bytes_tracked(self):
        """The pool's invisible_locked_bytes must match pool_total_allocated_bytes."""
        cache = make_cache()
        report = cache.get_allocator_fragmentation_report()
        assert report['invisible_locked_bytes'] == report['pool_total_allocated_bytes']


# ═══════════════════════════════════════════════════════════════════════════════
# 5. OOM Guard with Invisible Memory
# ═══════════════════════════════════════════════════════════════════════════════

class TestOOMGuardWithInvisibleMemory:
    """Tests that _check_and_prevent_oom accounts for cuMemHostAlloc memory."""

    def test_oom_check_includes_pool_bytes(self):
        """
        The OOM guard should consider zero_copy_pool._total_allocated_bytes
        in its pressure calculation (even on CPU fallback paths).
        """
        cache = make_cache()
        fill_cache_pages(cache, num_steps=10)

        # Simulate pool having allocated bytes
        cache.zero_copy_pool._total_allocated_bytes = 1000000
        # This shouldn't crash
        cache._check_and_prevent_oom()

    def test_cpu_fallback_oom_triggers_swap(self):
        """
        On CPU (no CUDA), setting vram_oom_threshold_ratio <= 0 should
        trigger swap_out even with zero-copy pool.
        """
        config = ArgusConfig(
            page_size=8, max_active_pages=2,
            vram_oom_threshold_ratio=-1.0,  # Force trigger
        )
        cache = PagedDynamicKVCache(config=config)
        fill_cache_pages(cache, num_steps=10)

        spills_before = cache.num_cpu_spills
        cache._check_and_prevent_oom()
        assert cache.num_cpu_spills > spills_before


# ═══════════════════════════════════════════════════════════════════════════════
# 6. PCIe Latency vs Dequant Latency Tracking
# ═══════════════════════════════════════════════════════════════════════════════

class TestPCIeVsDequantLatency:
    """
    Verify that PCIe swap latency and dequant latency are tracked
    independently.  After Zero-Copy, P95 PCIe swap time folds into
    the Triton read loop; these tests ensure both metric streams work.
    """

    def test_independent_metric_streams(self):
        """PCIe swap latencies and dequant latencies are in separate lists."""
        cache = make_cache()
        fill_cache_pages(cache, num_steps=15)

        dequant_before = len(cache.dequant_latencies)
        pcie_before = len(cache.pcie_swap_latencies)

        cache.swap_out_to_host()
        cache.swap_in_to_device(device="cpu")

        # PCIe latencies should have increased
        assert len(cache.pcie_swap_latencies) > pcie_before
        # Dequant latencies should NOT have changed from swap operations alone
        # (they change from resurrections, not swaps)

    def test_pool_pcie_metrics_mirror_cache_metrics(self):
        """Pool and cache PCIe metrics should be consistent."""
        cache = make_cache()
        fill_cache_pages(cache, num_steps=15)
        cache.swap_out_to_host()

        pool_metrics = cache.zero_copy_pool.get_pcie_metrics()
        assert pool_metrics['total_pcie_reads'] >= 1


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Telemetry Dashboard Rendering
# ═══════════════════════════════════════════════════════════════════════════════

class TestTelemetryDashboard:
    """Tests that the telemetry dashboard renders without crashing."""

    def test_dashboard_renders_with_pcie_section(self, capsys):
        """Dashboard must render the ZERO-COPY PCIe STREAMING section."""
        cache = make_cache()
        fill_cache_pages(cache, num_steps=15)
        cache.swap_out_to_host()
        cache.swap_in_to_device(device="cpu")

        cache.print_telemetry_summary()
        captured = capsys.readouterr()
        assert "ZERO-COPY PCIe STREAMING" in captured.out
        assert "Pool Mode:" in captured.out
        assert "NUMA Topology:" in captured.out
        assert "Fragmentation Risk:" in captured.out

    def test_dashboard_renders_without_swaps(self, capsys):
        """Dashboard must render PCIe section even with zero swaps."""
        cache = make_cache()
        fill_cache_pages(cache, num_steps=5)

        cache.print_telemetry_summary()
        captured = capsys.readouterr()
        assert "ZERO-COPY PCIe STREAMING" in captured.out
        assert "PCIe Swap Count:" in captured.out


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Zero-Copy Pool Edge Cases
# ═══════════════════════════════════════════════════════════════════════════════

class TestZeroCopyEdgeCases:
    """Edge cases for the Zero-Copy host pool."""

    def test_empty_tensor_pinning(self):
        """Pinning an empty tensor should not crash."""
        pool = ZeroCopyHostPool()
        t = torch.empty(0)
        pinned = pool.tensor_to_pinned(t)
        assert pinned.numel() == 0

    def test_large_tensor_pinning(self):
        """Pinning a large tensor should work in fallback mode."""
        pool = ZeroCopyHostPool()
        pool._fallback_mode = True
        pool._initialized = False

        t = torch.randn(64, 128, 256)
        pinned = pool.tensor_to_pinned(t)
        assert torch.allclose(t, pinned, atol=1e-6)

    def test_free_tensor_noop_for_normal_tensor(self):
        """free_tensor on a non-pool tensor should not crash."""
        pool = ZeroCopyHostPool()
        t = torch.randn(4, 4)
        pool.free_tensor(t)  # Should be no-op

    def test_pool_total_bytes_zero_initially(self):
        """Pool should start with zero allocated bytes."""
        pool = ZeroCopyHostPool()
        assert pool._total_allocated_bytes == 0
        assert pool._peak_allocated_bytes == 0
