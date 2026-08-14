"""
ARGUS Zero-Copy PCIe Streaming Host Memory Pool (C++ Backend Delegate)
"""

import torch
import argus_cpp_backend
from .logger import argus_log

class NUMATopology:
    """
    Detects and manages NUMA topology for optimal PCIe memory placement.
    On single-socket laptops / desktops (num_nodes <= 1), all NUMA binding
    is automatically bypassed to prevent crashes.
    """

    def __init__(self):
        self._num_nodes = 1
        self._gpu_numa_node = 0
        try:
            pool = argus_cpp_backend.ZeroCopyHostPool(0)
            self._num_nodes = pool.num_nodes
            self._gpu_numa_node = pool.gpu_numa_node
        except Exception:
            pass

    @property
    def num_nodes(self) -> int:
        return self._num_nodes

    @property
    def gpu_numa_node(self) -> int:
        return self._gpu_numa_node

    @property
    def is_multi_socket(self) -> bool:
        return self._num_nodes > 1

    def bind_to_gpu_node(self) -> bool:
        return self.is_multi_socket


class ZeroCopyHostPool:
    """
    Pinned + Device-Mapped host memory pool backed by C++ ZeroCopyHostPool.
    """

    def __init__(self, device_ordinal: int = 0):
        self._backend = argus_cpp_backend.ZeroCopyHostPool(device_ordinal)
        self.numa = NUMATopology()
        self.pcie_transfer_latencies = []
        self.pcie_read_count = 0
        self.pcie_bytes_transferred = 0

    @property
    def _total_allocated_bytes(self):
        return self._backend._total_allocated_bytes if self._backend else 0

    @_total_allocated_bytes.setter
    def _total_allocated_bytes(self, value):
        if self._backend:
            self._backend._total_allocated_bytes = value

    @property
    def _peak_allocated_bytes(self):
        return self._backend._peak_allocated_bytes if self._backend else 0

    @_peak_allocated_bytes.setter
    def _peak_allocated_bytes(self, value):
        if self._backend:
            self._backend._peak_allocated_bytes = value

    @property
    def _fallback_mode(self):
        return self._backend._fallback_mode if self._backend else True

    @_fallback_mode.setter
    def _fallback_mode(self, value):
        if self._backend:
            self._backend._fallback_mode = value

    @property
    def _initialized(self):
        return self._backend._initialized if self._backend else False

    @_initialized.setter
    def _initialized(self, value):
        if self._backend:
            self._backend._initialized = value

    def allocate(self, size_bytes: int):
        return self._backend.allocate(size_bytes) if self._backend else None

    def free(self, ptr):
        if self._backend:
            self._backend.free(ptr)

    def get_device_pointer(self, host_ptr):
        return self._backend.get_device_pointer(host_ptr) if self._backend else None

    def tensor_to_pinned(self, tensor: torch.Tensor) -> torch.Tensor:
        return self._backend.tensor_to_pinned(tensor) if self._backend else tensor

    def free_tensor(self, tensor: torch.Tensor):
        if self._backend:
            self._backend.free_tensor(tensor)

    def is_zero_copy_tensor(self, tensor: torch.Tensor) -> bool:
        if getattr(tensor, "_zero_copy_managed", False):
            return True
        return self._backend.is_zero_copy_tensor(tensor) if self._backend else False

    def get_fragmentation_report(self) -> dict:
        report = self._backend.get_fragmentation_report()
        if torch.cuda.is_available():
            report["pytorch_allocated_bytes"] = torch.cuda.memory_allocated()
            report["pytorch_reserved_bytes"] = torch.cuda.memory_reserved()
            total_mem = torch.cuda.get_device_properties(0).total_memory
            total_used = report["pytorch_allocated_bytes"] + self._total_allocated_bytes
            ratio = total_used / total_mem if total_mem > 0 else 0.0

            if ratio > 0.90:
                risk = "CRITICAL"
            elif ratio > 0.75:
                risk = "HIGH"
            elif ratio > 0.50:
                risk = "MEDIUM"
            else:
                risk = "LOW"

            report["total_physical_usage_bytes"] = total_used
            report["total_physical_usage_ratio"] = ratio
            report["device_total_memory"] = total_mem
            report["fragmentation_risk"] = risk
        return report

    def record_pcie_transfer(self, latency_ms: float, bytes_transferred: int):
        self.pcie_transfer_latencies.append(latency_ms)
        self.pcie_read_count += 1
        self.pcie_bytes_transferred += bytes_transferred

    def get_pcie_metrics(self) -> dict:
        lats = self.pcie_transfer_latencies
        n = len(lats)
        if n == 0:
            return {
                "avg_pcie_latency_ms": 0.0,
                "p50_pcie_latency_ms": 0.0,
                "p95_pcie_latency_ms": 0.0,
                "p99_pcie_latency_ms": 0.0,
                "total_pcie_reads": 0,
                "total_pcie_bytes": 0,
                "pcie_bandwidth_gbps": 0.0,
            }

        sorted_lat = sorted(lats)
        avg = sum(lats) / n
        p50 = sorted_lat[int(n * 0.50)]
        p95 = sorted_lat[min(n - 1, int(n * 0.95))]
        p99 = sorted_lat[min(n - 1, int(n * 0.99))]

        total_time_s = sum(lats) / 1000.0
        bw = ((self.pcie_bytes_transferred / (1024 ** 3)) / total_time_s
              if total_time_s > 0 else 0.0)

        return {
            "avg_pcie_latency_ms": avg,
            "p50_pcie_latency_ms": p50,
            "p95_pcie_latency_ms": p95,
            "p99_pcie_latency_ms": p99,
            "total_pcie_reads": self.pcie_read_count,
            "total_pcie_bytes": self.pcie_bytes_transferred,
            "pcie_bandwidth_gbps": bw,
        }

    def cleanup(self):
        if self._backend:
            self._backend = None
            self._total_allocated_bytes = 0

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass

    def __repr__(self) -> str:
        mode = "zero-copy" if self._initialized and not self._fallback_mode else "fallback"
        return (
            f"ZeroCopyHostPool(mode={mode}, "
            f"total={self._total_allocated_bytes / (1024**2):.1f}MB, "
            f"peak={self._peak_allocated_bytes / (1024**2):.1f}MB, "
            f"numa_nodes={self.numa.num_nodes})"
        )
