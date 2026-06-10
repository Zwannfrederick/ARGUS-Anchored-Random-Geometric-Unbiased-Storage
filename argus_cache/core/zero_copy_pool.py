"""
ARGUS Zero-Copy PCIe Streaming Host Memory Pool

Uses CUDA Driver API (cuMemHostAlloc with CU_MEMHOSTALLOC_DEVICEMAP)
to allocate page-locked host memory that is directly accessible from
GPU via PCIe bus without explicit cudaMemcpy transfers.

Key design decisions:
- Memory is allocated outside PyTorch's caching allocator to prevent
  fragmentation and allocator conflict (segfault prevention)
- NUMA affinity is applied on multi-socket systems (H100/L40S) for
  minimal cross-socket PCIe latency
- Single-socket/laptop systems automatically bypass NUMA binding
- Allocator fragmentation is tracked via shadow accounting so
  cuMemHostAlloc-locked bytes (invisible to torch.cuda.memory_allocated)
  never cause silent OOM
- PCIe transfer latency metrics are tracked separately from dequant
  latency to monitor bus saturation post-zero-copy migration
"""

import torch
import ctypes
import os
import time
from typing import Dict, Optional, Tuple, List

from .logger import argus_log

# ─── CUDA Driver API Constants ───────────────────────────────────────────────
CU_MEMHOSTALLOC_PORTABLE    = 0x01
CU_MEMHOSTALLOC_DEVICEMAP   = 0x02
CU_MEMHOSTALLOC_WRITECOMBINED = 0x04
CUDA_SUCCESS = 0


# ─── NUMA Topology Detector ─────────────────────────────────────────────────

class NUMATopology:
    """
    Detects and manages NUMA topology for optimal PCIe memory placement.
    On single-socket laptops / desktops (num_nodes <= 1), all NUMA binding
    is automatically bypassed to prevent crashes.
    """

    def __init__(self):
        self._num_nodes = self._detect_numa_nodes()
        self._gpu_numa_node = self._detect_gpu_numa_node()

    # ── Detection ─────────────────────────────────────────────────────────

    def _detect_numa_nodes(self) -> int:
        """Detect number of NUMA nodes via sysfs."""
        try:
            node_path = "/sys/devices/system/node"
            if os.path.exists(node_path):
                nodes = [d for d in os.listdir(node_path) if d.startswith("node")]
                return max(len(nodes), 1)
        except Exception:
            pass
        return 1

    def _detect_gpu_numa_node(self) -> int:
        """Detect which NUMA node the primary GPU is closest to (sysfs PCI)."""
        try:
            pci_path = "/sys/bus/pci/devices"
            if not os.path.exists(pci_path):
                return 0
            for device in sorted(os.listdir(pci_path)):
                class_path = os.path.join(pci_path, device, "class")
                if not os.path.exists(class_path):
                    continue
                with open(class_path) as f:
                    cls = f.read().strip()
                # 0x030200 = 3D controller, 0x030000 = VGA compatible
                if cls.startswith("0x0302") or cls.startswith("0x0300"):
                    numa_path = os.path.join(pci_path, device, "numa_node")
                    if os.path.exists(numa_path):
                        with open(numa_path) as f:
                            node = int(f.read().strip())
                        if node >= 0:
                            return node
        except Exception:
            pass
        return 0

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def num_nodes(self) -> int:
        return self._num_nodes

    @property
    def gpu_numa_node(self) -> int:
        return self._gpu_numa_node

    @property
    def is_multi_socket(self) -> bool:
        return self._num_nodes > 1

    # ── NUMA Binding ──────────────────────────────────────────────────────

    def bind_to_gpu_node(self) -> bool:
        """Set NUMA memory policy to prefer GPU-local node.  No-op on single-socket."""
        if not self.is_multi_socket:
            return False
        try:
            libnuma = ctypes.CDLL("libnuma.so.1", mode=ctypes.RTLD_GLOBAL)
            libnuma.numa_available.restype = ctypes.c_int
            if libnuma.numa_available() < 0:
                return False
            libnuma.numa_set_preferred(ctypes.c_int(self._gpu_numa_node))
            return True
        except (OSError, AttributeError):
            return False


# ─── Zero-Copy Host Memory Pool ─────────────────────────────────────────────

class ZeroCopyHostPool:
    """
    Pinned + Device-Mapped host memory pool using CUDA Driver API.

    Every tensor stored via ``tensor_to_pinned()`` is backed by
    ``cuMemHostAlloc(..., CU_MEMHOSTALLOC_DEVICEMAP)`` and thus is
    readable directly from GPU kernels via PCIe without cudaMemcpy.

    The pool keeps shadow accounting of all allocations so that
    ``get_fragmentation_report()`` can compare physical occupancy against
    what ``torch.cuda.memory_allocated()`` reports (which is blind to
    driver-level allocations).
    """

    def __init__(self, device_ordinal: int = 0):
        self._cuda_driver = None
        # host_ptr_int -> (ctypes.c_void_p, byte_size)
        self._allocations: Dict[int, Tuple[ctypes.c_void_p, int]] = {}
        # host_ptr_int -> device ctypes.c_void_p
        self._device_pointers: Dict[int, ctypes.c_void_p] = {}
        self._total_allocated_bytes: int = 0
        self._peak_allocated_bytes: int = 0
        self._device_ordinal = device_ordinal
        self._initialized = False
        self._fallback_mode = False

        # NUMA topology
        self.numa = NUMATopology()

        # PCIe transfer metrics  (tracked separately from dequant latency)
        self.pcie_transfer_latencies: List[float] = []
        self.pcie_read_count: int = 0
        self.pcie_bytes_transferred: int = 0

        self._try_init()

    # ── Initialization ────────────────────────────────────────────────────

    def _try_init(self):
        """Load libcuda and set up function signatures."""
        if not torch.cuda.is_available():
            self._fallback_mode = True
            argus_log("INFO",
                      "ZeroCopyHostPool: CUDA unavailable → fallback pin_memory mode",
                      line_no=0)
            return

        try:
            self._cuda_driver = ctypes.CDLL("libcuda.so.1")

            # cuMemHostAlloc(void **pp, size_t bytesize, unsigned int flags)
            self._cuda_driver.cuMemHostAlloc.argtypes = [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.c_size_t,
                ctypes.c_uint,
            ]
            self._cuda_driver.cuMemHostAlloc.restype = ctypes.c_int

            # cuMemHostGetDevicePointer_v2(CUdeviceptr *pdptr, void *p, unsigned int flags)
            self._cuda_driver.cuMemHostGetDevicePointer_v2.argtypes = [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.c_void_p,
                ctypes.c_uint,
            ]
            self._cuda_driver.cuMemHostGetDevicePointer_v2.restype = ctypes.c_int

            # cuMemFreeHost(void *p)
            self._cuda_driver.cuMemFreeHost.argtypes = [ctypes.c_void_p]
            self._cuda_driver.cuMemFreeHost.restype = ctypes.c_int

            # NUMA affinity (multi-socket only)
            if self.numa.is_multi_socket:
                bound = self.numa.bind_to_gpu_node()
                if bound:
                    argus_log("INFO",
                              f"ZeroCopyHostPool: NUMA affinity → node {self.numa.gpu_numa_node} "
                              f"(GPU-local) on {self.numa.num_nodes}-node system",
                              line_no=0)
                else:
                    argus_log("WARNING",
                              f"ZeroCopyHostPool: NUMA bind failed on {self.numa.num_nodes}-node "
                              f"system — cross-socket PCIe latency may be elevated",
                              line_no=0)
            else:
                argus_log("INFO",
                          "ZeroCopyHostPool: Single-socket system → NUMA binding bypassed",
                          line_no=0)

            # ── Capability probe: verify cuMemHostAlloc+DEVICEMAP actually works ──
            # Mobile GPUs (Optimus/MUXless) reject CU_MEMHOSTALLOC_DEVICEMAP with
            # CUDA_ERROR_INVALID_VALUE (201) even when libcuda loads fine.
            # Without this probe, _initialized stays True while every real
            # allocation silently falls back to pin_memory() — hiding the failure.
            _probe_min_bytes = 65536  # 64 KB — smallest reliable cuMemHostAlloc size
            _probe_ptr = ctypes.c_void_p()
            _probe_flags = CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_DEVICEMAP
            _probe_result = self._cuda_driver.cuMemHostAlloc(
                ctypes.byref(_probe_ptr),
                ctypes.c_size_t(_probe_min_bytes),
                ctypes.c_uint(_probe_flags),
            )
            if _probe_result != CUDA_SUCCESS:
                # Device-mapped pinned memory not supported on this GPU/driver combo.
                # Fall back to pin_memory() silently so callers work correctly.
                self._fallback_mode = True
                argus_log("WARNING",
                          f"ZeroCopyHostPool: cuMemHostAlloc+DEVICEMAP not supported "
                          f"(err={_probe_result}, likely Optimus/MUXless dGPU) "
                          f"→ falling back to pin_memory() mode",
                          line_no=0)
                return
            # Probe succeeded — free the test allocation and mark as ready
            self._cuda_driver.cuMemFreeHost(ctypes.c_void_p(_probe_ptr.value))

            self._initialized = True
            argus_log("INFO",
                      "ZeroCopyHostPool: CUDA Driver API initialised "
                      "(cuMemHostAlloc + DEVICEMAP) — Zero-Copy PCIe active",
                      line_no=0)

        except (OSError, AttributeError) as exc:
            self._fallback_mode = True
            argus_log("WARNING",
                      f"ZeroCopyHostPool: CUDA Driver API unavailable ({exc}) "
                      f"→ fallback pin_memory mode",
                      line_no=0)

    # ── Raw Allocation / Free ─────────────────────────────────────────────

    def allocate(self, size_bytes: int) -> Optional[ctypes.c_void_p]:
        """Allocate page-locked, device-mapped host memory."""
        if self._fallback_mode or not self._initialized:
            return None

        ptr = ctypes.c_void_p()
        flags = CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_DEVICEMAP

        result = self._cuda_driver.cuMemHostAlloc(
            ctypes.byref(ptr),
            ctypes.c_size_t(size_bytes),
            ctypes.c_uint(flags),
        )
        if result != CUDA_SUCCESS:
            argus_log("WARNING",
                      f"ZeroCopyHostPool: cuMemHostAlloc failed (err={result}), "
                      f"size={size_bytes} bytes",
                      line_no=0)
            return None

        # Map to device address space
        dev_ptr = ctypes.c_void_p()
        result = self._cuda_driver.cuMemHostGetDevicePointer_v2(
            ctypes.byref(dev_ptr), ptr, ctypes.c_uint(0),
        )
        if result != CUDA_SUCCESS:
            self._cuda_driver.cuMemFreeHost(ptr)
            argus_log("WARNING",
                      f"ZeroCopyHostPool: cuMemHostGetDevicePointer failed (err={result})",
                      line_no=0)
            return None

        ptr_val = ptr.value
        self._allocations[ptr_val] = (ptr, size_bytes)
        self._device_pointers[ptr_val] = dev_ptr
        self._total_allocated_bytes += size_bytes
        self._peak_allocated_bytes = max(
            self._peak_allocated_bytes, self._total_allocated_bytes
        )
        return ptr

    def free(self, ptr: ctypes.c_void_p):
        """Free a previously allocated host block."""
        if ptr is None or self._fallback_mode:
            return
        ptr_val = ptr.value if hasattr(ptr, "value") else ptr
        if ptr_val in self._allocations:
            _, size = self._allocations[ptr_val]
            self._cuda_driver.cuMemFreeHost(ctypes.c_void_p(ptr_val))
            self._total_allocated_bytes -= size
            del self._allocations[ptr_val]
            self._device_pointers.pop(ptr_val, None)

    def get_device_pointer(self, host_ptr) -> Optional[ctypes.c_void_p]:
        """Return the device-mapped pointer for *host_ptr*."""
        ptr_val = host_ptr.value if hasattr(host_ptr, "value") else host_ptr
        return self._device_pointers.get(ptr_val)

    # ── Tensor-Level API ──────────────────────────────────────────────────

    def tensor_to_pinned(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Copy *tensor* into pinned, device-mapped host memory and return
        the new tensor.

        If the CUDA Driver API is unavailable, falls back to
        ``torch.Tensor.pin_memory()``.

        The returned tensor carries private attributes
        ``_zero_copy_ptr``, ``_zero_copy_pool``, ``_zero_copy_nbytes``
        so it can be freed later by ``free_tensor()``.
        """
        if self._fallback_mode or not self._initialized:
            return self._fallback_pin(tensor)

        nbytes = tensor.nelement() * tensor.element_size()
        if nbytes == 0:
            return tensor

        ptr = self.allocate(nbytes)
        if ptr is None:
            return self._fallback_pin(tensor)

        # Wrap the raw host pointer as a PyTorch tensor  (zero-copy view)
        buf = (ctypes.c_byte * nbytes).from_address(ptr.value)
        pinned = torch.frombuffer(buf, dtype=tensor.dtype).reshape(tensor.shape)

        # Copy data into the pinned buffer
        src = tensor.detach().cpu() if tensor.is_cuda else tensor.detach()
        pinned.copy_(src)

        # Tag for later cleanup
        pinned._zero_copy_ptr = ptr
        pinned._zero_copy_pool = self
        pinned._zero_copy_nbytes = nbytes
        pinned._zero_copy_managed = True
        return pinned

    def free_tensor(self, tensor: torch.Tensor):
        """Free the pinned backing memory of a tensor produced by ``tensor_to_pinned``."""
        ptr = getattr(tensor, "_zero_copy_ptr", None)
        if ptr is not None:
            self.free(ptr)
            tensor._zero_copy_ptr = None

    def is_zero_copy_tensor(self, tensor: torch.Tensor) -> bool:
        """Return True if *tensor* is backed by cuMemHostAlloc memory."""
        return getattr(tensor, "_zero_copy_managed", False)

    # ── Fallback ──────────────────────────────────────────────────────────

    @staticmethod
    def _fallback_pin(tensor: torch.Tensor) -> torch.Tensor:
        cpu_t = tensor.detach().cpu() if tensor.is_cuda else tensor.detach()
        if torch.cuda.is_available():
            try:
                return cpu_t.pin_memory()
            except Exception:
                pass
        return cpu_t

    # ── Fragmentation & OOM Safety ────────────────────────────────────────

    def get_fragmentation_report(self) -> dict:
        """
        Compare CUDA Driver-locked bytes (invisible to PyTorch) against
        PyTorch's own view of GPU memory.  Returns a risk assessment that
        callers can use to pre-empt OOM.

        Keys:
          pool_total_allocated_bytes   – bytes held by cuMemHostAlloc
          pool_peak_allocated_bytes    – high-water mark
          pool_num_allocations         – live allocation count
          pytorch_allocated_bytes      – torch.cuda.memory_allocated()
          pytorch_reserved_bytes       – torch.cuda.memory_reserved()
          invisible_locked_bytes       – pool bytes invisible to PyTorch
          total_physical_usage_bytes   – PyTorch + pool
          total_physical_usage_ratio   – fraction of device total
          device_total_memory          – device VRAM
          fragmentation_risk           – LOW / MEDIUM / HIGH / CRITICAL
        """
        report = {
            "pool_total_allocated_bytes": self._total_allocated_bytes,
            "pool_peak_allocated_bytes": self._peak_allocated_bytes,
            "pool_num_allocations": len(self._allocations),
            "pytorch_allocated_bytes": 0,
            "pytorch_reserved_bytes": 0,
            "invisible_locked_bytes": self._total_allocated_bytes,
            "fragmentation_risk": "LOW",
        }

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

    # ── PCIe Transfer Metrics ─────────────────────────────────────────────

    def record_pcie_transfer(self, latency_ms: float, bytes_transferred: int):
        """Record a single PCIe read for latency / bandwidth bookkeeping."""
        self.pcie_transfer_latencies.append(latency_ms)
        self.pcie_read_count += 1
        self.pcie_bytes_transferred += bytes_transferred

    def get_pcie_metrics(self) -> dict:
        """
        Return PCIe transfer performance statistics.

        After Zero-Copy migration the P95 PCIe swap time that was
        previously a separate cudaMemcpy is folded into the Triton
        read loop (Average Dequant Latency).  This method lets
        callers monitor both independently.
        """
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

    # ── Cleanup ───────────────────────────────────────────────────────────

    def cleanup(self):
        """Free every live allocation."""
        for ptr_val, (ptr, _) in list(self._allocations.items()):
            try:
                self._cuda_driver.cuMemFreeHost(ptr)
            except Exception:
                pass
        self._allocations.clear()
        self._device_pointers.clear()
        self._total_allocated_bytes = 0

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass

    # ── Repr ──────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        mode = "zero-copy" if self._initialized and not self._fallback_mode else "fallback"
        return (
            f"ZeroCopyHostPool(mode={mode}, "
            f"allocs={len(self._allocations)}, "
            f"total={self._total_allocated_bytes / (1024**2):.1f}MB, "
            f"peak={self._peak_allocated_bytes / (1024**2):.1f}MB, "
            f"numa_nodes={self.numa.num_nodes})"
        )
