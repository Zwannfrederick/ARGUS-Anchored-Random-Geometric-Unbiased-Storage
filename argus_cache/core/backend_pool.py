"""Contiguous block pool allocator for quantized backend pages (Stage S5).

Allocates unified, contiguous memory chunks for quantized pages (q8_0, q4_0),
eliminating per-page malloc fragmentation and enabling direct batch PCIe streaming / GPU residency.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple
import torch

from argus_cache.core.page_table import CodecKind, PlacementLocation


class ContiguousBlockPool:
    """Contiguous memory pool for fixed-size quantized KV page blocks."""

    def __init__(
        self,
        codec: CodecKind,
        placement: PlacementLocation,
        max_slots: int,
        page_size: int,
        num_heads: int,
        head_dim: int,
        device: str = "cpu",
        batch_size: int = 1,
    ):
        self.codec = CodecKind(codec)
        self.placement = PlacementLocation(placement)
        if min(max_slots, page_size, num_heads, head_dim, batch_size) <= 0:
            raise ValueError("Pool capacity and geometry must be positive")
        self.max_slots = max_slots
        self.page_size = page_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device = device
        self.batch_size = batch_size

        # Calculate exact block bytes per page for K and V
        # Total elements per K or V page = num_heads * page_size * head_dim
        elements_per_page = batch_size * num_heads * page_size * head_dim
        if codec not in (CodecKind.ACTIVE_FP16, CodecKind.ACTIVE_BF16) and elements_per_page % 32 != 0:
            raise ValueError(f"Total elements per page ({elements_per_page}) must be multiple of 32 for GGML block layout")

        num_blocks = elements_per_page // 32
        if codec == CodecKind.GGML_Q8_0:
            # 34 bytes per block (2 bytes fp16 scale + 32 bytes int8 quants)
            self.bytes_per_page = num_blocks * 34
        elif codec == CodecKind.GGML_Q4_0:
            # 18 bytes per block (2 bytes fp16 scale + 16 bytes packed uint8 nibbles)
            self.bytes_per_page = num_blocks * 18
        elif codec in (CodecKind.ACTIVE_FP16, CodecKind.ACTIVE_BF16):
            self.bytes_per_page = elements_per_page * 2
        else:
            raise ValueError(f"Unsupported pool codec: {codec}")

        self._allocate_storage()

        # Slot free list
        self.free_slots: List[int] = list(range(max_slots - 1, -1, -1))
        self.allocated_slots: Set[int] = set()

    def _allocate_storage(self) -> None:

        # Allocate contiguous storage for Key and Value
        # [max_slots, bytes_per_page] uint8 tensor
        if self.placement == PlacementLocation.DISK:
            raise ValueError("Use DiskBlockPool for disk placement")
        target_device = torch.device(self.device)
        if self.placement == PlacementLocation.GPU_DEVICE:
            if not torch.cuda.is_available():
                raise RuntimeError("GPU pool requested without CUDA access")
            if target_device.type != "cuda":
                target_device = torch.device("cuda")
        else:
            target_device = torch.device("cpu")
        pinned = self.placement == PlacementLocation.HOST_PINNED
        if pinned and not torch.cuda.is_available():
            raise RuntimeError("Pinned pool requested without CUDA access")
        options = dict(dtype=torch.uint8, device=target_device, pin_memory=pinned)
        self.k_storage = torch.zeros((self.max_slots, self.bytes_per_page), **options)
        self.v_storage = torch.zeros((self.max_slots, self.bytes_per_page), **options)

    def allocate_slot(self) -> int:
        """Allocates a slot index from the contiguous pool."""
        if not self.free_slots:
            raise RuntimeError(f"ContiguousBlockPool ({self.codec.name}, {self.placement.name}) out of memory ({self.max_slots} slots full)")
        slot = self.free_slots.pop()
        self.allocated_slots.add(slot)
        return slot

    def free_slot(self, slot: int) -> None:
        """Returns slot to the pool free list."""
        if slot not in self.allocated_slots:
            return
        self.allocated_slots.remove(slot)
        self.free_slots.append(slot)

    def write_page_bytes(self, slot: int, k_bytes: torch.Tensor, v_bytes: torch.Tensor) -> None:
        """Writes compressed bytes into the allocated slot."""
        self._validate_write(slot, k_bytes, v_bytes)
        self.k_storage[slot].copy_(k_bytes.flatten())
        self.v_storage[slot].copy_(v_bytes.flatten())

    def _validate_write(self, slot, k_bytes, v_bytes):
        self._validate_slot(slot)
        for data in (k_bytes, v_bytes):
            if data.dtype != torch.uint8 or data.numel() != self.bytes_per_page:
                raise ValueError(f"Expected exactly {self.bytes_per_page} uint8 bytes per K/V page")

    def _validate_slot(self, slot):
        if slot not in self.allocated_slots:
            raise ValueError(f"Slot {slot} is not allocated")

    def read_page_bytes(self, slot: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reads compressed key and value views from the allocated slot."""
        self._validate_slot(slot)
        return self.k_storage[slot], self.v_storage[slot]

    def capacity_bytes(self) -> int:
        return self.max_slots * self.bytes_per_page * 2

    def live_bytes(self) -> int:
        return len(self.allocated_slots) * self.bytes_per_page * 2

    def fragmentation_ratio(self) -> float:
        if not self.allocated_slots:
            return 0.0
        return 1.0 - (len(self.allocated_slots) / self.max_slots)
