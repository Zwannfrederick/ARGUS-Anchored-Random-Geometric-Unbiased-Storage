"""Structure-of-Arrays (SoA) Page Table ABI for ARGUS (Stage S5).

Separates precision (ACTIVE, q8_0, q4_0) from placement (GPU, PINNED_HOST, PAGEABLE_HOST).
Avoids raw pointer / object traversal on the decode hot path by maintaining a contiguous,
vectorized descriptor table with generation counters for stale async eviction protection.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch


class CodecKind(enum.IntEnum):
    """Supported physical compression formats."""
    ACTIVE_FP16 = 0
    GGML_Q8_0 = 1
    GGML_Q4_0 = 2
    CUSTOM_PLUGIN = 99


class PlacementLocation(enum.IntEnum):
    """Memory residency tiers."""
    GPU_DEVICE = 0
    HOST_PINNED = 1
    HOST_PAGEABLE = 2


@dataclass
class PageDescriptor:
    """Logical page descriptor."""
    page_id: int
    logical_pos: int
    token_count: int
    codec: CodecKind
    placement: PlacementLocation
    pool_slot: int
    byte_offset: int
    generation: int
    k_scale: float = 1.0
    v_scale: float = 1.0


class StructureOfArraysPageTable:
    """High-performance Structure-of-Arrays (SoA) page table for hot-path decode dispatch."""

    def __init__(self, capacity: int = 4096, device: str = "cpu"):
        self.capacity = capacity
        self.device = device
        self.num_pages = 0
        self.global_generation = 1

        # Vectorized Structure-of-Arrays tensors for fast GPU/C++ batch dispatch
        self.page_ids = torch.full((capacity,), -1, dtype=torch.int32, device=device)
        self.logical_positions = torch.zeros((capacity,), dtype=torch.int64, device=device)
        self.token_counts = torch.zeros((capacity,), dtype=torch.int32, device=device)
        self.codecs = torch.zeros((capacity,), dtype=torch.int32, device=device)
        self.placements = torch.zeros((capacity,), dtype=torch.int32, device=device)
        self.pool_slots = torch.full((capacity,), -1, dtype=torch.int32, device=device)
        self.byte_offsets = torch.zeros((capacity,), dtype=torch.int64, device=device)
        self.generations = torch.zeros((capacity,), dtype=torch.int64, device=device)
        self.k_scales = torch.ones((capacity,), dtype=torch.float32, device=device)
        self.v_scales = torch.ones((capacity,), dtype=torch.float32, device=device)

        # Fast lookup mapping: page_id -> entry_index
        self._page_id_to_idx: Dict[int, int] = {}

    def allocate_page(
        self,
        page_id: int,
        logical_pos: int,
        token_count: int,
        codec: CodecKind = CodecKind.ACTIVE_FP16,
        placement: PlacementLocation = PlacementLocation.GPU_DEVICE,
        pool_slot: int = -1,
        byte_offset: int = 0,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
    ) -> int:
        """Publishes a new page descriptor into the table."""
        if self.num_pages >= self.capacity:
            self._grow_capacity(self.capacity * 2)

        idx = self.num_pages
        self.page_ids[idx] = page_id
        self.logical_positions[idx] = logical_pos
        self.token_counts[idx] = token_count
        self.codecs[idx] = int(codec)
        self.placements[idx] = int(placement)
        self.pool_slots[idx] = pool_slot
        self.byte_offsets[idx] = byte_offset
        self.generations[idx] = self.global_generation
        self.k_scales[idx] = k_scale
        self.v_scales[idx] = v_scale

        self._page_id_to_idx[page_id] = idx
        self.num_pages += 1
        self.global_generation += 1
        return idx

    def update_placement(
        self,
        page_id: int,
        new_placement: PlacementLocation,
        new_codec: Optional[CodecKind] = None,
        new_pool_slot: Optional[int] = None,
        new_byte_offset: Optional[int] = None,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
    ) -> None:
        """Updates placement / precision in-place with generation bump."""
        idx = self._page_id_to_idx.get(page_id)
        if idx is None:
            raise KeyError(f"Page ID {page_id} not found in page table")

        self.placements[idx] = int(new_placement)
        if new_codec is not None:
            self.codecs[idx] = int(new_codec)
        if new_pool_slot is not None:
            self.pool_slots[idx] = new_pool_slot
        if new_byte_offset is not None:
            self.byte_offsets[idx] = new_byte_offset
        if k_scale is not None:
            self.k_scales[idx] = k_scale
        if v_scale is not None:
            self.v_scales[idx] = v_scale

        self.generations[idx] = self.global_generation
        self.global_generation += 1

    def get_descriptor(self, page_id: int) -> PageDescriptor:
        """Retrieves logical descriptor for a page."""
        idx = self._page_id_to_idx.get(page_id)
        if idx is None:
            raise KeyError(f"Page ID {page_id} not found in page table")

        return PageDescriptor(
            page_id=int(self.page_ids[idx].item()),
            logical_pos=int(self.logical_positions[idx].item()),
            token_count=int(self.token_counts[idx].item()),
            codec=CodecKind(int(self.codecs[idx].item())),
            placement=PlacementLocation(int(self.placements[idx].item())),
            pool_slot=int(self.pool_slots[idx].item()),
            byte_offset=int(self.byte_offsets[idx].item()),
            generation=int(self.generations[idx].item()),
            k_scale=float(self.k_scales[idx].item()),
            v_scale=float(self.v_scales[idx].item()),
        )

    def is_valid_generation(self, page_id: int, expected_gen: int) -> bool:
        """Protects against stale asynchronous eviction access."""
        idx = self._page_id_to_idx.get(page_id)
        if idx is None:
            return False
        return int(self.generations[idx].item()) == expected_gen

    def remove_page(self, page_id: int) -> None:
        """Removes a page from the table and maintains compactness."""
        idx = self._page_id_to_idx.pop(page_id, None)
        if idx is None:
            return

        last_idx = self.num_pages - 1
        if idx != last_idx:
            # Move last element into deleted slot
            last_page_id = int(self.page_ids[last_idx].item())
            self.page_ids[idx] = self.page_ids[last_idx]
            self.logical_positions[idx] = self.logical_positions[last_idx]
            self.token_counts[idx] = self.token_counts[last_idx]
            self.codecs[idx] = self.codecs[last_idx]
            self.placements[idx] = self.placements[last_idx]
            self.pool_slots[idx] = self.pool_slots[last_idx]
            self.byte_offsets[idx] = self.byte_offsets[last_idx]
            self.generations[idx] = self.generations[last_idx]
            self.k_scales[idx] = self.k_scales[last_idx]
            self.v_scales[idx] = self.v_scales[last_idx]
            self._page_id_to_idx[last_page_id] = idx

        # Reset last entry
        self.page_ids[last_idx] = -1
        self.num_pages -= 1
        self.global_generation += 1

    def clear(self) -> None:
        """Invalidates all descriptors."""
        self.page_ids.fill_(-1)
        self._page_id_to_idx.clear()
        self.num_pages = 0
        self.global_generation += 1

    def _grow_capacity(self, new_cap: int) -> None:
        """Doubles tensor capacity."""
        pad = new_cap - self.capacity
        self.page_ids = torch.cat([self.page_ids, torch.full((pad,), -1, dtype=torch.int32, device=self.device)])
        self.logical_positions = torch.cat([self.logical_positions, torch.zeros((pad,), dtype=torch.int64, device=self.device)])
        self.token_counts = torch.cat([self.token_counts, torch.zeros((pad,), dtype=torch.int32, device=self.device)])
        self.codecs = torch.cat([self.codecs, torch.zeros((pad,), dtype=torch.int32, device=self.device)])
        self.placements = torch.cat([self.placements, torch.zeros((pad,), dtype=torch.int32, device=self.device)])
        self.pool_slots = torch.cat([self.pool_slots, torch.full((pad,), -1, dtype=torch.int32, device=self.device)])
        self.byte_offsets = torch.cat([self.byte_offsets, torch.zeros((pad,), dtype=torch.int64, device=self.device)])
        self.generations = torch.cat([self.generations, torch.zeros((pad,), dtype=torch.int64, device=self.device)])
        self.k_scales = torch.cat([self.k_scales, torch.ones((pad,), dtype=torch.float32, device=self.device)])
        self.v_scales = torch.cat([self.v_scales, torch.ones((pad,), dtype=torch.float32, device=self.device)])
        self.capacity = new_cap
