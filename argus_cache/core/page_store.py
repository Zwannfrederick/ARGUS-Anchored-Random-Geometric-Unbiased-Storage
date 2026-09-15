"""Transactional placement of existing encoded pages across bounded pools."""

import threading

import torch

from .direct_attention import DirectPagedAttentionEngine
from .page_table import StructureOfArraysPageTable


class PageStore:
    """Own page descriptors and slots; serialize reads against placement changes.

    Pool capacities are hard payload limits. Callers include descriptor memory and
    decode/transfer staging separately in their process budget.
    """

    def __init__(self, pools, *, descriptor_capacity=64):
        self.pools = {}
        for pool in pools:
            key = (pool.codec, pool.placement)
            if key in self.pools:
                raise ValueError(f"Duplicate pool for {key}")
            if pool.allocated_slots:
                raise ValueError("PageStore requires empty pools so every slot has an owner")
            self.pools[key] = pool
        self.table = StructureOfArraysPageTable(capacity=descriptor_capacity)
        # ponytail: one lock per store; split by sequence if concurrent decode is needed.
        self._lock = threading.RLock()
        self._pinned = set()

    def add(self, page_id, logical_pos, token_count, codec, placement, keys, values, *, pinned=False):
        with self._lock:
            try:
                self.table.get_descriptor(page_id)
            except KeyError:
                pass
            else:
                raise ValueError(f"Page ID {page_id} already exists")
            if not 0 <= page_id < 2**31 or logical_pos < 0:
                raise ValueError("Invalid page ID or logical position")
            pool = self.pools[(codec, placement)]
            if not 0 < token_count <= pool.page_size:
                raise ValueError("Token count must fit one pool page")
            # Validate payload before eviction can change placement.
            for data in (keys, values):
                if data.dtype != torch.uint8 or data.numel() != pool.bytes_per_page:
                    raise ValueError("Encoded K/V bytes do not match pool geometry")
            self._make_room(pool)
            slot = pool.allocate_slot()
            try:
                pool.write_page_bytes(slot, keys, values)
                self.table.allocate_page(page_id, logical_pos, token_count, codec, placement, slot)
            except BaseException:
                pool.free_slot(slot)
                raise
            if pinned:
                self._pinned.add(page_id)

    def _make_room(self, pool):
        if pool.free_slots:
            return
        colder = sorted(
            (p for (codec, placement), p in self.pools.items()
             if codec == pool.codec and placement > pool.placement),
            key=lambda p: p.placement,
        )
        if not colder:
            raise RuntimeError("PageStore capacity exhausted")
        candidates = [
            self.table.get_descriptor(page_id)
            for page_id in self.table.page_ids[:self.table.num_pages].tolist()
            if page_id not in self._pinned
        ]
        candidates = [p for p in candidates if p.codec == pool.codec and p.placement == pool.placement]
        if not candidates:
            raise RuntimeError("All pages in the full pool are pinned")
        # ponytail: oldest-position eviction; add access-based policy if measured reuse requires it.
        victim = min(candidates, key=lambda p: p.logical_pos)
        target = colder[0]
        self._make_room(target)
        self.move(victim.page_id, target.placement, expected_generation=victim.generation)

    def move(self, page_id, placement, *, expected_generation=None):
        """Publish destination only after a successful copy; never discard source on I/O failure."""
        with self._lock:
            descriptor = self.table.get_descriptor(page_id)
            if expected_generation is not None and descriptor.generation != expected_generation:
                raise ValueError("Stale page generation")
            if descriptor.placement == placement:
                return
            if page_id in self._pinned:
                raise ValueError("Pinned page cannot be moved")
            source = self.pools[(descriptor.codec, descriptor.placement)]
            target = self.pools[(descriptor.codec, placement)]
            geometry = ("batch_size", "page_size", "num_heads", "head_dim", "bytes_per_page")
            if any(getattr(source, name) != getattr(target, name) for name in geometry):
                raise ValueError("Source and target pool geometry differ")
            slot = target.allocate_slot()
            try:
                target.write_page_bytes(slot, *source.read_page_bytes(descriptor.pool_slot))
                if not self.table.is_valid_generation(page_id, descriptor.generation):
                    raise ValueError("Page changed during transfer")
            except BaseException:
                target.free_slot(slot)
                raise
            self.table.update_placement(page_id, placement, new_pool_slot=slot)
            source.free_slot(descriptor.pool_slot)

    def remove(self, page_id):
        with self._lock:
            descriptor = self.table.get_descriptor(page_id)
            self.pools[(descriptor.codec, descriptor.placement)].free_slot(descriptor.pool_slot)
            self.table.remove_page(page_id)
            self._pinned.discard(page_id)

    def decode(self, query, *, scale=None, query_position=None, sliding_window=None):
        with self._lock:
            try:
                return DirectPagedAttentionEngine.decode_single_token(
                    query, self.table, self.pools, scale=scale, prefetch=True,
                    query_position=query_position, sliding_window=sliding_window,
                )
            finally:
                for pool in self.pools.values():
                    cancel = getattr(pool, "cancel_prefetch", None)
                    if cancel is not None:
                        cancel()

    def clear(self):
        with self._lock:
            for pool in self.pools.values():
                cancel = getattr(pool, "cancel_prefetch", None)
                if cancel is not None:
                    cancel()
            for page_id in self.table.page_ids[:self.table.num_pages].tolist():
                self.remove(page_id)

    def usage(self):
        with self._lock:
            return {
                f"{codec.name}/{placement.name}": {
                    "live_bytes": pool.live_bytes(), "capacity_bytes": pool.capacity_bytes(),
                }
                for (codec, placement), pool in self.pools.items()
            }
