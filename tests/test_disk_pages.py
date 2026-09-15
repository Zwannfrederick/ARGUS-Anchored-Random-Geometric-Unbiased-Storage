"""Disk pages must be exact bytes and usable by the actual attention reader."""

import pytest
import torch

from argus_cache.core.backend_pool import ContiguousBlockPool
from argus_cache.core.direct_attention import DirectPagedAttentionEngine
from argus_cache.core.disk_pool import DiskBlockPool
from argus_cache.core.page_table import CodecKind, PlacementLocation, StructureOfArraysPageTable
from argus_cache.core.page_store import PageStore


def _options():
    return dict(codec=CodecKind.ACTIVE_FP16, max_slots=2, page_size=8, num_heads=1, head_dim=4)


def test_disk_round_trip_prefetch_corruption_and_capacity(tmp_path):
    with DiskBlockPool(directory=tmp_path, **_options()) as pool:
        slot = pool.allocate_slot()
        other = pool.allocate_slot()
        with pytest.raises(RuntimeError, match="out of memory"):
            pool.allocate_slot()
        keys = torch.arange(pool.bytes_per_page, dtype=torch.uint8)
        values = keys.flip(0)
        pool.write_page_bytes(slot, keys, values)
        pool.write_page_bytes(other, values, keys)
        assert pool.prefetch_page_bytes(slot)
        assert not pool.prefetch_page_bytes(other)
        with pytest.raises(RuntimeError, match="prefetch"):
            pool.free_slot(slot)
        with pytest.raises(RuntimeError, match="prefetch"):
            pool.write_page_bytes(slot, values, keys)
        restored = pool.read_page_bytes(slot)
        assert torch.equal(restored[0], keys) and torch.equal(restored[1], values)
        # Damage one payload byte without changing record length.
        with pool._path(slot).open("r+b") as stream:
            stream.write(b"\xff")
        with pytest.raises(OSError, match="checksum"):
            pool.read_page_bytes(slot)
        pool.free_slot(slot)
        assert pool.allocate_slot() == slot
    assert not list(tmp_path.iterdir())
    pool.close()
    with pytest.raises(RuntimeError, match="closed"):
        pool.allocate_slot()


def test_failed_disk_replace_preserves_previous_page(tmp_path, monkeypatch):
    with DiskBlockPool(directory=tmp_path, **_options()) as pool:
        slot = pool.allocate_slot()
        data = torch.zeros(pool.bytes_per_page, dtype=torch.uint8)
        pool.write_page_bytes(slot, data, data)

        def fail(*args):
            raise OSError("disk full")

        monkeypatch.setattr("argus_cache.core.disk_pool.os.replace", fail)
        with pytest.raises(OSError, match="disk full"):
            pool.write_page_bytes(slot, data + 1, data + 1)
        assert torch.equal(pool.read_page_bytes(slot)[0], data)
        assert len(list(pool._path(slot).parent.iterdir())) == 1


def test_attention_reads_same_codec_from_ram_and_disk_with_partial_page(tmp_path):
    torch.manual_seed(7)
    geometry = _options()
    resident = ContiguousBlockPool(placement=PlacementLocation.HOST_PAGEABLE, **geometry)
    with DiskBlockPool(directory=tmp_path, **geometry) as disk:
        table = StructureOfArraysPageTable(capacity=1)
        keys, values = [], []
        for page_id, (pool, count) in enumerate(((resident, 8), (disk, 3))):
            k = torch.randn(1, 1, 8, 4, dtype=torch.float16)
            v = torch.randn_like(k)
            slot = pool.allocate_slot()
            pool.write_page_bytes(slot, k.view(torch.uint8).flatten(), v.view(torch.uint8).flatten())
            table.allocate_page(page_id, page_id * 8, count, pool.codec, pool.placement, slot)
            keys.append(k[:, :, :count])
            values.append(v[:, :, :count])
        query = torch.randn(1, 4, 1, 4)
        actual = DirectPagedAttentionEngine.decode_single_token(
            query, table, {(p.codec, p.placement): p for p in (resident, disk)}
        )
        expected = torch.nn.functional.scaled_dot_product_attention(
            query, torch.cat(keys, dim=2).float(), torch.cat(values, dim=2).float(), enable_gqa=True
        )
        torch.testing.assert_close(actual, expected)
        with pytest.raises(ValueError, match="matching pool"):
            DirectPagedAttentionEngine.decode_single_token(query, table, {resident.codec: resident})


def test_page_and_byte_validation_prevents_aliasing():
    table = StructureOfArraysPageTable(capacity=1)
    table.allocate_page(1, 0, 8)
    with pytest.raises(ValueError, match="already exists"):
        table.allocate_page(1, 8, 8)
    assert table.num_pages == 1
    with pytest.raises(ValueError):
        StructureOfArraysPageTable(capacity=0)
    pool = ContiguousBlockPool(placement=PlacementLocation.HOST_PAGEABLE, **_options())
    slot = pool.allocate_slot()
    oversized = torch.zeros(pool.bytes_per_page + 1, dtype=torch.uint8)
    with pytest.raises(ValueError, match="exactly"):
        pool.write_page_bytes(slot, oversized, oversized)


def test_move_preserves_ownership_on_failure_and_rejects_stale_generation(tmp_path, monkeypatch):
    resident = ContiguousBlockPool(placement=PlacementLocation.HOST_PAGEABLE, **_options())
    with DiskBlockPool(directory=tmp_path, **_options()) as disk:
        store = PageStore([resident, disk])
        data = torch.zeros(resident.bytes_per_page, dtype=torch.uint8)
        store.add(7, 0, 8, resident.codec, resident.placement, data, data)
        before = store.table.get_descriptor(7)
        original = disk.write_page_bytes

        def fail(*args):
            raise OSError("disk full")

        monkeypatch.setattr(disk, "write_page_bytes", fail)
        with pytest.raises(OSError, match="disk full"):
            store.move(7, disk.placement)
        assert store.table.get_descriptor(7) == before
        assert not disk.allocated_slots
        assert len(resident.allocated_slots) == 1
        monkeypatch.setattr(disk, "write_page_bytes", original)
        store.move(7, disk.placement, expected_generation=before.generation)
        assert not resident.allocated_slots
        with pytest.raises(ValueError, match="Stale"):
            store.move(7, resident.placement, expected_generation=before.generation)
        store.move(7, resident.placement)
        store.clear()
        assert store.table.num_pages == 0
        assert not disk.allocated_slots and not resident.allocated_slots


def test_bf16_disk_attention_window_uses_positions_not_descriptor_order(tmp_path):
    geometry = dict(_options(), codec=CodecKind.ACTIVE_BF16, max_slots=3)
    with DiskBlockPool(directory=tmp_path, **geometry) as disk:
        store = PageStore([disk])
        torch.manual_seed(37)
        keys = torch.randn(1, 1, 24, 4, dtype=torch.bfloat16)
        values = torch.randn_like(keys)
        # Compacted tables need not be in logical token order.
        for page in (2, 0, 1):
            k = keys[:, :, page * 8:page * 8 + 8].contiguous()
            v = values[:, :, page * 8:page * 8 + 8].contiguous()
            store.add(page, page * 8, 8, disk.codec, disk.placement,
                      k.view(torch.uint8).flatten(), v.view(torch.uint8).flatten())
        query = torch.randn(1, 4, 1, 4)
        actual = store.decode(query, query_position=12, sliding_window=5)
        mask = (torch.arange(24) <= 12) & (torch.arange(24) > 7)
        expected = torch.nn.functional.scaled_dot_product_attention(
            query, keys.float(), values.float(), attn_mask=mask[None, :], enable_gqa=True,
        )
        torch.testing.assert_close(actual, expected)
        assert disk._pending is None


@pytest.mark.parametrize("codec,payload_size,packed", [
    (CodecKind.GGML_Q8_0, 32, 253), (CodecKind.GGML_Q4_0, 16, 0x55),
])
def test_quantized_disk_pages_are_consumed_without_a_resident_mirror(tmp_path, codec, payload_size, packed):
    with DiskBlockPool(directory=tmp_path, **dict(_options(), codec=codec)) as disk:
        scale = torch.tensor([0.25], dtype=torch.float16).view(torch.uint8)
        block = torch.cat((scale, torch.full((payload_size,), packed, dtype=torch.uint8)))
        store = PageStore([disk])
        store.add(1, 0, 8, codec, disk.placement, block, block)
        output = store.decode(torch.zeros(1, 4, 1, 4))
        torch.testing.assert_close(output, torch.full_like(output, -0.75))
        assert not hasattr(disk, "k_storage")


def test_full_ram_pool_spills_old_pages_but_respects_pinning(tmp_path):
    resident = ContiguousBlockPool(placement=PlacementLocation.HOST_PAGEABLE, **_options())
    with DiskBlockPool(directory=tmp_path, **_options()) as disk:
        store = PageStore([resident, disk])
        data = torch.zeros(resident.bytes_per_page, dtype=torch.uint8)
        for page in range(4):
            store.add(page, page * 8, 8, resident.codec, resident.placement, data, data, pinned=(page == 0))
        assert store.table.get_descriptor(0).placement == resident.placement
        assert store.table.get_descriptor(1).placement == disk.placement
        assert store.table.get_descriptor(2).placement == disk.placement
        assert len(resident.allocated_slots) == 2 and len(disk.allocated_slots) == 2
        with pytest.raises(RuntimeError, match="capacity exhausted"):
            store.add(4, 32, 8, resident.codec, resident.placement, data, data)
        assert store.table.num_pages == 4
        store.clear()


def test_move_from_disk_with_outstanding_prefetch_leaves_no_orphan_slot(tmp_path, monkeypatch):
    resident = ContiguousBlockPool(placement=PlacementLocation.HOST_PAGEABLE, **_options())
    with DiskBlockPool(directory=tmp_path, **_options()) as disk:
        store = PageStore([resident, disk])
        data = torch.arange(disk.bytes_per_page, dtype=torch.uint8)
        store.add(3, 0, 8, disk.codec, disk.placement, data, data)
        slot = store.table.get_descriptor(3).pool_slot
        assert disk.prefetch_page_bytes(slot)
        # Publishing the new placement must not strand the source slot behind the prefetch.
        store.move(3, resident.placement)
        assert store.table.get_descriptor(3).placement == resident.placement
        assert not disk.allocated_slots

        store.move(3, disk.placement)

        def fail(*args, **kwargs):
            raise PermissionError("read-only directory")

        monkeypatch.setattr("pathlib.Path.unlink", fail)
        # A leftover record file is harmless: the slot is rewritten atomically before reuse.
        store.move(3, resident.placement)
        assert not disk.allocated_slots
        monkeypatch.undo()
        store.clear()
        assert not resident.allocated_slots
