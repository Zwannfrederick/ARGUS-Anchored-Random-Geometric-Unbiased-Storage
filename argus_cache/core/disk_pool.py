"""Fixed-capacity compressed page storage with atomic replacement and one-page prefetch.

Files are temporary inference state, not a durable checkpoint. The directory may
be on NVMe, but this class makes no claim about the underlying storage hardware.
"""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import os
from pathlib import Path
import tempfile
import threading

import torch

from .backend_pool import ContiguousBlockPool
from .page_table import PlacementLocation


class DiskBlockPool(ContiguousBlockPool):
    """Same byte-page contract as a resident pool; no context-sized RAM allocation.

    Capacity includes a checksum per slot and one replacement scratch record.
    Host staging is bounded to one prefetched record plus the current read/write.
    """

    def __init__(self, *, directory, **kwargs):
        self._directory_parent = directory
        self._lock = threading.RLock()
        self._pending = None
        self._closed = False
        super().__init__(placement=PlacementLocation.DISK, **kwargs)
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="argus-disk")

    def _allocate_storage(self):
        self._directory = tempfile.TemporaryDirectory(prefix="argus-kv-", dir=self._directory_parent)
        self.record_bytes = self.bytes_per_page * 2 + 32

    def _path(self, slot):
        return Path(self._directory.name) / str(slot)

    def _validate_slot(self, slot):
        if self._closed:
            raise RuntimeError("Disk pool is closed")
        super()._validate_slot(slot)

    def allocate_slot(self):
        with self._lock:
            if self._closed:
                raise RuntimeError("Disk pool is closed")
            return super().allocate_slot()

    def _require_idle_slot(self, slot):
        if self._pending is not None and self._pending[0] == slot:
            raise RuntimeError("Cannot mutate a page with an outstanding prefetch")

    def write_page_bytes(self, slot, k_bytes, v_bytes):
        with self._lock:
            self._validate_write(slot, k_bytes, v_bytes)
            self._require_idle_slot(slot)
            digest = hashlib.sha256()
            scratch = None
            try:
                with tempfile.NamedTemporaryFile(dir=self._directory.name, delete=False) as stream:
                    scratch = stream.name
                    for data in (k_bytes, v_bytes):
                        payload = data.detach().to("cpu").contiguous().numpy().tobytes()
                        digest.update(payload)
                        if stream.write(payload) != len(payload):
                            raise OSError("Short KV page write")
                    checksum = digest.digest()
                    if stream.write(checksum) != len(checksum):
                        raise OSError("Short KV checksum write")
                # The old record survives all failures before publication.
                os.replace(scratch, self._path(slot))
            finally:
                if scratch is not None and os.path.exists(scratch):
                    os.unlink(scratch)

    def _read_record(self, slot):
        with self._path(slot).open("rb") as stream:
            raw = bytearray(stream.read(self.record_bytes + 1))
        if len(raw) != self.record_bytes:
            raise OSError("Truncated or oversized KV page")
        payload = memoryview(raw)[:-32]
        if hashlib.sha256(payload).digest() != raw[-32:]:
            raise OSError("KV page checksum mismatch")
        data = torch.frombuffer(payload, dtype=torch.uint8)
        return data[:self.bytes_per_page], data[self.bytes_per_page:]

    def prefetch_page_bytes(self, slot):
        """Reserve one slot until read/cancel; return False when the queue is full."""
        with self._lock:
            self._validate_slot(slot)
            if self._pending is not None:
                return self._pending[0] == slot
            self._pending = (slot, self._worker.submit(self._read_record, slot))
            return True

    def read_page_bytes(self, slot):
        with self._lock:
            self._validate_slot(slot)
            if self._pending is not None and self._pending[0] == slot:
                _, future = self._pending
                try:
                    return future.result()
                finally:
                    self._pending = None
            return self._read_record(slot)

    def cancel_prefetch(self):
        with self._lock:
            if self._pending is not None:
                _, future = self._pending
                future.cancel()
                # Running I/O must finish before its slot may be reused.
                if not future.cancelled():
                    try:
                        future.result()
                    finally:
                        self._pending = None
                else:
                    self._pending = None

    def free_slot(self, slot):
        with self._lock:
            self._require_idle_slot(slot)
            if slot in self.allocated_slots:
                try:
                    self._path(slot).unlink(missing_ok=True)
                except OSError:
                    # A stale record is harmless (the next write replaces it atomically);
                    # an orphaned slot would permanently shrink capacity.
                    pass
                super().free_slot(slot)

    def capacity_bytes(self):
        return (self.max_slots + 1) * self.record_bytes

    def live_bytes(self):
        return len(self.allocated_slots) * self.record_bytes

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._worker.shutdown(wait=True)
            self._pending = None
            self._directory.cleanup()
            self.allocated_slots.clear()
            self.free_slots.clear()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
