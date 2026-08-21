"""Bounded block I/O and payload arenas for the external gate."""
from __future__ import annotations

import hashlib
import threading
import time
import uuid
from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import Any, Dict, Protocol

from .errors import IntegrityError
from .model_map import WeightBlock
from .telemetry import NullTelemetry, TelemetrySink


class ArenaAllocation(Protocol):
    allocation_id: str
    length: int

    def writable_view(self) -> memoryview:
        ...

    def descriptor(self) -> Dict[str, Any]:
        ...

    def close(self) -> None:
        ...


@dataclass
class HeapAllocation:
    allocation_id: str
    length: int
    _data: bytearray

    def writable_view(self) -> memoryview:
        return memoryview(self._data)

    def descriptor(self) -> Dict[str, Any]:
        return {
            "transport": "heap",
            "allocation_id": self.allocation_id,
            "offset": 0,
            "length": self.length,
        }

    def close(self) -> None:
        self._data.clear()


class SharedMemoryAllocation:
    def __init__(self, allocation_id: str, length: int):
        self.allocation_id = allocation_id
        self.length = int(length)
        self._segment = shared_memory.SharedMemory(create=True, size=self.length)
        self._closed = False

    def writable_view(self) -> memoryview:
        if self._closed:
            raise RuntimeError("shared-memory allocation is closed")
        return self._segment.buf

    def descriptor(self) -> Dict[str, Any]:
        return {
            "transport": "shared_memory",
            "name": self._segment.name,
            "offset": 0,
            "length": self.length,
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._segment.close()
        try:
            self._segment.unlink()
        except FileNotFoundError:
            pass


class PayloadArena:
    """Allocates only the resident payload selected by the physical budget."""

    def __init__(self, transport: str):
        if transport not in {"heap", "shared_memory"}:
            raise ValueError(f"unsupported arena transport: {transport}")
        self.transport = transport

    def allocate(self, length: int) -> ArenaAllocation:
        allocation_id = uuid.uuid4().hex
        if self.transport == "shared_memory":
            return SharedMemoryAllocation(allocation_id, length)
        return HeapAllocation(allocation_id, int(length), bytearray(int(length)))


class FileRangeStore:
    """Reads exactly one manifest range; never materializes an entire shard."""

    def __init__(self, integrity: str = "first_load", telemetry: TelemetrySink | None = None):
        if integrity not in {"always", "first_load", "none"}:
            raise ValueError(f"invalid integrity mode: {integrity}")
        self.integrity = integrity
        self.telemetry = telemetry or NullTelemetry()
        self._verified: set[str] = set()
        self._lock = threading.Lock()
        self._reads = 0
        self._bytes_read = 0

    def _must_verify(self, block: WeightBlock) -> bool:
        if self.integrity == "none":
            return False
        if self.integrity == "always":
            return True
        with self._lock:
            return block.block_id not in self._verified

    def read_into(self, block: WeightBlock, target: memoryview) -> None:
        if target.nbytes != block.length:
            raise ValueError(f"target for {block.block_id!r} has {target.nbytes} bytes, expected {block.length}")
        started = time.perf_counter()
        verify = self._must_verify(block)
        digest = hashlib.sha256() if verify else None
        view = target.cast("B")
        position = 0
        try:
            with block.shard.open("rb", buffering=0) as handle:
                handle.seek(block.offset)
                while position < block.length:
                    end = min(position + (8 << 20), block.length)
                    count = handle.readinto(view[position:end])
                    if not count:
                        raise IntegrityError(
                            f"short read for {block.block_id!r}: {position}/{block.length} bytes"
                        )
                    if digest is not None:
                        digest.update(view[position:position + count])
                    position += count
            if digest is not None:
                actual = digest.hexdigest()
                if actual != block.sha256:
                    raise IntegrityError(
                        f"sha256 mismatch for {block.block_id!r}: expected {block.sha256}, got {actual}"
                    )
                with self._lock:
                    self._verified.add(block.block_id)
            duration_ms = (time.perf_counter() - started) * 1000.0
            with self._lock:
                self._reads += 1
                self._bytes_read += block.length
            self.telemetry.emit(
                "block_read", "ok", duration_ms=duration_ms,
                metadata={"block_id": block.block_id, "bytes": block.length, "verified": verify},
            )
        except Exception as exc:
            self.telemetry.emit(
                "block_read", "error", duration_ms=(time.perf_counter() - started) * 1000.0,
                metadata={"block_id": block.block_id, "error": type(exc).__name__},
            )
            raise
        finally:
            view.release()

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {"reads": self._reads, "bytes_read": self._bytes_read, "verified_blocks": len(self._verified)}

