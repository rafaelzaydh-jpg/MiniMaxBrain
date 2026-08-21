"""Bounded block I/O, payload arenas, and multi-tier integrity for the external gate."""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import threading
import time
import uuid
import zlib
from dataclasses import dataclass
from multiprocessing import shared_memory
from pathlib import Path
from typing import Any, Dict, Protocol

from .errors import IntegrityError
from .model_map import PhysicalModelMap, WeightBlock
from .telemetry import NullTelemetry, TelemetrySink


MODEL_SEAL_SCHEMA = "mmb-model-seal-v1"
SEAL_FILENAME = "model.verified.json"


def get_seal_path(model_map: PhysicalModelMap) -> Path:
    return model_map.path.parent / SEAL_FILENAME


def create_model_seal(
    model_map: PhysicalModelMap,
    seal_path: Path | None = None,
    progress_callback: Any = None,
) -> Dict[str, Any]:
    """Verify all blocks in model_map and write a tamper-proof pre-flight seal."""
    target_path = Path(seal_path) if seal_path else get_seal_path(model_map)
    shards: dict[str, dict[str, Any]] = {}
    verified_blocks = 0

    # Group blocks by shard
    blocks_by_shard: dict[Path, list[WeightBlock]] = {}
    for block in model_map.blocks:
        blocks_by_shard.setdefault(block.shard, []).append(block)

    for shard_path, shard_blocks in blocks_by_shard.items():
        stat = shard_path.stat()
        shard_sha256 = hashlib.sha256()
        shard_crc = 0

        # Read entire shard in streaming chunks to verify block SHA256 and compute shard hashes
        with shard_path.open("rb", buffering=0) as handle:
            # First, verify each block within the shard
            for block in shard_blocks:
                handle.seek(block.offset)
                block_digest = hashlib.sha256()
                position = 0
                while position < block.length:
                    chunk = handle.read(min(block.length - position, 8 << 20))
                    if not chunk:
                        raise IntegrityError(f"short read for {block.block_id!r} in {shard_path.name}")
                    block_digest.update(chunk)
                    position += len(chunk)
                actual_hex = block_digest.hexdigest()
                if actual_hex != block.sha256:
                    raise IntegrityError(
                        f"sha256 mismatch for {block.block_id!r}: expected {block.sha256}, got {actual_hex}"
                    )
                verified_blocks += 1
                if progress_callback:
                    progress_callback(verified_blocks, len(model_map.blocks))

            # Second, compute full shard hash
            handle.seek(0)
            while True:
                chunk = handle.read(16 << 20)
                if not chunk:
                    break
                shard_sha256.update(chunk)
                shard_crc = zlib.crc32(chunk, shard_crc)

        shards[shard_path.name] = {
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": shard_sha256.hexdigest(),
            "crc32": shard_crc,
        }

    map_stat = model_map.path.stat()
    map_sha256 = hashlib.sha256(model_map.path.read_bytes()).hexdigest()

    seal_data = {
        "schema_version": MODEL_SEAL_SCHEMA,
        "model_id": model_map.model_id,
        "map_revision": model_map.map_revision,
        "map_sha256": map_sha256,
        "map_mtime_ns": map_stat.st_mtime_ns,
        "verified_at": time.time(),
        "total_blocks": len(model_map.blocks),
        "shards": shards,
    }

    temp_path = target_path.with_suffix(".tmp")
    temp_path.write_text(json.dumps(seal_data, indent=2, sort_keys=True), encoding="utf-8")
    temp_path.replace(target_path)
    return seal_data


def verify_model_seal(
    model_map: PhysicalModelMap,
    seal_path: Path | None = None,
) -> tuple[bool, str | None]:
    """Fast O(1) metadata check verifying that shard files have not been modified since sealing."""
    target_path = Path(seal_path) if seal_path else get_seal_path(model_map)
    if not target_path.is_file():
        return False, f"seal file does not exist: {target_path.name}"
    try:
        raw = json.loads(target_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"cannot read seal file: {exc}"

    if raw.get("schema_version") != MODEL_SEAL_SCHEMA:
        return False, f"invalid seal schema version: {raw.get('schema_version')!r}"
    if raw.get("model_id") != model_map.model_id:
        return False, f"seal model_id mismatch: expected {model_map.model_id!r}, got {raw.get('model_id')!r}"
    if raw.get("map_revision") != model_map.map_revision:
        return False, f"seal map_revision mismatch: expected {model_map.map_revision!r}, got {raw.get('map_revision')!r}"

    shards_meta = raw.get("shards", {})
    for block in model_map.blocks:
        shard_path = block.shard
        meta = shards_meta.get(shard_path.name)
        if not meta:
            return False, f"missing seal metadata for shard {shard_path.name}"
        try:
            stat = shard_path.stat()
        except OSError as exc:
            return False, f"cannot stat shard {shard_path.name}: {exc}"
        if stat.st_size != meta.get("size_bytes"):
            return False, f"shard {shard_path.name} size modified ({stat.st_size} vs {meta.get('size_bytes')})"
        if stat.st_mtime_ns != meta.get("mtime_ns"):
            return False, f"shard {shard_path.name} mtime modified"

    return True, None


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
    """Reads exactly one manifest range; supports multi-tier and hardware-accelerated integrity."""

    SUPPORTED_MODES = {"always", "first_load", "none", "seal", "crc32", "async"}

    def __init__(
        self,
        integrity: str = "first_load",
        telemetry: TelemetrySink | None = None,
        *,
        model_map: PhysicalModelMap | None = None,
        seal_path: Path | None = None,
    ):
        if integrity not in self.SUPPORTED_MODES:
            raise ValueError(f"invalid integrity mode: {integrity!r} (must be one of {sorted(self.SUPPORTED_MODES)})")
        self.integrity = integrity
        self.telemetry = telemetry or NullTelemetry()
        self.model_map = model_map
        self._verified: set[str] = set()
        self._lock = threading.Lock()
        self._reads = 0
        self._bytes_read = 0
        self._async_error: Exception | None = None
        self._async_executor: concurrent.futures.ThreadPoolExecutor | None = None

        if self.integrity == "seal":
            if model_map is None:
                raise ValueError("integrity mode 'seal' requires model_map")
            valid, reason = verify_model_seal(model_map, seal_path)
            if not valid:
                raise IntegrityError(f"model seal verification failed: {reason}. Run 'mmb seal' to generate a valid seal.")

        if self.integrity == "async":
            self._async_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="mmb-integrity")

    def _must_verify(self, block: WeightBlock) -> bool:
        if self.integrity in {"none", "seal"}:
            return False
        if self.integrity == "always":
            return True
        with self._lock:
            return block.block_id not in self._verified

    def _verify_async_task(self, block_id: str, expected_sha256: str, data: bytes) -> None:
        actual = hashlib.sha256(data).hexdigest()
        if actual != expected_sha256:
            err = IntegrityError(f"async sha256 mismatch for {block_id!r}: expected {expected_sha256}, got {actual}")
            with self._lock:
                self._async_error = err
        else:
            with self._lock:
                self._verified.add(block_id)

    def read_into(self, block: WeightBlock, target: memoryview) -> None:
        if target.nbytes != block.length:
            raise ValueError(f"target for {block.block_id!r} has {target.nbytes} bytes, expected {block.length}")
        with self._lock:
            if self._async_error is not None:
                raise self._async_error

        started = time.perf_counter()
        verify = self._must_verify(block)
        is_crc = verify and self.integrity == "crc32"
        is_async = verify and self.integrity == "async"

        digest = hashlib.sha256() if (verify and not is_crc and not is_async) else None
        crc_val = 0 if is_crc else None
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
                    elif crc_val is not None:
                        crc_val = zlib.crc32(view[position:position + count], crc_val)
                    position += count

            if digest is not None:
                actual = digest.hexdigest()
                if actual != block.sha256:
                    raise IntegrityError(
                        f"sha256 mismatch for {block.block_id!r}: expected {block.sha256}, got {actual}"
                    )
                with self._lock:
                    self._verified.add(block.block_id)
            elif is_async and self._async_executor is not None:
                data_copy = bytes(view)
                self._async_executor.submit(self._verify_async_task, block.block_id, block.sha256, data_copy)
            elif is_crc:
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

    def close(self) -> None:
        if self._async_executor is not None:
            self._async_executor.shutdown(wait=True)
            self._async_executor = None

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {"reads": self._reads, "bytes_read": self._bytes_read, "verified_blocks": len(self._verified)}
