"""Bounded block I/O, payload arenas, and fail-closed integrity for the external gate."""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Protocol

from .errors import IntegrityError
from .model_map import PhysicalModelMap, WeightBlock


MODEL_SEAL_SCHEMA = "mmb-model-seal-v2"
SEAL_FILENAME = "model.verified.json"


def get_seal_path(model_map: PhysicalModelMap) -> Path:
    return model_map.path.parent / SEAL_FILENAME


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        while True:
            chunk = handle.read(16 << 20)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)


def _required_auxiliary_files(model_map: PhysicalModelMap) -> dict[str, Path]:
    """Return semantic bundle files that must be bound by the seal."""
    base = model_map.path.parent.resolve()
    result: dict[str, Path] = {}
    layout_path = base / "model.mmb-layout.json"
    if not layout_path.is_file():
        return result

    result["model.mmb-layout.json"] = layout_path
    try:
        layout = json.loads(layout_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IntegrityError(f"cannot read model layout while sealing: {exc}") from exc

    meta = layout.get("metadata_gguf") if isinstance(layout, dict) else None
    if isinstance(meta, dict) and isinstance(meta.get("file_name"), str):
        candidate = (base / meta["file_name"]).resolve()
        try:
            rel = candidate.relative_to(base)
        except ValueError as exc:
            raise IntegrityError("metadata GGUF path escapes model directory") from exc
        if not candidate.is_file():
            raise IntegrityError(f"metadata GGUF does not exist: {meta['file_name']!r}")
        result[rel.as_posix()] = candidate
    return result


def create_model_seal(
    model_map: PhysicalModelMap,
    seal_path: Path | None = None,
    progress_callback: Any = None,
) -> Dict[str, Any]:
    """Verify model bytes and write a reproducible pre-flight integrity seal."""
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

        shards[shard_path.name] = {
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": shard_sha256.hexdigest(),
        }

    map_stat = model_map.path.stat()
    map_sha256 = hashlib.sha256(model_map.path.read_bytes()).hexdigest()

    auxiliary_files: dict[str, dict[str, Any]] = {}
    for relative_name, path in _required_auxiliary_files(model_map).items():
        stat = path.stat()
        auxiliary_files[relative_name] = {
            "size_bytes": stat.st_size,
            "sha256": _sha256_path(path),
        }

    seal_data = {
        "schema_version": MODEL_SEAL_SCHEMA,
        "model_id": model_map.model_id,
        "map_revision": model_map.map_revision,
        "map_sha256": map_sha256,
        "map_mtime_ns": map_stat.st_mtime_ns,
        "verified_at": time.time(),
        "total_blocks": len(model_map.blocks),
        "shards": shards,
        "auxiliary_files": auxiliary_files,
    }

    temp_path = target_path.with_suffix(".tmp")
    temp_path.write_text(json.dumps(seal_data, indent=2, sort_keys=True), encoding="utf-8")
    temp_path.replace(target_path)
    return seal_data


def verify_model_seal(
    model_map: PhysicalModelMap,
    seal_path: Path | None = None,
) -> tuple[bool, str | None]:
    """Cryptographically verify the sealed map and every shard before use."""
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

    try:
        current_map_sha256 = hashlib.sha256(model_map.path.read_bytes()).hexdigest()
    except OSError as exc:
        return False, f"cannot hash model map: {exc}"
    if raw.get("map_sha256") != current_map_sha256:
        return False, "model map sha256 modified since sealing"

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

        expected_sha256 = meta.get("sha256")
        if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
            return False, f"seal is missing a valid sha256 for shard {shard_path.name}"
        digest = hashlib.sha256()
        try:
            with shard_path.open("rb", buffering=0) as handle:
                while True:
                    chunk = handle.read(16 << 20)
                    if not chunk:
                        break
                    digest.update(chunk)
        except OSError as exc:
            return False, f"cannot hash shard {shard_path.name}: {exc}"
        actual_sha256 = digest.hexdigest()
        if actual_sha256 != expected_sha256:
            return False, f"shard {shard_path.name} sha256 modified since sealing"

    try:
        required_aux = _required_auxiliary_files(model_map)
    except IntegrityError as exc:
        return False, str(exc)
    sealed_aux = raw.get("auxiliary_files")
    if not isinstance(sealed_aux, dict):
        return False, "seal is missing auxiliary_files"
    if set(sealed_aux) != set(required_aux):
        return False, "sealed auxiliary file set does not match current model bundle"
    for relative_name, path in required_aux.items():
        meta = sealed_aux.get(relative_name)
        if not isinstance(meta, dict):
            return False, f"invalid seal metadata for auxiliary file {relative_name}"
        try:
            stat = path.stat()
        except OSError as exc:
            return False, f"cannot stat auxiliary file {relative_name}: {exc}"
        if stat.st_size != meta.get("size_bytes"):
            return False, f"auxiliary file {relative_name} size modified since sealing"
        expected = meta.get("sha256")
        if not isinstance(expected, str) or len(expected) != 64:
            return False, f"seal is missing a valid sha256 for auxiliary file {relative_name}"
        try:
            actual = _sha256_path(path)
        except OSError as exc:
            return False, f"cannot hash auxiliary file {relative_name}: {exc}"
        if actual != expected:
            return False, f"auxiliary file {relative_name} sha256 modified since sealing"

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


class PayloadArena:
    """Allocate only resident payload bytes on the local heap."""

    def allocate(self, length: int) -> ArenaAllocation:
        allocation_id = uuid.uuid4().hex
        return HeapAllocation(allocation_id, int(length), bytearray(int(length)))


class FileRangeStore:
    """Read exactly one manifest range under the configured integrity policy."""

    SUPPORTED_MODES = {"always", "first_load", "none", "seal"}

    def __init__(
        self,
        integrity: str = "first_load",
        *,
        model_map: PhysicalModelMap | None = None,
        seal_path: Path | None = None,
    ):
        if integrity not in self.SUPPORTED_MODES:
            raise ValueError(f"invalid integrity mode: {integrity!r} (must be one of {sorted(self.SUPPORTED_MODES)})")
        self.integrity = integrity
        self.model_map = model_map
        self._verified: set[str] = set()
        self._lock = threading.Lock()
        self._reads = 0
        self._bytes_read = 0

        if self.integrity == "seal":
            if model_map is None:
                raise ValueError("integrity mode 'seal' requires model_map")
            valid, reason = verify_model_seal(model_map, seal_path)
            if not valid:
                raise IntegrityError(f"model seal verification failed: {reason}. Run 'mmb seal' to generate a valid seal.")

    def _must_verify(self, block: WeightBlock) -> bool:
        if self.integrity in {"none", "seal"}:
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
        except Exception as exc:
            raise
        finally:
            view.release()

    def close(self) -> None:
        return

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {"reads": self._reads, "bytes_read": self._bytes_read, "verified_blocks": len(self._verified)}
