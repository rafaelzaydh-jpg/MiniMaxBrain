"""Streaming packer for model-specific converters to produce pageable blocks."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, Mapping

from .errors import ManifestError
from .model_map import MODEL_MAP_SCHEMA
from .units import parse_bytes, parse_count


PACK_PLAN_SCHEMA = "mmb-pack-plan-v1"
_ROOT = {"schema_version", "model", "alignment", "shard_size", "blocks"}
_MODEL = {"id", "architecture", "parameter_count", "quantization", "backend_contract", "map_revision"}
_QUANT = {"name", "bits_per_weight"}
_BLOCK = {"id", "kind", "source", "layer", "expert"}


def _reject(value: Mapping[str, Any], allowed: set[str], where: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ManifestError(f"unknown field(s) at {where}: {', '.join(unknown)}")


def _string(value: Mapping[str, Any], key: str, where: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result.strip():
        raise ManifestError(f"{where}.{key} must be a non-empty string")
    return result.strip()


def _source(base: Path, raw: str) -> Path:
    candidate = (base / raw).resolve()
    try:
        candidate.relative_to(base.resolve())
    except ValueError as exc:
        raise ManifestError(f"block source escapes pack-plan directory: {raw!r}") from exc
    if not candidate.is_file():
        raise ManifestError(f"block source does not exist: {raw!r}")
    return candidate


def _copy_block(source: Path, target, digest: hashlib._Hash) -> int:
    total = 0
    with source.open("rb", buffering=0) as handle:
        while True:
            chunk = handle.read(8 << 20)
            if not chunk:
                break
            target.write(chunk)
            digest.update(chunk)
            total += len(chunk)
    return total


def pack_from_plan(plan_path: str | Path, output_dir: str | Path) -> Path:
    plan_file = Path(plan_path).resolve()
    try:
        root = json.loads(plan_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"cannot read pack plan: {exc}") from exc
    if not isinstance(root, dict):
        raise ManifestError("pack-plan root must be an object")
    _reject(root, _ROOT, "root")
    if root.get("schema_version") != PACK_PLAN_SCHEMA:
        raise ManifestError(f"schema_version must be {PACK_PLAN_SCHEMA!r}")
    model = root.get("model")
    if not isinstance(model, dict):
        raise ManifestError("model must be an object")
    _reject(model, _MODEL, "model")
    quant = model.get("quantization")
    if not isinstance(quant, dict):
        raise ManifestError("model.quantization must be an object")
    _reject(quant, _QUANT, "model.quantization")
    bits = quant.get("bits_per_weight")
    if isinstance(bits, bool) or not isinstance(bits, (int, float)) or not 0 < float(bits) <= 32:
        raise ManifestError("model.quantization.bits_per_weight must be in (0, 32]")
    alignment = root.get("alignment", 4096)
    if isinstance(alignment, bool) or not isinstance(alignment, int) or not 1 <= alignment <= (16 << 20):
        raise ManifestError("alignment must be an integer between 1 and 16MiB")
    shard_size = parse_bytes(root.get("shard_size", "64GiB"), field="shard_size")
    if shard_size < alignment:
        raise ManifestError("shard_size must be at least one alignment unit")
    raw_blocks = root.get("blocks")
    if not isinstance(raw_blocks, list) or not raw_blocks:
        raise ManifestError("blocks must be a non-empty array")

    prepared: list[dict[str, Any]] = []
    ids: set[str] = set()
    routes: set[tuple[int, int]] = set()
    for index, raw in enumerate(raw_blocks):
        where = f"blocks[{index}]"
        if not isinstance(raw, dict):
            raise ManifestError(f"{where} must be an object")
        _reject(raw, _BLOCK, where)
        block_id = _string(raw, "id", where)
        if block_id in ids:
            raise ManifestError(f"duplicate block id: {block_id}")
        ids.add(block_id)
        kind = _string(raw, "kind", where)
        if kind not in {"core", "expert"}:
            raise ManifestError(f"{where}.kind must be core or expert")
        item: dict[str, Any] = {
            "id": block_id,
            "kind": kind,
            "source": _source(plan_file.parent, _string(raw, "source", where)),
        }
        if kind == "expert":
            layer, expert = raw.get("layer"), raw.get("expert")
            if isinstance(layer, bool) or not isinstance(layer, int) or layer < 0:
                raise ManifestError(f"{where}.layer must be an integer >= 0")
            if isinstance(expert, bool) or not isinstance(expert, int) or expert < 0:
                raise ManifestError(f"{where}.expert must be an integer >= 0")
            if (layer, expert) in routes:
                raise ManifestError(f"duplicate route layer={layer}, expert={expert}")
            routes.add((layer, expert))
            item.update(layer=layer, expert=expert)
        elif "layer" in raw or "expert" in raw:
            raise ManifestError(f"{where} core block cannot declare layer/expert")
        prepared.append(item)

    output = Path(output_dir).resolve()
    if output.exists():
        raise ManifestError(f"output directory already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(f"{output.name}.partial-{uuid.uuid4().hex}")
    partial.mkdir()
    shards_dir = partial / "shards"
    shards_dir.mkdir()
    manifest_blocks: list[dict[str, Any]] = []
    shard_index = -1
    shard_handle = None
    shard_offset = 0
    try:
        for item in prepared:
            length = item["source"].stat().st_size
            if length <= 0:
                raise ManifestError(f"block source is empty: {item['source']}")
            aligned_offset = ((shard_offset + alignment - 1) // alignment) * alignment
            if shard_handle is None or (aligned_offset + length > shard_size and shard_offset > 0):
                if shard_handle is not None:
                    shard_handle.flush()
                    os.fsync(shard_handle.fileno())
                    shard_handle.close()
                shard_index += 1
                shard_name = f"model-{shard_index:05d}.mmbw"
                shard_handle = (shards_dir / shard_name).open("wb", buffering=0)
                shard_offset = 0
                aligned_offset = 0
            else:
                shard_name = f"model-{shard_index:05d}.mmbw"
            padding = aligned_offset - shard_offset
            if padding:
                shard_handle.write(b"\0" * padding)
            digest = hashlib.sha256()
            actual = _copy_block(item["source"], shard_handle, digest)
            block = {
                "id": item["id"],
                "kind": item["kind"],
                "shard": f"shards/{shard_name}",
                "offset": aligned_offset,
                "length": actual,
                "sha256": digest.hexdigest(),
                "alignment": alignment,
            }
            if item["kind"] == "expert":
                block.update(layer=item["layer"], expert=item["expert"])
            manifest_blocks.append(block)
            shard_offset = aligned_offset + actual
        if shard_handle is not None:
            shard_handle.flush()
            os.fsync(shard_handle.fileno())
            shard_handle.close()
            shard_handle = None
        manifest = {
            "schema_version": MODEL_MAP_SCHEMA,
            "model": {
                "id": _string(model, "id", "model"),
                "architecture": _string(model, "architecture", "model"),
                "parameter_count": parse_count(model.get("parameter_count"), field="model.parameter_count"),
                "quantization": {
                    "name": _string(quant, "name", "model.quantization"),
                    "bits_per_weight": float(bits),
                },
                "backend_contract": _string(model, "backend_contract", "model"),
                "map_revision": _string(model, "map_revision", "model"),
            },
            "blocks": manifest_blocks,
        }
        manifest_path = partial / "model.mmb-map.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(partial, output)
        return output / "model.mmb-map.json"
    except Exception:
        if shard_handle is not None:
            shard_handle.close()
        shutil.rmtree(partial, ignore_errors=True)
        raise
