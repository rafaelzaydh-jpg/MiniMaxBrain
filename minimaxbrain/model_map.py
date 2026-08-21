"""Strict physical map for independently pageable model blocks.

The external gate deliberately knows physical identity only. Semantic topology,
hidden states and router interpretation belong to the future internal gate.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Mapping, Sequence

from .errors import ManifestError, UnknownBlockError


MODEL_MAP_SCHEMA = "mmb-physical-model-map-v1"
_ROOT_FIELDS = {"schema_version", "model", "blocks"}
_MODEL_FIELDS = {
    "id", "architecture", "parameter_count", "quantization", "backend_contract", "map_revision"
}
_QUANTIZATION_FIELDS = {"name", "bits_per_weight"}
_BLOCK_FIELDS = {
    "id", "kind", "shard", "offset", "length", "sha256", "layer", "expert", "alignment"
}


def _object(value: Any, where: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ManifestError(f"{where} must be an object")
    return value


def _reject_unknown(value: Mapping[str, Any], allowed: set[str], where: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ManifestError(f"unknown field(s) at {where}: {', '.join(unknown)}")


def _required_string(value: Mapping[str, Any], key: str, where: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result.strip():
        raise ManifestError(f"{where}.{key} must be a non-empty string")
    return result.strip()


def _required_int(value: Mapping[str, Any], key: str, where: str, *, minimum: int = 0) -> int:
    result = value.get(key)
    if isinstance(result, bool) or not isinstance(result, int) or result < minimum:
        raise ManifestError(f"{where}.{key} must be an integer >= {minimum}")
    return result


@dataclass(frozen=True)
class WeightBlock:
    block_id: str
    kind: str
    shard: Path
    shard_name: str
    offset: int
    length: int
    sha256: str
    layer: int | None = None
    expert: int | None = None
    alignment: int = 1

    @property
    def route_key(self) -> tuple[int, int] | None:
        if self.layer is None or self.expert is None:
            return None
        return self.layer, self.expert


@dataclass(frozen=True)
class PhysicalModelMap:
    path: Path
    model_id: str
    architecture: str
    parameter_count: int
    quantization_name: str
    bits_per_weight: float
    backend_contract: str
    map_revision: str
    blocks: tuple[WeightBlock, ...]
    by_id: Mapping[str, WeightBlock]
    by_route: Mapping[tuple[int, int], WeightBlock]

    @property
    def core_blocks(self) -> tuple[WeightBlock, ...]:
        return tuple(block for block in self.blocks if block.kind == "core")

    @property
    def expert_blocks(self) -> tuple[WeightBlock, ...]:
        return tuple(block for block in self.blocks if block.kind == "expert")

    @property
    def stored_bytes(self) -> int:
        return sum(block.length for block in self.blocks)

    @property
    def core_bytes(self) -> int:
        return sum(block.length for block in self.core_blocks)

    @property
    def largest_expert_bytes(self) -> int:
        return max((block.length for block in self.expert_blocks), default=0)

    @property
    def identity(self) -> str:
        body = f"{self.model_id}\0{self.map_revision}\0{self.path.resolve()}".encode("utf-8")
        return hashlib.sha256(body).hexdigest()

    def block(self, block_id: str) -> WeightBlock:
        try:
            return self.by_id[str(block_id)]
        except KeyError as exc:
            raise UnknownBlockError(f"block {block_id!r} is not present in model map") from exc

    def route_block(self, layer: int, expert: int) -> WeightBlock:
        try:
            return self.by_route[(int(layer), int(expert))]
        except KeyError as exc:
            raise UnknownBlockError(f"no physical block for layer={layer}, expert={expert}") from exc


def _resolve_shard(base: Path, raw: str) -> tuple[Path, str]:
    shard_name = raw.replace("\\", "/")
    candidate = (base / shard_name).resolve()
    try:
        candidate.relative_to(base.resolve())
    except ValueError as exc:
        raise ManifestError(f"shard escapes model directory: {raw!r}") from exc
    if not candidate.is_file():
        raise ManifestError(f"shard does not exist: {raw!r}")
    return candidate, shard_name


def load_model_map(path: str | Path) -> PhysicalModelMap:
    manifest_path = Path(path).resolve()
    if not manifest_path.is_file():
        raise ManifestError(f"model map does not exist: {manifest_path}")
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"cannot read model map: {exc}") from exc
    root = _object(raw, "root")
    _reject_unknown(root, _ROOT_FIELDS, "root")
    if root.get("schema_version") != MODEL_MAP_SCHEMA:
        raise ManifestError(f"schema_version must be {MODEL_MAP_SCHEMA!r}")

    model = _object(root.get("model"), "model")
    _reject_unknown(model, _MODEL_FIELDS, "model")
    quant = _object(model.get("quantization"), "model.quantization")
    _reject_unknown(quant, _QUANTIZATION_FIELDS, "model.quantization")
    bits = quant.get("bits_per_weight")
    if isinstance(bits, bool) or not isinstance(bits, (int, float)) or not 0 < float(bits) <= 32:
        raise ManifestError("model.quantization.bits_per_weight must be in (0, 32]")

    raw_blocks = root.get("blocks")
    if not isinstance(raw_blocks, list) or not raw_blocks:
        raise ManifestError("blocks must be a non-empty array")
    blocks: list[WeightBlock] = []
    ids: set[str] = set()
    routes: set[tuple[int, int]] = set()
    ranges: dict[Path, list[tuple[int, int, str]]] = {}
    base = manifest_path.parent
    for index, item in enumerate(raw_blocks):
        where = f"blocks[{index}]"
        block_raw = _object(item, where)
        _reject_unknown(block_raw, _BLOCK_FIELDS, where)
        block_id = _required_string(block_raw, "id", where)
        if block_id in ids:
            raise ManifestError(f"duplicate block id: {block_id}")
        ids.add(block_id)
        kind = _required_string(block_raw, "kind", where)
        if kind not in {"core", "expert"}:
            raise ManifestError(f"{where}.kind must be 'core' or 'expert'")
        shard, shard_name = _resolve_shard(base, _required_string(block_raw, "shard", where))
        offset = _required_int(block_raw, "offset", where)
        length = _required_int(block_raw, "length", where, minimum=1)
        alignment = block_raw.get("alignment", 1)
        if isinstance(alignment, bool) or not isinstance(alignment, int) or alignment < 1:
            raise ManifestError(f"{where}.alignment must be an integer >= 1")
        if offset % alignment:
            raise ManifestError(f"{where}.offset is not aligned to {alignment} bytes")
        digest = _required_string(block_raw, "sha256", where).lower()
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise ManifestError(f"{where}.sha256 must be 64 lowercase hexadecimal characters")
        layer = block_raw.get("layer")
        expert = block_raw.get("expert")
        if kind == "expert":
            if isinstance(layer, bool) or not isinstance(layer, int) or layer < 0:
                raise ManifestError(f"{where}.layer must be an integer >= 0 for expert blocks")
            if isinstance(expert, bool) or not isinstance(expert, int) or expert < 0:
                raise ManifestError(f"{where}.expert must be an integer >= 0 for expert blocks")
            route = (layer, expert)
            if route in routes:
                raise ManifestError(f"duplicate physical route layer={layer}, expert={expert}")
            routes.add(route)
        elif layer is not None or expert is not None:
            raise ManifestError(f"{where} core block cannot declare layer/expert")
        if offset + length > shard.stat().st_size:
            raise ManifestError(f"{where} extends beyond shard {shard_name!r}")
        block = WeightBlock(
            block_id=block_id,
            kind=kind,
            shard=shard,
            shard_name=shard_name,
            offset=offset,
            length=length,
            sha256=digest,
            layer=layer,
            expert=expert,
            alignment=alignment,
        )
        blocks.append(block)
        ranges.setdefault(shard, []).append((offset, offset + length, block_id))

    for shard, shard_ranges in ranges.items():
        ordered = sorted(shard_ranges)
        for previous, current in zip(ordered, ordered[1:]):
            if current[0] < previous[1]:
                raise ManifestError(
                    f"overlapping blocks in {shard.name!r}: {previous[2]!r} and {current[2]!r}"
                )

    by_id = MappingProxyType({block.block_id: block for block in blocks})
    by_route = MappingProxyType({block.route_key: block for block in blocks if block.route_key is not None})
    return PhysicalModelMap(
        path=manifest_path,
        model_id=_required_string(model, "id", "model"),
        architecture=_required_string(model, "architecture", "model"),
        parameter_count=_required_int(model, "parameter_count", "model", minimum=1),
        quantization_name=_required_string(quant, "name", "model.quantization"),
        bits_per_weight=float(bits),
        backend_contract=_required_string(model, "backend_contract", "model"),
        map_revision=_required_string(model, "map_revision", "model"),
        blocks=tuple(blocks),
        by_id=by_id,
        by_route=by_route,
    )


def public_map_summary(model_map: PhysicalModelMap) -> Dict[str, Any]:
    layers = sorted({block.layer for block in model_map.expert_blocks if block.layer is not None})
    return {
        "schema_version": MODEL_MAP_SCHEMA,
        "model_id": model_map.model_id,
        "map_revision": model_map.map_revision,
        "architecture": model_map.architecture,
        "parameter_count": model_map.parameter_count,
        "quantization": {
            "name": model_map.quantization_name,
            "bits_per_weight": model_map.bits_per_weight,
        },
        "backend_contract": model_map.backend_contract,
        "blocks": len(model_map.blocks),
        "core_blocks": len(model_map.core_blocks),
        "expert_blocks": len(model_map.expert_blocks),
        "layers": len(layers),
        "stored_bytes": model_map.stored_bytes,
        "core_bytes": model_map.core_bytes,
        "largest_expert_bytes": model_map.largest_expert_bytes,
        "identity": model_map.identity,
    }

