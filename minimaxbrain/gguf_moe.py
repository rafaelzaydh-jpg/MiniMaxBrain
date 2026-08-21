"""Streaming GGUF MoE adapter for the MiniMaxBrain physical paging contract.

The adapter accepts GGUF models whose expert weights use the fused
``blk.N.ffn_*_exps.weight`` layout.  Supported architectures include
GraniteMoE, Qwen 3.5/3/2 MoE, and any future architecture that follows
the same naming convention.  It copies raw GGML-encoded slices; it does
not decode, dequantize, route, or execute them.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import uuid
from pathlib import Path
from typing import Any, BinaryIO

from .errors import ManifestError
from .gguf import GGUFModel, GGUFTensor, load_gguf
from .model_map import MODEL_MAP_SCHEMA, PhysicalModelMap


GGUF_MOE_LAYOUT_SCHEMA = "mmb-gguf-moe-layout-v2"
GGUF_MOE_BACKEND_CONTRACT = "mmb-raw-ggml-expert-segments-v2"
_EXPERT_NAME = re.compile(r"^blk\.(\d+)\.ffn_(down|gate|up)_exps\.weight$")
_ROLES = ("down", "gate", "up")

# Maps GGUF architecture strings to the metadata key prefix used for
# block_count, expert_count, and expert_used_count.  Every architecture
# listed here must use the ``blk.N.ffn_{down,gate,up}_exps.weight``
# fused expert tensor layout.
_MOE_ARCHITECTURES: dict[str, str] = {
    "granitemoe": "granitemoe",
    "qwen35moe":  "qwen35moe",
    "qwen3moe":   "qwen3moe",
    "qwen2moe":   "qwen2moe",
}


def _required_metadata_int(model: GGUFModel, key: str, *, minimum: int = 1) -> int:
    value = model.metadata.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ManifestError(f"GGUF metadata {key!r} must be an integer >= {minimum}")
    return value


def _align(handle: BinaryIO, alignment: int) -> int:
    position = handle.tell()
    aligned = ((position + alignment - 1) // alignment) * alignment
    padding = aligned - position
    if padding:
        handle.write(b"\0" * padding)
    return aligned


def _copy_range(
    source: BinaryIO,
    target: BinaryIO,
    *,
    offset: int,
    length: int,
    digest: hashlib._Hash,
) -> None:
    source.seek(offset)
    remaining = length
    while remaining:
        chunk = source.read(min(8 << 20, remaining))
        if not chunk:
            raise ManifestError(f"short GGUF read at offset {offset}: {length - remaining}/{length}")
        target.write(chunk)
        digest.update(chunk)
        remaining -= len(chunk)


def _source_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        while True:
            chunk = handle.read(8 << 20)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)


def _expert_groups(
    model: GGUFModel,
    *,
    layer_count: int,
    expert_count: int,
) -> dict[int, dict[str, GGUFTensor]]:
    groups: dict[int, dict[str, GGUFTensor]] = {}
    for tensor in model.tensors:
        match = _EXPERT_NAME.fullmatch(tensor.name)
        if match is None:
            continue
        layer, role = int(match.group(1)), match.group(2)
        if layer >= layer_count:
            raise ManifestError(f"expert tensor {tensor.name!r} exceeds declared layer count")
        if role in groups.setdefault(layer, {}):
            raise ManifestError(f"duplicate {role} expert tensor in layer {layer}")
        if tensor.dimensions[-1] != expert_count:
            raise ManifestError(
                f"expert axis for {tensor.name!r} is {tensor.dimensions[-1]}, expected {expert_count}"
            )
        if tensor.payload_length is None or tensor.payload_length % expert_count:
            raise ManifestError(f"cannot split encoded expert tensor {tensor.name!r} evenly")
        groups[layer][role] = tensor
    expected = set(_ROLES)
    for layer in range(layer_count):
        actual = set(groups.get(layer, {}))
        if actual != expected:
            missing = ", ".join(sorted(expected - actual)) or "none"
            extra = ", ".join(sorted(actual - expected)) or "none"
            raise ManifestError(f"layer {layer} has invalid expert tensors; missing={missing}, extra={extra}")
    return groups


def pack_moe_gguf(
    gguf_path: str | Path,
    output_dir: str | Path,
    *,
    alignment: int = 4096,
) -> Path:
    """Split fused MoE expert tensors into one pageable block per expert.

    Supports any GGUF architecture registered in ``_MOE_ARCHITECTURES``
    that uses the ``blk.N.ffn_{down,gate,up}_exps.weight`` layout.

    The output is a self-contained MiniMaxBrain model directory.  Core tensors
    are individual pinned blocks and every expert block concatenates its raw
    ``down``, ``gate`` and ``up`` encoded slices in that documented order.
    """
    if isinstance(alignment, bool) or not isinstance(alignment, int) or not 1 <= alignment <= (16 << 20):
        raise ManifestError("alignment must be an integer between 1 and 16MiB")
    model = load_gguf(gguf_path)
    meta_prefix = _MOE_ARCHITECTURES.get(model.architecture)
    if meta_prefix is None:
        supported = ", ".join(sorted(_MOE_ARCHITECTURES))
        raise ManifestError(
            f"GGUF architecture {model.architecture!r} is not a supported MoE layout; "
            f"supported: {supported}"
        )
    layer_count = _required_metadata_int(model, f"{meta_prefix}.block_count")
    expert_count = _required_metadata_int(model, f"{meta_prefix}.expert_count")
    active_experts = _required_metadata_int(model, f"{meta_prefix}.expert_used_count")
    if active_experts > expert_count:
        raise ManifestError(f"{meta_prefix}.expert_used_count exceeds expert_count")
    groups = _expert_groups(model, layer_count=layer_count, expert_count=expert_count)
    expert_names = {tensor.name for roles in groups.values() for tensor in roles.values()}
    for tensor in model.tensors:
        if "_exps" in tensor.name and tensor.name not in expert_names:
            raise ManifestError(f"unsupported fused expert tensor: {tensor.name!r}")
        if tensor.payload_length is None:
            raise ManifestError(f"cannot determine encoded payload length for {tensor.name!r}")

    output = Path(output_dir).resolve()
    if output.exists():
        raise ManifestError(f"output directory already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(f"{output.name}.partial-{uuid.uuid4().hex}")
    partial.mkdir()
    shard_name = "model-00000.mmbw"
    shard_path = partial / shard_name
    blocks: list[dict[str, Any]] = []
    core_layout: list[dict[str, Any]] = []
    layer_layout: list[dict[str, Any]] = []
    source_hash = _source_sha256(model.path)
    metadata_name = "model.mmb-meta.gguf"
    metadata_path = partial / metadata_name
    try:
        # Preserve the GGUF metadata/tensor directory without any weight payload.
        # This is sufficient for llama_model_init_from_user/gguf metadata parsing
        # and avoids duplicating the original model weights in the MMB bundle.
        with model.path.open("rb", buffering=0) as source_meta, metadata_path.open("wb", buffering=0) as meta_out:
            remaining = model.tensor_data_offset
            meta_digest = hashlib.sha256()
            while remaining:
                chunk = source_meta.read(min(8 << 20, remaining))
                if not chunk:
                    raise ManifestError("short GGUF read while creating metadata-only GGUF")
                meta_out.write(chunk)
                meta_digest.update(chunk)
                remaining -= len(chunk)
            meta_out.flush()
            os.fsync(meta_out.fileno())
        metadata_hash = meta_digest.hexdigest()

        with model.path.open("rb", buffering=0) as source, shard_path.open("wb", buffering=0) as target:
            for tensor in model.tensors:
                if tensor.name in expert_names:
                    continue
                assert tensor.payload_length is not None
                offset = _align(target, alignment)
                digest = hashlib.sha256()
                _copy_range(
                    source,
                    target,
                    offset=tensor.absolute_offset,
                    length=tensor.payload_length,
                    digest=digest,
                )
                block_id = f"core/{tensor.name}"
                blocks.append({
                    "id": block_id,
                    "kind": "core",
                    "shard": shard_name,
                    "offset": offset,
                    "length": tensor.payload_length,
                    "sha256": digest.hexdigest(),
                    "alignment": alignment,
                })
                core_layout.append({
                    "block_id": block_id,
                    "tensor": tensor.name,
                    "shape": list(tensor.dimensions),
                    "ggml_type": tensor.ggml_type,
                    "encoded_length": tensor.payload_length,
                })

            for layer in range(layer_count):
                roles = groups[layer]
                segments: list[dict[str, Any]] = []
                segment_offset = 0
                for role in _ROLES:
                    tensor = roles[role]
                    assert tensor.payload_length is not None
                    length = tensor.payload_length // expert_count
                    segments.append({
                        "role": role,
                        "tensor": tensor.name,
                        "offset": segment_offset,
                        "encoded_length": length,
                        "ggml_type": tensor.ggml_type,
                        "expert_shape": list(tensor.dimensions[:-1]),
                        "source_tensor_shape": list(tensor.dimensions),
                        "expert_axis": len(tensor.dimensions) - 1,
                        "expert_index_stride_bytes": length,
                    })
                    segment_offset += length
                layer_layout.append({
                    "layer": layer,
                    "block_id_pattern": f"expert/{layer}/{{expert}}",
                    "encoded_block_length": segment_offset,
                    "segments": segments,
                })
                for expert in range(expert_count):
                    offset = _align(target, alignment)
                    digest = hashlib.sha256()
                    for role in _ROLES:
                        tensor = roles[role]
                        assert tensor.payload_length is not None
                        slice_length = tensor.payload_length // expert_count
                        _copy_range(
                            source,
                            target,
                            offset=tensor.absolute_offset + expert * slice_length,
                            length=slice_length,
                            digest=digest,
                        )
                    blocks.append({
                        "id": f"expert/{layer}/{expert}",
                        "kind": "expert",
                        "shard": shard_name,
                        "offset": offset,
                        "length": segment_offset,
                        "sha256": digest.hexdigest(),
                        "alignment": alignment,
                        "layer": layer,
                        "expert": expert,
                    })
            target.flush()
            os.fsync(target.fileno())

        parameter_count = sum(tensor.elements for tensor in model.tensors)
        payload_bytes = sum(int(tensor.payload_length or 0) for tensor in model.tensors)
        bits_per_weight = payload_bytes * 8.0 / parameter_count
        manifest = {
            "schema_version": MODEL_MAP_SCHEMA,
            "model": {
                "id": str(model.metadata.get("general.name") or model.path.stem),
                "architecture": model.architecture,
                "parameter_count": parameter_count,
                "quantization": {
                    "name": f"GGUF-ftype-{model.metadata.get('general.file_type', 'unknown')}",
                    "bits_per_weight": round(bits_per_weight, 6),
                },
                "backend_contract": GGUF_MOE_BACKEND_CONTRACT,
                "map_revision": f"sha256:{source_hash}",
            },
            "blocks": blocks,
        }
        layout = {
            "schema_version": GGUF_MOE_LAYOUT_SCHEMA,
            "source": {
                "file_name": model.path.name,
                "size_bytes": model.path.stat().st_size,
                "sha256": source_hash,
                "gguf_version": model.version,
                "architecture": model.architecture,
            },
            "backend_contract": GGUF_MOE_BACKEND_CONTRACT,
            "metadata_gguf": {
                "file_name": metadata_name,
                "size_bytes": model.tensor_data_offset,
                "sha256": metadata_hash,
            },
            "expert_axis": -1,
            "expert_storage": "one contiguous block per (layer, expert), segmented as down/gate/up",
            "expert_count": expert_count,
            "active_experts_per_token": active_experts,
            "layer_count": layer_count,
            "segment_order": list(_ROLES),
            "encoding": "raw GGML bytes; no dequantization",
            "core_tensors": core_layout,
            "layers": layer_layout,
        }
        (partial / "model.mmb-map.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (partial / "model.mmb-layout.json").write_text(
            json.dumps(layout, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(partial, output)
        return output / "model.mmb-map.json"
    except Exception:
        shutil.rmtree(partial, ignore_errors=True)
        raise


def validate_moe_layout(model_map: PhysicalModelMap) -> dict[str, Any]:
    """Validate the semantic MoE layout against the physical model map.

    This is the portable validator used by ``mmb check``.  The native backend
    independently validates the same invariants before admitting a model.
    """
    layout_path = model_map.path.parent / "model.mmb-layout.json"
    try:
        raw = json.loads(layout_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ManifestError("model.mmb-layout.json does not exist") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"cannot read model.mmb-layout.json: {exc}") from exc
    if not isinstance(raw, dict):
        raise ManifestError("model.mmb-layout.json root must be an object")

    schema = raw.get("schema_version")
    if schema not in {"mmb-gguf-moe-layout-v1", GGUF_MOE_LAYOUT_SCHEMA}:
        raise ManifestError(f"unsupported MoE layout schema: {schema!r}")
    if raw.get("backend_contract") != model_map.backend_contract:
        raise ManifestError("layout/backend contract mismatch")

    def _positive_int(value: Any, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ManifestError(f"{name} must be an integer >= 1")
        return value

    layer_count = _positive_int(raw.get("layer_count"), "layout.layer_count")
    expert_count = _positive_int(raw.get("expert_count"), "layout.expert_count")
    active = _positive_int(raw.get("active_experts_per_token"), "layout.active_experts_per_token")
    if active > expert_count:
        raise ManifestError("layout.active_experts_per_token cannot exceed expert_count")

    expected_routes = layer_count * expert_count
    if len(model_map.expert_blocks) != expected_routes:
        raise ManifestError(
            f"expert route cardinality mismatch: expected {expected_routes}, "
            f"found {len(model_map.expert_blocks)}"
        )

    expected_route_set = {
        (layer, expert)
        for layer in range(layer_count)
        for expert in range(expert_count)
    }
    actual_route_set = set(model_map.by_route)
    if actual_route_set != expected_route_set:
        missing = sorted(expected_route_set - actual_route_set)[:8]
        extra = sorted(actual_route_set - expected_route_set)[:8]
        raise ManifestError(f"expert route topology mismatch: missing={missing}, extra={extra}")

    metadata_summary: dict[str, Any] | None = None
    if schema == GGUF_MOE_LAYOUT_SCHEMA:
        meta = raw.get("metadata_gguf")
        if not isinstance(meta, dict):
            raise ManifestError("layout.metadata_gguf must be an object")
        name = meta.get("file_name")
        size = meta.get("size_bytes")
        digest = meta.get("sha256")
        if not isinstance(name, str) or not name:
            raise ManifestError("layout.metadata_gguf.file_name must be a non-empty string")
        if isinstance(size, bool) or not isinstance(size, int) or size < 1:
            raise ManifestError("layout.metadata_gguf.size_bytes must be an integer >= 1")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(ch not in "0123456789abcdef" for ch in digest)
        ):
            raise ManifestError("layout.metadata_gguf.sha256 must be lowercase SHA-256")

        base = model_map.path.parent.resolve()
        meta_path = (base / name).resolve()
        try:
            meta_path.relative_to(base)
        except ValueError as exc:
            raise ManifestError("metadata GGUF path escapes model directory") from exc
        if not meta_path.is_file():
            raise ManifestError(f"metadata GGUF does not exist: {name!r}")
        if meta_path.stat().st_size != size:
            raise ManifestError("metadata GGUF size mismatch")
        actual = _source_sha256(meta_path)
        if actual != digest:
            raise ManifestError("metadata GGUF SHA-256 mismatch")
        metadata_summary = {"file_name": name, "size_bytes": size, "sha256": digest}

    core_items = raw.get("core_tensors")
    if not isinstance(core_items, list):
        raise ManifestError("layout.core_tensors must be an array")
    core_by_id = {block.block_id: block for block in model_map.core_blocks}
    seen_core_ids: set[str] = set()
    for index, item in enumerate(core_items):
        where = f"layout.core_tensors[{index}]"
        if not isinstance(item, dict):
            raise ManifestError(f"{where} must be an object")
        block_id = item.get("block_id")
        tensor = item.get("tensor")
        encoded = item.get("encoded_length")
        shape = item.get("shape")
        ggml_type = item.get("ggml_type")
        if not isinstance(block_id, str) or block_id not in core_by_id:
            raise ManifestError(f"{where}.block_id does not reference a core block")
        if block_id in seen_core_ids:
            raise ManifestError(f"duplicate core layout block_id: {block_id}")
        seen_core_ids.add(block_id)
        if not isinstance(tensor, str) or not tensor:
            raise ManifestError(f"{where}.tensor must be a non-empty string")
        if isinstance(encoded, bool) or not isinstance(encoded, int) or encoded < 1:
            raise ManifestError(f"{where}.encoded_length must be an integer >= 1")
        if encoded != core_by_id[block_id].length:
            raise ManifestError(f"{where}.encoded_length does not match physical block")
        if (
            not isinstance(shape, list)
            or not 1 <= len(shape) <= 4
            or any(isinstance(dim, bool) or not isinstance(dim, int) or dim < 1 for dim in shape)
        ):
            raise ManifestError(f"{where}.shape must contain 1..4 positive integer dimensions")
        if isinstance(ggml_type, bool) or not isinstance(ggml_type, int) or ggml_type < 0:
            raise ManifestError(f"{where}.ggml_type must be an integer >= 0")
    if seen_core_ids != set(core_by_id):
        missing = sorted(set(core_by_id) - seen_core_ids)[:8]
        raise ManifestError(f"layout is missing core tensor blocks: {missing}")

    layers = raw.get("layers")
    if not isinstance(layers, list) or len(layers) != layer_count:
        raise ManifestError("layout.layers cardinality does not match layer_count")
    seen_layers: set[int] = set()
    for index, item in enumerate(layers):
        where = f"layout.layers[{index}]"
        if not isinstance(item, dict):
            raise ManifestError(f"{where} must be an object")
        layer = item.get("layer")
        encoded_block_length = item.get("encoded_block_length")
        segments = item.get("segments")
        if isinstance(layer, bool) or not isinstance(layer, int) or not 0 <= layer < layer_count:
            raise ManifestError(f"{where}.layer is invalid")
        if layer in seen_layers:
            raise ManifestError(f"duplicate layer layout: {layer}")
        seen_layers.add(layer)
        if (
            isinstance(encoded_block_length, bool)
            or not isinstance(encoded_block_length, int)
            or encoded_block_length < 1
        ):
            raise ManifestError(f"{where}.encoded_block_length must be an integer >= 1")
        if not isinstance(segments, list) or len(segments) != 3:
            raise ManifestError(f"{where}.segments must contain exactly down/gate/up")

        ranges: list[tuple[int, int, str]] = []
        roles: set[str] = set()
        for seg_index, segment in enumerate(segments):
            seg_where = f"{where}.segments[{seg_index}]"
            if not isinstance(segment, dict):
                raise ManifestError(f"{seg_where} must be an object")
            role = segment.get("role")
            offset = segment.get("offset")
            length = segment.get("encoded_length")
            shape = segment.get("expert_shape")
            ggml_type = segment.get("ggml_type")
            if role not in _ROLES or role in roles:
                raise ManifestError(f"{seg_where}.role is invalid or duplicated")
            roles.add(role)
            if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
                raise ManifestError(f"{seg_where}.offset must be an integer >= 0")
            if isinstance(length, bool) or not isinstance(length, int) or length < 1:
                raise ManifestError(f"{seg_where}.encoded_length must be an integer >= 1")
            if offset + length > encoded_block_length:
                raise ManifestError(f"{seg_where} extends beyond encoded expert block")
            if (
                not isinstance(shape, list)
                or not 1 <= len(shape) <= 4
                or any(isinstance(dim, bool) or not isinstance(dim, int) or dim < 1 for dim in shape)
            ):
                raise ManifestError(f"{seg_where}.expert_shape is invalid")
            if isinstance(ggml_type, bool) or not isinstance(ggml_type, int) or ggml_type < 0:
                raise ManifestError(f"{seg_where}.ggml_type must be an integer >= 0")
            if schema == GGUF_MOE_LAYOUT_SCHEMA:
                stride = segment.get("expert_index_stride_bytes")
                if isinstance(stride, bool) or not isinstance(stride, int) or stride != length:
                    raise ManifestError(f"{seg_where}.expert_index_stride_bytes must match encoded_length")
                source_shape = segment.get("source_tensor_shape")
                axis = segment.get("expert_axis")
                if (
                    not isinstance(source_shape, list)
                    or not 1 <= len(source_shape) <= 4
                    or source_shape[-1] != expert_count
                ):
                    raise ManifestError(f"{seg_where}.source_tensor_shape is incompatible with expert_count")
                if axis != len(source_shape) - 1:
                    raise ManifestError(f"{seg_where}.expert_axis must point at the final source dimension")
            ranges.append((offset, offset + length, role))

        if roles != set(_ROLES):
            raise ManifestError(f"{where} does not define down/gate/up exactly once")
        cursor = 0
        for start, end, _role in sorted(ranges):
            if start != cursor:
                raise ManifestError(f"{where}.segments must exactly and contiguously cover the block")
            cursor = end
        if cursor != encoded_block_length:
            raise ManifestError(f"{where}.segments do not cover encoded_block_length")
        for expert in range(expert_count):
            if model_map.route_block(layer, expert).length != encoded_block_length:
                raise ManifestError(
                    f"expert block length mismatch at layer={layer}, expert={expert}"
                )

    return {
        "schema_version": schema,
        "backend_contract": model_map.backend_contract,
        "layer_count": layer_count,
        "expert_count": expert_count,
        "active_experts_per_token": active,
        "core_tensors": len(core_items),
        "expert_routes": expected_routes,
        "metadata_gguf": metadata_summary,
    }


# Backward-compatible alias for callers that referenced the old name.
pack_granitemoe_gguf = pack_moe_gguf
