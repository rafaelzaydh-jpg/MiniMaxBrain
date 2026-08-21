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
from .model_map import MODEL_MAP_SCHEMA


GGUF_MOE_LAYOUT_SCHEMA = "mmb-gguf-moe-layout-v1"
GGUF_MOE_BACKEND_CONTRACT = "mmb-raw-ggml-expert-concat-v1"
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
    try:
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
                "sha256": source_hash,
                "gguf_version": model.version,
                "architecture": model.architecture,
            },
            "backend_contract": GGUF_MOE_BACKEND_CONTRACT,
            "expert_axis": -1,
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


# Backward-compatible alias for callers that referenced the old name.
pack_granitemoe_gguf = pack_moe_gguf
