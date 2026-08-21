"""Dependency-free GGUF directory reader used by real-model diagnostics.

This module does not execute GGUF models and is not part of the gate's routing
authority. It exposes the immutable tensor ranges needed by a model-specific
converter and by physical paging benchmarks.
"""
from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Dict

from .errors import ManifestError


GGUF_MAGIC = b"GGUF"
_VALUE_TYPES = {
    0: ("B", 1),
    1: ("b", 1),
    2: ("H", 2),
    3: ("h", 2),
    4: ("I", 4),
    5: ("i", 4),
    6: ("f", 4),
    7: ("?", 1),
    10: ("Q", 8),
    11: ("q", 8),
    12: ("d", 8),
}

# GGML block size and encoded bytes per block. Unknown future types remain
# inspectable through their range spans even when exact payload size is absent.
_GGML_LAYOUTS = {
    0: (1, 4),    # F32
    1: (1, 2),    # F16
    2: (32, 18),  # Q4_0
    3: (32, 20),  # Q4_1
    6: (32, 22),  # Q5_0
    7: (32, 24),  # Q5_1
    8: (32, 34),  # Q8_0
    9: (32, 36),  # Q8_1
    10: (256, 84),   # Q2_K
    11: (256, 110),  # Q3_K
    12: (256, 144),  # Q4_K
    13: (256, 176),  # Q5_K
    14: (256, 210),  # Q6_K
    15: (256, 292),  # Q8_K
    16: (256, 66),   # IQ2_XXS
    17: (256, 74),   # IQ2_XS
    18: (256, 98),   # IQ3_XXS
    19: (256, 50),   # IQ1_S
    20: (32, 18),    # IQ4_NL
    21: (256, 110),  # IQ3_S
    22: (256, 82),   # IQ2_S
    23: (256, 136),  # IQ4_XS
    24: (1, 1),      # I8
    25: (1, 2),      # I16
    26: (1, 4),      # I32
    27: (1, 8),      # I64
    28: (1, 8),      # F64
    30: (1, 2),      # BF16
}


class _Reader:
    def __init__(self, handle: BinaryIO, file_size: int):
        self.handle = handle
        self.file_size = file_size

    def read(self, length: int) -> bytes:
        if length < 0 or self.handle.tell() + length > self.file_size:
            raise ManifestError("truncated or invalid GGUF directory")
        value = self.handle.read(length)
        if len(value) != length:
            raise ManifestError("truncated GGUF file")
        return value

    def unpack(self, fmt: str) -> Any:
        return struct.unpack("<" + fmt, self.read(struct.calcsize("<" + fmt)))[0]

    def string(self) -> str:
        length = int(self.unpack("Q"))
        if length > min(self.file_size, 64 << 20):
            raise ManifestError(f"GGUF string length is unreasonable: {length}")
        try:
            return self.read(length).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ManifestError("GGUF string is not valid UTF-8") from exc

    def value(self, value_type: int, *, depth: int = 0) -> Any:
        if depth > 4:
            raise ManifestError("GGUF metadata arrays are nested too deeply")
        if value_type in _VALUE_TYPES:
            return self.unpack(_VALUE_TYPES[value_type][0])
        if value_type == 8:
            return self.string()
        if value_type == 9:
            element_type = int(self.unpack("I"))
            count = int(self.unpack("Q"))
            if count > 10_000_000:
                raise ManifestError(f"GGUF metadata array is unreasonable: {count} elements")
            return [self.value(element_type, depth=depth + 1) for _ in range(count)]
        raise ManifestError(f"unsupported GGUF metadata value type: {value_type}")


@dataclass(frozen=True)
class GGUFTensor:
    name: str
    dimensions: tuple[int, ...]
    ggml_type: int
    relative_offset: int
    absolute_offset: int
    span_length: int
    payload_length: int | None

    @property
    def elements(self) -> int:
        return math.prod(self.dimensions)


@dataclass(frozen=True)
class GGUFModel:
    path: Path
    version: int
    alignment: int
    tensor_data_offset: int
    metadata: Dict[str, Any]
    tensors: tuple[GGUFTensor, ...]

    @property
    def architecture(self) -> str | None:
        value = self.metadata.get("general.architecture")
        return value if isinstance(value, str) else None

    def tensor(self, name: str) -> GGUFTensor:
        for item in self.tensors:
            if item.name == name:
                return item
        raise KeyError(name)


def _payload_length(dimensions: tuple[int, ...], ggml_type: int) -> int | None:
    layout = _GGML_LAYOUTS.get(ggml_type)
    if layout is None:
        return None
    block_size, encoded_bytes = layout
    elements = math.prod(dimensions)
    if elements % block_size:
        return None
    return elements // block_size * encoded_bytes


def load_gguf(path: str | Path) -> GGUFModel:
    file_path = Path(path).resolve()
    if not file_path.is_file():
        raise ManifestError(f"GGUF file does not exist: {file_path}")
    file_size = file_path.stat().st_size
    with file_path.open("rb") as handle:
        reader = _Reader(handle, file_size)
        if reader.read(4) != GGUF_MAGIC:
            raise ManifestError("file is not GGUF")
        version = int(reader.unpack("I"))
        if version not in {2, 3}:
            raise ManifestError(f"unsupported GGUF version: {version}")
        tensor_count = int(reader.unpack("Q"))
        metadata_count = int(reader.unpack("Q"))
        if tensor_count > 10_000_000 or metadata_count > 1_000_000:
            raise ManifestError("GGUF directory counts are unreasonable")
        metadata: Dict[str, Any] = {}
        for _ in range(metadata_count):
            key = reader.string()
            if key in metadata:
                raise ManifestError(f"duplicate GGUF metadata key: {key}")
            metadata[key] = reader.value(int(reader.unpack("I")))
        raw_tensors: list[tuple[str, tuple[int, ...], int, int]] = []
        names: set[str] = set()
        for _ in range(tensor_count):
            name = reader.string()
            if name in names:
                raise ManifestError(f"duplicate GGUF tensor name: {name}")
            names.add(name)
            dimensions_count = int(reader.unpack("I"))
            if not 1 <= dimensions_count <= 8:
                raise ManifestError(f"invalid dimension count for GGUF tensor {name!r}")
            dimensions = tuple(int(reader.unpack("Q")) for _ in range(dimensions_count))
            if any(value <= 0 for value in dimensions):
                raise ManifestError(f"invalid shape for GGUF tensor {name!r}")
            ggml_type = int(reader.unpack("I"))
            relative_offset = int(reader.unpack("Q"))
            raw_tensors.append((name, dimensions, ggml_type, relative_offset))
        alignment = metadata.get("general.alignment", 32)
        if isinstance(alignment, bool) or not isinstance(alignment, int) or not 1 <= alignment <= (16 << 20):
            raise ManifestError("GGUF general.alignment is invalid")
        directory_end = handle.tell()
        tensor_data_offset = ((directory_end + alignment - 1) // alignment) * alignment

    ordered = sorted(raw_tensors, key=lambda item: item[3])
    tensors: list[GGUFTensor] = []
    for index, (name, dimensions, ggml_type, relative_offset) in enumerate(ordered):
        absolute = tensor_data_offset + relative_offset
        next_absolute = (
            tensor_data_offset + ordered[index + 1][3]
            if index + 1 < len(ordered)
            else file_size
        )
        if absolute < tensor_data_offset or next_absolute <= absolute or next_absolute > file_size:
            raise ManifestError(f"invalid data range for GGUF tensor {name!r}")
        payload = _payload_length(dimensions, ggml_type)
        span = next_absolute - absolute
        if payload is not None and payload > span:
            raise ManifestError(f"GGUF tensor {name!r} payload exceeds its physical span")
        tensors.append(GGUFTensor(
            name=name,
            dimensions=dimensions,
            ggml_type=ggml_type,
            relative_offset=relative_offset,
            absolute_offset=absolute,
            span_length=span,
            payload_length=payload,
        ))
    return GGUFModel(
        path=file_path,
        version=version,
        alignment=alignment,
        tensor_data_offset=tensor_data_offset,
        metadata=metadata,
        tensors=tuple(tensors),
    )


def gguf_summary(model: GGUFModel) -> Dict[str, Any]:
    type_counts: dict[str, int] = {}
    for tensor in model.tensors:
        key = str(tensor.ggml_type)
        type_counts[key] = type_counts.get(key, 0) + 1
    moe_tensors = [
        item for item in model.tensors
        if "_exps" in item.name or ".experts." in item.name or "expert" in item.name.lower()
    ]
    return {
        "path": str(model.path),
        "file_bytes": model.path.stat().st_size,
        "version": model.version,
        "alignment": model.alignment,
        "architecture": model.architecture,
        "tensor_data_offset": model.tensor_data_offset,
        "metadata_entries": len(model.metadata),
        "tensors": len(model.tensors),
        "tensor_types": type_counts,
        "moe_tensors": len(moe_tensors),
        "moe_tensor_samples": [
            {
                "name": item.name,
                "shape": list(item.dimensions),
                "type": item.ggml_type,
                "offset": item.absolute_offset,
                "payload_bytes": item.payload_length,
                "span_bytes": item.span_length,
            }
            for item in moe_tensors[:24]
        ],
    }

