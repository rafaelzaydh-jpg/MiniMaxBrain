from __future__ import annotations

import struct

import pytest

from minimaxbrain.errors import ManifestError
from minimaxbrain.gguf import gguf_summary, load_gguf
from minimaxbrain.gguf_moe import GGUF_MOE_BACKEND_CONTRACT, pack_granitemoe_gguf, validate_moe_layout
from minimaxbrain.model_map import load_model_map
from minimaxbrain.storage import create_model_seal, verify_model_seal


def _u32(value: int) -> bytes:
    return struct.pack("<I", value)


def _u64(value: int) -> bytes:
    return struct.pack("<Q", value)


def _string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return _u64(len(encoded)) + encoded


def _small_gguf() -> bytes:
    body = bytearray(b"GGUF" + _u32(3) + _u64(2) + _u64(2))
    body += _string("general.architecture") + _u32(8) + _string("granitemoe")
    body += _string("general.alignment") + _u32(4) + _u32(32)
    body += _string("token_embd.weight") + _u32(1) + _u64(2) + _u32(0) + _u64(0)
    body += _string("blk.0.ffn_gate_exps.weight") + _u32(1) + _u64(2) + _u32(0) + _u64(32)
    body += b"\0" * ((32 - len(body) % 32) % 32)
    body += struct.pack("<2f", 1.0, 2.0)
    body += b"\0" * 24
    body += struct.pack("<2f", 3.0, 4.0)
    return bytes(body)


def _small_packable_moe_gguf() -> bytes:
    metadata = [
        ("general.architecture", 8, _string("granitemoe")),
        ("general.alignment", 4, _u32(32)),
        ("granitemoe.block_count", 4, _u32(1)),
        ("granitemoe.expert_count", 4, _u32(2)),
        ("granitemoe.expert_used_count", 4, _u32(1)),
    ]
    tensors = [
        ("token_embd.weight", (2,), 0, 0),
        ("blk.0.ffn_down_exps.weight", (1, 2), 0, 32),
        ("blk.0.ffn_gate_exps.weight", (1, 2), 0, 64),
        ("blk.0.ffn_up_exps.weight", (1, 2), 0, 96),
    ]
    body = bytearray(b"GGUF" + _u32(3) + _u64(len(tensors)) + _u64(len(metadata)))
    for key, value_type, encoded in metadata:
        body += _string(key) + _u32(value_type) + encoded
    for name, dimensions, ggml_type, offset in tensors:
        body += _string(name) + _u32(len(dimensions))
        for dimension in dimensions:
            body += _u64(dimension)
        body += _u32(ggml_type) + _u64(offset)
    body += b"\0" * ((32 - len(body) % 32) % 32)
    for values in ((1.0, 2.0), (10.0, 11.0), (20.0, 21.0), (30.0, 31.0)):
        body += struct.pack("<2f", *values) + b"\0" * 24
    return bytes(body)


def test_dependency_free_gguf_reader_exposes_real_ranges(tmp_path):
    path = tmp_path / "tiny.gguf"
    path.write_bytes(_small_gguf())

    model = load_gguf(path)
    summary = gguf_summary(model)

    assert model.version == 3
    assert model.architecture == "granitemoe"
    assert model.tensor("token_embd.weight").payload_length == 8
    assert model.tensor("token_embd.weight").span_length == 32
    assert summary["moe_tensors"] == 1
    assert summary["moe_tensor_samples"][0]["name"] == "blk.0.ffn_gate_exps.weight"


def test_gguf_reader_fails_closed_on_wrong_magic(tmp_path):
    path = tmp_path / "not.gguf"
    path.write_bytes(b"NOPE" + b"\0" * 64)

    with pytest.raises(ManifestError, match="not GGUF"):
        load_gguf(path)


def test_granitemoe_packer_splits_fused_experts_without_dequantizing(tmp_path):
    source = tmp_path / "tiny-moe.gguf"
    source.write_bytes(_small_packable_moe_gguf())

    manifest_path = pack_granitemoe_gguf(source, tmp_path / "packed", alignment=32)
    model_map = load_model_map(manifest_path)
    meta_path = manifest_path.parent / "model.mmb-meta.gguf"

    assert meta_path.is_file()
    assert meta_path.stat().st_size == load_gguf(source).tensor_data_offset
    assert model_map.backend_contract == GGUF_MOE_BACKEND_CONTRACT
    layout = validate_moe_layout(model_map)
    assert layout["schema_version"] == "mmb-gguf-moe-layout-v2"
    assert layout["metadata_gguf"]["file_name"] == "model.mmb-meta.gguf"
    assert layout["expert_routes"] == 2
    assert len(model_map.core_blocks) == 1
    assert len(model_map.expert_blocks) == 2
    first = model_map.route_block(0, 0)
    second = model_map.route_block(0, 1)
    assert first.length == second.length == 12
    with first.shard.open("rb") as handle:
        handle.seek(first.offset)
        assert struct.unpack("<3f", handle.read(first.length)) == (10.0, 20.0, 30.0)
        handle.seek(second.offset)
        assert struct.unpack("<3f", handle.read(second.length)) == (11.0, 21.0, 31.0)


def test_layout_validator_rejects_tampered_metadata_gguf(tmp_path):
    source = tmp_path / "tiny-moe.gguf"
    source.write_bytes(_small_packable_moe_gguf())

    manifest_path = pack_granitemoe_gguf(source, tmp_path / "packed", alignment=32)
    model_map = load_model_map(manifest_path)
    meta_path = manifest_path.parent / "model.mmb-meta.gguf"

    data = bytearray(meta_path.read_bytes())
    data[-1] ^= 0x01
    meta_path.write_bytes(data)

    with pytest.raises(ManifestError, match="metadata GGUF SHA-256 mismatch"):
        validate_moe_layout(model_map)


def test_model_seal_binds_layout_and_metadata(tmp_path):
    source = tmp_path / "tiny-moe.gguf"
    source.write_bytes(_small_packable_moe_gguf())
    manifest_path = pack_granitemoe_gguf(source, tmp_path / "packed", alignment=32)
    model_map = load_model_map(manifest_path)

    seal = create_model_seal(model_map)
    assert seal["schema_version"] == "mmb-model-seal-v2"
    assert set(seal["auxiliary_files"]) == {"model.mmb-layout.json", "model.mmb-meta.gguf"}
    assert verify_model_seal(model_map) == (True, None)

    layout_path = manifest_path.parent / "model.mmb-layout.json"
    layout = layout_path.read_text(encoding="utf-8")
    layout_path.write_text(layout.replace('"expert_storage"', '"expert_storagf"', 1), encoding="utf-8")

    valid, reason = verify_model_seal(model_map)
    assert valid is False
    assert reason and "model.mmb-layout.json sha256 modified" in reason
