from __future__ import annotations

import json
import struct
from pathlib import Path

from minimaxbrain.bundle_config import prepare_bundle_config
from minimaxbrain.config import load_external_bundle
from minimaxbrain.gguf_moe import pack_moe_gguf


def _u32(value: int) -> bytes:
    return struct.pack("<I", value)


def _u64(value: int) -> bytes:
    return struct.pack("<Q", value)


def _string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return _u64(len(encoded)) + encoded


def _tiny_moe() -> bytes:
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


def test_prepare_existing_bundle_without_reconversion(tmp_path: Path):
    source = tmp_path / "tiny.gguf"
    source.write_bytes(_tiny_moe())
    bundle = tmp_path / "bundle"
    pack_moe_gguf(source, bundle, alignment=32)

    shard = bundle / "model-00000.mmbw"
    before = shard.stat().st_mtime_ns
    config_path = prepare_bundle_config(bundle, expert_cache_bytes=1024)
    after = shard.stat().st_mtime_ns

    assert config_path == bundle / "gate.json"
    assert before == after

    raw = json.loads(config_path.read_text(encoding="utf-8"))
    assert raw["model_map"] == "model.mmb-map.json"

    config, model_map = load_external_bundle(config_path)
    assert config.memory.cache_capacity_bytes - model_map.core_bytes == 1024


def test_prepare_is_non_destructive_when_gate_exists(tmp_path: Path):
    source = tmp_path / "tiny.gguf"
    source.write_bytes(_tiny_moe())
    bundle = tmp_path / "bundle"
    pack_moe_gguf(source, bundle, alignment=32)

    config_path = prepare_bundle_config(bundle, expert_cache_bytes=1024)
    original = config_path.read_text(encoding="utf-8")
    prepare_bundle_config(bundle, expert_cache_bytes=2048)
    assert config_path.read_text(encoding="utf-8") == original
