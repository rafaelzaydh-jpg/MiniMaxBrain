from __future__ import annotations

import struct

import pytest

from tools.qwen36_acceptance import _require_qwen36_base


def _u32(value: int) -> bytes:
    return struct.pack("<I", value)


def _u64(value: int) -> bytes:
    return struct.pack("<Q", value)


def _string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return _u64(len(encoded)) + encoded


def _qwen36_directory(*, layers: int = 40, experts: int = 256, active: int = 8) -> bytes:
    metadata = [
        ("general.architecture", 8, _string("qwen35moe")),
        ("general.alignment", 4, _u32(32)),
        ("qwen35moe.block_count", 4, _u32(layers)),
        ("qwen35moe.expert_count", 4, _u32(experts)),
        ("qwen35moe.expert_used_count", 4, _u32(active)),
    ]
    tensors: list[tuple[str, tuple[int, ...], int, int]] = []
    offset = 0

    # One core tensor keeps the GGUF representative of a complete base file.
    tensors.append(("token_embd.weight", (1,), 0, offset))
    offset += 32

    for layer in range(layers):
        for role in ("down", "gate", "up"):
            name = f"blk.{layer}.ffn_{role}_exps.weight"
            # F32 payload = experts * 4 bytes; round physical spans to 32 bytes.
            tensors.append((name, (1, experts), 0, offset))
            payload = experts * 4
            offset += ((payload + 31) // 32) * 32

    body = bytearray(b"GGUF" + _u32(3) + _u64(len(tensors)) + _u64(len(metadata)))
    for key, value_type, encoded in metadata:
        body += _string(key) + _u32(value_type) + encoded
    for name, dimensions, ggml_type, tensor_offset in tensors:
        body += _string(name) + _u32(len(dimensions))
        for dimension in dimensions:
            body += _u64(dimension)
        body += _u32(ggml_type) + _u64(tensor_offset)

    body += b"\0" * ((32 - len(body) % 32) % 32)
    body += b"\0" * offset
    return bytes(body)


def test_qwen36_preflight_accepts_expected_base_topology(tmp_path):
    gguf = tmp_path / "qwen36.gguf"
    gguf.write_bytes(_qwen36_directory())

    info = _require_qwen36_base(gguf)

    assert info["architecture"] == "qwen35moe"
    assert info["layers"] == 40
    assert info["experts"] == 256
    assert info["active_experts"] == 8


def test_qwen36_preflight_rejects_other_qwen35moe_sizes(tmp_path):
    gguf = tmp_path / "other.gguf"
    gguf.write_bytes(_qwen36_directory(layers=48))

    with pytest.raises(ValueError, match="not the expected Qwen3.6-35B-A3B"):
        _require_qwen36_base(gguf)
