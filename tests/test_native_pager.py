from __future__ import annotations

import struct
from pathlib import Path

import pytest

from minimaxbrain.gguf_moe import pack_moe_gguf
from minimaxbrain.native import NativePager, find_native_backend


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


@pytest.mark.skipif(find_native_backend() is None, reason="native MMB backend not built")
def test_native_pager_reads_real_mmb_segments(tmp_path: Path):
    source = tmp_path / "tiny.gguf"
    source.write_bytes(_tiny_moe())
    pack_moe_gguf(source, tmp_path / "bundle", alignment=32)

    with NativePager(tmp_path / "bundle", 12, verify_sha256=True) as pager:
        assert pager.model_info == {
            "layer_count": 1,
            "expert_count": 2,
            "active_experts_per_token": 1,
        }
        with pager.acquire(0, [0, 0], router_request=True) as lease:
            assert lease.count == 1
            assert lease.expert_id(0) == 0
            assert struct.unpack("<f", lease.segment(0, "down").copy_bytes())[0] == 10.0
            assert struct.unpack("<f", lease.segment(0, "gate").copy_bytes())[0] == 20.0
            assert struct.unpack("<f", lease.segment(0, "up").copy_bytes())[0] == 30.0

        with pager.acquire(0, [1]) as lease:
            assert lease.expert_id(0) == 1
            assert struct.unpack("<f", lease.segment(0, "gate").copy_bytes())[0] == 21.0

        stats = pager.stats()
        assert stats["cache_misses"] >= 2
        assert stats["evictions"] >= 1
        assert stats["real_router_requests"] == 1
        assert stats["paged_experts_used"] is False
        assert stats["paged_moe_kernel_available"] is True
        assert stats["native_runtime_available"] is True
