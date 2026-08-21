from __future__ import annotations

import hashlib
import json
import socket
import threading
from pathlib import Path

import pytest

from minimaxbrain.client import ExternalGateClient
from minimaxbrain.config import load_external_bundle
from minimaxbrain.errors import BudgetError, ConfigurationError, IntegrityError, ManifestError, ProtocolError
from minimaxbrain.external import ExternalGate
from minimaxbrain.model_map import MODEL_MAP_SCHEMA, load_model_map
from minimaxbrain.model_memory import ModelMemory
from minimaxbrain.protocol import IPC_PROTOCOL, validate_request
from minimaxbrain.packer import PACK_PLAN_SCHEMA, pack_from_plan
from minimaxbrain.server import ExternalGateServer
from minimaxbrain.storage import create_model_seal, verify_model_seal
from minimaxbrain.telemetry import MemoryTelemetry


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _fixture(tmp_path: Path, *, transport: str = "heap", slots: int | None = None):
    model_dir = tmp_path / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    payloads = {
        "core": b"CORE",
        "l0e0": b"EXPERT00",
        "l0e1": b"EXPERT01",
        "l1e0": b"EXPERT10",
    }
    shard = b"".join(payloads.values())
    (model_dir / "weights.bin").write_bytes(shard)
    offset = 0
    blocks = []
    for block_id, data in payloads.items():
        block = {
            "id": block_id,
            "kind": "core" if block_id == "core" else "expert",
            "shard": "weights.bin",
            "offset": offset,
            "length": len(data),
            "sha256": _sha(data),
            "alignment": 1,
        }
        if block_id != "core":
            block["layer"] = int(block_id[1])
            block["expert"] = int(block_id[3])
        blocks.append(block)
        offset += len(data)
    model_map = {
        "schema_version": MODEL_MAP_SCHEMA,
        "model": {
            "id": "tiny-moe",
            "architecture": "test-moe",
            "parameter_count": 1000,
            "quantization": {"name": "raw-test", "bits_per_weight": 8},
            "backend_contract": "test-backend-v1",
            "map_revision": "fixture-1",
        },
        "blocks": blocks,
    }
    (model_dir / "model.mmb-map.json").write_text(json.dumps(model_map), encoding="utf-8")

    memory = {
        "ram_budget": 12 if slots is None else None,
        "resident_experts": slots,
        "kv_cache": 0,
        "scratch": 0,
        "transport": transport,
        "lease_timeout_seconds": 30,
    }
    config = {
        "schema_version": "mmb-external-gate-config-v1",
        "model_map": "model/model.mmb-map.json",
        "memory": memory,
        "io": {"workers": 2, "prefetch_queue": 4, "integrity": "always"},
        "server": {
            "host": "127.0.0.1",
            "port": _free_port(),
            "api_token": None,
            "max_request_bytes": 65536,
        },
        "telemetry": {"enabled": False},
    }
    config_path = tmp_path / "mmb.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path, payloads


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_physical_map_is_strict_and_indexes_routes(tmp_path):
    config_path, _ = _fixture(tmp_path)
    config, model_map = load_external_bundle(config_path)

    assert model_map.core_bytes == 4
    assert model_map.largest_expert_bytes == 8
    assert model_map.route_block(0, 1).block_id == "l0e1"
    assert config.memory.cache_capacity_bytes == 12


def test_physical_map_rejects_shard_escape(tmp_path):
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"data")
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    manifest = {
        "schema_version": MODEL_MAP_SCHEMA,
        "model": {
            "id": "x", "architecture": "x", "parameter_count": 1,
            "quantization": {"name": "x", "bits_per_weight": 8},
            "backend_contract": "x", "map_revision": "1",
        },
        "blocks": [{
            "id": "core", "kind": "core", "shard": "../outside.bin",
            "offset": 0, "length": 4, "sha256": _sha(b"data"), "alignment": 1,
        }],
    }
    path = model_dir / "map.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ManifestError, match="escapes"):
        load_model_map(path)


def test_expert_count_mode_derives_worst_case_byte_budget(tmp_path):
    config_path, _ = _fixture(tmp_path, slots=2)
    config, model_map = load_external_bundle(config_path)

    assert config.memory.budget_mode == "resident_experts"
    assert config.memory.max_resident_experts == 2
    assert config.memory.ram_budget_bytes == model_map.core_bytes + 2 * model_map.largest_expert_bytes


def test_external_gate_prefetches_acquires_releases_and_evicts(tmp_path):
    config_path, _ = _fixture(tmp_path)
    config, model_map = load_external_bundle(config_path)
    telemetry = MemoryTelemetry()

    with ExternalGate(config, model_map, telemetry=telemetry) as gate:
        assert gate.cache.snapshot()["blocks"][0]["block_id"] == "core"
        result = gate.prefetch([{"block_id": "l0e0", "priority": 1}])
        assert result["accepted"] == 1
        lease = gate.acquire(["l0e0"], request_id="token-0-layer-0")
        assert lease["blocks"][0]["block_id"] == "l0e0"
        assert gate.release(lease["lease_id"])["released"] is True

        second = gate.acquire_routes([{"layer": 0, "expert": 1}])
        gate.release(second["lease_id"])
        snapshot = gate.snapshot()
        assert snapshot["memory"]["used_bytes"] <= 12
        assert snapshot["memory"]["resident_experts"] == 1
        assert snapshot["memory"]["evictions"] >= 1
        assert snapshot["io"]["bytes_read"] >= 20
    assert any(item["name"] == "acquire" for item in telemetry.events)


def test_external_gate_feeds_optional_structural_memory(tmp_path):
    config_path, _ = _fixture(tmp_path)
    config, model_map = load_external_bundle(config_path)

    with ModelMemory(tmp_path / "model-memory.sqlite3", model_map) as memory:
        with ExternalGate(config, model_map, model_memory=memory) as gate:
            first = gate.acquire_routes([{"layer": 0, "expert": 0}])
            gate.release(first["lease_id"])
            second = gate.acquire_routes([{"layer": 0, "expert": 0}])
            gate.release(second["lease_id"])

            learned = memory.route_profile(0)["items"][0]
            assert learned["expert"] == 0
            assert learned["requests"] == 2
            assert learned["misses"] == 1
            assert learned["hits"] == 1
            assert gate.snapshot()["model_memory"]["observed_routes"] == 1


def test_external_gate_auto_opens_configured_model_memory(tmp_path):
    config_path, _ = _fixture(tmp_path)
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    raw["model_memory"] = {"enabled": True, "path": "state/routes.sqlite3"}
    config_path.write_text(json.dumps(raw), encoding="utf-8")
    config, model_map = load_external_bundle(config_path)

    with ExternalGate(config, model_map) as gate:
        lease = gate.acquire_routes([{"layer": 1, "expert": 0}])
        gate.release(lease["lease_id"])
        assert gate.snapshot()["model_memory"]["observed_routes"] == 1

    assert config.model_memory.path.is_file()
    with ModelMemory(config.model_memory.path, model_map) as reopened:
        assert reopened.route_profile(1)["items"][0]["requests"] == 1


def test_model_memory_path_cannot_escape_configuration_directory(tmp_path):
    config_path, _ = _fixture(tmp_path)
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    raw["model_memory"] = {"enabled": True, "path": "../outside.sqlite3"}
    config_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ConfigurationError, match="escapes"):
        load_external_bundle(config_path)


def test_bad_speculation_does_not_break_compulsory_route(tmp_path):
    config_path, _ = _fixture(tmp_path)
    config, model_map = load_external_bundle(config_path)

    with ExternalGate(config, model_map) as gate:
        assert gate.prefetch([{"block_id": "not-real"}])["rejected"] == 1
        lease = gate.acquire(["l1e0"])
        assert lease["blocks"][0]["block_id"] == "l1e0"
        gate.release(lease["lease_id"])


def test_speculation_cannot_evict_a_warm_compulsory_block(tmp_path):
    config_path, _ = _fixture(tmp_path)
    config, model_map = load_external_bundle(config_path)

    with ExternalGate(config, model_map) as gate:
        warm = gate.acquire(["l0e0"])
        gate.release(warm["lease_id"])
        gate.prefetch([{"block_id": "l0e1", "priority": 1}])
        again = gate.acquire(["l0e0"])
        gate.release(again["lease_id"])
        assert gate.snapshot()["routing"]["hits"] >= 1
        assert gate.cache.contains("l0e0") is True


def test_active_lease_prevents_budget_eviction(tmp_path):
    config_path, _ = _fixture(tmp_path)
    config, model_map = load_external_bundle(config_path)

    with ExternalGate(config, model_map) as gate:
        first = gate.acquire(["l0e0"])
        with pytest.raises(BudgetError):
            gate.acquire(["l0e1"])
        gate.release(first["lease_id"])
        second = gate.acquire(["l0e1"])
        gate.release(second["lease_id"])


def test_integrity_failure_is_fail_closed(tmp_path):
    config_path, _ = _fixture(tmp_path)
    config, model_map = load_external_bundle(config_path)
    shard = model_map.path.parent / "weights.bin"
    raw = bytearray(shard.read_bytes())
    raw[4] ^= 0xFF
    shard.write_bytes(raw)

    with ExternalGate(config, model_map) as gate:
        with pytest.raises(IntegrityError, match="sha256 mismatch"):
            gate.acquire(["l0e0"])


def test_strict_ipc_contract_rejects_ambiguous_acquire():
    with pytest.raises(ProtocolError, match="exactly one"):
        validate_request({
            "protocol": IPC_PROTOCOL,
            "op": "acquire",
            "api_token": None,
            "map_revision": "fixture-1",
            "block_ids": ["a"],
            "routes": [{"layer": 0, "expert": 0}],
        })


def test_independent_server_exposes_shared_memory_payload(tmp_path):
    config_path, payloads = _fixture(tmp_path, transport="shared_memory")
    config, model_map = load_external_bundle(config_path)
    gate = ExternalGate(config, model_map)
    gate.start()
    server = ExternalGateServer(gate)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    client = ExternalGateClient(config.server.host, config.server.port, timeout_seconds=5)
    try:
        assert client.hello()["model_id"] == "tiny-moe"
        with client.acquire(["l0e0"], request_id="ipc-test") as lease:
            assert len(lease.blocks) == 1
            view = lease.blocks[0].view()
            try:
                assert bytes(view) == payloads["l0e0"]
            finally:
                view.release()
        assert client.stats()["routing"]["active_external_leases"] == 0
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        gate.close()


def test_streaming_packer_produces_aligned_valid_map(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "core.bin").write_bytes(b"core-data")
    (source / "expert.bin").write_bytes(b"expert-data")
    plan = {
        "schema_version": PACK_PLAN_SCHEMA,
        "model": {
            "id": "packed", "architecture": "test", "parameter_count": "2T",
            "quantization": {"name": "q4", "bits_per_weight": 4},
            "backend_contract": "test-v1", "map_revision": "1",
        },
        "alignment": 16,
        "shard_size": "1KiB",
        "blocks": [
            {"id": "core", "kind": "core", "source": "source/core.bin"},
            {"id": "l0e0", "kind": "expert", "source": "source/expert.bin", "layer": 0, "expert": 0},
        ],
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    manifest_path = pack_from_plan(plan_path, tmp_path / "packed")
    model_map = load_model_map(manifest_path)

    assert model_map.parameter_count == 2_000_000_000_000
    assert model_map.block("l0e0").offset == 16
    assert model_map.block("l0e0").sha256 == _sha(b"expert-data")


def test_model_seal_creation_and_tamper_detection(tmp_path):
    config_path, payloads = _fixture(tmp_path)
    config, model_map = load_external_bundle(config_path)

    # Initial state: not sealed
    valid, reason = verify_model_seal(model_map)
    assert not valid
    assert "does not exist" in str(reason)

    # Create seal
    seal_data = create_model_seal(model_map)
    assert seal_data["model_id"] == "tiny-moe"
    assert seal_data["total_blocks"] == 4

    # Verify valid seal
    valid, reason = verify_model_seal(model_map)
    assert valid
    assert reason is None

    # Tamper with shard content by modifying weights.bin
    shard_file = tmp_path / "model" / "weights.bin"
    shard_file.write_bytes(shard_file.read_bytes() + b"\x00")

    # Verification must fail closed immediately
    valid, reason = verify_model_seal(model_map)
    assert not valid
    assert "size modified" in str(reason)


def test_external_gate_seal_integrity_mode(tmp_path):
    config_path, payloads = _fixture(tmp_path)
    config, model_map = load_external_bundle(config_path)

    # Update config to integrity: seal without sealing first -> must fail on gate start
    config_data = json.loads(config_path.read_text(encoding="utf-8"))
    config_data["io"]["integrity"] = "seal"
    config_path.write_text(json.dumps(config_data), encoding="utf-8")

    config, model_map = load_external_bundle(config_path)
    with pytest.raises(IntegrityError, match="model seal verification failed"):
        with ExternalGate(config, model_map) as gate:
            pass

    # Seal the model
    create_model_seal(model_map)

    # Gate should start and read blocks at full speed
    with ExternalGate(config, model_map) as gate:
        lease = gate.acquire(["l0e0"], request_id="seal-test")
        assert len(lease["blocks"]) == 1
        gate.release(lease["lease_id"])


def test_external_gate_crc32_and_async_modes(tmp_path):
    for mode in ("crc32", "async"):
        config_path, payloads = _fixture(tmp_path / f"test_{mode}")
        config_data = json.loads(config_path.read_text(encoding="utf-8"))
        config_data["io"]["integrity"] = mode
        config_path.write_text(json.dumps(config_data), encoding="utf-8")

        config, model_map = load_external_bundle(config_path)
        with ExternalGate(config, model_map) as gate:
            lease = gate.acquire(["l0e0"], request_id=f"test-{mode}")
            assert len(lease["blocks"]) == 1
            gate.release(lease["lease_id"])

