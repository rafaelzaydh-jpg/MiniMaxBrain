from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from minimaxbrain.errors import ModelMemoryError
from minimaxbrain.model_map import MODEL_MAP_SCHEMA, load_model_map
from minimaxbrain.model_memory import MODEL_MEMORY_SCHEMA, ModelMemory


def _model_map(tmp_path: Path, *, revision: str = "map-1"):
    model_dir = tmp_path / revision
    model_dir.mkdir()
    payloads = {
        "core": b"CORE",
        "l0e0": b"EXPERT00",
        "l0e1": b"EXPERT01",
        "l1e0": b"EXPERT10",
    }
    shard = b"".join(payloads.values())
    (model_dir / "weights.mmbw").write_bytes(shard)
    blocks = []
    offset = 0
    for block_id, payload in payloads.items():
        item = {
            "id": block_id,
            "kind": "core" if block_id == "core" else "expert",
            "shard": "weights.mmbw",
            "offset": offset,
            "length": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "alignment": 1,
        }
        if block_id != "core":
            item["layer"] = int(block_id[1])
            item["expert"] = int(block_id[3])
        blocks.append(item)
        offset += len(payload)
    manifest = {
        "schema_version": MODEL_MAP_SCHEMA,
        "model": {
            "id": "tiny-moe",
            "architecture": "test-moe",
            "parameter_count": 1000,
            "quantization": {"name": "raw-test", "bits_per_weight": 8},
            "backend_contract": "test-backend-v1",
            "map_revision": revision,
        },
        "blocks": blocks,
    }
    path = model_dir / "model.mmb-map.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return load_model_map(path)


def test_model_memory_registers_reduced_map_and_reopens_idempotently(tmp_path):
    model_map = _model_map(tmp_path)
    database = tmp_path / "memory" / "model.sqlite3"

    with ModelMemory(database, model_map) as memory:
        snapshot = memory.snapshot()
        assert snapshot["schema_version"] == MODEL_MEMORY_SCHEMA
        assert snapshot["map_revision"] == "map-1"
        assert snapshot["nodes"] == 7  # model + two layers + four physical blocks
        assert snapshot["edges"] == 6
        assert memory.node("expert:0:1")["payload"]["block_id"] == "l0e1"

    with ModelMemory(database, model_map) as reopened:
        assert reopened.snapshot()["nodes"] == 7
        assert len(reopened.history()["items"]) == 1


def test_model_memory_revision_conflict_preserves_current_node(tmp_path):
    model_map = _model_map(tmp_path)
    with ModelMemory(tmp_path / "model.sqlite3", model_map) as memory:
        created = memory.put_node("tensor:router:0", "tensor", {"shape": [16, 32]})
        revised = memory.put_node(
            "tensor:router:0",
            "tensor",
            {"shape": [16, 64]},
            expected_revision=created["revision"],
        )
        assert revised["revision"] == 2
        with pytest.raises(ModelMemoryError, match="revision conflict"):
            memory.put_node(
                "tensor:router:0",
                "tensor",
                {"shape": [1]},
                expected_revision=1,
            )
        assert memory.node("tensor:router:0")["payload"] == {"shape": [16, 64]}


def test_model_memory_neighbors_are_explicitly_bounded_and_paged(tmp_path):
    model_map = _model_map(tmp_path)
    with ModelMemory(tmp_path / "model.sqlite3", model_map) as memory:
        model_node = "model:tiny-moe"
        first = memory.neighbors(model_node, limit=1)
        assert len(first["items"]) == 1
        assert first["next_cursor"] is not None
        second = memory.neighbors(model_node, limit=1, cursor=first["next_cursor"])
        assert len(second["items"]) == 1
        assert first["items"][0]["edge_id"] != second["items"][0]["edge_id"]
        with pytest.raises(ModelMemoryError, match=r"\[1, 256\]"):
            memory.neighbors(model_node, limit=1000)


def test_model_memory_learns_physical_route_profile_without_model_weights(tmp_path):
    model_map = _model_map(tmp_path)
    with ModelMemory(tmp_path / "model.sqlite3", model_map) as memory:
        memory.record_route(0, 1, state="miss", duration_ms=12.0)
        memory.record_route(0, 1, state="hit", duration_ms=2.0)
        memory.record_route(0, 0, state="prefetch_wait", duration_ms=4.0)

        profile = memory.route_profile(0)["items"]
        assert [item["expert"] for item in profile] == [1, 0]
        expert = profile[0]
        assert expert["requests"] == 2
        assert expert["hits"] == 1
        assert expert["misses"] == 1
        assert expert["requested_bytes"] == 16
        assert expert["admitted_bytes"] == 8
        assert expert["average_latency_ms"] == 7.0
        assert memory.snapshot()["observed_routes"] == 2


def test_model_memory_keeps_map_revisions_physically_separate(tmp_path):
    first_map = _model_map(tmp_path, revision="map-1")
    second_map = _model_map(tmp_path, revision="map-2")
    database = tmp_path / "model.sqlite3"

    with ModelMemory(database, first_map) as first:
        first.record_route(0, 0, state="miss")
        assert first.snapshot()["observed_routes"] == 1

    with ModelMemory(database, second_map) as second:
        assert second.snapshot()["map_revision"] == "map-2"
        assert second.snapshot()["observed_routes"] == 0
        assert all(item["map_revision"] == "map-2" for item in second.history()["items"])
