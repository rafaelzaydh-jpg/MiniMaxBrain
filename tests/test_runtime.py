import hashlib
import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from minimaxbrain.errors import BackendUnavailableError, ConfigurationError
from minimaxbrain.runtime import InferenceMode, MMBRuntime
from minimaxbrain.server_http import start_mmb_server


def _fixture(tmp_path: Path) -> Path:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    payload = b"core" + b"expert00"
    shard = model_dir / "model-00000.mmbw"
    shard.write_bytes(payload)

    manifest = {
        "schema_version": "mmb-physical-model-map-v1",
        "model": {
            "id": "tiny-moe",
            "architecture": "test",
            "parameter_count": 12,
            "quantization": {"name": "raw", "bits_per_weight": 8},
            "backend_contract": "test-v1",
            "map_revision": "fixture-1",
        },
        "blocks": [
            {
                "id": "core",
                "kind": "core",
                "shard": shard.name,
                "offset": 0,
                "length": 4,
                "sha256": hashlib.sha256(b"core").hexdigest(),
                "alignment": 1,
            },
            {
                "id": "expert/0/0",
                "kind": "expert",
                "shard": shard.name,
                "offset": 4,
                "length": 8,
                "sha256": hashlib.sha256(b"expert00").hexdigest(),
                "alignment": 1,
                "layer": 0,
                "expert": 0,
            },
        ],
    }
    (model_dir / "model.mmb-map.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    (model_dir / "model.mmb-layout.json").write_text(
        json.dumps(
            {
                "schema_version": "mmb-gguf-moe-layout-v1",
                "source": {
                    "file_name": "missing-reference.gguf",
                    "sha256": "0" * 64,
                    "gguf_version": 3,
                    "architecture": "test",
                },
                "layer_count": 1,
                "expert_count": 1,
                "active_experts_per_token": 1,
            }
        ),
        encoding="utf-8",
    )
    config = {
        "schema_version": "mmb-external-gate-config-v1",
        "model_map": "model.mmb-map.json",
        "memory": {
            "ram_budget": "1MiB",
            "resident_experts": None,
            "kv_cache": 0,
            "scratch": 0,
            "lease_timeout_seconds": 120,
        },
        "io": {"workers": 1, "prefetch_queue": 0, "integrity": "first_load"},
        "server": {
            "host": "127.0.0.1",
            "port": 55321,
            "api_token": None,
            "max_request_bytes": 65536,
        },
        "telemetry": {"enabled": False},
        "model_memory": {"enabled": False, "path": "state.sqlite3"},
    }
    config_path = model_dir / "gate.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path


def _free_port() -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_runtime_fails_closed_without_real_backend(tmp_path: Path):
    runtime = MMBRuntime(_fixture(tmp_path))
    try:
        assert runtime.ready is False
        assert runtime.inference_mode is InferenceMode.UNAVAILABLE
        assert runtime.backend_error
        assert runtime.stats()["gguf_path"] is None
        assert runtime.stats()["paged_experts_used"] is False

        with pytest.raises(BackendUnavailableError):
            list(runtime.stream_generate("oi", max_tokens=4))
    finally:
        runtime.close()


def test_http_health_and_chat_are_fail_closed(tmp_path: Path):
    runtime = MMBRuntime(_fixture(tmp_path))
    port = _free_port()
    server = start_mmb_server(runtime, host="127.0.0.1", port=port)
    try:
        with pytest.raises(urllib.error.HTTPError) as health_error:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health")
        assert health_error.value.code == 503
        health = json.loads(health_error.value.read().decode("utf-8"))
        assert health["status"] == "not_ready"
        assert health["inference_mode"] == "unavailable"

        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            data=json.dumps(
                {
                    "messages": [{"role": "user", "content": "oi"}],
                    "max_tokens": 4,
                    "stream": False,
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as chat_error:
            urllib.request.urlopen(request)
        assert chat_error.value.code == 503
        payload = json.loads(chat_error.value.read().decode("utf-8"))
        assert payload["error"]["code"] == "BACKEND_UNAVAILABLE"
    finally:
        server.shutdown()
        server.server_close()
        runtime.close()


def test_message_contract_preserves_full_history():
    history = [
        {"role": "system", "content": "Seja breve."},
        {"role": "user", "content": "Meu nome e Ana."},
        {"role": "assistant", "content": "Entendido."},
        {"role": "user", "content": "Qual e meu nome?"},
    ]
    assert MMBRuntime._validate_messages(history) == history


def test_http_requires_bearer_token_when_configured(tmp_path: Path):
    config_path = _fixture(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["server"]["api_token"] = "0123456789abcdef"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    runtime = MMBRuntime(config_path)
    port = _free_port()
    server = start_mmb_server(runtime, host="127.0.0.1", port=port)
    try:
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            data=json.dumps({
                "messages": [{"role": "user", "content": "oi"}],
                "max_tokens": 4,
            }).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as unauthorized:
            urllib.request.urlopen(request)
        assert unauthorized.value.code == 401

        request.add_header("Authorization", "Bearer 0123456789abcdef")
        with pytest.raises(urllib.error.HTTPError) as unavailable:
            urllib.request.urlopen(request)
        assert unavailable.value.code == 503
    finally:
        server.shutdown()
        server.server_close()
        runtime.close()


def test_http_refuses_non_loopback_without_token(tmp_path: Path):
    runtime = MMBRuntime(_fixture(tmp_path))
    try:
        with pytest.raises(ConfigurationError, match="api_token"):
            start_mmb_server(runtime, host="0.0.0.0", port=_free_port())
    finally:
        runtime.close()
