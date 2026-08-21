"""Current-contract configuration for the independent external gate."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping

from .errors import BudgetError, ConfigurationError
from .model_map import PhysicalModelMap, load_model_map
from .units import parse_bytes


CONFIG_SCHEMA = "mmb-external-gate-config-v1"
_ROOT_FIELDS = {"schema_version", "model_map", "memory", "io", "server", "telemetry", "model_memory"}
_MEMORY_FIELDS = {
    "ram_budget", "resident_experts", "kv_cache", "scratch", "transport", "lease_timeout_seconds"
}
_IO_FIELDS = {"workers", "prefetch_queue", "integrity"}
_SERVER_FIELDS = {"host", "port", "api_token", "max_request_bytes"}
_TELEMETRY_FIELDS = {"enabled"}
_MODEL_MEMORY_FIELDS = {"enabled", "path"}


def _object(value: Any, where: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigurationError(f"{where} must be an object")
    return value


def _reject_unknown(value: Mapping[str, Any], allowed: set[str], where: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ConfigurationError(f"unknown field(s) at {where}: {', '.join(unknown)}")


def _int(value: Any, where: str, *, minimum: int, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigurationError(f"{where} must be an integer >= {minimum}")
    if maximum is not None and value > maximum:
        raise ConfigurationError(f"{where} must be <= {maximum}")
    return value


@dataclass(frozen=True)
class MemoryConfig:
    ram_budget_bytes: int
    max_resident_experts: int | None
    kv_cache_bytes: int
    scratch_bytes: int
    transport: str
    lease_timeout_seconds: float
    budget_mode: str

    @property
    def cache_capacity_bytes(self) -> int:
        return self.ram_budget_bytes - self.kv_cache_bytes - self.scratch_bytes


@dataclass(frozen=True)
class IoConfig:
    workers: int
    prefetch_queue: int
    integrity: str


@dataclass(frozen=True)
class ServerConfig:
    host: str
    port: int
    api_token: str | None
    max_request_bytes: int


@dataclass(frozen=True)
class ModelMemoryConfig:
    enabled: bool
    path: Path


@dataclass(frozen=True)
class ExternalGateConfig:
    path: Path
    model_map_path: Path
    memory: MemoryConfig
    io: IoConfig
    server: ServerConfig
    telemetry_enabled: bool
    model_memory: ModelMemoryConfig


def _resolve_inside(base: Path, value: str, field: str) -> Path:
    candidate = (base / value).resolve()
    try:
        candidate.relative_to(base.resolve())
    except ValueError as exc:
        raise ConfigurationError(f"{field} escapes configuration directory") from exc
    return candidate


def load_external_config(path: str | Path, model_map: PhysicalModelMap | None = None) -> ExternalGateConfig:
    config_path = Path(path).resolve()
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"cannot read configuration: {exc}") from exc
    root = _object(raw, "root")
    _reject_unknown(root, _ROOT_FIELDS, "root")
    if root.get("schema_version") != CONFIG_SCHEMA:
        raise ConfigurationError(f"schema_version must be {CONFIG_SCHEMA!r}")
    raw_map = root.get("model_map")
    if not isinstance(raw_map, str) or not raw_map.strip():
        raise ConfigurationError("model_map must be a non-empty relative path")
    model_map_path = _resolve_inside(config_path.parent, raw_map, "model_map")

    memory = _object(root.get("memory"), "memory")
    _reject_unknown(memory, _MEMORY_FIELDS, "memory")
    raw_ram = memory.get("ram_budget")
    raw_slots = memory.get("resident_experts")
    if (raw_ram is None) == (raw_slots is None):
        raise ConfigurationError("memory must set exactly one of ram_budget or resident_experts")
    slots: int | None = None
    budget_mode = "ram"
    if raw_slots is not None:
        slots = _int(raw_slots, "memory.resident_experts", minimum=1)
        budget_mode = "resident_experts"
        if model_map is None:
            raise ConfigurationError("resident_experts mode requires a loaded model map")
    kv = parse_bytes(memory.get("kv_cache", 0), field="memory.kv_cache")
    scratch = parse_bytes(memory.get("scratch", 0), field="memory.scratch")
    if raw_ram is not None:
        ram = parse_bytes(raw_ram, field="memory.ram_budget")
    else:
        assert model_map is not None and slots is not None
        # Worst-case sizing makes an expert-count budget physically honest even
        # when expert blocks have different encoded lengths.
        ram = kv + scratch + model_map.core_bytes + slots * model_map.largest_expert_bytes
    transport = memory.get("transport", "heap")
    if transport not in {"heap", "shared_memory"}:
        raise ConfigurationError("memory.transport must be 'heap' or 'shared_memory'")
    lease_timeout = memory.get("lease_timeout_seconds", 120.0)
    if isinstance(lease_timeout, bool) or not isinstance(lease_timeout, (int, float)) or lease_timeout <= 0:
        raise ConfigurationError("memory.lease_timeout_seconds must be positive")

    io = _object(root.get("io", {}), "io")
    _reject_unknown(io, _IO_FIELDS, "io")
    workers = _int(io.get("workers", 2), "io.workers", minimum=1, maximum=64)
    queue_depth = _int(io.get("prefetch_queue", 32), "io.prefetch_queue", minimum=0, maximum=100000)
    integrity = io.get("integrity", "first_load")
    if integrity not in {"always", "first_load", "none"}:
        raise ConfigurationError("io.integrity must be always, first_load, or none")

    server = _object(root.get("server", {}), "server")
    _reject_unknown(server, _SERVER_FIELDS, "server")
    host = server.get("host", "127.0.0.1")
    if not isinstance(host, str) or not host.strip():
        raise ConfigurationError("server.host must be a non-empty string")
    port = _int(server.get("port", 55321), "server.port", minimum=1, maximum=65535)
    token = server.get("api_token")
    if token is not None and (not isinstance(token, str) or len(token) < 16):
        raise ConfigurationError("server.api_token must be null or at least 16 characters")
    if host.strip().lower() not in {"127.0.0.1", "localhost", "::1"} and token is None:
        raise ConfigurationError("server.api_token is required when binding outside loopback")
    max_request = _int(
        server.get("max_request_bytes", 1 << 20), "server.max_request_bytes", minimum=1024, maximum=64 << 20
    )

    telemetry = _object(root.get("telemetry", {}), "telemetry")
    _reject_unknown(telemetry, _TELEMETRY_FIELDS, "telemetry")
    telemetry_enabled = telemetry.get("enabled", True)
    if not isinstance(telemetry_enabled, bool):
        raise ConfigurationError("telemetry.enabled must be boolean")

    model_memory_raw = _object(root.get("model_memory", {}), "model_memory")
    _reject_unknown(model_memory_raw, _MODEL_MEMORY_FIELDS, "model_memory")
    model_memory_enabled = model_memory_raw.get("enabled", False)
    if not isinstance(model_memory_enabled, bool):
        raise ConfigurationError("model_memory.enabled must be boolean")
    model_memory_path_raw = model_memory_raw.get("path", "model-memory.sqlite3")
    if not isinstance(model_memory_path_raw, str) or not model_memory_path_raw.strip():
        raise ConfigurationError("model_memory.path must be a non-empty relative path")
    model_memory_path = _resolve_inside(config_path.parent, model_memory_path_raw, "model_memory.path")

    result = ExternalGateConfig(
        path=config_path,
        model_map_path=model_map_path,
        memory=MemoryConfig(
            ram_budget_bytes=ram,
            max_resident_experts=slots,
            kv_cache_bytes=kv,
            scratch_bytes=scratch,
            transport=transport,
            lease_timeout_seconds=float(lease_timeout),
            budget_mode=budget_mode,
        ),
        io=IoConfig(workers=workers, prefetch_queue=queue_depth, integrity=integrity),
        server=ServerConfig(host=host.strip(), port=port, api_token=token, max_request_bytes=max_request),
        telemetry_enabled=telemetry_enabled,
        model_memory=ModelMemoryConfig(enabled=model_memory_enabled, path=model_memory_path),
    )
    if model_map is not None:
        validate_feasibility(result, model_map)
    return result


def validate_feasibility(config: ExternalGateConfig, model_map: PhysicalModelMap) -> None:
    capacity = config.memory.cache_capacity_bytes
    if capacity <= 0:
        raise BudgetError("RAM budget is entirely consumed by KV cache and scratch reservations")
    minimum = model_map.core_bytes + model_map.largest_expert_bytes
    if capacity < minimum:
        raise BudgetError(
            f"cache capacity {capacity} cannot hold core ({model_map.core_bytes}) plus one largest expert "
            f"({model_map.largest_expert_bytes})"
        )
    if config.memory.max_resident_experts is not None and config.memory.max_resident_experts < 1:
        raise BudgetError("at least one expert slot is required")


def load_external_bundle(path: str | Path) -> tuple[ExternalGateConfig, PhysicalModelMap]:
    """Load the physical map before final budget validation.

    The two-stage load is necessary because expert-count mode derives a safe
    byte budget from the largest encoded expert in the physical map.
    """
    config_path = Path(path).resolve()
    try:
        root = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"cannot read configuration: {exc}") from exc
    if not isinstance(root, dict):
        raise ConfigurationError("root must be an object")
    raw_map = root.get("model_map")
    if not isinstance(raw_map, str) or not raw_map.strip():
        raise ConfigurationError("model_map must be a non-empty relative path")
    map_path = _resolve_inside(config_path.parent, raw_map, "model_map")
    model_map = load_model_map(map_path)
    return load_external_config(config_path, model_map), model_map
