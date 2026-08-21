"""Create the small runtime config that sits next to an existing MMB bundle."""
from __future__ import annotations

import json
from pathlib import Path

from .gguf_moe import validate_moe_layout
from .model_map import load_model_map


def prepare_bundle_config(
    bundle_dir: str | Path,
    *,
    expert_cache_bytes: int = 1 << 30,
    kv_cache_bytes: int = 0,
    scratch_bytes: int = 0,
    overwrite: bool = False,
) -> Path:
    """Validate an MMB bundle and create ``gate.json`` without reconversion.

    Config v1 budgets core + expert cache together. The direct runtime later
    subtracts the physical core size and passes only ``expert_cache_bytes`` to
    the native pager.
    """
    bundle = Path(bundle_dir).resolve()
    if not bundle.is_dir():
        raise ValueError(f"MMB bundle directory not found: {bundle}")
    if expert_cache_bytes <= 0:
        raise ValueError("expert_cache_bytes must be positive")
    if kv_cache_bytes < 0 or scratch_bytes < 0:
        raise ValueError("kv_cache_bytes and scratch_bytes must be non-negative")

    map_path = bundle / "model.mmb-map.json"
    metadata_path = bundle / "model.mmb-meta.gguf"
    if not map_path.is_file():
        raise ValueError(f"bundle is missing {map_path.name}")
    if not metadata_path.is_file():
        raise ValueError(f"bundle is missing {metadata_path.name}")

    model_map = load_model_map(map_path)
    validate_moe_layout(model_map)
    if not model_map.expert_blocks:
        raise ValueError("bundle contains no routed expert blocks")
    if expert_cache_bytes < model_map.largest_expert_bytes:
        raise ValueError(
            "expert cache cannot hold the largest expert: "
            f"cache={expert_cache_bytes}, largest={model_map.largest_expert_bytes}"
        )

    config_path = bundle / "gate.json"
    if config_path.exists() and not overwrite:
        return config_path

    ram_budget = (
        int(model_map.core_bytes)
        + int(expert_cache_bytes)
        + int(kv_cache_bytes)
        + int(scratch_bytes)
    )
    config = {
        "schema_version": "mmb-external-gate-config-v1",
        "model_map": "model.mmb-map.json",
        "memory": {
            "ram_budget": ram_budget,
            "kv_cache": int(kv_cache_bytes),
            "scratch": int(scratch_bytes),
            "transport": "heap",
            "lease_timeout_seconds": 120.0,
        },
        "io": {
            "integrity": "first_load",
        },
        "server": {
            "host": "127.0.0.1",
            "port": 8080,
            "api_token": None,
            "max_request_bytes": 1 << 20,
        },
    }
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return config_path
