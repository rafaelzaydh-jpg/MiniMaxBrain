"""Command-line surface for the independent MiniMaxBrain external gate."""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, Sequence

from .config import load_external_bundle
from .errors import MMBError
from .external import ExternalGate
from .gguf import gguf_summary, load_gguf
from .gguf_moe import pack_moe_gguf
from .model_map import load_model_map, public_map_summary
from .packer import pack_from_plan
from .server import serve_gate
from .storage import create_model_seal, verify_model_seal
from .units import format_bytes, parse_bytes, parse_count


def _json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _inspect(args: argparse.Namespace) -> int:
    model_map = load_model_map(args.model_map)
    result = public_map_summary(model_map)
    result["human"] = {
        "stored": format_bytes(model_map.stored_bytes),
        "core": format_bytes(model_map.core_bytes),
        "largest_expert": format_bytes(model_map.largest_expert_bytes),
    }
    _json(result)
    return 0


def _gguf_inspect(args: argparse.Namespace) -> int:
    _json(gguf_summary(load_gguf(args.gguf)))
    return 0


def _gguf_pack_moe(args: argparse.Namespace) -> int:
    manifest_path = pack_moe_gguf(args.gguf, args.output, alignment=args.alignment)
    _json({"ok": True, "model_map": str(manifest_path)})
    return 0


def _seal(args: argparse.Namespace) -> int:
    config, model_map = load_external_bundle(args.config)
    seal_data = create_model_seal(model_map)
    _json({
        "ok": True,
        "seal": {
            "model_id": seal_data["model_id"],
            "map_revision": seal_data["map_revision"],
            "verified_at": seal_data["verified_at"],
            "total_blocks": seal_data["total_blocks"],
            "shards": list(seal_data["shards"].keys()),
        },
    })
    return 0


def _check(args: argparse.Namespace) -> int:
    config, model_map = load_external_bundle(args.config)
    sealed, seal_reason = verify_model_seal(model_map)
    result = {
        "ok": True,
        "model": public_map_summary(model_map),
        "integrity": {
            "configured_mode": config.io.integrity,
            "sealed": sealed,
            "seal_status": "valid" if sealed else (seal_reason or "not sealed"),
        },
        "budget": {
            "mode": config.memory.budget_mode,
            "ram_budget_bytes": config.memory.ram_budget_bytes,
            "cache_capacity_bytes": config.memory.cache_capacity_bytes,
            "kv_cache_bytes": config.memory.kv_cache_bytes,
            "scratch_bytes": config.memory.scratch_bytes,
            "max_resident_experts": config.memory.max_resident_experts,
        },
        "model_memory": {
            "enabled": config.model_memory.enabled,
            "path": str(config.model_memory.path),
        },
    }
    _json(result)
    return 0


def _smoke(args: argparse.Namespace) -> int:
    config, model_map = load_external_bundle(args.config)
    with ExternalGate(config, model_map) as gate:
        candidates = [item.block_id for item in model_map.expert_blocks[: max(1, args.blocks)]]
        if not candidates:
            candidates = [item.block_id for item in model_map.core_blocks[:1]]
        if not candidates:
            raise RuntimeError("model map contains no blocks")
        gate.prefetch([{"block_id": block_id, "priority": index} for index, block_id in enumerate(candidates)])
        acquired = gate.acquire(candidates, request_id="cli-smoke")
        gate.release(acquired["lease_id"])
        result = gate.snapshot()
        result["smoke"] = {"ok": True, "blocks": candidates}
        _json(result)
    return 0


def _serve(args: argparse.Namespace) -> int:
    config, model_map = load_external_bundle(args.config)
    gate = ExternalGate(config, model_map)
    print(
        f"MiniMaxBrain External Gate listening on {config.server.host}:{config.server.port} "
        f"for {model_map.model_id}",
        file=sys.stderr,
        flush=True,
    )
    serve_gate(gate)
    return 0


def _pack(args: argparse.Namespace) -> int:
    manifest_path = pack_from_plan(args.plan, args.output)
    _json({"ok": True, "model_map": str(manifest_path)})
    return 0


def _estimate(args: argparse.Namespace) -> int:
    parameters = parse_count(args.parameters, field="parameters")
    expert_fraction = float(args.expert_fraction)
    if not 0 < expert_fraction <= 1:
        raise ValueError("--expert-fraction must be in (0, 1]")
    bits = float(args.bits)
    if not 0 < bits <= 32:
        raise ValueError("--bits must be in (0, 32]")
    blocks = int(args.expert_blocks)
    active = int(args.cold_blocks_per_token)
    if blocks < 1 or active < 0:
        raise ValueError("expert block counts must be non-negative and total must be positive")
    overhead = float(args.encoding_overhead)
    bandwidth = parse_bytes(args.storage_bandwidth, field="storage_bandwidth")
    encoded = int(parameters * bits / 8.0 * overhead)
    expert_bytes = int(encoded * expert_fraction)
    shared_bytes = encoded - expert_bytes
    average_block = expert_bytes // blocks
    cold_bytes = average_block * active
    _json({
        "assumptions": {
            "parameters": parameters,
            "bits_per_weight": bits,
            "encoding_overhead_multiplier": overhead,
            "expert_parameter_fraction": expert_fraction,
            "expert_blocks": blocks,
            "cold_expert_blocks_read_per_token": active,
            "storage_bandwidth_bytes_per_second": bandwidth,
        },
        "physical_lower_bound": {
            "encoded_model_bytes": encoded,
            "shared_non_expert_bytes": shared_bytes,
            "average_expert_block_bytes": average_block,
            "cold_bytes_per_token": cold_bytes,
            "io_seconds_per_token": None if bandwidth == 0 else round(cold_bytes / bandwidth, 6),
            "io_tokens_per_second_ceiling": None if cold_bytes == 0 else round(bandwidth / cold_bytes, 6),
        },
        "warning": (
            "This is an I/O lower bound, not a semantic model estimate. A normal MoE may require "
            "multiple expert blocks per layer and many layers per token. Core, KV, scratch, compute, "
            "random-I/O latency, and cache misses must be added."
        ),
    })
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mmb", description="MiniMaxBrain independent external weight gate")
    sub = parser.add_subparsers(dest="command", required=True)
    inspect_parser = sub.add_parser("inspect", help="validate and summarize a physical model map")
    inspect_parser.add_argument("--model-map", required=True)
    inspect_parser.set_defaults(handler=_inspect)
    gguf_parser = sub.add_parser("gguf-inspect", help="inspect tensor ranges in a real GGUF model")
    gguf_parser.add_argument("--gguf", required=True)
    gguf_parser.set_defaults(handler=_gguf_inspect)
    gguf_pack_parser = sub.add_parser(
        "gguf-pack-moe",
        help="split MoE GGUF expert tensors into MiniMaxBrain pageable blocks (GraniteMoE, Qwen MoE)",
    )
    gguf_pack_parser.add_argument("--gguf", required=True)
    gguf_pack_parser.add_argument("--output", required=True)
    gguf_pack_parser.add_argument("--alignment", type=int, default=4096)
    gguf_pack_parser.set_defaults(handler=_gguf_pack_moe)
    seal_parser = sub.add_parser("seal", help="verify all blocks and write a tamper-proof integrity seal")
    seal_parser.add_argument("--config", required=True)
    seal_parser.set_defaults(handler=_seal)
    check_parser = sub.add_parser("check", help="validate a complete gate configuration and seal status")
    check_parser.add_argument("--config", required=True)
    check_parser.set_defaults(handler=_check)
    smoke_parser = sub.add_parser("smoke", help="physically prefetch, acquire, and release blocks")
    smoke_parser.add_argument("--config", required=True)
    smoke_parser.add_argument("--blocks", type=int, default=1)
    smoke_parser.set_defaults(handler=_smoke)
    serve_parser = sub.add_parser("serve", help="run the independent local IPC service")
    serve_parser.add_argument("--config", required=True)
    serve_parser.set_defaults(handler=_serve)
    pack_parser = sub.add_parser("pack", help="stream converter-produced blocks into aligned pageable shards")
    pack_parser.add_argument("--plan", required=True)
    pack_parser.add_argument("--output", required=True)
    pack_parser.set_defaults(handler=_pack)
    estimate_parser = sub.add_parser("estimate", help="calculate a transparent SSD-I/O lower bound")
    estimate_parser.add_argument("--parameters", required=True, help="for example 2T")
    estimate_parser.add_argument("--bits", type=float, default=4.0)
    estimate_parser.add_argument("--expert-fraction", type=float, default=0.95)
    estimate_parser.add_argument("--expert-blocks", type=int, required=True)
    estimate_parser.add_argument("--cold-blocks-per-token", type=int, required=True)
    estimate_parser.add_argument("--storage-bandwidth", default="7GB")
    estimate_parser.add_argument("--encoding-overhead", type=float, default=1.08)
    estimate_parser.set_defaults(handler=_estimate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (MMBError, ValueError, OSError) as exc:
        code = getattr(exc, "code", "INVALID_ARGUMENT")
        print(f"{code}: {exc}", file=sys.stderr)
        return 2
