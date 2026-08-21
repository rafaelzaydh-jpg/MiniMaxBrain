"""Exercise the external gate with deterministic routes over real MoE bytes."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from minimaxbrain import ExternalGate, load_external_bundle


def _route(token: int, layer: int, active: int, experts: int) -> list[int]:
    # Five is coprime to 32, so the standard Granite test has no duplicates.
    result = [((token * 7) + (layer * 3) + (index * 5)) % experts for index in range(active)]
    if len(set(result)) != active:
        raise ValueError("deterministic route produced duplicate experts")
    return result


def run(config_path: Path, *, tokens: int, prefetch: str = "next-layer") -> dict[str, object]:
    if prefetch not in {"none", "next-layer"}:
        raise ValueError("prefetch must be 'none' or 'next-layer'")
    config, model_map = load_external_bundle(config_path)
    layers = sorted({block.layer for block in model_map.expert_blocks if block.layer is not None})
    experts = 1 + max(block.expert for block in model_map.expert_blocks if block.expert is not None)
    active = min(8, experts)
    acquire_ms: list[float] = []
    active_payload_bytes = 0
    total_started = time.perf_counter()
    gate = ExternalGate(config, model_map)
    startup_started = time.perf_counter()
    gate.start()
    startup_seconds = time.perf_counter() - startup_started
    routing_started = time.perf_counter()
    try:
        for token in range(tokens):
            for position, layer in enumerate(layers):
                if prefetch == "next-layer":
                    window: list[dict[str, object]] = []
                    for distance, candidate_layer in enumerate(layers[position:position + 2]):
                        for index, expert in enumerate(_route(token, candidate_layer, active, experts)):
                            window.append({
                                "block_id": model_map.route_block(candidate_layer, expert).block_id,
                                "priority": distance * active + index,
                            })
                    gate.prefetch(window)
                current = _route(token, layer, active, experts)
                lease = gate.acquire_routes(
                    [{"layer": layer, "expert": expert} for expert in current],
                    request_id=f"real-moe-token-{token}-layer-{layer}",
                )
                acquire_ms.append(float(lease["duration_ms"]))
                active_payload_bytes += sum(int(block["length"]) for block in lease["blocks"])
                gate.release(lease["lease_id"])
        snapshot = gate.snapshot()
    finally:
        gate.close()
    routing_seconds = time.perf_counter() - routing_started
    total_seconds = time.perf_counter() - total_started
    compact_memory = dict(snapshot["memory"])
    compact_memory.pop("blocks", None)
    compact_snapshot = {
        "model": snapshot["model"],
        "memory": compact_memory,
        "io": snapshot["io"],
        "routing": snapshot["routing"],
    }
    sorted_ms = sorted(acquire_ms)
    p95_index = max(0, min(len(sorted_ms) - 1, int(len(sorted_ms) * 0.95) - 1))
    return {
        "ok": True,
        "test_contract": "deterministic synthetic routes over real encoded expert weights",
        "prefetch": prefetch,
        "tokens": tokens,
        "layers": len(layers),
        "experts_per_layer": experts,
        "active_experts_per_layer": active,
        "route_acquires": len(acquire_ms),
        "expert_block_acquires": len(acquire_ms) * active,
        "active_payload_bytes": active_payload_bytes,
        "startup_seconds": round(startup_seconds, 6),
        "routing_seconds": round(routing_seconds, 6),
        "elapsed_seconds": round(total_seconds, 6),
        "acquire_ms": {
            "mean": round(statistics.mean(acquire_ms), 3),
            "median": round(statistics.median(acquire_ms), 3),
            "p95": round(sorted_ms[p95_index], 3),
            "max": round(max(acquire_ms), 3),
        },
        "final_snapshot": compact_snapshot,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=2)
    parser.add_argument("--prefetch", choices=("none", "next-layer"), default="next-layer")
    args = parser.parse_args()
    if args.tokens < 1:
        parser.error("--tokens must be >= 1")
    print(json.dumps(
        run(args.config.resolve(), tokens=args.tokens, prefetch=args.prefetch),
        ensure_ascii=False,
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
