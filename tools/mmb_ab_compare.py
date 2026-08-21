"""Alternating A/B benchmark: compulsory-only I/O versus exact MiniMaxBrain prefetch."""
from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mmb_real_moe_smoke import run


def _summary(results: list[dict[str, object]]) -> dict[str, object]:
    routing = [float(item["routing_seconds"]) for item in results]
    startup = [float(item["startup_seconds"]) for item in results]
    acquire = [float(item["acquire_ms"]["mean"]) for item in results]  # type: ignore[index]
    bytes_read = [int(item["final_snapshot"]["io"]["bytes_read"]) for item in results]  # type: ignore[index]
    evictions = [int(item["final_snapshot"]["memory"]["evictions"]) for item in results]  # type: ignore[index]
    dropped = [int(item["final_snapshot"]["io"]["prefetch_dropped"]) for item in results]  # type: ignore[index]
    return {
        "runs": len(results),
        "routing_seconds": {
            "median": round(statistics.median(routing), 6),
            "min": round(min(routing), 6),
            "max": round(max(routing), 6),
        },
        "startup_seconds_median": round(statistics.median(startup), 6),
        "mean_acquire_ms_median": round(statistics.median(acquire), 3),
        "bytes_read": {
            "median": int(statistics.median(bytes_read)),
            "observed": sorted(set(bytes_read)),
        },
        "evictions": {
            "median": int(statistics.median(evictions)),
            "observed": sorted(set(evictions)),
        },
        "prefetch_dropped": {
            "median": int(statistics.median(dropped)),
            "observed": sorted(set(dropped)),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()
    if args.tokens < 1 or args.rounds < 1:
        parser.error("--tokens and --rounds must be >= 1")
    config = args.config.resolve()
    grouped: dict[str, list[dict[str, object]]] = {"none": [], "next-layer": []}
    execution_order: list[str] = []
    for round_index in range(args.rounds):
        order = ("none", "next-layer") if round_index % 2 == 0 else ("next-layer", "none")
        for mode in order:
            grouped[mode].append(run(config, tokens=args.tokens, prefetch=mode))
            execution_order.append(mode)
            gc.collect()
    without = _summary(grouped["none"])
    with_help = _summary(grouped["next-layer"])
    without_seconds = float(without["routing_seconds"]["median"])  # type: ignore[index]
    with_seconds = float(with_help["routing_seconds"]["median"])  # type: ignore[index]
    print(json.dumps({
        "ok": True,
        "comparison": "same real blocks and routes; compulsory-only versus exact next-layer prefetch",
        "caveat": (
            "This is a physical paging A/B, not end-to-end token generation. Windows filesystem cache "
            "was not flushed. Exact prefetch is an upper bound equivalent to perfect route advice."
        ),
        "tokens_per_run": args.tokens,
        "rounds_per_mode": args.rounds,
        "execution_order": execution_order,
        "without_prefetch": without,
        "with_mmb_help": with_help,
        "effect": {
            "routing_speedup_x": round(without_seconds / with_seconds, 4),
            "routing_time_reduction_percent": round((1.0 - with_seconds / without_seconds) * 100.0, 2),
        },
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
