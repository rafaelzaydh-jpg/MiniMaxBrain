"""Cross-process shared-memory validation against real pageable MoE blocks."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from minimaxbrain import ExternalGateClient, load_external_bundle


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config, model_map = load_external_bundle(args.config.resolve())
    client = ExternalGateClient(
        config.server.host,
        config.server.port,
        api_token=config.server.api_token,
    )
    hello = client.hello()
    expert_ids = [(index * 5) % 32 for index in range(8)]
    blocks = [model_map.route_block(0, expert).block_id for expert in expert_ids]
    prefetch = client.prefetch(
        [{"block_id": block_id, "priority": index} for index, block_id in enumerate(blocks)],
        request_id="real-ipc-prefetch",
    )
    verified: list[dict[str, object]] = []
    with client.acquire_routes(
        [{"layer": 0, "expert": expert} for expert in expert_ids],
        request_id="real-ipc-acquire",
    ) as lease:
        for block in lease.blocks:
            view = block.view()
            try:
                actual = hashlib.sha256(view).hexdigest()
            finally:
                view.release()
            expected = str(block.descriptor["sha256"])
            if actual != expected:
                raise RuntimeError(f"shared memory digest mismatch for {block.block_id}")
            verified.append({
                "block_id": block.block_id,
                "bytes": int(block.descriptor["length"]),
                "sha256": actual,
            })
    stats = client.stats()
    memory = dict(stats["memory"])
    memory.pop("blocks", None)
    print(json.dumps({
        "ok": True,
        "hello": hello,
        "prefetch": prefetch,
        "mapped_and_sha256_verified": verified,
        "memory": memory,
        "io": stats["io"],
        "routing": stats["routing"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
