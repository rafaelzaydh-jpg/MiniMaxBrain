"""Exercise the normal direct MMB runtime without opening the original GGUF."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minimaxbrain.bundle_config import prepare_bundle_config
from minimaxbrain.runtime import InferenceMode, MMBRuntime


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Open an existing MMB bundle through mmb_backend, run one greedy "
            "chat turn and require real router/MMBW kernel activity."
        )
    )
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--cache-gib", type=float, default=1.0)
    parser.add_argument("--ctx", type=int, default=1024)
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--threads", type=int)
    parser.add_argument(
        "--prompt",
        default="Responda apenas com a palavra Brasil.",
    )
    args = parser.parse_args(argv)

    if args.cache_gib <= 0 or args.ctx < 128 or args.tokens <= 0:
        parser.error("--cache-gib and --tokens must be positive; --ctx must be >= 128")
    if args.threads is not None and args.threads <= 0:
        parser.error("--threads must be positive")

    bundle = args.bundle.resolve()
    config_path = prepare_bundle_config(
        bundle,
        expert_cache_bytes=int(args.cache_gib * (1 << 30)),
    )

    with MMBRuntime(
        config_path,
        n_ctx=args.ctx,
        n_threads=args.threads,
    ) as runtime:
        if not runtime.ready:
            print(
                f"DIRECT_RUNTIME_FAILED: {runtime.backend_error or 'backend unavailable'}",
                file=sys.stderr,
            )
            return 2
        if runtime.inference_mode is not InferenceMode.PAGED_MMB:
            print(
                f"DIRECT_RUNTIME_FAILED: unexpected mode {runtime.inference_mode.value}",
                file=sys.stderr,
            )
            return 3

        output: list[str] = []
        for piece, _ in runtime.stream_chat(
            [{"role": "user", "content": args.prompt}],
            max_tokens=args.tokens,
            temperature=0.0,
            top_p=1.0,
            top_k=0,
        ):
            output.append(piece)

        stats = runtime.stats()
        paged = bool(stats.get("paged_experts_used"))
        native = stats.get("native_pager") or {}
        router_requests = int(native.get("real_router_requests", 0))
        bytes_read = int(native.get("bytes_read", 0))

        print(f"inference_mode={runtime.inference_mode.value}")
        print(f"paged_kernel_used={'true' if paged else 'false'}")
        print(f"router_requests={router_requests}")
        print(f"bytes_read={bytes_read}")
        print(f"cache_hits={int(native.get('cache_hits', 0))}")
        print(f"cache_misses={int(native.get('cache_misses', 0))}")
        print(f"resident_bytes={int(native.get('resident_bytes', 0))}")
        print(f"peak_resident_bytes={int(native.get('peak_resident_bytes', 0))}")
        print("direct_text_begin")
        print("".join(output))
        print("direct_text_end")

        if not paged or router_requests <= 0 or bytes_read <= 0:
            print(
                "DIRECT_RUNTIME_FAILED: generation did not prove MMB-backed MoE activity",
                file=sys.stderr,
            )
            return 4

        print("DIRECT_RUNTIME_OK")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
