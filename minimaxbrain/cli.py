"""Command line for the minimal, fail-closed MiniMaxBrain runtime."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from .config import load_external_bundle
from .bundle_config import prepare_bundle_config
from .errors import IntegrityError, MMBError
from .external import ExternalGate
from .gguf import gguf_summary, load_gguf
from .gguf_moe import pack_moe_gguf, validate_moe_layout
from .model_map import load_model_map, public_map_summary
from .native import NativePager, find_native_backend
from .storage import create_model_seal, verify_model_seal
from .units import format_bytes


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


def _convert(args: argparse.Namespace) -> int:
    manifest_path = pack_moe_gguf(args.gguf, args.output, alignment=args.alignment)
    config_path = prepare_bundle_config(
        Path(args.output),
        expert_cache_bytes=int(float(args.cache_gib) * (1 << 30)),
    )
    _json({
        "ok": True,
        "model_map": str(manifest_path),
        "config": str(config_path),
    })
    return 0


def _prepare(args: argparse.Namespace) -> int:
    config_path = prepare_bundle_config(
        args.bundle,
        expert_cache_bytes=int(float(args.cache_gib) * (1 << 30)),
        overwrite=bool(args.force),
    )
    _json({"ok": True, "config": str(config_path)})
    return 0


def _seal(args: argparse.Namespace) -> int:
    _config, model_map = load_external_bundle(args.config)
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
    layout = validate_moe_layout(model_map)
    sealed, seal_reason = verify_model_seal(model_map)
    if config.io.integrity == "seal" and not sealed:
        raise IntegrityError(
            f"model seal verification failed: {seal_reason}. Run 'mmb seal' first."
        )

    native_validation: dict[str, Any] = {
        "available": find_native_backend() is not None,
        "validated": False,
    }
    if native_validation["available"] and model_map.expert_blocks:
        expert_budget = config.memory.cache_capacity_bytes - model_map.core_bytes
        if expert_budget < model_map.largest_expert_bytes:
            raise IntegrityError(
                "configured cache budget cannot fit one expert after reserving core tensors"
            )
        with NativePager(
            model_map.path.parent,
            expert_budget,
            verify_sha256=config.io.integrity != "none",
        ) as pager:
            native_validation.update({
                "validated": True,
                "model_info": pager.model_info(),
                "capabilities": pager.stats()["capabilities"],
            })

    _json({
        "ok": True,
        "model": public_map_summary(model_map),
        "layout": layout,
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
        "runtime": {
            "native_pager": native_validation,
            "paged_moe_executor": bool(
                native_validation.get("capabilities", {}).get("paged_moe_kernel_available", False)
            ),
        },
    })
    return 0


def _smoke(args: argparse.Namespace) -> int:
    """Exercise only the physical pager. This is not an inference benchmark."""
    config, model_map = load_external_bundle(args.config)
    expert_blocks = list(model_map.expert_blocks[: max(1, int(args.blocks))])
    if expert_blocks and find_native_backend() is not None:
        expert_budget = config.memory.cache_capacity_bytes - model_map.core_bytes
        with NativePager(
            model_map.path.parent,
            expert_budget,
            verify_sha256=config.io.integrity != "none",
        ) as pager:
            acquired_routes = []
            for block in expert_blocks:
                assert block.layer is not None and block.expert is not None
                with pager.acquire(block.layer, [block.expert]) as lease:
                    acquired_routes.append({
                        "layer": block.layer,
                        "expert": block.expert,
                        "segments": {
                            role: lease.segment(0, role).bytes
                            for role in ("down", "gate", "up")
                        },
                    })
            _json({
                "ok": True,
                "kind": "native_physical_pager_smoke",
                "inference": False,
                "routes": acquired_routes,
                "native": pager.stats(),
            })
        return 0

    # Portable Python pager remains a reference/fallback for bundle validation.
    with ExternalGate(config, model_map) as gate:
        candidates = [item.block_id for item in expert_blocks]
        if not candidates:
            candidates = [item.block_id for item in model_map.core_blocks[:1]]
        if not candidates:
            raise RuntimeError("model map contains no blocks")
        acquired = gate.acquire(candidates, request_id="cli-smoke")
        gate.release(acquired["lease_id"])
        snapshot = gate.snapshot()
        _json({
            "ok": True,
            "kind": "python_physical_pager_smoke",
            "inference": False,
            "blocks": candidates,
            "memory": snapshot["memory"],
            "io": snapshot["io"],
            "routing": snapshot["routing"],
        })
    return 0


def _runtime_config_path(args: argparse.Namespace) -> Path:
    if getattr(args, "config", None):
        return Path(args.config).resolve()
    bundle = Path(args.bundle).resolve()
    return prepare_bundle_config(
        bundle,
        expert_cache_bytes=int(float(getattr(args, "cache_gib", 1.0)) * (1 << 30)),
    )


def _create_runtime(args: argparse.Namespace):
    from .runtime import MMBRuntime

    return MMBRuntime(
        _runtime_config_path(args),
        n_ctx=getattr(args, "ctx", 2048),
        n_threads=getattr(args, "threads", None),
    )


def _web(args: argparse.Namespace) -> int:
    from .server_http import start_mmb_server
    import webbrowser

    engine = _create_runtime(args)
    host = args.host
    port = int(args.port)

    print("=" * 65)
    print("MiniMaxBrain — Web/API")
    print("=" * 65)
    print(f"Modelo:  {engine.model_map.model_id}")
    print(f"Backend: {engine.inference_mode.value}")
    print(f"Status:  {'ready' if engine.ready else 'not_ready'}")
    if engine.backend_error:
        print(f"Motivo:  {engine.backend_error}")
    print(f"Web:     http://{host}:{port}/")
    print(f"Health:  http://{host}:{port}/health")
    print(f"API:     http://{host}:{port}/v1/chat/completions")
    print("=" * 65)

    server = start_mmb_server(engine, host=host, port=port)
    if args.open_browser:
        try:
            webbrowser.open(f"http://{host}:{port}/")
        except Exception:
            pass

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nEncerrando...")
    finally:
        server.shutdown()
        server.server_close()
        engine.close()
    return 0


def _chat(args: argparse.Namespace) -> int:
    engine = _create_runtime(args)
    if not engine.ready:
        print(
            f"BACKEND_UNAVAILABLE: {engine.backend_error or 'backend de inferencia indisponivel'}",
            file=sys.stderr,
        )
        engine.close()
        return 2

    print("=" * 65)
    print("MiniMaxBrain — Chat (inferencia real)")
    print("=" * 65)
    print(f"Modelo:  {engine.model_map.model_id}")
    print(f"Backend: {engine.inference_mode.value}")
    print("Digite sua pergunta. ('sair' encerra)")
    print("-" * 65)

    messages: list[dict[str, str]] = []
    try:
        while True:
            prompt = input("\nVoce: ").strip()
            if not prompt:
                continue
            if prompt.lower() in {"sair", "exit", "quit", "q"}:
                break

            messages.append({"role": "user", "content": prompt})
            print("IA: ", end="", flush=True)
            response: list[str] = []
            for chunk, _stats in engine.stream_chat(
                messages,
                max_tokens=args.tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
            ):
                print(chunk, end="", flush=True)
                response.append(chunk)
            messages.append({"role": "assistant", "content": "".join(response)})
            print()
    except (KeyboardInterrupt, EOFError):
        print("\nSessao encerrada.")
    finally:
        engine.close()
    return 0


def _add_runtime_args(parser: argparse.ArgumentParser) -> None:
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", help="gate.json of the converted bundle")
    source.add_argument(
        "--bundle",
        help="MMB bundle directory; gate.json is created automatically if absent",
    )
    parser.add_argument(
        "--cache-gib",
        type=float,
        default=1.0,
        help="expert cache GiB when --bundle needs a new gate.json (default: 1)",
    )
    parser.add_argument("--ctx", type=int, default=2048, help="context size")
    parser.add_argument("--threads", type=int, help="CPU thread count")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mmb",
        description="MiniMaxBrain 0.3: direct paged-MMB llama.cpp runtime",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("inspect", help="validate and summarize a physical model map")
    p.add_argument("--model-map", required=True)
    p.set_defaults(handler=_inspect)

    p = sub.add_parser("gguf-inspect", help="inspect a GGUF before conversion")
    p.add_argument("--gguf", required=True)
    p.set_defaults(handler=_gguf_inspect)

    p = sub.add_parser("convert", help="convert a supported MoE GGUF into an MMB bundle")
    p.add_argument("--gguf", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--alignment", type=int, default=4096)
    p.add_argument("--cache-gib", type=float, default=1.0)
    p.set_defaults(handler=_convert)

    p = sub.add_parser(
        "prepare",
        help="create gate.json for an already converted MMB bundle",
    )
    p.add_argument("--bundle", required=True)
    p.add_argument("--cache-gib", type=float, default=1.0)
    p.add_argument("--force", action="store_true")
    p.set_defaults(handler=_prepare)

    p = sub.add_parser("seal", help="hash every block and write model.verified.json")
    p.add_argument("--config", required=True)
    p.set_defaults(handler=_seal)

    p = sub.add_parser("check", help="validate bundle, budget and seal")
    p.add_argument("--config", required=True)
    p.set_defaults(handler=_check)

    p = sub.add_parser(
        "smoke",
        help="exercise SSD->RAM pager only (not an inference benchmark)",
    )
    p.add_argument("--config", required=True)
    p.add_argument("--blocks", type=int, default=1)
    p.set_defaults(handler=_smoke)

    p = sub.add_parser("chat", help="terminal chat through real llama.cpp inference")
    _add_runtime_args(p)
    p.add_argument("--tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--top-k", type=int, default=40)
    p.set_defaults(handler=_chat)

    p = sub.add_parser("web", help="Web UI and OpenAI-compatible API")
    _add_runtime_args(p)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--open-browser", action="store_true")
    p.set_defaults(handler=_web)

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
