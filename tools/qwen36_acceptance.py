"""Prepare and run the MiniMaxBrain 0.3 acceptance flow for Qwen3.6-35B-A3B.

This tool is intentionally strict. It accepts only the base qwen35moe 35B-A3B
topology (40 trunk layers, 256 routed experts, top-8) and refuses MTP-only
GGUF files or unsupported routed-expert layouts.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minimaxbrain.gguf import load_gguf
from minimaxbrain.gguf_moe import pack_moe_gguf, validate_moe_layout
from minimaxbrain.model_map import load_model_map

ARCH = "qwen35moe"
LAYERS = 40
EXPERTS = 256
ACTIVE_EXPERTS = 8
EXPERT_SUFFIXES = (
    ".ffn_down_exps.weight",
    ".ffn_gate_exps.weight",
    ".ffn_up_exps.weight",
)


def _require_qwen36_base(gguf: Path) -> dict[str, int | str]:
    model = load_gguf(gguf)
    if model.architecture != ARCH:
        raise ValueError(
            f"expected architecture {ARCH!r}, got {model.architecture!r}"
        )

    def require_int(key: str) -> int:
        value = model.metadata.get(key)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"GGUF metadata {key!r} is missing or not an integer")
        return value

    layers = require_int(f"{ARCH}.block_count")
    experts = require_int(f"{ARCH}.expert_count")
    active = require_int(f"{ARCH}.expert_used_count")

    if (layers, experts, active) != (LAYERS, EXPERTS, ACTIVE_EXPERTS):
        raise ValueError(
            "GGUF is qwen35moe but not the expected Qwen3.6-35B-A3B base topology: "
            f"layers={layers}, experts={experts}, active={active}; "
            f"expected {LAYERS}/{EXPERTS}/{ACTIVE_EXPERTS}"
        )

    routed = [t for t in model.tensors if any(t.name.endswith(s) for s in EXPERT_SUFFIXES)]
    expected_routed = LAYERS * len(EXPERT_SUFFIXES)
    if len(routed) != expected_routed:
        raise ValueError(
            f"expected {expected_routed} routed expert tensors "
            f"({LAYERS} layers x down/gate/up), found {len(routed)}. "
            "Use the base model GGUF, not an MTP-only GGUF."
        )

    unsupported = [
        t.name for t in model.tensors
        if "_exps" in t.name and not any(t.name.endswith(s) for s in EXPERT_SUFFIXES)
    ]
    if unsupported:
        sample = ", ".join(unsupported[:3])
        raise ValueError(f"unsupported routed expert tensor layout: {sample}")

    return {
        "architecture": model.architecture,
        "layers": layers,
        "experts": experts,
        "active_experts": active,
        "file_bytes": gguf.stat().st_size,
    }


def _ensure_space(source: Path, output: Path) -> None:
    # Conversion stores all raw weights once in MMBW plus metadata/alignment.
    # Require a conservative source-size + 2 GiB free margin.
    parent = output.parent.resolve()
    parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(parent).free
    required = source.stat().st_size + (2 << 30)
    if free < required:
        raise OSError(
            f"insufficient free space in {parent}: "
            f"need at least {required / (1 << 30):.1f} GiB, "
            f"have {free / (1 << 30):.1f} GiB"
        )


def _validate_bundle(bundle: Path) -> None:
    model_map = load_model_map(bundle / "model.mmb-map.json")
    layout = validate_moe_layout(model_map)
    if model_map.architecture != ARCH:
        raise ValueError(
            f"bundle architecture is {model_map.architecture!r}, expected {ARCH!r}"
        )
    if (
        layout["layer_count"] != LAYERS
        or layout["expert_count"] != EXPERTS
        or layout["active_experts_per_token"] != ACTIVE_EXPERTS
    ):
        raise ValueError("bundle topology does not match Qwen3.6-35B-A3B")
    metadata = bundle / "model.mmb-meta.gguf"
    if not metadata.is_file():
        raise ValueError("bundle is missing model.mmb-meta.gguf")


def _run(cmd: list[str]) -> int:
    print("+", subprocess.list2cmdline(cmd), flush=True)
    return subprocess.run(cmd, cwd=ROOT).returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert/build/validate and run Qwen3.6-35B-A3B MMB 0.3 acceptance."
    )
    parser.add_argument("--gguf", required=True, type=Path, help="base Qwen3.6-35B-A3B GGUF")
    parser.add_argument("--bundle", type=Path, help="MMB bundle directory")
    parser.add_argument(
        "--convert",
        action="store_true",
        help="create --bundle from the GGUF before testing",
    )
    parser.add_argument("--build", action="store_true", help="build native backend before testing")
    parser.add_argument("--clean", action="store_true", help="clean native build (requires --build)")
    parser.add_argument("--build-dir", type=Path, default=ROOT / "native" / "build")
    parser.add_argument("--config", default="Release")
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--ctx", type=int, default=1024)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--cache-gib", type=float, default=1.0)
    parser.add_argument(
        "--prompt",
        default="Responda apenas com a palavra Brasil.",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="skip MMB block SHA-256 verification during acceptance only",
    )
    args = parser.parse_args(argv)

    gguf = args.gguf.resolve()
    if not gguf.is_file():
        parser.error(f"GGUF not found: {gguf}")
    if args.clean and not args.build:
        parser.error("--clean requires --build")
    if args.tokens <= 0 or args.ctx <= 0 or args.cache_gib <= 0:
        parser.error("--tokens, --ctx and --cache-gib must be positive")
    if args.threads is not None and args.threads <= 0:
        parser.error("--threads must be positive")

    try:
        info = _require_qwen36_base(gguf)
    except (ValueError, OSError) as exc:
        print(f"QWEN36_PREFLIGHT_FAILED: {exc}", file=sys.stderr)
        return 2

    print(
        "QWEN36_PREFLIGHT_OK "
        f"architecture={info['architecture']} layers={info['layers']} "
        f"experts={info['experts']} active={info['active_experts']} "
        f"size_gib={int(info['file_bytes']) / (1 << 30):.2f}",
        flush=True,
    )

    bundle = (
        args.bundle.resolve()
        if args.bundle
        else gguf.with_name(gguf.stem + "-mmbw")
    )

    if args.convert:
        if bundle.exists():
            print(
                f"CONVERT_REFUSED: output already exists: {bundle}. "
                "Remove it explicitly or choose another --bundle.",
                file=sys.stderr,
            )
            return 2
        try:
            _ensure_space(gguf, bundle)
            print(f"Converting to MMB 0.3: {bundle}", flush=True)
            pack_moe_gguf(gguf, bundle)
        except Exception as exc:
            # pack_moe_gguf uses domain-specific exceptions; keep the tool
            # self-contained and report the concrete conversion error.
            print(f"CONVERT_FAILED: {exc}", file=sys.stderr)
            return 2

    if not bundle.is_dir():
        print(
            f"BUNDLE_NOT_FOUND: {bundle}. Run again with --convert or provide --bundle.",
            file=sys.stderr,
        )
        return 2

    try:
        _validate_bundle(bundle)
    except Exception as exc:
        print(f"BUNDLE_INVALID: {exc}", file=sys.stderr)
        return 2
    print("BUNDLE_VALID", flush=True)

    acceptance = [
        sys.executable,
        str(ROOT / "tools" / "native_moe_acceptance.py"),
        "--gguf",
        str(gguf),
        "--bundle",
        str(bundle),
        "--tokens",
        str(args.tokens),
        "--ctx",
        str(args.ctx),
        "--cache-bytes",
        str(int(args.cache_gib * (1 << 30))),
        "--prompt",
        args.prompt,
        "--build-dir",
        str(args.build_dir.resolve()),
        "--config",
        args.config,
    ]
    if args.threads is not None:
        acceptance.extend(["--threads", str(args.threads)])
    if args.no_verify:
        acceptance.append("--no-verify")
    if args.build:
        acceptance.append("--build")
    if args.clean:
        acceptance.append("--clean")

    return _run(acceptance)


if __name__ == "__main__":
    raise SystemExit(main())
