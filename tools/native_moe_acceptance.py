"""Run the same-process GGUF vs MMBW MoE acceptance test."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _candidate_binaries(build_dir: Path, config: str) -> list[Path]:
    exe = "mmb_llama_acceptance.exe" if os.name == "nt" else "mmb_llama_acceptance"
    return [
        build_dir / config / exe,
        build_dir / exe,
        build_dir / "bin" / config / exe,
        build_dir / "bin" / exe,
    ]


def _find_binary(build_dir: Path, config: str) -> Path | None:
    for candidate in _candidate_binaries(build_dir, config):
        if candidate.is_file():
            return candidate
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare greedy llama.cpp GGUF inference against the MMB 0.3 "
            "metadata+core-MMBW+expert-MMBW loader in the same process."
        )
    )
    parser.add_argument("--gguf", required=True, type=Path)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--prompt", default="Hello")
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--cache-bytes", type=int, default=512 * 1024 * 1024)
    parser.add_argument("--ctx", type=int, default=2048)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument("--build", action="store_true", help="build native targets before running")
    parser.add_argument("--clean", action="store_true", help="clean native build before --build")
    parser.add_argument("--build-dir", type=Path, default=ROOT / "native" / "build")
    parser.add_argument("--config", default="Release")
    args = parser.parse_args(argv)

    if args.tokens <= 0 or args.cache_bytes <= 0 or args.ctx <= 0:
        parser.error("--tokens, --cache-bytes and --ctx must be positive")
    if args.threads is not None and args.threads <= 0:
        parser.error("--threads must be positive")
    if not args.gguf.is_file():
        parser.error(f"GGUF not found: {args.gguf}")
    if not args.bundle.is_dir():
        parser.error(f"MMB bundle directory not found: {args.bundle}")

    build_dir = args.build_dir.resolve()
    if args.build:
        cmd = [
            sys.executable,
            str(ROOT / "tools" / "build_native.py"),
            "--build-dir",
            str(build_dir),
            "--config",
            args.config,
        ]
        if args.clean:
            cmd.append("--clean")
        subprocess.run(cmd, check=True)

    binary = _find_binary(build_dir, args.config)
    if binary is None:
        candidates = "\n  ".join(str(p) for p in _candidate_binaries(build_dir, args.config))
        print(
            "mmb_llama_acceptance was not found. Build it with:\n"
            "  python tools/build_native.py --clean\n"
            f"Checked:\n  {candidates}",
            file=sys.stderr,
        )
        return 2

    cmd = [
        str(binary),
        "--gguf",
        str(args.gguf.resolve()),
        "--bundle",
        str(args.bundle.resolve()),
        "--prompt",
        args.prompt,
        "--tokens",
        str(args.tokens),
        "--cache-bytes",
        str(args.cache_bytes),
        "--ctx",
        str(args.ctx),
    ]
    if args.threads is not None:
        cmd.extend(["--threads", str(args.threads)])
    if args.no_verify:
        cmd.append("--no-verify")

    return subprocess.run(cmd).returncode


if __name__ == "__main__":
    raise SystemExit(main())
