"""Build MiniMaxBrain native backend and the pinned llama.cpp integration."""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _parse_set_output(stdout: str) -> dict[str, str]:
    env = os.environ.copy()
    for line in stdout.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key:
            env[key] = value
    return env


def _capture_from_batch(batch: Path, *, vsdevcmd: bool) -> dict[str, str] | None:
    """Run a VS environment batch file and return an environment containing cl.exe."""
    if not batch.is_file():
        return None

    if vsdevcmd:
        command = f'call "{batch}" -arch=x64 >nul && set'
    else:
        command = f'call "{batch}" >nul && set'

    captured = subprocess.run(
        ["cmd.exe", "/d", "/s", "/c", command],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if captured.returncode != 0:
        return None

    env = _parse_set_output(captured.stdout)
    if shutil.which("cl", path=env.get("PATH")) is None:
        return None
    return env


def _visual_studio_installations() -> list[Path]:
    """Discover VS installations without depending on a specific component ID."""
    roots: list[Path] = []

    # Prefer vswhere, but do not filter by VC component ID. New/Insiders
    # installations can expose different component metadata even though the
    # actual VC toolchain is installed and usable.
    vswhere_candidates: list[Path] = []
    for env_name in ("ProgramFiles(x86)", "ProgramFiles"):
        base = os.environ.get(env_name)
        if base:
            vswhere_candidates.append(
                Path(base) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
            )

    vswhere = next((path for path in vswhere_candidates if path.is_file()), None)
    if vswhere is not None:
        query = [
            str(vswhere),
            "-all",
            "-prerelease",
            "-products",
            "*",
            "-property",
            "installationPath",
        ]
        result = subprocess.run(
            query,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                line = line.strip()
                if line:
                    roots.append(Path(line))

    # Robust fallback for VS 2022/2026/Insiders layouts.
    for env_name in ("ProgramFiles", "ProgramFiles(x86)"):
        base = os.environ.get(env_name)
        if not base:
            continue
        vs_root = Path(base) / "Microsoft Visual Studio"
        if not vs_root.is_dir():
            continue
        for version_dir in vs_root.iterdir():
            if not version_dir.is_dir() or version_dir.name == "Installer":
                continue
            for edition_dir in version_dir.iterdir():
                if edition_dir.is_dir():
                    roots.append(edition_dir)

    # Preserve discovery order but remove duplicates case-insensitively.
    unique: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = os.path.normcase(os.path.normpath(str(root)))
        if key not in seen:
            seen.add(key)
            unique.append(root)
    return unique


def _capture_msvc_environment() -> tuple[dict[str, str] | None, Path | None]:
    """Return an x64 MSVC environment and the batch file used to create it."""
    if os.name != "nt":
        return None, None

    # If the caller is already inside a Developer Command Prompt, preserve it.
    if shutil.which("cl") is not None:
        return os.environ.copy(), None

    for install in _visual_studio_installations():
        candidates = [
            (install / "Common7" / "Tools" / "VsDevCmd.bat", True),
            (install / "VC" / "Auxiliary" / "Build" / "vcvars64.bat", False),
        ]
        for batch, is_vsdevcmd in candidates:
            env = _capture_from_batch(batch, vsdevcmd=is_vsdevcmd)
            if env is not None:
                return env, batch

    return None, None


def run(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    print("+", subprocess.list2cmdline(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build MiniMaxBrain native backend")
    parser.add_argument("--build-dir", default=str(ROOT / "native" / "build"))
    parser.add_argument("--config", default="Release")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--no-tests", action="store_true")
    parser.add_argument(
        "--no-llama",
        action="store_true",
        help="build pager only, without the pinned llama.cpp integration",
    )
    args = parser.parse_args(argv)

    if shutil.which("cmake") is None:
        raise SystemExit("CMake was not found in PATH")

    build_dir = Path(args.build_dir).resolve()
    if args.clean and build_dir.exists():
        shutil.rmtree(build_dir)

    env: dict[str, str] | None = None
    msvc_batch: Path | None = None

    if os.name == "nt":
        env, msvc_batch = _capture_msvc_environment()

    if os.name == "nt" and not args.no_llama:
        compiler = shutil.which("cl", path=(env or os.environ).get("PATH"))
        nmake = shutil.which("nmake", path=(env or os.environ).get("PATH"))
        if compiler is None:
            installs = "\n".join(f"  - {p}" for p in _visual_studio_installations())
            detail = f"\nVisual Studio installations detected:\n{installs}" if installs else ""
            raise SystemExit(
                "MSVC C++ x64/x86 tools were not found. "
                "No usable VsDevCmd.bat/vcvars64.bat environment could be activated."
                + detail
            )
        print(f"MSVC: {compiler}", flush=True)
        if nmake:
            print(f"NMake: {nmake}", flush=True)
        if msvc_batch is not None:
            print(f"MSVC environment: {msvc_batch}", flush=True)

    run(
        [
            "cmake",
            "-S",
            str(ROOT / "native"),
            "-B",
            str(build_dir),
            f"-DMMB_BUILD_TESTS={'OFF' if args.no_tests else 'ON'}",
            f"-DMMB_WITH_LLAMA={'OFF' if args.no_llama else 'ON'}",
            f"-DCMAKE_BUILD_TYPE={args.config}",
        ],
        env=env,
    )
    run(
        ["cmake", "--build", str(build_dir), "--config", args.config],
        env=env,
    )
    if not args.no_tests:
        run(
            [
                "ctest",
                "--test-dir",
                str(build_dir),
                "-C",
                args.config,
                "--output-on-failure",
            ],
            env=env,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
