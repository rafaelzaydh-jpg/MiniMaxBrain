from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "native" / "build" / "Release" / "mmb_backend.dll"
DEST_DIR = ROOT / "runtime" / "windows-x64"
DEST = DEST_DIR / "mmb_backend.dll"
MANIFEST = DEST_DIR / "manifest.json"


def main() -> int:
    if not SOURCE.is_file():
        raise SystemExit(f"backend build not found: {SOURCE}")

    DEST_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SOURCE, DEST)
    payload = DEST.read_bytes()
    llama_path = ROOT / "native" / "LLAMA_VERSION"
    llama_version = (
        llama_path.read_text(encoding="utf-8").strip()
        if llama_path.is_file()
        else "unknown"
    )
    manifest = {
        "schema": "mmb-prebuilt-runtime-v1",
        "platform": "windows-x64",
        "mmb_version": "0.3.1",
        "abi_version": 3,
        "file": DEST.name,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
        "llama_version": llama_version,
        "built_from": "native/build/Release/mmb_backend.dll",
        "notes": [
            "User runtime artifact; no Visual Studio/CMake build is required to run MiniMaxBrain.",
            "The DLL uses the Microsoft Visual C++ v14 runtime; starter.bat can bootstrap the official redistributable when necessary.",
        ],
    }
    MANIFEST.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[OK] promoted: {DEST}")
    print(f"[OK] sha256: {manifest['sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
