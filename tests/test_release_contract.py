from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_user_release_contains_prebuilt_windows_backend():
    dll = ROOT / "runtime" / "windows-x64" / "mmb_backend.dll"
    manifest_path = ROOT / "runtime" / "windows-x64" / "manifest.json"
    assert dll.is_file()
    assert manifest_path.is_file()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema"] == "mmb-prebuilt-runtime-v1"
    assert manifest["platform"] == "windows-x64"
    assert manifest["abi_version"] == 3
    assert manifest["bytes"] == dll.stat().st_size
    assert manifest["sha256"] == hashlib.sha256(dll.read_bytes()).hexdigest()


def test_starter_uses_prebuilt_backend_and_does_not_pip_install_project():
    starter = (ROOT / "starter.bat").read_text(encoding="utf-8")
    assert r"runtime\windows-x64\mmb_backend.dll" in starter
    assert "pip install" not in starter
    assert "bootstrap_portable_python.ps1" in starter
    assert "bootstrap_vc_runtime.ps1" in starter
    assert "Ferramentas de desenvolvedor" in starter


def test_native_loader_prioritizes_release_runtime_path():
    source = (ROOT / "minimaxbrain" / "native.py").read_text(encoding="utf-8")
    release_pos = source.index('root / "runtime" / "windows-x64"')
    build_pos = source.index('root / "native" / "build" / "Release"')
    assert release_pos < build_pos
