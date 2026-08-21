"""Fail-closed direct MiniMaxBrain inference runtime.

The normal application path is:

    Python -> mmb_backend -> llama.cpp/GGML -> real router
           -> MMBPager -> expert bytes from MMBW -> MUL_MAT_ID

The original GGUF and llama-server are not runtime dependencies. They remain
useful only for conversion and A/B acceptance.
"""
from __future__ import annotations

import json
import math
import os
import time
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Generator, Mapping, Sequence, Tuple

from .config import load_external_bundle
from .errors import BackendUnavailableError, InferenceError
from .native import NativeMMBRuntime, find_native_backend


class InferenceMode(str, Enum):
    UNAVAILABLE = "unavailable"
    PAGED_MMB = "paged_mmb"


def _process_rss_bytes(pid: int | None) -> int | None:
    """Best-effort resident-set measurement without third-party dependencies."""
    if not pid:
        return None

    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            PROCESS_VM_READ = 0x0010

            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            kernel32.OpenProcess.argtypes = [
                wintypes.DWORD,
                wintypes.BOOL,
                wintypes.DWORD,
            ]
            kernel32.OpenProcess.restype = wintypes.HANDLE
            kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
            kernel32.CloseHandle.restype = wintypes.BOOL
            psapi.GetProcessMemoryInfo.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
                wintypes.DWORD,
            ]
            psapi.GetProcessMemoryInfo.restype = wintypes.BOOL

            handle = kernel32.OpenProcess(
                PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_VM_READ,
                False,
                int(pid),
            )
            if not handle:
                return None
            try:
                counters = PROCESS_MEMORY_COUNTERS()
                counters.cb = ctypes.sizeof(counters)
                if not psapi.GetProcessMemoryInfo(
                    handle,
                    ctypes.byref(counters),
                    ctypes.sizeof(counters),
                ):
                    return None
                return int(counters.WorkingSetSize)
            finally:
                kernel32.CloseHandle(handle)
        except Exception:
            return None

    status = Path(f"/proc/{int(pid)}/status")
    try:
        for line in status.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


class MMBRuntime:
    """Persistent direct MMB runtime exposed to CLI and HTTP consumers.

    For compatibility, the old ``gguf_model_path`` / ``runner_path`` arguments
    remain accepted by the Python constructor but are deliberately ignored.
    Runtime inference never falls back to the original GGUF or llama-server.
    """

    def __init__(
        self,
        config_path: str | Path,
        gguf_model_path: str | Path | None = None,
        *,
        runner_path: str | Path | None = None,
        n_ctx: int = 2048,
        n_threads: int | None = None,
        backend_startup_timeout_seconds: float = 180.0,
    ):
        del gguf_model_path, runner_path, backend_startup_timeout_seconds

        self.config_path = Path(config_path).resolve()
        self.config, self.model_map = load_external_bundle(self.config_path)
        self.n_ctx = int(n_ctx)
        self.threads = int(n_threads or max(1, min(8, os.cpu_count() or 4)))
        if self.n_ctx < 128:
            raise ValueError("n_ctx must be >= 128")
        if self.threads < 1:
            raise ValueError("n_threads must be >= 1")

        self.layout = self._load_layout()
        self.num_layers = int(self.layout.get("layer_count") or 0)
        self.active_experts_per_layer = int(
            self.layout.get("active_experts_per_token") or 0
        )

        self.backend: NativeMMBRuntime | None = None
        self.backend_error: str | None = None
        self.inference_mode = InferenceMode.UNAVAILABLE
        self._closed = False
        self._init_backend()

    def _load_layout(self) -> Dict[str, Any]:
        path = self.config_path.parent / "model.mmb-layout.json"
        if not path.is_file():
            return {}
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return raw if isinstance(raw, dict) else {}

    @property
    def expert_cache_budget_bytes(self) -> int:
        # Config v1 expresses a model-resident budget that includes core.
        # The native runtime allocates core separately through llama.cpp, so the
        # pager receives only the remainder reserved for routed expert blocks.
        return (
            int(self.config.memory.cache_capacity_bytes)
            - int(self.model_map.core_bytes)
        )

    def _init_backend(self) -> None:
        if find_native_backend() is None:
            self.backend_error = (
                "mmb_backend native library not found. "
                "Build it with: python tools/build_native.py"
            )
            return

        expert_budget = self.expert_cache_budget_bytes
        if expert_budget < self.model_map.largest_expert_bytes:
            self.backend_error = (
                "configured expert-cache budget cannot hold the largest expert: "
                f"budget={expert_budget}, largest={self.model_map.largest_expert_bytes}"
            )
            return

        metadata = self.model_map.path.parent / "model.mmb-meta.gguf"
        if not metadata.is_file():
            self.backend_error = (
                "bundle is missing model.mmb-meta.gguf required by the direct runtime"
            )
            return

        try:
            self.backend = NativeMMBRuntime(
                self.model_map.path.parent,
                expert_budget,
                n_ctx=self.n_ctx,
                n_threads=self.threads,
                verify_sha256=self.config.io.integrity != "none",
            )
        except Exception as exc:
            self.backend = None
            self.backend_error = str(exc)
            return

        self.inference_mode = InferenceMode.PAGED_MMB
        self.backend_error = None

    @property
    def ready(self) -> bool:
        return (
            not self._closed
            and self.backend is not None
            and self.inference_mode is InferenceMode.PAGED_MMB
        )

    def require_ready(self) -> None:
        if not self.ready:
            raise BackendUnavailableError(
                self.backend_error or "direct MMB inference backend is unavailable"
            )

    @staticmethod
    def _validate_messages(
        messages: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, str]]:
        if not isinstance(messages, (list, tuple)) or not messages:
            raise InferenceError("messages must be a non-empty array")
        result: list[dict[str, str]] = []
        for index, message in enumerate(messages):
            if not isinstance(message, Mapping):
                raise InferenceError(f"messages[{index}] must be an object")
            role = message.get("role")
            content = message.get("content")
            if role not in {"system", "user", "assistant"}:
                raise InferenceError(
                    f"messages[{index}].role must be system, user, or assistant"
                )
            if not isinstance(content, str):
                raise InferenceError(
                    f"messages[{index}].content must be a string"
                )
            result.append({"role": str(role), "content": content})
        return result

    def validate_generation_params(
        self,
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> tuple[int, float, float, int]:
        if isinstance(max_tokens, bool):
            raise InferenceError(
                f"max_tokens must be between 1 and {self.n_ctx}"
            )
        try:
            normalized_max = int(max_tokens)
            normalized_temperature = float(temperature)
            normalized_top_p = float(top_p)
            normalized_top_k = int(top_k)
        except (TypeError, ValueError, OverflowError) as exc:
            raise InferenceError("invalid generation parameters") from exc

        if not 1 <= normalized_max <= self.n_ctx:
            raise InferenceError(
                f"max_tokens must be between 1 and {self.n_ctx}"
            )
        if (
            not math.isfinite(normalized_temperature)
            or normalized_temperature < 0.0
        ):
            raise InferenceError("temperature must be a finite value >= 0")
        if (
            not math.isfinite(normalized_top_p)
            or not 0.0 <= normalized_top_p <= 1.0
        ):
            raise InferenceError(
                "top_p must be a finite value between 0 and 1"
            )
        if normalized_top_k < 0:
            raise InferenceError("top_k must be >= 0")
        return (
            normalized_max,
            normalized_temperature,
            normalized_top_p,
            normalized_top_k,
        )

    def stream_chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        max_tokens: int = 64,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
    ) -> Generator[Tuple[str, Dict[str, Any]], None, None]:
        self.require_ready()
        max_tokens, temperature, top_p, top_k = self.validate_generation_params(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        normalized = self._validate_messages(messages)
        assert self.backend is not None

        started = time.perf_counter()
        chunks = 0
        for piece in self.backend.stream_chat(
            normalized,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        ):
            chunks += 1
            elapsed = max(time.perf_counter() - started, 1e-9)
            pager_stats = self.backend.stats()
            yield piece, {
                "chunk_idx": chunks,
                "latency_ms": round(elapsed * 1000.0, 2),
                "tokens_per_second": None,
                "inference_mode": self.inference_mode.value,
                "backend_rss_bytes": _process_rss_bytes(os.getpid()),
                "expert_cache_bytes": int(
                    pager_stats.get("resident_bytes", 0)
                ),
                "paged_experts_used": bool(
                    pager_stats.get("paged_experts_used", False)
                ),
                "router_requests": int(
                    pager_stats.get("real_router_requests", 0)
                ),
                "bytes_read": int(pager_stats.get("bytes_read", 0)),
            }

    def stream_generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 64,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
    ) -> Generator[Tuple[str, Dict[str, Any]], None, None]:
        if not isinstance(prompt, str):
            raise InferenceError("prompt must be a string")
        yield from self.stream_chat(
            [{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )

    def stats(self) -> Dict[str, Any]:
        pager_stats: Dict[str, Any] = {}
        if self.backend is not None:
            try:
                pager_stats = self.backend.stats()
            except Exception as exc:
                self.backend_error = str(exc)

        return {
            "model_id": self.model_map.model_id,
            "status": "ready" if self.ready else "not_ready",
            "ready": self.ready,
            "inference_mode": self.inference_mode.value,
            "backend_error": self.backend_error,
            "gguf_path": None,
            "backend_rss_bytes": _process_rss_bytes(os.getpid()),
            "expert_cache_bytes": int(
                pager_stats.get("resident_bytes", 0)
            ),
            "expert_cache_budget_bytes": self.expert_cache_budget_bytes,
            "paged_experts_used": bool(
                pager_stats.get("paged_experts_used", False)
            ),
            "ram_budget_bytes": self.config.memory.ram_budget_bytes,
            "pageable_backend_available": self.backend is not None,
            "paged_moe_kernel_available": bool(
                pager_stats.get("paged_moe_kernel_available", False)
            ),
            "native_runtime_available": bool(
                pager_stats.get("native_runtime_available", False)
            ),
            "native_pager_error": None if self.backend is not None else self.backend_error,
            "native_pager": pager_stats,
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.backend is not None:
            self.backend.close()
            self.backend = None
        self.inference_mode = InferenceMode.UNAVAILABLE

    def __enter__(self) -> "MMBRuntime":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
