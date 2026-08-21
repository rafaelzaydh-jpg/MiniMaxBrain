"""Stable ctypes bridge for the MiniMaxBrain native pager ABI.

The bridge intentionally targets only ``mmb_backend``.  It never mirrors
llama.cpp or GGML internal structs.  The current native library implements
verified MMB storage, cache and leases.  The same ABI also exposes the validated direct llama.cpp runtime when the library advertises ``MMB_CAP_NATIVE_RUNTIME``.
"""
from __future__ import annotations

import codecs
import ctypes
import os
import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Iterable, Mapping, Sequence

from .errors import BackendUnavailableError, InferenceError

MMB_ABI_VERSION = 3
MMB_CAP_PAGER = 0x00000001
MMB_CAP_PAGED_MOE_KERNEL = 0x00000002
MMB_CAP_NATIVE_RUNTIME = 0x00000004

_ROLE = {"down": 0, "gate": 1, "up": 2}


class _SegmentView(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("bytes", ctypes.c_uint64),
        ("ggml_type", ctypes.c_int32),
        ("n_dims", ctypes.c_uint32),
        ("ne", ctypes.c_int64 * 4),
    ]


class _PagerStats(ctypes.Structure):
    _fields_ = [
        ("cache_hits", ctypes.c_uint64),
        ("cache_misses", ctypes.c_uint64),
        ("bytes_read", ctypes.c_uint64),
        ("resident_bytes", ctypes.c_uint64),
        ("peak_resident_bytes", ctypes.c_uint64),
        ("loads", ctypes.c_uint64),
        ("evictions", ctypes.c_uint64),
        ("real_router_requests", ctypes.c_uint64),
        ("experts_used", ctypes.c_uint64),
        ("acquire_ns", ctypes.c_uint64),
        ("io_ns", ctypes.c_uint64),
        ("paged_experts_used", ctypes.c_uint32),
    ]


class _ChatMessage(ctypes.Structure):
    _fields_ = [
        ("role", ctypes.c_char_p),
        ("content", ctypes.c_char_p),
    ]


class _GenerationParams(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("max_tokens", ctypes.c_int32),
        ("temperature", ctypes.c_float),
        ("top_p", ctypes.c_float),
        ("top_k", ctypes.c_int32),
        ("seed", ctypes.c_uint32),
    ]


_STREAM_CALLBACK = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_void_p,
)


@dataclass(frozen=True)
class NativeSegment:
    address: int
    bytes: int
    ggml_type: int
    shape: tuple[int, ...]

    def copy_bytes(self) -> bytes:
        if not self.address or self.bytes <= 0:
            return b""
        return ctypes.string_at(self.address, self.bytes)


def _candidate_library_paths() -> list[Path]:
    root = Path(__file__).resolve().parents[1]
    names = (
        ("mmb_backend.dll",)
        if os.name == "nt"
        else ("libmmb_backend.so", "libmmb_backend.dylib")
    )
    dirs = [
        # User-facing release path. A GitHub ZIP/release can run without a
        # local C++ toolchain when this prebuilt backend is present.
        root / "runtime" / "windows-x64",
        # Developer/build fallbacks.
        root / "native" / "build" / "Release",
        root / "native" / "build",
        root / "native" / "bin",
        root / "runner",
    ]
    return [directory / name for directory in dirs for name in names]


def find_native_backend(explicit: str | Path | None = None) -> Path | None:
    if explicit is not None:
        path = Path(explicit).expanduser().resolve()
        return path if path.is_file() else None
    env = os.environ.get("MMB_BACKEND_LIBRARY")
    if env:
        path = Path(env).expanduser().resolve()
        if path.is_file():
            return path
    for path in _candidate_library_paths():
        if path.is_file():
            return path.resolve()
    return None


class NativeLibrary:
    def __init__(self, path: str | Path | None = None):
        resolved = find_native_backend(path)
        if resolved is None:
            raise BackendUnavailableError(
                "mmb_backend native library was not found. "
                "Expected the prebuilt release at runtime/windows-x64/"
                "mmb_backend.dll; developers can rebuild it with "
                "python tools/build_native.py"
            )
        self.path = resolved
        try:
            self.lib = ctypes.CDLL(str(resolved))
        except OSError as exc:
            hint = ""
            if os.name == "nt":
                hint = (
                    " If Windows reports a missing MSVCP/VCRUNTIME/VCOMP DLL, "
                    "install the Microsoft Visual C++ v14 x64 Redistributable."
                )
            raise BackendUnavailableError(
                f"cannot load native MMB backend {resolved}: {exc}.{hint}"
            ) from exc
        self._bind()
        abi = int(self.lib.mmb_abi_version())
        if abi != MMB_ABI_VERSION:
            raise BackendUnavailableError(
                f"MMB native ABI mismatch: library={abi}, python={MMB_ABI_VERSION}"
            )

    def _bind(self) -> None:
        lib = self.lib
        lib.mmb_abi_version.argtypes = []
        lib.mmb_abi_version.restype = ctypes.c_uint32
        lib.mmb_backend_capabilities.argtypes = []
        lib.mmb_backend_capabilities.restype = ctypes.c_uint32
        lib.mmb_backend_version.argtypes = []
        lib.mmb_backend_version.restype = ctypes.c_char_p
        lib.mmb_last_error.argtypes = []
        lib.mmb_last_error.restype = ctypes.c_char_p

        lib.mmb_pager_open.argtypes = [
            ctypes.c_char_p,
            ctypes.c_uint64,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        lib.mmb_pager_open.restype = ctypes.c_int
        lib.mmb_pager_close.argtypes = [ctypes.c_void_p]
        lib.mmb_pager_close.restype = None

        lib.mmb_pager_model_info.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint32),
        ]
        lib.mmb_pager_model_info.restype = ctypes.c_int

        lib.mmb_pager_acquire.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        lib.mmb_pager_acquire.restype = ctypes.c_int
        lib.mmb_lease_count.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
        lib.mmb_lease_count.restype = ctypes.c_int
        lib.mmb_lease_expert_id.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_uint32),
        ]
        lib.mmb_lease_expert_id.restype = ctypes.c_int
        lib.mmb_lease_segment.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.POINTER(_SegmentView),
        ]
        lib.mmb_lease_segment.restype = ctypes.c_int
        lib.mmb_pager_release.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        lib.mmb_pager_release.restype = ctypes.c_int
        lib.mmb_pager_get_stats.argtypes = [ctypes.c_void_p, ctypes.POINTER(_PagerStats)]
        lib.mmb_pager_get_stats.restype = ctypes.c_int

        if hasattr(lib, "mmb_runtime_open"):
            lib.mmb_runtime_open.argtypes = [
                ctypes.c_char_p,
                ctypes.c_uint64,
                ctypes.c_int,
                ctypes.c_uint32,
                ctypes.c_int32,
                ctypes.POINTER(ctypes.c_void_p),
            ]
            lib.mmb_runtime_open.restype = ctypes.c_int
            lib.mmb_runtime_close.argtypes = [ctypes.c_void_p]
            lib.mmb_runtime_close.restype = None
            lib.mmb_runtime_chat.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(_ChatMessage),
                ctypes.c_size_t,
                ctypes.POINTER(_GenerationParams),
                _STREAM_CALLBACK,
                ctypes.c_void_p,
            ]
            lib.mmb_runtime_chat.restype = ctypes.c_int
            lib.mmb_runtime_get_stats.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(_PagerStats),
            ]
            lib.mmb_runtime_get_stats.restype = ctypes.c_int

    @property
    def capabilities(self) -> int:
        return int(self.lib.mmb_backend_capabilities())

    @property
    def version(self) -> str:
        raw = self.lib.mmb_backend_version()
        return raw.decode("utf-8", errors="strict") if raw else ""

    def error(self) -> str:
        raw = self.lib.mmb_last_error()
        return raw.decode("utf-8", errors="replace") if raw else "native MMB error"

    def check(self, rc: int) -> None:
        if int(rc) != 0:
            raise InferenceError(self.error())


class NativeLease:
    def __init__(self, pager: "NativePager", handle: ctypes.c_void_p):
        self._pager = pager
        self._handle = handle
        self._released = False

    @property
    def count(self) -> int:
        out = ctypes.c_size_t()
        self._pager.library.check(
            self._pager.library.lib.mmb_lease_count(self._handle, ctypes.byref(out))
        )
        return int(out.value)

    def expert_id(self, index: int) -> int:
        out = ctypes.c_uint32()
        self._pager.library.check(
            self._pager.library.lib.mmb_lease_expert_id(
                self._handle, int(index), ctypes.byref(out)
            )
        )
        return int(out.value)

    def segment(self, index: int, role: str) -> NativeSegment:
        if role not in _ROLE:
            raise ValueError("role must be down, gate, or up")
        out = _SegmentView()
        self._pager.library.check(
            self._pager.library.lib.mmb_lease_segment(
                self._handle, int(index), _ROLE[role], ctypes.byref(out)
            )
        )
        return NativeSegment(
            address=int(out.data or 0),
            bytes=int(out.bytes),
            ggml_type=int(out.ggml_type),
            shape=tuple(int(out.ne[i]) for i in range(int(out.n_dims))),
        )

    def release(self) -> None:
        if self._released:
            return
        self._pager.library.check(
            self._pager.library.lib.mmb_pager_release(
                self._pager._handle, self._handle
            )
        )
        self._released = True
        self._handle = ctypes.c_void_p()

    def __enter__(self) -> "NativeLease":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()


class NativePager:
    def __init__(
        self,
        model_dir: str | Path,
        cache_capacity_bytes: int,
        *,
        verify_sha256: bool = True,
        library_path: str | Path | None = None,
    ):
        self.library = NativeLibrary(library_path)
        if not (self.library.capabilities & MMB_CAP_PAGER):
            raise BackendUnavailableError("native MMB library does not provide pager capability")
        self.model_dir = Path(model_dir).resolve()
        self._handle = ctypes.c_void_p()
        self._closed = False
        self.library.check(
            self.library.lib.mmb_pager_open(
                os.fsencode(self.model_dir),
                int(cache_capacity_bytes),
                int(bool(verify_sha256)),
                ctypes.byref(self._handle),
            )
        )

    @property
    def model_info(self) -> dict[str, int]:
        layers = ctypes.c_uint32()
        experts = ctypes.c_uint32()
        active = ctypes.c_uint32()
        self.library.check(
            self.library.lib.mmb_pager_model_info(
                self._handle,
                ctypes.byref(layers),
                ctypes.byref(experts),
                ctypes.byref(active),
            )
        )
        return {
            "layer_count": int(layers.value),
            "expert_count": int(experts.value),
            "active_experts_per_token": int(active.value),
        }

    def acquire(
        self,
        layer: int,
        experts: Sequence[int] | Iterable[int],
        *,
        router_request: bool = False,
    ) -> NativeLease:
        ids = [int(value) for value in experts]
        if not ids:
            raise ValueError("experts must not be empty")
        if any(value < 0 or value > 0xFFFFFFFF for value in ids):
            raise ValueError("expert IDs must be uint32")
        array_type = ctypes.c_uint32 * len(ids)
        raw_ids = array_type(*ids)
        handle = ctypes.c_void_p()
        self.library.check(
            self.library.lib.mmb_pager_acquire(
                self._handle,
                int(layer),
                raw_ids,
                len(ids),
                int(bool(router_request)),
                ctypes.byref(handle),
            )
        )
        return NativeLease(self, handle)

    def stats(self) -> dict[str, int | bool]:
        raw = _PagerStats()
        self.library.check(
            self.library.lib.mmb_pager_get_stats(self._handle, ctypes.byref(raw))
        )
        return {
            "cache_hits": int(raw.cache_hits),
            "cache_misses": int(raw.cache_misses),
            "bytes_read": int(raw.bytes_read),
            "resident_bytes": int(raw.resident_bytes),
            "peak_resident_bytes": int(raw.peak_resident_bytes),
            "loads": int(raw.loads),
            "evictions": int(raw.evictions),
            "real_router_requests": int(raw.real_router_requests),
            "experts_used": int(raw.experts_used),
            "acquire_ns": int(raw.acquire_ns),
            "io_ns": int(raw.io_ns),
            "paged_experts_used": bool(raw.paged_experts_used),
            "paged_moe_kernel_available": bool(
                self.library.capabilities & MMB_CAP_PAGED_MOE_KERNEL
            ),
            "native_runtime_available": bool(
                self.library.capabilities & MMB_CAP_NATIVE_RUNTIME
            ),
            "capabilities": {
                "pager": bool(self.library.capabilities & MMB_CAP_PAGER),
                "paged_moe_kernel_available": bool(
                    self.library.capabilities & MMB_CAP_PAGED_MOE_KERNEL
                ),
                "native_runtime_available": bool(
                    self.library.capabilities & MMB_CAP_NATIVE_RUNTIME
                ),
            },
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._handle:
            self.library.lib.mmb_pager_close(self._handle)
            self._handle = ctypes.c_void_p()

    def __enter__(self) -> "NativePager":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class NativeMMBRuntime:
    """Persistent direct MMB -> llama.cpp runtime.

    The model and llama context stay loaded for the lifetime of this object.
    Every chat request rebuilds the prompt from the full history and clears
    context memory first, so unrelated callers cannot inherit stale KV state.
    """

    DEFAULT_SEED = 0xFFFFFFFF

    def __init__(
        self,
        model_dir: str | Path,
        expert_cache_bytes: int,
        *,
        n_ctx: int = 2048,
        n_threads: int = 1,
        verify_sha256: bool = True,
        library_path: str | Path | None = None,
    ):
        self.library = NativeLibrary(library_path)
        required = MMB_CAP_PAGED_MOE_KERNEL | MMB_CAP_NATIVE_RUNTIME
        if (
            (self.library.capabilities & required) != required
            or not hasattr(self.library.lib, "mmb_runtime_open")
        ):
            raise BackendUnavailableError(
                "native MMB library does not provide the direct paged MoE runtime"
            )
        if int(expert_cache_bytes) <= 0:
            raise ValueError("expert_cache_bytes must be positive")
        if int(n_ctx) < 128:
            raise ValueError("n_ctx must be >= 128")
        if int(n_threads) <= 0:
            raise ValueError("n_threads must be positive")

        self.model_dir = Path(model_dir).resolve()
        self._handle = ctypes.c_void_p()
        self._closed = False
        self._call_lock = threading.Lock()

        self.library.check(
            self.library.lib.mmb_runtime_open(
                os.fsencode(self.model_dir),
                int(expert_cache_bytes),
                int(bool(verify_sha256)),
                int(n_ctx),
                int(n_threads),
                ctypes.byref(self._handle),
            )
        )

    @staticmethod
    def _encode_messages(
        messages: Sequence[Mapping[str, str]],
    ) -> tuple[ctypes.Array, list[bytes], list[bytes]]:
        if not messages:
            raise InferenceError("messages must not be empty")
        roles: list[bytes] = []
        contents: list[bytes] = []
        for index, message in enumerate(messages):
            role = message.get("role")
            content = message.get("content")
            if role not in {"system", "user", "assistant"}:
                raise InferenceError(f"invalid role at messages[{index}]")
            if not isinstance(content, str):
                raise InferenceError(f"messages[{index}].content must be a string")
            roles.append(str(role).encode("utf-8"))
            contents.append(content.encode("utf-8"))
        array_type = _ChatMessage * len(messages)
        raw = array_type(
            *(_ChatMessage(roles[i], contents[i]) for i in range(len(messages)))
        )
        return raw, roles, contents

    def stream_chat(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        seed: int = DEFAULT_SEED,
    ) -> Generator[str, None, None]:
        if self._closed or not self._handle:
            raise BackendUnavailableError("native MMB runtime is closed")

        raw_messages, roles, contents = self._encode_messages(messages)
        params = _GenerationParams(
            ctypes.sizeof(_GenerationParams),
            int(max_tokens),
            float(temperature),
            float(top_p),
            int(top_k),
            int(seed) & 0xFFFFFFFF,
        )

        events: queue.Queue[tuple[str, object]] = queue.Queue()
        cancel = threading.Event()

        @_STREAM_CALLBACK
        def on_piece(address, size, _userdata):
            if cancel.is_set():
                return 1
            try:
                payload = ctypes.string_at(address, int(size)) if size else b""
                events.put(("data", payload))
                return 0
            except BaseException as exc:  # callback must never unwind through C
                try:
                    events.put(("error", exc))
                finally:
                    cancel.set()
                return 1

        def worker() -> None:
            try:
                with self._call_lock:
                    if self._closed or not self._handle:
                        raise BackendUnavailableError("native MMB runtime is closed")
                    rc = int(
                        self.library.lib.mmb_runtime_chat(
                            self._handle,
                            raw_messages,
                            len(messages),
                            ctypes.byref(params),
                            on_piece,
                            None,
                        )
                    )
                    # rc == 1 means the callback explicitly cancelled.
                    if rc < 0 or (rc > 0 and not cancel.is_set()):
                        raise InferenceError(self.library.error())
            except BaseException as exc:
                events.put(("error", exc))
            finally:
                # Keep encoded strings/callback alive until the C call is done.
                _ = (roles, contents, on_piece)
                events.put(("done", None))

        thread = threading.Thread(target=worker, name="mmb-native-generate", daemon=True)
        thread.start()
        decoder = codecs.getincrementaldecoder("utf-8")("replace")

        try:
            while True:
                kind, payload = events.get()
                if kind == "data":
                    text = decoder.decode(payload, final=False)
                    if text:
                        yield text
                elif kind == "error":
                    assert isinstance(payload, BaseException)
                    if isinstance(payload, (BackendUnavailableError, InferenceError)):
                        raise payload
                    raise InferenceError(str(payload)) from payload
                else:
                    tail = decoder.decode(b"", final=True)
                    if tail:
                        yield tail
                    break
        finally:
            cancel.set()
            thread.join()

    def stats(self) -> dict[str, int | bool | dict[str, bool]]:
        if self._closed or not self._handle:
            return {
                "cache_hits": 0,
                "cache_misses": 0,
                "bytes_read": 0,
                "resident_bytes": 0,
                "peak_resident_bytes": 0,
                "loads": 0,
                "evictions": 0,
                "real_router_requests": 0,
                "experts_used": 0,
                "acquire_ns": 0,
                "io_ns": 0,
                "paged_experts_used": False,
                "paged_moe_kernel_available": True,
                "native_runtime_available": True,
            }
        raw = _PagerStats()
        self.library.check(
            self.library.lib.mmb_runtime_get_stats(self._handle, ctypes.byref(raw))
        )
        return {
            "cache_hits": int(raw.cache_hits),
            "cache_misses": int(raw.cache_misses),
            "bytes_read": int(raw.bytes_read),
            "resident_bytes": int(raw.resident_bytes),
            "peak_resident_bytes": int(raw.peak_resident_bytes),
            "loads": int(raw.loads),
            "evictions": int(raw.evictions),
            "real_router_requests": int(raw.real_router_requests),
            "experts_used": int(raw.experts_used),
            "acquire_ns": int(raw.acquire_ns),
            "io_ns": int(raw.io_ns),
            "paged_experts_used": bool(raw.paged_experts_used),
            "paged_moe_kernel_available": True,
            "native_runtime_available": True,
            "capabilities": {
                "pager": True,
                "paged_moe_kernel_available": True,
                "native_runtime_available": True,
            },
        }

    def close(self) -> None:
        if self._closed:
            return
        with self._call_lock:
            if self._closed:
                return
            self._closed = True
            if self._handle:
                self.library.lib.mmb_runtime_close(self._handle)
                self._handle = ctypes.c_void_p()

    def __enter__(self) -> "NativeMMBRuntime":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
