"""Independent external admission controller for pageable LLM weights."""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Sequence

from .cache import CacheLease, ResidentCache
from .config import ExternalGateConfig
from .errors import AdmissionError, LeaseError, UnknownBlockError
from .model_map import PhysicalModelMap, WeightBlock, public_map_summary
from .model_memory import ModelMemory
from .scheduler import LoadScheduler
from .storage import FileRangeStore, PayloadArena
from .telemetry import MMBTelemetry, NullTelemetry, TelemetrySink


@dataclass
class _ExternalLease:
    lease_id: str
    request_id: str
    cache_lease_ids: tuple[str, ...]
    created_at: float


class ExternalGate:
    """Physical authority over SSD -> RAM admission.

    Block IDs are opaque. This component never interprets prompts, hidden
    states, logits, expert meaning, or answer quality.
    """

    def __init__(
        self,
        config: ExternalGateConfig,
        model_map: PhysicalModelMap,
        *,
        telemetry: TelemetrySink | None = None,
        model_memory: ModelMemory | None = None,
    ):
        self.config = config
        self.model_map = model_map
        self.telemetry = telemetry or (MMBTelemetry() if config.telemetry_enabled else NullTelemetry())
        self._owns_model_memory = model_memory is None and config.model_memory.enabled
        if self._owns_model_memory:
            model_memory = ModelMemory(config.model_memory.path, model_map)
        if model_memory is not None and model_memory.model_identity != model_map.identity:
            raise AdmissionError("model memory is bound to a different physical model-map identity")
        self.model_memory = model_memory
        self.cache = ResidentCache(
            config.memory.cache_capacity_bytes,
            max_expert_entries=config.memory.max_resident_experts,
        )
        self.arena = PayloadArena(config.memory.transport)
        self.store = FileRangeStore(config.io.integrity, self.telemetry)
        self.scheduler = LoadScheduler(
            self.cache,
            self.store,
            self.arena,
            workers=config.io.workers,
            max_prefetch=config.io.prefetch_queue,
            telemetry=self.telemetry,
        )
        self._leases: dict[str, _ExternalLease] = {}
        self._lock = threading.RLock()
        self._admission_lock = threading.RLock()
        self._started = False
        self._closed = False
        self._hits = 0
        self._misses = 0
        self._prefetch_waits = 0

    def start(self) -> None:
        with self._lock:
            if self._closed:
                raise AdmissionError("external gate is closed")
            if self._started:
                return
            self.scheduler.load_pinned(self.model_map.core_blocks)
            self._started = True

    def _require_started(self) -> None:
        if not self._started:
            self.start()
        if self._closed:
            raise AdmissionError("external gate is closed")

    def _blocks(self, block_ids: Iterable[str]) -> list[WeightBlock]:
        blocks: list[WeightBlock] = []
        seen: set[str] = set()
        for raw in block_ids:
            block_id = str(raw)
            if block_id in seen:
                continue
            seen.add(block_id)
            blocks.append(self.model_map.block(block_id))
        return blocks

    def prefetch(self, items: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        """Accept ordered, fallible advice from a future internal gate."""
        self._require_started()
        ordered: list[tuple[int, int, WeightBlock]] = []
        rejected = 0
        for ordinal, item in enumerate(items):
            if not isinstance(item, dict):
                rejected += 1
                continue
            try:
                block = self.model_map.block(str(item.get("block_id") or ""))
                priority = int(item.get("priority", 100))
            except (UnknownBlockError, TypeError, ValueError):
                rejected += 1
                continue
            ordered.append((priority, ordinal, block))
        ordered.sort(key=lambda value: (value[0], value[1]))
        result = self.scheduler.prefetch([item[2] for item in ordered])
        result["rejected"] = rejected
        return result

    def _expire_stale_leases(self) -> int:
        deadline = time.monotonic() - self.config.memory.lease_timeout_seconds
        with self._lock:
            stale = [lease_id for lease_id, lease in self._leases.items() if lease.created_at <= deadline]
        for lease_id in stale:
            self.release(lease_id, missing_ok=True)
        return len(stale)

    def acquire(self, block_ids: Sequence[str], *, request_id: str | None = None) -> Dict[str, Any]:
        """Admit compulsory blocks and hold them until explicit release."""
        self._require_started()
        if not isinstance(block_ids, (list, tuple)) or not block_ids:
            raise AdmissionError("acquire requires at least one block_id")
        if len(block_ids) > 256:
            raise AdmissionError("one acquire request may contain at most 256 block IDs")
        self._expire_stale_leases()
        blocks = self._blocks(block_ids)
        acquired: list[CacheLease] = []
        observations: list[tuple[WeightBlock, str, float]] = []
        route_started = time.perf_counter()
        with self._admission_lock:
            try:
                for block in blocks:
                    block_started = time.perf_counter()
                    state = self.scheduler.ensure(block)
                    lease = self.cache.acquire(block.block_id)
                    if lease is None:
                        # A concurrent speculative admission may have displaced
                        # an unleased block between ensure and acquire.
                        state = self.scheduler.ensure(block)
                        lease = self.cache.acquire(block.block_id)
                    if lease is None:
                        raise AdmissionError(f"block {block.block_id!r} could not be leased after admission")
                    acquired.append(lease)
                    observations.append((block, state, (time.perf_counter() - block_started) * 1000.0))
                    if state == "hit":
                        self._hits += 1
                    elif state == "prefetch_wait":
                        self._prefetch_waits += 1
                    else:
                        self._misses += 1
            except Exception:
                for lease in acquired:
                    self.cache.release(lease.lease_id)
                raise
        external_id = uuid.uuid4().hex
        request_identity = str(request_id or uuid.uuid4().hex)
        with self._lock:
            self._leases[external_id] = _ExternalLease(
                lease_id=external_id,
                request_id=request_identity,
                cache_lease_ids=tuple(item.lease_id for item in acquired),
                created_at=time.monotonic(),
            )
        duration_ms = (time.perf_counter() - route_started) * 1000.0
        if self.model_memory is not None:
            for block, state, block_duration_ms in observations:
                if block.route_key is None:
                    continue
                try:
                    self.model_memory.record_route(
                        block.layer,
                        block.expert,
                        state=state,
                        duration_ms=block_duration_ms,
                    )
                except Exception as exc:
                    # The memory advises future routing; it is never allowed to
                    # break a compulsory weight admission that already succeeded.
                    self.telemetry.emit(
                        "model_memory", "error",
                        metadata={"block_id": block.block_id, "error": type(exc).__name__},
                    )
        self.telemetry.emit(
            "acquire", "ok", duration_ms=duration_ms,
            metadata={"request_id": request_identity, "blocks": len(acquired)},
        )
        return {
            "lease_id": external_id,
            "request_id": request_identity,
            "blocks": [item.descriptor for item in acquired],
            "duration_ms": round(duration_ms, 3),
        }

    def acquire_routes(self, routes: Sequence[Dict[str, Any]], *, request_id: str | None = None) -> Dict[str, Any]:
        block_ids: list[str] = []
        for item in routes:
            if not isinstance(item, dict):
                raise AdmissionError("each route must be an object")
            try:
                block_ids.append(self.model_map.route_block(int(item["layer"]), int(item["expert"])).block_id)
            except (KeyError, TypeError, ValueError) as exc:
                raise AdmissionError("each route requires integer layer and expert") from exc
        return self.acquire(block_ids, request_id=request_id)

    def release(self, lease_id: str, *, missing_ok: bool = False) -> Dict[str, Any]:
        with self._lock:
            lease = self._leases.pop(str(lease_id), None)
        if lease is None:
            if missing_ok:
                return {"lease_id": str(lease_id), "released": False}
            raise LeaseError(f"external lease {lease_id!r} does not exist")
        released = 0
        for cache_lease_id in lease.cache_lease_ids:
            released += int(self.cache.release(cache_lease_id))
        return {"lease_id": lease.lease_id, "released": True, "blocks_released": released}

    def snapshot(self) -> Dict[str, Any]:
        self._expire_stale_leases()
        with self._lock:
            active = len(self._leases)
        return {
            "model": public_map_summary(self.model_map),
            "memory": {
                "ram_budget_bytes": self.config.memory.ram_budget_bytes,
                "kv_cache_bytes": self.config.memory.kv_cache_bytes,
                "scratch_bytes": self.config.memory.scratch_bytes,
                "budget_mode": self.config.memory.budget_mode,
                "transport": self.config.memory.transport,
                **self.cache.snapshot(),
            },
            "io": {**self.store.stats(), **self.scheduler.snapshot()},
            "routing": {
                "hits": self._hits,
                "misses": self._misses,
                "prefetch_waits": self._prefetch_waits,
                "active_external_leases": active,
            },
            "model_memory": self.model_memory.snapshot() if self.model_memory is not None else None,
        }

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            lease_ids = list(self._leases)
        for lease_id in lease_ids:
            self.release(lease_id, missing_ok=True)
        self.scheduler.close()
        self.cache.close()
        if self._owns_model_memory and self.model_memory is not None:
            self.model_memory.close()
        with self._lock:
            self._closed = True

    def __enter__(self) -> "ExternalGate":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
