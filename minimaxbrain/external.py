"""Synchronous physical SSD -> RAM pager for MiniMaxBrain weights."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Sequence

from .cache import CacheLease, ResidentCache
from .config import ExternalGateConfig
from .errors import AdmissionError, LeaseError
from .model_map import PhysicalModelMap, WeightBlock, public_map_summary
from .storage import FileRangeStore, PayloadArena


@dataclass
class _ExternalLease:
    lease_id: str
    request_id: str
    cache_lease_ids: tuple[str, ...]
    created_at: float


class ExternalGate:
    """Physical authority over SSD -> RAM admission.

    The gate knows only physical blocks. It does not interpret prompts, hidden
    states, logits, or choose experts. Loads are deliberately synchronous until
    a real paged-MoE executor exists; speculative prefetch is not on the
    correctness path.
    """

    def __init__(
        self,
        config: ExternalGateConfig,
        model_map: PhysicalModelMap,
    ):
        self.config = config
        self.model_map = model_map
        self.cache = ResidentCache(
            config.memory.cache_capacity_bytes,
            max_expert_entries=config.memory.max_resident_experts,
        )
        self.arena = PayloadArena()
        self.store = FileRangeStore(
            config.io.integrity,
            model_map=self.model_map,
        )

        self._leases: dict[str, _ExternalLease] = {}
        self._lock = threading.RLock()
        self._admission_lock = threading.RLock()
        self._started = False
        self._closed = False
        self._hits = 0
        self._misses = 0
        self._lease_seq = 0
        self._last_expiry = time.monotonic()

    def _load(self, block: WeightBlock, *, pinned: bool) -> bool:
        if self.cache.contains(block.block_id):
            return False

        started = time.perf_counter()
        reservation = self.cache.reserve(block, pinned=pinned, speculative=False)
        if reservation is None:
            return False

        allocation = None
        view = None
        try:
            allocation = self.arena.allocate(block.length)
            view = allocation.writable_view()
            self.store.read_into(block, view)
            view.release()
            view = None
            committed = self.cache.commit(reservation, block, allocation)
            allocation = None
            return committed
        except Exception as exc:
            self.cache.cancel(reservation)
            if view is not None:
                view.release()
            if allocation is not None:
                allocation.close()
            raise

    def start(self) -> None:
        with self._lock:
            if self._closed:
                raise AdmissionError("external gate is closed")
            if self._started:
                return
        try:
            with self._admission_lock:
                for block in self.model_map.core_blocks:
                    self._load(block, pinned=True)
        except Exception:
            self.close()
            raise
        with self._lock:
            self._started = True

    def _require_started(self) -> None:
        if self._closed:
            raise AdmissionError("external gate is closed")
        if not self._started:
            self.start()

    def _blocks(self, block_ids: Iterable[str]) -> list[WeightBlock]:
        return [self.model_map.block(str(block_id)) for block_id in block_ids]

    def _expire_stale_leases(self) -> int:
        now = time.monotonic()
        if now - self._last_expiry < 5.0:
            return 0
        self._last_expiry = now
        deadline = now - self.config.memory.lease_timeout_seconds
        with self._lock:
            stale = [
                lease_id
                for lease_id, lease in self._leases.items()
                if lease.created_at <= deadline
            ]
        for lease_id in stale:
            self.release(lease_id, missing_ok=True)
        return len(stale)

    def acquire(
        self,
        block_ids: Sequence[str],
        *,
        request_id: str | None = None,
    ) -> Dict[str, Any]:
        """Admit compulsory blocks and hold them until explicit release."""
        self._require_started()
        if not isinstance(block_ids, (list, tuple)) or not block_ids:
            raise AdmissionError("acquire requires at least one block_id")
        if len(block_ids) > 256:
            raise AdmissionError("one acquire request may contain at most 256 block IDs")

        self._expire_stale_leases()
        blocks = self._blocks(block_ids)
        acquired: list[CacheLease] = []
        route_started = time.perf_counter()

        with self._admission_lock:
            try:
                for block in blocks:
                    lease = self.cache.acquire(block.block_id)
                    if lease is not None:
                        self._hits += 1
                    else:
                        self._load(block, pinned=False)
                        lease = self.cache.acquire(block.block_id)
                        if lease is None:
                            raise AdmissionError(
                                f"block {block.block_id!r} could not be leased after admission"
                            )
                        self._misses += 1
                    acquired.append(lease)
            except Exception:
                for lease in acquired:
                    self.cache.release(lease.lease_id)
                raise

        with self._lock:
            self._lease_seq += 1
            external_id = f"lease-{self._lease_seq}"
            request_identity = str(request_id or external_id)
            self._leases[external_id] = _ExternalLease(
                lease_id=external_id,
                request_id=request_identity,
                cache_lease_ids=tuple(item.lease_id for item in acquired),
                created_at=time.monotonic(),
            )

        duration_ms = (time.perf_counter() - route_started) * 1000.0
        return {
            "lease_id": external_id,
            "request_id": request_identity,
            "blocks": [item.descriptor for item in acquired],
            "duration_ms": round(duration_ms, 3),
        }

    def acquire_routes(
        self,
        routes: Sequence[Dict[str, Any]],
        *,
        request_id: str | None = None,
    ) -> Dict[str, Any]:
        block_ids: list[str] = []
        for item in routes:
            if not isinstance(item, dict):
                raise AdmissionError("each route must be an object")
            try:
                layer = int(item["layer"])
                expert = int(item["expert"])
            except (KeyError, TypeError, ValueError) as exc:
                raise AdmissionError(
                    "each route requires integer layer and expert"
                ) from exc
            block_ids.append(self.model_map.route_block(layer, expert).block_id)
        return self.acquire(block_ids, request_id=request_id)

    def release(
        self,
        lease_id: str,
        *,
        missing_ok: bool = False,
    ) -> Dict[str, Any]:
        with self._lock:
            lease = self._leases.pop(str(lease_id), None)
        if lease is None:
            if missing_ok:
                return {"lease_id": str(lease_id), "released": False}
            raise LeaseError(f"external lease {lease_id!r} does not exist")

        released = 0
        for cache_lease_id in lease.cache_lease_ids:
            released += int(self.cache.release(cache_lease_id))
        return {
            "lease_id": lease.lease_id,
            "released": True,
            "blocks_released": released,
        }

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
                **self.cache.snapshot(),
            },
            "io": self.store.stats(),
            "routing": {
                "hits": self._hits,
                "misses": self._misses,
                "active_external_leases": active,
            },
        }

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            lease_ids = list(self._leases)
        for lease_id in lease_ids:
            self.release(lease_id, missing_ok=True)
        self.store.close()
        self.cache.close()
        with self._lock:
            self._closed = True
            self._started = False

    def __enter__(self) -> "ExternalGate":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
