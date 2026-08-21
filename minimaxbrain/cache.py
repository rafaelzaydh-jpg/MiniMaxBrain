"""Lease-aware byte LRU with explicit reservations and no hidden overcommit."""
from __future__ import annotations

import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict

from .errors import BudgetError, LeaseError
from .model_map import WeightBlock
from .storage import ArenaAllocation


@dataclass(frozen=True)
class CacheReservation:
    token: str
    block_id: str
    length: int
    pinned: bool
    speculative: bool


@dataclass
class _Entry:
    block: WeightBlock
    allocation: ArenaAllocation
    pinned: bool
    speculative: bool
    leases: int
    last_used: float


@dataclass(frozen=True)
class CacheLease:
    lease_id: str
    block_id: str
    descriptor: Dict[str, Any]


class ResidentCache:
    def __init__(self, capacity_bytes: int, *, max_expert_entries: int | None = None):
        if capacity_bytes <= 0:
            raise ValueError("cache capacity must be positive")
        self.capacity_bytes = int(capacity_bytes)
        self.max_expert_entries = max_expert_entries
        self._entries: OrderedDict[str, _Entry] = OrderedDict()
        self._reservations: dict[str, CacheReservation] = {}
        self._reserved_ids: set[str] = set()
        self._leases: dict[str, str] = {}
        self._lock = threading.RLock()
        self._evictions = 0

    def _used_bytes(self) -> int:
        return sum(entry.block.length for entry in self._entries.values())

    def _reserved_bytes(self) -> int:
        return sum(item.length for item in self._reservations.values())

    def _expert_count(self) -> int:
        return sum(1 for entry in self._entries.values() if entry.block.kind == "expert")

    def _reserved_expert_count(self) -> int:
        return sum(
            1 for item in self._reservations.values()
            if item.block_id not in self._entries
        )

    def contains(self, block_id: str) -> bool:
        with self._lock:
            return str(block_id) in self._entries

    def is_loading(self, block_id: str) -> bool:
        with self._lock:
            return str(block_id) in self._reserved_ids

    def _eviction_candidate(
        self,
        *,
        require_expert: bool = False,
        speculative_only: bool = False,
    ) -> str | None:
        candidates: list[tuple[int, float, str]] = []
        for block_id, entry in self._entries.items():
            if entry.pinned or entry.leases:
                continue
            if require_expert and entry.block.kind != "expert":
                continue
            if speculative_only and not entry.speculative:
                continue
            candidates.append((0 if entry.speculative else 1, entry.last_used, block_id))
        return min(candidates)[2] if candidates else None

    def _evict(self, block_id: str) -> None:
        entry = self._entries.pop(block_id)
        entry.allocation.close()
        self._evictions += 1

    def reserve(self, block: WeightBlock, *, pinned: bool = False, speculative: bool = False) -> CacheReservation | None:
        with self._lock:
            if block.block_id in self._entries:
                return None
            if block.block_id in self._reserved_ids:
                raise BudgetError(f"block {block.block_id!r} already has an in-flight reservation")
            if block.length > self.capacity_bytes:
                raise BudgetError(
                    f"block {block.block_id!r} ({block.length}) is larger than cache capacity ({self.capacity_bytes})"
                )
            if block.kind == "expert" and self.max_expert_entries is not None:
                while self._expert_count() + self._reserved_expert_count() >= self.max_expert_entries:
                    candidate = self._eviction_candidate(
                        require_expert=True,
                        speculative_only=speculative,
                    )
                    if candidate is None:
                        raise BudgetError("expert slot limit is occupied by pinned or leased blocks")
                    self._evict(candidate)
            while self._used_bytes() + self._reserved_bytes() + block.length > self.capacity_bytes:
                candidate = self._eviction_candidate(speculative_only=speculative)
                if candidate is None:
                    raise BudgetError("RAM budget is occupied by pinned, leased, or in-flight blocks")
                self._evict(candidate)
            reservation = CacheReservation(
                token=uuid.uuid4().hex,
                block_id=block.block_id,
                length=block.length,
                pinned=bool(pinned),
                speculative=bool(speculative),
            )
            self._reservations[reservation.token] = reservation
            self._reserved_ids.add(block.block_id)
            return reservation

    def commit(self, reservation: CacheReservation, block: WeightBlock, allocation: ArenaAllocation) -> bool:
        with self._lock:
            current = self._reservations.pop(reservation.token, None)
            if current != reservation:
                raise BudgetError("cache reservation is missing or stale")
            self._reserved_ids.discard(block.block_id)
            if allocation.length != block.length:
                raise BudgetError("allocation length does not match manifest block length")
            if block.block_id in self._entries:
                allocation.close()
                return False
            self._entries[block.block_id] = _Entry(
                block=block,
                allocation=allocation,
                pinned=reservation.pinned,
                speculative=reservation.speculative,
                leases=0,
                last_used=time.monotonic(),
            )
            return True

    def cancel(self, reservation: CacheReservation) -> None:
        with self._lock:
            current = self._reservations.pop(reservation.token, None)
            if current is not None:
                self._reserved_ids.discard(current.block_id)

    def acquire(self, block_id: str) -> CacheLease | None:
        with self._lock:
            entry = self._entries.get(str(block_id))
            if entry is None:
                return None
            entry.leases += 1
            entry.speculative = False
            entry.last_used = time.monotonic()
            self._entries.move_to_end(entry.block.block_id)
            lease_id = uuid.uuid4().hex
            self._leases[lease_id] = entry.block.block_id
            descriptor = {
                "block_id": entry.block.block_id,
                "kind": entry.block.kind,
                "sha256": entry.block.sha256,
                **entry.allocation.descriptor(),
            }
            return CacheLease(lease_id=lease_id, block_id=entry.block.block_id, descriptor=descriptor)

    def release(self, lease_id: str) -> bool:
        with self._lock:
            block_id = self._leases.pop(str(lease_id), None)
            if block_id is None:
                return False
            entry = self._entries.get(block_id)
            if entry is None or entry.leases <= 0:
                raise LeaseError(f"lease {lease_id!r} points to unavailable block")
            entry.leases -= 1
            entry.last_used = time.monotonic()
            return True

    def expire(self, older_than: float, lease_times: Dict[str, float]) -> list[str]:
        expired = [lease_id for lease_id, created in lease_times.items() if created <= older_than]
        for lease_id in expired:
            self.release(lease_id)
        return expired

    def close(self) -> None:
        with self._lock:
            for entry in list(self._entries.values()):
                entry.allocation.close()
            self._entries.clear()
            self._reservations.clear()
            self._reserved_ids.clear()
            self._leases.clear()

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            used = self._used_bytes()
            return {
                "capacity_bytes": self.capacity_bytes,
                "used_bytes": used,
                "reserved_bytes": self._reserved_bytes(),
                "free_bytes": self.capacity_bytes - used - self._reserved_bytes(),
                "resident_blocks": len(self._entries),
                "resident_experts": self._expert_count(),
                "active_leases": len(self._leases),
                "evictions": self._evictions,
                "blocks": [
                    {
                        "block_id": entry.block.block_id,
                        "kind": entry.block.kind,
                        "bytes": entry.block.length,
                        "pinned": entry.pinned,
                        "speculative": entry.speculative,
                        "leases": entry.leases,
                    }
                    for entry in self._entries.values()
                ],
            }
