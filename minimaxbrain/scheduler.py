"""Ordered speculative I/O plus compulsory load promotion."""
from __future__ import annotations

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Dict, Iterable, Sequence

from .cache import ResidentCache
from .errors import BudgetError
from .model_map import WeightBlock
from .storage import FileRangeStore, PayloadArena
from .telemetry import NullTelemetry, TelemetrySink


class LoadScheduler:
    """Coalesces loads and keeps speculation outside the correctness path."""

    def __init__(
        self,
        cache: ResidentCache,
        store: FileRangeStore,
        arena: PayloadArena,
        *,
        workers: int,
        max_prefetch: int,
        telemetry: TelemetrySink | None = None,
    ):
        self.cache = cache
        self.store = store
        self.arena = arena
        self.max_prefetch = int(max_prefetch)
        self.telemetry = telemetry or NullTelemetry()
        self._executor = ThreadPoolExecutor(max_workers=int(workers), thread_name_prefix="mmb-weight-io")
        self._pending: dict[str, Future[bool]] = {}
        self._pending_speculative: set[str] = set()
        self._lock = threading.RLock()
        self._closed = False
        self._prefetch_dropped = 0

    def _load(self, block: WeightBlock, *, pinned: bool, speculative: bool) -> bool:
        if self.cache.contains(block.block_id):
            return False
        started = time.perf_counter()
        reservation = self.cache.reserve(block, pinned=pinned, speculative=speculative)
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
            self.telemetry.emit(
                "block_admit", "ok", duration_ms=(time.perf_counter() - started) * 1000.0,
                metadata={
                    "block_id": block.block_id,
                    "bytes": block.length,
                    "pinned": pinned,
                    "speculative": speculative,
                    "committed": committed,
                },
            )
            return committed
        except Exception as exc:
            self.cache.cancel(reservation)
            if view is not None:
                view.release()
            if allocation is not None:
                allocation.close()
            self.telemetry.emit(
                "block_admit", "error", duration_ms=(time.perf_counter() - started) * 1000.0,
                metadata={"block_id": block.block_id, "error": type(exc).__name__},
            )
            raise

    def load_pinned(self, blocks: Iterable[WeightBlock]) -> None:
        for block in blocks:
            self._load(block, pinned=True, speculative=False)

    def _done(self, block_id: str, future: Future[bool]) -> None:
        with self._lock:
            if self._pending.get(block_id) is future:
                self._pending.pop(block_id, None)
                self._pending_speculative.discard(block_id)
        if future.cancelled():
            return
        try:
            future.exception()
        except Exception:
            pass

    def prefetch(self, blocks: Sequence[WeightBlock]) -> Dict[str, int]:
        accepted = 0
        already = 0
        dropped = 0
        with self._lock:
            if self._closed:
                return {"accepted": 0, "already_resident_or_pending": 0, "dropped": len(blocks)}
            for block in blocks:
                if self.cache.contains(block.block_id) or block.block_id in self._pending:
                    already += 1
                    continue
                if self.max_prefetch <= 0 or len(self._pending_speculative) >= self.max_prefetch:
                    dropped += 1
                    self._prefetch_dropped += 1
                    continue
                future = self._executor.submit(self._load, block, pinned=False, speculative=True)
                self._pending[block.block_id] = future
                self._pending_speculative.add(block.block_id)
                future.add_done_callback(lambda done, block_id=block.block_id: self._done(block_id, done))
                accepted += 1
        return {"accepted": accepted, "already_resident_or_pending": already, "dropped": dropped}

    def ensure(self, block: WeightBlock) -> str:
        """Make a compulsory block resident and report hit/prefetch/miss."""
        if self.cache.contains(block.block_id):
            return "hit"
        future: Future[bool] | None
        with self._lock:
            future = self._pending.get(block.block_id)
            if future is not None and future.cancel():
                self._pending.pop(block.block_id, None)
                self._pending_speculative.discard(block.block_id)
                future = None
        if future is not None:
            try:
                future.result()
                if self.cache.contains(block.block_id):
                    return "prefetch_wait"
            except BudgetError:
                # A speculative attempt may have lost a race against active
                # leases. Compulsory admission receives a fresh physical try.
                pass
        self._load(block, pinned=False, speculative=False)
        return "miss"

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            for future in self._pending.values():
                future.cancel()
        self._executor.shutdown(wait=True, cancel_futures=True)
        with self._lock:
            self._pending.clear()
            self._pending_speculative.clear()

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "pending": len(self._pending),
                "pending_speculative": len(self._pending_speculative),
                "prefetch_dropped": self._prefetch_dropped,
                "max_prefetch": self.max_prefetch,
            }

