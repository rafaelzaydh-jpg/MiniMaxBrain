"""Dependency-free physical telemetry for MiniMaxBrain."""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any, Dict, Protocol


class TelemetrySink(Protocol):
    def emit(self, name: str, status: str, *, duration_ms: float = 0.0, metadata: Dict[str, Any] | None = None) -> None:
        ...


class NullTelemetry:
    def emit(self, name: str, status: str, *, duration_ms: float = 0.0, metadata: Dict[str, Any] | None = None) -> None:
        return None


class MMBTelemetry:
    """Best-effort adapter to Python's standard logging subsystem."""

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.logger = logger or logging.getLogger("minimaxbrain.telemetry")

    def emit(self, name: str, status: str, *, duration_ms: float = 0.0, metadata: Dict[str, Any] | None = None) -> None:
        self.logger.info(
            "event=%s status=%s duration_ms=%.3f metadata=%r",
            name,
            status,
            float(duration_ms),
            dict(metadata or {}),
        )


class MemoryTelemetry:
    """Thread-safe test/embedding sink."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.events: list[dict[str, Any]] = []

    def emit(self, name: str, status: str, *, duration_ms: float = 0.0, metadata: Dict[str, Any] | None = None) -> None:
        with self._lock:
            self.events.append({
                "name": str(name),
                "status": str(status),
                "duration_ms": float(duration_ms),
                "metadata": dict(metadata or {}),
            })
