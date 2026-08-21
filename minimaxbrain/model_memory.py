"""Persistent, bounded structural memory for MiniMaxBrain model routing.

This is intentionally not conversational memory.  It stores a reduced model
graph and physical route observations on disk.  Reads are explicit and bounded,
so growing the graph never implies growing the gate's resident RAM set.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping

from .errors import ModelMemoryError, UnknownBlockError
from .model_map import PhysicalModelMap, WeightBlock


MODEL_MEMORY_SCHEMA = "mmb-model-memory-v1"
_ROUTE_STATES = {"hit", "miss", "prefetch_wait"}
_NODE_KINDS = {"model", "layer", "expert", "block", "tensor", "route_signature"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: Mapping[str, Any] | None) -> str:
    try:
        return json.dumps(dict(value or {}), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise ModelMemoryError("payload must be a JSON object") from exc


def _decode(value: str) -> Dict[str, Any]:
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ModelMemoryError("stored model-memory payload is invalid") from exc
    if not isinstance(decoded, dict):
        raise ModelMemoryError("stored model-memory payload is not an object")
    return decoded


def _text(value: Any, name: str) -> str:
    result = str(value or "").strip()
    if not result:
        raise ModelMemoryError(f"{name} must be a non-empty string")
    return result


class ModelMemory:
    """SQLite graph bound to one exact physical model-map identity."""

    def __init__(self, path: str | Path, model_map: PhysicalModelMap):
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.model_map = model_map
        self.model_identity = model_map.identity
        self._lock = threading.RLock()
        try:
            self._connection = sqlite3.connect(self.path, timeout=10.0, check_same_thread=False)
            self._connection.row_factory = sqlite3.Row
            self._connection.execute("PRAGMA foreign_keys = ON")
            self._connection.execute("PRAGMA journal_mode = WAL")
            self._connection.execute("PRAGMA synchronous = FULL")
            self._ensure_schema()
            self._register_map()
        except (OSError, sqlite3.Error) as exc:
            raise ModelMemoryError(f"cannot open model memory: {exc}") from exc

    def _ensure_schema(self) -> None:
        with self._connection:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS mmb_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS models (
                    identity TEXT PRIMARY KEY,
                    model_id TEXT NOT NULL,
                    map_revision TEXT NOT NULL,
                    architecture TEXT NOT NULL,
                    parameter_count INTEGER NOT NULL,
                    map_path TEXT NOT NULL,
                    registered_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS nodes (
                    model_identity TEXT NOT NULL REFERENCES models(identity) ON DELETE CASCADE,
                    node_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    revision INTEGER NOT NULL CHECK(revision >= 1),
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(model_identity,node_id)
                );
                CREATE INDEX IF NOT EXISTS idx_nodes_kind
                    ON nodes(model_identity,kind,node_id);
                CREATE TABLE IF NOT EXISTS edges (
                    model_identity TEXT NOT NULL REFERENCES models(identity) ON DELETE CASCADE,
                    edge_id TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    relation TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    revision INTEGER NOT NULL CHECK(revision >= 1),
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(model_identity,edge_id),
                    FOREIGN KEY(model_identity,source_id) REFERENCES nodes(model_identity,node_id),
                    FOREIGN KEY(model_identity,target_id) REFERENCES nodes(model_identity,node_id)
                );
                CREATE INDEX IF NOT EXISTS idx_edges_source
                    ON edges(model_identity,source_id,edge_id);
                CREATE INDEX IF NOT EXISTS idx_edges_target
                    ON edges(model_identity,target_id,edge_id);
                CREATE TABLE IF NOT EXISTS route_stats (
                    model_identity TEXT NOT NULL REFERENCES models(identity) ON DELETE CASCADE,
                    layer INTEGER NOT NULL CHECK(layer >= 0),
                    expert INTEGER NOT NULL CHECK(expert >= 0),
                    block_id TEXT NOT NULL,
                    requests INTEGER NOT NULL DEFAULT 0,
                    hits INTEGER NOT NULL DEFAULT 0,
                    misses INTEGER NOT NULL DEFAULT 0,
                    prefetch_waits INTEGER NOT NULL DEFAULT 0,
                    requested_bytes INTEGER NOT NULL DEFAULT 0,
                    admitted_bytes INTEGER NOT NULL DEFAULT 0,
                    total_latency_ms REAL NOT NULL DEFAULT 0.0,
                    last_used_at TEXT NOT NULL,
                    PRIMARY KEY(model_identity,layer,expert)
                );
                CREATE INDEX IF NOT EXISTS idx_route_profile
                    ON route_stats(model_identity,layer,requests DESC,expert);
                CREATE TABLE IF NOT EXISTS events (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_identity TEXT NOT NULL REFERENCES models(identity) ON DELETE CASCADE,
                    event_type TEXT NOT NULL,
                    entity_id TEXT NOT NULL,
                    entity_revision INTEGER,
                    map_revision TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_events_model_seq
                    ON events(model_identity,seq);
                """
            )
            row = self._connection.execute(
                "SELECT value FROM mmb_meta WHERE key='schema_version'"
            ).fetchone()
            if row is None:
                self._connection.execute(
                    "INSERT INTO mmb_meta(key,value) VALUES('schema_version',?)",
                    (MODEL_MEMORY_SCHEMA,),
                )
            elif str(row["value"]) != MODEL_MEMORY_SCHEMA:
                raise ModelMemoryError(
                    f"model-memory schema is {row['value']!r}, expected {MODEL_MEMORY_SCHEMA!r}"
                )

    def _register_map(self) -> None:
        now = _now()
        with self._lock, self._connection:
            existing = self._connection.execute(
                "SELECT * FROM models WHERE identity=?", (self.model_identity,)
            ).fetchone()
            expected = {
                "model_id": self.model_map.model_id,
                "map_revision": self.model_map.map_revision,
                "architecture": self.model_map.architecture,
                "parameter_count": self.model_map.parameter_count,
                "map_path": str(self.model_map.path.resolve()),
            }
            if existing is not None:
                actual = {key: existing[key] for key in expected}
                if actual != expected:
                    raise ModelMemoryError("physical model identity conflicts with stored metadata")
                return
            self._connection.execute(
                "INSERT INTO models(identity,model_id,map_revision,architecture,parameter_count,map_path,registered_at) "
                "VALUES(?,?,?,?,?,?,?)",
                (
                    self.model_identity,
                    self.model_map.model_id,
                    self.model_map.map_revision,
                    self.model_map.architecture,
                    self.model_map.parameter_count,
                    str(self.model_map.path.resolve()),
                    now,
                ),
            )
            self._insert_node(
                f"model:{self.model_map.model_id}",
                "model",
                {
                    "architecture": self.model_map.architecture,
                    "parameter_count": self.model_map.parameter_count,
                    "map_revision": self.model_map.map_revision,
                    "quantization": self.model_map.quantization_name,
                },
                now,
            )
            layer_ids: set[str] = set()
            for block in self.model_map.blocks:
                if block.route_key is None:
                    node_id = f"block:{block.block_id}"
                    self._insert_node(node_id, "block", self._block_payload(block), now)
                    self._insert_edge(
                        f"contains:model:{block.block_id}",
                        f"model:{self.model_map.model_id}",
                        "contains",
                        node_id,
                        {},
                        now,
                    )
                    continue
                layer_id = f"layer:{block.layer}"
                if layer_id not in layer_ids:
                    self._insert_node(layer_id, "layer", {"layer": block.layer}, now)
                    self._insert_edge(
                        f"contains:model:layer:{block.layer}",
                        f"model:{self.model_map.model_id}",
                        "contains",
                        layer_id,
                        {},
                        now,
                    )
                    layer_ids.add(layer_id)
                expert_id = f"expert:{block.layer}:{block.expert}"
                self._insert_node(expert_id, "expert", self._block_payload(block), now)
                self._insert_edge(
                    f"contains:layer:{block.layer}:expert:{block.expert}",
                    layer_id,
                    "contains",
                    expert_id,
                    {},
                    now,
                )
            self._event(
                "map_registered",
                f"model:{self.model_map.model_id}",
                1,
                {"nodes": 1 + len(layer_ids) + len(self.model_map.blocks)},
                now,
            )

    @staticmethod
    def _block_payload(block: WeightBlock) -> Dict[str, Any]:
        return {
            "block_id": block.block_id,
            "kind": block.kind,
            "layer": block.layer,
            "expert": block.expert,
            "length": block.length,
            "shard": block.shard_name,
            "offset": block.offset,
            "sha256": block.sha256,
        }

    def _insert_node(self, node_id: str, kind: str, payload: Mapping[str, Any], now: str) -> None:
        self._connection.execute(
            "INSERT INTO nodes(model_identity,node_id,kind,revision,payload_json,created_at,updated_at) "
            "VALUES(?,?,?,1,?,?,?)",
            (self.model_identity, node_id, kind, _json(payload), now, now),
        )

    def _insert_edge(
        self,
        edge_id: str,
        source_id: str,
        relation: str,
        target_id: str,
        payload: Mapping[str, Any],
        now: str,
    ) -> None:
        self._connection.execute(
            "INSERT INTO edges(model_identity,edge_id,source_id,relation,target_id,revision,payload_json,created_at,updated_at) "
            "VALUES(?,?,?,?,?,1,?,?,?)",
            (self.model_identity, edge_id, source_id, relation, target_id, _json(payload), now, now),
        )

    def _event(
        self,
        event_type: str,
        entity_id: str,
        revision: int | None,
        payload: Mapping[str, Any],
        now: str | None = None,
    ) -> None:
        self._connection.execute(
            "INSERT INTO events(model_identity,event_type,entity_id,entity_revision,map_revision,payload_json,created_at) "
            "VALUES(?,?,?,?,?,?,?)",
            (
                self.model_identity,
                event_type,
                entity_id,
                revision,
                self.model_map.map_revision,
                _json(payload),
                now or _now(),
            ),
        )

    def put_node(
        self,
        node_id: str,
        kind: str,
        payload: Mapping[str, Any],
        *,
        expected_revision: int | None = None,
    ) -> Dict[str, Any]:
        """Create or revise an internal-model node with optimistic locking."""
        node_id = _text(node_id, "node_id")
        kind = _text(kind, "kind")
        if kind not in _NODE_KINDS:
            raise ModelMemoryError(f"unsupported node kind: {kind}")
        encoded = _json(payload)
        now = _now()
        with self._lock, self._connection:
            row = self._connection.execute(
                "SELECT kind,revision,payload_json FROM nodes WHERE model_identity=? AND node_id=?",
                (self.model_identity, node_id),
            ).fetchone()
            if row is None:
                if expected_revision is not None:
                    raise ModelMemoryError(f"node {node_id!r} does not exist at revision {expected_revision}")
                revision = 1
                self._insert_node(node_id, kind, payload, now)
                event_type = "node_created"
            else:
                current = int(row["revision"])
                if expected_revision is None:
                    if str(row["kind"]) == kind and str(row["payload_json"]) == encoded:
                        return self.node(node_id)
                    raise ModelMemoryError(f"node {node_id!r} already exists; expected_revision is required")
                if expected_revision != current:
                    raise ModelMemoryError(
                        f"node {node_id!r} revision conflict: expected {expected_revision}, current {current}"
                    )
                revision = current + 1
                self._connection.execute(
                    "UPDATE nodes SET kind=?,revision=?,payload_json=?,updated_at=? "
                    "WHERE model_identity=? AND node_id=?",
                    (kind, revision, encoded, now, self.model_identity, node_id),
                )
                event_type = "node_revised"
            self._event(event_type, node_id, revision, {"kind": kind}, now)
        return self.node(node_id)

    def put_edge(
        self,
        edge_id: str,
        source_id: str,
        relation: str,
        target_id: str,
        payload: Mapping[str, Any] | None = None,
        *,
        expected_revision: int | None = None,
    ) -> Dict[str, Any]:
        """Create or revise an edge while pinning its exact revision."""
        edge_id = _text(edge_id, "edge_id")
        source_id = _text(source_id, "source_id")
        relation = _text(relation, "relation")
        target_id = _text(target_id, "target_id")
        encoded = _json(payload)
        now = _now()
        with self._lock, self._connection:
            row = self._connection.execute(
                "SELECT source_id,relation,target_id,revision,payload_json FROM edges "
                "WHERE model_identity=? AND edge_id=?",
                (self.model_identity, edge_id),
            ).fetchone()
            if row is None:
                if expected_revision is not None:
                    raise ModelMemoryError(f"edge {edge_id!r} does not exist at revision {expected_revision}")
                revision = 1
                try:
                    self._insert_edge(edge_id, source_id, relation, target_id, payload or {}, now)
                except sqlite3.IntegrityError as exc:
                    raise ModelMemoryError("edge endpoints must name existing nodes") from exc
                event_type = "edge_created"
            else:
                current = int(row["revision"])
                same = (
                    str(row["source_id"]) == source_id
                    and str(row["relation"]) == relation
                    and str(row["target_id"]) == target_id
                    and str(row["payload_json"]) == encoded
                )
                if expected_revision is None:
                    if same:
                        return self.edge(edge_id)
                    raise ModelMemoryError(f"edge {edge_id!r} already exists; expected_revision is required")
                if expected_revision != current:
                    raise ModelMemoryError(
                        f"edge {edge_id!r} revision conflict: expected {expected_revision}, current {current}"
                    )
                revision = current + 1
                try:
                    self._connection.execute(
                        "UPDATE edges SET source_id=?,relation=?,target_id=?,revision=?,payload_json=?,updated_at=? "
                        "WHERE model_identity=? AND edge_id=?",
                        (
                            source_id,
                            relation,
                            target_id,
                            revision,
                            encoded,
                            now,
                            self.model_identity,
                            edge_id,
                        ),
                    )
                except sqlite3.IntegrityError as exc:
                    raise ModelMemoryError("edge endpoints must name existing nodes") from exc
                event_type = "edge_revised"
            self._event(event_type, edge_id, revision, {"relation": relation}, now)
        return self.edge(edge_id)

    def node(self, node_id: str) -> Dict[str, Any]:
        with self._lock:
            row = self._connection.execute(
                "SELECT node_id,kind,revision,payload_json,created_at,updated_at FROM nodes "
                "WHERE model_identity=? AND node_id=?",
                (self.model_identity, _text(node_id, "node_id")),
            ).fetchone()
        if row is None:
            raise ModelMemoryError(f"unknown model-memory node: {node_id}")
        return {
            "node_id": str(row["node_id"]),
            "kind": str(row["kind"]),
            "revision": int(row["revision"]),
            "payload": _decode(str(row["payload_json"])),
            "created_at": str(row["created_at"]),
            "updated_at": str(row["updated_at"]),
            "map_revision": self.model_map.map_revision,
        }

    def edge(self, edge_id: str) -> Dict[str, Any]:
        with self._lock:
            row = self._connection.execute(
                "SELECT edge_id,source_id,relation,target_id,revision,payload_json,created_at,updated_at "
                "FROM edges WHERE model_identity=? AND edge_id=?",
                (self.model_identity, _text(edge_id, "edge_id")),
            ).fetchone()
        if row is None:
            raise ModelMemoryError(f"unknown model-memory edge: {edge_id}")
        return {
            "edge_id": str(row["edge_id"]),
            "source_id": str(row["source_id"]),
            "relation": str(row["relation"]),
            "target_id": str(row["target_id"]),
            "revision": int(row["revision"]),
            "payload": _decode(str(row["payload_json"])),
            "created_at": str(row["created_at"]),
            "updated_at": str(row["updated_at"]),
            "map_revision": self.model_map.map_revision,
        }

    def neighbors(self, node_id: str, *, limit: int = 64, cursor: str | None = None) -> Dict[str, Any]:
        """Return one bounded graph page; no implicit recursive expansion."""
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 256:
            raise ModelMemoryError("neighbor limit must be an integer in [1, 256]")
        node_id = _text(node_id, "node_id")
        after = str(cursor or "")
        with self._lock:
            rows = self._connection.execute(
                "SELECT edge_id,source_id,relation,target_id,revision,payload_json "
                "FROM edges WHERE model_identity=? AND (source_id=? OR target_id=?) AND edge_id>? "
                "ORDER BY edge_id LIMIT ?",
                (self.model_identity, node_id, node_id, after, limit + 1),
            ).fetchall()
        has_more = len(rows) > limit
        page = rows[:limit]
        items = [
            {
                "edge_id": str(row["edge_id"]),
                "source_id": str(row["source_id"]),
                "relation": str(row["relation"]),
                "target_id": str(row["target_id"]),
                "revision": int(row["revision"]),
                "payload": _decode(str(row["payload_json"])),
            }
            for row in page
        ]
        return {
            "node_id": node_id,
            "items": items,
            "next_cursor": str(page[-1]["edge_id"]) if has_more and page else None,
            "limit": limit,
            "map_revision": self.model_map.map_revision,
        }

    def record_route(
        self,
        layer: int,
        expert: int,
        *,
        state: str,
        duration_ms: float = 0.0,
    ) -> Dict[str, Any]:
        """Accumulate physical route behavior for future bounded prefetch advice."""
        if isinstance(layer, bool) or isinstance(expert, bool):
            raise ModelMemoryError("layer and expert must be non-negative integers")
        try:
            block = self.model_map.route_block(int(layer), int(expert))
        except (UnknownBlockError, TypeError, ValueError) as exc:
            raise ModelMemoryError(f"unknown route layer={layer}, expert={expert}") from exc
        state = str(state)
        if state not in _ROUTE_STATES:
            raise ModelMemoryError(f"route state must be one of {sorted(_ROUTE_STATES)}")
        try:
            latency = float(duration_ms)
        except (TypeError, ValueError) as exc:
            raise ModelMemoryError("duration_ms must be a non-negative number") from exc
        if latency < 0:
            raise ModelMemoryError("duration_ms must be a non-negative number")
        now = _now()
        state_column = {"hit": "hits", "miss": "misses", "prefetch_wait": "prefetch_waits"}[state]
        admitted = block.length if state == "miss" else 0
        with self._lock, self._connection:
            self._connection.execute(
                "INSERT INTO route_stats(model_identity,layer,expert,block_id,requests,hits,misses,prefetch_waits,"
                "requested_bytes,admitted_bytes,total_latency_ms,last_used_at) "
                "VALUES(?,?,?,?,1,?,?,?, ?,?,?,?) "
                "ON CONFLICT(model_identity,layer,expert) DO UPDATE SET "
                "requests=requests+1,"
                f"{state_column}={state_column}+1,"
                "requested_bytes=requested_bytes+excluded.requested_bytes,"
                "admitted_bytes=admitted_bytes+excluded.admitted_bytes,"
                "total_latency_ms=total_latency_ms+excluded.total_latency_ms,"
                "last_used_at=excluded.last_used_at",
                (
                    self.model_identity,
                    int(layer),
                    int(expert),
                    block.block_id,
                    1 if state == "hit" else 0,
                    1 if state == "miss" else 0,
                    1 if state == "prefetch_wait" else 0,
                    block.length,
                    admitted,
                    latency,
                    now,
                ),
            )
            self._event(
                "route_observed",
                f"expert:{int(layer)}:{int(expert)}",
                None,
                {"state": state, "duration_ms": latency, "block_id": block.block_id},
                now,
            )
        return self.route_profile(int(layer), limit=256, expert=int(expert))["items"][0]

    def route_profile(
        self,
        layer: int,
        *,
        limit: int = 32,
        expert: int | None = None,
    ) -> Dict[str, Any]:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 256:
            raise ModelMemoryError("profile limit must be an integer in [1, 256]")
        clauses = ["model_identity=?", "layer=?"]
        values: list[Any] = [self.model_identity, int(layer)]
        if expert is not None:
            clauses.append("expert=?")
            values.append(int(expert))
        values.append(limit)
        with self._lock:
            rows = self._connection.execute(
                "SELECT layer,expert,block_id,requests,hits,misses,prefetch_waits,requested_bytes,"
                "admitted_bytes,total_latency_ms,last_used_at FROM route_stats WHERE "
                + " AND ".join(clauses)
                + " ORDER BY requests DESC,expert LIMIT ?",
                tuple(values),
            ).fetchall()
        items = []
        for row in rows:
            requests = int(row["requests"])
            items.append({
                "layer": int(row["layer"]),
                "expert": int(row["expert"]),
                "block_id": str(row["block_id"]),
                "requests": requests,
                "hits": int(row["hits"]),
                "misses": int(row["misses"]),
                "prefetch_waits": int(row["prefetch_waits"]),
                "requested_bytes": int(row["requested_bytes"]),
                "admitted_bytes": int(row["admitted_bytes"]),
                "average_latency_ms": round(float(row["total_latency_ms"]) / requests, 6),
                "last_used_at": str(row["last_used_at"]),
            })
        return {
            "layer": int(layer),
            "items": items,
            "limit": limit,
            "map_revision": self.model_map.map_revision,
        }

    def history(self, *, cursor: int = 0, limit: int = 100) -> Dict[str, Any]:
        """Read an append-only bounded history page for audit/replay."""
        if isinstance(cursor, bool) or not isinstance(cursor, int) or cursor < 0:
            raise ModelMemoryError("history cursor must be an integer >= 0")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 1000:
            raise ModelMemoryError("history limit must be an integer in [1, 1000]")
        with self._lock:
            rows = self._connection.execute(
                "SELECT seq,event_type,entity_id,entity_revision,map_revision,payload_json,created_at "
                "FROM events WHERE model_identity=? AND seq>? ORDER BY seq LIMIT ?",
                (self.model_identity, cursor, limit + 1),
            ).fetchall()
        has_more = len(rows) > limit
        page = rows[:limit]
        items = [
            {
                "seq": int(row["seq"]),
                "event_type": str(row["event_type"]),
                "entity_id": str(row["entity_id"]),
                "entity_revision": int(row["entity_revision"]) if row["entity_revision"] is not None else None,
                "map_revision": str(row["map_revision"]),
                "payload": _decode(str(row["payload_json"])),
                "created_at": str(row["created_at"]),
            }
            for row in page
        ]
        return {
            "items": items,
            "next_cursor": int(page[-1]["seq"]) if has_more and page else None,
            "limit": limit,
            "map_revision": self.model_map.map_revision,
        }

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            nodes = int(self._connection.execute(
                "SELECT COUNT(*) FROM nodes WHERE model_identity=?", (self.model_identity,)
            ).fetchone()[0])
            edges = int(self._connection.execute(
                "SELECT COUNT(*) FROM edges WHERE model_identity=?", (self.model_identity,)
            ).fetchone()[0])
            routes = int(self._connection.execute(
                "SELECT COUNT(*) FROM route_stats WHERE model_identity=?", (self.model_identity,)
            ).fetchone()[0])
        return {
            "schema_version": MODEL_MEMORY_SCHEMA,
            "model_identity": self.model_identity,
            "model_id": self.model_map.model_id,
            "map_revision": self.model_map.map_revision,
            "nodes": nodes,
            "edges": edges,
            "observed_routes": routes,
            "database": str(self.path),
        }

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> "ModelMemory":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
