"""Versioned IPC contract between model runtime/internal advisor and external gate."""
from __future__ import annotations

from typing import Any, Dict, Mapping

from .errors import ProtocolError


IPC_PROTOCOL = "mmb-external-gate-ipc-v1"
_COMMON = {"protocol", "op", "api_token"}
_FIELDS = {
    "hello": _COMMON,
    "ping": _COMMON,
    "prefetch": _COMMON | {"request_id", "map_revision", "items"},
    "acquire": _COMMON | {"request_id", "map_revision", "block_ids", "routes"},
    "release": _COMMON | {"lease_id"},
    "stats": _COMMON,
}
_PREFETCH_ITEM_FIELDS = {
    "block_id", "priority", "earliest_step", "deadline_step", "confidence"
}
_ROUTE_FIELDS = {"layer", "expert"}


def validate_request(value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ProtocolError("request must be a JSON object")
    protocol = value.get("protocol")
    if protocol != IPC_PROTOCOL:
        raise ProtocolError(f"protocol must be {IPC_PROTOCOL!r}")
    op = value.get("op")
    if op not in _FIELDS:
        raise ProtocolError(f"unsupported operation: {op!r}")
    unknown = sorted(set(value) - _FIELDS[op])
    if unknown:
        raise ProtocolError(f"unknown field(s) for {op}: {', '.join(unknown)}")
    if op == "prefetch" and not isinstance(value.get("items"), list):
        raise ProtocolError("prefetch.items must be an array")
    if op in {"prefetch", "acquire"}:
        revision = value.get("map_revision")
        if not isinstance(revision, str) or not revision:
            raise ProtocolError(f"{op}.map_revision must be a non-empty string")
        request_id = value.get("request_id")
        if request_id is not None and (not isinstance(request_id, str) or not request_id):
            raise ProtocolError(f"{op}.request_id must be null or a non-empty string")
    if op == "prefetch":
        for index, item in enumerate(value["items"]):
            if not isinstance(item, dict):
                raise ProtocolError(f"prefetch.items[{index}] must be an object")
            item_unknown = sorted(set(item) - _PREFETCH_ITEM_FIELDS)
            if item_unknown:
                raise ProtocolError(
                    f"unknown field(s) at prefetch.items[{index}]: {', '.join(item_unknown)}"
                )
            if not isinstance(item.get("block_id"), str) or not item["block_id"]:
                raise ProtocolError(f"prefetch.items[{index}].block_id must be a non-empty string")
            priority = item.get("priority", 100)
            if isinstance(priority, bool) or not isinstance(priority, int):
                raise ProtocolError(f"prefetch.items[{index}].priority must be an integer")
            for key in ("earliest_step", "deadline_step"):
                step = item.get(key)
                if step is not None and (isinstance(step, bool) or not isinstance(step, int) or step < 0):
                    raise ProtocolError(f"prefetch.items[{index}].{key} must be null or an integer >= 0")
            confidence = item.get("confidence")
            if confidence is not None and (
                isinstance(confidence, bool)
                or not isinstance(confidence, (int, float))
                or not 0 <= float(confidence) <= 1
            ):
                raise ProtocolError(f"prefetch.items[{index}].confidence must be null or in [0, 1]")
    if op == "acquire":
        has_blocks = isinstance(value.get("block_ids"), list)
        has_routes = isinstance(value.get("routes"), list)
        if has_blocks == has_routes:
            raise ProtocolError("acquire must provide exactly one of block_ids or routes")
        if has_blocks and (
            not value["block_ids"]
            or any(not isinstance(item, str) or not item for item in value["block_ids"])
        ):
            raise ProtocolError("acquire.block_ids must contain non-empty strings")
        if has_routes:
            if not value["routes"]:
                raise ProtocolError("acquire.routes must not be empty")
            for index, item in enumerate(value["routes"]):
                if not isinstance(item, dict) or set(item) != _ROUTE_FIELDS:
                    raise ProtocolError(f"acquire.routes[{index}] must contain exactly layer and expert")
                if any(isinstance(item[key], bool) or not isinstance(item[key], int) or item[key] < 0 for key in _ROUTE_FIELDS):
                    raise ProtocolError(f"acquire.routes[{index}] values must be integers >= 0")
    if op == "release" and (not isinstance(value.get("lease_id"), str) or not value["lease_id"]):
        raise ProtocolError("release.lease_id must be a non-empty string")
    return value


def ok_response(op: str, result: Any) -> Dict[str, Any]:
    return {"protocol": IPC_PROTOCOL, "ok": True, "op": str(op), "result": result}


def error_response(op: str | None, code: str, detail: str) -> Dict[str, Any]:
    return {
        "protocol": IPC_PROTOCOL,
        "ok": False,
        "op": None if op is None else str(op),
        "error": {"code": str(code), "detail": str(detail)},
    }
