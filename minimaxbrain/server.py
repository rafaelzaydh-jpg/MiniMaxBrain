"""Small loopback JSON-lines control plane for the independent gate process."""
from __future__ import annotations

import hmac
import json
import socketserver
from typing import Any, Dict

from .errors import MMBError, ProtocolError
from .external import ExternalGate
from .protocol import IPC_PROTOCOL, error_response, ok_response, validate_request


class ExternalGateServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, gate: ExternalGate):
        if gate.config.memory.transport != "shared_memory":
            raise ProtocolError("independent server requires memory.transport='shared_memory'")
        self.gate = gate
        self.max_request_bytes = gate.config.server.max_request_bytes
        self.api_token = gate.config.server.api_token
        super().__init__((gate.config.server.host, gate.config.server.port), _RequestHandler)

    def authorized(self, supplied: Any) -> bool:
        if self.api_token is None:
            return supplied is None
        return isinstance(supplied, str) and hmac.compare_digest(self.api_token, supplied)

    def dispatch(self, request: Dict[str, Any]) -> Dict[str, Any]:
        op = request["op"]
        if op in {"hello", "ping"}:
            return ok_response(op, {
                "service": "MiniMaxBrain External Gate",
                "protocol": IPC_PROTOCOL,
                "model_id": self.gate.model_map.model_id,
                "map_revision": self.gate.model_map.map_revision,
                "transport": self.gate.config.memory.transport,
            })
        if op == "prefetch":
            if request["map_revision"] != self.gate.model_map.map_revision:
                raise ProtocolError("prefetch map_revision does not match the loaded physical map")
            return ok_response(op, self.gate.prefetch(request["items"]))
        if op == "acquire":
            if request["map_revision"] != self.gate.model_map.map_revision:
                raise ProtocolError("acquire map_revision does not match the loaded physical map")
            if "block_ids" in request:
                result = self.gate.acquire(request["block_ids"], request_id=request.get("request_id"))
            else:
                result = self.gate.acquire_routes(request["routes"], request_id=request.get("request_id"))
            return ok_response(op, result)
        if op == "release":
            return ok_response(op, self.gate.release(request["lease_id"]))
        if op == "stats":
            return ok_response(op, self.gate.snapshot())
        raise ProtocolError(f"unsupported operation: {op}")


class _RequestHandler(socketserver.StreamRequestHandler):
    server: ExternalGateServer

    def handle(self) -> None:
        op = None
        try:
            raw = self.rfile.readline(self.server.max_request_bytes + 1)
            if len(raw) > self.server.max_request_bytes or not raw.endswith(b"\n"):
                raise ProtocolError("request is too large or is not newline terminated")
            try:
                value = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ProtocolError(f"request is not valid UTF-8 JSON: {exc}") from exc
            request = validate_request(value)
            op = request.get("op")
            if not self.server.authorized(request.get("api_token")):
                response = error_response(op, "AUTH_FAILED", "invalid API token")
            else:
                response = self.server.dispatch(request)
        except MMBError as exc:
            response = error_response(op, exc.code, exc.detail)
        except Exception as exc:
            response = error_response(op, "INTERNAL_ERROR", type(exc).__name__)
        self.wfile.write(json.dumps(response, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n")


def serve_gate(gate: ExternalGate) -> None:
    gate.start()
    with ExternalGateServer(gate) as server:
        try:
            server.serve_forever(poll_interval=0.25)
        finally:
            gate.close()
