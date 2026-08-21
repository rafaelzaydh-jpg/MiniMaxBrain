"""Client contract for a tensor runtime or the future internal gate."""
from __future__ import annotations

import json
import socket
from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import Any, Dict, Sequence

from .errors import ProtocolError
from .protocol import IPC_PROTOCOL


class ExternalGateClient:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 55321,
        *,
        api_token: str | None = None,
        timeout_seconds: float = 120.0,
        max_response_bytes: int = 8 << 20,
    ):
        self.host = host
        self.port = int(port)
        self.api_token = api_token
        self.timeout_seconds = float(timeout_seconds)
        self.max_response_bytes = int(max_response_bytes)
        self._map_revision: str | None = None

    def _call(self, op: str, **payload: Any) -> Any:
        request = {"protocol": IPC_PROTOCOL, "op": op, "api_token": self.api_token, **payload}
        encoded = json.dumps(request, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n"
        with socket.create_connection((self.host, self.port), timeout=self.timeout_seconds) as sock:
            sock.settimeout(self.timeout_seconds)
            sock.sendall(encoded)
            chunks = bytearray()
            while True:
                chunk = sock.recv(min(65536, self.max_response_bytes + 1 - len(chunks)))
                if not chunk:
                    break
                chunks.extend(chunk)
                if b"\n" in chunk:
                    break
                if len(chunks) > self.max_response_bytes:
                    raise ProtocolError("external gate response exceeded configured limit")
        if len(chunks) > self.max_response_bytes:
            raise ProtocolError("external gate response exceeded configured limit")
        try:
            response = json.loads(bytes(chunks).split(b"\n", 1)[0].decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProtocolError(f"invalid response from external gate: {exc}") from exc
        if not isinstance(response, dict) or response.get("protocol") != IPC_PROTOCOL:
            raise ProtocolError("response has an invalid protocol envelope")
        if response.get("ok") is not True:
            error = response.get("error") if isinstance(response.get("error"), dict) else {}
            raise ProtocolError(f"{error.get('code', 'REMOTE_ERROR')}: {error.get('detail', '')}")
        return response.get("result")

    def hello(self) -> Dict[str, Any]:
        result = self._call("hello")
        self._map_revision = str(result["map_revision"])
        return result

    def _revision(self) -> str:
        if self._map_revision is None:
            self.hello()
        assert self._map_revision is not None
        return self._map_revision

    def prefetch(self, items: Sequence[Dict[str, Any]], *, request_id: str | None = None) -> Dict[str, Any]:
        return self._call(
            "prefetch", items=list(items), request_id=request_id, map_revision=self._revision()
        )

    def acquire(
        self,
        block_ids: Sequence[str],
        *,
        request_id: str | None = None,
    ) -> "RemoteLease":
        result = self._call(
            "acquire", block_ids=list(block_ids), request_id=request_id, map_revision=self._revision()
        )
        return RemoteLease(self, result)

    def acquire_routes(
        self,
        routes: Sequence[Dict[str, int]],
        *,
        request_id: str | None = None,
    ) -> "RemoteLease":
        result = self._call(
            "acquire", routes=list(routes), request_id=request_id, map_revision=self._revision()
        )
        return RemoteLease(self, result)

    def release(self, lease_id: str) -> Dict[str, Any]:
        return self._call("release", lease_id=str(lease_id))

    def stats(self) -> Dict[str, Any]:
        return self._call("stats")


@dataclass
class MappedBlock:
    descriptor: Dict[str, Any]
    segment: shared_memory.SharedMemory

    @property
    def block_id(self) -> str:
        return str(self.descriptor["block_id"])

    def view(self) -> memoryview:
        offset = int(self.descriptor.get("offset", 0))
        length = int(self.descriptor["length"])
        return self.segment.buf[offset:offset + length]

    def close(self) -> None:
        self.segment.close()


class RemoteLease:
    def __init__(self, client: ExternalGateClient, result: Dict[str, Any]):
        self.client = client
        self.result = result
        self.lease_id = str(result["lease_id"])
        self.blocks: list[MappedBlock] = []
        self._released = False
        try:
            for descriptor in result.get("blocks") or []:
                if descriptor.get("transport") != "shared_memory":
                    raise ProtocolError("remote acquisition did not return shared-memory descriptors")
                segment = shared_memory.SharedMemory(name=str(descriptor["name"]), create=False)
                self.blocks.append(MappedBlock(dict(descriptor), segment))
        except Exception:
            for block in self.blocks:
                block.close()
            self.client.release(self.lease_id)
            raise

    def release(self) -> None:
        if self._released:
            return
        for block in self.blocks:
            block.close()
        self.client.release(self.lease_id)
        self._released = True

    def __enter__(self) -> "RemoteLease":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()
