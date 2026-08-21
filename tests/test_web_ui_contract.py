from __future__ import annotations

import json
import re
import socket
import time
import urllib.request
from types import SimpleNamespace

from minimaxbrain.runtime import InferenceMode, MMBRuntime
from minimaxbrain.server_http import WEB_UI_HTML, start_mmb_server


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class _FakeBackend:
    def stream_chat(self, *_args, **_kwargs):
        for piece in ("Olá", " ", "mundo"):
            time.sleep(0.003)
            yield piece

    def stats(self):
        return {
            "resident_bytes": 64,
            "paged_experts_used": True,
            "real_router_requests": 3,
            "bytes_read": 128,
        }


def test_runtime_stream_exposes_live_tokens_per_second():
    runtime = MMBRuntime.__new__(MMBRuntime)
    runtime.n_ctx = 128
    runtime.backend = _FakeBackend()
    runtime.backend_error = None
    runtime.inference_mode = InferenceMode.PAGED_MMB
    runtime._closed = False
    runtime._last_generation_stats = {
        "tokens_generated": 0,
        "tokens_per_second": None,
        "ttft_ms": None,
        "generation_elapsed_ms": 0.0,
        "decode_elapsed_ms": 0.0,
    }

    events = list(
        runtime.stream_chat(
            [{"role": "user", "content": "oi"}],
            max_tokens=8,
            temperature=0.0,
            top_p=1.0,
            top_k=0,
        )
    )

    assert [piece for piece, _ in events] == ["Olá", " ", "mundo"]
    assert events[0][1]["tokens_generated"] == 1
    assert events[0][1]["tokens_per_second"] is None
    assert events[-1][1]["tokens_generated"] == 3
    assert events[-1][1]["tokens_per_second"] > 0
    assert events[-1][1]["ttft_ms"] >= 0
    assert runtime._last_generation_stats["tokens_generated"] == 3


class _FakeEngine:
    def __init__(self):
        self.ready = True
        self.backend_error = None
        self.inference_mode = InferenceMode.PAGED_MMB
        self.model_map = SimpleNamespace(model_id="fake-qwen")
        self.config = SimpleNamespace(
            server=SimpleNamespace(api_token=None, max_request_bytes=65536),
            memory=SimpleNamespace(ram_budget_bytes=1024),
        )

    def stats(self):
        return {
            "model_id": "fake-qwen",
            "status": "ready",
            "ready": True,
            "inference_mode": "paged_mmb",
            "backend_error": None,
            "backend_rss_bytes": 512,
            "expert_cache_bytes": 64,
            "expert_cache_budget_bytes": 256,
            "ram_budget_bytes": 1024,
            "native_pager": {
                "cache_hits": 3,
                "cache_misses": 1,
                "bytes_read": 128,
                "resident_bytes": 64,
                "peak_resident_bytes": 64,
                "real_router_requests": 2,
                "acquire_ns": 1000,
                "io_ns": 500,
            },
            "generation": {
                "tokens_generated": 2,
                "tokens_per_second": 1.5,
                "ttft_ms": 120.0,
                "generation_elapsed_ms": 900.0,
                "decode_elapsed_ms": 666.0,
            },
        }

    def validate_generation_params(self, *, max_tokens, temperature, top_p, top_k):
        return max_tokens, temperature, top_p, top_k

    def _validate_messages(self, messages):
        return MMBRuntime._validate_messages(messages)

    def stream_chat(self, *_args, **_kwargs):
        yield "Olá", {
            "tokens_generated": 1,
            "tokens_per_second": None,
            "ttft_ms": 100.0,
        }
        yield " mundo", {
            "tokens_generated": 2,
            "tokens_per_second": 2.0,
            "ttft_ms": 100.0,
        }


def test_http_routes_and_streaming_sse_are_connected():
    engine = _FakeEngine()
    port = _free_port()
    server = start_mmb_server(engine, host="127.0.0.1", port=port)
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health") as response:
            health = json.loads(response.read().decode("utf-8"))
            assert health["ready"] is True

        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/stats") as response:
            stats = json.loads(response.read().decode("utf-8"))
            assert stats["generation"]["tokens_per_second"] == 1.5

        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            data=json.dumps(
                {
                    "messages": [{"role": "user", "content": "oi"}],
                    "stream": True,
                    "max_tokens": 8,
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request) as response:
            raw = response.read().decode("utf-8")

        blocks = [
            block
            for block in re.split(r"\r?\n\r?\n", raw)
            if block.strip()
        ]
        assert blocks[-1].strip() == "data: [DONE]"

        events = []
        for block in blocks[:-1]:
            data_lines = [
                line[5:].lstrip()
                for line in re.split(r"\r?\n", block)
                if line.startswith("data:")
            ]
            events.append(json.loads("\n".join(data_lines)))

        assert "".join(
            event["choices"][0]["delta"]["content"] for event in events
        ) == "Olá mundo"
        assert events[-1]["mmb_stats"]["tokens_per_second"] == 2.0
    finally:
        server.shutdown()
        server.server_close()


def test_web_ui_uses_event_boundaries_and_exposes_tps():
    assert "/health" not in WEB_UI_HTML  # health is server/operator-facing, not polled by UI
    assert "/api/stats" in WEB_UI_HTML
    assert "/v1/chat/completions" in WEB_UI_HTML
    assert "tokens_per_second" in WEB_UI_HTML
    assert "tok/s" in WEB_UI_HTML
    assert r"/\r?\n\r?\n/" in WEB_UI_HTML
    assert "buffer.split('\\\\n')" not in WEB_UI_HTML
    assert "statusDot" not in WEB_UI_HTML


def test_web_ui_is_viewport_bounded_and_scrolls_only_chat_history():
    # The shell must never grow with messages. The middle grid row is the only
    # vertical conversation scroll container.
    assert "height:100dvh" in WEB_UI_HTML
    assert "grid-template-rows:auto minmax(0,1fr) auto" in WEB_UI_HTML
    assert ".chat-region{" in WEB_UI_HTML
    assert "position:absolute;" in WEB_UI_HTML
    assert "overflow-y:auto;" in WEB_UI_HTML
    assert "scrollbar-gutter:stable" in WEB_UI_HTML
    assert "scrollLatest" in WEB_UI_HTML


def test_web_ui_detaches_autoscroll_when_user_reads_older_messages():
    assert "let pinnedToBottom = true" in WEB_UI_HTML
    assert "function distanceFromBottom()" in WEB_UI_HTML
    assert "pinnedToBottom = distanceFromBottom() <= 140" in WEB_UI_HTML
    assert "if(!force && !pinnedToBottom) return" in WEB_UI_HTML
    assert "scrollEl.addEventListener('scroll', updatePinnedState" in WEB_UI_HTML
    assert "↓ Mais recente" in WEB_UI_HTML


def test_web_ui_composer_growth_is_bounded():
    assert "max-height:144px" in WEB_UI_HTML
    assert "Math.min(promptEl.scrollHeight, 144)" in WEB_UI_HTML
    assert "promptEl.style.overflowY" in WEB_UI_HTML
