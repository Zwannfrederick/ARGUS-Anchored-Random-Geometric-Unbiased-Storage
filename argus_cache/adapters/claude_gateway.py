"""Anthropic Messages API gateway: Claude Code <-> llama-server (OpenAI API).

Translates:
- Anthropic `/v1/messages` -> OpenAI `/v1/chat/completions`
- System prompts, multi-turn histories, and temperature/top_p controls
- Anthropic `tools` and `tool_use` / `tool_result` blocks <-> OpenAI `function` tool calls
- Server-Sent Events streaming with `input_json_delta` chunks
- Non-streaming responses and token usage tracking
- Zero external server dependencies: uses Python stdlib http.server and httpx

ARGUS Studio (:8090) serves the same routes from the same translator; this
stdlib server is for running the gateway without the Studio stack.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import httpx

from argus_cache.adapters.anthropic_translate import (
    AnthropicStreamTranslator,
    MessageFormatAdapter,
    discover_gateway_models,
    format_sse,
)

logger = logging.getLogger("argus.claude_gateway")

#: Default request context window, mirroring :data:`DEFAULT_LLAMA_SERVER_CTX`:
#: asking for more context than llama-server was started with does not grow the
#: window, it only misreports it.
#:
#: Throughput is governed by whether the KV cache fits in VRAM rather than by
#: how long the window is, so a large window is close to free -- right up until
#: it does not fit, where it stops loading at all. The backend constant records
#: which sizes were measured to load reliably; this tracks it rather than
#: restating the numbers.
DEFAULT_NUM_CTX = 131072

#: Claude Code authenticates local sessions with this placeholder key; a request
#: carrying it is never forwarded to Anthropic's cloud.
LOCAL_API_KEY = "argus-local"

_CLOUD_MODEL_PREFIXES = ("claude", "sonnet", "opus", "haiku", "fable")

STATS_FILE = Path.home() / ".argus_runtime" / "cache_stats.json"


def record_backend_telemetry(
    *,
    backend: str,
    model: str,
    input_tokens: Optional[int],
    output_tokens: Optional[int],
    state: str = "READY",
) -> Dict[str, Any]:
    """Persist what was measured about the last request, and nothing else.

    This deliberately publishes no tier occupancy and no savings percentage:
    the gateway sees tokens, not KV pages, so any tier breakdown rendered from
    here would be fiction presented as instrumentation.

    ``None`` is preserved rather than coerced to 0: zero tokens is a
    measurement, an absent count is not, and a dashboard cannot distinguish
    them once they are collapsed.
    """
    stats = {
        "state": state,
        "backend": backend,
        "model": model,
        "argus_managed": False,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "timestamp": time.time(),
    }
    try:
        STATS_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATS_FILE.write_text(json.dumps(stats))
    except OSError as exc:
        logger.warning("could not write telemetry to %s: %s", STATS_FILE, exc)
    return stats


def is_cloud_request(model: str, api_key: str, authorization: str) -> bool:
    """A Claude model name with real credentials goes to Anthropic, never the local model."""
    is_local = api_key == LOCAL_API_KEY or authorization.endswith(LOCAL_API_KEY)
    return not is_local and model.startswith(_CLOUD_MODEL_PREFIXES)


def estimate_input_tokens(body: bytes) -> int:
    """``count_tokens`` answer: ~4 bytes per token; Claude Code only uses it to budget."""
    return max(1, len(body) // 4)


class ClaudeGatewayHandler(BaseHTTPRequestHandler):
    """HTTP Request Handler translating between Anthropic and OpenAI protocols."""

    upstream_url: str = "http://127.0.0.1:8080/v1"
    num_ctx: int = DEFAULT_NUM_CTX

    def _send_json(self, status: int, payload: Any) -> None:
        body = payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def _start_sse(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/health":
            self._send_json(200, {"status": "ok", "upstream": self.upstream_url})
        elif path in ("/v1/models", "/models"):
            self._send_json(200, {"data": discover_gateway_models(self.upstream_url)})
        else:
            self._send_json(404, {"error": {"type": "not_found_error", "message": path}})

    def do_POST(self):
        parsed = urlparse(self.path)
        content_len = int(self.headers.get("Content-Length", 0))
        post_body = self.rfile.read(content_len) if content_len > 0 else b"{}"

        if parsed.path.startswith("/api/event_logging/"):
            self._send_json(200, {})
            return
        if parsed.path in ("/v1/messages/count_tokens", "/messages/count_tokens"):
            self._send_json(200, {"input_tokens": estimate_input_tokens(post_body)})
            return
        if parsed.path not in ("/v1/messages", "/messages"):
            self._send_json(404, {"error": {"type": "not_found_error", "message": parsed.path}})
            return

        try:
            data = json.loads(post_body.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as e:
            self._send_json(400, {"error": {"type": "invalid_request_error", "message": str(e)}})
            return

        model_name = data.get("model", "qwen3.6-35b-a3b")
        logger.info("request /v1/messages model=%s stream=%s", model_name, bool(data.get("stream")))
        if is_cloud_request(
            model_name,
            (self.headers.get("x-api-key") or "").strip(),
            (self.headers.get("authorization") or "").strip(),
        ):
            self._proxy_cloud(parsed, post_body, bool(data.get("stream")))
            return
        self._forward_local(data, model_name)

    def _proxy_cloud(self, parsed, post_body: bytes, stream: bool) -> None:
        skip = ("host", "content-length", "accept-encoding", "connection", "transfer-encoding")
        headers = {k: v for k, v in self.headers.items() if k.lower() not in skip}
        headers["content-type"] = self.headers.get("Content-Type", "application/json")
        url = f"https://api.anthropic.com{parsed.path}{'?' + parsed.query if parsed.query else ''}"
        try:
            with httpx.Client(timeout=240.0) as client:
                if not stream:
                    resp = client.post(url, content=post_body, headers=headers)
                    self._send_json(resp.status_code, resp.content)
                    return
                with client.stream("POST", url, content=post_body, headers=headers) as upstream:
                    if upstream.status_code != 200:
                        self._send_json(upstream.status_code, upstream.read())
                        return
                    self._start_sse()
                    for chunk in upstream.iter_bytes():
                        self.wfile.write(chunk)
                        self.wfile.flush()
        except httpx.HTTPError as err:
            logger.error("cloud proxy error: %s", err)
            self._send_json(502, {"error": {"type": "api_error", "message": f"Anthropic cloud proxy error: {err}"}})

    def _forward_local(self, data: Dict[str, Any], model_name: str) -> None:
        stream = bool(data.get("stream"))
        data = {**data, "max_tokens": data.get("max_tokens") or 4096}
        openai_req = MessageFormatAdapter.anthropic_request_to_openai(data, num_ctx=self.num_ctx)
        if stream:
            openai_req["stream_options"] = {"include_usage": True}
        endpoint = f"{self.upstream_url.rstrip('/')}/chat/completions"
        record_backend_telemetry(
            backend="llama-server", model=model_name, input_tokens=None, output_tokens=None, state="STREAMING"
        )
        try:
            with httpx.Client(timeout=900.0) as client:
                if not stream:
                    resp = client.post(endpoint, json=openai_req)
                    if resp.status_code != 200:
                        self._fail(model_name, resp.status_code, resp.content)
                        return
                    anthropic_resp = MessageFormatAdapter.openai_response_to_anthropic(resp.json(), model_name)
                    usage = anthropic_resp.get("usage", {})
                    record_backend_telemetry(
                        backend="llama-server",
                        model=model_name,
                        input_tokens=usage.get("input_tokens"),
                        output_tokens=usage.get("output_tokens"),
                    )
                    self._send_json(200, anthropic_resp)
                    return

                with client.stream("POST", endpoint, json=openai_req) as upstream:
                    if upstream.status_code != 200:
                        self._fail(model_name, upstream.status_code, upstream.read())
                        return
                    self._start_sse()
                    translator = AnthropicStreamTranslator(model_name)
                    self._write_events(translator.start())
                    for line in upstream.iter_lines():
                        chunk = AnthropicStreamTranslator.parse_sse_line(line)
                        if chunk:
                            self._write_events(translator.feed(chunk))
                    self._write_events(translator.finish())
                    record_backend_telemetry(
                        backend="llama-server",
                        model=model_name,
                        input_tokens=translator.input_tokens,
                        output_tokens=translator.output_tokens,
                    )
                    self.close_connection = True
        except (httpx.HTTPError, ValueError) as ex:
            logger.error("gateway error: %s", ex)
            self._fail(model_name, 502, {"error": {"type": "api_error", "message": str(ex)}})

    def _write_events(self, events) -> None:
        if events:
            self.wfile.write(b"".join(format_sse(*event) for event in events))
            self.wfile.flush()

    def _fail(self, model_name: str, status: int, body: Any) -> None:
        record_backend_telemetry(
            backend="llama-server", model=model_name, input_tokens=None, output_tokens=None, state="ERROR"
        )
        try:
            self._send_json(status, body)
        except OSError:
            pass  # headers already sent mid-stream; the client sees a truncated stream

    def handle(self):
        try:
            super().handle()
        except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError):
            pass

    def log_message(self, format, *args):
        # Silence default stderr logging during normal proxy operations
        pass


class QuietThreadingHTTPServer(ThreadingHTTPServer):
    """ThreadingHTTPServer that suppresses benign client disconnect stacktraces."""

    def handle_error(self, request, client_address):
        ex_type, _, _ = sys.exc_info()
        if ex_type in (ConnectionResetError, BrokenPipeError, ConnectionAbortedError):
            return
        super().handle_error(request, client_address)


def create_gateway_server(
    port: int = 8000,
    upstream_url: str = "http://127.0.0.1:8080/v1",
    num_ctx: int = DEFAULT_NUM_CTX,
    host: str = "127.0.0.1",
) -> ThreadingHTTPServer:
    """A QuietThreadingHTTPServer serving :class:`ClaudeGatewayHandler`.

    Binds loopback by default: the gateway forwards cloud credentials, so it is
    not something to expose on a LAN by accident.
    """
    ClaudeGatewayHandler.upstream_url = upstream_url
    ClaudeGatewayHandler.num_ctx = num_ctx
    return QuietThreadingHTTPServer((host, port), ClaudeGatewayHandler)


def main():
    parser = argparse.ArgumentParser(description="ARGUS Claude Code Anthropic Messages Gateway")
    parser.add_argument("--port", type=int, default=8000, help="Listening port for gateway")
    parser.add_argument("--host", default="127.0.0.1", help="Listening address (default: loopback)")
    parser.add_argument("--upstream", type=str, default="http://127.0.0.1:8080/v1", help="Upstream llama-server endpoint")
    parser.add_argument(
        "--num-ctx",
        type=int,
        default=DEFAULT_NUM_CTX,
        help=f"Context window to request per message (default: {DEFAULT_NUM_CTX}).",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    server = create_gateway_server(
        port=args.port, upstream_url=args.upstream, num_ctx=args.num_ctx, host=args.host
    )
    print(
        f"Starting ARGUS Claude Code Gateway on {args.host}:{args.port} "
        f"(upstream: {args.upstream}, num_ctx: {args.num_ctx})...",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
