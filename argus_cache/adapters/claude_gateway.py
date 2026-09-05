"""Anthropic Messages API Gateway for Claude Code <-> llama-server / OpenAI compatibility (Stage S10).

Translates:
- Anthropic `/v1/messages` -> OpenAI `/v1/chat/completions`
- System prompts, multi-turn histories, and temperature/top_p controls
- Anthropic `tools` and `tool_use` / `tool_result` blocks <-> OpenAI `function` tool calls
- Full bidirectional Server-Sent Events (SSE) streaming with `input_json_delta` chunks
- Non-streaming responses and token usage tracking
- Zero external server dependencies: uses Python stdlib http.server and httpx
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import httpx

logger = logging.getLogger("argus.claude_gateway")


STATS_FILE = Path.home() / ".argus_runtime" / "cache_stats.json"

BACKENDS = ("auto", "ollama", "openai")

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

OLLAMA_HOST = "http://127.0.0.1:11434"


def build_ollama_options(
    options: Optional[Dict[str, Any]], *, num_ctx: int
) -> Dict[str, Any]:
    """Request options with the context window applied.

    Kept separate from the request path so the window is a configured value
    rather than a literal buried in a handler.
    """
    merged = dict(options or {})
    merged["num_ctx"] = num_ctx
    return merged


def _probe_openai_upstream(upstream_url: str) -> bool:
    """Whether an OpenAI-compatible server answers at ``upstream_url``."""
    try:
        with httpx.Client(timeout=1.0) as client:
            resp = client.get(f"{upstream_url.rstrip('/')}/models")
            return resp.status_code == 200
    except httpx.HTTPError:
        return False


def resolve_backend(backend: str, upstream_url: str) -> str:
    """Decide which upstream protocol to speak.

    Two backends serve local models and they are not interchangeable:

    * ``ollama`` exposes a native Anthropic ``/v1/messages`` endpoint, so
      requests pass through untranslated;
    * ``openai`` is llama-server, which speaks ``/v1/chat/completions`` and
      therefore needs :class:`MessageFormatAdapter`.

    The distinction matters beyond protocol: ``--cpu-moe`` and per-run context
    sizing are llama-server flags that Ollama does not surface, and on a small
    GPU they were worth roughly 2x decode throughput. Routing was previously
    pinned to Ollama, which left the llama-server path in this module
    unreachable.
    """
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {', '.join(BACKENDS)}; got {backend!r}")
    if backend != "auto":
        return backend
    return "openai" if _probe_openai_upstream(upstream_url) else "ollama"


class SseUsageTracker:
    """Extracts Anthropic ``usage`` counts from a Server-Sent Events stream.

    Token counts are carried by the protocol itself: ``message_start`` reports
    ``input_tokens`` and ``message_delta`` reports the running
    ``output_tokens``. Reading them is the only way to report a real number --
    the alternative this replaced inferred a count from the byte length of each
    chunk, which tracked nothing.

    Chunks arrive on arbitrary byte boundaries, so a partial trailing line is
    buffered until the rest of it shows up.
    """

    def __init__(self) -> None:
        self.input_tokens: Optional[int] = None
        self.output_tokens: Optional[int] = None
        self._buffer = b""

    def feed(self, chunk: bytes) -> None:
        """Consume one chunk of the upstream stream. Never raises."""
        self._buffer += chunk
        while b"\n" in self._buffer:
            line, self._buffer = self._buffer.split(b"\n", 1)
            self._consume_line(line)

    def _consume_line(self, line: bytes) -> None:
        if not line.startswith(b"data:"):
            return
        try:
            event = json.loads(line[len(b"data:") :].strip())
        except (json.JSONDecodeError, UnicodeDecodeError):
            return
        if not isinstance(event, dict):
            return
        # message_start nests usage under `message`; message_delta puts it at
        # the top level.
        for holder in (event.get("message"), event):
            if not isinstance(holder, dict):
                continue
            usage = holder.get("usage")
            if not isinstance(usage, dict):
                continue
            if isinstance(usage.get("input_tokens"), int):
                self.input_tokens = usage["input_tokens"]
            if isinstance(usage.get("output_tokens"), int):
                self.output_tokens = usage["output_tokens"]


def parse_ollama_residency(
    payload: Dict[str, Any], *, model: str
) -> Optional[Dict[str, Any]]:
    """Where a loaded model actually sits, from Ollama's ``/api/ps`` response.

    On a small GPU this is the number that explains throughput: ``size_vram``
    against ``size`` is the share of the model that reached the device, and the
    remainder is being read from host RAM on every token.

    Returns None when the model is not loaded -- an absent model has no
    residency, and reporting zeros would read as "entirely on CPU".
    """
    for entry in payload.get("models", []):
        if entry.get("name") != model:
            continue
        total = int(entry.get("size", 0))
        vram = int(entry.get("size_vram", 0))
        return {
            "total_bytes": total,
            "vram_bytes": vram,
            "cpu_bytes": total - vram,
            "gpu_fraction": (vram / total) if total else 0.0,
            "context_length": entry.get("context_length"),
        }
    return None


def record_backend_telemetry(
    *,
    backend: str,
    model: str,
    input_tokens: Optional[int],
    output_tokens: Optional[int],
    state: str = "READY",
    residency: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Persist what was measured about the last request, and nothing else.

    This deliberately publishes no tier occupancy and no savings percentage.
    The gateway proxies to an external runtime that owns its KV cache in
    another process -- ``OllamaAdapter.capabilities.manages_kv_cache`` is False
    -- so ARGUS neither places those pages nor can observe them. Any tier
    breakdown rendered here would be fiction presented as instrumentation.

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
        "residency": residency,
        "timestamp": time.time(),
    }
    try:
        STATS_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATS_FILE.write_text(json.dumps(stats))
    except OSError as exc:
        logger.warning("could not write telemetry to %s: %s", STATS_FILE, exc)
    return stats


class MessageFormatAdapter:
    """Translates schemas between Anthropic Messages API and OpenAI Chat Completions API."""

    @staticmethod
    def anthropic_tools_to_openai(tools: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
        """Converts Anthropic tool schemas to OpenAI tool schemas."""
        if not tools:
            return None
        openai_tools = []
        for t in tools:
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": t.get("name"),
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
                },
            })
        return openai_tools

    @classmethod
    def anthropic_messages_to_openai(
        cls,
        messages: List[Dict[str, Any]],
        system_prompt: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """Converts Anthropic conversation history to OpenAI message list."""
        openai_msgs: List[Dict[str, Any]] = []

        # Handle system prompt
        if system_prompt:
            if isinstance(system_prompt, str):
                openai_msgs.append({"role": "system", "content": system_prompt})
            elif isinstance(system_prompt, list):
                combined = "\n".join(b.get("text", "") for b in system_prompt if isinstance(b, dict) and b.get("type") == "text")
                if combined:
                    openai_msgs.append({"role": "system", "content": combined})

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")

            if isinstance(content, str):
                openai_msgs.append({"role": role, "content": content})
                continue

            if not isinstance(content, list):
                openai_msgs.append({"role": role, "content": str(content or "")})
                continue

            # Content is a list of blocks
            text_parts = []
            tool_calls = []
            tool_results = []

            for block in content:
                if not isinstance(block, dict):
                    continue
                b_type = block.get("type")
                if b_type == "text":
                    text_parts.append(block.get("text", ""))
                elif b_type == "tool_use":
                    tool_calls.append({
                        "id": block.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                        "type": "function",
                        "function": {
                            "name": block.get("name"),
                            "arguments": json.dumps(block.get("input", {})),
                        },
                    })
                elif b_type == "tool_result":
                    res_content = block.get("content", "")
                    if isinstance(res_content, list):
                        res_content = "\n".join(rb.get("text", "") for rb in res_content if isinstance(rb, dict))
                    tool_results.append({
                        "role": "tool",
                        "tool_call_id": block.get("tool_use_id"),
                        "content": str(res_content),
                    })

            if role == "assistant":
                msg_dict: Dict[str, Any] = {"role": "assistant"}
                if text_parts:
                    msg_dict["content"] = "\n".join(text_parts)
                if tool_calls:
                    msg_dict["tool_calls"] = tool_calls
                if not text_parts and not tool_calls:
                    msg_dict["content"] = ""
                openai_msgs.append(msg_dict)
            elif role == "user":
                if tool_results:
                    for tr in tool_results:
                        openai_msgs.append(tr)
                if text_parts:
                    openai_msgs.append({"role": "user", "content": "\n".join(text_parts)})
            else:
                openai_msgs.append({"role": role, "content": "\n".join(text_parts)})

        return openai_msgs

    @classmethod
    def anthropic_request_to_openai(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """Translates full Anthropic request payload to OpenAI payload."""
        messages = data.get("messages", [])
        system = data.get("system")
        openai_messages = cls.anthropic_messages_to_openai(messages, system)

        openai_req: Dict[str, Any] = {
            "model": data.get("model", "qwen3.8-27b"),
            "messages": openai_messages,
            "stream": data.get("stream", False),
        }

        if "max_tokens" in data:
            openai_req["max_tokens"] = data["max_tokens"]
        if "temperature" in data:
            openai_req["temperature"] = data["temperature"]
        if "top_p" in data:
            openai_req["top_p"] = data["top_p"]
        if "seed" in data:
            openai_req["seed"] = data["seed"]

        tools = cls.anthropic_tools_to_openai(data.get("tools"))
        if tools:
            openai_req["tools"] = tools

        return openai_req

    @classmethod
    def openai_response_to_anthropic(cls, openai_resp: Dict[str, Any], requested_model: str) -> Dict[str, Any]:
        """Translates non-streaming OpenAI response back to Anthropic Messages format."""
        choice = openai_resp.get("choices", [{}])[0]
        message = choice.get("message", {})
        finish_reason = choice.get("finish_reason", "stop")

        content_blocks: List[Dict[str, Any]] = []

        # Reasoning / Thinking content (e.g. DeepSeek-R1 / Qwen3.6 thinking tokens)
        reasoning = message.get("reasoning_content") or message.get("reasoning") or message.get("thinking")
        if reasoning:
            content_blocks.append({"type": "thinking", "thinking": reasoning})

        # Text content
        text = message.get("content")
        if text:
            content_blocks.append({"type": "text", "text": text})

        # Tool calls
        tool_calls = message.get("tool_calls", [])
        for tc in tool_calls:
            fn = tc.get("function", {})
            try:
                args = json.loads(fn.get("arguments", "{}"))
            except Exception:
                args = {"raw": fn.get("arguments", "")}

            content_blocks.append({
                "type": "tool_use",
                "id": tc.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                "name": fn.get("name"),
                "input": args,
            })

        # Stop reason mapping
        stop_reason = "end_turn"
        if finish_reason == "tool_calls" or tool_calls:
            stop_reason = "tool_use"
        elif finish_reason == "length":
            stop_reason = "max_tokens"

        usage = openai_resp.get("usage", {})
        return {
            "id": f"msg_{uuid.uuid4().hex[:24]}",
            "type": "message",
            "role": "assistant",
            "model": requested_model,
            "content": content_blocks,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
            },
        }


class ClaudeGatewayHandler(BaseHTTPRequestHandler):
    """HTTP Request Handler translating between Anthropic and OpenAI protocols."""

    upstream_url: str = "http://127.0.0.1:8080/v1"
    backend: str = "ollama"
    num_ctx: int = DEFAULT_NUM_CTX

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok", "upstream": self.upstream_url}).encode("utf-8"))
        elif parsed.path in ("/v1/models", "/models"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            model_items = [
                {"id": "claude-3-7-sonnet-20250219", "object": "model"},
                {"id": "claude-3-5-sonnet-20241022", "object": "model"},
                {"id": "claude-3-opus-20240229", "object": "model"},
                {"id": "claude-3-5-haiku-20241022", "object": "model"},
            ]
            # Dynamically fetch available models from Ollama
            try:
                with httpx.Client(timeout=0.5) as probe:
                    ollama_resp = probe.get("http://127.0.0.1:11434/api/tags")
                    if ollama_resp.status_code == 200:
                        for m in ollama_resp.json().get("models", []):
                            m_name = m.get("name")
                            if m_name:
                                model_items.append({"id": m_name, "object": "model"})
                                if m_name.endswith(":latest"):
                                    model_items.append({"id": m_name[:-7], "object": "model"})
            except Exception:
                pass

            self.wfile.write(json.dumps({"data": model_items}).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path not in ("/v1/messages", "/messages"):
            self.send_response(404)
            self.end_headers()
            return

        content_len = int(self.headers.get("Content-Length", 0))
        post_body = self.rfile.read(content_len)
        try:
            data = json.loads(post_body.decode("utf-8"))
        except Exception as e:
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": {"type": "invalid_request_error", "message": str(e)}}).encode("utf-8"))
            return

        stream = data.get("stream", False)
        model_name = data.get("model", "qwen3.8-27b")
        print(f"[{time.strftime('%H:%M:%S')}] 📥 Request: /v1/messages | Model: {model_name} | Stream: {stream}", flush=True)

        # Smart Hybrid Routing:
        # If the requested model is an official Claude model (Opus/Sonnet/Haiku/Fable), proxy directly to Anthropic Cloud.
        is_cloud_model = any(model_name.startswith(p) for p in ("claude", "sonnet", "opus", "haiku", "fable"))
        if is_cloud_model:
            cloud_headers = {}
            for h_key, h_val in self.headers.items():
                if h_key.lower() not in ("host", "content-length"):
                    cloud_headers[h_key] = h_val
            cloud_headers["content-type"] = "application/json"

            try:
                with httpx.Client(timeout=240.0) as cloud_client:
                    if not stream:
                        cloud_resp = cloud_client.post("https://api.anthropic.com/v1/messages", json=data, headers=cloud_headers)
                        self.send_response(cloud_resp.status_code)
                        self.send_header("Content-Type", "application/json")
                        self.end_headers()
                        self.wfile.write(cloud_resp.content)
                        return
                    else:
                        with cloud_client.stream("POST", "https://api.anthropic.com/v1/messages", json=data, headers=cloud_headers) as cloud_stream:
                            self.send_response(cloud_stream.status_code)
                            if cloud_stream.status_code == 200:
                                self.send_header("Content-Type", "text/event-stream")
                                self.send_header("Cache-Control", "no-cache")
                                self.send_header("Connection", "keep-alive")
                                self.end_headers()
                                for chunk in cloud_stream.iter_bytes():
                                    self.wfile.write(chunk)
                                    self.wfile.flush()
                            else:
                                self.send_header("Content-Type", "application/json")
                                self.end_headers()
                                self.wfile.write(cloud_stream.read())
                            return
            except Exception as cloud_err:
                logger.error(f"Cloud proxy error: {cloud_err}")
                self.send_response(502)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": {"type": "api_error", "message": f"Anthropic Cloud Proxy error: {cloud_err}"}}).encode("utf-8"))
                return

        # Ensure max_tokens is present (required by Anthropic API spec & Ollama)
        if "max_tokens" not in data or not data["max_tokens"]:
            data["max_tokens"] = 4096

        # Local routing. llama-server does not speak the Anthropic wire
        # format, so only the Ollama backend is forwarded untranslated;
        # the OpenAI path below handles the other one.
        if self.backend == "ollama":
          try:
            ollama_endpoint = "http://127.0.0.1:11434/v1/messages"
            ollama_headers = {
                "content-type": "application/json",
                "x-api-key": "ollama",
                "anthropic-version": self.headers.get("anthropic-version", "2023-06-01"),
            }
            if "anthropic-beta" in self.headers:
                ollama_headers["anthropic-beta"] = self.headers["anthropic-beta"]

            # Map model name to live installed Ollama tag if needed
            local_data = dict(data)
            try:
                with httpx.Client(timeout=0.3) as probe:
                    tag_resp = probe.get("http://127.0.0.1:11434/api/tags")
                    if tag_resp.status_code == 200:
                        installed = [m["name"] for m in tag_resp.json().get("models", [])]
                        req_m = local_data.get("model", "")
                        matched = next((im for im in installed if im == req_m or im.startswith(f"{req_m}:") or im.split(":")[0] == req_m), None)
                        if matched:
                            local_data["model"] = matched
            except Exception:
                pass

            # Sanitize messages to prevent Jinja 'System message must be at the beginning' exception
            raw_msgs = local_data.get("messages", [])
            sanitized_msgs = []
            has_top_level_system = bool(local_data.get("system"))
            for i, msg in enumerate(raw_msgs):
                if msg.get("role") == "system":
                    if i != 0 or has_top_level_system:
                        sanitized_msgs.append({
                            **msg,
                            "role": "user",
                            "content": f"[System Context]: {msg.get('content', '')}" if isinstance(msg.get("content"), str) else msg.get("content")
                        })
                    else:
                        sanitized_msgs.append(msg)
                else:
                    sanitized_msgs.append(msg)
            local_data["messages"] = sanitized_msgs

            # Ensure 64K context window support in Ollama
            local_data["options"] = build_ollama_options(
                local_data.get("options"), num_ctx=self.num_ctx
            )

            # In flight: the request is running but the backend has not
            # reported usage yet, so there are no token counts to publish.
            record_backend_telemetry(
                backend="ollama",
                model=local_data.get("model", model_name),
                input_tokens=None,
                output_tokens=None,
                state="STREAMING",
            )

            with httpx.Client(timeout=240.0) as client:
                if not stream:
                    resp = client.post(ollama_endpoint, json=local_data, headers=ollama_headers)
                    usage = self._usage_from_response(resp)
                    record_backend_telemetry(
                        backend="ollama",
                        model=local_data.get("model", model_name),
                        input_tokens=usage.get("input_tokens"),
                        output_tokens=usage.get("output_tokens"),
                        residency=self._probe_residency(local_data.get("model", model_name)),
                    )
                    self.send_response(resp.status_code)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(resp.content)
                    return
                else:
                    with client.stream("POST", ollama_endpoint, json=local_data, headers=ollama_headers) as upstream_stream:
                        self.send_response(upstream_stream.status_code)
                        if upstream_stream.status_code == 200:
                            self.send_header("Content-Type", "text/event-stream")
                            self.send_header("Cache-Control", "no-cache")
                            self.send_header("Connection", "keep-alive")
                            self.end_headers()
                            # The stream carries its own usage counts; read
                            # them instead of inferring a count from chunk size.
                            tracker = SseUsageTracker()
                            for chunk in upstream_stream.iter_bytes():
                                self.wfile.write(chunk)
                                self.wfile.flush()
                                tracker.feed(chunk)
                            record_backend_telemetry(
                                backend="ollama",
                                model=local_data.get("model", model_name),
                                input_tokens=tracker.input_tokens,
                                output_tokens=tracker.output_tokens,
                                residency=self._probe_residency(local_data.get("model", model_name)),
                            )
                        else:
                            self.send_header("Content-Type", "application/json")
                            self.end_headers()
                            err_body = upstream_stream.read()
                            print(f"[{time.strftime('%H:%M:%S')}] ❌ Ollama stream error ({upstream_stream.status_code}): {err_body.decode('utf-8')}", flush=True)
                            self.wfile.write(err_body)
                            record_backend_telemetry(
                                backend="ollama",
                                model=local_data.get("model", model_name),
                                input_tokens=None,
                                output_tokens=None,
                                state="ERROR",
                            )
                        self.close_connection = True
                        return
          except Exception as ollama_err:
              logger.error(f"Ollama native forward error: {ollama_err}")
              self.send_response(502)
              self.send_header("Content-Type", "application/json")
              self.end_headers()
              self.wfile.write(json.dumps({"error": {"type": "gateway_error", "message": str(ollama_err)}}).encode("utf-8"))
              return

        # Fallback for llama-server OpenAI endpoint
        openai_req = MessageFormatAdapter.anthropic_request_to_openai(data)
        upstream_endpoint = f"{self.upstream_url.rstrip('/')}/chat/completions"

        try:
            with httpx.Client(timeout=120.0) as client:
                if not stream:
                    resp = client.post(upstream_endpoint, json=openai_req)
                    if resp.status_code != 200:
                        self.send_response(resp.status_code)
                        self.send_header("Content-Type", "application/json")
                        self.end_headers()
                        self.wfile.write(resp.content)
                        return

                    openai_resp = resp.json()
                    anthropic_resp = MessageFormatAdapter.openai_response_to_anthropic(openai_resp, model_name)
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps(anthropic_resp).encode("utf-8"))
                    return

                # Streaming SSE Response
                with client.stream("POST", upstream_endpoint, json=openai_req) as upstream_stream:
                    if upstream_stream.status_code != 200:
                        self.send_response(upstream_stream.status_code)
                        self.send_header("Content-Type", "application/json")
                        self.end_headers()
                        err_body = upstream_stream.read()
                        self.wfile.write(err_body)
                        return

                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "keep-alive")
                    self.end_headers()

                    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
                    self._send_sse_event("message_start", {
                        "type": "message_start",
                        "message": {
                            "id": msg_id,
                            "type": "message",
                            "role": "assistant",
                            "model": model_name,
                            "content": [],
                            "stop_reason": None,
                            "stop_sequence": None,
                            "usage": {"input_tokens": 0, "output_tokens": 0},
                        },
                    })

                    current_block_index = 0
                    in_thinking_block = False
                    in_text_block = False
                    in_tool_block = False
                    current_tool_id = None
                    current_tool_name = None
                    total_output_tokens = 0

                    for line in upstream_stream.iter_lines():
                        if not line or not line.startswith("data:"):
                            continue
                        payload = line[5:].strip()
                        if payload == "[DONE]":
                            break
                        try:
                            chunk = json.loads(payload)
                        except Exception:
                            continue

                        # Extract server-reported usage if available
                        chunk_usage = chunk.get("usage")
                        if isinstance(chunk_usage, dict) and "completion_tokens" in chunk_usage:
                            total_output_tokens = max(total_output_tokens, int(chunk_usage["completion_tokens"]))

                        choice = chunk.get("choices", [{}])[0]
                        delta = choice.get("delta", {})

                        reasoning_delta = delta.get("reasoning_content") or delta.get("reasoning") or delta.get("thinking")
                        text_delta = delta.get("content")

                        # 1. Reasoning/Thinking block
                        if reasoning_delta:
                            total_output_tokens += max(1, len(reasoning_delta) // 4)
                            if in_text_block:
                                self._send_sse_event("content_block_stop", {"type": "content_block_stop", "index": current_block_index})
                                current_block_index += 1
                                in_text_block = False
                            if in_tool_block:
                                self._send_sse_event("content_block_stop", {"type": "content_block_stop", "index": current_block_index})
                                current_block_index += 1
                                in_tool_block = False
                            if not in_thinking_block:
                                self._send_sse_event("content_block_start", {"type": "content_block_start", "index": current_block_index, "content_block": {"type": "thinking", "thinking": ""}})
                                in_thinking_block = True
                            self._send_sse_event("content_block_delta", {"type": "content_block_delta", "index": current_block_index, "delta": {"type": "thinking_delta", "thinking": reasoning_delta}})

                        # 2. Visible Text block
                        if text_delta:
                            total_output_tokens += max(1, len(text_delta) // 4)
                            if in_thinking_block:
                                self._send_sse_event("content_block_stop", {"type": "content_block_stop", "index": current_block_index})
                                current_block_index += 1
                                in_thinking_block = False
                            if in_tool_block:
                                self._send_sse_event("content_block_stop", {"type": "content_block_stop", "index": current_block_index})
                                current_block_index += 1
                                in_tool_block = False
                            if not in_text_block:
                                self._send_sse_event("content_block_start", {"type": "content_block_start", "index": current_block_index, "content_block": {"type": "text", "text": ""}})
                                in_text_block = True
                            self._send_sse_event("content_block_delta", {"type": "content_block_delta", "index": current_block_index, "delta": {"type": "text_delta", "text": text_delta}})

                        # 3. Tool calls delta
                        tool_calls = delta.get("tool_calls", [])
                        for tc in tool_calls:
                            if in_thinking_block:
                                self._send_sse_event("content_block_stop", {"type": "content_block_stop", "index": current_block_index})
                                current_block_index += 1
                                in_thinking_block = False
                            if in_text_block:
                                self._send_sse_event("content_block_stop", {"type": "content_block_stop", "index": current_block_index})
                                current_block_index += 1
                                in_text_block = False
                            fn = tc.get("function", {})
                            if tc.get("id"):
                                current_tool_id = tc["id"]
                            if fn.get("name"):
                                current_tool_name = fn["name"]
                                self._send_sse_event("content_block_start", {"type": "content_block_start", "index": current_block_index, "content_block": {"type": "tool_use", "id": current_tool_id, "name": current_tool_name, "input": {}}})
                                in_tool_block = True
                            args_delta = fn.get("arguments")
                            if args_delta and in_tool_block:
                                total_output_tokens += max(1, len(args_delta) // 4)
                                self._send_sse_event("content_block_delta", {"type": "content_block_delta", "index": current_block_index, "delta": {"type": "input_json_delta", "partial_json": args_delta}})

                if in_thinking_block or in_text_block or in_tool_block:
                    self._send_sse_event("content_block_stop", {"type": "content_block_stop", "index": current_block_index})

                stop_reason = "tool_use" if in_tool_block else "end_turn"
                final_output_tokens = max(1, total_output_tokens)
                self._send_sse_event("message_delta", {"type": "message_delta", "delta": {"stop_reason": stop_reason, "stop_sequence": None}, "usage": {"output_tokens": final_output_tokens}})
                self._send_sse_event("message_stop", {"type": "message_stop"})
                try:
                    self.wfile.flush()
                except Exception:
                    pass
                self.close_connection = True

        except Exception as ex:
            logger.error(f"Gateway error: {ex}")
            try:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": {"type": "api_error", "message": str(ex)}}).encode("utf-8"))
            except Exception:
                pass

    @staticmethod
    def _usage_from_response(resp) -> Dict[str, Any]:
        """Token counts from a non-streaming Anthropic response body.

        Returns an empty mapping when the body is not parseable, so callers
        publish ``None`` rather than a fabricated count.
        """
        try:
            usage = resp.json().get("usage", {})
        except (ValueError, AttributeError):
            return {}
        return usage if isinstance(usage, dict) else {}

    @staticmethod
    def _probe_residency(model: str) -> Optional[Dict[str, Any]]:
        """Ask Ollama where the model is actually resident. Best effort."""
        try:
            with httpx.Client(timeout=0.5) as probe:
                resp = probe.get("http://127.0.0.1:11434/api/ps")
                if resp.status_code == 200:
                    return parse_ollama_residency(resp.json(), model=model)
        except (httpx.HTTPError, ValueError):
            pass
        return None

    def _send_sse_event(self, event_type: str, data: Dict[str, Any]):
        msg = f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
        self.wfile.write(msg.encode("utf-8"))
        self.wfile.flush()

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
    backend: str = "auto",
    num_ctx: int = DEFAULT_NUM_CTX,
) -> ThreadingHTTPServer:
    """Creates a QuietThreadingHTTPServer configured with the ClaudeGatewayHandler.

    The backend is resolved once here rather than per request: probing the
    upstream on every message would add a round trip to each call, and the
    answer cannot change without restarting the server anyway.
    """
    ClaudeGatewayHandler.upstream_url = upstream_url
    ClaudeGatewayHandler.backend = resolve_backend(backend, upstream_url)
    ClaudeGatewayHandler.num_ctx = num_ctx
    server = QuietThreadingHTTPServer(("0.0.0.0", port), ClaudeGatewayHandler)
    return server


def main():
    parser = argparse.ArgumentParser(description="ARGUS Claude Code Anthropic Messages Gateway")
    parser.add_argument("--port", type=int, default=8000, help="Listening port for gateway")
    parser.add_argument("--upstream", type=str, default="http://127.0.0.1:8080/v1", help="Upstream llama-server endpoint")
    parser.add_argument(
        "--backend",
        choices=BACKENDS,
        default="auto",
        help=(
            "Upstream protocol. 'openai' is llama-server (supports --cpu-moe "
            "and per-run context sizing); 'ollama' speaks Anthropic natively. "
            "'auto' probes the upstream."
        ),
    )
    parser.add_argument(
        "--num-ctx",
        type=int,
        default=DEFAULT_NUM_CTX,
        help=f"Context window to request per message (default: {DEFAULT_NUM_CTX}).",
    )
    args = parser.parse_args()

    server = create_gateway_server(
        port=args.port,
        upstream_url=args.upstream,
        backend=args.backend,
        num_ctx=args.num_ctx,
    )
    print(
        f"Starting ARGUS Claude Code Gateway on port {args.port} "
        f"(backend: {ClaudeGatewayHandler.backend}, upstream: {args.upstream}, "
        f"num_ctx: {args.num_ctx})..."
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
