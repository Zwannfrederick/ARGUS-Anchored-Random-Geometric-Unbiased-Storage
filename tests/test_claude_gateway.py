"""Tests for Claude Code Anthropic Messages Gateway (Stage S10).

Validates:
1. Anthropic tools -> OpenAI function tools schema translation
2. Multi-turn system, text, tool_use, and tool_result message translation
3. Non-streaming OpenAI response -> Anthropic Messages response conversion
4. Server endpoint route handling (/health, /v1/models)
"""

import json
import threading
import time
import httpx

from argus_cache.adapters import claude_gateway
from argus_cache.adapters.claude_gateway import (
    MessageFormatAdapter,
    create_gateway_server,
)


def test_anthropic_tools_to_openai():
    """Translates Anthropic JSON schema tool definition to OpenAI function definition."""
    anthropic_tools = [
        {
            "name": "edit_file",
            "description": "Edits a file in place",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        }
    ]

    openai_tools = MessageFormatAdapter.anthropic_tools_to_openai(anthropic_tools)
    assert len(openai_tools) == 1
    assert openai_tools[0]["type"] == "function"
    assert openai_tools[0]["function"]["name"] == "edit_file"
    assert openai_tools[0]["function"]["parameters"]["required"] == ["path", "content"]


def test_anthropic_messages_to_openai_with_tool_roundtrip():
    """Validates multi-turn user/assistant/tool_use/tool_result translation."""
    messages = [
        {"role": "user", "content": "Can you check the file?"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me read the file."},
                {
                    "type": "tool_use",
                    "id": "call_12345",
                    "name": "read_file",
                    "input": {"path": "main.py"},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "call_12345",
                    "content": "print('hello world')",
                }
            ],
        },
    ]

    system = "You are a coding assistant."
    openai_msgs = MessageFormatAdapter.anthropic_messages_to_openai(messages, system)

    assert len(openai_msgs) == 4
    # System
    assert openai_msgs[0]["role"] == "system"
    assert openai_msgs[0]["content"] == "You are a coding assistant."
    # User text
    assert openai_msgs[1]["role"] == "user"
    assert openai_msgs[1]["content"] == "Can you check the file?"
    # Assistant tool call
    assert openai_msgs[2]["role"] == "assistant"
    assert openai_msgs[2]["content"] == "Let me read the file."
    assert len(openai_msgs[2]["tool_calls"]) == 1
    assert openai_msgs[2]["tool_calls"][0]["id"] == "call_12345"
    assert openai_msgs[2]["tool_calls"][0]["function"]["name"] == "read_file"
    assert json.loads(openai_msgs[2]["tool_calls"][0]["function"]["arguments"]) == {"path": "main.py"}
    # Tool result
    assert openai_msgs[3]["role"] == "tool"
    assert openai_msgs[3]["tool_call_id"] == "call_12345"
    assert openai_msgs[3]["content"] == "print('hello world')"


def test_openai_response_to_anthropic():
    """Translates OpenAI tool-call response into Anthropic format."""
    openai_resp = {
        "id": "chatcmpl-123",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "I will execute the test.",
                    "tool_calls": [
                        {
                            "id": "call_999",
                            "type": "function",
                            "function": {
                                "name": "run_test",
                                "arguments": "{\"test_name\": \"test_parity\"}",
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {
            "prompt_tokens": 150,
            "completion_tokens": 30,
        },
    }

    anthropic_resp = MessageFormatAdapter.openai_response_to_anthropic(openai_resp, "qwen3.8-27b")

    assert anthropic_resp["type"] == "message"
    assert anthropic_resp["role"] == "assistant"
    assert anthropic_resp["stop_reason"] == "tool_use"
    assert len(anthropic_resp["content"]) == 2
    assert anthropic_resp["content"][0]["type"] == "text"
    assert anthropic_resp["content"][0]["text"] == "I will execute the test."
    assert anthropic_resp["content"][1]["type"] == "tool_use"
    assert anthropic_resp["content"][1]["name"] == "run_test"
    assert anthropic_resp["content"][1]["input"] == {"test_name": "test_parity"}
    assert anthropic_resp["usage"]["input_tokens"] == 150
    assert anthropic_resp["usage"]["output_tokens"] == 30


def test_gateway_server_health_and_models_endpoints():
    """Starts live test server on ephemeral port and tests endpoints using httpx."""
    server = create_gateway_server(port=0, upstream_url="http://127.0.0.1:8080/v1")
    port = server.server_address[1]

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.05)

    try:
        with httpx.Client(base_url=f"http://127.0.0.1:{port}") as client:
            # Health check
            resp = client.get("/health")
            assert resp.status_code == 200
            assert resp.json()["status"] == "ok"

            # Models list. Only the static Anthropic ids are asserted: the
            # handler also appends whatever the local llama-server serves, and
            # pinning a name there would tie this test to one machine.
            resp_models = client.get("/v1/models")
            assert resp_models.status_code == 200
            ids = {m["id"] for m in resp_models.json()["data"]}
            assert "claude-3-7-sonnet-20250219" in ids
            assert "claude-3-5-haiku-20241022" in ids
    finally:
        server.shutdown()
        server.server_close()


# ── telemetry honesty ───────────────────────────────────────────────────────


def test_telemetry_reports_measured_token_counts(tmp_path, monkeypatch):
    """Token counts come from the backend's own usage block, never estimated.

    The replaced implementation guessed output tokens from SSE byte length
    (``len(chunk) // 20``) and, on the non-streaming path, wrote a hardcoded
    50. Both produced numbers that looked measured and were not.
    """
    monkeypatch.setattr(claude_gateway, "STATS_FILE", tmp_path / "cache_stats.json")

    stats = claude_gateway.record_backend_telemetry(
        backend="llama-server",
        model="qwen3.6-35b-a3b",
        input_tokens=5480,
        output_tokens=16,
    )

    assert stats["input_tokens"] == 5480
    assert stats["output_tokens"] == 16
    assert json.loads((tmp_path / "cache_stats.json").read_text()) == stats


def test_telemetry_never_fabricates_argus_tiers(tmp_path, monkeypatch):
    """No tier occupancy or savings figure may be attributed to an external backend.

    The gateway sees tokens, not KV pages, so any t0/t1/t2 page count or
    savings percentage would be invented. This is the regression that matters.
    """
    monkeypatch.setattr(claude_gateway, "STATS_FILE", tmp_path / "cache_stats.json")

    stats = claude_gateway.record_backend_telemetry(
        backend="llama-server",
        model="qwen3.6-35b-a3b",
        input_tokens=5480,
        output_tokens=16,
    )

    assert stats["argus_managed"] is False
    forbidden = ("t0_pages", "t1_pages", "t2_pages", "savings_pct", "total_pages")
    assert [key for key in forbidden if key in stats] == []


def test_telemetry_leaves_unmeasured_counts_as_none(tmp_path, monkeypatch):
    """An unmeasured count is reported as None, not coerced to zero.

    Zero is a measurement; None is an absence. Collapsing the two is exactly
    how the fabricated numbers became indistinguishable from real ones.
    """
    monkeypatch.setattr(claude_gateway, "STATS_FILE", tmp_path / "cache_stats.json")

    stats = claude_gateway.record_backend_telemetry(
        backend="llama-server",
        model="qwen3.6-35b-a3b",
        input_tokens=None,
        output_tokens=None,
        state="STREAMING",
    )

    assert stats["input_tokens"] is None
    assert stats["output_tokens"] is None
    assert stats["state"] == "STREAMING"


# ── streaming translation ───────────────────────────────────────────────────


def _events(translator, chunks):
    out = translator.start()
    for chunk in chunks:
        out += translator.feed(chunk)
    return out + translator.finish()


def _delta(**delta):
    return {"choices": [{"delta": delta}]}


def test_stream_translator_opens_one_block_per_kind_in_order():
    """thinking -> text -> tool_use each get their own block; switching closes the last."""
    from argus_cache.adapters.anthropic_translate import AnthropicStreamTranslator

    events = _events(AnthropicStreamTranslator("m"), [
        _delta(reasoning_content="hmm"),
        _delta(content="Hi"),
        _delta(content=" there"),
        _delta(tool_calls=[{"id": "t1", "function": {"name": "read", "arguments": '{"p":'}}]),
        _delta(tool_calls=[{"function": {"arguments": '"x"}'}}]),
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}], "usage": {"prompt_tokens": 9, "completion_tokens": 7}},
    ])
    kinds = [(e, d.get("content_block", {}).get("type") or d.get("delta", {}).get("type")) for e, d in events]
    assert kinds == [
        ("message_start", None),
        ("content_block_start", "thinking"), ("content_block_delta", "thinking_delta"), ("content_block_stop", None),
        ("content_block_start", "text"), ("content_block_delta", "text_delta"), ("content_block_delta", "text_delta"),
        ("content_block_stop", None),
        ("content_block_start", "tool_use"), ("content_block_delta", "input_json_delta"),
        ("content_block_delta", "input_json_delta"), ("content_block_stop", None),
        ("message_delta", None), ("message_stop", None),
    ]
    indices = [d["index"] for e, d in events if e.startswith("content_block")]
    assert indices == [0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
    final = events[-2][1]
    assert final["delta"]["stop_reason"] == "tool_use"
    assert final["usage"]["output_tokens"] == 7


def test_stream_translator_reports_length_stop_as_max_tokens():
    """A truncated answer must say so; reporting end_turn hides the cut-off."""
    from argus_cache.adapters.anthropic_translate import AnthropicStreamTranslator

    events = _events(AnthropicStreamTranslator("m"), [
        _delta(content="partial"),
        {"choices": [{"delta": {}, "finish_reason": "length"}]},
    ])
    assert events[-2][1]["delta"]["stop_reason"] == "max_tokens"


def test_parse_sse_line_ignores_non_data_and_done():
    from argus_cache.adapters.anthropic_translate import AnthropicStreamTranslator

    parse = AnthropicStreamTranslator.parse_sse_line
    assert parse(": keep-alive") is None
    assert parse("data: [DONE]") is None
    assert parse("data: {not json") is None
    assert parse('data: {"a": 1}') == {"a": 1}


def test_cloud_routing_needs_real_credentials():
    """A Claude model name with the local placeholder key stays on the local model."""
    assert claude_gateway.is_cloud_request("claude-sonnet-4", "sk-real", "")
    assert not claude_gateway.is_cloud_request("claude-sonnet-4", "argus-local", "")
    assert not claude_gateway.is_cloud_request("claude-sonnet-4", "", "Bearer argus-local")
    assert not claude_gateway.is_cloud_request("qwen3.6-35b-a3b", "sk-real", "")


def test_gateway_streams_anthropic_events_from_a_live_openai_upstream():
    """End to end: Anthropic request in, OpenAI SSE upstream, Anthropic SSE out."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class FakeLlamaServer(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for chunk in (_delta(content="Mer"), _delta(content="haba"),
                          {"choices": [{"delta": {}, "finish_reason": "stop"}],
                           "usage": {"prompt_tokens": 12, "completion_tokens": 2}}):
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")

        def log_message(self, *args):
            pass

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), FakeLlamaServer)
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    gateway = create_gateway_server(port=0, upstream_url=f"http://127.0.0.1:{upstream.server_address[1]}/v1")
    threading.Thread(target=gateway.serve_forever, daemon=True).start()
    try:
        resp = httpx.post(
            f"http://127.0.0.1:{gateway.server_address[1]}/v1/messages",
            headers={"x-api-key": "argus-local"},
            json={"model": "qwen", "stream": True, "max_tokens": 50,
                  "messages": [{"role": "user", "content": "naber"}]},
            timeout=10,
        )
        assert resp.status_code == 200
        events = [line[7:] for line in resp.text.splitlines() if line.startswith("event: ")]
        assert events[0] == "message_start" and events[-1] == "message_stop"
        text = "".join(
            json.loads(line[6:])["delta"]["text"]
            for line in resp.text.splitlines()
            if line.startswith("data: ") and '"text_delta"' in line
        )
        assert text == "Merhaba"
    finally:
        for server in (gateway, upstream):
            server.shutdown()
            server.server_close()


def test_thinking_toggle_translated_to_chat_template_kwargs():
    """Fast turns keep Qwen3.6 thinking off; explicit thinking enables it."""
    fast = MessageFormatAdapter.anthropic_request_to_openai({"messages": [{"role": "user", "content": "hi"}]})
    assert fast["chat_template_kwargs"] == {"enable_thinking": False}

    deep = MessageFormatAdapter.anthropic_request_to_openai({
        "messages": [{"role": "user", "content": "solve"}],
        "thinking": {"type": "enabled", "budget_tokens": 2048},
    })
    assert deep["chat_template_kwargs"] == {"enable_thinking": True}


def test_system_and_developer_messages_never_appear_after_index_zero():
    """Qwen3.6 Jinja raises 'System message must be at the beginning' if system/developer appears at index > 0."""
    messages = [
        {"role": "system", "content": "Leading system block"},
        {"role": "developer", "content": [{"type": "text", "text": "Leading developer block"}]},
        {"role": "user", "content": "2+2 kaçtır?"},
        {"role": "system", "content": "<system-reminder>Stay concise.</system-reminder>"},
        {"role": "developer", "content": "Tool policy reminder"},
    ]
    openai_msgs = MessageFormatAdapter.anthropic_messages_to_openai(
        messages, system_prompt=[{"type": "text", "text": "Top-level system"}]
    )

    assert openai_msgs[0]["role"] == "system"
    assert "Top-level system" in openai_msgs[0]["content"]
    assert "Leading system block" in openai_msgs[0]["content"]
    assert "Leading developer block" in openai_msgs[0]["content"]
    assert all(m["role"] != "system" for m in openai_msgs[1:])
    assert all(m["role"] in ("system", "user", "assistant", "tool") for m in openai_msgs)
    assert any("Stay concise." in m["content"] and m["role"] == "user" for m in openai_msgs[1:])


def test_count_tokens_and_event_logging_endpoints():
    """Claude Code 2.x preflight endpoints return 200 instead of 404."""
    server = create_gateway_server(port=0, upstream_url="http://127.0.0.1:8080/v1")
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.05)

    try:
        with httpx.Client(base_url=f"http://127.0.0.1:{port}") as client:
            r_cnt = client.post("/v1/messages/count_tokens", json={"messages": [{"role": "user", "content": "hello world"}]})
            assert r_cnt.status_code == 200
            assert r_cnt.json()["input_tokens"] >= 1

            r_evt = client.post("/api/event_logging/batch", json={"events": []})
            assert r_evt.status_code == 200

            r_evt_v2 = client.post("/api/event_logging/v2/batch", json={"events": []})
            assert r_evt_v2.status_code == 200
    finally:
        server.shutdown()
        server.server_close()


def test_models_endpoint_includes_live_llama_server_model_stem():
    """Gateway /v1/models surfaces the active llama-server GGUF stem for Claude /model discovery."""
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class _MockUpstream(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            payload = {"data": [{"id": "/models/Qwen3.6-35B-A3B-uncensored-heretic.i1-Q4_K_S.gguf", "object": "model"}]}
            self.wfile.write(json.dumps(payload).encode("utf-8"))

        def log_message(self, format, *args):
            pass

    upstream = HTTPServer(("127.0.0.1", 0), _MockUpstream)
    up_port = upstream.server_address[1]
    threading.Thread(target=upstream.serve_forever, daemon=True).start()

    server = create_gateway_server(port=0, upstream_url=f"http://127.0.0.1:{up_port}/v1")
    gw_port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    time.sleep(0.05)

    try:
        with httpx.Client(base_url=f"http://127.0.0.1:{gw_port}") as client:
            resp = client.get("/v1/models")
            assert resp.status_code == 200
            ids = {m["id"] for m in resp.json()["data"]}
            assert "Qwen3.6-35B-A3B-uncensored-heretic.i1-Q4_K_S" in ids
    finally:
        server.shutdown()
        server.server_close()
        upstream.shutdown()
        upstream.server_close()


def test_local_openai_translation_caps_excessive_max_tokens():
    """Switching /model from Opus (32k max_tokens) to local Qwen clamps output budget safely."""
    req_128k = MessageFormatAdapter.anthropic_request_to_openai(
        {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 32000},
        num_ctx=131072,
    )
    assert req_128k["max_tokens"] == 16384

    req_32k = MessageFormatAdapter.anthropic_request_to_openai(
        {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 32000},
        num_ctx=32768,
    )
    assert req_32k["max_tokens"] == 4096
