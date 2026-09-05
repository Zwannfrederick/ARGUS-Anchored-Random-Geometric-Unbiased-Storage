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
import pytest

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
            # handler also appends whatever a local Ollama reports, and pinning
            # a name there made this test depend on one machine's model
            # library. It failed the moment that library changed
            # (qwen3.8-27b -> qwen3.6-35b-a3b) while the gateway was correct.
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
        backend="ollama",
        model="qwen3.6-35b-a3b:latest",
        input_tokens=5480,
        output_tokens=16,
    )

    assert stats["input_tokens"] == 5480
    assert stats["output_tokens"] == 16
    assert json.loads((tmp_path / "cache_stats.json").read_text()) == stats


def test_telemetry_never_fabricates_argus_tiers(tmp_path, monkeypatch):
    """No tier occupancy or savings figure may be attributed to an external backend.

    ARGUS cannot see llama.cpp's KV cache -- ``OllamaAdapter.manages_kv_cache``
    is False -- so any t0/t1/t2 page count or savings percentage would be
    invented. This is the regression that matters.
    """
    monkeypatch.setattr(claude_gateway, "STATS_FILE", tmp_path / "cache_stats.json")

    stats = claude_gateway.record_backend_telemetry(
        backend="ollama",
        model="qwen3.6-35b-a3b:latest",
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
        backend="ollama",
        model="qwen3.6-35b-a3b:latest",
        input_tokens=None,
        output_tokens=None,
        state="STREAMING",
    )

    assert stats["input_tokens"] is None
    assert stats["output_tokens"] is None
    assert stats["state"] == "STREAMING"


def test_usage_is_extracted_from_the_anthropic_sse_stream():
    """Streaming token counts are read from the protocol, not guessed from bytes."""
    stream = (
        b'event: message_start\n'
        b'data: {"type":"message_start","message":{"usage":{"input_tokens":5480,"output_tokens":0}}}\n\n'
        b'event: content_block_delta\n'
        b'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"hi"}}\n\n'
        b'event: message_delta\n'
        b'data: {"type":"message_delta","usage":{"output_tokens":16}}\n\n'
    )

    tracker = claude_gateway.SseUsageTracker()
    for line in stream.splitlines(keepends=True):
        tracker.feed(line)

    assert tracker.input_tokens == 5480
    assert tracker.output_tokens == 16


def test_usage_tracker_reports_none_before_any_usage_arrives():
    """A stream that failed before reporting usage yields no counts at all."""
    tracker = claude_gateway.SseUsageTracker()

    assert tracker.input_tokens is None
    assert tracker.output_tokens is None


def test_usage_tracker_survives_a_split_sse_chunk():
    """SSE events arrive on arbitrary byte boundaries; a split must not lose usage."""
    tracker = claude_gateway.SseUsageTracker()
    payload = b'data: {"type":"message_delta","usage":{"output_tokens":16}}\n\n'

    tracker.feed(payload[:30])
    tracker.feed(payload[30:])

    assert tracker.output_tokens == 16


def test_residency_split_is_read_from_the_backend():
    """The GPU/CPU split comes from Ollama's reported size_vram, not a guess.

    On a 4 GB card this is the one number that actually explains throughput:
    how much of the model landed on the GPU.
    """
    residency = claude_gateway.parse_ollama_residency(
        {
            "models": [
                {
                    "name": "qwen3.6-35b-a3b:latest",
                    "size": 24_000_000_000,
                    "size_vram": 2_800_000_000,
                    "context_length": 32768,
                }
            ]
        },
        model="qwen3.6-35b-a3b:latest",
    )

    assert residency["total_bytes"] == 24_000_000_000
    assert residency["vram_bytes"] == 2_800_000_000
    assert residency["cpu_bytes"] == 21_200_000_000
    assert residency["gpu_fraction"] == pytest.approx(0.1167, abs=1e-4)
    assert residency["context_length"] == 32768


def test_residency_is_none_when_the_model_is_not_loaded():
    """A model that is not resident yields no residency block, not zeros."""
    assert claude_gateway.parse_ollama_residency({"models": []}, model="absent") is None


# ── backend routing ─────────────────────────────────────────────────────────


def test_explicit_backend_choice_is_not_probed():
    """An operator who names the backend is obeyed without a network probe."""
    assert claude_gateway.resolve_backend("openai", "http://127.0.0.1:8080/v1") == "openai"
    assert claude_gateway.resolve_backend("ollama", "http://127.0.0.1:8080/v1") == "ollama"


def test_auto_backend_detects_llama_server(monkeypatch):
    """A reachable OpenAI /models endpoint means llama-server is upstream.

    This is what unlocks the tuned configuration: `--cpu-moe` and per-request
    context are llama-server flags that Ollama does not expose, so the gateway
    has to be able to route there at all.
    """
    monkeypatch.setattr(claude_gateway, "_probe_openai_upstream", lambda url: True)

    assert claude_gateway.resolve_backend("auto", "http://127.0.0.1:8080/v1") == "openai"


def test_auto_backend_falls_back_to_ollama(monkeypatch):
    """With no OpenAI upstream answering, Ollama's native endpoint is used."""
    monkeypatch.setattr(claude_gateway, "_probe_openai_upstream", lambda url: False)

    assert claude_gateway.resolve_backend("auto", "http://127.0.0.1:8080/v1") == "ollama"


def test_unknown_backend_rejected():
    with pytest.raises(ValueError, match="backend"):
        claude_gateway.resolve_backend("nonsense", "http://127.0.0.1:8080/v1")


def test_request_context_length_is_configurable_not_hardcoded():
    """num_ctx must follow configuration rather than a literal in the request path.

    It was pinned to 65536, which silently capped the window at 64k even though
    160k was measured to run at essentially the same throughput.
    """
    options = claude_gateway.build_ollama_options({}, num_ctx=131072)

    assert options["num_ctx"] == 131072


def test_request_context_length_preserves_caller_options():
    """Existing request options survive; only num_ctx is supplied."""
    options = claude_gateway.build_ollama_options({"temperature": 0.2}, num_ctx=32768)

    assert options["temperature"] == 0.2
    assert options["num_ctx"] == 32768
