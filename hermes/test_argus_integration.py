"""ARGUS paged-KV runtime wiring for the Hermes model server and gateway."""

from __future__ import annotations

import importlib
import json
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

import httpx
from starlette.applications import Starlette
from starlette.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _reload_config(**env):
    with mock.patch.dict(os.environ, env, clear=False):
        for key in ("HERMES_ARGUS_KV_DIR", "HERMES_ARGUS_KV_RESIDENT_BYTES", "HERMES_LLAMA_SERVER_BIN"):
            if key not in env:
                os.environ.pop(key, None)
        import hermes.config as config
        return importlib.reload(config)


class ServerEnvironmentTests(unittest.TestCase):
    def test_stock_runtime_has_no_argus_variables(self):
        config = _reload_config()
        env = config.get_server_env()
        self.assertFalse(config.argus_enabled())
        self.assertNotIn("ARGUS_KV_DIR", env)
        self.assertNotIn("-nkvo", config.argus_server_args())

    def test_argus_runtime_sets_budgets_and_no_ollama_backend_override(self):
        config = _reload_config(
            HERMES_ARGUS_KV_DIR="/data/kv",
            HERMES_ARGUS_KV_RESIDENT_BYTES="1073741824",
            HERMES_LLAMA_SERVER_BIN="/opt/argus/llama-server",
        )
        env = config.get_server_env()
        self.assertTrue(config.argus_enabled())
        self.assertEqual(env["ARGUS_KV_DIR"], "/data/kv")
        self.assertEqual(env["ARGUS_KV_RESIDENT_BYTES"], "1073741824")
        self.assertEqual(env["ARGUS_KV_STATS_PATH"], str(config.ARGUS_KV_STATS_PATH))
        self.assertTrue(int(env["ARGUS_KV_MAX_BYTES"]) > 0)
        # The Ollama CUDA backend must not replace the ARGUS build's own GGML libraries.
        self.assertNotIn("GGML_BACKEND_PATH", env)
        self.assertEqual(str(config.LLAMA_SERVER_BIN), "/opt/argus/llama-server")
        self.assertEqual(config.argus_server_args(), ["-nkvo", "-fa", "on"])

    def test_invalid_resident_budget_is_rejected(self):
        with self.assertRaises(ValueError):
            _reload_config(HERMES_ARGUS_KV_DIR="/data/kv", HERMES_ARGUS_KV_RESIDENT_BYTES="1GB")

    def tearDown(self):
        _reload_config()


class ArgusStatusTests(unittest.TestCase):
    def test_status_reads_published_stats(self):
        from hermes import argus_chat

        with mock.patch.object(argus_chat, "ARGUS_KV_STATS_PATH") as path:
            path.read_text.return_value = json.dumps({"live_bytes": 10, "resident_bytes": 4})
            status = argus_chat.read_argus_stats()
        self.assertEqual(status["resident_bytes"], 4)

    def test_status_is_none_when_runtime_has_not_published(self):
        from hermes import argus_chat

        with mock.patch.object(argus_chat, "ARGUS_KV_STATS_PATH") as path:
            path.read_text.side_effect = FileNotFoundError
            self.assertIsNone(argus_chat.read_argus_stats())

    def test_chat_payload_validation(self):
        from hermes.argus_chat import validate_chat_payload

        body = validate_chat_payload({
            "messages": [{"role": "user", "content": "Merhaba"}],
            "max_tokens": 99999,
            "temperature": 9,
            "tools": [{"type": "function"}],
        })
        self.assertEqual(body["messages"], [{"role": "user", "content": "Merhaba"}])
        self.assertLessEqual(body["max_tokens"], 4096)
        self.assertLessEqual(body["temperature"], 2.0)
        self.assertNotIn("tools", body)
        self.assertTrue(body["stream"])
        for bad in (
            {},
            {"messages": []},
            {"messages": [{"role": "tool", "content": "x"}]},
            {"messages": [{"role": "user", "content": 5}]},
            {"messages": [{"role": "user", "content": "x" * 200_001}]},
        ):
            with self.assertRaises(ValueError):
                validate_chat_payload(bad)


class ChatRouteTests(unittest.TestCase):
    def _client(self, handler):
        from hermes.argus_chat import build_routes

        upstream = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        auth = lambda request: request.headers.get("authorization") == "Bearer good"
        return TestClient(Starlette(routes=build_routes(auth, upstream)))

    def test_routes_require_auth_and_validate(self):
        client = self._client(lambda request: httpx.Response(500))
        self.assertEqual(client.get("/api/argus/status").status_code, 401)
        self.assertEqual(client.post("/api/chat/completions", json={}).status_code, 401)
        bad = client.post("/api/chat/completions", json={"messages": []}, headers={"Authorization": "Bearer good"})
        self.assertEqual(bad.status_code, 400)
        self.assertEqual(client.get("/chat").status_code, 200)

    def test_chat_relays_stream_without_tools(self):
        seen = {}

        def handler(request):
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, content=b'data: {"choices":[{"delta":{"content":"Selam"}}]}\n\ndata: [DONE]\n\n')

        client = self._client(handler)
        response = client.post(
            "/api/chat/completions",
            json={"messages": [{"role": "user", "content": "Merhaba"}], "tools": [{"type": "function"}]},
            headers={"Authorization": "Bearer good"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("Selam", response.text)
        self.assertNotIn("tools", seen["body"])
        status = client.get("/api/argus/status", headers={"Authorization": "Bearer good"}).json()
        self.assertFalse(status["generating"])

    def test_upstream_failure_is_reported_as_stream_error(self):
        client = self._client(lambda request: httpx.Response(503, content=b"loading"))
        response = client.post("/api/chat/completions", json={"messages": [{"role": "user", "content": "x"}]},
                               headers={"Authorization": "Bearer good"})
        self.assertIn("Model server rejected the request", response.text)


if __name__ == "__main__":
    unittest.main()
