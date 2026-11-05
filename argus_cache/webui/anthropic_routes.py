"""Anthropic Messages API on the Studio port, so Claude Code can use :8090 directly.

Same translation as the stdlib gateway (``argus_cache.adapters.claude_gateway``),
served async. Everything is answered by the local model: Studio is the local
stack, so there is no cloud passthrough here.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Dict

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from argus_cache.adapters.anthropic_translate import (
    AnthropicStreamTranslator,
    MessageFormatAdapter,
    discover_gateway_models,
    format_sse,
)
from argus_cache.adapters.claude_gateway import (
    DEFAULT_NUM_CTX,
    estimate_input_tokens,
    record_backend_telemetry,
)


def _error(status: int, kind: str, message: str) -> JSONResponse:
    return JSONResponse({"type": "error", "error": {"type": kind, "message": message}}, status_code=status)


def register_anthropic_routes(app: FastAPI, upstream_url: str, num_ctx: int = DEFAULT_NUM_CTX) -> None:
    endpoint = f"{upstream_url.rstrip('/')}/chat/completions"

    @app.get("/v1/models")
    def anthropic_models() -> Dict[str, Any]:
        return {"data": discover_gateway_models(upstream_url)}

    @app.post("/v1/messages/count_tokens")
    async def anthropic_count_tokens(request: Request) -> Dict[str, int]:
        return {"input_tokens": estimate_input_tokens(await request.body())}

    @app.post("/api/event_logging/{rest:path}")
    async def anthropic_event_logging(rest: str) -> Dict[str, Any]:
        return {}

    @app.post("/v1/messages")
    async def anthropic_messages(request: Request):
        try:
            data = json.loads(await request.body())
        except ValueError as exc:
            return _error(400, "invalid_request_error", str(exc))
        if not isinstance(data, dict) or not isinstance(data.get("messages"), list):
            return _error(400, "invalid_request_error", "body must be an object with a messages list")

        model = str(data.get("model") or "local")
        data = {**data, "max_tokens": data.get("max_tokens") or 4096}
        openai_req = MessageFormatAdapter.anthropic_request_to_openai(data, num_ctx=num_ctx)
        record_backend_telemetry(backend="llama-server", model=model, input_tokens=None, output_tokens=None, state="STREAMING")

        if not data.get("stream"):
            try:
                async with httpx.AsyncClient(timeout=900.0) as client:
                    resp = await client.post(endpoint, json=openai_req)
            except httpx.HTTPError as exc:
                return _error(502, "api_error", f"llama-server unreachable: {exc}")
            if resp.status_code != 200:
                return _error(resp.status_code, "api_error", resp.text[:500])
            out = MessageFormatAdapter.openai_response_to_anthropic(resp.json(), model)
            record_backend_telemetry(
                backend="llama-server", model=model,
                input_tokens=out["usage"].get("input_tokens"), output_tokens=out["usage"].get("output_tokens"),
            )
            return out

        openai_req["stream_options"] = {"include_usage": True}
        return StreamingResponse(
            _stream(endpoint, openai_req, model),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )


async def _stream(endpoint: str, openai_req: Dict[str, Any], model: str) -> AsyncIterator[bytes]:
    translator = AnthropicStreamTranslator(model)
    try:
        async with httpx.AsyncClient(timeout=900.0) as client:
            async with client.stream("POST", endpoint, json=openai_req) as upstream:
                if upstream.status_code != 200:
                    detail = (await upstream.aread()).decode("utf-8", "replace")[:500]
                    yield format_sse("error", {"type": "error", "error": {"type": "api_error", "message": detail}})
                    return
                for event in translator.start():
                    yield format_sse(*event)
                async for line in upstream.aiter_lines():
                    chunk = AnthropicStreamTranslator.parse_sse_line(line)
                    if chunk:
                        for event in translator.feed(chunk):
                            yield format_sse(*event)
    except httpx.HTTPError as exc:
        yield format_sse("error", {"type": "error", "error": {"type": "api_error", "message": str(exc)}})
        record_backend_telemetry(backend="llama-server", model=model, input_tokens=None, output_tokens=None, state="ERROR")
        return
    for event in translator.finish():
        yield format_sse(*event)
    record_backend_telemetry(
        backend="llama-server", model=model,
        input_tokens=translator.input_tokens, output_tokens=translator.output_tokens,
    )
