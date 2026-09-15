"""Plain chat and ARGUS KV status routes for the Neo gateway.

Plain chat has no tools and no desktop control: it only forwards validated
messages to the local llama-server and streams the answer back.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import httpx
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response, StreamingResponse
from starlette.routing import Route

try:
    from hermes.config import (ARGUS_KV_MAX_BYTES, ARGUS_KV_RESIDENT_BYTES, ARGUS_KV_STATS_PATH,
                               BASE_URL, argus_enabled)
except ModuleNotFoundError:
    from config import (ARGUS_KV_MAX_BYTES, ARGUS_KV_RESIDENT_BYTES, ARGUS_KV_STATS_PATH,
                        BASE_URL, argus_enabled)

logger = logging.getLogger("argus_chat")

CHAT_PAGE = Path(__file__).resolve().parent / "web" / "chat.html"
MAX_MESSAGES = 200
MAX_MESSAGE_CHARS = 200_000
MAX_TOKENS = 4096
ROLES = {"system", "user", "assistant"}


def read_argus_stats() -> Optional[Dict[str, Any]]:
    """Latest counters published by the patched llama-server, or None before the first publish."""
    try:
        return json.loads(ARGUS_KV_STATS_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def validate_chat_payload(body: Any) -> Dict[str, Any]:
    """Keep only plain chat fields; reject anything that isn't a bounded message list."""
    if not isinstance(body, dict):
        raise ValueError("JSON object required")
    messages = body.get("messages")
    if not isinstance(messages, list) or not 0 < len(messages) <= MAX_MESSAGES:
        raise ValueError(f"messages must be a list of 1..{MAX_MESSAGES} items")
    clean = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in ROLES:
            raise ValueError("each message needs role system, user or assistant")
        content = message.get("content")
        if not isinstance(content, str) or len(content) > MAX_MESSAGE_CHARS:
            raise ValueError(f"message content must be text up to {MAX_MESSAGE_CHARS} characters")
        clean.append({"role": message["role"], "content": content})
    payload: Dict[str, Any] = {"messages": clean, "stream": True}
    for key, low, high, cast in (("max_tokens", 1, MAX_TOKENS, int), ("temperature", 0.0, 2.0, float)):
        if key in body:
            try:
                payload[key] = min(max(cast(body[key]), low), high)
            except (TypeError, ValueError):
                raise ValueError(f"{key} must be a number") from None
    return payload


def build_routes(auth_ok: Callable[[Request], bool], http_client: httpx.AsyncClient) -> list[Route]:
    # ponytail: one generation at a time matches llama-server -np 1; add a queue if slots grow.
    generation_lock = asyncio.Lock()

    async def status(request: Request) -> Response:
        if not auth_ok(request):
            return JSONResponse({"error": "Unauthorized"}, status_code=401)
        return JSONResponse({
            "enabled": argus_enabled(),
            "max_bytes": int(ARGUS_KV_MAX_BYTES) if argus_enabled() else None,
            "resident_budget_bytes": int(ARGUS_KV_RESIDENT_BYTES) if ARGUS_KV_RESIDENT_BYTES else None,
            "stats": read_argus_stats() if argus_enabled() else None,
            "generating": generation_lock.locked(),
        })

    async def chat(request: Request) -> Response:
        if not auth_ok(request):
            return JSONResponse({"error": "Unauthorized"}, status_code=401)
        try:
            payload = validate_chat_payload(await request.json())
        except (ValueError, json.JSONDecodeError) as error:
            return JSONResponse({"error": str(error)}, status_code=400)
        if generation_lock.locked():
            return JSONResponse({"error": "Model is busy with another reply"}, status_code=429)

        async def relay():
            # Taken inside the generator so a client that disconnects before streaming
            # starts can never strand the lock; the fast path of acquire() doesn't yield.
            if generation_lock.locked():
                yield f"data: {json.dumps({'error': 'Model is busy with another reply'})}\n\n"
                return
            await generation_lock.acquire()
            try:
                async with http_client.stream("POST", f"{BASE_URL}/v1/chat/completions",
                                              json=payload, timeout=None) as upstream:
                    if upstream.status_code != 200:
                        detail = (await upstream.aread()).decode(errors="replace")[:500]
                        logger.warning("llama-server chat failed: %s %s", upstream.status_code, detail)
                        yield f"data: {json.dumps({'error': 'Model server rejected the request'})}\n\n"
                        return
                    async for chunk in upstream.aiter_bytes():
                        yield chunk
            except httpx.HTTPError as error:
                logger.warning("llama-server unreachable: %s", error)
                yield f"data: {json.dumps({'error': 'Model server is unreachable'})}\n\n"
            finally:
                generation_lock.release()

        return StreamingResponse(relay(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    async def page(_: Request) -> Response:
        return FileResponse(CHAT_PAGE, media_type="text/html")

    return [
        Route("/chat", page, methods=["GET"]),
        Route("/api/argus/status", status, methods=["GET"]),
        Route("/api/chat/completions", chat, methods=["POST"]),
    ]
