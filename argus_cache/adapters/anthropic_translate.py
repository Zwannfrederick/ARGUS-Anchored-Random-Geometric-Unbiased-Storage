"""Schema translation between Anthropic Messages API and OpenAI Chat Completions API."""

from __future__ import annotations

import json
import uuid
from typing import Any, Dict, List, Optional, Tuple

import httpx

from argus_cache.adapters.claude_session import model_label_for


def discover_gateway_models(upstream_url: str) -> List[Dict[str, str]]:
    """Build the /v1/models list: Anthropic defaults plus whatever llama-server serves."""
    items: List[Dict[str, str]] = [
        {"id": "claude-3-7-sonnet-20250219", "object": "model"},
        {"id": "claude-3-5-sonnet-20241022", "object": "model"},
        {"id": "claude-3-opus-20240229", "object": "model"},
        {"id": "claude-3-5-haiku-20241022", "object": "model"},
    ]
    seen = {m["id"] for m in items}

    def _add(model_id: Optional[str], display_name: Optional[str] = None) -> None:
        if not model_id or model_id in seen:
            return
        seen.add(model_id)
        entry = {"id": model_id, "object": "model"}
        if display_name:
            entry["display_name"] = display_name
        items.append(entry)

    try:
        with httpx.Client(timeout=0.5) as probe:
            up_resp = probe.get(f"{upstream_url.rstrip('/')}/models")
            if up_resp.status_code == 200:
                for m in up_resp.json().get("data", []):
                    raw_id = m.get("id") if isinstance(m, dict) else None
                    if raw_id:
                        stem = model_label_for(raw_id)
                        _add(stem, f"ARGUS: {stem}")
                        _add(raw_id)
    except (httpx.HTTPError, ValueError, AttributeError):
        pass  # upstream still loading: the Anthropic defaults are a usable list

    return items


class MessageFormatAdapter:
    """Translates schemas between Anthropic Messages API and OpenAI Chat Completions API."""

    @staticmethod
    def anthropic_tools_to_openai(tools: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
        """Converts Anthropic tool schemas to OpenAI tool schemas."""
        if not tools:
            return None
        openai_tools = []
        for t in tools:
            if not isinstance(t, dict) or not t.get("name"):
                continue
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": t.get("name"),
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
                },
            })
        return openai_tools

    @staticmethod
    def _extract_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(
                str(b.get("text", ""))
                for b in content
                if isinstance(b, dict) and b.get("type") == "text" and b.get("text")
            )
        return str(content or "")

    @classmethod
    def anthropic_messages_to_openai(
        cls,
        messages: List[Dict[str, Any]],
        system_prompt: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """Converts Anthropic conversation history to OpenAI message list."""
        system_parts: List[str] = []
        body_msgs: List[Dict[str, Any]] = []

        if system_prompt:
            sys_top = cls._extract_text(system_prompt)
            if sys_top:
                system_parts.append(sys_top)

        for msg in messages:
            role = msg.get("role") or "user"
            content = msg.get("content")

            if role in ("system", "developer"):
                sys_txt = cls._extract_text(content)
                if sys_txt:
                    if not body_msgs:
                        system_parts.append(sys_txt)
                    else:
                        body_msgs.append({"role": "user", "content": sys_txt})
                continue

            if role not in ("user", "assistant", "tool"):
                role = "user"

            if isinstance(content, str):
                body_msgs.append({"role": role, "content": content})
                continue

            if not isinstance(content, list):
                body_msgs.append({"role": role, "content": str(content or "")})
                continue

            text_parts = []
            thinking_parts = []
            tool_calls = []
            tool_results = []

            for block in content:
                if not isinstance(block, dict):
                    continue
                b_type = block.get("type")
                if b_type == "text":
                    text_parts.append(block.get("text", ""))
                elif b_type == "thinking" and block.get("thinking"):
                    thinking_parts.append(str(block.get("thinking")))
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
                if thinking_parts:
                    msg_dict["reasoning_content"] = "\n".join(thinking_parts)
                if tool_calls:
                    msg_dict["tool_calls"] = tool_calls
                if not text_parts and not tool_calls:
                    msg_dict["content"] = ""
                body_msgs.append(msg_dict)
            elif role == "user":
                if tool_results:
                    body_msgs.extend(tool_results)
                if text_parts or not tool_results:
                    body_msgs.append({"role": "user", "content": "\n".join(text_parts)})
            else:
                body_msgs.append({"role": role, "content": "\n".join(text_parts)})

        openai_msgs: List[Dict[str, Any]] = []
        if system_parts:
            combined_sys = "\n\n".join(system_parts)
            if body_msgs:
                openai_msgs.append({"role": "system", "content": combined_sys})
            else:
                body_msgs.append({"role": "user", "content": combined_sys})
        openai_msgs.extend(body_msgs)
        return openai_msgs

    @classmethod
    def anthropic_request_to_openai(
        cls, data: Dict[str, Any], *, num_ctx: Optional[int] = None
    ) -> Dict[str, Any]:
        """Translates full Anthropic request payload to OpenAI payload."""
        messages = data.get("messages", [])
        system = data.get("system")
        openai_messages = cls.anthropic_messages_to_openai(messages, system)

        thinking_cfg = data.get("thinking")
        enable_thinking = isinstance(thinking_cfg, dict) and thinking_cfg.get("type") in ("enabled", "adaptive")

        openai_req: Dict[str, Any] = {
            "model": data.get("model", "qwen3.6-35b-a3b"),
            "messages": openai_messages,
            "stream": data.get("stream", False),
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
        }

        if "max_tokens" in data and data["max_tokens"]:
            max_out = int(data["max_tokens"])
            if num_ctx is not None:
                max_out = min(max_out, min(16384, max(4096, num_ctx // 8)))
            openai_req["max_tokens"] = max_out
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

        reasoning = message.get("reasoning_content") or message.get("reasoning") or message.get("thinking")
        if reasoning:
            content_blocks.append({"type": "thinking", "thinking": reasoning})

        text = message.get("content")
        if text:
            content_blocks.append({"type": "text", "text": text})

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


_FINISH_TO_STOP = {"length": "max_tokens", "tool_calls": "tool_use"}

Event = Tuple[str, Dict[str, Any]]


class AnthropicStreamTranslator:
    """OpenAI ``chat.completion.chunk`` stream -> Anthropic Messages SSE events.

    Transport-free so the stdlib gateway and the Studio server share one state
    machine: feed it parsed chunks, write out the ``(event, data)`` pairs it
    returns. Reasoning, text and tool calls each open their own content block,
    and switching kind closes the previous block, as the Anthropic stream does.
    """

    def __init__(self, model: str) -> None:
        self.model = model
        self.index = 0
        self.open_kind: Optional[str] = None  # "thinking" | "text" | "tool_use"
        self.finish_reason: Optional[str] = None
        self.input_tokens: Optional[int] = None
        self.output_tokens: Optional[int] = None
        self._estimated_output = 0

    def start(self) -> List[Event]:
        return [("message_start", {
            "type": "message_start",
            "message": {
                "id": f"msg_{uuid.uuid4().hex[:24]}",
                "type": "message",
                "role": "assistant",
                "model": self.model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        })]

    def _close(self) -> List[Event]:
        if self.open_kind is None:
            return []
        self.open_kind = None
        event = ("content_block_stop", {"type": "content_block_stop", "index": self.index})
        self.index += 1
        return [event]

    def _open(self, kind: str, block: Dict[str, Any]) -> List[Event]:
        events = self._close()
        self.open_kind = kind
        events.append(("content_block_start", {
            "type": "content_block_start", "index": self.index, "content_block": block,
        }))
        return events

    def _delta(self, delta: Dict[str, Any]) -> Event:
        return ("content_block_delta", {"type": "content_block_delta", "index": self.index, "delta": delta})

    def feed(self, chunk: Dict[str, Any]) -> List[Event]:
        usage = chunk.get("usage")
        if isinstance(usage, dict):
            if isinstance(usage.get("prompt_tokens"), int):
                self.input_tokens = usage["prompt_tokens"]
            if isinstance(usage.get("completion_tokens"), int):
                self.output_tokens = usage["completion_tokens"]
        choices = chunk.get("choices") or []
        if not choices:
            return []
        choice = choices[0]
        self.finish_reason = choice.get("finish_reason") or self.finish_reason
        delta = choice.get("delta") or {}
        events: List[Event] = []

        reasoning = delta.get("reasoning_content") or delta.get("reasoning") or delta.get("thinking")
        if reasoning:
            self._estimated_output += max(1, len(reasoning) // 4)
            if self.open_kind != "thinking":
                events += self._open("thinking", {"type": "thinking", "thinking": ""})
            events.append(self._delta({"type": "thinking_delta", "thinking": reasoning}))

        text = delta.get("content")
        if text:
            self._estimated_output += max(1, len(text) // 4)
            if self.open_kind != "text":
                events += self._open("text", {"type": "text", "text": ""})
            events.append(self._delta({"type": "text_delta", "text": text}))

        for tc in delta.get("tool_calls") or []:
            fn = tc.get("function") or {}
            if fn.get("name"):
                events += self._open("tool_use", {
                    "type": "tool_use",
                    "id": tc.get("id") or f"toolu_{uuid.uuid4().hex[:16]}",
                    "name": fn["name"],
                    "input": {},
                })
            args = fn.get("arguments")
            if args and self.open_kind == "tool_use":
                self._estimated_output += max(1, len(args) // 4)
                events.append(self._delta({"type": "input_json_delta", "partial_json": args}))
        return events

    def finish(self) -> List[Event]:
        saw_tool = self.open_kind == "tool_use" or self.finish_reason == "tool_calls"
        events = self._close()
        stop = "tool_use" if saw_tool else _FINISH_TO_STOP.get(self.finish_reason or "", "end_turn")
        out = self.output_tokens if self.output_tokens is not None else max(1, self._estimated_output)
        events.append(("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": stop, "stop_sequence": None},
            "usage": {"output_tokens": out},
        }))
        events.append(("message_stop", {"type": "message_stop"}))
        return events

    @staticmethod
    def parse_sse_line(line: str) -> Optional[Dict[str, Any]]:
        """One ``data:`` line of an OpenAI stream as a dict; None for anything else."""
        if not line.startswith("data:"):
            return None
        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            return None
        try:
            chunk = json.loads(payload)
        except ValueError:
            return None
        return chunk if isinstance(chunk, dict) else None


def format_sse(event: str, data: Dict[str, Any]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode("utf-8")
