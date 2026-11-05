"""Streaming multi-round agent loop, cancellation manager, and chronological timeline recorder."""

from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx

from argus_cache.webui.compaction import estimate_active_tokens
from argus_cache.webui.permissions import PermissionBroker
from argus_cache.webui.research_tools import audit_citations
from argus_cache.webui.store import StudioStore
from argus_cache.webui.tools import ToolRunner

_JSON_TOOL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_QWEN_XML_TOOL_RE = re.compile(
    r"<tool_call>\s*<function=([a-zA-Z0-9_]+)>\s*(.*?)\s*</function>\s*</tool_call>",
    re.DOTALL,
)
_QWEN_PARAM_RE = re.compile(r"<parameter=([a-zA-Z0-9_]+)>\s*(.*?)\s*</parameter>", re.DOTALL)
_ANY_TOOL_BLOCK_RE = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL)
_THINK_TAG_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_FAKE_TOOL_ECHO_RE = re.compile(r"\[Araç Çıktısı\s*[—\-:][^\]]*\].*", re.DOTALL)


def strip_fake_tool_echo(text: str) -> str:
    if "[Araç Çıktısı" not in (text or ""):
        return text or ""
    return _FAKE_TOOL_ECHO_RE.sub("", text).strip()


def fallback_session_title(prompt: str) -> str:
    cleaned = re.sub(r"\s+", " ", (prompt or "").strip()).strip("\"'`.,:;!?-— ")
    if not cleaned:
        return "Yeni Sohbet"
    words = cleaned.split(" ")
    short = " ".join(words[:6])
    return (short[:48].rstrip() + ("…" if len(short) > 48 else "")) or "Yeni Sohbet"


def clean_ai_session_title(raw: str, fallback_prompt: str = "") -> str:
    txt = _THINK_TAG_RE.sub("", raw or "")
    txt = _ANY_TOOL_BLOCK_RE.sub("", txt)
    txt = strip_fake_tool_echo(txt)
    lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
    txt = lines[0] if lines else ""
    txt = re.sub(r"^(?:başlık|baslik|title|sohbet başlığı)\s*[:\-—]\s*", "", txt, flags=re.IGNORECASE)
    txt = re.sub(r"[#*`\"'«»“”]+", "", txt)
    txt = re.sub(r"\s+", " ", txt).strip(" .,:;!?-—/\\|")
    if len(txt) < 2:
        return fallback_session_title(fallback_prompt)
    if len(txt) > 52:
        txt = txt[:52].rsplit(" ", 1)[0].strip() or txt[:52]
    return txt


async def _generate_ai_session_title(
    *,
    client: httpx.AsyncClient,
    upstream_url: str,
    llm_messages: List[Dict[str, Any]],
    final_answer: str,
    tool_schemas: List[Dict[str, Any]],
    fallback_prompt: str,
) -> str:
    """Generate a concise 2-5 word session title reusing the active KV-cache prefix."""
    title_msgs = list(llm_messages)
    if final_answer:
        title_msgs.append({"role": "assistant", "content": final_answer[:1000]})
    title_msgs.append({
        "role": "user",
        "content": (
            "Bu sohbete konuyu özetleyen 2-5 kelimelik kısa, net bir başlık ver "
            "(kullanıcının dilinde; tırnak, nokta veya açıklama eklemeden yalnızca başlığı yaz)."
        ),
    })
    payload: Dict[str, Any] = {
        "messages": title_msgs,
        "stream": True,
        "temperature": 0.2,
        "max_tokens": 24,
        "n_predict": 24,
        "reasoning_budget_tokens": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if tool_schemas:
        payload["tools"] = tool_schemas

    pieces: List[str] = []
    try:
        async with client.stream("POST", f"{upstream_url}/chat/completions", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                raw = line[6:].strip()
                if raw == "[DONE]":
                    break
                try:
                    chunk = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                c_piece = (choices[0].get("delta") or {}).get("content") or ""
                if c_piece:
                    pieces.append(c_piece)
                    if "\n" in c_piece or sum(len(x) for x in pieces) >= 72:
                        break
    except Exception:
        return fallback_session_title(fallback_prompt)
    return clean_ai_session_title("".join(pieces), fallback_prompt)


def sse_frame(event: str, payload: Dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _coerce_xml_val(raw_val: str) -> Any:
    val = raw_val.strip()
    if not val:
        return ""
    if val[0] in ("[", "{", '"') or val in ("true", "false", "null") or re.fullmatch(r"-?\d+(?:\.\d+)?", val):
        try:
            return json.loads(val)
        except json.JSONDecodeError:
            pass
    return val


def parse_fallback_tool_calls(text: str) -> List[Dict[str, str]]:
    calls: List[Dict[str, str]] = []
    for idx, m in enumerate(_JSON_TOOL_RE.finditer(text or "")):
        try:
            obj = json.loads(m.group(1))
            name = obj.get("name")
            args = obj.get("arguments", {})
            if name:
                calls.append({
                    "id": f"call_json_{idx}_{uuid.uuid4().hex[:6]}",
                    "name": str(name),
                    "arguments": json.dumps(args, ensure_ascii=False) if isinstance(args, dict) else str(args),
                })
        except json.JSONDecodeError:
            continue

    if calls:
        return calls

    for idx, m in enumerate(_QWEN_XML_TOOL_RE.finditer(text or "")):
        fn_name = m.group(1).strip()
        body = m.group(2)
        args_dict: Dict[str, Any] = {}
        for pm in _QWEN_PARAM_RE.finditer(body):
            args_dict[pm.group(1).strip()] = _coerce_xml_val(pm.group(2))
        if fn_name:
            calls.append({
                "id": f"call_xml_{idx}_{uuid.uuid4().hex[:6]}",
                "name": fn_name,
                "arguments": json.dumps(args_dict, ensure_ascii=False),
            })
    return calls


class StreamManager:
    """Coordinates active SSE/LLM tasks so single-slot `llama-server` cancels deterministically."""

    def __init__(self, broker: PermissionBroker) -> None:
        self.broker = broker
        self._tasks: Dict[str, asyncio.Task[None]] = {}

    def register(self, session_id: str, task: asyncio.Task[None]) -> None:
        self._tasks[session_id] = task

    def unregister(self, session_id: str, task: asyncio.Task[None]) -> None:
        if self._tasks.get(session_id) is task:
            self._tasks.pop(session_id, None)

    def is_active(self, session_id: Optional[str] = None) -> bool:
        if session_id:
            t = self._tasks.get(session_id)
            return bool(t and not t.done())
        return any(not t.done() for t in self._tasks.values())

    async def stop_session(self, session_id: str) -> bool:
        self.broker.cancel_session_pending(session_id)
        task = self._tasks.get(session_id)
        if not task or task.done():
            self._tasks.pop(session_id, None)
            return False
        task.cancel()
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=1.5)
        except (asyncio.CancelledError, asyncio.TimeoutError, Exception):
            pass
        self._tasks.pop(session_id, None)
        return True

    async def stop_all(self) -> int:
        sids = list(self._tasks.keys())
        count = 0
        for sid in sids:
            if await self.stop_session(sid):
                count += 1
        return count


async def run_agent_stream(
    *,
    store: StudioStore,
    stream_mgr: StreamManager,
    runner: ToolRunner,
    upstream_url: str,
    session_id: str,
    clean_title: str,
    mode: str,
    temperature: float,
    enable_thinking: bool,
    llm_messages: List[Dict[str, Any]],
    tool_schemas: List[Dict[str, Any]],
    auto_compacted_info: Optional[Dict[str, Any]],
    needs_ai_title: bool = False,
    user_prompt_for_title: str = "",
) -> AsyncIterator[str]:
    await stream_mgr.stop_all()

    event_queue: asyncio.Queue[Optional[tuple[str, Dict[str, Any]]]] = asyncio.Queue()
    timeline: List[Dict[str, Any]] = []

    def emit(ev: str, data: Dict[str, Any]) -> None:
        if ev == "pc_screenshot":
            timeline.append({
                "type": "screenshot",
                "url": data.get("url", ""),
                "window": data.get("window", ""),
                "scope": data.get("scope", ""),
            })
        event_queue.put_nowait((ev, data))

    runner.emit_event = emit

    async def _worker() -> None:
        emit("session_meta", {"session_id": session_id, "title": clean_title})
        if auto_compacted_info:
            emit("compaction_notice", auto_compacted_info)

        final_text_parts: List[str] = []
        final_reasoning_parts: List[str] = []
        recorded_tools: List[Dict[str, Any]] = []
        last_timings: Dict[str, Any] = {}
        t_start = time.perf_counter()
        out_tokens = 0
        empty_nudges = 0
        max_rounds = 18 if mode == "research" else 12
        cur_round_content: List[str] = []
        cur_round_reasoning: List[str] = []

        try:
            stream_timeout = httpx.Timeout(connect=15.0, read=None, write=60.0, pool=15.0)
            async with httpx.AsyncClient(timeout=stream_timeout) as client:
                for _round in range(max_rounds):
                    round_no = _round + 1
                    cur_round_content = []
                    cur_round_reasoning = []
                    tc_accum: Dict[int, Dict[str, str]] = {}
                    saw_thinking = False
                    saw_writing = False

                    emit("round_start", {"round": round_no})
                    emit(
                        "phase",
                        {
                            "phase": "prefill",
                            "round": round_no,
                            "label": "Bağlam işleniyor…" if _round == 0 else f"Araç çıktıları değerlendiriliyor (Adım {round_no})…",
                        },
                    )

                    payload: Dict[str, Any] = {
                        "messages": llm_messages,
                        "stream": True,
                        "temperature": temperature,
                        "n_predict": -1,
                        "reasoning_budget_tokens": -1 if enable_thinking else 0,
                        "chat_template_kwargs": {"enable_thinking": enable_thinking},
                    }
                    if tool_schemas:
                        payload["tools"] = tool_schemas

                    async with client.stream(
                        "POST",
                        f"{upstream_url}/chat/completions",
                        json=payload,
                    ) as resp:
                        resp.raise_for_status()
                        async for line in resp.aiter_lines():
                            if not line or not line.startswith("data: "):
                                continue
                            raw = line[6:].strip()
                            if raw == "[DONE]":
                                break
                            try:
                                chunk = json.loads(raw)
                            except json.JSONDecodeError:
                                continue

                            timings = chunk.get("timings")
                            if timings and timings.get("predicted_per_second"):
                                last_timings = {
                                    "decode_tps": round(float(timings["predicted_per_second"]), 1),
                                    "prefill_tps": round(float(timings.get("prompt_per_second", 0.0)), 1),
                                }

                            choices = chunk.get("choices") or []
                            if not choices:
                                continue
                            delta = choices[0].get("delta") or {}

                            r_piece = delta.get("reasoning_content") or ""
                            if r_piece:
                                if not saw_thinking:
                                    saw_thinking = True
                                    emit("phase", {"phase": "thinking", "round": round_no, "label": "Düşünüyor…"})
                                cur_round_reasoning.append(r_piece)
                                final_reasoning_parts.append(r_piece)
                                out_tokens += 1
                                emit("reasoning_delta", {"round": round_no, "text": r_piece})

                            c_piece = delta.get("content") or ""
                            if c_piece:
                                if not saw_writing:
                                    saw_writing = True
                                    emit("phase", {"phase": "writing", "round": round_no, "label": "Yazıyor…"})
                                cur_round_content.append(c_piece)
                                out_tokens += 1
                                emit("content_delta", {"round": round_no, "text": c_piece})

                            for tc in delta.get("tool_calls") or []:
                                idx = int(tc.get("index") or 0)
                                slot = tc_accum.setdefault(
                                    idx,
                                    {"id": tc.get("id") or f"call_{idx}_{uuid.uuid4().hex[:6]}", "name": "", "arguments": ""},
                                )
                                if tc.get("id"):
                                    slot["id"] = tc["id"]
                                fn = tc.get("function") or {}
                                if fn.get("name"):
                                    slot["name"] += fn["name"]
                                if fn.get("arguments"):
                                    slot["arguments"] += fn["arguments"]

                    joined_reasoning = "".join(cur_round_reasoning)
                    joined_content = "".join(cur_round_content)
                    cleaned_stream = False

                    if "<think>" in joined_content:
                        for tm in _THINK_TAG_RE.finditer(joined_content):
                            joined_reasoning += tm.group(1)
                            final_reasoning_parts.append(tm.group(1))
                        joined_content = _THINK_TAG_RE.sub("", joined_content).strip()
                        cleaned_stream = True

                    completed_calls = [tc_accum[k] for k in sorted(tc_accum) if tc_accum[k]["name"]]
                    if not completed_calls and "<tool_call>" in joined_content:
                        completed_calls = parse_fallback_tool_calls(joined_content)
                        if completed_calls:
                            joined_content = _ANY_TOOL_BLOCK_RE.sub("", joined_content).strip()
                            cleaned_stream = True
                    if not completed_calls and "<tool_call>" in joined_reasoning:
                        completed_calls = parse_fallback_tool_calls(joined_reasoning)
                        if completed_calls:
                            joined_reasoning = _ANY_TOOL_BLOCK_RE.sub("", joined_reasoning).strip()

                    saw_fake_tool_echo = False
                    if not completed_calls and "[Araç Çıktısı" in joined_content:
                        saw_fake_tool_echo = True
                        joined_content = strip_fake_tool_echo(joined_content)
                        cleaned_stream = True

                    if cleaned_stream:
                        emit("round_content_replace", {"round": round_no, "text": joined_content})
                        emit("content_replace", {"text": "".join(final_text_parts) + joined_content})

                    if joined_reasoning.strip():
                        timeline.append({"type": "reasoning", "round": round_no, "text": joined_reasoning.strip()})
                    if joined_content.strip() and not saw_fake_tool_echo:
                        timeline.append({"type": "text", "round": round_no, "text": joined_content.strip()})

                    if not completed_calls:
                        if saw_fake_tool_echo and tool_schemas and empty_nudges < 2:
                            empty_nudges += 1
                            emit("round_content_replace", {"round": round_no, "text": ""})
                            emit("content_replace", {"text": "".join(final_text_parts)})
                            emit(
                                "phase",
                                {
                                    "phase": "continuing",
                                    "round": round_no + 1,
                                    "label": "Gerçek araç çağrısı yürütülüyor…",
                                },
                            )
                            llm_messages.append({
                                "role": "user",
                                "content": (
                                    "[OTOMATİK DÜZELTME]: Metin içinde sahte '[Araç Çıktısı]' yazdın ama hiçbir aracı "
                                    "gerçekten çağırmadın. Sahte çıktı yazmak YASAKTIR; işlemi yapmak için doğrudan "
                                    "ilgili aracı (`pc_click`, `pc_focus_or_launch`, `pc_capture_screen` vb.) gerçek "
                                    "tool_call olarak çağır."
                                ),
                            })
                            continue

                        # Auto-continue if the model only emitted reasoning (or finished a tool turn empty) without answering!
                        if not joined_content.strip() and (joined_reasoning.strip() or _round > 0) and empty_nudges < 2:
                            empty_nudges += 1
                            emit(
                                "phase",
                                {
                                    "phase": "continuing",
                                    "round": round_no + 1,
                                    "label": "Planlanan adım otomatik sürdürülüyor…",
                                },
                            )
                            if joined_reasoning.strip():
                                llm_messages.append({"role": "assistant", "content": ""})
                            llm_messages.append({
                                "role": "user",
                                "content": (
                                    "[OTOMATİK AKIŞ DEVAMI]: İç düşüncede planladığın aracı henüz çağırmadın ve kullanıcıya "
                                    "görünür yanıt üretmedin. Kullanıcı girdisi beklemeden doğrudan planladığın aracı "
                                    "çağır veya bulgularını yanıt olarak yaz."
                                ),
                            })
                            continue

                        if joined_content.strip():
                            final_text_parts.append(joined_content)
                        break

                    if joined_content.strip():
                        final_text_parts.append(joined_content.strip() + "\n\n")

                    llm_messages.append({
                        "role": "assistant",
                        "content": joined_content or "",
                        "tool_calls": [
                            {
                                "id": c["id"],
                                "type": "function",
                                "function": {"name": c["name"], "arguments": c["arguments"]},
                            }
                            for c in completed_calls
                        ],
                    })

                    async def _exec_one(call: Dict[str, str]) -> tuple[Dict[str, str], str, Dict[str, Any]]:
                        emit("phase", {"phase": "tool", "round": round_no, "tool": call["name"], "label": f"Araç çalışıyor: {call['name']}…"})
                        tool_tl_item: Dict[str, Any] = {
                            "type": "tool",
                            "round": round_no,
                            "id": call["id"],
                            "name": call["name"],
                            "arguments": call["arguments"],
                            "preview": "",
                        }
                        timeline.append(tool_tl_item)
                        emit("tool_start", {"round": round_no, "id": call["id"], "name": call["name"], "arguments": call["arguments"]})
                        t_out = await runner.execute(call["name"], call["arguments"])
                        prev = t_out[:280] + ("…" if len(t_out) > 280 else "")
                        out_digest = t_out[:2800] + ("…" if len(t_out) > 2800 else "")
                        tool_tl_item["preview"] = prev
                        emit("tool_end", {"id": call["id"], "name": call["name"], "arguments": call["arguments"], "preview": prev})
                        rec = {
                            "id": call["id"],
                            "name": call["name"],
                            "arguments": call["arguments"],
                            "preview": prev,
                            "output": out_digest,
                        }
                        return call, t_out, rec

                    tool_results = await asyncio.gather(*(_exec_one(c) for c in completed_calls))
                    for call, tool_output, evt_rec in tool_results:
                        recorded_tools.append(evt_rec)
                        llm_messages.append({
                            "role": "tool",
                            "tool_call_id": call["id"],
                            "name": call["name"],
                            "content": tool_output,
                        })

                if not last_timings and out_tokens > 0:
                    elapsed = max(0.01, time.perf_counter() - t_start)
                    last_timings = {"decode_tps": round(out_tokens / elapsed, 1)}

                final_answer = "".join(final_text_parts).strip()
                if mode == "research" or runner.python_verified:
                    c_audit = audit_citations(final_answer, runner.verified_urls)
                    if mode == "research" and c_audit["unverified_urls"]:
                        warn_note = (
                            "\n\n> ⚠️ **Bağımsız Kaynak Denetimi:** Yanıtta geçen şu adresler bu oturumdaki "
                            "`web_search`/`read_webpage` kayıtlarında doğrulanamadı: "
                            + ", ".join(f"`{u}`" for u in c_audit["unverified_urls"])
                        )
                        final_answer += warn_note
                        timeline.append({"type": "text", "round": 99, "text": warn_note.strip()})
                        emit("content_delta", {"round": 99, "text": warn_note})
                    emit(
                        "grounding_audit",
                        {
                            "mode": mode,
                            "python_verified": runner.python_verified,
                            "verified_source_count": max(len(c_audit["verified_urls"]), len(runner.citations)),
                            "unverified_urls": c_audit["unverified_urls"],
                        },
                    )

                store.add_message(
                    session_id,
                    role="assistant",
                    content=final_answer,
                    reasoning="".join(final_reasoning_parts).strip(),
                    citations=runner.citations,
                    tool_events=recorded_tools,
                    timeline=timeline,
                    timings=last_timings,
                )

                if needs_ai_title:
                    emit("phase", {"phase": "titling", "round": 99, "label": "Sohbet başlığı oluşturuluyor…"})
                    ai_title = await _generate_ai_session_title(
                        client=client,
                        upstream_url=upstream_url,
                        llm_messages=llm_messages,
                        final_answer=final_answer,
                        tool_schemas=tool_schemas,
                        fallback_prompt=user_prompt_for_title,
                    )
                    store.update_session(session_id, title=ai_title, custom_title=True)
                    emit("session_meta", {"session_id": session_id, "title": ai_title})

                if last_timings:
                    emit("timings", last_timings)
                refreshed_sess = store.get_session(session_id)
                emit(
                    "done",
                    {
                        "ok": True,
                        "active_est_tokens": estimate_active_tokens(refreshed_sess) if refreshed_sess else 0,
                    },
                )
        except asyncio.CancelledError:
            partial_r = "".join(cur_round_reasoning).strip()
            partial_c = "".join(cur_round_content).strip()
            if partial_r and (not timeline or timeline[-1].get("text") != partial_r):
                timeline.append({"type": "reasoning", "round": 0, "text": partial_r})
            if partial_c:
                final_text_parts.append(partial_c)
                timeline.append({"type": "text", "round": 0, "text": partial_c})
            timeline.append({"type": "stopped", "text": "⏹ İstek kullanıcı tarafından durduruldu."})
            partial_ans = "".join(final_text_parts).strip()
            if partial_ans or final_reasoning_parts or recorded_tools:
                store.add_message(
                    session_id,
                    role="assistant",
                    content=partial_ans or "_(İstek durduruldu)_",
                    reasoning="".join(final_reasoning_parts).strip(),
                    citations=runner.citations,
                    tool_events=recorded_tools,
                    timeline=timeline,
                    timings=last_timings,
                )
            emit("stopped", {"session_id": session_id})
            raise
        except Exception as exc:
            emit("error", {"message": str(exc)})
        finally:
            event_queue.put_nowait(None)

    task = asyncio.create_task(_worker())
    stream_mgr.register(session_id, task)

    try:
        while True:
            try:
                item = await asyncio.wait_for(event_queue.get(), timeout=4.0)
            except asyncio.TimeoutError:
                yield ": keepalive\n\n"
                continue
            if item is None:
                break
            ev, data = item
            yield sse_frame(ev, data)
    finally:
        if not task.done():
            await stream_mgr.stop_session(session_id)
        else:
            stream_mgr.unregister(session_id, task)
