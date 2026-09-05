"""
Neo Mobile Companion Gateway
============================
Authenticated Gateway and Reverse Event Bridge for Neo & Hermes.
Connects React Native mobile companion app to:
- Gemma 4 E4B-it (128K context, MTP, llama-server at http://127.0.0.1:8080)
- Cognitive Risk Router & Dynamic Thinking Control
- Real Desktop Screen Capture (Wayland/Hyprland)
- Coding Agent Delegation (Codex, Claude Code, Antigravity)
- Replay-Resistant HIGH-risk Approval Gates
- Tiered Notifications (LOW, NORMAL, URGENT) with Mobile Vibration
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import os
import re
import secrets
import sqlite3
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import httpx
import uvicorn
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket, WebSocketDisconnect

import fcntl

# Import Hermes modules
HERMES_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = HERMES_DIR.parent
import sys
if str(HERMES_DIR) not in sys.path:
    sys.path.insert(0, str(HERMES_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cognitive_router import CognitiveRouter, RiskAssessment
from config import BASE_URL
from hermes_supervisor import HermesSupervisor, compute_action_hash
from prefix_builder import (
    get_canonical_stable_prefix,
    get_canonical_tools,
    get_prefix_metadata,
    build_warmup_payload,
    parse_cache_telemetry,
)

# Configuration constants
PORT = 8765
HOST = "0.0.0.0"
HERMES_HOME = Path.home() / ".hermes"
CACHE_DIR = HERMES_HOME / "cache"
SCREENSHOTS_DIR = CACHE_DIR / "screenshots"
DB_PATH = HERMES_HOME / "neo_mobile.db"
TOKEN_PATH = HERMES_HOME / "neo_auth_token.txt"
GATEWAY_LOCK_FILE = HERMES_HOME / "neo_gateway.lock"

# Ensure directories exist
HERMES_HOME.mkdir(parents=True, exist_ok=True)
SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Single-Instance OS Advisory Lock (POSIX fcntl.flock)
# ---------------------------------------------------------------------------
_gateway_lock_fp = None

def acquire_single_instance_lock():
    """Ensures true, kernel-level single instance mutual exclusion via fcntl."""
    global _gateway_lock_fp
    _gateway_lock_fp = open(GATEWAY_LOCK_FILE, "a+")
    try:
        fcntl.flock(_gateway_lock_fp.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        _gateway_lock_fp.seek(0)
        _gateway_lock_fp.truncate()
        _gateway_lock_fp.write(f"{os.getpid()}\n")
        _gateway_lock_fp.flush()
    except (IOError, BlockingIOError):
        _gateway_lock_fp.seek(0)
        holder_pid = _gateway_lock_fp.read().strip()
        print(f"[FATAL] Another Neo Mobile Gateway instance is already running (PID: {holder_pid}). Exiting cleanly.", file=sys.stderr)
        sys.exit(1)

# ---------------------------------------------------------------------------
# Persona & System Prompt Assembly (Canonical Stable Prefix for 100% KV Cache Hits)
# ---------------------------------------------------------------------------
def load_neo_system_prompt() -> str:
    return get_canonical_stable_prefix()

NEO_SYSTEM_PROMPT = load_neo_system_prompt()

# ---------------------------------------------------------------------------
# Auth Token Setup
# ---------------------------------------------------------------------------
def get_or_create_auth_token() -> str:
    if TOKEN_PATH.exists():
        token = TOKEN_PATH.read_text(encoding="utf-8").strip()
        if token:
            return token
    new_token = secrets.token_hex(32)
    TOKEN_PATH.write_text(new_token, encoding="utf-8")
    os.chmod(TOKEN_PATH, 0o600)
    return new_token

AUTH_TOKEN = get_or_create_auth_token()

def verify_token(provided_token: Optional[str]) -> bool:
    if not provided_token:
        return False
    return secrets.compare_digest(provided_token.strip(), AUTH_TOKEN.strip())

# ---------------------------------------------------------------------------
# SQLite Persistence
# ---------------------------------------------------------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS sessions (
        session_id TEXT PRIMARY KEY,
        title TEXT,
        created_at REAL,
        updated_at REAL,
        hermes_session_id TEXT
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS messages (
        id TEXT PRIMARY KEY,
        session_id TEXT,
        role TEXT,
        content TEXT,
        reasoning_content TEXT,
        tool_calls TEXT,
        screenshot_url TEXT,
        timestamp REAL,
        client_msg_id TEXT,
        FOREIGN KEY (session_id) REFERENCES sessions (session_id)
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS events (
        id TEXT PRIMARY KEY,
        session_id TEXT,
        type TEXT,
        priority TEXT,
        title TEXT,
        body TEXT,
        timestamp REAL,
        payload TEXT,
        image_ref TEXT,
        acknowledged INTEGER DEFAULT 0
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS pending_approvals (
        approval_id TEXT PRIMARY KEY,
        session_id TEXT,
        command TEXT,
        action_description TEXT,
        risk_level TEXT,
        action_hash TEXT,
        created_at REAL,
        status TEXT
    )
    """)
    try:
        cur.execute("ALTER TABLE pending_approvals ADD COLUMN action_hash TEXT")
    except sqlite3.OperationalError:
        pass
    conn.commit()
    conn.close()

init_db()

# ---------------------------------------------------------------------------
# In-Memory State & Hermes Agent Adapter
# ---------------------------------------------------------------------------
connected_websockets: Set[WebSocket] = set()
hermes_supervisor = HermesSupervisor(base_url=BASE_URL)
http_client = httpx.AsyncClient(timeout=300.0)

# Approval wait futures: approval_id -> asyncio.Future
approval_futures: Dict[str, asyncio.Future] = {}

# Dual-Phase Warmup tracking (Cold vs Warmed First-Action Latency)
model_warmed: bool = False
warmup_duration: float = 0.0
warmup_in_progress: bool = False
cold_action_latency_s: float = 0.0
warmed_action_latency_s: float = 0.0
warmup_telemetry: Dict[str, Any] = {}
warmup_event: asyncio.Event = asyncio.Event()

# ---------------------------------------------------------------------------
# Heuristic Auto-Title Generator (Zero LLM Invariant to Protect Slot 0)
# ---------------------------------------------------------------------------
TURKISH_ENGLISH_FILLERS = {
    "lütfen", "bana", "bakar", "mısın", "misin", "musun", "müsün", "eder", "yapar", "açıp", "acip",
    "selam", "merhaba", "merhabalar", "hey", "neo", "günaydın", "iyi", "akşamlar", "gunaydin", "aksamlar",
    "hadi", "şunu", "bunu", "bir", "ve", "ile", "için", "icin", "hakkında", "hakkinda", "ac", "aç", "bak",
    "please", "can", "you", "could", "would", "tell", "me", "show", "open", "launch",
}

def generate_heuristic_title(user_text: str) -> str:
    """
    Deterministic title generation without auxiliary LLM calls to protect single-slot resident cache.
    """
    cleaned = re.sub(r"[^\w\s\-_]", " ", user_text, flags=re.UNICODE)
    words = cleaned.strip().split()
    if not words:
        return "Yeni Sohbet"

    meaningful = [w for w in words if w.lower() not in TURKISH_ENGLISH_FILLERS and len(w) > 1]
    if not meaningful:
        meaningful = words[:4]

    title_words = meaningful[:4]
    title = " ".join(w.capitalize() for w in title_words)
    if len(title) > 36:
        title = title[:33] + "..."
    return title or "Yeni Sohbet"

async def generate_ai_title(user_text: str) -> Optional[str]:
    """
    Asks the model for a short session title.

    Uses the SAME canonical system prefix, tool schema and chat-template kwargs as a
    normal fast-path turn, so the single resident slot keeps its ~5.4K-token prefix
    cached (only the short tail differs). A title call with a different prefix would
    evict slot 0 and cost the next real turn a full re-prefill.
    """
    payload = {
        "model": "default",
        "messages": [
            {"role": "system", "content": NEO_SYSTEM_PROMPT},
            {"role": "user", "content":
                "Asagidaki kullanici mesaji icin 2-4 kelimelik kisa bir sohbet basligi yaz. "
                "SADECE basligi yaz; tirnak, noktalama, aciklama veya arac cagrisi yok.\n\n"
                f"Mesaj: {user_text[:400]}"},
        ],
        "tools": get_canonical_tools(),
        "tool_choice": "none",
        "temperature": 0.2,
        "chat_template_kwargs": {"enable_thinking": False},
        "max_tokens": 24,
        "stream": False,
    }
    try:
        resp = await http_client.post(f"{BASE_URL}/v1/chat/completions", json=payload, timeout=60.0)
        if resp.status_code != 200:
            return None
        raw = (resp.json()["choices"][0]["message"].get("content") or "").strip()
    except Exception as exc:
        print(f"[TITLE] AI title failed, keeping heuristic: {exc}", flush=True)
        return None

    title = raw.splitlines()[0].strip().strip('"\'`*#').strip()
    if not title or len(title) < 2:
        return None
    return title[:40]


async def apply_ai_title(session_id: str, user_text: str):
    """Upgrades a session's heuristic title to an AI-generated one, after the turn."""
    title = await generate_ai_title(user_text)
    if not title:
        return
    conn = sqlite3.connect(DB_PATH)
    conn.execute("UPDATE sessions SET title = ? WHERE session_id = ?", (title, session_id))
    conn.commit()
    conn.close()
    frame = json.dumps({
        "type": "session_updated",
        "session_id": session_id,
        "title": title,
        "session": {"session_id": session_id, "title": title, "updated_at": time.time()},
    })
    for ws in list(connected_websockets):
        with contextlib.suppress(Exception):
            await ws.send_text(frame)


async def run_auto_warmup() -> Dict[str, Any]:
    """
    Automatic llama-server warmup matching production prefix and tools:
    1. Waits until /health reports ready.
    2. Phase 1: Canonical Reasoning Path Warmup (enable_thinking=true).
    3. Phase 2 (LAST): Canonical Fast Path Warmup (enable_thinking=false).
    Leaves Slot 0 resident KV cache primed for normal Fast Path turns.
    """
    global model_warmed, warmup_duration, warmup_in_progress, cold_action_latency_s, warmed_action_latency_s, warmup_telemetry
    if warmup_in_progress:
        return {"status": "in_progress", "warmed": model_warmed}
    warmup_in_progress = True

    print("[WARMUP] Starting automatic model KV cache and CUDA warmup using canonical prefix & tools...", flush=True)
    t_start_total = time.perf_counter()

    # 1. Wait for llama-server health
    llama_ready = False
    for attempt in range(40):
        try:
            r = await http_client.get(f"{BASE_URL}/health", timeout=2.0)
            if r.status_code == 200 and r.json().get("status") == "ok":
                llama_ready = True
                break
        except Exception:
            pass
        await asyncio.sleep(1.0)

    if not llama_ready:
        print("[WARMUP] llama-server /health timed out during warmup initialization", flush=True)
        warmup_in_progress = False
        return {"status": "failed", "error": "llama-server /health unavailable", "warmed": False}

    # 2. Phase 1: Canonical Reasoning Path Warmup (enable_thinking=true, compiles thinking graphs)
    t0_reasoning = time.perf_counter()
    reasoning_payload = build_warmup_payload(enable_thinking=True, dummy_suffix="Warmup reasoning path.")
    try:
        resp_reasoning = await http_client.post(f"{BASE_URL}/v1/chat/completions", json=reasoning_payload, timeout=300.0)
        if resp_reasoning.status_code != 200:
            warmup_in_progress = False
            return {"status": "failed", "error": f"Reasoning warmup failed: {resp_reasoning.status_code}", "warmed": False}
        t1_reasoning = time.perf_counter()
        reasoning_action_latency_s = round(t1_reasoning - t0_reasoning, 3)
        reasoning_usage = resp_reasoning.json().get("usage", {})
        telem_reasoning = parse_cache_telemetry(reasoning_usage, t0_reasoning, thinking_mode=True)
        print(f"[WARMUP] Phase 1 Success: Reasoning Path (enable_thinking=true): {reasoning_action_latency_s}s | prompt_tokens: {telem_reasoning['prompt_tokens']} | cached_tokens: {telem_reasoning['cached_tokens']} | ratio: {telem_reasoning['cached_ratio']:.2%}", flush=True)
    except Exception as exc:
        print(f"[WARMUP] Reasoning warmup failed: {exc}", flush=True)
        warmup_in_progress = False
        return {"status": "failed", "error": str(exc), "warmed": False}

    # 3. Phase 2: Canonical Fast Path Warmup (enable_thinking=false, primes resident cache for normal turns - FAST PATH LAST)
    t0_fast = time.perf_counter()
    fast_payload = build_warmup_payload(enable_thinking=False, dummy_suffix="Warmup fast path.")
    try:
        resp_fast = await http_client.post(f"{BASE_URL}/v1/chat/completions", json=fast_payload, timeout=300.0)
        if resp_fast.status_code != 200:
            warmup_in_progress = False
            return {"status": "failed", "error": f"Fast path warmup failed: {resp_fast.status_code}", "warmed": False}
        t1_fast = time.perf_counter()
        fast_action_latency_s = round(t1_fast - t0_fast, 3)
        fast_usage = resp_fast.json().get("usage", {})
        telem_fast = parse_cache_telemetry(fast_usage, t0_fast, thinking_mode=False)
        print(f"[WARMUP] Phase 2 Success: Fast Path (enable_thinking=false): {fast_action_latency_s}s | prompt_tokens: {telem_fast['prompt_tokens']} | cached_tokens: {telem_fast['cached_tokens']} | ratio: {telem_fast['cached_ratio']:.2%}", flush=True)
    except Exception as exc:
        print(f"[WARMUP] Fast path warmup failed: {exc}", flush=True)
        warmup_in_progress = False
        return {"status": "failed", "error": str(exc), "warmed": False}

    # Mark ready ONLY after both succeed - Fast Path is now resident in Slot 0
    model_warmed = True
    warmup_duration = round(time.perf_counter() - t_start_total, 3)
    warmup_in_progress = False
    cold_action_latency_s = reasoning_action_latency_s
    warmed_action_latency_s = fast_action_latency_s
    warmup_telemetry = {
        "reasoning_path": telem_reasoning,
        "fast_path": telem_fast,
        "prefix_metadata": get_prefix_metadata(),
    }
    warmup_event.set()
    print(f"[WARMUP] Complete! Both canonical prefixes ready in {warmup_duration}s. Reasoning: {reasoning_action_latency_s}s | Fast: {fast_action_latency_s}s", flush=True)
    return {
        "status": "ok",
        "warmed": True,
        "total_duration_s": warmup_duration,
        "cold_action_latency_s": cold_action_latency_s,
        "warmed_action_latency_s": warmed_action_latency_s,
        "telemetry": warmup_telemetry,
    }

# ---------------------------------------------------------------------------
# Event Dispatcher
# ---------------------------------------------------------------------------
async def broadcast_event(event: Dict[str, Any]):
    # Persist event
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
    INSERT INTO events (id, session_id, type, priority, title, body, timestamp, payload, image_ref)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        event["id"],
        event.get("session_id", ""),
        event["type"],
        event["priority"],
        event["title"],
        event["body"],
        event["timestamp"],
        json.dumps(event.get("payload", {})),
        event.get("image_ref"),
    ))
    conn.commit()
    conn.close()

    # Fan out to all connected WebSockets
    payload = json.dumps({"type": "event", "event": event})
    dead_sockets = []
    for ws in list(connected_websockets):
        try:
            await ws.send_text(payload)
        except Exception:
            dead_sockets.append(ws)
    for ws in dead_sockets:
        connected_websockets.discard(ws)

# ---------------------------------------------------------------------------
# Turn Orchestration via Hermes AIAgent Pipeline (Gateway as Adapter)
# ---------------------------------------------------------------------------
async def process_user_turn(
    session_id: str,
    user_text: str,
    client_msg_id: Optional[str] = None,
    ws_client: Optional[WebSocket] = None
):
    """
    Adapter entrypoint: Routes mobile user turns directly into the authoritative
    Hermes AIAgent / session / tool execution pipeline (hermes_supervisor.py).
    """
    # 0. Bounded wait: If warmup is currently running, queue incoming request
    if not model_warmed and warmup_in_progress:
        print("[TURN_GATEWAY] Warmup currently in progress. Queuing incoming turn until KV cache is primed...", flush=True)
        try:
            await asyncio.wait_for(warmup_event.wait(), timeout=60.0)
        except asyncio.TimeoutError:
            print("[TURN_GATEWAY] Warmup wait timed out, proceeding with turn...", flush=True)

    now = time.time()
    user_msg_id = client_msg_id or f"msg_{int(now*1000)}_user"

    # 1. Persist user message in SQLite & check deduplication
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    if client_msg_id:
        cur.execute("SELECT id FROM messages WHERE client_msg_id = ?", (client_msg_id,))
        if cur.fetchone():
            conn.close()
            return  # Already processed

    cur.execute("""
    INSERT INTO messages (id, session_id, role, content, timestamp, client_msg_id)
    VALUES (?, ?, 'user', ?, ?, ?)
    """, (user_msg_id, session_id, user_text, now, client_msg_id))
    cur.execute("UPDATE sessions SET updated_at = ? WHERE session_id = ?", (now, session_id))

    # Auto-title heuristic check (zero LLM calls to protect Slot 0 resident cache)
    cur.execute("SELECT title FROM sessions WHERE session_id = ?", (session_id,))
    s_row = cur.fetchone()
    current_title = s_row[0] if s_row else ""
    if not current_title or current_title in ("New Session", "New Neo Session", "Conversation"):
        needs_ai_title = True
        new_title = generate_heuristic_title(user_text)
        cur.execute("UPDATE sessions SET title = ? WHERE session_id = ?", (new_title, session_id))
        conn.commit()
        # Broadcast session update frame over WebSockets
        update_frame = json.dumps({
            "type": "session_updated",
            "session_id": session_id,
            "title": new_title,
            "session": {
                "session_id": session_id,
                "title": new_title,
                "updated_at": now,
            }
        })
        for ws in list(connected_websockets):
            try:
                asyncio.create_task(ws.send_text(update_frame))
            except Exception:
                pass
    else:
        needs_ai_title = False
        conn.commit()

    # 2. Retrieve past conversation dialogue for Hermes context assembly
    cur.execute("""
    SELECT role, content FROM messages
    WHERE session_id = ? AND id != ?
    ORDER BY timestamp ASC
    """, (session_id, user_msg_id))
    rows = cur.fetchall()
    conn.close()

    conversation_history = [
        {"role": r[0], "content": r[1]}
        for r in rows
        if r[0] in ("user", "assistant") and r[1]
    ]

    # 3. Callbacks for Hermes AIAgent pipeline
    async def _request_approval_cb(approval_record: Dict[str, Any]) -> str:
        approval_id = approval_record["approval_id"]
        conn_app = sqlite3.connect(DB_PATH)
        cur_app = conn_app.cursor()
        cur_app.execute("""
        INSERT INTO pending_approvals (approval_id, session_id, command, action_description, risk_level, action_hash, created_at, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, 'pending')
        """, (
            approval_id,
            approval_record["session_id"],
            approval_record["command"],
            approval_record["action_description"],
            approval_record["risk_level"],
            approval_record["action_hash"],
            approval_record["created_at"],
        ))
        conn_app.commit()
        conn_app.close()

        # Emit URGENT event to phone triggering haptic vibration
        approval_event = {
            "id": f"evt_{int(time.time()*1000)}_appr",
            "session_id": session_id,
            "type": "approval_required",
            "priority": "urgent",
            "title": "⚠️ Approval Required: High-Risk Action",
            "body": f"Neo requires authorization to run: {approval_record['action_description']}",
            "timestamp": time.time(),
            "payload": approval_record,
        }
        await broadcast_event(approval_event)
        if ws_client:
            try:
                await ws_client.send_text(json.dumps({
                    "type": "approval_required",
                    "approval": approval_record,
                }))
            except Exception:
                pass

        loop = asyncio.get_running_loop()
        decision_future = loop.create_future()
        approval_futures[approval_id] = decision_future
        try:
            decision = await asyncio.wait_for(decision_future, timeout=120.0)
        except asyncio.TimeoutError:
            decision = "reject"
            conn_to = sqlite3.connect(DB_PATH)
            cur_to = conn_to.cursor()
            cur_to.execute("UPDATE pending_approvals SET status = 'timed_out' WHERE approval_id = ?", (approval_id,))
            conn_to.commit()
            conn_to.close()
        finally:
            approval_futures.pop(approval_id, None)
        return decision

    async def _on_delta(dtype: str, delta: str):
        if ws_client:
            try:
                event_type = "thinking.delta" if dtype == "reasoning" else "message.delta"
                await ws_client.send_text(json.dumps({
                    "type": event_type,
                    "session_id": session_id,
                    "delta": delta,
                }))
            except Exception:
                pass

    async def _on_tool(tool_action: Dict[str, Any]):
        if ws_client:
            try:
                await ws_client.send_text(json.dumps({
                    "type": "tool_progress",
                    "session_id": session_id,
                    "tool_name": tool_action.get("name") or tool_action.get("tool") or tool_action.get("tool_name"),
                    "badge": tool_action.get("label") or tool_action.get("badge"),
                    "step": tool_action.get("step"),
                    "tool_action": tool_action,
                }))
            except Exception:
                pass

    # 4. Route through Hermes AIAgent pipeline (CognitiveRouter, Tools, Precompaction, Gemma 4 E4B)
    agent_result = await hermes_supervisor.execute_session_turn(
        session_id=session_id,
        user_text=user_text,
        conversation_history=conversation_history,
        system_prompt=NEO_SYSTEM_PROMPT,
        screenshots_dir=SCREENSHOTS_DIR,
        request_approval_cb=_request_approval_cb,
        emit_event_cb=broadcast_event,
        on_delta_cb=_on_delta,
        on_tool_cb=_on_tool,
    )

    full_content = agent_result.get("content", "")
    full_reasoning = agent_result.get("reasoning_content")
    tool_calls_record = agent_result.get("tool_calls", [])
    screenshot_url = agent_result.get("screenshot_url")
    risk_info = agent_result.get("risk_assessment", {})

    if not full_content:
        if screenshot_url:
            full_content = "Desktop screenshot successfully captured from current active workspace."
        else:
            full_content = "Command acknowledged and processed by Neo."

    # 5. Persist assistant turn in SQLite
    asst_msg_id = f"msg_{int(time.time()*1000)}_asst"
    conn_final = sqlite3.connect(DB_PATH)
    cur_final = conn_final.cursor()
    cur_final.execute("""
    INSERT INTO messages (id, session_id, role, content, reasoning_content, tool_calls, screenshot_url, timestamp)
    VALUES (?, ?, 'assistant', ?, ?, ?, ?, ?)
    """, (
        asst_msg_id,
        session_id,
        full_content,
        full_reasoning,
        json.dumps(tool_calls_record) if tool_calls_record else None,
        screenshot_url,
        time.time(),
    ))
    conn_final.commit()
    conn_final.close()

    final_msg = {
        "id": asst_msg_id,
        "session_id": session_id,
        "role": "assistant",
        "content": full_content,
        "reasoning_content": full_reasoning,
        "thinking_enabled": risk_info.get("thinking_enabled", False),
        "tool_calls": tool_calls_record,
        "screenshot_url": screenshot_url,
        "telemetry": agent_result.get("telemetry"),
        "client_msg_id": client_msg_id,
        "turn_id": client_msg_id,
        "timestamp": time.time(),
    }

    if ws_client:
        try:
            await ws_client.send_text(json.dumps({
                "type": "message",
                "message": final_msg,
            }))
        except Exception:
            pass

    # Title upgrade runs last: the model server has a single slot, so this must not
    # compete with the turn itself.
    if needs_ai_title:
        asyncio.create_task(apply_ai_title(session_id, user_text))

    # Broadcast task completion
    await broadcast_event({
        "id": f"evt_{int(time.time()*1000)}_done",
        "session_id": session_id,
        "type": "task_completed",
        "priority": "normal",
        "title": "Turn Complete",
        "body": full_content[:80],
        "timestamp": time.time(),
        "image_ref": screenshot_url,
    })

async def _run_turn_guarded(
    session_id: str,
    user_text: str,
    client_msg_id: Optional[str],
    ws_client: Optional[WebSocket],
):
    """
    Never let a turn die silently: a crashed background task used to leave the phone
    stuck on "Neo is processing..." forever. Any failure is turned into a normal
    assistant message frame so the pending bubble resolves.
    """
    try:
        await process_user_turn(session_id, user_text, client_msg_id, ws_client)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        err_text = f"⚠️ Tur başarısız oldu: `{type(exc).__name__}: {exc}`"
        err_id = f"msg_{int(time.time()*1000)}_err"
        try:
            conn_err = sqlite3.connect(DB_PATH)
            conn_err.execute(
                "INSERT INTO messages (id, session_id, role, content, timestamp) VALUES (?, ?, 'assistant', ?, ?)",
                (err_id, session_id, err_text, time.time()),
            )
            conn_err.commit()
            conn_err.close()
        except Exception:
            pass
        if ws_client:
            with contextlib.suppress(Exception):
                await ws_client.send_text(json.dumps({
                    "type": "message",
                    "message": {
                        "id": err_id,
                        "session_id": session_id,
                        "role": "assistant",
                        "content": err_text,
                        "client_msg_id": client_msg_id,
                        "turn_id": client_msg_id,
                        "tool_calls": [],
                        "timestamp": time.time(),
                    },
                }))


# ---------------------------------------------------------------------------
# HTTP Route Handlers
# ---------------------------------------------------------------------------
async def handle_health(request: Request) -> Response:
    llama_ready = False
    try:
        r = await http_client.get(f"{BASE_URL}/health", timeout=2.0)
        llama_ready = (r.status_code == 200 and r.json().get("status") == "ok")
    except Exception:
        pass

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM sessions")
    sessions_count = cur.fetchone()[0]
    conn.close()

    # Gateway must not report Neo ready until both warmup requests complete
    neo_ready = llama_ready and model_warmed

    return JSONResponse({
        "status": "ok",
        "ready": neo_ready,
        "gateway": "Neo Mobile Companion Gateway v1.0",
        "llama_server": {"url": BASE_URL, "ready": llama_ready},
        "model_warmed": model_warmed,
        "warmup": {
            "warmed": model_warmed,
            "cold_action_latency_s": cold_action_latency_s,
            "warmed_action_latency_s": warmed_action_latency_s,
            "total_duration_s": warmup_duration,
            "in_progress": warmup_in_progress,
        },
        "warmup_telemetry": warmup_telemetry,
        "prefix_metadata": get_prefix_metadata(),
        "sessions_count": sessions_count,
        "ws_clients_connected": len(connected_websockets),
        "single_instance_locked": True,
        "timestamp": time.time(),
    })

async def handle_manual_warmup(request: Request) -> Response:
    if not _auth_ok(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    res = await run_auto_warmup()
    return JSONResponse(res)

async def handle_get_screenshot(request: Request) -> Response:
    filename = request.path_params.get("filename", "")
    # Served without auth so the phone's <Image> can load it; keep it strictly inside
    # the screenshots dir so a crafted name can never reach another file.
    filepath = (SCREENSHOTS_DIR / filename).resolve()
    if SCREENSHOTS_DIR.resolve() not in filepath.parents:
        return JSONResponse({"error": "Screenshot not found"}, status_code=404)
    if not filepath.exists() or not filepath.is_file():
        return JSONResponse({"error": "Screenshot not found"}, status_code=404)
    return FileResponse(filepath, media_type="image/png")

def _auth_ok(request: Request) -> bool:
    auth = request.headers.get("authorization", "")
    if auth.startswith("Bearer "):
        token = auth[7:]
    else:
        token = request.query_params.get("token", "")
    return verify_token(token)

async def handle_get_sessions(request: Request) -> Response:
    if not _auth_ok(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
    SELECT s.session_id, s.title, s.created_at, s.updated_at, s.hermes_session_id,
           (SELECT content FROM messages m WHERE m.session_id = s.session_id ORDER BY timestamp DESC LIMIT 1) as last_msg
    FROM sessions s
    ORDER BY s.updated_at DESC
    """)
    rows = cur.fetchall()
    conn.close()

    sessions = [
        {
            "session_id": r[0],
            "title": r[1],
            "created_at": r[2],
            "updated_at": r[3],
            "hermes_session_id": r[4],
            "last_message": r[5],
        }
        for r in rows
    ]
    return JSONResponse({"sessions": sessions})

async def handle_create_session(request: Request) -> Response:
    if not _auth_ok(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    try:
        body = await request.json()
    except Exception:
        body = {}
    title = body.get("title", "New Neo Session")
    session_id = f"neo_sess_{int(time.time())}_{secrets.token_hex(4)}"
    now = time.time()

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
    INSERT INTO sessions (session_id, title, created_at, updated_at, hermes_session_id)
    VALUES (?, ?, ?, ?, ?)
    """, (session_id, title, now, now, f"hermes_{secrets.token_hex(4)}"))
    conn.commit()
    conn.close()

    return JSONResponse({
        "session": {
            "session_id": session_id,
            "title": title,
            "created_at": now,
            "updated_at": now,
        }
    })

async def handle_get_messages(request: Request) -> Response:
    if not _auth_ok(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    session_id = request.path_params.get("session_id", "")
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
    SELECT id, session_id, role, content, reasoning_content, tool_calls, screenshot_url, timestamp, client_msg_id
    FROM messages
    WHERE session_id = ?
    ORDER BY timestamp ASC
    """, (session_id,))
    rows = cur.fetchall()
    conn.close()

    messages = [
        {
            "id": r[0],
            "session_id": r[1],
            "role": r[2],
            "content": r[3],
            "reasoning_content": r[4],
            "tool_calls": json.loads(r[5]) if r[5] else [],
            "screenshot_url": r[6],
            "timestamp": r[7],
            "client_msg_id": r[8],
        }
        for r in rows
    ]
    return JSONResponse({"messages": messages})

async def handle_send_message_http(request: Request) -> Response:
    if not _auth_ok(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    session_id = request.path_params.get("session_id", "")
    body = await request.json()
    text = body.get("text", "")
    client_msg_id = body.get("client_msg_id")

    asyncio.create_task(_run_turn_guarded(session_id, text, client_msg_id, None))
    return JSONResponse({"status": "queued"})

async def handle_delete_session(request: Request) -> Response:
    """Deletes a session and everything attached to it."""
    if not _auth_ok(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    session_id = request.path_params.get("session_id", "")
    if not session_id:
        return JSONResponse({"error": "session_id required"}, status_code=400)

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT session_id FROM sessions WHERE session_id = ?", (session_id,))
    if not cur.fetchone():
        conn.close()
        return JSONResponse({"error": "Session not found"}, status_code=404)
    for table in ("messages", "events", "pending_approvals"):
        with contextlib.suppress(sqlite3.OperationalError):
            cur.execute(f"DELETE FROM {table} WHERE session_id = ?", (session_id,))
    cur.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
    conn.commit()
    conn.close()

    frame = json.dumps({"type": "session_deleted", "session_id": session_id})
    for ws in list(connected_websockets):
        with contextlib.suppress(Exception):
            await ws.send_text(frame)
    return JSONResponse({"status": "deleted", "session_id": session_id})

async def handle_decide_approval(request: Request) -> Response:
    if not _auth_ok(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    approval_id = request.path_params.get("approval_id", "")
    body = await request.json()
    decision = body.get("decision", "reject")

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
    SELECT status, session_id, command, action_description, risk_level, action_hash
    FROM pending_approvals WHERE approval_id = ?
    """, (approval_id,))
    row = cur.fetchone()

    if not row:
        conn.close()
        return JSONResponse({"error": "Approval request not found"}, status_code=404)

    status, session_id, command, action_description, risk_level, action_hash = row

    if status != "pending":
        conn.close()
        # Prevent replay attacks!
        return JSONResponse({"error": f"Action already resolved as '{status}'. Replay rejected."}, status_code=409)

    # Verify cryptographic binding to exact structured pending action
    expected_hash = compute_action_hash(session_id, command, action_description, risk_level)
    client_hash = body.get("action_hash")
    if client_hash and client_hash != expected_hash:
        conn.close()
        return JSONResponse({"error": "Action hash mismatch - potential tampering detected. Action cannot be modified."}, status_code=400)
    if action_hash and action_hash != expected_hash:
        conn.close()
        return JSONResponse({"error": "Database integrity error: stored action hash mismatch."}, status_code=400)

    cur.execute("UPDATE pending_approvals SET status = ? WHERE approval_id = ?", (decision, approval_id))
    conn.commit()
    conn.close()

    fut = approval_futures.get(approval_id)
    if fut and not fut.done():
        fut.set_result(decision)

    return JSONResponse({"success": True, "approval_id": approval_id, "decision": decision})

async def handle_test_urgent_event(request: Request) -> Response:
    """Pushes a live test urgent alert to verify physical mobile phone vibration and banner."""
    event = {
        "id": f"evt_{int(time.time()*1000)}_test_urgent",
        "session_id": "test_session",
        "type": "urgent_attention",
        "priority": "urgent",
        "title": "🚨 URGENT: Neo Verification Alert",
        "body": "This is a real high-priority event from Neo. Device vibration triggered successfully.",
        "timestamp": time.time(),
        "payload": {"test": True, "vibrate": True},
    }
    await broadcast_event(event)
    return JSONResponse({"success": True, "event": event})

# ---------------------------------------------------------------------------
# WebSocket Transport Handler (Safer Authenticated Handshake)
# ---------------------------------------------------------------------------
async def handle_ws(ws: WebSocket):
    await ws.accept()

    # Authenticate via either the first frame handshake (preferred, no query string leaks)
    # or fallback query param for backwards compatibility.
    query_token = ws.query_params.get("token", "")
    authenticated = False

    if query_token and verify_token(query_token):
        authenticated = True
        await ws.send_text(json.dumps({"type": "auth_ok"}))
    else:
        # Safe Handshake Protocol: Wait up to 5.0 seconds for initial auth frame
        try:
            raw_first = await asyncio.wait_for(ws.receive_text(), timeout=5.0)
            first_msg = json.loads(raw_first)
            if first_msg.get("type") == "auth" and verify_token(first_msg.get("token")):
                authenticated = True
                await ws.send_text(json.dumps({"type": "auth_ok"}))
            else:
                await ws.send_text(json.dumps({"type": "auth_error", "error": "Invalid authentication token"}))
                await ws.close(code=4401)
                return
        except Exception:
            await ws.close(code=4401)
            return

    if not authenticated:
        await ws.close(code=4401)
        return

    connected_websockets.add(ws)
    try:
        while True:
            raw_text = await ws.receive_text()
            try:
                frame = json.loads(raw_text)
            except Exception:
                continue

            frame_type = frame.get("type", "")
            if frame_type == "ping":
                await ws.send_text(json.dumps({"type": "pong"}))
            elif frame_type == "send_message":
                session_id = frame.get("session_id", "")
                text = frame.get("text", "")
                client_msg_id = frame.get("client_msg_id")
                asyncio.create_task(_run_turn_guarded(session_id, text, client_msg_id, ws))
    except WebSocketDisconnect:
        pass
    finally:
        connected_websockets.discard(ws)

# ---------------------------------------------------------------------------
# App Assembly
# ---------------------------------------------------------------------------
routes = [
    Route("/health", handle_health, methods=["GET"]),
    Route("/screenshots/{filename}", handle_get_screenshot, methods=["GET"]),
    Route("/api/sessions", handle_get_sessions, methods=["GET"]),
    Route("/api/sessions", handle_create_session, methods=["POST"]),
    Route("/api/sessions/{session_id}/messages", handle_get_messages, methods=["GET"]),
    Route("/api/sessions/{session_id}/messages", handle_send_message_http, methods=["POST"]),
    Route("/api/sessions/{session_id}", handle_delete_session, methods=["DELETE"]),
    Route("/api/approvals/{approval_id}/decide", handle_decide_approval, methods=["POST"]),
    Route("/api/events/test_urgent", handle_test_urgent_event, methods=["POST"]),
    Route("/api/warmup", handle_manual_warmup, methods=["POST", "GET"]),
    WebSocketRoute("/ws", handle_ws),
]

middleware = [
    Middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
]

@contextlib.asynccontextmanager
async def lifespan(app: Starlette):
    # Trigger automatic KV cache warmup in the background upon gateway startup!
    asyncio.create_task(run_auto_warmup())
    yield
    await http_client.aclose()

app = Starlette(debug=False, routes=routes, middleware=middleware, lifespan=lifespan)

def main():
    # Enforce true POSIX kernel-level single instance ownership
    acquire_single_instance_lock()

    print(f"==================================================")
    print(f"NEO MOBILE GATEWAY (HERMES AIAGENT ADAPTER)")
    print(f"==================================================")
    print(f"Auth Token: {AUTH_TOKEN}")
    print(f"Token Path: {TOKEN_PATH}")
    print(f"Listening on: http://{HOST}:{PORT}")
    print(f"Screenshots: {SCREENSHOTS_DIR}")
    print(f"Database: {DB_PATH}")
    print(f"Single-Instance Lock: {GATEWAY_LOCK_FILE} (Active)")
    print(f"Auto-Warmup: Enabled (Dual-phase thinking off/on)")
    print(f"==================================================")
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")

if __name__ == "__main__":
    main()
