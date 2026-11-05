"""
Hermes Supervisor Engine
========================
PC & Browser Agent Supervisor powered by Gemma 4 E4B-it.
Features:
- Adaptive Cognitive Gating (Low risk -> Thinking OFF; Medium/High -> Thinking ON)
- Zero Artificial Truncation (No token limit cap)
- Multimodal Vision Support (Screenshots, accessibility tree, DOM snippets)
- Native Tool Calling & Destructive Safety Gating
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import re
import secrets
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import httpx

try:
    from hermes.config import BASE_URL
    from hermes.cognitive_router import CognitiveRouter, RiskAssessment
    from hermes import agent_bridge, wayland_ui
    from hermes.prefix_builder import (
        CANONICAL_HERMES_TOOLS as HERMES_TOOLS,
        get_canonical_tools,
        get_canonical_stable_prefix,
        build_dynamic_user_content,
        parse_cache_telemetry,
        get_prefix_metadata,
    )
except ModuleNotFoundError:
    from config import BASE_URL
    from cognitive_router import CognitiveRouter, RiskAssessment
    import agent_bridge
    import wayland_ui
    from prefix_builder import (
        CANONICAL_HERMES_TOOLS as HERMES_TOOLS,
        get_canonical_tools,
        get_canonical_stable_prefix,
        build_dynamic_user_content,
        parse_cache_telemetry,
        get_prefix_metadata,
    )

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Precompaction constants: Trigger at ~60K tokens (preserving latest 5 pairs verbatim)
COMPACTION_TOKEN_THRESHOLD = 60000
VERBATIM_PAIRS_COUNT = 5  # 5 user/assistant pairs = 10 messages

def estimate_tokens(item: Union[str, List[Dict[str, Any]], Dict[str, Any]]) -> int:
    """Estimates tokens for Gemma/E4B (~3.6 chars per token average). Supports str, message dict, or list."""
    if not item:
        return 0
    if isinstance(item, str):
        return max(1, int(len(item) / 3.6))
    if isinstance(item, dict):
        text = str(item.get("content") or "") + str(item.get("reasoning_content") or "")
        return max(1, int(len(text) / 3.6))
    if isinstance(item, (list, tuple)):
        return sum(estimate_tokens(m) for m in item)
    return max(1, int(len(str(item)) / 3.6))

def compute_action_hash(session_id: str, command: str, action_description: str, risk_level: str) -> str:
    """Cryptographically binds approval parameters into an immutable SHA-256 digest."""
    canonical = json.dumps({
        "session_id": session_id,
        "command": command,
        "action_description": action_description,
        "risk_level": risk_level,
    }, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

def compact_history_if_needed(messages: List[Dict[str, Any]], max_tokens: int = COMPACTION_TOKEN_THRESHOLD) -> Tuple[List[Dict[str, Any]], bool]:
    """
    Evaluates conversation token count.
    If total tokens > max_tokens (~60K), compresses older history preceding the latest 5 user/assistant pairs.
    Preserves the latest 5 user/assistant pairs VERBATIM.
    """
    total_tokens = sum(estimate_tokens(str(m.get("content", "")) + str(m.get("reasoning_content", ""))) for m in messages)
    if total_tokens <= max_tokens:
        return messages, False

    # Separate system messages, older history, and latest verbatim pairs
    system_msgs = [m for m in messages if m.get("role") == "system"]
    non_system = [m for m in messages if m.get("role") != "system"]

    verbatim_msg_count = VERBATIM_PAIRS_COUNT * 2
    if len(non_system) <= verbatim_msg_count:
        return messages, False

    older = non_system[:-verbatim_msg_count]
    recent_verbatim = non_system[-verbatim_msg_count:]

    # Construct concise structured summary of the older turns
    summary_lines = []
    for idx, msg in enumerate(older):
        role = msg.get("role", "unknown")
        content = (msg.get("content") or "").strip()
        tc = msg.get("tool_calls")
        tc_info = f" [Tools: {','.join(t.get('name', '') for t in tc)}]" if tc else ""
        snippet = content[:150] + ("..." if len(content) > 150 else "")
        summary_lines.append(f"- Turn {idx+1} ({role}){tc_info}: {snippet}")

    summary_block = (
        "[PRECOMPACTED SESSION HISTORY SUMMARY]\n"
        f"Older dialogue ({len(older)} turns) condensed to preserve 128K context capacity:\n"
        + "\n".join(summary_lines)
    )

    compacted = [
        *system_msgs,
        {"role": "system", "content": summary_block},
        *recent_verbatim,
    ]
    return compacted, True


# HERMES_TOOLS is canonically imported from prefix_builder above

CANONICAL_HERMES_TOOL_NAMES = {
    "wayland_focus_or_launch",
    "wayland_trigger_shortcut",
    "wayland_list_windows",
    "inspect_accessibility_tree",
    "capture_screenshot",
    "execute_terminal_command",
    "ui_type_text",
    "hermes_skill",
    "ui_press_key",
    "ui_click",
    "send_chat_message",
    "notify_user",
    "ask_approval",
    "route_coding_agent",
}

def _encode_screenshot_as_data_uri(screenshots_dir: Path, shot_url: str) -> Optional[str]:
    """Turns a '/screenshots/<file>.png' ref into a base64 data URI for vision input."""
    # ponytail: full-resolution frame sent as-is; downscale here if CPU mmproj encoding gets slow.
    try:
        path = screenshots_dir / Path(shot_url).name
        if not path.is_file():
            return None
        raw = path.read_bytes()
        if not raw:
            return None
        return "data:image/png;base64," + base64.b64encode(raw).decode("ascii")
    except Exception as exc:
        print(f"[VISION] Screenshot encode failed: {exc}", flush=True)
        return None


def _png_size(path: Path) -> Optional[Tuple[int, int]]:
    """Reads width/height straight from the PNG IHDR header (no image library needed)."""
    try:
        with open(path, "rb") as fh:
            head = fh.read(24)
        if len(head) < 24 or head[12:16] != b"IHDR":
            return None
        return int.from_bytes(head[16:20], "big"), int.from_bytes(head[20:24], "big")
    except Exception:
        return None


def get_tool_public_label(tool_name: str, args: Dict[str, Any]) -> str:
    """Provides user-friendly Turkish progress text for mobile UI without leaking internal details."""
    if tool_name == "wayland_list_windows":
        return "Masaüstü pencereleri kontrol ediliyor..."
    elif tool_name == "wayland_focus_or_launch":
        app = args.get("app_name", "uygulama")
        return f"{app.capitalize()} açılıyor / odaklanılıyor..."
    elif tool_name == "wayland_trigger_shortcut":
        act = args.get("action", "")
        return f"Masaüstü kısayolu tetikleniyor ({act})..."
    elif tool_name == "capture_screenshot":
        return ("Tüm masaüstü görüntüsü alınıyor..." if args.get("scope") == "full"
                else "Aktif pencere görüntüsü alınıyor...")
    elif tool_name == "execute_terminal_command":
        cmd = (args.get("command") or "")[:35]
        return f"Terminal komutu yürütülüyor: {cmd}..."
    elif tool_name == "route_coding_agent":
        ag = args.get("agent", "kodlama ajanı")
        return f"{ag.capitalize()} kodlama ajanına delege ediliyor..."
    elif tool_name == "ui_type_text":
        txt = (args.get("text") or "")[:40]
        return f"Metin yazılıyor: \"{txt}\"..."
    elif tool_name == "ui_press_key":
        mods = "+".join(args.get("modifiers") or [])
        combo = f"{mods}+{args.get('key', '')}" if mods else str(args.get("key", ""))
        return f"Tuşa basılıyor ({combo})..."
    elif tool_name == "ui_click":
        if args.get("text"):
            return f"Ekranda aranıp tıklanıyor: \"{args['text']}\"..."
        if args.get("mark") is not None:
            return f"Tıklanıyor (öğe {args['mark']})..."
        return f"Tıklanıyor ({args.get('x')},{args.get('y')})..."

    elif tool_name == "send_chat_message":
        return f"{args.get('chat')} sohbetine mesaj gönderiliyor..."
    elif tool_name == "notify_user":
        return f"Telefona bildirim gonderiliyor ({args.get('priority', 'normal')})..."
    elif tool_name == "hermes_skill":
        if args.get("name"):
            return f"Hermes yeteneği çalıştırılıyor: {args['name']}..."
        return f"Hermes yetenekleri aranıyor: {args.get('query') or 'tümü'}..."
    elif tool_name == "inspect_accessibility_tree":
        app = args.get("app_name")
        return f"Arayüz erişilebilirlik ağacı taranıyor ({app or 'genel'})..."
    return f"İşlem yürütülüyor: {tool_name}..."

def parse_and_strip_tool_call(
    tool_call_accumulator: Dict[str, Any],
    raw_content: str,
) -> Tuple[Optional[str], Dict[str, Any], str]:
    """
    Priority Hierarchy:
    A) Native OpenAI tool_calls (streaming delta accumulator)
    B) <tool_call>{...}</tool_call> Fallback
    C) Strict JSON Fallback (markdown code block or raw JSON object)
    Returns: (tool_name, parsed_args, sanitized_content)
    """
    clean_content = raw_content

    # A. Native OpenAI tool_calls
    acc_name = tool_call_accumulator.get("name", "")
    for known in CANONICAL_HERMES_TOOL_NAMES:
        if known in acc_name:
            acc_name = known
            break
    if acc_name in CANONICAL_HERMES_TOOL_NAMES:
        parsed_args: Dict[str, Any] = {}
        raw_args = tool_call_accumulator.get("arguments", "")
        if raw_args:
            try:
                parsed_args = json.loads(raw_args)
            except Exception:
                pass
        return acc_name, parsed_args, clean_content.strip()

    # B. <tool_call>{...}</tool_call> Fallback
    tc_match = re.search(r"<tool_call>[\s\r\n]*({.*?})[\s\r\n]*</tool_call>", raw_content, re.DOTALL)
    if tc_match:
        try:
            raw_data = json.loads(tc_match.group(1))
            cand = raw_data.get("name") or raw_data.get("tool_name")
            if cand in CANONICAL_HERMES_TOOL_NAMES:
                args = raw_data.get("arguments") or raw_data.get("params") or {}
                clean_content = raw_content[:tc_match.start()] + raw_content[tc_match.end():]
                return cand, args, clean_content.strip()
        except Exception:
            pass

    # C. Strict JSON Fallback
    # Check ```json ... ``` or ``` ... ```
    json_blocks = list(re.finditer(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", raw_content))
    for block in json_blocks:
        try:
            raw_data = json.loads(block.group(1))
            cand = raw_data.get("tool_name") or raw_data.get("name")
            if cand in CANONICAL_HERMES_TOOL_NAMES:
                args = raw_data.get("params") or raw_data.get("arguments") or raw_data.get("parameters") or {}
                clean_content = raw_content[:block.start()] + raw_content[block.end():]
                return cand, args, clean_content.strip()
        except Exception:
            pass

    # Raw JSON object fallback containing tool_name or name
    raw_obj_matches = list(re.finditer(r"(\{[\s\r\n]*\"(?:tool_name|name)\"[\s\r\n]*:[\s\S]*?\})", raw_content))
    for obj_m in raw_obj_matches:
        try:
            raw_data = json.loads(obj_m.group(1))
            cand = raw_data.get("tool_name") or raw_data.get("name")
            if cand in CANONICAL_HERMES_TOOL_NAMES:
                args = raw_data.get("params") or raw_data.get("arguments") or raw_data.get("parameters") or {}
                clean_content = raw_content[:obj_m.start()] + raw_content[obj_m.end():]
                return cand, args, clean_content.strip()
        except Exception:
            pass

    return None, {}, clean_content.strip()


class HermesSupervisor:
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url.rstrip("/")
        self.router = CognitiveRouter()
        self.client = httpx.Client(timeout=300.0)

    def dispatch(
        self,
        prompt: str,
        screenshot_path: Optional[Union[str, Path]] = None,
        terminal_output: Optional[str] = None,
        dom_snippet: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Main execution method for Hermes. Evaluates risk, sets cognitive mode, and invokes Gemma 4.
        """
        # Step 1: Pre-action Risk Assessment
        risk: RiskAssessment = self.router.evaluate_risk(
            prompt=prompt,
            terminal_context=terminal_output,
            dom_context=dom_snippet,
        )

        # Step 2: Assemble Context & Messages (Stable Canonical Prefix First, Volatile Tail Second)
        sys_msg = system_prompt or get_canonical_stable_prefix()
        messages: List[Dict[str, Any]] = [{"role": "system", "content": sys_msg}]

        # Dynamic suffix assembly: text first, observations strictly at the tail
        user_turn_content = build_dynamic_user_content(
            user_text=prompt,
            screenshot_path=screenshot_path,
            terminal_output=terminal_output,
            dom_snippet=dom_snippet,
        )
        messages.append({"role": "user", "content": user_turn_content})

        # Step 3: Configure Payload (Canonical Tools, Exact Match, Stream Usage Telemetry)
        payload: Dict[str, Any] = {
            "model": "default",
            "messages": messages,
            "tools": tools or get_canonical_tools(),
            "tool_choice": "auto",
            "temperature": 0.1 if not risk.thinking_enabled else 0.4,
            "chat_template_kwargs": {
                "enable_thinking": risk.thinking_enabled
            },
            "max_tokens": 4096,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        # Step 4: Streaming Execution & Telemetry
        t0 = time.perf_counter()
        t_first_token: Optional[float] = None
        first_tool_call_time: Optional[float] = None
        usage_info: Optional[Dict[str, Any]] = None

        chunks: List[str] = []
        reasoning_chunks: List[str] = []
        tool_call_accumulator: Dict[str, Any] = {}
        tokens_emitted = 0

        with self.client.stream("POST", f"{self.base_url}/v1/chat/completions", json=payload) as resp:
            if resp.status_code != 200:
                return {
                    "success": False,
                    "error": f"HTTP {resp.status_code}: {resp.read().decode('utf-8')[:300]}",
                    "risk_assessment": risk.__dict__,
                }

            for line in resp.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[6:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    delta_json = json.loads(data_str)
                    now = time.perf_counter()
                    if t_first_token is None:
                        t_first_token = now

                    choices = delta_json.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        if "reasoning_content" in delta and delta["reasoning_content"]:
                            reasoning_chunks.append(delta["reasoning_content"])
                            tokens_emitted += 1
                        if "content" in delta and delta["content"]:
                            chunks.append(delta["content"])
                            tokens_emitted += 1
                        if "tool_calls" in delta and delta["tool_calls"]:
                            if first_tool_call_time is None:
                                first_tool_call_time = now
                            tc = delta["tool_calls"][0]
                            if "function" in tc:
                                fn = tc["function"]
                                if "name" in fn and fn["name"]:
                                    curr = tool_call_accumulator.get("name", "")
                                    incoming = fn["name"]
                                    if not curr:
                                        tool_call_accumulator["name"] = incoming
                                    elif curr != incoming and not incoming.startswith(curr):
                                        tool_call_accumulator["name"] = curr + incoming
                                if "arguments" in fn and fn["arguments"]:
                                    tool_call_accumulator["arguments"] = (
                                        tool_call_accumulator.get("arguments", "") + fn["arguments"]
                                    )
                            tokens_emitted += 1
                    if "usage" in delta_json and delta_json["usage"]:
                        usage_info = delta_json["usage"]
                except Exception:
                    pass

        t_end = time.perf_counter()
        total_wall_s = t_end - t0
        ttft_ms = ((t_first_token - t0) * 1000.0) if t_first_token else 0.0
        action_latency_ms = ((first_tool_call_time - t0) * 1000.0) if first_tool_call_time else (total_wall_s * 1000.0)
        decode_wall_s = (t_end - t_first_token) if (t_end and t_first_token) else total_wall_s
        decode_tps = (tokens_emitted / decode_wall_s) if decode_wall_s > 0 else 0.0

        cache_telem = parse_cache_telemetry(
            usage=usage_info,
            start_time=t0,
            ttft=(t_first_token - t0) if t_first_token else None,
            thinking_mode=risk.thinking_enabled,
            tool_decision_latency=(first_tool_call_time - t0) if first_tool_call_time else None,
        )

        # Parse arguments
        parsed_args = {}
        if tool_call_accumulator.get("arguments"):
            try:
                parsed_args = json.loads(tool_call_accumulator["arguments"])
            except Exception:
                pass

        # Fallback regex parser for inline <tool_call>
        tool_name = tool_call_accumulator.get("name", "")
        for known in [
            "wayland_focus_or_launch", "wayland_trigger_shortcut", "wayland_list_windows",
            "inspect_accessibility_tree", "capture_screenshot", "execute_terminal_command",
            "ask_approval", "route_coding_agent"
        ]:
            if known in tool_name:
                tool_name = known
                break
        if not tool_name:
            full_text = "".join(chunks)
            match = re.search(r"<tool_call>[\s\r\n]*({.*?})[\s\r\n]*</tool_call>", full_text, re.DOTALL)
            if match:
                try:
                    raw_data = json.loads(match.group(1))
                    tool_name = raw_data.get("name", "")
                    parsed_args = raw_data.get("arguments", {})
                except Exception:
                    pass

        return {
            "success": True,
            "risk_assessment": risk.__dict__,
            "tool_call": {
                "name": tool_name,
                "arguments": parsed_args,
            } if tool_name else None,
            "content": "".join(chunks),
            "reasoning_content": "".join(reasoning_chunks) if risk.thinking_enabled else None,
            "telemetry": {
                "ttft_ms": round(ttft_ms, 2),
                "action_latency_ms": round(action_latency_ms, 2),
                "total_wall_s": round(total_wall_s, 3),
                "decode_tps": round(decode_tps, 2),
                "tokens_emitted": tokens_emitted,
                **cache_telem,
            }
        }

    async def _confirm_sent(
        self,
        screenshots_dir: Path,
        turn_state: Dict[str, Any],
        used_modifier: bool,
    ) -> Tuple[bool, str]:
        """True once the draft has left the input box, retrying with the other Enter combo.

        A model that presses Enter and reports success is the failure this exists to stop:
        the text sat in the box while the turn declared the message delivered.
        """
        draft = (turn_state or {}).get("draft_text") or ""
        probe = " ".join(draft.split()[:4])
        if not probe:
            return True, ""

        for attempt in range(2):
            time.sleep(0.7)
            url, _ = self._grab_frame(screenshots_dir, "active_window", turn_state)
            if not url:
                return True, ""  # cannot see; do not block on a blind guess
            still_there = wayland_ui.band_texts(
                str(screenshots_dir / Path(url).name), 0.14, "bottom")
            folded = wayland_ui._fold_loose(probe)
            if not any(folded in wayland_ui._fold_loose(line) for line in still_there):
                return True, ""
            if attempt == 0:
                # Enter inserted a newline, so undo it and try the other combination.
                wayland_ui.simulate_key("BackSpace")
                if used_modifier:
                    wayland_ui.simulate_key("Return")
                else:
                    wayland_ui.simulate_key("Return", ["ctrl"])
        return False, ("Hem Return hem ctrl+Return denendi, metin kutudan cikmadi. "
                       "Dogru pencerede ve yazi kutusunda oldugundan emin ol.")

    async def send_chat_message(
        self,
        app: str,
        chat: str,
        text: str,
        screenshots_dir: Path,
        turn_state: Optional[Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], Optional[str]]:
        """Open a conversation and send one message, verifying every step.

        Exists because the model cannot hold the sequence together. Given the pieces --
        click the row, check the header, type, check the box, press Enter, check it left --
        it re-clicked the same row five times, typed into a side panel, and announced
        delivery of a message still sitting in the input box. The steps are reliable; the
        planning is not, so the planning lives here.
        """
        def frame() -> Tuple[Optional[str], Optional[Path]]:
            url, _ = self._grab_frame(screenshots_dir, "active_window", turn_state)
            return url, (screenshots_dir / Path(url).name) if url else None

        launch = wayland_ui.focus_or_launch(app)
        if launch.get("error"):
            return {"error": f"'{app}' acilamadi: {launch['error']}"}, None
        time.sleep(1.2)

        url, png = frame()
        if not png:
            return {"error": "Ekran goruntusu alinamadi."}, None
        hits = wayland_ui.find_text(str(png), chat)
        if not hits:
            return ({"error": f"'{chat}' sohbeti ekranda bulunamadi. Ad dogru mu, sohbet "
                              f"listede gorunuyor mu?"}, url)
        wayland_ui.click_at(*[c + o for c, o in
                              zip((hits[0]["x"], hits[0]["y"]),
                                  (turn_state or {}).get("shot_origin", (0, 0)))])
        time.sleep(1.2)

        url, png = frame()
        header = wayland_ui.band_texts(str(png), 0.14, "top") if png else []
        folded = wayland_ui._fold_loose(chat)
        if header and not any(folded in wayland_ui._fold_loose(h) for h in header):
            return ({"error": f"'{chat}' acilamadi: ustte bu isim gorunmuyor.",
                     "screen_header": header[:6]}, url)

        wayland_ui.clear_focused_field()
        if not wayland_ui.simulate_typing(text):
            return {"error": "Metin yazilamadi (ydotoold/wl-copy calisiyor mu?)."}, url
        time.sleep(0.5)

        probe = " ".join(text.split()[:4])
        url, png = frame()
        box = wayland_ui.band_texts(str(png), 0.14, "bottom") if png else []
        if box and not any(wayland_ui._fold_loose(probe) in wayland_ui._fold_loose(b)
                           for b in box):
            return ({"error": "Metin yazi kutusuna dusmedi; klavye odagi baska yerde.",
                     "input_area": box[:4]}, url)

        for combo in (([], "Return"), (["ctrl"], "Return")):
            wayland_ui.simulate_key(combo[1], combo[0] or None)
            time.sleep(0.9)
            url, png = frame()
            box = wayland_ui.band_texts(str(png), 0.14, "bottom") if png else []
            gone = not box or not any(
                wayland_ui._fold_loose(probe) in wayland_ui._fold_loose(b) for b in box)
            if gone:
                if turn_state is not None:
                    turn_state.pop("draft_pending", None)
                return ({"sent": True, "chat": chat, "text": text,
                         "note": "Gonderildi ve yazi kutusunun bosaldigi dogrulandi."}, url)
            wayland_ui.simulate_key("BackSpace")  # Enter only inserted a newline

        return ({"error": "Mesaj gonderilemedi: Return ve ctrl+Return denendi, metin kutuda "
                          "kaldi.", "chat": chat}, url)

    def _grab_frame(
        self,
        screenshots_dir: Path,
        scope: str,
        turn_state: Optional[Dict],
    ) -> Tuple[Optional[str], Tuple[int, int]]:
        """Takes a frame and records its screen origin/size on the turn, so that
        ui_click can translate coordinates read off THIS image back to the screen."""
        origin, size = (0, 0), None
        if scope == "active_window":
            win = wayland_ui.get_active_window() or {}
            if win.get("at") and win.get("size"):
                origin, size = tuple(win["at"]), tuple(win["size"])
        url = self.capture_screenshot(screenshots_dir, scope)
        if not url:
            return None, (0, 0)
        if size is None:
            size = _png_size(screenshots_dir / Path(url).name) or (0, 0)
        if turn_state is not None:
            turn_state["shot_origin"] = origin
            turn_state["shot_size"] = size
        return url, size

    @staticmethod
    def capture_screenshot(destination_dir: Path, scope: str = "active_window") -> Optional[str]:
        """
        Captures a native Wayland frame using grim or the CUA driver.

        scope="active_window" crops to the focused window. The model sees images
        downscaled to roughly 896x896, so a full 1080p desktop shot loses the small
        text that matters for UI work (chat titles, input fields). Cropping to the
        window keeps that text legible.
        """
        destination_dir.mkdir(parents=True, exist_ok=True)
        screenshot_id = f"snap_{int(time.time()*1000)}_{secrets.token_hex(4)}"
        filename = f"{screenshot_id}.png"
        filepath = destination_dir / filename

        grim_cmd = ["/usr/bin/grim"]
        if scope == "active_window":
            win = wayland_ui.get_active_window() or {}
            at, size = win.get("at"), win.get("size")
            if at and size and size[0] > 0 and size[1] > 0:
                grim_cmd += ["-g", f"{at[0]},{at[1]} {size[0]}x{size[1]}"]
        grim_cmd.append(str(filepath))

        try:
            proc = subprocess.run(grim_cmd, capture_output=True, timeout=5)
            if proc.returncode == 0 and filepath.exists() and filepath.stat().st_size > 0:
                return f"/screenshots/{filename}"
        except Exception:
            pass

        try:
            cua_bin = Path.home() / ".local" / "bin" / "cua-driver"
            if cua_bin.exists():
                proc = subprocess.run([str(cua_bin), "--screenshot", str(filepath)], capture_output=True, timeout=5)
                if proc.returncode == 0 and filepath.exists() and filepath.stat().st_size > 0:
                    return f"/screenshots/{filename}"
        except Exception:
            pass

        return None

    @staticmethod
    async def execute_terminal_command(
        command: str,
        cwd: Optional[str] = None,
        timeout: int = 60,
    ) -> Dict[str, Any]:
        """Executes a command-oriented terminal task (git, docker, systemctl, etc.)."""
        loop = asyncio.get_running_loop()
        target_cwd = cwd or str(PROJECT_ROOT)

        def _run():
            try:
                proc = subprocess.run(
                    command,
                    shell=True,
                    cwd=target_cwd,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
                return {
                    "success": proc.returncode == 0,
                    "returncode": proc.returncode,
                    "stdout": proc.stdout.strip(),
                    "stderr": proc.stderr.strip(),
                }
            except subprocess.TimeoutExpired:
                return {"success": False, "error": f"Command timed out after {timeout}s"}
            except Exception as e:
                return {"success": False, "error": str(e)}

        return await loop.run_in_executor(None, _run)



    @staticmethod
    async def delegate_agent(
        agent: str,
        task: str,
        working_dir: Optional[str] = None,
        resume: bool = False,
    ) -> str:
        """
        Executes real delegation to Codex, Claude Code, or Antigravity CLI:
        - Navigates to target project directory (cwd).
        - Supports resume flags: agy -c, claude -c, codex resume --last.
        - Actively supervises Antigravity with a 2-second check loop until completion.
        - Runs Claude in auto-mode and Codex with exec/resume.
        """
        loop = asyncio.get_running_loop()
        cwd = working_dir or str(PROJECT_ROOT)
        env = dict(os.environ)
        home_bin = str(Path.home() / ".local" / "bin")
        npm_bin = str(Path.home() / ".npm-global" / "bin")
        env["PATH"] = f"{home_bin}:{npm_bin}:{env.get('PATH', '')}"

        agy_bin = str(Path.home() / ".local" / "bin" / "agy")
        claude_bin = str(Path.home() / ".npm-global" / "bin" / "claude")
        codex_bin = "/usr/bin/codex"

        if agent == "antigravity":
            bin_path = agy_bin if os.path.exists(agy_bin) else "agy"
            if resume:
                cmd = [bin_path, "-c", task] if task else [bin_path, "-c"]
            else:
                cmd = [bin_path, "-p", task]
        elif agent == "claude_code":
            bin_path = claude_bin if os.path.exists(claude_bin) else "claude"
            if resume:
                cmd = [bin_path, "-c", task] if task else [bin_path, "-c"]
            else:
                cmd = [bin_path, "-p", task]
        elif agent == "codex":
            bin_path = codex_bin if os.path.exists(codex_bin) else "codex"
            if resume:
                cmd = [bin_path, "resume", "--last"]
            else:
                cmd = [bin_path, "exec", "--skip-git-repo-check", task]
        else:
            return f"Unknown agent: {agent}"

        def _execute_supervised():
            try:
                proc = subprocess.Popen(
                    cmd,
                    cwd=cwd,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )

                # Antigravity requires active supervision every 2 seconds
                poll_interval = 2.0
                while proc.poll() is None:
                    time.sleep(poll_interval)

                stdout, stderr = proc.communicate(timeout=10)
                if proc.returncode == 0:
                    return stdout.strip() or "Task completed successfully with no terminal output."
                return f"Agent {agent} exited with code {proc.returncode}.\nStdout: {stdout[:500]}\nStderr: {stderr[:500]}"
            except subprocess.TimeoutExpired:
                proc.kill()
                return f"Agent {agent} timed out."
            except Exception as e:
                return f"Delegation error: {e}"

        return await loop.run_in_executor(None, _execute_supervised)

    async def execute_tool(
        self,
        tool_name: str,
        parsed_args: Dict[str, Any],
        session_id: str,
        user_text: str,
        screenshots_dir: Path,
        emit_event_cb: Callable[[Dict[str, Any]], Any],
        request_approval_cb: Optional[Callable[[Dict[str, Any]], Any]] = None,
        turn_state: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Any, Optional[str], str]:
        """
        Executes a canonical Hermes tool safely.
        Returns: (result_payload, screenshot_url, label)
        """
        screenshot_url = None
        try:
            if tool_name == "wayland_list_windows":
                windows = wayland_ui.list_windows()
                label = f"Pencereler kontrol edildi ({len(windows)} açık)"
                return {"count": len(windows), "windows": windows}, None, label

            elif tool_name == "wayland_focus_or_launch":
                app_name = parsed_args.get("app_name", "")
                await emit_event_cb({
                    "id": f"evt_{int(time.time()*1000)}_ui_focus",
                    "session_id": session_id,
                    "type": "task_progress",
                    "priority": "low",
                    "title": "Desktop UI Action",
                    "body": f"Focusing or launching '{app_name}' in Wayland UI...",
                    "timestamp": time.time(),
                })
                ui_res = wayland_ui.focus_or_launch(app_name)
                status_desc = "mevcut pencereye odaklandı" if ui_res.get("action") == "focused_existing" else "başlatıcı üzerinden açıldı"
                ws_info = f" (Workspace {ui_res.get('workspace')})" if "workspace" in ui_res else ""
                label = f"{app_name.capitalize()}: {status_desc}{ws_info}"
                return ui_res, None, label

            elif tool_name == "wayland_trigger_shortcut":
                action = parsed_args.get("action", "")
                sc_res = wayland_ui.trigger_shortcut(action)
                label = f"Kısayol: {sc_res.get('description', action)}"
                return sc_res, None, label

            elif tool_name == "inspect_accessibility_tree":
                app_target = parsed_args.get("app_name")
                tree = wayland_ui.inspect_accessibility_tree(app_target)
                label = f"AT-SPI Ağacı: {tree.get('count', 0)} uygulama"
                return tree, None, label

            elif tool_name == "capture_screenshot":
                scope = parsed_args.get("scope") or "active_window"
                screenshot_url, size = self._grab_frame(screenshots_dir, scope, turn_state)
                if not screenshot_url:
                    return {"error": "capture failed"}, None, "Ekran görüntüsü alınamadı"
                if parsed_args.get("mark_elements"):
                    # Set-of-Mark: the encoder squeezes this frame to 224x224, so the model
                    # cannot name a pixel. Numbering the elements turns "where" into "which".
                    src = screenshots_dir / Path(screenshot_url).name
                    els = wayland_ui.detect_elements(str(src))
                    marked = src.with_name(f"mark_{src.name}")
                    if els and wayland_ui.annotate_elements(str(src), els, str(marked)):
                        if turn_state is not None:
                            turn_state["marks"] = [
                                {"xy": ((e["box"][0] + e["box"][2]) // 2,
                                        (e["box"][1] + e["box"][3]) // 2),
                                 "label": e["label"] or e["kind"]} for e in els
                            ]
                        return ({
                            "url": f"/screenshots/{marked.name}",
                            "image_width": size[0], "image_height": size[1],
                            "elements": [{"mark": i, "kind": e["kind"], "label": e["label"]}
                                         for i, e in enumerate(els, start=1)],
                            "note": "Her ogenin yaninda turuncu numarasi var. Tiklamak icin "
                                    "ui_click'e o numarayi `mark` olarak ver; koordinat verme.",
                        }, f"/screenshots/{marked.name}", f"{len(els)} öğe numaralandı")

                res = {
                    "url": screenshot_url,
                    "image_width": size[0],
                    "image_height": size[1],
                    "note": "Tikla: ui_click'e bu goruntudeki piksel koordinatlarini ver.",
                }
                return res, screenshot_url, ("Aktif pencere görüntüsü alındı"
                                             if scope == "active_window" else "Masaüstü görüntüsü alındı")

            elif tool_name == "execute_terminal_command":
                cmd = parsed_args.get("command", "")
                cwd = parsed_args.get("cwd")

                cmd_risk = self.router.evaluate_risk(prompt=cmd)
                if cmd_risk.requires_approval and request_approval_cb:
                    appr_id = f"appr_{int(time.time()*1000)}_{secrets.token_hex(4)}"
                    desc = f"Terminal Komutu: {cmd}"
                    act_hash = compute_action_hash(session_id, cmd, desc, cmd_risk.level)
                    rec = {
                        "approval_id": appr_id,
                        "session_id": session_id,
                        "command": cmd,
                        "action_description": desc,
                        "risk_level": cmd_risk.level,
                        "action_hash": act_hash,
                        "created_at": time.time(),
                        "status": "pending",
                    }
                    decision = await request_approval_cb(rec)
                    if decision != "approve":
                        return {"error": "Kullanıcı komutu onaylamadı veya reddetti.", "rejected": True}, None, f"Reddedildi: {cmd[:30]}"

                cmd_res = await self.execute_terminal_command(cmd, cwd)
                label = f"Terminal: {cmd[:35]}"
                return cmd_res, None, label

            elif tool_name == "ui_type_text":
                text_arg = parsed_args.get("text", "")
                target_app = (parsed_args.get("target_app") or "").strip()
                # Fail closed: typing goes to whatever holds focus, so a stale focus
                # silently delivers the message to the wrong application. Verify first.
                active = wayland_ui.get_active_window() or {}
                active_id = f"{active.get('class', '')} {active.get('title', '')}".lower()
                if not target_app:
                    return ({"error": "target_app zorunlu: metnin hangi uygulamaya yazılacağını belirt."},
                            None, "Reddedildi: target_app eksik")
                if not active:
                    return ({"error": "Yazilmadi: odakta hicbir pencere yok. Muhtemelen Rofi "
                                      "baslatici acik (layer yuzeyi, toplevel pencere degil). "
                                      "capture_screenshot ile bak; Rofi aciksa ui_press_key ile "
                                      "'Return' bas, sonra hedef uygulamayi odakla."},
                            None, "Reddedildi: odakta pencere yok (Rofi mi açık?)")
                if target_app.lower() not in active_id:
                    return ({"error": f"Yazılmadı. Odaktaki pencere '{active.get('class') or 'yok'}' "
                                      f"(başlık: {active.get('title') or '-'}), hedef '{target_app}' değil. "
                                      f"Once wayland_focus_or_launch ile '{target_app}' odakla, sonra tekrar dene.",
                             "focused": active.get("class"), "expected": target_app},
                            None, f"Reddedildi: odak {active.get('class') or 'yok'}, hedef {target_app}")
                if turn_state is not None:
                    turn_state.pop("marks", None)  # numbering belongs to the frame it came from
                # Clear the field first. A stale draft left in a chat box silently prefixes
                # the message -- "Kardemır COO" from an abandoned search once rode out in
                # front of a real one -- and the model has no way to see that it happened.
                if not parsed_args.get("append"):
                    wayland_ui.clear_focused_field()
                ok = wayland_ui.simulate_typing(text_arg)
                if not ok:
                    return {"error": "Metin yazilamadi (ydotoold ve wl-copy calisiyor mu?)"}, None, "Metin yazılamadı"
                # The window title carries no conversation name, so "am I in the right
                # chat?" cannot be answered in code. Hand back a frame instead: the model
                # sees where the draft actually landed before it is allowed to send.
                time.sleep(0.25)
                shot_url, size = self._grab_frame(screenshots_dir, "active_window", turn_state)

                # Confirm the text actually landed. Keystrokes go to whatever holds focus,
                # and a side panel silently swallowed a whole message once -- after which
                # "is the draft gone?" at send time read as success instead of as never-typed.
                probe = " ".join(text_arg.split()[:4])
                if shot_url and probe:
                    box = wayland_ui.band_texts(
                        str(screenshots_dir / Path(shot_url).name), 0.14, "bottom")
                    folded = wayland_ui._fold_loose(probe)
                    if box and not any(folded in wayland_ui._fold_loose(b) for b in box):
                        if turn_state is not None:
                            turn_state["type_failed"] = True
                        return ({"error": "Yazi kutuya dusmedi: yazdigin metin ekranin altinda "
                                          "gorunmuyor. Klavye odagi baska yerde, muhtemelen bir "
                                          "yan panel acik. Escape'e bas veya mesaj kutusuna "
                                          "`ui_click` ile tikla, sonra tekrar yaz.",
                                 "input_area": box[:4]},
                                shot_url, "Yazı kutuya düşmedi")

                if turn_state is not None:
                    turn_state["draft_pending"] = True
                    turn_state["draft_text"] = text_arg
                    turn_state.pop("type_failed", None)
                res = {
                    "typed": text_arg,
                    "ok": True,
                    "sent": False,
                    "verify": ("Metin SADECE yazildi, GONDERILMEDI. Asagidaki goruntude once "
                               "sohbet basligini oku: dogru kisi/grup mu? Taslak dogru kutuya mi "
                               "dusmus? Dogruysa Return'e bas. Yanlissa gonderme: ctrl+a sonra "
                               "BackSpace ile temizle ve dogru sohbeti ac."),
                    "image_width": size[0],
                    "image_height": size[1],
                }
                return res, shot_url, f"Yazıldı (gönderilmedi): {text_arg[:30]}"

            elif tool_name == "ui_press_key":
                key_arg = (parsed_args.get("key") or "").strip()
                mods = parsed_args.get("modifiers") or []
                if isinstance(mods, str):
                    mods = [mods]
                if key_arg.lower() in ("ctrl", "control", "alt", "shift", "logo", "super", "meta"):
                    return ({"error": f"'{key_arg}' bir modifier, tuş degil. Gercek tusu `key` icinde ver ve "
                                      f"modifierlari `modifiers` listesine koy. Ornek: ctrl+shift+f icin "
                                      f"key='f', modifiers=['ctrl','shift']."},
                            None, f"Gecersiz tus: {key_arg}")
                # Return right after typing IS the send. Make the model state, on the
                # record, which conversation it read in the header -- the label below puts
                # that claim in front of the user, so a wrong recipient is visible at once.
                recipient = (parsed_args.get("confirmed_recipient") or "").strip()
                # Any Return with a draft pending is a send, modifiers included: the model
                # slipped an unverified recipient past this gate once by reaching for
                # ctrl+Return, which the old `not mods` condition waved straight through.
                is_send = (key_arg.lower() in ("return", "enter")
                           and turn_state is not None and turn_state.get("draft_pending"))
                # Pressing Return after a failed write sends nothing, and letting it come
                # back "ok" is how a turn ends up announcing a message that never existed.
                if (key_arg.lower() in ("return", "enter") and turn_state is not None
                        and turn_state.get("type_failed")):
                    return ({"error": "Gonderilecek bir sey yok: son yazma denemesi basarisiz "
                                      "oldu, metin kutuya hic dusmedi. Once odagi mesaj "
                                      "kutusuna al ve tekrar yaz."},
                            None, "Reddedildi: yazılmamış mesaj")
                if is_send and not recipient:
                    return ({"error": "Gonderim engellendi. Az once metin yazdin; Return'e basmak "
                                      "onu GONDERIR. Son ekran goruntusundeki sohbet basligini oku ve "
                                      "`confirmed_recipient` alanina yaz. Baslik kullanicinin istedigi "
                                      "kisi/grup degilse GONDERME: ctrl+a, BackSpace ile temizle."},
                            None, "Reddedildi: alıcı doğrulanmadı")
                if is_send:
                    # Check the claim against the screen. Left unchecked the field is just
                    # something the model types: it once passed "Muhammed" -- a word from the
                    # message body -- while parked in a completely different conversation.
                    shot_url, _ = self._grab_frame(screenshots_dir, "active_window", turn_state)
                    header = wayland_ui.top_band_texts(
                        str(screenshots_dir / Path(shot_url).name)) if shot_url else []
                    folded = wayland_ui._fold(recipient)
                    if header and not any(folded in wayland_ui._fold(h) for h in header):
                        # Point at the name it actually clicked to get here: handed only a
                        # list of header fragments it just re-sent the same wrong guess.
                        opened = turn_state.get("last_opened") or ""
                        hint = (f" Sohbeti '{opened}' yazisina tiklayarak actin; "
                                f"dogru sohbetteysen confirmed_recipient olarak BUNU ver."
                                if opened else "")
                        return ({"error": f"Gonderim engellendi: ekranin ustunde '{recipient}' "
                                          f"yazmiyor.{hint} Dogru sohbette degilsen GONDERME; "
                                          f"once dogru sohbeti ac.",
                                 "screen_header": header[:6]},
                                shot_url, f"Reddedildi: ekranda '{recipient[:20]}' yok")

                if turn_state is not None:
                    turn_state.pop("marks", None)  # numbering belongs to the frame it came from
                repeat = max(1, min(int(parsed_args.get("repeat") or 1), 20))
                ok = True
                for _ in range(repeat):
                    ok = wayland_ui.simulate_key(key_arg, mods) and ok
                combo = ("+".join(mods) + "+" + key_arg) if mods else key_arg
                if is_send and ok:
                    # Whether Enter sends or inserts a newline is a per-app setting; this
                    # ZapZap needs Ctrl+Enter. Rather than make the model guess, check the
                    # input box and try the other combination once if the draft is still there.
                    ok_sent, note = await self._confirm_sent(
                        screenshots_dir, turn_state, bool(mods))
                    if not ok_sent:
                        turn_state["draft_pending"] = True
                        return ({"error": "Gonderilemedi: metin hala yazi kutusunda duruyor. "
                                          + note},
                                None, "Gönderilemedi (metin kutuda)")
                if turn_state is not None:
                    turn_state["draft_pending"] = False
                    turn_state.pop("draft_text", None)
                res = {"key": combo, "repeat": repeat, "ok": ok} if ok else {"error": f"'{combo}' gonderilemedi"}
                if not ok:
                    return res, None, f"Tuş gönderilemedi: {combo}"
                if recipient:
                    res["sent_to"] = recipient
                    return res, None, f"Gönderildi → {recipient[:40]}"
                return res, None, f"Tuş: {combo} x{repeat}"

            elif tool_name == "ui_click":
                # One clicking tool with three ways to name a target. Separate tools for
                # text / mark / coordinates only made the model mix up their parameters:
                # it fetched a numbered frame, then called this with `text` anyway.
                needle = (parsed_args.get("text") or "").strip()
                mark = parsed_args.get("mark")

                if needle:
                    # Always acts on a frame taken right now: using a stale one is how a
                    # click lands on whatever scrolled into that spot in the meantime.
                    shot_url, size = self._grab_frame(screenshots_dir, "active_window", turn_state)
                    if not shot_url:
                        return {"error": "Ekran goruntusu alinamadi."}, None, "Görüntü alınamadı"
                    hits = wayland_ui.find_text(
                        str(screenshots_dir / Path(shot_url).name), needle)
                    if not hits:
                        return ({"error": f"'{needle}' ekranda bulunamadi.",
                                 "hint": "Gorunur durumda mi? Listeyi kaydirman veya once ilgili "
                                         "pencereyi acman gerekebilir. Yaziyi ekranda gorundugu "
                                         "gibi ve kisa yaz. Hedefte hic yazi yoksa "
                                         "capture_screenshot'i mark_elements=true ile cagirip "
                                         "numarasiyla tikla."},
                                shot_url, f"Bulunamadı: {needle[:30]}")

                    occurrence = parsed_args.get("occurrence")
                    if occurrence:
                        idx = int(occurrence)
                        if not 1 <= idx <= len(hits):
                            return ({"error": f"occurrence {idx} gecersiz; "
                                              f"{len(hits)} eslesme var."},
                                    shot_url, "Geçersiz occurrence")
                    elif len(hits) == 1:
                        idx = 1
                    else:
                        # Refusing an ambiguous match was a dead end: told to retry with
                        # `occurrence`, the model just reissued the identical call until the
                        # loop guard killed the turn. So a repeat advances to the next match
                        # by itself -- its one reflex becomes the way through.
                        cycle = turn_state.setdefault("click_cycle", {}) if turn_state is not None else {}
                        key = needle.lower()
                        idx = cycle.get(key, 0) % len(hits) + 1
                        cycle[key] = idx
                    hit = hits[idx - 1]
                    cx, cy = hit["x"], hit["y"]
                    what = hit["text"]
                    if turn_state is not None:
                        turn_state["last_opened"] = hit["text"]
                    if len(hits) > 1:
                        what = f"{hit['text']} ({idx}/{len(hits)}. eşleşme)"

                elif mark is not None:
                    marks = (turn_state or {}).get("marks") or []
                    try:
                        n = int(mark)
                    except (TypeError, ValueError):
                        return {"error": "mark tamsayi olmali."}, None, "Geçersiz mark"
                    if not marks:
                        return ({"error": "Numarali oge yok. Once capture_screenshot'i "
                                          "mark_elements=true ile cagir."},
                                None, "Reddedildi: işaretli görüntü yok")
                    if not 1 <= n <= len(marks):
                        return ({"error": f"{n} numarali oge yok; 1..{len(marks)} arasi."},
                                None, f"Geçersiz numara: {n}")
                    cx, cy = marks[n - 1]["xy"]
                    # Echo the label: numbering is recomputed per frame, so element 12 in
                    # one screenshot is not element 12 in the next. Saying what was hit is
                    # how the model notices it reused a number out of habit.
                    what = f"öğe {n} ({marks[n - 1]['label'][:40]})"

                else:
                    if turn_state is None or "shot_size" not in turn_state:
                        return ({"error": "Once capture_screenshot cagir. Tiklama koordinatlari "
                                          "en son alinan goruntuye gore yorumlanir."},
                                None, "Reddedildi: önce ekran görüntüsü gerekli")
                    try:
                        cx, cy = int(parsed_args.get("x")), int(parsed_args.get("y"))
                    except (TypeError, ValueError):
                        return ({"error": "Hedefi belirt: `text` (ustundeki yazi), `mark` "
                                          "(numarali oge) veya `x`+`y`."},
                                None, "Hedef belirtilmedi")
                    sw, sh = turn_state["shot_size"]
                    if sw and sh and not (0 <= cx <= sw and 0 <= cy <= sh):
                        return ({"error": f"({cx},{cy}) goruntunun disinda. Goruntu {sw}x{sh}."},
                                None, f"Koordinat görüntü dışı: {cx},{cy}")
                    what = f"({cx},{cy})"

                ox, oy = (turn_state or {}).get("shot_origin", (0, 0))
                if turn_state is not None:
                    turn_state.pop("marks", None)  # the click is about to change the frame
                click_res = wayland_ui.click_at(ox + cx, oy + cy,
                                                parsed_args.get("button") or "left",
                                                bool(parsed_args.get("double")))
                if click_res.get("error"):
                    return click_res, None, f"Tıklama başarısız: {click_res['error'][:50]}"

                # Hand back what the screen looks like now: the model must see where it
                # actually landed before it does anything irreversible next.
                time.sleep(0.6)
                after_url, after_size = self._grab_frame(
                    screenshots_dir, "active_window", turn_state)
                res = {"clicked": what, "at": [cx, cy],
                       "image_width": after_size[0], "image_height": after_size[1],
                       "verify": "Tiklama sonrasi goruntu asagida. Bekledigin yere mi gitti?"}
                if needle and turn_state is not None:
                    seen = turn_state.setdefault("clicked_texts", {})
                    seen[needle.lower()] = seen.get(needle.lower(), 0) + 1
                    if seen[needle.lower()] > 1:
                        # Said at the moment of the mistake, which lands better than a rule
                        # in the prompt: it re-clicked the same chat row five times running.
                        res["warning"] = (
                            f"'{needle}' yazisina bu turda {seen[needle.lower()]} kez tikladin. "
                            f"Tekrar tiklama — hedef zaten acik. SIRADAKI adima gec: mesaj "
                            f"yazacaksan dogrudan `ui_type_text` cagir.")
                if needle and len(hits) > 1:
                    res["note"] = (f"'{needle}' ekranda {len(hits)} yerde gecti; {idx}. "
                                   f"olanina tiklandi. Yanlis yere gittiyse AYNI cagriyi "
                                   f"aynen tekrarla, siradakini denerim.")
                return res, after_url, f"Tıklandı: {what[:35]}"


            elif tool_name == "hermes_skill":
                skill = (parsed_args.get("name") or "").strip()
                query = (parsed_args.get("query") or "").strip()
                # Search and run are one tool because a two-tool split proved to be a
                # trap: the model kept calling the runner with the searcher's `query`.
                if not skill:
                    res = agent_bridge.search(query, int(parsed_args.get("limit") or 12))
                    n = len(res.get("matches", []))
                    return res, None, f"{n} yetenek bulundu"

                skill_args = parsed_args.get("arguments")
                if isinstance(skill_args, str):
                    try:
                        skill_args = json.loads(skill_args)
                    except ValueError:
                        return ({"error": "arguments gecerli bir JSON nesnesi olmali."},
                                None, "Geçersiz argüman")
                skill_args = skill_args or {}
                # These reach the shell and the filesystem, so they pass through the same
                # approval gate as execute_terminal_command rather than around it.
                if skill in agent_bridge.HIGH_RISK_TOOLS and request_approval_cb:
                    decision = await request_approval_cb({
                        "tool": "hermes_skill",
                        "action": f"{skill}({json.dumps(skill_args, ensure_ascii=False)[:200]})",
                        "risk": "high",
                    })
                    if decision != "approve":
                        return ({"error": "Kullanıcı onaylamadı.", "rejected": True},
                                None, f"Reddedildi: {skill}")
                res = await agent_bridge.dispatch(skill, skill_args)
                failed = isinstance(res, dict) and res.get("error")
                return res, None, (f"{skill} başarısız" if failed else f"{skill} çalıştırıldı")

            elif tool_name == "send_chat_message":
                app = (parsed_args.get("app") or "").strip()
                chat = (parsed_args.get("chat") or "").strip()
                body = parsed_args.get("text") or ""
                if not (app and chat and body):
                    return ({"error": "app, chat ve text zorunlu."}, None,
                            "Reddedildi: eksik alan")
                res, shot = await self.send_chat_message(
                    app, chat, body, screenshots_dir, turn_state)
                label = (f"Gönderildi → {chat[:30]}" if res.get("sent")
                         else f"Gönderilemedi → {chat[:30]}")
                return res, shot, label

            elif tool_name == "notify_user":
                prio = parsed_args.get("priority") or "normal"
                if prio not in ("low", "normal", "urgent"):
                    prio = "normal"
                title = parsed_args.get("title") or "Neo"
                body = parsed_args.get("body") or ""
                await emit_event_cb({
                    "id": f"evt_{int(time.time()*1000)}_notify",
                    "session_id": session_id,
                    "type": "user_notification",
                    "priority": prio,
                    "title": title,
                    "body": body,
                    "timestamp": time.time(),
                })
                return ({"sent": True, "priority": prio}, None,
                        f"Bildirim gonderildi ({prio}): {title[:30]}")

            elif tool_name == "route_coding_agent":
                agent = parsed_args.get("agent", "antigravity")
                instruction = parsed_args.get("instruction", user_text)
                wdir = parsed_args.get("working_dir")
                resume = bool(parsed_args.get("resume", False))
                await emit_event_cb({
                    "id": f"evt_{int(time.time()*1000)}_delegate",
                    "session_id": session_id,
                    "type": "task_progress",
                    "priority": "low",
                    "title": f"Delegating to {agent.capitalize()}",
                    "body": f"Running task via {agent} (resume={resume})...",
                    "timestamp": time.time(),
                })
                delegation_res = await self.delegate_agent(agent, instruction, wdir, resume)
                label = f"{agent.capitalize()} Ajanı Tamamlandı"
                return {"status": "completed", "output": delegation_res[:1000]}, None, label

            else:
                return {"error": f"Bilinmeyen araç: {tool_name}"}, None, f"Hata: {tool_name}"

        except Exception as exc:
            return {"error": str(exc)}, None, f"Hata: {tool_name} ({exc})"

    async def execute_session_turn(
        self,
        session_id: str,
        user_text: str,
        conversation_history: List[Dict[str, Any]],
        system_prompt: str,
        screenshots_dir: Path,
        request_approval_cb: Callable[[Dict[str, Any]], Any],
        emit_event_cb: Callable[[Dict[str, Any]], Any],
        on_delta_cb: Optional[Callable[[str, str], Any]] = None,
        on_tool_cb: Optional[Callable[[Dict[str, Any]], Any]] = None,
    ) -> Dict[str, Any]:
        """
        The authoritative Hermes AIAgent turn execution pipeline.
        Orchestrates:
        1. Cognitive Risk Routing
        2. Cryptographic Approval Gating
        3. Precompaction of older dialogue (>60K tokens) preserving latest 5 pairs verbatim
        4. Multi-step Autonomous Agent Loop with Native Tool Semantics
        5. Streaming Gemma-4 E4B inference with dynamic thinking
        """
        now = time.time()
        risk: RiskAssessment = self.router.evaluate_risk(prompt=user_text)

        # 1. Emit progress event
        await emit_event_cb({
            "id": f"evt_{int(now*1000)}_eval",
            "session_id": session_id,
            "type": "task_progress",
            "priority": "low",
            "title": "Evaluating Request",
            "body": f"Risk level: {risk.level.upper()} | Thinking: {'ON' if risk.thinking_enabled else 'OFF'}",
            "timestamp": now,
            "payload": {"risk_level": risk.level, "thinking_enabled": risk.thinking_enabled},
        })

        # 2. Cryptographic Approval Gate
        if risk.requires_approval:
            approval_id = f"appr_{int(time.time()*1000)}_{secrets.token_hex(4)}"
            action_desc = f"Execute action with high impact: '{user_text}'"
            action_hash = compute_action_hash(session_id, user_text, action_desc, risk.level)

            approval_record = {
                "approval_id": approval_id,
                "session_id": session_id,
                "command": user_text,
                "action_description": action_desc,
                "risk_level": risk.level,
                "action_hash": action_hash,
                "created_at": time.time(),
                "status": "pending",
            }

            decision = await request_approval_cb(approval_record)
            if decision != "approve":
                refusal = f"⚠️ **Action Blocked**: High-risk action was rejected or timed out.\nTarget: `{user_text}`"
                return {
                    "role": "assistant",
                    "content": refusal,
                    "reasoning_content": None,
                    "tool_calls": [],
                    "screenshot_url": None,
                    "risk_assessment": risk.__dict__,
                    "action_blocked": True,
                }

        # 3. Context Assembly with Precompaction (>60K tokens)
        canonical_sys = system_prompt or get_canonical_stable_prefix()
        raw_history = [
            {"role": "system", "content": canonical_sys},
            *conversation_history,
            {"role": "user", "content": user_text},
        ]
        compacted_messages, was_compacted = compact_history_if_needed(raw_history, max_tokens=COMPACTION_TOKEN_THRESHOLD)

        if was_compacted:
            await emit_event_cb({
                "id": f"evt_{int(time.time()*1000)}_compact",
                "session_id": session_id,
                "type": "task_progress",
                "priority": "low",
                "title": "Context Precompaction",
                "body": "Dialogue exceeded 60K tokens. Compacted older history while preserving latest 5 pairs verbatim.",
                "timestamp": time.time(),
            })

        # 4. Multi-step Autonomous Agent Loop with Native Tool Semantics (max_steps=5)
        max_steps = 12
        step = 0
        loop_guard_history: List[Tuple[str, str]] = []
        consecutive_failures: Dict[str, int] = {}
        last_tool_error = ""
        turn_state: Dict[str, Any] = {}
        all_tool_calls_record: List[Dict[str, Any]] = []
        current_messages = list(compacted_messages)

        final_content = ""
        reasoning_parts: List[str] = []
        final_reasoning = None
        final_screenshot_url: Optional[str] = None
        last_usage_info: Optional[Dict[str, Any]] = None
        t0_exec = time.perf_counter()
        t_first_token: Optional[float] = None
        first_tool_call_time: Optional[float] = None

        while step < max_steps:
            # Thinking mode stays CONSTANT for the whole turn on purpose: flipping
            # enable_thinking mid-loop changes the chat template prefix and costs a full
            # ~5.4K-token re-prefill (~200s on this box). Reasoning is still captured and
            # surfaced from every step below whenever the model emits it.
            step_thinking = risk.thinking_enabled
            payload = {
                "model": "default",
                "messages": current_messages,
                "tools": get_canonical_tools(),
                "tool_choice": "auto",
                # Router gates only the first step (low-risk one-shots stay sub-second).
                # Once the turn becomes a multi-step tool loop, thinking is always on so the
                # phone can show why Neo is doing what it is doing.
                "temperature": 0.1 if not step_thinking else 0.3,
                "chat_template_kwargs": {"enable_thinking": step_thinking},
                "max_tokens": 4096,
                "stream": True,
                "stream_options": {"include_usage": True},
            }

            content_chunks: List[str] = []
            reasoning_chunks: List[str] = []
            tool_call_accumulator: Dict[str, Any] = {}

            async with httpx.AsyncClient(timeout=300.0) as async_client:
                async with async_client.stream("POST", f"{self.base_url}/v1/chat/completions", json=payload) as resp:
                    async for line in resp.aiter_lines():
                        if not line or not line.startswith("data: "):
                            continue
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            delta_json = json.loads(data_str)
                            if t_first_token is None:
                                t_first_token = time.perf_counter()
                            if "usage" in delta_json and delta_json["usage"]:
                                last_usage_info = delta_json["usage"]

                            choices = delta_json.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {})
                                if "reasoning_content" in delta and delta["reasoning_content"]:
                                    r_tok = delta["reasoning_content"]
                                    reasoning_chunks.append(r_tok)
                                    if on_delta_cb:
                                        await on_delta_cb("reasoning", r_tok)
                                if "content" in delta and delta["content"]:
                                    c_tok = delta["content"]
                                    content_chunks.append(c_tok)
                                    # Suppress raw streaming if inside tool call markers
                                    raw_so_far = "".join(content_chunks)
                                    if "<tool_call" not in raw_so_far and not raw_so_far.strip().startswith("```json") and not raw_so_far.strip().startswith("{"):
                                        if on_delta_cb:
                                            await on_delta_cb("content", c_tok)
                                if "tool_calls" in delta and delta["tool_calls"]:
                                    if first_tool_call_time is None:
                                        first_tool_call_time = time.perf_counter()
                                    tc = delta["tool_calls"][0]
                                    if "function" in tc:
                                        fn = tc["function"]
                                        if "name" in fn and fn["name"]:
                                            curr = tool_call_accumulator.get("name", "")
                                            incoming = fn["name"]
                                            if not curr:
                                                tool_call_accumulator["name"] = incoming
                                            elif curr != incoming and not incoming.startswith(curr):
                                                tool_call_accumulator["name"] = curr + incoming
                                        if "arguments" in fn and fn["arguments"]:
                                            tool_call_accumulator["arguments"] = tool_call_accumulator.get("arguments", "") + fn["arguments"]
                        except Exception:
                            pass

            raw_step_content = "".join(content_chunks).strip()
            raw_step_reasoning = "".join(reasoning_chunks).strip()
            if raw_step_reasoning:
                reasoning_parts.append(raw_step_reasoning)

            tool_name, parsed_args, clean_content = parse_and_strip_tool_call(tool_call_accumulator, raw_step_content)

            if not tool_name:
                # Terminal step: model provided text response with no tool
                final_content = clean_content
                break

            # Loop Guard: Halt consecutive duplicate invocations
            arg_key = json.dumps(parsed_args, sort_keys=True)
            # Counted across the turn, not just back to back: a look-click-look loop puts a
            # capture_screenshot between every repeat, so "consecutive" never triggered while
            # the model clicked the same row five times.
            if loop_guard_history.count((tool_name, arg_key)) >= 2:
                print(f"[LOOP_GUARD] Triggered for {tool_name} with {arg_key}", flush=True)
                final_content = (f"`{tool_name}` aracını bu turda aynı argümanlarla üçüncü kez "
                                 f"çağırdım ve ekranda bir ilerleme olmadı, durduruyorum. "
                                 f"Görevi tamamlayamadım.")
                break

            # A tool that keeps erroring is a loop even when the arguments differ each
            # time -- varying them is exactly how a confused model burns the step budget.
            if consecutive_failures.get(tool_name, 0) >= 3:
                print(f"[LOOP_GUARD] {tool_name} failed 3x consecutively", flush=True)
                final_content = (f"`{tool_name}` aracını üst üste 3 kez hatayla çağırdım, "
                                 f"son hata: {last_tool_error}. Bu yoldan devam etmiyorum — "
                                 f"görevi tamamlayamadım.")
                break

            loop_guard_history.append((tool_name, arg_key))

            # Emit tool progress badge to phone
            tool_call_id = f"call_{secrets.token_hex(6)}"
            public_label = get_tool_public_label(tool_name, parsed_args)

            tool_action_record = {
                "id": tool_call_id,
                "name": tool_name,
                "status": "pending",
                "label": public_label,
                "arguments": parsed_args,
                "step": step + 1,
            }
            all_tool_calls_record.append(tool_action_record)
            if on_tool_cb:
                await on_tool_cb(tool_action_record)

            # Execute tool safely
            tool_res, shot_url, status_summary = await self.execute_tool(
                tool_name=tool_name,
                parsed_args=parsed_args,
                session_id=session_id,
                user_text=user_text,
                screenshots_dir=screenshots_dir,
                emit_event_cb=emit_event_cb,
                request_approval_cb=request_approval_cb,
                turn_state=turn_state,
            )

            if shot_url:
                final_screenshot_url = shot_url

            has_error = isinstance(tool_res, dict) and bool(tool_res.get("error"))
            tool_action_record["status"] = "failure" if has_error else "success"
            if has_error:
                consecutive_failures[tool_name] = consecutive_failures.get(tool_name, 0) + 1
                last_tool_error = str(tool_res.get("error"))[:150]
            else:
                consecutive_failures.pop(tool_name, None)
            tool_action_record["label"] = status_summary
            tool_action_record["result"] = tool_res
            if on_tool_cb:
                await on_tool_cb(tool_action_record)

            # Append assistant message with native tool_calls
            current_messages.append({
                "role": "assistant",
                "content": clean_content if clean_content else None,
                "tool_calls": [{
                    "id": tool_call_id,
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": json.dumps(parsed_args, ensure_ascii=False),
                    }
                }]
            })

            # Append native tool response message
            tool_msg_content = json.dumps(tool_res, ensure_ascii=False) if not isinstance(tool_res, str) else tool_res
            current_messages.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": tool_msg_content,
            })

            # Feed the captured frame back to the model as a real image so it can
            # actually SEE the desktop instead of receiving a URL string it cannot open.
            # llama-server runs with --mmproj (Gemma 4 E4B vision), so image_url parts work.
            if shot_url:
                image_part = _encode_screenshot_as_data_uri(screenshots_dir, shot_url)
                if image_part:
                    # Only the newest frame stays as an image; older ones collapse to a
                    # placeholder so a 12-step loop does not resend megabytes of base64 each step.
                    for msg in current_messages:
                        if isinstance(msg.get("content"), list):
                            msg["content"] = [
                                {"type": "text", "text": "[Önceki ekran görüntüsü — güncel değil, çıkarıldı]"}
                                if part.get("type") == "image_url" else part
                                for part in msg["content"]
                            ]
                    current_messages.append({
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "[Otomatik gözlem] Az önce alınan ekran görüntüsü aşağıda"
                                    + (f" ({turn_state['shot_size'][0]}x{turn_state['shot_size'][1]} piksel)"
                                       if turn_state.get("shot_size") else "")
                                    + ". Ekranda ne olduğunu incele. Bir şeye tıklaman gerekiyorsa "
                                      "`ui_click` aracına BU GÖRÜNTÜDEKİ piksel koordinatlarını ver "
                                      "(sol üst köşe 0,0). Sonra göreve devam et."
                                ),
                            },
                            {"type": "image_url", "image_url": {"url": image_part}},
                        ],
                    })

            if clean_content:
                final_content = clean_content

            step += 1

        final_reasoning = "\n\n".join(reasoning_parts) if reasoning_parts else None

        if not final_content:
            if final_screenshot_url:
                final_content = "Masaüstü ekran görüntüsü başarıyla alındı."
            elif all_tool_calls_record:
                final_content = "İşlem adımları başarıyla tamamlandı."
            else:
                final_content = "Komut işlendi."

        cache_telem = parse_cache_telemetry(
            usage=last_usage_info,
            start_time=t0_exec,
            ttft=(t_first_token - t0_exec) if t_first_token else None,
            thinking_mode=risk.thinking_enabled,
            tool_decision_latency=(first_tool_call_time - t0_exec) if first_tool_call_time else None,
        )
        print(
            f"[CACHE_TELEMETRY] prompt_tokens={cache_telem['prompt_tokens']} "
            f"cached_tokens={cache_telem['cached_tokens']} "
            f"cached_ratio={cache_telem['cached_ratio']:.2%} "
            f"newly_eval={cache_telem['newly_evaluated_tokens']} "
            f"ttft={cache_telem['ttft']}s total={cache_telem['total_latency']}s "
            f"thinking={cache_telem['thinking_mode']} prefix_sha={cache_telem['stable_prefix_sha256'][:8]}",
            flush=True,
        )

        return {
            "role": "assistant",
            "content": final_content,
            "reasoning_content": final_reasoning,
            "tool_calls": all_tool_calls_record,
            "screenshot_url": final_screenshot_url,
            "risk_assessment": risk.__dict__,
            "compacted": was_compacted,
            "telemetry": cache_telem,
        }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Hermes Local PC & Browser Supervisor")
    parser.add_argument("prompt", type=str, help="User prompt / action command")
    parser.add_argument("--screenshot", type=str, default=None, help="Path to UI screenshot PNG")
    parser.add_argument("--terminal", type=str, default=None, help="Terminal traceback or output")
    parser.add_argument("--dom", type=str, default=None, help="Active DOM HTML snippet")
    args = parser.parse_args()

    supervisor = HermesSupervisor()
    result = supervisor.dispatch(
        prompt=args.prompt,
        screenshot_path=args.screenshot,
        terminal_output=args.terminal,
        dom_snippet=args.dom,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
