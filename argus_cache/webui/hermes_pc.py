"""Hermes Wayland / Hyprland PC & Desktop Control adapter for ARGUS Studio (Background + Foreground)."""

from __future__ import annotations

import asyncio
from collections import Counter
import json
import logging
import re
import secrets
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from argus_cache.webui.hermes_vision import (
    fast_click,
    fast_detect_elements,
    fast_find_text,
    fast_read_screen_text,
    focus_or_launch_smart,
    is_studio_window,
    list_clients_full,
    remove_stale_headless_monitors,
    transient_target_focus,
)
from argus_cache.webui.permissions import PermissionBroker

HERMES_DIR = Path(__file__).resolve().parents[2] / "experiments" / "hermes"
if str(HERMES_DIR) not in sys.path:
    sys.path.insert(0, str(HERMES_DIR))

# The Hermes desktop agent lives in experiments/hermes and is not part of the
# wheel. Without it Studio still runs; it just offers no PC tools.
try:
    import agent_bridge  # type: ignore  # noqa: E402
    import wayland_ui  # type: ignore  # noqa: E402
    from cognitive_router import CognitiveRouter  # type: ignore  # noqa: E402
except ImportError as _exc:
    HERMES_UNAVAILABLE: Optional[str] = f"Hermes desktop agent not installed ({_exc})"
    logging.getLogger("argus.studio").warning("%s; PC tools disabled", HERMES_UNAVAILABLE)
    _ROUTER = None
else:
    HERMES_UNAVAILABLE = None
    _ROUTER = CognitiveRouter()

#: Tools whose output is a reading of the screen. It is stale once the turn
#: ends, so chat history replays a marker instead of the old screen text.
SCREEN_STATE_TOOLS = frozenset({
    "pc_list_windows", "pc_focus_or_launch", "pc_shortcut", "pc_capture_screen",
    "pc_click", "pc_type_text", "pc_press_key",
})

#: Keys that change nothing on screen and are undone by pressing them again
#: (volume, media, brightness). Every other bind opens a window, changes the
#: layout or (zapret) the network, so it waits for the user's approval.
_SILENT_KEY_PREFIXES = ("XF86Audio", "XF86MonBrightness")

#: Hyprland modmask bits, in the order a person would say the combination.
_MODMASK = ((64, "super"), (4, "ctrl"), (8, "alt"), (1, "shift"))


def parse_live_shortcuts(binds: List[Dict[str, Any]], known_keys) -> Dict[str, Tuple[List[str], str]]:
    """``hyprctl binds -j`` -> {the user's own bind description: (modifiers, key)}.

    The tool used to copy commands out of hyprland.lua and drifted whenever the
    config changed. Reading the compositor's live binds and pressing the key
    combination keeps it exactly in step. Mouse, catch-all and release binds
    are skipped, as are keys ydotool cannot press.
    """
    counts = Counter((b.get("description") or "").strip() for b in binds)
    table: Dict[str, Tuple[List[str], str]] = {}
    for b in binds:
        desc, key = (b.get("description") or "").strip(), str(b.get("key") or "")
        if desc in ("", "-") or b.get("mouse") or b.get("catch_all") or b.get("release"):
            continue
        if key.lower() not in known_keys:
            continue
        name = f"{desc} {key}" if counts[desc] > 1 else desc
        table[name] = ([m for bit, m in _MODMASK if int(b.get("modmask") or 0) & bit], key)
    return table


def live_shortcuts() -> Dict[str, Tuple[List[str], str]]:
    """The binds Hyprland has right now; empty when hyprctl is unavailable."""
    try:
        r = subprocess.run(["hyprctl", "binds", "-j"], capture_output=True, text=True, timeout=3)
        return parse_live_shortcuts(json.loads(r.stdout), wayland_ui.KEYCODES)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        logging.getLogger("argus.studio").warning("could not read Hyprland binds: %s", exc)
        return {}

PC_HINT_RE = re.compile(
    r"(\bpc\b|bilgisayar|masaüstü|masaustu|pencere|workspace|ekran görüntüsü|ekran goruntusu|screenshot|"
    r"ekranda|ekrandaki|ekranı|ekrani|ne açık|ne acik|ne görünüyor|ne gorunuyor|"
    r"uygulama|odakla|başlat|baslat|tıkla|tikla|tuşa|tusa|kısayol|kisayol|sesi\s+(?:aç|ac|kıs|kis|kapat)|"
    r"şarkı|sarki|müzik|muzik|durdur|oynat|sonraki parça|terminalde|komut çalıştır|komut calistir|"
    r"\b(?:chrome|kitty|spotify|discord|zapzap|whatsapp|signal|rofi|cava|waybar|dolphin|hyprctl)\b|"
    r"mesaj gönder|mesaj gonder|mesaj at|hermes)",
    re.IGNORECASE,
)

PC_TOOL_NAMES = frozenset({
    "pc_list_windows",
    "pc_focus_or_launch",
    "pc_shortcut",
    "pc_capture_screen",
    "pc_click",
    "pc_type_text",
    "pc_press_key",
    "pc_terminal_command",
    "hermes_skill",
})


def _fn(name: str, desc: str, props: Dict[str, Any], req: List[str]) -> Dict[str, Any]:
    schema: Dict[str, Any] = {"type": "object", "properties": props}
    if req:
        schema["required"] = req
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": schema}}


def build_hermes_pc_schemas() -> List[Dict[str, Any]]:
    if HERMES_UNAVAILABLE:
        return []
    shortcuts = live_shortcuts()
    return [
        _fn("pc_list_windows", "Açık masaüstü pencerelerini ve workspace numaralarını listeler.", {}, []),
        _fn(
            "pc_focus_or_launch",
            "Uygulamayı arka planda/masaüstünde açar veya odaklar ve ekranındaki metinleri OCR ile hemen döndürür (ör: zapzap, chrome, kitty, spotify, discord, signal).",
            {"app_name": {"type": "string"}, "include_screen": {"type": "boolean"}},
            ["app_name"],
        ),
        _fn(
            "pc_shortcut",
            "Kullanıcının kendi Hyprland kısayolunu tuş kombinasyonuna basarak çalıştırır (ses/medya/parlaklık "
            "dışındakiler onay ister). Yalnızca kullanıcı açıkça bir masaüstü eylemi istediğinde çağır.",
            {"action": {"type": "string", **({"enum": sorted(shortcuts)} if shortcuts else {})}},
            ["action"],
        ),
        _fn(
            "pc_capture_screen",
            "Hedef uygulamanın veya ekranın temiz görüntüsünü alır ve ekrandaki tüm metin içeriğini OCR ile döndürür.",
            {
                "scope": {"type": "string", "enum": ["active_window", "full"]},
            },
            [],
        ),
        _fn(
            "pc_click",
            "Ekrandaki öğeye görünür `text` (önerilen) veya `x`+`y` ile tıklar ve güncel ekran metnini döndürür.",
            {
                "text": {"type": "string"},
                "occurrence": {"type": "integer"},
                "x": {"type": "integer"},
                "y": {"type": "integer"},
                "button": {"type": "string", "enum": ["left", "right", "middle"]},
                "double": {"type": "boolean"},
            },
            [],
        ),
        _fn(
            "pc_type_text",
            "Hedef uygulamaya Türkçe karakter korumalı metin yazar (Enter basmaz).",
            {
                "text": {"type": "string"},
                "target_app": {"type": "string"},
                "append": {"type": "boolean"},
            },
            ["text", "target_app"],
        ),
        _fn(
            "pc_press_key",
            "Hedef pencereye tuş/kısayol gönderir (Return, Escape, Tab, Down, f + modifiers). Mesaj gönderirken `confirmed_recipient` zorunludur.",
            {
                "key": {"type": "string"},
                "modifiers": {"type": "array", "items": {"type": "string", "enum": ["ctrl", "alt", "shift", "logo"]}},
                "repeat": {"type": "integer"},
                "confirmed_recipient": {"type": "string"},
            },
            ["key"],
        ),
        _fn(
            "pc_terminal_command",
            "İzin onaylı bash komutu çalıştırır.",
            {"command": {"type": "string"}, "cwd": {"type": "string"}},
            ["command"],
        ),
        _fn(
            "hermes_skill",
            "Hermes ajan kataloğunda ek araç arar (`query`) veya çalıştırır (`name` + `arguments`).",
            {"query": {"type": "string"}, "name": {"type": "string"}, "arguments": {"type": "object"}},
            [],
        ),
    ]


def _png_size(path: Path) -> Tuple[int, int]:
    try:
        head = path.read_bytes()[:24]
        if len(head) >= 24 and head[12:16] == b"IHDR":
            return int.from_bytes(head[16:20], "big"), int.from_bytes(head[20:24], "big")
    except OSError:
        pass
    return (0, 0)


def _grab_screen_with_ocr(
    screenshots_dir: Path,
    scope: str,
    mark_elements: bool,
    pc_state: Dict[str, Any],
    emit_event: Callable[[str, Dict[str, Any]], None],
) -> Dict[str, Any]:
    del mark_elements
    remove_stale_headless_monitors()
    screenshots_dir.mkdir(parents=True, exist_ok=True)
    snap_name = f"snap_{int(time.time() * 1000)}_{secrets.token_hex(4)}.png"
    filepath = screenshots_dir / snap_name

    origin = (0, 0)
    size: Optional[Tuple[int, int]] = None
    target = dict(pc_state.get("target_window") or {})
    win_label = ""
    ok = False

    if scope == "active_window" and (target.get("address") or target.get("class")):
        for client in list_clients_full():
            if (target.get("address") and client["address"] == target["address"]) or (
                not target.get("address") and target.get("class") and client["class"] == target["class"]
            ):
                target.update(client)
                pc_state["target_window"] = target
                break
        at, sz = target.get("at"), target.get("size")
        if at and sz and int(sz[0]) > 0 and int(sz[1]) > 0:
            origin, size = (int(at[0]), int(at[1])), (int(sz[0]), int(sz[1]))
        bg_tag = " [Arka Plan]" if target.get("background") else ""
        win_label = f"{target.get('class', 'uygulama')} — {target.get('title', '')}{bg_tag}".strip(" —")

        stable_id = str(target.get("stable_id") or "").strip()
        if stable_id:
            try:
                proc = subprocess.run(["/usr/bin/grim", "-T", stable_id, str(filepath)], capture_output=True, timeout=5)
                ok = proc.returncode == 0 and filepath.is_file() and filepath.stat().st_size > 0
            except Exception:
                ok = False

        if not ok:
            with transient_target_focus(wayland_ui, pc_state):
                grim_cmd = ["/usr/bin/grim"]
                if size and size[0] > 0 and size[1] > 0:
                    grim_cmd += ["-g", f"{origin[0]},{origin[1]} {size[0]}x{size[1]}"]
                grim_cmd.append(str(filepath))
                try:
                    proc = subprocess.run(grim_cmd, capture_output=True, timeout=5)
                    ok = proc.returncode == 0 and filepath.is_file() and filepath.stat().st_size > 0
                except Exception:
                    ok = False
    else:
        grim_cmd = ["/usr/bin/grim"]
        if scope == "active_window":
            win = wayland_ui.get_active_window() or {}
            if is_studio_window(win):
                candidates = [w for w in list_clients_full() if w.get("mapped", True) and not is_studio_window(w)]
                if candidates and candidates[0].get("stable_id"):
                    c0 = candidates[0]
                    origin = (int(c0["at"][0]), int(c0["at"][1]))
                    size = (int(c0["size"][0]), int(c0["size"][1]))
                    win_label = f"{c0['class']} — {c0['title']}".strip(" —")
                    grim_cmd += ["-T", str(c0["stable_id"])]
            if "-T" not in grim_cmd:
                at, sz = win.get("at"), win.get("size")
                if at and sz and sz[0] > 0 and sz[1] > 0:
                    origin, size = (int(at[0]), int(at[1])), (int(sz[0]), int(sz[1]))
                    grim_cmd += ["-g", f"{origin[0]},{origin[1]} {size[0]}x{size[1]}"]
                win_label = f"{win.get('class', 'masaüstü')} — {win.get('title', '')}".strip(" —")
        grim_cmd.append(str(filepath))
        try:
            proc = subprocess.run(grim_cmd, capture_output=True, timeout=5)
            ok = proc.returncode == 0 and filepath.is_file() and filepath.stat().st_size > 0
        except Exception:
            ok = False

    if not ok:
        return {"error": "Ekran görüntüsü alınamadı (`grim` veya aktif Wayland oturumu bulunamadı)."}

    if not size or size[0] <= 0:
        size = _png_size(filepath)
    pc_state["shot_origin"] = origin
    pc_state["shot_size"] = size
    pc_state["last_snap_path"] = str(filepath)
    pc_state["last_snap_ts"] = time.monotonic()

    els = fast_detect_elements(wayland_ui, str(filepath), max_elements=60)
    screen_text = fast_read_screen_text(
        wayland_ui,
        str(filepath),
        width=int(size[0]) if size else 0,
        height=int(size[1]) if size else 0,
    )
    url = f"/screenshots/{snap_name}"
    pc_state["marks"] = [
        {"xy": ((e["box"][0] + e["box"][2]) // 2, (e["box"][1] + e["box"][3]) // 2), "label": e["label"]}
        for e in els
    ]
    if not win_label:
        active = wayland_ui.get_active_window() or {}
        win_label = f"{active.get('class', 'masaüstü')} — {active.get('title', '')}".strip(" —")
    emit_event("pc_screenshot", {"url": url, "scope": scope, "window": win_label})

    return {
        "url": url,
        "focused_window": win_label,
        "size": f"{size[0]}x{size[1]}",
        "elements_summary": screen_text,
    }


async def execute_hermes_pc_tool(
    name: str,
    args: Dict[str, Any],
    *,
    session_id: str,
    workspace_dir: str,
    screenshots_dir: Path,
    broker: PermissionBroker,
    pc_state: Dict[str, Any],
    emit_event: Callable[[str, Dict[str, Any]], None],
    pc_mode: str = "background",
) -> str:
    if HERMES_UNAVAILABLE:
        return HERMES_UNAVAILABLE
    use_background = pc_mode != "foreground"

    if name == "pc_list_windows":
        wins = await asyncio.to_thread(wayland_ui.list_windows)
        if not wins:
            return "Açık Wayland/Hyprland penceresi bulunamadı."
        rows = [f"- [WS {w['workspace_id']}] {w['class']} — {w['title']} (PID {w['pid']})" for w in wins]
        return f"Açık Pencereler ({len(wins)} adet):\n" + "\n".join(rows)

    if name == "pc_focus_or_launch":
        app_name = str(args.get("app_name") or args.get("app") or "").strip()
        if not app_name:
            return "Hata: `app_name` belirtilmedi."
        res = await asyncio.to_thread(
            focus_or_launch_smart,
            wayland_ui,
            app_name,
            background=use_background,
            pc_state=pc_state,
        )
        if args.get("include_screen", True) and pc_state.get("target_window"):
            pc_state["last_app"] = app_name
            await asyncio.sleep(0.35)
            snap = await asyncio.to_thread(_grab_screen_with_ocr, screenshots_dir, "active_window", False, pc_state, emit_event)
            if "error" not in snap:
                return (
                    f"Uygulama hazır: {json.dumps(res, ensure_ascii=False)}\n"
                    f"Güncel Ekran ({snap['size']}, {snap['focused_window']}):\n"
                    f"ÖNEMLİ: Kullanıcının sorduğu sohbet/kişi/sekme/not şu an `[Açık Başlık]` içinde açık değilse "
                    f"(yalnızca `Sol Panel` listesinde görünüyorsa), önizlemeyi okuyup durma; hemen `pc_click(text=...)` çağırarak o öğeyi aç!\n"
                    f"{snap['elements_summary']}"
                )
        return json.dumps(res, ensure_ascii=False, indent=2)

    if name == "pc_shortcut":
        action = str(args.get("action", "")).strip()
        table = live_shortcuts()
        if action not in table:
            return f"Bilinmeyen kısayol `{action}`. Geçerli kısayollar: {', '.join(sorted(table)) or '(Hyprland okunamadı)'}"
        mods, key = table[action]
        if not key.startswith(_SILENT_KEY_PREFIXES):
            allowed = await broker.ensure_permission(
                session_id=session_id,
                operation="desktop",
                target=Path(workspace_dir or Path.home()),
                workspace_dir=workspace_dir or None,
                summary=f"Masaüstü Kısayolu: {action}",
                diff_preview="",
                emit_event=emit_event,
            )
            if not allowed:
                return f"Kullanıcı '{action}' masaüstü kısayolunu reddetti."
        ok = await asyncio.to_thread(wayland_ui.simulate_key, key, mods)
        combo = "+".join([*mods, key]).upper()
        return json.dumps({"action": action, "shortcut": combo, "pressed": bool(ok)}, ensure_ascii=False)

    if name == "pc_capture_screen":
        scope = str(args.get("scope") or "active_window")
        snap = await asyncio.to_thread(_grab_screen_with_ocr, screenshots_dir, scope, False, pc_state, emit_event)
        if "error" in snap:
            return snap["error"]
        return (
            f"Ekran görüntüsü alındı ({snap['size']}, odak: {snap['focused_window']}).\n"
            f"Ekranda okunan içerik (kullanıcı ekranda ne olduğunu soruyorsa aşağıdaki gerçek metinleri özetle; "
            f"tıklama gerekiyorsa `pc_click(text=...)` çağır):\n\n{snap['elements_summary']}"
        )

    if name == "pc_click":
        needle = str(args.get("text") or "").strip()
        mark = args.get("mark")
        if needle:
            last_path = pc_state.get("last_snap_path")
            fresh = bool(last_path) and Path(last_path).is_file() and (time.monotonic() - float(pc_state.get("last_snap_ts") or 0)) < 20.0
            hits = await asyncio.to_thread(fast_find_text, wayland_ui, str(last_path), needle) if fresh else []
            if not hits:
                snap = await asyncio.to_thread(_grab_screen_with_ocr, screenshots_dir, "active_window", False, pc_state, emit_event)
                if "error" in snap:
                    return snap["error"]
                png_path = pc_state.get("last_snap_path") or str(screenshots_dir / Path(snap["url"]).name)
                hits = await asyncio.to_thread(fast_find_text, wayland_ui, png_path, needle)
                if not hits:
                    return f"Ekranda '{needle}' metni bulunamadı. Güncel öğeler:\n{snap['elements_summary']}"
            occ = max(1, min(len(hits), int(args.get("occurrence") or 1)))
            cx, cy = hits[occ - 1]["x"], hits[occ - 1]["y"]
            target_desc = f"'{hits[occ - 1]['text']}' ({occ}/{len(hits)})"
            pc_state["last_opened"] = hits[occ - 1]["text"]
        elif mark is not None:
            marks = pc_state.get("marks") or []
            idx = int(mark)
            if not (1 <= idx <= len(marks)):
                return f"Geçersiz `mark={idx}`. Mevcut aralık: 1..{len(marks)}."
            cx, cy = marks[idx - 1]["xy"]
            target_desc = f"mark #{idx} ({marks[idx - 1]['label']})"
        elif args.get("x") is not None and args.get("y") is not None:
            cx, cy = int(args["x"]), int(args["y"])
            target_desc = f"({cx},{cy})"
        else:
            return "Hata: `text`, `mark` veya `x`+`y` belirtmelisin."

        ox, oy = pc_state.get("shot_origin", (0, 0))
        btn = str(args.get("button") or "left")
        dbl = bool(args.get("double", False))

        def _do_click() -> Dict[str, Any]:
            with transient_target_focus(wayland_ui, pc_state) as focused_win:
                eff_ox, eff_oy = ox, oy
                if focused_win and focused_win.get("at"):
                    eff_ox, eff_oy = int(focused_win["at"][0]), int(focused_win["at"][1])
                return fast_click(wayland_ui, eff_ox + cx, eff_oy + cy, btn, dbl)

        clk = await asyncio.to_thread(_do_click)
        if clk.get("error"):
            return f"Tıklama başarısız: {clk['error']}"
        await asyncio.sleep(0.20)
        after = await asyncio.to_thread(_grab_screen_with_ocr, screenshots_dir, "active_window", False, pc_state, emit_event)
        return f"Tıklandı: {target_desc}.\nSonraki ekran ({after.get('focused_window', '')}):\n{after.get('elements_summary', '')}"

    if name == "pc_type_text":
        text_arg = str(args.get("text") or "")
        target_app = str(args.get("target_app") or "").strip()
        pinned = pc_state.get("target_window") or {}
        pinned_id = f"{pinned.get('class', '')} {pinned.get('title', '')}".lower()
        use_pinned = bool(target_app) and bool(pinned.get("address")) and target_app.lower() in pinned_id

        if not use_pinned:
            active = await asyncio.to_thread(wayland_ui.get_active_window) or {}
            active_id = f"{active.get('class', '')} {active.get('title', '')}".lower()
            if not target_app or target_app.lower() not in active_id:
                return (
                    f"Yazma reddedildi: Odaktaki pencere '{active.get('class') or 'yok'}' ({active.get('title') or '-'}), "
                    f"beklenen '{target_app}'. Önce `pc_focus_or_launch` ile hedef uygulamayı hazırla."
                )

        def _do_type() -> bool:
            with transient_target_focus(wayland_ui, pc_state):
                if not args.get("append"):
                    wayland_ui.clear_focused_field()
                return wayland_ui.simulate_typing(text_arg)

        ok = await asyncio.to_thread(_do_type)
        if not ok:
            return "Metin yazılamadı (`wl-copy` / `ydotool` kontrol et)."
        pc_state["draft_pending"] = True
        await asyncio.sleep(0.2)
        after = await asyncio.to_thread(_grab_screen_with_ocr, screenshots_dir, "active_window", False, pc_state, emit_event)
        return (
            f"Metin yazıldı (HENÜZ GÖNDERİLMEDİ): '{text_arg[:60]}'.\n"
            f"Alıcı doğruysa `pc_press_key(key='Return', confirmed_recipient='...')` çağır.\n"
            f"Güncel ekran:\n{after.get('elements_summary', '')}"
        )

    if name == "pc_press_key":
        key_arg = str(args.get("key") or "").strip()
        mods = [str(m) for m in (args.get("modifiers") or [])]
        recipient = str(args.get("confirmed_recipient") or "").strip()
        is_send = key_arg.lower() in ("return", "enter") and bool(pc_state.get("draft_pending"))
        if is_send and not recipient:
            return "Gönderim engellendi: Yazılmış taslağı göndermek için `confirmed_recipient` alanına alıcı/sohbet başlığını yazmalısın."
        repeat = max(1, min(15, int(args.get("repeat") or 1)))

        def _do_keys() -> bool:
            all_ok = True
            with transient_target_focus(wayland_ui, pc_state):
                for _ in range(repeat):
                    all_ok = wayland_ui.simulate_key(key_arg, mods or None) and all_ok
            return all_ok

        ok = await asyncio.to_thread(_do_keys)
        if is_send and ok:
            pc_state["draft_pending"] = False
        return f"Tuş gönderildi: {'+'.join(mods + [key_arg])} x{repeat} (başarılı={ok})."

    if name == "pc_terminal_command":
        cmd = str(args.get("command") or "").strip()
        if not cmd:
            return "Komut boş olamaz."
        cwd_path = Path(str(args.get("cwd") or workspace_dir or Path.home())).expanduser().resolve(strict=False)
        risk = _ROUTER.evaluate_risk(prompt=cmd)
        op_kind = "exec_critical" if risk.requires_approval else "exec"
        risk_badge = "KRİTİK / YIKICI KOMUT" if risk.requires_approval else "Terminal Komutu"
        allowed = await broker.ensure_permission(
            session_id=session_id,
            operation=op_kind,
            target=cwd_path,
            workspace_dir=workspace_dir or None,
            summary=f"{risk_badge} ({cwd_path}): {cmd}",
            diff_preview=f"$ cd {cwd_path}\n$ {cmd}",
            emit_event=emit_event,
        )
        if not allowed:
            return f"Kullanıcı terminal komutunu onaylamadı: `{cmd}`"

        def _run_cmd() -> str:
            proc = subprocess.run(
                cmd,
                shell=True,
                cwd=str(cwd_path) if cwd_path.is_dir() else str(Path.home()),
                capture_output=True,
                text=True,
                timeout=45,
            )
            out = (proc.stdout or "").strip()
            err = (proc.stderr or "").strip()
            combined = "\n".join(x for x in (out, f"[stderr]\n{err}" if err else "") if x).strip()
            if len(combined) > 6_000:
                combined = combined[:6_000] + "\n…(çıktı kısaltıldı)"
            return f"Çıkış Kodu: {proc.returncode}\n{combined or '(çıktı yok)'}"

        return await asyncio.to_thread(_run_cmd)

    if name == "hermes_skill":
        skill = str(args.get("name") or "").strip()
        query = str(args.get("query") or "").strip()
        if not skill:
            res = await asyncio.to_thread(agent_bridge.search, query, 10)
            return json.dumps(res, ensure_ascii=False, indent=2)
        skill_args = args.get("arguments") or {}
        if skill in agent_bridge.HIGH_RISK_TOOLS:
            allowed = await broker.ensure_permission(
                session_id=session_id,
                operation="exec_critical",
                target=Path(workspace_dir or Path.home()),
                workspace_dir=workspace_dir or None,
                summary=f"Hermes Aracı: {skill}",
                diff_preview=json.dumps(skill_args, ensure_ascii=False, indent=2)[:800],
                emit_event=emit_event,
            )
            if not allowed:
                return f"Kullanıcı '{skill}' Hermes aracını onaylamadı."
        res = await agent_bridge.dispatch(skill, skill_args)
        return json.dumps(res, ensure_ascii=False, indent=2)[:6_000]

    return f"Bilinmeyen Hermes aracı: {name}"
