"""
Wayland / Hyprland UI Interaction Engine for Neo
=================================================
Enables Neo to interact directly with the PC Graphical User Interface (Wayland/Hyprland)
instead of executing blind background shell commands.

Capabilities:
- Live window inventory (active windows, workspaces, PIDs)
- Smart Focus-or-Launch (brings existing apps into focus rather than spawning duplicates)
- Native Wayland shortcut execution (Rofi, Zapret, Siber Pano, Cava, Kitty, Media, Window ops)
- Virtual keyboard typing and hotkey simulation via wtype
"""

from __future__ import annotations

import functools
import json
import os
import shutil
import re
import subprocess
import time
from typing import Any, Dict, List, Optional, Union

HYPRCTL_BIN = "/usr/bin/hyprctl"
WTYPE_BIN = "/usr/bin/wtype"
YDOTOOL_BIN = "/usr/bin/ydotool"
TESSERACT_BIN = "/usr/bin/tesseract"
WL_COPY_BIN = "/usr/bin/wl-copy"
WL_PASTE_BIN = "/usr/bin/wl-paste"

USER_HOME = os.path.expanduser("~")
HYPR_SCRIPTS_DIR = os.path.join(USER_HOME, ".config", "hypr", "scripts")
# ydotoold runs as a user service on a per-user socket; the client defaults to
# /tmp/.ydotool_socket and would otherwise never find it.
YDOTOOL_SOCKET = os.path.join(USER_HOME, ".ydotool_socket")
ROFI_DIR = os.path.join(USER_HOME, ".config", "rofi")

# Known application binary / class aliases
APP_MAP = {
    "zapzap": {"class": "com.rtosta.zapzap", "exec": "zapzap"},
    "whatsapp": {"class": "com.rtosta.zapzap", "exec": "zapzap"},
    "antigravity": {"class": "antigravity-ide", "exec": "antigravity"},
    "ide": {"class": "antigravity-ide", "exec": "antigravity"},
    "chatgpt": {"class": "chatgpt", "exec": "chatgpt"},
    "signal": {"class": "signal", "exec": "signal-desktop"},
    "kitty": {"class": "kitty", "exec": "kitty"},
    "terminal": {"class": "kitty", "exec": "kitty"},
    "chrome": {"class": "google-chrome", "exec": "google-chrome-stable"},
    "tarayıcı": {"class": "google-chrome", "exec": "google-chrome-stable"},
    "browser": {"class": "google-chrome", "exec": "google-chrome-stable"},
    "spotify": {"class": "Spotify", "exec": "spotify"},
    "discord": {"class": "discord", "exec": "discord"},
    "cava": {"class": "cava", "exec": f"kitty --class cava --session {USER_HOME}/.config/cava/kitty_session.conf"},
}


def run_hyprctl_dispatch(lua_dispatcher_call: str) -> Dict[str, Any]:
    """Issues a dispatcher call to Hyprland's Lua IPC engine."""
    cmd = [HYPRCTL_BIN, "dispatch", lua_dispatcher_call]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        success = res.returncode == 0
        return {
            "success": success,
            "stdout": res.stdout.strip(),
            "stderr": res.stderr.strip(),
            "command": " ".join(cmd),
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def list_windows() -> List[Dict[str, Any]]:
    """Returns a list of all currently open Wayland client windows."""
    cmd = [HYPRCTL_BIN, "clients", "-j"]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        if res.returncode != 0:
            return []
        data = json.loads(res.stdout)
        windows = []
        for item in data:
            windows.append({
                "address": item.get("address", ""),
                "class": item.get("class", ""),
                "title": item.get("title", ""),
                "workspace_id": item.get("workspace", {}).get("id", 0),
                "workspace_name": str(item.get("workspace", {}).get("name", "")),
                "pid": item.get("pid", 0),
                "floating": item.get("floating", False),
            })
        return windows
    except Exception:
        return []


def get_active_window() -> Optional[Dict[str, Any]]:
    """Returns details of the currently focused window."""
    cmd = [HYPRCTL_BIN, "activewindow", "-j"]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        if res.returncode == 0 and res.stdout.strip():
            return json.loads(res.stdout)
    except Exception:
        pass
    return None


def focus_window_by_query(query: str) -> bool:
    """
    Attempts to focus a window matching class, title, or address.
    Uses Hyprland 0.56+ Lua dispatch syntax: hl.dsp.focus({ window = '...' }).
    """
    q_lower = query.lower().strip()
    windows = list_windows()

    # 1. Exact or partial class match
    target = None
    for w in windows:
        if q_lower == w["class"].lower() or q_lower in w["class"].lower():
            target = w
            break

    # 2. Title match
    if not target:
        for w in windows:
            if q_lower in w["title"].lower():
                target = w
                break

    if target:
        # Switch to workspace first if needed
        ws_id = target.get("workspace_id")
        if ws_id and ws_id > 0:
            run_hyprctl_dispatch(f"hl.dsp.focus({{ workspace = {ws_id} }})")

        # Focus window via class
        res = run_hyprctl_dispatch(f"hl.dsp.focus({{ window = 'class:{target['class']}' }})")
        return res.get("success", False)

    # Fallback to direct string query
    res = run_hyprctl_dispatch(f"hl.dsp.focus({{ window = '{query}' }})")
    return res.get("success", False)


def launch_via_desktop_ui(app_name: str, fallback_exec: Optional[str] = None) -> Dict[str, Any]:
    """
    Launches an application through the user's real desktop UI (UI-First):
    1. Triggers the user's actual desktop launcher (ALT + SPACE / Rofi drun).
    2. Searches/types the app name in Rofi.
    3. Selects and launches via Return.
    4. Falls back to Hyprland compositor exec if launcher is unavailable.
    """
    rofi_theme = f"{ROFI_DIR}/launchers/type-1/style-1.rasi"
    if os.path.exists(rofi_theme) and os.path.exists("/usr/bin/rofi"):
        # UI-First: launch via Rofi desktop launcher search
        # -x so the pattern does not also match unrelated processes (power-profiles-daemon
        # contains "rofi"); double quotes so the filter does not break the nested quoting
        # of hyprctl's exec_cmd argument.
        safe_name = re.sub(r'[^\w .-]', '', app_name)[:40]
        cmd = f'pkill -x rofi; rofi -show drun -theme {rofi_theme} -filter "{safe_name}"' 
        run_hyprctl_dispatch(f"hl.dsp.exec_cmd('{cmd}')")
        # Wait for Rofi to actually map before confirming. The old code slept 0.3s and
        # sent Return through wtype, which Rofi never receives — so nothing launched.
        if not _wait_for_rofi():
            return {
                "action": "rofi_did_not_open",
                "app": app_name,
                "success": False,
                "hint": "Rofi acilmadi. execute_terminal_command ile dogrudan calistirmayi dene.",
            }
        # Rofi is filtered to the app name and its first entry is the primary .desktop
        # match, so confirm it here. Leaving the choice to the model meant it pressed
        # arrows and then typed into a launcher that owns the keyboard. Enter goes via
        # uinput (wtype does not reach Rofi) once the surface is actually mapped.
        time.sleep(0.4)
        selected = press_key_uinput(28)  # KEY_ENTER
        return {
            "action": "launched_via_rofi_ui",
            "app": app_name,
            "success": selected,
            "hint": ("Rofi'de ilk eslesme secildi. Pencere acilmazsa capture_screenshot al; "
                     "Rofi hala aciksa ui_press_key ile 'Down'/'Return' kullan "
                     "(Rofi fare olaylarini ALMAZ)."),
        }

    target_exec = fallback_exec or app_name
    res = run_hyprctl_dispatch(f"hl.dsp.exec_cmd('{target_exec}')")
    return {
        "action": "launched_via_compositor",
        "app": app_name,
        "command": target_exec,
        "success": res.get("success", False),
    }


def focus_or_launch(app_name: str, fallback_exec: Optional[str] = None) -> Dict[str, Any]:
    """
    Smart Focus-or-Launch (UI-First Policy):
    1. Checks if the application window is already running on any workspace.
    2. If found, switches to that workspace and focuses the existing window (prevents duplicates).
    3. If not found, launches through the desktop UI (ALT+SPACE -> Rofi search -> Enter).
    """
    app_key = app_name.lower().strip()
    meta = APP_MAP.get(app_key)

    target_class = meta["class"] if meta else app_key
    target_exec = fallback_exec or (meta["exec"] if meta else app_name)

    windows = list_windows()
    for w in windows:
        if target_class.lower() in w["class"].lower() or app_key in w["title"].lower():
            ws_id = w.get("workspace_id", 1)
            # Switch to workspace & focus window
            run_hyprctl_dispatch(f"hl.dsp.focus({{ workspace = {ws_id} }})")
            run_hyprctl_dispatch(f"hl.dsp.focus({{ window = 'class:{w['class']}' }})")
            return {
                "action": "focused_existing",
                "app": app_name,
                "class": w["class"],
                "title": w["title"],
                "workspace": ws_id,
            }

    # Not running -> move to a free workspace first so the new window does not land on
    # top of whatever the user is currently working in.
    free_ws = find_free_workspace()
    if free_ws is not None:
        run_hyprctl_dispatch(f"hl.dsp.focus({{ workspace = {free_ws} }})")
        time.sleep(0.3)

    res = launch_via_desktop_ui(app_name, target_exec)
    res["target_workspace"] = free_ws
    if free_ws is None:
        res["workspaces_full"] = True
        res["hint_workspaces"] = ("Bos workspace kalmadi; uygulama mevcut workspace'te acilacak. "
                                  "notify_user ile kullaniciya haber ver.")
    w = _wait_for_window(target_class, timeout_s=12.0) or _wait_for_window(app_key, timeout_s=0.1)
    if w:
        run_hyprctl_dispatch(f"hl.dsp.focus({{ workspace = {w.get('workspace_id', 1)} }})")
        run_hyprctl_dispatch(f"hl.dsp.focus({{ window = 'class:{w['class']}' }})")
        time.sleep(0.4)
        res.update({
            "action": "launched_and_focused",
            "class": w["class"],
            "title": w["title"],
            "workspace": w.get("workspace_id", 1),
        })
        return res
    res["warning"] = (f"'{app_name}' penceresi 12sn icinde gorunmedi. Rofi hala acik olabilir: "
                      f"capture_screenshot ile bak ve dogru girdiye ui_click ile tikla.")
    return res


def trigger_shortcut(action: str, **kwargs) -> Dict[str, Any]:
    """
    Executes or simulates the user's bound Wayland/Hyprland shortcut.
    Supported actions:
    - rofi / app_launcher: ALT + SPACE
    - terminal: CTRL + ALT + T (kitty)
    - zapret / toggle_zapret: SUPER + Z (~/.config/hypr/scripts/toggle-zapret.sh)
    - siber_pano / clipboard: SUPER + C (~/.config/hypr/scripts/siber-pano.sh)
    - cava: SUPER + A
    - lock: SUPER + S (hyprlock)
    - waybar_reload: SUPER + W
    - screenshot: SUPER + SHIFT + S (grim + slurp + swappy)
    - screen_record: SHIFT + code:107 (~/.config/hypr/scripts/siber-kayit.sh)
    - close_window: SUPER + Q
    - toggle_floating: SUPER + V
    - fullscreen: SUPER + F
    - split_layout: SUPER + P
    - workspace: SUPER + 1..10
    - media_play_pause: playerctl play-pause
    - media_next: playerctl next
    - media_prev: playerctl previous
    - volume_up: wpctl set-volume -l 1.5 @DEFAULT_AUDIO_SINK@ 5%+
    - volume_down: wpctl set-volume @DEFAULT_AUDIO_SINK@ 5%-
    - volume_mute: wpctl set-mute @DEFAULT_AUDIO_SINK@ toggle
    """
    action_key = action.lower().replace("-", "_").strip()

    if action_key in ("rofi", "app_launcher", "launcher"):
        rofi_theme = f"{ROFI_DIR}/launchers/type-1/style-1.rasi"
        cmd = f"pkill rofi; rofi -show drun -theme {rofi_theme}"
        res = run_hyprctl_dispatch(f"hl.dsp.exec_cmd('{cmd}')")
        return {"shortcut": "ALT+SPACE", "description": "Rofi Application Launcher triggered", "result": res}

    if action_key in ("terminal", "kitty"):
        res = run_hyprctl_dispatch("hl.dsp.exec_cmd('kitty')")
        return {"shortcut": "CTRL+ALT+T", "description": "Kitty Terminal launched", "result": res}

    if action_key in ("zapret", "toggle_zapret", "zapzap_bypass", "bypass"):
        script = f"{HYPR_SCRIPTS_DIR}/toggle-zapret.sh"
        res = run_hyprctl_dispatch(f"hl.dsp.exec_cmd('{script}')")
        return {"shortcut": "SUPER+Z", "description": "Zapret DPI bypass toggled", "result": res}

    if action_key in ("siber_pano", "clipboard", "pano"):
        script = f"{HYPR_SCRIPTS_DIR}/siber-pano.sh"
        res = run_hyprctl_dispatch(f"hl.dsp.exec_cmd('{script}')")
        return {"shortcut": "SUPER+C", "description": "Siber Pano clipboard manager opened", "result": res}

    if action_key in ("cava", "visualizer"):
        cmd = f"pkill -x cava || kitty --class cava --session {USER_HOME}/.config/cava/kitty_session.conf"
        res = run_hyprctl_dispatch(f"hl.dsp.exec_cmd('{cmd}')")
        return {"shortcut": "SUPER+A", "description": "Cava audio visualizer toggled", "result": res}

    if action_key in ("lock", "lock_screen"):
        res = run_hyprctl_dispatch("hl.dsp.exec_cmd('hyprlock')")
        return {"shortcut": "SUPER+S", "description": "Hyprlock screen lock triggered", "result": res}

    if action_key in ("waybar", "waybar_reload"):
        cmd = "pkill -x waybar; pkill -x waybar-lua-bin; /home/zwannfrederick/.local/bin/waybar"
        res = run_hyprctl_dispatch(f"hl.dsp.exec_cmd('{cmd}')")
        return {"shortcut": "SUPER+W", "description": "Waybar reloaded", "result": res}

    if action_key in ("screenshot", "screen_capture"):
        cmd = 'grim -g "$(slurp)" - | swappy -f -'
        res = run_hyprctl_dispatch(f"hl.dsp.exec_cmd('{cmd}')")
        return {"shortcut": "SUPER+SHIFT+S", "description": "Interactive screenshot tool opened", "result": res}

    if action_key in ("close_window", "killactive", "close"):
        res = run_hyprctl_dispatch("hl.dsp.window.close()")
        return {"shortcut": "SUPER+Q", "description": "Active window closed", "result": res}

    if action_key in ("toggle_floating", "float"):
        res = run_hyprctl_dispatch("hl.dsp.window.float({ action = 'toggle' })")
        return {"shortcut": "SUPER+V", "description": "Window floating mode toggled", "result": res}

    if action_key in ("fullscreen", "toggle_fullscreen"):
        res = run_hyprctl_dispatch("hl.dsp.window.fullscreen({ mode = 0 })")
        return {"shortcut": "SUPER+F", "description": "Window fullscreen toggled", "result": res}

    if action_key in ("split_layout", "layout_split"):
        res = run_hyprctl_dispatch("hl.dsp.layout('togglesplit')")
        return {"shortcut": "SUPER+P", "description": "Layout split direction toggled", "result": res}

    if action_key.startswith("workspace"):
        ws = kwargs.get("workspace_id")
        if not ws:
            # Extract number from string like "workspace_2"
            digits = "".join(filter(str.isdigit, action_key))
            ws = int(digits) if digits else 1
        res = run_hyprctl_dispatch(f"hl.dsp.focus({{ workspace = {ws} }})")
        return {"shortcut": f"SUPER+{ws}", "description": f"Switched to workspace {ws}", "result": res}

    # Media and Volume
    if action_key in ("play_pause", "media_play_pause"):
        subprocess.run(["playerctl", "play-pause"], check=False)
        return {"action": "media_play_pause", "status": "executed"}

    if action_key in ("media_next", "next_track"):
        subprocess.run(["playerctl", "next"], check=False)
        return {"action": "media_next", "status": "executed"}

    if action_key in ("media_prev", "prev_track"):
        subprocess.run(["playerctl", "previous"], check=False)
        return {"action": "media_prev", "status": "executed"}

    if action_key in ("volume_up", "raise_volume"):
        subprocess.run(["wpctl", "set-volume", "-l", "1.5", "@DEFAULT_AUDIO_SINK@", "5%+"], check=False)
        return {"action": "volume_up", "status": "executed"}

    if action_key in ("volume_down", "lower_volume"):
        subprocess.run(["wpctl", "set-volume", "@DEFAULT_AUDIO_SINK@", "5%-"], check=False)
        return {"action": "volume_down", "status": "executed"}

    if action_key in ("volume_mute", "mute"):
        subprocess.run(["wpctl", "set-mute", "@DEFAULT_AUDIO_SINK@", "toggle"], check=False)
        return {"action": "volume_mute", "status": "executed"}

    return {"error": f"Unknown shortcut action: {action}"}


def simulate_typing(text: str) -> bool:
    """
    Types text into whatever currently holds keyboard focus.

    Goes through the Wayland clipboard + Ctrl+V rather than synthesising keystrokes:
    ydotool's `type` maps characters onto US-layout keycodes, so on this box (layout
    `trq`) every non-ASCII character was dropped and punctuation shifted -- "durağına
    yürüyor." arrived as "durana yryorç". Pasting is layout-independent and Unicode-safe.
    The clipboard is restored afterwards so the user's own copy buffer survives.
    """
    if not text:
        return True
    if not os.path.exists(WL_COPY_BIN) or not os.path.exists(YDOTOOL_BIN):
        return _simulate_typing_keystrokes(text)

    saved: Optional[str] = None
    try:
        r = subprocess.run([WL_PASTE_BIN, "--no-newline"], capture_output=True,
                           text=True, timeout=3)
        if r.returncode == 0:
            saved = r.stdout
    except Exception:
        pass

    try:
        subprocess.run([WL_COPY_BIN], input=text, text=True, timeout=5, check=True)
    except Exception:
        return _simulate_typing_keystrokes(text)

    time.sleep(0.08)  # give the focused client a moment to see the new selection
    ok = simulate_key("v", ["ctrl"])
    time.sleep(0.12)  # paste must land before the clipboard is handed back

    if saved is not None:
        try:
            subprocess.run([WL_COPY_BIN], input=saved, text=True, timeout=5)
        except Exception:
            pass
    return ok


def _simulate_typing_keystrokes(text: str) -> bool:
    """Keystroke fallback for when the clipboard path is unavailable. ASCII-only in
    practice: see simulate_typing for why this mangles non-US-layout characters."""
    if os.path.exists(YDOTOOL_BIN):
        try:
            env = {**os.environ, "YDOTOOL_SOCKET": YDOTOOL_SOCKET}
            r = subprocess.run([YDOTOOL_BIN, "type", "--key-delay", "12", "--", text],
                               capture_output=True, text=True, timeout=30, env=env)
            if r.returncode == 0:
                return True
        except Exception:
            pass
    if not os.path.exists(WTYPE_BIN):
        return False
    try:
        subprocess.run([WTYPE_BIN, text], check=True, timeout=5)
        return True
    except Exception:
        return False


def simulate_key(key: str, modifiers: Optional[List[str]] = None) -> bool:
    """
    Presses a key with optional modifiers, e.g. key='f', modifiers=['ctrl'].

    Uses ydotool/uinput so the event reaches Rofi, Electron apps and compositor
    keybindings alike; falls back to wtype when ydotool is missing.
    """
    code = KEYCODES.get(key.strip().lower())
    if code and os.path.exists(YDOTOOL_BIN):
        mods = [MODCODES[m.lower()] for m in (modifiers or []) if m.lower() in MODCODES]
        seq = [f"{m}:1" for m in mods] + [f"{code}:1", f"{code}:0"] + \
              [f"{m}:0" for m in reversed(mods)]
        try:
            env = {**os.environ, "YDOTOOL_SOCKET": YDOTOOL_SOCKET}
            r = subprocess.run([YDOTOOL_BIN, "key"] + seq,
                               capture_output=True, text=True, timeout=10, env=env)
            if r.returncode == 0:
                return True
        except Exception:
            pass
    if not os.path.exists(WTYPE_BIN):
        return False
    cmd = [WTYPE_BIN]
    if modifiers:
        for m in modifiers:
            cmd.extend(["-M", m])
    cmd.extend(["-k", key])
    if modifiers:
        for m in modifiers:
            cmd.extend(["-m", m])
    try:
        subprocess.run(cmd, check=True, timeout=5)
        return True
    except Exception:
        return False


def get_cursor_pos() -> Optional[Tuple[int, int]]:
    """Current pointer position as reported by Hyprland."""
    try:
        r = subprocess.run([HYPRCTL_BIN, "cursorpos"], capture_output=True, text=True, timeout=5)
        x, y = r.stdout.strip().split(",")
        return int(x), int(y)
    except Exception:
        return None


# Linux input-event-codes for the keys an agent actually needs. wtype speaks the
# Wayland virtual-keyboard protocol, which Rofi and Electron apps ignore; ydotool
# injects at uinput so it reaches every client.
KEYCODES = {
    "escape": 1, "esc": 1, "backspace": 14, "tab": 15, "return": 28, "enter": 28,
    "space": 57, "delete": 111, "insert": 110,
    "up": 103, "down": 108, "left": 105, "right": 106,
    "home": 102, "end": 107, "pageup": 104, "pagedown": 109,
    "minus": 12, "equal": 13, "comma": 51, "dot": 52, "period": 52, "slash": 53,
    "semicolon": 39, "apostrophe": 40, "grave": 41, "backslash": 43,
    "leftbracket": 26, "rightbracket": 27,
}
for _i, _c in enumerate("abcdefghijklmnopqrstuvwxyz"):
    KEYCODES[_c] = [30, 48, 46, 32, 18, 33, 34, 35, 23, 36, 37, 38, 50,
                    49, 24, 25, 16, 19, 31, 20, 22, 47, 17, 45, 21, 44][_i]
for _i, _c in enumerate("1234567890"):
    KEYCODES[_c] = 2 + _i
for _i in range(1, 11):
    KEYCODES[f"f{_i}"] = 58 + _i
KEYCODES["f11"], KEYCODES["f12"] = 87, 88

MODCODES = {"ctrl": 29, "control": 29, "shift": 42, "alt": 56,
            "logo": 125, "super": 125, "meta": 125}


def press_key_uinput(keycode: int) -> bool:
    """Presses one key via ydotool (uinput). Reaches clients that ignore wtype's
    Wayland virtual-keyboard protocol, such as Rofi and Electron apps."""
    if not os.path.exists(YDOTOOL_BIN):
        return False
    try:
        env = {**os.environ, "YDOTOOL_SOCKET": YDOTOOL_SOCKET}
        r = subprocess.run([YDOTOOL_BIN, "key", f"{keycode}:1", f"{keycode}:0"],
                           capture_output=True, text=True, timeout=5, env=env)
        return r.returncode == 0
    except Exception:
        return False


def find_free_workspace(max_id: int = 10) -> Optional[int]:
    """
    Lowest workspace id (1..max_id) that currently holds no windows.

    New apps go to an empty workspace so they never land on top of what the user is
    already doing. Returns None when every workspace is occupied.
    """
    occupied = {w.get("workspace_id") for w in list_windows()}
    for wid in range(1, max_id + 1):
        if wid not in occupied:
            return wid
    return None


def _rofi_is_open() -> bool:
    """Rofi draws as a layer-shell surface, so it never appears in `hyprctl clients`."""
    try:
        res = subprocess.run([HYPRCTL_BIN, "layers", "-j"], capture_output=True, text=True, timeout=5)
        return "rofi" in res.stdout.lower()
    except Exception:
        return False


def _wait_for_rofi(timeout_s: float = 4.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if _rofi_is_open():
            return True
        time.sleep(0.25)
    return False


def _wait_for_window(match: str, timeout_s: float = 10.0) -> Optional[Dict[str, Any]]:
    """Polls Hyprland until a window whose class/title matches appears."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        for w in list_windows():
            if match.lower() in w["class"].lower() or match.lower() in w["title"].lower():
                return w
        time.sleep(0.4)
    return None


def click_at(x: int, y: int, button: str = "left", double: bool = False) -> Dict[str, Any]:
    """
    Moves the pointer to absolute screen coordinates and clicks, via ydotool.

    wtype only speaks the Wayland virtual-keyboard protocol, which some toolkits
    (notably Electron) ignore. ydotool injects at the kernel uinput layer, so it
    reaches every client regardless of toolkit.
    """
    if not os.path.exists(YDOTOOL_BIN):
        return {"error": "ydotool kurulu degil ('sudo pacman -S ydotool'). Tiklama yapilamiyor."}
    codes = {"left": "0xC0", "right": "0xC1", "middle": "0xC2"}
    code = codes.get(button, "0xC0")
    env = {**os.environ, "YDOTOOL_SOCKET": YDOTOOL_SOCKET}
    try:
        # ydotool's --absolute mode reports success but does not move the pointer on this
        # setup, so steer by relative deltas against Hyprland's own cursor reading and
        # converge (a single delta can undershoot with pointer acceleration).
        target = (int(x), int(y))
        for _ in range(6):
            cur = get_cursor_pos()
            if cur is None:
                return {"error": "Imlec konumu okunamadi (hyprctl cursorpos)."}
            dx, dy = target[0] - cur[0], target[1] - cur[1]
            if abs(dx) <= 2 and abs(dy) <= 2:
                break
            move = subprocess.run(
                [YDOTOOL_BIN, "mousemove", "-x", str(dx), "-y", str(dy)],
                capture_output=True, text=True, timeout=5, env=env,
            )
            if move.returncode != 0:
                return {"error": f"ydotool mousemove basarisiz: {move.stderr.strip() or move.returncode}. "
                                 f"ydotoold calisiyor mu?"}
            time.sleep(0.08)
        landed = get_cursor_pos()
        if landed and (abs(landed[0] - target[0]) > 6 or abs(landed[1] - target[1]) > 6):
            return {"error": f"Imlec hedefe ulasamadi: istenen {target}, ulasilan {landed}."}
        # Press and release are sent separately with a dwell between them: the combined
        # 0xC0 code lands press+release in the same instant, which QtWebEngine chat rows
        # (ZapZap) register as a hover rather than a click -- the pointer went to exactly
        # the right pixel and the conversation never opened.
        down = hex(int(code, 16) & 0x4F)
        up = hex(int(code, 16) & 0x8F)
        time.sleep(0.12)  # let hover state settle before pressing
        for _ in range(2 if double else 1):
            for stage, pause in ((down, 0.07), (up, 0.05)):
                clk = subprocess.run([YDOTOOL_BIN, "click", stage], capture_output=True,
                                     text=True, timeout=5, env=env)
                if clk.returncode != 0:
                    return {"error": f"ydotool click basarisiz: "
                                     f"{clk.stderr.strip() or clk.returncode}"}
                time.sleep(pause)
        return {"clicked": True, "x": int(x), "y": int(y), "landed": landed,
                "button": button, "double": double}
    except Exception as e:
        return {"error": str(e)}


def inspect_accessibility_tree(app_name: Optional[str] = None, max_depth: int = 3) -> Dict[str, Any]:
    """
    Inspects the native AT-SPI accessibility tree of active desktop applications.
    Enables semantic UI understanding (roles, names, descriptions, child hierarchies)
    to perform UI automation via semantic element selection rather than blind coordinate guessing.
    """
    try:
        import gi
        gi.require_version("Atspi", "2.0")
        from gi.repository import Atspi
    except Exception as e:
        return {"success": False, "error": f"AT-SPI not available: {e}", "elements": []}

    def _dump_node(acc, depth: int):
        if depth > max_depth or not acc:
            return None
        try:
            role = acc.get_role_name() or "unknown"
            name = acc.get_name() or ""
            desc = acc.get_description() or ""
            count = acc.get_child_count()
        except Exception:
            return None

        node: Dict[str, Any] = {"role": role, "name": name}
        if desc:
            node["description"] = desc

        if depth < max_depth and count > 0:
            children = []
            for i in range(min(10, count)):
                c = acc.get_child_at_index(i)
                cd = _dump_node(c, depth + 1)
                if cd:
                    children.append(cd)
            if children:
                node["children"] = children
        return node

    try:
        desktop = Atspi.get_desktop(0)
        app_count = desktop.get_child_count()
        results = []

        for i in range(app_count):
            app = desktop.get_child_at_index(i)
            if not app:
                continue
            app_name_current = app.get_name() or ""
            if app_name:
                if app_name.lower() not in app_name_current.lower():
                    continue
            dumped = _dump_node(app, depth=0)
            if dumped:
                results.append(dumped)

        return {
            "success": True,
            "target": app_name or "all_accessible_apps",
            "count": len(results),
            "elements": results,
        }
    except Exception as e:
        return {"success": False, "error": str(e), "elements": []}


@functools.lru_cache(maxsize=1)
def _ocr_lang() -> str:
    """Best available tesseract language set. Asking for a language whose traineddata is
    missing makes tesseract exit non-zero, which would silently disable OCR entirely, so
    the request is narrowed to what is actually installed."""
    try:
        r = subprocess.run([TESSERACT_BIN, "--list-langs"],
                           capture_output=True, text=True, timeout=10)
        have = {ln.strip() for ln in r.stdout.splitlines()[1:] if ln.strip()}
    except Exception:
        return "eng"
    wanted = [l for l in ("tur", "eng") if l in have]
    if wanted:
        return "+".join(wanted)
    # Any latin-alphabet model still reads UI labels well enough to locate them.
    return next((l for l in sorted(have) if l != "osd"), "eng")


def find_text(image_path: str, needle: str, lang: Optional[str] = None) -> List[Dict[str, Any]]:
    """Locates *needle* in a screenshot via OCR, returning boxes in that image's pixels.

    The model cannot localise UI elements in a 1868x988 frame -- given "open the Kardemir
    COO chat" it emitted round guesses (100,450), (150,550) and burned the whole step
    budget without ever hitting the row. OCR answers "where is this text" deterministically,
    for any application, with no accessibility tree and no per-app coordinates.

    Words are stitched back into lines because a chat title spans several OCR words, and
    matching happens on the normalised line so case and Turkish diacritics do not matter.
    """
    if not needle.strip():
        return []
    lines = _ocr_lines(image_path, lang)

    # Exact first, then a pass that treats OCR's habitual confusions as equal: tesseract
    # reads "Otoparçasan" as "Otopargasan" at 89% confidence -- the cedilla looks like a
    # descender at UI point sizes, and no language pack fixes that.
    hits = _match_lines(lines, needle, _fold)
    return hits if hits else _match_lines(lines, needle, _fold_loose)


def _match_lines(lines, needle: str, fold) -> List[Dict[str, Any]]:
    target = fold(needle)
    hits: List[Dict[str, Any]] = []
    for words in lines:
        # Match against the whole line, then click only the words that actually matched:
        # a chat row spans the full column, and its centre can sit far from a short label.
        span = _matching_span(words, target, fold)
        if span is None:
            continue
        matched = words[span[0]:span[1]]
        x0 = min(w["box"][0] for w in matched)
        y0 = min(w["box"][1] for w in matched)
        x1 = max(w["box"][2] for w in matched)
        y1 = max(w["box"][3] for w in matched)
        hits.append({
            "text": " ".join(w["text"] for w in matched),
            "line": " ".join(w["text"] for w in words),
            "x": (x0 + x1) // 2,
            "y": (y0 + y1) // 2,
            "box": [x0, y0, x1, y1],
            "confidence": round(sum(w["conf"] for w in matched) / len(matched), 1),
        })
    # Leftmost first, not topmost. Desktop apps are master-detail: the list you act on is
    # the left column, while the same name in the right pane is the open item's header --
    # clicking that opened an info panel four runs in a row, stealing keyboard focus.
    hits.sort(key=lambda h: (h["x"], h["y"]))
    return hits


def _matching_span(words: List[Dict[str, Any]], target: str, fold=None) -> Optional[tuple]:
    """Shortest contiguous run of *words* whose folded text contains *target*."""
    fold = fold or _fold
    if target not in fold(" ".join(w["text"] for w in words)):
        return None
    best = None
    for i in range(len(words)):
        for j in range(i + 1, len(words) + 1):
            if target in fold(" ".join(w["text"] for w in words[i:j])):
                if best is None or (j - i) < (best[1] - best[0]):
                    best = (i, j)
                break
    return best


_FOLD = str.maketrans("ıİşŞğĞüÜöÖçÇ", "iisSgGuUoOcC")


def _fold(text: str) -> str:
    """Casefold plus Turkish diacritic stripping, so OCR noise on 'ğ' still matches."""
    return " ".join(text.translate(_FOLD).lower().split())


# Glyphs tesseract routinely swaps for one another at UI point sizes, collapsed to one
# representative each. Only ever used as a second pass, so precision comes first.
_LOOSE = str.maketrans({c: t for t, group in
                        (("c", "cçgğq"), ("i", "iıl1|!j"), ("s", "sş5"),
                         ("o", "oö0"), ("u", "uüv"), ("n", "nñ"))
                        for c in group})


def _fold_loose(text: str) -> str:
    return " ".join(_fold(text).translate(_LOOSE).split())


def _ocr_lines(image_path: str, lang: Optional[str] = None) -> List[List[Dict[str, Any]]]:
    """OCR one image into UI labels: lists of words, each word carrying box and confidence.

    Words are grouped geometrically rather than by tesseract's own block model. A desktop
    app is several columns side by side, and the block model happily welds a sidebar row
    onto the chat header beside it, producing a box spanning the whole window -- useless
    both for matching and for pointing at.
    """
    if not os.path.exists(TESSERACT_BIN):
        return []
    try:
        r = subprocess.run(
            [TESSERACT_BIN, image_path, "stdout", "-l", lang or _ocr_lang(),
             "--psm", "11", "tsv"],
            capture_output=True, text=True, timeout=30)
    except Exception:
        return []
    if r.returncode != 0:
        return []

    words: List[Dict[str, Any]] = []
    for row in r.stdout.splitlines()[1:]:
        f = row.split("\t")
        if len(f) < 12 or not f[11].strip():
            continue
        try:
            conf = float(f[10])
            left, top, width, height = (int(f[6]), int(f[7]), int(f[8]), int(f[9]))
        except ValueError:
            continue
        # Deliberately permissive: tesseract scores short UI tokens low even when it read
        # them correctly ("coo" on a chat row came back at 14), and find_text matches an exact phrase, so
        # a stray low-confidence word costs nothing while a dropped one loses the target.
        if conf < 10 or height <= 0:
            continue
        words.append({"text": f[11].strip(), "box": [left, top, left + width, top + height],
                      "conf": conf})

    words.sort(key=lambda w: (w["box"][1], w["box"][0]))
    lines: List[List[Dict[str, Any]]] = []
    for word in words:
        x0, y0, x1, y1 = word["box"]
        height = y1 - y0
        for line in lines:
            last = line[-1]
            ly0, ly1 = last["box"][1], last["box"][3]
            same_row = abs((y0 + y1) / 2 - (ly0 + ly1) / 2) < height * 0.6
            # A single space is about a third of the cap height; allow a wide-ish gap for
            # padded UI labels but not the empty gulf between two columns.
            adjacent = 0 <= x0 - last["box"][2] < height * 1.5
            if same_row and adjacent:
                line.append(word)
                break
        else:
            lines.append([word])
    return lines




# The vision encoder takes 224x224 with 16px patches -- 196 cells for a 1868x988 frame,
# so one cell spans ~133x71 screen pixels and a whole chat row fits inside one. The model
# therefore cannot name a pixel; it can only name a thing. Numbering the things on the
# image turns "where do I click" into "which one", which is a choice it can make.
ICON_MIN_PX = 14
ICON_MAX_PX = 96
_EDGE_DOWNSCALE = 4


def detect_elements(image_path: str, max_elements: int = 100) -> List[Dict[str, Any]]:
    """Clickable candidates on screen: OCR text runs plus icon-sized graphical blobs."""
    elements: List[Dict[str, Any]] = []
    for words in _ocr_lines(image_path):
        text = " ".join(w["text"] for w in words).strip()
        mean_conf = sum(w["conf"] for w in words) / len(words)
        if len(text) < 2 or mean_conf < 45:
            continue
        x0 = min(w["box"][0] for w in words)
        y0 = min(w["box"][1] for w in words)
        x1 = max(w["box"][2] for w in words)
        y1 = max(w["box"][3] for w in words)
        elements.append({"kind": "text", "label": text[:60], "box": [x0, y0, x1, y1]})

    for box in _icon_boxes(image_path):
        if any(_overlaps(box, e["box"]) for e in elements):
            continue
        elements.append({"kind": "icon", "label": "", "box": box})

    # Truncation drops icons first: a labelled element is one the model can also name,
    # and losing the bottom half of a list is worse than losing a few decorations.
    if len(elements) > max_elements:
        texts = [e for e in elements if e["kind"] == "text"]
        icons = [e for e in elements if e["kind"] == "icon"]
        elements = texts[:max_elements] + icons[:max(0, max_elements - len(texts))]
    elements.sort(key=lambda e: (e["box"][1], e["box"][0]))
    return elements


def _overlaps(a: List[int], b: List[int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _icon_boxes(image_path: str) -> List[List[int]]:
    """Icon-sized high-contrast blobs, found by flood-filling a downscaled edge mask.

    # ponytail: plain BFS labelling on a /4 image (~115k px) because neither OpenCV nor
    # scipy is installed; swap in cv2.connectedComponents if this ever shows up in a profile.
    """
    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        return []
    try:
        img = Image.open(image_path).convert("L")
    except Exception:
        return []

    step = _EDGE_DOWNSCALE
    small = np.asarray(img.resize((img.width // step, img.height // step)), dtype=np.int16)
    gx = np.abs(np.diff(small, axis=1, prepend=small[:, :1]))
    gy = np.abs(np.diff(small, axis=0, prepend=small[:1, :]))
    mask = (gx + gy) > 40

    seen = np.zeros_like(mask, dtype=bool)
    h, w = mask.shape
    boxes: List[List[int]] = []
    ys, xs = np.nonzero(mask)
    for sy, sx in zip(ys.tolist(), xs.tolist()):
        if seen[sy, sx]:
            continue
        stack, pts = [(sy, sx)], []
        seen[sy, sx] = True
        while stack:
            y, x = stack.pop()
            pts.append((y, x))
            if len(pts) > 4000:  # a whole window border, not an icon
                break
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        y0 = min(p[0] for p in pts) * step
        y1 = (max(p[0] for p in pts) + 1) * step
        x0 = min(p[1] for p in pts) * step
        x1 = (max(p[1] for p in pts) + 1) * step
        bw, bh = x1 - x0, y1 - y0
        if not (ICON_MIN_PX <= bw <= ICON_MAX_PX and ICON_MIN_PX <= bh <= ICON_MAX_PX):
            continue
        if not 0.4 <= bw / bh <= 2.5:  # icons are roughly square
            continue
        boxes.append([x0, y0, x1, y1])
    return boxes


def annotate_elements(image_path: str, elements: List[Dict[str, Any]], out_path: str) -> bool:
    """Draw a numbered box around every element, so the model can answer with a number."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return False
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception:
        return False
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/TTF/DejaVuSans-Bold.ttf", 17)
    except Exception:
        font = ImageFont.load_default()

    for i, el in enumerate(elements, start=1):
        x0, y0, x1, y1 = el["box"]
        draw.rectangle([x0 - 2, y0 - 2, x1 + 2, y1 + 2], outline=(255, 64, 0), width=2)
        tag = str(i)
        bw = int(draw.textlength(tag, font=font)) + 8
        # Prefer the left margin: UI rows stack ~26px apart, so a badge placed above lands
        # on the previous row's label and hides exactly what the model needs to read.
        if x0 - bw - 4 >= 0:
            bx, by = x0 - bw - 4, max(0, (y0 + y1) // 2 - 10)
        elif y0 >= 22:
            bx, by = x0 - 2, y0 - 20
        else:
            bx, by = x0 - 2, y0
        draw.rectangle([bx, by, bx + bw, by + 20], fill=(255, 64, 0))
        draw.text((bx + 4, by + 1), tag, fill=(255, 255, 255), font=font)
    try:
        img.save(out_path)
        return True
    except Exception:
        return False


def band_texts(image_path: str, fraction: float = 0.12, where: str = "top") -> List[str]:
    """Text lines in the top strip of a frame -- where chat apps put the conversation title.

    Used to check a claimed recipient against the screen before a message is sent: the
    model will otherwise put anything in that field (it once claimed "Muhammed", a word
    from the message body, while sitting in a different chat).
    """
    try:
        from PIL import Image
        img = Image.open(image_path)
    except Exception:
        return []
    h = max(1, int(img.height * fraction))
    band = (img.crop((0, 0, img.width, h)) if where == "top"
            else img.crop((0, img.height - h, img.width, img.height)))
    tmp = f"{image_path}.{where}band.png"
    try:
        band.save(tmp)
        return [" ".join(w["text"] for w in line) for line in _ocr_lines(tmp)]
    except Exception:
        return []
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


def top_band_texts(image_path: str, fraction: float = 0.12) -> List[str]:
    return band_texts(image_path, fraction, "top")


def clear_focused_field() -> None:
    """Empty the focused text field without touching anything else.

    Not ctrl+a: outside a text input that selects the whole document, and in a web view
    (ZapZap is QtWebEngine) the leftover page selection then swallows the next click --
    rows stopped opening even though the pointer was on the right pixel. End+shift+Home
    is scoped to the field and is a no-op when nothing editable has focus.
    """
    simulate_key("End")
    simulate_key("Home", ["shift"])
    simulate_key("BackSpace")
