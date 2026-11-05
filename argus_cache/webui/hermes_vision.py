"""Fast cached OCR and non-disruptive Hyprland headless background execution for Hermes PC."""

from __future__ import annotations

import functools
import json
import os
import subprocess
import tempfile
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Tuple

HYPRCTL_BIN = "/usr/bin/hyprctl"
GRIM_BIN = "/usr/bin/grim"
TESSERACT_BIN = "/usr/bin/tesseract"
HEADLESS_WS = 9
MAX_OCR_WIDTH = 1600

_GPU_HEAVY_APPS = {
    "zapzap": "env QTWEBENGINE_CHROMIUM_FLAGS='--disable-gpu' zapzap",
    "whatsapp": "env QTWEBENGINE_CHROMIUM_FLAGS='--disable-gpu' zapzap",
    "discord": "discord --disable-gpu",
    "signal": "signal-desktop --disable-gpu",
    "spotify": "spotify --disable-gpu",
}


def ensure_wayland_env() -> None:
    """Auto-discover active Wayland & Hyprland sockets when running inside a background systemd user service."""
    runtime_dir = os.environ.get("XDG_RUNTIME_DIR") or f"/run/user/{os.getuid()}"
    if os.path.isdir(runtime_dir):
        os.environ.setdefault("XDG_RUNTIME_DIR", runtime_dir)
        w_disp = os.environ.get("WAYLAND_DISPLAY")
        if not w_disp or not os.path.exists(os.path.join(runtime_dir, w_disp)):
            for cand in ("wayland-1", "wayland-0"):
                if os.path.exists(os.path.join(runtime_dir, cand)):
                    os.environ["WAYLAND_DISPLAY"] = cand
                    break
        hypr_dir = os.path.join(runtime_dir, "hypr")
        sig = os.environ.get("HYPRLAND_INSTANCE_SIGNATURE")
        if os.path.isdir(hypr_dir) and (not sig or not os.path.isdir(os.path.join(hypr_dir, sig))):
            try:
                entries = [
                    os.path.join(hypr_dir, d)
                    for d in os.listdir(hypr_dir)
                    if os.path.isdir(os.path.join(hypr_dir, d))
                ]
                if entries:
                    newest = max(entries, key=os.path.getmtime)
                    os.environ["HYPRLAND_INSTANCE_SIGNATURE"] = os.path.basename(newest)
            except OSError:
                pass


def hypr_json(subcmd: List[str]) -> Any:
    ensure_wayland_env()
    try:
        res = subprocess.run([HYPRCTL_BIN, *subcmd, "-j"], capture_output=True, text=True, timeout=3)
        if res.returncode == 0 and res.stdout.strip():
            return json.loads(res.stdout)
    except Exception:
        pass
    return None


def hypr_dispatch(*args: str) -> bool:
    ensure_wayland_env()
    try:
        res = subprocess.run([HYPRCTL_BIN, "dispatch", *args], capture_output=True, text=True, timeout=4)
        out = (res.stdout or "").strip().lower()
        return res.returncode == 0 and (out in ("ok", "") or "error" not in out and "invalid" not in out and "not found" not in out)
    except Exception:
        return False


def list_clients_full() -> List[Dict[str, Any]]:
    data = hypr_json(["clients"])
    if not isinstance(data, list):
        return []
    out: List[Dict[str, Any]] = []
    for item in data:
        ws = item.get("workspace") or {}
        out.append({
            "address": str(item.get("address") or ""),
            "stable_id": str(item.get("stableId") or ""),
            "class": str(item.get("class") or ""),
            "title": str(item.get("title") or ""),
            "workspace_id": int(ws.get("id") or 0),
            "workspace_name": str(ws.get("name") or ""),
            "monitor": item.get("monitor"),
            "at": item.get("at") or [0, 0],
            "size": item.get("size") or [0, 0],
            "pid": int(item.get("pid") or 0),
            "mapped": bool(item.get("mapped", True)),
        })
    return out


def list_monitors() -> List[Dict[str, Any]]:
    data = hypr_json(["monitors"])
    return data if isinstance(data, list) else []


def is_headless_name(name: str) -> bool:
    up = (name or "").upper()
    return up.startswith("HEADLESS") or up == "HERMES_BG"


def is_studio_window(win: Optional[Dict[str, Any]]) -> bool:
    if not win:
        return False
    title = str(win.get("title") or "").lower()
    return "argus studio" in title or "127.0.0.1:8090" in title


def _low_free_vram() -> bool:
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=1.0,
        )
        if res.returncode == 0 and res.stdout.strip():
            return float(res.stdout.strip().splitlines()[0]) < 1100.0
    except Exception:
        pass
    return False


def _focus_window_ipc(wayland_ui: Any, ws_id: int, addr: str, wclass: str = "") -> None:
    if ws_id > 0:
        if not hypr_dispatch("workspace", str(ws_id)):
            wayland_ui.run_hyprctl_dispatch(f"hl.dsp.focus({{ workspace = {ws_id} }})")
    if addr:
        if not hypr_dispatch("focuswindow", f"address:{addr}"):
            res = wayland_ui.run_hyprctl_dispatch(f"hl.dsp.focus({{ window = 'address:{addr}' }})")
            if not res.get("success") and wclass:
                wayland_ui.run_hyprctl_dispatch(f"hl.dsp.focus({{ window = 'class:{wclass}' }})")


def save_user_focus(wayland_ui: Any) -> Dict[str, Any]:
    win = wayland_ui.get_active_window() or {}
    cur = wayland_ui.get_cursor_pos()
    return {
        "address": str(win.get("address") or ""),
        "class": str(win.get("class") or ""),
        "workspace_id": int((win.get("workspace") or {}).get("id") or 0),
        "cursor": cur,
    }


def restore_user_focus(wayland_ui: Any, snapshot: Dict[str, Any]) -> None:
    addr = str(snapshot.get("address") or "")
    wclass = str(snapshot.get("class") or "")
    ws_id = int(snapshot.get("workspace_id") or 0)
    cur = snapshot.get("cursor")
    if ws_id > 0 or addr:
        _focus_window_ipc(wayland_ui, ws_id, addr, wclass)
    if cur and isinstance(cur, tuple) and len(cur) == 2:
        hypr_dispatch("movecursor", str(int(cur[0])), str(int(cur[1])))


def remove_stale_headless_monitors() -> None:
    """Remove any virtual headless output so physical display boundaries remain intact."""
    for m in list_monitors():
        name = str(m.get("name") or "")
        if is_headless_name(name):
            try:
                subprocess.run([HYPRCTL_BIN, "output", "remove", name], capture_output=True, text=True, timeout=3)
            except Exception:
                pass


def find_matching_window(app_name: str, target_class: str) -> Optional[Dict[str, Any]]:
    q = app_name.lower().strip()
    tc = target_class.lower().strip()
    candidates = [w for w in list_clients_full() if w.get("mapped", True)]
    for w in candidates:
        if tc and tc in w["class"].lower() and not is_studio_window(w):
            return w
    for w in candidates:
        if q and (q in w["class"].lower() or q in w["title"].lower()) and not is_studio_window(w):
            return w
    return None


def focus_or_launch_smart(
    wayland_ui: Any,
    app_name: str,
    *,
    background: bool = True,
    pc_state: Dict[str, Any],
) -> Dict[str, Any]:
    remove_stale_headless_monitors()
    app_key = app_name.lower().strip()
    meta = wayland_ui.APP_MAP.get(app_key)
    target_class = meta["class"] if meta else app_key
    base_exec = meta["exec"] if meta else app_name
    exec_cmd = _GPU_HEAVY_APPS[app_key] if (app_key in _GPU_HEAVY_APPS and _low_free_vram()) else base_exec

    existing = find_matching_window(app_key, target_class)

    if background:
        if existing:
            pc_state["target_window"] = {**existing, "headless_monitor": "", "background": True}
            return {
                "action": "focused_in_background",
                "mode": "background_toplevel",
                "app": app_name,
                "class": existing["class"],
                "title": existing["title"],
                "workspace": existing["workspace_id"],
            }

        snap = save_user_focus(wayland_ui)
        free_ws = getattr(wayland_ui, "find_free_workspace", lambda: None)() or HEADLESS_WS
        launched = hypr_dispatch("exec", f"[workspace {free_ws} silent] {exec_cmd}")
        if not launched:
            wayland_ui.run_hyprctl_dispatch(f"hl.dsp.exec_cmd('{exec_cmd}')")
        deadline = time.time() + 8.5
        found: Optional[Dict[str, Any]] = None
        while time.time() < deadline:
            found = find_matching_window(app_key, target_class)
            if found:
                break
            time.sleep(0.3)
        restore_user_focus(wayland_ui, snap)
        if found:
            pc_state["target_window"] = {**found, "headless_monitor": "", "background": True}
            return {
                "action": "launched_in_background",
                "mode": "background_toplevel",
                "app": app_name,
                "class": found["class"],
                "title": found["title"],
                "workspace": found["workspace_id"],
            }

    if existing:
        _focus_window_ipc(wayland_ui, existing["workspace_id"], existing["address"], existing["class"])
        time.sleep(0.2)
        pc_state["target_window"] = {**existing, "headless_monitor": "", "background": False}
        return {
            "action": "focused_existing",
            "mode": "foreground",
            "app": app_name,
            "class": existing["class"],
            "title": existing["title"],
            "workspace": existing["workspace_id"],
        }

    res = wayland_ui.focus_or_launch(app_name, fallback_exec=exec_cmd)
    found = find_matching_window(app_key, target_class)
    if not found and wayland_ui._rofi_is_open():
        wayland_ui.simulate_key("Escape")
        hypr_dispatch("exec", exec_cmd)
        deadline = time.time() + 6.0
        while time.time() < deadline and not found:
            time.sleep(0.3)
            found = find_matching_window(app_key, target_class)
    if found:
        pc_state["target_window"] = {**found, "headless_monitor": "", "background": False}
        res.update({"action": "launched_and_focused", "class": found["class"], "title": found["title"]})
    return res


@contextmanager
def transient_target_focus(
    wayland_ui: Any,
    pc_state: Dict[str, Any],
    *,
    post_delay: float = 0.28,
) -> Iterator[Optional[Dict[str, Any]]]:
    """Briefly focus the pinned background/target window for input, let Hyprland animation settle, then restore."""
    target = pc_state.get("target_window")
    if not target or not target.get("address"):
        yield None
        return
    is_bg = bool(target.get("background"))
    snap = save_user_focus(wayland_ui)
    addr = str(target["address"])
    ws_id = int(target.get("workspace_id") or 0)
    wclass = str(target.get("class") or "")
    try:
        _focus_window_ipc(wayland_ui, ws_id, addr, wclass)
        deadline = time.monotonic() + 0.30
        while time.monotonic() < deadline:
            active = wayland_ui.get_active_window() or {}
            if str(active.get("address") or "") == addr:
                if active.get("at"):
                    target["at"] = active["at"]
                break
            time.sleep(0.04)
        time.sleep(0.16)
        yield target
        if post_delay > 0:
            time.sleep(post_delay)
    finally:
        if is_bg and (snap.get("address") != addr or snap.get("workspace_id") != ws_id):
            restore_user_focus(wayland_ui, snap)


def fast_click(wayland_ui: Any, x: int, y: int, button: str = "left", double: bool = False) -> Dict[str, Any]:
    """Teleport Wayland cursor via Hyprland `movecursor` when available, avoiding 6-step relative drift."""
    if hypr_dispatch("movecursor", str(int(x)), str(int(y))):
        time.sleep(0.04)
        cur = wayland_ui.get_cursor_pos()
        if cur and abs(cur[0] - int(x)) <= 4 and abs(cur[1] - int(y)) <= 4:
            return wayland_ui.click_at(int(x), int(y), button=button, double=double)
    return wayland_ui.click_at(int(x), int(y), button=button, double=double)


@functools.lru_cache(maxsize=1)
def _best_single_lang(wayland_ui: Any) -> str:
    """Single-model `tur` already recognizes full ASCII + Turkish diacritics in 1 LSTM pass (~2x faster than `tur+eng`)."""
    avail = wayland_ui._ocr_lang()
    if "tur" in avail.split("+"):
        return "tur"
    return avail.split("+")[0] if avail else "eng"


def _center_y(word: Dict[str, Any]) -> float:
    return (word["box"][1] + word["box"][3]) / 2


def group_words_into_lines(words: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """OCR words -> reading-order lines: rows by vertical centre, then left to right.

    Grouping in (top, left) order broke chat bubbles: a right-hand word whose box
    sat one pixel higher opened its own line first, so the line came back split
    and reversed. Rows tolerate that jitter; a wide horizontal gap still splits a
    row, which keeps side-by-side columns (incoming/outgoing bubbles) apart.
    """
    rows: List[List[Dict[str, Any]]] = []
    for word in sorted(words, key=_center_y):
        height = max(1, word["box"][3] - word["box"][1])
        if rows:
            row = rows[-1]
            row_center = sum(_center_y(w) for w in row) / len(row)
            if abs(_center_y(word) - row_center) < height * 0.6:
                row.append(word)
                continue
        rows.append([word])

    lines: List[List[Dict[str, Any]]] = []
    for row in rows:
        ordered = sorted(row, key=lambda w: w["box"][0])
        line = [ordered[0]]
        for word in ordered[1:]:
            height = max(1, word["box"][3] - word["box"][1])
            if word["box"][0] - line[-1]["box"][2] < height * 1.5:
                line.append(word)
            else:
                lines.append(line)
                line = [word]
        lines.append(line)
    return lines


def find_sidebar_split(boxes: List[List[int]], *, width: int, height: int) -> int:
    """x of the widest text-free vertical gutter between 20% and 50% of the width.

    A fixed 33% split put ZapZap's chat-list timestamps (ending ~640 px of 1868)
    into the conversation. The gutter must have text on both sides; with none
    the old 33% fraction stands.
    """
    lo, hi = int(width * 0.2), int(width * 0.5)
    header_y = int(height * 0.12)
    covered = bytearray(max(0, hi - lo))
    for x0, y0, x1, y1 in boxes:
        if y1 < header_y:
            continue
        for x in range(max(x0, lo), min(x1, hi)):
            covered[x - lo] = 1
    best_len, best_mid, start = 0, None, None
    for i, c in enumerate(covered):
        if not c and start is None:
            start = i
        elif c and start is not None:
            if start > 0 and i - start > best_len:
                best_len, best_mid = i - start, lo + (start + i) // 2
            start = None
    return best_mid if best_mid is not None and best_len >= 12 else int(width * 0.33)


@functools.lru_cache(maxsize=16)
def _cached_ocr_lines(image_path: str, mtime_ns: int, file_size: int, lang: str) -> Tuple[Tuple[Dict[str, Any], ...], ...]:
    del mtime_ns, file_size
    if not os.path.exists(TESSERACT_BIN):
        return ()
    scale = 1.0
    ocr_target = image_path
    tmp_bmp: Optional[str] = None
    try:
        from PIL import Image

        img = Image.open(image_path).convert("L")
        if img.width > MAX_OCR_WIDTH:
            scale = MAX_OCR_WIDTH / float(img.width)
            img = img.resize(
                (max(1, int(img.width * scale)), max(1, int(img.height * scale))),
                Image.Resampling.BILINEAR,
            )
        fd, tmp_bmp = tempfile.mkstemp(prefix="argus_ocr_", suffix=".bmp")
        os.close(fd)
        img.save(tmp_bmp, format="BMP")
        ocr_target = tmp_bmp
    except Exception:
        scale = 1.0
        ocr_target = image_path

    try:
        env = {**os.environ, "OMP_THREAD_LIMIT": "2"}
        r = subprocess.run(
            [TESSERACT_BIN, ocr_target, "stdout", "-l", lang, "--psm", "11", "tsv"],
            capture_output=True,
            text=True,
            timeout=18,
            env=env,
        )
        if r.returncode != 0:
            return ()
        inv = 1.0 / scale
        words: List[Dict[str, Any]] = []
        for row in r.stdout.splitlines()[1:]:
            f = row.split("\t")
            if len(f) < 12 or not f[11].strip():
                continue
            try:
                conf = float(f[10])
                left = int(round(int(f[6]) * inv))
                top = int(round(int(f[7]) * inv))
                w = int(round(int(f[8]) * inv))
                h = int(round(int(f[9]) * inv))
            except ValueError:
                continue
            if conf < 12 or h <= 0:
                continue
            words.append({"text": f[11].strip(), "box": [left, top, left + w, top + h], "conf": conf})

        return tuple(tuple(ln) for ln in group_words_into_lines(words))
    except Exception:
        return ()
    finally:
        if tmp_bmp:
            try:
                os.unlink(tmp_bmp)
            except OSError:
                pass


def fast_ocr_lines(wayland_ui: Any, image_path: str) -> List[List[Dict[str, Any]]]:
    try:
        st = os.stat(image_path)
    except OSError:
        return []
    cached = _cached_ocr_lines(os.path.abspath(image_path), st.st_mtime_ns, st.st_size, _best_single_lang(wayland_ui))
    return [list(ln) for ln in cached]


def fast_detect_elements(wayland_ui: Any, image_path: str, max_elements: int = 80) -> List[Dict[str, Any]]:
    elements: List[Dict[str, Any]] = []
    for words in fast_ocr_lines(wayland_ui, image_path):
        text = " ".join(w["text"] for w in words).strip()
        mean_conf = sum(w["conf"] for w in words) / len(words)
        if len(text) < 2 or mean_conf < 32:
            continue
        x0 = min(w["box"][0] for w in words)
        y0 = min(w["box"][1] for w in words)
        x1 = max(w["box"][2] for w in words)
        y1 = max(w["box"][3] for w in words)
        elements.append({"kind": "text", "label": text, "box": [x0, y0, x1, y1]})

    elements.sort(key=lambda e: (e["box"][1], e["box"][0]))
    return elements[:max_elements]


def fast_read_screen_text(
    wayland_ui: Any,
    image_path: str,
    *,
    width: int = 0,
    height: int = 0,
    max_chars: int = 4600,
) -> str:
    """Extract clean, unboxed, spatially grouped full-screen text without splitting chat timelines."""
    lines_out: List[Tuple[int, int, int, str]] = []
    use_zones = width >= 900 and height >= 480
    ocr_lines = fast_ocr_lines(wayland_ui, image_path)
    sidebar_x = find_sidebar_split(
        [[min(w["box"][0] for w in ln), min(w["box"][1] for w in ln), max(w["box"][2] for w in ln), max(w["box"][3] for w in ln)]
         for ln in ocr_lines],
        width=width,
        height=height,
    ) if use_zones else 0
    right_bubble_x = int(width * 0.56)
    header_y = int(height * 0.12)

    for order, words in enumerate(ocr_lines):
        text = " ".join(w["text"] for w in words).strip()
        mean_conf = sum(w["conf"] for w in words) / len(words)
        # Read ticks and bubble tails OCR as ",, 5. ,"; they carry no text.
        if sum(ch.isalnum() for ch in text) < 2 or mean_conf < 30:
            continue
        x0 = min(w["box"][0] for w in words)
        y0 = min(w["box"][1] for w in words)
        x1 = max(w["box"][2] for w in words)
        cx = (x0 + x1) // 2
        if not use_zones:
            zone = 0
            label = text
        elif y0 < header_y:
            zone = 0  # Üst Başlık / Aktif Pencere veya Açık Sohbet Başlığı
            label = f"[Açık Başlık] {text}" if cx >= sidebar_x else text
        elif cx < sidebar_x:
            zone = 1  # Sol Panel (Menü / Sohbet / Dosya Listesi)
            label = text
        else:
            zone = 2  # Ana İçerik (tam genişlik kronolojik akış; sağa yaslı balonlar işaretli)
            label = f"[Sağ] {text}" if x0 >= right_bubble_x else text
        lines_out.append((zone, order, x0, label))

    if not lines_out:
        return "(Ekranda OCR ile okunabilir metin bulunamadı.)"

    lines_out.sort(key=lambda it: (it[0], it[1], it[2]))
    zone_titles = {
        0: "Üst Çubuk / Başlık",
        1: "Sol Panel (Menü / Sohbet Listesi — listedeki öğeyi açmak için `pc_click(text=...)` çağır)",
        2: "Ana İçerik (Şu An Açık Olan Görünüm — yukarıdan aşağıya kronolojik, `[Sağ]` sağa yaslı/gönderilen)",
    }
    sections: Dict[int, List[str]] = {}
    for zone, _y0, _x0, txt in lines_out:
        bucket = sections.setdefault(zone, [])
        if not bucket or bucket[-1] != txt:
            bucket.append(txt)

    if not use_zones or len(sections) == 1:
        merged = "\n".join(txt for _z, _y, _x, txt in lines_out)
        return merged[:max_chars]

    blocks: List[str] = []
    for z in (0, 1, 2):
        if z in sections:
            blocks.append(f"### {zone_titles[z]}\n" + "\n".join(sections[z]))
    full = "\n\n".join(blocks)
    return full[:max_chars] + ("\n…" if len(full) > max_chars else "")


def fast_find_text(wayland_ui: Any, image_path: str, needle: str) -> List[Dict[str, Any]]:
    if not needle.strip():
        return []
    lines = fast_ocr_lines(wayland_ui, image_path)
    hits = wayland_ui._match_lines(lines, needle, wayland_ui._fold)
    return hits if hits else wayland_ui._match_lines(lines, needle, wayland_ui._fold_loose)

