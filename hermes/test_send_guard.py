"""Guards against the ZapZap incident: a message typed with mangled Turkish
characters and sent into whichever conversation happened to be open."""
import asyncio
from unittest.mock import patch

from hermes import wayland_ui
from hermes.hermes_supervisor import HermesSupervisor


def _run(sup, name, args, state, shots=None):
    from pathlib import Path
    return asyncio.run(sup.execute_tool(
        name, args, session_id="t", user_text="", screenshots_dir=Path(shots or "/tmp"),
        emit_event_cb=None, request_approval_cb=None, turn_state=state,
    ))


def test_typing_goes_through_the_clipboard_so_unicode_survives():
    text = "Kezban durağına yürüyor. ışğüöç"
    with patch.object(wayland_ui.subprocess, "run") as run, \
         patch.object(wayland_ui, "simulate_key", return_value=True) as key:
        run.return_value.returncode = 0
        run.return_value.stdout = ""
        assert wayland_ui.simulate_typing(text) is True
    assert key.call_args[0] == ("v", ["ctrl"])
    pasted = [c for c in run.call_args_list if c[0][0] == [wayland_ui.WL_COPY_BIN]]
    assert pasted[0].kwargs["input"] == text, "clipboard must carry the text verbatim"


def test_return_after_typing_is_refused_without_a_confirmed_recipient():
    sup = HermesSupervisor()
    state = {"draft_pending": True}
    for args in ({"key": "Return"}, {"key": "Return", "modifiers": ["ctrl"]}):
        res, _, label = _run(sup, "ui_press_key", args, state)
        assert "error" in res and "confirmed_recipient" in res["error"], args
        assert state["draft_pending"] is True, "a refused send must not clear the draft"

    # A recipient the screen does not show is a fabrication, not a confirmation.
    with patch.object(wayland_ui, "simulate_key", return_value=True), \
         patch.object(wayland_ui, "top_band_texts", return_value=["kardemir coo", "Arda, Yusuf"]), \
         patch.object(HermesSupervisor, "_grab_frame",
                      lambda self, d, sc, ts: ("/screenshots/f.png", (10, 10))):
        state["last_opened"] = "kardemir coo"
        bad, _, _ = _run(sup, "ui_press_key",
                         {"key": "Return", "confirmed_recipient": "Muhammed"}, state)
        assert "error" in bad and bad["screen_header"], bad
        # Naming the chat it actually clicked beats handing back header fragments.
        assert "kardemir coo" in bad["error"], bad["error"]
        assert state["draft_pending"] is True, "a blocked send must not clear the draft"

        res, _, label = _run(sup, "ui_press_key",
                             {"key": "Return", "confirmed_recipient": "kardemir coo"}, state)
    assert res["sent_to"] == "kardemir coo"
    assert "kardemir coo" in label, "the user must see who it went to"
    assert state["draft_pending"] is False


def test_return_without_a_pending_draft_is_ordinary_navigation():
    sup = HermesSupervisor()
    with patch.object(wayland_ui, "simulate_key", return_value=True):
        res, _, _ = _run(sup, "ui_press_key", {"key": "Return"}, {})
    assert res["ok"] is True




def test_hermes_skill_reaches_the_agent_registry():
    sup = HermesSupervisor()
    res, _, label = _run(sup, "hermes_skill", {"query": "web search"}, {})
    assert any(m["name"] == "web_search" for m in res["matches"]), res

    res, _, _ = _run(sup, "hermes_skill",
                     {"name": "read_file", "arguments": {"path": __file__}}, {})
    assert "test_hermes_skill_reaches_the_agent_registry" in str(res), res


def test_high_risk_skills_go_through_the_approval_gate():
    sup = HermesSupervisor()
    asked = []

    async def deny(req):
        asked.append(req)
        return "reject"

    res, _, _ = asyncio.run(sup.execute_tool(
        "hermes_skill", {"name": "terminal", "arguments": {"command": "rm -rf /"}},
        session_id="t", user_text="", screenshots_dir=None,
        emit_event_cb=None, request_approval_cb=deny, turn_state={},
    ))
    assert res["rejected"] is True and asked, res


def test_capture_screenshot_records_the_frame_for_ui_click(tmp=None):
    """The screenshot path stayed untested once, and a decorator landing on the wrong
    function shipped a broken capture to the phone. Exercise it end to end."""
    import tempfile
    from pathlib import Path

    sup = HermesSupervisor()
    with tempfile.TemporaryDirectory() as d:
        shots = Path(d)
        png = shots / "frame.png"
        png.write_bytes(b"")
        state = {}
        with patch.object(HermesSupervisor, "capture_screenshot",
                          staticmethod(lambda dest, scope="active_window": "/screenshots/frame.png")), \
             patch.object(wayland_ui, "get_active_window",
                          return_value={"at": [10, 20], "size": [800, 600]}):
            res, url, _ = asyncio.run(sup.execute_tool(
                "capture_screenshot", {"scope": "active_window"},
                session_id="t", user_text="", screenshots_dir=shots,
                emit_event_cb=None, request_approval_cb=None, turn_state=state,
            ))
    assert "error" not in res, res
    assert url == "/screenshots/frame.png"
    assert res["image_width"] == 800 and res["image_height"] == 600
    # ui_click translates against exactly these, so a wrong origin misplaces every click.
    assert state["shot_origin"] == (10, 20) and state["shot_size"] == (800, 600)


def test_ocr_locates_a_label_and_returns_only_its_box():
    """The model guessed round pixel coordinates and never hit the target; ui_click_text
    exists so it names a label instead. Verify OCR puts the box on the label, not the row."""
    import tempfile
    from pathlib import Path
    from PIL import Image, ImageDraw

    from PIL import ImageFont
    font = ImageFont.truetype("/usr/share/fonts/TTF/DejaVuSans.ttf", 34)
    img = Image.new("RGB", (1800, 600), "white")
    d = ImageDraw.Draw(img)
    d.text((120, 120), "Notlar", fill="black", font=font)
    d.text((120, 360), "kardemir coo", fill="black", font=font)
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "shot.png")
        img.save(path)

        hits = wayland_ui.find_text(path, "kardemir coo")
        assert len(hits) == 1, hits
        x0, y0, x1, y1 = hits[0]["box"]
        assert 350 < hits[0]["y"] < 410, hits[0]  # drawn at y=360
        assert x1 - x0 < 900, "the box must hug the label, not span the whole line"
        assert hits[0]["x"] == (x0 + x1) // 2

        assert wayland_ui.find_text(path, "definitely absent") == []

        # OCR reads "ç" as "g" at UI sizes, so matching falls back to treating the glyphs
        # tesseract confuses as equal -- but only after an exact pass finds nothing.
        assert wayland_ui._fold_loose("Otoparçasan") == wayland_ui._fold_loose("Otopargasan")
        assert wayland_ui._fold("Otoparçasan") != wayland_ui._fold("Otopargasan")
        assert wayland_ui.find_text(path, "Notlar")[0]["text"] == "Notlar"


def test_marked_screenshot_lets_the_model_click_by_number():
    """Set-of-Mark: the encoder is 224x224, so the model picks an element, not a pixel."""
    import shutil, tempfile
    from pathlib import Path

    if not Path("/tmp/win.png").exists():
        return  # sample frame absent; the OCR test already covers detection
    sup = HermesSupervisor()
    with tempfile.TemporaryDirectory() as d:
        shots = Path(d)
        shutil.copy("/tmp/win.png", shots / "frame.png")
        state = {}
        with patch.object(HermesSupervisor, "capture_screenshot",
                          staticmethod(lambda dest, scope="active_window": "/screenshots/frame.png")), \
             patch.object(wayland_ui, "get_active_window",
                          return_value={"at": [24, 66], "size": [1872, 988]}):
            res, url, _ = asyncio.run(sup.execute_tool(
                "capture_screenshot", {"scope": "active_window", "mark_elements": True},
                session_id="t", user_text="", screenshots_dir=shots,
                emit_event_cb=None, request_approval_cb=None, turn_state=state,
            ))
            assert res["elements"], res
            assert url.startswith("/screenshots/mark_"), url
            assert (shots / Path(url).name).exists(), "annotated frame must be saved"
            assert len(state["marks"]) == len(res["elements"])

            target = next(e for e in res["elements"] if "kardemir" in e["label"].lower())
            expected = state["marks"][target["mark"] - 1]["xy"]
            clicks = []
            with patch.object(wayland_ui, "click_at",
                              side_effect=lambda x, y, b, d: clicks.append((x, y)) or {"clicked": True}), \
                 patch.object(HermesSupervisor, "capture_screenshot",
                              staticmethod(lambda dest, scope="active_window": "/screenshots/frame.png")):
                out, _, label = asyncio.run(sup.execute_tool(
                    "ui_click", {"mark": target["mark"]},
                    session_id="t", user_text="", screenshots_dir=shots,
                    emit_event_cb=None, request_approval_cb=None, turn_state=state,
                ))
            assert "error" not in out, out
            assert "kardemir" in out["clicked"].lower(), out["clicked"]
            # Window origin (24,66) must be added back on: marks are image-space.
            assert clicks == [(24 + expected[0], 66 + expected[1])], (clicks, expected)
            # The click changed the screen, so that numbering no longer describes it.
            assert "marks" not in state, "stale marks must not survive a click"

            bad, _, _ = asyncio.run(sup.execute_tool(
                "ui_click", {"mark": 1},
                session_id="t", user_text="", screenshots_dir=shots,
                emit_event_cb=None, request_approval_cb=None, turn_state=state,
            ))
            assert "error" in bad and "mark_elements" in bad["error"], bad


async def _noop_event(_evt):
    return None


def test_every_advertised_tool_has_an_executor():
    """The model is offered these names in the prefix; a name with no branch behind it is
    a dead tool it will keep retrying. Two branches were lost to line-based edits already,
    each time silently, so this compares the two lists directly."""
    from hermes.hermes_supervisor import CANONICAL_HERMES_TOOL_NAMES
    from hermes.prefix_builder import get_canonical_tools

    advertised = {t["function"]["name"] for t in get_canonical_tools()}
    assert advertised == set(CANONICAL_HERMES_TOOL_NAMES), (
        advertised ^ set(CANONICAL_HERMES_TOOL_NAMES))

    import tempfile
    from pathlib import Path

    sup = HermesSupervisor()
    with tempfile.TemporaryDirectory() as d:
        for name in sorted(advertised):
            if name == "ask_approval":
                continue  # answered by the router, not by execute_tool
            res, _, label = asyncio.run(sup.execute_tool(
                name, {}, session_id="t", user_text="", screenshots_dir=Path(d),
                emit_event_cb=_noop_event, request_approval_cb=None, turn_state={},
            ))
            # Empty args must produce a real refusal, never "unknown tool" or a crash.
            assert "Bilinmeyen araç" not in str(res), f"{name} has no executor branch"
            assert not label.startswith("Hata:"), f"{name}: {label}"


def test_repeating_an_ambiguous_click_walks_to_the_next_match():
    """The model's only reflex on a refusal is to reissue the identical call, so a repeat
    must advance through the matches instead of dead-ending on 'pass occurrence'."""
    import tempfile
    from pathlib import Path

    sup = HermesSupervisor()
    two = [{"text": "Notlar", "line": "Notlar", "x": 100, "y": 62,
            "box": [0, 0, 0, 0], "confidence": 90},
           {"text": "Notlar", "line": "Notlar", "x": 100, "y": 283,
            "box": [0, 0, 0, 0], "confidence": 90}]
    with tempfile.TemporaryDirectory() as d:
        shots = Path(d)
        (shots / "frame.png").write_bytes(b"")
        state, clicks = {}, []
        with patch.object(HermesSupervisor, "capture_screenshot",
                          staticmethod(lambda dest, scope="active_window": "/screenshots/frame.png")), \
             patch.object(wayland_ui, "get_active_window",
                          return_value={"at": [0, 0], "size": [900, 900]}), \
             patch.object(wayland_ui, "find_text", return_value=two), \
             patch.object(wayland_ui, "click_at",
                          side_effect=lambda x, y, b, dbl: clicks.append(y) or {"clicked": True}):
            for n in range(3):
                res, _, _ = asyncio.run(sup.execute_tool(
                    "ui_click", {"text": "Notlar"}, session_id="t", user_text="",
                    screenshots_dir=shots, emit_event_cb=None,
                    request_approval_cb=None, turn_state=state,
                ))
                assert "error" not in res, res
    # Leftmost first, then the next match, then back round -- never the same dead end.
    assert clicks == [62, 283, 62], clicks
    # And a repeat is called out where it happens, not only in the prompt.
    assert "warning" in res and "tikladin" in res["warning"], res


def test_typing_clears_a_stale_draft_first():
    """A leftover draft in a chat box once rode out glued to the front of a real message."""
    sup = HermesSupervisor()
    keys = []
    with patch.object(wayland_ui, "simulate_key",
                      side_effect=lambda k, m=None: keys.append((k, m)) or True), \
         patch.object(wayland_ui, "simulate_typing", return_value=True), \
         patch.object(wayland_ui, "get_active_window",
                      return_value={"class": "zapzap", "title": "ZapZap",
                                    "at": [0, 0], "size": [100, 100]}), \
         patch.object(HermesSupervisor, "_grab_frame",
                      lambda self, d, sc, ts: ("/screenshots/f.png", (100, 100))):
        res, _, _ = _run(sup, "ui_type_text",
                         {"text": "merhaba", "target_app": "zapzap"}, {})
        assert res["ok"] is True, res
        assert keys == [("a", ["ctrl"]), ("BackSpace", None)], keys

        keys.clear()
        _run(sup, "ui_type_text",
             {"text": "devami", "target_app": "zapzap", "append": True}, {})
        assert keys == [], "append must leave the field alone"


def test_a_send_is_only_reported_once_the_draft_leaves_the_box():
    """Enter sends in some apps and inserts a newline in others. The turn must not claim
    delivery while the text is still sitting in the input box."""
    sup = HermesSupervisor()

    def run(bottom_lines_sequence):
        state = {"draft_pending": True, "draft_text": "Muhammed yolda, Kezban duragina"}
        seen = iter(bottom_lines_sequence)
        keys = []
        with patch.object(wayland_ui, "simulate_key",
                          side_effect=lambda k, m=None: keys.append((k, m)) or True), \
             patch.object(wayland_ui, "top_band_texts", return_value=["Notlar"]), \
             patch.object(wayland_ui, "band_texts", side_effect=lambda *a, **k: next(seen)), \
             patch.object(HermesSupervisor, "_grab_frame",
                          lambda self, d, sc, ts: ("/screenshots/f.png", (10, 10))):
            res, _, label = _run(sup, "ui_press_key",
                                 {"key": "Return", "confirmed_recipient": "Notlar"}, state)
        return res, state, keys

    # Draft gone on the first look: a clean send.
    res, state, _ = run([[""]])
    assert res.get("sent_to") == "Notlar", res
    assert state["draft_pending"] is False

    # Enter only made a newline; the retry with ctrl+Return clears the box.
    res, state, keys = run([["Muhammed yolda, Kezban duragina"], [""]])
    assert res.get("sent_to") == "Notlar", res
    assert ("Return", ["ctrl"]) in keys, keys

    # Neither combination worked: say so instead of reporting success.
    res, state, _ = run([["Muhammed yolda, Kezban duragina"]] * 2)
    assert "error" in res and "yazi kutusunda" in res["error"], res
    assert state["draft_pending"] is True, "an unsent draft must stay pending"


def test_typing_is_refused_when_the_text_never_reaches_the_box():
    """Keystrokes go wherever focus is. A side panel once swallowed a whole message, and
    the send check then read the missing draft as proof of delivery."""
    sup = HermesSupervisor()
    active = {"class": "zapzap", "title": "ZapZap", "at": [0, 0], "size": [100, 100]}
    with patch.object(wayland_ui, "simulate_key", return_value=True), \
         patch.object(wayland_ui, "simulate_typing", return_value=True), \
         patch.object(wayland_ui, "get_active_window", return_value=active), \
         patch.object(HermesSupervisor, "_grab_frame",
                      lambda self, d, sc, ts: ("/screenshots/f.png", (10, 10))):

        state = {}
        with patch.object(wayland_ui, "band_texts", return_value=["Bir mesaj yazın"]):
            res, _, label = _run(sup, "ui_type_text",
                                 {"text": "Kezban durağına yürüyor", "target_app": "zapzap"},
                                 state)
        assert "error" in res and "odagi" in res["error"], res
        assert "draft_pending" not in state, "nothing was typed, so nothing is pending"

        # Return after a failed write must not come back "ok": that is how a turn ends up
        # announcing a message that was never composed.
        nothing, _, _ = _run(sup, "ui_press_key",
                             {"key": "Return", "confirmed_recipient": "Notlar"}, state)
        assert "error" in nothing and "basarisiz" in nothing["error"], nothing

        with patch.object(wayland_ui, "band_texts",
                          return_value=["Kezban duragina yuruyor"]):
            res, _, _ = _run(sup, "ui_type_text",
                             {"text": "Kezban durağına yürüyor", "target_app": "zapzap"},
                             state)
        assert res.get("ok") is True, res
        assert state["draft_pending"] is True


def test_matches_are_ordered_left_to_right_so_the_list_wins_over_the_header():
    """Master-detail layouts put the actionable list on the left; the same name on the
    right is the open item's header, and clicking it opened an info panel that stole
    keyboard focus four runs running."""
    if not __import__("pathlib").Path("/tmp/e2ewin.png").exists():
        return
    hits = wayland_ui.find_text("/tmp/e2ewin.png", "Notlar")
    if len(hits) < 2:
        return
    assert hits == sorted(hits, key=lambda h: (h["x"], h["y"])), hits
    assert hits[0]["x"] < hits[1]["x"], "the leftmost match must be picked by default"


if __name__ == "__main__":
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        fn(); print("ok:", fn.__name__)
