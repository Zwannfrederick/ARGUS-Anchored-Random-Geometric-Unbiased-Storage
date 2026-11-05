"""Tests for multi-turn agent continuation, stop cancellation, chronological timeline, and background Hermes."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

from argus_cache.webui import hermes_pc, hermes_vision
from argus_cache.webui.agent_loop import StreamManager, parse_fallback_tool_calls
from argus_cache.webui.permissions import PermissionBroker
from argus_cache.webui.server import build_chat_llm_messages, create_studio_app
from argus_cache.webui.store import StudioStore
from argus_cache.webui.tools import ToolRunner, build_tool_schemas


def test_parse_fallback_tool_calls_supports_json_and_qwen_xml():
    xml_text = (
        "Önce hesaplayalım.\n"
        "<tool_call>\n"
        "<function=verify_with_python>\n"
        "<parameter=code>\nprint(2 + 2)\n</parameter>\n"
        "<parameter=hypothesis>\ntoplama testi\n</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )
    calls = parse_fallback_tool_calls(xml_text)
    assert len(calls) == 1
    assert calls[0]["name"] == "verify_with_python"
    args = json.loads(calls[0]["arguments"])
    assert args["code"] == "print(2 + 2)"
    assert args["hypothesis"] == "toplama testi"


def test_research_mode_exposes_callable_audit_citations_and_background_pc_modes():
    research_names = [
        s["function"]["name"]
        for s in build_tool_schemas(
            enable_web=True,
            web_mode="auto",
            pc_mode="off",
            has_docs=False,
            has_workspace=False,
            mode="research",
            prompt="kuantum kimya literatür taraması yap",
        )
    ]
    assert "audit_citations" in research_names
    assert "verify_with_python" in research_names

    bg_pc_names = [
        s["function"]["name"]
        for s in build_tool_schemas(
            enable_web=False,
            web_mode="off",
            pc_mode="background",
            has_docs=False,
            has_workspace=False,
            mode="assistant",
            prompt="notlara tıkla",
        )
    ]
    assert "pc_focus_or_launch" in bg_pc_names
    assert "pc_click" in bg_pc_names


def test_empty_after_thinking_auto_continues_and_saves_chronological_timeline(tmp_path, monkeypatch):
    import httpx as httpx_mod

    round_idx = 0

    class FakeStreamResp:
        def __init__(self, lines):
            self._lines = lines

        def raise_for_status(self):
            return None

        async def aiter_lines(self):
            for ln in self._lines:
                yield ln

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

    class FakeAsyncClient:
        def __init__(self, **kwargs):
            pass

        def stream(self, method, url, *, json=None, **kwargs):
            nonlocal round_idx
            round_idx += 1
            if round_idx == 1:
                # Bug reproducer: model finishes <think> with empty content and zero tool_calls!
                return FakeStreamResp([
                    'data: {"choices":[{"delta":{"reasoning_content":"Önce Python ile doğrulama yapmalıyım."}}]}',
                    "data: [DONE]",
                ])
            if round_idx == 2:
                # Auto-continued round: emits visible step intro + Qwen XML tool call!
                return FakeStreamResp([
                    'data: {"choices":[{"delta":{"content":"Hesaplamayı başlatıyorum.\\n<tool_call><function=verify_with_python><parameter=code>print(21 * 2)</parameter></function></tool_call>"}}]}',
                    "data: [DONE]",
                ])
            return FakeStreamResp([
                'data: {"choices":[{"delta":{"reasoning_content":"Sonuç 42 çıktı.","content":"Doğrulanan sonuç **42** ve kaynak https://ornek-uydurma.org/makale"}}]}',
                "data: [DONE]",
            ])

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

    monkeypatch.setattr(httpx_mod, "AsyncClient", FakeAsyncClient)

    app = create_studio_app(storage_dir=tmp_path / "studio", upstream_url="http://127.0.0.1:9/v1")
    client = TestClient(app)
    sess = client.post("/api/sessions", json={"title": "AR-GE Akış Testi", "mode": "research"}).json()

    resp = client.post(
        "/api/chat/stream",
        json={
            "session_id": sess["id"],
            "message": "Hesapla ve doğrula",
            "mode": "research",
            "enable_thinking": True,
            "web_mode": "off",
            "pc_mode": "off",
        },
    )
    assert resp.status_code == 200
    assert round_idx == 3
    assert "event: phase" in resp.text
    assert "event: grounding_audit" in resp.text

    saved_sess = client.get(f"/api/sessions/{sess['id']}").json()
    assistant_msg = saved_sess["messages"][-1]
    assert "Hesaplamayı başlatıyorum." in assistant_msg["content"]
    assert "Doğrulanan sonuç **42**" in assistant_msg["content"]
    assert "Bağımsız Kaynak Denetimi" in assistant_msg["content"]

    tl_types = [item["type"] for item in assistant_msg["timeline"]]
    # Strictly chronological: Round 1 reasoning -> Round 2 intro text -> Round 2 tool -> Round 3 reasoning -> Round 3 text -> warning text
    assert tl_types[:5] == ["reasoning", "text", "tool", "reasoning", "text"]
    assert assistant_msg["timeline"][2]["name"] == "verify_with_python"
    assert "42" in assistant_msg["timeline"][2]["preview"]


def test_stream_manager_stop_cancels_inflight_task_and_pending_permissions(tmp_path):
    async def _run():
        broker = PermissionBroker(tmp_path / "notes")
        mgr = StreamManager(broker)
        cancelled_flag = False

        async def slow_worker():
            nonlocal cancelled_flag
            try:
                allowed = await broker.ensure_permission(
                    session_id="sess_1",
                    operation="exec",
                    target=tmp_path,
                    workspace_dir=None,
                    summary="sleep 30",
                    diff_preview="",
                    emit_event=lambda _e, _d: None,
                )
                if not allowed:
                    await asyncio.sleep(30)
            except asyncio.CancelledError:
                cancelled_flag = True
                raise

        task = asyncio.create_task(slow_worker())
        mgr.register("sess_1", task)
        await asyncio.sleep(0.03)
        assert mgr.is_active("sess_1") is True
        assert len(broker._pending) == 1

        stopped = await mgr.stop_session("sess_1")
        assert stopped is True
        assert cancelled_flag is True
        assert len(broker._pending) == 0
        assert mgr.is_active("sess_1") is False

    asyncio.run(_run())


def test_fast_ocr_cache_and_background_hermes_focus_without_stealing_chrome(tmp_path):
    # 1. Verify _cached_ocr_lines LRU cache hits on identical (path, mtime_ns, size)
    hermes_vision._cached_ocr_lines.cache_clear()
    sample_img = tmp_path / "zapzap.png"
    Image.new("RGB", (640, 400), color=(24, 30, 28)).save(sample_img)

    fake_tsv = (
        "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"
        "5\t1\t1\t1\t1\t1\t40\t80\t90\t24\t92.0\tNotlarim\n"
        "5\t1\t1\t1\t2\t1\t40\t140\t120\t24\t89.0\tMarket_Listesi\n"
    )

    class ProcOk:
        returncode = 0
        stdout = fake_tsv

    with (
        patch.object(hermes_vision.os.path, "exists", return_value=True),
        patch.object(hermes_vision, "_best_single_lang", return_value="tur"),
        patch("argus_cache.webui.hermes_vision.subprocess.run", return_value=ProcOk()) as mock_run,
    ):
        lines_1 = hermes_vision.fast_ocr_lines(hermes_pc.wayland_ui, str(sample_img))
        lines_2 = hermes_vision.fast_ocr_lines(hermes_pc.wayland_ui, str(sample_img))
        assert len(lines_1) == 2
        assert lines_1 == lines_2
        assert mock_run.call_count == 1  # Second call is 0.0ms LRU cache hit!

    # 2. Verify background pc_focus_or_launch pins ZapZap (NOT Chrome) and pc_click reuses cached OCR
    store = StudioStore(tmp_path / "studio")
    sess = store.create_session(title="Hermes Arka Plan")
    broker = PermissionBroker(store.notes_dir)
    emitted = []
    runner = ToolRunner(
        store=store,
        broker=broker,
        session_id=sess["id"],
        workspace_dir="",
        emit_event=lambda ev, d: emitted.append((ev, d)),
        pc_mode="background",
    )

    zap_win = {
        "address": "0xabc123",
        "stable_id": "18000014",
        "class": "com.rtosta.zapzap",
        "title": "ZapZap",
        "workspace_id": 4,
        "headless_monitor": "",
        "background": True,
        "at": [26, 66],
        "size": [1868, 988],
    }

    def fake_focus_smart(_wui, app_name, *, background=True, pc_state=None):
        pc_state["target_window"] = dict(zap_win)
        return {"action": "focused_in_background", "mode": "background_toplevel", "app": app_name, "class": "com.rtosta.zapzap", "title": "ZapZap", "workspace": 4}

    tess_calls = 0
    grim_args_seen = []

    def fake_subproc(cmd, **kwargs):
        nonlocal tess_calls
        if cmd and "grim" in str(cmd[0]):
            grim_args_seen.append(list(cmd))
            out_file = Path(cmd[-1])
            Image.new("RGB", (1868, 988), color=(20, 24, 22)).save(out_file)
            return ProcOk()
        if cmd and "tesseract" in str(cmd[0]):
            tess_calls += 1
        return ProcOk()

    clicked_coords = []

    with (
        patch.object(hermes_pc, "focus_or_launch_smart", side_effect=fake_focus_smart),
        patch.object(hermes_pc, "list_clients_full", return_value=[zap_win]),
        patch.object(hermes_pc, "remove_stale_headless_monitors"),
        patch.object(hermes_vision, "_best_single_lang", return_value="tur"),
        patch.object(hermes_vision.os.path, "exists", return_value=True),
        patch("subprocess.run", side_effect=fake_subproc),
        patch.object(hermes_pc, "fast_click", side_effect=lambda _wui, x, y, button="left", double=False: clicked_coords.append((x, y)) or {"ok": True}),
        patch.object(hermes_pc, "transient_target_focus"),
    ):
        hermes_vision._cached_ocr_lines.cache_clear()

        focus_out = asyncio.run(runner.execute("pc_focus_or_launch", '{"app": "zapzap"}'))
        assert "Arka Plan" in focus_out
        assert "com.rtosta.zapzap" in focus_out
        assert "Notlarim" in focus_out
        assert any("-T" in g and "18000014" in g for g in grim_args_seen)
        assert any(ev == "pc_screenshot" and "zapzap" in d["window"] for ev, d in emitted)
        assert tess_calls == 1

        # Now click "Notlarim" -> must reuse fresh last_snap_path from LRU cache without extra pre-click OCR!
        click_out = asyncio.run(runner.execute("pc_click", '{"text": "Notlarim"}'))
        assert "Notlarim" in click_out
        assert len(clicked_coords) == 1
        assert clicked_coords[0][0] > 26
        # Only 1 extra OCR call for post-click verification snapshot (0 pre-click OCR!)
        assert tess_calls == 2


def test_hermes_screen_capture_emits_clean_unboxed_png_and_full_zone_text(tmp_path):
    """Ensure YOLO/square box labeling (`annotate_elements`, `_icon_boxes`, `mark_elements`) is completely gone."""
    pc_schemas = hermes_pc.build_hermes_pc_schemas()
    cap_schema = next(s for s in pc_schemas if s["function"]["name"] == "pc_capture_screen")
    click_schema = next(s for s in pc_schemas if s["function"]["name"] == "pc_click")
    assert "mark_elements" not in cap_schema["function"]["parameters"]["properties"]
    assert "mark" not in click_schema["function"]["parameters"]["properties"]

    fake_tsv = (
        "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"
        "5\t1\t1\t1\t1\t1\t80\t40\t220\t24\t95.0\tMakine_Ogrenmesi\n"
        "5\t1\t2\t1\t1\t1\t60\t300\t140\t20\t91.0\tchap2_data.pdf\n"
        "5\t1\t3\t1\t1\t1\t600\t280\t380\t24\t94.0\tBagging_ve_Boosting_Karsilastirmasi\n"
        "5\t1\t4\t1\t1\t1\t1550\t320\t160\t22\t90.0\tBilgi_Kartlari\n"
    )

    class ProcOk:
        returncode = 0
        stdout = fake_tsv

    def fake_subproc(cmd, **kwargs):
        if cmd and "grim" in str(cmd[0]):
            Image.new("RGB", (1920, 1080), color=(18, 20, 22)).save(Path(cmd[-1]))
        return ProcOk()

    emitted = []
    store = StudioStore(tmp_path / "studio")
    sess = store.create_session(title="Temiz Ekran")
    runner = ToolRunner(
        store=store,
        broker=PermissionBroker(store.notes_dir),
        session_id=sess["id"],
        workspace_dir="",
        emit_event=lambda ev, d: emitted.append((ev, d)),
    )

    with (
        patch.object(hermes_vision, "_best_single_lang", return_value="tur"),
        patch.object(hermes_vision.os.path, "exists", return_value=True),
        patch("subprocess.run", side_effect=fake_subproc),
        patch.object(hermes_pc.wayland_ui, "annotate_elements", side_effect=AssertionError("annotate_elements must never run")),
        patch.object(hermes_pc.wayland_ui, "_icon_boxes", side_effect=AssertionError("_icon_boxes must never run")),
    ):
        hermes_vision._cached_ocr_lines.cache_clear()
        out = asyncio.run(runner.execute("pc_capture_screen", '{"scope": "full"}'))

    shot_events = [d for ev, d in emitted if ev == "pc_screenshot"]
    assert len(shot_events) == 1
    assert "/screenshots/snap_" in shot_events[0]["url"]
    assert "mark_" not in shot_events[0]["url"]
    assert "Ana İçerik" in out
    assert "Bagging_ve_Boosting_Karsilastirmasi" in out
    assert "chap2_data.pdf" in out
    assert "Bilgi_Kartlari" in out


def test_prefix_stable_schemas_and_turn2_tool_digest_memory():
    """Verify KV prefix invariance across diverse prompts and Turn-2 screen/tool memory preservation."""
    t1 = build_tool_schemas(mode="assistant", web_mode="auto", pc_mode="auto", prompt="selam", prefix_stable=True)
    t2 = build_tool_schemas(mode="assistant", web_mode="auto", pc_mode="auto", prompt="ekranda ne var anlat", prefix_stable=True)
    t3 = build_tool_schemas(mode="assistant", web_mode="auto", pc_mode="auto", prompt="2026 makale araştır ve notlar.md yaz", prefix_stable=True)
    assert t1 == t2 == t3

    r1 = build_tool_schemas(mode="research", web_mode="auto", pc_mode="off", prompt="merhaba", prefix_stable=True)
    r2 = build_tool_schemas(mode="research", web_mode="auto", pc_mode="off", prompt="SymPy ile entropi doğrula", prefix_stable=True)
    assert r1 == r2

    prior = [
        {"role": "user", "content": "ekran görüntüsü al"},
        {
            "role": "assistant",
            "content": "Ekran görüntüsü alındı.",
            "tool_events": [
                {
                    "id": "call_1",
                    "name": "pc_capture_screen",
                    "arguments": "{}",
                    "preview": "Soru 1 Bagging...",
                    "output": "### Ana İçerik\nSoru 1: Bagging ve Bootstrap olasılığı 0.632",
                }
            ],
        },
    ]
    sess_turn2 = {"research_state_md": "# Durum v1\n- Hipotez A", "research_phase": "hypothesize"}
    sess_turn3 = {"research_state_md": "# Durum v2\n- Hipotez A doğrulandı", "research_phase": "verify"}

    msgs_2 = build_chat_llm_messages(sess_turn2, mode="research", user_message="ne var ekranda anlatsana", prior_msgs=prior)
    msgs_3 = build_chat_llm_messages(sess_turn3, mode="research", user_message="sıradaki deneye geç", prior_msgs=prior)

    # System prompt + entire historical prefix (native OpenAI tool_calls + role="tool" + assistant) stay 100% identical!
    assert msgs_2[:-1] == msgs_3[:-1]
    assert msgs_2[2]["role"] == "assistant" and msgs_2[2]["tool_calls"][0]["function"]["name"] == "pc_capture_screen"
    assert msgs_2[3]["role"] == "tool" and "Soru 1: Bagging ve Bootstrap olasılığı 0.632" in msgs_2[3]["content"]
    assert "[Araç Çıktısı" not in msgs_2[4]["content"]
    # Dynamic research_state_md sits strictly AFTER user_message at the very end of the last turn
    assert msgs_2[-1]["content"].startswith("ne var ekranda anlatsana\n\n<kalici_arastirma_durumu")


def test_ai_session_naming_and_fake_tool_echo_recovery(tmp_path, monkeypatch):
    """Untitled sessions get an AI-generated title via KV-cached follow-up, and fake '[Araç Çıktısı]' echoes auto-recover."""
    import httpx as httpx_mod

    seen_payloads = []

    class FakeStreamResp:
        def __init__(self, lines):
            self._lines = lines

        def raise_for_status(self):
            return None

        async def aiter_lines(self):
            for ln in self._lines:
                yield ln

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

    class FakeAsyncClient:
        def __init__(self, **kwargs):
            pass

        def stream(self, method, url, *, json=None, **kwargs):
            seen_payloads.append(json)
            idx = len(seen_payloads)
            if idx == 1:
                # Model accidentally tries to emit fake "[Araç Çıktısı — ...]" text instead of a real tool_call!
                return FakeStreamResp([
                    'data: {"choices":[{"delta":{"content":"[Araç Çıktısı — pc_focus_or_launch]\\nSahte ekran metni"}}]}',
                    "data: [DONE]",
                ])
            if idx == 2:
                # Auto-correction nudge forces clean assistant reply
                return FakeStreamResp([
                    'data: {"choices":[{"delta":{"content":"ZapZap Notlar bölümündeki son mesajınız: bitkisi."}}]}',
                    "data: [DONE]",
                ])
            # idx == 3: KV-prefix-preserving AI session title generation!
            return FakeStreamResp([
                'data: {"choices":[{"delta":{"content":"\\"ZapZap Notlar Kontrolü\\"."}}]}',
                "data: [DONE]",
            ])

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

    monkeypatch.setattr(httpx_mod, "AsyncClient", FakeAsyncClient)

    app = create_studio_app(storage_dir=tmp_path / "studio", upstream_url="http://127.0.0.1:9/v1")
    client = TestClient(app)
    sess = client.post("/api/sessions", json={"title": "Yeni Sohbet", "mode": "assistant"}).json()
    assert sess["custom_title"] == 0

    resp = client.post(
        "/api/chat/stream",
        json={
            "session_id": sess["id"],
            "message": "zapzapa baksana notlara en son ne yazmışım",
            "mode": "assistant",
            "enable_thinking": False,
            "web_mode": "off",
            "pc_mode": "auto",
        },
    )
    assert resp.status_code == 200
    assert len(seen_payloads) == 3
    # Title request reuses identical tool schemas so llama-server hits 100% KV prefix cache
    assert seen_payloads[2].get("tools") == seen_payloads[0].get("tools")
    assert '"title": "ZapZap Notlar Kontrolü"' in resp.text

    updated_sess = client.get(f"/api/sessions/{sess['id']}").json()
    assert updated_sess["title"] == "ZapZap Notlar Kontrolü"
    assert updated_sess["custom_title"] == 1
    assert "[Araç Çıktısı" not in updated_sess["messages"][-1]["content"]
    assert "ZapZap Notlar bölümündeki son mesajınız" in updated_sess["messages"][-1]["content"]


def test_zapzap_chat_layout_keeps_right_bubbles_chronological_and_persists_pc_state(tmp_path):
    """Right-aligned outgoing bubbles stay in chronological Ana İçerik instead of splitting into a fake Sağ Panel."""
    hermes_vision._cached_ocr_lines.cache_clear()
    sample_img = tmp_path / "zapzap_full.png"
    Image.new("RGB", (1868, 988), color=(18, 22, 20)).save(sample_img)

    fake_tsv = (
        "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"
        "5\t1\t1\t1\t1\t1\t700\t45\t180\t24\t95.0\tErkek_Nesesi\n"
        "5\t1\t2\t1\t1\t1\t90\t360\t80\t22\t94.0\tNotlar\n"
        "5\t1\t3\t1\t1\t1\t680\t300\t220\t22\t92.0\tGelen_Mesaj_Bir\n"
        "5\t1\t4\t1\t1\t1\t1380\t420\t240\t22\t93.0\tBenim_Yazdigim_Mesaj\n"
        "5\t1\t5\t1\t1\t1\t690\t540\t210\t22\t91.0\tGelen_Mesaj_Iki\n"
    )

    class ProcOk:
        returncode = 0
        stdout = fake_tsv

    with (
        patch.object(hermes_vision.os.path, "exists", return_value=True),
        patch.object(hermes_vision, "_best_single_lang", return_value="tur"),
        patch("argus_cache.webui.hermes_vision.subprocess.run", return_value=ProcOk()),
    ):
        txt = hermes_vision.fast_read_screen_text(hermes_pc.wayland_ui, str(sample_img), width=1868, height=988)

    assert "Sağ Panel" not in txt
    assert "[Açık Başlık] Erkek_Nesesi" in txt
    assert "Notlar" in txt
    assert "[Sağ] Benim_Yazdigim_Mesaj" in txt
    assert txt.index("Gelen_Mesaj_Bir") < txt.index("[Sağ] Benim_Yazdigim_Mesaj") < txt.index("Gelen_Mesaj_Iki")


