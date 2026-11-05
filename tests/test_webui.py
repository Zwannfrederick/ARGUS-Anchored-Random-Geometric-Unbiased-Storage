"""Unit and integration tests for ARGUS Studio WebUI (.md conversion, permissions, tools, SSRF, FastAPI)."""

from __future__ import annotations

import asyncio
import json
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from argus_cache.webui.compaction import should_auto_compact
from argus_cache.webui.doc_markdown import (
    DIRECT_CONTEXT_CHAR_LIMIT,
    build_prompt_attachment_block,
    convert_file_to_markdown,
    html_to_markdown,
    query_sections,
)
from argus_cache.webui.permissions import PermissionBroker, is_sensitive_path
from argus_cache.webui.server import create_studio_app
from argus_cache.webui.store import StudioStore
from argus_cache.webui.tools import ToolRunner, build_tool_schemas
from argus_cache.webui.web_search import _assert_safe_public_url, _unwrap_ddg_url


def _make_sample_docx(path: Path) -> None:
    doc_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
    <w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
      <w:body>
        <w:p>
          <w:pPr><w:pStyle w:val="Heading1"/></w:pPr>
          <w:r><w:t>Termodinamik Ders Notu</w:t></w:r>
        </w:p>
        <w:p>
          <w:r><w:rPr><w:b/></w:rPr><w:t>Birinci Yasa:</w:t></w:r>
          <w:r><w:t> Enerji yoktan var edilemez.</w:t></w:r>
        </w:p>
        <w:tbl>
          <w:tr>
            <w:tc><w:p><w:r><w:t>Kavram</w:t></w:r></w:p></w:tc>
            <w:tc><w:p><w:r><w:t>Sembol</w:t></w:r></w:p></w:tc>
          </w:tr>
          <w:tr>
            <w:tc><w:p><w:r><w:t>Entropi</w:t></w:r></w:p></w:tc>
            <w:tc><w:p><w:r><w:t>S</w:t></w:r></w:p></w:tc>
          </w:tr>
        </w:tbl>
      </w:body>
    </w:document>"""
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("word/document.xml", doc_xml)


def test_docx_converts_headings_bold_and_tables_to_markdown(tmp_path):
    docx_file = tmp_path / "fizik.docx"
    _make_sample_docx(docx_file)

    conv = convert_file_to_markdown(docx_file)

    assert conv.fmt == "docx"
    assert "# Termodinamik Ders Notu" in conv.markdown
    assert "**Birinci Yasa:**" in conv.markdown
    assert "| Kavram | Sembol |" in conv.markdown
    assert "| Entropi | S |" in conv.markdown


def test_html_to_markdown_strips_scripts_and_preserves_structure():
    raw = """
    <html><head><script>alert(1)</script><style>body{}</style></head>
    <body>
      <nav>Menü</nav>
      <h1>Başlık</h1>
      <p>Bu bir <a href="https://example.com/makale">kaynak</a> metnidir.</p>
      <ul><li>Madde bir</li><li>Madde iki</li></ul>
    </body></html>
    """
    md = html_to_markdown(raw)
    assert "alert(1)" not in md
    assert "Menü" not in md
    assert "# Başlık" in md
    assert "[kaynak](https://example.com/makale)" in md
    assert "- Madde bir" in md


def test_large_document_uses_index_instead_of_raw_dump(tmp_path):
    store = StudioStore(tmp_path / "studio")
    sess = store.create_session(title="Ders")
    sections = [
        {"index": i, "title": f"Bölüm {i}: Konu {i}", "content": f"## Bölüm {i}\n\n" + (f"Detay_{i} " * 400)}
        for i in range(1, 12)
    ]
    big_md = "\n\n".join(s["content"] for s in sections)
    assert len(big_md) > DIRECT_CONTEXT_CHAR_LIMIT

    saved = store.save_document(
        session_id=sess["id"],
        filename="kapsamli_kitap.pdf",
        fmt="pdf",
        markdown=big_md,
        sections=sections,
        page_count=11,
    )
    doc = store.get_document(saved["id"])
    prompt_block = build_prompt_attachment_block(doc, "Detay_9 konusunu anlat")

    assert "<ekli_belge_indeksi" in prompt_block
    assert len(prompt_block) < len(big_md)
    assert "Detay_9" in prompt_block

    queried = query_sections(doc["sections"], query="Detay_7")
    assert "Bölüm 7" in queried


def test_sensitive_paths_and_ssrf_are_blocked():
    assert is_sensitive_path(Path.home() / ".ssh" / "id_ed25519") is True
    assert is_sensitive_path(Path("/srv/app/.env")) is True
    assert is_sensitive_path(Path("/srv/app/notlar.md")) is False

    with pytest.raises(PermissionError):
        _assert_safe_public_url("http://127.0.0.1:8080/props")

    unwrapped = _unwrap_ddg_url("//duckduckgo.com/l/?uddg=https%3A%2F%2Fkernel.org%2Fdoc")
    assert unwrapped == "https://kernel.org/doc"


def test_write_and_edit_tools_require_explicit_user_approval(tmp_path):
    store = StudioStore(tmp_path / "studio")
    broker = PermissionBroker(store.notes_dir)
    sess = store.create_session(title="Kod")
    ws = tmp_path / "workspace"
    ws.mkdir()

    events = []

    def emit(ev, payload):
        events.append((ev, payload))
        if ev == "permission_request":
            decision = "deny" if "reddet" in payload["target_path"] else "allow_once"
            broker.resolve(payload["req_id"], decision)

    runner = ToolRunner(
        store=store,
        broker=broker,
        session_id=sess["id"],
        workspace_dir=str(ws),
        emit_event=emit,
    )

    ok_res = asyncio.run(runner.execute("write_local_file", '{"path": "ozet.md", "content": "# Özet\\nMerhaba"}'))
    assert "Başarıyla kaydedildi" in ok_res
    assert (ws / "ozet.md").read_text(encoding="utf-8") == "# Özet\nMerhaba"
    assert any(ev == "permission_request" and "+# Özet" in p["diff_preview"] for ev, p in events)

    deny_res = asyncio.run(runner.execute("write_local_file", '{"path": "reddet.md", "content": "x"}'))
    assert "reddetti" in deny_res
    assert not (ws / "reddet.md").exists()


def test_studio_fastapi_upload_and_canvas_endpoints(tmp_path):
    app = create_studio_app(storage_dir=tmp_path / "studio")
    client = TestClient(app)

    assert client.get("/health").json()["status"] == "ok"
    sess = client.post("/api/sessions", json={"title": "Fizik 101", "mode": "study"}).json()

    docx_path = tmp_path / "ders.docx"
    _make_sample_docx(docx_path)
    with docx_path.open("rb") as fp:
        up = client.post(
            "/api/upload",
            data={"session_id": sess["id"]},
            files={"file": ("ders.docx", fp, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
        )
    assert up.status_code == 200
    up_data = up.json()
    assert up_data["fmt"] == "docx"

    dl = client.get(f"/api/docs/{up_data['id']}?download=1")
    assert dl.status_code == 200
    assert "# Termodinamik Ders Notu" in dl.text

    patched = client.patch(
        f"/api/sessions/{sess['id']}",
        json={"canvas_title": "Vize Özeti", "canvas_md": "# Vize\n- Madde 1"},
    ).json()
    assert patched["canvas_title"] == "Vize Özeti"
    assert "# Vize" in patched["canvas_md"]


def test_contextual_tool_gating_keeps_casual_chat_zero_bloat():
    assert build_tool_schemas(web_mode="auto", prompt="naber nasılsın") == []

    web_names = [t["function"]["name"] for t in build_tool_schemas(web_mode="auto", prompt="2026 Linux haberlerini araştır")]
    assert "web_search" in web_names and "read_webpage" in web_names

    doc_names = [t["function"]["name"] for t in build_tool_schemas(web_mode="off", has_docs=True, prompt="ilk bölümü anlat")]
    assert doc_names == ["read_uploaded_doc"]

    file_names = [t["function"]["name"] for t in build_tool_schemas(web_mode="off", prompt="Masaüstü klasöründeki notlar.md dosyasını oku")]
    assert "read_local_file" in file_names and "write_local_file" in file_names

    study_names = [t["function"]["name"] for t in build_tool_schemas(web_mode="off", mode="study", prompt="konuyu anlat")]
    assert study_names == ["publish_quiz", "publish_flashcards", "update_canvas"]

    draft_names = [t["function"]["name"] for t in build_tool_schemas(web_mode="off", mode="draft", prompt="yeni fikir yaz")]
    assert draft_names == ["update_canvas"]

    research_names = [t["function"]["name"] for t in build_tool_schemas(web_mode="auto", mode="research", prompt="entropi hipotezi")]
    assert "web_search" in research_names
    assert "verify_with_python" in research_names
    assert "update_research_state" in research_names

    pc_names = [t["function"]["name"] for t in build_tool_schemas(web_mode="off", mode="assistant", pc_mode="auto", prompt="açık pencereleri listele")]
    assert "pc_list_windows" in pc_names and "pc_terminal_command" in pc_names

    hist_names = [t["function"]["name"] for t in build_tool_schemas(web_mode="off", has_history=True, prompt="devam et")]
    assert hist_names == ["search_chat_history"]


def test_non_destructive_compaction_and_history_search_tool(tmp_path):
    app = create_studio_app(storage_dir=tmp_path / "studio", upstream_url="http://127.0.0.1:9/v1")
    store: StudioStore = app.state.store
    broker: PermissionBroker = app.state.broker
    client = TestClient(app)

    sess = client.post("/api/sessions", json={"title": "Uzun Proje"}).json()
    sid = sess["id"]

    store.add_message(sid, role="user", content="İlk kararımız GIZLI_MIMARI_ANAHTARI_99 olsun.")
    store.add_message(
        sid,
        role="assistant",
        content="Tamam, anahtarı kaydettim ve config dosyasını yazdım.",
        tool_events=[{"id": "c1", "name": "write_local_file", "arguments": '{"path":"cfg.md"}', "preview": "cfg.md yazıldı"}],
    )
    for i in range(16):
        store.add_message(sid, role="user" if i % 2 == 0 else "assistant", content=f"Ara tartışma adımı {i}")

    before = store.get_session(sid)
    assert len(before["messages"]) == 18
    assert should_auto_compact(before) is True

    comp_resp = client.post(f"/api/sessions/{sid}/compact", json={"keep_recent": 4})
    assert comp_resp.status_code == 200
    cdata = comp_resp.json()
    assert cdata["compacted"] is True
    assert "Sohbet Özeti" in cdata["compacted_summary"]
    assert "Yapılan İşlemler" in cdata["compacted_summary"]
    assert "write_local_file" in cdata["compacted_summary"]

    # Zero messages deleted!
    after = client.get(f"/api/sessions/{sid}").json()
    assert len(after["messages"]) == 18
    assert after["compacted_until_msg_id"] > 0

    runner = ToolRunner(
        store=store,
        broker=broker,
        session_id=sid,
        workspace_dir="",
        emit_event=lambda *_: None,
    )
    recalled = asyncio.run(runner.execute("search_chat_history", '{"query": "GIZLI_MIMARI_ANAHTARI_99"}'))
    assert "GIZLI_MIMARI_ANAHTARI_99" in recalled
    assert "Arşivlenmiş" in recalled

    range_out = asyncio.run(runner.execute("search_chat_history", '{"start_id": 1, "end_id": 2}'))
    assert "write_local_file" in range_out


def test_parallel_multi_query_web_search_merges_and_deduplicates(tmp_path, monkeypatch):
    import argus_cache.webui.tools as tools_mod

    async def fake_parallel(queries, *, max_results_per_query=4):
        return [
            (
                queries[0],
                [{"title": "Kaynak A", "url": "https://example.org/a", "snippet": "Detay A", "domain": "example.org"}],
            ),
            (
                queries[1],
                [
                    {"title": "Kaynak A", "url": "https://example.org/a", "snippet": "Detay A", "domain": "example.org"},
                    {"title": "Kaynak B", "url": "https://example.org/b", "snippet": "Detay B", "domain": "example.org"},
                ],
            ),
        ]

    monkeypatch.setattr(tools_mod, "parallel_web_search", fake_parallel)
    store = StudioStore(tmp_path / "studio")
    sess = store.create_session()
    emitted = []
    runner = ToolRunner(
        store=store,
        broker=PermissionBroker(store.notes_dir),
        session_id=sess["id"],
        workspace_dir="",
        emit_event=lambda ev, d: emitted.append((ev, d)),
    )
    out = asyncio.run(runner.execute("web_search", '{"queries": ["linux kernel", "qwen moe"]}'))
    assert "### Sorgu: `linux kernel`" in out
    assert "### Sorgu: `qwen moe`" in out
    assert len(runner.citations) == 2
    assert any(ev == "citations" and len(d["items"]) == 2 for ev, d in emitted)


def test_scientific_python_verifier_and_safety_sandbox(tmp_path):
    from argus_cache.webui.research_tools import run_scientific_python

    sympy_out = run_scientific_python("import sympy as sp\nx = sp.Symbol('x')\nsp.integrate(3*x**2 + 2*x, (x, 0, 2))")
    assert "✓ Bağımsız Python Doğrulama Çıktısı:" in sympy_out
    assert "12" in sympy_out

    blocked_net = run_scientific_python("import socket\nsocket.gethostname()")
    assert "Güvenlik sınırı" in blocked_net and "socket" in blocked_net

    blocked_open = run_scientific_python("open('/etc/passwd').read()")
    assert "Güvenlik sınırı" in blocked_open and "open" in blocked_open


def test_research_state_persistence_and_citation_hallucination_guard(tmp_path, monkeypatch):
    import argus_cache.webui.tools as tools_mod
    from argus_cache.webui.research_tools import audit_citations

    async def fake_search(query, *, max_results=5):
        return [{"title": "NIST Kimya Verisi", "url": "https://webbook.nist.gov/chemistry/h2o", "snippet": "Entalpi", "domain": "webbook.nist.gov"}]

    monkeypatch.setattr(tools_mod, "web_search", fake_search)
    store = StudioStore(tmp_path / "studio")
    sess = store.create_session(title="Termodinamik AR-GE", mode="research")
    emitted = []
    runner = ToolRunner(
        store=store,
        broker=PermissionBroker(store.notes_dir),
        session_id=sess["id"],
        workspace_dir="",
        emit_event=lambda ev, d: emitted.append((ev, d)),
    )

    asyncio.run(runner.execute("web_search", '{"query": "water enthalpy nist"}'))
    assert "https://webbook.nist.gov/chemistry/h2o" in runner.verified_urls

    py_res = asyncio.run(runner.execute("verify_with_python", '{"code": "round(8.314462 * 298.15, 2)"}'))
    assert "2478.96" in py_res
    assert runner.python_verified is True

    state_res = asyncio.run(
        runner.execute(
            "update_research_state",
            json.dumps({
                "phase": "3_dogrula",
                "question": "Standart sıcaklıkta RT çarpımı nedir?",
                "hypotheses": [
                    {"id": "H1", "statement": "RT ≈ 2.48 kJ/mol", "status": "DOGRULANDI", "evidence": "Python: 2478.96 J/mol"}
                ],
                "verified_findings": ["RT = 2478.96 J/mol olarak hesaplandı."],
                "critiques": ["İdeal gaz varsayımı yüksek basınçta sapar."],
                "next_experiment": "Van der Waals düzeltmesini hesapla.",
                "sources": [
                    "https://webbook.nist.gov/chemistry/h2o",
                    "https://uydurma-akademi-makale.fake/2099/kesif",
                ],
            }),
        )
    )
    assert "UYARI (Uydurma Kaynak Engeli)" in state_res
    assert "https://uydurma-akademi-makale.fake/2099/kesif" in state_res

    saved = store.get_session(sess["id"])
    assert saved["research_phase"] == "3_dogrula"
    assert "https://webbook.nist.gov/chemistry/h2o" in saved["research_state_md"]
    assert "Uydurma Kaynak Koruması" in saved["research_state_md"]
    disk_file = store.research_state_file(sess["id"])
    assert disk_file.is_file()
    assert "H1" in disk_file.read_text(encoding="utf-8")

    post_audit = audit_citations(
        "Kaynaklar: https://webbook.nist.gov/chemistry/h2o ve https://sahte.org/x",
        runner.verified_urls,
    )
    assert post_audit["verified_urls"] == ["https://webbook.nist.gov/chemistry/h2o"]
    assert post_audit["unverified_urls"] == ["https://sahte.org/x"]
    assert post_audit["all_verified"] is False


def test_interactive_quiz_and_flashcards_tools(tmp_path):
    store = StudioStore(tmp_path / "studio")
    sess = store.create_session(title="Fizik Vize", mode="study")
    emitted = []
    runner = ToolRunner(
        store=store,
        broker=PermissionBroker(store.notes_dir),
        session_id=sess["id"],
        workspace_dir="",
        emit_event=lambda ev, d: emitted.append((ev, d)),
    )

    q_out = asyncio.run(
        runner.execute(
            "publish_quiz",
            json.dumps({
                "title": "Newton Mekaniği Testi",
                "questions": [
                    {
                        "question": "Net kuvvet sıfırsa cismin ivmesi nedir?",
                        "options": ["0 m/s²", "9.8 m/s²", "Sabit artar", "Kütleye bağlıdır"],
                        "correct_index": 0,
                        "explanation": "F = m·a gereği F_net = 0 ise a = 0 olur.",
                        "hint": "Newton'un birinci yasasını hatırla.",
                        "difficulty": "Kolay",
                    }
                ],
            }),
        )
    )
    assert "Newton Mekaniği Testi" in q_out
    assert any(ev == "study_quiz" and len(d["questions"]) == 1 for ev, d in emitted)

    f_out = asyncio.run(
        runner.execute(
            "publish_flashcards",
            json.dumps({
                "title": "Termodinamik Kartları",
                "cards": [
                    {"front": "Entropi (S)", "back": "Sistemin mikro-durum düzensizlik ölçüsü (J/K)", "tag": "2. Yasa", "mnemonic": "S = k·ln(W)"}
                ],
            }),
        )
    )
    assert "Termodinamik Kartları" in f_out
    assert any(ev == "study_flashcards" and len(d["cards"]) == 1 for ev, d in emitted)

    reloaded = store.get_session(sess["id"])
    assert reloaded["study_quiz"]["title"] == "Newton Mekaniği Testi"
    assert reloaded["study_flashcards"]["cards"][0]["front"] == "Entropi (S)"


def test_hermes_pc_tools_focus_guard_and_exec_permission(tmp_path, monkeypatch):
    import argus_cache.webui.hermes_pc as pc_mod

    monkeypatch.setattr(
        pc_mod.wayland_ui,
        "list_windows",
        lambda: [{"workspace_id": 2, "class": "kitty", "title": "zsh", "pid": 4242}],
    )
    monkeypatch.setattr(
        pc_mod.wayland_ui,
        "get_active_window",
        lambda: {"class": "google-chrome", "title": "ARGUS Studio"},
    )

    store = StudioStore(tmp_path / "studio")
    broker = PermissionBroker(store.notes_dir)
    sess = store.create_session(title="Hermes PC")
    emitted = []
    runner = ToolRunner(
        store=store,
        broker=broker,
        session_id=sess["id"],
        workspace_dir=str(tmp_path),
        emit_event=lambda ev, d: emitted.append((ev, d)),
    )

    win_out = asyncio.run(runner.execute("pc_list_windows", "{}"))
    assert "[WS 2] kitty — zsh" in win_out

    # Wrong window focused -> typing into kitty must be rejected safely!
    guard_out = asyncio.run(runner.execute("pc_type_text", '{"text": "echo selam", "target_app": "kitty"}'))
    assert "Yazma reddedildi" in guard_out

    # Terminal command requires explicit user permission and respects denial / approval
    async def run_denied_and_approved():
        def handle_event(ev, d):
            emitted.append((ev, d))
            if ev == "permission_request":
                broker.resolve(d["req_id"], decision=handle_event.decision)

        handle_event.decision = "deny"
        runner.emit_event = handle_event
        denied = await runner.execute("pc_terminal_command", '{"command": "echo MERHABA_HERMES"}')
        assert "onaylamadı" in denied

        handle_event.decision = "allow_once"
        approved = await runner.execute("pc_terminal_command", '{"command": "echo MERHABA_HERMES"}')
        assert "Çıkış Kodu: 0" in approved
        assert "MERHABA_HERMES" in approved

    asyncio.run(run_denied_and_approved())


def test_thinking_toggle_sets_unrestricted_budget_when_enabled(tmp_path, monkeypatch):
    import httpx as httpx_mod

    captured_payloads = []
    captured_timeouts = []

    class FakeStreamResp:
        def raise_for_status(self):
            return None

        async def aiter_lines(self):
            yield 'data: {"choices":[{"delta":{"reasoning_content":"uzun muhakeme...","content":"sonuç"}}]}'
            yield "data: [DONE]"

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

    class FakeAsyncClient:
        def __init__(self, *, timeout=None, **kwargs):
            captured_timeouts.append(timeout)

        def stream(self, method, url, *, json=None, **kwargs):
            captured_payloads.append(json)
            return FakeStreamResp()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

    monkeypatch.setattr(httpx_mod, "AsyncClient", FakeAsyncClient)

    app = create_studio_app(storage_dir=tmp_path / "studio", upstream_url="http://127.0.0.1:9/v1")
    client = TestClient(app)
    sess = client.post("/api/sessions", json={"title": "Düşünme Testi"}).json()

    # 1. Thinking ON -> NO token limit (n_predict=-1, reasoning_budget_tokens=-1, read timeout=None, no max_tokens)
    r_on = client.post(
        "/api/chat/stream",
        json={"session_id": sess["id"], "message": "selam", "enable_thinking": True, "web_mode": "off", "pc_mode": "off"},
    )
    assert r_on.status_code == 200
    assert captured_payloads[0]["chat_template_kwargs"] == {"enable_thinking": True}
    assert captured_payloads[0]["n_predict"] == -1
    assert captured_payloads[0]["reasoning_budget_tokens"] == -1
    assert "max_tokens" not in captured_payloads[0]
    assert captured_timeouts[0].read is None

    # 2. Thinking OFF -> reasoning_budget_tokens=0 (immediate skip), answer n_predict=-1
    r_off = client.post(
        "/api/chat/stream",
        json={"session_id": sess["id"], "message": "selam", "enable_thinking": False, "web_mode": "off", "pc_mode": "off"},
    )
    assert r_off.status_code == 200
    assert captured_payloads[1]["chat_template_kwargs"] == {"enable_thinking": False}
    assert captured_payloads[1]["n_predict"] == -1
    assert captured_payloads[1]["reasoning_budget_tokens"] == 0







def test_shortcuts_come_from_the_live_hyprland_bind_list():
    """The tool copied commands out of hyprland.lua and drifted when it changed.
    It now reads the compositor's binds and presses the key combination itself."""
    from argus_cache.webui.hermes_pc import parse_live_shortcuts

    binds = [
        {"modmask": 12, "key": "T", "description": "Terminal aç"},
        {"modmask": 64, "key": "1", "description": "Çalışma alanına geç"},
        {"modmask": 64, "key": "2", "description": "Çalışma alanına geç"},
        {"modmask": 0, "key": "XF86AudioRaiseVolume", "description": "Sesi artır"},
        {"modmask": 64, "key": "mouse:272", "description": "Sürükle", "mouse": True},
        {"modmask": 64, "key": "SUPER_L", "description": "Güç menüsü", "release": True},
        {"modmask": 0, "key": "x", "description": "-"},
    ]
    table = parse_live_shortcuts(binds, {"t", "1", "2", "xf86audioraisevolume", "x", "super_l"})
    assert table == {
        "Terminal aç": (["ctrl", "alt"], "T"),
        "Çalışma alanına geç 1": (["super"], "1"),
        "Çalışma alanına geç 2": (["super"], "2"),
        "Sesi artır": ([], "XF86AudioRaiseVolume"),
    }


def test_desktop_shortcuts_that_open_windows_need_approval(tmp_path, monkeypatch):
    """Only volume/media/brightness keys run silently; anything that pops a window asks.

    A memory question ("what was my last message?") once made the model fire
    rofi and launch a terminal on the user's screen without asking.
    """
    from argus_cache.webui import hermes_pc

    if hermes_pc.HERMES_UNAVAILABLE:
        pytest.skip(hermes_pc.HERMES_UNAVAILABLE)
    pressed = []
    monkeypatch.setattr(hermes_pc, "live_shortcuts", lambda: {
        "Sesi artır": ([], "XF86AudioRaiseVolume"), "Uygulama başlatıcı": (["alt"], "SPACE"),
    })
    monkeypatch.setattr(hermes_pc.wayland_ui, "simulate_key", lambda key, mods: pressed.append(key) or True)
    broker = PermissionBroker(tmp_path)

    async def run(action):
        events = []
        task = asyncio.create_task(hermes_pc.execute_hermes_pc_tool(
            "pc_shortcut", {"action": action}, session_id="s", workspace_dir="",
            screenshots_dir=tmp_path, broker=broker, pc_state={},
            emit_event=lambda kind, data: events.append((kind, data)),
        ))
        await asyncio.sleep(0.05)
        for kind, data in events:
            if kind == "permission_request":
                assert data["operation"] == "desktop"
                broker.resolve(data["req_id"], "deny")
        await task
        return [kind for kind, _ in events]

    assert asyncio.run(run("Sesi artır")) == []
    assert pressed == ["XF86AudioRaiseVolume"]
    assert asyncio.run(run("Uygulama başlatıcı")) == ["permission_request"]
    assert pressed == ["XF86AudioRaiseVolume"]


def _word(text, x0, y0, x1, y1, conf=90.0):
    return {"text": text, "box": [x0, y0, x1, y1], "conf": conf}


def test_ocr_line_keeps_reading_order_when_right_words_sit_a_pixel_higher():
    """A chat bubble line came back split and reversed: the right half's boxes
    were 1 px higher, so it opened its own line before the left half arrived."""
    from argus_cache.webui.hermes_vision import group_words_into_lines

    words = [
        _word("yoo", 1244, 341, 1270, 356), _word("ai", 1276, 341, 1290, 356),
        _word("yaptı", 1296, 341, 1340, 356), _word("net", 1600, 340, 1625, 355),
        _word("bi", 1631, 340, 1645, 355), _word("hüküm", 1651, 340, 1700, 355),
    ]
    lines = group_words_into_lines(words)
    assert [" ".join(w["text"] for w in ln) for ln in lines] == ["yoo ai yaptı", "net bi hüküm"]


def test_ocr_keeps_side_by_side_columns_apart():
    """Two bubbles on the same row (left incoming, right outgoing) stay separate lines."""
    from argus_cache.webui.hermes_vision import group_words_into_lines

    words = [_word("Merhaba", 746, 400, 820, 415), _word("Selam", 1500, 401, 1550, 416)]
    assert [[w["text"] for w in ln] for ln in group_words_into_lines(words)] == [["Merhaba"], ["Selam"]]


def test_sidebar_split_follows_the_layout_not_a_fixed_fraction():
    """ZapZap's chat list ends at ~640 px on a 1868 px window; a fixed 33% split
    (616 px) pushed the list's timestamps into the conversation."""
    from argus_cache.webui.hermes_vision import find_sidebar_split

    boxes = [[223, y, 640, y + 15] for y in range(200, 900, 40)]        # list rows incl. timestamps
    boxes += [[746, y, 1200, y + 15] for y in range(200, 900, 60)]      # incoming bubbles
    boxes += [[1250, y, 1790, y + 15] for y in range(230, 900, 60)]     # outgoing bubbles
    split = find_sidebar_split(boxes, width=1868, height=988)
    assert 640 <= split <= 746
    assert find_sidebar_split([], width=1868, height=988) == int(1868 * 0.33)


def test_old_screen_reads_are_not_replayed_as_current_screen():
    """A past turn's OCR dump of one chat made the model answer about another.

    Screen state goes stale the moment the turn ends; history keeps it under a
    fixed 'old screen, read again' label so it is never taken for the current one.
    """
    from argus_cache.webui.server import build_chat_llm_messages

    prior = [
        {"id": 1, "role": "user", "content": "erkek neşesine bak"},
        {"id": 2, "role": "assistant", "content": "Baktım.", "tool_events": [
            {"id": "c1", "name": "pc_capture_screen", "arguments": "{}", "output": "ESKI EKRAN: Erkek Neşesi mesajları"},
            {"id": "c2", "name": "pc_terminal_command", "arguments": "{}", "output": "Çıkış Kodu: 0\nok"},
        ]},
    ]
    msgs = build_chat_llm_messages({}, mode="assistant", user_message="notlara bak", prior_msgs=prior)
    tool_text = {m["name"]: m["content"] for m in msgs if m.get("role") == "tool"}
    from argus_cache.webui.server import STALE_SCREEN_LABEL

    assert tool_text["pc_capture_screen"].startswith(STALE_SCREEN_LABEL)
    assert "ESKI EKRAN: Erkek Neşesi" in tool_text["pc_capture_screen"]  # follow-ups still work
    assert tool_text["pc_terminal_command"] == "Çıkış Kodu: 0\nok"


def test_compaction_counts_the_tool_output_that_history_replays():
    """Tool outputs are replayed every turn; a trigger blind to them let context
    grow unchecked through tool-heavy turns made of short messages."""
    from argus_cache.webui.compaction import AUTO_COMPACT_CHAR_THRESHOLD, estimate_active_tokens, should_auto_compact

    big = "x" * 2000
    msgs = [{"id": i, "role": "assistant" if i % 2 else "user", "content": "kısa",
             "tool_events": [{"name": "web_search", "output": big}] * 3 if i % 2 else []}
            for i in range(1, 13)]
    sess = {"messages": msgs, "auto_compact": 1}
    assert estimate_active_tokens(sess) > 6 * 3 * 2000 // 4
    assert 6 * 3 * 2000 >= AUTO_COMPACT_CHAR_THRESHOLD
    assert should_auto_compact(sess)


def test_chat_lines_drop_timestamps_and_mark_quoted_replies():
    """WhatsApp OCR read "16:03 ✓✓" as a message ("Siz: 160347") and gave a reply's
    quoted text to the replier ("Yusuf: Sınavdan çıktım yatıyorum")."""
    from argus_cache.webui.hermes_vision import annotate_chat_lines

    lines = [
        "[Sağ] 160347", "[Sağ] Sınavdan çıktım yatıyorum", "[Sağ] 03 y",
        "Yusuf", "Siz", "Sınavdan çıktım yatıyorum", "go 16:05",
        "Yusuf", "Siz", "Sınavdan çıktım yat…", "adam benden önce çıkacaktı djslsmsimsis",
        "[Sağ] Yusuf", "[Sağ] adam benden önce çıkacaktı dislsmsimsis", "[Sağ] Amcı",
        "[Sağ] 1609",
    ]
    assert annotate_chat_lines(lines) == [
        "[Sağ] Sınavdan çıktım yatıyorum",
        "Yusuf", "Siz", "↪ alıntı: Sınavdan çıktım yatıyorum", "go 16:05",
        "Yusuf", "Siz", "↪ alıntı: Sınavdan çıktım yat…", "adam benden önce çıkacaktı djslsmsimsis",
        "[Sağ] Yusuf", "[Sağ] ↪ alıntı: adam benden önce çıkacaktı dislsmsimsis", "[Sağ] Amcı",
    ]
