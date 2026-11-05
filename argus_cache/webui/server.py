"""FastAPI + SSE server for ARGUS Studio WebUI (port 8090)."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from argus_cache.webui import DEFAULT_STUDIO_PORT
from argus_cache.webui.anthropic_routes import register_anthropic_routes
from argus_cache.webui.agent_loop import StreamManager, run_agent_stream, strip_fake_tool_echo
from argus_cache.webui.compaction import (
    REPLAYED_TOOL_OUTPUT_CHARS,
    compact_session_context,
    get_uncompacted_messages,
    should_auto_compact,
)
from argus_cache.webui.doc_markdown import build_prompt_attachment_block, convert_file_to_markdown
from argus_cache.webui.hermes_pc import SCREEN_STATE_TOOLS
from argus_cache.webui.hermes_vision import find_matching_window
from argus_cache.webui.permissions import PermissionBroker
from argus_cache.webui.store import StudioStore, is_untitled_name
from argus_cache.webui.tools import ToolRunner, build_tool_schemas

RUNTIME_DIR = Path.home() / ".argus_runtime"
STUDIO_DIR = RUNTIME_DIR / "studio"
STATIC_DIR = Path(__file__).parent / "static"

_TOOL_DISCIPLINE = (
    "\nKullanıcı web araması, bilimsel hesaplama, kaynak denetimi, test/flashcard, masaüstü/ekran okuma, "
    "geçmiş mesaj sorgulama, yerel dosya veya Canvas işlemi istediğinde tahmin yürütmek yerine ilgili aracı çağır; "
    "ekran/araç çıktısında gerçekten bulunmayan metin veya butonları asla uydurma. "
    "Kullanıcı 'ilk mesajım neydi / az önce ne sordum / ne konuşmuştuk' diye sorduğunda son mesajı tekrarlamak yerine "
    "yukarıdaki geçmiş `user` mesajlarını sırayla kontrol et (bağlam sıkıştırılmışsa `search_chat_history` çağır). "
    "`update_canvas` aracını YALNIZCA kullanıcı açıkça not/taslak/Canvas istediğinde çağır."
)

_MODE_PROMPTS = {
    "assistant": (
        "Sen ARGUS Studio yerel yapay zeka ve Hermes PC asistanısın. Kullanıcının dilinde (Türkçe ise akıcı Türkçe) "
        "doğrudan, doğru ve net yanıt ver; iç düşünceyi cevaba sızdırma. Masaüstü/uygulama işlemlerinde "
        "Hermes varsayılan olarak kullanıcının ekranını bozmayan arka plan pencere yakalama modunda çalışır:\n"
        "1. Uygulamayı hazırlamak için `pc_focus_or_launch(app_name=...)` çağır; bu araç güncel ekran OCR metnini döndürür.\n"
        "2. ÇOK ADIMLI UYGULAMA GEZİNTİSİ (KRİTİK): ZapZap/WhatsApp, Discord, Telegram veya tarayıcı gibi sol panelli "
        "uygulamalarda önce `Üst Çubuk / Başlık` içindeki `[Açık Başlık]` satırına bak. Kullanıcının sorduğu sohbet, kişi, "
        "grup, klasör veya sekme (ör. `Notlar`, `Favoriler`, kişi/grup adı) şu an açık olan başlık DEĞİLSE — `Sol Panel` "
        "listesinde 1 satırlık önizlemesini görsen bile — ASLA önizlemeyi veya başka açık sohbetin mesajlarını cevap diye sunma! "
        "Hemen bir sonraki adımda `pc_click(text='Notlar')` (veya listede yoksa arama kutusuna tıklayıp `pc_type_text` + `pc_click`) "
        "çağırarak hedef sohbeti/sekmeyi aç, yeni açılan `Ana İçerik` mesajlarını oku ve ancak ondan sonra yanıt ver.\n"
        "3. `Ana İçerik` bölümünde `[Sağ]` etiketi sağa yaslı balonları (mesajlaşma uygulamalarında kullanıcının kendi gönderdiği "
        "`Siz` mesajlarını), etiketsiz satırlar ise karşı tarafın mesajlarını gösterir.\n"
        "4. ASLA metin olarak sahte `[Araç Çıktısı — ...]` yazma; her ekran okuma ve tıklama gerçek bir araç çağrısı olmalıdır."
        + _TOOL_DISCIPLINE
    ),
    "research": (
        "Sen ARGUS Bilimsel AR-GE ve Doğrulanmış Araştırma ajanısın. Çalışma döngün: "
        "1. Araştır (`web_search(queries=[...])` + `read_webpage`) -> "
        "2. Hipotez Kur (test edilebilir öneriler) -> "
        "3. Doğrula (`verify_with_python` ile SymPy/NumPy/math bağımsız hesaplama) -> "
        "4. Eleştir (`audit_citations` + sınır koşulları ve karşıt kanıt kontrolü) -> "
        "5. Kaydet (`update_research_state` ile hipotezleri, elenenleri ve sıradaki deneyi `research_state.md`'ye yaz).\n"
        "KESİN KURALLAR:\n"
        "- İç düşüncede (`<think>`) bir araç çağırmaya karar verdiğinde ASLA boş yanıtla durup kullanıcıdan girdi bekleme; "
        "aracı hemen çağır, sonucunu incele ve görev tamamlanana kadar çok adımlı döngüyü sürdür.\n"
        "- Fizik, kimya, matematik ve sayısal iddialarda salt iç düşünceye güvenmek YASAKTIR; `verify_with_python` ile doğrula.\n"
        "- Araç çıktısında bizzat görmediğin hiçbir URL, DOI veya makale adını ASLA uydurma; `audit_citations` ile kaynakları denetle."
        + _TOOL_DISCIPLINE
    ),
    "study": (
        "Sen ARGUS Ders Çalışma & Akademik Etüt Stüdyosu asistanısın. Kullanıcının dilinde net tanımlar, formüller ve somut örneklerle anlat. "
        "Test/quiz/sınav istendiğinde `publish_quiz` aracını (cevapları sohbette açık etmeden!), bilgi/ezber kartı istendiğinde "
        "`publish_flashcards` aracını, ders notu/özet kağıdı istendiğinde `update_canvas` aracını, sayısal problem çözümünde ise "
        "`verify_with_python` aracını mutlaka çağır."
        + _TOOL_DISCIPLINE
    ),
    "draft": (
        "Sen ARGUS Fikir & Taslak Stüdyosu asistanısın. Ham fikirleri yapılandırılmış Markdown dokümanlarına dönüştür; "
        "taslakları her adımda `update_canvas` aracıyla sağdaki Canvas paneline yaz, sohbette yalnızca kısa özet ve kritik tasarım kararlarını paylaş."
        + _TOOL_DISCIPLINE
    ),
}


def build_chat_llm_messages(
    sess: Dict[str, Any],
    *,
    mode: str,
    user_message: str,
    prior_msgs: List[Dict[str, Any]],
    ws_dir: str = "",
    attachment_blocks: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Build KV-prefix-stable chat messages: fixed mode system prompt first, dynamic state last."""
    llm_messages: List[Dict[str, Any]] = [{"role": "system", "content": _MODE_PROMPTS[mode]}]

    if sess.get("compacted_summary"):
        llm_messages.append({
            "role": "user",
            "content": (
                f'<sikistirilmis_baglam_ozeti kapsanan_mesajlar="1..{sess.get("compacted_until_msg_id", 0)}">\n'
                f"{sess['compacted_summary']}\n"
                "</sikistirilmis_baglam_ozeti>\n"
                "(Tam geçmiş detayları için `search_chat_history` aracını çağır.)"
            ),
        })
        llm_messages.append({"role": "assistant", "content": "Önceki bağlam özeti alındı."})

    for m in prior_msgs:
        role = m.get("role")
        if role == "user":
            llm_messages.append({"role": "user", "content": m["content"]})
        elif role == "assistant":
            t_events = [ev for ev in (m.get("tool_events") or []) if ev.get("name")]
            if t_events:
                t_calls: List[Dict[str, Any]] = []
                t_msgs: List[Dict[str, Any]] = []
                for idx, ev in enumerate(t_events):
                    tc_id = str(ev.get("id") or f"call_hist_{m.get('id', 0)}_{idx}")
                    raw_args = ev.get("arguments")
                    if isinstance(raw_args, dict):
                        args_str = json.dumps(raw_args, ensure_ascii=False)
                    elif isinstance(raw_args, str) and raw_args.strip():
                        try:
                            json.loads(raw_args)
                            args_str = raw_args
                        except json.JSONDecodeError:
                            args_str = "{}"
                    else:
                        args_str = "{}"
                    t_calls.append({
                        "id": tc_id,
                        "type": "function",
                        "function": {"name": str(ev["name"]), "arguments": args_str},
                    })
                    digest = str(ev.get("output") or ev.get("preview") or "(çıktı kaydedildi)").strip()
                    if ev["name"] in SCREEN_STATE_TOOLS:
                        # Fixed text per message keeps the KV prefix stable turn to turn.
                        digest = STALE_SCREEN_LABEL + digest
                    t_msgs.append({
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "name": str(ev["name"]),
                        "content": digest[:REPLAYED_TOOL_OUTPUT_CHARS],
                    })
                llm_messages.append({"role": "assistant", "content": "", "tool_calls": t_calls})
                llm_messages.extend(t_msgs)
            clean_body = strip_fake_tool_echo(m.get("content") or "")
            if clean_body or not t_events:
                llm_messages.append({"role": "assistant", "content": clean_body})

    tail_blocks: List[str] = []
    if ws_dir:
        tail_blocks.append(f"<calisma_klasoru>{ws_dir}</calisma_klasoru>")
    if attachment_blocks:
        tail_blocks.extend(attachment_blocks)
    if sess.get("research_state_md") and mode == "research":
        tail_blocks.append(
            f'<kalici_arastirma_durumu dosya="research_state.md" asama="{sess.get("research_phase") or ""}">\n'
            f"{sess['research_state_md'][:6000]}\n</kalici_arastirma_durumu>"
        )
    if sess.get("canvas_md") and mode in ("study", "draft"):
        tail_blocks.append(
            f'<mevcut_canvas baslik="{sess.get("canvas_title") or "Taslak"}">\n'
            f"{sess['canvas_md'][:6000]}\n</mevcut_canvas>"
        )

    user_prompt_full = user_message + ("\n\n" + "\n\n".join(tail_blocks) if tail_blocks else "")
    llm_messages.append({"role": "user", "content": user_prompt_full})
    return llm_messages


#: Prefixed to screen readings replayed from earlier turns, so the model treats
#: them as history (fine for "explain question 1 from that screen") rather than
#: as the current screen (wrong for "what is the latest message in Notlar").
STALE_SCREEN_LABEL = (
    "[ESKİ EKRAN OKUMASI — önceki turdan, GÜNCEL DEĞİL. Ekranın şu anki hali sorulursa "
    "`pc_capture_screen` veya `pc_focus_or_launch` ile yeniden oku.]\n"
)


class SessionCreateReq(BaseModel):
    title: str = "Yeni Sohbet"
    mode: str = "assistant"
    workspace_dir: str = ""


class SessionPatchReq(BaseModel):
    title: Optional[str] = None
    mode: Optional[str] = None
    workspace_dir: Optional[str] = None
    canvas_title: Optional[str] = None
    canvas_md: Optional[str] = None
    auto_compact: Optional[bool] = None
    research_state_md: Optional[str] = None


class CompactReq(BaseModel):
    keep_recent: int = 4


class PermissionDecisionReq(BaseModel):
    decision: str  # "allow_once" | "allow_dir" | "deny"


class ChatStreamReq(BaseModel):
    session_id: str
    message: str
    attachment_ids: List[str] = []
    mode: str = "assistant"
    enable_web: bool = True
    web_mode: str = "auto"  # "auto" | "always" | "off"
    pc_mode: str = "auto"  # "auto" | "background" | "foreground" | "always" | "off"
    enable_thinking: bool = False
    workspace_dir: str = ""
    temperature: float = 0.6


def create_studio_app(
    *,
    storage_dir: Path = STUDIO_DIR,
    upstream_url: str = "http://127.0.0.1:8080/v1",
) -> FastAPI:
    app = FastAPI(title="ARGUS Studio WebUI", version="1.1.0")
    store = StudioStore(storage_dir)
    broker = PermissionBroker(store.notes_dir)
    stream_mgr = StreamManager(broker)
    pc_states: Dict[str, Dict[str, Any]] = {}
    app.state.store = store
    app.state.broker = broker
    app.state.stream_mgr = stream_mgr
    app.state.pc_states = pc_states
    app.state.upstream_url = upstream_url.rstrip("/")
    status_cache: Dict[str, Any] = {"ts": 0.0, "data": None}

    @app.get("/health")
    async def health() -> Dict[str, Any]:
        return {
            "status": "ok",
            "service": "argus-studio",
            "upstream": app.state.upstream_url,
            "busy": stream_mgr.is_active(),
        }

    @app.get("/api/status")
    async def api_status() -> Dict[str, Any]:
        now = time.monotonic()
        if status_cache["data"] is not None and (now - status_cache["ts"]) < 8.0:
            cached = dict(status_cache["data"])
            cached["busy"] = stream_mgr.is_active()
            return cached

        up_base = app.state.upstream_url[:-3] if app.state.upstream_url.endswith("/v1") else app.state.upstream_url
        backend_ok = False
        model_name = "Yüklü değil"
        ctx_len = None
        try:
            async with httpx.AsyncClient(timeout=1.5) as client:
                hr = await client.get(f"{up_base}/health")
                backend_ok = hr.status_code == 200
                pr = await client.get(f"{up_base}/props")
                if pr.status_code == 200:
                    pdata = pr.json()
                    mpath = pdata.get("model_path") or pdata.get("model_alias", "")
                    model_name = Path(mpath).stem if mpath else "llama-server"
                    ctx_len = pdata.get("default_generation_settings", {}).get("n_ctx")
        except httpx.HTTPError:
            pass

        default_ws = ""
        cfg_path = RUNTIME_DIR / "config.json"
        if cfg_path.is_file():
            try:
                default_ws = str(json.loads(cfg_path.read_text(encoding="utf-8")).get("workspace", ""))
            except (OSError, json.JSONDecodeError):
                pass

        result = {
            "backend_ok": backend_ok,
            "model_name": model_name,
            "context_length": ctx_len,
            "notes_dir": str(store.notes_dir),
            "default_workspace": default_ws,
            "busy": stream_mgr.is_active(),
        }
        status_cache["ts"] = now
        status_cache["data"] = result
        return result

    @app.get("/api/sessions")
    async def list_sessions() -> Dict[str, Any]:
        return {"sessions": store.list_sessions()}

    @app.post("/api/sessions")
    async def create_session(req: SessionCreateReq) -> Dict[str, Any]:
        return store.create_session(title=req.title, mode=req.mode, workspace_dir=req.workspace_dir)

    @app.get("/api/sessions/{session_id}")
    async def get_session(session_id: str) -> Dict[str, Any]:
        sess = store.get_session(session_id)
        if not sess:
            raise HTTPException(status_code=404, detail="Oturum bulunamadı")
        return sess

    @app.patch("/api/sessions/{session_id}")
    async def patch_session(session_id: str, req: SessionPatchReq) -> Dict[str, Any]:
        patched_title: Optional[str] = None
        patched_custom: Optional[bool] = None
        if req.title is not None:
            untitled = is_untitled_name(req.title)
            patched_title = "Yeni Sohbet" if untitled else req.title.strip()
            patched_custom = not untitled
        sess = store.update_session(
            session_id,
            title=patched_title,
            custom_title=patched_custom,
            mode=req.mode,
            workspace_dir=req.workspace_dir,
            canvas_title=req.canvas_title,
            canvas_md=req.canvas_md,
            auto_compact=req.auto_compact,
            research_state_md=req.research_state_md,
        )
        if not sess:
            raise HTTPException(status_code=404, detail="Oturum bulunamadı")
        return sess

    @app.post("/api/sessions/{session_id}/compact")
    async def manual_compact_session(session_id: str, req: CompactReq = CompactReq()) -> Dict[str, Any]:
        if not store.get_session(session_id):
            raise HTTPException(status_code=404, detail="Oturum bulunamadı")
        return await compact_session_context(
            store,
            session_id,
            upstream_url=app.state.upstream_url,
            keep_recent=max(0, req.keep_recent),
            force=True,
        )

    @app.delete("/api/sessions/{session_id}")
    async def delete_session(session_id: str) -> Dict[str, bool]:
        await stream_mgr.stop_session(session_id)
        pc_states.pop(session_id, None)
        store.delete_session(session_id)
        return {"ok": True}

    @app.post("/api/sessions/{session_id}/stop")
    async def stop_session_stream(session_id: str) -> Dict[str, Any]:
        stopped = await stream_mgr.stop_session(session_id)
        return {"ok": True, "stopped": stopped}

    @app.post("/api/chat/stop")
    async def stop_all_streams() -> Dict[str, Any]:
        stopped_count = await stream_mgr.stop_all()
        return {"ok": True, "stopped_count": stopped_count}

    @app.post("/api/upload")
    async def upload_document(
        session_id: str = Form(...),
        file: UploadFile = File(...),
    ) -> Dict[str, Any]:
        if not store.get_session(session_id):
            raise HTTPException(status_code=404, detail="Oturum bulunamadı")
        filename = file.filename or "belge.txt"
        suffix = Path(filename).suffix
        raw_bytes = await file.read()
        if len(raw_bytes) > 64 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Dosya boyutu 64 MB sınırını aşıyor.")

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(raw_bytes)
            tmp_path = Path(tmp.name)

        try:
            converted = await asyncio.to_thread(convert_file_to_markdown, tmp_path, display_name=filename)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Markdown dönüşüm hatası: {exc}") from exc
        finally:
            tmp_path.unlink(missing_ok=True)

        saved = store.save_document(
            session_id=session_id,
            filename=filename,
            fmt=converted.fmt,
            markdown=converted.markdown,
            sections=[s.to_dict() for s in converted.sections],
            page_count=converted.page_count,
        )
        saved["outline"] = converted.outline(20)
        saved["preview_md"] = converted.markdown[:2_000]
        return saved

    @app.get("/api/docs/{doc_id}")
    async def get_document(doc_id: str, download: int = 0):
        doc = store.get_document(doc_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Belge bulunamadı")
        md_path = Path(doc["md_path"])
        if not md_path.is_file():
            raise HTTPException(status_code=404, detail="Markdown dosyası bulunamadı")
        if download:
            stem = Path(doc["filename"]).stem
            return PlainTextResponse(
                md_path.read_text(encoding="utf-8", errors="replace"),
                media_type="text/markdown; charset=utf-8",
                headers={"Content-Disposition": f'attachment; filename="{stem}.md"'},
            )
        return {
            "id": doc["id"],
            "filename": doc["filename"],
            "fmt": doc["fmt"],
            "page_count": doc["page_count"],
            "est_tokens": doc["est_tokens"],
            "markdown": md_path.read_text(encoding="utf-8", errors="replace"),
            "sections": doc["sections"],
        }

    @app.post("/api/permissions/{req_id}")
    async def resolve_permission(req_id: str, body: PermissionDecisionReq) -> Dict[str, Any]:
        resolved = broker.resolve(req_id, body.decision)
        if not resolved:
            raise HTTPException(status_code=404, detail="Bekleyen izin isteği bulunamadı veya zaman aşımına uğradı.")
        return resolved

    @app.post("/api/chat/stream")
    async def chat_stream(req: ChatStreamReq) -> StreamingResponse:
        sess = store.get_session(req.session_id)
        if not sess:
            raise HTTPException(status_code=404, detail="Oturum bulunamadı")

        ws_dir = req.workspace_dir.strip() if req.workspace_dir is not None else sess.get("workspace_dir", "")
        mode = req.mode if req.mode in _MODE_PROMPTS else "assistant"
        store.update_session(req.session_id, mode=mode, workspace_dir=ws_dir)

        attached_meta: List[Dict[str, Any]] = []
        attachment_blocks: List[str] = []
        for doc_id in req.attachment_ids:
            doc = store.get_document(doc_id)
            if doc:
                attached_meta.append({
                    "id": doc["id"],
                    "filename": doc["filename"],
                    "fmt": doc["fmt"],
                    "page_count": doc["page_count"],
                    "est_tokens": doc["est_tokens"],
                })
                attachment_blocks.append(build_prompt_attachment_block(doc, req.message))

        first_user_trunc = ""
        if sess.get("messages") and sess["messages"][0].get("role") == "user":
            first_user_trunc = re.sub(r"\s+", " ", sess["messages"][0]["content"].strip())[:48]
        needs_ai_title = bool(req.message.strip()) and (not bool(sess.get("custom_title"))) and (
            is_untitled_name(sess["title"]) or (bool(first_user_trunc) and sess["title"] == first_user_trunc)
        )
        clean_title = sess["title"]

        auto_compacted_info: Optional[Dict[str, Any]] = None
        if should_auto_compact(sess):
            c_res = await compact_session_context(
                store,
                req.session_id,
                upstream_url=app.state.upstream_url,
            )
            if c_res.get("compacted") and c_res.get("session"):
                sess = c_res["session"]
                auto_compacted_info = {
                    "compacted_count": c_res["compacted_count"],
                    "compacted_until_msg_id": c_res["compacted_until_msg_id"],
                    "compacted_summary": c_res["compacted_summary"],
                }

        prior_msgs = get_uncompacted_messages(sess)[-16:]

        store.add_message(
            req.session_id,
            role="user",
            content=req.message,
            attachments=attached_meta,
        )

        llm_messages = build_chat_llm_messages(
            sess,
            mode=mode,
            user_message=req.message,
            prior_msgs=prior_msgs,
            ws_dir=ws_dir,
            attachment_blocks=attachment_blocks,
        )

        exec_pc_mode = "foreground" if req.pc_mode == "foreground" else "background"
        sess_pc_state = pc_states.setdefault(req.session_id, {})
        if not sess_pc_state.get("target_window"):
            for pm in reversed(sess.get("messages") or []):
                for pev in reversed(pm.get("tool_events") or []):
                    if pev.get("name") == "pc_focus_or_launch":
                        try:
                            pargs = json.loads(pev.get("arguments") or "{}")
                        except json.JSONDecodeError:
                            pargs = {}
                        app_hint = str(pargs.get("app_name") or pargs.get("app") or "").strip()
                        if app_hint:
                            recovered = find_matching_window(app_hint, app_hint)
                            if recovered:
                                sess_pc_state["target_window"] = {
                                    **recovered,
                                    "headless_monitor": "",
                                    "background": exec_pc_mode != "foreground",
                                }
                        break
                if sess_pc_state.get("target_window"):
                    break

        runner = ToolRunner(
            store=store,
            broker=broker,
            session_id=req.session_id,
            workspace_dir=ws_dir,
            emit_event=lambda _ev, _data: None,
            pc_mode=exec_pc_mode,
            pc_state=sess_pc_state,
        )
        has_docs = bool(attached_meta) or bool(sess.get("documents"))
        tool_schemas = build_tool_schemas(
            enable_web=req.enable_web,
            web_mode=req.web_mode,
            pc_mode=req.pc_mode,
            has_docs=has_docs,
            has_workspace=bool(ws_dir),
            mode=mode,
            prompt=req.message,
            prefix_stable=True,
        )

        return StreamingResponse(
            run_agent_stream(
                store=store,
                stream_mgr=stream_mgr,
                runner=runner,
                upstream_url=app.state.upstream_url,
                session_id=req.session_id,
                clean_title=clean_title,
                mode=mode,
                temperature=req.temperature,
                enable_thinking=req.enable_thinking,
                llm_messages=llm_messages,
                tool_schemas=tool_schemas,
                auto_compacted_info=auto_compacted_info,
                needs_ai_title=needs_ai_title,
                user_prompt_for_title=req.message,
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    register_anthropic_routes(app, upstream_url)

    app.mount("/screenshots", StaticFiles(directory=str(store.screenshots_dir)), name="screenshots")
    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/")
    async def index_page():
        return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-store"})

    return app


def main() -> None:
    import os

    parser = argparse.ArgumentParser(description="ARGUS Studio WebUI Server")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_STUDIO_PORT)
    parser.add_argument("--upstream", type=str, default="http://127.0.0.1:8080/v1")
    args = parser.parse_args()
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    (RUNTIME_DIR / "gateway.pid").write_text(str(os.getpid()), encoding="utf-8")
    app = create_studio_app(upstream_url=args.upstream)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
