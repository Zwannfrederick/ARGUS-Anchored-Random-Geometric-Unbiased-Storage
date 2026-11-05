"""OpenAI-format tool definitions and permission-gated local/web tool executors for ARGUS Studio."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Set

from argus_cache.webui.compaction import format_history_search_results
from argus_cache.webui.doc_markdown import convert_file_to_markdown, query_sections
from argus_cache.webui.hermes_pc import PC_HINT_RE, PC_TOOL_NAMES, build_hermes_pc_schemas, execute_hermes_pc_tool
from argus_cache.webui.permissions import PermissionBroker, build_unified_diff
from argus_cache.webui.research_tools import PYTHON_HINT_RE, build_research_schemas, execute_research_tool
from argus_cache.webui.store import StudioStore
from argus_cache.webui.study_tools import build_study_schemas, execute_study_tool
from argus_cache.webui.web_search import parallel_web_search, read_webpage, read_webpages_parallel, web_search

_BINARY_DOC_EXTS = {".pdf", ".docx", ".odt", ".pptx", ".xlsx", ".csv", ".tsv", ".doc", ".rtf", ".html", ".htm"}

_WEB_HINT_RE = re.compile(
    r"(https?://|www\.|\bweb\b|internet|araştır|arastir|\bara\b|güncel|guncel|haber|son dakika|"
    r"fiyat|dolar|euro|hava durumu|2025|2026|2027|sürüm|surum|release|wikipedia|kaynak bul|search|latest|news)",
    re.IGNORECASE,
)
_FILE_HINT_RE = re.compile(
    r"(dosya|klasör|klasor|dizin|kaydet|diske|düzenle|duzenle|masaüstü|masaustu|desktop|indirilenler|"
    r"belgeler|listele|~/|\./|\.\./|/home/|/tmp/|/etc/|"
    r"\.(?:md|pdf|docx?|odt|pptx|xlsx|csv|tsv|py|js|ts|json|txt|html?|sh|cpp|c|h|rs|go|ya?ml|sql)\b|"
    r"\bfile\b|\bfolder\b|\bdirectory\b)",
    re.IGNORECASE,
)
_CANVAS_HINT_RE = re.compile(
    r"(taslak|canvas|çalışma kağıdı|calisma kagidi|ders notu|özet çıkar|ozet cikar|not çıkar|not cikar|"
    r"plan oluştur|plan olustur|makale|prd|şablon|sablon|doküman hazırla|dokuman hazirla)",
    re.IGNORECASE,
)
_HISTORY_HINT_RE = re.compile(
    r"(geçmiş|gecmis|önceki|onceki|daha önce|daha once|hatırla|hatirla|ne konuşmuştuk|ne konusmustuk|"
    r"ilk başta|ilk basta|eski mesaj|yukarıda|yukarida|arşiv|arsiv)",
    re.IGNORECASE,
)
_STUDY_HINT_RE = re.compile(
    r"(quiz|test hazırla|test hazirla|sınav|sinav|soru hazırla|soru hazirla|çoktan seçmeli|coktan secmeli|"
    r"flashcard|bilgi kart|ezber kart|ödev|odev)",
    re.IGNORECASE,
)


def _fn(name: str, desc: str, props: Dict[str, Any], req: List[str]) -> Dict[str, Any]:
    schema: Dict[str, Any] = {"type": "object", "properties": props}
    if req:
        schema["required"] = req
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": schema}}


def _history_schema() -> Dict[str, Any]:
    return _fn(
        "search_chat_history",
        "Sıkıştırılmış olsa bile silinmeyen tüm geçmiş mesajlarda ve yapılan işlemlerde sorgu veya #ID aralığı arar.",
        {
            "query": {"type": "string"},
            "start_id": {"type": "integer"},
            "end_id": {"type": "integer"},
            "limit": {"type": "integer"},
        },
        [],
    )


def _doc_schema() -> Dict[str, Any]:
    return _fn(
        "read_uploaded_doc",
        "Ekli belgenin Markdown bölümlerinde arar veya sayfa/bölüm okur.",
        {
            "doc_id": {"type": "string"},
            "query": {"type": "string"},
            "start_section": {"type": "integer"},
            "end_section": {"type": "integer"},
        },
        ["doc_id"],
    )


def _web_schemas() -> List[Dict[str, Any]]:
    return [
        _fn(
            "web_search",
            "İnternette tek (`query`) veya paralel çoklu (`queries`) sorguyla güncel kaynak arar.",
            {
                "query": {"type": "string"},
                "queries": {"type": "array", "items": {"type": "string"}},
                "max_results": {"type": "integer"},
            },
            [],
        ),
        _fn(
            "read_webpage",
            "Tek (`url`) veya paralel çoklu (`urls`) web sayfasını temiz Markdown olarak okur.",
            {
                "url": {"type": "string"},
                "urls": {"type": "array", "items": {"type": "string"}},
                "focus_query": {"type": "string"},
            },
            [],
        ),
    ]


def _local_file_schemas() -> List[Dict[str, Any]]:
    return [
        _fn("list_directory", "Yerel klasör içeriğini listeler.", {"path": {"type": "string"}}, []),
        _fn(
            "search_local_files",
            "Yerel dosyalarda metin veya dosya adı arar.",
            {"query": {"type": "string"}, "path": {"type": "string"}},
            ["query"],
        ),
        _fn(
            "read_local_file",
            "Yerel dosyayı okur (PDF/Word otomatik Markdown olur).",
            {
                "path": {"type": "string"},
                "start_line": {"type": "integer"},
                "end_line": {"type": "integer"},
                "query": {"type": "string"},
            },
            ["path"],
        ),
        _fn(
            "write_local_file",
            "Yerel dosya yazar (diff önizlemesiyle kullanıcıdan izin alır).",
            {"path": {"type": "string"}, "content": {"type": "string"}},
            ["path", "content"],
        ),
        _fn(
            "edit_local_file",
            "Yerel dosyada metin değiştirir (diff ile izin alır).",
            {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}},
            ["path", "old_text", "new_text"],
        ),
    ]


def _canvas_schema() -> Dict[str, Any]:
    return _fn(
        "update_canvas",
        "YALNIZCA kullanıcı açıkça taslak, ders notu, çalışma kağıdı veya Canvas güncellemesi istediğinde sağdaki Markdown panelini günceller; normal sohbette çağırma.",
        {"title": {"type": "string"}, "markdown": {"type": "string"}},
        ["title", "markdown"],
    )


def build_tool_schemas(
    *,
    enable_web: bool = True,
    web_mode: str = "auto",
    pc_mode: str = "auto",
    has_docs: bool = False,
    has_workspace: bool = False,
    has_canvas: bool = False,
    has_history: bool = False,
    has_pc_history: bool = False,
    has_web_history: bool = False,
    mode: str = "assistant",
    prompt: str = "",
    prefix_stable: bool = False,
) -> List[Dict[str, Any]]:
    """Build deterministic OpenAI tool schemas.

    When ``prefix_stable=True`` (used by live Studio sessions), tool selection and order
    depend strictly on session mode and explicit toggles — never on per-turn prompt regexes
    or message counts — so Token 0 of the Jinja KV prefix cache stays 100% invariant across turns.
    """
    effective_web = web_mode if web_mode in ("auto", "always", "off") else ("auto" if enable_web else "off")

    if prefix_stable:
        tools: List[Dict[str, Any]] = []
        if effective_web != "off":
            tools.extend(_web_schemas())
        if mode == "research":
            tools.extend(build_research_schemas())
        elif mode in ("assistant", "study"):
            tools.append(build_research_schemas()[0])
        if mode == "study":
            tools.extend(build_study_schemas())
        if pc_mode != "off" and (mode == "assistant" or pc_mode in ("always", "background", "foreground")):
            tools.extend(build_hermes_pc_schemas())
        tools.append(_history_schema())
        tools.append(_doc_schema())
        if mode in ("assistant", "draft") or has_workspace:
            tools.extend(_local_file_schemas())
        tools.append(_canvas_schema())
        return tools

    tools = []
    want_web = (
        (mode == "research" and effective_web != "off")
        or effective_web == "always"
        or (effective_web == "auto" and (has_web_history or bool(_WEB_HINT_RE.search(prompt))))
    )
    if want_web:
        tools.extend(_web_schemas())

    if mode == "research":
        tools.extend(build_research_schemas())
    elif bool(PYTHON_HINT_RE.search(prompt)):
        tools.append(build_research_schemas()[0])

    if mode == "study" or bool(_STUDY_HINT_RE.search(prompt)):
        tools.extend(build_study_schemas())

    want_pc = pc_mode in ("always", "background", "foreground") or (
        mode == "assistant" and pc_mode == "auto" and (has_pc_history or bool(PC_HINT_RE.search(prompt)))
    )
    if want_pc:
        tools.extend(build_hermes_pc_schemas())

    if has_history or bool(_HISTORY_HINT_RE.search(prompt)):
        tools.append(_history_schema())

    if has_docs:
        tools.append(_doc_schema())

    if has_workspace or bool(_FILE_HINT_RE.search(prompt)):
        tools.extend(_local_file_schemas())

    if mode in ("study", "draft", "research") or has_canvas or bool(_CANVAS_HINT_RE.search(prompt)):
        tools.append(_canvas_schema())

    return tools


@dataclass
class ToolRunner:
    store: StudioStore
    broker: PermissionBroker
    session_id: str
    workspace_dir: str
    emit_event: Callable[[str, Dict[str, Any]], None]
    pc_mode: str = "background"
    citations: List[Dict[str, str]] = field(default_factory=list)
    verified_urls: Set[str] = field(default_factory=set)
    pc_state: Dict[str, Any] = field(default_factory=dict)
    python_verified: bool = False

    def __post_init__(self) -> None:
        self.verified_urls.update(self.store.get_verified_urls(self.session_id))

    def _resolve_path(self, raw_path: str) -> Path:
        p = Path(raw_path).expanduser()
        if not p.is_absolute():
            base = Path(self.workspace_dir).expanduser() if self.workspace_dir else self.store.notes_dir
            p = base / p
        return p.resolve(strict=False)

    async def execute(self, name: str, raw_args: str) -> str:
        try:
            args = json.loads(raw_args) if raw_args and raw_args.strip() else {}
        except json.JSONDecodeError as exc:
            return f"Araç argümanları geçerli JSON değil: {exc}"

        try:
            if name == "web_search":
                raw_qs = args.get("queries")
                q_list: List[str] = []
                if isinstance(raw_qs, list):
                    q_list.extend(str(x).strip() for x in raw_qs if str(x).strip())
                single_q = str(args.get("query", "")).strip()
                if single_q and single_q not in q_list:
                    q_list.insert(0, single_q)
                if not q_list:
                    return "Arama sorgusu (`query` veya `queries`) belirtilmedi."

                seen = {c["url"] for c in self.citations}
                max_r = int(args.get("max_results", 4 if len(q_list) > 1 else 5))

                if len(q_list) == 1:
                    items = await web_search(q_list[0], max_results=max_r)
                    if not items:
                        return f"'{q_list[0]}' için sonuç bulunamadı."
                    for it in items:
                        self.verified_urls.add(it["url"])
                        if it["url"] not in seen:
                            self.citations.append(it)
                            seen.add(it["url"])
                    self.emit_event("citations", {"items": self.citations})
                    lines = [
                        f"{i}. **{it['title']}** ({it['domain']})\n   URL: {it['url']}\n   Özet: {it['snippet']}"
                        for i, it in enumerate(items, 1)
                    ]
                    body = "\n\n".join(lines)
                else:
                    grouped = await parallel_web_search(q_list, max_results_per_query=max_r)
                    sections: List[str] = []
                    for q_str, items in grouped:
                        for it in items:
                            self.verified_urls.add(it["url"])
                            if it["url"] not in seen:
                                self.citations.append(it)
                                seen.add(it["url"])
                        if not items:
                            sections.append(f"### Sorgu: `{q_str}`\nSonuç bulunamadı.")
                            continue
                        rows = [
                            f"- **{it['title']}** ({it['domain']}) — {it['url']}\n  {it['snippet']}"
                            for it in items
                        ]
                        sections.append(f"### Sorgu: `{q_str}`\n" + "\n".join(rows))
                    self.emit_event("citations", {"items": self.citations})
                    body = "\n\n".join(sections)

                return (
                    "Arama sonuçları (sayfa içi tam veri gerekiyorsa en uygun URL'ler için `read_webpage` çağır; "
                    "yeterliyse bu bulgularla kullanıcının sorusunu doğrudan yanıtla):\n\n"
                    + body
                )

            if name == "read_webpage":
                raw_urls = args.get("urls")
                u_list: List[str] = []
                if isinstance(raw_urls, list):
                    u_list.extend(str(u).strip() for u in raw_urls if str(u).strip())
                single_u = str(args.get("url", "")).strip()
                if single_u and single_u not in u_list:
                    u_list.insert(0, single_u)
                focus = str(args.get("focus_query", "")).strip()
                if len(u_list) > 1:
                    out = await read_webpages_parallel(u_list, focus_query=focus)
                    for u in u_list:
                        if "okunamadı" not in out:
                            self.verified_urls.add(u)
                    return out
                if len(u_list) == 1:
                    out = await read_webpage(u_list[0], focus_query=focus)
                    if not out.startswith("Sayfa okunamadı"):
                        self.verified_urls.add(u_list[0])
                    return out
                return "Okunacak `url` veya `urls` belirtilmedi."

            if name == "search_chat_history":
                sess = self.store.get_session(self.session_id)
                if not sess:
                    return "Oturum bulunamadı."
                s_id = int(args["start_id"]) if args.get("start_id") is not None else None
                e_id = int(args["end_id"]) if args.get("end_id") is not None else None
                lim = int(args.get("limit", 8))
                matches = self.store.search_messages(
                    self.session_id,
                    query=str(args.get("query", "")),
                    start_id=s_id,
                    end_id=e_id,
                    limit=lim,
                )
                return format_history_search_results(
                    matches,
                    total_count=len(sess.get("messages", [])),
                    compacted_until_id=int(sess.get("compacted_until_msg_id") or 0),
                )

            if name == "read_uploaded_doc":
                doc_id = str(args.get("doc_id", "")).strip()
                doc = self.store.get_document(doc_id)
                if not doc:
                    return f"Belge bulunamadı: {doc_id}"
                return query_sections(
                    doc["sections"],
                    query=str(args.get("query", "")),
                    start=int(args.get("start_section", 1)),
                    end=int(args.get("end_section", 3)),
                )

            if name == "list_directory":
                target = self._resolve_path(str(args.get("path", ".")))
                allowed = await self.broker.ensure_permission(
                    session_id=self.session_id,
                    operation="read",
                    target=target,
                    workspace_dir=self.workspace_dir or None,
                    summary=f"Klasör listeleme: {target}",
                    diff_preview="",
                    emit_event=self.emit_event,
                )
                if not allowed:
                    return f"Kullanıcı '{target}' klasörüne erişim izni vermedi."
                if not target.is_dir():
                    return f"Klasör bulunamadı: {target}"
                entries = sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))[:120]
                lines = [
                    f"{'[DİZİN]' if e.is_dir() else f'[{e.stat().st_size} B]':<12} {e.name}"
                    for e in entries
                    if not e.name.startswith(".git")
                ]
                return f"Klasör: {target}\n" + ("\n".join(lines) if lines else "(boş klasör)")

            if name == "search_local_files":
                q = str(args.get("query", "")).strip()
                target = self._resolve_path(str(args.get("path", ".")))
                allowed = await self.broker.ensure_permission(
                    session_id=self.session_id,
                    operation="read",
                    target=target,
                    workspace_dir=self.workspace_dir or None,
                    summary=f"Klasörde arama ('{q}'): {target}",
                    diff_preview="",
                    emit_event=self.emit_event,
                )
                if not allowed:
                    return f"Kullanıcı '{target}' klasöründe arama izni vermedi."
                rg = shutil.which("rg")
                if rg and target.exists():
                    res = subprocess.run(
                        [rg, "-n", "-i", "--max-count", "4", "--max-filesize", "1M", q, str(target)],
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                    out = "\n".join(res.stdout.splitlines()[:50]).strip()
                    return out or f"'{q}' için eşleşme bulunamadı."
                hits: List[str] = []
                for root_dir, dirs, files in os.walk(target):
                    dirs[:] = [d for d in dirs if not d.startswith(".") and d != "node_modules" and d != ".venv"]
                    for fn in files:
                        fp = Path(root_dir) / fn
                        if q.lower() in fn.lower():
                            hits.append(f"[DOSYA] {fp}")
                        if len(hits) >= 40:
                            break
                return "\n".join(hits) if hits else f"'{q}' için eşleşme bulunamadı."

            if name == "read_local_file":
                target = self._resolve_path(str(args.get("path", "")))
                allowed = await self.broker.ensure_permission(
                    session_id=self.session_id,
                    operation="read",
                    target=target,
                    workspace_dir=self.workspace_dir or None,
                    summary=f"Dosya okuma: {target}",
                    diff_preview="",
                    emit_event=self.emit_event,
                )
                if not allowed:
                    return f"Kullanıcı '{target}' dosyasını okuma izni vermedi."
                if not target.is_file():
                    return f"Dosya bulunamadı: {target}"
                start_l = max(1, int(args.get("start_line", 1)))
                end_l = max(start_l, int(args.get("end_line", 160)))
                query = str(args.get("query", "")).strip()
                if target.suffix.lower() in _BINARY_DOC_EXTS:
                    conv = convert_file_to_markdown(target)
                    if len(conv.markdown) <= 9_500 and not query:
                        return conv.markdown
                    sec_dicts = [s.to_dict() for s in conv.sections]
                    outline = "\n".join(conv.outline(25))
                    picked = query_sections(sec_dicts, query=query, start=start_l, end=min(end_l, start_l + 3))
                    return f"# {conv.title} (Markdown Dönüşümü — {len(conv.sections)} bölüm)\nİÇİNDEKİLER:\n{outline}\n\n{picked}"

                lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
                end_clamped = min(len(lines), min(end_l, start_l + 220))
                numbered = [f"{idx:>4}: {lines[idx - 1]}" for idx in range(start_l, end_clamped + 1)]
                return (
                    f"Dosya: {target} (satır {start_l}-{end_clamped} / toplam {len(lines)})\n"
                    + "\n".join(numbered)
                )

            if name == "write_local_file":
                target = self._resolve_path(str(args.get("path", "")))
                content = str(args.get("content", ""))
                old_text = target.read_text(encoding="utf-8", errors="replace") if target.is_file() else None
                diff = build_unified_diff(target, old_text, content)
                action_label = "Dosya güncelleme" if old_text is not None else "Yeni dosya oluşturma"
                allowed = await self.broker.ensure_permission(
                    session_id=self.session_id,
                    operation="write",
                    target=target,
                    workspace_dir=self.workspace_dir or None,
                    summary=f"{action_label}: {target}",
                    diff_preview=diff,
                    emit_event=self.emit_event,
                )
                if not allowed:
                    return f"Kullanıcı '{target}' dosyasına yazma işlemini reddetti."
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")
                return f"Başarıyla kaydedildi: {target} ({len(content)} karakter)."

            if name == "edit_local_file":
                target = self._resolve_path(str(args.get("path", "")))
                if not target.is_file():
                    return f"Düzenlenecek dosya bulunamadı: {target}"
                old_snippet = str(args.get("old_text", ""))
                new_snippet = str(args.get("new_text", ""))
                existing = target.read_text(encoding="utf-8", errors="replace")
                if old_snippet not in existing:
                    return f"old_text '{target.name}' içinde birebir bulunamadı. Önce read_local_file ile güncel satırları kontrol et."
                updated = existing.replace(old_snippet, new_snippet, 1)
                diff = build_unified_diff(target, existing, updated)
                allowed = await self.broker.ensure_permission(
                    session_id=self.session_id,
                    operation="edit",
                    target=target,
                    workspace_dir=self.workspace_dir or None,
                    summary=f"Dosya düzenleme: {target}",
                    diff_preview=diff,
                    emit_event=self.emit_event,
                )
                if not allowed:
                    return f"Kullanıcı '{target}' düzenlemesini reddetti."
                target.write_text(updated, encoding="utf-8")
                return f"Başarıyla düzenlendi: {target}"

            if name == "update_canvas":
                title = str(args.get("title", "Taslak")).strip() or "Taslak"
                md = str(args.get("markdown", ""))
                self.store.update_session(self.session_id, canvas_title=title, canvas_md=md)
                self.emit_event("canvas_update", {"title": title, "markdown": md})
                return f"Canvas güncellendi: '{title}' ({len(md)} karakter Markdown)."

            if name in ("verify_with_python", "update_research_state", "read_research_state", "audit_citations"):
                res = await execute_research_tool(
                    name,
                    args,
                    store=self.store,
                    session_id=self.session_id,
                    verified_urls=self.verified_urls,
                    emit_event=self.emit_event,
                )
                if name == "verify_with_python" and res.startswith("✓"):
                    self.python_verified = True
                return res

            if name in ("publish_quiz", "publish_flashcards"):
                return execute_study_tool(
                    name,
                    args,
                    store=self.store,
                    session_id=self.session_id,
                    emit_event=self.emit_event,
                )

            if name in PC_TOOL_NAMES:
                return await execute_hermes_pc_tool(
                    name,
                    args,
                    session_id=self.session_id,
                    workspace_dir=self.workspace_dir,
                    screenshots_dir=self.store.screenshots_dir,
                    broker=self.broker,
                    pc_state=self.pc_state,
                    emit_event=self.emit_event,
                    pc_mode=self.pc_mode,
                )

            return f"Bilinmeyen araç: {name}"
        except Exception as exc:
            return f"Araç hatası ({name}): {exc}"

