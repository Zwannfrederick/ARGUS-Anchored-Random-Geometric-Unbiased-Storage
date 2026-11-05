"""SQLite storage for ARGUS Studio sessions, messages, converted Markdown docs, and canvas drafts."""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_SESSION_TITLES = frozenset({
    "",
    "yeni sohbet",
    "yeni çalışma",
    "yeni calisma",
    "yeni çalışma / sohbet",
    "yeni calisma / sohbet",
    "adsız",
    "adsız sohbet",
    "untitled",
    "new chat",
})


def is_untitled_name(title: Optional[str]) -> bool:
    return (title or "").strip().lower() in DEFAULT_SESSION_TITLES


class StudioStore:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir.expanduser().resolve(strict=False)
        self.docs_dir = self.base_dir / "docs"
        self.notes_dir = self.base_dir / "notes"
        self.screenshots_dir = self.base_dir / "screenshots"
        self.research_dir = self.base_dir / "research"
        self.db_path = self.base_dir / "studio.db"
        for d in (self.docs_dir, self.notes_dir, self.screenshots_dir, self.research_dir):
            d.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=15.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    mode TEXT NOT NULL DEFAULT 'assistant',
                    workspace_dir TEXT NOT NULL DEFAULT '',
                    canvas_title TEXT NOT NULL DEFAULT '',
                    canvas_md TEXT NOT NULL DEFAULT '',
                    compacted_summary TEXT NOT NULL DEFAULT '',
                    compacted_until_msg_id INTEGER NOT NULL DEFAULT 0,
                    auto_compact INTEGER NOT NULL DEFAULT 1,
                    custom_title INTEGER NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    reasoning TEXT NOT NULL DEFAULT '',
                    citations_json TEXT NOT NULL DEFAULT '[]',
                    tool_events_json TEXT NOT NULL DEFAULT '[]',
                    timeline_json TEXT NOT NULL DEFAULT '[]',
                    attachments_json TEXT NOT NULL DEFAULT '[]',
                    timings_json TEXT NOT NULL DEFAULT '{}',
                    created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    fmt TEXT NOT NULL,
                    page_count INTEGER NOT NULL,
                    char_count INTEGER NOT NULL,
                    est_tokens INTEGER NOT NULL,
                    md_path TEXT NOT NULL,
                    sections_json TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);
                CREATE INDEX IF NOT EXISTS idx_docs_session ON documents(session_id);
                """
            )
            existing_cols = {
                r["name"] for r in conn.execute("PRAGMA table_info(sessions)").fetchall()
            }
            migrations = [
                ("compacted_summary", "TEXT NOT NULL DEFAULT ''"),
                ("compacted_until_msg_id", "INTEGER NOT NULL DEFAULT 0"),
                ("auto_compact", "INTEGER NOT NULL DEFAULT 1"),
                ("research_state_md", "TEXT NOT NULL DEFAULT ''"),
                ("research_phase", "TEXT NOT NULL DEFAULT ''"),
                ("study_quiz_json", "TEXT NOT NULL DEFAULT ''"),
                ("study_flashcards_json", "TEXT NOT NULL DEFAULT ''"),
                ("custom_title", "INTEGER NOT NULL DEFAULT 0"),
            ]
            for col_name, col_def in migrations:
                if col_name not in existing_cols:
                    conn.execute(f"ALTER TABLE sessions ADD COLUMN {col_name} {col_def}")
            msg_cols = {
                r["name"] for r in conn.execute("PRAGMA table_info(messages)").fetchall()
            }
            if "timeline_json" not in msg_cols:
                conn.execute("ALTER TABLE messages ADD COLUMN timeline_json TEXT NOT NULL DEFAULT '[]'")

    def create_session(
        self,
        *,
        title: str = "Yeni Sohbet",
        mode: str = "assistant",
        workspace_dir: str = "",
    ) -> Dict[str, Any]:
        sid = uuid.uuid4().hex[:12]
        now = time.time()
        untitled = is_untitled_name(title)
        clean_title = "Yeni Sohbet" if untitled else title.strip()
        custom_title = 0 if untitled else 1
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO sessions(id, title, custom_title, mode, workspace_dir, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (sid, clean_title, custom_title, mode, workspace_dir, now, now),
            )
        return self.get_session(sid) or {}

    def list_sessions(self, limit: int = 80) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, title, custom_title, mode, workspace_dir, canvas_title, compacted_until_msg_id, "
                "auto_compact, research_phase, updated_at, created_at "
                "FROM sessions ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def research_state_file(self, session_id: str) -> Path:
        return self.research_dir / session_id / "research_state.md"

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
            if row is None:
                return None
            msgs = conn.execute(
                "SELECT * FROM messages WHERE session_id = ? ORDER BY id ASC",
                (session_id,),
            ).fetchall()
            docs = conn.execute(
                "SELECT id, filename, fmt, page_count, char_count, est_tokens, created_at "
                "FROM documents WHERE session_id = ? ORDER BY created_at DESC",
                (session_id,),
            ).fetchall()
        session = dict(row)
        r_file = self.research_state_file(session_id)
        if r_file.is_file():
            session["research_state_md"] = r_file.read_text(encoding="utf-8", errors="replace")
        session["research_state_path"] = str(r_file)
        session["study_quiz"] = json.loads(session.pop("study_quiz_json", "") or "null")
        session["study_flashcards"] = json.loads(session.pop("study_flashcards_json", "") or "null")
        session["messages"] = [
            {
                "id": m["id"],
                "role": m["role"],
                "content": m["content"],
                "reasoning": m["reasoning"],
                "citations": json.loads(m["citations_json"] or "[]"),
                "tool_events": json.loads(m["tool_events_json"] or "[]"),
                "timeline": json.loads(m["timeline_json"] or "[]"),
                "attachments": json.loads(m["attachments_json"] or "[]"),
                "timings": json.loads(m["timings_json"] or "{}"),
                "created_at": m["created_at"],
            }
            for m in msgs
        ]
        session["documents"] = [dict(d) for d in docs]
        return session

    def get_verified_urls(self, session_id: str) -> set[str]:
        """Return all URLs genuinely retrieved via tools across this session's history."""
        sess = self.get_session(session_id)
        if not sess:
            return set()
        urls: set[str] = set()
        for m in sess.get("messages", []):
            for c in m.get("citations") or []:
                u = str(c.get("url") or "").strip()
                if u:
                    urls.add(u)
        return urls

    def update_session(
        self,
        session_id: str,
        *,
        title: Optional[str] = None,
        custom_title: Optional[bool] = None,
        mode: Optional[str] = None,
        workspace_dir: Optional[str] = None,
        canvas_title: Optional[str] = None,
        canvas_md: Optional[str] = None,
        compacted_summary: Optional[str] = None,
        compacted_until_msg_id: Optional[int] = None,
        auto_compact: Optional[bool] = None,
        research_state_md: Optional[str] = None,
        research_phase: Optional[str] = None,
        study_quiz: Optional[Dict[str, Any]] = None,
        study_flashcards: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        if research_state_md is not None:
            r_file = self.research_state_file(session_id)
            r_file.parent.mkdir(parents=True, exist_ok=True)
            r_file.write_text(research_state_md, encoding="utf-8")
        fields: List[str] = ["updated_at = ?"]
        vals: List[Any] = [time.time()]
        for col, val in (
            ("title", title),
            ("custom_title", int(custom_title) if custom_title is not None else None),
            ("mode", mode),
            ("workspace_dir", workspace_dir),
            ("canvas_title", canvas_title),
            ("canvas_md", canvas_md),
            ("compacted_summary", compacted_summary),
            ("compacted_until_msg_id", compacted_until_msg_id),
            ("auto_compact", int(auto_compact) if auto_compact is not None else None),
            ("research_state_md", research_state_md),
            ("research_phase", research_phase),
            ("study_quiz_json", json.dumps(study_quiz, ensure_ascii=False) if study_quiz is not None else None),
            (
                "study_flashcards_json",
                json.dumps(study_flashcards, ensure_ascii=False) if study_flashcards is not None else None,
            ),
        ):
            if val is not None:
                fields.append(f"{col} = ?")
                vals.append(val)
        vals.append(session_id)
        with self._connect() as conn:
            conn.execute(f"UPDATE sessions SET {', '.join(fields)} WHERE id = ?", vals)
        return self.get_session(session_id)

    def delete_session(self, session_id: str) -> None:
        with self._connect() as conn:
            docs = conn.execute("SELECT md_path FROM documents WHERE session_id = ?", (session_id,)).fetchall()
            conn.execute("DELETE FROM documents WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        for d in docs:
            Path(d["md_path"]).unlink(missing_ok=True)
        r_file = self.research_state_file(session_id)
        if r_file.is_file():
            r_file.unlink(missing_ok=True)
            try:
                r_file.parent.rmdir()
            except OSError:
                pass

    def add_message(
        self,
        session_id: str,
        *,
        role: str,
        content: str,
        reasoning: str = "",
        citations: Optional[List[Dict[str, Any]]] = None,
        tool_events: Optional[List[Dict[str, Any]]] = None,
        timeline: Optional[List[Dict[str, Any]]] = None,
        attachments: Optional[List[Dict[str, Any]]] = None,
        timings: Optional[Dict[str, Any]] = None,
    ) -> None:
        now = time.time()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO messages(
                    session_id, role, content, reasoning,
                    citations_json, tool_events_json, timeline_json, attachments_json, timings_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    role,
                    content,
                    reasoning,
                    json.dumps(citations or [], ensure_ascii=False),
                    json.dumps(tool_events or [], ensure_ascii=False),
                    json.dumps(timeline or [], ensure_ascii=False),
                    json.dumps(attachments or [], ensure_ascii=False),
                    json.dumps(timings or {}, ensure_ascii=False),
                    now,
                ),
            )
            conn.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (now, session_id))

    def save_document(
        self,
        *,
        session_id: str,
        filename: str,
        fmt: str,
        markdown: str,
        sections: List[Dict[str, Any]],
        page_count: int,
    ) -> Dict[str, Any]:
        doc_id = uuid.uuid4().hex[:10]
        md_path = self.docs_dir / f"{doc_id}.md"
        md_path.write_text(markdown, encoding="utf-8")
        est_tokens = max(1, (len(markdown) + 3) // 4)
        now = time.time()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO documents(
                    id, session_id, filename, fmt, page_count, char_count,
                    est_tokens, md_path, sections_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    doc_id,
                    session_id,
                    filename,
                    fmt,
                    page_count,
                    len(markdown),
                    est_tokens,
                    str(md_path),
                    json.dumps(sections, ensure_ascii=False),
                    now,
                ),
            )
        return {
            "id": doc_id,
            "session_id": session_id,
            "filename": filename,
            "fmt": fmt,
            "page_count": page_count,
            "char_count": len(markdown),
            "est_tokens": est_tokens,
            "md_path": str(md_path),
        }

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
        if row is None:
            return None
        doc = dict(row)
        doc["sections"] = json.loads(doc.pop("sections_json") or "[]")
        return doc

    def search_messages(
        self,
        session_id: str,
        *,
        query: str = "",
        start_id: Optional[int] = None,
        end_id: Optional[int] = None,
        limit: int = 8,
    ) -> List[Dict[str, Any]]:
        """Search or retrieve preserved full message history (never deleted during compaction)."""
        sess = self.get_session(session_id)
        if not sess:
            return []
        msgs = sess.get("messages", [])
        cap = max(1, min(20, int(limit)))

        if start_id is not None or end_id is not None:
            lo = int(start_id) if start_id is not None else 0
            hi = int(end_id) if end_id is not None else 10**12
            return [m for m in msgs if lo <= int(m["id"]) <= hi][:cap]

        def _norm(s: str) -> str:
            return s.replace("İ", "i").replace("I", "ı").lower()

        q_clean = _norm(query.strip())
        if not q_clean:
            return msgs[-cap:]

        terms = [t for t in q_clean.split() if len(t) >= 2] or [q_clean]
        scored: List[tuple[int, Dict[str, Any]]] = []
        for m in msgs:
            tool_blob = " ".join(
                f"{t.get('name', '')} {t.get('arguments', '')} {t.get('preview', '')}"
                for t in (m.get("tool_events") or [])
            )
            att_blob = " ".join(a.get("filename", "") for a in (m.get("attachments") or []))
            hay = _norm(f"{m['content']} {tool_blob} {att_blob}")
            score = (6 if q_clean in hay else 0) + sum(hay.count(t) for t in terms)
            if score > 0:
                scored.append((score, m))

        scored.sort(key=lambda item: (item[0], item[1]["id"]), reverse=True)
        picked = sorted((m for _, m in scored[:cap]), key=lambda m: m["id"])
        return picked

