"""Non-destructive context compaction and full-history recall helpers for ARGUS Studio.

Compaction NEVER deletes rows from SQLite `messages`. Instead it maintains a rolling
structured summary (`Sohbet Özeti & Kararlar` + `Yapılan İşlemler & Çıktılar`) and advances
`compacted_until_msg_id`, while `search_chat_history` queries the full preserved archive.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

import httpx

from argus_cache.webui.store import StudioStore

AUTO_COMPACT_MSG_THRESHOLD = 16
AUTO_COMPACT_CHAR_THRESHOLD = 22_000
DEFAULT_KEEP_RECENT = 6
#: Characters of each tool output that history replays to the model per turn.
REPLAYED_TOOL_OUTPUT_CHARS = 2200

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def get_uncompacted_messages(sess: Dict[str, Any]) -> List[Dict[str, Any]]:
    until_id = int(sess.get("compacted_until_msg_id") or 0)
    return [m for m in sess.get("messages", []) if int(m["id"]) > until_id]


def replayed_chars(m: Dict[str, Any]) -> int:
    """What a stored message costs in context: its text plus the tool output replayed with it."""
    tools = sum(
        min(REPLAYED_TOOL_OUTPUT_CHARS, len(str(t.get("output") or t.get("preview") or "")))
        for t in (m.get("tool_events") or [])
    )
    return len(m.get("content") or "") + tools


def estimate_active_tokens(sess: Dict[str, Any]) -> int:
    summary_chars = len(sess.get("compacted_summary") or "")
    msg_chars = sum(replayed_chars(m) for m in get_uncompacted_messages(sess))
    return max(1, (summary_chars + msg_chars + 3) // 4)


def should_auto_compact(sess: Dict[str, Any]) -> bool:
    if not bool(sess.get("auto_compact", 1)):
        return False
    uncompacted = get_uncompacted_messages(sess)
    if len(uncompacted) <= DEFAULT_KEEP_RECENT:
        return False
    total_chars = sum(replayed_chars(m) for m in uncompacted)
    return len(uncompacted) >= AUTO_COMPACT_MSG_THRESHOLD or total_chars >= AUTO_COMPACT_CHAR_THRESHOLD


def _deterministic_summary(prior_summary: str, messages: List[Dict[str, Any]]) -> str:
    topic_lines: List[str] = []
    action_lines: List[str] = []
    for m in messages:
        mid = m["id"]
        snippet = re.sub(r"\s+", " ", (m.get("content") or "").strip())[:180]
        if m["role"] == "user" and snippet:
            topic_lines.append(f"- **[#{mid} Kullanıcı]:** {snippet}")
        elif m["role"] == "assistant" and snippet:
            topic_lines.append(f"- **[#{mid} Yanıt]:** {snippet}")
        for att in m.get("attachments") or []:
            action_lines.append(f"- **[#{mid} Belge Eklendi]:** `{att.get('filename')}` (.md)")
        for te in m.get("tool_events") or []:
            prev = re.sub(r"\s+", " ", str(te.get("preview") or te.get("arguments") or ""))[:140]
            action_lines.append(f"- **[#{mid} Araç `{te.get('name')}`]:** {prev}")

    parts: List[str] = []
    if prior_summary.strip():
        parts.append(prior_summary.strip())
    parts.append("### Sohbet Özeti & Kararlar\n" + ("\n".join(topic_lines[-14:]) or "- Genel sohbet akışı."))
    parts.append("### Yapılan İşlemler & Çıktılar\n" + ("\n".join(action_lines[-14:]) or "- Henüz araç/dosya işlemi yapılmadı."))
    return "\n\n".join(parts)[-6_000:]


def _format_slice_for_summarizer(messages: List[Dict[str, Any]]) -> str:
    blocks: List[str] = []
    for m in messages:
        role_tr = "KULLANICI" if m["role"] == "user" else "ASİSTAN"
        tools = [
            f"{t.get('name')}({str(t.get('arguments', ''))[:100]}) -> {str(t.get('preview', ''))[:120]}"
            for t in (m.get("tool_events") or [])
        ]
        atts = [a.get("filename", "") for a in (m.get("attachments") or [])]
        meta = ""
        if atts:
            meta += f"\n[Ekli Belgeler: {', '.join(atts)}]"
        if tools:
            meta += "\n[Çalışan Araçlar: " + " | ".join(tools) + "]"
        body = (m.get("content") or "")[:1_400]
        blocks.append(f"[Mesaj #{m['id']} - {role_tr}]{meta}\n{body}")
    return "\n\n".join(blocks)


async def compact_session_context(
    store: StudioStore,
    session_id: str,
    *,
    upstream_url: str,
    keep_recent: int = DEFAULT_KEEP_RECENT,
    force: bool = False,
) -> Dict[str, Any]:
    """Summarize older messages into `compacted_summary` without deleting any message rows."""
    sess = store.get_session(session_id)
    if not sess:
        raise ValueError("Oturum bulunamadı")

    uncompacted = get_uncompacted_messages(sess)
    if not uncompacted:
        return {"compacted": False, "reason": "Sıkıştırılacak yeni mesaj yok.", "session": sess}

    if force:
        eff_keep = min(keep_recent, max(0, len(uncompacted) - 2)) if len(uncompacted) >= 2 else 0
    else:
        eff_keep = keep_recent

    to_compact = uncompacted[:-eff_keep] if eff_keep > 0 else uncompacted
    if not to_compact:
        return {"compacted": False, "reason": "Mesaj sayısı sıkıştırma eşiğinin altında.", "session": sess}

    prior_summary = (sess.get("compacted_summary") or "").strip()
    transcript = _format_slice_for_summarizer(to_compact)[-14_000:]
    new_until_id = int(to_compact[-1]["id"])

    prompt_parts = []
    if prior_summary:
        prompt_parts.append(f"ÖNCEKİ ÖZET:\n{prior_summary}\n")
    prompt_parts.append(f"YENİ EKLENEN SOHBET KESİTİ (Mesaj #{to_compact[0]['id']}..#{new_until_id}):\n{transcript}")
    compaction_input = "\n".join(prompt_parts)

    summary_md = ""
    try:
        async with httpx.AsyncClient(timeout=45.0) as client:
            resp = await client.post(
                f"{upstream_url.rstrip('/')}/chat/completions",
                json={
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "Sen bağlam sıkıştırma uzmanısın. Verilen geçmiş konuşmayı ve varsa önceki özeti birleştirip "
                                "Türkçe, yoğun ve bilgi kaybı olmayan tek bir Markdown özeti üret. Kesinlikle şu iki başlığı kullan:\n"
                                "### Sohbet Özeti & Kararlar\n"
                                "(Konuşulan ana konular, kullanıcı tercihleri, teknik kararlar, önemli mesaj #ID referansları)\n\n"
                                "### Yapılan İşlemler & Çıktılar\n"
                                "(Okunan/yazılan yerel dosyalar, web aramaları, incelenen PDF/Word belgeleri, güncellenen Canvas taslakları)"
                            ),
                        },
                        {"role": "user", "content": compaction_input},
                    ],
                    "stream": False,
                    "temperature": 0.2,
                    "max_tokens": 700,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
            if resp.status_code == 200:
                raw = ((resp.json().get("choices") or [{}])[0].get("message") or {}).get("content") or ""
                summary_md = _THINK_RE.sub("", raw).strip()
    except httpx.HTTPError:
        summary_md = ""

    if not summary_md:
        summary_md = _deterministic_summary(prior_summary, to_compact)

    updated = store.update_session(
        session_id,
        compacted_summary=summary_md,
        compacted_until_msg_id=new_until_id,
    )
    return {
        "compacted": True,
        "compacted_count": len(to_compact),
        "compacted_until_msg_id": new_until_id,
        "compacted_summary": summary_md,
        "active_est_tokens": estimate_active_tokens(updated or sess),
        "session": updated,
    }


def format_history_search_results(
    matches: List[Dict[str, Any]],
    *,
    total_count: int,
    compacted_until_id: int,
) -> str:
    if not matches:
        return f"Arşivde eşleşen geçmiş mesaj bulunamadı (toplam saklanan mesaj: {total_count})."
    out = [
        f"Sohbet Arşivi Sonuçları ({len(matches)} eşleşme / toplam {total_count} silinmemiş mesaj, "
        f"özet sınırı: #{compacted_until_id}):"
    ]
    for m in matches:
        mid = int(m["id"])
        zone = "Arşivlenmiş" if mid <= compacted_until_id else "Aktif"
        role_tr = "Kullanıcı" if m["role"] == "user" else "Asistan"
        tools = ", ".join(str(t.get("name")) for t in (m.get("tool_events") or []) if t.get("name"))
        atts = ", ".join(str(a.get("filename")) for a in (m.get("attachments") or []) if a.get("filename"))
        header = f"--- [Mesaj #{mid} | {role_tr} | {zone}]"
        if atts:
            header += f" (Belgeler: {atts})"
        if tools:
            header += f" (Araçlar: {tools})"
        body = (m.get("content") or "").strip()
        if len(body) > 1_800:
            body = body[:1_800] + "\n…(tamamı için start_id/end_id ile okuyabilirsin)"
        out.append(f"{header}\n{body}")
    return "\n\n".join(out)

