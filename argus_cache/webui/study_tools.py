"""Interactive Quiz & Flashcards builders and executors for ARGUS Study Studio."""

from __future__ import annotations

from typing import Any, Callable, Dict, List

from argus_cache.webui.store import StudioStore


def _fn(name: str, desc: str, props: Dict[str, Any], req: List[str]) -> Dict[str, Any]:
    schema: Dict[str, Any] = {"type": "object", "properties": props}
    if req:
        schema["required"] = req
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": schema}}


def build_study_schemas() -> List[Dict[str, Any]]:
    return [
        _fn(
            "publish_quiz",
            "Sağdaki Stüdyo panelinde anında puanlanan, ipuçlu ve çözümlü interaktif çoktan seçmeli Test/Quiz başlatır. Cevapları düz metne dökmek yerine sınav/test istendiğinde mutlaka bu aracı kullan.",
            {
                "title": {"type": "string"},
                "questions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "question": {"type": "string"},
                            "options": {"type": "array", "items": {"type": "string"}, "description": "4 seçenek dizisi."},
                            "correct_index": {"type": "integer", "description": "Doğru seçeneğin 0-tabanlı indeksi (0..3)."},
                            "explanation": {"type": "string", "description": "Adım adım çözüm ve neden doğru olduğu."},
                            "hint": {"type": "string", "description": "Cevabı ele vermeyen kısa ipucu."},
                            "difficulty": {"type": "string", "enum": ["Kolay", "Orta", "Zor"]},
                        },
                        "required": ["question", "options", "correct_index", "explanation"],
                    },
                },
            },
            ["title", "questions"],
        ),
        _fn(
            "publish_flashcards",
            "Sağdaki Stüdyo panelinde çevrilebilir (ön yüz / arka yüz), hafıza ipuçlu ve ilerleme takipli interaktif Flashcards (Bilgi Kartları) destesi oluşturur.",
            {
                "title": {"type": "string"},
                "cards": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "front": {"type": "string", "description": "Kartın ön yüzü: kavram, soru veya formül adı."},
                            "back": {"type": "string", "description": "Kartın arka yüzü: net tanım, formül veya yanıt."},
                            "tag": {"type": "string", "description": "Konu etiketi (ör: Termodinamik, Türev)."},
                            "mnemonic": {"type": "string", "description": "Akılda tutmayı kolaylaştıran kısa kodlama/ipucu."},
                        },
                        "required": ["front", "back"],
                    },
                },
            },
            ["title", "cards"],
        ),
    ]


def execute_study_tool(
    name: str,
    args: Dict[str, Any],
    *,
    store: StudioStore,
    session_id: str,
    emit_event: Callable[[str, Dict[str, Any]], None],
) -> str:
    if name == "publish_quiz":
        title = str(args.get("title") or "Konu Tarama Testi").strip()
        raw_qs = args.get("questions") or []
        clean_qs: List[Dict[str, Any]] = []
        for idx, q in enumerate(raw_qs):
            if not isinstance(q, dict):
                continue
            opts = [str(o).strip() for o in (q.get("options") or []) if str(o).strip()]
            if len(opts) < 2 or not str(q.get("question") or "").strip():
                continue
            c_idx = int(q.get("correct_index", 0))
            if not (0 <= c_idx < len(opts)):
                c_idx = 0
            clean_qs.append({
                "id": idx + 1,
                "question": str(q["question"]).strip(),
                "options": opts[:6],
                "correct_index": c_idx,
                "explanation": str(q.get("explanation") or "").strip(),
                "hint": str(q.get("hint") or "").strip(),
                "difficulty": str(q.get("difficulty") or "Orta"),
            })
        if not clean_qs:
            return "Hata: Geçerli en az 1 çoktan seçmeli soru (`options` >= 2) gerekli."
        quiz_obj = {"title": title, "questions": clean_qs}
        store.update_session(session_id, study_quiz=quiz_obj)
        emit_event("study_quiz", quiz_obj)
        return (
            f"İnteraktif test sağdaki Stüdyo panelinde yayınlandı: '{title}' ({len(clean_qs)} soru). "
            "Sohbette doğru şıkları açıkça yazma ki kullanıcı sağ panelde kendisi çözebilsin!"
        )

    if name == "publish_flashcards":
        title = str(args.get("title") or "Bilgi Kartları").strip()
        raw_cards = args.get("cards") or []
        clean_cards: List[Dict[str, Any]] = []
        for idx, c in enumerate(raw_cards):
            if not isinstance(c, dict):
                continue
            front = str(c.get("front") or "").strip()
            back = str(c.get("back") or "").strip()
            if not front or not back:
                continue
            clean_cards.append({
                "id": idx + 1,
                "front": front,
                "back": back,
                "tag": str(c.get("tag") or "Kavram").strip(),
                "mnemonic": str(c.get("mnemonic") or "").strip(),
            })
        if not clean_cards:
            return "Hata: Geçerli en az 1 bilgi kartı (`front` ve `back`) gerekli."
        deck_obj = {"title": title, "cards": clean_cards}
        store.update_session(session_id, study_flashcards=deck_obj)
        emit_event("study_flashcards", deck_obj)
        return f"İnteraktif Flashcards destesi sağdaki Stüdyo panelinde açıldı: '{title}' ({len(clean_cards)} kart)."

    return f"Bilinmeyen etüt aracı: {name}"

