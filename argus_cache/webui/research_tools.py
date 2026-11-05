"""Verified Scientific Research Loop, Python/SymPy verifier, citation guard, and persistent research_state.md."""

from __future__ import annotations

import ast
import asyncio
import os
import re
import subprocess
import sys
import tempfile
from typing import Any, Callable, Dict, List, Set

from argus_cache.webui.store import StudioStore

RESEARCH_PHASES = {
    "1_arastir": "1. Araştır (Literatür & Kaynak)",
    "2_hipotez": "2. Hipotez Kur (Test Edilebilir Öneri)",
    "3_dogrula": "3. Doğrula (Python / SymPy / Hesaplama)",
    "4_elestir": "4. Eleştir (Çelişki & Sınır Analizi)",
    "5_kaydet": "5. Kaydet (Kalıcı Hafıza & Sonraki Deney)",
}

_URL_RE = re.compile(r"https?://[^\s\)\]>\"'`]+", re.IGNORECASE)
_BANNED_PY_MODULES = frozenset({
    "socket", "urllib", "http", "httpx", "requests", "aiohttp",
    "subprocess", "shutil", "ctypes", "multiprocessing", "pty", " webbrowser",
})
_BANNED_PY_CALLS = frozenset({"system", "popen", "remove", "unlink", "rmdir", "removedirs", "kill", "fork", "execv"})

PYTHON_HINT_RE = re.compile(
    r"(hesapla|doğrula|dogrula|denklem|integral|türev|turev|matris|olasılık|olasilik|istatistik|"
    r"fizik|kimya|mol|stokiyometri|termodinamik|kuantum|sympy|numpy|simülasyon|simulasyon|formül|formul)",
    re.IGNORECASE,
)


def _fn(name: str, desc: str, props: Dict[str, Any], req: List[str]) -> Dict[str, Any]:
    schema: Dict[str, Any] = {"type": "object", "properties": props}
    if req:
        schema["required"] = req
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": schema}}


def build_research_schemas() -> List[Dict[str, Any]]:
    return [
        _fn(
            "verify_with_python",
            "Bilimsel, matematiksel, fiziksel ve kimyasal iddiaları Python (`math`, `numpy`, `sympy`, `statistics`, `decimal`, `fractions`) ile bağımsız olarak hesaplar ve doğrular. Son ifadeyi otomatik yazdırır.",
            {
                "code": {"type": "string", "description": "Çalıştırılacak saf hesaplama/doğrulama Python kodu."},
                "purpose": {"type": "string", "description": "Hangi hipotezin veya denklemin test edildiği."},
            },
            ["code"],
        ),
        _fn(
            "update_research_state",
            "Uzun araştırmalarda hipotezleri, Python doğrulamalarını, eleştirileri, doğrulanmış kaynakları ve sıradaki deneyi kalıcı `research_state.md` dosyasına yazar. Uydurma/okunmamış URL'leri reddeder.",
            {
                "phase": {"type": "string", "enum": list(RESEARCH_PHASES.keys())},
                "question": {"type": "string"},
                "hypotheses": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "statement": {"type": "string"},
                            "status": {"type": "string", "enum": ["TEST_EDILIYOR", "DOGRULANDI", "ELENDI"]},
                            "evidence": {"type": "string"},
                        },
                        "required": ["id", "statement", "status"],
                    },
                },
                "verified_findings": {"type": "array", "items": {"type": "string"}},
                "critiques": {"type": "array", "items": {"type": "string"}},
                "next_experiment": {"type": "string"},
                "sources": {"type": "array", "items": {"type": "string"}},
            },
            ["phase", "question"],
        ),
        _fn(
            "read_research_state",
            "Mevcut oturumun kalıcı `research_state.md` araştırma durum dosyasını okur.",
            {},
            [],
        ),
        _fn(
            "audit_citations",
            "Metindeki veya URL listesindeki kaynakların bu oturumda gerçekten `web_search`/`read_webpage` ile doğrulanıp doğrulanmadığını denetler.",
            {
                "text": {"type": "string", "description": "Denetlenecek kaynak metni veya özet."},
                "urls": {"type": "array", "items": {"type": "string"}, "description": "Kontrol edilecek URL listesi."},
            },
            [],
        ),
    ]


def _normalize_url(u: str) -> str:
    cleaned = u.strip().rstrip(".,;:!?)")
    return cleaned[:-1] if cleaned.endswith("/") else cleaned


def extract_urls(text: str) -> List[str]:
    seen: List[str] = []
    for m in _URL_RE.findall(text or ""):
        nu = _normalize_url(m)
        if nu and nu not in seen:
            seen.append(nu)
    return seen


def audit_citations(text: str, verified_urls: Set[str]) -> Dict[str, Any]:
    norm_verified = {_normalize_url(u) for u in verified_urls if u}
    cited = extract_urls(text)
    matched: List[str] = []
    unverified: List[str] = []
    for cu in cited:
        if cu in norm_verified or any(cu.startswith(v) or v.startswith(cu) for v in norm_verified if len(v) > 12):
            matched.append(cu)
        else:
            unverified.append(cu)
    return {
        "cited_count": len(cited),
        "verified_urls": matched,
        "unverified_urls": unverified,
        "all_verified": len(unverified) == 0,
    }


def _prepare_safe_scientific_code(code: str) -> str:
    tree = ast.parse(code, mode="exec")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0].strip()
                if root in _BANNED_PY_MODULES:
                    raise ValueError(f"Güvenlik sınırı: `{root}` modülü bilimsel doğrulama kum havuzunda kapalıdır.")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0].strip()
            if root in _BANNED_PY_MODULES:
                raise ValueError(f"Güvenlik sınırı: `{root}` modülü bilimsel doğrulama kum havuzunda kapalıdır.")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and node.func.attr in _BANNED_PY_CALLS:
                raise ValueError(f"Güvenlik sınırı: `{node.func.attr}` çağrısı bilimsel kum havuzunda kapalıdır.")
            if isinstance(node.func, ast.Name) and node.func.id in ("open", "exec", "eval", "compile", "__import__"):
                raise ValueError(f"Güvenlik sınırı: `{node.func.id}` çağrısı bilimsel kum havuzunda kapalıdır.")

    if tree.body and isinstance(tree.body[-1], ast.Expr):
        last_expr = tree.body[-1].value
        if not (isinstance(last_expr, ast.Call) and isinstance(last_expr.func, ast.Name) and last_expr.func.id == "print"):
            tree.body[-1] = ast.Expr(
                value=ast.Call(
                    func=ast.Name(id="print", ctx=ast.Load()),
                    args=[ast.Call(func=ast.Name(id="repr", ctx=ast.Load()), args=[last_expr], keywords=[])],
                    keywords=[],
                )
            )
            ast.fix_missing_locations(tree)
            return ast.unparse(tree)
    return code


def run_scientific_python(code: str, timeout_s: float = 15.0) -> str:
    try:
        prepared = _prepare_safe_scientific_code(code)
    except SyntaxError as exc:
        return f"Python Sözdizimi Hatası: {exc}"
    except ValueError as exc:
        return str(exc)

    with tempfile.TemporaryDirectory(prefix="argus_sci_") as tmpdir:
        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "PYTHONIOENCODING": "utf-8",
            "MPLCONFIGDIR": tmpdir,
            "HOME": tmpdir,
        }
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-c", prepared],
                cwd=tmpdir,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            return f"Hesaplama zaman aşımına uğradı ({timeout_s:.0f}s sınırı)."

        out = (proc.stdout or "").strip()
        err = (proc.stderr or "").strip()
        if proc.returncode != 0:
            return f"Hesaplama Hatası (kod {proc.returncode}):\n{err or out}"
        result = out if out else "(Kod hatasız çalıştı fakat çıktı üretmedi; sonuçları `print(...)` ile yazdır.)"
        if len(result) > 5_000:
            result = result[:5_000] + "\n…(çıktı kısaltıldı)"
        return f"✓ Bağımsız Python Doğrulama Çıktısı:\n{result}"


def render_research_state_markdown(
    *,
    phase: str,
    question: str,
    hypotheses: List[Dict[str, Any]],
    verified_findings: List[str],
    critiques: List[str],
    next_experiment: str,
    verified_sources: List[str],
    rejected_sources: List[str],
) -> str:
    phase_label = RESEARCH_PHASES.get(phase, phase)
    lines = [
        "# 🔬 Kalıcı Araştırma Durumu (`research_state.md`)",
        "",
        f"- **Aktif Aşama:** `{phase_label}`",
        f"- **Araştırma Sorusu:** {question.strip()}",
        "",
        "## 1. Hipotez Matrisi",
    ]
    if hypotheses:
        status_icons = {"DOGRULANDI": "✅ DOĞRULANDI", "ELENDI": "❌ ELENDİ", "TEST_EDILIYOR": "🧪 TEST EDİLİYOR"}
        for h in hypotheses:
            st = status_icons.get(str(h.get("status", "")).upper(), "🧪 TEST EDİLİYOR")
            ev = f" — *Kanıt:* {h['evidence']}" if h.get("evidence") else ""
            lines.append(f"- **{h.get('id', 'H?')}** [{st}]: {h.get('statement', '')}{ev}")
    else:
        lines.append("- *(Henüz hipotez kaydedilmedi)*")

    lines.extend(["", "## 2. Doğrulanmış Bulgular & Hesaplamalar"])
    if verified_findings:
        lines.extend(f"- {f}" for f in verified_findings)
    else:
        lines.append("- *(Henüz doğrulanmış bulgu eklenmedi)*")

    lines.extend(["", "## 3. Eleştirel Analiz & Sınır Koşulları"])
    if critiques:
        lines.extend(f"- {c}" for c in critiques)
    else:
        lines.append("- *(Açık çelişki veya sınır koşulu notu yok)*")

    lines.extend([
        "",
        "## 4. Sıradaki Deney / Açık Adım",
        next_experiment.strip() or "*(Belirtilmedi)*",
        "",
        "## 5. Araçla Doğrulanmış Kaynaklar",
    ])
    if verified_sources:
        lines.extend(f"- {u}" for u in verified_sources)
    else:
        lines.append("- *(Henüz web/belge kaynağı eklenmedi)*")

    if rejected_sources:
        lines.extend([
            "",
            "> [!WARNING]",
            "> **Uydurma Kaynak Koruması:** Aşağıdaki linkler `web_search`/`read_webpage` kayıtlarında bulunmadığı için reddedildi:",
            *[f"> - `{u}`" for u in rejected_sources],
        ])
    return "\n".join(lines) + "\n"


async def execute_research_tool(
    name: str,
    args: Dict[str, Any],
    *,
    store: StudioStore,
    session_id: str,
    verified_urls: Set[str],
    emit_event: Callable[[str, Dict[str, Any]], None],
) -> str:
    if name == "verify_with_python":
        code = str(args.get("code") or "").strip()
        if not code:
            return "Hata: `code` alanı boş."
        return await asyncio.to_thread(run_scientific_python, code)

    if name == "read_research_state":
        sess = store.get_session(session_id)
        md = (sess or {}).get("research_state_md") or ""
        return md if md.strip() else "Bu oturumda henüz `research_state.md` oluşturulmadı."

    if name == "audit_citations":
        raw_text = str(args.get("text") or "")
        url_list = [str(u).strip() for u in (args.get("urls") or []) if str(u).strip()]
        combined = "\n".join([raw_text, *url_list]).strip()
        audit = audit_citations(combined, verified_urls)
        lines = [
            f"Kaynak Denetimi Sonucu (Toplam incelenen URL: {audit['cited_count']}):",
            f"- Doğrulanmış ({len(audit['verified_urls'])}): {', '.join(audit['verified_urls']) or '(yok)'}",
            f"- Doğrulanmamış / Uydurma Riski ({len(audit['unverified_urls'])}): {', '.join(audit['unverified_urls']) or '(yok — tümü temiz)'}",
        ]
        return "\n".join(lines)

    if name == "update_research_state":
        phase = str(args.get("phase") or "1_arastir").strip()
        if phase not in RESEARCH_PHASES:
            phase = "1_arastir"
        question = str(args.get("question") or "Araştırma Problemi").strip()
        hypotheses = [h for h in (args.get("hypotheses") or []) if isinstance(h, dict)]
        findings = [str(x).strip() for x in (args.get("verified_findings") or []) if str(x).strip()]
        critiques = [str(x).strip() for x in (args.get("critiques") or []) if str(x).strip()]
        next_exp = str(args.get("next_experiment") or "").strip()
        raw_sources = [str(x).strip() for x in (args.get("sources") or []) if str(x).strip()]

        joined_check = "\n".join(raw_sources + findings)
        audit = audit_citations(joined_check, verified_urls)
        md = render_research_state_markdown(
            phase=phase,
            question=question,
            hypotheses=hypotheses,
            verified_findings=findings,
            critiques=critiques,
            next_experiment=next_exp,
            verified_sources=audit["verified_urls"],
            rejected_sources=audit["unverified_urls"],
        )
        updated = store.update_session(session_id, research_state_md=md, research_phase=phase)
        path_str = (updated or {}).get("research_state_path") or str(store.research_state_file(session_id))
        emit_event(
            "research_state",
            {
                "phase": phase,
                "phase_label": RESEARCH_PHASES[phase],
                "markdown": md,
                "path": path_str,
                "unverified_urls": audit["unverified_urls"],
            },
        )
        msg = f"`research_state.md` güncellendi ({RESEARCH_PHASES[phase]}) → `{path_str}`."
        if audit["unverified_urls"]:
            msg += (
                "\nUYARI (Uydurma Kaynak Engeli): Şu URL'ler `web_search` veya `read_webpage` çıktısında "
                f"görülmediği için doğrulanmış kaynaklardan çıkarıldı: {', '.join(audit['unverified_urls'])}. "
                "Lütfen yalnızca araçla gerçekten okuduğun kaynakları yaz."
            )
        return msg

    return f"Bilinmeyen araştırma aracı: {name}"

