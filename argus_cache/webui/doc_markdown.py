"""Convert PDF, Word (.docx/.doc/.odt), Slides (.pptx), Sheets (.xlsx/.csv), HTML, and code into structured Markdown.

Design goal: never dump raw binary or giant unindexed blobs into the LLM context.
Large documents are split into logical Markdown sections (pages/headings/slides) with an outline
and keyword retrieval so only relevant sections enter the prompt unless requested in full.
"""

from __future__ import annotations

import csv
import io
import re
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin

DIRECT_CONTEXT_CHAR_LIMIT = 11_000
EXCERPT_CHAR_LIMIT = 2_400

_CODE_LANGS = {
    ".py": "python", ".js": "javascript", ".ts": "typescript", ".tsx": "tsx",
    ".jsx": "jsx", ".c": "c", ".cpp": "cpp", ".cc": "cpp", ".h": "c",
    ".hpp": "cpp", ".cu": "cuda", ".rs": "rust", ".go": "go", ".java": "java",
    ".sh": "bash", ".zsh": "zsh", ".json": "json", ".yaml": "yaml",
    ".yml": "yaml", ".toml": "toml", ".sql": "sql", ".css": "css",
    ".lua": "lua", ".kt": "kotlin", ".rb": "ruby", ".php": "php",
}


@dataclass
class DocSection:
    index: int
    title: str
    content: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "index": self.index,
            "title": self.title,
            "content": self.content,
            "chars": len(self.content),
        }


@dataclass
class ConvertedDocument:
    title: str
    fmt: str
    markdown: str
    sections: List[DocSection]
    page_count: int

    @property
    def char_count(self) -> int:
        return len(self.markdown)

    @property
    def est_tokens(self) -> int:
        return max(1, (len(self.markdown) + 3) // 4)

    def outline(self, limit: int = 30) -> List[str]:
        items = [f"[{s.index}] {s.title} (~{max(1, len(s.content)//4)} tok)" for s in self.sections[:limit]]
        if len(self.sections) > limit:
            items.append(f"... (+{len(self.sections) - limit} bölüm daha)")
        return items


def _clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    # Rejoin PDF hyphenated line-breaks across lowercase words
    text = re.sub(r"([a-zçğıöşü])-\n([a-zçğıöşü])", r"\1\2", text)
    lines = [ln.rstrip() for ln in text.split("\n")]
    cleaned: List[str] = []
    blank_run = 0
    for ln in lines:
        if not ln.strip():
            blank_run += 1
            if blank_run <= 2:
                cleaned.append("")
        else:
            blank_run = 0
            cleaned.append(ln)
    return "\n".join(cleaned).strip()


def _format_table(rows: List[List[str]]) -> str:
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    if width == 0:
        return ""
    norm = [[(c or "").replace("\n", " ").replace("|", "\\|").strip() for c in r] + [""] * (width - len(r)) for r in rows]
    header = norm[0]
    if not any(header) and len(norm) > 1:
        header = [f"Sütun {i + 1}" for i in range(width)]
        body = norm
    else:
        body = norm[1:]
    out = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * width) + " |",
    ]
    for row in body:
        if any(row):
            out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


class _MarkdownHTMLParser(HTMLParser):
    _SKIP_TAGS = {"script", "style", "noscript", "svg", "iframe", "nav", "footer", "aside", "form"}

    def __init__(self, base_url: str = "") -> None:
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self._skip_depth = 0
        self._parts: List[str] = []
        self._href_stack: List[Tuple[str, int]] = []
        self._in_pre = False
        self._in_code = False
        self._table_rows: Optional[List[List[str]]] = None
        self._cur_row: Optional[List[str]] = None
        self._cur_cell: Optional[List[str]] = None

    def _emit(self, txt: str) -> None:
        if self._skip_depth > 0:
            return
        if self._cur_cell is not None:
            self._cur_cell.append(txt)
        else:
            self._parts.append(txt)

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        t = tag.lower()
        if t in self._SKIP_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth > 0:
            return
        attr_map = dict(attrs)
        if t in ("h1", "h2", "h3", "h4", "h5", "h6"):
            level = int(t[1])
            self._emit(f"\n\n{'#' * level} ")
        elif t in ("p", "div", "section", "article", "blockquote"):
            self._emit("\n\n> " if t == "blockquote" else "\n\n")
        elif t == "br":
            self._emit("\n")
        elif t == "li":
            self._emit("\n- ")
        elif t in ("strong", "b"):
            self._emit("**")
        elif t in ("em", "i"):
            self._emit("*")
        elif t == "pre":
            self._in_pre = True
            self._emit("\n\n```\n")
        elif t == "code" and not self._in_pre:
            self._in_code = True
            self._emit("`")
        elif t == "a":
            href = (attr_map.get("href") or "").strip()
            if href and self.base_url:
                href = urljoin(self.base_url, href)
            self._href_stack.append((href, len(self._parts)))
            self._emit("[")
        elif t == "table":
            self._table_rows = []
        elif t == "tr" and self._table_rows is not None:
            self._cur_row = []
        elif t in ("td", "th") and self._cur_row is not None:
            self._cur_cell = []

    def handle_endtag(self, tag: str) -> None:
        t = tag.lower()
        if t in self._SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if self._skip_depth > 0:
            return
        if t in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self._emit("\n\n")
        elif t in ("strong", "b"):
            self._emit("**")
        elif t in ("em", "i"):
            self._emit("*")
        elif t == "pre":
            self._in_pre = False
            self._emit("\n```\n\n")
        elif t == "code" and self._in_code:
            self._in_code = False
            self._emit("`")
        elif t == "a" and self._href_stack:
            href, _ = self._href_stack.pop()
            if href and not href.startswith("javascript:"):
                self._emit(f"]({href})")
            else:
                self._emit("]")
        elif t in ("td", "th") and self._cur_row is not None and self._cur_cell is not None:
            self._cur_row.append("".join(self._cur_cell).strip())
            self._cur_cell = None
        elif t == "tr" and self._table_rows is not None and self._cur_row is not None:
            if any(self._cur_row):
                self._table_rows.append(self._cur_row)
            self._cur_row = None
        elif t == "table" and self._table_rows is not None:
            tbl_md = _format_table(self._table_rows)
            self._table_rows = None
            if tbl_md:
                self._parts.append(f"\n\n{tbl_md}\n\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth > 0:
            return
        if self._in_pre:
            self._emit(data)
        else:
            collapsed = re.sub(r"[ \t\n]+", " ", data)
            if collapsed:
                self._emit(collapsed)


def html_to_markdown(html: str, *, base_url: str = "") -> str:
    parser = _MarkdownHTMLParser(base_url=base_url)
    parser.feed(html)
    raw = "".join(parser._parts)
    raw = re.sub(r"\[\s*\]\([^)]*\)", "", raw)
    return _clean_text(raw)


def _convert_pdf(path: Path, title: str) -> ConvertedDocument:
    pdftotext = shutil.which("pdftotext")
    if not pdftotext:
        raise RuntimeError("PDF dönüşümü için 'pdftotext' bulunamadı.")
    res = subprocess.run(
        [pdftotext, "-enc", "UTF-8", str(path), "-"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if res.returncode != 0 and not res.stdout:
        raise RuntimeError(f"PDF okunamadı: {res.stderr.strip() or 'pdftotext hatası'}")

    raw_pages = res.stdout.split("\x0c")
    sections: List[DocSection] = []
    md_blocks = [f"# {title}\n"]
    for idx, page in enumerate(raw_pages, start=1):
        cleaned = _clean_text(page)
        if not cleaned:
            continue
        first_line = next((ln.strip() for ln in cleaned.splitlines() if ln.strip()), "")
        short_hint = first_line[:64] + ("…" if len(first_line) > 64 else "")
        sec_title = f"Sayfa {idx}" + (f" — {short_hint}" if short_hint else "")
        sec_md = f"## {sec_title}\n\n{cleaned}"
        sections.append(DocSection(index=len(sections) + 1, title=sec_title, content=sec_md))
        md_blocks.append(sec_md)

    if not sections:
        msg = "_Bu PDF metin katmanı içermiyor (taranmış görüntü olabilir)._"
        sections.append(DocSection(index=1, title="Sayfa 1", content=msg))
        md_blocks.append(msg)

    return ConvertedDocument(
        title=title,
        fmt="pdf",
        markdown="\n\n".join(md_blocks).strip(),
        sections=sections,
        page_count=max(1, len([p for p in raw_pages if p.strip()])),
    )


def _local_tag(el: ET.Element) -> str:
    return el.tag.rsplit("}", 1)[-1]


def _docx_paragraph_to_md(p_el: ET.Element, w_ns: str) -> str:
    style_val = ""
    is_list = False
    ppr = p_el.find(f"{{{w_ns}}}pPr")
    if ppr is not None:
        pstyle = ppr.find(f"{{{w_ns}}}pStyle")
        if pstyle is not None:
            style_val = pstyle.attrib.get(f"{{{w_ns}}}val", "") or pstyle.attrib.get("val", "")
        if ppr.find(f"{{{w_ns}}}numPr") is not None:
            is_list = True

    runs: List[str] = []
    for child in p_el.iter():
        tag = _local_tag(child)
        if tag == "r":
            rpr = child.find(f"{{{w_ns}}}rPr")
            bold = rpr is not None and rpr.find(f"{{{w_ns}}}b") is not None
            italic = rpr is not None and rpr.find(f"{{{w_ns}}}i") is not None
            txt = "".join((t.text or "") for t in child.findall(f".//{{{w_ns}}}t"))
            if child.find(f"{{{w_ns}}}br") is not None:
                txt += "\n"
            if not txt:
                continue
            if bold and txt.strip():
                txt = f"**{txt.strip()}** " if txt.endswith(" ") else f"**{txt}**"
            if italic and txt.strip():
                txt = f"*{txt}*"
            runs.append(txt)

    line = "".join(runs).strip()
    if not line:
        return ""
    low = style_val.lower()
    for lvl in range(1, 7):
        if f"heading{lvl}" in low or f"heading {lvl}" in low or f"balk{lvl}" in low or f"baslik{lvl}" in low:
            return f"{'#' * lvl} {line.strip('* ')}"
    if "title" in low:
        return f"# {line.strip('* ')}"
    if is_list or "list" in low:
        return f"- {line}"
    return line


def _convert_docx(path: Path, title: str) -> ConvertedDocument:
    w_ns = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    blocks: List[str] = [f"# {title}"]
    with zipfile.ZipFile(path, "r") as zf:
        xml_bytes = zf.read("word/document.xml")
    root = ET.fromstring(xml_bytes)
    body = root.find(f"{{{w_ns}}}body")
    if body is None:
        body = root

    for el in list(body):
        tag = _local_tag(el)
        if tag == "p":
            md_line = _docx_paragraph_to_md(el, w_ns)
            if md_line:
                blocks.append(md_line)
        elif tag == "tbl":
            rows: List[List[str]] = []
            for tr in el.findall(f".//{{{w_ns}}}tr"):
                row_cells: List[str] = []
                for tc in tr.findall(f"./{{{w_ns}}}tc"):
                    cell_txts = [_docx_paragraph_to_md(p, w_ns) for p in tc.findall(f".//{{{w_ns}}}p")]
                    row_cells.append(" ".join(t for t in cell_txts if t))
                if any(row_cells):
                    rows.append(row_cells)
            tbl_md = _format_table(rows)
            if tbl_md:
                blocks.append(tbl_md)

    full_md = _clean_text("\n\n".join(blocks))
    sections = _split_markdown_sections(full_md, fallback_title=title)
    return ConvertedDocument(title=title, fmt="docx", markdown=full_md, sections=sections, page_count=len(sections))


def _convert_odt(path: Path, title: str) -> ConvertedDocument:
    with zipfile.ZipFile(path, "r") as zf:
        root = ET.fromstring(zf.read("content.xml"))
    blocks = [f"# {title}"]
    for el in root.iter():
        tag = _local_tag(el)
        if tag in ("h", "p"):
            txt = "".join(el.itertext()).strip()
            if not txt:
                continue
            if tag == "h":
                blocks.append(f"## {txt}")
            else:
                blocks.append(txt)
    full_md = _clean_text("\n\n".join(blocks))
    sections = _split_markdown_sections(full_md, fallback_title=title)
    return ConvertedDocument(title=title, fmt="odt", markdown=full_md, sections=sections, page_count=len(sections))


def _convert_pptx(path: Path, title: str) -> ConvertedDocument:
    sections: List[DocSection] = []
    md_blocks = [f"# {title}"]
    with zipfile.ZipFile(path, "r") as zf:
        slides = sorted(
            [n for n in zf.namelist() if re.match(r"^ppt/slides/slide\d+\.xml$", n)],
            key=lambda s: int(re.search(r"(\d+)", s).group(1)),
        )
        for idx, sname in enumerate(slides, start=1):
            root = ET.fromstring(zf.read(sname))
            paras: List[str] = []
            for p_el in root.iter():
                if _local_tag(p_el) == "p":
                    t = "".join(t_el.text or "" for t_el in p_el.iter() if _local_tag(t_el) == "t").strip()
                    if t:
                        paras.append(f"- {t}")
            if not paras:
                continue
            head = paras[0].lstrip("- ").strip()[:64]
            sec_title = f"Slayt {idx} — {head}"
            sec_md = f"## {sec_title}\n\n" + "\n".join(paras)
            sections.append(DocSection(index=len(sections) + 1, title=sec_title, content=sec_md))
            md_blocks.append(sec_md)

    full_md = "\n\n".join(md_blocks).strip()
    return ConvertedDocument(title=title, fmt="pptx", markdown=full_md, sections=sections, page_count=max(1, len(sections)))


def _convert_xlsx(path: Path, title: str, max_rows: int = 200) -> ConvertedDocument:
    sections: List[DocSection] = []
    md_blocks = [f"# {title}"]
    with zipfile.ZipFile(path, "r") as zf:
        shared: List[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            s_root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in s_root:
                if _local_tag(si) == "si":
                    shared.append("".join(t.text or "" for t in si.iter() if _local_tag(t) == "t"))
        sheets = sorted([n for n in zf.namelist() if re.match(r"^xl/worksheets/sheet\d+\.xml$", n)])
        for idx, sh in enumerate(sheets, start=1):
            root = ET.fromstring(zf.read(sh))
            rows: List[List[str]] = []
            total_rows = 0
            for r_el in root.iter():
                if _local_tag(r_el) != "row":
                    continue
                total_rows += 1
                if len(rows) >= max_rows:
                    continue
                vals: List[str] = []
                for c_el in r_el:
                    if _local_tag(c_el) != "c":
                        continue
                    ctype = c_el.attrib.get("t", "")
                    v_el = next((ch for ch in c_el if _local_tag(ch) in ("v", "is")), None)
                    raw = "".join(v_el.itertext()).strip() if v_el is not None else ""
                    if ctype == "s" and raw.isdigit() and int(raw) < len(shared):
                        raw = shared[int(raw)]
                    vals.append(raw)
                if any(vals):
                    rows.append(vals)
            if not rows:
                continue
            tbl = _format_table(rows)
            note = f"\n\n_(Toplam {total_rows} satırdan ilk {len(rows)} satır gösteriliyor.)_" if total_rows > len(rows) else ""
            sec_title = f"Tablo {idx}"
            sec_md = f"## {sec_title}\n\n{tbl}{note}"
            sections.append(DocSection(index=len(sections) + 1, title=sec_title, content=sec_md))
            md_blocks.append(sec_md)

    full_md = "\n\n".join(md_blocks).strip()
    return ConvertedDocument(title=title, fmt="xlsx", markdown=full_md, sections=sections, page_count=max(1, len(sections)))


def _convert_csv(path: Path, title: str, delimiter: str = ",", max_rows: int = 200) -> ConvertedDocument:
    text = path.read_text(encoding="utf-8", errors="replace")
    reader = list(csv.reader(io.StringIO(text), delimiter=delimiter))
    shown = reader[:max_rows]
    tbl = _format_table(shown)
    note = f"\n\n_(Toplam {len(reader)} satırdan ilk {len(shown)} satır gösteriliyor.)_" if len(reader) > len(shown) else ""
    md = f"# {title}\n\n{tbl}{note}".strip()
    sections = [DocSection(index=1, title=title, content=md)]
    return ConvertedDocument(title=title, fmt="csv", markdown=md, sections=sections, page_count=1)


def _convert_legacy_office(path: Path, title: str) -> ConvertedDocument:
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if not soffice:
        raise RuntimeError("Eski .doc/.rtf dönüşümü için LibreOffice (soffice) bulunamadı.")
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(
            [soffice, "--headless", "--convert-to", "txt:Text", "--outdir", tmp, str(path)],
            capture_output=True,
            timeout=45,
        )
        txts = list(Path(tmp).glob("*.txt"))
        if not txts:
            raise RuntimeError(f"{path.name} metne dönüştürülemedi.")
        content = _clean_text(txts[0].read_text(encoding="utf-8", errors="replace"))
    full_md = f"# {title}\n\n{content}".strip()
    sections = _split_markdown_sections(full_md, fallback_title=title)
    return ConvertedDocument(title=title, fmt=path.suffix.lstrip(".").lower(), markdown=full_md, sections=sections, page_count=len(sections))


def _split_markdown_sections(markdown: str, *, fallback_title: str, chunk_chars: int = 3200) -> List[DocSection]:
    """Split Markdown by headings or bounded character chunks so sections stay scannable."""
    parts = re.split(r"(?m)^(?=#{1,3}\s+)", markdown)
    sections: List[DocSection] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        first_ln = part.splitlines()[0].lstrip("# ").strip() or fallback_title
        if len(part) <= chunk_chars * 1.4:
            sections.append(DocSection(index=len(sections) + 1, title=first_ln[:72], content=part))
        else:
            paras = part.split("\n\n")
            buf: List[str] = []
            cur = 0
            part_idx = 1
            for p in paras:
                if cur + len(p) > chunk_chars and buf:
                    sections.append(
                        DocSection(
                            index=len(sections) + 1,
                            title=f"{first_ln[:56]} (Kısım {part_idx})",
                            content="\n\n".join(buf),
                        )
                    )
                    buf = [p]
                    cur = len(p)
                    part_idx += 1
                else:
                    buf.append(p)
                    cur += len(p)
            if buf:
                sections.append(
                    DocSection(
                        index=len(sections) + 1,
                        title=f"{first_ln[:56]}" + (f" (Kısım {part_idx})" if part_idx > 1 else ""),
                        content="\n\n".join(buf),
                    )
                )
    if not sections:
        sections.append(DocSection(index=1, title=fallback_title, content=markdown))
    return sections


def convert_file_to_markdown(path: Path, *, display_name: Optional[str] = None) -> ConvertedDocument:
    """Convert any supported document or source file into structured Markdown."""
    name = display_name or path.name
    ext = Path(name).suffix.lower()
    if ext == ".pdf":
        return _convert_pdf(path, name)
    if ext == ".docx":
        return _convert_docx(path, name)
    if ext == ".odt":
        return _convert_odt(path, name)
    if ext == ".pptx":
        return _convert_pptx(path, name)
    if ext == ".xlsx":
        return _convert_xlsx(path, name)
    if ext == ".csv":
        return _convert_csv(path, name, delimiter=",")
    if ext == ".tsv":
        return _convert_csv(path, name, delimiter="\t")
    if ext in (".doc", ".rtf"):
        return _convert_legacy_office(path, name)
    if ext in (".html", ".htm"):
        raw_html = path.read_text(encoding="utf-8", errors="replace")
        md = f"# {name}\n\n" + html_to_markdown(raw_html)
        secs = _split_markdown_sections(md, fallback_title=name)
        return ConvertedDocument(title=name, fmt="html", markdown=md, sections=secs, page_count=len(secs))

    raw = path.read_text(encoding="utf-8", errors="replace")
    if ext in (".md", ".markdown"):
        md = _clean_text(raw)
        secs = _split_markdown_sections(md, fallback_title=name)
        return ConvertedDocument(title=name, fmt="md", markdown=md, sections=secs, page_count=len(secs))
    if ext in _CODE_LANGS:
        lang = _CODE_LANGS[ext]
        md = f"# {name}\n\n```{lang}\n{raw.rstrip()}\n```"
        secs = _split_markdown_sections(raw, fallback_title=name)
        return ConvertedDocument(title=name, fmt=lang, markdown=md, sections=secs, page_count=len(secs))

    cleaned = _clean_text(raw)
    md = f"# {name}\n\n{cleaned}"
    secs = _split_markdown_sections(md, fallback_title=name)
    return ConvertedDocument(title=name, fmt=ext.lstrip(".") or "txt", markdown=md, sections=secs, page_count=len(secs))


def query_sections(sections: List[Dict[str, object]], query: str = "", start: int = 1, end: int = 3) -> str:
    """Retrieve either keyword-matched sections or an explicit 1-based range."""
    if not sections:
        return "Belge bölümü bulunamadı."
    if query.strip():
        terms = [w.lower() for w in re.findall(r"\w{2,}", query) if len(w) >= 2]
        scored = []
        for s in sections:
            hay = f"{s.get('title', '')}\n{s.get('content', '')}".lower()
            score = sum(hay.count(t) * (3 if t in str(s.get("title", "")).lower() else 1) for t in terms)
            if score > 0:
                scored.append((score, s))
        scored.sort(key=lambda x: x[0], reverse=True)
        picked = [s for _, s in scored[:4]]
        if not picked:
            picked = sections[:2]
    else:
        lo = max(0, start - 1)
        hi = max(lo + 1, min(len(sections), end))
        picked = sections[lo:hi]

    out = [f"### [{s['index']}/{len(sections)}] {s['title']}\n\n{s['content']}" for s in picked]
    return "\n\n---\n\n".join(out)


def build_prompt_attachment_block(doc_record: Dict[str, object], user_prompt: str) -> str:
    """Format an attached Markdown document for the prompt without context bloat."""
    doc_id = doc_record["id"]
    title = doc_record["filename"]
    md_text = Path(str(doc_record["md_path"])).read_text(encoding="utf-8", errors="replace")
    sections: List[Dict[str, object]] = list(doc_record.get("sections") or [])
    est_tok = doc_record.get("est_tokens", max(1, len(md_text) // 4))

    if len(md_text) <= DIRECT_CONTEXT_CHAR_LIMIT:
        return (
            f'<ekli_belge id="{doc_id}" dosya="{title}" format="markdown" yaklasik_token="{est_tok}">\n'
            f"{md_text}\n"
            f"</ekli_belge>"
        )

    outline_lines = [
        f"  [{s['index']}] {s['title']}" for s in sections[:35]
    ]
    if len(sections) > 35:
        outline_lines.append(f"  ... (+{len(sections) - 35} bölüm daha)")
    matched = query_sections(sections, query=user_prompt, start=1, end=2)
    return (
        f'<ekli_belge_indeksi id="{doc_id}" dosya="{title}" bolum_sayisi="{len(sections)}" '
        f'yaklasik_token="{est_tok}" not="Belge büyük olduğu için Markdown indeksine çevrildi. '
        f'Diğer sayfa/bölümleri okumak için read_uploaded_doc(doc_id=\'{doc_id}\', ...) aracını çağır.">\n'
        f"İÇİNDEKİLER:\n" + "\n".join(outline_lines) + "\n\n"
        f"SORUYLA EŞLEŞEN / İLK BÖLÜMLER:\n{matched[:EXCERPT_CHAR_LIMIT * 2]}\n"
        f"</ekli_belge_indeksi>"
    )

