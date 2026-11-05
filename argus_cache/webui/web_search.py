"""Free, API-keyless Web Search and SSRF-safe Markdown Webpage Reader."""

from __future__ import annotations

import asyncio
import html
import ipaddress
import re
import socket
from typing import Dict, List, Tuple
from urllib.parse import parse_qs, unquote, urlparse

import httpx

from argus_cache.webui.doc_markdown import html_to_markdown

_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)


def _unwrap_ddg_url(raw_url: str) -> str:
    raw_url = html.unescape(raw_url.strip())
    if raw_url.startswith("//"):
        raw_url = "https:" + raw_url
    parsed = urlparse(raw_url)
    if "duckduckgo.com" in parsed.netloc and parsed.path.startswith("/l/"):
        qs = parse_qs(parsed.query)
        if "uddg" in qs and qs["uddg"]:
            return unquote(qs["uddg"][0])
    return raw_url


def _strip_tags(fragment: str) -> str:
    txt = re.sub(r"<[^>]+>", " ", fragment)
    return re.sub(r"\s+", " ", html.unescape(txt)).strip()


def _parse_ddg_html(page: str, max_results: int) -> List[Dict[str, str]]:
    results: List[Dict[str, str]] = []
    blocks = re.split(r'<div[^>]+class="[^"]*result\b[^"]*"[^>]*>', page)[1:]
    for blk in blocks:
        m_link = re.search(
            r'<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
            blk,
            re.DOTALL,
        )
        if not m_link:
            continue
        url = _unwrap_ddg_url(m_link.group(1))
        if not url.startswith(("http://", "https://")) or "duckduckgo.com/y.js" in url:
            continue
        title = _strip_tags(m_link.group(2))
        m_snip = re.search(
            r'class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</(?:a|div|span)>',
            blk,
            re.DOTALL,
        )
        snippet = _strip_tags(m_snip.group(1)) if m_snip else ""
        if title:
            results.append({
                "title": title,
                "url": url,
                "snippet": snippet,
                "domain": urlparse(url).netloc.removeprefix("www."),
            })
        if len(results) >= max_results:
            break
    return results


async def _wikipedia_fallback(client: httpx.AsyncClient, query: str, max_results: int) -> List[Dict[str, str]]:
    results: List[Dict[str, str]] = []
    for lang in ("tr", "en"):
        if len(results) >= max_results:
            break
        try:
            resp = await client.get(
                f"https://{lang}.wikipedia.org/w/api.php",
                params={
                    "action": "query",
                    "list": "search",
                    "srsearch": query,
                    "srlimit": max(2, max_results - len(results)),
                    "format": "json",
                },
            )
            if resp.status_code != 200:
                continue
            for item in resp.json().get("query", {}).get("search", []):
                title = item.get("title", "")
                if not title:
                    continue
                slug = title.replace(" ", "_")
                results.append({
                    "title": f"{title} ({lang.upper()} Wikipedia)",
                    "url": f"https://{lang}.wikipedia.org/wiki/{slug}",
                    "snippet": _strip_tags(item.get("snippet", "")),
                    "domain": f"{lang}.wikipedia.org",
                })
                if len(results) >= max_results:
                    break
        except httpx.HTTPError:
            continue
    return results


async def _search_with_client(client: httpx.AsyncClient, query: str, max_results: int) -> List[Dict[str, str]]:
    q = query.strip()
    if not q:
        return []
    try:
        resp = await client.post("https://html.duckduckgo.com/html/", data={"q": q})
        if resp.status_code == 200:
            items = _parse_ddg_html(resp.text, max_results)
            if items:
                return items
    except httpx.HTTPError:
        pass
    return await _wikipedia_fallback(client, q, max_results)


async def web_search(query: str, max_results: int = 6) -> List[Dict[str, str]]:
    """Search the web without paid API keys; returns title, url, snippet, domain."""
    max_results = max(1, min(10, int(max_results)))
    headers = {"User-Agent": _USER_AGENT, "Accept-Language": "tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7"}
    async with httpx.AsyncClient(headers=headers, follow_redirects=True, timeout=10.0) as client:
        return await _search_with_client(client, query, max_results)


async def parallel_web_search(
    queries: List[str],
    *,
    max_results_per_query: int = 4,
) -> List[Tuple[str, List[Dict[str, str]]]]:
    """Execute multiple web search queries concurrently over a shared HTTP pool."""
    cleaned = [q.strip() for q in queries if q and q.strip()][:6]
    if not cleaned:
        return []
    per_q = max(1, min(8, int(max_results_per_query)))
    sem = asyncio.Semaphore(5)
    headers = {"User-Agent": _USER_AGENT, "Accept-Language": "tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7"}

    async with httpx.AsyncClient(headers=headers, follow_redirects=True, timeout=11.0) as client:
        async def _one(q: str) -> Tuple[str, List[Dict[str, str]]]:
            async with sem:
                items = await _search_with_client(client, q, per_q)
                return q, items

        return list(await asyncio.gather(*(_one(q) for q in cleaned)))


def _assert_safe_public_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("Yalnızca geçerli http/https adresleri okunabilir.")
    host = parsed.hostname
    try:
        infos = socket.getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80))
    except socket.gaierror as exc:
        raise ValueError(f"Alan adı çözümlenemedi: {host}") from exc

    for info in infos:
        addr = info[4][0]
        ip = ipaddress.ip_address(addr)
        if ip.is_loopback or ip.is_private or ip.is_link_local or ip.is_multicast or ip.is_reserved:
            raise PermissionError(f"Yerel/özel ağ adreslerine ({host}) erişim güvenlik nedeniyle engellendi.")


async def read_webpage(url: str, *, focus_query: str = "", max_chars: int = 9_000) -> str:
    """Fetch a public webpage and convert it into clean, bounded Markdown."""
    _assert_safe_public_url(url)
    headers = {"User-Agent": _USER_AGENT, "Accept-Language": "tr-TR,tr;q=0.9,en;q=0.8"}
    async with httpx.AsyncClient(headers=headers, follow_redirects=True, timeout=14.0) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        final_url = str(resp.url)
        _assert_safe_public_url(final_url)
        ctype = resp.headers.get("content-type", "").lower()
        body = resp.text

    if "html" in ctype or "<html" in body[:500].lower():
        md = html_to_markdown(body, base_url=final_url)
    else:
        md = body.strip()

    if len(md) <= max_chars:
        return f"Kaynak: {final_url}\n\n{md}"

    if focus_query.strip():
        terms = [w.lower() for w in re.findall(r"\w{2,}", focus_query)]
        paras = [p.strip() for p in md.split("\n\n") if p.strip()]
        scored = sorted(
            ((sum(p.lower().count(t) for t in terms), idx, p) for idx, p in enumerate(paras)),
            key=lambda x: x[0],
            reverse=True,
        )
        top_indices = sorted({0, 1} | {idx for sc, idx, _ in scored[:10] if sc > 0})
        focused = "\n\n".join(paras[i] for i in top_indices if i < len(paras))
        if len(focused) >= 400:
            return f"Kaynak: {final_url} (odak: '{focus_query}')\n\n{focused[:max_chars]}"

    return f"Kaynak: {final_url} (ilk {max_chars} karakter)\n\n{md[:max_chars]}\n\n_(Sayfa devamı kısaltıldı.)_"


async def read_webpages_parallel(
    urls: List[str],
    *,
    focus_query: str = "",
    max_chars_each: int = 5_500,
) -> str:
    """Fetch and convert multiple public webpages to Markdown concurrently."""
    cleaned = [u.strip() for u in urls if u and u.strip()][:5]
    if not cleaned:
        return "Okunacak URL belirtilmedi."

    async def _safe_one(u: str) -> str:
        try:
            return await read_webpage(u, focus_query=focus_query, max_chars=max_chars_each)
        except Exception as exc:
            return f"Kaynak hatası ({u}): {exc}"

    parts = await asyncio.gather(*(_safe_one(u) for u in cleaned))
    return "\n\n---\n\n".join(parts)

