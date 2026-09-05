"""Bridge from Neo's twelve canonical tools to hermes-agent's tool registry.

hermes-agent already ships ~83 registered tools (browser automation, web search,
vision, file editing) that Neo has no equivalent of. They cannot simply be appended
to Neo's tool list: the canonical prefix is what keeps the KV cache at 99%+, and
83 schemas would grow it roughly sevenfold. So the registry is reached through two
dispatcher tools instead -- search the catalogue, then call one by name -- which
costs two schemas in the prefix and leaves the rest out of context until used.

Excluded on purpose:
  desktop_ui   - hermes-agent's own TUI panes, meaningless outside its CLI.
  computer_use - cua-driver returns no image and sees only the frame for Electron
                 apps on this box; Neo's Wayland stack (wayland_ui) is the path
                 that actually works here.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

AGENT_ROOT = Path(__file__).resolve().parent / "hermes-agent"

# Neo owns these capabilities with a working implementation; hermes-agent's versions
# are either CLI-bound or broken on this machine.
EXCLUDED_TOOLSETS = frozenset({"desktop_ui", "computer_use"})

# Reached through hermes_skill, these run code or touch the filesystem, so they go
# through the same approval gate as execute_terminal_command.
HIGH_RISK_TOOLS = frozenset({
    "terminal", "execute_code", "write_file", "patch", "process_manage", "delegate_task",
})

_lock = threading.Lock()
_catalog: Optional[Dict[str, Dict[str, str]]] = None
_registry: Any = None
_load_error: Optional[str] = None


def _load() -> Tuple[Optional[Dict[str, Dict[str, str]]], Optional[str]]:
    """Import hermes-agent once and snapshot the tools usable on this box.

    Availability is resolved here rather than per call: the check functions shell out
    (is a browser up? is a token set?) and hermes-agent memoizes them per turn anyway.
    """
    global _catalog, _registry, _load_error
    with _lock:
        if _catalog is not None or _load_error is not None:
            return _catalog, _load_error
        try:
            if str(AGENT_ROOT) not in sys.path:
                sys.path.insert(0, str(AGENT_ROOT))
            from tools import registry as agent_registry  # type: ignore

            agent_registry.discover_builtin_tools()
            reg = agent_registry.registry
            toolset_ok = reg.check_toolset_requirements()
            catalog: Dict[str, Dict[str, str]] = {}
            for entry in reg.get_all_entries():
                if entry.toolset in EXCLUDED_TOOLSETS or not toolset_ok.get(entry.toolset):
                    continue
                catalog[entry.name] = {
                    "toolset": entry.toolset,
                    "description": (entry.description or "").strip(),
                }
            _registry, _catalog = reg, catalog
        except Exception as exc:  # noqa: BLE001 - surfaced to the model, never raised
            _load_error = f"{type(exc).__name__}: {exc}"
        return _catalog, _load_error


def search(query: str, limit: int = 12) -> Dict[str, Any]:
    """Return catalogue entries matching *query*, or the whole catalogue when empty."""
    catalog, err = _load()
    if err:
        return {"error": f"hermes-agent arac kataloğu yuklenemedi: {err}"}

    terms = [t for t in (query or "").lower().split() if t]
    if not terms:
        hits = sorted(catalog.items())
    else:
        scored = []
        for name, meta in catalog.items():
            haystack = f"{name} {meta['toolset']} {meta['description']}".lower()
            score = sum(2 if t in name.lower() else 1 for t in terms if t in haystack)
            if score:
                scored.append((-score, name, meta))
        hits = [(n, m) for _, n, m in sorted(scored)]

    matches = [
        {"name": n, "toolset": m["toolset"], "description": m["description"][:240]}
        for n, m in hits[:limit]
    ]
    # Searching for an exact tool name means "how do I call this one", so answer that
    # in the same round trip rather than making the model ask again.
    if (query or "").strip() in catalog:
        matches[0]["parameters"] = describe(query.strip())["parameters"]
    return {
        "total_available": len(catalog),
        "matches": matches,
        "note": "Calistirmak icin hermes_skill(name=..., arguments={...}) kullan. "
                "Argumanlari bilmiyorsan once tam adi arayip `parameters` alanina bak.",
    }


def describe(name: str) -> Dict[str, Any]:
    """Full JSON schema for one tool, so its arguments can be filled in correctly."""
    catalog, err = _load()
    if err:
        return {"error": err}
    if name not in catalog:
        return {"error": f"'{name}' yok veya bu makinede kullanilamiyor. hermes_skill_search ile ara."}
    # get_definitions returns the OpenAI-wrapped form (with parameters); get_schema
    # returns a bare function object whose parameters live at the top level.
    defs = _registry.get_definitions({name}, quiet=True)
    fn = (defs[0].get("function") if defs else None) or {}
    return {
        "name": name,
        "toolset": catalog[name]["toolset"],
        "description": fn.get("description", ""),
        "parameters": fn.get("parameters", {}),
    }


async def dispatch(name: str, arguments: Dict[str, Any]) -> Any:
    """Run one hermes-agent tool and return its result as parsed JSON when possible.

    Dispatch is synchronous and may spin its own event loop for async handlers, so it
    is kept off Neo's loop entirely.
    """
    catalog, err = _load()
    if err:
        return {"error": f"hermes-agent yuklenemedi: {err}"}
    if name not in catalog:
        hint = search(name, limit=5).get("matches", [])
        return {
            "error": f"'{name}' kayitli degil veya bu makinede kullanilamiyor.",
            "did_you_mean": [h["name"] for h in hint],
        }
    if not isinstance(arguments, dict):
        return {"error": "arguments bir JSON nesnesi olmali."}

    raw = await asyncio.to_thread(_registry.dispatch, name, arguments)
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except ValueError:
            return {"result": raw}
    return raw


def demo() -> None:
    catalog, err = _load()
    assert err is None, err
    assert catalog, "catalogue must not be empty"
    assert not any(m["toolset"] in EXCLUDED_TOOLSETS for m in catalog.values())

    hits = search("web arama")["matches"]
    assert any(h["name"] == "web_search" for h in hits), [h["name"] for h in hits]

    # A name match must outrank a description-only match.
    exact = search("browser_navigate")["matches"][0]
    assert exact["name"] == "browser_navigate"
    assert "parameters" in exact, "an exact name match must carry the call signature"

    assert describe("web_search")["parameters"].get("properties"), describe("web_search")
    assert "error" in describe("definitely_not_a_tool")

    missing = asyncio.run(dispatch("definitely_not_a_tool", {}))
    assert "error" in missing and "did_you_mean" in missing

    # Real dispatch, no network: a read of this very file.
    out = asyncio.run(dispatch("read_file", {"path": __file__}))
    assert "hermes-agent's tool registry" in json.dumps(out, ensure_ascii=False), out
    print(f"OK: {len(catalog)} tools bridged, dispatch verified")


if __name__ == "__main__":
    demo()
