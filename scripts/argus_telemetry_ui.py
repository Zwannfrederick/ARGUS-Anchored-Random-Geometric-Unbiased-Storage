#!/usr/bin/env python3
"""ARGUS Live Telemetry & Monitoring Dashboard (Stage S10 / Waybar Integration).

Displays real-time:
- Operational State & Health Probes (OFF, STARTING, READY, ERROR)
- GPU VRAM Allocation, Utilization & Headroom
- KV Cache Tier Occupancy (ACTIVE FP16 vs q8_0 vs q4_0)
- Upstream llama-server / Ollama Backend Connectivity
- Live Gateway Event Stream Logs
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich.align import Align
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from scripts.argus_service import LOG_FILE, get_detailed_status


def create_header(status: dict) -> Panel:
    state = status["state"]
    if state == "READY":
        state_badge = "[bold white on green]  ● READY  [/]"
    elif state == "STARTING":
        state_badge = "[bold black on yellow]  ◐ STARTING  [/]"
    elif state == "ERROR":
        state_badge = "[bold white on red]  ! ERROR  [/]"
    else:
        state_badge = "[bold white on bright_black]  ● OFF  [/]"

    grid = Table.grid(expand=True)
    grid.add_column(justify="left", ratio=1)
    grid.add_column(justify="right", ratio=1)
    grid.add_row(
        Text("⚡ ARGUS HYBRID ATTENTION TELEMETRY", style="bold cyan"),
        Text.from_markup(f"Status: {state_badge}"),
    )
    return Panel(grid, style="cyan", border_style="cyan")


def create_system_table(status: dict) -> Panel:
    table = Table(box=None, expand=True, show_header=False)
    table.add_column("Metric", style="bold white", width=18)
    table.add_column("Value", style="yellow")

    table.add_row("Gateway URL", f"{status['gateway_url']} ({'🟢 Live' if status['gateway_ok'] else '🔴 Down'})")
    table.add_row("LLM Backend", f"{status['upstream_url']} ({'🟢 Connected' if status['backend_ok'] else '🟡 ' + status['backend_type']})")
    table.add_row("Active Model", status["model_name"])
    table.add_row(
        "KV Cache",
        f"{status.get('kv_cache_type', '?')} "
        f"(owned by {status.get('kv_managed_by', 'backend')})",
    )
    ctx = status.get("context_length")
    table.add_row("Context Length", f"{ctx:,}" if ctx else "[dim]no model loaded[/]")

    v = status["vram"]
    if v["available"]:
        pct = v["utilization_pct"]
        color = "green" if pct < 75 else "yellow" if pct < 90 else "red"
        bar_len = 20
        filled = int((pct / 100) * bar_len)
        bar = "█" * filled + "░" * (bar_len - filled)
        table.add_row("GPU Hardware", f"{v['gpu_name']}")
        table.add_row("VRAM Used", f"[{color}]{bar} {v['used_mib']} / {v['total_mib']} MiB ({pct}%)[/]")
    else:
        table.add_row("GPU Hardware", "[white]CPU Fallback Mode[/]")

    if status.get("error_message"):
        table.add_row("Notice", f"[bold red]⚠️ {status['error_message']}[/]")

    return Panel(table, title="[bold]Runtime Metrics[/]", border_style="blue")


CACHE_STATS_FILE = Path.home() / ".argus_runtime" / "cache_stats.json"


def get_live_cache_stats() -> dict:
    if CACHE_STATS_FILE.exists():
        try:
            return json.loads(CACHE_STATS_FILE.read_text())
        except (OSError, json.JSONDecodeError):
            pass
    return {
        "state": "READY",
        "backend": None,
        "model": None,
        "argus_managed": False,
        "input_tokens": None,
        "output_tokens": None,
        "residency": None,
    }


def _fmt_gib(value) -> str:
    """Bytes as GiB, or an explicit dash when the figure was never measured."""
    if not isinstance(value, (int, float)):
        return "[dim]—[/]"
    return f"{value / (1024 ** 3):.2f} GiB"


def _fmt_count(value) -> str:
    """A count, distinguishing a measured zero from an absent measurement."""
    if not isinstance(value, int):
        return "[dim]—[/]"
    return f"{value:,}"


def create_tier_table() -> Panel:
    """Render what the backend actually reported.

    This panel used to show a T0/T1/T2 page cascade and a "% Memory Saved"
    figure. None of it was measured: the gateway proxies to an external
    runtime that owns its KV cache in another process, so ARGUS places no
    pages there and can observe none. Those rows were computed from the token
    count and fixed byte constants, which is why they are gone rather than
    zeroed -- a zeroed tier still asserts the tier exists.
    """
    stats = get_live_cache_stats()
    residency = stats.get("residency") or {}

    table = Table(expand=True, show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan", width=22)
    table.add_column("Value", justify="right", style="green")
    table.add_column("Source", style="dim")

    table.add_row("Backend", stats.get("backend") or "[dim]—[/]", "gateway")
    table.add_row("Model", stats.get("model") or "[dim]—[/]", "request")
    table.add_row("Input tokens", _fmt_count(stats.get("input_tokens")), "backend usage")
    table.add_row("Output tokens", _fmt_count(stats.get("output_tokens")), "backend usage")

    if residency:
        gpu_pct = residency.get("gpu_fraction", 0.0) * 100
        table.add_row("Model size", _fmt_gib(residency.get("total_bytes")), "ollama /api/ps")
        table.add_row(
            "Resident on GPU",
            f"{_fmt_gib(residency.get('vram_bytes'))} ({gpu_pct:.0f}%)",
            "ollama /api/ps",
        )
        table.add_row("Resident on CPU", _fmt_gib(residency.get("cpu_bytes")), "ollama /api/ps")
        table.add_row("Context length", _fmt_count(residency.get("context_length")), "ollama /api/ps")
    else:
        table.add_row("Residency", "[dim]model not loaded[/]", "ollama /api/ps")

    managed = stats.get("argus_managed", False)
    note = (
        "[green]ARGUS-managed KV cache[/]"
        if managed
        else "[yellow]KV cache owned by the backend process — not ARGUS-managed[/]"
    )
    title = f"[bold]Backend Telemetry[/]  [dim]•[/]  {note}"
    return Panel(table, title=title, border_style="magenta")


def create_logs_panel() -> Panel:
    if not LOG_FILE.exists():
        log_text = "[dim]No gateway logs recorded yet.[/dim]"
    else:
        try:
            raw = subprocess.run(["tail", "-n", "8", str(LOG_FILE)], capture_output=True, text=True).stdout
            lines = raw.strip().split("\n") if raw.strip() else []
            formatted_lines = []
            for l in lines:
                if "❌" in l or "error" in l.lower():
                    formatted_lines.append(f"[bold red]{l}[/]")
                elif "📥" in l or "Request:" in l:
                    formatted_lines.append(f"[bold cyan]{l}[/]")
                elif "Starting" in l or "READY" in l:
                    formatted_lines.append(f"[bold green]{l}[/]")
                else:
                    formatted_lines.append(f"[dim white]{l}[/]")
            log_text = "\n".join(formatted_lines) or "[dim]Waiting for traffic...[/dim]"
        except Exception:
            log_text = "[dim]Error reading log file[/dim]"

    return Panel(
        Text.from_markup(log_text),
        title="[bold]Recent Gateway Logs (Active Session)[/]",
        border_style="bright_black",
    )


def make_layout(status: dict) -> Layout:
    layout = Layout()
    layout.split(
        Layout(name="header", size=3),
        Layout(name="main", size=10),
        Layout(name="tiers", size=9),
        Layout(name="logs", ratio=1),
    )
    layout["header"].update(create_header(status))
    layout["main"].update(create_system_table(status))
    layout["tiers"].update(create_tier_table())
    layout["logs"].update(create_logs_panel())
    return layout


def main():
    console = Console()
    console.clear()

    with Live(console=console, screen=True, refresh_per_second=2) as live:
        try:
            while True:
                st = get_detailed_status()
                live.update(make_layout(st))
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
