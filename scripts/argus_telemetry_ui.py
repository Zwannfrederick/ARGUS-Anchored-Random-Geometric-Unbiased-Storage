#!/usr/bin/env python3
"""ARGUS Live Telemetry & Monitoring Dashboard (Stage S10 / Waybar Integration).

Displays real-time:
- Operational State & Health Probes (OFF, STARTING, READY, ERROR)
- GPU VRAM Allocation, Utilization & Headroom
- KV Cache Tier Occupancy (ACTIVE FP16 vs q8_0 vs q4_0)
- Upstream llama-server connectivity
- Live Gateway Event Stream Logs
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from scripts.argus_service import LOG_FILE, RUNTIME_DIR, get_detailed_status

ENGINE_LOG_FILE = RUNTIME_DIR / "llama-server.log"


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


def _tail_lines(path: Path, count: int = 40) -> list[str]:
    if not path.exists():
        return []
    try:
        raw = subprocess.run(["tail", "-n", str(count), str(path)], capture_output=True, text=True, timeout=1.0).stdout
        return [ln for ln in raw.splitlines() if ln.strip()]
    except Exception:
        return []


def _style_log_line(prefix: str, line: str) -> str:
    safe = escape(line.strip())
    low = line.lower()
    if "❌" in line or "error" in low or "exception" in low or "traceback" in low:
        return f"[bold red]{prefix}{safe}[/]"
    if "📥" in line or "request:" in low or "argus p3" in low or "lcp similarity" in low:
        return f"[bold cyan]{prefix}{safe}[/]"
    if "print_timing" in low or "tokens per second" in low or "listening on" in low or "starting" in low or "ready" in low:
        return f"[bold green]{prefix}{safe}[/]"
    return f"[dim white]{prefix}{safe}[/]"


def create_logs_panel() -> Panel:
    gw_lines = [_style_log_line("[GW] ", ln) for ln in _tail_lines(LOG_FILE, 8)]
    eng_raw = [
        ln for ln in _tail_lines(ENGINE_LOG_FILE, 60)
        if not ln.startswith("create_tensor:") and not ln.startswith("llama_model_loader: - kv")
    ][-12:]
    eng_lines = [_style_log_line("[LLM] ", ln) for ln in eng_raw]

    sections: list[str] = []
    if gw_lines:
        sections.append("[bold cyan]── Claude Gateway (gateway.log) ──[/]\n" + "\n".join(gw_lines))
    if eng_lines:
        sections.append("[bold green]── ARGUS Engine (llama-server.log) ──[/]\n" + "\n".join(eng_lines))

    log_text = "\n\n".join(sections) if sections else "[dim]Henüz log kaydı yok (istek bekleniyor)...[/dim]"

    return Panel(
        Text.from_markup(log_text),
        title="[bold]Canlı Loglar (Studio & ARGUS llama-server)  •  Çıkış: Ctrl+C[/]",
        border_style="cyan",
    )


def make_layout(status: dict) -> Layout:
    layout = Layout()
    layout.split(
        Layout(create_header(status), name="header", size=3),
        Layout(create_system_table(status), name="main", size=11),
        Layout(create_logs_panel(), name="logs", ratio=1),
    )
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
