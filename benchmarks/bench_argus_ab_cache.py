#!/usr/bin/env python3
"""
Controlled A/B Benchmark: ARGUS KV/Cache Layer vs. Native Vanilla Inference
===========================================================================
Measures whether the ARGUS KV/cache layer introduces overhead on inference throughput
across short (4K), medium (16K), and long (32K, 64K) context coding workloads.

A/B Modes:
  Mode A: ARGUS ON (Production ARGUS service entrypoint on port 8008,
          translating messages, tracking SSE tokens & residency)
  Mode B: ARGUS BYPASS / VANILLA (Direct llama-server native KV inference on port 8080)

Fixed Config:
  Model: Qwen3.6-35B-A3B-UD-Q4_K_M-MTP.gguf
  spec_draft_n_max: 2
  spec_draft_p_min: 0.0
  load-mode: none
  Flash Attention: ON (-fa on)
  KV cache: q4_0 / q4_0
  Seed: 42, Temperature: 0.2
  Context Sweep: 4K (3.5K prompt + 512 gen) -> 16K -> 32K -> 64K
  Interleaving: A -> B -> A -> B -> A -> B per context level.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

GATEWAY_URL = "http://127.0.0.1:8008"
UPSTREAM_URL = "http://127.0.0.1:8080"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "measurements"

CODING_TASK = (
    "\n# Task: Write a thread-safe, high-performance distributed LRU cache manager in Python "
    "with doubly-linked lists, lock-free reads, and TTL expiration. Include docstrings and full typing."
)

BASE_SNIPPET = (
    "class PagedBufferNode:\n"
    "    def __init__(self, node_id: int, size: int):\n"
    "        self.node_id = node_id\n"
    "        self.size = size\n"
    "        self.data = bytearray(size)\n"
    "    def clear(self) -> None:\n"
    "        self.data = bytearray(self.size)\n"
    "    def slice(self, start: int, end: int) -> bytes:\n"
    "        return bytes(self.data[start:end])\n\n"
)


# ==============================================================================
# Telemetry Helpers
# ==============================================================================

def get_gpu_telemetry() -> Dict[str, Any]:
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=memory.used,memory.total,temperature.gpu,clocks.current.graphics,power.draw",
            "--format=csv,noheader,nounits",
        ]
        res = subprocess.check_output(cmd, encoding="utf-8").strip().split(",")
        return {
            "vram_used_mb": float(res[0].strip()),
            "vram_total_mb": float(res[1].strip()),
            "gpu_temp_c": float(res[2].strip()),
            "gpu_clock_mhz": float(res[3].strip()),
            "gpu_power_w": float(res[4].strip()),
        }
    except Exception:
        return {
            "vram_used_mb": 0.0,
            "vram_total_mb": 4096.0,
            "gpu_temp_c": 0.0,
            "gpu_clock_mhz": 0.0,
            "gpu_power_w": 0.0,
        }


def get_system_telemetry(pid: Optional[int]) -> Dict[str, Any]:
    # Memory
    ram_used_mb = 0.0
    swap_used_mb = 0.0
    try:
        with open("/proc/meminfo") as f:
            lines = {line.split(":")[0]: int(line.split(":")[1].split()[0]) for line in f}
            ram_used_mb = (lines.get("MemTotal", 0) - lines.get("MemAvailable", 0)) / 1024.0
            swap_used_mb = (lines.get("SwapTotal", 0) - lines.get("SwapFree", 0)) / 1024.0
    except Exception:
        pass

    # CPU Freq
    freqs = []
    for p in glob.glob("/sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq"):
        try:
            with open(p) as f:
                freqs.append(int(f.read().strip()) / 1000.0)
        except Exception:
            pass
    avg_mhz = sum(freqs) / len(freqs) if freqs else 0.0

    # Page Faults
    min_flt = 0
    maj_flt = 0
    if pid and os.path.exists(f"/proc/{pid}/stat"):
        try:
            with open(f"/proc/{pid}/stat") as f:
                fields = f.read().split()
                min_flt = int(fields[9])
                maj_flt = int(fields[11])
        except Exception:
            pass

    return {
        "ram_used_mb": round(ram_used_mb, 1),
        "swap_used_mb": round(swap_used_mb, 1),
        "cpu_avg_mhz": round(avg_mhz, 1),
        "minor_faults": min_flt,
        "major_faults": maj_flt,
    }


# ==============================================================================
# Process & Architecture Audit
# ==============================================================================

def run_process_audit(client: httpx.Client) -> Dict[str, Any]:
    print("=" * 80)
    print("               PRE-FLIGHT PROCESS & ARCHITECTURE AUDIT")
    print("=" * 80)

    # 1. Find llama-server PID
    llama_pid = None
    try:
        pid_out = subprocess.check_output(["pgrep", "-f", "llama-server"], encoding="utf-8").strip()
        llama_pid = int(pid_out.splitlines()[0])
    except Exception:
        pass

    # 2. Find gateway PID
    gateway_pid = None
    try:
        pid_out = subprocess.check_output(["pgrep", "-f", "claude_gateway"], encoding="utf-8").strip()
        gateway_pid = int(pid_out.splitlines()[0])
    except Exception:
        pass

    print(f"[*] llama-server PID : {llama_pid}")
    print(f"[*] claude_gateway PID: {gateway_pid}")

    # Read cmdline of llama-server
    llama_cmdline = ""
    if llama_pid and os.path.exists(f"/proc/{llama_pid}/cmdline"):
        with open(f"/proc/{llama_pid}/cmdline", "rb") as f:
            llama_cmdline = f.read().replace(b"\x00", b" ").decode("utf-8", errors="replace").strip()
    print(f"[*] llama-server Cmd : {llama_cmdline[:110]}...")

    # Check /proc/<pid>/maps for ARGUS libraries
    argus_maps = []
    if llama_pid and os.path.exists(f"/proc/{llama_pid}/maps"):
        with open(f"/proc/{llama_pid}/maps", "r") as f:
            for line in f:
                if "argus" in line.lower():
                    argus_maps.append(line.strip())

    has_argus_in_llama = len(argus_maps) > 0
    print(f"[*] ARGUS C++ lib mapped in llama-server : {has_argus_in_llama} (Found {len(argus_maps)} mappings)")

    # Read gateway status
    gw_status = "UNKNOWN"
    try:
        r = client.get(f"{GATEWAY_URL}/health", timeout=5.0)
        gw_status = "ONLINE" if r.status_code == 200 else f"HTTP {r.status_code}"
    except Exception as e:
        gw_status = f"UNREACHABLE ({e})"
    print(f"[*] ARGUS Gateway Health: {gw_status}")

    audit_result = {
        "llama_pid": llama_pid,
        "gateway_pid": gateway_pid,
        "llama_cmdline": llama_cmdline,
        "argus_in_llama_server_maps": has_argus_in_llama,
        "argus_maps_count": len(argus_maps),
        "gateway_status": gw_status,
        "verified_at": datetime.now().isoformat(),
    }
    print("=" * 80)
    return audit_result


# ==============================================================================
# Prompt Construction
# ==============================================================================

def generate_context_prompt(client: httpx.Client, target_tokens: int) -> Tuple[str, int]:
    """Generates a realistic code prompt prefix tokenized to approximately target_tokens."""
    # Each repeat of BASE_SNIPPET is approx 89.5 tokens
    repeats = max(1, int(target_tokens / 89.5))
    prefix = BASE_SNIPPET * repeats
    full_text = prefix + CODING_TASK

    # Verify exact tokens via /tokenize
    try:
        r = client.post(f"{UPSTREAM_URL}/tokenize", json={"content": full_text}, timeout=30.0)
        actual_tokens = len(r.json().get("tokens", []))
    except Exception:
        actual_tokens = target_tokens

    return full_text, actual_tokens


# ==============================================================================
# Run Execution (Mode A vs Mode B)
# ==============================================================================

def execute_request(
    client: httpx.Client,
    mode: str,
    prompt: str,
    max_tokens: int,
    llama_pid: Optional[int],
    timeout_s: float = 1200.0,
) -> Dict[str, Any]:
    """
    Executes a single inference request:
      - mode == 'ARGUS_ON': routes via Gateway (port 8008, /v1/messages, Anthropic format)
      - mode == 'BYPASS': routes directly to llama-server (port 8080, /v1/chat/completions, OpenAI format)
    """
    gpu_before = get_gpu_telemetry()
    sys_before = get_system_telemetry(llama_pid)

    t0 = time.perf_counter()

    if mode == "ARGUS_ON":
        # Mode A: ARGUS Claude Gateway with SSE streaming to isolate TTFT & Decode TPS
        payload = {
            "model": "qwen3.6",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.2,
            "seed": 42,
            "stream": True,
        }
        url = f"{GATEWAY_URL}/v1/messages"
        headers = {"Content-Type": "application/json", "x-api-key": "test"}

        t_first = None
        output_tokens = 0
        input_tokens = 0
        raw_events = []

        try:
            with client.stream("POST", url, json=payload, headers=headers, timeout=timeout_s) as response:
                if response.status_code != 200:
                    return {
                        "success": False,
                        "error": f"HTTP {response.status_code}",
                        "total_time_s": time.perf_counter() - t0,
                        "stability": f"FAIL_HTTP_{response.status_code}",
                    }

                for line in response.iter_lines():
                    if line.startswith("data:"):
                        if t_first is None:
                            t_first = time.perf_counter()
                        data_str = line[5:].strip()
                        if data_str and data_str != "[DONE]":
                            try:
                                evt = json.loads(data_str)
                                e_type = evt.get("type")
                                if e_type == "content_block_delta":
                                    output_tokens += 1
                                elif e_type == "message_start":
                                    input_tokens = evt.get("message", {}).get("usage", {}).get("input_tokens", 0)
                                elif e_type == "message_delta":
                                    u = evt.get("usage", {})
                                    if "output_tokens" in u:
                                        output_tokens = u["output_tokens"]
                            except Exception:
                                pass

            t_end = time.perf_counter()
            dt_total = t_end - t0
            prompt_ms = ((t_first - t0) * 1000.0) if t_first else 0.0
            decode_time_s = (t_end - t_first) if t_first else dt_total
            decode_tps = (output_tokens / decode_time_s) if decode_time_s > 0 else 0.0
            prompt_tps = (input_tokens / (prompt_ms / 1000.0)) if prompt_ms > 0 else 0.0

            prompt_tokens = input_tokens
            completion_tokens = output_tokens
            decode_ms = decode_time_s * 1000.0
            draft_n = None
            draft_accepted = None
            acceptance_pct = None

        except httpx.ReadTimeout:
            return {
                "success": False,
                "error": "ReadTimeout after 1200s (capacity limit / OOM)",
                "total_time_s": time.perf_counter() - t0,
                "stability": "TIMEOUT_CAPACITY_LIMIT",
            }
        except Exception as exc:
            return {
                "success": False,
                "error": str(exc),
                "total_time_s": time.perf_counter() - t0,
                "stability": "CRASH_OR_OOM",
            }
    else:
        # Mode B: Direct llama-server Vanilla Bypass (Non-streaming to get exact hardware timings)
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.2,
            "seed": 42,
        }
        url = f"{UPSTREAM_URL}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}

        try:
            resp = client.post(url, json=payload, headers=headers, timeout=timeout_s)
            dt_total = time.perf_counter() - t0

            if resp.status_code != 200:
                return {
                    "success": False,
                    "error": f"HTTP {resp.status_code}: {resp.text[:200]}",
                    "total_time_s": dt_total,
                    "stability": "FAIL_HTTP",
                }

            data = resp.json()
        except httpx.ReadTimeout:
            return {
                "success": False,
                "error": "ReadTimeout after 1200s (vanilla capacity limit / OOM)",
                "total_time_s": time.perf_counter() - t0,
                "stability": "TIMEOUT_CAPACITY_LIMIT",
            }
        except Exception as exc:
            return {
                "success": False,
                "error": str(exc),
                "total_time_s": time.perf_counter() - t0,
                "stability": "CRASH_OR_OOM",
            }

        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        timings = data.get("timings", {})

        prompt_ms = timings.get("prompt_ms", 0.0)
        prompt_tps = timings.get("prompt_per_second", 0.0)
        decode_ms = timings.get("predicted_ms", dt_total * 1000.0)
        decode_tps = timings.get("predicted_per_second", (completion_tokens / dt_total) if dt_total > 0 else 0.0)
        draft_n = timings.get("draft_n")
        draft_accepted = timings.get("draft_n_accepted")
        acceptance_pct = (
            round((draft_accepted / draft_n) * 100.0, 2)
            if (draft_n is not None and draft_accepted is not None and draft_n > 0)
            else None
        )
    gpu_after = get_gpu_telemetry()
    sys_after = get_system_telemetry(llama_pid)

    return {
        "success": True,
        "stability": "STABLE",
        "total_time_s": round(dt_total, 3),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "prompt_ms": round(prompt_ms, 2),
        "prompt_tps": round(prompt_tps, 2),
        "decode_ms": round(decode_ms, 2),
        "decode_tps": round(decode_tps, 2),
        "draft_n": draft_n,
        "draft_accepted": draft_accepted,
        "acceptance_pct": acceptance_pct,
        "gpu_vram_mb": gpu_after["vram_used_mb"],
        "gpu_temp_c": gpu_after["gpu_temp_c"],
        "gpu_clock_mhz": gpu_after["gpu_clock_mhz"],
        "gpu_power_w": gpu_after["gpu_power_w"],
        "ram_used_mb": sys_after["ram_used_mb"],
        "swap_used_mb": sys_after["swap_used_mb"],
        "cpu_avg_mhz": sys_after["cpu_avg_mhz"],
        "minor_faults_delta": sys_after["minor_faults"] - sys_before["minor_faults"],
        "major_faults_delta": sys_after["major_faults"] - sys_before["major_faults"],
    }


# ==============================================================================
# Main Benchmark Orchestrator
# ==============================================================================

def run_benchmark(
    target_contexts: Optional[List[str]] = None,
    max_tokens: int = 512,
    runs_per_mode: int = 3,
    smoke_test: bool = False,
) -> None:
    client = httpx.Client(timeout=1200.0)
    audit = run_process_audit(client)
    llama_pid = audit.get("llama_pid")

    if smoke_test:
        context_targets = [("SMOKE", 128)]
        max_tokens = 32
        runs_per_mode = 1
    else:
        all_targets = [
            ("4K", 3500),
            ("16K", 15500),
            ("32K", 31500),
            ("64K", 63000),
        ]
        if target_contexts:
            context_targets = [t for t in all_targets if t[0] in target_contexts]
        else:
            context_targets = all_targets

    results_by_ctx: Dict[str, Any] = {}

    print(f"\nStarting Context Sweep: {[t[0] for t in context_targets]}")
    print(f"Protocol: Deep Warmup -> {runs_per_mode}x{max_tokens} tokens interleaved (A -> B -> A -> B...)")
    print("-" * 80)

    for ctx_label, target_tokens in context_targets:
        print(f"\n[{ctx_label}] Context Target: Generating prompt ~{target_tokens} tokens...")
        prompt_text, actual_tokens = generate_context_prompt(client, target_tokens)
        print(f"[{ctx_label}] Actual Prompt Tokens: {actual_tokens} (Prefix + LRU Coding Task)")

        # ----------------------------------------------------------------------
        # Deep Warmup Run (Mode B direct to establish prompt cache & hot weights)
        # ----------------------------------------------------------------------
        warmup_tokens = 16 if smoke_test else 64
        print(f"[{ctx_label}] Deep Warmup in progress ({warmup_tokens} tokens, warming KV cache & MoE weights)...", end="", flush=True)
        t_w0 = time.perf_counter()
        w_res = execute_request(client, "BYPASS", prompt_text, warmup_tokens, llama_pid, timeout_s=900.0)
        t_warmup = time.perf_counter() - t_w0
        if not w_res["success"]:
            print(f" ❌ WARMUP FAILED: {w_res.get('error')}")
            results_by_ctx[ctx_label] = {
                "actual_prompt_tokens": actual_tokens,
                "warmup_error": w_res.get("error"),
                "stability": "CAPACITY_LIMIT_WARMUP",
            }
            continue
        else:
            print(f" ✅ Done ({t_warmup:.1f}s | Prompt: {w_res.get('prompt_tps')} tok/s | Decode: {w_res.get('decode_tps')} tok/s)")

        # ----------------------------------------------------------------------
        # Interleaved Runs
        # ----------------------------------------------------------------------
        runs_a: List[Dict[str, Any]] = []
        runs_b: List[Dict[str, Any]] = []

        for run_idx in range(1, runs_per_mode + 1):
            print(f"\n  [Run {run_idx}/{runs_per_mode}] Interleaved Pair:")

            # Mode A: ARGUS ON (Port 8008)
            print(f"    -> [A: ARGUS ON] {max_tokens} tokens via Gateway...", end="", flush=True)
            res_a = execute_request(client, "ARGUS_ON", prompt_text, max_tokens, llama_pid)
            if res_a["success"]:
                runs_a.append(res_a)
                print(f" ✅ {res_a['decode_tps']:.2f} tok/s ({res_a['total_time_s']:.1f}s | TTFT: {res_a['prompt_ms']:.0f}ms | VRAM: {res_a['gpu_vram_mb']}MB)")
            else:
                print(f" ❌ FAILED: {res_a.get('error')}")
                runs_a.append(res_a)

            time.sleep(1.0)  # Brief pause between modes

            # Mode B: ARGUS BYPASS (Port 8080)
            print(f"    -> [B: BYPASS]   {max_tokens} tokens direct upstream...", end="", flush=True)
            res_b = execute_request(client, "BYPASS", prompt_text, max_tokens, llama_pid)
            if res_b["success"]:
                runs_b.append(res_b)
                acc_str = f" | Acc: {res_b['acceptance_pct']}%" if res_b['acceptance_pct'] is not None else ""
                print(f" ✅ {res_b['decode_tps']:.2f} tok/s ({res_b['total_time_s']:.1f}s{acc_str} | TTFT: {res_b['prompt_ms']:.0f}ms | VRAM: {res_b['gpu_vram_mb']}MB)")
            else:
                print(f" ❌ FAILED: {res_b.get('error')}")
                runs_b.append(res_b)

            time.sleep(1.0)

        # ----------------------------------------------------------------------
        # Aggregate Context Results
        # ----------------------------------------------------------------------
        successful_a = [r for r in runs_a if r["success"]]
        successful_b = [r for r in runs_b if r["success"]]

        median_a_tps = statistics.median([r["decode_tps"] for r in successful_a]) if successful_a else None
        median_b_tps = statistics.median([r["decode_tps"] for r in successful_b]) if successful_b else None

        delta_pct = (
            round(((median_a_tps - median_b_tps) / median_b_tps) * 100.0, 2)
            if (median_a_tps is not None and median_b_tps is not None and median_b_tps > 0)
            else None
        )

        results_by_ctx[ctx_label] = {
            "actual_prompt_tokens": actual_tokens,
            "warmup_time_s": round(t_warmup, 2),
            "mode_a_argus": {
                "median_decode_tps": median_a_tps,
                "min_decode_tps": min([r["decode_tps"] for r in successful_a]) if successful_a else None,
                "max_decode_tps": max([r["decode_tps"] for r in successful_a]) if successful_a else None,
                "median_ttft_ms": statistics.median([r["prompt_ms"] for r in successful_a]) if successful_a else None,
                "median_total_time_s": statistics.median([r["total_time_s"] for r in successful_a]) if successful_a else None,
                "vram_mb": successful_a[-1]["gpu_vram_mb"] if successful_a else None,
                "ram_mb": successful_a[-1]["ram_used_mb"] if successful_a else None,
                "stability": "STABLE" if len(successful_a) == runs_per_mode else f"{len(successful_a)}/{runs_per_mode}_PASS",
                "raw_runs": runs_a,
            },
            "mode_b_vanilla": {
                "median_decode_tps": median_b_tps,
                "min_decode_tps": min([r["decode_tps"] for r in successful_b]) if successful_b else None,
                "max_decode_tps": max([r["decode_tps"] for r in successful_b]) if successful_b else None,
                "median_prompt_ms": statistics.median([r["prompt_ms"] for r in successful_b]) if successful_b else None,
                "median_acceptance_pct": statistics.median([r["acceptance_pct"] for r in successful_b if r["acceptance_pct"] is not None]) if any(r["acceptance_pct"] is not None for r in successful_b) else None,
                "vram_mb": successful_b[-1]["gpu_vram_mb"] if successful_b else None,
                "ram_mb": successful_b[-1]["ram_used_mb"] if successful_b else None,
                "stability": "STABLE" if len(successful_b) == runs_per_mode else f"{len(successful_b)}/{runs_per_mode}_PASS",
                "raw_runs": runs_b,
            },
            "delta_pct_vs_vanilla": delta_pct,
        }

        # Incremental save after each context level completes
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        report_file = OUTPUT_DIR / "argus-ab-cache-comparison-2026-09-04.json"
        partial_report = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "hardware": "NVIDIA GeForce RTX 3050 Ti Laptop GPU (4 GB VRAM) + Intel Core i5-11300H + 32 GB RAM",
            "model": "Qwen3.6-35B-A3B-UD-Q4_K_M-MTP.gguf",
            "fixed_config": {
                "spec_draft_n_max": 2,
                "spec_draft_p_min": 0.0,
                "load_mode": "none",
                "flash_attn": "on",
                "kv_cache_type": "q4_0",
                "num_ctx": 65536,
                "seed": 42,
                "temperature": 0.2,
            },
            "audit": audit,
            "context_sweep_results": results_by_ctx,
        }
        report_file.write_text(json.dumps(partial_report, indent=2), encoding="utf-8")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_file = OUTPUT_DIR / "argus-ab-cache-comparison-2026-09-04.json"

    full_report = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": "NVIDIA GeForce RTX 3050 Ti Laptop GPU (4 GB VRAM) + Intel Core i5-11300H + 32 GB RAM",
        "model": "Qwen3.6-35B-A3B-UD-Q4_K_M-MTP.gguf",
        "fixed_config": {
            "spec_draft_n_max": 2,
            "spec_draft_p_min": 0.0,
            "load_mode": "none",
            "flash_attn": "on",
            "kv_cache_type": "q4_0",
            "num_ctx": 65536,
            "seed": 42,
            "temperature": 0.2,
        },
        "audit": audit,
        "context_sweep_results": results_by_ctx,
    }

    report_file.write_text(json.dumps(full_report, indent=2), encoding="utf-8")
    print(f"\n[+] Ham sonuçlar kaydedildi: {report_file}")

    # ==========================================================================
    # Render Scoreboard Table
    # ==========================================================================
    print("\n" + "=" * 105)
    print("                                   FINAL A/B BENCHMARK SCOREBOARD")
    print("=" * 105)
    print(f"{'Context':<8} | {'ARGUS mode':<14} | {'Median tok/s':<13} | {'Δ vs vanilla':<13} | {'TTFT':<10} | {'VRAM':<10} | {'RAM':<10} | {'Acceptance':<11} | {'Stability':<10}")
    print("-" * 105)

    for ctx_label, data in results_by_ctx.items():
        if "warmup_error" in data:
            print(f"{ctx_label:<8} | {'VANILLA/ARGUS':<14} | {'N/A':<13} | {'N/A':<13} | {'N/A':<10} | {'N/A':<10} | {'N/A':<10} | {'N/A':<11} | {'CAPACITY LIMIT':<10}")
            continue

        a_data = data["mode_a_argus"]
        b_data = data["mode_b_vanilla"]
        delta = data["delta_pct_vs_vanilla"]
        delta_str = f"{delta:+.1f}%" if delta is not None else "N/A"

        # Mode A Row
        ttft_a = f"{a_data['median_ttft_ms']:.0f}ms" if a_data.get('median_ttft_ms') else "N/A"
        vram_a = f"{a_data['vram_mb']:.0f} MB" if a_data['vram_mb'] else "N/A"
        ram_a = f"{a_data['ram_mb']:.0f} MB" if a_data['ram_mb'] else "N/A"
        tps_a = f"{a_data['median_decode_tps']:.2f}" if a_data['median_decode_tps'] else "N/A"

        print(f"{ctx_label:<8} | {'A (ARGUS ON)':<14} | {tps_a:<13} | {delta_str:<13} | {ttft_a:<10} | {vram_a:<10} | {ram_a:<10} | {'-':<11} | {a_data['stability']:<10}")

        # Mode B Row
        ttft_b = f"{b_data['median_prompt_ms']:.0f}ms" if b_data['median_prompt_ms'] else "N/A"
        vram_b = f"{b_data['vram_mb']:.0f} MB" if b_data['vram_mb'] else "N/A"
        ram_b = f"{b_data['ram_mb']:.0f} MB" if b_data['ram_mb'] else "N/A"
        tps_b = f"{b_data['median_decode_tps']:.2f}" if b_data['median_decode_tps'] else "N/A"
        acc_b = f"{b_data['median_acceptance_pct']:.1f}%" if b_data['median_acceptance_pct'] else "N/A"

        print(f"{'':<8} | {'B (VANILLA)':<14} | {tps_b:<13} | {'[BASELINE]':<13} | {ttft_b:<10} | {vram_b:<10} | {ram_b:<10} | {acc_b:<11} | {b_data['stability']:<10}")
        print("-" * 105)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A/B Benchmark: ARGUS KV Cache vs Vanilla")
    parser.add_argument("--smoke-test", action="store_true", help="Run quick single-sample smoke test")
    parser.add_argument("--contexts", nargs="+", default=None, help="Contexts to run (e.g. 4K 16K 32K 64K)")
    parser.add_argument("--max-tokens", type=int, default=512, help="Output tokens per run")
    parser.add_argument("--runs", type=int, default=3, help="Runs per mode per context")
    args = parser.parse_args()

    run_benchmark(
        target_contexts=args.contexts,
        max_tokens=args.max_tokens,
        runs_per_mode=args.runs,
        smoke_test=args.smoke_test,
    )
