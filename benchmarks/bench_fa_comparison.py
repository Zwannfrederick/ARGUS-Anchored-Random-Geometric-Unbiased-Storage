#!/usr/bin/env python3
"""Single-variable benchmark comparing Flash Attention ON vs OFF on n=2.

Evaluates Qwen 3.6 35B A3B MTP + N-gram (n=2) with -fa on vs -fa off
to test whether upstream Flash Attention kernel regressions exist on CPU-MoE.
Uses identical prompt, seed, temperature, 512 tokens, and computes 3-run medians.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import httpx

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_FILE = Path.home() / ".argus_runtime" / "config.json"
RESULTS_FILE = PROJECT_ROOT / "docs" / "measurements" / "flash-attn-comparison-n2-2026-09-04.json"

PROMPT = (
    "Python ile LRU Cache (En Son Kullanılan Önbellek) veri yapısını O(1) get ve put "
    "karmaşıklığıyla OrderedDict kullanmadan, çift yönlü bağlı liste (doubly linked list) "
    "ve hash table ile sıfırdan implement et. Kod temiz ve type-hintli olsun."
)


def update_runtime_config(fa_mode: str) -> None:
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    cfg = {}
    if CONFIG_FILE.exists():
        try:
            cfg = json.loads(CONFIG_FILE.read_text())
        except Exception:
            cfg = {}
    cfg["spec_draft_n_max"] = 2
    cfg["flash_attn"] = fa_mode
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2))


def restart_service(fa_mode: str) -> None:
    print(f"\n🔄 [FA={fa_mode.upper()}] Restarting ARGUS service with n=2 and --flash-attn {fa_mode}...")
    ctl = PROJECT_ROOT / "scripts" / "argus_ctl.sh"
    subprocess.run([str(ctl), "stop"], capture_output=True)

    # Ensure any lingering llama-server is terminated
    for _ in range(20):
        res = subprocess.run(["pgrep", "-f", "llama-server"], capture_output=True, text=True)
        if not res.stdout.strip():
            break
        time.sleep(0.5)
    else:
        subprocess.run(["pkill", "-9", "-f", "llama-server"], capture_output=True)
        time.sleep(1.0)

    time.sleep(1.0)
    proc = subprocess.run(
        [str(ctl), "start", "--spec-draft-n-max", "2", "--flash-attn", fa_mode],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        print(f"❌ Failed to start service: {proc.stderr}")
        sys.exit(1)

    # Poll health
    health_url = "http://127.0.0.1:8080/health"
    healthy = False
    for _ in range(60):
        try:
            with httpx.Client(timeout=1.0) as client:
                if client.get(health_url).status_code == 200:
                    healthy = True
                    break
        except Exception:
            pass
        time.sleep(1.0)

    if not healthy:
        raise RuntimeError(f"llama-server failed to answer health on FA={fa_mode}")

    # Verify /proc cmdline has exact fa flag
    res = subprocess.run(["pgrep", "-f", "llama-server"], capture_output=True, text=True)
    pids = res.stdout.strip().split()
    cmdline_verified = False
    for pid in pids:
        try:
            cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().decode("utf-8", errors="ignore").replace("\x00", " ")
            if f"-fa {fa_mode}" in cmdline:
                print(f"   ✅ llama-server healthy (PID {pid}, verified cmdline has -fa {fa_mode})")
                cmdline_verified = True
                break
        except Exception:
            pass

    if not cmdline_verified:
        print(f"   ℹ️ llama-server healthy (FA={fa_mode})")


def run_inference(max_tokens: int = 512, temperature: float = 0.2, seed: int = 42) -> Dict[str, Any]:
    url = "http://127.0.0.1:8080/v1/chat/completions"
    payload = {
        "model": "qwen3.6-35b-a3b",
        "messages": [
            {"role": "user", "content": PROMPT}
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "seed": seed,
    }
    t0 = time.perf_counter()
    with httpx.Client(timeout=300.0) as client:
        resp = client.post(url, json=payload)
    elapsed = time.perf_counter() - t0
    resp.raise_for_status()
    data = resp.json()
    timings = data.get("timings", {})
    return {
        "elapsed_seconds": elapsed,
        "timings": timings,
        "completion_tokens": data.get("usage", {}).get("completion_tokens", 0),
    }


def main():
    parser = argparse.ArgumentParser(description="Flash Attention Comparison Benchmark (ON vs OFF)")
    parser.add_argument("--runs", type=int, default=3, help="Runs per configuration")
    parser.add_argument("--max-tokens", type=int, default=512, help="Tokens to generate per run")
    args = parser.parse_args()

    modes = ["on", "off"]
    results: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": "NVIDIA GeForce RTX 3050 Ti Laptop GPU (4 GB VRAM) + Intel Core i5-11300H + 32 GB RAM",
        "model": "Qwen3.6-35B-A3B-UD-Q4_K_M-MTP.gguf",
        "commit": "d222767c7",
        "draft_depth": 2,
        "prompt": PROMPT,
        "max_tokens": args.max_tokens,
        "runs_per_mode": args.runs,
        "comparison": {},
    }

    print("=" * 80)
    print("🚀 ARGUS FLASH ATTENTION BENCHMARK: FA ON vs FA OFF (n=2)")
    print(f"   Runs: {args.runs} | Max Tokens: {args.max_tokens} | Model: Qwen 3.6 35B A3B MTP")
    print("=" * 80)

    for fa_mode in modes:
        update_runtime_config(fa_mode)
        restart_service(fa_mode)

        # Deep warmup
        print(f"   🔥 Deep warmup run (128 tokens) to exercise MoE router & warm page-cache...")
        try:
            w_res = run_inference(max_tokens=128)
            w_tps = w_res["timings"].get("predicted_per_second", 0.0)
            print(f"      Warmup complete ({w_tps:.2f} tok/s)")
        except Exception as ex:
            print(f"   ⚠️ Warmup warning: {ex}")

        m_runs = []
        for r in range(1, args.runs + 1):
            print(f"   ▶ Run {r}/{args.runs} (generating {args.max_tokens} tokens)...", end="", flush=True)
            res = run_inference(max_tokens=args.max_tokens)
            timings = res["timings"]
            tps = timings.get("predicted_per_second", 0.0)
            draft_n = timings.get("draft_n", 0)
            draft_acc = timings.get("draft_n_accepted", 0)
            acc_rate = (draft_acc / draft_n * 100.0) if draft_n > 0 else 0.0
            pred_n = timings.get("predicted_n", 0)
            eval_ms = timings.get("predicted_ms", 0.0)
            prompt_ms = timings.get("prompt_ms", 0.0)
            prompt_tps = timings.get("prompt_per_second", 0.0)

            run_stat = {
                "run": r,
                "tokens_per_second": round(tps, 2),
                "predicted_tokens": pred_n,
                "predicted_ms": round(eval_ms, 1),
                "prompt_ms": round(prompt_ms, 1),
                "prompt_tps": round(prompt_tps, 2),
                "draft_n": draft_n,
                "draft_accepted": draft_acc,
                "acceptance_rate_pct": round(acc_rate, 2),
            }
            m_runs.append(run_stat)
            print(f" -> {tps:.2f} tok/s | Acceptance: {acc_rate:.1f}% ({draft_acc}/{draft_n}) | TTFT: {prompt_ms:.0f}ms")
            time.sleep(1.0)

        tps_list = [x["tokens_per_second"] for x in m_runs]
        acc_list = [x["acceptance_rate_pct"] for x in m_runs]
        ttft_list = [x["prompt_ms"] for x in m_runs]

        median_tps = statistics.median(tps_list)
        min_tps = min(tps_list)
        max_tps = max(tps_list)

        median_acc = statistics.median(acc_list)
        min_acc = min(acc_list)
        max_acc = max(acc_list)

        median_ttft = statistics.median(ttft_list)
        min_ttft = min(ttft_list)
        max_ttft = max(ttft_list)

        results["comparison"][f"fa_{fa_mode}"] = {
            "flash_attn": fa_mode,
            "median_tokens_per_second": round(median_tps, 2),
            "min_tokens_per_second": round(min_tps, 2),
            "max_tokens_per_second": round(max_tps, 2),
            "median_acceptance_pct": round(median_acc, 2),
            "min_acceptance_pct": round(min_acc, 2),
            "max_acceptance_pct": round(max_acc, 2),
            "median_prompt_ms": round(median_ttft, 1),
            "raw_runs": m_runs,
        }

    # Scoreboard
    print("\n" + "=" * 80)
    print("📊 FLASH ATTENTION COMPARISON (n=2, MEDIAN + MIN/MAX SCOREBOARD)")
    print("=" * 80)
    print(f"{'FA Mode':<10} | {'Median Speed (Min - Max)':<25} | {'Acceptance (Med [Min-Max])':<26} | {'TTFT Prefill':<14} | {'Delta vs FA ON':<14}")
    print("-" * 80)

    base_tps = results["comparison"]["fa_on"]["median_tokens_per_second"]
    for fa_mode in modes:
        data = results["comparison"][f"fa_{fa_mode}"]
        tps = data["median_tokens_per_second"]
        tps_range = f"{tps:.2f} ({data['min_tokens_per_second']:.1f}-{data['max_tokens_per_second']:.1f}) tok/s"
        acc = data["median_acceptance_pct"]
        acc_range = f"{acc:.1f}% [{data['min_acceptance_pct']:.1f}-{data['max_acceptance_pct']:.1f}%]"
        ttft = f"{data['median_prompt_ms']:.0f} ms"
        delta = f"{((tps - base_tps) / base_tps) * 100:+.1f}%" if fa_mode != "on" else "BASELINE"
        print(f"FA {fa_mode.upper():<7} | {tps_range:<25} | {acc_range:<26} | {ttft:<14} | {delta:>12}")
    print("=" * 80)

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    print(f"\n📁 Detailed telemetry saved to: {RESULTS_FILE}\n")


if __name__ == "__main__":
    main()
