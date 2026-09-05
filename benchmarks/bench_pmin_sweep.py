#!/usr/bin/env python3
"""Single-variable benchmark sweep for speculative draft probability threshold (--spec-draft-p-min).

Values tested: 0.00, 0.50, 0.65, 0.75, 0.85.
Fixed baseline:
  - Model: Qwen3.6-35B-A3B-UD-Q4_K_M-MTP.gguf
  - Context: 65,536 (-c 65536)
  - KV Cache: q4_0 (-ctk q4_0 -ctv q4_0)
  - Flash Attention: ON (-fa on)
  - Load Mode: NONE (--load-mode none)
  - Speculative: draft-mtp,ngram-mod (n_max=2)
  - Prompt: Standard LRU Cache implementation
  - Hyperparams: seed=42, temperature=0.2, max_tokens=512
  - Warmup: 128-token deep warmup per p-min value
  - Runs: 3 runs per value, calculating median, min/max, TTFT, and hardware telemetry.
  - Verification: /proc/<pid>/cmdline verified after each restart.
  - Winner Selection: based on median decode tok/s.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_FILE = Path.home() / ".argus_runtime" / "config.json"
RESULTS_FILE = PROJECT_ROOT / "docs" / "measurements" / "pmin-sweep-2026-09-04.json"
MODEL_PATH = PROJECT_ROOT / "scratch" / "models" / "Qwen3.6-35B-A3B-UD-Q4_K_M-MTP.gguf"
BENCHMARK_LOG = Path.home() / ".argus_runtime" / "pmin_benchmark_server.log"

PROMPT = (
    "Python ile LRU Cache (En Son Kullanılan Önbellek) veri yapısını O(1) get ve put "
    "karmaşıklığıyla OrderedDict kullanmadan, çift yönlü bağlı liste (doubly linked list) "
    "ve hash table ile sıfırdan implement et. Kod temiz ve type-hintli olsun."
)


def get_hardware_telemetry() -> Dict[str, Any]:
    telemetry: Dict[str, Any] = {
        "gpu_temp_c": None,
        "gpu_power_w": None,
        "gpu_clock_mhz": None,
        "gpu_mem_used_mib": None,
        "cpu_avg_mhz": None,
    }
    # GPU Telemetry
    try:
        res = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=temperature.gpu,power.draw,clocks.current.graphics,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=1.5,
        )
        if res.returncode == 0 and res.stdout.strip():
            parts = [p.strip() for p in res.stdout.strip().split(",")]
            if len(parts) >= 4:
                telemetry["gpu_temp_c"] = float(parts[0])
                telemetry["gpu_power_w"] = float(parts[1])
                telemetry["gpu_clock_mhz"] = float(parts[2])
                telemetry["gpu_mem_used_mib"] = float(parts[3])
    except Exception:
        pass

    # CPU Telemetry
    try:
        with open("/proc/cpuinfo") as f:
            freqs = [float(line.split(":")[1].strip()) for line in f if "cpu MHz" in line]
            if freqs:
                telemetry["cpu_avg_mhz"] = round(sum(freqs) / len(freqs), 1)
    except Exception:
        pass

    return telemetry


def stop_any_running_service() -> None:
    ctl = PROJECT_ROOT / "scripts" / "argus_ctl.sh"
    subprocess.run([str(ctl), "stop"], capture_output=True)

    for _ in range(20):
        res = subprocess.run(["pgrep", "-f", "llama-server"], capture_output=True, text=True)
        if not res.stdout.strip():
            break
        time.sleep(0.5)
    else:
        subprocess.run(["pkill", "-9", "-f", "llama-server"], capture_output=True)
        time.sleep(1.0)


def start_isolated_llama_server(p_min: float) -> Tuple[int, float]:
    stop_any_running_service()
    time.sleep(1.0)

    BENCHMARK_LOG.parent.mkdir(parents=True, exist_ok=True)
    p_min_str = f"{p_min:.2f}"

    cmd = [
        "/usr/lib/ollama/llama-server",
        "--model", str(MODEL_PATH),
        "--port", "8080",
        "--host", "127.0.0.1",
        "--no-webui",
        "-c", "65536",
        "-ngl", "99",
        "--cpu-moe",
        "-ctk", "q4_0",
        "-ctv", "q4_0",
        "-fa", "on",
        "-np", "1",
        "--load-mode", "none",
        "--spec-type", "draft-mtp,ngram-mod",
        "--spec-draft-n-max", "2",
        "--spec-draft-p-min", p_min_str,
    ]

    env = dict(os.environ)
    cuda_dir = Path("/usr/lib/ollama/cuda_v13")
    if cuda_dir.is_dir():
        env["LD_LIBRARY_PATH"] = f"{cuda_dir}:{cuda_dir.parent}:" + env.get("LD_LIBRARY_PATH", "")
        env["GGML_BACKEND_PATH"] = str(cuda_dir / "libggml-cuda.so")

    t_boot_start = time.perf_counter()
    log_fp = open(BENCHMARK_LOG, "w")
    proc = subprocess.Popen(cmd, stdout=log_fp, stderr=subprocess.STDOUT, env=env, start_new_session=True)

    print(f"\n🔄 [p-min={p_min_str}] Starting isolated llama-server (PID: {proc.pid})...")
    print(f"   Flags: --load-mode none, -fa on, spec draft-mtp,ngram-mod (n=2, p-min={p_min_str})")

    # Poll health
    health_url = "http://127.0.0.1:8080/health"
    healthy = False
    for _ in range(180):
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server exited prematurely with code {proc.returncode}")
        try:
            with httpx.Client(timeout=1.0) as client:
                if client.get(health_url).status_code == 200:
                    healthy = True
                    break
        except Exception:
            pass
        time.sleep(1.0)

    boot_duration = time.perf_counter() - t_boot_start
    if not healthy:
        raise RuntimeError(f"llama-server failed to become healthy within 180s on p-min={p_min_str}")

    # Verify cmdline from /proc
    cmdline = Path(f"/proc/{proc.pid}/cmdline").read_bytes().decode("utf-8", errors="ignore").replace("\x00", " ")
    expected_flag = f"--spec-draft-p-min {p_min_str}"
    if expected_flag in cmdline:
        print(f"   ✅ llama-server healthy in {boot_duration:.1f}s | Verified cmdline contains: {expected_flag}")
    else:
        print(f"   ⚠️ WARNING: {expected_flag} not found in cmdline: {cmdline[:200]}...")

    return proc.pid, boot_duration


def run_inference(max_tokens: int = 512, temperature: float = 0.2, seed: int = 42) -> Dict[str, Any]:
    url = "http://127.0.0.1:8080/v1/chat/completions"
    payload = {
        "model": "qwen3.6-35b-a3b",
        "messages": [{"role": "user", "content": PROMPT}],
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
    hw = get_hardware_telemetry()
    return {
        "elapsed_seconds": elapsed,
        "timings": timings,
        "hardware": hw,
        "completion_tokens": data.get("usage", {}).get("completion_tokens", 0),
    }


def main():
    parser = argparse.ArgumentParser(description="--spec-draft-p-min Single-Variable Sweep Benchmark")
    parser.add_argument("--p-values", nargs="+", type=float, default=[0.00, 0.50, 0.65, 0.75, 0.85], help="p-min values")
    parser.add_argument("--runs", type=int, default=3, help="Runs per configuration")
    parser.add_argument("--max-tokens", type=int, default=512, help="Tokens to generate per run")
    args = parser.parse_args()

    results: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": "NVIDIA GeForce RTX 3050 Ti Laptop GPU (4 GB VRAM) + Intel Core i5-11300H + 32 GB RAM",
        "model": "Qwen3.6-35B-A3B-UD-Q4_K_M-MTP.gguf",
        "commit": "d222767c7",
        "fixed_config": {
            "spec_draft_n_max": 2,
            "spec_type": "draft-mtp,ngram-mod",
            "load_mode": "none",
            "flash_attn": "on",
            "kv_cache_type": "q4_0",
            "num_ctx": 65536,
            "seed": 42,
            "temperature": 0.2,
        },
        "prompt": PROMPT,
        "max_tokens": args.max_tokens,
        "runs_per_val": args.runs,
        "sweep": {},
    }

    print("=" * 85)
    print("🚀 ARGUS SINGLE-VARIABLE BENCHMARK: --spec-draft-p-min SWEEP")
    print(f"   Values: {args.p_values} | Runs: {args.runs} | Max Tokens: {args.max_tokens}")
    print(f"   Fixed: n=2, --load-mode none, FA ON, q4_0 KV, ctx 64K, seed=42, temp=0.2")
    print("=" * 85)

    current_pid = None
    try:
        for p_val in args.p_values:
            p_key = f"{p_val:.2f}"
            current_pid, boot_time = start_isolated_llama_server(p_val)

            # Deep warmup (128 tokens)
            print(f"   🔥 Deep warmup run (128 tokens)...")
            try:
                w_res = run_inference(max_tokens=128)
                w_tps = w_res["timings"].get("predicted_per_second", 0.0)
                w_ttft = w_res["timings"].get("prompt_ms", 0.0)
                print(f"      Warmup complete: {w_tps:.2f} tok/s | First TTFT: {w_ttft:.0f}ms")
            except Exception as ex:
                print(f"   ⚠️ Warmup warning: {ex}")

            p_runs = []
            for r in range(1, args.runs + 1):
                print(f"   ▶ Run {r}/{args.runs} (512 tokens)...", end="", flush=True)
                res = run_inference(max_tokens=args.max_tokens)
                timings = res["timings"]
                hw = res["hardware"]

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
                    "gpu_temp_c": hw.get("gpu_temp_c"),
                    "gpu_power_w": hw.get("gpu_power_w"),
                    "cpu_avg_mhz": hw.get("cpu_avg_mhz"),
                }
                p_runs.append(run_stat)
                gpu_info = f"GPU: {hw.get('gpu_temp_c')}C" if hw.get("gpu_temp_c") else ""
                cpu_info = f"CPU: {hw.get('cpu_avg_mhz')}MHz" if hw.get("cpu_avg_mhz") else ""
                telemetry_str = f"[{gpu_info}, {cpu_info}]" if gpu_info else ""
                print(f" -> {tps:.2f} tok/s | Acc: {acc_rate:.1f}% ({draft_acc}/{draft_n}) | TTFT: {prompt_ms:.0f}ms {telemetry_str}")
                time.sleep(1.0)

            # Statistics
            tps_list = [x["tokens_per_second"] for x in p_runs]
            acc_list = [x["acceptance_rate_pct"] for x in p_runs]
            accepted_tokens_list = [x["draft_accepted"] for x in p_runs]
            ttft_list = [x["prompt_ms"] for x in p_runs]

            median_tps = statistics.median(tps_list)
            min_tps = min(tps_list)
            max_tps = max(tps_list)

            median_acc = statistics.median(acc_list)
            mean_acc_tokens = round(statistics.mean(accepted_tokens_list), 1)
            median_ttft = statistics.median(ttft_list)

            results["sweep"][f"p_{p_key}"] = {
                "spec_draft_p_min": p_val,
                "boot_duration_seconds": round(boot_time, 1),
                "median_tokens_per_second": round(median_tps, 2),
                "min_tokens_per_second": round(min_tps, 2),
                "max_tokens_per_second": round(max_tps, 2),
                "median_acceptance_pct": round(median_acc, 2),
                "mean_accepted_tokens": mean_acc_tokens,
                "median_prompt_ms": round(median_ttft, 1),
                "raw_runs": p_runs,
            }

    finally:
        # Stop isolated server
        stop_any_running_service()

    # Determine winner based strictly on median decode tok/s
    best_key = max(results["sweep"].keys(), key=lambda k: results["sweep"][k]["median_tokens_per_second"])
    best_p_min = results["sweep"][best_key]["spec_draft_p_min"]
    base_tps = results["sweep"]["p_0.00"]["median_tokens_per_second"]
    results["winner"] = {
        "best_spec_draft_p_min": best_p_min,
        "best_median_tokens_per_second": results["sweep"][best_key]["median_tokens_per_second"],
        "baseline_tokens_per_second": base_tps,
        "delta_pct": round(((results["sweep"][best_key]["median_tokens_per_second"] - base_tps) / base_tps) * 100, 2),
    }

    # Summary Table
    print("\n" + "=" * 90)
    print("📊 SPECULATIVE DRAFT P-MIN SWEEP RESULTS (SCOREBOARD)")
    print("=" * 90)
    print(f"{'p-min':<8} | {'Median Speed (Min - Max)':<25} | {'Acceptance':<12} | {'Avg Acc Tokens':<16} | {'TTFT':<10} | {'Delta vs 0.00':<12}")
    print("-" * 90)

    for p_val in args.p_values:
        p_key = f"{p_val:.2f}"
        data = results["sweep"][f"p_{p_key}"]
        tps = data["median_tokens_per_second"]
        tps_range = f"{tps:.2f} ({data['min_tokens_per_second']:.1f}-{data['max_tokens_per_second']:.1f}) tok/s"
        acc = f"{data['median_acceptance_pct']:.1f}%"
        acc_toks = f"{data['mean_accepted_tokens']} toks"
        ttft = f"{data['median_prompt_ms']:.0f} ms"
        delta = f"{((tps - base_tps) / base_tps) * 100:+.1f}%" if p_val != 0.00 else "BASELINE"
        star = " 🏆 (WINNER)" if p_val == best_p_min else ""
        print(f"{p_key:<8} | {tps_range:<25} | {acc:<12} | {acc_toks:<16} | {ttft:<10} | {delta:>11}{star}")
    print("=" * 90)

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    print(f"\n📁 Detailed results saved to: {RESULTS_FILE}")

    # Restore regular ARGUS service with the winning p-min configuration
    print(f"\n🔄 Restoring ARGUS service with winning configuration (--spec-draft-p-min {best_p_min:.2f})...")
    ctl = PROJECT_ROOT / "scripts" / "argus_ctl.sh"
    subprocess.run(
        [
            str(ctl), "start",
            "--spec-draft-n-max", "2",
            "--spec-draft-p-min", f"{best_p_min:.2f}",
            "--flash-attn", "on",
            "--load-mode", "none",
        ],
        capture_output=True,
    )
    # Update config.json
    cfg = {}
    if CONFIG_FILE.exists():
        try:
            cfg = json.loads(CONFIG_FILE.read_text())
        except Exception:
            cfg = {}
    cfg["spec_draft_n_max"] = 2
    cfg["spec_draft_p_min"] = best_p_min
    cfg["flash_attn"] = "on"
    cfg["load_mode"] = "none"
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2))
    print("✅ ARGUS service restored and operational.\n")


if __name__ == "__main__":
    main()
