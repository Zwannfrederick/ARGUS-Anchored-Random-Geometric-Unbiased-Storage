#!/usr/bin/env python3
"""Controlled prefill & memory-policy benchmark harness for ARGUS llama-server.

Supports two safe execution modes:
1. `--mode live` (non-destructive, zero restart): benchmarks an already-running server
   across exact token counts (including 78-token short chat & 680-token regression case).
2. `--mode matrix` (maintenance window): sweeps `ARGUS_P3_MIN_TOKENS` across
   `64, 128, 256, 512, auto` (and optional `--sweep-mmap` across `mmap+random`,
   `mmap+normal`, `mmap+sequential`, `none`), capturing TTFT, prefill tok/s,
   major page faults (`majflt`), swap-ins (`pswpin`), RSS/PSS, and P3 engagements.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from argus_cache.adapters.llama_server import ARGUS_LLAMA_SERVER, llama_server_command

DEFAULT_PROMPT_TOKENS = [64, 78, 128, 256, 512, 680, 1024, 2048]
DEFAULT_THRESHOLDS = ["64", "128", "256", "512", "auto"]
P3_LOG_RE = re.compile(r"ARGUS P3: (?:batch engaged|layer-major prefill)")


def read_proc_majflt(pid: Optional[int]) -> int:
    if not pid:
        return 0
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[-1].split()
        return int(fields[9])  # field 12 overall = majflt
    except (OSError, ValueError, IndexError):
        return 0


def read_vmstat_pswpin() -> int:
    try:
        for line in Path("/proc/vmstat").read_text().splitlines():
            if line.startswith("pswpin "):
                return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        pass
    return 0


def read_rss_pss_mib(pid: Optional[int]) -> Tuple[float, float]:
    if not pid:
        return (0.0, 0.0)
    rss_kb = 0
    pss_kb = 0
    try:
        for line in Path(f"/proc/{pid}/smaps_rollup").read_text().splitlines():
            if line.startswith("Rss:"):
                rss_kb = int(line.split()[1])
            elif line.startswith("Pss:"):
                pss_kb = int(line.split()[1])
    except (OSError, ValueError, IndexError):
        pass
    return (round(rss_kb / 1024.0, 1), round(pss_kb / 1024.0, 1))


def find_server_pid(port: int) -> Optional[int]:
    for name in ("backend.pid", "llama-server.pid"):
        pid_file = Path.home() / ".argus_runtime" / name
        if pid_file.is_file():
            try:
                pid = int(pid_file.read_text().strip())
                os.kill(pid, 0)
                return pid
            except (OSError, ValueError):
                pass
    try:
        out = subprocess.check_output(["ss", "-tlnp", f"sport = :{port}"], text=True, timeout=2)
        m = re.search(r"pid=(\d+)", out)
        if m:
            return int(m.group(1))
    except Exception:
        pass
    return None


def craft_exact_token_prompt(client: httpx.Client, base_url: str, target_tokens: int) -> Tuple[str, int]:
    """Build a cache-busted prompt whose tokenized length matches `target_tokens` closely."""
    nonce = f"BENCH-{uuid.uuid4().hex[:12]}: "
    seed_unit = "kuantum termodinamik entropi matris spektrum analiz doğrulama "
    words = (seed_unit * ((target_tokens // 4) + 32)).split()

    lo, hi = 1, len(words)
    best_prompt = nonce + " ".join(words[: max(1, target_tokens // 2)])
    best_count = target_tokens

    # Reserve ~12 tokens for chat template framing when using /v1/chat/completions
    raw_goal = max(4, target_tokens - 12)
    try:
        while lo <= hi:
            mid = (lo + hi) // 2
            cand = nonce + " ".join(words[:mid])
            r = client.post(f"{base_url}/tokenize", json={"content": cand, "add_special": False}, timeout=10.0)
            if r.status_code != 200:
                break
            toks = r.json().get("tokens") or []
            cnt = len(toks)
            best_prompt, best_count = cand, cnt + 12
            if cnt == raw_goal:
                break
            if cnt < raw_goal:
                lo = mid + 1
            else:
                hi = mid - 1
    except httpx.HTTPError:
        pass
    return best_prompt, best_count


def measure_single_prefill(
    client: httpx.Client,
    base_url: str,
    target_tokens: int,
    pid: Optional[int],
    log_path: Optional[Path] = None,
) -> Dict[str, Any]:
    prompt_text, est_tokens = craft_exact_token_prompt(client, base_url, target_tokens)
    log_offset = log_path.stat().st_size if (log_path and log_path.is_file()) else 0
    maj0 = read_proc_majflt(pid)
    swp0 = read_vmstat_pswpin()

    payload = {
        "messages": [{"role": "user", "content": prompt_text}],
        "stream": True,
        "max_tokens": 1,
        "temperature": 0.0,
        "cache_prompt": False,
        "timings_per_token": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    t0 = time.perf_counter()
    ttft_ms: Optional[float] = None
    prompt_n = est_tokens
    prompt_ms = 0.0
    prefill_tps = 0.0

    with client.stream("POST", f"{base_url}/v1/chat/completions", json=payload, timeout=300.0) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line or not line.startswith("data: "):
                continue
            raw = line[6:].strip()
            if raw == "[DONE]":
                break
            try:
                chunk = json.loads(raw)
            except json.JSONDecodeError:
                continue
            choices = chunk.get("choices") or []
            if choices and ttft_ms is None:
                delta = choices[0].get("delta") or {}
                if delta.get("content") or delta.get("reasoning_content"):
                    ttft_ms = (time.perf_counter() - t0) * 1000.0
            timings = chunk.get("timings")
            if timings and timings.get("prompt_n"):
                prompt_n = int(timings["prompt_n"])
                prompt_ms = float(timings.get("prompt_ms") or 0.0)
                prefill_tps = float(timings.get("prompt_per_second") or 0.0)

    wall_ms = (time.perf_counter() - t0) * 1000.0
    if ttft_ms is None:
        ttft_ms = wall_ms
    if prefill_tps <= 0.0 and ttft_ms > 0:
        prefill_tps = prompt_n / (ttft_ms / 1000.0)

    maj_delta = max(0, read_proc_majflt(pid) - maj0)
    swp_delta = max(0, read_vmstat_pswpin() - swp0)
    rss_mib, pss_mib = read_rss_pss_mib(pid)

    p3_batches = 0
    if log_path and log_path.is_file():
        with log_path.open("rb") as fp:
            fp.seek(log_offset)
            new_log = fp.read().decode("utf-8", "replace")
        p3_batches = len(P3_LOG_RE.findall(new_log))

    return {
        "target_tokens": target_tokens,
        "actual_tokens": prompt_n,
        "ttft_ms": round(ttft_ms, 1),
        "prompt_eval_ms": round(prompt_ms or ttft_ms, 1),
        "prefill_tps": round(prefill_tps, 2),
        "majflt": maj_delta,
        "pswpin": swp_delta,
        "rss_mib": rss_mib,
        "pss_mib": pss_mib,
        "p3_batches": p3_batches,
    }


def wait_healthy(base_url: str, timeout_s: float = 240.0) -> bool:
    deadline = time.time() + timeout_s
    with httpx.Client(timeout=2.0) as client:
        while time.time() < deadline:
            try:
                if client.get(f"{base_url}/health").status_code == 200:
                    return True
            except httpx.HTTPError:
                pass
            time.sleep(0.5)
    return False


def spawn_bench_server(
    *,
    binary: str,
    model_path: str,
    port: int,
    num_ctx: int,
    kv_type: str,
    load_mode: str,
    cache_ram: int,
    p3_min_tokens: str,
    mmap_policy: str,
    log_path: Path,
) -> subprocess.Popen:
    cmd = llama_server_command(
        model_path=model_path,
        port=port,
        num_ctx=num_ctx,
        kv_cache_type=kv_type,
        binary=binary,
        spec_type="ngram-mod",
        spec_draft_n_max=4,
        flash_attn="on",
        load_mode=load_mode,
        webui=False,
        cache_ram=cache_ram,
    )
    bin_dir = str(Path(binary).parent)
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = f"{bin_dir}:/opt/cuda/lib64:" + env.get("LD_LIBRARY_PATH", "")
    env["ARGUS_AUTO"] = "1"
    env["ARGUS_KV_VRAM_MARGIN"] = "268435456"
    env["ARGUS_P3_MIN_TOKENS"] = p3_min_tokens
    env["ARGUS_P3_LOG"] = "1"
    if mmap_policy:
        env["ARGUS_MMAP_POLICY"] = mmap_policy

    log_fp = log_path.open("w", encoding="utf-8")
    return subprocess.Popen(cmd, stdout=log_fp, stderr=subprocess.STDOUT, env=env, start_new_session=True)


def stop_proc(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=15)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def format_markdown_report(runs: List[Dict[str, Any]]) -> str:
    lines = [
        "# ARGUS P3 Eşik & Bellek Politikası Benchmark Raporu",
        "",
        "| Konfigürasyon | Hedef Tok | Gerçek Tok | TTFT (ms) | Prefill (tok/s) | P3 Batch | MajFlt | SwapIn | RSS / PSS (MiB) |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for block in runs:
        label = block["label"]
        for row in block["results"]:
            lines.append(
                f"| `{label}` | {row['target_tokens']} | {row['actual_tokens']} | "
                f"{row['ttft_ms']:.1f} | **{row['prefill_tps']:.2f}** | {row['p3_batches']} | "
                f"{row['majflt']} | {row['pswpin']} | {row['rss_mib']:.0f} / {row['pss_mib']:.0f} |"
            )
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="ARGUS P3 Prefill Threshold & Memory Policy Benchmark")
    ap.add_argument("--mode", choices=["live", "matrix"], default="live")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--tokens", type=str, default=",".join(str(x) for x in DEFAULT_PROMPT_TOKENS))
    ap.add_argument("--thresholds", type=str, default=",".join(DEFAULT_THRESHOLDS))
    ap.add_argument("--sweep-mmap", action="store_true", help="Also sweep mmap+random, mmap+normal, load_mode=none")
    ap.add_argument("--binary", type=str, default=ARGUS_LLAMA_SERVER)
    ap.add_argument("--model", type=str, default="")
    ap.add_argument("--num-ctx", type=int, default=131072)
    ap.add_argument("--cache-ram", type=int, default=2048)
    ap.add_argument("--out", type=str, default=str(REPO_ROOT / "docs" / "benchmarks" / "p3_threshold_matrix.md"))
    args = ap.parse_args()

    token_list = [int(x.strip()) for x in args.tokens.split(",") if x.strip()]
    thresholds = [x.strip() for x in args.thresholds.split(",") if x.strip()]
    base_url = f"http://127.0.0.1:{args.port}"
    all_runs: List[Dict[str, Any]] = []

    if args.mode == "live":
        pid = find_server_pid(args.port)
        log_path = Path.home() / ".argus_runtime" / "llama-server.log"
        print(f"🔎 Live non-destructive benchmark against {base_url} (PID={pid})...")
        rows = []
        with httpx.Client() as client:
            for tok in token_list:
                res = measure_single_prefill(client, base_url, tok, pid, log_path if log_path.is_file() else None)
                print(
                    f"  tok={res['actual_tokens']:4d} | TTFT={res['ttft_ms']:8.1f} ms | "
                    f"prefill={res['prefill_tps']:7.2f} tok/s | P3={res['p3_batches']} | majflt={res['majflt']}"
                )
                rows.append(res)
        all_runs.append({"label": "live_baseline", "results": rows})
    else:
        cfg_file = Path.home() / ".argus_runtime" / "config.json"
        cfg = json.loads(cfg_file.read_text()) if cfg_file.is_file() else {}
        model_path = args.model or cfg.get("llama_server_model", "")
        if not model_path or not Path(model_path).is_file():
            print("❌ Model path required (--model or ~/.argus_runtime/config.json).", file=sys.stderr)
            return 1

        matrix_specs: List[Tuple[str, str, str, str]] = [
            (f"p3={th},mmap=random", th, "mmap", "random") for th in thresholds
        ]
        if args.sweep_mmap:
            matrix_specs.extend([
                ("p3=auto,mmap=normal", "auto", "mmap", "normal"),
                ("p3=auto,mmap=sequential", "auto", "mmap", "sequential"),
                ("p3=auto,load_mode=none", "auto", "none", ""),
            ])

        for label, th, lmode, mpol in matrix_specs:
            with tempfile.NamedTemporaryFile(prefix="argus_p3_bench_", suffix=".log", delete=False) as tf:
                tmp_log = Path(tf.name)
            print(f"\n🚀 Launching [{label}] on port {args.port}...")
            proc = spawn_bench_server(
                binary=args.binary,
                model_path=model_path,
                port=args.port,
                num_ctx=args.num_ctx,
                kv_type=cfg.get("kv_cache_type", "q4_0"),
                load_mode=lmode,
                cache_ram=args.cache_ram,
                p3_min_tokens=th,
                mmap_policy=mpol,
                log_path=tmp_log,
            )
            try:
                if not wait_healthy(base_url):
                    print(f"❌ Server failed to become healthy for {label}", file=sys.stderr)
                    continue
                rows = []
                with httpx.Client() as client:
                    for tok in token_list:
                        res = measure_single_prefill(client, base_url, tok, proc.pid, tmp_log)
                        print(
                            f"  [{label}] tok={res['actual_tokens']:4d} | TTFT={res['ttft_ms']:8.1f} ms | "
                            f"prefill={res['prefill_tps']:7.2f} tok/s | P3={res['p3_batches']} | majflt={res['majflt']}"
                        )
                        rows.append(res)
                all_runs.append({"label": label, "results": rows})
            finally:
                stop_proc(proc)
                tmp_log.unlink(missing_ok=True)
                time.sleep(1.0)

    out_md = Path(args.out)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(format_markdown_report(all_runs), encoding="utf-8")
    out_md.with_suffix(".json").write_text(json.dumps(all_runs, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n✅ Saved report to {out_md} and {out_md.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

