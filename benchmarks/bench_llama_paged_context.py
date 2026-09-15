"""Real-model context ladder: stock llama.cpp vs ARGUS paged KV, with a needle check.

Each run starts a fresh llama-server, fills the context with filler text that hides
one fact near the start, asks for the fact, and records server timings, process
memory, GPU memory and ARGUS residency counters. Nothing here is synthetic KV:
every token is prefilled by the model and decode attends to the whole context.

Run explicitly (minutes to hours):
  python benchmarks/bench_llama_paged_context.py --server BUILD/bin/llama-server \
      --model MODEL.gguf --kv-dir /physical/disk/dir --contexts 8192 32768 \
      --modes stock-gpu-kv stock-host-kv argus-paged --resident-bytes 67108864 \
      --output docs/measurements/result.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import socket
import subprocess
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

NEEDLE = "The vault access code is {code}."
QUESTION = "\n\nQuestion: What is the vault access code mentioned at the beginning? Answer with the number only.\nAnswer:"
FILLER = [
    "The river bends twice before it reaches the old mill near the northern hills.",
    "A committee reviewed the harbour schedule and postponed the lighthouse repairs.",
    "Farmers in the valley rotate barley and lentils to keep the soil productive.",
    "The museum catalogued forty ceramic bowls recovered from the dry lake bed.",
    "Engineers measured the bridge cables every spring after the thaw had passed.",
    "Local bakers start before dawn so the market stalls open with fresh bread.",
]


def http_json(url, payload=None, timeout=7200):
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def memory_sampler(pid, stop, peaks):
    """Peak process RSS and ARGUS-mapped RSS from /proc, sampled every 200 ms."""
    while not stop.is_set():
        try:
            rss = argus = 0
            in_argus = False
            for line in Path(f"/proc/{pid}/smaps").read_text().splitlines():
                if line[:1].isalnum() and " " in line and "-" in line.split(" ")[0]:
                    in_argus = "/argus-ggml-" in line or "[anon:argus-disk-" in line
                elif line.startswith("Rss:"):
                    kib = int(line.split()[1]) * 1024
                    rss += kib
                    argus += kib if in_argus else 0
            peaks["rss_bytes"] = max(peaks.get("rss_bytes", 0), rss)
            peaks["argus_mapped_rss_bytes"] = max(peaks.get("argus_mapped_rss_bytes", 0), argus)
            used = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            ).stdout
            for row in used.splitlines():
                app_pid, mib = (part.strip() for part in row.split(","))
                if int(app_pid) == pid:
                    peaks["vram_bytes"] = max(peaks.get("vram_bytes", 0), int(mib) << 20)
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
        stop.wait(0.2)


def build_prompt(base, target_tokens, code):
    tokens_per_line = len(http_json(f"{base}/tokenize", {"content": FILLER[0] + " "})["tokens"])
    rng = random.Random(code)
    lines = [NEEDLE.format(code=code)]
    lines += [rng.choice(FILLER) for _ in range(max(1, target_tokens // max(tokens_per_line, 1)))]
    prompt = " ".join(lines) + QUESTION
    # Trim to the exact budget with the model's own tokenizer.
    while len(http_json(f"{base}/tokenize", {"content": prompt})["tokens"]) > target_tokens:
        lines = lines[:-max(1, len(lines) // 50)]
        prompt = " ".join(lines) + QUESTION
    return prompt


def run_one(args, mode, context):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = {k: v for k, v in os.environ.items() if not k.startswith("ARGUS_KV_")}
    command = [args.server, "-m", args.model, "-c", str(context), "-np", "1", "-ngl", str(args.gpu_layers),
               "-fa", "on", "-t", str(args.threads), "-ctk", args.kv_type, "-ctv", args.kv_type,
               "--no-webui", "--host", "127.0.0.1", "--port", str(port), "-b", "2048", "-ub", str(args.ubatch)]
    storage = tempfile.TemporaryDirectory(prefix="argus-ladder-", dir=args.kv_dir)
    stats_path = Path(storage.name) / "stats.json"
    if mode != "stock-gpu-kv":
        command.append("-nkvo")
    if mode in {"argus-paged", "argus-direct"}:
        env.update(ARGUS_KV_DIR=storage.name, ARGUS_KV_MAX_BYTES=str(args.max_kv_bytes),
                   ARGUS_KV_RESIDENT_BYTES=str(args.resident_bytes), ARGUS_KV_STATS_PATH=str(stats_path))
    if mode == "argus-direct":
        env["ARGUS_KV_STAGING_BYTES"] = str(args.staging_bytes)
    log = tempfile.TemporaryFile(mode="w+")
    process = subprocess.Popen(command, env=env, stdout=log, stderr=log)
    result = {"mode": mode, "context": context, "command": command}
    stop, peaks = threading.Event(), {}
    try:
        base = f"http://127.0.0.1:{port}"
        deadline = time.monotonic() + 600
        while True:
            if process.poll() is not None:
                raise RuntimeError("server exited during startup")
            try:
                if http_json(f"{base}/health", timeout=2).get("status") == "ok":
                    break
            except OSError:
                pass
            if time.monotonic() > deadline:
                raise TimeoutError("server startup")
            time.sleep(0.5)
        code = random.Random(context).randint(100000, 999999)
        prompt = build_prompt(base, context - args.predict - 64, code)
        sampler = threading.Thread(target=memory_sampler, args=(process.pid, stop, peaks), daemon=True)
        sampler.start()
        started = time.perf_counter()
        completion = http_json(f"{base}/completion", {
            "prompt": prompt, "n_predict": args.predict, "temperature": 0, "seed": 42,
            "cache_prompt": False, "ignore_eos": True,
        })
        result.update(
            wall_seconds=time.perf_counter() - started,
            prompt_tokens=completion["timings"]["prompt_n"],
            ttft_seconds=completion["timings"]["prompt_ms"] / 1000,
            prefill_tokens_per_second=completion["timings"]["prompt_per_second"],
            decode_tokens=completion["timings"]["predicted_n"],
            decode_tokens_per_second=completion["timings"]["predicted_per_second"],
            tpot_seconds=completion["timings"]["predicted_ms"] / 1000 / max(completion["timings"]["predicted_n"], 1),
            answer=completion["content"][:80],
            needle_found=str(code) in completion["content"],
        )
        if stats_path.exists():
            result["argus"] = json.loads(stats_path.read_text())
    except Exception as error:  # the ladder records OOM/timeouts instead of aborting
        result["error"] = f"{type(error).__name__}: {error}"
    finally:
        stop.set()
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        log.seek(0)
        lines = log.read().splitlines()
        result["peaks"] = peaks
        result["kv_logs"] = [l for l in lines if "KV buffer size" in l or "ARGUS_KV" in l or "offloaded" in l][-12:]
        if "error" in result:
            result["log_tail"] = lines[-15:]
        storage.cleanup()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--server", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--kv-dir", required=True, help="directory on the physical device being measured")
    parser.add_argument("--contexts", type=int, nargs="+", required=True)
    parser.add_argument("--modes", nargs="+", default=["stock-gpu-kv", "stock-host-kv", "argus-paged"],
                        choices=["stock-gpu-kv", "stock-host-kv", "argus-paged", "argus-direct"])
    parser.add_argument("--resident-bytes", type=int, required=True, help="ARGUS_KV_RESIDENT_BYTES for argus-paged")
    parser.add_argument("--max-kv-bytes", type=int, default=64 << 30)
    parser.add_argument("--staging-bytes", type=int, default=4 << 20)
    parser.add_argument("--ubatch", type=int, default=256)
    parser.add_argument("--kv-type", default="q8_0")
    parser.add_argument("--gpu-layers", default="99")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--predict", type=int, default=32)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    mount = subprocess.run(["findmnt", "-no", "SOURCE,FSTYPE", "-T", args.kv_dir], capture_output=True, text=True).stdout.strip()
    report = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": {"platform": platform.platform(), "cpu_count": os.cpu_count(), "kv_dir_mount": mount},
        "model": args.model, "kv_type": args.kv_type, "resident_bytes": args.resident_bytes,
        "staging_bytes": args.staging_bytes,
        "runs": [],
    }
    for context in args.contexts:
        for mode in args.modes:
            run = run_one(args, mode, context)
            report["runs"].append(run)
            Path(args.output).write_text(json.dumps(report, indent=2))
            print(json.dumps({k: run.get(k) for k in ("mode", "context", "ttft_seconds", "decode_tokens_per_second",
                                                       "needle_found", "error")}), flush=True)


if __name__ == "__main__":
    main()
