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
import hashlib
import json
import os
import platform
import random
import socket
import statistics
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


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
    def tokenize(text, special=False):
        return http_json(f"{base}/tokenize", {"content": text, "add_special": special})["tokens"]

    head = tokenize(NEEDLE.format(code=code) + " ", True)
    tail = tokenize(QUESTION)
    remaining = target_tokens - len(head) - len(tail)
    if remaining < 0:
        raise ValueError("prompt budget cannot hold needle and question")
    rng = random.Random(code)
    filler = []
    while len(filler) < remaining:
        filler.extend(tokenize(" ".join(rng.choice(FILLER) for _ in range(128))))
    # Token-ID prompts preserve the exact length and both ends of the task.
    return head + filler[:remaining] + tail


def summarize(runs):
    summaries = []
    for mode, context in sorted({(run["mode"], run["context"]) for run in runs}):
        group = [run for run in runs if run["mode"] == mode and run["context"] == context]
        good = [run for run in group if "error" not in run]
        summary = {"mode": mode, "context": context, "completed": len(good), "failed": len(group) - len(good)}
        for key in ("wall_seconds", "prefill_seconds", "decode_tokens_per_second", "tpot_seconds"):
            values = [run[key] for run in good]
            if values:
                summary[key] = {"min": min(values), "median": statistics.median(values), "max": max(values)}
        summaries.append(summary)
    return summaries


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
    if mode.startswith("argus-"):
        env.update(ARGUS_KV_DIR=storage.name, ARGUS_KV_MAX_BYTES=str(args.max_kv_bytes),
                   ARGUS_KV_RESIDENT_BYTES=str(args.resident_bytes), ARGUS_KV_STATS_PATH=str(stats_path))
    if mode == "argus-direct" or mode.startswith("argus-cuda-"):
        env["ARGUS_KV_STAGING_BYTES"] = str(args.staging_bytes)
    if mode.startswith("argus-cuda-"):
        if args.attention_path:
            env["ARGUS_KV_ATTENTION_PATH"] = args.attention_path
        env.update(ARGUS_KV_GPU_BYTES=str(args.gpu_bytes), ARGUS_KV_PINNED_BYTES=str(args.pinned_bytes),
                   ARGUS_KV_POLICY="on" if mode == "argus-cuda-on" else "off")
        if args.profile:
            env["ARGUS_KV_PROFILE"] = "cpu" if args.profile == "cpu" else "1"
        if mode == "argus-cuda-control":
            env["ARGUS_KV_GPU_CONTROL"] = "1"
        if args.ram_bytes:
            env["ARGUS_KV_RAM_BYTES"] = str(args.ram_bytes)
    log = tempfile.TemporaryFile(mode="w+")
    process = None
    result = {"mode": mode, "context": context, "command": command,
              "argus_environment": {k: v for k, v in env.items() if k.startswith("ARGUS_KV_")}}
    stop, peaks = threading.Event(), {}
    sampler = None
    try:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=log)
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
        result.update(input_tokens=len(prompt), input_sha256=hashlib.sha256(json.dumps(prompt).encode()).hexdigest(),
                      needle_code=code)
        request = {"prompt": prompt, "n_predict": args.predict, "temperature": 0, "seed": 42,
                   "cache_prompt": False, "ignore_eos": True}
        for _ in range(args.warmups):
            http_json(f"{base}/completion", request, timeout=args.request_timeout)
        before = json.loads(stats_path.read_text()) if stats_path.exists() else {}
        sampler = threading.Thread(target=memory_sampler, args=(process.pid, stop, peaks), daemon=True)
        sampler.start()
        started = time.perf_counter()
        completion = http_json(f"{base}/completion", request, timeout=args.request_timeout)
        result.update(
            wall_seconds=time.perf_counter() - started,
            prompt_tokens=completion["timings"]["prompt_n"],
            prefill_seconds=completion["timings"]["prompt_ms"] / 1000,
            prefill_tokens_per_second=completion["timings"]["prompt_per_second"],
            decode_tokens=completion["timings"]["predicted_n"],
            decode_tokens_per_second=completion["timings"]["predicted_per_second"],
            tpot_seconds=1 / completion["timings"]["predicted_per_second"],
            answer=completion["content"],
            output_sha256=hashlib.sha256(completion["content"].encode()).hexdigest(),
            needle_found=str(code) in completion["content"],
            truncated=completion.get("truncated"),
            tokens_cached=completion.get("tokens_cached"),
        )
        if stats_path.exists():
            result["argus"] = json.loads(stats_path.read_text())
            counters = ("read_bytes", "written_bytes", "committed_pages", "cuda_attention_calls",
                        "policy_promotions", "policy_demotions", "policy_rejected", "policy_nanoseconds")
            counters += tuple(k for k in result["argus"] if k.startswith("profile_"))
            result["argus_delta"] = {k: result["argus"][k] - before.get(k, 0)
                                     for k in counters if k in result["argus"]}
        if result["prompt_tokens"] != len(prompt) or result["truncated"]:
            raise RuntimeError("server did not evaluate the complete token-ID prompt")
        if result["decode_tokens"] != args.predict:
            raise RuntimeError("server did not generate the requested number of tokens")
        if mode.startswith("argus-cuda-"):
            delta = result.get("argus_delta", {})
            if delta.get("cuda_attention_calls", 0) <= 0:
                raise RuntimeError("requested ARGUS CUDA path was not observed")
            if (delta.get("policy_nanoseconds", 0) > 0) != (mode == "argus-cuda-on"):
                raise RuntimeError("policy timing does not match requested off/on mode")
            for key, budget in (("peak_gpu_bytes", args.gpu_bytes), ("peak_pinned_bytes", args.pinned_bytes),
                                ("peak_staging_bytes", args.staging_bytes)):
                if result["argus"][key] > budget:
                    raise RuntimeError(f"native {key} exceeded budget")
            if args.profile == "cuda" and delta.get("profile_prefill_kernel_gpu_ns", 0) <= 0:
                raise RuntimeError("CUDA event profiling was requested but not observed")
            if args.profile == "cpu" and (delta.get("profile_prefill_attention_ns", 0) <= 0 or
                                           delta.get("profile_prefill_kernel_gpu_ns", 0) != 0):
                raise RuntimeError("CPU-only attribution was requested but not observed")
            if mode == "argus-cuda-control":
                if any(result["argus"][key] for key in ("read_bytes", "written_bytes", "disk_bytes")):
                    raise RuntimeError("GPU-resident control performed disk payload I/O or allocated disk backing")
                if args.profile and delta.get("profile_decode_d2d_bytes", 0) <= 0:
                    raise RuntimeError("GPU-resident control did not stage resident KV")
    except Exception as error:  # the ladder records OOM/timeouts instead of aborting
        result["error"] = f"{type(error).__name__}: {error}"
    finally:
        stop.set()
        if sampler is not None:
            sampler.join(timeout=6)
        if process is not None:
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
        log.close()
        storage.cleanup()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--server", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--kv-dir", required=True, help="directory on the physical device being measured")
    parser.add_argument("--contexts", type=int, nargs="+", required=True)
    parser.add_argument("--modes", nargs="+", default=["stock-gpu-kv", "stock-host-kv", "argus-paged"],
                        choices=["stock-gpu-kv", "stock-host-kv", "argus-paged", "argus-direct",
                                 "argus-cuda-off", "argus-cuda-on", "argus-cuda-control"])
    parser.add_argument("--profile", nargs="?", const="cuda", choices=["cpu", "cuda"],
                        help="diagnostic CPU scopes, optionally CUDA events; measure overhead separately")
    parser.add_argument("--attention-path", choices=["staged", "direct", "batched"], help="controlled CUDA datapath A/B")
    parser.add_argument("--resident-bytes", type=int, required=True, help="ARGUS_KV_RESIDENT_BYTES for argus-paged")
    parser.add_argument("--max-kv-bytes", type=int, default=64 << 30)
    parser.add_argument("--staging-bytes", type=int, default=4 << 20)
    parser.add_argument("--gpu-bytes", type=int, default=4 << 20)
    parser.add_argument("--pinned-bytes", type=int, default=4 << 20)
    parser.add_argument("--ram-bytes", type=int, default=0, help="0 disables the optional RAM tier")
    parser.add_argument("--warmups", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1, help="fresh server per repeat; warmups run within it")
    parser.add_argument("--request-timeout", type=float, default=7200)
    parser.add_argument("--ubatch", type=int, default=256)
    parser.add_argument("--kv-type", default="q8_0")
    parser.add_argument("--gpu-layers", default="99")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--predict", type=int, default=32)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.repeats < 1 or args.warmups < 0 or args.predict < 1 or args.request_timeout <= 0:
        parser.error("repeats/predict/timeout must be positive; warmups must be nonnegative")
    if min(args.contexts) <= args.predict + 64 or args.ram_bytes < 0:
        parser.error("contexts must exceed predict + 64; ram-bytes must be nonnegative")
    if min(args.gpu_bytes, args.pinned_bytes, args.resident_bytes, args.staging_bytes,
           args.max_kv_bytes, args.ubatch, args.threads) <= 0:
        parser.error("budgets, ubatch and threads must be positive")
    if any(mode.startswith("argus-cuda-") for mode in args.modes) and args.kv_type != "f16":
        parser.error("CUDA attention currently requires --kv-type f16")
    mount = subprocess.run(["findmnt", "-no", "SOURCE,FSTYPE", "-T", args.kv_dir], capture_output=True, text=True).stdout.strip()
    report = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": {"platform": platform.platform(), "cpu_count": os.cpu_count(), "kv_dir_mount": mount},
        "model": args.model, "kv_type": args.kv_type, "resident_bytes": args.resident_bytes,
        "staging_bytes": args.staging_bytes,
        "settings": vars(args),
        "sha256": {"model": file_sha256(args.model), "server": file_sha256(args.server),
                   "harness": file_sha256(__file__)},
        "notes": ["prefill_seconds is server prompt_ms, not measured TTFT",
                  "ARGUS peaks are process-lifetime peaks including warmup; argus_delta excludes warmup",
                  "profile *_ns CPU scopes are inclusive; *_exclusive_ns subtract nested CPU scopes; GPU events overlap CPU waits",
                  "disk_read/write_ns measure blocking O_DIRECT syscall elapsed time, not device-only service time",
                  "profile phases use query/row count >1 for prefill, 1 for decode; matched here to ubatch=64 and 4016 prompt tokens",
                  "GPU control is diagnostic GPU-authoritative storage with policy off; no backing file or payload disk I/O",
                  "one fresh server per repeat; fixed input and seed; no prompt-cache reuse"],
        "runs": [],
    }
    library = Path(args.server).resolve().parent / "libllama.so"
    if library.exists():
        report["sha256"]["libllama"] = file_sha256(library)
    for context in args.contexts:
        for repeat in range(args.repeats):
            # Alternate mode order to reduce fixed-order thermal/cache bias.
            for mode in args.modes if repeat % 2 == 0 else reversed(args.modes):
                run = run_one(args, mode, context)
                run["repeat"] = repeat
                report["runs"].append(run)
                report["summary"] = summarize(report["runs"])
                Path(args.output).write_text(json.dumps(report, indent=2))
                print(json.dumps({k: run.get(k) for k in ("mode", "context", "prefill_seconds", "decode_tokens_per_second",
                                                       "needle_found", "error")}), flush=True)


if __name__ == "__main__":
    main()
