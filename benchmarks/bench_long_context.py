"""v0.8 long-context runs on a real model, with the machine's coexistence record.

One fresh llama-server per (context, configuration). The context is filled with real
prefilled tokens (filler text hiding one fact near the start), then the model answers a
question about the fact while decode attends to every filled cell. Recorded per run:
prefill and decode rates, VRAM/RSS peaks, the server's CPU use, and what the rest of the
machine felt: swap traffic, major faults, iowait, the lowest MemAvailable, and the wake-up
latency of a small periodic probe task (a stand-in for desktop responsiveness).

Configurations come from a JSON file committed next to the results:
  {"server_args": ["-ngl", "99", "--cpu-moe", ...],
   "configs": {"stock-nkvo": {"args": ["-nkvo"], "env": {}},
               "k1-nkvo": {"args": ["-nkvo"], "env": {"ARGUS_KV_CPU_ATTENTION": "k1"}}}}

  python benchmarks/bench_long_context.py --server BUILD/bin/llama-server --model MODEL.gguf \
      --configs configs.json --contexts 32768 --output result.json
"""

from __future__ import annotations

import argparse
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
from pathlib import Path

from bench_llama_paged_context import build_prompt, file_sha256, http_json, memory_sampler

PROBE_PERIOD = 0.010  # seconds; the probe asks to wake every 10 ms and records how late it was


def read_vmstat():
    fields = dict(line.split() for line in Path("/proc/vmstat").read_text().splitlines())
    return {k: int(fields[k]) for k in ("pswpin", "pswpout", "pgmajfault")}


def read_cpu_jiffies():
    values = [int(x) for x in Path("/proc/stat").read_text().splitlines()[0].split()[1:]]
    return {"total": sum(values), "iowait": values[4]}


def mem_available():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable missing")


def process_cpu_seconds(pid):
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")


class Coexistence:
    """Samples what the rest of the machine experiences while a request runs."""

    def __init__(self):
        self.stop = threading.Event()
        self.lateness = []
        self.min_available = mem_available()
        self.threads = [threading.Thread(target=self._probe, daemon=True),
                        threading.Thread(target=self._memory, daemon=True)]

    def _probe(self):
        while not self.stop.is_set():
            asked = time.perf_counter()
            time.sleep(PROBE_PERIOD)
            self.lateness.append(time.perf_counter() - asked - PROBE_PERIOD)

    def _memory(self):
        while not self.stop.wait(0.25):
            self.min_available = min(self.min_available, mem_available())

    def __enter__(self):
        self.vm, self.cpu = read_vmstat(), read_cpu_jiffies()
        for thread in self.threads:
            thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        for thread in self.threads:
            thread.join()
        vm, cpu = read_vmstat(), read_cpu_jiffies()
        late = sorted(self.lateness) or [0.0]
        self.record = {
            "swap_in_pages": vm["pswpin"] - self.vm["pswpin"],
            "swap_out_pages": vm["pswpout"] - self.vm["pswpout"],
            "major_faults": vm["pgmajfault"] - self.vm["pgmajfault"],
            "iowait_fraction": (cpu["iowait"] - self.cpu["iowait"]) / max(1, cpu["total"] - self.cpu["total"]),
            "min_mem_available_bytes": self.min_available,
            "probe_wakeups": len(self.lateness),
            "probe_late_ms": {"p50": 1e3 * late[len(late) // 2], "p99": 1e3 * late[int(len(late) * 0.99)],
                              "max": 1e3 * late[-1]},
        }
        return False


def start_server(args, config, context, log):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    command = [args.server, "-m", args.model, "-c", str(context), "--host", "127.0.0.1", "--port", str(port),
               "--no-webui", *args.spec["server_args"], *config["args"]]
    env = {k: v for k, v in os.environ.items() if not k.startswith("ARGUS_")}
    env.update(config.get("env", {}))
    process = subprocess.Popen(command, env=env, stdout=log, stderr=log)
    base = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 900
    while True:
        if process.poll() is not None:
            raise RuntimeError("server exited during startup")
        try:
            if http_json(f"{base}/health", timeout=2).get("status") == "ok":
                return process, base, command
        except OSError:
            pass
        if time.monotonic() > deadline:
            raise TimeoutError("server startup")
        time.sleep(0.5)


def run_one(args, name, context):
    config = args.spec["configs"][name]
    result = {"config": name, "context": context}
    log = tempfile.TemporaryFile(mode="w+")
    process = None
    stop, peaks = threading.Event(), {}
    sampler = None
    try:
        process, base, command = start_server(args, config, context, log)
        result.update(command=command[1:], env=config.get("env", {}))
        code = random.Random(context).randint(100000, 999999)
        prompt = build_prompt(base, context - args.predict - 64, code)
        request = {"prompt": prompt, "n_predict": args.predict, "temperature": 0, "seed": 42,
                   "cache_prompt": False, "ignore_eos": True}
        if args.warmup_tokens:
            # Pages the model in (mmap): the first pass over the experts reads them from disk.
            warmup = {"prompt": prompt[:args.warmup_tokens], "n_predict": 4, "temperature": 0, "cache_prompt": False}
            http_json(f"{base}/completion", warmup, timeout=args.request_timeout)
        sampler = threading.Thread(target=memory_sampler, args=(process.pid, stop, peaks), daemon=True)
        sampler.start()
        cpu_before = process_cpu_seconds(process.pid)
        started = time.perf_counter()
        with Coexistence() as coexistence:
            completion = http_json(f"{base}/completion", request, timeout=args.request_timeout)
        wall = time.perf_counter() - started
        timings = completion["timings"]
        result.update(
            wall_seconds=wall,
            prompt_tokens=timings["prompt_n"],
            prefill_seconds=timings["prompt_ms"] / 1000,
            prefill_tokens_per_second=timings["prompt_per_second"],
            decode_tokens=timings["predicted_n"],
            decode_seconds=timings["predicted_ms"] / 1000,
            decode_tokens_per_second=timings["predicted_per_second"],
            server_cpu_cores=(process_cpu_seconds(process.pid) - cpu_before) / wall,
            answer=completion["content"],
            needle_found=str(code) in completion["content"],
            coexistence=coexistence.record,
        )
        if result["prompt_tokens"] != len(prompt) or completion.get("truncated"):
            raise RuntimeError("server did not evaluate the complete token-ID prompt")
        if result["decode_tokens"] != args.predict:
            raise RuntimeError("server did not generate the requested number of tokens")
    except Exception as error:  # a failed configuration (OOM, timeout) is a result, not an abort
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
        k1_requested = config.get("env", {}).get("ARGUS_KV_CPU_ATTENTION") == "k1"
        if k1_requested and "error" not in result and not any(l.startswith("ARGUS K1:") for l in lines):
            result["error"] = "RuntimeError: requested K1 path was not observed in the server log"
        if config.get("env", {}).get("ARGUS_PREFILL") == "layers" and "error" not in result and \
                not any(l.startswith("ARGUS P3: layer-major prefill of") for l in lines):
            result["error"] = "RuntimeError: requested layer-major prefill was not observed in the server log"
        if config.get("env", {}).get("ARGUS_KV_VRAM_BYTES") and "error" not in result and \
                not any(l.startswith("ARGUS KV tiers: cells") for l in lines):
            result["error"] = "RuntimeError: requested KV tiers were not observed in the server log"
        result["server_log"] = [l for l in lines if any(s in l for s in (
            "KV buffer", "compute buffer", "offloaded", "ARGUS", "model buffer", "CUDA0", "error"))][-24:]
        if "error" in result:
            result["log_tail"] = lines[-20:]
        log.close()
    return result


def summarize(runs):
    out = []
    for name, context in sorted({(r["config"], r["context"]) for r in runs}, key=lambda x: (x[1], x[0])):
        good = [r for r in runs if r["config"] == name and r["context"] == context and "error" not in r]
        row = {"config": name, "context": context, "completed": len(good)}
        for key in ("prefill_seconds", "decode_tokens_per_second"):
            values = [r[key] for r in good]
            if values:
                row[key] = {"min": min(values), "median": statistics.median(values), "max": max(values)}
        out.append(row)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--server", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--configs", required=True, help="JSON file: server_args and named configs")
    parser.add_argument("--only", nargs="+", help="run only these configs from the file")
    parser.add_argument("--contexts", type=int, nargs="+", required=True)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--predict", type=int, default=32)
    parser.add_argument("--warmup-tokens", type=int, default=0, help="unmeasured request of this many prompt tokens first")
    parser.add_argument("--request-timeout", type=float, default=6 * 3600)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    args.spec = json.loads(Path(args.configs).read_text())
    names = args.only or list(args.spec["configs"])
    unknown = set(names) - set(args.spec["configs"])
    if unknown:
        parser.error(f"unknown configs: {sorted(unknown)}")
    if args.repeats < 1 or args.predict < 1 or min(args.contexts) <= args.predict + 64:
        parser.error("repeats and predict must be positive; contexts must exceed predict + 64")
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()
    report = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git": {"commit": git("rev-parse", "HEAD"), "dirty_files": len(git("status", "--porcelain").splitlines())},
        "host": {"platform": platform.platform(), "cpu_count": os.cpu_count(),
                 "turbo_disabled": Path("/sys/devices/system/cpu/intel_pstate/no_turbo").read_text().strip()
                 if Path("/sys/devices/system/cpu/intel_pstate/no_turbo").exists() else None},
        "model": Path(args.model).name, "configs": args.spec, "settings": {k: v for k, v in vars(args).items() if k != "spec"},
        "sha256": {"model": file_sha256(args.model), "server": file_sha256(args.server), "harness": file_sha256(__file__)},
        "notes": ["prefill_seconds and decode rate are the server's own timings",
                  "probe_late_ms: how late a thread asking to sleep 10 ms woke up, during the request",
                  "one fresh server per run; fixed prompt and seed; no prompt-cache reuse",
                  "warmup_tokens > 0: one unmeasured request of that many tokens first, in the same server"],
        "runs": [],
    }
    library = Path(args.server).resolve().parent / "libllama.so"
    if library.exists():
        report["sha256"]["libllama"] = file_sha256(library)
    for context in args.contexts:
        for repeat in range(args.repeats):
            for name in names if repeat % 2 == 0 else reversed(names):
                run = run_one(args, name, context)
                run["repeat"] = repeat
                report["runs"].append(run)
                report["summary"] = summarize(report["runs"])
                Path(args.output).write_text(json.dumps(report, indent=2))
                print(json.dumps({k: run.get(k) for k in ("config", "context", "prefill_seconds",
                                                       "decode_tokens_per_second", "needle_found", "error")}), flush=True)


if __name__ == "__main__":
    main()
