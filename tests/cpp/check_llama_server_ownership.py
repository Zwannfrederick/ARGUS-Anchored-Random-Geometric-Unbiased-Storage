"""Run explicitly: python check_llama_server_ownership.py SERVER MODEL STORAGE_DIR [GPU_LAYERS].

Compares stock, ARGUS host KV and ARGUS paged KV (bounded resident bytes) servers.
"""

import json
import os
import re
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request


def check(binary, model, directory, gpu_layers="0"):
    results = {}
    with tempfile.TemporaryDirectory(prefix="argus-server-audit-", dir=directory) as storage:
        for mode in ("stock", "argus", "paged"):
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            env = os.environ.copy()
            env.pop("ARGUS_KV_DIR", None)
            env.pop("ARGUS_KV_MAX_BYTES", None)
            env.pop("ARGUS_KV_RESIDENT_BYTES", None)
            stats_path = Path(storage) / f"{mode}-stats.json"
            if mode != "stock":
                env.update(ARGUS_KV_DIR=str(Path(storage).resolve()),
                           ARGUS_KV_MAX_BYTES="16777216", ARGUS_KV_STATS_PATH=str(stats_path))
            if mode == "paged":
                env.update(ARGUS_KV_RESIDENT_BYTES="262144", ARGUS_KV_BLOCK_CELLS="16")
            with tempfile.TemporaryFile(mode="w+") as log:
                process = subprocess.Popen(
                    [binary, "-m", model, "-c", "256", "-np", "1", "-ngl", gpu_layers, "-fa", "on",
                     "-nkvo", "-t", "2", "--verbose", "--host", "127.0.0.1", "--port", str(port)],
                    env=env, stdout=log, stderr=log,
                )
                try:
                    base = f"http://127.0.0.1:{port}"
                    deadline = time.monotonic() + 30
                    while time.monotonic() < deadline:
                        if process.poll() is not None:
                            raise RuntimeError("server exited before health check")
                        try:
                            with urllib.request.urlopen(base + "/health", timeout=1) as response:
                                if response.status == 200:
                                    break
                        except OSError:
                            time.sleep(0.1)
                    else:
                        raise TimeoutError("server startup")
                    request = urllib.request.Request(
                        base + "/completion",
                        data=json.dumps(dict(prompt="Once upon a time", n_predict=64,
                                             temperature=0, seed=42, cache_prompt=False)).encode(),
                        headers={"Content-Type": "application/json"},
                    )
                    with urllib.request.urlopen(request, timeout=60) as response:
                        completion = json.load(response)
                    maps = Path(f"/proc/{process.pid}/maps").read_text()
                    results[mode] = dict(
                        content=completion["content"],
                        tokens_predicted=completion["tokens_predicted"],
                        maps=[line for line in maps.splitlines() if "/argus-ggml-" in line],
                    )
                    time.sleep(1.1)
                    with urllib.request.urlopen(request, timeout=60) as response:
                        json.load(response)
                    if stats_path.exists():
                        results[mode]["stats"] = json.loads(stats_path.read_text())
                finally:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    log.seek(0)
                    logs = log.read()
                    if mode not in results:
                        print(logs[-5000:], file=sys.stderr)
                    else:
                        results[mode]["allocation_logs"] = [
                            line for line in logs.splitlines() if "ARGUS_KV" in line
                        ]
                        results[mode]["kv_layers"] = sum(
                            "dev = ARGUS_HOST" in line for line in logs.splitlines())
                        results[mode]["gpu_logs"] = [
                            line for line in logs.splitlines()
                            if "offloaded" in line or "CUDA0 model buffer" in line
                        ]
    print(json.dumps(results, indent=2))
    for mode in ("argus", "paged"):
        assert results["stock"]["content"] == results[mode]["content"], mode
        assert results[mode]["tokens_predicted"] == 64
        assert results[mode]["maps"] and not results["stock"]["maps"]
        assert any("allocate bytes=" in line for line in results[mode]["allocation_logs"])
        assert any("free bytes=" in line and "live_bytes=0" in line
                   for line in results[mode]["allocation_logs"])
    paged = results["paged"]["stats"]
    assert paged["read_bytes"] > 0 and paged["paged_out_bytes"] > 0, paged
    # The sample can land while another worker still holds its current block plus one
    # prefetched block of one layer; each K/V tensor may also keep one boundary page.
    layers = results["paged"]["kv_layers"]
    cell_bytes_per_layer = paged["live_bytes"] // (256 * layers)
    staging = 2 * 16 * 2 * cell_bytes_per_layer
    assert paged["resident_bytes"] <= paged["resident_budget_bytes"] + staging + 2 * layers * 4096, paged
    assert "stats" not in results["argus"] or results["argus"]["stats"]["paged_out_bytes"] == 0
    if int(gpu_layers) > 0:
        for result in results.values():
            assert any(re.search(r"offloaded [1-9][0-9]*/[0-9]+ layers to GPU", line)
                       for line in result["gpu_logs"])


if __name__ == "__main__":
    check(*sys.argv[1:])
