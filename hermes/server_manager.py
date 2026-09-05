"""
Hermes Server Manager
=====================
Lifecycle manager for llama-server serving Gemma 4 E4B-it:
- Multi-modal projector (mmproj-BF16)
- Speculative decoding draft model (MTP draft)
- CUDA GPU acceleration (24 layers offloaded to RTX 3050)
- Configurable KV cache (default: 65,536 tokens = ~1.3 GB RAM)
- Per-request reasoning support (--reasoning auto)
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import httpx

try:
    from hermes.config import (
        BASE_URL,
        DEFAULT_HOST,
        DEFAULT_NUM_CTX,
        DEFAULT_PORT,
        GPU_LAYERS,
        HERMES_ROOT,
        KV_CACHE_TYPE,
        LLAMA_SERVER_BIN,
        MMPROJ_PATH,
        MODEL_PATH,
        MTP_DRAFT_PATH,
        SPEC_DRAFT_N_MAX,
        SPEC_DRAFT_P_MIN,
        get_cuda_env,
    )
except ModuleNotFoundError:
    from config import (
        BASE_URL,
        DEFAULT_HOST,
        DEFAULT_NUM_CTX,
        DEFAULT_PORT,
        GPU_LAYERS,
        HERMES_ROOT,
        KV_CACHE_TYPE,
        LLAMA_SERVER_BIN,
        MMPROJ_PATH,
        MODEL_PATH,
        MTP_DRAFT_PATH,
        SPEC_DRAFT_N_MAX,
        SPEC_DRAFT_P_MIN,
        get_cuda_env,
    )

PID_FILE = HERMES_ROOT / "hermes_server.pid"
LOG_FILE = HERMES_ROOT / "hermes_server.log"


class ServerManager:
    def __init__(self, port: int = DEFAULT_PORT, num_ctx: int = DEFAULT_NUM_CTX):
        self.port = port
        self.num_ctx = num_ctx
        self.base_url = f"http://{DEFAULT_HOST}:{self.port}"

    def get_pid(self) -> Optional[int]:
        if not PID_FILE.exists():
            return None
        try:
            pid = int(PID_FILE.read_text().strip())
            os.kill(pid, 0)
            return pid
        except (ValueError, OSError):
            PID_FILE.unlink(missing_ok=True)
            return None

    def is_healthy(self) -> bool:
        try:
            with httpx.Client(timeout=1.5) as client:
                r = client.get(f"{self.base_url}/health")
                return r.status_code == 200 and r.json().get("status") == "ok"
        except Exception:
            return False

    def start(self, num_ctx: Optional[int] = None, gpu_layers: int = GPU_LAYERS) -> Tuple[bool, str]:
        if self.is_healthy():
            return True, f"Server already running and healthy at {self.base_url}"

        ctx = num_ctx or self.num_ctx
        if not MODEL_PATH.exists():
            return False, f"Model file not found: {MODEL_PATH}"
        if not LLAMA_SERVER_BIN.exists():
            return False, f"llama-server binary not found: {LLAMA_SERVER_BIN}"

        cmd = [
            str(LLAMA_SERVER_BIN),
            "--model", str(MODEL_PATH),
            "--mmproj", str(MMPROJ_PATH),
            "--no-mmproj-offload",
            "--spec-draft-model", str(MTP_DRAFT_PATH),
            "--spec-type", "draft-mtp,ngram-mod",
            "--spec-draft-n-max", str(SPEC_DRAFT_N_MAX),
            "--spec-draft-p-min", str(SPEC_DRAFT_P_MIN),
            "--reasoning", "auto",
            "-np", "1",
            "--port", str(self.port),
            "--host", DEFAULT_HOST,
            "-c", str(ctx),
            "-ngl", str(gpu_layers),
            "-fa", "on",
            "-ctk", KV_CACHE_TYPE,
            "-ctv", KV_CACHE_TYPE,
            "--no-webui",
            "--jinja",
        ]

        env = get_cuda_env()
        log_fp = open(LOG_FILE, "a")
        proc = subprocess.Popen(
            cmd,
            stdout=log_fp,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=env,
        )
        PID_FILE.write_text(str(proc.pid))

        # Wait for health check
        for _ in range(30):
            time.sleep(1.0)
            if self.is_healthy():
                return True, f"Server booted successfully (PID: {proc.pid}, Context: {ctx:,}, URL: {self.base_url})"
            if proc.poll() is not None:
                return False, f"Server crashed on startup. See {LOG_FILE}"

        return False, f"Server timed out waiting for /health. Check {LOG_FILE}"

    def stop(self) -> Tuple[bool, str]:
        pid = self.get_pid()
        if not pid:
            # Fallback: check fuser for port
            try:
                subprocess.run(["fuser", "-k", f"{self.port}/tcp"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                pass
            return True, "No running Hermes server found."

        try:
            os.kill(pid, signal.SIGTERM)
            for _ in range(10):
                time.sleep(0.5)
                try:
                    os.kill(pid, 0)
                except OSError:
                    break
            else:
                os.kill(pid, signal.SIGKILL)
        except OSError:
            pass

        PID_FILE.unlink(missing_ok=True)
        return True, f"Hermes server (PID {pid}) stopped and VRAM released."


if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else "status"
    mgr = ServerManager()
    if action == "start":
        ok, msg = mgr.start()
        print(msg)
        sys.exit(0 if ok else 1)
    elif action == "stop":
        ok, msg = mgr.stop()
        print(msg)
    elif action == "status":
        healthy = mgr.is_healthy()
        pid = mgr.get_pid()
        print(f"Status: {'HEALTHY' if healthy else 'STOPPED'} | PID: {pid} | URL: {mgr.base_url}")
