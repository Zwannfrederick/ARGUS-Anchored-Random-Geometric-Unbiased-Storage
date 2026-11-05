#!/usr/bin/env python3
"""Foreground systemd exec launcher for ARGUS CUDA llama-server (reads ~/.argus_runtime/config.json)."""

from __future__ import annotations

import os
import sys
from argparse import Namespace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.argus_service import (  # noqa: E402
    BACKEND_PID_FILE,
    apply_config,
    ensure_runtime_dir,
    plan_llama_server_launch,
    warm_model_page_cache,
)

DEFAULT_FALLBACK_MODEL = str(
    PROJECT_ROOT / "scratch" / "models" / "Qwen3.6-35B-A3B-uncensored-heretic.i1-Q4_K_S.gguf"
)


def main() -> None:
    ensure_runtime_dir()
    args = apply_config(Namespace())
    args.llama_server_model = args.llama_server_model or DEFAULT_FALLBACK_MODEL
    if not os.path.isfile(args.llama_server_model):
        raise SystemExit(f"ARGUS model file not found: {args.llama_server_model}")

    cmd, env, load_mode, summary = plan_llama_server_launch(args)
    print(f"ARGUS llama-server [{summary}]", file=sys.stderr)
    if load_mode == "mmap":
        warm_model_page_cache(args.llama_server_model)

    nice_level = int(args.nice_level or 0)
    if nice_level > 0:
        try:
            os.nice(nice_level)
        except OSError as exc:
            print(f"ARGUS nice({nice_level}) failed: {exc}", file=sys.stderr)

    BACKEND_PID_FILE.write_text(str(os.getpid()))
    os.execvpe(cmd[0], cmd, env)


if __name__ == "__main__":
    main()
