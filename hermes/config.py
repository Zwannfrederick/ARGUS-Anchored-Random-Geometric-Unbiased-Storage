"""
Hermes Supervisor Configuration
===============================
Hardware profile: RTX 3050 Laptop (4GB VRAM) + 32GB System RAM.
Model profile: Gemma 4 E4B-it (4.5B active / 8B total PLE) + mmproj-BF16 + MTP draft.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Any

HERMES_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = HERMES_ROOT.parent
MODELS_DIR = PROJECT_ROOT / "scratch" / "models"

# Model paths
MODEL_PATH = MODELS_DIR / "gemma-4-E4B-it-Q4_K_M.gguf"
MMPROJ_PATH = MODELS_DIR / "mmproj-gemma-4-E4B-BF16.gguf"
MTP_DRAFT_PATH = MODELS_DIR / "mtp-gemma-4-E4B-it.gguf"

# Runtime binaries and libraries
LLAMA_SERVER_BIN = Path(os.environ.get("HERMES_LLAMA_SERVER_BIN", "/usr/lib/ollama/llama-server"))
LLAMA_CUDA_DIR = Path("/usr/lib/ollama/cuda_v13")

# ARGUS v0.5 paged KV: llama-server patched with integrations/llama.cpp.
# The KV directory should be on a physical filesystem; tmpfs pages cannot be evicted.
ARGUS_KV_DIR = os.environ.get("HERMES_ARGUS_KV_DIR", "")
ARGUS_KV_MAX_BYTES = os.environ.get("HERMES_ARGUS_KV_MAX_BYTES", str(64 << 30))
ARGUS_KV_RESIDENT_BYTES = os.environ.get("HERMES_ARGUS_KV_RESIDENT_BYTES", "")
ARGUS_KV_STATS_PATH = HERMES_ROOT / "argus_kv_stats.json"
for _name, _value in (("HERMES_ARGUS_KV_MAX_BYTES", ARGUS_KV_MAX_BYTES),
                      ("HERMES_ARGUS_KV_RESIDENT_BYTES", ARGUS_KV_RESIDENT_BYTES)):
    if _value and (not _value.isdigit() or int(_value) <= 0):
        raise ValueError(f"{_name} must be a positive byte count, got {_value!r}")

# Server settings
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8080
BASE_URL = f"http://{DEFAULT_HOST}:{DEFAULT_PORT}"

# Context & KV Cache Options:
# - 32K context  (32,768)  -> ~648 MB RAM
# - 64K context  (65,536)  -> ~1.3 GB RAM (Recommended: daily fast PC/browser loop)
# - 128K context (131,072) -> ~2.6 GB RAM (Deep project / long terminal sessions)
# - 256K context (262,144) -> ~5.2 GB RAM (Archive capacity ceiling)
DEFAULT_NUM_CTX = int(os.environ.get("HERMES_NUM_CTX", 131072))
KV_CACHE_TYPE = "q4_0"
GPU_LAYERS = int(os.environ.get("HERMES_GPU_LAYERS", 24))

# Speculative Decoding (MTP)
SPEC_DRAFT_N_MAX = 2
SPEC_DRAFT_P_MIN = 0.0

def argus_enabled() -> bool:
    return bool(ARGUS_KV_DIR)


def argus_server_args() -> list[str]:
    """ARGUS owns host KV, so GPU KV offload is off and the paged kernel needs FA layout."""
    return ["-nkvo", "-fa", "on"] if argus_enabled() else []


def get_server_env() -> Dict[str, str]:
    """Environment for llama-server: ARGUS budgets, or Ollama's CUDA ggml libraries."""
    env = dict(os.environ)
    if argus_enabled():
        env.update(
            ARGUS_KV_DIR=ARGUS_KV_DIR,
            ARGUS_KV_MAX_BYTES=ARGUS_KV_MAX_BYTES,
            ARGUS_KV_STATS_PATH=str(ARGUS_KV_STATS_PATH),
        )
        if ARGUS_KV_RESIDENT_BYTES:
            env["ARGUS_KV_RESIDENT_BYTES"] = ARGUS_KV_RESIDENT_BYTES
    elif LLAMA_CUDA_DIR.is_dir():
        cuda_path = str(LLAMA_CUDA_DIR)
        parent_path = str(LLAMA_CUDA_DIR.parent)
        existing = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{cuda_path}:{parent_path}:{existing}".rstrip(":")
        env["GGML_BACKEND_PATH"] = str(LLAMA_CUDA_DIR / "libggml-cuda.so")
    return env
