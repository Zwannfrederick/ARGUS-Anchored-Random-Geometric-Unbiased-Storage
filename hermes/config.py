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
LLAMA_SERVER_BIN = Path("/usr/lib/ollama/llama-server")
LLAMA_CUDA_DIR = Path("/usr/lib/ollama/cuda_v13")

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

def get_cuda_env() -> Dict[str, str]:
    """Returns environment variables required for Ollama CUDA ggml acceleration."""
    env = dict(os.environ)
    if LLAMA_CUDA_DIR.is_dir():
        cuda_path = str(LLAMA_CUDA_DIR)
        parent_path = str(LLAMA_CUDA_DIR.parent)
        existing = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{cuda_path}:{parent_path}:{existing}".rstrip(":")
        env["GGML_BACKEND_PATH"] = str(LLAMA_CUDA_DIR / "libggml-cuda.so")
    return env
