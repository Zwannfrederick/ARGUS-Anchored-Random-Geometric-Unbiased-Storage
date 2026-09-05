#!/usr/bin/env python3
"""Resumable download script for unsloth/Qwen3.6-35B-A3B-MTP-GGUF."""

import os
import sys
import time
from pathlib import Path
from huggingface_hub import hf_hub_download

REPO_ID = "unsloth/Qwen3.6-35B-A3B-MTP-GGUF"
FILENAME = "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
TARGET_DIR = Path("scratch/models").resolve()
TARGET_FILE = TARGET_DIR / "Qwen3.6-35B-A3B-UD-Q4_K_M-MTP.gguf"

def main():
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    print(f"📥 Starting download of {FILENAME} from {REPO_ID}...")
    print(f"📁 Destination: {TARGET_FILE}")
    start = time.time()
    try:
        downloaded_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=FILENAME,
            local_dir=str(TARGET_DIR),
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        if Path(downloaded_path) != TARGET_FILE and Path(downloaded_path).exists():
            os.rename(downloaded_path, TARGET_FILE)
        duration = round(time.time() - start, 1)
        size_gb = round(TARGET_FILE.stat().st_size / (1024**3), 2)
        print(f"✅ Download completed successfully in {duration}s! Size: {size_gb} GB")
    except Exception as e:
        print(f"❌ Download failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
