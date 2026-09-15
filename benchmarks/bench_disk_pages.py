"""Filled-cache disk/attention capacity check, NOT an end-to-end model benchmark.

Run from the repository root with PYTHONPATH=. and a storage directory on the
device being measured. /tmp may be tmpfs and must not be called an NVMe result.
"""

import argparse
import json
import platform
import resource
import subprocess
import time

import torch

from argus_cache import CodecKind, DiskBlockPool, PageStore


def run(directory, tokens, page_size):
    if tokens <= 0 or page_size <= 0:
        raise ValueError("Token count and page size must be positive")
    page_count = (tokens + page_size - 1) // page_size
    started = time.perf_counter()
    with DiskBlockPool(directory=directory, codec=CodecKind.ACTIVE_FP16,
                       max_slots=page_count, page_size=page_size, num_heads=1,
                       head_dim=4) as pool:
        store = PageStore([pool])
        keys = torch.zeros(1, 1, page_size, 4, dtype=torch.float16)
        values = torch.empty_like(keys)
        total_value = 0
        for page_id in range(page_count):
            count = min(page_size, tokens - page_id * page_size)
            value = page_id % 7
            values.fill_(value)
            store.add(page_id, page_id * page_size, count, pool.codec, pool.placement,
                      keys.view(torch.uint8).flatten(), values.view(torch.uint8).flatten())
            total_value += count * value
        fill_seconds = time.perf_counter() - started
        query = torch.zeros(1, 1, 1, 4)
        started = time.perf_counter()
        output = store.decode(query)
        decode_seconds = time.perf_counter() - started
        torch.testing.assert_close(output, torch.full_like(query, total_value / tokens), rtol=1e-5, atol=1e-5)
        return {
            "kind": "synthetic-filled-disk-cache", "model_generation": False,
            "tokens": tokens, "pages": page_count, "page_size": page_size,
            "geometry": {"layers": 1, "batch": 1, "kv_heads": 1, "head_dim": 4},
            "fill_seconds": fill_seconds, "decode_seconds": decode_seconds,
            "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "pool_usage": store.usage(), "parity_passed": True,
            "directory": str(directory), "python": platform.python_version(),
            "torch": torch.__version__,
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True)
    parser.add_argument("--tokens", type=int, default=1048576)
    parser.add_argument("--page-size", type=int, default=1024)
    args = parser.parse_args()
    print(json.dumps(run(args.directory, args.tokens, args.page_size), indent=2))
