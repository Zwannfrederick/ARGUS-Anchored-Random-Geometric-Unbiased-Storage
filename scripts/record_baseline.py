"""Record a reproducible snapshot of repository and environment state.

Every measurement committed under docs/measurements/ must be traceable to one
of these snapshots. Without the git commit and the exact torch/CUDA versions, a
benchmark number cannot be reproduced or invalidated later -- which makes it an
assertion rather than a measurement.

Usage::

    python scripts/record_baseline.py --output docs/measurements/baseline-<date>.json
    python scripts/record_baseline.py --output /tmp/quick.json --skip-tests
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

#: Sources whose size the C++ port is judged by. Tracking them per snapshot is
#: how "the split actually landed" becomes checkable rather than remembered.
TRACKED_FILES = [
    "argus_cache/core/memory_manager.py",
    "argus_cache/core/telemetry.py",
    "argus_cache/core/jl_operators.py",
    "argus_cache/core/pool_allocator.py",
    "argus_cache/core/granularity.py",
    "argus_cache/core/host_spill.py",
    "argus_cache/csrc/manager.cpp",
    "argus_cache/csrc/quantization_kernels.cu",
    "argus_cache/csrc/tier_codec.cpp",
]


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()


def _git_dirty() -> bool:
    """A snapshot taken on a dirty tree cannot be reproduced from the commit."""
    status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=REPO, text=True
    )
    return bool(status.strip())


def _environment() -> dict:
    import torch

    env = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        env["gpu"] = props.name
        env["gpu_total_mib"] = props.total_memory // (1024 * 1024)
        env["cuda"] = torch.version.cuda
        env["capability"] = f"{props.major}.{props.minor}"
    try:
        import triton

        env["triton"] = triton.__version__
    except ImportError:
        env["triton"] = None
    return env


def _file_sizes() -> dict:
    sizes = {}
    for rel in TRACKED_FILES:
        path = REPO / rel
        if path.exists():
            sizes[rel] = len(path.read_text(encoding="utf-8").splitlines())
        else:
            # Modules not yet extracted report None rather than 0, so a missing
            # file is distinguishable from an empty one.
            sizes[rel] = None
    return sizes


def _run_tests() -> dict:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q", "--tb=no"],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    lines = [ln for ln in proc.stdout.strip().splitlines() if ln.strip()]
    return {
        "exit_code": proc.returncode,
        "summary": lines[-1] if lines else "",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip the test run (fast path for harness self-tests).",
    )
    args = parser.parse_args()

    snapshot = {
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "environment": _environment(),
        "file_sizes": _file_sizes(),
    }
    if not args.skip_tests:
        snapshot["test_summary"] = _run_tests()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(snapshot, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
