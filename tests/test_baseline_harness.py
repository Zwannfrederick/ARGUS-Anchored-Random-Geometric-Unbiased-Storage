"""The measurement provenance harness must record enough to reproduce a run.

A benchmark number without its git commit and exact torch/CUDA versions cannot
be reproduced or invalidated later, which makes it an assertion rather than a
measurement. These tests pin what a snapshot has to carry.
"""

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def test_record_baseline_emits_required_keys(tmp_path):
    """A baseline snapshot without provenance is not evidence."""
    out = tmp_path / "baseline.json"
    result = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts" / "record_baseline.py"),
            "--output",
            str(out),
            "--skip-tests",
        ],
        capture_output=True,
        text=True,
        cwd=REPO,
    )
    assert result.returncode == 0, result.stderr

    data = json.loads(out.read_text(encoding="utf-8"))
    for key in ("git_commit", "environment", "file_sizes"):
        assert key in data, f"missing {key}"
    assert data["environment"]["torch"].startswith("2.12"), (
        "baseline must record the pinned torch version"
    )
    assert len(data["git_commit"]) == 40


def test_record_baseline_tracks_the_refactored_sources(tmp_path):
    """The files whose size the port is judged by must be in every snapshot."""
    out = tmp_path / "baseline.json"
    subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts" / "record_baseline.py"),
            "--output",
            str(out),
            "--skip-tests",
        ],
        capture_output=True,
        text=True,
        cwd=REPO,
        check=True,
    )

    sizes = json.loads(out.read_text(encoding="utf-8"))["file_sizes"]
    assert sizes["argus_cache/core/memory_manager.py"] > 0
    assert sizes["argus_cache/csrc/manager.cpp"] > 0
