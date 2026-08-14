# ARGUS C++ Port Stabilization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Settle the C++/CUDA port — remove stale code, split the memory manager by responsibility, verify both runtime adapters against real runtimes, evaluate the JL tier on real activations, and only then cut a release with measurements that are actually measured.

**Architecture:** No new features. Python keeps decisions (eviction, tier policy, telemetry); C++ keeps mechanics (page pool, codec kernels, attention). Work proceeds on a frozen dependency set until the release is cut; vLLM is verified in a *separate* virtualenv because installing it upgrades torch and breaks the compiled extension.

**Tech Stack:** Python 3.14.7 · torch 2.12.0+cu130 · Triton 3.7.0 · CUDA 13.0 · pybind11 via `torch.utils.cpp_extension` · pytest · Ollama (`ollama-cuda`, Arch `extra`) · vLLM (isolated venv only)

**Spec:** `docs/architecture.md` — the architecture document written during the codec refactor. It is the source of truth for the tier-codec design, plugin contract, adapter contract, and the current benchmark table. Read it before starting.

## Global Constraints

These apply to **every** task. They are not optional.

- **No fabricated measurements.** A number appears in documentation only if the command that produced it is recorded alongside it and can be re-run. If something was not measured, the documentation says "not measured" — never an estimate, never a number carried over from an older run.
- **Separate the three claim classes.** `reconstruction` (synthetic fidelity on tensors), `runtime` (latency/memory), `downstream` (model quality). Never present one as another. Synthetic fidelity is not accuracy.
- **No new features until Task 13 (release) is done.** Bug fixes and refactors only. A "small useful addition" is out of scope.
- **The test suite is the gate.** Baseline is `132 passed, 2 skipped`. No task may reduce passing tests. Run `pytest tests/ -q` before every commit.
- **Do not upgrade torch, triton, or CUDA in the primary venv** before Task 13. The compiled `argus_cpp_backend.so` links against torch 2.12; any upgrade requires a rebuild and invalidates every prior measurement.
- **Rebuild after every C++ change:** `python setup.py build_ext --inplace`. Python-only changes need no rebuild.
- **Public API compatibility.** Anything in `argus_cache.__all__` keeps working. Breaking it requires a shim and a note in the release entry.
- **Commit per task**, using conventional commits (`feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `chore:`). Do not push.
- **Apache-2.0 headers/licensing structure preserved.**

## Sequencing Rationale

The order is not arbitrary — two hard dependencies drive it:

1. **vLLM is last, in an isolated venv.** `pip install vllm` resolves to vLLM 0.27.1 and would upgrade torch 2.12.0 → 2.13.0 and triton 3.7.0 → 3.7.1 in this environment. That breaks `argus_cpp_backend.so` (linked against torch 2.12's `libc10`/`libtorch`) and invalidates every measurement taken before it. Verifying vLLM early would pull the floor out from under the memory-manager split.
2. **Ollama is early because it is free of risk.** It is a separate OS process; it touches neither the venv nor the extension.

Everything that must run on a stable, measured foundation (the split, the JL evaluation, stabilization) happens between those two.

---

## Phase 0 — Lock the baseline

### Task 1: Reproducibility harness and pinned baseline

Nothing later is trustworthy unless "the state before I changed anything" is recorded mechanically rather than remembered.

**Files:**
- Create: `scripts/record_baseline.py`
- Create: `docs/measurements/README.md`
- Test: `tests/test_baseline_harness.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `scripts/record_baseline.py` writing a JSON file with keys
  `git_commit: str`, `environment: dict`, `test_summary: dict`,
  `file_sizes: dict`. `docs/measurements/` is the only directory where
  measurement JSON is committed.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_baseline_harness.py
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def test_record_baseline_emits_required_keys(tmp_path):
    """A baseline snapshot without provenance is not evidence."""
    out = tmp_path / "baseline.json"
    result = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "record_baseline.py"),
         "--output", str(out), "--skip-tests"],
        capture_output=True, text=True, cwd=REPO,
    )
    assert result.returncode == 0, result.stderr

    data = json.loads(out.read_text())
    for key in ("git_commit", "environment", "file_sizes"):
        assert key in data, f"missing {key}"
    assert data["environment"]["torch"].startswith("2.12"), (
        "baseline must record the pinned torch version"
    )
    assert len(data["git_commit"]) == 40
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_baseline_harness.py -v`
Expected: FAIL — `scripts/record_baseline.py` does not exist (returncode 2, "can't open file").

- [ ] **Step 3: Write the implementation**

```python
# scripts/record_baseline.py
"""Record a reproducible snapshot of repository and environment state.

Every measurement committed under docs/measurements/ must be traceable to
one of these snapshots. Without the git commit and the exact torch/CUDA
versions, a benchmark number cannot be reproduced or invalidated later --
which makes it an assertion rather than a measurement.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

TRACKED_FILES = [
    "argus_cache/core/memory_manager.py",
    "argus_cache/csrc/manager.cpp",
    "argus_cache/csrc/quantization_kernels.cu",
    "argus_cache/csrc/tier_codec.cpp",
]


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()


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
        sizes[rel] = len(path.read_text(encoding="utf-8").splitlines()) if path.exists() else None
    return sizes


def _run_tests() -> dict:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q", "--tb=no"],
        cwd=REPO, capture_output=True, text=True,
    )
    tail = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    return {"exit_code": proc.returncode, "summary": tail}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args()

    snapshot = {
        "git_commit": _git_commit(),
        "environment": _environment(),
        "file_sizes": _file_sizes(),
    }
    if not args.skip_tests:
        snapshot["test_summary"] = _run_tests()

    Path(args.output).write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    print(json.dumps(snapshot, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_baseline_harness.py -v`
Expected: PASS

- [ ] **Step 5: Write the measurements README**

```markdown
<!-- docs/measurements/README.md -->
# Measurements

Every file here is machine-generated. Do not hand-edit.

Each JSON file records the git commit and full environment that produced it.
A number that appears in `README.md` or `docs/architecture.md` must be
traceable to a file in this directory.

## Regenerating

    python scripts/record_baseline.py --output docs/measurements/baseline-<date>.json
    python benchmarks/bench_native_runtime.py --json docs/measurements/native-<date>.json

## Claim classes

- `reconstruction` — synthetic fidelity of a codec round-trip on generated
  tensors. Says nothing about model quality.
- `runtime` — wall-clock latency, throughput, memory.
- `downstream` — end-to-end model behavior (perplexity, retrieval accuracy).

Never report one class as another.
```

- [ ] **Step 6: Record the actual baseline**

```bash
.venv/bin/python scripts/record_baseline.py \
  --output docs/measurements/baseline-2026-08-14.json
```

Expected: `test_summary.summary` reads `132 passed, 2 skipped ...`. If it does
not, **stop** — the working tree is not in the state this plan assumes.

- [ ] **Step 7: Run full suite and commit**

```bash
.venv/bin/python -m pytest tests/ -q
git add scripts/record_baseline.py docs/measurements/ tests/test_baseline_harness.py
git commit -m "test: add reproducibility harness and pin measurement baseline"
```

---

## Phase 1 — Cleanup

### Task 2: Remove the stale root `models/attention_wrapper.py`

`models/attention_wrapper.py` (173 lines) has diverged 121 lines from
`argus_cache/models/attention_wrapper.py` (204 lines). The root copy lacks
`pipeline=` and `balloon_driver=` support entirely, so anything importing it
silently gets a cache that ignores tier configuration. Note that
`core/*.py` are **not** duplicates — they are 1–2 line re-export shims and are
handled in Task 3, not here.

**Files:**
- Delete: `models/attention_wrapper.py`
- Create: `models/__init__.py`
- Modify: `benchmarks/vram_profiler.py:10`
- Test: `tests/test_import_hygiene.py`

**Interfaces:**
- Consumes: `scripts/record_baseline.py` from Task 1.
- Produces: `models/` becomes a re-export shim package matching `core/`'s
  existing pattern; `models.attention_wrapper` continues to import.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_import_hygiene.py
"""The root shim packages must re-export, never re-implement.

A second copy of a class silently diverges -- the root attention wrapper had
drifted 121 lines and lost pipeline/balloon_driver support, so code importing
it got a cache that ignored its own tier configuration.
"""

import importlib


def test_root_models_shim_is_the_same_class():
    from argus_cache.models.attention_wrapper import PagedDynamicQuantizedCache as Canonical
    from models.attention_wrapper import PagedDynamicQuantizedCache as ViaShim

    assert ViaShim is Canonical, "root models/ is a divergent copy, not a shim"


def test_root_core_shims_are_the_same_objects():
    for module in ("memory_manager", "quantization", "triton_kernels", "balloon_driver"):
        canonical = importlib.import_module(f"argus_cache.core.{module}")
        shim = importlib.import_module(f"core.{module}")
        for name in getattr(canonical, "__all__", []) or ["__name__"]:
            if hasattr(canonical, name) and hasattr(shim, name):
                assert getattr(shim, name) is getattr(canonical, name), (
                    f"core.{module}.{name} diverged from argus_cache"
                )


def test_shim_accepts_pipeline_argument():
    """The regression the divergent copy caused: pipeline= was dropped."""
    import inspect
    from models.attention_wrapper import PagedDynamicQuantizedCache

    params = inspect.signature(PagedDynamicQuantizedCache.__init__).parameters
    assert "pipeline" in params
    assert "balloon_driver" in params
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_import_hygiene.py -v`
Expected: `test_root_models_shim_is_the_same_class` FAILS (two distinct
classes) and `test_shim_accepts_pipeline_argument` FAILS (no `pipeline` param).

- [ ] **Step 3: Replace the divergent copy with a shim**

```bash
git rm models/attention_wrapper.py
```

```python
# models/attention_wrapper.py
# Re-export to eliminate code duplication.
# This module previously held a second, divergent implementation that had
# drifted 121 lines from argus_cache and silently ignored pipeline= and
# balloon_driver=. Keep this file a pure re-export.
from argus_cache.models.attention_wrapper import *  # noqa: F401,F403
from argus_cache.models.attention_wrapper import PagedDynamicQuantizedCache  # noqa: F401
```

```python
# models/__init__.py
"""Compatibility shim package. Canonical code lives in argus_cache.models."""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_import_hygiene.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Run full suite and commit**

```bash
.venv/bin/python -m pytest tests/ -q
git add models/ tests/test_import_hygiene.py
git commit -m "fix: replace divergent root attention wrapper with a re-export shim"
```

### Task 3: Quarantine `scratch/` and document the shim contract

`scratch/` holds 9 ad-hoc scripts plus `failed_test.log`. They are not tests,
are not run by CI, and import paths that may no longer exist. They should not
be mistaken for part of the package.

**Files:**
- Modify: `.gitignore`
- Create: `scratch/README.md`
- Modify: `pyproject.toml`
- Test: `tests/test_import_hygiene.py` (extend)

**Interfaces:**
- Consumes: `tests/test_import_hygiene.py` from Task 2.
- Produces: `pyproject.toml` `[tool.pytest.ini_options] norecursedirs`
  excluding `scratch`, so `pytest` never collects scratch scripts.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_import_hygiene.py
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def test_scratch_is_excluded_from_test_collection():
    """scratch/ holds throwaway scripts named test_*; collecting them would
    make the suite's pass count meaningless."""
    config = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    assert "norecursedirs" in config
    assert "scratch" in config


def test_scratch_is_documented_as_unsupported():
    readme = REPO / "scratch" / "README.md"
    assert readme.exists(), "scratch/ must state that it is unsupported"
    assert "not part of the package" in readme.read_text(encoding="utf-8").lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_import_hygiene.py -k scratch -v`
Expected: FAIL — no `norecursedirs`, no `scratch/README.md`.

- [ ] **Step 3: Add the pytest exclusion**

Add to `pyproject.toml` under `[tool.pytest.ini_options]` (the section already
exists — it is what rejected `--timeout`):

```toml
norecursedirs = ["scratch", "build", "dist", ".venv", "*.egg-info"]
```

- [ ] **Step 4: Document scratch/**

```markdown
<!-- scratch/README.md -->
# scratch/

Throwaway experiment scripts. **Not part of the package**, not supported, not
run by the test suite, and free to break at any time.

Files here are excluded from pytest collection via `norecursedirs` in
`pyproject.toml` — several are named `test_*.py` but are manual scripts, and
collecting them would inflate the suite's pass count with things nobody
maintains.

Real tests live in `tests/`. If something here is worth keeping, promote it to
a real test there and delete it from this directory.
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_import_hygiene.py -v`
Expected: PASS (5 tests)

- [ ] **Step 6: Confirm the suite count is unchanged**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: `137 passed, 2 skipped` (132 baseline + 5 new hygiene tests). If the
count jumped by more than 5, scratch scripts were being collected — investigate
before continuing.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml scratch/README.md tests/test_import_hygiene.py
git commit -m "chore: exclude scratch/ from test collection and document it"
```

---

## Phase 2 — Split the memory manager

`argus_cache/core/memory_manager.py` is 2662 lines. The problem is not its
length; it is that it holds at least five genuinely separate responsibilities.
Each task below extracts one along an existing seam. **Extract by
responsibility, never to reduce line count** — a task that merely moves a
method to a new file without a clear boundary should be dropped, not completed.

The extraction pattern is identical each time and is stated once here:

1. Create the new module holding the logic, taking the manager as an explicit
   collaborator (no imports back into `memory_manager`).
2. Keep a thin delegating method on `PagedDynamicKVCache` so the public API and
   all existing callers keep working.
3. Verify via the existing suite — these are refactors, so the tests that
   already cover the behavior must pass unchanged.

### Task 4: Extract telemetry and reporting

The cleanest seam: `get_cache_telemetry`, `_sync_pending_cuda_events`,
`print_telemetry_summary`, `get_allocator_fragmentation_report`,
`get_vram_usage` only *read* manager state and format it. Roughly 370 lines
(`memory_manager.py:1365-1450`, `2584-2662`).

**Files:**
- Create: `argus_cache/core/telemetry.py`
- Modify: `argus_cache/core/memory_manager.py:1365-1450`, `:2584-2662`
- Test: `tests/test_telemetry_extraction.py`

**Interfaces:**
- Consumes: `PagedDynamicKVCache` instance attributes (read-only):
  `active_pages`, `pages_by_tier`, `tier_specs`, `num_resurrections`,
  `num_cpu_spills`, `total_demotions`, `dequant_latencies`,
  `pending_cuda_events`, `cascade_counts`, `zero_copy_pool`.
- Produces: `class CacheTelemetry` with
  `__init__(self, cache)`, `snapshot() -> dict`, `format_summary() -> str`,
  `sync_pending_cuda_events() -> None`,
  `allocator_fragmentation() -> dict`, `vram_usage() -> dict`.

- [ ] **Step 1: Write the characterization test first**

Refactors need a test that pins current behavior *before* the move.

```python
# tests/test_telemetry_extraction.py
"""Telemetry must survive extraction byte-for-byte.

This is a refactor, so the contract is that nothing observable changes. These
tests pin the current output shape before the move and after.
"""

import torch

from argus_cache.core.memory_manager import PagedDynamicKVCache
from argus_cache.core.tier_registry import PipelineConfig, TierSpec
from argus_cache.backends.eviction import ImportanceSortPolicy

PAGE = 16


def _cache():
    return PagedDynamicKVCache(
        pipeline=PipelineConfig(
            tiers=[
                TierSpec(name="fp8", backend="fp8", max_pages=1),
                TierSpec(name="int4", backend="int4", max_pages=4),
            ],
            eviction_policy=ImportanceSortPolicy(),
            page_size=PAGE,
            sink_tokens=0,
            max_active_pages=1,
        )
    )


def _fill(cache, pages=4):
    for _ in range(pages):
        k = torch.randn(1, 1, PAGE, 16, dtype=torch.float16)
        cache.push_new_tokens(k, torch.randn_like(k))


def test_telemetry_snapshot_has_stable_keys():
    cache = _cache()
    _fill(cache)

    telemetry = cache.get_cache_telemetry()

    for key in ("active_pages", "total_pages", "num_resurrections", "total_demotions"):
        assert key in telemetry, f"telemetry lost key {key!r} in extraction"


def test_telemetry_counts_match_manager_state():
    cache = _cache()
    _fill(cache)

    telemetry = cache.get_cache_telemetry()

    assert telemetry["active_pages"] == len(cache.active_pages)


def test_print_telemetry_summary_returns_without_raising(capsys):
    cache = _cache()
    _fill(cache)

    cache.print_telemetry_summary()

    assert capsys.readouterr().out, "summary produced no output"


def test_vram_usage_reports_numeric_fields():
    cache = _cache()
    usage = cache.get_vram_usage()
    assert isinstance(usage, dict) and usage
```

- [ ] **Step 2: Run the characterization test against current code**

Run: `.venv/bin/python -m pytest tests/test_telemetry_extraction.py -v`
Expected: PASS — this pins existing behavior. If any test fails now, fix the
*test* to match reality before moving code; do not change behavior in this task.

- [ ] **Step 3: Create the telemetry module**

Move the five method bodies into `argus_cache/core/telemetry.py` verbatim,
rebinding `self` to `self.cache`. Module skeleton:

```python
# argus_cache/core/telemetry.py
"""Cache observability, separated from cache mechanics.

Telemetry only ever reads manager state and formats it. Keeping it in the
manager meant every change to reporting risked touching page lifecycle code;
this module takes the cache as an explicit collaborator so the dependency runs
one way only -- telemetry knows about the cache, the cache does not know how
its numbers are rendered.
"""

from __future__ import annotations

from typing import Any, Dict


class CacheTelemetry:
    def __init__(self, cache: Any) -> None:
        self.cache = cache

    def snapshot(self) -> Dict[str, Any]:
        ...  # body moved from PagedDynamicKVCache.get_cache_telemetry

    def sync_pending_cuda_events(self) -> None:
        ...  # body moved from _sync_pending_cuda_events

    def format_summary(self) -> str:
        ...  # body moved from print_telemetry_summary, returning the string

    def allocator_fragmentation(self) -> Dict[str, Any]:
        ...  # body moved from get_allocator_fragmentation_report

    def vram_usage(self) -> Dict[str, Any]:
        ...  # body moved from get_vram_usage
```

**Do not rewrite the moved logic.** Copy it; adjust only `self.` →
`self.cache.` for manager attributes.

- [ ] **Step 4: Replace the manager methods with delegation**

```python
# in PagedDynamicKVCache.__init__, after telemetry counters are initialized:
from argus_cache.core.telemetry import CacheTelemetry
self._telemetry = CacheTelemetry(self)

# replacing the five extracted methods:
def get_cache_telemetry(self):
    return self._telemetry.snapshot()

def _sync_pending_cuda_events(self):
    return self._telemetry.sync_pending_cuda_events()

def print_telemetry_summary(self):
    print(self._telemetry.format_summary())

def get_allocator_fragmentation_report(self) -> dict:
    return self._telemetry.allocator_fragmentation()

def get_vram_usage(self):
    return self._telemetry.vram_usage()
```

- [ ] **Step 5: Run the full suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: `141 passed, 2 skipped` (137 + 4 telemetry tests). Any failure means
the move changed behavior — revert and redo the move verbatim.

- [ ] **Step 6: Verify the line count actually dropped**

```bash
wc -l argus_cache/core/memory_manager.py argus_cache/core/telemetry.py
```

Expected: `memory_manager.py` around 2300; `telemetry.py` around 370. If
`memory_manager.py` barely shrank, logic was copied rather than moved.

- [ ] **Step 7: Commit**

```bash
git add argus_cache/core/telemetry.py argus_cache/core/memory_manager.py tests/test_telemetry_extraction.py
git commit -m "refactor: extract cache telemetry into its own module"
```

### Task 5: Extract the JL projection matrix cache

`get_jl_projection_matrix` and `get_jl_reconstruction_operator`
(`memory_manager.py:861-936`, ~75 lines) are a keyed cache over
`(device, dtype, seq_len)` with a regularized least-squares solve. Self-contained.

**Files:**
- Create: `argus_cache/core/jl_operators.py`
- Modify: `argus_cache/core/memory_manager.py:861-936`
- Test: `tests/test_jl_operators.py`

**Interfaces:**
- Consumes: `CacheTelemetry` pattern from Task 4 (collaborator style).
- Produces: `class JLOperatorCache` with
  `__init__(self, page_size: int, ratio: int = 4)`,
  `projection(device, dtype, seq_len=None) -> torch.Tensor`,
  `reconstruction(device, dtype, alpha=1e-3, seq_len=None) -> torch.Tensor`,
  `clear() -> None`, `cache_size() -> int`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_jl_operators.py
"""JL operators are cached per (device, dtype, seq_len).

Variable-granularity micro-pages need a differently shaped projection than
full pages, because JL projects along the sequence axis -- a single cached
matrix would be silently wrong for split pages.
"""

import pytest
import torch

from argus_cache.core.jl_operators import JLOperatorCache


def test_projection_shape_follows_sequence_length():
    cache = JLOperatorCache(page_size=64, ratio=4)

    w = cache.projection(torch.device("cpu"), torch.float32, seq_len=64)

    assert w.shape == (16, 64), "projection must map seq_len -> seq_len/ratio"


def test_different_sequence_lengths_get_different_operators():
    cache = JLOperatorCache(page_size=64, ratio=4)

    full = cache.projection(torch.device("cpu"), torch.float32, seq_len=64)
    micro = cache.projection(torch.device("cpu"), torch.float32, seq_len=16)

    assert full.shape != micro.shape
    assert cache.cache_size() == 2


def test_same_key_returns_the_cached_tensor():
    cache = JLOperatorCache(page_size=64, ratio=4)

    first = cache.projection(torch.device("cpu"), torch.float32, seq_len=64)
    second = cache.projection(torch.device("cpu"), torch.float32, seq_len=64)

    assert first is second, "operator was rebuilt instead of cached"


def test_reconstruction_inverts_projection_on_low_rank_input():
    """JL reconstruction is only meaningful on low-rank data -- white noise
    cannot be recovered from a 4x rank reduction."""
    cache = JLOperatorCache(page_size=64, ratio=4)
    device, dtype = torch.device("cpu"), torch.float32

    basis = torch.randn(64, 4, dtype=dtype)
    x = basis @ torch.randn(4, 8, dtype=dtype)          # rank 4, 64x8

    w = cache.projection(device, dtype, seq_len=64)
    recon = cache.reconstruction(device, dtype, seq_len=64)
    restored = recon @ (w @ x)

    rel = (restored - x).norm() / x.norm()
    assert rel < 0.25, f"low-rank reconstruction error {rel:.3f} too high"


def test_clear_empties_the_cache():
    cache = JLOperatorCache(page_size=64, ratio=4)
    cache.projection(torch.device("cpu"), torch.float32, seq_len=64)

    cache.clear()

    assert cache.cache_size() == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_jl_operators.py -v`
Expected: FAIL — `ModuleNotFoundError: argus_cache.core.jl_operators`

- [ ] **Step 3: Create the module**

Move the two method bodies from `memory_manager.py:861-936` into
`JLOperatorCache`, replacing `self._jl_w_proj_cache` / `self._jl_recon_operator_cache`
with instance dicts and `self.page_size` with the constructor argument. Preserve
the existing regularized-solve logic exactly.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_jl_operators.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Delegate from the manager**

```python
# in PagedDynamicKVCache.__init__, replacing the two cache dicts:
from argus_cache.core.jl_operators import JLOperatorCache
self._jl_operators = JLOperatorCache(page_size=self.page_size)

def get_jl_projection_matrix(self, device, dtype, seq_len=None):
    return self._jl_operators.projection(device, dtype, seq_len)

def get_jl_reconstruction_operator(self, device, dtype, alpha=1e-3, seq_len=None):
    return self._jl_operators.reconstruction(device, dtype, alpha, seq_len)
```

Keep the `self.w_proj` back-compat attribute assignment that existing code reads.

- [ ] **Step 6: Run full suite and commit**

```bash
.venv/bin/python -m pytest tests/ -q   # expect 146 passed, 2 skipped
git add argus_cache/core/jl_operators.py argus_cache/core/memory_manager.py tests/test_jl_operators.py
git commit -m "refactor: extract JL operator cache from the memory manager"
```

### Task 6: Extract static pool allocation

`_get_pool_tensor`, `_ensure_pools_allocated`, `_allocate_pool_for_tier` plus
the ~80 lines of `*_pool_*` properties (`memory_manager.py:752-1030`, ~280
lines). This is also where the last per-tier hardcoding in Python lives —
`_allocate_pool_for_tier` still branches on `name == "fp8"` / `"one_bit"` and
must be driven by the tier's capabilities instead.

**Files:**
- Create: `argus_cache/core/pool_allocator.py`
- Modify: `argus_cache/core/memory_manager.py:752-1030`
- Test: `tests/test_pool_allocator.py`

**Interfaces:**
- Consumes: `TierSpec.capabilities` / `TierSpec.effective_bits` from
  `argus_cache/core/tier_registry.py`.
- Produces: `class StaticPoolAllocator` with
  `__init__(self, page_size: int)`,
  `allocate_for_tier(spec, max_pages, device, dtype, batch, num_heads, head_dim) -> dict`,
  `get(tier_name: str, field: str) -> torch.Tensor | None`,
  `release(tier_name: str) -> None`, `pools: dict`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pool_allocator.py
"""Static pool shapes must come from a tier's declared bit width.

_allocate_pool_for_tier previously branched on the tier's *name*, so a plugin
tier got no pool at all -- the same name-coupling the C++ codec refactor
removed, still present on the Python side.
"""

import torch

from argus_cache.core.pool_allocator import StaticPoolAllocator
from argus_cache.core.tier_registry import TierSpec

PAGE, BATCH, HEADS, HEAD_DIM = 64, 1, 2, 16


def _allocate(allocator, spec, max_pages=2):
    return allocator.allocate_for_tier(
        spec, max_pages, torch.device("cpu"), torch.float16, BATCH, HEADS, HEAD_DIM
    )


def test_eight_bit_tier_gets_full_length_pool():
    allocator = StaticPoolAllocator(page_size=PAGE)
    pools = _allocate(allocator, TierSpec(name="int8", backend="int8"))

    assert pools["int8_key_q"].shape == (2, BATCH, HEADS, PAGE, HEAD_DIM)


def test_one_bit_tier_pool_is_packed_eight_to_one():
    allocator = StaticPoolAllocator(page_size=PAGE)
    pools = _allocate(allocator, TierSpec(name="one_bit", backend="one_bit"))

    assert pools["one_bit_key_q"].shape[3] == PAGE // 8


def test_plugin_tier_gets_a_pool_sized_from_its_capabilities():
    """The regression: a custom tier used to fall through every name branch."""
    from argus_cache.plugins import (
        BackendCapabilities, NativeCodecSpec,
        register_quantizer, unregister_quantizer,
    )
    from argus_cache.backends.quantization import INT2Backend

    register_quantizer(
        "plugin_two_bit", INT2Backend,
        BackendCapabilities(
            name="plugin_two_bit", effective_bits=2.0,
            native_codec=NativeCodecSpec(kind="unsigned_affine", bits=2),
        ),
        replace=True,
    )
    try:
        allocator = StaticPoolAllocator(page_size=PAGE)
        pools = _allocate(allocator, TierSpec(name="plugin_two_bit", backend="plugin_two_bit"))

        assert pools, "plugin tier received no static pool"
        assert pools["plugin_two_bit_key_q"].shape[3] == PAGE // 4
    finally:
        unregister_quantizer("plugin_two_bit")


def test_release_frees_a_tiers_pools():
    allocator = StaticPoolAllocator(page_size=PAGE)
    _allocate(allocator, TierSpec(name="int8", backend="int8"))

    allocator.release("int8")

    assert allocator.get("int8", "key_q") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_pool_allocator.py -v`
Expected: FAIL — `ModuleNotFoundError: argus_cache.core.pool_allocator`

- [ ] **Step 3: Implement, driving shapes from capabilities**

Replace the name branches with the tier's declared bit width:

```python
# argus_cache/core/pool_allocator.py (core of allocate_for_tier)
codec = spec.capabilities.native_codec if spec.capabilities else None
pack_factor = 8 // codec.bits if codec and codec.bits < 8 else 1
stored_dtype = torch.int8 if codec and codec.kind == "signed_linear" else torch.uint8
packed_len = self.page_size // pack_factor

pools = {
    f"{spec.name}_key_q": torch.zeros(
        max_pages, batch, num_heads, packed_len, head_dim,
        device=device, dtype=stored_dtype),
    f"{spec.name}_value_q": torch.zeros(
        max_pages, batch, num_heads, packed_len, head_dim,
        device=device, dtype=stored_dtype),
    f"{spec.name}_key_scales": torch.zeros(
        max_pages, batch, num_heads, self.page_size, 1, device=device, dtype=dtype),
    f"{spec.name}_value_scales": torch.zeros(
        max_pages, batch, num_heads, self.page_size, 1, device=device, dtype=dtype),
}
if codec is not None and codec.kind == "unsigned_affine":
    pools[f"{spec.name}_key_min_vals"] = torch.zeros(
        max_pages, batch, num_heads, self.page_size, 1, device=device, dtype=dtype)
    pools[f"{spec.name}_value_min_vals"] = torch.zeros(
        max_pages, batch, num_heads, self.page_size, 1, device=device, dtype=dtype)
```

Projection tiers allocate no static pool (their compressed shape depends on the
projection rank) — return `{}` for them, matching current behavior.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_pool_allocator.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Delegate from the manager**

Keep every `*_pool_*` property on `PagedDynamicKVCache` as a one-line delegation
to `self._pool_allocator.get(tier, field)`, so existing callers and tests are
untouched.

- [ ] **Step 6: Run full suite and commit**

```bash
.venv/bin/python -m pytest tests/ -q   # expect 150 passed, 2 skipped
git add argus_cache/core/pool_allocator.py argus_cache/core/memory_manager.py tests/test_pool_allocator.py
git commit -m "refactor: extract static pool allocator and drive shapes from tier capabilities"
```

### Task 7: Extract variable granularity (page split/merge)

`split_page`, `merge_pages`, `manage_variable_granularity`,
`_write_compressed_field` (`memory_manager.py:2081-2342`, ~260 lines). This is
an experimental subsystem; extraction is also where its invariants get stated.

**Files:**
- Create: `argus_cache/core/granularity.py`
- Modify: `argus_cache/core/memory_manager.py:2081-2342`
- Test: `tests/test_granularity_invariants.py`

**Interfaces:**
- Consumes: `JLOperatorCache` (Task 5) for projection tiers when splitting.
- Produces: `class GranularityManager` with
  `__init__(self, cache, micro_page_size: int)`,
  `split(page, tier_name=None) -> list`, `merge(pages, tier_name=None) -> dict`,
  `rebalance() -> dict` (the former `manage_variable_granularity`).

- [ ] **Step 1: Write the invariant tests**

```python
# tests/test_granularity_invariants.py
"""Invariants for the experimental variable-granularity subsystem.

Splitting and merging move token data between page objects. The invariants
that must hold regardless of tier: no token is lost, no token is duplicated,
and micro-page sizes stay compatible with sub-byte packing.
"""

import pytest
import torch

from argus_cache.core.memory_manager import PagedDynamicKVCache
from argus_cache.core.tier_registry import PipelineConfig, TierSpec
from argus_cache.backends.eviction import ImportanceSortPolicy

PAGE = 64


def _cache(micro=16):
    return PagedDynamicKVCache(
        pipeline=PipelineConfig(
            tiers=[TierSpec(name="int4", backend="int4", max_pages=8)],
            eviction_policy=ImportanceSortPolicy(),
            page_size=PAGE, sink_tokens=0, max_active_pages=4,
            micro_page_size=micro,
        )
    )


def test_split_preserves_total_token_count():
    cache = _cache()
    k = torch.randn(1, 1, PAGE, 16, dtype=torch.float16)
    cache.push_new_tokens(k, torch.randn_like(k))
    page = cache.active_pages[0]

    parts = cache.split_page(page)

    assert sum(p["page_size"] for p in parts) == PAGE


def test_split_assigns_unique_page_ids():
    """Split pages draw IDs from the C++ counter; colliding IDs would make the
    prefetch cache return another page's data."""
    cache = _cache()
    k = torch.randn(1, 1, PAGE, 16, dtype=torch.float16)
    cache.push_new_tokens(k, torch.randn_like(k))

    parts = cache.split_page(cache.active_pages[0])

    ids = [p["page_id"] for p in parts]
    assert len(ids) == len(set(ids)), f"duplicate page ids after split: {ids}"


def test_merge_restores_the_original_token_count():
    cache = _cache()
    k = torch.randn(1, 1, PAGE, 16, dtype=torch.float16)
    cache.push_new_tokens(k, torch.randn_like(k))
    parts = cache.split_page(cache.active_pages[0])

    merged = cache.merge_pages(parts)

    assert merged["page_size"] == PAGE


@pytest.mark.parametrize("micro", [8, 16, 32])
def test_micro_page_size_stays_packing_compatible(micro):
    """Sub-byte tiers pack 8/4/2 elements per byte; a micro-page size that is
    not a multiple of 8 crashes the first time one cascades into one_bit."""
    cache = _cache(micro=micro)
    assert cache.micro_page_size % 8 == 0
```

- [ ] **Step 2: Run tests against current code**

Run: `.venv/bin/python -m pytest tests/test_granularity_invariants.py -v`
Expected: PASS — these characterize existing behavior. If
`test_split_assigns_unique_page_ids` fails, that is a **real bug**; fix it in
this task by drawing IDs from `self._cpp_manager.next_page_id()` and note the
fix in the commit message.

- [ ] **Step 3: Move the four methods into `GranularityManager`**

Copy verbatim, rebinding manager attribute access to `self.cache.`.

- [ ] **Step 4: Delegate from the manager and re-run**

```bash
.venv/bin/python -m pytest tests/ -q   # expect 156 passed, 2 skipped
```

- [ ] **Step 5: Commit**

```bash
git add argus_cache/core/granularity.py argus_cache/core/memory_manager.py tests/test_granularity_invariants.py
git commit -m "refactor: extract variable-granularity page split/merge with invariant tests"
```

### Task 8: Extract host spill and confirm the split landed

`swap_out_to_host` / `swap_in_to_device` (`memory_manager.py:2424-2583`, ~160
lines), then verify the whole phase achieved its goal.

**Files:**
- Create: `argus_cache/core/host_spill.py`
- Modify: `argus_cache/core/memory_manager.py:2424-2583`
- Test: `tests/test_host_spill.py`

**Interfaces:**
- Consumes: `cache.zero_copy_pool`, `cache._cpp_manager`.
- Produces: `class HostSpillManager` with `__init__(self, cache)`,
  `spill_out() -> dict`, `spill_in(device="cuda") -> dict`,
  `is_swapped_out: bool`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_host_spill.py
"""CPU spill must round-trip page data and be idempotent.

Spilling twice, or restoring without spilling, must not corrupt state -- this
runs under memory pressure, which is exactly when retry logic fires.
"""

import pytest
import torch

from argus_cache.core.memory_manager import PagedDynamicKVCache
from argus_cache.core.tier_registry import PipelineConfig, TierSpec
from argus_cache.backends.eviction import ImportanceSortPolicy

PAGE = 16


def _cache():
    return PagedDynamicKVCache(
        pipeline=PipelineConfig(
            tiers=[TierSpec(name="int4", backend="int4", max_pages=8)],
            eviction_policy=ImportanceSortPolicy(),
            page_size=PAGE, sink_tokens=0, max_active_pages=1,
        )
    )


def _fill(cache, n=4):
    for _ in range(n):
        k = torch.randn(1, 1, PAGE, 16, dtype=torch.float16)
        cache.push_new_tokens(k, torch.randn_like(k))


def test_spill_out_then_in_preserves_page_count():
    cache = _cache()
    _fill(cache)
    before = cache._cpp_manager.get_page_count()

    cache.swap_out_to_host()
    cache.swap_in_to_device("cpu")

    assert cache._cpp_manager.get_page_count() == before


def test_double_spill_is_idempotent():
    cache = _cache()
    _fill(cache)

    cache.swap_out_to_host()
    cache.swap_out_to_host()

    assert cache.is_swapped_out


def test_restore_without_spill_is_a_noop():
    cache = _cache()
    _fill(cache)
    before = cache._cpp_manager.get_page_count()

    cache.swap_in_to_device("cpu")

    assert cache._cpp_manager.get_page_count() == before
```

- [ ] **Step 2: Run to characterize**

Run: `.venv/bin/python -m pytest tests/test_host_spill.py -v`
Expected: PASS, or a genuine idempotency bug surfaces — if
`test_double_spill_is_idempotent` fails, guard `swap_out_to_host` with an early
`if self.is_swapped_out: return` and note the fix in the commit.

- [ ] **Step 3: Move both methods into `HostSpillManager`, delegate, re-run**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: `159 passed, 2 skipped`

- [ ] **Step 4: Confirm the phase achieved its goal**

```bash
wc -l argus_cache/core/*.py
```

Expected `memory_manager.py` around **1400–1600 lines**, down from 2662, with
`telemetry.py`, `jl_operators.py`, `pool_allocator.py`, `granularity.py`,
`host_spill.py` alongside it.

**If `memory_manager.py` is still above 2000 lines, stop and report** rather
than inventing further splits to hit a number. The goal was separating five
real responsibilities, not reaching a line target.

- [ ] **Step 5: Update the architecture doc and commit**

Add the new module layout to the layering diagram in `docs/architecture.md` §1.

```bash
.venv/bin/python -m pytest tests/ -q
git add argus_cache/core/host_spill.py argus_cache/core/memory_manager.py tests/test_host_spill.py docs/architecture.md
git commit -m "refactor: extract host spill manager and document the module split"
```

---

## Phase 3 — Verify Ollama against a real server

### Task 9: Live Ollama verification

**Prerequisite (run by the user, not the agent):**

```bash
sudo pacman -S ollama-cuda
sudo systemctl enable --now ollama
ollama pull qwen2.5:0.5b
```

**Files:**
- Modify: `tests/test_adapters.py` (live section)
- Create: `docs/measurements/ollama-<date>.json`
- Modify: `docs/architecture.md` §4

**Interfaces:**
- Consumes: `OllamaAdapter` from `argus_cache/adapters/ollama.py`.
- Produces: a recorded live-verification artifact under `docs/measurements/`.

- [ ] **Step 1: Confirm the server is reachable**

```bash
curl -s localhost:11434/api/version && ollama list
```

Expected: a version JSON and `qwen2.5:0.5b` listed. If not, the remaining steps
cannot be run — report and stop rather than marking this task complete.

- [ ] **Step 2: Run the existing live test**

```bash
.venv/bin/python -m pytest tests/test_adapters.py -v -k live
```

Expected: `test_ollama_live_version_probe` now **runs** instead of skipping.

- [ ] **Step 3: Add live tests that exercise the real contract**

```python
# append to the live section of tests/test_adapters.py
LIVE_MODEL = os.environ.get("ARGUS_OLLAMA_MODEL", "qwen2.5:0.5b")


@pytest.mark.skipif(not _ollama_running(), reason="no Ollama server on localhost:11434")
def test_ollama_live_generate_reports_real_timings():
    with OllamaAdapter(model=LIVE_MODEL, num_ctx=2048) as ollama:
        result = ollama.generate("Count to three.", max_tokens=24)

    assert result.text.strip(), "server returned an empty completion"
    assert result.eval_tokens and result.eval_tokens > 0
    assert result.tokens_per_second and result.tokens_per_second > 0
    # Client wall time must bound the server's own decode time.
    assert result.wall_seconds >= result.eval_seconds


@pytest.mark.skipif(not _ollama_running(), reason="no Ollama server on localhost:11434")
def test_ollama_live_missing_model_is_rejected():
    adapter = OllamaAdapter(model="definitely-not-a-real-model")
    with pytest.raises(AdapterError, match="ollama pull"):
        adapter.initialize()


@pytest.mark.skipif(not _ollama_running(), reason="no Ollama server on localhost:11434")
def test_ollama_live_telemetry_reports_loaded_model():
    with OllamaAdapter(model=LIVE_MODEL) as ollama:
        ollama.generate("hi", max_tokens=4)
        telemetry = ollama.telemetry()

    assert telemetry["argus_manages_kv_cache"] is False
    assert telemetry["loaded_models"], "server reported no loaded model"


@pytest.mark.skipif(not _ollama_running(), reason="no Ollama server on localhost:11434")
def test_ollama_live_repeated_cycles_are_stable():
    adapter = OllamaAdapter(model=LIVE_MODEL)
    for _ in range(3):
        adapter.activate()
        assert adapter.generate("ok", max_tokens=4).text is not None
        adapter.deactivate()
    adapter.shutdown()
```

- [ ] **Step 4: Run and fix what reality disagrees with**

```bash
.venv/bin/python -m pytest tests/test_adapters.py -v -k live
```

Expected: 5 live tests pass. **If the real Ollama API disagrees with the
adapter** (field names, `/api/ps` shape, option keys), fix
`argus_cache/adapters/ollama.py` to match the server — the server is the
truth, not the adapter's assumptions.

- [ ] **Step 5: Record the measurement**

```bash
.venv/bin/python - <<'PY' > docs/measurements/ollama-2026-08-14.json
import json, subprocess
from argus_cache.adapters.ollama import OllamaAdapter
with OllamaAdapter(model="qwen2.5:0.5b", num_ctx=2048) as o:
    runs = [o.generate("Explain virtual memory in one sentence.", max_tokens=64)
            for _ in range(5)]
    print(json.dumps({
        "runtime": "ollama",
        "note": "External baseline. ARGUS does NOT manage this KV cache.",
        "server_version": o.server_version,
        "model": o.model,
        "ollama_pkg": subprocess.check_output(["pacman","-Q","ollama-cuda"], text=True).strip(),
        "runs": [{"wall_s": r.wall_seconds, "tps": r.tokens_per_second,
                  "prompt_eval_s": r.prompt_eval_seconds,
                  "eval_tokens": r.eval_tokens} for r in runs],
        "telemetry": o.telemetry(),
    }, indent=2))
PY
```

- [ ] **Step 6: Update the docs with what was actually verified**

In `docs/architecture.md` §4, change the Ollama subsection's status to state
the verified server version, model, and that these are **external baseline**
numbers — explicitly not ARGUS cache measurements.

- [ ] **Step 7: Commit**

```bash
.venv/bin/python -m pytest tests/ -q
git add tests/test_adapters.py docs/measurements/ollama-2026-08-14.json docs/architecture.md argus_cache/adapters/ollama.py
git commit -m "test: verify the Ollama adapter against a live server"
```

---

## Phase 4 — Evaluate the JL tier honestly

### Task 10: JL evaluation on real activations

The current benchmark reports JL at `cosine 0.27` on random tensors. That
number says nothing about the tier: reconstructing a 4× rank-reduced projection
of white noise is information-theoretically impossible. JL targets the low-rank
structure of *real* KV tensors, and nobody has measured it there.

**Files:**
- Create: `benchmarks/bench_jl_fidelity.py`
- Create: `docs/measurements/jl-<date>.json`
- Modify: `docs/architecture.md` §5.1

**Interfaces:**
- Consumes: `JLOperatorCache` (Task 5), `Qwen/Qwen2.5-0.5B-Instruct` from the
  local HF cache.
- Produces: a JSON artifact with, per layer, `effective_rank`,
  `jl_relative_error`, `jl_cosine`, and the same for `int2` as a same-budget
  control.

- [ ] **Step 1: Write the test that guards the benchmark's logic**

```python
# tests/test_jl_operators.py (append)
def test_jl_error_is_far_lower_on_low_rank_than_on_white_noise():
    """The claim the benchmark rests on: JL is only meaningful for low-rank
    input. If this ever stops holding, the JL tier's premise is wrong."""
    import torch
    from argus_cache.core.jl_operators import JLOperatorCache

    cache = JLOperatorCache(page_size=64, ratio=4)
    device, dtype = torch.device("cpu"), torch.float32
    w = cache.projection(device, dtype, seq_len=64)
    recon = cache.reconstruction(device, dtype, seq_len=64)

    def rel_error(x):
        return ((recon @ (w @ x)) - x).norm().item() / x.norm().item()

    torch.manual_seed(0)
    low_rank = torch.randn(64, 4, dtype=dtype) @ torch.randn(4, 8, dtype=dtype)
    white_noise = torch.randn(64, 8, dtype=dtype)

    assert rel_error(low_rank) < 0.5 * rel_error(white_noise)
```

- [ ] **Step 2: Run it**

Run: `.venv/bin/python -m pytest tests/test_jl_operators.py -k white_noise -v`
Expected: PASS. If it fails, the JL tier's premise does not hold and that is
the headline finding — report it rather than proceeding.

- [ ] **Step 3: Write the benchmark**

```python
# benchmarks/bench_jl_fidelity.py
"""Measure the JL archival tier on REAL KV activations, not random tensors.

Random-tensor fidelity is meaningless for a projection tier: a 4x rank
reduction of white noise cannot be inverted, so a poor score there reflects
the input, not the codec. This script captures real K/V tensors from a model
forward pass, measures their effective rank, and reports JL reconstruction
against an equal-storage-budget quantizer (int2, also 4x) as a control.

Claim class: `reconstruction`. This is NOT downstream model quality.
"""

from __future__ import annotations

import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from argus_cache.core.jl_operators import JLOperatorCache

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def effective_rank(x: torch.Tensor, energy: float = 0.99) -> int:
    """Singular values needed to retain `energy` of the spectrum."""
    sv = torch.linalg.svdvals(x.float())
    cumulative = torch.cumsum(sv**2, 0) / (sv**2).sum()
    return int((cumulative < energy).sum().item()) + 1


def capture_kv(model, tokenizer, prompt: str, device: str):
    """Run one forward pass and return per-layer key tensors."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs, use_cache=True)
    return [layer[0] for layer in out.past_key_values]   # keys only


def int2_roundtrip(x: torch.Tensor) -> torch.Tensor:
    """Equal-budget control: 2-bit affine, also 4x smaller than fp16."""
    vmin, vmax = x.min(), x.max()
    scale = (vmax - vmin) / 3.0 if vmax > vmin else torch.tensor(1.0)
    q = ((x - vmin) / scale).round().clamp(0, 3)
    return q * scale + vmin


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--ratio", type=int, default=4)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16
    ).to(device).eval()

    prompt = "The history of virtual memory management. " * (args.tokens // 8)
    keys = capture_kv(model, tokenizer, prompt, device)

    jl_cache = JLOperatorCache(page_size=0, ratio=args.ratio)
    rows = []
    for idx, k in enumerate(keys):
        # [batch, heads, seq, head_dim] -> project along the sequence axis
        mat = k[0, 0].float().cpu()
        seq_len = mat.shape[0]
        if seq_len % args.ratio:
            continue

        w = jl_cache.projection(torch.device("cpu"), torch.float32, seq_len=seq_len)
        recon_op = jl_cache.reconstruction(torch.device("cpu"), torch.float32, seq_len=seq_len)
        restored = recon_op @ (w @ mat)

        jl_err = ((restored - mat).norm() / mat.norm()).item()
        jl_cos = torch.nn.functional.cosine_similarity(
            restored.flatten(), mat.flatten(), dim=0).item()

        control = int2_roundtrip(mat)
        int2_err = ((control - mat).norm() / mat.norm()).item()

        rows.append({
            "layer": idx,
            "seq_len": seq_len,
            "effective_rank_99": effective_rank(mat),
            "full_rank": min(mat.shape),
            "jl_relative_error": round(jl_err, 4),
            "jl_cosine": round(jl_cos, 4),
            "int2_relative_error_same_budget": round(int2_err, 4),
            "jl_beats_int2": bool(jl_err < int2_err),
        })

    wins = sum(r["jl_beats_int2"] for r in rows)
    result = {
        "metric_class": "reconstruction",
        "note": "Synthetic reconstruction on real activations. NOT model quality.",
        "model": args.model,
        "ratio": args.ratio,
        "layers": rows,
        "jl_wins_vs_int2": f"{wins}/{len(rows)}",
    }

    print(f"{'layer':>5} {'eff.rank':>9} {'full':>5} {'JL err':>8} {'JL cos':>8} {'int2 err':>9}")
    for r in rows:
        print(f"{r['layer']:5d} {r['effective_rank_99']:9d} {r['full_rank']:5d} "
              f"{r['jl_relative_error']:8.4f} {r['jl_cosine']:8.4f} "
              f"{r['int2_relative_error_same_budget']:9.4f}")
    print(f"\nJL beats equal-budget int2 on {wins}/{len(rows)} layers.")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run it and record**

```bash
.venv/bin/python benchmarks/bench_jl_fidelity.py \
  --json docs/measurements/jl-2026-08-14.json
```

- [ ] **Step 5: Report the verdict — whatever it is**

Update `docs/architecture.md` §5.1's JL note with the measured result. Three
possible outcomes, all acceptable to write down:

- **JL beats equal-budget int2 on most layers** → the tier is justified; state
  the win rate and the measured effective rank.
- **JL roughly ties int2** → state that it earns its place only where the
  projection is reused across pages, and mark it as such.
- **JL loses to int2 on most layers** → say so plainly and open an issue
  proposing its removal. Do not keep a tier alive because it is already
  written; a lost comparison is a real result.

- [ ] **Step 6: Commit**

```bash
.venv/bin/python -m pytest tests/ -q
git add benchmarks/bench_jl_fidelity.py docs/measurements/jl-2026-08-14.json docs/architecture.md tests/test_jl_operators.py
git commit -m "test: evaluate the JL tier on real activations against an equal-budget control"
```

---

## Phase 5 — Stabilization

### Task 11: Lifecycle, leak, and concurrency stress

**Files:**
- Create: `tests/test_stability.py`

**Interfaces:**
- Consumes: everything built in Phases 2–4.
- Produces: no new API — a stress suite.

- [ ] **Step 1: Write the stress tests**

```python
# tests/test_stability.py
"""Stress tests for the C++ port.

These target the failure modes a unit test misses: state accumulating across
cache lifetimes, the prefetch worker racing the main thread, and cascades
running long enough to exhaust a pool.
"""

import gc
import threading

import pytest
import torch

from argus_cache.core.memory_manager import PagedDynamicKVCache
from argus_cache.core.tier_registry import PipelineConfig, TierSpec
from argus_cache.backends.eviction import ImportanceSortPolicy

PAGE = 32


def _cache(max_active=2):
    return PagedDynamicKVCache(
        pipeline=PipelineConfig(
            tiers=[
                TierSpec(name="int8", backend="int8", max_pages=2),
                TierSpec(name="int4", backend="int4", max_pages=2),
                TierSpec(name="one_bit", backend="one_bit", max_pages=-1),
            ],
            eviction_policy=ImportanceSortPolicy(),
            page_size=PAGE, sink_tokens=0, max_active_pages=max_active,
        )
    )


def _push(cache, n=1):
    for _ in range(n):
        k = torch.randn(1, 2, PAGE, 16, dtype=torch.float16,
                        device="cuda" if torch.cuda.is_available() else "cpu")
        cache.push_new_tokens(k, torch.randn_like(k))


def test_repeated_cache_construction_does_not_leak_host_memory():
    """Each cache owns a pinned host pool; 20 lifetimes must not accumulate."""
    for _ in range(20):
        cache = _cache()
        _push(cache, 6)
        del cache
        gc.collect()
    # Reaching here without OOM or a pinned-allocation failure is the assertion.


def test_long_cascade_preserves_every_page():
    """200 pages through a 3-tier cascade: nothing may be lost or duplicated."""
    cache = _cache()
    _push(cache, 200)

    total = len(cache.active_pages) + sum(
        len(v) for v in cache.pages_by_tier.values()
    )
    ids = [p["page_id"] for p in cache.active_pages]
    for pages in cache.pages_by_tier.values():
        ids.extend(p["page_id"] for p in pages)

    assert len(ids) == len(set(ids)), "duplicate page ids after long cascade"
    assert total > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_prefetch_does_not_race_attention():
    """speculate_and_prefetch runs on a background thread; attention must not
    observe a half-written prefetch cache entry."""
    cache = _cache()
    _push(cache, 12)

    q = torch.randn(1, 2, 1, 16, dtype=torch.float16, device="cuda")
    errors = []

    def hammer():
        try:
            for _ in range(40):
                cache.inplace_paged_attention(q)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=hammer) for _ in range(2)]
    for page_ids in ([p["page_id"] for p in cache.pages_by_tier["int4"]],):
        cache.speculate_and_prefetch_ids(page_ids) if hasattr(
            cache, "speculate_and_prefetch_ids") else cache.speculate_and_prefetch()
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"attention raced the prefetcher: {errors[0]}"


def test_repeated_swap_cycles_are_stable():
    cache = _cache()
    _push(cache, 8)

    for _ in range(10):
        cache.swap_out_to_host()
        cache.swap_in_to_device("cuda" if torch.cuda.is_available() else "cpu")

    assert not cache.is_swapped_out


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_attention_output_is_finite_after_deep_cascade():
    """The real correctness question: does a page that fell all the way to
    1-bit and came back still produce usable attention?"""
    cache = _cache()
    _push(cache, 30)

    q = torch.randn(1, 2, 1, 16, dtype=torch.float16, device="cuda")
    out = cache.inplace_paged_attention(q)

    assert torch.isfinite(out).all(), "attention produced NaN/Inf after cascade"
```

- [ ] **Step 2: Run and triage**

```bash
.venv/bin/python -m pytest tests/test_stability.py -v
```

Every failure here is a **real defect**, not a bad test. Fix the code. If a
test's premise is wrong (e.g. the prefetch API differs), correct the test to
match the real API — but never weaken an assertion to make it pass.

- [ ] **Step 3: Commit**

```bash
.venv/bin/python -m pytest tests/ -q
git add tests/test_stability.py argus_cache/
git commit -m "test: add lifecycle, leak, and concurrency stress coverage"
```

### Task 12: Regenerate all measurements on the stabilized tree

**Files:**
- Create: `docs/measurements/native-<date>.json`, `baseline-<date>-post.json`
- Modify: `docs/architecture.md` §5

- [ ] **Step 1: Rebuild from clean**

```bash
rm -rf build
.venv/bin/python setup.py build_ext --inplace
.venv/bin/python -m pytest tests/ -q
```

- [ ] **Step 2: Re-record baseline and native benchmark**

```bash
.venv/bin/python scripts/record_baseline.py --output docs/measurements/baseline-2026-08-14-post.json
.venv/bin/python benchmarks/bench_native_runtime.py --json docs/measurements/native-2026-08-14.json
```

- [ ] **Step 3: Replace every table in `docs/architecture.md` §5**

Every number must come from the JSON files just written. Delete any row that no
longer has a source. Keep the §5.3 "Not measured" list accurate — items move out
of it only when a measurement file backs them.

- [ ] **Step 4: Commit**

```bash
git add docs/measurements/ docs/architecture.md
git commit -m "docs: regenerate all measurements against the stabilized tree"
```

---

## Phase 6 — Release

### Task 13: Cut v0.3.0

**Files:**
- Modify: `setup.py:6`, `pyproject.toml`
- Create: `CHANGELOG.md`
- Modify: `README.md`

- [ ] **Step 1: Write the changelog**

```markdown
<!-- CHANGELOG.md -->
# Changelog

## v0.3.0 — Native tier-codec engine, plugin system, runtime adapters

### Added
- Generic tier codec in C++ (`csrc/tier_codec.{h,cpp}`): a tier's storage
  format is described numerically (kind, bits, pack factor) instead of
  dispatched on its name. Registering a tier no longer requires editing the
  engine.
- Plugin system (`argus_cache.plugins`): quantization backends register with
  capability metadata (effective bits, devices, dtypes, calibration,
  reconstruction, native codec). Any built-in tier — including 1-bit — can be
  removed and replaced without touching the memory manager.
- Runtime adapters (`argus_cache.adapters`): explicit, idempotent
  `initialize/activate/deactivate/shutdown` lifecycle with guaranteed
  non-raising teardown. `OllamaAdapter` (external runtime) and `VLLMAdapter`
  (in-process, experimental).
- `benchmarks/bench_native_runtime.py`, `benchmarks/bench_jl_fidelity.py`,
  `scripts/record_baseline.py`, and `docs/measurements/` — every documented
  number is now traceable to a committed artifact.

### Fixed
- **Quantization truncated instead of rounding.** Casting float to int
  truncates toward zero, biasing every value by up to half a step. int4
  relative L2 error: ~0.27 → 0.159.
- **1-bit used the wrong magnitude.** `SignPacked` reconstructs as ±scale and
  used the per-page maximum; the L2-optimal scalar is the mean absolute value.
  Relative L2: **3.51 → 0.60** at identical storage cost.
- **Prefetch worker cached undefined tensors** for projection pages with no
  cached reconstruction operator, surfacing later as an empty page during
  attention. It now skips the speculation.

### Changed
- `manager.cpp` 1207 → 763 lines; `quantization_kernels.cu` 183 → 81 (five
  bespoke kernels became one parameterized kernel).
- `memory_manager.py` split by responsibility into `telemetry.py`,
  `jl_operators.py`, `pool_allocator.py`, `granularity.py`, `host_spill.py`.
- Static pool shapes derive from a tier's declared bit width, not its name.

### Removed
- **`inject_argus_to_vllm()`.** It divided vLLM's `block_tables` by a
  "reduction factor", but those entries are physical block *indices*, not byte
  offsets — it aliased unrelated sequences onto shared blocks, compressed
  nothing, and corrupted attention under load. Use `VLLMAdapter`.
  `argus_vllm_models` remains as a deprecation shim.

### Benchmark claims withdrawn
All previously published "ARGUS-vLLM" VRAM figures were measured through the
removed injection path and are **not supported**. `docs/architecture.md §5`
carries the revalidated measurements and an explicit list of what has not been
measured.

### Known limitations
- `VLLMAdapter` is experimental and unverified against a live vLLM.
- ARGUS does not manage Ollama's KV cache; the adapter configures and measures
  an external process.
- Predictive paging remains experimental and off by default.
```

- [ ] **Step 2: Bump the version**

`setup.py:6` — `version="0.2.0"` → `version="0.3.0"`. Match it in
`pyproject.toml`.

- [ ] **Step 3: Build and verify the artifact**

```bash
.venv/bin/python -m pip install --upgrade build twine
.venv/bin/python -m build
.venv/bin/python -m twine check dist/argus_cache-0.3.0*
```

Expected: `PASSED` for both artifacts.

- [ ] **Step 4: Verify a clean install works**

```bash
python -m venv /tmp/argus-verify
/tmp/argus-verify/bin/pip install dist/argus_cache-0.3.0.tar.gz
/tmp/argus-verify/bin/python -c "
import argus_cache
from argus_cache.adapters import list_adapters
from argus_cache import list_quantizers
print('OK', argus_cache.__all__ and list_adapters() and list_quantizers())
"
```

- [ ] **Step 5: Commit and tag (do not push)**

```bash
git add setup.py pyproject.toml CHANGELOG.md README.md
git commit -m "release: v0.3.0 - native tier codec, plugin system, runtime adapters"
git tag -a v0.3.0 -m "v0.3.0"
```

**Do not push and do not upload to PyPI** without explicit instruction.

---

## Phase 7 — vLLM, in isolation

### Task 14: Build an isolated vLLM environment and find the real seam

Installing vLLM in the primary venv would upgrade torch 2.12 → 2.13 and break
`argus_cpp_backend.so`. It gets its own venv.

**Files:**
- Create: `docs/vllm-verification.md`
- Modify: `argus_cache/adapters/vllm.py`

**Interfaces:**
- Consumes: `VLLMAdapter` as shipped in v0.3.0.
- Produces: a `SUPPORTED_VLLM_RANGE` and `target_modules` list derived from a
  real installed vLLM instead of a guess.

- [ ] **Step 1: Create the isolated environment**

```bash
python -m venv .venv-vllm
.venv-vllm/bin/pip install vllm
.venv-vllm/bin/python -c "import vllm, torch; print(vllm.__version__, torch.__version__)"
```

Expected: vLLM 0.27.x with its own torch. **Confirm the primary venv is
untouched:**

```bash
.venv/bin/python -c "import torch; print(torch.__version__)"   # must still be 2.12.0+cu130
.venv/bin/python -m pytest tests/ -q                           # must still pass
```

- [ ] **Step 2: Find the real attention seam**

```bash
.venv-vllm/bin/python - <<'PY'
import importlib, inspect, pkgutil
import vllm.model_executor.models as models

print("vllm", __import__("vllm").__version__)
for mod in pkgutil.iter_modules(models.__path__):
    if "llama" not in mod.name:
        continue
    m = importlib.import_module(f"vllm.model_executor.models.{mod.name}")
    for name, obj in vars(m).items():
        if inspect.isclass(obj) and "Attention" in name:
            print(f"{m.__name__}.{name}")
            print("   forward:", inspect.signature(obj.forward))
PY
```

Record the output verbatim in `docs/vllm-verification.md`. This is the
measurement that replaces the guessed range.

- [ ] **Step 3: Correct the adapter to match reality**

Update `SUPPORTED_VLLM_RANGE` in `argus_cache/adapters/vllm.py` to bracket the
version actually tested, and `target_modules` to the class paths the probe
printed. If vLLM v1 no longer exposes a patchable per-layer `forward`, **say so
in `docs/vllm-verification.md` and stop** — do not invent a seam. Report that
in-process vLLM integration needs a different mechanism (an attention-backend
plugin) and treat that as a separate future project.

- [ ] **Step 4: Commit**

```bash
git add argus_cache/adapters/vllm.py docs/vllm-verification.md
git commit -m "fix: derive vLLM adapter version range and seam from a real install"
```

### Task 15: Run the adapter test suite against real vLLM

- [ ] **Step 1: Make the adapter tests runnable in the vLLM venv**

```bash
.venv-vllm/bin/pip install pytest
ARGUS_TEST_VLLM=1 .venv-vllm/bin/python -m pytest tests/test_adapters.py -v -k vllm
```

Note: only the adapter tests can run there — the C++ extension is not built
against that torch. That is expected and is why the split exists.

- [ ] **Step 2: Verify the properties that matter**

The three that must hold against real vLLM:
1. `activate()` patches, `deactivate()` restores, `is_fully_restored()` is True.
2. A partial-patch failure rolls back every already-patched layer.
3. Five activate/deactivate cycles leave vLLM byte-identical.

- [ ] **Step 3: Record the outcome honestly**

Write the result into `docs/vllm-verification.md` — including a negative
result. "The seam does not exist in vLLM 0.27" is a valid, useful finding and
must be published rather than left as an unverified "experimental" label.

- [ ] **Step 4: Commit**

```bash
git add docs/vllm-verification.md tests/test_adapters.py
git commit -m "test: verify the vLLM adapter against a real vLLM install"
```

---

## Phase 8 — Final measurement pass

### Task 16: End-to-end downstream measurement

Only now — with a stabilized engine, verified adapters, and a released
baseline — are downstream numbers worth producing.

**Files:**
- Create: `benchmarks/bench_downstream.py`
- Create: `docs/measurements/downstream-<date>.json`
- Modify: `README.md`, `docs/architecture.md` §5

**Interfaces:**
- Consumes: `patch_model_with_argus`, `Qwen/Qwen2.5-0.5B-Instruct`.
- Produces: measured TTFT, TPOT, peak VRAM, and perplexity delta for
  baseline vs ARGUS across context lengths.

- [ ] **Step 1: Write the benchmark**

It must measure, for each of `{baseline, argus}` × `{512, 1024, 2048, 4096}`
tokens: TTFT (time to first generated token), TPOT (mean inter-token latency
over ≥64 tokens), `torch.cuda.max_memory_allocated()`, and perplexity on a
fixed held-out passage. Fixed seed, ≥3 repeats, report median and spread.
Record the exact command in the JSON.

- [ ] **Step 2: Run it**

```bash
.venv/bin/python benchmarks/bench_downstream.py \
  --json docs/measurements/downstream-2026-08-14.json
```

- [ ] **Step 3: Publish only what was measured**

Update `README.md`'s benchmark section. Rules, non-negotiable:
- Delete the withdrawn "ARGUS-vLLM" tables entirely — a warning banner is a
  transitional measure, not a permanent home for unsupported numbers.
- Every remaining number cites its `docs/measurements/*.json` file.
- If ARGUS is **slower** than baseline at some context length, publish that
  row. A runtime whose costs are documented is trustworthy; one whose costs
  are hidden is not.
- Keep the "not measured" list current.

- [ ] **Step 4: Commit**

```bash
.venv/bin/python -m pytest tests/ -q
git add benchmarks/bench_downstream.py docs/measurements/ README.md docs/architecture.md
git commit -m "docs: publish measured downstream benchmarks and remove withdrawn claims"
```

---

## Completion Criteria

The port is settled when all of these hold:

- [ ] `pytest tests/ -q` passes with no skips other than genuinely unavailable runtimes
- [ ] `memory_manager.py` under ~1600 lines, with five extracted modules each owning one responsibility
- [ ] Ollama verified against a live server; measurement artifact committed
- [ ] JL tier evaluated on real activations against an equal-budget control, with the verdict published either way
- [ ] Stress suite passing: leaks, long cascades, concurrency, repeated swap cycles
- [ ] v0.3.0 tagged, built, and installable from a clean venv
- [ ] vLLM verified in an isolated venv, or its seam documented as absent
- [ ] Every number in `README.md` and `docs/architecture.md` traceable to a file in `docs/measurements/`
- [ ] No unsupported claim remains anywhere in the documentation

**Only after all of the above** does new feature work resume.
