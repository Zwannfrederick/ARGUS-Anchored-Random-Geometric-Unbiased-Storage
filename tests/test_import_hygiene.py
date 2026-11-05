"""There must be exactly one copy of each module, under argus_cache.

A second copy of a class silently diverges. The root attention wrapper had
drifted 121 lines from the canonical one in argus_cache and lost `pipeline=`
and `balloon_driver=` support entirely -- so anything importing it got a cache
that ignored its own tier configuration, with no error to say so. Root `core/`
and `models/` were reduced to re-export shims to stop that, and removed in
v0.5.2 once nothing imported them: a package that does not exist cannot drift,
and installing the tree no longer puts top-level `core`/`models` names into a
user's environment. These tests hold that line.
"""

import inspect
import re
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parent.parent


def test_no_second_copy_at_the_repository_root():
    """The duplicate that drifted lived here; it must not come back."""
    for name in ("core", "models"):
        assert not (REPO / name).exists(), (
            f"{name}/ is back at the repository root, where it can diverge from "
            "argus_cache and be installed as a top-level package"
        )


def test_only_argus_cache_is_packaged():
    """The stray top-level package shipped in 0.5.0 and 0.5.1 came from here.

    find_packages() walks the repository, so any directory with an __init__.py
    beside argus_cache/ becomes a top-level package in the user's environment.
    Verifying the built sdist would be stronger, but building it runs setup.py,
    and CUDAExtension resolves CUDA library paths as it is constructed, so that
    check needs CUDA_HOME and stays local.
    """
    from setuptools import find_packages

    top_level = {name.split(".")[0] for name in find_packages(where=str(REPO))}
    assert top_level == {"argus_cache"}, (
        f"packaging would install {sorted(top_level)}; only argus_cache belongs "
        "there, and anything else collides with that name in a user's environment"
    )


def test_nothing_imports_the_removed_root_packages():
    offenders = []
    for path in list((REPO / "tests").rglob("*.py")) + list((REPO / "benchmarks").rglob("*.py")):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if re.match(r"\s*(from|import) (core|models)\.", line):
                offenders.append(f"{path.relative_to(REPO)}:{number}")
    assert not offenders, "import argus_cache.core/.models instead: " + ", ".join(offenders)


def test_canonical_wrapper_keeps_the_arguments_the_copy_dropped():
    """The regression the divergent copy caused: pipeline= was silently dropped."""
    from argus_cache.models.attention_wrapper import PagedDynamicQuantizedCache

    params = inspect.signature(PagedDynamicQuantizedCache.__init__).parameters
    assert "pipeline" in params
    assert "balloon_driver" in params


def test_scratch_is_excluded_from_test_collection():
    """scratch/ and experiments/ hold throwaway or archived scripts named test_*;
    collecting them would make the suite's pass count meaningless."""
    config = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    assert "norecursedirs" in config
    assert "scratch" in config
    assert "experiments" in config


def test_scratch_is_documented_as_unsupported():
    readme = REPO / "scratch" / "README.md"
    assert readme.exists(), "scratch/ must state that it is unsupported"
    assert "not part of the package" in readme.read_text(encoding="utf-8").lower()
    exp_readme = REPO / "experiments" / "README.md"
    assert exp_readme.exists(), "experiments/ must document archived research/experiments"


def test_adapter_import_does_not_load_native_cache_runtime():
    """Adapters must be testable in a vLLM env with a different torch ABI."""
    code = (
        "import sys; import argus_cache.adapters; "
        "assert 'argus_cache.core.memory_manager' not in sys.modules; "
        "assert 'argus_cpp_backend' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_every_annotation_in_the_package_resolves():
    """Annotations must name imported types.

    Python 3.14 defers annotation evaluation (PEP 649), so an unimported
    ``Dict`` passes here and raises NameError at import time on 3.10-3.13:
    v0.7.0 shipped exactly that in ``PagedDynamicKVCache.snapshot``.
    """
    import inspect
    import typing

    import argus_cache.core.memory_manager as mm

    for _, cls in inspect.getmembers(mm, inspect.isclass):
        if cls.__module__ != mm.__name__:
            continue
        for _, fn in inspect.getmembers(cls, inspect.isfunction):
            typing.get_type_hints(fn)


def test_cuda_event_sync_needs_no_undefined_names():
    """sync_pending_cuda_events used ``torch`` without importing it."""
    from types import SimpleNamespace

    from argus_cache.core.telemetry import CacheTelemetry

    class _Event:
        def elapsed_time(self, other):
            return 1.5

    cache = SimpleNamespace(pending_cuda_events=[(_Event(), _Event())], dequant_times_ms=[])
    CacheTelemetry(cache).sync_pending_cuda_events()
