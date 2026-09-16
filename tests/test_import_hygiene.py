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
    """scratch/ holds throwaway scripts named test_*; collecting them would
    make the suite's pass count meaningless."""
    config = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    assert "norecursedirs" in config
    assert "scratch" in config


def test_scratch_is_documented_as_unsupported():
    readme = REPO / "scratch" / "README.md"
    assert readme.exists(), "scratch/ must state that it is unsupported"
    assert "not part of the package" in readme.read_text(encoding="utf-8").lower()


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
