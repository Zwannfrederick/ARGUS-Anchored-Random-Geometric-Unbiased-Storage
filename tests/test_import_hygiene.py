"""The root shim packages must re-export, never re-implement.

A second copy of a class silently diverges. The root attention wrapper had
drifted 121 lines from the canonical one in argus_cache and lost `pipeline=`
and `balloon_driver=` support entirely -- so anything importing it got a cache
that ignored its own tier configuration, with no error to say so.
"""

import importlib
import inspect
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def test_root_models_shim_is_the_same_class():
    from argus_cache.models.attention_wrapper import PagedDynamicQuantizedCache as Canonical
    from models.attention_wrapper import PagedDynamicQuantizedCache as ViaShim

    assert ViaShim is Canonical, "root models/ is a divergent copy, not a shim"


def test_root_core_shims_are_the_same_objects():
    for module in ("memory_manager", "quantization", "triton_kernels", "balloon_driver"):
        canonical = importlib.import_module(f"argus_cache.core.{module}")
        shim = importlib.import_module(f"core.{module}")
        shared = set(vars(canonical)) & set(vars(shim))
        interesting = [
            name
            for name in shared
            if not name.startswith("_")
            and (inspect.isclass(getattr(canonical, name)) or inspect.isfunction(getattr(canonical, name)))
        ]
        assert interesting, f"core.{module} re-exported nothing"
        for name in interesting:
            assert getattr(shim, name) is getattr(canonical, name), (
                f"core.{module}.{name} diverged from argus_cache"
            )


def test_shim_accepts_pipeline_argument():
    """The regression the divergent copy caused: pipeline= was silently dropped."""
    from models.attention_wrapper import PagedDynamicQuantizedCache

    params = inspect.signature(PagedDynamicQuantizedCache.__init__).parameters
    assert "pipeline" in params
    assert "balloon_driver" in params
