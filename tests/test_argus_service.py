"""Service configuration resolution.

The waybar module and any other launcher that cannot pass arguments call
``argus_ctl.sh start`` bare. Without a persisted configuration that invocation
silently takes the untuned path -- Ollama instead of llama-server -- which
gives up the largest measured speedup (12.09 -> 19.14 tok/s from ``--cpu-moe``)
precisely when a user clicks the button rather than typing a command.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "argus_service",
    Path(__file__).resolve().parent.parent / "scripts" / "argus_service.py",
)
argus_service = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(argus_service)


class _Args:
    """Stand-in for the argparse namespace, with everything unset."""

    def __init__(self, **kwargs):
        self.llama_server_model = None
        self.num_ctx = None
        self.kv_cache_type = None
        self.llama_server_port = None
        self.cwd = None
        for key, value in kwargs.items():
            setattr(self, key, value)


def test_config_supplies_settings_when_no_flags_were_given(tmp_path):
    """A bare `start` picks up the persisted backend, which is what waybar does."""
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "llama_server_model": "/models/qwen.gguf",
        "num_ctx": 229376,
        "kv_cache_type": "iq4_nl",
    }))

    args = argus_service.apply_config(_Args(), config_path=config)

    assert args.llama_server_model == "/models/qwen.gguf"
    assert args.num_ctx == 229376
    assert args.kv_cache_type == "iq4_nl"


def test_explicit_flags_win_over_the_config(tmp_path):
    """Typing a flag must override the stored value, not be silently ignored."""
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"num_ctx": 229376, "kv_cache_type": "iq4_nl"}))

    args = argus_service.apply_config(
        _Args(num_ctx=32768, kv_cache_type="q8_0"), config_path=config
    )

    assert args.num_ctx == 32768
    assert args.kv_cache_type == "q8_0"


def test_absent_config_leaves_built_in_defaults(tmp_path):
    """No config is a normal state, not an error."""
    args = argus_service.apply_config(_Args(), config_path=tmp_path / "nope.json")

    assert args.llama_server_model is None
    assert args.num_ctx == argus_service.DEFAULT_LLAMA_SERVER_CTX


def test_unreadable_config_does_not_prevent_startup(tmp_path):
    """A corrupt config must degrade to defaults rather than crash the service.

    This runs from a window-manager click with output discarded, so an
    exception here would look like the button simply not working.
    """
    config = tmp_path / "config.json"
    config.write_text("{ this is not json")

    args = argus_service.apply_config(_Args(), config_path=config)

    assert args.num_ctx == argus_service.DEFAULT_LLAMA_SERVER_CTX


def test_config_rejects_an_unusable_kv_type(tmp_path):
    """A typo in the stored KV type is reported, not carried into llama-server.

    llama.cpp fails late and obscurely on an unknown cache type, so the value
    is checked where it can still be attributed to the config file.
    """
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"kv_cache_type": "q3_k"}))

    with pytest.raises(ValueError, match="kv_cache_type"):
        argus_service.apply_config(_Args(), config_path=config)


def test_get_vram_metrics_skips_nvidia_smi_when_dgpu_is_suspended(tmp_path, monkeypatch):
    """Waybar polling when ARGUS is OFF must not wake an Optimus laptop dGPU from D3cold."""
    power_status = tmp_path / "runtime_status"
    power_status.write_text("suspended\n")
    monkeypatch.setattr(argus_service, "_DGPU_RUNTIME_STATUS", power_status)

    called = False

    def _forbid_run(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("nvidia-smi should not run while dGPU is suspended")

    monkeypatch.setattr(argus_service.subprocess, "run", _forbid_run)
    metrics = argus_service.get_vram_metrics(allow_wake=False)

    assert called is False
    assert metrics["available"] is False
