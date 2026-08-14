"""Runtime adapter tests.

Neither Ollama nor vLLM is assumed to be installed. The adapters take an
injected transport / resolve their targets by dotted path, so their lifecycle,
failure handling, and rollback behavior are testable against fakes — which is
what most of these tests exercise. Tests that need the real runtime skip.

The property that matters most here: an adapter failure must never leave the
runtime half-patched or ARGUS's state disturbed.
"""

import os
import sys
import types

import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from argus_cache.adapters import (
    AdapterError,
    AdapterState,
    RuntimeKind,
    get_adapter,
    list_adapters,
)
from argus_cache.adapters.ollama import OllamaAdapter, OllamaGeneration
from argus_cache.adapters.vllm import (
    PROBED_VLLM_VERSIONS,
    SUPPORTED_VLLM_RANGE,
    VLLMAdapter,
    inject_argus_to_vllm,
)


# ── registry ────────────────────────────────────────────────────────────────


def test_adapter_registry_lists_both_runtimes():
    assert list_adapters() == ["ollama", "vllm"]


def test_unknown_adapter_raises():
    with pytest.raises(AdapterError, match="unknown runtime adapter"):
        get_adapter("sglang")


def test_adapters_declare_honest_capabilities():
    """Ollama is an external process; it must not claim to be ARGUS-cached."""
    assert get_adapter("ollama").capabilities.kind is RuntimeKind.EXTERNAL
    assert get_adapter("ollama").capabilities.manages_kv_cache is False
    assert get_adapter("vllm").capabilities.kind is RuntimeKind.IN_PROCESS
    assert get_adapter("vllm").capabilities.manages_kv_cache is False


# ── Ollama ──────────────────────────────────────────────────────────────────


class FakeTransport:
    """Scriptable stand-in for HttpTransport."""

    def __init__(self, responses=None, fail_on=None):
        self.responses = responses or {}
        self.fail_on = fail_on or set()
        self.calls = []

    def request(self, method, path, payload=None):
        self.calls.append((method, path, payload))
        if path in self.fail_on:
            raise AdapterError(f"simulated failure for {path}")
        return self.responses.get(path, {})


def _ok_transport(**overrides):
    responses = {
        "/api/version": {"version": "0.5.7"},
        "/api/tags": {"models": [{"name": "llama3.2:latest"}]},
        "/api/generate": {
            "response": "hello",
            "model": "llama3.2",
            "prompt_eval_duration": 120_000_000,
            "eval_duration": 500_000_000,
            "prompt_eval_count": 12,
            "eval_count": 40,
        },
        "/api/ps": {"models": [{"name": "llama3.2", "size": 100, "size_vram": 80}]},
    }
    responses.update(overrides)
    return FakeTransport(responses)


def test_ollama_lifecycle_is_explicit_and_idempotent():
    adapter = OllamaAdapter(model="llama3.2", transport=_ok_transport())
    assert adapter.state is AdapterState.CREATED

    adapter.initialize()
    assert adapter.state is AdapterState.INITIALIZED
    adapter.initialize()  # idempotent
    assert adapter.state is AdapterState.INITIALIZED

    adapter.activate()
    assert adapter.is_active
    adapter.activate()  # idempotent
    assert adapter.is_active

    adapter.deactivate()
    assert adapter.state is AdapterState.INITIALIZED
    adapter.deactivate()  # idempotent

    adapter.shutdown()
    assert adapter.state is AdapterState.SHUTDOWN
    adapter.shutdown()  # idempotent


def test_ollama_reports_server_version_and_model():
    adapter = OllamaAdapter(model="llama3.2", transport=_ok_transport())
    adapter.initialize()
    assert adapter.server_version == "0.5.7"


def test_ollama_missing_model_fails_with_actionable_message():
    transport = _ok_transport(**{"/api/tags": {"models": [{"name": "mistral:latest"}]}})
    adapter = OllamaAdapter(model="llama3.2", transport=transport)

    with pytest.raises(AdapterError, match="ollama pull llama3.2"):
        adapter.initialize()
    assert adapter.state is AdapterState.FAILED


def test_ollama_unreachable_server_fails_cleanly():
    adapter = OllamaAdapter(
        model="llama3.2", transport=FakeTransport(fail_on={"/api/version"})
    )

    with pytest.raises(AdapterError):
        adapter.activate()
    assert adapter.state is AdapterState.FAILED
    assert not adapter.is_active


def test_ollama_generate_requires_active_adapter():
    adapter = OllamaAdapter(model="llama3.2", transport=_ok_transport())
    adapter.initialize()

    with pytest.raises(AdapterError, match="not active"):
        adapter.generate("hi")


def test_ollama_generate_converts_nanosecond_timings():
    adapter = OllamaAdapter(model="llama3.2", transport=_ok_transport())
    adapter.activate()

    result = adapter.generate("hi", max_tokens=40)

    assert isinstance(result, OllamaGeneration)
    assert result.text == "hello"
    assert result.prompt_eval_seconds == pytest.approx(0.12)
    assert result.eval_seconds == pytest.approx(0.5)
    assert result.tokens_per_second == pytest.approx(80.0)


def test_ollama_tokens_per_second_is_none_without_server_timings():
    adapter = OllamaAdapter(
        model="llama3.2", transport=_ok_transport(**{"/api/generate": {"response": "x"}})
    )
    adapter.activate()
    assert adapter.generate("hi").tokens_per_second is None


def test_ollama_kv_cache_type_is_validated():
    with pytest.raises(ValueError, match="kv_cache_type"):
        OllamaAdapter(model="m", kv_cache_type="q2_k")


def test_ollama_options_are_passed_through():
    adapter = OllamaAdapter(
        model="llama3.2", num_ctx=8192, kv_cache_type="q8_0", transport=_ok_transport()
    )
    options = adapter.effective_options()

    assert options["num_ctx"] == 8192
    assert options["cache_type_k"] == options["cache_type_v"] == "q8_0"


def test_ollama_telemetry_never_claims_argus_owns_the_cache():
    adapter = OllamaAdapter(model="llama3.2", transport=_ok_transport())
    adapter.activate()

    telemetry = adapter.telemetry()
    assert telemetry["argus_manages_kv_cache"] is False
    assert telemetry["loaded_models"][0]["size_vram_bytes"] == 80


def test_ollama_telemetry_failure_is_swallowed():
    """Telemetry is best-effort: a dashboard poll must not raise."""
    transport = _ok_transport()
    adapter = OllamaAdapter(model="llama3.2", transport=transport)
    adapter.activate()
    transport.fail_on.add("/api/ps")

    assert adapter.telemetry() == {}


def test_ollama_deactivate_survives_transport_failure():
    """Teardown must complete even when the server has gone away."""
    transport = _ok_transport()
    adapter = OllamaAdapter(model="llama3.2", transport=transport)
    adapter.activate()

    transport.fail_on.add("/api/generate")
    adapter.deactivate()  # must not raise

    assert adapter.state is AdapterState.INITIALIZED
    assert adapter.last_error is not None


def test_ollama_context_manager_shuts_down():
    adapter = OllamaAdapter(model="llama3.2", transport=_ok_transport())
    with adapter as a:
        assert a.is_active
    assert adapter.state is AdapterState.SHUTDOWN


def test_ollama_repeated_init_cleanup_cycles():
    """Repeated lifecycle cycles must not accumulate state."""
    adapter = OllamaAdapter(model="llama3.2", transport=_ok_transport())
    for _ in range(5):
        adapter.initialize()
        adapter.activate()
        adapter.deactivate()
    adapter.shutdown()
    assert adapter.state is AdapterState.SHUTDOWN


# ── vLLM ────────────────────────────────────────────────────────────────────


@pytest.fixture
def fake_vllm(monkeypatch):
    """Install a minimal fake vLLM package for the compatibility probe."""
    vllm = types.ModuleType("vllm")
    vllm.__version__ = "0.9.0"
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    return vllm


def test_legacy_injection_helper_refuses_with_an_explanation():
    """The old block_tables division was incorrect; it must not silently run."""
    with pytest.raises(AdapterError, match="block-table"):
        inject_argus_to_vllm()


def test_vllm_missing_is_reported_clearly(monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm", None)
    adapter = VLLMAdapter(cache_factory=lambda: object())

    with pytest.raises(AdapterError, match="not installed"):
        adapter.initialize()


def test_vllm_has_no_guessed_supported_range():
    assert SUPPORTED_VLLM_RANGE == ()
    assert "0.27.1" in PROBED_VLLM_VERSIONS


def test_vllm_probe_refuses_false_forward_hook_integration(fake_vllm):
    adapter = VLLMAdapter(cache_factory=lambda: object())

    with pytest.raises(AdapterError, match="KVConnectorBase_V1"):
        adapter.initialize()
    assert adapter.state is AdapterState.FAILED
    assert adapter.vllm_version == "0.9.0"
    assert adapter.is_fully_restored()
    assert adapter.cache is None


def test_vllm_unsafe_override_cannot_reenable_noop_patch(fake_vllm):
    adapter = VLLMAdapter(cache_factory=lambda: object(), strict_version=False)
    with pytest.raises(AdapterError, match="activation is refused"):
        adapter.activate()
    assert adapter.is_fully_restored()


def test_vllm_probe_never_constructs_an_argus_cache(fake_vllm):
    calls = []
    adapter = VLLMAdapter(cache_factory=lambda: calls.append("constructed"))
    with pytest.raises(AdapterError):
        adapter.initialize()
    assert calls == []


def test_vllm_telemetry_never_claims_cache_ownership(fake_vllm):
    adapter = VLLMAdapter()
    with pytest.raises(AdapterError):
        adapter.initialize()
    assert adapter.telemetry() == {
        "runtime": "vllm",
        "vllm_version": "0.9.0",
        "argus_manages_kv_cache": False,
        "integration_available": False,
    }


def test_vllm_failed_probe_can_be_shutdown_idempotently(fake_vllm):
    adapter = VLLMAdapter()
    with pytest.raises(AdapterError):
        adapter.initialize()
    adapter.shutdown()
    adapter.shutdown()
    assert adapter.state is AdapterState.SHUTDOWN
    assert adapter.is_fully_restored()


def test_vllm_target_paths_are_never_monkey_patched(fake_vllm):
    sentinel = type("Attention", (), {"forward": lambda self: "native"})
    original = sentinel.forward
    adapter = VLLMAdapter(target_modules=["fake.module.Attention"])
    with pytest.raises(AdapterError):
        adapter.initialize()
    assert sentinel.forward is original


def test_vllm_repeated_probes_do_not_mutate_runtime_module(fake_vllm):
    before = dict(vars(fake_vllm))
    for _ in range(5):
        with pytest.raises(AdapterError):
            VLLMAdapter().initialize()
    assert vars(fake_vllm) == before


# ── live runtimes (skipped unless present) ──────────────────────────────────


def _ollama_running() -> bool:
    try:
        OllamaAdapter(model="_probe_").transport.request("GET", "/api/version")
        return True
    except Exception:
        return False


def _live_model() -> str:
    """Pick a model that is actually on the server.

    Hardcoding a tag makes the live suite fail on any machine that pulled a
    different one -- which is a property of the test environment, not of the
    adapter. Prefer an explicit override, else the first installed model.
    """
    override = os.environ.get("ARGUS_OLLAMA_MODEL")
    if override:
        return override
    try:
        tags = OllamaAdapter(model="_probe_").transport.request("GET", "/api/tags")
        models = [m.get("name") for m in tags.get("models", []) if m.get("name")]
        return models[0] if models else "qwen2.5:0.5b"
    except Exception:
        return "qwen2.5:0.5b"


@pytest.mark.skipif(not _ollama_running(), reason="no Ollama server on localhost:11434")
def test_ollama_live_version_probe():
    adapter = OllamaAdapter(model=_live_model())
    adapter.initialize()
    assert adapter.server_version


@pytest.mark.skipif(not _ollama_running(), reason="no Ollama server on localhost:11434")
def test_ollama_live_generate_reports_real_timings():
    with OllamaAdapter(model=_live_model(), num_ctx=2048) as ollama:
        result = ollama.generate("Count to three.", max_tokens=24)

    assert result.text.strip(), "server returned an empty completion"
    assert result.eval_tokens and result.eval_tokens > 0
    assert result.tokens_per_second and result.tokens_per_second > 0
    # Client wall time must bound the server's own decode time.
    assert result.wall_seconds >= result.eval_seconds


@pytest.mark.skipif(not _ollama_running(), reason="no Ollama server on localhost:11434")
def test_ollama_live_missing_model_is_rejected():
    """The adapter must fail with an actionable message, not a raw 404 --
    a missing model is the most common live failure and the fix is one
    command the error should name."""
    adapter = OllamaAdapter(model="definitely-not-a-real-model")
    with pytest.raises(AdapterError, match="ollama pull"):
        adapter.initialize()


@pytest.mark.skipif(not _ollama_running(), reason="no Ollama server on localhost:11434")
def test_ollama_live_telemetry_reports_loaded_model():
    with OllamaAdapter(model=_live_model()) as ollama:
        ollama.generate("hi", max_tokens=4)
        telemetry = ollama.telemetry()

    # Ollama runs in its own process: ARGUS cannot own its KV cache, and the
    # adapter must keep saying so rather than implying credit for its numbers.
    assert telemetry["argus_manages_kv_cache"] is False
    assert telemetry["loaded_models"], "server reported no loaded model"


@pytest.mark.skipif(not _ollama_running(), reason="no Ollama server on localhost:11434")
def test_ollama_live_repeated_cycles_are_stable():
    adapter = OllamaAdapter(model=_live_model())
    for _ in range(3):
        adapter.activate()
        assert adapter.generate("ok", max_tokens=4).text is not None
        adapter.deactivate()
    adapter.shutdown()


@pytest.mark.skipif(
    "vllm" not in sys.modules and not os.environ.get("ARGUS_TEST_VLLM"),
    reason="vLLM not installed (set ARGUS_TEST_VLLM=1 to force)",
)
def test_vllm_live_probe_fails_closed():
    adapter = VLLMAdapter(cache_factory=lambda: object())
    with pytest.raises(AdapterError, match="KVConnectorBase_V1"):
        adapter.initialize()
    assert adapter.vllm_version
    assert adapter.capabilities.manages_kv_cache is False
    assert adapter.is_fully_restored()
