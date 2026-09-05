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

from argus_cache.adapters.ollama import (
    DEFAULT_LLAMA_SERVER_CTX,
    llama_server_command,
    recommended_server_environment,
)
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


#: Cold-loading a 20 GB model costs the better part of a minute before any
#: token is produced, which exceeds the adapter's 60 s default. That is a
#: property of the machine's model, not of the adapter, so the live tests state
#: their own budget rather than the adapter lowering its guard for everyone.
LIVE_TIMEOUT = 300.0


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
    adapter = OllamaAdapter(model=_live_model(), timeout=LIVE_TIMEOUT)
    adapter.initialize()
    assert adapter.server_version


@pytest.mark.skipif(not _ollama_running(), reason="no Ollama server on localhost:11434")
def test_ollama_live_generate_reports_real_timings():
    with OllamaAdapter(
        model=_live_model(), num_ctx=2048, timeout=LIVE_TIMEOUT
    ) as ollama:
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
    with OllamaAdapter(model=_live_model(), timeout=LIVE_TIMEOUT) as ollama:
        ollama.generate("hi", max_tokens=4)
        telemetry = ollama.telemetry()

    # Ollama runs in its own process: ARGUS cannot own its KV cache, and the
    # adapter must keep saying so rather than implying credit for its numbers.
    assert telemetry["argus_manages_kv_cache"] is False
    assert telemetry["loaded_models"], "server reported no loaded model"


@pytest.mark.skipif(not _ollama_running(), reason="no Ollama server on localhost:11434")
def test_ollama_live_repeated_cycles_are_stable():
    adapter = OllamaAdapter(model=_live_model(), timeout=LIVE_TIMEOUT)
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


# ── server tuning knobs ─────────────────────────────────────────────────────


def test_recommended_server_environment_sets_measured_kv_defaults():
    """The KV knobs that were measured to matter are set, not left to folklore.

    Measured on a 4 GB RTX 3050 Ti with qwen3.6-35b-a3b at 262144 context:
    f16 KV gave 2.77 tok/s, q8_0 KV plus flash attention gave 8.63 tok/s. The
    2 GB the quantized cache gives back is what keeps the weights resident
    instead of paging from disk.
    """
    env = recommended_server_environment({})

    assert env["OLLAMA_KV_CACHE_TYPE"] == "q8_0"
    assert env["OLLAMA_FLASH_ATTENTION"] == "1"


def test_recommended_server_environment_preserves_the_caller_environment():
    """Tuning is additive; unrelated variables survive untouched."""
    env = recommended_server_environment({"PATH": "/usr/bin", "HOME": "/home/x"})

    assert env["PATH"] == "/usr/bin"
    assert env["HOME"] == "/home/x"


def test_recommended_server_environment_does_not_override_an_explicit_choice():
    """An operator who already set a KV type keeps it.

    Flash attention is a precondition for a quantized KV cache in llama.cpp, so
    silently replacing a deliberate f16 choice would change behaviour behind
    the operator's back.
    """
    env = recommended_server_environment({"OLLAMA_KV_CACHE_TYPE": "f16"})

    assert env["OLLAMA_KV_CACHE_TYPE"] == "f16"


def test_recommended_server_environment_rejects_an_unsupported_kv_type():
    with pytest.raises(ValueError, match="kv_cache_type"):
        recommended_server_environment({}, kv_cache_type="q3_k")


# ── llama-server launch configuration ───────────────────────────────────────


def test_llama_server_command_keeps_experts_on_cpu():
    """--cpu-moe is the single largest lever on a small GPU, so it is not optional.

    Measured on an RTX 3050 Ti (4 GB) with qwen3.6-35b-a3b at 32k: 12.09 tok/s
    without it, 19.14 tok/s with it. A mixture-of-experts model activates 8 of
    256 experts per token, so pinning the experts to CPU frees VRAM for the
    attention weights and the KV cache, which is what actually needs to be
    resident.
    """
    cmd = llama_server_command(model_path="/m.gguf", port=8080)

    assert "--cpu-moe" in cmd


def test_llama_server_command_requests_a_quantized_kv_cache():
    """A quantized KV cache is what keeps long context inside VRAM.

    Throughput is governed by whether the cache fits on the device: roughly
    18 tok/s while it does and 8.8 tok/s once it spills to host memory. q4_0
    moved the ceiling from 96k to 160k without costing throughput.
    """
    cmd = llama_server_command(model_path="/m.gguf", port=8080)

    assert "-ctk" in cmd and "-ctv" in cmd
    assert cmd[cmd.index("-ctk") + 1] == "q4_0"
    assert cmd[cmd.index("-ctv") + 1] == "q4_0"
    # llama.cpp requires flash attention for a quantized KV cache; without it
    # the request is silently ineffective.
    assert "-fa" in cmd


def test_llama_server_command_carries_model_and_port():
    cmd = llama_server_command(model_path="/models/q.gguf", port=9099)

    assert cmd[cmd.index("--model") + 1] == "/models/q.gguf"
    assert cmd[cmd.index("--port") + 1] == "9099"


def test_llama_server_context_is_configurable():
    cmd = llama_server_command(model_path="/m.gguf", port=8080, num_ctx=32768)

    assert cmd[cmd.index("-c") + 1] == "32768"


def test_llama_server_command_rejects_unsupported_kv_type():
    with pytest.raises(ValueError, match="kv_cache_type"):
        llama_server_command(model_path="/m.gguf", port=8080, kv_cache_type="q3_k")


def test_llama_server_accepts_types_ollama_does_not():
    """llama-server's KV vocabulary is wider than Ollama's, and iq4_nl matters.

    Ollama's OLLAMA_KV_CACHE_TYPE takes only f16/q8_0/q4_0. llama-server also
    takes iq4_nl, which measured 224k of context in 4 GB of VRAM at 16.64 tok/s
    where q4_0 could not load past 160k. Validating llama-server flags against
    Ollama's shorter list rejects the better setting.
    """
    cmd = llama_server_command(model_path="/m.gguf", port=8080, kv_cache_type="iq4_nl")

    assert cmd[cmd.index("-ctk") + 1] == "iq4_nl"


def test_ollama_environment_still_rejects_llama_only_types():
    """The narrower Ollama list is not widened by the llama-server change."""
    with pytest.raises(ValueError, match="kv_cache_type"):
        recommended_server_environment({}, kv_cache_type="iq4_nl")


def test_default_context_leaves_vram_headroom():
    """The shipped context must load reliably, not sit on the VRAM cliff.

    Measured on the 4 GB RTX 3050 Ti with ``--cpu-moe`` and a q4_0 KV cache
    (usable VRAM 3770 MiB):

    ======  =========  ===========================
    ctx     VRAM       outcome
    ======  =========  ===========================
    262144  --         OOM (needs 969 MiB more)
    196608  --         OOM (needs 777 MiB more)
    163840  3696 MiB   98% -- loaded once, aborted
                       once in CUDA on the retry
    131072  3420 MiB   91% -- loaded 2/2
    ======  =========  ===========================

    163840 fits arithmetically and still fails intermittently: at 98% the
    allocation outcome depends on driver overhead and fragmentation rather
    than on anything the caller controls. A default that OOMs some of the time
    is worse than a smaller one that always works, so the shipped value keeps
    real headroom.
    """
    assert DEFAULT_LLAMA_SERVER_CTX <= 131072


def test_llama_server_command_supports_speculative_types():
    cmd = llama_server_command(
        model_path="/m.gguf",
        port=8080,
        spec_type="draft-mtp,ngram-mod",
        spec_draft_n_max=3,
    )
    assert "--spec-type" in cmd
    assert cmd[cmd.index("--spec-type") + 1] == "draft-mtp,ngram-mod"
    assert "--spec-draft-n-max" in cmd
    assert cmd[cmd.index("--spec-draft-n-max") + 1] == "3"


def test_llama_server_command_supports_draft_model():
    cmd = llama_server_command(
        model_path="/m.gguf",
        port=8080,
        spec_type="draft-simple",
        draft_model="/draft.gguf",
    )
    assert "--spec-type" in cmd
    assert cmd[cmd.index("--spec-type") + 1] == "draft-simple"
    assert "--spec-draft-model" in cmd
    assert cmd[cmd.index("--spec-draft-model") + 1] == "/draft.gguf"

