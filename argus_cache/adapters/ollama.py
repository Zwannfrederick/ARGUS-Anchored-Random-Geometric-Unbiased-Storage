"""Ollama runtime adapter.

**Scope, stated up front.** Ollama runs as a separate server process wrapping
llama.cpp, which owns its own KV cache in its own address space. ARGUS cannot
manage that cache — there is no in-process attention path to intercept. Any
claim that ARGUS compresses Ollama's KV cache would be false.

What this adapter legitimately does:

* manages the lifecycle of the connection to an Ollama server explicitly,
  instead of the ad-hoc calls this integration previously consisted of;
* validates that the server is reachable and that a requested model exists,
  before a benchmark or application starts;
* surfaces the KV-cache knobs Ollama *does* expose (context length, and the
  server's own ``OLLAMA_KV_CACHE_TYPE`` quantization setting) as configuration
  rather than environment folklore;
* reports per-request timing and the server's memory accounting, so ARGUS
  benchmarks can compare against it as an external baseline;
* fails closed: every network error is converted to :class:`AdapterError` and
  never touches ARGUS's cache-manager state.

The transport is injected (``transport=``) so the adapter is testable without
a running server, and so callers may substitute their own HTTP client.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from .base import AdapterCapabilities, AdapterError, RuntimeAdapter, RuntimeKind

DEFAULT_HOST = "http://localhost:11434"
DEFAULT_TIMEOUT = 60.0

#: KV cache types Ollama accepts. Anything else is rejected rather than passed
#: through, because llama.cpp fails late and obscurely on an unknown type.
KV_CACHE_TYPES = ("f16", "q8_0", "q4_0")

#: Quantized KV is the default because it was measured to matter, not because
#: it sounds thrifty. See ``recommended_server_environment``.
DEFAULT_SERVER_KV_CACHE_TYPE = "q8_0"

#: llama-server accepts a wider set than Ollama's env var does, so validating
#: llama-server flags against Ollama's shorter list would reject usable types.
#: Note iq4_nl is the same width as q4_0 -- a non-linear codebook, not a smaller
#: one -- so it cannot buy context, only fidelity at equal size. None of the
#: types beyond f16/q8_0/q4_0 have been measured here.
LLAMA_KV_CACHE_TYPES = (
    "f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1",
)


#: Default llama-server binary shipped alongside Ollama.
DEFAULT_LLAMA_SERVER = "/usr/lib/ollama/llama-server"

#: KV width llama-server is started with. q4_0 rather than iq4_nl because q4_0
#: is the rung every number in this module was measured on.
DEFAULT_SERVER_LLAMA_KV_TYPE = "q4_0"

#: Largest context measured to load *reliably* on a 4 GB card with the experts
#: on CPU and a q4_0 KV cache: 3420 of 3770 MiB, 2/2 loads. 163840 fits on paper
#: too -- 3696 MiB -- but at 98% occupancy it aborted inside CUDA on a repeat
#: run, so the ceiling arithmetic allows is not the ceiling that ships. At
#: 131072 the measured decode was 15.8-17.6 tok/s. See
#: ``test_default_context_leaves_vram_headroom``.
DEFAULT_LLAMA_SERVER_CTX = 131072


def llama_server_command(
    *,
    model_path: str,
    port: int,
    num_ctx: int = DEFAULT_LLAMA_SERVER_CTX,
    kv_cache_type: str = "q4_0",
    binary: str = DEFAULT_LLAMA_SERVER,
    host: str = "127.0.0.1",
    spec_type: Optional[str] = None,
    spec_draft_n_max: int = 2,
    draft_model: Optional[str] = None,
    flash_attn: str = "on",
    load_mode: Optional[str] = None,
    spec_draft_p_min: Optional[float] = None,
) -> List[str]:
    """Argument vector for a llama-server tuned for a small-VRAM machine.

    Ollama cannot serve this configuration: it does not expose ``--cpu-moe``,
    which on a mixture-of-experts model is the difference between 12.09 and
    19.14 tok/s (measured, RTX 3050 Ti 4 GB, qwen3.6-35b-a3b at 32k). Running
    llama-server directly is the only way to reach it.

    The three settings that matter, and why:

    * ``--cpu-moe`` pins the experts to host memory. Only 8 of 256 are active
      per token, so they are poor tenants of scarce VRAM; the attention weights
      and KV cache are what benefit from being resident.
    * ``-ctk``/``-ctv`` quantize the KV cache. Whether that cache fits in VRAM
      is the only thing that moves throughput -- about 18 tok/s while it does,
      8.8 tok/s once it spills -- and q4_0 raised the ceiling from 96k to 160k
      at no measured cost in speed.
    * ``-fa`` is a precondition: llama.cpp needs flash attention for a
      quantized KV cache, and silently ignores the request without it.
    * Speculative decoding (MTP + N-Gram) dramatically boosts token generation
      rate on CPU-MoE setups by verifying multiple tokens per RAM read.

    The default is ``q4_0`` because that is the rung the numbers above were
    measured on. ``iq4_nl`` is the same width with a non-linear codebook and
    should dominate it, but it has not been measured here -- defaulting to it
    would mean shipping an unmeasured setting justified by another setting's
    data.

    On quality: a multi-key retrieval probe -- four codes recalled from 31k
    tokens against eight confusable distractors, each answer required to bind
    the right code to the right project name -- scored 4/4 at f16, q8_0 and
    q4_0 alike, with byte-identical answers
    (``docs/measurements/kv-quantization-retrieval-2026-08-16.json``).

    Read that narrowly. It rules out retrieval damage at this width and
    length, which was the specific worry. It does not measure reasoning, code
    generation, or long-range coherence, it ran at 31k rather than at the
    131072 window shipped, and three identical answers suggest the probe still
    sits inside the model's comfortable range rather than at its margin.
    """
    if kv_cache_type not in LLAMA_KV_CACHE_TYPES:
        raise ValueError(
            f"kv_cache_type must be one of {', '.join(LLAMA_KV_CACHE_TYPES)}; "
            f"got {kv_cache_type!r}"
        )
    cmd = [
        binary,
        "--model", model_path,
        "--port", str(port),
        "--host", host,
        "--no-webui",
        "-c", str(num_ctx),
        "-ngl", "99",
        "--cpu-moe",
        "-ctk", kv_cache_type,
        "-ctv", kv_cache_type,
        "-fa", flash_attn,
        "-np", "1",
    ]
    if spec_type:
        cmd.extend(["--spec-type", spec_type])
        if spec_draft_n_max > 0:
            cmd.extend(["--spec-draft-n-max", str(spec_draft_n_max)])
        if spec_draft_p_min is not None:
            cmd.extend(["--spec-draft-p-min", str(spec_draft_p_min)])
        if draft_model:
            cmd.extend(["--spec-draft-model", draft_model])
    if load_mode:
        cmd.extend(["--load-mode", load_mode])
    return cmd


def recommended_server_environment(
    base: Optional[Dict[str, str]] = None,
    *,
    kv_cache_type: str = DEFAULT_SERVER_KV_CACHE_TYPE,
    flash_attention: bool = True,
) -> Dict[str, str]:
    """Environment for ``ollama serve`` that shrinks the KV cache.

    These are server-process knobs: Ollama reads them at startup, so they
    cannot be set per request, and a server started without them cannot be
    retuned by a client. That is why they belong here rather than in
    :class:`OllamaAdapter`, which only configures individual requests.

    Measured on an RTX 3050 Ti Laptop (4 GB) with ``qwen3.6-35b-a3b`` at
    262144 context: ``f16`` KV gave 2.77 tok/s at a 28 GB footprint, while
    ``q8_0`` plus flash attention gave 8.63 tok/s at 26 GB. The speedup is not
    arithmetic getting cheaper -- it is the 2 GB the smaller cache gives back,
    which is the difference between the weights staying resident and being
    paged off disk on every token. At short contexts, where the cache is small
    either way, the same setting changes nothing measurable.

    Flash attention is included because llama.cpp requires it for a quantized
    KV cache; without it the quantization request is silently ineffective.

    Args:
        base: Environment to extend. Not mutated.
        kv_cache_type: One of :data:`KV_CACHE_TYPES`.
        flash_attention: Whether to request flash attention.

    Returns:
        A new environment dict. Values already present in ``base`` win, so an
        operator's explicit choice is never overridden.
    """
    if kv_cache_type not in KV_CACHE_TYPES:
        raise ValueError(
            f"kv_cache_type must be one of {', '.join(KV_CACHE_TYPES)}; "
            f"got {kv_cache_type!r}"
        )
    env = dict(base or {})
    env.setdefault("OLLAMA_KV_CACHE_TYPE", kv_cache_type)
    env.setdefault("OLLAMA_FLASH_ATTENTION", "1" if flash_attention else "0")
    return env


class HttpTransport:
    """Minimal JSON-over-HTTP transport backed by the standard library.

    Kept behind an interface with a single method so tests (and users who want
    connection pooling or auth) can substitute their own without the adapter
    growing a hard dependency on `requests`.
    """

    def __init__(self, host: str = DEFAULT_HOST, timeout: float = DEFAULT_TIMEOUT):
        self.host = host.rstrip("/")
        self.timeout = timeout

    def request(
        self, method: str, path: str, payload: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        url = f"{self.host}{path}"
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        req = urllib.request.Request(
            url,
            data=data,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:500]
            raise AdapterError(
                f"Ollama returned HTTP {exc.code} for {method} {path}: {detail}"
            ) from exc
        except urllib.error.URLError as exc:
            raise AdapterError(
                f"cannot reach Ollama at {self.host} ({exc.reason}). "
                f"Is `ollama serve` running?"
            ) from exc
        except OSError as exc:
            raise AdapterError(f"transport error talking to {self.host}: {exc}") from exc

        if not body.strip():
            return {}
        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise AdapterError(
                f"Ollama returned a non-JSON body for {method} {path}: {body[:200]!r}"
            ) from exc


@dataclass
class OllamaGeneration:
    """Result of one generate call, with the timings Ollama reports.

    Durations arrive in nanoseconds; they are converted once here so callers
    never have to remember the unit.
    """

    text: str
    model: str
    #: Wall-clock seconds measured client-side, including transport.
    wall_seconds: float
    #: Server-reported prompt evaluation time — the closest thing Ollama
    #: exposes to TTFT for a non-streaming request.
    prompt_eval_seconds: Optional[float] = None
    eval_seconds: Optional[float] = None
    prompt_tokens: Optional[int] = None
    eval_tokens: Optional[int] = None

    @property
    def tokens_per_second(self) -> Optional[float]:
        """Decode throughput, or None when the server reported no timings."""
        if not self.eval_tokens or not self.eval_seconds:
            return None
        return self.eval_tokens / self.eval_seconds


class OllamaAdapter(RuntimeAdapter):
    """Lifecycle-managed client for an external Ollama server.

    Example::

        with OllamaAdapter(model="llama3.2", num_ctx=8192) as ollama:
            result = ollama.generate("Explain paging.")
            print(result.tokens_per_second)

    Args:
        model: Model tag to use. Verified to exist during ``initialize``.
        host: Base URL of the Ollama server.
        num_ctx: Context window to request. Ollama silently clamps this to the
            model's trained maximum, so ``effective_options`` reports what was
            asked for, not what was granted.
        kv_cache_type: Ollama's own KV cache quantization ("f16", "q8_0",
            "q4_0"). Passed through as a request option. This is *Ollama's*
            quantization, not ARGUS's — see the module docstring.
        transport: Injected transport; defaults to :class:`HttpTransport`.
        timeout: Request timeout in seconds.
    """

    capabilities = AdapterCapabilities(
        name="ollama",
        kind=RuntimeKind.EXTERNAL,
        # Explicitly false. Ollama's KV cache lives in another process.
        manages_kv_cache=False,
        provides_telemetry=True,
        reversible=True,
        notes=(
            "External llama.cpp server. ARGUS configures and measures it but "
            "does not manage its KV cache."
        ),
    )

    def __init__(
        self,
        model: str,
        host: str = DEFAULT_HOST,
        num_ctx: Optional[int] = None,
        kv_cache_type: Optional[str] = None,
        transport: Optional[HttpTransport] = None,
        timeout: float = DEFAULT_TIMEOUT,
        **options: Any,
    ) -> None:
        super().__init__(**options)
        if not model:
            raise ValueError("OllamaAdapter requires a model tag")
        if kv_cache_type is not None and kv_cache_type not in KV_CACHE_TYPES:
            raise ValueError(
                f"kv_cache_type must be one of {', '.join(KV_CACHE_TYPES)}; "
                f"got {kv_cache_type!r}"
            )
        self.model = model
        self.num_ctx = num_ctx
        self.kv_cache_type = kv_cache_type
        self.transport = transport or HttpTransport(host=host, timeout=timeout)
        self.server_version: Optional[str] = None
        self._available_models: List[str] = []
        self._request_count = 0
        self._total_wall_seconds = 0.0

    # ── lifecycle ───────────────────────────────────────────────────────────

    def _do_initialize(self) -> None:
        version = self.transport.request("GET", "/api/version")
        self.server_version = version.get("version")

        tags = self.transport.request("GET", "/api/tags")
        self._available_models = [
            m.get("name", "") for m in tags.get("models", []) if m.get("name")
        ]
        if self._available_models and not self._model_present():
            raise AdapterError(
                f"model {self.model!r} is not present on the Ollama server. "
                f"Available: {', '.join(self._available_models) or '(none)'}. "
                f"Pull it with `ollama pull {self.model}`."
            )

    def _do_activate(self) -> None:
        # Ollama loads models lazily; a zero-token generate warms the model so
        # the first real request isn't billed for load time. Nothing is patched
        # in this process, which is why deactivate has nothing to undo.
        self.transport.request(
            "POST",
            "/api/generate",
            {
                "model": self.model,
                "prompt": "",
                "stream": False,
                "options": self._request_options(),
            },
        )

    def _do_deactivate(self) -> None:
        # Nothing to restore: this adapter never mutates in-process state.
        # Asking the server to unload the model is best-effort and must not
        # fail teardown, so errors are swallowed by the base class.
        self.transport.request(
            "POST",
            "/api/generate",
            {"model": self.model, "prompt": "", "keep_alive": 0},
        )

    # ── operations ──────────────────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        **overrides: Any,
    ) -> OllamaGeneration:
        """Run one non-streaming completion and return it with timings.

        Raises:
            AdapterError: if the adapter is not active or the server fails.
        """
        self._require_active("generate")

        options = self._request_options()
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        options.update(overrides)

        started = time.perf_counter()
        response = self.transport.request(
            "POST",
            "/api/generate",
            {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": options,
            },
        )
        wall = time.perf_counter() - started

        self._request_count += 1
        self._total_wall_seconds += wall

        return OllamaGeneration(
            text=response.get("response", ""),
            model=response.get("model", self.model),
            wall_seconds=wall,
            prompt_eval_seconds=_ns_to_s(response.get("prompt_eval_duration")),
            eval_seconds=_ns_to_s(response.get("eval_duration")),
            prompt_tokens=response.get("prompt_eval_count"),
            eval_tokens=response.get("eval_count"),
        )

    def generate_many(self, prompts: Iterable[str], **kwargs: Any) -> List[OllamaGeneration]:
        """Sequential batch. Ollama serializes requests per model anyway, so
        this exists for convenience rather than throughput."""
        return [self.generate(p, **kwargs) for p in prompts]

    def effective_options(self) -> Dict[str, Any]:
        """The options this adapter sends with every request.

        Note these are *requests*: Ollama clamps ``num_ctx`` to the model's
        trained window without reporting it, so this is not a measurement of
        what the server actually applied.
        """
        return self._request_options()

    # ── telemetry ───────────────────────────────────────────────────────────

    def _do_telemetry(self) -> Dict[str, Any]:
        running = self.transport.request("GET", "/api/ps")
        loaded = [
            {
                "name": m.get("name"),
                "size_bytes": m.get("size"),
                "size_vram_bytes": m.get("size_vram"),
            }
            for m in running.get("models", [])
        ]
        return {
            "runtime": "ollama",
            "server_version": self.server_version,
            "model": self.model,
            "loaded_models": loaded,
            "requests": self._request_count,
            "total_wall_seconds": round(self._total_wall_seconds, 6),
            # Restated on every telemetry read so a dashboard can never
            # present these numbers as ARGUS cache metrics.
            "argus_manages_kv_cache": False,
        }

    # ── internals ───────────────────────────────────────────────────────────

    def _model_present(self) -> bool:
        # Ollama reports tags as "name:tag"; a bare name matches its :latest.
        return any(
            available == self.model or available.split(":", 1)[0] == self.model
            for available in self._available_models
        )

    def _request_options(self) -> Dict[str, Any]:
        options: Dict[str, Any] = {}
        if self.num_ctx is not None:
            options["num_ctx"] = self.num_ctx
        if self.kv_cache_type is not None:
            options["cache_type_k"] = self.kv_cache_type
            options["cache_type_v"] = self.kv_cache_type
        return options

    def _require_active(self, what: str) -> None:
        if not self.is_active:
            raise AdapterError(
                f"cannot {what}: OllamaAdapter is {self.state.value}, not active. "
                f"Call activate() or use the adapter as a context manager."
            )


def _ns_to_s(value: Optional[int]) -> Optional[float]:
    """Ollama reports durations in nanoseconds."""
    return None if value is None else value / 1e9
