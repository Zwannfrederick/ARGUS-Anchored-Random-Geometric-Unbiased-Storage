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
        if kv_cache_type is not None and kv_cache_type not in {"f16", "q8_0", "q4_0"}:
            raise ValueError(
                f"kv_cache_type must be one of f16, q8_0, q4_0; got {kv_cache_type!r}"
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
