"""Runtime adapter contract.

An adapter is the seam between ARGUS's cache-management core and a specific
inference runtime (vLLM, Ollama, and later SGLang). Everything runtime-specific
lives behind this interface so the core engine never learns what it is running
under — the constraint that makes a new runtime an additive change.

Two kinds of runtime exist, and conflating them is how ARGUS's integration
claims previously drifted:

``IN_PROCESS``
    The runtime executes in this Python process and exposes its attention /
    KV-cache path, so ARGUS can actually own the cache. vLLM is in this class.

``EXTERNAL``
    The runtime is a separate process with its own memory manager (Ollama
    wraps llama.cpp behind an HTTP server). ARGUS cannot manage its KV cache.
    An adapter for such a runtime configures and measures it; it must not
    claim to be caching for it.

Lifecycle is explicit and idempotent: ``initialize`` → ``activate`` →
``deactivate`` → ``shutdown``. Every one of them may be called twice safely,
and a failure anywhere must leave ARGUS's own state untouched — an adapter
error is reported, never propagated into the cache manager.
"""

from __future__ import annotations

import abc
import enum
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger("argus.adapters")


class RuntimeKind(enum.Enum):
    """Whether ARGUS can own the runtime's KV cache."""

    #: Runs in this process; ARGUS can manage its KV cache directly.
    IN_PROCESS = "in_process"
    #: Separate process with its own cache; ARGUS can configure and observe
    #: it, but not manage its memory.
    EXTERNAL = "external"


class AdapterState(enum.Enum):
    CREATED = "created"
    INITIALIZED = "initialized"
    ACTIVE = "active"
    SHUTDOWN = "shutdown"
    FAILED = "failed"


class AdapterError(RuntimeError):
    """Raised for adapter-level failures.

    Callers may always catch this and continue: adapters guarantee they roll
    back their own partial state before raising, so ARGUS's cache manager is
    never left half-patched.
    """


@dataclass
class AdapterCapabilities:
    """What an adapter can actually do, stated rather than implied."""

    name: str
    kind: RuntimeKind
    #: True only when ARGUS's paged cache genuinely backs this runtime's
    #: attention. False for EXTERNAL runtimes and for adapters that merely
    #: observe.
    manages_kv_cache: bool = False
    #: True when the adapter can report cache/memory telemetry.
    provides_telemetry: bool = False
    #: True when deactivate() fully restores the runtime's original behavior.
    reversible: bool = True
    notes: str = ""


class RuntimeAdapter(abc.ABC):
    """Base class for runtime adapters.

    Subclasses implement the four ``_do_*`` hooks. The public methods handle
    state transitions, idempotency, and failure isolation, so no adapter has
    to re-implement that and get it subtly wrong.
    """

    capabilities: AdapterCapabilities

    def __init__(self, **options: Any) -> None:
        self.options: Dict[str, Any] = dict(options)
        self._state = AdapterState.CREATED
        self._last_error: Optional[BaseException] = None

    # ── state ───────────────────────────────────────────────────────────────

    @property
    def state(self) -> AdapterState:
        return self._state

    @property
    def is_active(self) -> bool:
        return self._state is AdapterState.ACTIVE

    @property
    def last_error(self) -> Optional[BaseException]:
        """The exception from the most recent failed transition, if any."""
        return self._last_error

    # ── lifecycle ───────────────────────────────────────────────────────────

    def initialize(self) -> None:
        """Prepare the adapter: import the runtime, open connections, probe.

        Idempotent. Raises AdapterError if the runtime is unavailable.
        """
        if self._state in (AdapterState.INITIALIZED, AdapterState.ACTIVE):
            return
        self._transition(self._do_initialize, AdapterState.INITIALIZED, "initialize")

    def activate(self) -> None:
        """Install ARGUS into the runtime. Idempotent; initializes if needed."""
        if self._state is AdapterState.ACTIVE:
            return
        if self._state is not AdapterState.INITIALIZED:
            self.initialize()
        self._transition(self._do_activate, AdapterState.ACTIVE, "activate")

    def deactivate(self) -> None:
        """Restore the runtime's original behavior.

        Idempotent and non-raising: teardown must always run to completion so a
        failure here cannot leave the runtime permanently patched. Failures are
        logged and recorded in :attr:`last_error`.
        """
        if self._state is not AdapterState.ACTIVE:
            return
        try:
            self._do_deactivate()
        except Exception as exc:  # noqa: BLE001 - teardown must not raise
            self._last_error = exc
            logger.warning(
                "[%s] deactivate failed; runtime may retain ARGUS hooks: %s",
                self.capabilities.name,
                exc,
            )
        finally:
            self._state = AdapterState.INITIALIZED

    def shutdown(self) -> None:
        """Release every resource. Idempotent and non-raising."""
        if self._state is AdapterState.SHUTDOWN:
            return
        if self._state is AdapterState.ACTIVE:
            self.deactivate()
        try:
            self._do_shutdown()
        except Exception as exc:  # noqa: BLE001 - teardown must not raise
            self._last_error = exc
            logger.warning("[%s] shutdown failed: %s", self.capabilities.name, exc)
        finally:
            self._state = AdapterState.SHUTDOWN

    def __enter__(self) -> "RuntimeAdapter":
        self.activate()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.shutdown()

    # ── telemetry ───────────────────────────────────────────────────────────

    def telemetry(self) -> Dict[str, Any]:
        """Runtime-reported metrics. Never raises; returns {} when unavailable."""
        if not self.capabilities.provides_telemetry:
            return {}
        try:
            return self._do_telemetry()
        except Exception as exc:  # noqa: BLE001 - telemetry is best-effort
            logger.debug("[%s] telemetry unavailable: %s", self.capabilities.name, exc)
            return {}

    # ── hooks ───────────────────────────────────────────────────────────────

    @abc.abstractmethod
    def _do_initialize(self) -> None: ...

    @abc.abstractmethod
    def _do_activate(self) -> None: ...

    @abc.abstractmethod
    def _do_deactivate(self) -> None: ...

    def _do_shutdown(self) -> None:
        """Optional. Default is a no-op beyond deactivation."""

    def _do_telemetry(self) -> Dict[str, Any]:
        return {}

    # ── internals ───────────────────────────────────────────────────────────

    def _transition(self, action, target: AdapterState, label: str) -> None:
        try:
            action()
        except AdapterError as exc:
            self._state = AdapterState.FAILED
            self._last_error = exc
            raise
        except Exception as exc:  # noqa: BLE001
            self._state = AdapterState.FAILED
            self._last_error = exc
            raise AdapterError(
                f"[{self.capabilities.name}] {label} failed: {exc}"
            ) from exc
        self._state = target
        logger.info("[%s] %s complete", self.capabilities.name, label)
