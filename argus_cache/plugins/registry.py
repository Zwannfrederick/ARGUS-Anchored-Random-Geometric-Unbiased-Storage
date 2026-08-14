"""Plugin registry for ARGUS quantization backends.

The registry is the stable contract between ARGUS's cache manager and the
quantizers it moves pages through. A backend is identified by a name, paired
with :class:`BackendCapabilities` describing what it costs and what it accepts,
and constructed lazily through a factory so registering a backend never
allocates GPU state.

The point is substitution: disabling ``one_bit`` and dropping in a different
archival quantizer is a registry operation, not an edit to the memory manager.

    >>> from argus_cache.plugins import register_quantizer, unregister_quantizer
    >>> unregister_quantizer("one_bit")
    >>> register_quantizer("my_codec", MyBackend, caps)

Thread safety: registration is guarded by a lock, so plugins may be registered
from setup code running on any thread. Instances handed out by
:func:`get_quantizer` are not themselves synchronized.
"""

from __future__ import annotations

import threading
from typing import Callable, Dict, List, Optional

import torch

from .capabilities import BackendCapabilities

#: A zero-argument callable returning a fresh backend instance.
QuantizerFactory = Callable[[], object]

# Methods a quantization backend must expose. Checked at registration so a
# malformed plugin fails loudly at setup rather than mid-generation.
_REQUIRED_METHODS = ("compress", "decompress", "decompress_batch", "memory_bytes")


class PluginError(RuntimeError):
    """Raised for invalid plugin registration or lookup."""


class _Entry:
    __slots__ = ("factory", "capabilities", "_instance", "_lock")

    def __init__(self, factory: QuantizerFactory, capabilities: BackendCapabilities):
        self.factory = factory
        self.capabilities = capabilities
        self._instance: Optional[object] = None
        self._lock = threading.Lock()

    def instance(self) -> object:
        # Backends are stateless value-transformers in practice, so one shared
        # instance per registration keeps JL projection caches warm instead of
        # rebuilding them per page.
        if self._instance is None:
            with self._lock:
                if self._instance is None:
                    self._instance = self.factory()
        return self._instance


class QuantizerRegistry:
    """Name → (factory, capabilities) mapping for quantization backends."""

    def __init__(self) -> None:
        self._entries: Dict[str, _Entry] = {}
        self._lock = threading.RLock()

    # ── registration ────────────────────────────────────────────────────────

    def register(
        self,
        name: str,
        factory: QuantizerFactory,
        capabilities: BackendCapabilities,
        *,
        replace: bool = False,
    ) -> None:
        """Register a quantization backend under ``name``.

        Args:
            name: Registry key, also the default tier name.
            factory: Zero-argument callable returning a backend instance.
            capabilities: What the backend costs and accepts.
            replace: Permit overwriting an existing registration. Off by
                default so a typo'd name cannot silently shadow a built-in.

        Raises:
            PluginError: on a name clash without ``replace``, a
                non-callable factory, a capabilities/name mismatch, or a
                backend missing a required method.
        """
        if not isinstance(name, str) or not name:
            raise PluginError("plugin name must be a non-empty string")
        if not callable(factory):
            raise PluginError(f"factory for {name!r} must be callable")
        if not isinstance(capabilities, BackendCapabilities):
            raise PluginError(
                f"capabilities for {name!r} must be a BackendCapabilities, "
                f"got {type(capabilities).__name__}"
            )
        if capabilities.name != name:
            raise PluginError(
                f"capabilities.name ({capabilities.name!r}) does not match the "
                f"registration name ({name!r})"
            )

        with self._lock:
            if name in self._entries and not replace:
                raise PluginError(
                    f"quantizer {name!r} is already registered; "
                    f"pass replace=True to override it"
                )
            self._entries[name] = _Entry(factory, capabilities)

    def unregister(self, name: str) -> None:
        """Remove a backend. Raises PluginError if it was never registered."""
        with self._lock:
            if name not in self._entries:
                raise PluginError(f"quantizer {name!r} is not registered")
            del self._entries[name]

    def clear(self) -> None:
        """Drop every registration. Intended for test isolation."""
        with self._lock:
            self._entries.clear()

    # ── lookup ──────────────────────────────────────────────────────────────

    def __contains__(self, name: object) -> bool:
        return name in self._entries

    def get(self, name: str) -> object:
        """Return the shared backend instance for ``name``.

        The instance is constructed on first use and validated against the
        required backend protocol, so a factory returning the wrong kind of
        object is caught here rather than during a cache demotion.
        """
        entry = self._entry(name)
        instance = entry.instance()
        missing = [m for m in _REQUIRED_METHODS if not callable(getattr(instance, m, None))]
        if missing:
            raise PluginError(
                f"quantizer {name!r} ({type(instance).__name__}) is missing "
                f"required method(s): {', '.join(missing)}"
            )
        return instance

    def capabilities(self, name: str) -> BackendCapabilities:
        return self._entry(name).capabilities

    def names(self) -> List[str]:
        with self._lock:
            return sorted(self._entries)

    def available(
        self,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
        *,
        max_effective_bits: Optional[float] = None,
        require_reconstruction: bool = False,
    ) -> List[str]:
        """Names of backends matching a capability filter, cheapest last.

        This is the capability-based replacement for hardcoded tier lists:
        callers ask for "backends usable on this device at or below N bits"
        rather than naming INT2 or ONE_BIT.
        """
        with self._lock:
            entries = list(self._entries.items())

        matches = []
        for name, entry in entries:
            caps = entry.capabilities
            if not caps.supports(device=device, dtype=dtype):
                continue
            if max_effective_bits is not None and caps.effective_bits > max_effective_bits:
                continue
            if require_reconstruction and not caps.supports_reconstruction:
                continue
            matches.append((caps.effective_bits, name))

        return [name for _, name in sorted(matches, reverse=True)]

    def _entry(self, name: str) -> _Entry:
        with self._lock:
            entry = self._entries.get(name)
        if entry is None:
            raise PluginError(
                f"quantizer {name!r} is not registered; "
                f"available: {', '.join(self.names()) or '(none)'}"
            )
        return entry


#: Process-wide registry. ARGUS's built-ins register into it on import of
#: ``argus_cache.plugins``; applications add their own the same way.
REGISTRY = QuantizerRegistry()


def register_quantizer(
    name: str,
    factory: QuantizerFactory,
    capabilities: BackendCapabilities,
    *,
    replace: bool = False,
) -> None:
    """Register a quantization backend in the process-wide registry."""
    REGISTRY.register(name, factory, capabilities, replace=replace)


def unregister_quantizer(name: str) -> None:
    """Remove a quantization backend from the process-wide registry."""
    REGISTRY.unregister(name)


def get_quantizer(name: str) -> object:
    """Return the shared instance of a registered backend."""
    return REGISTRY.get(name)


def get_capabilities(name: str) -> BackendCapabilities:
    """Return the capabilities declared by a registered backend."""
    return REGISTRY.capabilities(name)


def list_quantizers() -> List[str]:
    """Names of every registered backend."""
    return REGISTRY.names()


def available_quantizers(
    device: Optional[torch.device | str] = None,
    dtype: Optional[torch.dtype] = None,
    *,
    max_effective_bits: Optional[float] = None,
    require_reconstruction: bool = False,
) -> List[str]:
    """Registered backends matching a capability filter, most expensive first."""
    return REGISTRY.available(
        device=device,
        dtype=dtype,
        max_effective_bits=max_effective_bits,
        require_reconstruction=require_reconstruction,
    )
