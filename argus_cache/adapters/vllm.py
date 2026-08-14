"""Fail-closed vLLM compatibility probe.

ARGUS does not currently integrate with vLLM's KV cache.  The former adapter
monkey-patched ``LlamaAttention.forward`` and attached an ``_argus_cache``
attribute to each layer, but vLLM never read that attribute: its paged block
pool remained the sole cache used by attention.  Reporting that path as an
active cache integration was therefore incorrect.

vLLM 0.27.1 exposes two relevant, supported extension mechanisms:

* ``KVConnectorBase_V1`` for transferring/offloading vLLM-owned KV blocks;
* a custom ``AttentionBackend`` for controlling cache layout and attention.

A working ARGUS integration needs both sides of that contract (or a native
vLLM offloading backend).  A model-class ``forward`` wrapper is not a KV-cache
seam.  Until that implementation exists this adapter probes the installed
version and then refuses activation with an actionable error.  It never
patches runtime code and never claims to manage vLLM memory.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from .base import AdapterCapabilities, AdapterError, RuntimeAdapter, RuntimeKind

# There is deliberately no supported range.  0.27.1 was probed and found to
# require a KVConnector/custom-attention implementation that ARGUS does not yet
# ship.  An empty tuple is machine-checkable and cannot accidentally admit a
# future release merely because its version compares inside a guessed range.
SUPPORTED_VLLM_RANGE: Tuple[()] = ()
PROBED_VLLM_VERSIONS = ("0.27.1",)


class VLLMAdapter(RuntimeAdapter):
    """Probe vLLM and fail closed until a real KV-block integration exists.

    ``cache_factory``, ``target_modules`` and ``strict_version`` remain
    accepted for source compatibility with v0.3.0 callers.  They cannot opt
    back into the ineffective model-class monkey patch.
    """

    capabilities = AdapterCapabilities(
        name="vllm",
        kind=RuntimeKind.IN_PROCESS,
        manages_kv_cache=False,
        provides_telemetry=True,
        reversible=True,
        notes=(
            "Unavailable: vLLM 0.27.1 requires a KVConnector/custom attention "
            "backend; ARGUS does not currently ship one."
        ),
    )

    def __init__(
        self,
        cache_factory: Optional[Any] = None,
        target_modules: Optional[list[str]] = None,
        strict_version: bool = True,
        **options: Any,
    ) -> None:
        super().__init__(**options)
        self.cache_factory = cache_factory
        self.target_modules = list(target_modules or [])
        self.strict_version = strict_version
        self.vllm_version: Optional[str] = None

    def _do_initialize(self) -> None:
        try:
            import vllm
        except ImportError as exc:
            raise AdapterError(
                "vLLM is not installed in this environment. Install it in an "
                "isolated environment before probing compatibility."
            ) from exc

        self.vllm_version = getattr(vllm, "__version__", "unknown")
        raise AdapterError(
            f"vLLM {self.vllm_version} is not integrated with ARGUS. The "
            "model-class forward hook used before v0.3.1 did not own or alter "
            "vLLM's KV blocks. A correct implementation must use vLLM's "
            "KVConnectorBase_V1 and/or a registered custom AttentionBackend; "
            "activation is refused to prevent a false cache-management claim."
        )

    def _do_activate(self) -> None:  # pragma: no cover - initialize always refuses
        raise AdapterError("vLLM integration is unavailable")

    def _do_deactivate(self) -> None:
        return None

    def is_fully_restored(self) -> bool:
        """Always true: this fail-closed adapter never mutates vLLM."""
        return True

    @property
    def cache(self) -> None:
        return None

    def _do_telemetry(self) -> Dict[str, Any]:
        return {
            "runtime": "vllm",
            "vllm_version": self.vllm_version,
            "argus_manages_kv_cache": False,
            "integration_available": False,
        }


def inject_argus_to_vllm(*_args: Any, **_kwargs: Any) -> bool:
    """Removed unsafe legacy injection helper."""
    raise AdapterError(
        "inject_argus_to_vllm() has been removed: it divided vLLM block-table "
        "indices by a 'reduction factor', aliasing unrelated sequences without "
        "compressing KV data. A correct replacement requires vLLM's "
        "KVConnector/custom AttentionBackend interfaces."
    )
