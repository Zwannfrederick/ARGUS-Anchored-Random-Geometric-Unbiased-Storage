"""Runtime adapters.

Adapters live at the edge of ARGUS. The core cache manager never imports one,
and no adapter appears in the hot path of another — which is what allows an
SGLang adapter to be added later by writing one file here and registering it,
without touching the engine.

Adapter modules are imported lazily so that importing ``argus_cache`` never
requires vLLM (or any other runtime) to be installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Type

from .base import (
    AdapterCapabilities,
    AdapterError,
    AdapterState,
    RuntimeAdapter,
    RuntimeKind,
)

if TYPE_CHECKING:  # pragma: no cover
    from .ollama import OllamaAdapter
    from .vllm import VLLMAdapter

__all__ = [
    "AdapterCapabilities",
    "AdapterError",
    "AdapterState",
    "RuntimeAdapter",
    "RuntimeKind",
    "OllamaAdapter",
    "VLLMAdapter",
    "get_adapter",
    "list_adapters",
]

#: name -> "module:class", resolved on first use.
_ADAPTERS: Dict[str, str] = {
    "ollama": "argus_cache.adapters.ollama:OllamaAdapter",
    "vllm": "argus_cache.adapters.vllm:VLLMAdapter",
}


def list_adapters() -> List[str]:
    """Names of every registered runtime adapter."""
    return sorted(_ADAPTERS)


def get_adapter(name: str) -> Type[RuntimeAdapter]:
    """Return an adapter class by name, importing its module on demand."""
    try:
        target = _ADAPTERS[name]
    except KeyError:
        raise AdapterError(
            f"unknown runtime adapter {name!r}; available: {', '.join(list_adapters())}"
        ) from None

    module_path, _, class_name = target.partition(":")
    module = __import__(module_path, fromlist=[class_name])
    return getattr(module, class_name)


def __getattr__(name: str) -> Any:
    # Lazy re-export so `from argus_cache.adapters import VLLMAdapter` works
    # without importing the Ollama transport (or vice versa) at package import.
    if name == "OllamaAdapter":
        return get_adapter("ollama")
    if name == "VLLMAdapter":
        return get_adapter("vllm")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
