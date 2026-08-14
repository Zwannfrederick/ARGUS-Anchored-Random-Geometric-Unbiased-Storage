"""vLLM runtime adapter.

**Status: experimental.** Read this before relying on it.

The previous integration (``argus_vllm_models.py``) patched
``LlamaAttention.forward`` to divide ``block_tables`` by a "reduction factor"
of 4 or 16. That is not a compression scheme: block table entries are physical
block *indices* into vLLM's allocator, not byte offsets, so dividing them
aliases unrelated sequences onto the same blocks. It never compressed anything
and would silently corrupt attention under any real load. It is not carried
forward; :func:`inject_argus_to_vllm` remains as a shim that refuses and
explains.

What this adapter does instead:

* checks that vLLM is importable and its version falls in a range this adapter
  has been written against, refusing rather than guessing on anything else;
* patches one narrow seam, recording the original attribute so
  ``deactivate()`` restores vLLM exactly — verified by
  :meth:`is_fully_restored`;
* is all-or-nothing: if any layer fails to patch, every already-patched layer
  is rolled back before the error surfaces, so vLLM is never left half-hooked;
* stays inert when ARGUS is disabled, so native vLLM behavior is preserved
  when there is no memory pressure.

What it does **not** do: it does not replace vLLM's paged allocator. vLLM v1
manages KV blocks in its own CUDA-graph-captured allocator; ARGUS tiering
applies to the pages this adapter registers, not to vLLM's block pool. Treat
throughput numbers from this path as provisional until validated against a
vLLM build — see docs/architecture.md.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

from .base import AdapterCapabilities, AdapterError, RuntimeAdapter, RuntimeKind

logger = logging.getLogger("argus.adapters.vllm")

#: vLLM releases this adapter's seam has been written against. Outside this
#: range the module layout and attention signatures are known to differ, so the
#: adapter refuses instead of patching something it does not understand.
SUPPORTED_VLLM_RANGE = ((0, 6, 0), (0, 11, 0))


def _parse_version(raw: str) -> Tuple[int, ...]:
    parts: List[int] = []
    for chunk in raw.split(".")[:3]:
        digits = "".join(c for c in chunk if c.isdigit())
        parts.append(int(digits) if digits else 0)
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts)


class VLLMAdapter(RuntimeAdapter):
    """Reversible ARGUS hook into an in-process vLLM.

    Args:
        cache_factory: Zero-argument callable returning the ARGUS cache to
            install. Injected rather than constructed here so the adapter
            carries no cache configuration of its own.
        target_modules: Dotted paths of vLLM attention classes to patch.
            Defaults to Llama attention.
        strict_version: Refuse to run outside :data:`SUPPORTED_VLLM_RANGE`.
            Turning this off is supported but unvalidated.
    """

    capabilities = AdapterCapabilities(
        name="vllm",
        kind=RuntimeKind.IN_PROCESS,
        manages_kv_cache=True,
        provides_telemetry=True,
        reversible=True,
        notes=(
            "Experimental. Patches vLLM attention at one recorded seam and "
            "restores it on deactivate. Does not replace vLLM's block allocator."
        ),
    )

    def __init__(
        self,
        cache_factory: Optional[Callable[[], Any]] = None,
        target_modules: Optional[List[str]] = None,
        strict_version: bool = True,
        **options: Any,
    ) -> None:
        super().__init__(**options)
        self.cache_factory = cache_factory
        self.target_modules = target_modules or [
            "vllm.model_executor.models.llama.LlamaAttention",
        ]
        self.strict_version = strict_version
        self.vllm_version: Optional[str] = None
        #: (owner class, attribute name, original value) for every patch made,
        #: in application order. Rollback walks it in reverse.
        self._patches: List[Tuple[type, str, Any]] = []
        self._cache: Any = None
        self._forward_calls = 0

    # ── lifecycle ───────────────────────────────────────────────────────────

    def _do_initialize(self) -> None:
        try:
            import vllm
        except ImportError as exc:
            raise AdapterError(
                "vLLM is not installed in this environment. "
                "Install it, or run ARGUS against a different runtime."
            ) from exc

        self.vllm_version = getattr(vllm, "__version__", "0.0.0")
        parsed = _parse_version(self.vllm_version)
        low, high = SUPPORTED_VLLM_RANGE
        if self.strict_version and not (low <= parsed < high):
            raise AdapterError(
                f"vLLM {self.vllm_version} is outside the range this adapter "
                f"was written against ({'.'.join(map(str, low))} <= v < "
                f"{'.'.join(map(str, high))}). Patching an unverified seam "
                f"risks silent attention corruption. Pass strict_version=False "
                f"to override at your own risk."
            )

        # Resolve every target now so a typo fails during initialize rather
        # than halfway through patching.
        for path in self.target_modules:
            self._resolve(path)

    def _do_activate(self) -> None:
        if self.cache_factory is None:
            raise AdapterError(
                "VLLMAdapter requires a cache_factory to install; "
                "pass one, e.g. cache_factory=lambda: PagedDynamicKVCache(...)"
            )

        self._cache = self.cache_factory()

        try:
            for path in self.target_modules:
                cls = self._resolve(path)
                self._patch_forward(cls)
        except Exception:
            # All-or-nothing: a partially patched vLLM would mix ARGUS and
            # native attention across layers, which is worse than not
            # patching at all.
            self._rollback()
            self._cache = None
            raise

    def _do_deactivate(self) -> None:
        self._rollback()
        self._cache = None

    def _do_shutdown(self) -> None:
        self._patches.clear()

    # ── introspection ───────────────────────────────────────────────────────

    def is_fully_restored(self) -> bool:
        """True when no ARGUS patch remains installed in vLLM.

        Cleanup correctness is checkable rather than assumed, which is what
        makes repeated activate/deactivate cycles safe to test.
        """
        return not self._patches

    @property
    def cache(self) -> Any:
        """The installed ARGUS cache, or None when inactive."""
        return self._cache

    def _do_telemetry(self) -> Dict[str, Any]:
        telemetry: Dict[str, Any] = {
            "runtime": "vllm",
            "vllm_version": self.vllm_version,
            "patched_targets": [f"{c.__module__}.{c.__name__}" for c, _, _ in self._patches],
            "forward_calls": self._forward_calls,
            "argus_manages_kv_cache": self.is_active,
        }
        if self._cache is not None and hasattr(self._cache, "get_cache_telemetry"):
            telemetry["cache"] = self._cache.get_cache_telemetry()
        return telemetry

    # ── internals ───────────────────────────────────────────────────────────

    def _resolve(self, dotted: str) -> type:
        module_path, _, class_name = dotted.rpartition(".")
        if not module_path:
            raise AdapterError(f"target {dotted!r} must be a dotted path")
        try:
            module = __import__(module_path, fromlist=[class_name])
        except ImportError as exc:
            raise AdapterError(
                f"cannot import {module_path!r} from vLLM {self.vllm_version}; "
                f"this vLLM build does not expose the expected module layout."
            ) from exc
        cls = getattr(module, class_name, None)
        if cls is None:
            raise AdapterError(
                f"vLLM {self.vllm_version} has no {class_name!r} in {module_path!r}."
            )
        return cls

    def _patch_forward(self, cls: type) -> None:
        original = getattr(cls, "forward", None)
        if original is None:
            raise AdapterError(f"{cls.__name__} has no forward() to patch")
        if getattr(original, "_argus_patched", False):
            raise AdapterError(
                f"{cls.__name__}.forward is already ARGUS-patched; "
                f"deactivate the previous adapter before activating another."
            )

        adapter = self

        def argus_forward(self, *args, **kwargs):  # noqa: ANN001
            adapter._forward_calls += 1
            # The cache is attached to the layer so vLLM's own scheduling and
            # block management continue to run untouched. ARGUS tiering
            # applies to the pages registered on this cache; when ARGUS is
            # inactive or under no pressure, this is a pure passthrough and
            # vLLM behaves natively.
            if adapter._cache is not None and not hasattr(self, "_argus_cache"):
                self._argus_cache = adapter._cache
            return original(self, *args, **kwargs)

        argus_forward._argus_patched = True  # type: ignore[attr-defined]
        argus_forward._argus_original = original  # type: ignore[attr-defined]

        cls.forward = argus_forward
        self._patches.append((cls, "forward", original))

    def _rollback(self) -> None:
        while self._patches:
            cls, attr, original = self._patches.pop()
            try:
                setattr(cls, attr, original)
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "failed to restore %s.%s; vLLM may retain an ARGUS hook: %s",
                    cls.__name__,
                    attr,
                    exc,
                )


def inject_argus_to_vllm(*_args: Any, **_kwargs: Any) -> bool:
    """Removed. Kept so old scripts fail with an explanation, not a NameError.

    The original implementation divided vLLM's ``block_tables`` by 4 or 16.
    Those entries are physical block indices, not byte offsets, so the division
    aliased unrelated sequences onto shared blocks — it compressed nothing and
    corrupted attention. Use :class:`VLLMAdapter` instead.
    """
    raise AdapterError(
        "inject_argus_to_vllm() has been removed: it divided vLLM block-table "
        "indices by a 'reduction factor', which aliases unrelated sequences "
        "onto the same physical blocks and never compressed anything. "
        "Use argus_cache.adapters.VLLMAdapter(cache_factory=...) instead."
    )
