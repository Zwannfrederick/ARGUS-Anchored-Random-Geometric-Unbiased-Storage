"""Deprecated compatibility shim for the old vLLM injection script.

This module used to patch ``LlamaAttention.forward`` so it divided vLLM's
``block_tables`` by a "reduction factor" of 4 (INT4) or 16 (1-bit). Block table
entries are physical block *indices* into vLLM's allocator, not byte offsets,
so dividing them mapped unrelated sequences onto the same physical blocks. It
compressed nothing and corrupted attention whenever more than a handful of
blocks were live.

The real integration now lives in :mod:`argus_cache.adapters.vllm`, which
patches one recorded seam, restores it on deactivate, and refuses to run
against untested vLLM versions::

    from argus_cache.adapters import VLLMAdapter
    from argus_cache import PagedDynamicKVCache

    with VLLMAdapter(cache_factory=lambda: PagedDynamicKVCache()) as adapter:
        ...

This file is kept only so existing scripts fail with an explanation instead of
an ImportError. It will be removed in a future release.
"""

import warnings

from argus_cache.adapters import AdapterError, VLLMAdapter  # noqa: F401
from argus_cache.adapters.vllm import inject_argus_to_vllm  # noqa: F401

warnings.warn(
    "argus_vllm_models is deprecated; use argus_cache.adapters.VLLMAdapter. "
    "The previous inject_argus_to_vllm() implementation was incorrect and has "
    "been removed.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["VLLMAdapter", "inject_argus_to_vllm", "AdapterError"]

if __name__ == "__main__":
    print(__doc__)
