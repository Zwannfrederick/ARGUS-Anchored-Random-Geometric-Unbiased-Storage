"""ARGUS plugin system.

ARGUS is a KV-cache *management* runtime, not a single quantization algorithm.
The six tiers it ships with (fp8, int8, int4, int2, one_bit, jl) are ordinary
plugins registered here at import time; the cache manager reaches them only
through :mod:`argus_cache.plugins.registry` and reasons about them only through
:class:`~argus_cache.plugins.capabilities.BackendCapabilities`.

Replacing a tier therefore needs no change to the manager::

    from argus_cache.plugins import (
        BackendCapabilities, NativeCodecSpec,
        register_quantizer, unregister_quantizer,
    )

    unregister_quantizer("one_bit")
    register_quantizer(
        "my_archival",
        MyArchivalBackend,
        BackendCapabilities(
            name="my_archival",
            effective_bits=1.5,
            native_codec=NativeCodecSpec(kind="unsigned_affine", bits=2),
        ),
    )

.. note::
   A backend's ``native_codec`` describes how the **C++ engine** stores the
   tier. The Python backend classes in :mod:`argus_cache.backends.quantization`
   pack along a different axis than the native kernels do, so a page compressed
   by one must never be decoded by the other. The manager keeps them apart;
   read back native-tier pages via ``peek_decompress_page``.
"""

from __future__ import annotations

from .capabilities import NATIVE_KINDS, BackendCapabilities, NativeCodecSpec
from .registry import (
    REGISTRY,
    PluginError,
    QuantizerRegistry,
    available_quantizers,
    get_capabilities,
    get_quantizer,
    list_quantizers,
    register_quantizer,
    unregister_quantizer,
)

__all__ = [
    "BackendCapabilities",
    "NativeCodecSpec",
    "NATIVE_KINDS",
    "PluginError",
    "QuantizerRegistry",
    "REGISTRY",
    "register_quantizer",
    "unregister_quantizer",
    "get_quantizer",
    "get_capabilities",
    "list_quantizers",
    "available_quantizers",
    "register_builtin_quantizers",
]


def register_builtin_quantizers(*, replace: bool = True) -> None:
    """Register ARGUS's six built-in tiers.

    Called once on import. Exposed so tests (and applications that cleared the
    registry) can restore the defaults.
    """
    # Imported lazily: argus_cache.backends imports torch and the core
    # quantization kernels, and this module is imported very early.
    from argus_cache.backends.quantization import (
        FP8Backend,
        INT2Backend,
        INT4Backend,
        INT8Backend,
        JLProjectionBackend,
        OneBitBackend,
    )

    builtins = (
        (
            FP8Backend,
            BackendCapabilities(
                name="fp8",
                effective_bits=8.0,
                native_codec=NativeCodecSpec(kind="signed_linear", bits=8),
                description="Simulated FP8 (E4M3-style) symmetric scaling.",
            ),
        ),
        (
            INT8Backend,
            BackendCapabilities(
                name="int8",
                effective_bits=8.0,
                native_codec=NativeCodecSpec(kind="signed_linear", bits=8),
                description="Symmetric per-tensor INT8 quantization.",
            ),
        ),
        (
            INT4Backend,
            BackendCapabilities(
                name="int4",
                effective_bits=4.0,
                native_codec=NativeCodecSpec(kind="unsigned_affine", bits=4),
                description="Asymmetric INT4, two elements packed per byte.",
            ),
        ),
        (
            INT2Backend,
            BackendCapabilities(
                name="int2",
                effective_bits=2.0,
                native_codec=NativeCodecSpec(kind="unsigned_affine", bits=2),
                description="Asymmetric INT2, four elements packed per byte.",
            ),
        ),
        (
            OneBitBackend,
            BackendCapabilities(
                name="one_bit",
                effective_bits=1.0,
                native_codec=NativeCodecSpec(kind="sign_packed", bits=1),
                description="Sign binarization with a per-page magnitude, "
                "eight elements packed per byte.",
            ),
        ),
        (
            JLProjectionBackend,
            BackendCapabilities(
                name="jl",
                effective_bits=4.0,
                # The projection matrix is built on the device the page lives
                # on; reconstruction is a regularized least-squares solve.
                native_codec=NativeCodecSpec(
                    kind="projection", bits=16, compression_ratio=0.25
                ),
                description="Laplacian-regularized Johnson-Lindenstrauss "
                "random projection along the sequence axis.",
            ),
        ),
    )

    for factory, caps in builtins:
        register_quantizer(caps.name, factory, caps, replace=replace)


register_builtin_quantizers()
