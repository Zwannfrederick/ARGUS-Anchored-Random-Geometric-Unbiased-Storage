"""Capability metadata for ARGUS quantization and storage plugins.

Policy code in ARGUS used to branch on tier *names* — ``if spec.name == 'jl'``,
``if tier == 'one_bit'``. That made every policy decision implicitly a
statement about which quantizers exist, so removing a tier meant auditing the
manager. Capabilities replace those name checks with questions about what a
backend actually does: how many bits it costs, whether it is lossy, whether it
can reconstruct, which devices and dtypes it accepts.

A plugin declares its capabilities once; the manager and policy engine read
them instead of the plugin's identity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import FrozenSet, Optional

import torch

# Native storage formats the C++ engine can execute directly. A plugin that
# declares one gets the fused CUDA dequant path; a plugin that declares none
# still works, but its pages spill uncompressed on the native side and are
# compressed/decompressed in Python.
NATIVE_KINDS = frozenset(
    {
        "signed_linear",
        "unsigned_affine",
        "sign_packed",
        "projection",
        "passthrough",
        "ggml_q8_0",
        "ggml_q4_0",
    }
)

# Bit widths the generic native kernel can pack. Sub-byte widths must divide 8
# evenly, since packing is byte-aligned.
_VALID_NATIVE_BITS = {
    "signed_linear": (8,),
    "unsigned_affine": (1, 2, 4),
    "sign_packed": (1,),
    "projection": (16,),
    "passthrough": (16,),
    "ggml_q8_0": (8,),
    "ggml_q4_0": (4,),
}

_GGML_BLOCK_COMPRESSION_RATIOS = {
    "ggml_q8_0": 34.0 / 64.0,
    "ggml_q4_0": 18.0 / 64.0,
}


@dataclass(frozen=True)
class NativeCodecSpec:
    """How the C++ engine should store this backend's pages.

    Mirrors ``argus::TierCodec``. Declaring one is what promotes a plugin from
    "Python-side only" to a tier the native engine compresses itself.
    """

    kind: str
    bits: int
    #: Storage cost relative to fp16. Derived from ``bits`` for quantized
    #: kinds; projection backends must state it, since their cost depends on
    #: the projection rank rather than a bit width.
    compression_ratio: Optional[float] = None

    def __post_init__(self) -> None:
        if self.kind not in NATIVE_KINDS:
            raise ValueError(
                f"unknown native codec kind {self.kind!r}; "
                f"expected one of {sorted(NATIVE_KINDS)}"
            )
        valid_bits = _VALID_NATIVE_BITS[self.kind]
        if self.bits not in valid_bits:
            raise ValueError(
                f"native codec kind {self.kind!r} supports bits "
                f"{valid_bits}, got {self.bits}"
            )
        if self.compression_ratio is None:
            ratio = _GGML_BLOCK_COMPRESSION_RATIOS.get(
                self.kind, self.bits / 16.0
            )
            object.__setattr__(self, "compression_ratio", ratio)
        if not 0.0 < self.compression_ratio <= 1.0:
            raise ValueError(
                f"compression_ratio must be in (0, 1], got {self.compression_ratio}"
            )

    @property
    def effective_bits(self) -> float:
        return 16.0 * self.compression_ratio


@dataclass(frozen=True)
class BackendCapabilities:
    """What a quantization backend costs and what it can handle.

    Attributes:
        name: Registry key for the backend.
        effective_bits: Average bits of storage per original fp16 element,
            including quantization metadata. Used by policy code to order
            tiers by cost without knowing their names.
        lossy: False only for backends that reconstruct exactly.
        supported_devices: Device *types* ("cuda", "cpu") the backend accepts.
        supported_dtypes: Input dtypes the backend accepts.
        requires_calibration: True when the backend needs a calibration pass
            over representative data before it produces valid output. The
            manager refuses to install such a backend until it is calibrated.
        supports_reconstruction: False for write-only/archival backends that
            cannot return tensors (they may only be terminal tiers).
        native_codec: Native storage format, if the C++ engine can execute
            this backend directly.
        description: Free-form text for telemetry and `list_quantizers`.
    """

    name: str
    effective_bits: float
    lossy: bool = True
    supported_devices: FrozenSet[str] = frozenset({"cuda", "cpu"})
    supported_dtypes: FrozenSet[torch.dtype] = field(
        default_factory=lambda: frozenset(
            {torch.float16, torch.bfloat16, torch.float32}
        )
    )
    requires_calibration: bool = False
    supports_reconstruction: bool = True
    native_codec: Optional[NativeCodecSpec] = None
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("BackendCapabilities.name must be non-empty")
        if not 0.0 < self.effective_bits <= 16.0:
            raise ValueError(
                f"effective_bits must be in (0, 16] for {self.name!r}, "
                f"got {self.effective_bits}"
            )
        if not self.supported_devices:
            raise ValueError(f"{self.name!r} must support at least one device")
        if not self.supported_dtypes:
            raise ValueError(f"{self.name!r} must support at least one dtype")
        if not self.lossy and self.native_codec is not None:
            if self.native_codec.kind != "passthrough":
                raise ValueError(
                    f"{self.name!r} claims to be lossless but declares a lossy "
                    f"native codec {self.native_codec.kind!r}"
                )

    @property
    def compression_ratio(self) -> float:
        """Storage cost as a fraction of fp16."""
        return self.effective_bits / 16.0

    @property
    def is_native(self) -> bool:
        """True when the C++ engine can compress this tier without Python."""
        return self.native_codec is not None

    def supports(
        self,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> bool:
        """Capability check used instead of name-based feature detection."""
        if device is not None:
            device_type = torch.device(device).type
            if device_type not in self.supported_devices:
                return False
        if dtype is not None and dtype not in self.supported_dtypes:
            return False
        return True
