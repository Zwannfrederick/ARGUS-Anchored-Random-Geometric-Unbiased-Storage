"""Pressure-aware activation policy for the ARGUS data plane.

The policy deliberately separates estimation from cache mechanics.  This
makes the decision testable without a GPU and keeps request routing out of the
memory manager's already hot path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


CacheMode = Literal["latency", "balanced", "capacity"]


@dataclass(frozen=True)
class CacheGeometry:
    """The model dimensions needed to estimate an exact K/V cache."""

    layers: int
    batch_size: int
    kv_heads: int
    head_dim: int
    element_size: int

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor, model_config=None) -> "CacheGeometry":
        if tensor.ndim != 4:
            raise ValueError("KV tensors must have shape [batch, heads, tokens, head_dim]")
        layers = None
        if model_config is not None:
            for name in ("num_hidden_layers", "n_layer", "num_layers"):
                value = getattr(model_config, name, None)
                if value is not None:
                    layers = int(value)
                    break
        if layers is None:
            raise ValueError("model_config must declare its number of hidden layers")
        return cls(
            layers=layers,
            batch_size=int(tensor.shape[0]),
            kv_heads=int(tensor.shape[1]),
            head_dim=int(tensor.shape[3]),
            element_size=int(tensor.element_size()),
        )

    def exact_bytes(self, tokens: int) -> int:
        if tokens < 0:
            raise ValueError("tokens must be non-negative")
        # Two tensors (K and V) exist at every transformer layer.
        return (
            2
            * self.layers
            * self.batch_size
            * self.kv_heads
            * self.head_dim
            * tokens
            * self.element_size
        )


@dataclass(frozen=True)
class MemorySnapshot:
    """Free and total device memory captured at the request boundary."""

    free_bytes: int
    total_bytes: int

    def __post_init__(self):
        if self.total_bytes <= 0:
            raise ValueError("total_bytes must be positive")
        if not 0 <= self.free_bytes <= self.total_bytes:
            raise ValueError("free_bytes must be between zero and total_bytes")

    @classmethod
    def current_cuda(cls, device=None) -> "MemorySnapshot | None":
        if not torch.cuda.is_available():
            return None
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        return cls(int(free_bytes), int(total_bytes))


@dataclass(frozen=True)
class ActivationDecision:
    activate: bool
    reason: str
    exact_bytes: int
    estimated_savings_bytes: int
    projected_utilization: float | None


@dataclass(frozen=True)
class AdaptiveCachePolicy:
    """Choose exact-cache bypass or ARGUS once per request.

    ``balanced`` requires both a capacity gain and projected memory pressure.
    Different activate/deactivate thresholds provide hysteresis between
    requests.  A caller may still escalate from exact to ARGUS during a
    request if the original token estimate proves too small, but it should not
    switch back until the request is reset.
    """

    mode: CacheMode = "balanced"
    expected_tokens: int | None = None
    min_savings_bytes: int = 64 * 1024 * 1024
    activate_ratio: float = 0.90
    deactivate_ratio: float = 0.75

    def __post_init__(self):
        if self.mode not in ("latency", "balanced", "capacity"):
            raise ValueError(f"unknown cache mode: {self.mode!r}")
        if self.expected_tokens is not None and self.expected_tokens <= 0:
            raise ValueError("expected_tokens must be positive when provided")
        if self.min_savings_bytes < 0:
            raise ValueError("min_savings_bytes must be non-negative")
        if not 0.0 <= self.deactivate_ratio < self.activate_ratio <= 1.0:
            raise ValueError(
                "deactivate_ratio must be lower than activate_ratio and both must be in [0, 1]"
            )

    def decide(
        self,
        *,
        geometry: CacheGeometry,
        tokens: int,
        page_size: int,
        max_active_pages: int,
        tier_effective_bits: float,
        memory: MemorySnapshot | None,
        was_active: bool,
    ) -> ActivationDecision:
        predicted_tokens = max(tokens, self.expected_tokens or 0)
        exact_bytes = geometry.exact_bytes(predicted_tokens)
        savings = self._estimated_savings(
            geometry,
            predicted_tokens,
            page_size,
            max_active_pages,
            tier_effective_bits,
        )

        if self.mode == "latency":
            return ActivationDecision(False, "latency-mode", exact_bytes, savings, None)
        if self.mode == "capacity":
            return ActivationDecision(True, "capacity-mode", exact_bytes, savings, None)
        if savings < self.min_savings_bytes:
            return ActivationDecision(False, "insufficient-gain", exact_bytes, savings, None)
        if memory is None:
            return ActivationDecision(False, "memory-unavailable", exact_bytes, savings, None)

        allocated = memory.total_bytes - memory.free_bytes
        projected = min(1.0, (allocated + exact_bytes) / memory.total_bytes)
        if was_active:
            activate = projected > self.deactivate_ratio
            reason = "hysteresis-hold" if activate else "pressure-relieved"
        else:
            activate = projected >= self.activate_ratio
            reason = "memory-pressure" if activate else "exact-cache-fits"
        return ActivationDecision(activate, reason, exact_bytes, savings, projected)

    @staticmethod
    def _estimated_savings(
        geometry: CacheGeometry,
        tokens: int,
        page_size: int,
        max_active_pages: int,
        tier_effective_bits: float,
    ) -> int:
        if page_size <= 0 or max_active_pages < 0:
            raise ValueError("page_size must be positive and max_active_pages non-negative")
        if tier_effective_bits <= 0:
            raise ValueError("tier_effective_bits must be positive")

        active_tokens = min(tokens, page_size * max_active_pages)
        cold_tokens = max(0, tokens - active_tokens)
        exact_cold = geometry.exact_bytes(cold_tokens)
        original_bits = geometry.element_size * 8
        compressed_cold = exact_cold * tier_effective_bits / original_bits
        return max(0, int(exact_cold - compressed_cold))
