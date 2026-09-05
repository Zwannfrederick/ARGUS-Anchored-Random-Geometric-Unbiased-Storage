"""Hybrid cache ownership contract for architectures with mixed linear/full attention (Stage S4).

Explicitly separates:
1. Growing KV-cache memory for full-attention layers (owned by ARGUS PagedDynamicKVCache)
2. Fixed-size recurrent/conv state for linear-attention layers (e.g. Qwen3.8 Gated DeltaNet)

Provides deterministic state digests to prove that ARGUS operations never mutate recurrent state,
and validates lifecycle invariants (rollback, sequence operations, fail-closed guards).
"""

from __future__ import annotations

import enum
import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

from argus_cache.core.memory_manager import ArgusConfig, PagedDynamicKVCache
from argus_cache.core.tier_registry import PipelineConfig, TierSpec


class LayerRole(str, enum.Enum):
    """Explicit capability role for each layer in a hybrid model."""
    FULL_ATTENTION_KV = "full_attention_kv"
    RECURRENT_STATE = "recurrent_state"
    CONV_STATE = "conv_state"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class HybridTopology:
    """Topology breakdown derived strictly from validated model/GGUF metadata."""
    num_total_layers: int
    full_attention_interval: int
    num_full_attention_layers: int
    num_recurrent_layers: int
    layer_roles: Dict[int, LayerRole]
    q_heads: int
    kv_heads: int
    head_dim: int

    @classmethod
    def from_config(cls, config: Any) -> HybridTopology:
        """Derive and validate layer roles from HuggingFace or GGUF metadata."""
        model_type = getattr(config, "model_type", "").lower()
        num_layers = int(getattr(config, "num_hidden_layers", getattr(config, "n_layer", 0)))
        interval = int(getattr(config, "full_attention_interval", 4))
        q_heads = int(getattr(config, "num_attention_heads", getattr(config, "n_head", 24)))
        kv_heads = int(getattr(config, "num_key_value_heads", getattr(config, "n_head_kv", 4)))
        head_dim = int(getattr(config, "head_dim", 256))

        if num_layers <= 0:
            raise ValueError(f"Invalid num_hidden_layers: {num_layers}")
        if interval <= 0:
            raise ValueError(f"Invalid full_attention_interval: {interval}")

        # Derive roles
        roles: Dict[int, LayerRole] = {}
        num_full_attn = 0
        num_recr = 0

        # For Qwen3.5/Qwen3.8 hybrid models:
        # Layers where (il + 1) % interval == 0 (or il % interval == interval - 1) are full attention
        for il in range(num_layers):
            if (il + 1) % interval == 0:
                roles[il] = LayerRole.FULL_ATTENTION_KV
                num_full_attn += 1
            else:
                roles[il] = LayerRole.RECURRENT_STATE
                num_recr += 1

        # Strict validation for Qwen3.5/Qwen3.8 64-layer contracts
        if num_layers == 64 and interval == 4:
            if num_full_attn != 16 or num_recr != 48:
                raise ValueError(
                    f"Contradictory topology: expected 16 full-attention and 48 recurrent layers, "
                    f"got {num_full_attn} full-attention and {num_recr} recurrent layers."
                )

        return cls(
            num_total_layers=num_layers,
            full_attention_interval=interval,
            num_full_attention_layers=num_full_attn,
            num_recurrent_layers=num_recr,
            layer_roles=roles,
            q_heads=q_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
        )

    def is_full_attention(self, layer_idx: int) -> bool:
        return self.layer_roles.get(layer_idx) == LayerRole.FULL_ATTENTION_KV

    def is_recurrent(self, layer_idx: int) -> bool:
        return self.layer_roles.get(layer_idx) == LayerRole.RECURRENT_STATE

    def exact_bf16_kv_bytes_per_token(self) -> int:
        """Exact BF16/FP16 growing KV memory per token for full-attention layers only."""
        # 2 (K and V) * num_full_attn_layers * kv_heads * head_dim * 2 bytes (fp16/bf16)
        return 2 * self.num_full_attention_layers * self.kv_heads * self.head_dim * 2

    def q8_0_kv_bytes_per_token(self) -> float:
        """q8_0 block layout memory per token (34 bytes per 32 elements)."""
        elements = 2 * self.num_full_attention_layers * self.kv_heads * self.head_dim
        return (elements / 32) * 34

    def q4_0_kv_bytes_per_token(self) -> float:
        """q4_0 block layout memory per token (18 bytes per 32 elements)."""
        elements = 2 * self.num_full_attention_layers * self.kv_heads * self.head_dim
        return (elements / 32) * 18


class HybridQwenCache:
    """Manages hybrid model memory: ARGUS owns growing KV, runtime owns recurrent state."""

    def __init__(
        self,
        topology: HybridTopology,
        page_size: int = 128,
        pipeline: Optional[PipelineConfig] = None,
        max_active_pages: int = 2,
    ):
        self.topology = topology
        self.page_size = page_size
        self.pipeline = pipeline
        self.max_active_pages = max_active_pages

        # ARGUS-owned caches for full-attention layers: {layer_idx: PagedDynamicKVCache}
        self.attn_caches: Dict[int, PagedDynamicKVCache] = {}
        # Runtime-owned recurrent state buffers: {layer_idx: torch.Tensor}
        self.recurrent_states: Dict[int, torch.Tensor] = {}
        # Runtime-owned conv state buffers: {layer_idx: torch.Tensor}
        self.conv_states: Dict[int, torch.Tensor] = {}

        # Initialize full-attention layers with PagedDynamicKVCache
        for il, role in self.topology.layer_roles.items():
            if role == LayerRole.FULL_ATTENTION_KV:
                self.attn_caches[il] = PagedDynamicKVCache(
                    page_size=self.page_size,
                    pipeline=self.pipeline,
                    max_active_pages=self.max_active_pages,
                )

    def update_attention_layer(
        self,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> PagedDynamicKVCache:
        """Push new key/value tokens to ARGUS-owned full-attention layer."""
        if not self.topology.is_full_attention(layer_idx):
            raise ValueError(f"Layer {layer_idx} is not a full-attention layer (role: {self.topology.layer_roles.get(layer_idx)})")
        cache = self.attn_caches[layer_idx]
        cache.push_new_tokens(k, v)
        return cache

    def update_recurrent_state(
        self,
        layer_idx: int,
        state: torch.Tensor,
        conv_state: Optional[torch.Tensor] = None,
    ) -> None:
        """Update runtime-owned recurrent / DeltaNet state."""
        if not self.topology.is_recurrent(layer_idx):
            raise ValueError(f"Layer {layer_idx} is not a recurrent layer")
        self.recurrent_states[layer_idx] = state
        if conv_state is not None:
            self.conv_states[layer_idx] = conv_state

    def get_recurrent_state(self, layer_idx: int) -> Optional[torch.Tensor]:
        return self.recurrent_states.get(layer_idx)

    def compute_recurrent_state_digest(self) -> str:
        """Calculates deterministic cryptographic hash of all linear-attention recurrent states."""
        hasher = hashlib.sha256()
        for layer_idx in sorted(self.recurrent_states.keys()):
            state = self.recurrent_states[layer_idx]
            hasher.update(layer_idx.to_bytes(4, byteorder="big"))
            hasher.update(state.detach().cpu().contiguous().numpy().tobytes())
        for layer_idx in sorted(self.conv_states.keys()):
            conv = self.conv_states[layer_idx]
            hasher.update(f"conv_{layer_idx}".encode("utf-8"))
            hasher.update(conv.detach().cpu().contiguous().numpy().tobytes())
        return hasher.hexdigest()

    def snapshot(self) -> Dict[str, Any]:
        """Creates a snapshot for rollback upon cancellation, speculative rejection, or branching."""
        recurrent_clones = {k: v.clone() for k, v in self.recurrent_states.items()}
        conv_clones = {k: v.clone() for k, v in self.conv_states.items()}
        attn_snaps = {k: cache.snapshot() for k, cache in self.attn_caches.items()}
        return {
            "recurrent_states": recurrent_clones,
            "conv_states": conv_clones,
            "attn_snapshots": attn_snaps,
            "digest": self.compute_recurrent_state_digest(),
        }

    def restore(self, snapshot: Dict[str, Any]) -> None:
        """Restores recurrent state and ARGUS attention caches from snapshot and verifies integrity."""
        self.recurrent_states = {k: v.clone() for k, v in snapshot["recurrent_states"].items()}
        self.conv_states = {k: v.clone() for k, v in snapshot["conv_states"].items()}
        current_digest = self.compute_recurrent_state_digest()
        if current_digest != snapshot["digest"]:
            raise RuntimeError(
                f"Recurrent state corruption during restore! Expected digest {snapshot['digest']}, got {current_digest}"
            )
        if "attn_snapshots" in snapshot:
            for k, snap in snapshot["attn_snapshots"].items():
                if k in self.attn_caches:
                    self.attn_caches[k].restore(snap)

    def _cache_seq_length(self, cache: PagedDynamicKVCache) -> int:
        length = 0
        if cache.sink_k is not None and cache.sink_k.numel() > 0:
            length += cache.sink_k.shape[-2]
        if cache.anchor_k is not None and cache.anchor_k.numel() > 0:
            length += cache.anchor_k.shape[-2]
        num_pages = len(cache.active_pages) + sum(len(pages) for pages in cache.pages_by_tier.values())
        length += num_pages * cache.page_size
        if cache.k_buffer is not None and cache.k_buffer.numel() > 0:
            length += cache.k_buffer.shape[-2]
        return length

    def get_seq_length(self) -> int:
        """Returns sequence length of full-attention layers."""
        for cache in self.attn_caches.values():
            return self._cache_seq_length(cache)
        return 0

    def _cache_memory_bytes(self, cache: PagedDynamicKVCache) -> int:
        try:
            _, _, _, comp_bytes = cache.get_cache_telemetry()
            if comp_bytes > 0:
                return comp_bytes
        except Exception:
            pass
        total_tokens = self._cache_seq_length(cache)
        return total_tokens * self.topology.kv_heads * self.topology.head_dim * 2 * 2

    def memory_breakdown(self) -> Dict[str, Any]:
        """Reports independent memory breakdown between ARGUS attention KV and recurrent state."""
        attn_bytes = sum(self._cache_memory_bytes(c) for c in self.attn_caches.values())
        recr_bytes = sum(s.element_size() * s.nelement() for s in self.recurrent_states.values())
        conv_bytes = sum(s.element_size() * s.nelement() for s in self.conv_states.values())
        return {
            "attention_kv_bytes": attn_bytes,
            "recurrent_state_bytes": recr_bytes,
            "conv_state_bytes": conv_bytes,
            "total_hybrid_bytes": attn_bytes + recr_bytes + conv_bytes,
            "num_attention_layers": len(self.attn_caches),
            "num_recurrent_layers": len(self.recurrent_states),
        }
