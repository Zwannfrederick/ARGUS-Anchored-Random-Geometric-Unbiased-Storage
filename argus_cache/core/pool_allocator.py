"""Preallocated per-tier storage for compressed pages.

Compressed pages live in fixed-size pools rather than in freshly allocated
tensors, so a demotion is a copy into a slot instead of an allocator call on
the inference hot path.

A pool's shape is determined by *how the tier stores bits*, never by its name.
Sub-byte codecs pack ``8 // bits`` values into each byte along the sequence
axis, so their ``_q`` pools are correspondingly shorter; affine codecs need a
zero-point pool that signed and sign-packed ones do not. Reading that from the
tier's declared capabilities is what lets a third-party plugin tier get a pool
at all -- the previous name-based branches silently allocated nothing for it,
and the tier then fell back to uncompressed spill with no error.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch


class StaticPoolAllocator:
    """Owns the ``{f"{tier}_{field}": tensor}`` pool mapping for all tiers."""

    def __init__(self, page_size: int) -> None:
        self.page_size = page_size
        #: Callers (including PagedDynamicKVCache.pools_by_tier) hold a
        #: reference to this dict, so it is mutated in place, never rebound.
        self.pools: Dict[str, torch.Tensor] = {}

    def allocate_for_tier(
        self,
        spec,
        max_pages: int,
        device,
        dtype,
        batch: int,
        num_heads: int,
        head_dim: int,
    ) -> Dict[str, torch.Tensor]:
        """Allocate and register the pools for one tier; return just those pools.

        Returns an empty mapping for tiers that cannot use a static pool
        (projection tiers, and tiers that declare no native storage format).
        """
        if max_pages < 0:
            # An unbounded tier (max_pages=-1, the archival floor) has no fixed
            # capacity to preallocate; it allocates per page instead. Passing
            # the sentinel straight to torch.zeros raised "Dimension size must
            # be non-negative", which is how this surfaced.
            return {}

        caps = getattr(spec, "capabilities", None)
        codec = caps.native_codec if caps is not None else None
        if codec is None or codec.kind in ("projection", "passthrough"):
            # A projection tier's compressed extent depends on the projection
            # rank; a passthrough tier stores the page verbatim. Neither has a
            # fixed per-page byte layout to preallocate.
            return {}

        pack_factor = 8 // codec.bits if codec.bits < 8 else 1
        stored_dtype = torch.int8 if codec.kind == "signed_linear" else torch.uint8
        packed_len = self.page_size // pack_factor

        def q_pool():
            return torch.zeros(
                max_pages, batch, num_heads, packed_len, head_dim,
                device=device, dtype=stored_dtype,
            )

        def param_pool():
            return torch.zeros(
                max_pages, batch, num_heads, self.page_size, 1,
                device=device, dtype=dtype,
            )

        allocated = {
            f"{spec.name}_key_q": q_pool(),
            f"{spec.name}_value_q": q_pool(),
            f"{spec.name}_key_scales": param_pool(),
            f"{spec.name}_value_scales": param_pool(),
        }
        if codec.kind == "unsigned_affine":
            # Only an affine codec stores a zero-point alongside the scale.
            allocated[f"{spec.name}_key_min_vals"] = param_pool()
            allocated[f"{spec.name}_value_min_vals"] = param_pool()

        self.pools.update(allocated)
        return allocated

    def ensure(
        self,
        tier_name: str,
        field: str,
        max_pages: int,
        shape,
        dtype,
        device,
    ) -> torch.Tensor:
        """Return a pool, allocating it lazily at an explicitly given shape.

        Used for fields whose extent is only known once a page arrives.
        """
        key = f"{tier_name}_{field}"
        pool = self.pools.get(key)
        if pool is None:
            if max_pages < 0:
                raise ValueError(
                    f"tier {tier_name!r} is unbounded (max_pages={max_pages}) "
                    "and cannot back a fixed-size pool"
                )
            pool = torch.zeros(max_pages, *shape, dtype=dtype, device=device)
            self.pools[key] = pool
        return pool

    def get(self, tier_name: str, field: str) -> Optional[torch.Tensor]:
        return self.pools.get(f"{tier_name}_{field}")

    def release(self, tier_name: str) -> None:
        """Drop every pool belonging to a tier."""
        prefix = f"{tier_name}_"
        for key in [k for k in self.pools if k.startswith(prefix)]:
            del self.pools[key]

    def reset(self) -> None:
        """Drop all pools, keeping the mapping's identity for held references."""
        self.pools.clear()

    def total_bytes(self) -> int:
        return sum(t.nelement() * t.element_size() for t in self.pools.values())
