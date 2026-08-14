"""Cached Johnson-Lindenstrauss projection and reconstruction operators.

JL is the only tier that is not a scalar codec: it compresses by projecting
along the *sequence* axis, so its operators depend on the page's sequence
length rather than on a bit width. Micro-pages produced by variable
granularity therefore need their own operators -- reusing a full-page matrix
against a half-length page is a shape error at best and a silent decode
against the wrong basis at worst.

Building the reconstruction operator costs an N x N inverse, so both are
cached per ``(device, dtype, seq_len)``.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import torch

#: Seed for the projection basis. It must be fixed: a page compressed in one
#: process and decompressed in another has to decode against the same basis.
_PROJECTION_SEED = 42

CacheKey = Tuple[torch.device, torch.dtype, int]


class JLOperatorCache:
    """Keyed cache of JL projection matrices and their reconstruction operators.

    Args:
        page_size: default sequence length, used when a caller passes no
            ``seq_len``. Also defines which operators count as "full page".
        ratio: compression ratio along the sequence axis. A page of length N
            projects to ``max(1, N // ratio)`` rows.
        on_full_page_cuda: optional observer called as ``(kind, tensor)`` with
            ``kind`` in ``{"projection", "reconstruction"}`` whenever a
            full-page CUDA operator is built. The native manager holds exactly
            one of each for its auto-cascade; publishing a CPU-built or
            micro-page operator would poison the next CUDA demotion, so those
            never fire the observer.
    """

    def __init__(
        self,
        page_size: int,
        ratio: int = 4,
        on_full_page_cuda: Optional[Callable[[str, torch.Tensor], None]] = None,
    ) -> None:
        self.page_size = page_size
        self.ratio = ratio
        self._on_full_page_cuda = on_full_page_cuda
        self._projections: Dict[CacheKey, torch.Tensor] = {}
        self._reconstructions: Dict[CacheKey, torch.Tensor] = {}

    def _publishable(self, device, seq_len: int) -> bool:
        return seq_len == self.page_size and torch.device(device).type == "cuda"

    def projection(self, device, dtype, seq_len: Optional[int] = None) -> torch.Tensor:
        """Return the ``[seq_len // ratio, seq_len]`` projection matrix."""
        seq_len = seq_len or self.page_size
        key: CacheKey = (device, dtype, seq_len)
        cached = self._projections.get(key)
        if cached is None:
            n = seq_len
            m = max(1, seq_len // self.ratio)
            torch.manual_seed(_PROJECTION_SEED)  # Keep it deterministic
            raw_randn = torch.randn(n, m, dtype=torch.float32, device=device)
            q, _ = torch.linalg.qr(raw_randn)
            cached = q.t().to(dtype)  # [M, N]
            self._projections[key] = cached
            if self._on_full_page_cuda is not None and self._publishable(device, seq_len):
                self._on_full_page_cuda("projection", cached)
        return cached

    def reconstruction(
        self,
        device,
        dtype,
        alpha: float = 1e-3,
        seq_len: Optional[int] = None,
    ) -> torch.Tensor:
        """Return the ``[seq_len, seq_len // ratio]`` reconstruction operator.

        This is the smoothness-regularized least-squares pseudo-inverse: among
        all x satisfying ``W x = q`` it picks the one minimizing ``x' L x`` for
        a 1-D Laplacian L, which is the right prior for KV activations along a
        sequence axis. ``alpha`` regularizes L so it stays invertible.
        """
        seq_len = seq_len or self.page_size
        key: CacheKey = (device, dtype, seq_len)
        cached = self._reconstructions.get(key)
        if cached is None:
            w_proj = self.projection(device, dtype, seq_len)
            _, n = w_proj.shape

            # Standard 1D Laplacian L [N, N] with Neumann ends.
            L = torch.zeros(n, n, dtype=torch.float32, device=device)
            for i in range(n):
                L[i, i] = 2.0
                if i > 0:
                    L[i, i - 1] = -1.0
                if i < n - 1:
                    L[i, i + 1] = -1.0
            L[0, 0] = 1.0
            L[n - 1, n - 1] = 1.0

            A = L + alpha * torch.eye(n, dtype=torch.float32, device=device)
            A_inv = torch.inverse(A)

            W = w_proj.to(torch.float32)
            W_A_inv_WT = torch.matmul(torch.matmul(W, A_inv), W.t())
            inv_term = torch.inverse(W_A_inv_WT)

            recon_operator = torch.matmul(torch.matmul(A_inv, W.t()), inv_term)

            cached = recon_operator.to(dtype)
            self._reconstructions[key] = cached
            if self._on_full_page_cuda is not None and self._publishable(device, seq_len):
                self._on_full_page_cuda("reconstruction", cached)
        return cached

    def clear(self) -> None:
        self._projections.clear()
        self._reconstructions.clear()

    def cache_size(self) -> int:
        """Number of distinct projection keys currently cached."""
        return len(self._projections)
