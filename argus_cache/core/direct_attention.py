"""Direct ACTIVE + q8_0 single-token attention engine (Stage S6).

Implements exact tile-by-tile online-softmax recurrence directly over descriptor tables
and contiguous quantized block pools without materializing a context-sized FP16 KV tensor.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple
import torch

from argus_cache.core.page_table import CodecKind, PlacementLocation, StructureOfArraysPageTable
from argus_cache.core.backend_pool import ContiguousBlockPool


class DirectPagedAttentionEngine:
    """Exact single-token paged attention engine over heterogeneous precision pools."""

    @staticmethod
    def dequantize_q8_0_page(
        raw_bytes: torch.Tensor,
        batch: int,
        num_heads: int,
        page_size: int,
        head_dim: int,
        target_dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Dequantizes a contiguous q8_0 page block into a single page tile."""
        num_elements = batch * num_heads * page_size * head_dim
        num_blocks = num_elements // 32
        
        # Block layout: 2 bytes fp16 scale, 32 bytes int8 values
        blocks = raw_bytes[: num_blocks * 34].contiguous().view(num_blocks, 34)
        scales = blocks[:, :2].contiguous().view(torch.float16).to(target_dtype) # [num_blocks, 1]
        quants = blocks[:, 2:].contiguous().view(torch.int8).to(target_dtype)     # [num_blocks, 32]
        
        dequantized_flat = (quants * scales).flatten()[:num_elements]
        return dequantized_flat.view(batch, num_heads, page_size, head_dim)

    @staticmethod
    def dequantize_q4_0_page(
        raw_bytes: torch.Tensor,
        batch: int,
        num_heads: int,
        page_size: int,
        head_dim: int,
        target_dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Dequantizes a contiguous q4_0 page block into a single page tile."""
        num_elements = batch * num_heads * page_size * head_dim
        num_blocks = num_elements // 32

        blocks = raw_bytes[: num_blocks * 18].contiguous().view(num_blocks, 18)
        scales = blocks[:, :2].contiguous().view(torch.float16).to(target_dtype) # [num_blocks, 1]
        packed = blocks[:, 2:].contiguous()                                       # [num_blocks, 16]

        low = (packed & 0x0F).to(target_dtype) - 8.0
        high = ((packed >> 4) & 0x0F).to(target_dtype) - 8.0
        quants = torch.cat([low, high], dim=1)                                    # [num_blocks, 32]

        dequantized_flat = (quants * scales).flatten()[:num_elements]
        return dequantized_flat.view(batch, num_heads, page_size, head_dim)

    @classmethod
    @torch.inference_mode()
    def decode_single_token(
        cls,
        query: torch.Tensor,
        page_table: StructureOfArraysPageTable,
        pools: Dict[CodecKind, ContiguousBlockPool],
        active_pages: Optional[Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = None,
        scale: Optional[float] = None,
    ) -> torch.Tensor:
        """Executes exact paged attention across the descriptor table without full KV reconstruction."""
        batch, q_heads, q_len, head_dim = query.shape
        if q_len != 1:
            raise ValueError(f"DirectPagedAttentionEngine supports q_len=1 decode only (got {q_len})")

        if scale is None:
            scale = 1.0 / math.sqrt(head_dim)

        device = query.device
        dtype = query.dtype
        q_f32 = query.to(torch.float32)

        running_max = torch.full((batch, q_heads, 1, 1), float("-inf"), dtype=torch.float32, device=device)
        running_sum = torch.zeros((batch, q_heads, 1, 1), dtype=torch.float32, device=device)
        running_out = torch.zeros((batch, q_heads, 1, head_dim), dtype=torch.float32, device=device)

        num_pages = page_table.num_pages
        if num_pages == 0:
            return torch.zeros_like(query)

        active_dict = active_pages or {}

        for i in range(num_pages):
            page_id = int(page_table.page_ids[i].item())
            codec = CodecKind(int(page_table.codecs[i].item()))
            slot = int(page_table.pool_slots[i].item())
            token_count = int(page_table.token_counts[i].item())

            # Load tile for this page
            if codec == CodecKind.ACTIVE_FP16:
                if page_id in active_dict:
                    k_tile, v_tile = active_dict[page_id]
                    k_tile = k_tile.to(device=device, dtype=torch.float32)
                    v_tile = v_tile.to(device=device, dtype=torch.float32)
                else:
                    pool = pools.get(CodecKind.ACTIVE_FP16)
                    k_bytes, v_bytes = pool.read_page_bytes(slot)
                    k_tile = k_bytes.view(torch.float16).to(device=device, dtype=torch.float32).view(batch, -1, token_count, head_dim)
                    v_tile = v_bytes.view(torch.float16).to(device=device, dtype=torch.float32).view(batch, -1, token_count, head_dim)
            elif codec == CodecKind.GGML_Q8_0:
                pool = pools[CodecKind.GGML_Q8_0]
                k_bytes, v_bytes = pool.read_page_bytes(slot)
                k_tile = cls.dequantize_q8_0_page(k_bytes, batch, pool.num_heads, pool.page_size, head_dim, torch.float32).to(device)
                v_tile = cls.dequantize_q8_0_page(v_bytes, batch, pool.num_heads, pool.page_size, head_dim, torch.float32).to(device)
                if token_count < pool.page_size:
                    k_tile = k_tile[:, :, :token_count, :]
                    v_tile = v_tile[:, :, :token_count, :]
            elif codec == CodecKind.GGML_Q4_0:
                pool = pools[CodecKind.GGML_Q4_0]
                k_bytes, v_bytes = pool.read_page_bytes(slot)
                k_tile = cls.dequantize_q4_0_page(k_bytes, batch, pool.num_heads, pool.page_size, head_dim, torch.float32).to(device)
                v_tile = cls.dequantize_q4_0_page(v_bytes, batch, pool.num_heads, pool.page_size, head_dim, torch.float32).to(device)
                if token_count < pool.page_size:
                    k_tile = k_tile[:, :, :token_count, :]
                    v_tile = v_tile[:, :, :token_count, :]
            else:
                raise NotImplementedError(f"Unsupported codec in direct attention engine: {codec}")

            # GQA Repeat if needed
            kv_heads = k_tile.shape[1]
            if q_heads != kv_heads:
                assert q_heads % kv_heads == 0, f"q_heads ({q_heads}) must be multiple of kv_heads ({kv_heads})"
                groups = q_heads // kv_heads
                k_tile = k_tile.repeat_interleave(groups, dim=1)
                v_tile = v_tile.repeat_interleave(groups, dim=1)

            # Online Softmax Recurrence
            # scores: [B, q_heads, 1, token_count]
            scores = torch.matmul(q_f32, k_tile.transpose(-1, -2)) * scale
            tile_max = scores.amax(dim=-1, keepdim=True)
            next_max = torch.maximum(running_max, tile_max)

            is_first = torch.isneginf(running_max)
            rescale = torch.where(is_first, torch.zeros_like(running_max), torch.exp(running_max - next_max))
            tile_exp = torch.exp(scores - next_max)
            tile_exp = torch.where(torch.isneginf(tile_max), torch.zeros_like(tile_exp), tile_exp)

            running_out = running_out * rescale + torch.matmul(tile_exp, v_tile)
            running_sum = running_sum * rescale + tile_exp.sum(dim=-1, keepdim=True)
            running_max = next_max

        # Final normalize and cast back to query dtype
        output = (running_out / running_sum.clamp_min(1e-8)).to(dtype)
        return output
