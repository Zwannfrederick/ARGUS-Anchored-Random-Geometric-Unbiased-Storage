import torch
from typing import Dict, Any
from argus_cache.core.tier_registry import QuantizationBackend
from argus_cache.core.quantization import (
    quantize_to_fp8_simulated,
    dequantize_from_fp8_simulated,
    quantize_to_int8,
    dequantize_from_int8,
    quantize_to_int4_packed,
    dequantize_from_int4_packed,
    quantize_to_int2_packed,
    dequantize_from_int2_packed,
    quantize_to_1bit_packed,
    dequantize_from_1bit_packed,
    quantize_to_jl_projection,
    dequantize_from_jl_projection,
)

class FP8Backend(QuantizationBackend):
    """Wraps simulated FP8 quantization backend."""
    def compress(self, tensor: torch.Tensor, **kwargs) -> Dict[str, Any]:
        q, scales = quantize_to_fp8_simulated(tensor, dim=-1)
        return {"q": q, "scales": scales}

    def decompress(self, compressed: Dict[str, Any], **kwargs) -> torch.Tensor:
        return dequantize_from_fp8_simulated(compressed["q"], compressed["scales"])

    def decompress_batch(self, compressed_list: list, **kwargs) -> list:
        if not compressed_list:
            return []
        qs = torch.stack([c["q"] for c in compressed_list], dim=0)
        scales = torch.stack([c["scales"] for c in compressed_list], dim=0)
        decompressed = dequantize_from_fp8_simulated(qs, scales)
        return list(decompressed.unbind(dim=0))

    def memory_bytes(self, compressed: Dict[str, Any]) -> int:
        return (compressed["q"].nelement() * compressed["q"].element_size() +
                compressed["scales"].nelement() * compressed["scales"].element_size())


class INT8Backend(QuantizationBackend):
    """Wraps symmetric INT8 quantization backend."""
    def compress(self, tensor: torch.Tensor, **kwargs) -> Dict[str, Any]:
        q, scales = quantize_to_int8(tensor, dim=-1)
        return {"q": q, "scales": scales}

    def decompress(self, compressed: Dict[str, Any], **kwargs) -> torch.Tensor:
        return dequantize_from_int8(compressed["q"], compressed["scales"])

    def decompress_batch(self, compressed_list: list, **kwargs) -> list:
        if not compressed_list:
            return []
        qs = torch.stack([c["q"] for c in compressed_list], dim=0)
        scales = torch.stack([c["scales"] for c in compressed_list], dim=0)
        decompressed = dequantize_from_int8(qs, scales)
        return list(decompressed.unbind(dim=0))

    def memory_bytes(self, compressed: Dict[str, Any]) -> int:
        return (compressed["q"].nelement() * compressed["q"].element_size() +
                compressed["scales"].nelement() * compressed["scales"].element_size())


class INT4Backend(QuantizationBackend):
    """Wraps packed asymmetric INT4 quantization backend."""
    def compress(self, tensor: torch.Tensor, **kwargs) -> Dict[str, Any]:
        seq_dim = kwargs.get("seq_dim", -2)
        quant_dim = kwargs.get("quant_dim", -1)
        q, scales, min_vals = quantize_to_int4_packed(tensor, seq_dim=seq_dim, quant_dim=quant_dim)
        return {"q": q, "scales": scales, "min_vals": min_vals}

    def decompress(self, compressed: Dict[str, Any], **kwargs) -> torch.Tensor:
        seq_dim = kwargs.get("seq_dim", -2)
        return dequantize_from_int4_packed(
            compressed["q"], compressed["scales"], compressed["min_vals"], seq_dim=seq_dim
        )

    def decompress_batch(self, compressed_list: list, **kwargs) -> list:
        if not compressed_list:
            return []
        if len(compressed_list) == 1:
            return [self.decompress(compressed_list[0], **kwargs)]
        
        seq_dim = kwargs.get("seq_dim", -2)
        qs = torch.cat([c["q"] for c in compressed_list], dim=seq_dim)
        scales = torch.cat([c["scales"] for c in compressed_list], dim=seq_dim)
        min_vals = torch.cat([c["min_vals"] for c in compressed_list], dim=seq_dim)
        
        decompressed = dequantize_from_int4_packed(qs, scales, min_vals, seq_dim=seq_dim)
        split_sizes = [c["q"].shape[seq_dim] * 2 for c in compressed_list]
        return list(torch.split(decompressed, split_sizes, dim=seq_dim))

    def memory_bytes(self, compressed: Dict[str, Any]) -> int:
        return (compressed["q"].nelement() * compressed["q"].element_size() +
                compressed["scales"].nelement() * compressed["scales"].element_size() +
                compressed["min_vals"].nelement() * compressed["min_vals"].element_size())


class INT2Backend(QuantizationBackend):
    """Wraps packed asymmetric INT2 quantization backend."""
    def compress(self, tensor: torch.Tensor, **kwargs) -> Dict[str, Any]:
        seq_dim = kwargs.get("seq_dim", -2)
        quant_dim = kwargs.get("quant_dim", -1)
        q, scales, min_vals = quantize_to_int2_packed(tensor, seq_dim=seq_dim, quant_dim=quant_dim)
        return {"q": q, "scales": scales, "min_vals": min_vals}

    def decompress(self, compressed: Dict[str, Any], **kwargs) -> torch.Tensor:
        seq_dim = kwargs.get("seq_dim", -2)
        return dequantize_from_int2_packed(
            compressed["q"], compressed["scales"], compressed["min_vals"], seq_dim=seq_dim
        )

    def decompress_batch(self, compressed_list: list, **kwargs) -> list:
        if not compressed_list:
            return []
        if len(compressed_list) == 1:
            return [self.decompress(compressed_list[0], **kwargs)]
            
        seq_dim = kwargs.get("seq_dim", -2)
        qs = torch.cat([c["q"] for c in compressed_list], dim=seq_dim)
        scales = torch.cat([c["scales"] for c in compressed_list], dim=seq_dim)
        min_vals = torch.cat([c["min_vals"] for c in compressed_list], dim=seq_dim)
        
        decompressed = dequantize_from_int2_packed(qs, scales, min_vals, seq_dim=seq_dim)
        split_sizes = [c["q"].shape[seq_dim] * 4 for c in compressed_list]
        return list(torch.split(decompressed, split_sizes, dim=seq_dim))

    def memory_bytes(self, compressed: Dict[str, Any]) -> int:
        return (compressed["q"].nelement() * compressed["q"].element_size() +
                compressed["scales"].nelement() * compressed["scales"].element_size() +
                compressed["min_vals"].nelement() * compressed["min_vals"].element_size())


class OneBitBackend(QuantizationBackend):
    """Wraps packed binarized 1-Bit quantization backend."""
    def compress(self, tensor: torch.Tensor, **kwargs) -> Dict[str, Any]:
        seq_dim = kwargs.get("seq_dim", -2)
        quant_dim = kwargs.get("quant_dim", -1)
        q, scales = quantize_to_1bit_packed(tensor, seq_dim=seq_dim, quant_dim=quant_dim)
        return {"q": q, "scales": scales}

    def decompress(self, compressed: Dict[str, Any], **kwargs) -> torch.Tensor:
        seq_dim = kwargs.get("seq_dim", -2)
        return dequantize_from_1bit_packed(compressed["q"], compressed["scales"], seq_dim=seq_dim)

    def decompress_batch(self, compressed_list: list, **kwargs) -> list:
        if not compressed_list:
            return []
        if len(compressed_list) == 1:
            return [self.decompress(compressed_list[0], **kwargs)]
            
        seq_dim = kwargs.get("seq_dim", -2)
        qs = torch.cat([c["q"] for c in compressed_list], dim=seq_dim)
        scales = torch.cat([c["scales"] for c in compressed_list], dim=seq_dim)
        
        decompressed = dequantize_from_1bit_packed(qs, scales, seq_dim=seq_dim)
        split_sizes = [c["q"].shape[seq_dim] * 8 for c in compressed_list]
        return list(torch.split(decompressed, split_sizes, dim=seq_dim))

    def memory_bytes(self, compressed: Dict[str, Any]) -> int:
        return (compressed["q"].nelement() * compressed["q"].element_size() +
                compressed["scales"].nelement() * compressed["scales"].element_size())


class JLProjectionBackend(QuantizationBackend):
    """Wraps Laplacian-regularized Johnson-Lindenstrauss random projection backend."""
    def compress(self, tensor: torch.Tensor, **kwargs) -> Dict[str, Any]:
        w_proj = kwargs.get("w_proj")
        if w_proj is None:
            ratio = kwargs.get("ratio", 4)
            compressed, w_proj = quantize_to_jl_projection(tensor, ratio=ratio)
        else:
            compressed = torch.matmul(w_proj, tensor)
        
        return {"q": compressed, "w_proj": w_proj}

    def decompress(self, compressed: Dict[str, Any], **kwargs) -> torch.Tensor:
        recon_operator = kwargs.get("recon_operator")
        alpha = kwargs.get("alpha", 1e-3)
        return dequantize_from_jl_projection(
            compressed["q"], compressed["w_proj"], recon_operator=recon_operator, alpha=alpha
        )

    def decompress_batch(self, compressed_list: list, **kwargs) -> list:
        if not compressed_list:
            return []
        if len(compressed_list) == 1:
            return [self.decompress(compressed_list[0], **kwargs)]
            
        recon_operator = kwargs.get("recon_operator")
        first_q = compressed_list[0]["q"]
        if recon_operator is not None and all(c["q"].shape == first_q.shape for c in compressed_list):
            qs = torch.stack([c["q"] for c in compressed_list], dim=0)
            decompressed = torch.matmul(recon_operator, qs)
            return list(decompressed.unbind(dim=0))
            
        return [self.decompress(c, **kwargs) for c in compressed_list]

    def memory_bytes(self, compressed: Dict[str, Any]) -> int:
        return compressed["q"].nelement() * compressed["q"].element_size()
