from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol, List, Dict, Any
import torch

class QuantizationBackend(Protocol):
    """Interface for pluggable quantization backends in ARGUS."""
    def compress(self, tensor: torch.Tensor, **kwargs) -> Dict[str, Any]:
        """Compresses the input tensor and returns a metadata dictionary with compressed weights."""
        ...
    def decompress(self, compressed: Dict[str, Any], **kwargs) -> torch.Tensor:
        """Restores the original tensor (approximately) from the compressed representation."""
        ...
    def decompress_batch(self, compressed_list: List[Dict[str, Any]], **kwargs) -> List[torch.Tensor]:
        """Restores multiple tensors in a batch to minimize kernel launch overhead."""
        ...
    def memory_bytes(self, compressed: Dict[str, Any]) -> int:
        """Returns the estimated memory usage in bytes of the compressed representation."""
        ...

class EvictionPolicy(Protocol):
    """Interface for pluggable page replacement/eviction policies in ARGUS."""
    def select_victim(self, pages: List[Dict[str, Any]], context: Dict[str, Any]) -> Dict[str, Any]:
        """Selects a page to demote or evict."""
        ...
    def on_access(self, page: Dict[str, Any], attention_score: float, step: int):
        """Callback to update page statistics when a page is accessed."""
        ...

class ScoringFunction(Protocol):
    """Interface for page importance scoring functions."""
    def score(self, page: Dict[str, Any], step: int, **kwargs) -> float:
        """Calculates a page's importance score."""
        ...

@dataclass
class TierSpec:
    """Defines a single memory cache tier in the cascading hierarchy."""
    name: str                                    # e.g., "fp8", "int8", "one_bit", "jl"
    backend: QuantizationBackend                 # Pluggable compression backend
    max_pages: int = 2                           # Maximum pages allowed in this tier
    priority: int = 0                            # Priority of the tier (lower = colder)
    use_outlier_isolation: bool = True            # Enable outlier isolation sidecar
    use_static_pool: bool = True                 # Use pre-allocated static pool
    description: str = ""                        # Text description

@dataclass
class PipelineConfig:
    """Top-level configuration mapping the cascading memory pipeline."""
    tiers: List[TierSpec] = field(default_factory=list)
    eviction_policy: Optional[EvictionPolicy] = None
    scoring_function: Optional[ScoringFunction] = None
    page_size: int = 4096
    sink_tokens: int = 4
    threshold_sigma: float = 3.0
    vram_oom_threshold_ratio: float = 0.90
    max_active_pages: int = 2
    importance_alpha: float = 0.5
    importance_beta: float = 0.3
    importance_gamma: float = 0.2
    resurrection_threshold: float = 0.01
    micro_page_size: Optional[int] = None
    balloon_driver: Optional[Any] = None
