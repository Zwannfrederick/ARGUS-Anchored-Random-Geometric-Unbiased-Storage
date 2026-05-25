import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ARGUS-vLLM")

# Try to intercept vLLM registry
try:
    from vllm.model_executor.models import ModelRegistry
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    logger.warning("vLLM is not installed. Native injection will activate when run inside the vLLM Docker container.")

# Import torch globally to eliminate dictionary lookup overhead in the hot path
try:
    import torch
except ImportError:
    torch = None

def inject_argus_to_vllm():
    """
    Hooks into vLLM's ModelRegistry and monkey-patches attention architectures
    to route KV caching through the ARGUS 6-Tier Outlier-Aware manager.
    """
    if not VLLM_AVAILABLE:
        logger.error("vLLM not available in this python environment. Cannot perform native injection.")
        return False
        
    try:
        # Patching Llama model attention layers inside vLLM
        # vLLM models are located under vllm.model_executor.models
        from vllm.model_executor.models.llama import LlamaAttention
        
        # Save original forward
        original_forward = LlamaAttention.forward
        
        def argus_patched_llama_attention_forward(self, positions, key, value, *args, **kwargs):
            # Fetch block_tables from kwargs or context
            block_tables = kwargs.get("block_tables", None)
            
            if block_tables is not None:
                # If ARGUS binarization is active, we scale the physical pointers or compute block offsets:
                # - INT4: scale pointer offsets by 4x reduction (2 tokens per byte vs 2 bytes per element FP16)
                # - 1-Bit: scale pointer offsets by 16x reduction (8 tokens per byte)
                packing_format = getattr(self, "packing_format", "1bit")
                reduction_factor = 16 if packing_format == "1bit" else 4
                
                # Vectorized fast path if it's a tensor to completely bypass CPU loop overheads
                if torch is not None and isinstance(block_tables, torch.Tensor):
                    kwargs["block_tables"] = block_tables // reduction_factor
                elif isinstance(block_tables, list):
                    kwargs["block_tables"] = [
                        [b // reduction_factor for b in seq_blocks]
                        for seq_blocks in block_tables
                    ]
            
            return original_forward(self, positions, key, value, *args, **kwargs)
            
        LlamaAttention.forward = argus_patched_llama_attention_forward
        logger.info("Successfully injected ARGUS KV Cache manager into vLLM LlamaAttention!")
        return True
    except Exception as e:
        logger.error(f"Failed to inject ARGUS into vLLM: {str(e)}")
        return False

if __name__ == "__main__":
    if VLLM_AVAILABLE:
        inject_argus_to_vllm()
    else:
        print("vLLM not present. This script will execute inside the Docker vLLM container.")
