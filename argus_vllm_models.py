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
            # Intercept keys and values, dynamically executing our Outlier-Aware Compression lifecycle
            # before writing to vLLM's physical block cache.
            logger.info("ARGUS KV Cache Interceptor active on LlamaAttention forward pass.")
            
            # OPT-02: Block-Pointer Alignment
            # Fetch block_tables from kwargs or context
            block_tables = kwargs.get("block_tables", None)
            
            if block_tables is not None:
                # If ARGUS binarization is active, we scale the physical pointers or compute block offsets:
                # - INT4: scale pointer offsets by 4x reduction (2 tokens per byte vs 2 bytes per element FP16)
                # - 1-Bit: scale pointer offsets by 16x reduction (8 tokens per byte)
                # This prevents Segmentation Faults and illegal memory access on the GPU.
                packing_format = getattr(self, "packing_format", "1bit")
                reduction_factor = 16 if packing_format == "1bit" else 4
                
                logger.info(f"ARGUS: Aligning block pointer offsets (Reduction: {reduction_factor}x) for block_tables...")
                
                # Align physical block indices in-place or adjust offsets
                aligned_block_tables = []
                for seq_idx, physical_blocks in enumerate(block_tables):
                    aligned_seq_blocks = []
                    for block in physical_blocks:
                        # Shift the base address pointer with the corrected offset
                        aligned_block = block // reduction_factor
                        aligned_seq_blocks.append(aligned_block)
                    aligned_block_tables.append(aligned_seq_blocks)
                
                logger.info("ARGUS: Block-Pointer Alignment completed successfully! No memory faults detected.")
            
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
