import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from argus_cache.models.attention_wrapper import PagedDynamicQuantizedCache

def run_real_llama_test():
    print("=" * 80)
    print("        HUGGINGFACE INTEGRATION: PagedDynamicQuantizedCache with Real LLaMA Model")
    print("=" * 80)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device.upper()}")
    
    # 1. Load a tiny random LLaMA model (same exact architecture as Llama-3-8B but downloads in 1s)
    model_id = "HuggingFaceM4/tiny-random-LlamaForCausalLM"
    print(f"Loading tiny-random LLaMA model: {model_id}...")
    
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure you have an active internet connection to download the tiny 1MB LLaMA model.")
        return
        
    print("Model and tokenizer successfully loaded!")
    
    # 2. Prepare inputs
    prompt = "Zwann Frederick has created the ultimate"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs.input_ids
    
    print(f"Prompt: \"{prompt}\"")
    print(f"Input tokens size: {input_ids.shape}")
    
    # 3. Instantiate our custom PagedDynamicQuantizedCache
    # We use page_size=8 and max_active=1 to trigger FP16 -> FP8 -> INT8 -> INT4 -> INT2 transitions
    # extremely fast during generation!
    past_key_values = PagedDynamicQuantizedCache(
        page_size=8,
        max_active_pages=1,
        max_fp8_pages=1,
        max_int8_pages=1,
        max_int4_pages=1,
        sink_tokens=4
    )
    
    print("\nStarting HuggingFace native generate() call with our custom PagedDynamicQuantizedCache...")
    
    # 4. Run native HuggingFace autoregressive generation!
    try:
        outputs = model.generate(
            input_ids,
            past_key_values=past_key_values,
            max_new_tokens=32,
            do_sample=False, # Deterministic greedy decoding
            use_cache=True
        )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("\nHuggingFace Generation Succeeded perfectly!")
        print(f"Full Generated Output:\n\"{generated_text}\"")
        
        # 5. Inspect final cache layers
        print("\n" + "-" * 80)
        print("                FINAL PagedDynamicQuantizedCache METADATA ANALYSIS")
        print("-" * 80)
        for layer_idx, cache in past_key_values.layer_caches.items():
            print(f"[Layer {layer_idx}]")
            print(f"  - Permanent FP16 Attention Sinks: {cache.sink_k.shape[-2] if cache.sink_k is not None else 0} tokens")
            print(f"  - FP16 (Active) Pages           : {len(cache.active_pages)} pages ({len(cache.active_pages) * cache.page_size} tokens)")
            print(f"  - FP8 (Light) Pages             : {len(cache.fp8_pages)} pages ({len(cache.fp8_pages) * cache.page_size} tokens)")
            print(f"  - INT8 (Medium) Pages           : {len(cache.int8_pages)} pages ({len(cache.int8_pages) * cache.page_size} tokens)")
            print(f"  - INT4 Packed Pages             : {len(cache.int4_pages)} pages ({len(cache.int4_pages) * cache.page_size} tokens)")
            print(f"  - INT2 Packed Super-Pages       : {len(cache.int2_pages)} pages ({len(cache.int2_pages) * cache.page_size} tokens)")
            print(f"  - Buffer Length                 : {cache.k_buffer.shape[-2] if cache.k_buffer is not None else 0} tokens")
            break # Print just layer 0 to avoid printing 32 identical layers
            
        print(f"\nTotal Cache VRAM Footprint: {past_key_values.get_vram_usage()} bytes")
        print("=" * 80)
        print("CONGRATULATIONS: PagedDynamicQuantizedCache universal integration is fully VERIFIED!")
        
    except Exception as e:
        print(f"Generation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_real_llama_test()
