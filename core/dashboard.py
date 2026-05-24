import os
import sys

def render_dashboard(cache, step, mode="paged_quantized", text_output="", prefetch_hits=0, prefetch_misses=0):
    """
    Renders an elite, high-fidelity terminal dashboard showing real-time KV Cache metrics,
    7-tier page allocations, VRAM savings, and speculative prefetch statistics.
    """
    # ANSI color codes
    CYAN = "\033[96m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    MAGENTA = "\033[95m"
    RED = "\033[91m"
    PURPLE = "\033[35m"
    BOLD = "\033[1m"
    RESET = "\033[0m"
    WHITE = "\033[97m"
    GRAY = "\033[90m"
    
    # Try getting terminal width, default to 80
    try:
        columns, _ = os.get_terminal_size()
    except OSError:
        columns = 80
        
    width = min(columns - 4, 76)
    
    # Clear terminal step-by-step
    sys.stdout.write("\033[H")
    
    print(f"\n{CYAN}{BOLD}⚡ ARGUS: ANCHORED RANDOM GEOMETRIC UNBIASED STORAGE{RESET}".center(width + 15))
    print(f"{GRAY}{'─' * width}{RESET}")
    
    # Mode and status info
    status_str = f"{GREEN}{BOLD}RUNNING (ACTIVE){RESET}"
    if hasattr(cache, "is_swapped_out") and cache.is_swapped_out:
        status_str = f"{YELLOW}{BOLD}SWAPPED TO HOST RAM (GUARD ACTIVE){RESET}"
        
    print(f"{BOLD}Serving Mode: {RESET}{mode.upper():<20} | {BOLD}Status: {RESET}{status_str}")
    print(f"{BOLD}Current Generation Step: {RESET}{step:<10} | {BOLD}Prefetch Hits: {RESET}{GREEN}{prefetch_hits}{RESET} / Miss: {RED}{prefetch_misses}{RESET}")
    print(f"{GRAY}{'─' * width}{RESET}")
    
    # Page tiers and VRAM calculation
    # Retrieve metadata from cache
    # If the input cache is a wrapper (PagedDynamicQuantizedCache), get layer 0 cache
    if hasattr(cache, "layer_caches"):
        layer_cache = cache.layer_caches.get(0)
    else:
        layer_cache = cache

    if layer_cache is not None:
        # Determine dimensions
        heads = 4
        head_dim = 16
        if layer_cache.sink_k is not None:
            heads = layer_cache.sink_k.shape[1]
            head_dim = layer_cache.sink_k.shape[3]
        elif layer_cache.k_buffer is not None:
            heads = layer_cache.k_buffer.shape[1]
            head_dim = layer_cache.k_buffer.shape[3]
            
        p_size = layer_cache.page_size
        
        # Get tier page lists
        t1 = len(layer_cache.active_pages)
        t2 = len(layer_cache.fp8_pages)
        t3 = len(layer_cache.int8_pages)
        t4 = len(layer_cache.int4_pages)
        t5 = len(layer_cache.int2_pages)
        t6 = len(layer_cache.one_bit_pages)
        t7 = len(layer_cache.jl_pages)
        
        sink_tokens = layer_cache.sink_k.shape[-2] if layer_cache.sink_k is not None else 0
        anchor_tokens = layer_cache.anchor_k.shape[-2] if layer_cache.anchor_k is not None else 0
        buffer_tokens = layer_cache.k_buffer.shape[-2] if layer_cache.k_buffer is not None else 0
        
        total_tokens = sink_tokens + anchor_tokens + buffer_tokens + (t1 + t2 + t3 + t4 + t5 + t6 + t7) * p_size
        
        # Standard VRAM in bytes (FP16 = 2 bytes per element, key and value)
        std_bytes = total_tokens * 2 * heads * head_dim * 2
        paged_bytes = cache.get_vram_usage()
        
        saving = 0.0
        if std_bytes > 0:
            saving = ((std_bytes - paged_bytes) / std_bytes) * 100
            
        # Draw Tiers progress bar
        print(f"{BOLD}7-Tier Page Distribution Queue:{RESET}\n")
        
        def make_bar(pages, color, label):
            block_char = "█"
            empty_char = "░"
            bar_len = 10
            filled = min(pages, bar_len)
            bar = f"{color}{block_char * filled}{GRAY}{empty_char * (bar_len - filled)}{RESET}"
            print(f"  {label:<28} : {bar} | {BOLD}{pages}{RESET} pages ({pages * p_size} tokens)")
            
        make_bar(t1, CYAN, "Tier 1: FP16 (Active Pages)")
        make_bar(t2, BLUE, "Tier 2: FP8 (Light Quant)")
        make_bar(t3, GREEN, "Tier 3: INT8 (Medium Quant)")
        make_bar(t4, YELLOW, "Tier 4: INT4 (Heavy Quant)")
        make_bar(t5, MAGENTA, "Tier 5: INT2 (Super Heavy)")
        make_bar(t6, RED, "Tier 6: 1-Bit (Binarized)")
        make_bar(t7, PURPLE, "Tier 7: JL Ortho (Archive)")
        
        print(f"\n  {GRAY}Attention Sinks: {sink_tokens} | VIP Anchors: {anchor_tokens} | Temp Buffer: {buffer_tokens}{RESET}")
        print(f"{GRAY}{'─' * width}{RESET}")
        
        # Memory metrics
        print(f"{BOLD}VRAM Telemetry:{RESET}")
        std_kb = std_bytes / 1024
        pg_kb = paged_bytes / 1024
        saving_color = GREEN if saving > 0 else RED
        
        print(f"  - Standard FP16 KV Cache VRAM : {CYAN}{std_kb:7.2f} KB{RESET}")
        print(f"  - ARGUS Paged Quantized Cache : {MAGENTA}{pg_kb:7.2f} KB{RESET}")
        print(f"  - Net GPU VRAM Savings        : {saving_color}{BOLD}{saving:6.2f}%{RESET}")
    else:
        print(f"  {GRAY}No active KV cache tracked yet.{RESET}")
        print(f"{GRAY}{'─' * width}{RESET}")
        
    print(f"{GRAY}{'─' * width}{RESET}")
    print(f"{BOLD}Autoregressive Generation:{RESET}")
    # Print trailing 60 chars of text output elegantly
    clean_text = text_output.replace('\n', ' ')
    if len(clean_text) > width:
        clean_text = "..." + clean_text[-(width-5):]
    print(f"  {WHITE}\"{clean_text}\"{RESET}")
    print(f"{GRAY}{'─' * width}{RESET}\n")
    sys.stdout.flush()
