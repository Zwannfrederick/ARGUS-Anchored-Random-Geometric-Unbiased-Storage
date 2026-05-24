import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

def calculate_standard_vram(seq_len, layers=32, kv_heads=8, head_dim=128):
    # Standard FP16 KV Cache (2 bytes per element, key and value)
    return layers * 2 * kv_heads * head_dim * seq_len * 2

def calculate_argus_vram(seq_len, layers=32, kv_heads=8, head_dim=128, page_size=4096):
    # Sinks (8 tokens) and Anchors (average 32 tokens) in FP16
    sink_tokens = 8
    anchor_tokens = 32
    
    # Active limits
    max_active = 2
    max_fp8 = 2
    max_int8 = 2
    max_int4 = 2
    max_int2 = 2
    max_one_bit = 2
    
    total_bytes = 0
    
    # Static shared projection matrices across all layers and pages (stored once in VRAM)
    shared_matrix_bytes = 2 * (page_size // 4) * page_size * 2
    total_bytes += shared_matrix_bytes
    
    # Each layer has its own cache manager
    for layer in range(layers):
        tokens_left = seq_len
        
        # Sinks & Anchors (FP16 = 2 bytes * 2 for K and V)
        sinks_size = min(tokens_left, sink_tokens + anchor_tokens)
        total_bytes += sinks_size * 2 * kv_heads * head_dim * 2
        tokens_left -= sinks_size
        
        if tokens_left <= 0:
            continue
            
        # Active FP16 Pages (max_active * page_size)
        active_tokens = min(tokens_left, max_active * page_size)
        total_bytes += active_tokens * 2 * kv_heads * head_dim * 2
        tokens_left -= active_tokens
        
        if tokens_left <= 0:
            continue
            
        # FP8 Pages (max_fp8 * page_size) -> 1 byte + scales (float16) + sparse outliers
        fp8_tokens = min(tokens_left, max_fp8 * page_size)
        total_bytes += fp8_tokens * 2 * kv_heads * head_dim * 1 # quantized
        total_bytes += fp8_tokens * 2 * kv_heads * 2 # scales
        total_bytes += int(fp8_tokens * 0.02) * 2 * kv_heads * head_dim * 2
        tokens_left -= fp8_tokens
        
        if tokens_left <= 0:
            continue
            
        # INT8 Pages (max_int8 * page_size) -> 1 byte + scales (float16) + outliers
        int8_tokens = min(tokens_left, max_int8 * page_size)
        total_bytes += int8_tokens * 2 * kv_heads * head_dim * 1
        total_bytes += int8_tokens * 2 * kv_heads * 2 # scales
        total_bytes += int(int8_tokens * 0.02) * 2 * kv_heads * head_dim * 2
        tokens_left -= int8_tokens
        
        if tokens_left <= 0:
            continue
            
        # INT4 Pages (max_int4 * page_size) -> 0.5 byte + scales + mins + outliers
        int4_tokens = min(tokens_left, max_int4 * page_size)
        total_bytes += int4_tokens * 2 * kv_heads * head_dim * 0.5
        total_bytes += int4_tokens * 2 * kv_heads * 2 * 2 # scales and mins
        total_bytes += int(int4_tokens * 0.02) * 2 * kv_heads * head_dim * 2
        tokens_left -= int4_tokens
        
        if tokens_left <= 0:
            continue

        # INT2 Pages (max_int2 * page_size) -> 0.25 byte + scales + mins + outliers
        int2_tokens = min(tokens_left, max_int2 * page_size)
        total_bytes += int2_tokens * 2 * kv_heads * head_dim * 0.25
        total_bytes += int2_tokens * 2 * kv_heads * 2 * 2 # scales and mins
        total_bytes += int(int2_tokens * 0.02) * 2 * kv_heads * head_dim * 2
        tokens_left -= int2_tokens
        
        if tokens_left <= 0:
            continue
            
        # 1-Bit Pages (max_one_bit * page_size) -> 0.125 byte + scales + outliers
        one_bit_tokens = min(tokens_left, max_one_bit * page_size)
        total_bytes += one_bit_tokens * 2 * kv_heads * head_dim * 0.125
        total_bytes += one_bit_tokens * 2 * kv_heads * 2 # scales
        total_bytes += int(one_bit_tokens * 0.02) * 2 * kv_heads * head_dim * 2
        tokens_left -= one_bit_tokens
        
        if tokens_left <= 0:
            continue
            
        # JL Projected FP16 Archive (Compressed 4x along sequence dimension in FP16)
        jl_tokens = tokens_left
        projected_tokens = jl_tokens // 4
        total_bytes += projected_tokens * 2 * kv_heads * head_dim * 2
        
    return total_bytes

def generate_graph():
    seq_lens = np.arange(1000, 100001, 1000)
    standard_vram = []
    argus_vram = []
    
    for l in seq_lens:
        standard_vram.append(calculate_standard_vram(l) / (1024**3)) # to GB
        argus_vram.append(calculate_argus_vram(l) / (1024**3))
        
    # Styling Plot beautifully (Elite Dark Theme)
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    # Background color matches modern IDEs
    fig.patch.set_facecolor('#0f0f12')
    ax.set_facecolor('#141419')
    
    # Plot lines with smooth curve shading
    ax.plot(seq_lens, standard_vram, color='#FF5252', linewidth=2.5, label='Standard KV Cache (FP16)')
    ax.plot(seq_lens, argus_vram, color='#FFB300', linewidth=3.0, label='ARGUS 7-Tier Adaptive Cache')
    
    # Fill areas under curves for sleek contrast
    ax.fill_between(seq_lens, standard_vram, alpha=0.08, color='#FF5252')
    ax.fill_between(seq_lens, argus_vram, alpha=0.15, color='#FFB300')
    
    # Grid and labels
    ax.grid(color='#282833', linestyle='--', linewidth=0.7)
    
    ax.set_title('KV Cache Memory Scaling: Standard vs ARGUS (Beta Phase)', fontsize=14, fontweight='bold', pad=15, color='#FFFFFF')
    ax.set_xlabel('Context Length (Tokens)', fontsize=11, labelpad=10, color='#E0E0E0')
    ax.set_ylabel('Theoretical VRAM Consumption (GB)', fontsize=11, labelpad=10, color='#E0E0E0')
    
    # Format axes tick values warning-free using FuncFormatter
    def k_formatter(x, pos):
        return f'{int(x)//1000}k' if x > 0 else '0'
        
    ax.xaxis.set_major_formatter(FuncFormatter(k_formatter))
    ax.tick_params(axis='both', colors='#B0B0B0', labelsize=9)
    
    # Add an annotation showing the compression factor at 100k context
    standard_100k = standard_vram[-1]
    argus_100k = argus_vram[-1]
    ratio = standard_100k / argus_100k
    
    ax.annotate(f'{ratio:.1f}x Memory Reduction\n({standard_100k:.2f} GB vs {argus_100k:.2f} GB)',
                xy=(100000, argus_100k),
                xytext=(55000, 7.5),
                arrowprops=dict(facecolor='#FFB300', shrink=0.08, width=1.5, headwidth=7),
                fontsize=10.5, fontweight='semibold', color='#FFB300',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#1E1E24', edgecolor='#FFB300', alpha=0.9))
    
    ax.legend(loc='upper left', frameon=True, facecolor='#1A1A22', edgecolor='#2D2D3A', fontsize=10)
    
    plt.tight_layout()
    
    # Save the output visualization
    output_path = 'benchmarks/vram_scaling_graph.png'
    plt.savefig(output_path, facecolor=fig.get_facecolor(), edgecolor='none')
    print(f"Successfully generated scaling benchmark graph and saved to {output_path}")

if __name__ == "__main__":
    generate_graph()
