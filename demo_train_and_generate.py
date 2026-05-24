import torch
import torch.nn as nn
import torch.optim as optim
import random
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from argus_cache.core.memory_manager import PagedDynamicKVCache
from argus_cache.models.attention_wrapper import PagedDynamicQuantizedCache

# =====================================================================
# QAT COMPRESSION NOISE SIMULATOR FOR TRAINING RESILIENCE
# =====================================================================
def simulate_compression_noise(tensor):
    if torch.rand(1).item() < 0.3:
        # 1. Simulate FP8/INT8 noise
        max_val = torch.max(torch.abs(tensor)) + 1e-8
        scale = max_val / 127.0
        q = torch.round(tensor / scale).clamp(-128, 127)
        return q * scale
    elif torch.rand(1).item() < 0.3:
        # 2. Simulate INT4 packed noise
        min_val = torch.min(tensor)
        max_val = torch.max(tensor)
        scale = (max_val - min_val) / 15.0 + 1e-8
        q = torch.round((tensor - min_val) / scale).clamp(0, 15)
        return q * scale + min_val
    return tensor

# =====================================================================
# 1. SYNTHETIC CORPUS (TR POEM FOR CONCRETE TRAINED TEXT GENERATION)
# =====================================================================

text_corpus = """
zwann frederick bir yola cikti localde rtx ile canavar yaratti.
mamba ve transformer el ele verdi, paged quantize cache bellegi eritti.
oom hatalari geride kaldi, bitirme projesi kapilari acti.
akademik dunya gururla izler, bu temiz mimari sanayide eser.
"""

# Character-level vocabulary mapping
chars = sorted(list(set(text_corpus)))
vocab_size = len(chars)
char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}

# Helper functions to convert string to tensor and vice-versa
def string_to_tensor(string, device="cpu"):
    return torch.tensor([char_to_ix[c] for c in string], dtype=torch.long, device=device).unsqueeze(0)

def tensor_to_string(tensor):
    return "".join([ix_to_char[i.item()] for i in tensor.squeeze(0)])

# =====================================================================
# 2. MINI-MODEL LAYER COMPONENT DEFINITIONS
# =====================================================================

class CausalAttention(nn.Module):
    """Causal Attention with exact standard cache support."""
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, cache=None, layer_idx=0, qat_mode=False, token_ids=None):
        b, s, e = x.shape
        device = x.device
        
        q = self.q_proj(x).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        
        if self.training and qat_mode:
            k = simulate_compression_noise(k)
            v = simulate_compression_noise(v)
            
        if cache is not None:
            # We support both standard dict cache and HuggingFace PagedDynamicQuantizedCache
            if isinstance(cache, dict):
                if cache.get(layer_idx) is None:
                    cache[layer_idx] = (k, v)
                else:
                    k_prev, v_prev = cache[layer_idx]
                    k = torch.cat([k_prev, k], dim=-2)
                    v = torch.cat([v_prev, v], dim=-2)
                    cache[layer_idx] = (k, v)
            else:
                # Dynamic "Rhyme-Anchor" & Newline structural detection for our paged cache!
                is_anchor = None
                if token_ids is not None:
                    anchor_chars = ['\n', '.', ',']
                    anchor_indices_in_vocab = [char_to_ix[c] for c in anchor_chars if c in char_to_ix]
                    
                    is_anchor = torch.zeros_like(token_ids, dtype=torch.bool)
                    for idx_in_vocab in anchor_indices_in_vocab:
                        is_anchor = is_anchor | (token_ids == idx_in_vocab)
                    is_anchor = is_anchor[0] # sequence 1D mask
                    
                # Pass is_anchor via cache_kwargs
                k, v = cache.update(k, v, layer_idx, cache_kwargs={"is_anchor": is_anchor})

        # Causal Attention masking
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # Apply causal mask for autoregressive generation
        q_len, k_len = q.shape[-2], k.shape[-2]
        mask = torch.triu(torch.ones(q_len, k_len, device=device), diagonal=k_len - q_len + 1).bool()
        attn_weights = attn_weights.masked_fill(mask.unsqueeze(0).unsqueeze(1), float('-inf'))
        
        attn_probs = torch.softmax(attn_weights, dim=-1)
        out = torch.matmul(attn_probs, v)
        
        out = out.transpose(1, 2).contiguous().view(b, s, e)
        return self.out_proj(out)


class SSMLayer(nn.Module):
    """Recurrent SSM Scan Layer (behaves like Mamba)."""
    def __init__(self, embed_dim, state_dim=64):
        super().__init__()
        self.embed_dim = embed_dim
        self.state_dim = state_dim
        
        self.x_proj = nn.Linear(embed_dim, state_dim)
        self.delta_proj = nn.Linear(embed_dim, state_dim)
        self.out_proj = nn.Linear(state_dim, embed_dim)
        self.A = nn.Parameter(torch.randn(state_dim) * 0.05 - 0.1) # negative decay

    def forward(self, x, ssm_state=None):
        b, s, e = x.shape
        device = x.device
        
        inputs = self.x_proj(x)
        deltas = torch.sigmoid(self.delta_proj(x))
        
        h = ssm_state if ssm_state is not None else torch.zeros(b, self.state_dim, device=device)
        outputs = []
        for t in range(s):
            u_t = inputs[:, t, :]
            d_t = deltas[:, t, :]
            
            # Recurrent linear scan step
            alpha = torch.exp(self.A * d_t)
            beta = d_t
            h = alpha * h + beta * u_t
            outputs.append(h.unsqueeze(1))
            
        outputs = torch.cat(outputs, dim=1)
        return self.out_proj(outputs), h


# =====================================================================
# 3. UNIFIED LANGUAGE MODEL CLASS
# =====================================================================

class UnifiedLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, num_heads=4, mode="transformer"):
        super().__init__()
        self.mode = mode # "transformer", "qat_transformer", or "ssm"
        
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, 1000, embed_dim))
        
        if mode in ["transformer", "qat_transformer"]:
            self.layer = CausalAttention(embed_dim, num_heads)
        else:
            self.layer = SSMLayer(embed_dim, state_dim=128)
            
        self.ln = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, idx, cache=None, ssm_state=None):
        b, s = idx.shape
        x = self.token_emb(idx)
        
        if self.mode in ["transformer", "qat_transformer"]:
            # Add positional embedding
            x = x + self.pos_emb[:, :s, :]
            qat_mode = (self.mode == "qat_transformer")
            x = self.layer(x, cache=cache, qat_mode=qat_mode, token_ids=idx)
            next_state = None
        else:
            # SSM runs recurrently without positional embeddings
            x, next_state = self.layer(x, ssm_state=ssm_state)
            
        x = self.ln(x)
        logits = self.lm_head(x)
        return logits, next_state

# =====================================================================
# 4. TRAINING DEMO (CPU or GPU, extremely fast)
# =====================================================================

def train_demo():
    print("=" * 80)
    print("                DEMO: TRAINING CHARACTER-LEVEL GPT MODEL")
    print("=" * 80)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device.upper()}")
    
    # 1. Instantiate the three models
    embed_dim = 64
    num_heads = 4
    
    trans_model = UnifiedLanguageModel(vocab_size, embed_dim, num_heads, mode="transformer").to(device)
    ssm_model = UnifiedLanguageModel(vocab_size, embed_dim, num_heads, mode="ssm").to(device)
    qat_model = UnifiedLanguageModel(vocab_size, embed_dim, num_heads, mode="qat_transformer").to(device)
    
    criterion = nn.CrossEntropyLoss()
    opt_trans = optim.Adam(trans_model.parameters(), lr=0.005)
    opt_ssm = optim.Adam(ssm_model.parameters(), lr=0.005)
    opt_qat = optim.Adam(qat_model.parameters(), lr=0.005)
    
    # Simple training inputs/targets
    input_text = text_corpus[:-1]
    target_text = text_corpus[1:]
    
    x_tensor = string_to_tensor(input_text, device)
    y_tensor = string_to_tensor(target_text, device)
    
    # Train for 200 epochs
    epochs = 200
    for epoch in range(epochs):
        # 1. Train Transformer
        opt_trans.zero_grad()
        logits_t, _ = trans_model(x_tensor)
        loss_t = criterion(logits_t.view(-1, vocab_size), y_tensor.view(-1))
        loss_t.backward()
        opt_trans.step()
        
        # 2. Train SSM
        opt_ssm.zero_grad()
        logits_s, _ = ssm_model(x_tensor)
        loss_s = criterion(logits_s.view(-1, vocab_size), y_tensor.view(-1))
        loss_s.backward()
        opt_ssm.step()
        
        # 3. Train QAT Transformer
        opt_qat.zero_grad()
        logits_q, _ = qat_model(x_tensor)
        loss_q = criterion(logits_q.view(-1, vocab_size), y_tensor.view(-1))
        loss_q.backward()
        opt_qat.step()
        
        if (epoch + 1) % 50 == 0:
            print(f"Step {epoch+1:3d} | Transformer Loss: {loss_t.item():.4f} | Mamba SSM Loss: {loss_s.item():.4f} | QAT Trans Loss: {loss_q.item():.4f}")

    print("\nTraining completed successfully! Models have memorized the corpus pattern.")
    
    return trans_model, ssm_model, qat_model, device

# =====================================================================
# 5. AUTOREGRESSIVE TEXT GENERATION & ANALYSIS
# =====================================================================

def generate_text(model, mode, cache=None, ssm_state=None, prompt="zwann", gen_len=60, device="cpu"):
    """Autoregressive text generation using standard cache, SSM state, or our Paged Cache."""
    model.eval()
    
    current_tokens = string_to_tensor(prompt, device)
    generated = prompt
    
    # Helper function to apply character-level repetition penalty
    def get_penalized_argmax(logits, current_generated):
        next_token_logits = logits[:, -1, :].clone()  # Keep batch dim: [B, vocab]
        # Penalize recently generated characters (sliding window of 15) to prevent repetition loops
        recent_chars = current_generated[-15:]
        for char in recent_chars:
            if char in char_to_ix:
                char_idx = char_to_ix[char]
                next_token_logits[:, char_idx] -= 1.5
        return torch.argmax(next_token_logits, dim=-1)  # Returns [B]
    
    # If using transformer cache, initialize it
    # We will do incremental generation (token-by-token) to verify dynamic cache operation!
    state = ssm_state
    
    with torch.no_grad():
        if mode == "standard_transformer":
            # Initialize standard FP16 dict cache
            model_cache = {}
            # Warmup with prompt
            logits, _ = model(current_tokens, cache=model_cache)
            next_token = get_penalized_argmax(logits, generated)
            generated += ix_to_char[next_token.item()]
            
            # Generate next tokens autoregressively
            for _ in range(gen_len - 1):
                next_token_tensor = next_token.unsqueeze(0)
                logits, _ = model(next_token_tensor, cache=model_cache)
                next_token = get_penalized_argmax(logits, generated)
                generated += ix_to_char[next_token.item()]
                
        elif mode == "ssm":
            # Recurrent sequence scan
            logits, state = model(current_tokens, ssm_state=state)
            next_token = get_penalized_argmax(logits, generated)
            generated += ix_to_char[next_token.item()]
            
            for _ in range(gen_len - 1):
                next_token_tensor = next_token.unsqueeze(0)
                logits, state = model(next_token_tensor, ssm_state=state)
                next_token = get_penalized_argmax(logits, generated)
                generated += ix_to_char[next_token.item()]
                
        elif mode == "paged_quantized":
            # Powered by our 5-tier Paged Cache!
            from core.dashboard import render_dashboard
            import time
            
            paged_cache = PagedDynamicQuantizedCache(
                page_size=8,
                max_active_pages=1,
                max_fp8_pages=1,
                max_int8_pages=1,
                max_int4_pages=1
            )
            
            # Clear terminal at start of live dashboard
            sys.stdout.write("\033[2J\033[H")
            sys.stdout.flush()
            
            # Warmup with prompt
            logits, _ = model(current_tokens, cache=paged_cache)
            next_token = get_penalized_argmax(logits, generated)
            generated += ix_to_char[next_token.item()]
            
            # Show initial dashboard
            render_dashboard(paged_cache, step=0, mode="paged_quantized", text_output=generated)
            time.sleep(0.1)
            
            # Generate autoregressively
            for step in range(1, gen_len):
                next_token_tensor = next_token.unsqueeze(0)
                
                # Dynamic Swap Out showcase at step 20 (Guard activated!)
                if step == 20:
                    paged_cache.swap_out_to_host()
                    render_dashboard(paged_cache, step=step, mode="paged_quantized (Swapped Out)", text_output=generated)
                    time.sleep(0.8) # Let the user see the "SWAPPED OUT" state
                    
                # Auto Prefetching prediction
                paged_cache.speculate_and_prefetch()
                
                logits, _ = model(next_token_tensor, cache=paged_cache)
                next_token = get_penalized_argmax(logits, generated)
                generated += ix_to_char[next_token.item()]
                
                # Fetch layer 0 stats
                c = paged_cache.layer_caches[0]
                render_dashboard(paged_cache, step=step, mode="paged_quantized", text_output=generated, 
                                 prefetch_hits=c.prefetch_hits, prefetch_misses=c.prefetch_misses)
                time.sleep(0.1) # Smooth progression delay
                
            # Log cache stats at the end of generation
            c = paged_cache.layer_caches[0]
            print(f"  [Paged Cache Final Structure]")
            print(f"    - FP16 (Active) Pages: {len(c.active_pages)}")
            print(f"    - FP8 Tiers Pages    : {len(c.fp8_pages)}")
            print(f"    - INT8 Tiers Pages   : {len(c.int8_pages)}")
            print(f"    - INT4 Packed Pages  : {len(c.int4_pages)}")
            print(f"    - INT2 Packed Pages  : {len(c.int2_pages)}")
            print(f"    - Final VRAM Cost    : {paged_cache.get_vram_usage()} bytes")
            
    return generated.replace("\n", " ")

# =====================================================================
# 6. RUN EXPERIMENT DEMO
# =====================================================================

if __name__ == "__main__":
    trans_model, ssm_model, qat_model, device = train_demo()
    
    print("\n" + "=" * 80)
    print("                DEMO: GENERATING TEXT ACROSS 3 ARCHITECTURES")
    print("=" * 80)
    
    prompt = "zwann"
    
    # 1. Standard Transformer
    print("\n1. Standard Transformer Causal Generation (FP16 Cache):")
    text_std = generate_text(trans_model, "standard_transformer", prompt=prompt, device=device)
    print(f"  Generated Output: \"{text_std}\"")
    
    # 2. Mamba SSM Model
    print("\n2. Mamba SSM Layer Causal Generation (Recurrent Compression):")
    text_ssm = generate_text(ssm_model, "ssm", prompt=prompt, device=device)
    print(f"  Generated Output: \"{text_ssm}\"")
    
    # 3. Paged Quantized Transformer (Our Architecture using QAT Weights & Rhyme Outliers!)
    print("\n3. Our Paged 5-Tier Quantized Generation (FP16->FP8->INT8->INT4->JL-Proj Cache) with QAT weights:")
    text_paged = generate_text(qat_model, "paged_quantized", prompt=prompt, device=device)
    print(f"  Generated Output: \"{text_paged}\"")
    print("=" * 80)
