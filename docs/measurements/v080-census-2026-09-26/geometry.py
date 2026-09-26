"""KV bytes per token and MoE bytes per token for the local models, from their GGUF files.

Only layers that keep a growing KV cache count: Qwen3.5/3.6 use full attention on every
4th layer (the rest are linear-attention state); Gemma 4 keeps a sliding window of 512-1024
tokens on most layers, so only its global layers grow (and E4B's last 18 layers share KV).
Usage: PYTHONPATH=<llama.cpp>/gguf-py python3 geometry.py <models dir>
"""
import collections
import sys

import gguf

MODELS = {  # name: (file, growing-KV layers, KV heads, head dim, (experts used, experts total))
    "Qwen3.6-35B-A3B": ("Qwen3.6-35B-A3B-UD-Q4_K_M-MTP.gguf", 10, 2, 256, (8, 256)),
    "UI-Mate-9B": ("tencent_UI-Mate-9B-Q4_K_M.gguf", 8, 4, 256, None),
    "gemma-4-26B-A4B": ("gemma-4-26B-A4B-it-UD-Q4_K_M.gguf", 5, 2, 512, (8, 128)),
    "gemma-4-E4B": ("gemma-4-E4B-it-Q4_K_M.gguf", 4, 2, 512, None),
}
BYTES_PER_ELEMENT = {"f16": 2.0, "q8_0": 34 / 32, "q4_0": 18 / 32}

for name, (path, layers, kv_heads, head_dim, moe) in MODELS.items():
    reader = gguf.GGUFReader(f"{sys.argv[1]}/{path}")
    sizes = collections.Counter()
    for tensor in reader.tensors:
        sizes["all"] += tensor.n_bytes
        sizes["experts"] += tensor.n_bytes if "_exps" in tensor.name else 0
    per_token = {k: 2 * layers * kv_heads * head_dim * v for k, v in BYTES_PER_ELEMENT.items()}
    line = f"{name}: weights {sizes['all'] / 1e9:.2f} GB"
    if moe:
        line += f", experts {sizes['experts'] / 1e9:.2f} GB, active experts per token {sizes['experts'] * moe[0] / moe[1] / 1e6:.0f} MB"
    print(line)
    print("  KV per token: " + ", ".join(f"{k} {v / 1024:.1f} KiB" for k, v in per_token.items()))
    for tokens in (262144, 1048576):
        print(f"  {tokens // 1024}K tokens: " + ", ".join(f"{k} {v * tokens / 1e9:.2f} GB" for k, v in per_token.items()))
