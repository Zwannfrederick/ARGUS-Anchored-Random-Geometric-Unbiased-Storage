import json, os, subprocess, sys, pathlib
sys.path.insert(0, 'tests')
import test_native_llama_paged as t
tmp = pathlib.Path(sys.argv[1]); tmp.mkdir(exist_ok=True)
binary = t._compile(tmp, "tests/cpp/test_llama_paged_attention.cpp", ["-DARGUS_TEST_CUDA_TIER"], ["-lllama", "-lggml-base"])
env = {**os.environ, "ARGUS_KV_STAGING_BYTES": "4194304", "ARGUS_KV_GPU_BYTES": sys.argv[2], "ARGUS_KV_PINNED_BYTES": "4194304",
       "ARGUS_TEST_PROMPT_TOKENS": "600", "ARGUS_TEST_AUTO_POLICY": "1", "ARGUS_KV_POLICY": "on"}
for k in ("ARGUS_TEST_REPORT_LOGITS_ONLY", "ARGUS_KV_RAM_BYTES"): env.pop(k, None)
stats = {}
for path in ("staged", "batched"):
    r = subprocess.run([binary, t.MODEL, t._storage(), "f16"], env={**env, "ARGUS_KV_ATTENTION_PATH": path}, capture_output=True, text=True, timeout=600)
    stats[path] = json.loads(next(l for l in r.stdout.splitlines() if l.startswith('{"kv_type"')))["stats"]
a, b = stats["staged"], stats["batched"]
for k in sorted(set(a) | set(b)):
    if a.get(k) != b.get(k) and not k.startswith("profile_"): print(k, a.get(k), b.get(k))
