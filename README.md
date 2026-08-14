# ARGUS: Heterogeneous KV-cache memory management

ARGUS is an experimental runtime for storing transformer KV-cache pages at
different precision and residency levels. Recent pages remain in FP16; older
pages can be demoted through FP8, INT8, INT4, INT2, 1-bit, projection, and CPU
storage according to a configurable policy.

ARGUS is a **capacity-oriented research runtime**, not an inference speedup
engine. The current HuggingFace integration reduces peak VRAM at sufficiently
long contexts, but its decode latency is not yet suitable for low-latency
serving.

Türkçe belge: [README_TR.md](README_TR.md)

## What is proven today

The current end-to-end benchmark uses Qwen2.5-0.5B-Instruct on an RTX 3050 Ti
Laptop GPU (4 GB), FP16, batch 1, 64 generated tokens, three measured repeats,
one warm-up, and 1024-token ARGUS pages.

| context | baseline VRAM | ARGUS VRAM | VRAM change | baseline TPOT | ARGUS TPOT |
|---:|---:|---:|---:|---:|---:|
| 512 | 982.2 MiB | 988.2 MiB | +0.6% | 18.62 ms | 28.69 ms |
| 1,024 | 1007.1 MiB | 1007.1 MiB | 0.0% | 18.55 ms | 30.21 ms |
| 2,048 | 1056.8 MiB | 1056.9 MiB | 0.0% | 19.24 ms | 30.85 ms |
| 4,096 | 1149.4 MiB | 1137.4 MiB | -1.0% | 19.14 ms | 40.21 ms |
| 8,192 | 1340.4 MiB | 1280.5 MiB | **-4.5%** | 17.85 ms | **52.51 ms** |
| 16,384 | 1722.5 MiB | 1590.6 MiB | **-7.7%** | 18.84 ms | **79.78 ms** |

At 16K, TTFT is 1.819 s for the baseline and 3.992 s for ARGUS. Therefore:

- The VRAM saving is real and grows with context length in this experiment.
- Decode latency is the main unresolved problem: TPOT is 4.2x baseline at 16K.
- No tested row shows baseline OOM while ARGUS survives. OOM prevention and a
  larger usable context window remain hypotheses, not published results.
- The measured perplexity delta is -0.0176, but only the near-lossless FP8 tier
  was reached. This does not validate INT4, INT2, 1-bit, or JL downstream
  quality.

Full provenance and min/max timings are in
[`docs/measurements/downstream-2026-08-14.json`](docs/measurements/downstream-2026-08-14.json).
The analysis and discarded-harness explanation are in
[`docs/findings-2026-08-14.md`](docs/findings-2026-08-14.md).

## Why decode is slow

ARGUS compresses **storage**, not attention arithmetic. Its current
HuggingFace adapter reconstructs compressed pages to FP16 and assembles a
contiguous K/V tensor for the model on every decode step. That keeps the model
compatible, but repeated decompression and concatenation scale with the number
of resident pages.

The durable serving design is an attention backend that consumes the paged
cache directly and reconstructs only the blocks needed by the attention
kernel. Page-size tuning can reduce overhead, but cannot remove the contiguous
materialization cost.

## Architecture

```text
HuggingFace model
       |
       v
PagedDynamicQuantizedCache
       |
       +-- policy/configuration in Python
       |
       +-- native C++ page lifecycle and tier cascade
       |
       +-- CUDA/Triton codecs and attention experiments
       |
       +-- pinned CPU spill for archival pages
```

Compression tiers are plugins. The cache manager selects tiers through
capabilities and numeric codec metadata rather than hardcoded tier names.
Detailed ownership and extension points are documented in
[`docs/architecture.md`](docs/architecture.md).

## Runtime status

| runtime | status | what ARGUS manages |
|---|---|---|
| HuggingFace Transformers | Research path, measured | ARGUS owns the model's KV cache |
| Ollama | External adapter, live-tested | Nothing inside Ollama; configuration and timing only |
| vLLM | Unavailable, fails closed | Nothing; no false monkey patch is installed |
| SGLang | Not implemented | Nothing |

The former vLLM integration did not own vLLM KV blocks and could not compress
them. The current adapter intentionally refuses activation. A real integration
must use vLLM's KV connector and/or custom attention-backend interfaces; see
[`docs/vllm-verification.md`](docs/vllm-verification.md).

## Installation

The stabilization changes in this tree have not been published. Build and
install from the checkout:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
python setup.py build_ext --inplace
```

A CUDA-capable PyTorch installation and a working compiler toolchain are
required for the native extension.

## HuggingFace example

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from argus_cache import patch_model_with_argus

model_id = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.float16,
).to("cuda")

model = patch_model_with_argus(
    model,
    page_size=1024,
    max_active_pages=2,
    max_fp8_pages=2,
    sink_tokens=4,
)

inputs = tokenizer("Virtual memory for KV caches", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=64, use_cache=True)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

`page_size=1024` is the best latency/memory compromise measured on the test
machine, not a universal recommendation. Re-benchmark for each model, GPU,
context distribution, and service-level objective.

## Pluggable tiers

```python
from argus_cache import (
    BackendCapabilities,
    NativeCodecSpec,
    register_quantizer,
    unregister_quantizer,
)

unregister_quantizer("one_bit")
register_quantizer(
    "my_codec",
    MyBackend,
    BackendCapabilities(
        name="my_codec",
        effective_bits=2.0,
        native_codec=NativeCodecSpec(kind="unsigned_affine", bits=2),
    ),
)
```

## Reproducing the evidence

```bash
pytest tests/ -q
python benchmarks/bench_native_runtime.py --json native.json
python benchmarks/bench_jl_fidelity.py --json jl.json --tokens 512
python benchmarks/bench_downstream.py \
  --json downstream.json \
  --contexts 512 1024 2048 4096 8192 16384 \
  --new-tokens 64 --repeats 3 --warmups 1 --page-size 1024
```

Benchmark classes are kept separate:

- `reconstruction`: tensor codec error, not model accuracy;
- `runtime`: kernel/cache latency and memory;
- `downstream`: end-to-end TTFT, TPOT, peak VRAM, and model scoring.

## Known limits and next milestone

- Decode currently reconstructs all compressed pages for every generated
  token. This is the primary production blocker.
- Lossy archival tiers need downstream perplexity and retrieval evaluation.
- CPU-spill latency under real memory pressure and multi-user throughput have
  not been measured.
- OOM survival requires a larger model or longer-context experiment where the
  baseline actually exceeds device memory.
- Predictive paging is experimental and disabled by default.

The next meaningful milestone is not another headline compression ratio. It is
a paged attention integration that preserves the demonstrated VRAM curve while
bringing TPOT close enough to baseline for real serving.

## License

ARGUS is licensed under the [Apache 2.0 License](LICENSE).
