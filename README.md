# ARGUS

**A KV-cache memory manager for LLM runtimes. It treats the KV cache as a paged
memory hierarchy — GPU, pinned RAM, pageable RAM, disk — instead of one tensor
pinned to one device, so a context can outlive the VRAM that would normally hold
it.**

ARGUS is not an inference server and not a speedup engine. It is the layer
underneath one, owning where each KV page lives and at what precision. The
deepest integration is llama.cpp, where ARGUS owns the KV allocation, the page
writes and the attention reads.

**Status: v0.5.2. The mechanism works and is proven byte-exact. The policy that
decides where pages should go does not exist yet — that is v0.6.** Every
performance number below is therefore a cost measurement, not a win.

[![packaging](https://github.com/Zwannfrederick/ARGUS-Anchored-Random-Geometric-Unbiased-Storage/actions/workflows/packaging.yml/badge.svg)](https://github.com/Zwannfrederick/ARGUS-Anchored-Random-Geometric-Unbiased-Storage/actions/workflows/packaging.yml)

Türkçe belge: [README_TR.md](README_TR.md)

## Status at a glance

Every "proven" row links to the artifact that proves it. Every "not measured"
row is open, not pending publication.

| Claim | Status | Evidence |
|---|---|---|
| ARGUS owns llama.cpp KV allocation, writes and attention reads | Proven | [host ownership](docs/measurements/v050-llama-host-ownership-2026-09-15.json) |
| Output is byte-identical to stock llama.cpp | Proven | [UI-Mate parity](docs/measurements/v050-ui-mate-reference-parity-2026-09-16.json) |
| KV pages migrate GPU ↔ pinned ↔ pageable ↔ disk, source-preserving | Proven | [CUDA mechanism](docs/measurements/v050-cuda-mechanism-2026-09-16.json) |
| VRAM saving on the HuggingFace path grows with context | Proven, v0.4 | [downstream](docs/measurements/downstream-2026-08-14.json) |
| Precision trades memory for latency, monotonically | Proven, engine in isolation | [fused attention](docs/measurements/v040-fused-attention-benchmark.json) |
| ARGUS is faster than the runtime it sits under | **Not measured** — and not claimed | — |
| ARGUS at a realistic operating point | **Not measured**, and impossible before v0.6 | see [Why there is no speed result](#why-there-is-no-speed-result) |
| Long-context throughput at 262K | **Not measured**, deferred | [v0.6 plan](plans/argus-v0.6.0.md) |
| Quality under INT4, INT2, 1-bit or JL tiers | **Not measured** | only FP8 was reached, see [Quality](#quality) |

Local suite on the development machine: **383 passed, 13 skipped**. The CI badge
above covers packaging and repository hygiene only — hosted runners have no CUDA
device, so a green badge means the package ships the right files, not that ARGUS
works.

## What ARGUS is not

- **Not an inference server.** Sampling, batching, API serving and model loading
  stay in the runtime. ARGUS manages KV memory.
- **Not a speedup.** No measurement in this repository shows ARGUS decoding
  faster than the runtime beneath it. The v0.5 numbers show what it costs.
- **Not a compression benchmark.** Codec ratios are storage facts; they do not
  establish model quality. Only the FP8 tier has downstream evidence.

## Install

```bash
pip install torch                                        # must already be importable
pip install --no-build-isolation argus-cache             # core runtime
pip install --no-build-isolation "argus-cache[gateway]"  # + Anthropic Messages gateway
```

ARGUS ships as a source distribution and compiles a CUDA extension on your
machine, so a CUDA-capable PyTorch, the CUDA toolkit and a C++17 compiler must
already be present. There is no prebuilt wheel: a binary compiled against one
PyTorch ABI and CUDA version would be wrong for most installs.

`--no-build-isolation` is required, not optional. The build reads your installed
`torch` to configure the extension, and pip's isolated build hides it. Building
against a torch pip fetched into a throwaway environment would be worse than
failing — the extension would be compiled for an ABI your runtime does not have.

To work on ARGUS itself:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip && pip install -e .
python setup.py build_ext --inplace
```

## Use it with llama.cpp

This is the deepest integration and the one v0.5 closed on. KV layers are
allocated in an ARGUS-owned `O_DIRECT` store with enforced disk, metadata and
staging budgets; page writes are checksum-verified before the descriptor moves;
attention reads pages block by block with a bounded single-depth prefetch.

```bash
export ARGUS_KV_DIR=/path/on/fast/storage
export ARGUS_KV_MAX_BYTES=$((1 << 30))     # disk budget
export ARGUS_KV_STAGING_BYTES=$((4 << 20)) # staging budget, enables direct disk KV
llama-server -m model.gguf -c 4096 -fa on -ctk f16 -ctv f16
```

Unset, the binary follows the stock path. Budgets are hard: exceeding one fails
explicitly rather than silently growing. Setup, the patch and the full contract
are in [`integrations/llama.cpp/README.md`](integrations/llama.cpp/README.md).

## Use it with HuggingFace

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from argus_cache import AdaptiveCachePolicy, patch_model_with_argus

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct", dtype=torch.float16,
).to("cuda")

model = patch_model_with_argus(
    model,
    page_size=1024,
    max_active_pages=2,
    max_fp8_pages=2,
    sink_tokens=4,
    activation_policy=AdaptiveCachePolicy(
        mode="balanced",
        expected_tokens=16_448,  # prompt + maximum requested output
    ),
    pipeline_profile="balanced",
)
```

`page_size=1024` is the best latency/memory compromise measured on the test
machine, not a universal recommendation. Re-benchmark per model, GPU, context
distribution and service-level objective.

---

# Evidence

## llama.cpp KV ownership (v0.5)

**Ownership is real, not a loaded library.** Allocation and page-ID records,
write/read counters, a trace of attention consuming those pages, and a
controlled backend-shutdown test are all required before ownership is claimed; a
library appearing in the process maps proves nothing. On a real CPU model run:
six attention layers, 32,000 vocabulary logits, maximum logit difference **0**
across eight decode steps after prefill and state restore, 1,769,472 bytes
allocated and back to 0 after teardown.
Artifact: [`v050-llama-host-ownership-2026-09-15.json`](docs/measurements/v050-llama-host-ownership-2026-09-15.json).

**Output is byte-identical to stock.** Driven by the pinned upstream UI-Mate
message builder and parser at revision `1cb9e1e4`, under a hash-identical
payload, stock llama.cpp and ARGUS produced the same reasoning text, the same
coordinate, the same parsed action and the same 119 completion tokens on a
9B multimodal model with CUDA attention.
Artifact: [`v050-ui-mate-reference-parity-2026-09-16.json`](docs/measurements/v050-ui-mate-reference-parity-2026-09-16.json).

**Exactness holds under compression and restore.** Qwen2.5-0.5B with `q8_0` and
`q4_0` KV, 4 MiB staging, 1 MiB resident: attention and logit difference 0 over
19 steps including crop and state restore, peak staging 1.27 MiB.

**Migration preserves the source.** GPU, pinned RAM, pageable RAM and disk are
separately budgeted. A move reserves and verifies the destination before
releasing the source; failed transfers and stale revisions preserve the
published page. Verified for head dimensions 48/64/256, with Q8 and Q4 encoded
bytes carried unchanged across all four placements.
Artifact: [`v050-cuda-mechanism-2026-09-16.json`](docs/measurements/v050-cuda-mechanism-2026-09-16.json).

## Why there is no speed result

v0.5 delivers **mechanism only**. Nothing in the serving path promotes a page —
`argus_disk_move_page` is called only by tests — so every attention read goes to
disk. In the UI-Mate run above:

| | stock | ARGUS |
|---|---:|---:|
| prefill | 89.34 ms/token | 151.31 ms/token |
| decode | 165.58 ms/token | 2915.31 ms/token |
| one request | 260.4 s | 752.0 s |

One request read **14.98 GB** from disk. The tier budgets were not the binding
constraint: 4 MiB of GPU and 4 MiB of pinned were offered and only ~0.5 MiB was
ever used, because nothing decides to use them. Note also that
`ARGUS_KV_RESIDENT_BYTES` budgets the page descriptor table — 65,536 descriptors
for 256 MiB of KV — not a KV working set.

**This is the cost of having no placement policy.** It is not a tuned
configuration, not a starved one, and not an operating point. This repository
holds no ARGUS measurement at a realistic operating point, and none can exist
before automatic placement lands in v0.6. Closing this gap is the entire purpose
of v0.6; it is not a v0.5 claim.

## The HuggingFace research path (v0.4)

Historical context for where the project came from. Qwen2.5-0.5B-Instruct,
RTX 3050 Ti Laptop (4 GB), FP16, batch 1, 64 generated tokens, three measured
repeats, one warm-up, 1024-token pages.

| context | baseline VRAM | ARGUS VRAM | VRAM change | baseline TPOT | ARGUS TPOT |
|---:|---:|---:|---:|---:|---:|
| 512 | 982.2 MiB | 988.2 MiB | +0.6% | 18.62 ms | 28.69 ms |
| 1,024 | 1007.1 MiB | 1007.1 MiB | 0.0% | 18.55 ms | 30.21 ms |
| 2,048 | 1056.8 MiB | 1056.9 MiB | 0.0% | 19.24 ms | 30.85 ms |
| 4,096 | 1149.4 MiB | 1137.4 MiB | -1.0% | 19.14 ms | 40.21 ms |
| 8,192 | 1340.4 MiB | 1280.5 MiB | **-4.5%** | 17.85 ms | **52.51 ms** |
| 16,384 | 1722.5 MiB | 1590.6 MiB | **-7.7%** | 18.84 ms | **79.78 ms** |

At 16K, TTFT is 1.819 s baseline against 3.992 s for ARGUS. What this does and
does not show:

- The VRAM saving is real and grows with context length in this experiment.
- Decode latency is the unresolved problem: TPOT is 4.2x baseline at 16K.
- **No tested row shows baseline OOM while ARGUS survives.** OOM prevention and a
  larger usable context window remain hypotheses, not results.

Artifact: [`downstream-2026-08-14.json`](docs/measurements/downstream-2026-08-14.json).
Analysis: [`docs/findings-2026-08-14.md`](docs/findings-2026-08-14.md).

## Precision versus latency, engine in isolation

A Structure-of-Arrays page table separates *precision* (`ACTIVE_FP16`,
`GGML_Q8_0`, `GGML_Q4_0`) from *placement* (`GPU_DEVICE`, `HOST_PINNED`,
`HOST_PAGEABLE`) in one contiguous descriptor table.
`DirectPagedAttentionEngine` runs an exact tile-by-tile online-softmax
recurrence over it: no context-sized FP16 KV tensor is materialized, and
reconstruction is bounded to one page tile.

Measured on the RTX 3050 Ti Laptop with Qwen-like geometry (24 query heads,
4 KV heads, head_dim 256, page 128):

| context | ACTIVE_FP16 | GGML_Q8_0 | GGML_Q4_0 |
|---:|---|---|---|
| 1,024 | 1.98 ms / 12.15 MiB | 3.15 ms / 10.28 MiB | 4.24 ms / 9.28 MiB |
| 4,096 | 7.51 ms / 24.15 MiB | 11.43 ms / 16.65 MiB | 15.86 ms / 12.65 MiB |
| 8,192 | 14.71 ms / 40.15 MiB | 22.55 ms / 25.15 MiB | 31.61 ms / 17.15 MiB |
| 16,384 | 29.26 ms / 72.15 MiB | 44.97 ms / 44.15 MiB | 62.77 ms / 26.15 MiB |
| 32,768 | 58.48 ms / 136.16 MiB | 90.15 ms / 76.16 MiB | 125.25 ms / 44.16 MiB |

The trade is monotone and steep: at 32K, q4_0 holds the context in 44.16 MiB
against FP16's 136.16 MiB — 3.1x less memory for 2.14x the latency. This is the
engine alone. **It is not wired into the HuggingFace decode path**, so these
numbers appear in no end-to-end result.

Artifact: [`v040-fused-attention-benchmark.json`](docs/measurements/v040-fused-attention-benchmark.json).

## Quality

The measured perplexity delta is **-0.0176**, but only the near-lossless FP8
tier was reached at that passage length. This does **not** validate INT4, INT2,
1-bit or JL quality — the artifact carries that caveat itself. The q4_0
retrieval probe is a single forgiving task at 31k tokens; reasoning, code
generation and long-range coherence under quantized KV are unmeasured.

## A negative result kept on purpose

An A/B sweep at 4K/16K/32K/64K against a local llama-server (Qwen3.6-35B-A3B,
q4_0 KV) appeared to show an ARGUS win. **The audit in the artifact shows ARGUS
was never loaded into the process**: `argus_in_llama_server_maps: false`,
`argus_maps_count: 0`, and peak VRAM byte-identical across both arms at every
context. The decode-rate gap (17.66 vs 11.88 tok/s at 16K) traces to the
gateway's prompt-prefix cache, not to ARGUS.

No ARGUS claim is drawn from this run. It ships in the repository because a
misattributed win is exactly the result that would otherwise go unchallenged.
The sweeps published beside it (`load-mode-comparison`, `pmin-sweep`,
`speculative-sweep-n2-n3-n4`) are llama.cpp runtime tuning and are labelled as
such.

Artifact: [`argus-ab-cache-comparison-2026-09-04.json`](docs/measurements/argus-ab-cache-comparison-2026-09-04.json).

---

# How it works

## Architecture

```text
runtime (llama.cpp / HuggingFace)
       |
       v
ARGUS KV memory layer
       |
       +-- page descriptor table: precision and placement are independent
       |
       +-- budgeted tiers: GPU, pinned RAM, pageable RAM, disk
       |
       +-- native C++ page lifecycle, checksum-verified writes
       |
       +-- CUDA / CPU attention reading pages in place
```

The governing principle is that **a logical KV page, its physical placement and
its physical precision are three separate things**. A page may be GPU/FP16,
pinned/Q8 or disk/Q4; GPU does not imply FP16. Compression tiers are plugins,
selected through capabilities and numeric codec metadata rather than hardcoded
names. Ownership and extension points:
[`docs/architecture.md`](docs/architecture.md).

## Model contracts stay out of the core

`AttentionAdapter` owns native eligibility, query preparation and output layout
per `config.model_type`. Applications add a contract with
`register_attention_adapter()`; an unregistered model never enters native page
attention accidentally and keeps its own attention implementation.

For models mixing full and linear attention (Qwen3.8 Gated DeltaNet, for
example), `argus_cache/models/hybrid_cache.py` states ownership explicitly:
ARGUS owns the growing KV of full-attention layers, and the fixed-size recurrent
and conv state of linear-attention layers is not ARGUS's to touch. Deterministic
state digests are asserted around every cache operation, and an unsupported
layer role fails closed rather than being paged.

## Runtime status

| runtime | status | what ARGUS manages |
|---|---|---|
| llama.cpp | Deepest integration; host and direct-disk KV, CPU and CUDA attention, byte-exact parity | KV allocation, budgets, page writes and attention reads |
| HuggingFace Transformers | Research path, measured | The model's KV cache |
| Ollama | External adapter, live-tested | Nothing inside Ollama; configuration and timing only |
| vLLM | Unavailable, fails closed | Nothing; no false monkey patch is installed |
| SGLang | Not implemented | Nothing |

The former vLLM integration did not own vLLM KV blocks and could not compress
them, so the current adapter refuses activation rather than pretending. A real
integration must use vLLM's KV connector or custom attention-backend interfaces;
see [`docs/vllm-verification.md`](docs/vllm-verification.md).

---

# Roadmap

## v0.6 — the placement policy

**Nothing in this section is implemented.** It is a decided contract, recorded
in [`plans/argus-v0.6.0.md`](plans/argus-v0.6.0.md), that v0.6 implements.

- **The policy is a module above the store.** The store keeps the mechanism;
  `ARGUS_KV_POLICY=off` reproduces v0.5 by deciding nothing, not through a
  second code path.
- **Content and placement become separate revision axes.** Today one store-wide
  counter is bumped by both writes and migrations, so a byte-preserving,
  checksum-verified move cancels an in-flight prefetch, and a write anywhere in
  the store drops a migration.
- **The store owns budgets.** Policy proposes; the store verifies or refuses
  explicitly, and refusals are counted rather than swallowed.
- **The first policy selects placement only.** Precision is defined but off,
  under the rule that precision reduction is one-way per page: promoting a q4
  page to f16 yields dequantized q4, never the original, and no policy may
  report that as recovered quality.

The 262K benchmark baseline, quality tolerance and success metric are
deliberately still open. Measurement follows the policy, because without
promotion there is no placement behaviour to measure.

## Known limits

- Native HuggingFace decode covers only the validated Qwen2 full-attention
  contract. Other models and masked/local-attention cases reconstruct K/V.
- Streaming attention is an ATen operation sequence, not one fused kernel, so it
  still carries per-page launch overhead.
- `DirectPagedAttentionEngine` is measured in isolation and is not wired into
  the HuggingFace decode path.
- The Python `PageStore` and the native store are separate; they are unified
  only if the v0.6 contract requires it.
- CPU-spill latency under real memory pressure and multi-user throughput are
  unmeasured.
- Predictive paging is experimental and disabled by default.
- The [1M-token synthetic disk check](docs/measurements/v050-disk-capacity-smoke-2026-09-15.json)
  uses one layer, one KV head and head dimension 4. It is not 1M-context model
  generation and not an NVMe performance claim.

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

Benchmark classes are kept separate on purpose:

- `reconstruction`: tensor codec error, not model accuracy;
- `runtime`: kernel and cache latency and memory;
- `downstream`: end-to-end TTFT, TPOT, peak VRAM and model scoring.

## License

ARGUS is licensed under the [Apache 2.0 License](LICENSE).
