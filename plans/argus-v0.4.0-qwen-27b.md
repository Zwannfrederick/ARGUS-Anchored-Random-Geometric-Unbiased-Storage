# ARGUS v0.4.0 — Qwen3.8-27B local agent runtime blueprint

**Date:** 2026-08-15
**Target host:** RTX 3050 Ti Laptop 4 GiB (SM 8.6), i5-11300H (4C/8T),
31 GiB RAM, 4.5 GiB swap
**Model:** `Qwen/Qwen3.8-27B`, pinned by immutable revision
**Weights:** validated GGUF quantization (Q4-class primary)
**Runtime:** pinned llama.cpp fork/build; Ollama is an optional wrapper
**Outcome:** run Qwen3.8-27B behind a local API that Claude Code can use for a
sustained tool-using workflow.

## 1. Release contract

v0.4.0 is ready only when all mandatory gates below pass on the target host.

1. The CUDA driver is healthy and the full native suite passes from a clean
   process after rebuilding for SM 8.6.
2. Qwen2 single-token native decode matches reconstructed reference attention
   across the S3 parity matrix.
3. The selected target-host `ACTIVE + q8_0` path consumes compressed pages
   directly on the backend that owns each attention layer and never creates a
   context-sized FP16 K/V tensor during decode. CUDA is required only where S2
   proves that the relevant K/V and compute are GPU-resident.
4. Unsupported model semantics cannot enter the native kernel.
5. A second standard full-attention family passes the adapter contract without
   model-name conditionals in the core.
6. Qwen3.8's 48 Gated DeltaNet layers retain llama.cpp-owned recurrent state;
   ARGUS owns only the 16 Gated Attention layers' growing K/V.
7. The Q4 capacity backend closes the 262,144-token native-context allocation
   and filled-cache gates before the release soak.
8. A local API completes the Claude Code acceptance workflow in S10
   without OOM, cache corruption, unbounded host growth, or teardown leaks.
9. Every published latency, memory, and quality statement is backed by a
   committed JSON measurement containing hardware, dependency versions,
   checkpoint revision, command, warmups, repeats, and min/median/max values.

### Explicit non-goals

- Ollama data-plane ownership is not a release blocker. Its official controls
  expose Flash Attention and global `q8_0`/`q4_0` KV modes, not an ARGUS page
  backend. Ollama remains an optional API/baseline adapter unless a supported
  ownership seam appears.
- INT2, 1-bit, JL, approximate page selection, multi-user throughput, and
  multi-GPU are out of scope. Q4 is a required capacity backend, not a claim
  that 4-bit KV is the preferred latency path.
- ARGUS does not make 27B weights fit in 4 GiB VRAM. GGUF weight quantization and
  CPU/partial-GPU placement are separate from KV-cache capacity.
- v0.4 is text-only and single-slot: no multimodal projector, separate MTP
  artifact, or parallel sequence. Vision, MTP, and `-np > 1` require separate
  capability and memory gates.

## 2. Model identity, context ladder, and feasibility gate

The target is the official `Qwen/Qwen3.8-27B` repository supplied for this
project. Before downloading weights, pin its immutable revision, tokenizer,
chat template, license, config, GGUF conversion/quantization provenance, and
the first llama.cpp revision that passes its reference vectors.

Qwen3.8-27B is a 64-layer hybrid: 48 Gated DeltaNet layers and 16 Gated
Attention layers. Full attention uses 24 query heads, 4 KV heads, and head
dimension 256. ARGUS must never apply the 64-layer growing-KV formula. For the
16 full-attention layers:

```text
exact BF16 KV/token = 2 * 16 * 4 * 256 * 2 bytes = 64 KiB
q8_0 KV/token       = 32 KiB payload * (34/32) = 34 KiB
q4_0 KV/token       = 16 KiB payload * (18/16) = 18 KiB
```

| retained tokens | BF16 KV | `q8_0` KV | `q4_0` KV | role |
|---:|---:|---:|---:|---|
| 32K | 2 GiB | 1.0625 GiB | 0.5625 GiB | smoke and correctness |
| 64K | 4 GiB | 2.125 GiB | 1.125 GiB | practical baseline |
| 128K | 8 GiB | 4.25 GiB | 2.25 GiB | primary long-agent release target |
| 262,144 | 16 GiB | 8.5 GiB | 4.5 GiB | native-context capacity target |
| 512K | 32 GiB | 17 GiB | 9 GiB | experimental, non-blocking |
| 1M | ~61 GiB | ~32.4 GiB | ~17.2 GiB | v0.5.0 extreme-context work |

The table uses llama.cpp block layouts (`q8_0`: 34 bytes/32 values; `q4_0`:
18 bytes/32 values), not payload-only estimates. It excludes alignment and
allocator overhead. In particular, 128K `q8_0` cannot be fully GPU-resident on
a 4 GiB device even before weights and workspace. `balanced = GPU Q8` is at
most a measured 32K/possibly-64K profile, not the 128K solution.

These are geometry estimates, not measured process peaks. Scale/metadata,
Gated DeltaNet recurrent/conv state, allocator/workspace, model weights, OS,
and Claude Code are additional. The 262,144 gate is therefore a whole-system
capacity test, not merely a kernel test. The 512K experiment may require a
deeper KV tier or storage spill; it cannot be promised by the Q8 fast path.

The host has enough nominal RAM for a roughly 14–18 GiB 4-bit 27B artifact plus
runtime state, but usable token speed will be dominated by CPU/PCIe weight
bandwidth. The release report must separate:

- model weight placement and quantization;
- growing full-attention KV;
- fixed linear-attention state;
- CUDA graph/workspace allocations;
- OS and Claude Code headroom.

The DeltaNet matrix state alone is estimated near 144 MiB per sequence at
FP32 (`48 * 48 * 128 * 128 * 4`), before convolution state and checkpoint
copies. The release manifest records measured recurrent/conv bytes and the
checkpoint multiplier. Swap is emergency headroom, never usable capacity;
sustained swap traffic or major-fault growth fails the gate.

### Context acceptance ladder

- **32K:** mandatory numerical parity, tool-call smoke, and teardown.
- **64K:** mandatory stock llama.cpp baseline and ARGUS comparison.
- **128K:** mandatory v0.4 primary workflow/soak gate.
- **262,144:** mandatory native-context capacity gate. If measured whole-system
  memory makes it impossible, v0.4 is blocked until the scope is explicitly
  renegotiated; the checkpoint is never silently skipped or relabeled.
- **512K:** experimental measurement, not a release blocker.
- **1M:** explicitly deferred to v0.5.0; YaRN/extension quality and storage
  strategy are separate research gates.

## 3. Dependency graph

```text
S0 freeze current tree
 └─ S1 GPU safety gate
     └─ S2 pin artifacts + stock baseline + ownership/topology feasibility
          ├─ S3 parity harness
          └─ S4 hybrid Qwen cache ownership contract
               └─ S5 backend-aware compressed pool + page-table ABI
                    ├─ S6 q8_0 direct-attention backend
                    └─ S6C q4_0 capacity backend
                         └─ S7 isolated backend benchmarks
                              ├─ S8 adaptive-policy integration
                              └─ S9 second full-attention adapter
                                   └─ S10 serving API + Claude Code
                                        └─ S11 27B soak and release
```

S3 test-data work and S4 design can be prepared independently after S2. No
CUDA implementation is accepted while the GPU gate is red, and no CUDA-only
design is accepted unless S2 proves it is on the target-host critical path.
Both S6 and S6C must close before S7; S8 and S9 may then proceed independently.

## 4. Construction steps

### S0 — Freeze the post-v0.3 correctness baseline

**Context:** The working tree currently contains adaptive activation, native
streaming attention, model-aware Hugging Face attention, tests, benchmarks,
and documentation that are not yet committed. Mixing those changes with a new
kernel would make regression attribution impossible.

**Work:**

- Review the complete diff and split it into reviewable commits: activation,
  streaming prototype, HF bridge, benchmarks/docs.
- Stop calling the current signed-linear 8-bit integer codec hardware FP8 in
  measurements. Keep `fp8` only as a deprecated compatibility alias or migrate
  it to an explicit `q8_linear`/llama.cpp-compatible Q8 descriptor.
- Record the exact dependency lock, compiler, CUDA toolkit, driver, GPU, and
  extension ABI.
- Add a release-gate command that stores stdout, exit status, environment, and
  measurement JSON under a dated artifact directory.
- Do not change public performance claims from preliminary `/tmp` results.

**Verification:** `git diff --check`; CPU-safe suite; clean-process extension
import; documentation links; no generated binaries staged.

**Exit:** A bisectable baseline exists before fused-kernel work starts.

**Rollback:** Revert only the relevant baseline commit; never reset the user's
working tree.

### S1 — Restore CUDA and close the full safety gate

**Context:** The extension builds for SM 8.6, but the current session reports
`cuda_available=False` and `nvidia-smi` cannot communicate with the driver.
The last full run under this state produced environment failures from CUDA-only
operations, not a valid regression signal.

**Work:**

- Restore the driver by reboot/service/package repair outside this plan's code
  scope; verify driver/toolkit/PyTorch compatibility.
- Rebuild from a clean extension build directory with explicit SM 8.6.
- Run the full suite twice in fresh processes.
- Run focused generation, resurrection, long cascade, concurrent access,
  repeated construction/destruction, host spill, and teardown tests.
- Capture NVML memory before/after 100 cache lifecycles and after process exit.

**Verification commands:**

```bash
nvidia-smi
.venv/bin/python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name())"
TORCH_CUDA_ARCH_LIST=8.6 .venv/bin/python setup.py build_ext --inplace
.venv/bin/python -m pytest -q
.venv/bin/python -m pytest -q
```

**Exit:** Two green full-suite runs, zero unexplained memory drift, no native
crash/hang, and a committed gate artifact.

**Rollback:** No kernel work is merged; diagnose driver or baseline regression.

### S2 — Pin artifacts and close ownership/topology feasibility

**Context:** The current ARGUS C++/CUDA extension is a PyTorch backend. A fused
kernel implemented only there does not accelerate llama.cpp. Runtime ownership
must be decided before designing the production page ABI.

**Work:**

- Pin the Qwen3.8 Hugging Face revision and archive `config.json`, tokenizer,
  chat template, generation config, and model-card claims.
- Produce or select GGUF Q4_K_M plus one higher-fidelity comparison artifact;
  record source hashes, converter revision, quantizer revision, sizes, and
  reference prompt logits/text.
- Pin a llama.cpp revision that explicitly recognizes the model architecture;
  do not map Qwen3.8 to QWEN35 merely because its high-level layout resembles it.
- Run stock `llama-cli`, `llama-bench`, and `llama-server` at 32K and 64K with
  explicit threads, GPU layers, Flash Attention, K/V type, mmap, and context.
- Use `q8_0` and `q4_0` as the production KV codec IDs and byte layouts. Keep
  the existing ARGUS signed-linear codec, if retained, under the distinct
  `q8_linear` ID; it cannot share memory math or parity claims with `q8_0`.
- Trace llama.cpp's hybrid cache: identify full-attention K/V buffers,
  recurrent/conv state, graph builders, cache position semantics, defrag,
  sequence copy/remove, context shift, and CUDA attention dispatch.
- Resolve the production seam at ggml graph/scheduler level: pin the llama.cpp
  fork and commit, identify the hybrid-memory factory, select a ggml custom op
  or backend-dispatch integration, define stream/event ownership, cover defrag
  and sequence lifecycle, and retain `--argus off` stock fallback in the same
  binary.
- Build a minimal proof that replaces or intercepts one full-attention layer's
  K/V allocation/read. Run the cache lifecycle without a model step and prove
  the DeltaNet buffer digest is unchanged. Memory telemetry must show ownership
  changed; a wrapper-only proof is rejected.
- Measure the real target-host placement matrix before choosing a kernel:
  `-ngl`, which of the 16 attention layers execute on CPU/GPU, weight residency,
  GPU KV budget, host bandwidth, PCIe traffic, workspace, RSS, and swap faults.
  Include allocation-only dry runs at 32K/64K/128K/262,144 and record disk
  preflight for model, comparison artifact, builds, and measurements.
- From those measurements, explicitly select the first production path:
  ggml CPU Q4/Q8 direct attention, per-backend implementations, or exact
  CPU/GPU online-softmax reduction that combines partial max/sum/V. A CUDA
  kernel is a primary dependency only for the layers proven GPU-resident.
- Choose one production integration architecture: a maintained llama.cpp fork,
  an upstreamable backend extension, or a narrow shared C ABI. Document why the
  rejected alternatives cannot own K/V safely.

**Exit:** Stock reference works; ownership is documented with file/symbol and
ggml scheduler anchors; the one-layer proof changes real allocated bytes
without changing output outside tolerance; and a signed topology decision
states where q8_0/q4_0 attention executes at every mandatory context.

**Rollback:** Keep stock llama.cpp as executable baseline; no ARGUS performance
claim is made if ownership or a physically viable 128K/262,144 topology cannot
be proven.

### S3 — Freeze a property-driven attention parity matrix

**Context:** “Runs” is insufficient. Three independent oracles are required:
FP32 mathematical attention; decoded reference over the exact same `q8_0` or
`q4_0` bytes; and stock llama.cpp/ggml layer output plus full-model logit/token
parity. Codec loss and attention-backend error are reported separately.

**Owned files:** `tests/test_attention_parity_matrix.py`, a new deterministic
fixture helper, and measurement schema only. Do not alter the kernel here.

**Matrix:**

- batch: 1, 2;
- Q heads: 1, 4, 8, 16, 24;
- KV heads / GQA ratios: 1:1, 2:1, 4:1, 6:1;
- head dimensions: 64, 128, 256;
- context: page boundary -1, exact boundary, +1, 1K, 4K, 16K;
- page sizes: 128, 256, 512, 1024;
- query dtype: FP16 and BF16 where the GPU supports it;
- storage: ACTIVE-only, Q8-only, ACTIVE+Q8, Q4 capacity reference, and
  GPU-resident/host-resident variants;
- layout/mask rejection: sliding, local, padding, q_len > 1, mismatched value
  dimension, invalid head ratios.

Qwen3.8 fixtures use real ggml tensor strides/layout, 24Q/4KV/256 geometry,
MRoPE-applied K, and the gated-attention output boundary. A PyTorch-shaped
contiguous fixture alone cannot release the llama.cpp path.

**Numerics:** Compare FP32-accumulated output using absolute/relative error,
cosine similarity, NaN/Inf checks, and deterministic seeds. Set tolerances from
observed error distributions and dtype theory; never widen a threshold only to
make a failure green. Keep separate limits for ACTIVE exact and quantized tiers.

**Exit:** Every supported cell passes; every unsupported cell raises or uses an
explicit fallback; failures print the full seed/shape/tier reproduction tuple.

**Rollback:** Tests remain useful even if the streaming prototype is retained.

### S4 — Implement the hybrid Qwen cache ownership contract

**Context:** Qwen3.8 has 48 Gated DeltaNet layers and 16 Gated Attention layers.
The DeltaNet state remains wholly runtime-owned in v0.4. ARGUS receives only
growing K/V from layer roles derived from the pinned model metadata.

**Work:**

- Represent cache capabilities explicitly: `recurrent_state`, `conv_state`,
  `full_attention_kv`, and unsupported state kinds.
- Derive layer roles from GGUF/model metadata and validate the expected 48/16
  split at model load. Fail closed on missing or contradictory metadata.
- Route full-attention allocation/update/read through the ownership seam from
  S2. Leave recurrent/conv update, snapshot, and rollback semantics unchanged.
- Preserve gated-attention output gates, partial/multimodal RoPE, logical
  positions, sequence operations, cache reuse, and context shift behavior.
- Add a state digest/debug mode that proves ARGUS operations never mutate the
  DeltaNet buffers.
- Pin explicit fail-closed behavior for append-only turns, identical-prefix
  reuse, middle-of-history mutation, `seq_cp`, partial `seq_rm`, cancellation
  rollback, in-process and file state restore, and reaching the context limit.
  Unsupported prefix mutation or context shift must return a clear error; it
  must not silently re-prefill or continue with stale recurrent state. The
  512K experiment cannot depend on unvalidated context shifting.

**Verification:** Cache-only operations preserve the recurrent-state digest.
ACTIVE/exact end-to-end runs require state/logit parity. Quantized Q8/Q4 runs
use declared output/logit/quality tolerances rather than impossible byte
equality after changed attention outputs. Also cover sequence remove/copy,
cancellation, state restore, prefix mutation, context limit, and repeated
sessions at a 32K configuration with tractable token samples.

**Exit:** All 16 and only those 16 growing K/V layers are ARGUS-owned; all 48
DeltaNet states remain runtime-owned; allocation telemetry agrees with the
64 KiB/token geometry before quantization.

**Rollback:** A build flag returns all cache ownership to stock llama.cpp.

### S5 — Separate precision from placement and freeze the backend ABI

**Context:** Current compression writes 8-bit pages into separate pinned-host
allocations. A CUDA kernel that walks those pointers would read the entire KV
over PCIe on every token and cannot serve 128K/262,144 efficiently. Precision
(`q8_0`, `q4_0`) and placement (`GPU`, pinned host, pageable host) are
independent axes, and CPU and CUDA backends share logical descriptors.

**Owned files:** llama.cpp integration/fork page-table and pool units plus an
optional mirrored prototype in ARGUS. No fused math yet.

**Work:**

- Add preallocated or growable contiguous backend-local compressed-page pools;
  create a GPU pool only for residency proven useful in S2.
- Define a versioned structure-of-arrays page table containing pool/backend
  offset, token count, K/V scale, codec kind, dtype, logical position, and
  residency generation. Avoid raw `std::vector<Page*>` traversal in hot loops.
- Keep ACTIVE FP16/BF16 and compressed Q8 pools distinct but addressable
  by one descriptor table.
- Add generation counters so async eviction cannot leave stale descriptors.
- Make host spill an explicit placement transition with stream/event ownership.
- Account for allocator capacity, live bytes, fragmentation, and transient
  bytes separately in telemetry.

**Invariants:** Unique page IDs; stable logical order; no alias after reuse;
descriptor publication occurs after data copy; eviction waits for readers;
reset invalidates all descriptors; no Python callback on the decode hot path.

**Exit:** Random allocate/demote/resurrect/free traces preserve bytes and order,
including concurrent attention/lifecycle access under compute-sanitizer.

**Rollback:** Feature flag selects the old pinned-page layout.

### S6 — Implement direct ACTIVE + q8_0 single-token attention

**Context:** The first production backend is exact attention over the stored
representation, not an approximate page selector. It supports q_len=1 only.
Its CPU, CUDA, or split implementation is the topology selected in S2.

**Owned files:** ggml CPU/custom-op and/or `.cu/.cuh` units in the selected
llama.cpp ownership path, narrow dispatch, and backend tests. The PyTorch
extension may host a reference launcher but is not production integration.
Avoid policy or model-name logic in backend math.

**Data path:**

```text
backend page table + ACTIVE/q8_0 pools + Q
  -> load/dequant tile
  -> QK dot products
  -> FP32 online max/sum recurrence
  -> weighted V accumulation
  -> cast output
```

**Backend v1 contract:**

- CUDA portions are SM 8.6-compatible C++; CPU portions use supported ggml
  primitives/SIMD. Triton may remain only a comparison prototype.
- Batch 1 first, then batch >1 only after parity and profiling.
- GQA mapping without materialized `repeat_interleave`.
- FP16/BF16 query and ACTIVE data; production compressed pages use llama.cpp
  `q8_0` block layout. `q8_linear`, if retained, has separate dispatch/tests.
- SM 8.6 has INT8 Tensor Core instructions but no hardware FP8 Tensor Core.
  The first implementation may dequantize Q8 tiles for floating-point QK/V or
  use a validated INT8 dot-product formulation; it must never label either as
  native FP8 execution.
- No context-sized FP16 K/V or score tensor.
- Stable online softmax across arbitrary page boundaries.
- One logical decode operation; launch count may be a small fixed number only
  when a measured cooperative-kernel limitation requires it.
- A split CPU/GPU path combines partial online-softmax statistics with the
  exact global-max rescaling equations; it never averages backend outputs.
- No QoS `.item()` synchronization on the normal serving path. Page-mass
  telemetry is sampled or accumulated asynchronously.

**Exit:** S3's three oracles pass; relevant sanitizers are clean; allocation
trace proves no full reconstruction; unsupported contracts fail before
dispatch; the chosen backend beats reconstructed attention at its residency.

**Rollback:** `--argus off` returns to stock ggml attention;
`streaming_attention=fused` remains opt-in until S7 and S8 pass.

### S6C — Implement the required q4_0 capacity backend

**Context:** `q8_0` alone cannot fit the mandatory long contexts on this host.
The 262,144-token gate cannot depend on a codec/backend absent from the graph.

**Work:**

- Freeze a versioned `q4_0` codec descriptor byte-compatible with the pinned
  llama.cpp block layout, including scales, alignment, and tail handling.
- Consume q4_0 directly in the CPU or split backend selected by S2 without a
  context-sized dequantized tensor or per-token PCIe scan.
- Compare against the same q4_0 bytes decoded by the reference oracle; then run
  full-model logit, deterministic task, and long-context retrieval quality
  gates. Quality thresholds are fixed before the release run.
- Test allocator/transient peaks at allocation-only and filled-cache
  32K/64K/128K/262,144 checkpoints.

**Exit:** q4_0 layout parity, numerical/quality gates, lifecycle tests, and the
262,144 filled-cache capacity test pass without OOM, sustained swap, or full
reconstruction.

**Rollback:** Disable the capacity profile and block v0.4 release; never fall
back silently to a lower-quality codec or relabel a smaller context.

### S7 — Build isolated backend benchmarks and a profiler gate

**Owned files:** `benchmarks/bench_fused_attention.py`, JSON schema, profiler
scripts, and committed target-host measurements.

**Measure independently:**

- backend duration (median, p95, min/max; CUDA events only for CUDA work);
- achieved DRAM and PCIe bandwidth;
- kernel launch count per layer/token;
- GPU pool bytes, host-pinned bytes, and peak transient bytes;
- occupancy/register pressure/shared memory;
- contiguous SDPA ratio and ATen-streaming ratio;
- page-table update overhead outside the kernel.

Use warmups and synchronization at measurement boundaries, not inside the hot
loop. Profile 1K/4K/8K/16K, then model and measure 32K/64K/128K/262,144 with
identical stored K/V. Report ACTIVE-only, q8_0-only, q4_0-only, mixed, CPU,
GPU, and split paths separately. Compare predicted bytes/token to hardware
counters so a hidden PCIe scan cannot pass.

**Performance gate:** In the existing Qwen2.5-0.5B laboratory case at 16K,
fused must be materially faster than the current ~80 ms downstream capacity
path and must not regress peak VRAM. That number is not transferred to 27B.
A separate Qwen3.8 llama.cpp target is set from its stock 32K/64K baseline
after S2; the goal is to minimize ARGUS overhead at the same residency point,
not to predeclare an absolute TPOT.

Before implementation tuning, derive a relative usability SLO from the stock
baseline: maximum TPOT regression at equal placement/codec, tool-turn timeout,
cancellation latency, and workflow success rate. The numerical values belong
in the S2 artifact and cannot be loosened after seeing ARGUS results.

**Rollback:** If the profiler gate fails, fused/direct dispatch remains disabled
and the failure artifact determines whether to revise S5/S6 or use stock ggml.

### S8 — Connect adaptive policy to measured fast paths

**Work:**

- `latency`: exact native/standard cache, zero manager allocation until needed.
- `balanced`: ACTIVE + q8_0 on the backend/placement proven by S2; GPU residency
  is used only within measured VRAM headroom.
- `capacity`: bounded q8_0 followed by the validated q4_0 capacity backend. Lower
  research tiers remain explicit and cannot silently activate.
- Include model weights, non-KV state, workspace reserve, and allocator slack in
  the pressure calculation; current KV-only geometry is insufficient for 27B.
- Calibrate hysteresis from real measurements and expose the decision inputs in
  telemetry.

**Exit:** Short-context overhead is within measurement noise; transitions do not
oscillate; no mid-request full-cache migration pause exceeds the declared budget.

**Rollback:** Disable the failing policy profile and route it to stock cache;
other independently green profiles remain available.

### S9 — Validate the adapter abstraction with a Llama-family model

**Work:** Add one Llama-family full-attention adapter using only the public
adapter registry. Its module-specific query/output/mask semantics live in the
adapter. Core manager, page pool, launcher, and CUDA source contain no `qwen`,
`llama`, or model-class branching.

**Exit:** Qwen2 and Llama parity matrices pass; deleting either adapter leaves
the other unchanged; sliding/local variants remain rejected until separately
validated.

**Rollback:** Remove the Llama adapter registration without touching core or
the Qwen path; no support claim is published for the removed family.

### S10 — Serve through llama.cpp and connect Claude Code

**Runtime:** Use the pinned patched llama.cpp integration selected in S2. Keep
stock llama-server as the byte/performance reference. Stock Ollama is only a
UX/API and baseline option; it does not wrap a patched cache automatically.
Only the patched llama.cpp binary, or a separately built and verified custom
Ollama, can provide ARGUS ownership evidence.

**API contract:** llama-server exposes its pinned OpenAI-compatible endpoint.
Claude Code connects through a pinned local Anthropic Messages-compatible
gateway (or a native equivalent) configured with `ANTHROPIC_BASE_URL` and local
authentication. The bridge must translate SSE streaming, `tool_use` and
`tool_result`, system prompts, multi-turn history, usage fields, cancellation,
timeouts, max-token errors, and finish reasons. Direct Claude Code-to-OpenAI
endpoint compatibility is never assumed, and the trace proves no cloud fallback.

**Exit:** Claude Code can inspect a repository, call at least three tools, edit
a file, run a test, recover from one tool error, and continue the same session.
No hidden cloud fallback is allowed in the acceptance trace.

**Rollback:** Use the pinned stock llama-server and disable the ARGUS build.
Gateway failure blocks Claude Code support but does not contaminate cache
correctness measurements.

### S11 — Qwen3.8-27B target-host soak and v0.4.0 release

**Pinned matrix:** official Qwen3.8 revision; Q4_K_M GGUF plus one comparison
artifact; 32K, 64K, 128K, and 262,144 checkpoints; 512K experimental attempt;
stock llama.cpp exact/q8 baseline; ARGUS latency, balanced, and capacity profiles.

Each context record distinguishes `allocated_context`, `prefilled_tokens`,
`decoded_tokens`, and `retained_multi_turn_tokens`. Merely launching with
`-c 128K` is not the 128K workflow gate: the primary acceptance run retains
approximately 128K real history. The 262,144 capacity gate similarly requires
a filled cache; a shorter deterministic correctness companion may be reported
separately but cannot replace it.

**Soak:** Minimum 60-minute Claude Code workflow and 10 sequential sessions.
Record load time, TTFT, TPOT, prompt/decode tokens, RSS, swap, VRAM, pinned host
bytes, cache compression, tool-call success, cancellation, teardown, and model
quality failures. Also record temperature, power limit, GPU/CPU clocks, thermal
throttling, CPU governor, fan state where available, major page faults, and
swap-in/out. Sustained swap or thermal collapse invalidates the run.

**Mandatory outcome:** The selected workflow completes without OOM/corruption
and memory returns to the declared idle envelope after unload. The release notes
state the measured token rate honestly; no minimum interactivity claim is made
before S2 data exists; the precommitted relative SLO from S7 must pass.

**Release:** Update version in both packaging sources, changelog, architecture,
supported-model matrix, reproducible setup, driver recovery notes, measurement
provenance, and known limitations. Tag only from a clean tree after a fresh
install and smoke test.

**Rollback:** Do not tag. Preserve the last green artifact/commit and publish
the failing context/SLO as a blocked release result rather than narrowing the
contract silently.

## 5. Adversarial failure catalog

- **False integration:** API responds but runtime still owns an untouched cache.
- **Host-Q8 “fast path”:** fused math reads every token over PCIe each step.
- **VRAM arithmetic failure:** 128K q8_0 is called GPU-resident although its
  4.25 GiB KV allocation already exceeds the device.
- **Codec conflation:** simulated signed INT8 is labeled hardware FP8.
- **Codec-layout conflation:** ARGUS `q8_linear` bytes are treated as llama.cpp
  `q8_0`, corrupting both memory accounting and numerical comparisons.
- **Reference mismatch:** native output compared against original FP16 rather
  than the reconstructed representation, mixing codec and kernel error.
- **Hybrid corruption:** linear-attention recurrent state is treated as K/V.
- **Mask elision:** custom attention registration causes the framework to drop
  padding/local masks.
- **Descriptor race:** page is evicted/reused while a kernel still reads it.
- **Benchmark synchronization error:** `.item()`, implicit copies, or missing
  CUDA synchronization makes timings incomparable.
- **Weight/KV confusion:** KV savings are presented as proof a 27B weight file
  fits in 4 GiB VRAM.
- **Agent demo theater:** a one-prompt completion is called a Claude Code
  workflow without tool calls, cancellation, or long-lived cache state.
- **Protocol wishful thinking:** Claude Code is pointed directly at an
  OpenAI-format llama-server without a tested Anthropic Messages bridge.
- **Reserved-context theater:** `-c 128K` succeeds but the run never fills or
  retains anything close to 128K tokens.

## 6. Plan mutation protocol

- A step may be split when its review diff exceeds one coherent ownership unit.
- New steps must declare dependencies, files, verification, exit, and rollback.
- A failed performance hypothesis is recorded with measurements; it is not
  silently removed.
- Model/runtime versions are pinned by revision. API drift creates a new
  compatibility step rather than an unreviewed workaround.
- Any correctness failure sends the branch back to the last green dependency;
  performance work never weakens a parity or fail-closed gate.

## 7. Primary references

- Qwen3.8-27B official target model:
  https://huggingface.co/Qwen/Qwen3.8-27B
- Qwen3.8 llama.cpp model implementation:
  https://github.com/ggml-org/llama.cpp/blob/master/src/models/qwen35.cpp
- llama.cpp public memory/sequence API:
  https://github.com/ggml-org/llama.cpp/blob/master/include/llama.h
- Qwen official llama.cpp/GGUF guide:
  https://github.com/QwenLM/Qwen3/blob/main/docs/source/run_locally/llama.cpp.md
- Qwen3.5-27B official model card and architecture:
  https://huggingface.co/Qwen/Qwen3.5-27B
- Qwen3.5-27B official configuration:
  https://huggingface.co/Qwen/Qwen3.5-27B/blob/main/config.json
- Qwen3.6-27B official release, including agentic coding positioning:
  https://qwen.ai/blog?id=qwen3.6-27b
- Hugging Face custom attention interface:
  https://huggingface.co/docs/transformers/main/en/attention_interface
- Hugging Face cache strategies:
  https://huggingface.co/docs/transformers/main/en/kv_cache
- FlashAttention exact IO-aware attention paper:
  https://arxiv.org/abs/2205.14135
- NVIDIA Ampere supported Tensor Core data types (SM 8.x; no FP8 MMA):
  https://docs.nvidia.com/cuda/ampere-tuning-guide/
- vLLM paged attention design:
  https://docs.vllm.ai/en/latest/design/paged_attention/
- vLLM custom attention backend registry:
  https://docs.vllm.ai/en/latest/api/vllm/v1/attention/backends/registry/
- Ollama official KV/Flash Attention configuration:
  https://github.com/ollama/ollama/blob/main/docs/faq.mdx
- Claude Code LLM gateway configuration (`ANTHROPIC_BASE_URL`):
  https://docs.anthropic.com/en/docs/claude-code/llm-gateway
- Claude Code CLI reference:
  https://docs.anthropic.com/en/docs/claude-code/cli-usage
