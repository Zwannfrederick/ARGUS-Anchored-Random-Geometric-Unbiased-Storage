# Experimental llama.cpp host KV ownership seam

Pinned upstream revision: `54315813269112dd0baed7112ec87ad93a8218ca`.
Patches (in order): `host-kv.patch`, then `cuda-kv.patch`.
Implementation: `argus_cache/csrc/ggml_*`.

The patch routes `llama_kv_cache` tensor allocation to ARGUS-owned, file-backed
host buffers when `ARGUS_KV_DIR` is set. The existing GGML tensor layout and
attention graph read/write those buffers. It requires `--no-kv-offload` and an
explicit `ARGUS_KV_MAX_BYTES` allocation limit. With the environment variable
unset, the same binary follows the stock allocation path.

The mapped host mode does not connect Python `PageStore`, implement heterogeneous
precision or enforce a resident RAM limit. File mappings use the OS page cache; the
byte limit bounds mapped KV allocation, not RSS. Without `ARGUS_KV_RESIDENT_BYTES`,
attention remains GGML's. The direct disk mode below is the budgeted path.
Other memory implementations may not use `llama_kv_cache`; no support is claimed
without observing allocation and real inference using the buffer.

## Experimental block attention

Setting `ARGUS_KV_RESIDENT_BYTES` selects ARGUS CPU block attention over the
mapped KV tensors. It requires `-nkvo -fa on` and the supported single-stream
contract; unsupported attention metadata refuses activation. K/V stay in GGML's
layout and precision. Full attention merges block softmax results without
reconstructing the full cache. `ARGUS_KV_BLOCK_CELLS` defaults to 256 and must be
a positive integer. Worker coordination is owned by each invocation and released
when its workers finish, including reuse of the same graph and idle workers.

The resident setting is a **target, not a hard memory limit**. `msync`, `madvise`
and `posix_fadvise` request writeback/eviction; the OS controls page-cache residency.
Worker scratch, prefetch, partial pages and state-copy operations can exceed the
target. `ARGUS_KV_STATS_PATH` samples mapping residency at most once per second:
its peak is a sampled peak, `read_bytes` counts worker reads, and
`paged_out_bytes` counts requested ranges, not confirmed physical disk I/O.
This does not close the v0.5 resident/staging budget gate.

The standalone kernel check covers FP32/FP16/BF16/Q8/Q4 against stored-value
references, with repeated graphs at 1/4/16 workers. Native model lifecycle and
server parity are separate checks. The stories15M lifecycle cases use FP32 and
FP16: its head dimension of 48 is incompatible with GGML's 32-element Q8/Q4
blocks. Quantized real-model parity uses a separate model (see below).

```sh
ARGUS_LLAMA_CPP_DIR="$PWD/scratch/llama.cpp-v050" \
ARGUS_TEST_GGUF="$PWD/scratch/models/stories15M.gguf" \
ARGUS_TEST_QUANT_GGUF="$PWD/scratch/models/qwen2.5-0.5b-instruct-q4_k_m.gguf" \
  .venv/bin/python -m pytest -q tests/test_native_llama_paged.py
```

## Direct disk KV (v0.5)

Setting `ARGUS_KV_STAGING_BYTES` together with `ARGUS_KV_DIR` places every KV
layer on the `ARGUS_DISK` buffer. It requires `-nkvo -fa on`. By default K/V bytes
keep GGML's codec (F32/F16/Q8_0/Q4_0) and live only in an unlinked `O_DIRECT` file;
tensor addresses are inaccessible handles, never resident KV.

- Appends go through an ARGUS `set_rows` op. Each 4 KiB page is written to a
  second slot, read back and checksum-verified before its descriptor moves, so
  a failed or short write keeps the previous page. Disk reservation is twice the
  KV size.
- Attention reads visible blocks from disk. A single-depth prefetch worker loads
  the next visible block; a full queue or a stale generation after crop/reset is
  refused, and destruction joins pending work.
- Budgets are enforced, not advisory: `ARGUS_KV_MAX_BYTES` bounds disk bytes,
  `ARGUS_KV_RESIDENT_BYTES` bounds page-descriptor and resident-handle metadata, and
  `ARGUS_KV_STAGING_BYTES` bounds all ARGUS attention scratch, bounce buffers and
  the prefetch stack process-wide. Exceeding any of them fails with an explicit
  error instead of growing. Staging must hold at least one encoded KV row plus a
  page; Qwen2.5-0.5B with `-ub 256` needs more than 1 MiB.
- GGML calls custom ops on every graph worker regardless of the `n_tasks` hint;
  disk attention and disk `set_rows` run only on worker 0.

Verified on 2026-09-15 with Qwen2.5-0.5B-Instruct Q4_K_M on CPU, 4 MiB staging and
1 MiB resident budget: for `q8_0` and `q4_0` KV, 19 compared steps including crop
and state restore had zero attention and logit error and zero greedy mismatches.
Peak staging was 1.27 MiB and peak metadata 16–28 KiB. Disk short-write,
corrupt-page, queue and budget-refusal cases pass in
`tests/cpp/test_ggml_disk_buffer.cpp`.

This proves budgeted placement and exact parity, not speed: every attention block
still comes from disk I/O. No long-context throughput or NVMe measurement is claimed.
`benchmarks/bench_llama_paged_context.py --modes argus-direct` runs the ladder.

The local CPU checkout/build now lives in `scratch/llama.cpp-v050` so it survives
cleanup of `/tmp`. It was restored from the pinned source archive; its local Git
repository has no upstream history and reports an unknown build commit. Use the
pinned archive revision above for provenance, not the binary's commit label.

## Build

Use a separate checkout at the pinned revision. Verify `git rev-parse HEAD`
before applying the patch; don't apply it blindly to another revision.

```sh
git -C /path/to/llama.cpp apply --check /path/to/ARGUS/integrations/llama.cpp/host-kv.patch
git -C /path/to/llama.cpp apply /path/to/ARGUS/integrations/llama.cpp/host-kv.patch
git -C /path/to/llama.cpp apply --check /path/to/ARGUS/integrations/llama.cpp/cuda-kv.patch
git -C /path/to/llama.cpp apply /path/to/ARGUS/integrations/llama.cpp/cuda-kv.patch
cmake -S /path/to/llama.cpp -B /path/to/llama.cpp/build \
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF \
  -DARGUS_CORE_DIR=/path/to/ARGUS/argus_cache/csrc
cmake --build /path/to/llama.cpp/build --target llama-cli llama-server -j2
```

Quote paths containing spaces. CMake is a build tool, not an added Python
runtime dependency.

## CUDA mechanism (v0.5 M2)

**v0.5 provides mechanism; v0.6 chooses policy.** With direct disk mode enabled,
set positive `ARGUS_KV_GPU_BYTES` and `ARGUS_KV_PINNED_BYTES` to select CUDA
attention. Both are hard ARGUS allocation limits, including their staging
buffers. `ARGUS_KV_RAM_BYTES` enables pageable-RAM placement; an unset RAM
budget disables that tier for policy. These limits exclude llama.cpp weights, graph outputs and workspace;
they are not limits on total VRAM or process RSS.

`argus_disk_move_page(tensor, offset, tier, expected_revision)` explicitly moves
one aligned 4 KiB physical backing page. GPU, pinned RAM and pageable RAM retain
the original codec bytes. The verified disk copy remains authoritative backing;
every write is verified on disk before invalidating a resident copy. Promotion
reserves its destination, copies and verifies before releasing the source.
Demotion verifies the retained disk copy before releasing the resident source.
Failed transfers and stale revisions preserve the published page. Atomicity is
per physical page, not a multi-page logical KV transaction.
On the v0.6 development branch, content and placement revisions are separate.
Prefetch and attention check store content revisions; migration takes the
allocation/page/content token returned by `argus_disk_page_revision(tensor, offset)`.
Byte-preserving migration does not invalidate attention or prefetch. A write to
another page does not invalidate the migration token; a write to its page or a
reset does. The released v0.5 mechanism used the shared revision instead.

The first CUDA attention kernel supports FP16 K/V, F32 queries, F16/F32 masks,
head dimensions up to 256, GQA and one sequence on CUDA device 0. Q8/Q4 attention
continues to use the CPU path; explicitly requesting CUDA for an unsupported
codec fails. FP16 is this kernel's current limit, not a placement/codec coupling.
The scheduler dispatches a real CUDA custom op; it never copies the entire
disk-backed tensor to the GPU. CUDA graph capture is disabled for external I/O.

Two 32-cell pinned/device tiles overlap transfer with the previous tile's
compute. GPU-resident source pages use device-to-device copies. Online FP32
softmax state spans all tiles, including partially or entirely masked rows.
`ARGUS_KV_NO_OVERLAP=1` selects the serial reference. All tiles, softmax state and
disk bounce buffers also count against the process-wide staging budget. Do not
sum staging and tier counters as disjoint physical allocations.

The default (`ARGUS_KV_POLICY` unset or `off`) performs no automatic placement.
The native mechanism lifecycle test explicitly promotes two pages. The scalar kernel and
per-invocation staging allocations establish correctness, not throughput targets.
No 262K performance improvement is claimed.

On 2026-09-16, RTX 3050 Ti checks passed for D=48/64/256, all four placements
(including unchanged Q8/Q4 encoded bytes, independently of kernel support),
overlap/serial equality, short writes, corrupt backing, stale migration, budget
refusal and teardown. A real stories15M lifecycle with a 600-token prefill,
decode, crop and restore compared 19 steps with zero greedy mismatches;
maximum attention error was 0.000344426 and logit error 0.0102015. It executed
126 CUDA attention calls and explicitly migrated two pages. These tolerance
results do not imply bitwise floating-point equality or hybrid lifecycle coverage.
For floating KV, this native test uses stock non-flash F32 accumulation as its
reference; the server workload separately compares against stock flash attention.
The [measurement record](../../docs/measurements/v050-cuda-mechanism-2026-09-16.json)
includes budgets and test scope.

```sh
ARGUS_LLAMA_CPP_DIR="$PWD/scratch/llama.cpp-v050" \
ARGUS_LLAMA_BUILD=build-cuda ARGUS_TEST_CUDA=1 \
ARGUS_TEST_GGUF="$PWD/scratch/models/stories15M.gguf" \
  .venv/bin/python -m pytest -q tests/test_native_llama_paged.py -k cuda
```

Both patches were reverse/applied in order and all eight affected files matched
the tested checkout. CPU-only builds can apply both patches; CUDA additions are
conditional on `GGML_CUDA`.

### Experimental placement policy (v0.6 development)

Set `ARGUS_KV_POLICY=on` to enable the separate `ggml_kv_policy.cpp` module on
the CUDA attention path. Unknown values fail explicitly. Keep an `off` run from
the same build as the reference. Existing checkouts must add the policy source
from the updated `cuda-kv.patch` to the llama target and rebuild.

After attention, written pages with at least two recorded reads are admitted
to GPU, then pinned RAM, then pageable RAM, subject to available tier budgets.
If a tier is full, a page with fewer reads can be demoted to its verified disk
backing. Equal-frequency pages are retained to avoid churning a sequential scan.
Before attention scratch is allocated, resident pages can be evicted across all
live stores to make room for the exact double-buffered tiles and softmax state.
The store still owns budget enforcement, checksum verification and stale-token checks.

Stats include `policy_promotions`, `policy_demotions`, `policy_rejected` and
`policy_nanoseconds` (wall time inside enabled policy calls, including moves).
Rejected moves preserve the source and increment the failure counter. No codec
conversion is performed. Selection uses read frequency, not model topology;
eviction scans descriptors and migrations are synchronous. This is an initial
policy, not a 262K throughput claim. The `cuda_policy` test compares the same
stories15M workload with `off/on`, no manual migrations, and both 4 MiB and
256 KiB GPU budgets.
The [smoke record](../../docs/measurements/v060-policy-smoke-2026-09-17.json)
contains both reports and tested source hashes; it does not measure throughput.

The context ladder accepts `argus-cuda-off` and `argus-cuda-on` with `--kv-type
f16`. Set `--gpu-bytes`, `--pinned-bytes` and optional `--ram-bytes`; use
`--warmups 1 --repeats 3` for repeated comparisons. Token-ID inputs have an exact
recorded length and hash. Each repeat starts a fresh server; workload warmups
run within it, with prompt caching disabled. Counter deltas exclude warmup;
ARGUS peak counters include the whole process lifetime. `prefill_seconds` is
server prompt time, not TTFT; TPOT is the inverse of the server's decode rate.

For attribution, add `--profile` (`ARGUS_KV_PROFILE=1`). `--profile cpu`
(`ARGUS_KV_PROFILE=cpu`) keeps CPU scopes and byte counters without timing events,
to separate observer overhead from the original wait/submission cost.
Counters are split into
prefill (query/append rows > 1), decode (one row), and other operations. CPU
`profile_*_ns` scopes are inclusive; `*_exclusive_ns` subtract nested CPU scopes.
They cover policy, set_rows, page writes, blocking O_DIRECT pread/pwrite,
checksums, page lookup/access accounting, descriptor scans, staging, copy
submission, explicit CUDA synchronization, tier allocation/free, and synchronous
tier reads/writes. Syscall time includes blocking plus kernel/CPU overhead;
it is not device-only service time. Allocation/free can themselves synchronize.
The measured 4016-token input with ubatch=64 has no single-token prefill tail.

Kernel and H2D/D2D staging use CUDA event brackets, read after existing waits;
profiling adds no synchronization points. Event intervals can include host
submission gaps for short operations and device scheduling/resource contention
with the other stream; transfer event latency is not isolated memcpy-engine
service time or a bandwidth benchmark. GPU event time overlaps CPU scope/wait
time and **must not be added** to the CPU exclusive partition. Event objects and
CPU clocks add overhead: compare the same binary and workload without `--profile`.
Top-level attention/append scopes omit llama.cpp scheduler/model work and stats
publication; the request residual is reported separately, not labelled GPU time.

`--modes argus-cuda-control` enables `ARGUS_KV_GPU_CONTROL=1`, a diagnostic
GPU-authoritative store with policy off. Written KV pages have no backing file
and cannot migrate away from GPU; payload disk read/write counters must remain
zero. Writes still use the same CPU set_rows conversion and separately budgeted,
verified GPU page replacement. At the 89eaf06 attribution baseline, attention traverses the same page lookup,
double-buffered staging, 32-cell kernel and synchronization path. The resident
prefill optimization below now bypasses staging; decode retains that reference
path. Masked unwritten padding retains logical-zero semantics. This isolates disk placement; it is not a
durability feature or a proposed optimized production path. Stats/model/log I/O
are outside the zero-KV-payload-I/O claim. The full KV plus scratch and replacement
page must fit the GPU budget or the run fails. Use 64 MiB GPU/pinned for the 4K
Qwen2.5-0.5B diagnostic; compare stock-host, off, on and control in the same run.

The [4K attribution report](../../docs/measurements/v060-4k-attribution-2026-09-17.md)
records paired profiler-disabled/event runs, CPU-only scopes, the original
2 MiB pressure point, and the GPU-only control. It separates overlapping timers
and observed measurement overhead; it does not claim 262K validation.

Resident prefill defaults to a single-launch pointer-table path; for D=64 views whose
rows are aligned to their size it uses the lane-per-cell kernel with a branch-free
value loop (`cells-mlp`). Controlled comparisons (`ARGUS_KV_ATTENTION_PATH`, or
`--attention-path` in the ladder): `cells` keeps the per-cell-branch loop, `batched`
the warp-per-cell kernel, `direct` the intermediate 32-cell resident kernel and
`staged` the reference path. All are bit-exact with `staged`.
Q>1 F16 attention is eligible. GPU pages are read in place; written pages on other
tiers are checksum-verified and copied into bounded per-invocation scratch (not a
placement change); if that scratch does not fit, the invocation falls back to staged.
Unwritten pages retain logical-zero semantics. The registry and both source stores
stay locked until the compute stream completes, protecting against writes, migration
and teardown. Pointer tables, cold-page scratch and optional scalar-path state are
charged to staging/GPU budgets. Read history is recorded with the same 32-cell access
granularity as staged attention; placement-policy decisions are unchanged. Decode
falls back to staged.

The first [direct-path measurement](../../docs/measurements/v060-datapath-direct-2026-09-18.json)
eliminates prefill payload D2D (0 bytes) and cuts explicit waits to 1512, but still
launches 101376 scalar kernels and takes 40.05 seconds prefill versus 41.24 in
the fresh [89eaf06 reference](../../docs/measurements/v060-datapath-before-2026-09-18.json).
This single profiled pair is not evidence of a material speedup; it exposes the
remaining scalar-kernel bottleneck. Decode still uses the staged path and this
sample is slower (3.49 versus 5.02 tok/s), so no decode improvement is claimed.

The [batched-path report](../../docs/measurements/v060-datapath-2026-09-18.md)
records the next optimization: four query/head warps per block process the full
context in one invocation per attention call, preserving the staged FP32
reduction and FMA order. Profiled prefill drops to 12.51 seconds with 1512 kernel
launches and zero payload D2D. One completion wait protects the page lease and
pointer table. Decode and non-resident inputs retain staged attention; the
redundant transfer-stream teardown wait is removed because each staging read
already drains its copies. This is a 4K datapath improvement, not 262K acceptance.

Contiguous `set_rows` writes now share the existing page-rounded encoding
buffer. A gap, repeated index, full buffer or strided target flushes the batch.
This reduces repeated disk writes to the same physical page without increasing
staging allocations or changing per-page verification/rollback. CPU regression
checks cover adjacent, boundary-crossing and repeated indices, plus Q8/Q4 bytes;
the native CUDA lifecycle checks cover F16 output parity.

The opt-in `benchmarks/check_ui_mate_workload.py` compares Turkish, visual labels,
coordinate grounding and a tool call on a generated fixture (Python 3.10+ and
Pillow). It executes no desktop action. It uses the projector's recommended
1024 minimum image tokens, UI-Mate's 0–999 relative coordinate convention and
a server JSON schema for coordinates. Initial
lower-resolution runs exposed coordinate drift despite matching text/tool
outputs; see the [raw workload record](../../docs/measurements/v050-ui-mate-cuda-workload-2026-09-16.json).
Do not treat this small fixture as full Neo or multimodal-quality acceptance.

`benchmarks/check_ui_mate_reference.py` separately uses the upstream UI-Mate
message builder and response parser at revision
`1cb9e1e44ce856e23b593992b02efbd489943fcb`. Supply a reviewed checkout through
`--reference-dir` (containing `agents/ui_mate_agent.py`, `agents/demo_workflow.py`
and a `revision.txt` with that revision), plus the same `--server`, `--model`,
`--mmproj`, `--kv-dir` and `--output` arguments. Its greedy comparison enables
thinking and evaluates the resulting action as data. No generated Python code
or desktop action is executed. Ad hoc visual-chat failures remain in the record;
the official-prompt run is a separate protocol, not a rescore of those failures.
The reference runner defaults to a 1200-second request timeout; use
`--request-timeout` to change it and `--modes argus-cuda` for a targeted retry.
The initial 600-second ARGUS timeout is retained in the measurement history.

## Ownership checks

`tests/cpp/test_ggml_host_buffer.cpp` checks allocation, tensor read/write/copy/view,
budget exhaustion, unchanged source data after failed allocation, and full budget
recovery at teardown. This check passed locally on 2026-09-15.

```sh
c++ -std=c++17 -Wall -Wextra -Werror \
  -Iargus_cache/csrc -I/path/to/llama.cpp/ggml/include -I/path/to/llama.cpp/ggml/src \
  tests/cpp/test_ggml_host_buffer.cpp argus_cache/csrc/ggml_host_buffer.cpp \
  -L/path/to/llama.cpp/build/bin -Wl,-rpath,/path/to/llama.cpp/build/bin \
  -lggml-base -o /tmp/argus-host-buffer-test
/tmp/argus-host-buffer-test
```

For generation parity, run the same model, prompt, seed and context in the same
binary twice, once without `ARGUS_KV_DIR`, once with it and a sufficient byte
limit. Capture `ARGUS_KV allocate/free` and `ARGUS_HOST KV buffer` logs alongside
outputs. An intentionally insufficient budget must refuse context creation.
Full v0.5 acceptance additionally requires layer-level read/write evidence,
logits/state lifecycle parity and integration with tiering.

`tests/cpp/test_llama_host_ownership.cpp` passed with `stories15M.gguf` on CPU:
all 32,000 logits match within 1e-5 for prefill and eight decode steps following
state restore, and a one-byte allocation budget refuses context creation.
The GGUF SHA-256 is
`61b50d457809a5194818fd22e6724b456cd7bb9a6264c52c8110684c53f3704a`.
This is a six-layer F16-KV test, not hybrid, GPU or long-context acceptance.

The CPU server check also passed: stock and ARGUS produced the same 16 greedy
tokens with prompt caching disabled. Only ARGUS had an `argus-ggml-*` mapping,
allocation logs reported 1,769,472 bytes, and teardown returned live bytes to zero.
Run the Linux process-map check explicitly with the same small model:

```sh
python tests/cpp/check_llama_server_ownership.py /path/to/llama-server /path/to/stories15M.gguf /path/to/storage
```

Append `99` to run the same check with GPU model offload. This passed on the
RTX 3050 Ti: 7/7 layers offloaded, 57.95 MiB CUDA model buffer, identical
16-token greedy output, and ARGUS host KV allocation/mapping with full teardown.
The test requires positive GPU offload evidence in logs. KV remains host-resident
(`-nkvo`); this does not prove GPU-resident ARGUS KV ownership or GPU attention.

```sh
c++ -std=c++17 -Wall -Wextra -Werror \
  -I/path/to/llama.cpp/include -I/path/to/llama.cpp/ggml/include \
  tests/cpp/test_llama_host_ownership.cpp \
  -L/path/to/llama.cpp/build/bin -Wl,-rpath,/path/to/llama.cpp/build/bin \
  -lllama -o /tmp/argus-llama-host-test
/tmp/argus-llama-host-test /path/to/stories15M.gguf
```

Mapping files are unlinked after space reservation and released at teardown;
they are temporary KV storage, not checkpoints. `/tmp` may be RAM-backed;
use a verified physical-storage directory for disk measurements.
