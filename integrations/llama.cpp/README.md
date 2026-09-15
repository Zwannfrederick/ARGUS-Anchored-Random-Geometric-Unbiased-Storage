# Experimental llama.cpp host KV ownership seam

Pinned upstream revision: `54315813269112dd0baed7112ec87ad93a8218ca`.
Patch: `host-kv.patch`. Implementation: `argus_cache/csrc/ggml_host_buffer.cpp`.

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
layer on the `ARGUS_DISK` buffer. It requires `-nkvo -fa on`. K/V bytes keep
GGML's codec (F32/F16/Q8_0/Q4_0) and live only in an unlinked `O_DIRECT` file;
tensor addresses are inaccessible handles, never resident KV.

- Appends go through an ARGUS `set_rows` op. Each 4 KiB page is written to a
  second slot, read back and checksum-verified before its descriptor moves, so
  a failed or short write keeps the previous page. Disk reservation is twice the
  KV size.
- Attention reads visible blocks from disk. A single-depth prefetch worker loads
  the next visible block; a full queue or a stale generation after crop/reset is
  refused, and destruction joins pending work.
- Budgets are enforced, not advisory: `ARGUS_KV_MAX_BYTES` bounds disk bytes,
  `ARGUS_KV_RESIDENT_BYTES` bounds page-descriptor metadata, and
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
cmake -S /path/to/llama.cpp -B /path/to/llama.cpp/build \
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF \
  -DARGUS_CORE_DIR=/path/to/ARGUS/argus_cache/csrc
cmake --build /path/to/llama.cpp/build --target llama-cli llama-server -j2
```

Quote paths containing spaces. CMake is a build tool, not an added Python
runtime dependency.

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
