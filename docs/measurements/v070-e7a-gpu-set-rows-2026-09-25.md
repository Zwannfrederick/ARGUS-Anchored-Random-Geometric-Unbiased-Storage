# v0.7 E7a: KV appends written on the GPU (GPU control) — ACCEPT

Audit finding L2.1: GPU-born K/V rows were copied to the host, encoded on the CPU, uploaded, and read back for verification. That CPU critical path explained most of the 1.05 s of idle GPU time in the 4K GPU-control prefill. E7a keeps the rows on the GPU for GPU-authoritative stores. The user chose this contract change. Raw data: [`v070-e7-2026-09-25/`](v070-e7-2026-09-25/).

## Change

- **New CUDA custom op.** `argus_ggml_disk_set_rows` builds a CUDA node when all of these hold (`argus_disk_gpu_appendable`):
  - the target store is GPU-control,
  - the KV type is F16,
  - rows never straddle a 4 KiB page.

  Otherwise the CPU append is unchanged.
- **Kind tag.** The node shares the single registered external marker with attention. A kind tag in `op_params` dispatches it, so no llama.cpp patch changed, and the scheduler already places the node on CUDA.
- **What the op does:**
  1. copies the row indices from the device (a few hundred bytes),
  2. resolves each row to its GPU page under the store lock (`argus_disk_gpu_rows`), giving unwritten pages a zeroed GPU page,
  3. publishes the new revisions,
  4. launches `encode_rows`: F32 → F16 with `__float2half_rn`, the same round-to-nearest-even as ggml's CPU `from_float`.
- **Contract change.** There is no transport to verify, so the D2H read-back is gone. A GPU-written page's digest is *pending* until the first host read (`read_page`) computes it, and every later host read checks against it. GPU-memory corruption between the write and that first host read is not detected. Attention never checked digests.

## Correctness

- `check_gpu_set_rows`: GPU-written bytes equal the CPU `from_float` encoding exactly. Coverage:
  - rows: 5 scattered rows, one page already written, one never-touched page that stays unwritten, freshly written pages that read zeros around the new rows;
  - values: random, ±0, subnormal, rounding ties, 65504/65520 (overflow to inf), ±1e9, ±inf.

  Both host reads match: the first computes the pending digests and the second verifies them. A non-GPU-control store keeps the CPU node.
- **Mutations.** Round-toward-zero conversion fails the test. Skipping the zero fill of fresh pages fails the test.
- The accounting check "prefill exclusive time == attention" is a test-local assumption, so the new check runs after it.
- Native suite: 15 passed, 2 skipped. stories15M's 576-byte rows do not divide a page, so its lifecycle tests keep the CPU path. Qwen's 256-byte rows take the GPU path, and the output hash is `a152ed56` on every run.
- The harness check "GPU-resident control did not stage resident KV" expected decode D2D copies, which is the pre-E5 behaviour; E5 had missed it. It now requires resident decode with zero D2D bytes.

## Results (4K, Qwen2.5-0.5B Q4_K_M, F16 KV, ubatch 64, GPU control)

**A/B, 5 alternating pairs, profiler off.** One `llama-server`; `libllama.so` switched with `LD_LIBRARY_PATH`. E6 = `e4eb49a`.

| | prefill median (range) | decode tok/s median |
|---|---:|---:|
| E6 | 2.193 s (2.180–2.292) | 39.61 |
| **E7a** | **1.975 s** (1.949–2.030) | **43.41** |

**−9.9% prefill, faster in 5/5, ranges do not overlap. Decode +9.6%.**

**Final baseline, 5 repeats, same session.** Load average rose from 4 to 7, and other processes drove IO pressure to 21–23%.

- stock-host: prefill 1.303 s, decode 37.97 tok/s.
- **ARGUS GPU control: prefill 2.027 s (1.953–2.060), 1.56x stock** (E6: 1.73x). Decode **43.13 tok/s, faster than stock**.

**Milestones.**
- **M3 (< 2.25 s) met.**
- M4 (< 2.00 s) was met by the A/B median but not by this final baseline, so it is not claimed. It needs a quiet-machine confirmation.

**Policy-on (3 repeats).** 8.80 / **28.77** / 8.44 s. Counters were identical in all three runs.
- E7a does not touch this path: a policy store keeps the CPU append, and only the dispatch check changed.
- The 28.8 s run coincided with 21–23% IO pressure from other processes, and this mode is disk-latency-bound.
- It is recorded here as an outlier, in the same family as the open E1 outlier, and not attributed to E7a.

## Next: E7b

The same GPU append for policy and disk stores, with a write-back flusher that publishes the disk copy in the background. That changes the durability contract: after a crash, disk can trail the GPU by the unflushed pages.
