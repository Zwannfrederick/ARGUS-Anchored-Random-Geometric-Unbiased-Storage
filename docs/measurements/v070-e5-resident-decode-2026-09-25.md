# v0.7 E5: single-token decode on the resident path — ACCEPT

The staged Q=1 path was the decode path of every CUDA mode since v0.5. For each decode token and each layer it:
- allocated pinned and GPU tiles and a stream,
- copied the GPU-resident pages 4 KiB at a time into scratch,
- launched one kernel per 32 cells.

At 4K context that came to 46,080 launches and 184,320 staging copies per request (audit: [control-audit P0.1](../plans/control-audit.md)). E5 removes the `reject_q1` gate, so decode uses the same resident kernel as prefill: one launch per layer, pages read in place through the pointer table, cold pages staged through `ArgusColdStaging`. Raw data: [`v070-e5-2026-09-25/`](v070-e5-2026-09-25/).

## Correctness

- **TDD.** The mechanism test previously *required* the Q=1 rejection. It now requires every resident path (direct, batched, cells, cells-mlp, cells-kc, cells-v2) to accept Q=1 and to equal the staged result bit for bit, at D = 48/64/256. It failed before the change and passes after.
- **Policy equivalence.** `test_cuda_mixed_resident_path_preserves_policy_semantics` caught a real divergence: 2167 staged promotions against 2441 resident, and 92 demotions against 373. The staged path ran `argus_kv_policy_observe` while its tile scratch (`2·tile + state` GPU, `2·tile` pinned) was still allocated, so the policy counted transient attention buffers as occupied budget. The resident path frees its scratch first.
  - **Fix:** the staged scratch now lives in a scope that closes before `observe`, matching the resident path. Transient buffers should not steer placement.
  - After the fix, promotions, demotions, rejections, written bytes and commits are equal at both GPU budgets (4 MiB and 256 KiB), and resident `read_bytes` ≤ staged (`policy_diff.py`).
- `test_cuda_gpu_control_has_no_disk_io_and_preserves_lifecycle` asserted `profile_decode_d2d_bytes > 0`, which encoded the old staging copies. It now asserts the new contract: decode accepted on the resident path, with zero D2D bytes.
- Native suite: 15 passed, 2 skipped (no quantized GGUF). Output hash `a152ed56` unchanged in every mode.
- The `reject_q1` counter is removed; nothing else read it.

## Results (4K, Qwen2.5-0.5B Q4_K_M, F16 KV, ubatch 64, 16 generated)

**GPU-control** (5 repeats, `e5-final-baseline.json`):

| | prefill | decode tok/s (min–max) | TPOT |
|---|---:|---:|---:|
| stock-host | 1.286 s | 39.06 (38.07–39.23) | 25.6 ms |
| ARGUS GPU-control, E4 (`a9240b2`, same day) | 2.328 s | 7.53 | — |
| **ARGUS GPU-control, E5** | 2.187 s | **39.31** (38.54–40.51) | 25.4 ms |

**Decode goes from 4.6x slower than stock to parity: 5.2x.** Resident decode ran 360 of 360 invocations. The prefill path is unchanged by E5; the stock baseline also moved (1.342 → 1.286 s), and the ratio moved with it (1.73x → 1.70x), so this is session variance and no prefill claim is made.

**Policy-on** (3 alternating pairs in one binary; `staged` forces the old decode path, `policy-on-decode-ab/`):

| pair | staged decode | resident decode |
|---|---:|---:|
| 1 | 6.73 | 18.49 |
| 2 | 6.07 | 15.57 |
| 3 | 6.13 | 14.29 |

**≈2.5x**, faster in 3/3. Policy counters are unchanged against E4 (12,768 promotions, 0 demotions, 12,768 commits).

**Stated plainly.** In the first policy-on batch, 2 of 3 runs decoded at only 2.3 tok/s (`e5-policy-on-first.json`) while disk IO pressure from other processes was elevated. Those runs had the same counters as the fast ones. The CPU profile (`policy-prof-*.json`) shows why this mode is so sensitive:
- decode time is dominated by `set_rows` → `write_page` (disk write plus verification read),
- then `disk_read`,
- every token, the tail pages of K and V are rewritten to disk and read back cold (2 cold pages per call, 720 per request).

That is logic-audit finding L0.1. Policy-on decode stays disk-latency-bound until it is fixed.

## What remains

- Policy-on decode: L0.1 (a rewrite drops the page's promotion) and L0.2 (LFU heat) are the next items.
- Q=1 occupies only 4 blocks (14 heads). Split-K would fill the GPU, but decode is already at stock parity at 4K; it only becomes worth doing at longer contexts, and it needs its own exactness decision.
