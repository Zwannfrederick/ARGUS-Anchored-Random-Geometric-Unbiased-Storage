# v0.7 E6: placement policy logic faults — two fixed, one withdrawn

Item 3 of the [audit action list](../plans/control-audit.md). These changes affect only the policy-on and disk modes. GPU-control does not run the policy and returns early in `write_page`, so its code paths are unchanged. Raw data: [`v070-e6-2026-09-25/`](v070-e6-2026-09-25/).

## 1. A rewrite dropped the page's promotion (audit L0.1) — FIXED

`write_page` (disk path) committed the new bytes to disk and then deleted the page's resident copy. Every decode append rewrites the tail page, so that page went back to disk every token, was read cold by the next attention call, and was promoted again.

**Change:** after the disk page is published, the resident copy is refreshed from the verified bytes, read back and checked. If the refresh fails, the copy is dropped and the published disk page stays authoritative. `read_page` still checks every resident read against the descriptor's checksum.

**Tests (TDD):**
- `check_policy`: rewriting a promoted page keeps it on the GPU, with correct contents, and makes no policy moves. Failed before the change.
- `check_mixed_residency`'s "write frontier" case encoded the old behaviour ("A rewrite drops residency"). It now requires that the frontier stays resident (cold pages 5 → 4).
- Not tested: the refresh-failure branch, because GPU/pinned corruption cannot be injected.

**Effect, 64 MiB budget, steady state (the measured request after a warmup):**

| | E5 (`077c816`) | E6 |
|---|---:|---:|
| promotions | 12,768 | **0** |
| cold pages (prefill + decode) | 1,800 + 720 | **0** |
| `read_bytes` | 156.9 MB | **52.3 MB** (−67%) |
| decode tok/s | 14.3–18.5 | 14.9–18.9 |

Wall clock is unchanged within noise. At this budget, decode is dominated by the durable write itself: `pwrite` plus the verification `pread` for every appended row (`policy-prof-*.json` in E5).

## 2. Promotion filled the attention scratch headroom (new, found by measurement) — FIXED

**How it was found.** A before/after A/B at the v0.6 tight budget (GPU 2 MiB, pinned 2 MiB), using `LD_LIBRARY_PATH` to switch `libllama.so` under one `llama-server`, showed that fix 1 alone changed almost nothing: 78,680 → 78,504 promotions, 78,496 demotions in both.

**Mechanism.**
1. `argus_kv_policy_prepare` force-evicts pages so the next invocation's scratch fits.
2. After attention, `observe` promoted pages straight into the room that scratch had just vacated.
3. The next `prepare` evicted them again.

That is about 40 pages per call over 1,872 calls. Cache and scratch were trading the same budget on every call.

**Change:** `prepare` records the headroom it reserved per tier. `observe` never promotes into that headroom, and only swaps out a colder page when the headroom is intact.

**Test (TDD):** `check_scratch_headroom`. With a 3-page GPU budget, a 1-page reservation and 4 invocations, it expects:
- exactly 2 cached GPU pages,
- no demotions,
- 4 promotions in total.

It failed before the change.

**Effect, tight budget (2 alternating pairs; `before` = `077c816`, `new` = E6):**

| | before | new |
|---|---:|---:|
| promotions / demotions | 78,680 / 78,496 | **36 / 34** |
| `read_bytes` | 3.04 GB | **2.14 GB** (−30%) |
| prefill | 51.3 / 54.0 s | **46.4 / 46.9 s** (−10–13%) |
| decode tok/s | 0.909 / 0.757 | **1.040 / 1.013** |
| resident prefill accepted | 0 | 60 |

The tight budget is still slow. The 48 MiB of KV has 2 MiB of GPU tier, so most pages come from disk on every call. That is the cost of the budget itself, no longer policy waste.

## 3. "LFU heat is inverted" (audit L0.2) — WITHDRAWN

The audit claimed that LFU evicts the page certain to be read next, and proposed resetting `access_count` on rewrite. Both parts were wrong for this workload:
- Under full causal attention, *every* visible page is read every step. "Certain to be read next" is true of all of them. For a cyclic scan larger than the tier, hit rate depends on keeping a *stable* subset, and monotone LFU counts already give that. LRU would be the pessimal choice.
- Resetting the count on each rewrite would keep the tail page's count near zero, so it would be chosen as the victim every token. That would recreate the ping-pong fix 1 removed.

What remains true is minor: `last_access_step` is recorded and never used by the policy (P3). The logic-flow report is corrected accordingly.

## Regression and correctness

- Native suite: 15 passed, 2 skipped. Hash `a152ed56` in every run of every mode.
- Policy-on at 64 MiB (3 repeats): prefill 8.81–9.36 s, decode 16.6–17.5 tok/s, counters stable.
- GPU-control (3 repeats, same batch): prefill 2.37–2.48 s, decode 35–37 tok/s. This is slightly below E5's 2.19 s / 39.3 tok/s, but no GPU-control code path changed, and the load average was 5.0–5.5 during this batch. Treat it as session variance.
