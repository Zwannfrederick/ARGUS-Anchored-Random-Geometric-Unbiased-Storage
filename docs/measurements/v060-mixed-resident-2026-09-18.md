# v0.6: partial-resident prefill through a mixed pointer table (4K)

Same workload, budgets and output hash as the earlier steps. The placement policy
algorithm is unchanged. [The census](v060-residency-census-2026-09-18.md) showed every
policy-on prefill invocation rejected by one small cold run: the pages the current
ubatch had just rewritten.

## Design

`argus_disk_read_resident` optionally takes `ArgusColdStaging`. Under the registry
and both store locks it now works as follows:

1. It counts written non-GPU pages.
2. It asks `reserve(n)` for host space. The CUDA side checks staging and GPU room for
   host bounce + device copy + the table/state scratch that follows. If there is not
   enough room it declines (`cold_scratch_budget`) instead of evicting, and the
   invocation falls back to the staged path unchanged.
3. It reads each cold page with the existing `read_page`: disk `pread` + checksum,
   or pinned/RAM tier copy + checksum. Verification failures therefore throw
   before anything is queued on the GPU.
4. `upload` sends all cold pages with one H2D copy on the compute stream.
5. It points their table entries at the device copy and runs the unchanged batched
   (or direct) kernel.

GPU pages are still borrowed directly; unwritten pages are still null.

Temporary staging is not placement. `page.resident`, placement/content revisions and
access history are untouched. The 32-cell `record_access` emulation and policy
`observe` run exactly as before. A drain guard owned by the request drains a queued
upload before its buffers are freed if anything fails later.

## Correctness

New mechanism coverage (D = 48, 64, 256; CUDA-event and CPU-only profiling). Every
case is float-vector equal to staged, has identical placement, placement revision and
content revision before/after, and the same access-count deltas as the staged path:

- all-GPU except the unmovable partial tail pages;
- one cold K page (disk);
- one cold V page (pinned);
- GPU + pinned + RAM + disk;
- a rewritten page (write frontier);
- every page cold;
- batched scratch at the exact staging limit (accepted) and one page below
  (declined, staged result identical);
- a corrupted cold read (throws; descriptors, access counts, GPU and staging usage
  unchanged; the next read succeeds);
- a concurrent writer blocked until borrowed/staged pages are released;
- the original mixed-tier test now runs through the mixed path, still within the
  1e-4 double-reference bound and equal to staged.

New native test (stories15M, policy on, 600-token prompt, staged vs batched):

- 4 MiB GPU budget: 24/24 prefill invocations accepted (1800 cold pages).
- 256 KiB: the policy keeps the GPU tier full, so all 24 decline on
  `cold_scratch_budget` to the staged path.
- In both: greedy tokens identical; policy promotions, demotions and rejections,
  written bytes, committed pages and attention calls identical.
- `read_bytes` is lower on the mixed path: with 576-byte cells, staged 32-cell tiles
  re-read the pages that straddle two tiles.

GPU tests `7 passed`, CPU/context tests `5 passed`.

## Result: 4K, profiler disabled, 3 repeats each

| Mode | Before | After |
|---|---:|---:|
| policy-on prefill median (min–max) | 46.152 s (46.071–46.548) | **12.062 s** (11.999–12.231) |
| GPU-control prefill median (min–max) | 5.178 s (5.146–5.199) | 5.277 s (5.204–5.304) |

Policy-on prefill is 3.83x faster. GPU-control never has cold pages; its only new
work is a second pass over the page descriptors (microseconds per invocation). The
+1.9% is treated as run-to-run drift: an identical control build measured 5.222 s in
an earlier repeat set. A 1–2% control regression is not excluded by these samples.

[policy-on before](v060-census-policy-baseline-2026-09-18.json), [policy-on after](v060-mixed-resident-policy-2026-09-18.json),
[control after](v060-mixed-resident-control-2026-09-18.json)

## Physical path (profiled, one repeat each; not mixed with the timings above)

| Policy-on prefill | Before ([profile](v060-datapath-policy-profile-2026-09-18.json)) | After ([profile](v060-mixed-resident-profiled-2026-09-18.json)) |
|---|---:|---:|
| Resident invocations / cold pages staged | 0 / 0 | 1512 / 12048 |
| Kernel launches | 101376 | 1512 |
| Kernel event seconds | 37.361 | 3.852 |
| Payload D2D bytes | 1609236480 | 0 |
| H2D bytes | 51707904 | 52592640 (cold pages 49348608 + tables) |
| Disk read / write bytes | 148045824 / 49348608 | 148045824 / 49348608 |
| Explicit sync calls | 302616 | 1512 |
| Policy promotions / demotions | 12768 / 0 | 12768 / 0 |

Disk traffic and policy decisions are identical; the cold pages are exactly the pages
the staged path also read from disk. An intermediate build synchronized once more per
invocation (a temporary `Drain` destroyed while armed); it was fixed before these
measurements.

## Remaining policy-on cost

Profiled policy-on prefill minus kernel time is still ~8.8 s against ~1.9 s for
GPU-control. The inclusive profile attributes most of it to the write path
(`set_rows` 6.1 s, including `write_page` 5.8 s: `pwrite` + verification `pread` and
checksum of every rewritten page) and `disk_read` 4.5 s, which includes those
verification reads as well as cold-page reads. The scopes overlap and must not be
summed; policy is 0.8 s inclusive. That is storage durability/placement cost, not attention datapath cost, and
is not changed here.
