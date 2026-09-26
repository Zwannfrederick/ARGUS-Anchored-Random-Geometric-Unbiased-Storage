# ARGUS v0.7 — experimental deep optimization campaign

Branch `v0.7-dev`, based on `391e37d` (v0.6.0 release; the accepted kernel checkpoint is
`3aea617`). v0.6 is the immutable reference. Exact by default: no fast math,
reassociation, tensor cores, changed reduction or mask order, tolerance change or
placement-semantics change. Category 2/3 ideas stay opt-in experiments.

## Workload and method

Qwen2.5-0.5B-Instruct Q4_K_M, F16 KV, 4K context, 4016-token prompt, 16 generated,
ubatch 64, GPU-control (`argus-cuda-control`) unless stated otherwise, RTX 3050 Ti Laptop.

- Profiler-off screening: ≥3 repeats. Close results: 5–7 alternating repeats.
- Report median, min/max, relative delta and stock ratio.
- Same-binary A/B through selectable variants (`ARGUS_KV_ATTENTION_PATH` or
  compile-time variants).
- The GPU must be idle; check `nvidia-smi` for other compute processes, such as an
  Ollama runner loaded by the live adapter test.

Every experiment records:
- hypothesis, the one causal factor and the expected profiler signature;
- implementation delta and correctness category;
- result (wall clock, NCU when relevant);
- ACCEPT or REJECT, with the rollback reason.

Acceptance prefers ≥5% total GPU-control prefill. Kernel-only wins and complexity-heavy
small wins are rejected. Negative results are documented and reverted.

Milestones (GPU-control prefill): M1 < 2.70 s, M2 < 2.50 s, M3 < 2.25 s,
M4 < 2.00 s, stretch < 1.75 s. No stock-parity claim without same-session evidence.

## Baseline (2026-09-18, `v0.7-dev` = v0.6.0 code, profiler off, 3 repeats; [JSON](../docs/measurements/v070-baseline-2026-09-18.json))

| mode | prefill median (min–max) | vs stock |
|---|---:|---:|
| stock-host | 1.331 s (1.322–1.331) | 1.00x |
| GPU-control | 2.778 s (2.773–2.789) | 2.09x |

Hash unchanged (`a152ed56…`); gap to stock ≈ 1.45 s.

## Macro attribution so far (ARGUS scopes only)

Profiled runs ([CPU scopes](../docs/measurements/v070-control-cpu-profiled-2026-09-18.json), prefill 3.059 s under profiling) and
[CUDA events](../docs/measurements/v070-control-cuda-profiled-2026-09-18.json) (3.094 s). Inclusive CPU scopes are not summed with events.

| component | time | notes |
|---|---:|---|
| A attention custom op | 1.398 s incl. | 1.315 s of it waits for the kernel; kernel events 1.303 s, 1512 launches |
| B KV write / `set_rows` | 0.612 s incl. | `write_page` 0.463 s: per-page `cudaMalloc`/`cudaFree`, H2D write, D2H verification read (`tier_read` 0.201 s), two checksums (0.132 s); row gather 0.149 s |
| C page table / lookup / descriptor scan | ≈0.03 s | 231k lookups, 1512 scans |
| D synchronization | inside A | one wait per invocation; it is kernel time |
| E copies | inside A/B | pointer tables 3.2 MB H2D; page payload only in B |
| G allocation / release | 0.081 s | inside B (12,048 pages + 1,512 tables) |
| H + I llama.cpp non-ARGUS work, scheduler, splits, profiler | ≈1.05 s remainder | **not decomposable without Nsight Systems** |

Stock-host's 1.33 s (its FlashAttention, per-split KV H2D copies, host `set_rows` and
the same model compute) is also not decomposed.

Working conclusion, to be confirmed by nsys: attention is still the largest ARGUS
component of the gap. The GPU-control write-verification path (B) is the second, and
it is a durability/diagnostic-semantics path, not free to change.

## Macro timeline (Nsight Systems, 2026-09-18)

`nsys profile -t cuda --cuda-graph-trace=node`, one warmup + one measured request per
mode. Window = the measured request's prefill: from the first GPU activity after the
previous request's last decode attention to the first decode attention of the measured
request. Measured prefill attention launches are ARGUS cells #1560–3071 and stock
FlashAttention #1920–3431, after 48 start-up launches per mode. Windows match the server's
reported prefill: ARGUS 2.798 s vs 2.800 s, stock 1.506 s vs 1.507 s. Tracing inflates
stock by ~13% and ARGUS by ~1% against profiler-off medians; use the rows for
attribution only. Script: [`v070-nsys-window.py`](../docs/measurements/v070-nsys-window.py).

| component (measured prefill) | ARGUS GPU-control | stock-host | ARGUS − stock |
|---|---:|---:|---:|
| attention kernels | **1.292 s** (cells, 1512) | **0.073 s** (FA 0.054 + fixup 0.019) | **+1.219 s** |
| model kernels (mmq 0.25, fixup 0.04, quantize 0.03, glu/bcast/norm/rope ~0.06) | ≈0.39 s | ≈0.38 s | ≈0 |
| H2D copies | 0.017 s (84 MB: page writes, tables) | **0.558 s** (1.69 GB: host KV to GPU per split) | **−0.54 s** |
| D2H copies | 0.049 s (149 MB: page-write verification, logits) | 0.030 s (99 MB) | +0.02 s |
| GPU idle inside the window | **1.049 s** | 0.470 s | **+0.58 s** |
| total | 2.798 s | 1.506 s | +1.29 s |

GPU idle in ARGUS is the CPU critical path between GPU work, dominated by `set_rows`
page writes. Runtime API inside the window: `cudaMemcpy` 0.237 s (24k synchronous page
writes/reads), `cudaMalloc` + `cudaFree` 0.077 s (13.6k each), launches 0.213 s. Syncs
(1.57 s) mostly wait for attention.

Conclusions:
- Attention alone exceeds the whole gap. Candidate F does not trigger; the attention
  campaign is the right first target.
- ARGUS already avoids stock's 0.54 s of per-split KV H2D, and gives about that back
  in its CPU-side page-write/verification path. That path is the second target, a
  separate campaign because it touches write verification semantics.
- Stock's attention uses tensor-core FlashAttention (arithmetic category 3) and is 17.7x
  faster than the exact kernel. Exact-order work closes part of this; the remainder is
  the price of the exactness contract, to be measured, not assumed.

## Pending measurements (need the user: installs and root)

1. ~~Nsight Systems~~: installed and captured (above).
2. ~~Experiment 0~~: captured and analysed below. Original request:
   **memory-pressure discrimination (Nsight Compute, root).** Same late
   launch (measured request, ubatch 60, layer 0, `--launch-skip 3000`), for
   `cells-mlp` (default) and `cells`. Use the `MemoryWorkloadAnalysis_Chart` and
   `breakdown:` metrics. The goal is to learn which unit the 76% "memory access
   throughput" is:
   - L1TEX data or tag pipe;
   - LSU input requests;
   - MIO queue or shared-memory pipe;
   - L2 or DRAM.

   It should also explain why the L1 hit rate fell from 94% to 70% while L2 misses
   stayed flat (in-flight duplicate requests vs. real evictions).

## Experiment matrix (one factor at a time; chosen only from measurements)

- **A — V MLP depth.** Unroll depth 2/4/8 of the branch-free value loop. Selected if
  E0 shows the LSU/L1 pressure is created by the depth-4 burst while latency is hidden.
- **B — GQA-aware V reuse.** Only if request maps show the 7 heads sharing a KV head
  miss in L1 on the same lines close together, and caching fails to absorb it.
- **C — explicit V staging** (shared tile or warp-cooperative load). Only if L1TEX
  request/tag pressure dominates.
- **D — V row pointer delivery.** Only if MIO/shared pointer loads are material.
- **E — K path.** Low priority (~11% of old stalls; ncu estimated ~4.6%).
- **F — non-attention.** If nsys shows attention is not the majority of the gap,
  stop attention work and attack the dominant component.

Policy-on (GPU-control 2.78 s vs policy-on 9.89 s) is a separate, later campaign. Its
v0.6 physical counters stay the semantic reference: 1512/1512 fast path, 12,048 cold
pages, D2D 0, H2D 52.59 MB, disk 148.0/49.3 MB, 1512 syncs, 12,768/0 promotions and
demotions. Decode is untouched until the prefill campaign reaches a measured stopping
point.

## Experiment 0 result (Nsight Compute, same late launch; `cells` vs `cells-mlp`)

Reports: [`v070-e0-2026-09-18/`](../docs/measurements/v070-e0-2026-09-18/). The ncu
durations are profiler-only and serve as ratios.

| metric | `cells` | `cells-mlp` |
|---|---:|---:|
| duration | 3.557 ms | 2.065 ms |
| memory-access throughput (% peak) | 43.7 | **76.3** |
| = `l1tex__data_pipe_lsu_wavefronts` (% peak, avg / busiest SM) | 43.7 / 50.7 | **76.3 / 81.9** |
| global load requests / sectors | 8.14 M / 42.09 M | 8.14 M / 42.09 M (identical) |
| L1 lookup hit / miss sectors | 39.57 M / 2.52 M | 29.92 M / **12.17 M** |
| L1 → L2 read sectors (xbar / lts) | 2.52 M / 2.46 M | **12.23 M / 12.50 M** |
| L2 hit rate / DRAM read | 95.1% / 4.85 MB | 96.9% / 4.74 MB |
| LSU instructions (= lsuin requests) | 17.93 M | 14.72 M |
| shared wavefronts / bank conflicts | 11.73 M / 0 | 12.01 M / 0 |
| issue active, eligible warps | 0.32, 0.59 | 0.57, 1.63 |

**Where the 76% comes from (measured).** The high-level metric is exactly the L1TEX
data pipe's LSU wavefronts. No other unit is close: the LSU instruction pipe peaks at
55% on the busiest SM, and L2/DRAM traffic is small.

**Which instructions fill that pipe (inferred from measured totals and per-PC
execution counts; there is no per-PC wavefront counter).**

| source | requests | wavefronts |
|---|---|---:|
| K rows: 8 `LDG.E.128` per lane per tile; lanes 256 B apart, so every lane hits its own line | ~0.87 M, 32 wavefronts each | ≈27.9 M |
| q, weight and pointer `LDS` | — | 12.0 M |
| V row loads (64 B per warp, one line) | ~6.9 M, 1 wavefront each | ≈6.9 M |
| **sum** | | **≈46.8 M** |

The sum matches the ≈46.6 M wavefronts implied by 76.3% of one wavefront per SM per
cycle over the kernel. **K loads are ~60% of the saturating unit's work.** Sectors
agree: K ≈27.9 M + V ≈13.9 M ≈ 41.8 M of the measured 42.09 M.

**Why L1 hit fell 94% → 70% (measured).** Requested traffic is byte-identical
(8.14 M requests, 42.09 M sectors), and DRAM is unchanged. What changed is that
9.7 M more sectors miss L1 and are fetched from L2, where they hit.
- With the branch-free loop, many warps issue value loads for the same KV lines
  (7 query heads × 64 tokens share each KV head) before the first fill arrives.
- Those loads count as L1 misses and are each fetched from L2 again.

These are duplicated in-flight misses, not capacity evictions. They cost L2 bandwidth
(still far from limiting) and put L2 latency on the first consumer of each 4-cell
group, where the remaining long-scoreboard stall sits.

### Hypothesis verdicts

| hypothesis | verdict | evidence |
|---|---|---|
| V MLP depth too aggressive | **UNRESOLVED**, secondary | Depth-4 bursts create the duplicate L1 misses (+9.7 M sectors to L2), but L2 is not limiting. The time cost would be L2 latency on the group's first value, not throughput. Needs the depth sweep to quantify. |
| L1/TEX data/tag/request pressure | **ACCEPT (data pipe)** | Data-pipe wavefronts 76–82% of peak = the whole memory-access metric; about 60% of it is the lane-scattered K-row loads |
| LSU pressure | REJECT as primary | LSU instructions fell 18%; LSU pipe ≤55%. `lg_throttle` 0.36 and `mio_throttle` 1.56 cycles/issue follow the data-pipe backlog. |
| MIO pressure | secondary symptom | mio_throttle rose 0.61 → 1.56; shared wavefronts unchanged (12.0 M), no bank conflicts |
| L2/DRAM pressure | REJECT | DRAM 4.7 MB; L2 hit 97%; L2 read sectors ×5 but L2 utilization stays far below L1's |
| GQA-induced redundant V traffic | **ACCEPT as L1→L2 duplication, REJECT as bandwidth limit** | In-flight duplicate misses of shared KV lines; L2 absorbs them |
| shared pointer path | REJECT | shared wavefronts equal before and after, 0 conflicts; its stall share did not grow |

## Next experiment (exactly one): coalesced K-row loads through a per-warp shared transpose

- **Measured evidence.**
  - L1TEX data-pipe wavefronts are the most utilized resource (76% avg, 82% busiest SM).
  - About 60% of them (≈27.9 M) come from the score phase's lane-scattered 16-byte K
    loads: 32 wavefronts per request, ncu's "21.4 of 32 bytes per sector used,
    33% excessive sectors".
  - `mio_throttle`/`lg_throttle` rose with that backlog.
- **Causal hypothesis.** Loading each 32-cell K tile with coalesced requests cuts K
  data-pipe wavefronts ~8x. That frees the L1 data pipe, which lowers queueing
  throttles and memory latency for the value path, and raises issue rate.
- **Single factor.** How the K rows of a tile reach the lane that scores them.
  Nothing else changes: the tree, the values, the V loop, the unroll depth and the
  mask handling all stay as they are.
- **Minimal code change** (new variant, e.g. `ARGUS_KV_ATTENTION_PATH=cells-kc`, and a
  template flag on `attention_resident_cells`; default unchanged):
  1. Load the tile's K rows with lanes 0–7 covering one 128-byte row, so 4 rows per
     `LDG.E.128` warp instruction, all lines fully used. Null pages load as zero, as
     today.
  2. Store them to a per-warp shared buffer padded to avoid bank conflicts.
  3. `__syncwarp()`.
  4. Each lane reads its own row back with `LDS.128`.
  5. The existing per-lane tree runs unchanged.

  Process the tile in chunks whose buffer keeps the block at the current 8 resident
  blocks/SM (register-limited). An 8-cell chunk is ~1.2 KiB per warp; a 16-cell chunk
  (~2.3 KiB per warp) only if the measured occupancy stays 8.
- **Invariants.**
  - Every product is `__fmul_rn(q[i], k[i])` on the same `k` values.
  - The same addition tree runs in the same lane, and the same score `__fmaf_rn`.
  - The prefix max, alpha/beta, and `sum`/`acc` chains are untouched.
  - Mask, null-page and unwritten semantics are unchanged.
  - No extra barrier beyond `__syncwarp`. No policy or placement change.
- **Exactness category 1.** Only the path of the K bytes changes; the arithmetic
  and its order are identical.
- **Expected SASS.**
  - The score phase shows 8 coalesced `LDG.E.128`, `STS.128` and `LDS.128` in place
    of today's 8 per-lane scattered `LDG.E.128`.
  - The value loop is identical to `cells-mlp`.
  - Registers ≤ 58 (target), no spill, static shared ≤ what keeps 8 blocks/SM.
- **Expected NCU signature** (same launch, `--launch-skip 3000`):
  - `l1tex__data_pipe_lsu_wavefronts` falls from ≈46.6 M to roughly 25–30 M
    (K 27.9 M → ~3.5 M global + ~7 M shared);
  - L1 memory-access throughput falls below ~55%;
  - K sector efficiency 32/32 bytes;
  - `mio_throttle` and `lg_throttle` decrease;
  - issue active above 0.57;
  - duration below 2.07 ms.
- **Expected effect (estimate).** Kernel −10% to −25%, about 0.13–0.32 s of the
  1.29 s attention time. GPU-control prefill ≈2.78 s → 2.5–2.65 s (−5% to −10%).
- **Benchmark methodology.**
  - Same binary, `cells-mlp` vs `cells-kc`, 4K GPU-control, profiler off.
  - 3 repeats each; 5–7 alternating repeats if the difference is 3–8%.
  - Then policy-on 3 repeats, as a regression check only.
- **Acceptance.**
  - Bit-exact everywhere, with the output hash unchanged.
  - GPU-control prefill median ≥5% faster, with non-overlapping ranges.
  - The NCU signature above observed.
  - Registers ≤ 64, no spill.
- **Rejection / rollback.** Any of the following rejects the variant; it is documented
  as a negative result and `cells-mlp` stays the default.
  - Parity failure.
  - Spill, or occupancy below 8 blocks/SM without a measured net win.
  - Data-pipe wavefronts not reduced.
  - Prefill gain < 5% (3–5%: inconclusive, re-measure with 7 repeats).
- **Correctness tests.**
  - Add `cells-kc` to every existing path list: the resident and mixed-residency
    loops at D = 48/64/256, shifted views, cold pages.
  - Run all Qwen cases: causal, scattered −∞, fully masked tile and rows, live null
    pages, F32 bias, ×24 scores, ±0 with ties.
  - Mutation checks: permute the K row assignment (must fail) and skip `__syncwarp`
    (review only).
  - Suites: 7/7 GPU, 5/5 CPU/context.
- **Profiler validation.** One `sudo ncu` capture of the same launch with
  `MemoryWorkloadAnalysis`, `WarpStateStats`, `SchedulerStats`, `LaunchStats`,
  `Occupancy`, `SourceCounters` and
  `breakdown:gpu__compute_memory_access_throughput.avg.pct_of_peak_sustained_elapsed`,
  compared against `v070-e0-2026-09-18/e0-cells-mlp-late`.

Deferred behind this experiment: the V MLP depth sweep (candidate A; revisit if long
scoreboard on the first group value dominates after K is relieved), GQA cooperative V
reuse and explicit V staging (V is ~15% of the data pipe), pointer delivery.

## E1 result (2026-09-19): coalesced K-row loads — ACCEPT, now the default

Report: [`v070-e1-cells-kc-2026-09-19.md`](../docs/measurements/v070-e1-cells-kc-2026-09-19.md).

- Bit-exact everywhere; hash `a152ed56…`; GPU 7/7, CPU/context 5/5.
- GPU-control prefill: 2.858 → 2.667 s (−6.7%, 7 alternating pairs, kc faster in 6/7).
- Prefill kernel events: 1.314 → 1.156 s (−12.0%); policy-on kernel events −14.5%.
- NCU, same launch as E0:
  - data-pipe wavefronts ≈46.6 → 35.9 M (global −58%, shared +9.4 M);
  - utilization 76 → 67%;
  - `lg_throttle` 0.38 → 0.03; long scoreboard 3.52 → 2.24;
  - issue 0.57 → 0.63;
  - duration −12%.
- Accepted under the occupancy exception: 66 registers, 7 blocks/SM, and a 100 KB
  shared carveout that leaves L1 at 28 KB.
- Weaker than predicted: 35.9 M wavefronts, not 25–30 M, and 67% utilization, not < 55%.

Final baseline, profiler off, 5 repeats, on an idle GPU:
- stock-host 1.341 s;
- GPU-control **2.709 s** (2.02x); M1 < 2.70 s not met;
- one repeat of 5.917 s with no other GPU process present: an unattributed ARGUS-side
  outlier, open item.

New dominant in-kernel bottleneck:
- The L1TEX data pipe is still the most utilized unit (67%), now 60% shared wavefronts:
  q/weight/pointer `LDS` 12.0 M, K tile 9.4 M.
- `mio_throttle` 1.51 and long scoreboard 2.24 lead the stalls.

Macro: attention is still the largest ARGUS component, and the CPU page-write path is
second. The next experiment is chosen from these measurements in a separate turn.

## E2 result (2026-09-23): the GPU-control page-write path — ACCEPT, both factors

Report: [`v070-e2-write-path-2026-09-23.md`](../docs/measurements/v070-e2-write-path-2026-09-23.md).
This opens the second campaign the nsys section named. It turned out not to touch the
verification contract at all: the same digests are compared over the same bytes before
the same publication, only cheaper and in flushes.

Chosen from a direct measurement of the per-page costs on this GPU, before any code
changed: FNV-1a 5.34 us/page (twice per page), blocking 4 KiB H2D 7.37 us, blocking
4 KiB D2H 8.76 us, `cudaMalloc` + `cudaFree` 3.34 us. Pinning the host side alone is
worthless at 4 KiB (D2H 7.83 us pinned vs 8.76 pageable); queueing a flush's copies on
one stream and waiting once is what pays (18.8 → 8.7 us per page).

- **E2a, hardware CRC32C digest** (`ARGUS_KV_CHECKSUM=fnv` keeps FNV-1a selectable):
  −4.3%, 2.672 → 2.557 s, faster in 7/7 alternating pairs.
- **E2b, whole-page runs committed together** (`ARGUS_KV_PAGE_COMMIT=page` keeps the
  per-page commit selectable): −5.6%, 2.529 → 2.387 s, faster in 6/7 pairs.

Profiled CPU scopes, per request, page/fnv → run/crc32c: `set_rows` 0.566 → 0.309 s,
`write_page` 0.461 → 0.214 s, `tier_read` 0.204 → 0.086 s, `checksum` 0.136 → 0.015 s.
`attention` (1.251 → 1.270 s) and `synchronization` (1.154 → 1.173 s) are flat across
all four variant cells — each factor moves only its own scope.

Costs: `peak_pinned_bytes` 32,768 → 65,536 (store-owned commit staging, released with
the store). `peak_staging_bytes` and `peak_gpu_bytes` unchanged.

Correctness: hash `a152ed56…` everywhere, GPU 7/7, CPU/context 5/5, policy-on counters
unchanged (25,632 committed, 25,536 promotions, 0 demotions).

Final baseline, profiler off, 5 repeats, idle GPU:
- stock-host 1.342 s (1.340–1.389);
- GPU-control **2.406 s** (2.370–2.424), **1.79x**, −11.2% against E1's 2.709 s;
- decode 7.08 → 8.08 tok/s;
- M1 (< 2.70 s) and M2 (< 2.50 s) met; M3 (< 2.25 s) not.
- Policy-on 9.178 s median (3 repeats) against E1's 10.04–10.43 s: no regression.

The E1 outlier did not reappear in 43 GPU-control prefills today (slowest 2.878 s).
24k blocking driver round trips per prefill were a credible source and are gone, but
this is the absence of the symptom, not an attributed fix. The open item stays open.

## Where the time is after E2

| component (per request, profiled) | time |
|---|---:|
| attention custom op, inclusive | 1.256 s (1.160 s of it waiting for the kernel) |
| `set_rows`, inclusive | 0.309 s |
| of which `write_page` | 0.214 s (`tier_read` 0.086, `allocation` 0.053, `tier_write` 0.040, `release` 0.030, `checksum` 0.014) |
| of which row gather | 0.096 s |

Attention is now ≈4x everything else ARGUS does on this path. The two candidates:

1. **Attention, in-kernel.** E1's exit state stands: the L1TEX data pipe at 67%, now
   60% shared-memory wavefronts — q/weight/pointer `LDS` 12.0 M and the K tile
   round-trip 9.4 M — with `mio_throttle` 1.51 the largest throttle. The q re-read is
   the biggest single piece of that 12.0 M: every lane reads all 64 query floats from
   shared memory on every tile, 64 broadcast `LDS` per warp per tile, because 64
   floats cannot stay in registers at 66 registers per thread. Whether vectorising
   those reads (`LDS.128` over four contiguous q values) cuts the instruction count
   without paying it back in registers is a measurable question and the natural E3.
2. **The page allocator**, worth ≈0.07 s: 12,048 4 KiB `cudaMalloc`/`cudaFree` pairs
   against 0.5 ms for the same pages carved from 4 MiB slabs. It makes a page's device
   memory shared with its neighbours', so a slab can be held alive by one live page —
   a VRAM trade that needs its own experiment and its own budget accounting.

## E3 candidate (2026-09-23): Q shared-load vectorization — REJECTED at the compiler gate

Report: [`v070-e3-q-lds-sass-2026-09-23.md`](../docs/measurements/v070-e3-q-lds-sass-2026-09-23.md).
`b0255fe` is the accepted checkpoint: GPU-control 2.406 s, 1.79x stock, decode
8.08 tok/s, policy-on 9.178 s, hash `a152ed56…`. M1 and M2 met; next is M3 < 2.25 s.

The candidate was a `cells-kq` path whose only change is vectorizing the query reads
out of `query_tile`. The gate before any benchmark was to prove in SASS that the Q
loads become a wider, fewer `LDS` form. They already are one:

- Lines 318–319 compile to **16 `LDS.128` per tile**, offsets `0x00`–`0xf0` in 16-byte
  steps from a warp-uniform base: the whole 256-byte query row, every float loaded
  exactly once, at the widest shared load sm_86 has, naturally aligned.
- Every `LDS` in the kernel is already `.128`; the only narrow shared ops are four
  `STS.64` and the once-per-row `query_tile` fill, and widening them is wavefront-
  neutral and is not a Q change.
- Q reads are warp-uniform broadcasts, so structurally conflict-free; E1's 0.37 M bank
  conflicts belong to the K tile swizzle.
- Q is 16 of the 48 warp-uniform `LDS.128` per tile, so ≈4.0 M of the 12.0 M
  "q/weight/pointer" wavefronts — ≈11% of the kernel's 35.9 M data-pipe wavefronts.
  The V loop's weight and row-pointer broadcasts are twice that.

Even a hypothetical complete removal of Q's shared traffic maps, through E1's measured
transfer function, to ≈−3% prefill; the vectorization actually proposed is worth zero.
The only remaining levers on Q — caching the row in registers (64 floats against a
66-register budget) or consuming more cells per read (a loop restructure) — are
excluded by the single-factor rule and, for the first, by arithmetic.

No replacement experiment is selected in this turn, as instructed.

## Possible next map (2026-09-23, after `231df30`)

Not a selected experiment — the candidate list with what each is worth, what evidence
it already has and what it still needs. Nothing here is committed to.

### Where the 2.406 s stands

| component (per request, profiled) | time | note |
|---|---:|---|
| attention, inclusive | 1.256 s | 1.160 s of it is the host waiting on the kernel |
| `set_rows`, inclusive | 0.309 s | `write_page` 0.214 + row gather 0.096 |
| everything else ARGUS | ≈0.03 s | page table, descriptor scan, copies |
| stock-host, whole prefill | 1.342 s | its FlashAttention alone is 0.073 s |

Kernel interior, still valid because the kernel has not changed since E1: 35.9 M
L1TEX data-pipe wavefronts at 67.2%, 14.5 M global and 21.4 M shared, `mio_throttle`
1.51 the largest throttle, long scoreboard 2.24, issue 0.63, 66 registers,
7 blocks/SM, 49.1% achieved occupancy, 100 KB shared carveout leaving L1 at 28 KB.

Per-tile instruction census (this turn's SASS work) with the E1 wavefront totals
distributed across it:

| traffic | per tile | ≈ wavefronts | share of 35.9 M |
|---|---|---:|---:|
| K tile shared round trip (302/307) | 8 `STS.128` + 8 `LDS.128`, lane-distinct | 9.4 M | 26% |
| V row loads (`__ldg` of one `half`, 397) | 64 `LDG`, 2 per cell per lane | ≈6.9 M | 19% |
| K row global loads (302) | 8 `LDG.E.128` | ≈6.9 M | 19% |
| Q broadcasts (318–319) | 16 `LDS.128` | ≈4.0 M | 11% |
| V-loop weight broadcasts (393) | 16 `LDS.128` | ≈4.0 M | 11% |
| V-loop pointer broadcasts (394) | 16 `LDS.128` | ≈4.0 M | 11% |
| mask, `STS.64`, output | rest | ≈0.7 M | 2% |

E1's transfer function, for sizing anything below: −23% data-pipe wavefronts gave
−12% kernel and −6.7% prefill, so roughly **0.29% of prefill per 1% of wavefronts**.

### Candidates, strongest first

1. **V values as `half2`: let each lane own two adjacent output dims.** Today a lane
   loads `value_row + lane` and `value_row + lane + 32` as two separate 2-byte `__ldg`,
   64 `LDG` per tile. If lane *l* owned dims `2l` and `2l+1` it would load one
   `half2`, 32 `LDG` per tile, each still one fully used wavefront. Estimated −3.4 M
   wavefronts (−9.5%) → ≈−5% kernel → ≈−2.8% prefill, plus 32 fewer issued
   instructions per tile against the top throttle, which the linear estimate does not
   capture. **Category 1**: every dim keeps its own `acc` chain, advanced cell by cell
   with the same `__fmaf_rn` and the same weights; only which lane holds which dim
   changes, and the final `output[row * D + …]` index changes with it (and can become
   a 64-bit store). Needs: a SASS gate that nvcc actually emits one 32-bit `LDG` per
   pair, and an exactness argument written out before any benchmark.
2. **The page allocator**, ≈0.07 s (−2.9%). Measured, low risk, no kernel involvement:
   12,048 4 KiB `cudaMalloc`/`cudaFree` pairs against 0.5 ms for the same pages carved
   from 4 MiB slabs. The cost is that a page's device memory becomes shared with its
   neighbours', so a slab can be held alive by one live page — it needs its own VRAM
   accounting and a bound on retention. Deferred on instruction, not on merit.
3. **The row gather in `set_rows`**, 0.096 s. Per-row `from_float` into the staging
   buffer. Worth looking at only after it is broken down; it is an upper bound, not an
   opportunity.
4. **Occupancy step 7 → 8 blocks/SM.** Shared (11 KiB) already allows 9 blocks; the
   limiter is exactly the two registers above 64. `__launch_bounds__(128, 8)` was
   tried in E1 and spilled. Any retry is a lottery on nvcc's allocator, so it is only
   worth attaching to another change that happens to free registers, never as an
   experiment of its own.
5. **Weight and pointer broadcasts via `__shfl` instead of shared** (8.0 M together).
   Removes data-pipe wavefronts but replaces 16 `LDS.128` with ~64 `SHFL` per tile and
   pushes the same warp onto the unit `mio_throttle` already names. Likely a loss;
   listed so it is not re-derived from scratch.

### What the map does not reach

M3 (< 2.25 s) needs −0.156 s and is reachable: candidate 2 plus candidate 1 is
≈0.14 s, plus whatever 3 yields. **M4 (< 2.00 s) needs −0.41 s and is not visible.**
At the current attribution it would have to come from attention, ≈−25% of kernel time,
from data-movement changes alone with the data pipe already down to 67%. Stock does
the same attention in 0.073 s against our 1.16 s because it uses tensor-core
FlashAttention; that 16x is the price of the Category 1 contract, and closing it is a
contract decision, not an optimization. Past M3, the honest options are an explicit
Category 2/3 experiment or a structural change that overlaps attention with the next
layer's writes — both larger than anything on this list.

### Open items carried forward

- The E1 multi-second outlier: not reproduced in 43 GPU-control prefills on
  2026-09-23, not attributed either. Still open.
- **Decode is untouched**: 8.08 tok/s against stock's 37.02, a 4.6x gap, far wider in
  relative terms than prefill's 1.79x. The prefill campaign was chosen first on
  purpose, but decode is what a user of this feature actually waits on, and no
  experiment in v0.7 has looked at it.

## E4 result (2026-09-25): V values as `half2` — ACCEPT, now the default

Report: [`v070-e4-v-half2-2026-09-25.md`](../docs/measurements/v070-e4-v-half2-2026-09-25.md).

- Category 1: only the lane that owns a dim changes (`2l, 2l+1` instead of `l, l+32`).
  One 32-bit `half2` `__ldg` replaces two 16-bit loads. Mutation-checked.
- SASS: 8 → 4 V loads per unrolled 4-cell group; reference kernels byte-identical to
  `4e63501`. 76 registers, 6 blocks/SM, no spill. `__launch_bounds__(128, 7)` (72
  registers plus an `LDL.64` per tile) was a +1.3% kernel loss; it is rejected and
  must not be retried.
- Kernel events −6.3% (5/5, ranges do not overlap). Profiler-off prefill −4.5%
  (2.442 → 2.332 s, 6/7). Accepted below the preferred 5% on the strength of the
  clean kernel signal and two runs that agree in direction; the call is stated in the
  report.
- Final baseline: GPU-control **2.328 s** (2.271–2.396), **1.73x** stock (1.342 s).
  Policy-on 8.976 s median, counters unchanged. Hash `a152ed56`.
- M3 (< 2.25 s) needs −0.078 s more.

The 2026-09-25 audits ([control](../docs/plans/control-audit.md),
[logic flow](../docs/plans/logic-flow-breakdown.md)) reorder what comes next. The
largest remaining levers are outside the kernel:
1. Decode goes through the staged Q=1 path.
2. The CPU-side `set_rows` round trip (1.05 s of idle GPU time).
3. Two policy logic faults in the product mode: a rewrite drops a page's promotion,
   and the LFU heat is never reset.

## E5 result (2026-09-25): single-token decode on the resident path — ACCEPT

Report: [`v070-e5-resident-decode-2026-09-25.md`](../docs/measurements/v070-e5-resident-decode-2026-09-25.md).

- `reject_q1` is removed, so decode now uses the resident kernel: one launch per layer,
  with pages read in place. Bit-exact with staged on every resident path (TDD).
- The equivalence test exposed a policy bug on the staged path: scratch was still
  alive while the policy observed. The scratch is now released before `observe`, and
  policy decisions are identical again at both budgets.
- GPU-control decode **7.53 → 39.31 tok/s** (stock 39.06): decode reaches stock parity.
- Policy-on decode ≈2.5x (6.1–6.7 → 14.3–18.5 tok/s). It is still disk-latency-bound
  because of L0.1.

## E6 result (2026-09-25): placement policy logic — two fixes, one withdrawn

Report: [`v070-e6-policy-logic-2026-09-25.md`](../docs/measurements/v070-e6-policy-logic-2026-09-25.md).

- **Rewrite keeps promotion** (audit L0.1). At 64 MiB, steady state: promotions
  12,768 → 0, cold pages 2,520 → 0, `read_bytes` −67%. Wall clock stays bound by the
  durable write.
- **Promotion no longer fills the attention scratch headroom** (new L0.3, found by
  measurement). At the tight budget (2 MiB GPU): promotions and demotions go from
  78,680 / 78,496 to 36 / 34, reads −30%, prefill −10–13%.
- The LFU "heat inversion" (audit L0.2) is withdrawn: it was wrong for full causal
  attention, and the proposed reset would have caused harm.

## E7 plan (2026-09-25): KV appends written on the GPU (audit L2.1)

Decision by the user: GPU-side `set_rows` in every mode, with the disk copy
maintained in the background. Two phases, each measured and committed on its own.

**Why.** nsys puts 1.05 s of idle GPU time in the 4K GPU-control prefill, dominated
by the CPU-side `set_rows` path:
1. the scheduler splits every layer to copy `K_cur`/`V_cur` D2H,
2. the CPU encodes F32 → F16,
3. the pages are uploaded,
4. every page is read back D2H for verification.

**E7a — GPU control.** A CUDA `set_rows` custom op, sharing the registered external
marker and dispatched by a kind tag in `op_params`:
- resolves each destination row to its GPU page (unwritten pages start as zeroed GPU pages),
- converts with `__float2half_rn` (RNE, the same rounding as ggml's CPU `from_float`),
- scatters the rows on the attention stream.

*Contract change, chosen by the user.* There is no transport, so the D2H
verification read-back goes. A GPU-written page's digest is pending until the first
host read (`read_page`) or flush computes it. Corruption of GPU memory between the
write and that first read is not detected. Attention never checked digests.

*Invariants.* Byte-identical page contents to the CPU path. The output hash stays
`a152ed56`. Revisions advance as before. Budgets are enforced, and allocation
failure aborts exactly like the CPU append.

*Tests first.* GPU-written bytes must equal the CPU encoding on random, subnormal,
overflow, ±0 and ±inf values. Rows are scattered and pages partially written; an
unwritten page must start at zeros. The node lands on the CUDA path only for
GPU-authoritative F16 stores.

*Acceptance.* GPU-control prefill ≥5% faster, hash unchanged, suites green.

**E7b — policy and disk modes (write-back).** Designed after E7a is measured.
Pages written on the GPU become dirty. A background flusher publishes the disk copy
with the existing double-slot and verification rules. Demotion and teardown wait
for the flush. Durability changes: after a crash, the disk copy can trail the GPU
by the unflushed pages.

## E7a result (2026-09-25): KV appends written on the GPU (GPU control) — ACCEPT

Report: [`v070-e7a-gpu-set-rows-2026-09-25.md`](../docs/measurements/v070-e7a-gpu-set-rows-2026-09-25.md).

- CUDA `set_rows` for GPU-control F16 stores: byte-identical RNE encoding
  (mutation-checked); the verification read-back is replaced by a digest that stays
  pending until the first host read.
- A/B (5/5): prefill **2.193 → 1.975 s (−9.9%)**, decode 39.6 → 43.4 tok/s.
- Final baseline under load: **2.027 s, 1.56x stock**; decode 43.1 tok/s, faster than
  stock's 38.0. **M3 met.** M4 was met only by the A/B median, so it is not claimed.
- The earlier claim that M4 could not be reached from data movement held for the
  kernel only; the write path was outside that estimate.

### Release gate (user, 2026-09-25): repeat the final baselines before pushing v0.7

On a quiet machine: stock-host vs GPU-control, 5 repeats, to confirm or refute M4
(< 2.00 s; the E7a A/B median was 1.975 s, the loaded baseline 2.027 s). Policy-on,
3 repeats, to see whether the 28.8 s outlier recurs without foreign IO pressure.

## E7b plan (2026-09-25): GPU appends with background disk write-back

**Scope.** Every CUDA disk store: policy-on and disk modes, i.e. any store with a GPU
tier budget, F16 KV and page-aligned rows.

**Write path.** The append resolves each row to a GPU page, obtained as follows:
- a page that is already GPU-resident is used as is;
- a written page resident elsewhere, or on disk only, has its verified bytes uploaded
  to a new GPU page — a promotion by write, which advances `placement_revision`;
- an unwritten page gets a zeroed GPU page.

If the GPU budget cannot hold the pages the call needs, the op copies the rows to the
host and takes the existing host append, unchanged. The store lock is held from row
resolution through the encode kernel's completion, so no host reader or flusher can
see a half-written page. The new revisions are then published and the pages marked
**dirty**, meaning the disk slot is older than the resident copy.

**Flusher.** One worker per store. Its stack is charged to staging, as with the
prefetch worker. For each dirty page, under the store lock, it:
1. copies the resident bytes,
2. computes the digest,
3. writes the inactive disk slot and reads it back for verification,
4. publishes the slot and digest, then clears the dirty flag.

Holding the lock serializes slot I/O with `write_page`, which also writes the inactive
slot. A worker error is kept and rethrown by the next storage call, as prefetch does.

**Readers and policy.**
- Readers already prefer the resident copy, so a dirty page is never read from its
  stale slot.
- Demoting a dirty page to disk flushes it synchronously first.
- Promoting it to another tier keeps it dirty.
- `clear(0)` drops dirty state together with the content.
- Teardown stops the worker before the store goes.
- An explicit `argus_disk_flush(tensor)` drains the dirty pages, for tests and
  for sync points.

**Durability contract (user decision).** After a crash, the disk copy can trail the
GPU copy by the dirty pages. Every published slot is still verified, and the previous
slot is kept until the new one verifies.

**Tests first.**
- A GPU append on a policy store is byte-exact with the CPU append; the page is
  GPU-resident and dirty.
- After a flush, demoting the page and reading it back returns the new bytes.
- Demoting without an explicit flush preserves the content.
- A short write during a flush leaves the page dirty and intact and surfaces the
  error; a retry succeeds.
- With a GPU budget too small for the pages, the append falls back to the host path.
- Policy equivalence between staged and mixed reads still holds.

**Acceptance.** Policy-on 4K prefill and decode improve, and the tight-budget path
does not regress (both same-binary A/B). Hash unchanged, suites green.

## E7b result (2026-09-25): GPU appends with background write-back — ACCEPT

Report: [`v070-e7b-write-back-2026-09-25.md`](../docs/measurements/v070-e7b-write-back-2026-09-25.md).

- Policy-on 4K: prefill **8.0–13.6 → 2.0–2.2 s**, decode **~19 → ~40 tok/s**.
  The product mode is now within a few percent of GPU control.
- Tight budget (2 MiB) unchanged: every append falls back to the host path there.
  GPU control unchanged.
- Measurement caught two design mistakes before acceptance: the flusher did its disk
  I/O under the store lock, and appends filled the attention scratch headroom.
- Contract: after a crash, disk can trail the GPU by the dirty pages. Every published
  slot is still verified. Policy off stays the reference host path.

### Release gate result (2026-09-26, `9206bba`)

Data: [`v070-release-gate-2026-09-26/`](../docs/measurements/v070-release-gate-2026-09-26/).
A first attempt was invalid: 3–5 `codebase-memory-mcp` re-index workers ran throughout
(load 7, stock itself hit a 3.1 s outlier). It is kept as `loaded-first-attempt-*`.
The valid run paused those indexers with SIGSTOP for its duration (`gate.sh`); load
stayed at the benchmark's own ~3 and IO pressure at 1%.

| mode | prefill median (range) | decode tok/s |
|---|---:|---:|
| stock-host | 1.328 s (1.326–1.346) | 37.4 |
| ARGUS GPU control | **2.000 s** (1.994–2.040) | **41.6** |
| ARGUS policy-on (3 repeats) | 2.160 s (2.146–2.201) | 38.7 |

- **1.51x stock prefill**, down from 2.09x at the v0.7 baseline. Decode is faster
  than stock in both ARGUS modes. Hash `a152ed56` on every ARGUS run.
- **M4 (< 2.00 s) is not met.** The median is exactly on the line: 3 of 5 runs were
  below it and the median was not. M3 (< 2.25 s) is met by GPU control and policy-on.
- The 28.8 s policy-on outlier did not recur without foreign IO pressure. The
  unattributed E1 outlier stays open, with no occurrence in this clean run.

## E8 (2026-09-26): GPU-control appends resolved on the device — ACCEPT

Data: [`v070-e8-2026-09-26/`](../docs/measurements/v070-e8-2026-09-26/).

The first append puts every page of a GPU-control store on the GPU, zeroed if unwritten,
and uploads one device table of page addresses. From then on the kernel reads each
row index on the device and resolves it through that table. The whole host round
trip is gone:
- the D2H copy of the indices and its sync,
- the per-call page resolution,
- the row-table allocation and upload,
- the closing sync.

What replaces it:
- **Host readers** (`read_page`, staged copies on another stream) wait for the
  store's write event.
- **Out-of-range rows.** The kernel sets a pinned error flag, and the next append
  refuses (tested, mutation-checked). A mutation of the event wait is
  timing-dependent, so it is covered by construction, not by a test.
- **Page table invalidation.** Any page replacement drops the table: host writes,
  `write_run`, the E7a path and `clear`.
- **Written flags.** Pages a GPU-control tensor spans are all marked written.
  Unwritten ones read as zero pages instead of null, which is numerically identical
  because masked cells never contribute.

Results:
- `set_rows` host exclusive time: 325 → 75 ms per request.
- GPU-control prefill A/B: 2.228 → 2.151 s (−3.5%), 5/5, hash `a152ed56`.
- This session was loaded by foreign desktop apps (load ≈ 5), so absolute times are
  above the release gate's 2.000 s.

## E9 (2026-09-26): GPU-control attention without a per-call wait — ACCEPT; M4 met

Data: [`v070-e9-2026-09-26/`](../docs/measurements/v070-e9-2026-09-26/).

Once E8 has built a GPU-control store's device page table, the default attention path
(`cells-v2`) reads K/V pages through that table directly. There is no per-call pointer
table to build, allocate or upload, and no stream wait at the end, so the next layer's
work queues behind the kernel.

Why that is safe:
- Nothing is borrowed. Any page replacement drops the table, and freeing GPU memory
  synchronizes the device first.
- The profiler still waits, but only when CUDA-event timing is requested.
- The reference paths keep the borrowing path.

Tests:
- A parity test: the table path and the borrowing path agree bit for bit on a
  non-blocking stream.
- A mutation that shifts the table by one page fails the test.
- Native 16 passed.

A/B against E8 (5/5, hash `a152ed56`):
- prefill 1.936 → 1.664 s (−14%);
- decode 43.6 → 55.9 tok/s.

**Final baseline** (MCP indexers paused, 5 repeats):

| mode | prefill median (range) | decode tok/s |
|---|---:|---:|
| stock-host | 1.350 s (1.333–1.366) | 37.5 |
| ARGUS GPU control | **1.679 s** (1.661–1.931) | **55.3** |
| ARGUS policy-on (3 repeats, path unchanged) | 2.177 s (2.159–2.182) | 38.4 |

- **M4 (< 2.00 s) is met.** All five runs are below the line; the median ratio is
  **1.24x** stock.
- The stretch goal (< 1.75 s) is met by the median, with 2 of 5 runs above it.
- Decode is 1.5x stock.
- Policy-on stores are not GPU-authoritative, so they keep the borrowing path. Giving
  them the same async read is the obvious next step for the product mode.

## E10 (2026-09-26): wait-free page-table attention for policy stores — ACCEPT

Data: [`v070-e10-2026-09-26/`](../docs/measurements/v070-e10-2026-09-26/). This is not a
copy of the GPU-control table: in a policy store, pages move between tiers while
attention kernels are still queued.

### Invariants extracted from the code

**1. Where a resident GPU page is allocated, freed or replaced**
- Promotion and demotion in `move_page`. The old copy is deleted after the new one
  is verified.
- The E7b append attaching a fresh GPU page to a written non-GPU page or an unwritten one.
- The E6 refresh failure in `write_page`, which deletes the copy.
- `clear(0)` and `release`, which delete every copy.
- In GPU control only, host `write_page`/`write_run` replace pages; policy stores
  never do that.

**2. How promotion/demotion relates to launches and stream order**
- Moves run on the host thread, inside `prepare`/`observe` around each attention
  call, between launches.
- A promoted page is filled and verified with synchronous copies before it becomes
  visible.
- Nothing orders a move with kernels already queued on the backend's non-blocking
  stream.

**3. Can a page be freed while an outstanding kernel still uses it?** No. Every free
is a `cudaFree`/`cudaFreeHost`, and on this driver both wait for all queued device
work, non-blocking streams included (`cudafree-sync-probe.cu`: about 12 ms while a
1 GiB copy was pending). `cudaMalloc` does not wait, and a legacy `cudaMemcpy` does
not order with the backend stream. This is the pre-existing device-wide
synchronization that E9 already relies on. Avoiding it needs a stream-ordered
allocator (`cudaFreeAsync`), i.e. an allocator redesign, and is out of scope.

**4. Lifetime of the mixed-resident cold scratch** One attention call. It is
reserved under the store locks, uploaded on the attention stream and freed after the
call's wait. The table path never uses it: any written page off the GPU sends that
call down the borrowing path.

**5. When CPU readers must see GPU writes** Policy appends are synchronous: E7b
finishes its encode under the store lock. So `read_page`, save and flush see them as
soon as the append returns. Two paths needed guarding:
- Host in-place page writes (the E6 refresh on the host append) are not ordered with
  queued reads. They now wait for the table's read event first.
- GPU control keeps E8's write event.

### Design

- **Table per policy store.** Each entry holds the page's GPU address or null.
- **Built only if everything fits.** The table exists only while every page, the table
  and the policy's scratch headroom fit the GPU budget together. It can then never
  displace a page, so placement decisions, promotion/demotion and budget accounting
  are those of the borrowing path.
- **Stream-ordered publication.** An entry changes only through `publish_entry`, a
  one-thread kernel on the attention stream, after every GPU-residency change:
  promotion, demotion, E7b attach, E6 refresh failure. Kernels queued earlier keep
  the entry they saw.
- **Residency gate at launch.** The host checks under the store lock that every
  written K/V page is on the GPU. Otherwise the call takes the borrowing path.
- **Unchanged pieces.** The 32-cell access history is recorded exactly as on the
  borrowing path, and `prepare`/`observe` are untouched.
- **What goes away.** No generation, epoch or refcount; no new global sync; no
  per-call table and no completion wait.
- **Profiling.** CUDA-event profiling keeps the borrowing path.

### Correctness gate

`check_policy_table`: every case runs with the kernel still queued behind a 512 MiB copy.
1. The table path is bit-exact with staged and with the borrowing path, and its
   per-page access-count deltas are identical.
2. Demote a page the queued kernel reads, then refill the freed memory with 0xff. The
   queued result stands, and the next call goes to the borrowing path (cold scratch).
3. Promote while another table read is queued. The next table read sees the
   published entry.
4. A GPU append is queued behind a table read. The read sees the old rows, and a host
   read right after the append sees the new ones.
5. Free the store with a table read still queued.

Mutations:
- no publication: fails, once the garbage page is kept alive so the promotion cannot
  land on the old address;
- no residency gate: fails;
- no access history: fails.

Suites: native 16 passed, Python 382 passed. The staged/mixed policy-semantics tests
at 4 MiB and 256 KiB are unchanged, because those stores never qualify for a table.

### Performance gate

Same `llama-server`; `libllama.so` switched between E9 and E10; profiler off;
MCP indexers paused; hash `a152ed56` everywhere.

| | E9 | E10 |
|---|---:|---:|
| policy-on prefill (5 pairs) | 2.138 s | **1.817 s (−15.0%, 5/5)** |
| policy-on decode | 38.2 tok/s | **48.4 tok/s** |
| GPU-control prefill (3 pairs) | 1.676 s | 1.671 s |

- Placement counters are identical in every pair: promotions 0, demotions 0,
  rejected 0, committed 12,768.
- `written_bytes`/`read_bytes` differ, and they are not placement counters. They
  measure how much the E7b background flusher published inside the request window.
  A shorter request flushes less inside the window: 8.4–9.4 MB against 13.5–13.7 MB.
- Every E10 attention call took the table path: 1512 calls, 0 cold pages.

### Final baseline after E10 (`2ec8f18`, MCP indexers paused)

| mode | prefill median (range) | decode tok/s |
|---|---:|---:|
| stock-host | 1.336 s (1.331–1.341) | 37.5 |
| ARGUS GPU control | **1.719 s** (1.666–1.759) | **55.6** |
| ARGUS policy-on (3 repeats) | **1.821 s** (1.814–1.826) | **48.2** |

- GPU control is 1.29x stock and policy-on 1.36x.
- Both ARGUS modes decode faster than stock.
- Placement counters: 0 promotions, 0 demotions, 12,768 commits.
- Hash `a152ed56` on every ARGUS run.
- M4 (< 2.00 s) is met by both modes.
- v0.7 stops here.

## v0.7.0 release (2026-09-26)

The optimization campaign is closed. Accepted final optimization: E10 `2ec8f18`.
Final-baseline checkpoint: `95c74fa`.

**Release audit.** The E10 lifetime solution covers every path that frees or
replaces a page:

| path | guard |
|---|---|
| `move_page` promotion/demotion | old copy freed via `cudaFree`/`cudaFreeHost` (waits for queued work); RAM copies never enter the table; then `publish_entry` |
| append attaching a fresh GPU page | policy: `publish_entry`; GPU control: table dropped; the replaced copy is never a GPU page |
| host `write_page` in-place refresh / failure | waits for the table's `read_event` before writing; on failure the copy is deleted and `publish_entry` runs |
| GPU-control host writes / `write_run` | GPU-control only; `cudaFree` waits; table dropped |
| `clear(0)` | copies and table freed through `cudaFree` (waits) |
| `release` / shutdown | flusher joined; copies and table freed through `cudaFree`; events destroyed (safe while recorded) |

**Release checks.**
- Test suites:
  - Python: 382 passed, 6 skipped.
  - Native, CUDA build: 18 passed, quantized-KV lifecycle included.
  - Native, CPU build: 11 passed, 7 CUDA-only skipped.
- Running the quantized-KV lifecycle checks for the first time on the CUDA build
  exposed a harness problem, not an ARGUS one:
  - With zero GPU layers, llama.cpp still offloaded host operations to the GPU,
    flash attention included (340 graph splits). The stock reference then differed
    from ARGUS's CPU attention by 0.0206 in the first layer.
  - The same checks pass with the current sources on the CPU build, and the CPU
    attention source is unchanged since v0.6.0.
  - The test now sets `op_offload = false` outside the CUDA-tier variant.
- Output hash is `a152ed56` in every ARGUS mode; stock is `007ddc77`.
- Placement counters:
  - policy on, 64 MiB: 0 promotions, 0 demotions, 0 rejections, 12,768 commits,
    0 cold pages;
  - GPU control: no policy activity, no disk bytes;
  - policy on, 2 MiB: 36 promotions, 34 demotions, 2.14 GB read — identical to
    E6/E7b.

**Performance evolution** (GPU-control prefill, 4K):

| checkpoint | prefill |
|---|---:|
| v0.6.0 baseline | 2.778 s |
| E1 | 2.709 s |
| E2 | 2.406 s |
| E4 | 2.328 s |
| E7a | ≈2.0 s |
| E9 | 1.68 s |
| final baseline | 1.719 s |

- Decode went from 8.0 to 55.6 tok/s.
- Policy on went from 9.2 s (E2) to 2.0 s (E7b) to 1.82 s (E10).

Methodology note: `codebase-memory-mcp` re-index workers were paused during every
final measurement. See the release-gate record.

v0.8 backlog: [`argus-v0.8.0.md`](argus-v0.8.0.md).
