# ARGUS v0.7 — experimental deep optimization campaign

Branch `v0.7-dev`, based on `adddea7` (v0.6.0 release; the accepted kernel checkpoint is
`1ca5d0d`). v0.6 is the immutable reference. Exact by default: no fast math,
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
