# v0.6: branch-free value loop in the lane-per-cell kernel (4K)

One causal factor changed: the control flow of the 32-cell sequential value loop in
`attention_resident_cells`. Arithmetic, cell order, the score phase, K path, shared
memory layout, registers and placement are unchanged. Same 4K workload as the
[lane-per-cell report](v060-kernel-cells-2026-09-18.md). ARGUS output hash is unchanged
(`a152ed56…`). Chosen from the [Nsight Compute attribution](v060-ncu-cells-2026-09-18.md).

## Change and exactness

Before (`78812c1`, still available as `ARGUS_KV_ATTENTION_PATH=cells`), each masked cell
was skipped with `continue`. NVCC compiled every unrolled cell as its own block:

`LDS` row pointer → `LDG` value → consume → branch

That left one cell's value load in flight per warp.

Now (`attention_resident_cells<true>`, the default, also `cells-mlp`):

- Masked lanes publish a null value row and `(0, 0)` weights in the parallel phase.
- The sequential loop has no per-cell branch. Each cell computes its `sum`/`acc`
  candidates, and `live ? next : old` keeps the old bits for masked cells.
- NVCC emitted these as **predicated** FFMAs (`@P FFMA`). A masked cell performs no
  arithmetic on `sum`/`acc` at all; no `fma(1, acc, 0)`-style substitution exists.
- Live cells run the same `__fmaf_rn`/`__fmul_rn` operations in the same order. The
  chains are true data dependencies through one register each (`sum` R40, `acc`
  R41/R42), so the hardware cannot reorder them.
- Null (unwritten) pages still read `0.0f`.

SASS of one 4-cell unrolled group (new kernel):

| | Address |
|---|---|
| 8 value loads `LDG.E.U16.CONSTANT` (4 cells × 2) | 0x2250–0x23c0 |
| first consumer `HADD2.F32` | 0x24d0 |

All four cells' loads are in flight before the first is consumed (before: loads at
0x2290 → consumer 0x22e0, then the next cell).

| Kernel | Registers | Stack / local / spill | Shared | Static SASS |
|---|---:|---:|---:|---:|
| `cells` (`78812c1`) | 58 | 0 / 0 / 0 | 3 KiB | 792 |
| `cells-mlp` | 58 | 0 / 0 / 0 | 3 KiB | 792 |

The staged, direct, both batched and the `cells` kernels' SASS is unchanged
(normalized diff 0 lines).

**Correctness.**

- Every existing exact case also runs `cells-mlp`: the resident and mixed-residency
  loops (D = 48/64/256, shifted views, cold pages), and the Qwen 14/2/D64/Q64 cases
  (causal, scattered −∞, fully masked tile and rows, live null pages, F32 bias, ×24
  scores, ±0 queries with tied keys). All bit-exact with staged.
- Two deliberate mutations fail: reversed cell order, and dropping the select so
  masked cells update `acc`.
- GPU `7 passed`, CPU/context `5 passed`.

## A/B: same binary, profiler disabled, 3 repeats each

| Mode | `cells` | `cells-mlp` | Change |
|---|---:|---:|---:|
| GPU-control prefill median (min–max) | 3.673 s (3.664–3.686) | **2.825 s** (2.815–2.847) | −23.1% |
| policy-on prefill median (min–max) | 11.091 s (11.051–11.388) | **9.891 s** (9.802–13.079) | −10.8% |

One policy-on repeat had a 13.08 s outlier; the median and the other two repeats are
9.80–9.89 s. Profiled policy-on kernel events: 2.67 s (`cells`) → 1.66 s.

## Nsight Compute: same late invocation (ubatch 60, layer 0; profiler-only durations)

| Metric | `cells` (P1) | `cells-mlp` (P7) |
|---|---:|---:|
| Duration (base clock) | 3.642 ms | **2.082 ms** |
| Warp instructions | 135.6 M | 129.4 M (−4.5%) |
| Registers / theoretical occupancy / waves | 58 / 66.7% / 1.40 | 58 / 66.7% / 1.40 |
| Achieved occupancy / active warps per scheduler | 50.2% / 5.92 | 54.7% / 6.61 |
| Eligible warps / scheduler / cycle | 0.60 | **1.63** |
| Issued / scheduler / cycle | 0.33 | **0.57** |
| Cycles between issues per warp | 18.1 | 11.6 |
| of which long_scoreboard | 12.0 | **3.6** |
| short_scoreboard / wait | 2.0 / 1.6 | 1.4 / 0.9 |
| not_selected / math_pipe_throttle | 0.84 / 0.43 | 1.87 / 0.80 |
| mio_throttle / lg_throttle | 0.61 / 0.12 | **1.56 / 0.36** |
| Long-scoreboard share of samples | 69.1% | **33.4%** |
| Value-consumer `HADD2` share of all samples | 56.9% | **21.2%** |
| Local-memory spill instructions | 0 | 0 |
| L1 sector hit rate | 94.0% | 70.2% |
| L2 sector hit rate / L2 lookup misses | 93.3% / 140 k sectors | 95.4% / 144 k sectors |
| Compute-memory access throughput (% peak) | 43.7% | 75.9% |

In `cells-mlp` the remaining value-load wait sits almost entirely on the **first** consumer
pair of each 4-cell group (`0x24d0`/`0x24e0`: 59% of long-scoreboard samples). The
next three cells' consumers barely stall, because their loads have landed.

The time reduction (1.75x) comes with only 4.5% fewer instructions and an unchanged
register count. It is explained by the hidden value-load latency, which is what the
experiment targeted:

- eligible warps 0.60 → 1.63;
- issued per cycle 0.33 → 0.57;
- long-scoreboard cycles 12.0 → 3.6 per issue.

New pressure appears in the LSU/L1 path: mio_throttle and lg_throttle roughly triple,
the memory-access throughput reaches ~76% of peak, and the L1 sector hit rate falls to
70%. L2 misses (DRAM traffic) are unchanged. The L1 hit drop fits several warps now
requesting the same lines while they are still in flight. That reading is an
inference: the reduced capture has no per-level utilization breakdown. Memory is
not saturated, but it is no longer idle. The next capture should say which unit
(L1 data, tag or LSU) is at 76% before any further value-path change.

## Final baseline (default `cells-mlp`; profiler disabled, 3 repeats, alternating order)

| Mode | Prefill median (min–max) | vs stock | Decode tok/s |
|---|---:|---:|---:|
| stock-host | 1.332 s (1.326–1.335) | 1.00x | 37.31 |
| GPU-control | **2.821 s** (2.779–2.873) | 2.12x | 8.00 |
| policy-on | **9.888 s** (9.779–9.929) | 7.42x | 6.37 |

Decode is the unchanged staged Q=1 path; no decode change is claimed.

Policy-on physical path (profiled, one repeat): 1512/1512 invocations resident on the
lane-per-cell kernel, 12048 cold pages staged, payload D2D 0, H2D 52,592,640 B, disk read
148,045,824 B / write 49,348,608 B, 1512 explicit syncs, promotions/demotions 12768/0.
These are identical to the mixed-resident and `78812c1` measurements. Kernel events 1.66 s.

## Progress on this workload (profiler-disabled medians)

| Checkpoint | GPU-control prefill | policy-on prefill | Source |
|---|---:|---:|---|
| staged scalar (forced, `579ba21` binary) | 40.908 s | — | [datapath](v060-datapath-2026-09-18.md) |
| first batched resident `579ba21` | 12.760 s | 48.319 s | [datapath](v060-datapath-2026-09-18.md) |
| D=64 specialized `0320456` | 5.514 s | — | [d64](v060-kernel-d64-2026-09-18.md) |
| row pages `de4bc58` | 5.178 s | — | [rows](v060-kernel-row-pages-2026-09-18.md) |
| mixed resident `ff1ca2d` / checkpoint `da1b7c4` | 5.217 s | 12.271 s | [checkpoint](v060-checkpoint-2026-09-18.md) |
| lane-per-cell `78812c1` | 3.685 s | 10.989 s | [cells](v060-kernel-cells-2026-09-18.md) |
| **branch-free value loop (this)** | **2.821 s** | **9.888 s** | this report |
| stock-host | 1.332 s | | this report |

GPU-control is 14.5x faster than forced staged and 4.5x faster than the first batched
resident path. Policy-on is 4.9x faster than its first measurement. Rows from different
sessions carry run-to-run drift of a few percent.

JSON: [A/B control](v060-kernel-cells-mlp-control-2026-09-18.json),
[A/B policy-on](v060-kernel-cells-mlp-policy-2026-09-18.json),
[baseline](v060-kernel-cells-mlp-baseline-2026-09-18.json),
[policy-on profiled](v060-kernel-cells-mlp-policy-profiled-2026-09-18.json);
Nsight Compute report `v060-ncu-cells-2026-09-18/p7-cells-mlp-late.ncu-repz`.
