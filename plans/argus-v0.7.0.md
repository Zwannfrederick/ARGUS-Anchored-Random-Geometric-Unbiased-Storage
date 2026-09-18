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

## Pending measurements (need the user: installs and root)

1. **Nsight Systems** (not installed). Install with `sudo pacman -S nsight-systems`.
   CUDA/OS-runtime tracing needs no root. Then run one GPU-control and one stock-host
   capture of the same workload. Goal: split H/I and stock's 1.33 s into GPU kernels,
   copies, CPU ops, split boundaries and idle gaps.
2. **Experiment 0: memory-pressure discrimination (Nsight Compute, root).** Same late
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
