# v0.6: exact lane-per-cell resident attention kernel (4K)

Same workload, budgets and placement policy as the [checkpoint](v060-checkpoint-2026-09-18.md).
ARGUS output hash is unchanged (`a152ed56…`) in every run. The warp-per-cell D=64 kernel
(`attention_resident_batch<64, true>`) stays in the binary as control/reference
(`ARGUS_KV_ATTENTION_PATH=batched`). Its SASS, and that of the staged, direct and generic
kernels, is instruction-for-instruction identical to `da1b7c4`.

## Kernel and why it is bit-exact

`attention_resident_cells` keeps one warp per query row and processes 32-cell tiles:

1. **Scores in parallel.** Lane c scores cell c. The old kernel's lane `l` formed
   `t_l = p_l + p_{l+32}`, then shuffle strides 16, 8, 4, 2, 1. The new lane forms
   `t_i`, `t_{i+16}` and `t_i + t_{i+16}` for i < 16, then strides 8, 4, 2, 1 in
   registers. That is the same binary tree over the same `__fmul_rn` products. IEEE
   addition is commutative, so which operand came from the "lower" lane cannot change
   a bit. The score is the same `__fmaf_rn(dot, scale, bias)`. K rows are read as four
   16-byte loads; half→float conversion is exact.
2. **Running maximum as a warp prefix scan** (masked lanes contribute −∞). The value
   of an `fmaxf` fold is order-independent:
   - `fmaxf` ignores a NaN operand in either order;
   - the running maximum is never NaN;
   - only the sign of a zero maximum can differ, and the maximum is used only in
     `previous − next` and `score − next`, whose values are equal for ±0.

   `next = fmaxf(previous, score)` and `alpha/beta = expf(…)` then see equal inputs.
3. **Order-dependent arithmetic unchanged.** `sum = __fmaf_rn(alpha, sum, beta)` and
   `acc[d] = __fmaf_rn(alpha, acc[d], __fmul_rn(beta, v[d]))` advance cell by cell in
   the original order, skipping masked cells. Every lane runs the sum chain, so the
   final broadcast is replaced by identical computation.

Eligibility is unchanged from the row path (D=64, each view's in-page offset a
multiple of 128 B, so rows are 16-byte aligned). Other views use the previous kernels.

## Correctness

- New Qwen-geometry cases (14/2 heads, D=64, 64 tokens, 600 cells of which 500
  written): causal; scattered −∞ with a fully masked 32-cell tile and fully masked
  rows; live unwritten (null-page) cells; F32 bias; ×24 query (alpha/beta underflow);
  +0/−0 queries with identical key rows (tied and signed-zero scores).
- In every case staged == direct == batched == cells bitwise. The cell kernel is
  confirmed to run by a new always-on counter (`resident_*_cell_kernel`).
- The resident and mixed-residency loops (D = 48/64/256, shifted views, cold pages)
  also run `cells`.
- Deliberate mutations are caught: changing the tree pairing, and reversing the cell
  order of the sequential chains. Both fail on the first D=64 mixed case.
- GPU `7 passed`, CPU/context `5 passed`.

## Kernels

| Kernel | Registers | Shared | Local/stack | Static SASS | Blocks/SM | Waves at 224 blocks |
|---|---:|---:|---:|---:|---:|---:|
| warp-per-cell D=64 rows (`batched`) | 34 | 0 | 0 | 648 | 12 | 0.93 |
| lane-per-cell (`cells`) | 58 | 3 KiB | 0 | 792 | 8 | 1.40 |

Per 32-cell tile, the new loop is about 324 SASS instructions for the parallel scores,
~76 for the scan/expf and ~20 per cell for the sequential chains. That is ~33 per
cell against ~133 before, so roughly 9e10 dynamic warp instructions against 3.6e11.
These are estimates from SASS, not counters: Nsight Compute is unavailable
(`RmProfilingAdminOnly=1`).

## A/B: 4K, profiler disabled, 3 repeats each, same binary

| Mode | `batched` (row kernel) | `cells` | Change |
|---|---:|---:|---:|
| GPU-control prefill median (min–max) | 5.188 s (5.152–5.193) | **3.748 s** (3.683–3.755) | −27.8% |
| policy-on prefill median (min–max) | 12.248 s (12.104–12.281) | **10.978 s** (10.976–11.084) | −10.4% |

Profiled, one repeat each (CUDA events; not mixed with the above):

| GPU-control | `batched` | `cells` |
|---|---:|---:|
| Prefill kernel events | 3.615 s | 2.202 s |
| Launches | 1512 | 1512 (all lane-per-cell) |

Policy-on with `cells`: 1512/1512 invocations resident on the lane-per-cell kernel,
12048 cold pages staged, 0 payload D2D, disk read bytes and promotions/demotions
identical to before (148045824; 12768/0), kernel events 2.67 s.

Negative sub-experiment, not kept: `__launch_bounds__(128, 12)` cut the new kernel to
40 registers (one wave) but spilled 120 bytes to the stack. Kernel events went from
2.20 s to 2.77 s.

The ~4x instruction reduction bought 1.64x in kernel time. At ~1.9 GHz this puts
estimated issue utilization near 27%, so the kernel is probably no longer
issue-bound. Candidate limits for the next evaluation (unmeasured): the global V loads
inside the sequential chain and the 1.4 waves.

## New baseline (default `cells`; profiler disabled, 3 repeats, alternating order)

| Mode | Prefill median (min–max) | vs stock | Decode tok/s |
|---|---:|---:|---:|
| stock-host | 1.332 s (1.328–1.334) | 1.00x | 37.30 |
| GPU-control | 3.685 s (3.675–3.742) | 2.77x | 8.14 |
| policy-on | 10.989 s (10.877–11.009) | 8.25x | 6.34 |

Decode remains the unchanged staged Q=1 path; its rate is not attributed to this change.

JSON: [control A/B](v060-kernel-cells-control-2026-09-18.json),
[policy-on A/B](v060-kernel-cells-policy-2026-09-18.json),
[profiled](v060-kernel-cells-profiled-2026-09-18.json),
[baseline](v060-kernel-cells-baseline-2026-09-18.json).
