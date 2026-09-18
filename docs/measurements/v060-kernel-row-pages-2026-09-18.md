# v0.6: resolve resident pages once per K/V row (4K)

Same workload, budgets and output hash as [the D=64 step](v060-kernel-d64-2026-09-18.md).

## Guarantee and change

Graph construction requires packed K/V: `nb[1] = D * 2` and `nb[2] = nb[1] * kv_heads`.
A (cell, kv_head) row therefore starts at `offset + (cell * kv_heads + kv_head) * 2D`.
When the view's in-page offset is a multiple of the row size `2D` (128 B for D=64, a
divisor of 4096), every row start is a multiple of 128 and the whole row lies in
one 4 KiB page. For those views the D=64 kernel loads the row's page pointer
once and reads its lanes with `__ldg` (`LDG.E.U16.CONSTANT`, global address space)
instead of pointer load → null branch → generic `LD.E.U16` for each element. Null
(unwritten) pages still multiply the query by `0.0f`. Unaligned views (the shifted
test view) keep the per-element path; D != 64 keeps the generic kernel.

Wider (`half2`) loads were rejected: lane `l` owns elements `l` and `l + 32` because
that ownership *is* the staged reduction tree; a `half2` load per lane would pair
`2l, 2l+1` and change the addition order.

Serial memory operations per unmasked cell: 9 → 5 (mask, K pointer, K row, V pointer,
V row; each row's two element loads issue together). D=64 row kernel: 34 registers,
648 SASS instructions (was 38 / 688).

## Result

| GPU-control, profiler off, 3 repeats | Prefill median (min–max) |
|---|---:|
| `0320456` D=64 | 5.514 s (5.492–5.521) |
| row pages | **5.178 s** (5.146–5.199) |

Profiled kernel events (one repeat each): 3.960 s → 3.609 s. Output hash unchanged;
GPU `5 passed`, CPU/context `5 passed` (the Qwen-geometry and unshifted D=64 cases use
the row path, the shifted view the element path).

[after](v060-kernel-row-pages-control-2026-09-18.json), [profiled](v060-kernel-row-pages-profiled-2026-09-18.json)

## Why only ~9% of kernel time

Removing four serial memory operations per cell should matter a lot if each warp were
latency bound. It did not. One wave keeps ~45 warps per SM (~11 per scheduler), and
the loop body is ~132 SASS instructions per cell. That makes roughly 11 × 130 ≈ 1.4K
issue cycles per cell step per scheduler, ≈ 0.9 µs at ~1.6 GHz, against the measured
≈ 1.1 µs per cell. Whole-prefill, ~3.5e11 warp instructions over 80 schedulers is
≈ 2.7 s at 1.6 GHz, against 3.6 s measured. The kernel now looks mostly
**instruction-issue bound**, with latency hidden by other warps. This fits the D=64
step giving 2.1x from fewer instructions and this step giving 9% from shorter chains.
It is an estimate from SASS and timing: Nsight Compute is not installed, so issue-slot
utilization was not measured.

Implication: next-cell register prefetch (adds instructions) and shared-memory K/V reuse
(removes only ~4 load instructions per cell) are not the next lever. The per-cell warp
reduction, lane-0 softmax and broadcasts (~130 instructions per 4 useful FMA
instructions) are.
