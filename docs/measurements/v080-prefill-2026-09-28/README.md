# Prefill against ubatch size (2026-09-28)

Why ARGUS tiers did not make prefill faster ([M1](../v080-m1-2026-09-27/README.md)): the
hypothesis was that prefill is paid per ubatch, for the expert weights that `--cpu-moe`
keeps in RAM and llama.cpp copies to the GPU for every batch, not for KV.

- Model: Qwen3.6-35B-A3B (UD-Q4_K_M), 16K filled context, ARGUS tiers
  (`ARGUS_KV_VRAM_BYTES=auto`, `ARGUS_KV_PREFILL=gpu`, `-nkvo`), 4 pinned threads.
- Configurations: [`configs.json`](configs.json) (`-b = -ub` = 512, 1024, 2048, 4096).
- Result: [`results/qwen36-16k-ubatch.json`](results/qwen36-16k-ubatch.json); machine state in
  [`results/machine.txt`](results/machine.txt). One run per point.

| ubatch | prefill | batches | per batch | decode tok/s | auto mirror budget | peak VRAM |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 168.7 s | 32 | 5.27 s | 23.2 | 1143 MiB | 2826 MiB |
| 1024 | 95.0 s | 16 | 5.94 s | 21.8 | 1063 MiB | 2876 MiB |
| 2048 | 55.7 s | 8 | 6.96 s | 22.7 | 919 MiB | 3046 MiB |
| 4096 | **35.7 s** | 4 | 8.93 s | 22.0 | 361 MiB | 3604 MiB |

**The hypothesis holds.** Time per batch fits a fixed part plus a part per token:
≈ 4.84 s + 1.01 ms × ubatch (least squares over the four points; each within 1.7%). So
prefill(N, u) ≈ (N / u) × 4.84 s + N × 1.01 ms, plus attention at long context.

- Doubling ubatch from 2048 to 4096 makes prefill 1.56x faster, and costs 558 MiB of the
  VRAM the mirror would otherwise get. Decode is the same here because 16K fits the mirror
  either way; at 64K+ that VRAM is decode speed. Choosing ubatch is a planner decision.
- The fixed 4.84 s is what a layer-major prefill (each layer's experts crossing PCIe once
  for the whole prompt) would remove. Projection for 128K (arithmetic, not a measurement):
  of the measured 540 s, 64 × 4.84 ≈ 310 s is the fixed part; the ~132 s per-token part and
the attention stay.
- **Where the fixed part goes (measured).** The same run repeated with `nvidia-smi dmon -s t`
  beside it ([`results/pcie-16k-ub2048.dmon.txt`](results/pcie-16k-ub2048.dmon.txt), prefill 55.7 s
  again): 125 GB crossed host-to-device during prefill, **15.6 GB per batch** — the experts
  the batch routes to, not all 20.7 GB — at up to 3.33 GB/s, the PCIe ceiling the census
  measured. 15.6 GB at ~3.2 GB/s is ≈ 4.9 s: the fixed part is expert transfer.
- **The link also idles.** The mean during prefill is 2.36 GB/s, with seconds at zero: the
  copies do not overlap the GPU's compute. About a quarter of the link time is unused.

## P3: layer-major prefill (measured)

`ARGUS_PREFILL=layers` runs every ubatch of a llama_decode call through one layer before the
next, so each layer's experts cross PCIe once per call instead of once per ubatch. The
expert cache borrows the KV mirrors' device memory while they are detached (VRAM shared by
phase); the mirrors are refilled from host KV afterwards. The server runs with `-b 131072`
so one call holds the whole prompt. Configurations: [`configs-layers.json`](configs-layers.json).

| context | batch-major (ub 2048) | layer-major (ub 2048) | speed-up | decode, batch → layer |
|---|---:|---:|---:|---:|
| 5.3K (ub 1024, one call) | 34.0 s | 18.0 s | 1.89x | 25.5 → 25.5, 48 greedy tokens identical |
| 16K | 55.9 s | 32.0 s | 1.75x | 22.8 → 23.1 |
| 32K | 114.3 s | **59.1 s** | **1.93x** | 21.27 → 20.93 |
| 64K | 242.2 s ¹ | **130.4 s** | **1.86x** | 12.31 ¹ → 12.21 |
| 128K | 540.3 s ¹ | not taken ² | — | 6.18 ¹ → 5.85 |

¹ From [M1](../v080-m1-2026-09-27/README.md) (same configuration, previous day).
² At 128K the automatic budget left 386.6 MiB, below the 498 MiB cache; prefill stayed
batch-major (the harness flags the run). The context's compute buffer still reserves ~500
MiB for llama.cpp's per-batch expert copies, which layer-major prefill never uses. Next
step: when P3 is on, run every prefill batch layer-major and reserve the compute buffer
without those copies, giving that memory to the cache and the mirrors.

- Results: `results/qwen36-layers.json` (16K/32K, before VRAM sharing: 32K decode fell to
  15.9 because the cache took mirror memory), `results/qwen36-shared-32k.json`,
  `results/qwen36-shared-long.json` (with VRAM sharing).
- Every layer-major run found the hidden fact; the harness requires the server log line
  `ARGUS P3: layer-major prefill of`.
