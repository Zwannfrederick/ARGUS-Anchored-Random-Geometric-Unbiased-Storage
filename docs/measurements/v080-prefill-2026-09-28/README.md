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
- **Open:** 21 GB of experts in 4.84 s would be 4.3 GB/s, above the 3.14 GB/s the census
  measured for PCIe host-to-device. Either not every expert crosses per batch, or the
  fixed part is not only transfer. The PCIe traffic has to be measured before layer-major
  prefill is designed.
