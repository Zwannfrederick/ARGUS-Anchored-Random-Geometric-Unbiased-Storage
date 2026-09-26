# v0.8 bandwidth census (2026-09-26)

The first measurement of v0.8: what each path between memory and compute actually
delivers on the development machine, before any algorithm is chosen.

- Machine: Intel i5-11300H (4 cores / 8 threads, 8 MiB L3), 31.9 GB RAM,
  RTX 3050 Ti Laptop (4 GB, driver 610.57.04), CUDA 13.3.
- Conditions: Android emulator closed, `codebase-memory-mcp` indexers paused for the
  whole run, no other GPU compute process. The commit, machine state and PCIe link state
  during the copies are in [`results/machine.txt`](results/machine.txt).
- Reproduce: `bash run.sh <llama.cpp dir with build-cuda> <output dir>` (≈ 1 minute).

| probe | source | result |
|---|---|---|
| PCIe copies | [`pcie.cu`](pcie.cu) | [`results/pcie.json`](results/pcie.json) |
| RAM streaming read | [`ram_bw.cpp`](ram_bw.cpp) | [`results/ram_bw.json`](results/ram_bw.json) |
| Decode attention, stock GGML flash attention | [`attn_bw.cpp`](attn_bw.cpp) | [`results/attn_cpu_4t.json`](results/attn_cpu_4t.json), [`attn_cpu_8t.json`](results/attn_cpu_8t.json), [`attn_cuda.json`](results/attn_cuda.json) |
| Model KV and MoE bytes per token | [`geometry.py`](geometry.py) | [`results/geometry.txt`](results/geometry.txt) |

## 1. The links

| link | measured | note |
|---|---:|---|
| GPU memory, read by attention (f16) | **184 GB/s** | 131K cells; near the card's peak |
| Host RAM, streaming read, 4 threads | **42.2 GB/s** | 1 thread 20.9; 8 threads no better than 4 |
| PCIe host → device, pinned, 256 MiB | **3.14 GB/s** | device → host 3.29; both at once 5.62 total |
| PCIe host → device, 4 KiB | 0.65 GB/s | small copies are latency-bound |

**The GPU link runs at PCIe Gen3 x4**, although the card supports x16 (read by
`nvidia-smi` during the copy). This is how the card is wired in this laptop, not a
power state. RAM is 13x faster than the link, GPU memory 59x.

**Consequence:** moving KV to the GPU is the worst use of bandwidth on this machine.
Reading 262K tokens of Qwen3.6 KV (5.37 GB f16) across the link would take 1.7 s per
decoded token. Only small things may cross it: queries, partial outputs, page summaries.

## 2. Decode attention (one full-attention layer of Qwen3.6-35B-A3B)

16 query heads, 2 KV heads, head dim 256; one query token; median of 10 calls.

| backend | KV type | 16K cells | 64K | 131K | effective KV read |
|---|---|---:|---:|---:|---:|
| CUDA | f16 | 0.20 ms | 0.75 ms | 1.46 ms | 184 GB/s |
| CUDA | q8_0 | 0.40 ms | 1.54 ms | 3.02 ms | 47 GB/s |
| CUDA | q4_0 | 0.53 ms | 2.01 ms | 4.01 ms | 19 GB/s |
| CPU, 4 threads | f16 | 7.3 ms | 32.9 ms | 54.3 ms | 4.9 GB/s |
| CPU, 8 threads | f16 | 5.0 ms | 26.1 ms | 51.1 ms | 5.2 GB/s |
| CPU, 8 threads | q8_0 | 5.6 ms | 22.4 ms | 44.9 ms | 3.2 GB/s |
| CPU, 8 threads | q4_0 | 6.1 ms | 24.4 ms | 55.2 ms | 1.4 GB/s |

CPU times moved between this run and an earlier one the same afternoon by up to 30% at
16K cells and 18% at 131K; GPU, link and RAM figures repeated within 2%.

**The CPU kernel reads every KV head once per query head.** In
`ggml_compute_forward_flash_attn_ext_f16_one_chunk` the outer loop runs over
(token, query head) and the inner loop over all KV cells of that head's KV head
(`ik2 = iq2 / rk2`). Qwen3.6 has 8 query heads per KV head, so the same bytes are read
8 times: 8 × 268 MB / 54.3 ms = **39.5 GB/s — the RAM limit**. The kernel is
bandwidth-bound on redundant reads, which is also why quantized KV is *not* faster on
the CPU: dequantization is repeated 8 times too.

On CUDA, quantized KV costs time (q8_0 2.1x, q4_0 2.8x of f16 at 131K): it buys
capacity only.

## 3. Bytes per decoded token

| model | growing-KV layers | KV / token (f16 · q8_0 · q4_0) | 262K tokens (f16 · q8_0 · q4_0) | weights | active experts / token |
|---|---:|---|---|---:|---:|
| Qwen3.6-35B-A3B | 10 of 40 | 20.0 · 10.6 · 5.6 KiB | 5.37 · 2.85 · 1.51 GB | 22.65 GB | 627 MB |
| gemma-4-26B-A4B | 5 global | 20.0 · 10.6 · 5.6 KiB | 5.37 · 2.85 · 1.51 GB | 16.93 GB | 897 MB |
| UI-Mate-9B | 8 of 32 | 32.0 · 17.0 · 9.0 KiB | 8.59 · 4.56 · 2.42 GB | 5.90 GB | dense |
| gemma-4-E4B | 4 global | 16.0 · 8.5 · 4.5 KiB | 4.29 · 2.28 · 1.21 GB | 4.96 GB | dense |

The q8_0 figure for Qwen3.6 (10.6 KiB) matches the value measured through Ollama on
2026-08-16. **At 262K the KV read per token is 4.5x (q8_0) to 8.6x (f16) the expert
weights read per token**: at long context, attention owns the RAM bandwidth.

## 4. What this predicts (arithmetic from the rates above, not a measurement)

Qwen3.6-35B-A3B, 262K tokens filled, attention of one decoded token (10 layers):

| arrangement | per token |
|---|---:|
| All KV on the GPU (f16) — does not fit 4 GB | 29 ms |
| Stock `-nkvo`: all KV in RAM, stock CPU kernel (f16) | ≈ 1.09 s |
| Copy the KV to the GPU every token (f16) | ≈ 1.71 s |
| Split, stock CPU kernel (≈ 1 GB of KV on the GPU, rest in RAM) | ≈ 0.88 s |
| Split, CPU kernel reading each KV head once, at RAM bandwidth (f16) | ≈ 0.10 s + 15 ms experts |
| Same, cold pages q8_0 (if dequantization keeps pace) | ≈ 0.05 s + 15 ms experts |

Two conclusions drive the v0.8 plan ([`plans/argus-v0.8.0.md`](../../../plans/argus-v0.8.0.md)):

1. **The CPU attention kernel is the bottleneck of long context on this machine**, not
   placement. Splitting work with the GPU gains little until the CPU side reads each
   KV head once; that alone is worth up to ~8x.
2. **Compression pays on the CPU only after that fix.** Today q8_0/q4_0 KV makes CPU
   attention no faster; once the kernel is bandwidth-bound, halving the bytes halves
   the time.

Still to measure: stock end to end with the context actually filled (the 2026-08-16
runs set a 262K context size; the attention cost above applies only once that many
tokens are in the cache), prefill time at long context, free VRAM next to the model,
and RAM bandwidth while expert weights are being read.
