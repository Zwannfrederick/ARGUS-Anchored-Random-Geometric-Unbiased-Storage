# v0.6: policy-on resident eligibility census (4K)

Placement policy is unchanged. Same workload as
[the datapath report](v060-datapath-2026-09-18.md): Qwen2.5-0.5B-Instruct Q4_K_M,
F16 KV, context 4096, 4016 input tokens, 16 generated, ubatch 64, 64 MiB GPU /
64 MiB pinned / 4 MiB staging budgets, RTX 3050 Ti Laptop. Output hash stays
`a152ed5676e2f9f4412d7aadffa7054cc438c021549ecb4f472a83c864acba0a`.

## Instrumentation

- Always on (atomic counters, published in stats): one rejection reason per
  invocation that misses the resident path, first failing check in order:
  `q1`, `forced_staged`, `alignment`, `gpu_table_budget`, `staging_budget`,
  `codec`, `nonresident_key_page`, `nonresident_value_page`; plus `accepted`.
- Profiling only (`ARGUS_KV_PROFILE`): a census of every K/V view page at
  eligibility time, before `record_access`: pages by tier, resident percentage,
  cold runs (written but not GPU-resident), hot/cold transitions, cold-page
  `access_count`, and distance behind the store's most recent page write.

K and V of all layers share one store, and V is written after K, so the "most
recent write" is V's last page: V cold-page distances are exact, K cold pages fall
into the `64+` bin by construction.

Profiler-off policy-on baseline on this binary: 46.152 s median prefill
(46.071–46.548, 3 repeats, [JSON](v060-census-policy-baseline-2026-09-18.json)).

The always-on counters have no measurable cost: GPU-control prefill
(profiler off, 3 repeats) was 12.461 s median before and 12.327 s after
([before](v060-census-control-before-2026-09-18.json), [after](v060-census-control-after-2026-09-18.json)).

## Result (profiled with `--profile cpu`, one measured request)

| Prefill census (warmup 1 / warmup 0) | [warm](v060-residency-census-warm-2026-09-18.json) | [cold start](v060-residency-census-cold-2026-09-18.json) |
|---|---:|---:|
| Invocations / rejected `nonresident_key_page` | 1512 / 1512 | 1512 / 1512 |
| Other rejection reasons | 0 | 0 |
| Invocations with cold K / cold V | 1512 / 1512 | 1512 / 1512 |
| Cold runs (per invocation per tensor) | 3024 (exactly 1) | 3024 (exactly 1) |
| Cold run length 2–3 / 4–7 / 8–15 | 48 / 2976 / 0 | 48 / 2208 / 768 |
| Page-visits GPU / disk / unwritten | 392880 / 12048 / 576 | 371904 / 15120 / 18480 |
| Page-visits pinned / pageable RAM | 0 / 0 | 0 / 0 |
| Resident % bins <50 / 50–90 / 90–95 / 95–99 | 0 / 192 / 192 / 1128 | 48 / 240 / 288 / 936 |
| V cold-page distance 0 / 1 / 2–3 / 4–7 | 1512 / 1512 / 3000 / 0 | 1512 / 1512 / 3000 / 1536 |
| Cold `access_count` 0 / 1 / 2–3 / 16+ | 0 / 0 / 0 / 12048 | 3072 / 6144 / 5904 / 0 |
| Max cold pages in one invocation | 16 (process lifetime) | 16 |

Decode: 360 invocations, all rejected as `q1` (by design).

## Interpretation

The research hypothesis was only partly right.

- Confirmed: residency is high (mostly 95–99%) and one single cold run per
  tensor rejects every prefill invocation. Budget, alignment and codec never reject.
- Confirmed: the cold run is the write frontier. In the warm request the V cold
  pages are exactly the four pages the current ubatch's `set_rows` just wrote
  (64 cells × 256 B = 4 pages; 3 in the final 48-token ubatch).
  `write_page` drops GPU residency on every write, so those pages are on disk.
- Refuted for the warm request: the `access_count >= 2` promotion threshold is
  not the cause. The cold pages there have counts ≥16: they were promoted in the
  warmup and invalidated by the rewrite.
- Cold start only: the previous ubatch's pages are still cold as well (count 1
  at the check, which precedes `record_access`), giving runs of up to 8 pages per
  tensor. Some just-written pages already have counts 2–3 because earlier views
  read them as unwritten padding.

Consequence for the datapath: at most 16 cold 4 KiB pages (64 KiB) block an
invocation whose other ~97% of pages are GPU-resident. A mixed pointer table that
stages only those pages keeps the batched kernel usable without any placement
change. This does not change or propose changing the policy.
