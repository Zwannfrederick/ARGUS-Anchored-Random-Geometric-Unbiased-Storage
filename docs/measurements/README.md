# Measurements

Every file here is machine-generated. Do not hand-edit.

Each JSON file records the git commit and full environment that produced it. A
number that appears in `README.md` or `docs/architecture.md` must be traceable
to a file in this directory. If it is not, it is not a measurement.

## Regenerating

```bash
python scripts/record_baseline.py --output docs/measurements/baseline-<date>.json
python benchmarks/bench_native_runtime.py --json docs/measurements/native-<date>.json
python benchmarks/bench_jl_fidelity.py --json docs/measurements/jl-<date>.json
```

## Claim classes

Keep these separate. Reporting one as another is the failure mode this
directory exists to prevent.

| class | what it measures | what it does **not** say |
|---|---|---|
| `reconstruction` | fidelity of a codec's compress/decompress round-trip | anything about model output quality |
| `runtime` | wall-clock latency, throughput, memory | anything about accuracy |
| `downstream` | end-to-end model behavior (perplexity, retrieval) | — |

## Reading `git_dirty`

A snapshot with `"git_dirty": true` was taken on a modified working tree and
**cannot be reproduced from its commit alone**. Such snapshots are acceptable
during development but must not back a published claim; regenerate on a clean
tree before release.
