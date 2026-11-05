# ARGUS P3 Eşik & Bellek Politikası Benchmark Raporu

| Konfigürasyon | Hedef Tok | Gerçek Tok | TTFT (ms) | Prefill (tok/s) | P3 Batch | MajFlt | SwapIn | RSS / PSS (MiB) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `live_baseline` | 78 | 78 | 36567.8 | **2.14** | 0 | 391349 | 4 | 15260 / 15252 |
| `live_baseline` | 680 | 681 | 55218.3 | **12.35** | 2 | 53607 | 0 | 16608 / 16600 |
