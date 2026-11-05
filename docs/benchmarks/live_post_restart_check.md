# ARGUS P3 Eşik & Bellek Politikası Benchmark Raporu

| Konfigürasyon | Hedef Tok | Gerçek Tok | TTFT (ms) | Prefill (tok/s) | P3 Batch | MajFlt | SwapIn | RSS / PSS (MiB) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `live_baseline` | 78 | 78 | 36535.3 | **2.14** | 2 | 49953 | 1 | 16194 / 16186 |
| `live_baseline` | 680 | 680 | 54742.3 | **12.43** | 2 | 78098 | 0 | 16808 / 16800 |
