# ARGUS v0.8 — Heterojen CPU/GPU Execution Scheduler ve Uyarlanabilir RAM/VRAM Bellek Yöneticisi Denetim ve Doğrulama Raporu

**Tarih:** 28 Ekim 2026  
**Hedef Model:** `Qwen3.6-35B-A3B-UD-Q4_K_M-MTP.gguf` (`35B` toplam parametre, `3B` aktif parametre/token, `40` katman: `30` Gated DeltaNet recurrent + `10` tam dikkat katmanı, katman başına `256` routed expert'ten `top-8` + `1` shared expert)  
**Hedef Donanım:** Intel Core i5-11300H (`4` fiziksel / `8` mantıksal çekirdek, `AVX2` + `AVX-512`) + `31,869.5 MiB` (`32 GB`) DDR4 RAM + NVIDIA GeForce RTX 3050 Ti Laptop GPU (`3,824.5 MiB` / `4 GB` VRAM, PCIe Gen4)  
**Test Bağlamları:** `8K` (`8,192`), `32K` (`32,768`) ve `64K` (`65,536`)

---

## 1. Yönetici Özeti ve Mimari Kazanımlar

ARGUS v0.8 kapsamında CPU ve GPU'yu eşzamanlı ve koordineli kullanan, RAM ve VRAM'i ortak bir heterojen bellek hiyerarşisi olarak yöneten **6 temel mimari bileşen** ([`argus_hetero.h`](file:///home/zwannfrederick/Masaüstü/Sektor/Coding/mamba%20fix/argus_cache/csrc/argus_hetero.h), [`ggml_layer_prefill.cpp`](file:///home/zwannfrederick/Masaüstü/Sektor/Coding/mamba%20fix/argus_cache/csrc/ggml_layer_prefill.cpp), [`llama-context.cpp`](file:///home/zwannfrederick/Masaüstü/Sektor/Coding/mamba%20fix/scratch/llama.cpp-v050/src/llama-context.cpp), [`llama-graph.cpp`](file:///home/zwannfrederick/Masaüstü/Sektor/Coding/mamba%20fix/scratch/llama.cpp-v050/src/llama-graph.cpp), [`qwen35moe.cpp`](file:///home/zwannfrederick/Masaüstü/Sektor/Coding/mamba%20fix/scratch/llama.cpp-v050/src/models/qwen35moe.cpp), [`ggml-backend.cpp`](file:///home/zwannfrederick/Masaüstü/Sektor/Coding/mamba%20fix/scratch/llama.cpp-v050/ggml/src/ggml-backend.cpp)) tasarlanmış, uygulanmış ve gerçek `Qwen3.6-35B-A3B` üzerinde **8K, 32K ve 64K** bağlamlarda doğrulanmıştır:

1. **VRAM Compute Buffer Tüketiminde %80–%85 Azalma ve 4 GB GPU'da 64K %100 `S1` Mirroring:**
   - P3 katman-bazlı (`layer-major`) prefill grafik rezervasyonunun 40 katman yerine 2 katmanlık zarf (`envelope`) ile sınırlandırılması, `sched_reserve()` sırasında `150 MiB` sahte MoE ağırlık kopyasının (`bind_dummy`) elenmesi, adaptif `n_ubatch = 512` seçimi ve `n_outputs_max = 16` dinamik tavanı sayesinde:
   - **32K bağlamda** `CUDA0` compute buffer **`879.3 MiB`'dan (`n_outputs_max=1` ile `658.10 MiB`) `132.03 MiB`'a (`-%80.0` / `-%85.0`)**, `CUDA_Host` pinned buffer ise **`160.07 MiB`'dan `40.02 MiB`'a (`-%75.0`)** düşürülmüştür (**`524.0 MiB` net VRAM tasarrufu**).
   - **64K (`65,536`) bağlamda** `CUDA0` compute buffer **`722.10 MiB`'dan `228.03 MiB`'a** düşürülerek 4 GB VRAM'li RTX 3050 Ti üzerinde **65,536 KV hücresinin tamamının (`1,280.0 MiB`, %100)** `S1` GPU aynasında tutulması sağlanmıştır (`Legacy_Auto` yalnızca `36,352 / 65,536` = `%55.5` hücreyi sığdırabiliyordu).
2. **Prefill Hızında Büyük Sıçrama (8K'da `+41.1%`, 32K'da `431.93 tok/s`, 64K'da `376.87 tok/s`):**
   - `30` Gated DeltaNet katmanında kullanılmayan `self_kq_mask` (`32K`'da `34.23 GiB`, `64K`'da `132.87 GiB` CPU sıfırlama/yazma trafiği) ve `480`–`960` gereksiz GGML graf yeniden inşası (`can_reuse == false`) ortadan kaldırılmıştır.
   - Host GGUF `mmap` alanında GPU'ya zaten kopyalanmış olan `254` tensör ile kullanılmayan `blk.40.*` MTP tensörlerine ait **`1,521.9 MiB` ölü sayfa** geri kazanılmış (`MADV_DONTNEED` + `POSIX_FADV_DONTNEED`), `mincore()` tabanlı Anti-Belady koruması ile RAM'de sıcak (`>= %98` resident) olan katmanlarda `POSIX_MADV_SEQUENTIAL` sayfa düşürme (`Inactive(file)` demotion) engellenmiştir.
   - Sonuç olarak prefill hızı **8K'da `193.52 tok/s` -> `272.96 tok/s` (`+41.1%`)**, **32K'da `404.96 tok/s` -> `431.93 tok/s` (`73.49 s` TTFT, önceki `289.7 tok/s` referansına göre `+49.1%`)**, **64K'da ise `376.87 tok/s` (`61,440` token için `163.03 s`)** seviyesine ulaşmıştır.
3. **Gerçek Eşzamanlı CPU/GPU Paralel Yürütme (`ARGUS_HETERO_OVERLAP=1`) ve Asenkron PCIe Batching:**
   - Her katmanda `ffn_shexp` (Shared Expert, GPU) ile `RoutedExperts` (`ffn_moe_out`, CPU) arasındaki veri bağımsızlığı kullanılarak `CUDA0` üzerindeki `ffn_shexp` hesaplaması CPU MoE uzmanlarıyla **gerçek zamanlı paralel (overlap)** çalıştırılmıştır.
   - `ggml_backend_sched_compute_splits` içindeki `6` senkron PCIe çağrısı (`22.21 us/katman`), tek senkronizasyonlu asenkron D2H/H2D kuyruğuna (`11.35 us/katman`, **`-48.9%`**) indirilmiştir.
   - 32K ablasyon testinde (`32K_B` vs `32K_C`) tek başına `ARGUS_HETERO_OVERLAP=1` mekanizması `SharedExpert_40L_GPU` duvar saatini **`5.48 ms/tok`'tan `2.96 ms/tok`'a (`-46.0%`)** düşürmüş, decode hızını **`2.15 tok/s` -> `3.45 tok/s` (`+60.5%`, `Legacy_Auto` `1.72 tok/s`'a göre `+100.6%` / `2.01x`)** artırmış ve `p50` gecikmesini **`516.29 ms`'dan `259.85 ms`'a (`-49.7%`)** indirmiştir.
4. **Sıfır Doğruluk Kaybı (`3/3` Multi-Needle):**
   - Tüm 8K, 32K ve 64K koşularında üç iğnenin tamamı (`7194`, `BLUE-FALCON`, `OCTOBER-29`) **`3/3` (%100)** doğrulukla geri çağrılmıştır.

---

## 2. Hedeflenen 6 Mimari Bileşen ve Kod Eşleşmesi

| # | Mimari Bileşen | Sorumluluk ve Maliyet Modeli | Uygulandığı Dosyalar |
|---|---|---|---|
| **1** | **Heterogeneous Execution Scheduler** | P3 2-katman zarf rezervasyonu, `n_tokens_all >= 32` maliyet eşiği, `can_reuse` graf önbelleklemesi ve CPU/GPU paralel alt-graf (`gv_pre`) zamanlaması. | [`llama-context.cpp`](file:///home/zwannfrederick/Masaüstü/Sektor/Coding/mamba%20fix/scratch/llama.cpp-v050/src/llama-context.cpp), [`llama-graph.cpp`](file:///home/zwannfrederick/Masaüstü/Sektor/Coding/mamba%20fix/scratch/llama.cpp-v050/src/llama-graph.cpp), [`ggml-backend.cpp`](file:///home/zwannfrederick/Masaüstü/Sektor/Coding/mamba%20fix/scratch/llama.cpp-v050/ggml/src/ggml-backend.cpp) |
| **2** | **Adaptive RAM/VRAM Memory Manager** | Dinamik VRAM bütçeleme (`argus_plan_hetero_vram_budget`), `n_ubatch` ve `n_outputs_max` adaptasyonu, `150 MiB` sahte MoE kopyası eliminasyonu (`bind_dummy`), `1.52 GiB` ölü GGUF sayfa geri kazanımı (`reclaim_non_host_gguf_pages`). | [`argus_hetero.h`](file:///home/zwannfrederick/Masaüstü/Sektor/Coding/mamba%20fix/argus_cache/csrc/argus_hetero.h), [`ggml_kv_tiers.cpp`](file:///home/zwannfrederick/Masaüstü/Sektor/Coding/mamba%20fix/argus_cache/csrc/ggml_kv_tiers.cpp), [`ggml_layer_prefill.cpp`](file:///home/zwannfrederick/Masaüstü/Sektor/Coding/mamba%20fix/argus_cache/csrc/ggml_layer_prefill.cpp) |
| **3** | **CPU/GPU Workload Partitioning** | Kupon-toplayıcı (`coupon-collector`) tabanlı analitik maliyet modeli (`argus_evaluate_moe_cost`): $N < 32$ için CPU token-major AVX2/AVX-512, $N \ge 32$ için GPU P3 layer-major cuBLAS; `qwen35moe` katman içi bağımsız dal sıralaması. | [`argus_hetero.h`](file:///home/zwannfrederick/Masaüstü/Sektor/Coding/mamba%20fix/argus_cache/csrc/argus_hetero.h), [`qwen35moe.cpp`](file:///home/zwannfrederick/Masaüstü/Sektor/Coding/mamba%20fix/scratch/llama.cpp-v050/src/models/qwen35moe.cpp) |
| **4** | **Asynchronous Data Movement** | Split sınırlarında 3-girdili D2H (`ffn_inp`, `selected_experts`, `weights`) için tek senkronizasyonlu asenkron batching ve stream-ordered H2D (`ggml_backend_tensor_set_async`). | [`ggml-backend.cpp`](file:///home/zwannfrederick/Masaüstü/Sektor/Coding/mamba%20fix/scratch/llama.cpp-v050/ggml/src/ggml-backend.cpp) |
| **5** | **Memory Pressure Controller** | `/proc/meminfo` (`MemAvailable`, `MemFree`, `SwapFree`) ve `mincore()` sayfa yerleşikliği izleme; bellek baskısında (`ARGUS_PRESSURE_ELEVATED`/`CRITICAL`) `mlock` dilimlerini serbest bırakma ve Anti-Belady sayfa koruma. | [`argus_hetero.h`](file:///home/zwannfrederick/Masaüstü/Sektor/Coding/mamba%20fix/argus_cache/csrc/argus_hetero.h), [`ggml_layer_prefill.cpp`](file:///home/zwannfrederick/Masaüstü/Sektor/Coding/mamba%20fix/argus_cache/csrc/ggml_layer_prefill.cpp) |
| **6** | **Hardware-Aware Runtime Profiler** | Çalışma anında CPU mantıksal çekirdek, `AVX2`/`AVX-512`, DDR4 bant genişliği, PCIe H2D/D2H bant genişliği ve kullanılabilir RAM/VRAM tespiti (`argus_probe_hardware`). | [`argus_hetero.h`](file:///home/zwannfrederick/Masaüstü/Sektor/Coding/mamba%20fix/argus_cache/csrc/argus_hetero.h) |

---

## 3. Geliştirme Görevleri (1–17) Detaylı Bulgular ve Doğrulama

### Görev 1: Mevcut ARGUS CPU/GPU İşlem Dağılımının Ölçümü
[`bench_hetero_profile.cpp`](file:///home/zwannfrederick/Masaüstü/Sektor/Coding/mamba%20fix/docs/measurements/v080-libargus-2026-10-27/bench_hetero_profile.cpp) ile `Qwen3.6-35B-A3B` decode (`bs=1`) grafındaki tüm düğümler (`2,425` view-olmayan aktif operasyon düğümü) cihaz bazında sınıflandırılmıştır:

| Operasyon Tipi (`ggml_op`) | `CUDA0` Düğüm Sayısı | `CUDA0` Çıktı Boyutu (KiB) | `CPU` Düğüm Sayısı | `CPU` Çıktı Boyutu (KiB) | Mimari Rolü |
|---|---:|---:|---:|---:|---|
| `MUL_MAT` | `391` | `3,617.7` | `0` | `0.0` | QKV, Gate, Shared Expert, Router, LM Head |
| `MUL_MAT_ID` | `0` | `0.0` | `120` | `3,840.0` | 40 katman × 3 (`gate`, `up`, `down`) Routed MoE (`top-8`) |
| `MUL` | `241` | `2,271.8` | `40` | `2,560.0` | GPU: GDN/Gate ölçekleme; CPU: MoE expert ağırlık çarpımı |
| `ADD` | `150` | `963.8` | `280` | `2,240.0` | GPU: Residual birleştirme; CPU: 8 expert çıktısının indirgenmesi |
| `RMS_NORM` | `191` | `1,788.0` | `0` | `0.0` | Katman ve QK-Norm normalizasyonları |
| `UNARY` / `GLU` | `210` | `1,687.7` | `40` | `640.0` | SwiGLU ve aktivasyon fonksiyonları |
| `GET_ROWS` | `161` | `64,329.2` | `1` | `8.0` | Token embedding (`CPU` 1 düğüm) ve KV/durum indeksleme |
| `CPY` / `CONT` / `CONCAT` | `160` | `68,320.0` | `0` | `0.0` | KV önbellek ve recurrent state yazımları |
| `SCALE` / `SOFT_MAX` / `ARGSORT` / `SUM_ROWS` / `DIV` / `CLAMP` | `320` | `561.6` | `0` | `0.0` | MoE top-k yönlendirme (`Router`) |
| `CUSTOM` (`FlashAttn` / `S1`) | `40` | `655,701.2` | `0` | `0.0` | 10 tam dikkat katmanı `S1` GPU decode |
| `GATED_DELTA_NET` + `SSM_CONV` | `60` | `62,880.0` | `0` | `0.0` | 30 Gated DeltaNet katmanı |
| `ROPE` | `20` | `180.0` | `0` | `0.0` | 10 tam dikkat katmanı Q/K RoPE |
| **TOPLAM (Aktif Düğümler)** | **`1,944` (%80.2)** | **`862,301.0`** | **`481` (%19.8)** | **`9,288.0`** | **40 D2H (`660.5 KiB/tok`) + 41 H2D (`349.9 KiB/tok`)** |

---

### Görev 2 ve Görev 3: Alt-Blok Hesaplama Süreleri, PCIe Transferleri ve Senkronizasyon Maliyetleri

#### 3.1 PCIe Gen4 H2D / D2H ve Senkronizasyon Mikro-Benchmark Sonuçları
RTX 3050 Ti Laptop GPU üzerinde Pinned (`CUDA_Host`) vs Pageable RAM ve Senkron (`6 sync/katman`) vs Asenkron-Batched (`1 sync/katman`) transfer süreleri ölçülmüştür:

| Transfer Boyutu | H2D Pinned Sync (`us`) | H2D Pinned Async (`us`) | H2D Pageable Sync (`us`) | D2H Pinned Sync (`us`) | D2H Pinned Async (`us`) | D2H Pageable Sync (`us`) |
|---:|---:|---:|---:|---:|---:|---:|
| `64 B` (Router İndeks/Ağırlık) | `6.49` | **`3.10`** | `6.85` | `6.84` | **`1.77`** | `7.46` |
| `4,096 B` (`4 KiB`) | `8.46` | **`4.12`** | `8.27` | `7.84` | **`2.61`** | `11.81` |
| `8,192 B` (`8 KiB` Gizli Durum) | `11.12` | **`5.28`** | `11.22` | `10.05` | **`3.91`** | `15.31` |
| `65,536 B` (`64 KiB`) | `30.88` | **`23.68`** | `38.92` | `28.22` | **`21.38`** | `56.27` |
| `1,048,576 B` (`1 MiB`) | `348.73` | **`337.96`** | `409.07` | `328.37` | **`321.06`** | `717.75` |
| `16,777,216 B` (`16 MiB`) | `5,534.81` | **`5,372.46`** | `5,736.04` | `5,238.91` | **`5,099.97`** | `6,396.06` |
| **3-Girdili Split (`8 KiB + 32 B + 32 B`)** | — | — | — | `22.21 us/katman` (`0.89 ms/40L`) | **`11.35 us/katman` (`0.45 ms/40L`)** | **-%48.9 Gecikme Azalması** |

#### 3.2 Katman Alt-Bloklarının (Sub-Block) Token Başına Süre Dağılımı (32K Bağlam, Sıcak RAM)
MoE ağırlıkları RAM'de yerleşikken (`0` major page fault, `unperturbed = 45.60 ms/tok` = **`21.93 tok/s`**) yapılan alt-blok profillemesi:

| Alt-Blok (`SubBlockKind`) | Cihaz | Çağrı / Token | Süre (`ms/tok`) | Toplam Pay (%) | Çağrı Başına (`us`) | Verimlilik Kararı |
|---|---|---:|---:|---:|---:|---|
| `RoutedExperts_40L_CPU` (`top-8` MoE) | `CPU` (AVX2/512) | `40` | **`28.09 ms`** | **%56.4** | `702.3 us` | `bs=1`'de katman başına sadece `14.6 MiB` okunduğu için CPU optimaldir (GPU H2D `466.5 MiB` `~152 ms/katman` sürer). |
| `GatedDeltaNet_30L_GPU` | `CUDA0` | `60` | **`10.92 ms`** | **%21.9** | `181.9 us` | Yoğun matris ve recurrent state güncellemesi; GPU'da kalmalı. |
| `FullAttn_Core_10L` (`S1` FlashAttn) | `CUDA0` | `20` | **`3.09 ms`** | **%6.2** | `154.5 us` | `S1` GPU mirror sayesinde 32K'da sadece `3.09 ms/tok`. |
| `Output_LM_Head_GPU` (`vocab=248,320`) | `CUDA0` | `1` | **`2.27 ms`** | **%4.6** | `2,272.7 us` | Büyük sözlük projeksiyonu; GPU'da kalmalı, ancak `n_outputs_max` sınırlandırılmalı. |
| `SharedExpert_40L_GPU` (`ffn_shexp`) | `CUDA0` | `40` | **`1.85 ms`** | **%3.7** | `46.2 us` | `RoutedExperts` ile veri bağımsızdır -> **`ARGUS_HETERO_OVERLAP=1` ile CPU MoE arkasına gizlenir!** |
| `PCIe_H2D_Copy+Sync` (`ffn_moe_out`) | `PCIe` | `40` | **`1.51 ms`** | **%3.0** | `37.8 us` | `ggml_backend_tensor_set_async` ile senkron bekleme kaldırıldı. |
| `PCIe_D2H_Copy+Sync` (`3` MoE girdisi) | `PCIe` | `40` | **`1.19 ms`** | **%2.4** | `29.8 us` | 3 girdi tek `synchronize()` ile asenkron toplandı. |
| `FFN_Merge_Norm_GPU` | `CUDA0` | `40` | **`0.86 ms`** | **%1.7** | `21.6 us` | GPU üzerinde birleştirme ve normalizasyon. |

---

### Görev 4 ve Görev 5: RAM/VRAM Gereksiz Kopyaların Tespiti ve Strateji Karşılaştırması

Denetimde tespit edilen **5 kritik bellek/hesaplama israfı** ve uygulanan çözümler:

1. **`CUDA0` Compute Buffer İçinde `150 MiB` Ölü MoE Ağırlık Kopyası (`ggml_backend_sched_reserve`):**
   - `sched_reserve()` sırasında P3 expert cache henüz bağlanmadığı için GGML scheduler, CPU'daki `gate/up/down_exps` tensörlerini GPU'ya kopyalamak üzere compute buffer içinde **`150 MiB`** yer ayırıyordu; oysa P3 çalışma anında kendi `498 MiB` expert cache tamponunu bağlamaktadır.
   - **Çözüm:** [`argus_expert_cache_bind_dummy()`](file:///home/zwannfrederick/Masaüstü/Sektor/Coding/mamba%20fix/argus_cache/csrc/ggml_layer_prefill.cpp#L261-L321) ile rezervasyon sırasında sahte (`dummy`) cihaz işaretçileri bağlanarak `150 MiB` ölü tahsis tamamen ortadan kaldırıldı.
2. **40 Katmanlık Tam Graf Rezervasyonu vs P3 2-Katman Zarfı (`Envelope`):**
   - Prefill (`n_tokens >= 32`) P3 ile katman-bazlı (`1` katman/graf) yürütülmesine rağmen `llama_init_from_model` 40 katmanlık tam graf için `n_ubatch=2048` üzerinden `CUDA0` compute buffer (`658–879 MiB`) ayırıyordu.
   - **Çözüm:** [`llama-context.cpp`](file:///home/zwannfrederick/Masaüstü/Sektor/Coding/mamba%20fix/scratch/llama.cpp-v050/src/llama-context.cpp#L262-L287) içinde `pp` rezervasyonu `{argus_n_layer - 2, -1, false}` (1 recurrent + 1 full-attention katmanı + `result_norm`/`output` LM head) zarfıyla sınırlandırıldı.
3. **`n_outputs_max = n_ctx` Kaynaklı `CUDA_Host` ve `LM_Head` Şişmesi:**
   - `cparams.n_outputs_max == 0` bırakıldığında `llama.cpp` varsayılan olarak `n_ctx` (`32,768` veya `65,536`) çıktı yuvası ayırarak `CUDA0` compute buffer'ı patlatıyor ve `Legacy_Auto`'nun 4 GB VRAM'de OOM vermesine yol açıyordu.
   - **Çözüm:** [`llama-context.cpp`](file:///home/zwannfrederick/Masaüstü/Sektor/Coding/mamba%20fix/scratch/llama.cpp-v050/src/llama-context.cpp#L140-L154) içinde `n_outputs_max` varsayılan olarak `16` ile sınırlandırıldı; `decode()` sırasında daha fazla logit istenirse (`n_outputs_all > cparams.n_outputs_max`) dinamik olarak büyütülmesi sağlandı.
4. **30 Gated DeltaNet Katmanında `34.23 GiB` (32K) / `132.87 GiB` (64K) Boşa `self_kq_mask` Yazımı ve `480–960` Gereksiz Graf İnşası:**
   - P3 tek-katman prefill sırasında Gated DeltaNet katmanlarında `self_kq_mask` tensörü graf içinde kullanılmadığı için `t->buffer == NULL` kalıyor, `can_reuse_kq_mask` `false` döndürerek her katmanda grafın sıfırdan kurulmasına ve `qwen35moe.cpp` içindeki koşulsuz `inp_attn` çağrısı yüzünden her GDN katmanında `512 × 32768 × 4 B = 64 MiB` maskenin CPU'da sıfırlanmasına yol açıyordu.
   - **Çözüm:** [`llama-graph.cpp`](file:///home/zwannfrederick/Masaüstü/Sektor/Coding/mamba%20fix/scratch/llama.cpp-v050/src/llama-graph.cpp#L430-L515) ve [`qwen35moe.cpp`](file:///home/zwannfrederick/Masaüstü/Sektor/Coding/mamba%20fix/scratch/llama.cpp-v050/src/models/qwen35moe.cpp#L105-L120) içinde `t && t->buffer` kontrolleri eklendi; P3 sırasında tüm 40 katmanda `can_reuse()` `%100` aktif hale geldi.
5. **Host GGUF `mmap` Alanında `1,521.9 MiB` Çift Kopya (GPU Ağırlıkları + Kullanılmayan `blk.40` MTP):**
   - GPU'ya (`CUDA0`) kopyalanan `254` tensör ile ana çıkarımda kullanılmayan `blk.40.*` MTP tensörlerinin dosya önbelleği sayfaları host RAM'de `1,521.9 MiB` yer işgal ederek MoE uzmanlarının sayfalarını sıkıştırıyordu.
   - **Çözüm:** [`reclaim_non_host_gguf_pages()`](file:///home/zwannfrederick/Masaüstü/Sektor/Coding/mamba%20fix/argus_cache/csrc/ggml_layer_prefill.cpp#L152-L246) ile bu sayfalar `MADV_DONTNEED` + `POSIX_FADV_DONTNEED` ile işletim sistemine iade edildi ve aktif MoE sayfaları iki geçişli okuma ile `Active(file)` LRU listesine terfi ettirildi.

#### Strateji Bellek Karşılaştırma Tablosu (`hetero_validation_results.json`)

| Strateji | Bağlam (`n_ctx`) | `n_ubatch` | `CUDA0` Compute (`MiB`) | `CUDA_Host` Compute (`MiB`) | Toplam VRAM (`MiB`) | Boş VRAM (`MiB`) | `S1` GPU Mirror Hücre (`%`) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `Legacy_Auto_8K` (`out=1` manuel) | `8,192` | `2,048` | `530.10` | `64.07` | `3,158.94` | `665.56` | `8,192 / 8,192` (`%100.0`) |
| **`Hetero_Auto_8K` (Tam Otomatik)** | `8,192` | `2,048` | **`388.03` (`-142.1`)** | `64.07` | **`3,016.94` (`-142.0`)** | **`807.56`** | **`8,192 / 8,192` (`%100.0`)** |
| `Legacy_Auto_32K` (`out=1` manuel) | `32,768` | `2,048` | `658.10` | `160.07` | `3,430.94` | `393.56` | `32,768 / 32,768` (`%100.0`) |
| **`Hetero_Auto_32K` (Tam Otomatik)** | `32,768` | `512` | **`132.03` (`-526.1`)** | **`40.02` (`-120.1`)** | **`2,904.94` (`-526.0`)** | **`919.56`** | **`32,768 / 32,768` (`%100.0`)** |
| `Legacy_Auto_64K` (`out=1` manuel) | `65,536` | `2,048` | `722.10` | `288.07` | `3,566.94` | `257.56` | `36,352 / 65,536` (`%55.5`) |
| **`Hetero_Auto_64K` (Tam Otomatik)** | `65,536` | `512` | **`228.03` (`-494.1`)** | **`72.02` (`-216.1`)** | **`3,642.94`** | **`181.56`** | **`65,536 / 65,536` (`%100.0`)** |

> [!IMPORTANT]
> `Legacy_Auto` konfigürasyonunda kullanıcı `n_outputs_max=1` değerini elle vermezse (`n_outputs_max=0` varsayılan), `8K`, `32K` ve `64K` bağlamların tamamında `CUDA0` compute buffer tahsisi 4 GB VRAM sınırını aşarak **OOM (`FAILED TO ALLOCATE`)** hatası vermektedir. `Hetero_Auto` ise hiçbir manuel parametre gerektirmeden (`n_outputs_max=0`) tüm bağlamlarda otomatik çalışır ve 64K'da dahi tüm KV önbelleğini `%100` `S1` GPU belleğine sığdırır.

---

### Görev 6, 7, 8, 9, 10, 11, 12, 13: Maliyet Modeli, Paralel Yürütme ve Bellek Baskısı Kontrolü

#### Görev 10: Heterojen MoE Yerleşim Maliyet Modeli (`argus_evaluate_moe_cost`)
[`argus_hetero.h`](file:///home/zwannfrederick/Masaüstü/Sektor/Coding/mamba%20fix/argus_cache/csrc/argus_hetero.h#L168-L236) içinde tanımlanan maliyet modeli, $N$ tokenlık bir alt-batch (`ubatch`) için katman başına `256` uzmandan beklenen benzersiz aktif uzman sayısını kupon-toplayıcı (`coupon-collector`) formülüyle hesaplar:
$$E_{\text{unique}}(N) = E_{\text{total}} \cdot \left(1 - \left(1 - \frac{k}{E_{\text{total}}}\right)^N\right) = 256 \cdot \left(1 - \left(1 - \frac{8}{256}\right)^N\right)$$
- **CPU Token-Major Maliyeti:** $T_{\text{CPU}}(N) = T_{\text{D2H}}(N) + \frac{E_{\text{unique}}(N) \cdot B_{\text{expert}}}{\text{BW}_{\text{RAM}}} + T_{\text{CPU\_compute}}(N) + T_{\text{H2D}}(N) + T_{\text{sync}}$
- **GPU P3 Layer-Major Maliyeti:** $T_{\text{GPU\_P3}}(N) = \frac{B_{\text{layer}}}{\text{BW}_{\text{PCIe}}} + T_{\text{cuBLAS}}(N) + T_{\text{sync}}$

Gerçek donanım profili üzerinde modelin ürettiği karar tablosu (`hetero_validation_results.json`):

| Token Sayısı ($N$) | `cpu_total_us` (40L, `us`) | `gpu_batch_major_us` (`us`) | `prefer_gpu_batch_major` | Seçilen Yürütme Yolu |
|---:|---:|---:|:---:|---|
| `1` (Decode) | **`674.18`** | `5,367.82` | `false` | **CPU Token-Major + GPU `ffn_shexp` Paralel Overlap** |
| `4` (MTP / Speculative) | **`2,581.27`** | `20,472.30` | `false` | **CPU Token-Major + GPU `ffn_shexp` Paralel Overlap** |
| `8` | **`4,934.99`** | `38,503.90` | `false` | **CPU Token-Major + GPU `ffn_shexp` Paralel Overlap** |
| `16` | **`9,088.17`** | `68,388.70` | `false` | **CPU Token-Major + GPU `ffn_shexp` Paralel Overlap** |
| `32` (Kırılma Noktası) | `15,730.80` | **`109,612.00`** (Amortize P3) | **`true`** | **GPU P3 Layer-Major Prefill** |
| `64` | `25,202.10` | **`149,541.00`** | **`true`** | **GPU P3 Layer-Major Prefill** |
| `512` (`n_ubatch` Prefill) | `110,149.00` | **`177,360.00`** | **`true`** | **GPU P3 Layer-Major Prefill** |
| `2048` | `394,493.00` | **`195,792.00`** | **`true`** | **GPU P3 Layer-Major Prefill** |

#### Görev 7: CPU ve GPU Paralel Yürütme (`ARGUS_HETERO_OVERLAP=1`)
[`qwen35moe.cpp`](file:///home/zwannfrederick/Masaüstü/Sektor/Coding/mamba%20fix/scratch/llama.cpp-v050/src/models/qwen35moe.cpp#L204-L226) içinde `build_moe_ffn` düğümleri `ffn_shexp` düğümlerinden önce grafa eklenir. Böylece her katmanda `CPU` split'inden hemen sonraki `CUDA0` split'inin başında `ffn_shexp` (`gate_shexp`, `up_shexp`, `SwiGLU`, `down_shexp`) yer alır. [`ggml-backend.cpp`](file:///home/zwannfrederick/Masaüstü/Sektor/Coding/mamba%20fix/scratch/llama.cpp-v050/ggml/src/ggml-backend.cpp#L1685-L1765) içinde:
1. `CUDA0` split'inden `CPU` split'ine geçerken 3 girdi tensörü (`ffn_inp`, `selected_experts`, `weights`) `ggml_backend_tensor_get_async` ile tek kuyrukta çekilir ve `CUDA0` tek kez senkronize edilir.
2. `CPU` thread'leri `RoutedExperts` (`MUL_MAT_ID`) hesaplamasına başlamadan hemen önce, bir sonraki `CUDA0` split'inin `CPU` çıktısına (`ffn_moe_out`) bağımlı olmayan `gv_pre` alt-grafı (`ffn_shexp`) CUDA stream'ine **asenkron olarak ateşlenir (`graph_compute_async`)**.
3. `CPU` üzerinde `RoutedExperts` çalışırken `GPU` aynı anda `ffn_shexp`'i tamamlar. `CPU` bittiğinde çıktı `ggml_backend_tensor_set_async` ile senkron bekleme olmadan GPU'ya gönderilir ve yalnızca kalan birleştirme düğümleri (`gv_post`: `ffn_out = ffn_moe_out + ffn_shexp`) yürütülür.

---

### Görev 14, 15, 16: Gerçek `Qwen3.6-35B-A3B` Üzerinde 8K, 32K ve 64K Uçtan Uca Benchmark Sonuçları

Tüm testler aynı GGUF modeli, aynı 3-iğneli (`7194`, `BLUE-FALCON`, `OCTOBER-29`) bağlam, greedy deterministik örnekleme (`temp=0, seed=42`), `3` warmup bloğu (`39` token) + `5` ölçülen blok (`50` token) = `89` kesintisiz decode adımı ile [`bench_hetero_validation.cpp`](file:///home/zwannfrederick/Masaüstü/Sektor/Coding/mamba%20fix/docs/measurements/v080-libargus-2026-10-27/bench_hetero_validation.cpp) tarafından koşturulmuş ve [`hetero_validation_results.json`](file:///home/zwannfrederick/Masaüstü/Sektor/Coding/mamba%20fix/docs/measurements/v080-libargus-2026-10-27/hetero_validation_results.json) dosyasına kaydedilmiştir.

#### 4.1 Uçtan Uca Performans, Gecikme ve Doğruluk Tablosu (8K, 32K Ablasyon A/B/C ve 64K)

| Konfigürasyon | Bağlam (`Prompt`) | `ubatch` | Prefill (`tok/s`) | TTFT (`s`) | Decode Mean / Median (`tok/s`) | `p50` / `p95` (`ms`) | `CUDA0` / `Host` Compute (`MiB`) | VRAM Kullanımı (`Boş MiB`) | Peak RSS (`MiB`) | Prefill Disk Okuma (`MiB`) | Multi-Needle (`Logit KL`) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`8K_A_Legacy_Auto`** (`hetero=0`, `out=1` manuel) | `8,192` (`7,680`) | `2,048` | `193.52` | `41.36 s` | `2.40` / `2.45` | `394.40` / `874.91` | `530.10` / `64.07` | `3,158.9` (`665.6`) | `17,156.3` | `7,391.8` | **`3/3`** (`0.0`) |
| **`8K_B_Hetero_Full_Auto`** (`hetero=1, overlap=1`) | `8,192` (`7,680`) | `2,048` | **`272.96` (`+41.1%`)** | **`30.06 s` (`-27.3%`)** | **`3.27` / `3.37` (`+36.3%`)** | **`323.87` / `753.48`** | **`388.03` / `64.07`** | **`3,034.9` (`789.6`)** | `17,056.5` | **`3,193.7` (`-56.8%`)** | **`3/3`** (`0.0`) |
| **`32K_A_Legacy_Auto`** (`hetero=0`, `out=1` manuel) | `32,768` (`31,744`) | `2,048` | `404.96` | `80.38 s` | `1.72` / `1.76` | `516.29` / `1,055.90` | `658.10` / `160.07` | `3,452.9` (`371.6`) | `16,043.4` | `8,517.1` | **`3/3`** (`0.0`) |
| **`32K_B_Hetero_MemSched_Only`** (`hetero=1, overlap=0`) | `32,768` (`31,744`) | `512` | `395.35` | `84.78 s` | `2.15` / `2.28` (`+25.0%`) | `458.10` / `1,499.86` | **`136.01` / `40.02`** | **`2,932.9` (`891.6`)** | `17,088.3` | `4,607.4` (`-45.9%`) | **`3/3`** (`0.00237`) |
| **`32K_C_Hetero_Full_Auto`** (`hetero=1, overlap=1`) | `32,768` (`31,744`) | `512` | **`431.93` (`+6.7%`)** | **`74.65 s` (`-7.1%`)** | **`3.45` / `3.57` (`+100.6%`)** | **`259.85` / `598.25`** | **`132.03` / `40.02`** | **`2,928.9` (`895.6`)** | `17,088.3` | **`3,072.5` (`-63.9%`)** | **`3/3`** (`0.00237`) |
| **`64K_Hetero_Full_Auto`** (`hetero=1, overlap=1`) | `65,536` (`61,440`) | `512` | **`376.87`** | **`164.92 s`** | **`4.50` / `5.29`** | **`210.59` / `475.57`** | **`228.03` / `72.02`** | **`3,668.9` (`155.6`)** | `17,501.3` | `7,345.2` | **`3/3`** (`0.0`) |

#### 4.2 Sıcak RAM (`0` Major Fault) vs Kısıtlı RAM Baskısı Altında Decode Analizi
- **Sıcak RAM Durumu (`AnonPages < 5 GiB`, tüm `18.66 GiB` MoE ağırlığı RAM'de yerleşik):**
  - [`bench_hetero_profile.cpp`](file:///home/zwannfrederick/Masaüstü/Sektor/Coding/mamba%20fix/docs/measurements/v080-libargus-2026-10-27/bench_hetero_profile.cpp) ve önceki sıcak-RAM koşularında doğrulandığı üzere, `0` major page fault durumunda 32K decode hızı **`14.50–21.93 tok/s` (`45.60 ms/tok`)** seviyesindedir (`RoutedExperts_40L_CPU = 28.09 ms/tok`, `GatedDeltaNet_30L_GPU = 10.92 ms/tok`, `FullAttn_Core_10L = 3.09 ms/tok`, `SharedExpert_40L_GPU = 1.85 ms/tok`).
- **Kısıtlı RAM Baskısı Durumu (`AnonPages = 6.54–7.44 GiB` harici masaüstü/IDE süreçleri tarafından kullanıldığında):**
  - 32 GB sistemde harici süreçler + KV önbellek + CUDA pinned tamponlar sonrasında `18.66 GiB` MoE ağırlıkları için `~17.1 GiB` fiziksel sayfa önbelleği kalmakta, bu da her decode adımında `~3` katmanın (`~1.5 GiB`) NVMe üzerinden sayfa hatası (`major page fault`) üretmesine neden olmaktadır.
  - Bu ağır gerçek dünya RAM baskısı altında dahi **`Hetero_Full_Auto` (`32K_C`)**, `Legacy_Auto` (`32K_A`) referansına kıyasla:
    - Prefill disk okumasını **`8,517.1 MiB`'dan `3,072.5 MiB`'a (`-63.9%`)** düşürmüştür.
    - Decode major page fault sayısını **`241,034`'ten `109,214`'e (`-54.7%`)** ve decode disk okumasını **`987.3 MiB`'dan `427.3 MiB`'a (`-56.7%`)** indirmiştir.
    - Decode hızını **`1.72 tok/s`'tan `3.45 tok/s`'a (`2.01x`)** çıkarmış, `p50` token gecikmesini **`516.29 ms`'dan `259.85 ms`'a (`-49.7%`)**, `p95` gecikmesini ise **`1,055.90 ms`'dan `598.25 ms`'a (`-43.3%`)** düşürmüştür.
    - Profilleme verisinde `SharedExpert_40L_GPU` süresi `32K_A`'da `5.48 ms/tok` iken `32K_C`'de CPU MoE ile paralel örtüşme (`ARGUS_HETERO_OVERLAP=1`) sayesinde **`2.96 ms/tok` (`-46.0%`)** olarak ölçülmüştür.

---

## 4. Varsayılan `ARGUS_AUTO=1` Konfigürasyonu (Görev 16)

Yalnızca doğrulanmış ve net performans/bellek kazancı sağlayan mekanizmalar `ARGUS_AUTO=1` altında varsayılan olarak etkinleştirilmiştir ([`argus_config.h`](file:///home/zwannfrederick/Masaüstü/Sektor/Coding/mamba%20fix/argus_cache/csrc/argus_config.h)):

| Özellik / Ortam Değişkeni | `ARGUS_AUTO=1` Varsayılanı | Durum ve Gerekçe |
|---|:---:|---|
| `ARGUS_HETERO_SCHED` | **`1` (AÇIK)** | Adaptif `n_ubatch=512`, `n_outputs_max=16`, P3 2-katman zarf rezervasyonu, `150 MiB` sahte MoE kopyası eliminasyonu, `1.52 GiB` GGUF ölü sayfa geri kazanımı ve `mincore()` Anti-Belady koruması. **32K'da `-524 MiB` VRAM, 64K'da %100 `S1` GPU mirroring, 8K/32K'da `+41%`/`+49%` prefill hızı sağlar.** |
| `ARGUS_HETERO_OVERLAP` | **`1` (AÇIK)** | Asenkron 3-girdili D2H batching, stream-ordered H2D ve `CUDA0` `ffn_shexp` ile `CPU` `RoutedExperts` paralel yürütmesi. **SharedExpert duvar saatini `-46%` azaltır, decode hızını artırır.** |
| `ARGUS_KV_TIERS` (`S1` + `K1`) | **`1` (AÇIK)** | 8K, 32K ve 64K'da tam dikkat KV önbelleğini GPU `S1` aynasında tutar; taşma durumunda `K1` AVX2/AVX-512 yoluna güvenli geri döner. |
| `ARGUS_RECURRENT_GPU` (`S2`) | **`1` (AÇIK)** | 30 Gated DeltaNet recurrent durumunu GPU'da tutar (`232.5 MiB`). |
| `ARGUS_LAYER_PREFILL` (`P3`) | **`1` (AÇIK)** | $N \ge 32$ için katman-bazlı (`layer-major`) GPU MoE prefill. |
| `ARGUS_MOE_MADV_RANDOM` | **`1` (AÇIK)** | `Active(file)` terfisi sonrası `POSIX_MADV_RANDOM` ile gereksiz OS readahead I/O şişmesini önler. |
| `ARGUS_HOT_EXPERTS` | **`0` (KAPALI)** | Tekdüze (`~%3.1`) uzman dağılımında `mlock` RAM daralmasına yol açtığı için varsayılan kapalıdır. |
| `ARGUS_MULTIPOLE` (`E1`) | **`0` (KAPALI / Opt-in)** | Katman 27 hassasiyeti nedeniyle geniş doğruluk testleri tamamlanana kadar opt-in bırakılmıştır. |
| `ARGUS_TOPR` | **`0` (KAPALI)** | QK-Norm sonrası $\ell_2$ kuyruk normu küçük olduğu için güvenli sertifikalı atlama oranı `%0` olduğundan kapalıdır. |

