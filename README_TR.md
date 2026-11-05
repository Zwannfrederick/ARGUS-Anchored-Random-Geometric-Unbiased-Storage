# ARGUS (`libargus`)

**LLM çalışma zamanları için otonom C++/CUDA KV-cache ve MoE bant genişliği
orkestratörü. KV önbelleğini ve uzman (expert) ağırlıklarını tek bir cihaza
sabitlemek yerine, `libargus` bunları faz-paylaşımlı bir bellek hiyerarşisinde
(GPU VRAM, pinned RAM, pageable RAM ve doğrulanmış `O_DIRECT` disk) yönetir;
böylece uzun bağlamlar ve büyük MoE modelleri dar VRAM bütçelerine sığarken
hem prefill hem de decode aşamalarında host-offload taban çizgilerini geride
bırakır.**

ARGUS bir çıkarım (inference) sunucusu değildir. Bir sunucunun (`llama.cpp` / GGML)
altındaki sistem katmanıdır: her KV hücresinin veya sayfasının nerede duracağını,
nasıl yazılacağını, prefill sırasında MoE uzman ağırlıklarının PCIe üzerinden nasıl
aktarılacağını ve decode attention'ın GPU ile CPU katmanlarını aynı anda nasıl
okuyacağını yönetir.

**Durum: v0.8.0.** RTX 3050 Ti Laptop (4 GB VRAM, PCIe Gen3 x4, 4 çekirdekli Intel
i5-11300H) üzerinde ölçülen sonuçlar:
- **Heterojen CPU/GPU Execution Scheduler ve Uyarlanabilir RAM/VRAM Bellek Yöneticisi (`ARGUS_AUTO=1`):**
  `Qwen3.6-35B-A3B` (`Q4_K_M`, 4 GB VRAM + 32 GB RAM) üzerinde heterojen zamanlayıcı (`ARGUS_HETERO_SCHED=1`, `ARGUS_HETERO_OVERLAP=1`), `CUDA0` compute buffer içindeki `150 MiB` ölü MoE rezervasyonunu kaldırır, P3 graf rezervasyonunu 2-katmanlı zarfla sınırlar (`u_batch=512`, `n_outputs_max=16`), 30 Gated DeltaNet katmanındaki `480–960` gereksiz graf yeniden inşasını ve `34.23–132.87 GiB` boşa `kq_mask` yazımını önler, host GGUF `mmap` alanında `1,521.9 MiB` ölü sayfayı geri kazanır ve asenkron PCIe batching (`-%48.9` split sync gecikmesi) ile GPU `ffn_shexp` ve CPU `RoutedExperts` hesaplamalarını paralel örtüştürür. **32K** bağlamda `CUDA0` compute buffer **%80.0–%85.0** küçülür (`879.3 MiB` / `658.1 MiB` → **`132.03 MiB`**, **`524.0 MiB` net VRAM tasarrufu**), prefill hızı **`431.93 tok/s`** (`73.49 s`, önceki referansa göre **`+49.1%`**) seviyesine çıkar, decode hızı kısıtlı RAM baskısında **2.01 katına** (`p50: 516.29 ms → 259.85 ms`) ve sıcak RAM'de **`14.50–21.93 tok/s`** (`45.60 ms/tok`) düzeyine ulaşır. **64K (`65,536`)** bağlamda ise 4 GB VRAM üzerinde tüm KV hücreleri (**%100**, `1,280.0 MiB`) `S1` GPU aynasına sığar (`376.87 tok/s` prefill, `5.29 tok/s` medyan decode, 8K/32K/64K'da `3/3` multi-needle doğruluğu) ([Heterojen mimari raporu](docs/measurements/v080-libargus-2026-10-27/hetero_architecture_report.md)).
- **32K–128K dolu bağlamda 35B MoE (`Qwen3.6-35B-A3B` Q4_K_M, `--cpu-moe`):**
  64K ve 128K bağlamda stock GPU KV başlatılamaz (VRAM OOM). Stock `-nkvo` ile
  karşılaştırıldığında `libargus` (`S1` bölünmüş decode + `K1` CPU attention + `P2`
  GPU prefill) **64K'da 6.6 kat daha hızlı decode** (`12.31` vs `1.86 tok/s`) ve
  **128K'da 6.0 kat daha hızlı decode** (`6.18` vs `1.02 tok/s`) sunar. `P3`
  katman-öncelikli (layer-major) prefill ise her katmanın MoE uzmanlarını PCIe
  üzerinden mikro-batch başına bir kez yerine prompt başına yalnızca bir kez
  aktararak 32K/64K prefill süresini **1.93 kat / 1.86 kat** kısaltır (`32K'da
  114.3 s → 59.1 s`; `64K'da 242.2 s → 130.4 s`)
  ([M1](docs/measurements/v080-m1-2026-09-27/README.md),
  [P3](docs/measurements/v080-prefill-2026-09-28/README.md)).
- **Karışık Hassasiyetli Katmanlar (`C1`), Multipole / V-Centroid Sınırlı-Kütleli Yaklaşık Atlama (`E1`) ve Seyrek MoE `MADV_RANDOM` (`ARGUS_AUTO=1`):**
  8K dolu bağlamda (`Qwen2.5-0.5B-Instruct` Q4_K_M), varsayılan `ARGUS_AUTO=1` ve
  asimetrik `K=q8_0, V=q4_0` host KV (**-%59.4 host KV RAM** + `f16` GPU aynası)
  prefill süresini **0.815 s'ye (9,936 tok/s, stock'tan 1.10x daha hızlı)** düşürür,
  decode hızını **23.18 tok/s**'den (stock `-nkvo`) **175.45 tok/s'ye (7.57x)**
  çıkarır ve tepe RSS'yi **75.0 MiB** azaltır. Dar 16 MiB bölünmüş ayna (1280 GPU +
  6912 host hücresi) altında ise `ARGUS_AUTO=1`, `f16` ile **79.54 tok/s (3.43x)** ve
  `K=q8_0, V=q4_0` ile **69.60 tok/s (3.00x)** hıza ulaşarak her modda bağlama
  gizlenmiş 6 haneli anahtarı (needle) eksiksiz bulur. QK-Norm içermeyen sentetik tensörler
  üzerinde yapılan 16K–128K kernel testlerinde `K1` exact, stock GGML CPU
  FlashAttention'dan **9.9x–10.1x hızlıdır** (64K'da `5.79 ms` vs `57.43 ms`; 128K'da
  `11.35 ms` vs `114.72 ms`, `q8_0`'da `10.95 ms`); `K1 + E1` sınırlı-kütleli blok atlama
  (`ARGUS_KV_SKIP_EPS=1e-4`) ise `< 6.1e-7` bağıl hata ile 64K `f16` üzerinde **0.91 ms
  (62.8x hızlanma)** ve 128K `f16`/`q4_0` üzerinde **2.56 ms (44.9x hızlanma; `q8_0`
  üzerinde stock `f16`'ya göre %47 daha az RAM ile 44.1x hızlanma)** sağlar
  ([v0.8 kaldıraçları](docs/measurements/v080-libargus-2026-10-27/README.md)).
- **Doğrudan disk sayfalı KV (v0.7 yolu, 4K bağlam, `Qwen2.5-0.5B`):**
  Tüm KV GPU'dayken prefill stock llama.cpp süresinin **1.29 katı**, placement
  policy GPU/pinned/RAM/disk katmanlarını yönetirken **1.36 katıdır**; decode hızı
  stock'un **37.5 tok/s** değerine karşı **55.6** ve **48.2 tok/s**'dir
  ([v0.7 baseline](docs/measurements/v070-e10-2026-09-26/)).

[![packaging](https://github.com/Zwannfrederick/ARGUS-Anchored-Random-Geometric-Unbiased-Storage/actions/workflows/packaging.yml/badge.svg)](https://github.com/Zwannfrederick/ARGUS-Anchored-Random-Geometric-Unbiased-Storage/actions/workflows/packaging.yml)

English documentation: [README.md](README.md)

---

## İçindekiler

- [Bir bakışta durum](#bir-bakışta-durum)
- [ARGUS nedir ve ne değildir](#argus-nedir-ve-ne-değildir)
- [Mimari: `libargus`'un Beş Kaldıracı](#mimari-libargusun-beş-kaldıracı)
- [`libargus`'u llama.cpp ile kullanmak](#libargusu-llamacpp-ile-kullanmak) — derleme, otonom mod, katmanlar ve tüm ayarlar
- [Yerel kullanım: ARGUS Studio, Claude Code ve servisler](#yerel-kullanım-argus-studio-claude-code-ve-servisler)
- [v0.8 sonuçları: Uzun bağlam katmanları, MoE prefill, C1 ve E1](#v08-sonuçları-uzun-bağlam-katmanları-moe-prefill-c1-ve-e1)
- [v0.7 sonuçları: Doğrudan disk sayfalı KV](#v07-sonuçları-doğrudan-disk-sayfalı-kv)
- [Stock karşılaştırmasını okumak](#stock-karşılaştırmasını-okumak)
- [HuggingFace ile kullanım (eski araştırma yolu)](#huggingface-ile-kullanım-eski-araştırma-yolu)
- [Ölçüm metodolojisi](#ölçüm-metodolojisi)
- [Bilinen sınırlar](#bilinen-sınırlar)
- [Önceki kanıtlar (v0.4–v0.6)](#önceki-kanıtlar-v04v06)
- [Yol haritası](#yol-haritası)

---

## Bir bakışta durum

Burada tahmin veya projeksiyon yoktur. Her sayı açıp inceleyebileceğiniz kayıtlı bir
koşudur; her davranışın arkasında, o davranış bozulursa kırılan bir test vardır;
ölçülmemiş olan her şey açıkça **Ölçülmedi** olarak işaretlenmiştir.

| İddia | Durum | Kanıt |
|---|---|---|
| `ARGUS_AUTO=1` otonom olarak K1 CPU decode, S1 VRAM aynaları, P2 GPU prefill, P3 katman-öncelikli prefill, seyrek MoE `MADV_RANDOM` ve Heterojen CPU/GPU Execution Scheduler & Uyarlanabilir RAM/VRAM Yöneticisini (`ARGUS_HETERO_SCHED=1`, `ARGUS_HETERO_OVERLAP=1`) etkinleştirir (E1, Multipole, Hot-Expert Cache ve Top-$r$ opt-in olarak korunur) | Kanıtlandı | [`argus_config.h`](argus_cache/csrc/argus_config.h), [`argus_hetero.h`](argus_cache/csrc/argus_hetero.h), [`hetero_architecture_report.md`](docs/measurements/v080-libargus-2026-10-27/hetero_architecture_report.md) |
| K1 CPU decode attention (`f32`/`f16`/`q8_0`/`q4_0`), `float64` referansına `2e-6` yakınlığındadır ve `{1,2,3,4,8}` iş parçacığında bit-özdeştir | Kanıtlandı (10 geometri, SIMD + skaler; `Qwen3.6-35B-A3B`'nin tüm 10 tam-attention katmanında `< 3e-7` izole hata) | [`test_ggml_decode_attention.cpp`](tests/cpp/test_ggml_decode_attention.cpp), [`v0.8 raporu`](docs/measurements/v080-libargus-2026-10-27/README.md) |
| S1 bölünmüş decode attention (GPU `f16` ayna + `f16`/`q8_0`/`q4_0` host KV üzerinde CPU `K1`), `float64` referansıyla `2e-6` içinde eşleşir | Kanıtlandı (7 `f16` + 2 `C1` karışık hassasiyet senaryosu) | [`test_ggml_split_attention.cpp`](tests/cpp/test_ggml_split_attention.cpp) |
| E1 + Multipole sınırlı-kütleli yaklaşık atlama (`ARGUS_KV_SKIP_EPS=1e-4`, `ARGUS_MULTIPOLE=1`), gerçek `Qwen3.6-35B-A3B` üzerinde `%12.0–%12.6` K-blok ve `%34.8–%60.2` V-hücre budarken attention hatasını `2.4x–11.4x`, 8K ortalama KL sapmasını `%40` (`2.50e-5`) ve 32K maksimum logit hatasını `2.79x` (`1.064` vs `2.969`) düşürür | Kanıtlandı (`Qwen3.6-35B-A3B` 8K & 32K 8-modlu ablasyon, `3/3` needle; sentetik QK-Norm'suz kernel benchmark'ında 64K `f16`'da 62.8x) | [`real_a3b_validation.json`](docs/measurements/v080-libargus-2026-10-27/real_a3b_validation.json), [`README.md`](docs/measurements/v080-libargus-2026-10-27/README.md) |
| C1 karışık hassasiyetli katmanlar (host RAM'de `q8_0`, `q4_0` veya asimetrik `K=q8_0, V=q4_0` + GPU'da `f16` ayna) host KV RAM'ini %47–%72 azaltırken decode'u stock `-nkvo`'ya göre 2.9x–7.6x hızlandırır | `Qwen2.5-0.5B` 8K bağlamda kanıtlandı (`f16`, `q8_0` ve `K=q8_0/V=q4_0` modlarında `needle_found: true`; `auto` ayna ile `0.815 s` prefill ve `175.45 tok/s` decode) | [`bench_e2e_result.json`](docs/measurements/v080-libargus-2026-10-27/README.md) |
| Stock GPU KV'nin OOM olduğu `Qwen3.6-35B-A3B` (`--cpu-moe`) modelinde 64K ve 128K decode | Kanıtlandı: 64K'da stock `-nkvo`'dan **6.6 kat hızlı** (`12.31` vs `1.86 tok/s`), 128K'da **6.0 kat hızlı** (`6.18` vs `1.02 tok/s`), needle bulundu | [M1 raporu](docs/measurements/v080-m1-2026-09-27/README.md) |
| `Qwen3.6-35B-A3B` üzerinde faz-paylaşımlı VRAM arenası ile P3 katman-öncelikli MoE prefill | 5.3K–64K arasında kanıtlandı (32K'da **1.93x**, 64K'da **1.86x** daha hızlı); 128K uyarlanabilir VRAM bütçe planlayıcısı birim testiyle doğrulandı, 128K canlı P3 koşusu **Ölçülmedi** | [P3 raporu](docs/measurements/v080-prefill-2026-09-28/README.md), [`test_ggml_decode_attention.cpp`](tests/cpp/test_ggml_decode_attention.cpp) |
| Doğrudan disk KV sayfaları GPU ↔ pinned ↔ pageable ↔ disk arasında checksum (`crc32c`) korumalı taşınır, bozuk sayfalar reddedilir | Kanıtlandı | [`test_ggml_cuda_mechanism.cpp`](tests/cpp/test_ggml_cuda_mechanism.cpp), [`test_ggml_disk_buffer.cpp`](tests/cpp/test_ggml_disk_buffer.cpp) |
| Desteklenmeyen attention özellikleri veya `-np > 1` yükleme sırasında temizce llama.cpp'nin kendi KV önbelleğine düşer | Kanıtlandı | [`test_unsupported_model_falls_back_to_stock_kv_at_load`](tests/test_native_llama_paged.py) |
| 262K bağlamda uçtan uca throughput | **Ölçülmedi** | — |
| `Gemma-4-26B-A4B` ve `UI-Mate-9B` üzerinde 32K–128K uçtan uca koşular | **Ölçülmedi** (kernel geometrileri `K1`/`S1` içinde doğrulandı) | — |
| HuggingFace araştırma yolunda INT4, INT2, 1-bit veya JL katmanlarında model kalitesi | **Ölçülmedi**; yalnızca FP8 katmanına ulaşıldı | [Kalite](#kalite-huggingface-yolu) |

Geliştirme makinesindeki yerel doğrulama paketleri (Python 3.14): Python **424 geçti, 27 atlandı**
(import/depo hijyeni dahil); native `llama.cpp` K1 (`simd` + `scalar`) ve S1/C1
CUDA bölünmüş attention testleri **tamamı geçti**.

---

## ARGUS nedir ve ne değildir

- **Bir inference sunucusu değildir.** Örnekleme (sampling), batching, HTTP sunumu ve
  GGUF yükleme `llama.cpp` içinde kalır. `libargus`, odaklanmış üç yama
  (`host-kv.patch`, `cuda-kv.patch`, `v08-kv.patch`) aracılığıyla `llama.cpp`'nin KV
  önbelleğine, graf oluşturucusuna ve yürütme döngüsüne bağlanır.
- **Tek kütüphanede iki tamamlayıcı çalışma ailesi:**
  1. **v0.8 Bellek-İçi ve MoE Orkestratörü (`ARGUS_AUTO=1`):** Doğrudan `llama.cpp`'nin
     `-nkvo` host KV önbelleği (`f16`, `q8_0`, `q4_0`) üzerinde çalışır. Sıcak öneki
     (prefix) `f16` olarak GPU VRAM'de aynalar (`S1`), kalan host KV'yi gruplanmış
     SIMD CPU decode (`K1`) ve isteğe bağlı sertifikalı blok atlama (`E1`) ile
     hesaplar, prefill attention'ı GPU'da çalıştırır (`P2`) ve VRAM arenasını prefill
     MoE uzman önbellekleri ile decode KV aynaları arasında paylaştırır (`P3`).
  2. **v0.7 Doğrudan Disk Sayfalı Depo (`ARGUS_KV_DIR`):** KV'yi unlink edilmiş bir
     `O_DIRECT` dosyasında tutar ve 4 KiB checksum korumalı sayfaları katı bayt
     bütçeleri dahilinde GPU, pinned RAM, pageable RAM ve disk arasında terfi/demote
     ettirir.
- **Dürüst kapsam.** Deneysel yan projeler (`hermes`, `neo-mobile`, konteyner
  deneyleri) [`experiments/`](experiments/README.md) altında arşivlenmiş ve paketin
  dışına çıkarılmıştır.

---

## Mimari: `libargus`'un Beş Kaldıracı

```text
llama.cpp (-nkvo -fa on) + libargus (ARGUS_AUTO=1)
  |
  +-- [Prefill Fazı] -----------------------------------------------------------+
  |   1. P3 Katman-Öncelikli MoE Prefill (ggml_layer_prefill.cpp):              |
  |      KV aynalarını ayırır, faz-paylaşımlı VRAM arenasını ödünç alarak bir   |
  |      katmanın MoE uzmanlarını tüm ubatch'ler boyunca GPU'da tutar (PCIe     |
  |      üzerinden N_ubatch kez yerine katman başına 1 kez aktarım), ardından   |
  |      KV aynalarını host'tan yeniden doldurur.                               |
  |   2. P2 GPU Prefill Attention (ggml_kv_tiers.cpp):                          |
  |      FlashAttention için host K/V görünümünü batch başına bir kez GPU'ya    |
  |      aktarır.                                                               |
  |                                                                             |
  +-- [Decode Fazı] ------------------------------------------------------------+
      1. C1 Karışık Hassasiyetli Write-Through (ggml_split_attention.cu):
         Tek bir CUDA kernel'i yeni KV satırlarını GPU aynasına F16 olarak ve
         pinned host belleğine F16, Q8_0 veya Q4_0 olarak yazar.
      2. S1 Bölünmüş Decode Attention (ggml_split_attention.cu + ggml_kv_tiers.cpp):
         GPU, VRAM'deki aynalanmış [0, G) hücreleri üzerinde kısmi softmax
         (max, sum, ağırlıklı V) hesaplarken CPU da RAM'deki [G, n_kv) hücreleri
         üzerinde kısmi sonuçları hesaplar; ikisi tam log-sum-exp indirgemesiyle
         birleştirilir.
      3. K1 Gruplanmış SIMD CPU Decode + E1 Sertifikalı Atlama (ggml_decode_attention.cpp):
         Her KV başlığını (head) tüm GQA sorgu başlıkları için AVX2/AVX-512
         yazmaç-içi çözme (F16, Q8_0, Q4_0) ile tek seferde okur.
         ARGUS_KV_SKIP_EPS > 0 olduğunda, önbelleğe alınmış [k_min, k_max]
         koordinat kutularını ve sink taban değerlerini (watermark) kullanarak
         olası en yüksek softmax katkısı epsilon'un altında kalan blokları atlar.
```

---

## `libargus`'u llama.cpp ile kullanmak

### 1. `llama.cpp`'yi `libargus` ile derlemek

`libargus`, sabitlenmiş `54315813269112dd0baed7112ec87ad93a8218ca` `llama.cpp`
revizyonunu yamalar (bkz. [`integrations/llama.cpp/README.md`](integrations/llama.cpp/README.md)).
Üç yamayı sırayla uygulayın ve CMake'i `argus_cache/csrc` dizinine yönlendirin:

```bash
git -C /path/to/llama.cpp apply /path/to/ARGUS/integrations/llama.cpp/host-kv.patch
git -C /path/to/llama.cpp apply /path/to/ARGUS/integrations/llama.cpp/cuda-kv.patch
git -C /path/to/llama.cpp apply /path/to/ARGUS/integrations/llama.cpp/v08-kv.patch
cmake -S /path/to/llama.cpp -B /path/to/llama.cpp/build-cuda \
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DARGUS_CORE_DIR=/path/to/ARGUS/argus_cache/csrc
cmake --build /path/to/llama.cpp/build-cuda --target llama-server -j4
```

Hiçbir `ARGUS_*` ortam değişkeni tanımlanmadığında, derlenen ikili dosya (binary)
stock `llama.cpp` ile birebir aynı şekilde çalışır.

### 2. Önerilen v0.8 Modları (`ARGUS_AUTO=1`)

```bash
# 1. Sıfır ayarlı otonom mod (F16 host KV + otomatik F16 VRAM aynası + K1 + P2 + P3 + MADV_RANDOM)
export ARGUS_AUTO=1
llama-server -m model.gguf -c 65536 -ngl 99 --cpu-moe -fa on -nkvo -b 65536 -ub 2048

# 2. C1 Asimetrik Karışık Hassasiyet (~%60 daha az host KV RAM, F16 GPU aynası) + Opt-in E1 Multipole Sınırlı-Kütleli Atlama
export ARGUS_AUTO=1
export ARGUS_KV_SKIP_EPS=1e-4
export ARGUS_MULTIPOLE=1
llama-server -m model.gguf -c 65536 -ngl 99 -fa on -nkvo -ctk q8_0 -ctv q4_0
```

### 3. v0.8 Yapılandırma Referansı (`argus_config.h`)

| Değişken | Değerler | Anlamı |
|---|---|---|
| `ARGUS_AUTO` | `1` / `on` / `0` / `off` | Otonom orkestrasyonu etkinleştirir (`ARGUS_KV_CPU_ATTENTION=k1`, `ARGUS_KV_VRAM_BYTES=auto`, `ARGUS_KV_PREFILL=gpu`, `ARGUS_PREFILL=layers`, `ARGUS_MADV_RANDOM=1`; `ARGUS_KV_SKIP_EPS=0.0`, `ARGUS_MULTIPOLE=0`, `ARGUS_MOE_CACHE=0` ve `ARGUS_TOP_R=0` opt-in olarak korunur). Aşağıdaki herhangi bir açık değişken kendi ayarını geçersiz kılar. |
| `ARGUS_KV_CPU_ATTENTION` | `k1` / `stock` | Host KV üzerinde K1 gruplanmış SIMD CPU decode attention'ı seçer. |
| `ARGUS_K1_MAX_TOKENS` | tam sayı (varsayılan `16`) | K1 / S1 decode yolu tarafından işlenen maksimum batch token sayısı (tek token decode ve spekülatif taslak batch'lerini kapsar). |
| `ARGUS_KV_SKIP_EPS` | `[0, 1)` aralığında float (varsayılan `0.0`) | Opt-in E1 sınırlı-kütleli yaklaşık atlama eşiği (veya `1e-4` için `ARGUS_E1=1`). `0.0` %100 exact'tir; `1e-4`, soğuk 32 hücreli K-bloklarını ve V-hücrelerini budar. |
| `ARGUS_MULTIPOLE` | `1` / `on` / `0` / `off` (varsayılan `0`) | `ARGUS_KV_SKIP_EPS > 0` olduğunda opt-in Multipole monopol/dipol K-kutusu sınırlarını ve V-centroid telafisini (`W_skip * mu_V`) etkinleştirir. |
| `ARGUS_MADV_RANDOM` | `1` / `on` / `0` / `off` (`ARGUS_AUTO` içinde varsayılan `1`) | Host MoE uzman tensörlerine (`--cpu-moe`) `POSIX_MADV_RANDOM` uygulayarak işletim sistemi sıralı readahead şişmesini engeller. |
| `ARGUS_MOE_CACHE` | `1` / `on` / `0` / `off` (varsayılan `0`) | Yükseltilmiş `RLIMIT_MEMLOCK` (`ulimit -l`) bulunan sistemler için opt-in frekans takipli sıcak uzman önbelleği (`mlock` / `MADV_COLD`, `ARGUS_HOT_EXPERTS=32`). |
| `ARGUS_TOP_R` | `1` / `on` / `0` / `off` (varsayılan `0`) | K-kutuları için opt-in Top-$r$ (`r=32`) Cauchy-Schwarz kuyruk sınırı (QK-Norm modelleri düz kanal spektrumuna sahip olduğundan varsayılan olarak kapalıdır). |
| `ARGUS_KV_VRAM_BYTES` | `auto` veya bayt sayısı | `-nkvo` host KV'sinin (`f16`, `q8_0`, `q4_0`) S1 `F16` GPU aynalarını etkinleştirir. `auto`, hesaplama tamponları ayrıldıktan sonra kalan boş VRAM'e göre arenayı boyutlandırır. |
| `ARGUS_KV_VRAM_MARGIN` | bayt sayısı (varsayılan `256 MiB`) | `ARGUS_KV_VRAM_BYTES=auto` tarafından bırakılan boş VRAM payı (P3 paylaşımlı MoE uzman önbelleğini sığdırmak gerektiğinde otomatik olarak `64 MiB` güvenlik tabanına kadar iner). |
| `ARGUS_KV_PREFILL` | `gpu` / `cpu` | P2: KV host belleğindeyken (`-nkvo`) prefill FlashAttention'ı GPU üzerinde çalıştırır. |
| `ARGUS_PREFILL` | `layers` / `batches` | P3: çoklu-ubatch prefill'i katman-öncelikli çalıştırarak `--cpu-moe` uzman ağırlıklarının PCIe üzerinden prompt başına bir kez geçmesini sağlar ve VRAM arenasını S1 KV aynalarıyla paylaşır. |

*(v0.7 doğrudan disk `ARGUS_KV_DIR` ayarları için bkz. [`integrations/llama.cpp/README.md`](integrations/llama.cpp/README.md).)*

---

## Yerel kullanım: ARGUS Studio, Claude Code ve servisler

Depo, geliştirildiği yerel ortamı da içerir: ARGUS ile derlenmiş `llama-server` ve
hem sohbet arayüzü hem de Claude Code için Anthropic Messages uç noktası olan
**ARGUS Studio**. Yolda Ollama yoktur.

| Port | Ne | Not |
|---|---|---|
| `8080` | ARGUS `llama-server`, yalnızca OpenAI API | llama.cpp'nin kendi web sayfası kapalı (`"webui": false`); yerine Studio var |
| `8090` | ARGUS Studio | sohbet arayüzü; Claude Code için `/v1/messages`, `/v1/models`, `/v1/messages/count_tokens` |

```bash
pip install --no-build-isolation "argus-cache[studio]"   # Studio için FastAPI/uvicorn
argus-studio --upstream http://127.0.0.1:8080/v1         # arayüz :8090'da
ANTHROPIC_BASE_URL=http://127.0.0.1:8090 ANTHROPIC_API_KEY=argus-local claude
```

`scripts/install_systemd_service.sh`, `argus.target` altında iki kullanıcı servisi
kurar. Servisler açılışta **başlamaz**: 18 GB'lık MoE modelini görüntü yöneticisiyle
aynı anda yüklemek giriş ekranını bekletiyordu (ölçüm: motor 10. saniyede SDDM ile
başladı, masaüstü 62 s sonra geldi). Hedefi oturum açıldıktan sonra compositor
başlatır (`systemctl --user start argus.target`); kurulan `ARGUS Studio` başlatıcısı
da aynısını yapıp Studio'yu uygulama penceresi olarak açar.

`--load-mode mmap` ile expert'ler ilk dokunulana kadar diskte kalır; açılıştan sonraki
ilk istek SSD okumasını ödüyordu (14 tokenlık prefill **9.7 s**). Başlatıcı artık
GGUF'u boşta G/Ç ve CPU önceliğiyle (`ionice -c3`, `nice 19`; BFQ bunu uygular) page
cache'e okur; yalnızca kimsenin kullanmadığı disk zamanını harcar ve geri alınabilir
sıradan önbellek bırakır. Aynı soğuk başlangıç sonrasında: **0.65 s**.

Hermes masaüstü araçları (pencere listesi, OCR, tıklama, kullanıcının kendi Hyprland
kısayolları) yalnızca `experiments/hermes` varsa yüklenir; PyPI kurulumu Studio'yu
bunlar olmadan çalıştırır. Kısayollar canlı olarak `hyprctl binds -j`'den okunup tuş
kombinasyonu olarak basılır, bu yüzden `hyprland.lua` değişince bayatlamaz;
ses/medya/parlaklık dışındaki her kısayol önce onay ister.

---

## v0.8 sonuçları: Uzun bağlam katmanları, MoE prefill, C1 ve E1

### 1. 32K–128K Dolu Bağlamda 35B MoE (`Qwen3.6-35B-A3B` UD-Q4_K_M, `--cpu-moe`, `f16` KV)

RTX 3050 Ti Laptop (4 GB) + i5-11300H (4 sabitlenmiş iş parçacığı, turbo kapalı)
üzerinde [`benchmarks/bench_long_context.py`](benchmarks/bench_long_context.py) ile
ölçülmüştür. Her bağlam, başlangıca yakın bir yere 6 haneli bir kod gizleyen prefill
token'larıyla tamamen doldurulmuş ve ardından 32 token decode edilmiştir
([M1 verisi](docs/measurements/v080-m1-2026-09-27/README.md),
[P3 verisi](docs/measurements/v080-prefill-2026-09-28/README.md)):

| Bağlam | Stock GPU KV Decode | Stock `-nkvo` Decode | `libargus` K1 (`-nkvo`) | `libargus` S1 Katmanlar (`auto`) | En İyi Stock'a Göre S1 | Prefill: Batch-Öncelikli (`P2`) | Prefill: Katman-Öncelikli (`P3`) | `P3` Hızlanması |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **32K** | **22.47 tok/s** | 3.15 tok/s | 13.23 tok/s | 20.12 tok/s | 0.90x | 114.3 s | **59.1 s** | **1.93x** |
| **64K** | *VRAM OOM* | 1.86 tok/s | 9.49 tok/s | **12.31 tok/s** | **6.6x** | 242.2 s | **130.4 s** | **1.86x** |
| **128K** | *VRAM OOM* | 1.02 tok/s | 4.41 tok/s | **6.18 tok/s** | **6.0x** | 540.3 s | *Canlı ölçülmedi* ¹ | — |

¹ *128K'da orijinal statik 256 MiB marjı 386.6 MiB boş VRAM bırakıyordu (en büyük katmanın uzmanları için gereken 498 MiB'den az). `argus_config.h` içindeki `argus_plan_vram_budget`, `free_vram >= shared_expert_bytes + 64 MiB` olduğunda marjı 64 MiB tabanına uyarlayarak bunu çözer (birim testlerinde doğrulandı; canlı 128K P3 tekrar koşusu henüz kaydedilmedi).*

### 1.1 Gerçek `Qwen3.6-35B-A3B` Uçtan Uca Doğrulaması ve SOTA-3 Ablasyonu (`8192` ve `32768`, 3 Warmup + 5 Ölçümlü Koşu)

`Qwen3.6-35B-A3B-UD-Q4_K_M-MTP.gguf` üzerinde [`bench_real_a3b_validation.cpp`](docs/measurements/v080-libargus-2026-10-27/bench_real_a3b_validation.cpp) ([JSON raporu](docs/measurements/v080-libargus-2026-10-27/real_a3b_validation.json), [tam rapor](docs/measurements/v080-libargus-2026-10-27/README.md)) ile ölçülmüştür (seed `2026`, deterministik greedy örnekleme, 32 decode token'ı, %10 / %50 / %85 derinlikte 3 needle: `657446`, `839204`, `412975`):

| Bağlam | Mod | Prefill (`tok/s` / `s`) | TTFT (`s`) | Decode Medyan (`tok/s`) | Token Gecikmesi `p50 / p95` (`ms`) | K-Blok Atlama % | V-Hücre Atlama % | Logit `max_abs` (vs Stock / vs E1-OFF) | $\text{KL}_{\text{mean}}$ (vs Stock / vs E1-OFF) | Needle (`%10/%50/%85`) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| **8192** | `Stock llama.cpp` *(disk I/O darboğazı)* | 25.1 (`321.68 s`) | 321.85 | **5.73** | `168.0 / 237.6` | %0.00 | %0.00 | `0 / —` | `0 / —` | **3/3** |
| **8192** | `ARGUS AUTO (E1 OFF)` | **114.9 (`70.21 s`)** | **70.27** | **18.55** | **`48.3` / `65.0`** | %0.00 | %0.00 | `2.14 / 0` | `3.20e-5 / 0` | **3/3** |
| **8192** | `A: ARGUS AUTO (SOTA-3 OFF)` | 114.9 (`70.21 s`) | 70.27 | **18.31** | `48.9 / 66.5` | **%12.49** | **%38.98** | `2.27 / 1.60` | `3.65e-5 / 4.16e-5` | **3/3** |
| **8192** | `B: A + MADV_RANDOM` | 114.9 (`70.21 s`) | 70.27 | **18.04** | `49.7 / 68.7` | **%12.49** | **%38.98** | `2.27 / 1.60` | `3.65e-5 / 4.16e-5` | **3/3** |
| **8192** | `C: B + Hot-Expert Cache` *(%68.1 isabet)* | 114.9 (`70.21 s`) | 70.27 | **17.95** | `50.4 / 72.3` | **%12.49** | **%38.98** | `2.27 / 1.60` | `3.65e-5 / 4.16e-5` | **3/3** |
| **8192** | `D: C + Multipole / V-Centroid` | 114.9 (`70.21 s`) | 70.27 | **18.25** | `49.4` / **`64.4`** | **%12.57** | **%39.01** | `7.26 / 7.95` | `3.93e-5` / **`2.50e-5`** | **3/3** |
| **8192** | `E: D + Top-r` *(K-skip'i sıfırlar)* | 114.9 (`70.21 s`) | 70.27 | **17.49** | `50.0 / 70.0` | `%0.00` | `%36.18` | `3.78 / 3.94` | `5.18e-5 / 3.64e-5` | **3/3** |
| **32768** | `Stock llama.cpp` | 33.1 (`986.81 s`) | 987.18 | **2.69** | `361.7 / 457.5` | %0.00 | %0.00 | `0 / —` | `0 / —` | **3/3** |
| **32768** | `ARGUS AUTO (E1 OFF)` | **322.7 (`101.15 s`)** | **101.24** | **10.90** *(14.22¹)* | `85.7 / 149.6` | %0.00 | %0.00 | `1.51 / 0` | `2.68e-5 / 0` | **3/3** |
| **32768** | `A: ARGUS AUTO (SOTA-3 OFF)` | 322.7 (`101.15 s`) | 101.25 | **10.56** *(14.29¹)* | `84.4 / 131.0` | **%12.17** | **%34.83** | `2.38 / 2.46` | `2.10e-5 / 2.18e-5` | **3/3** |
| **32768** | `B: A + MADV_RANDOM` | 322.7 (`101.15 s`) | 101.24 | **10.79** | `84.7` / **`111.7`** | **%12.17** | **%34.83** | `2.64 / 2.72` | `4.63e-5 / 2.57e-5` | **3/3** |
| **32768** | `C: B + Hot-Expert Cache` *(%68.7 isabet)* | 322.7 (`101.15 s`) | 101.24 | **11.15** | **`81.7`** / `129.3` | **%12.17** | **%34.83** | `2.85 / 2.97` | `4.39e-5 / 3.68e-5` | **3/3** |
| **32768** | `D: C + Multipole / V-Centroid` | 322.7 (`101.15 s`) | 101.25 | **10.65** | `86.9 / 114.8` | **%12.03** | **%34.81** | **`1.94 / 1.06`** | **`2.85e-5 / 2.35e-5`** | **3/3** |
| **32768** | `E: D + Top-r` *(K-skip'i sıfırlar)* | 322.7 (`101.15 s`) | 101.25 | **10.49** | `91.1 / 147.5` | `%0.00` | `%32.73` | `2.52 / 2.44` | `1.74e-5 / 2.26e-5` | **3/3** |

¹ *Önceki yüksüz host koşusunda (`MemAvailable > 23 GiB`, `RSS ~ 22.1 GiB`) `ARGUS AUTO` 32K'da `Stock`'un `2.86 tok/s` değerine karşı `14.22–14.29 tok/s`'ye ulaşmıştır. Yukarıdaki kesintisiz 8-modlu ablasyonda (`MemAvailable = 15.2 GiB`, `RSS = 10.6 GiB`) `ARGUS AUTO`, `Stock`'un `2.69 tok/s` değerine karşı `10.90–11.15 tok/s` (`4.05x–4.15x`) decode ve `322.7 tok/s` (`9.76x`) prefill sunarken; `MADV_RANDOM` 32K `p95` gecikmesini `-%25.3` (`149.6 -> 111.7 ms`) azaltmış, `Multipole / V-Centroid` ise `E1 OFF` referansına göre 32K maksimum logit hatasını `2.79x` (`2.969 -> 1.064`) düşürmüştür. Bkz. [tam ablasyon ve Katman 27 matematik denetimi](docs/measurements/v080-libargus-2026-10-27/README.md).*

### 1.2 `Qwen3.6-35B-A3B` Üzerinde 32K Çalışma Zamanı Darboğaz İzolasyonu (5-Soruluk Sistem Denetimi)

[`bench_32k_bottleneck_audit.cpp`](docs/measurements/v080-libargus-2026-10-27/bench_32k_bottleneck_audit.cpp) ([JSON telemetri](docs/measurements/v080-libargus-2026-10-27/bottleneck_32k_audit.json), [soğuk readahead JSON](docs/measurements/v080-libargus-2026-10-27/q3_cold_readahead.json), [Bölüm 3.5 raporu](docs/measurements/v080-libargus-2026-10-27/README.md#35-32k-runtime-bottleneck-isolation-on-qwen36-35b-a3b)) ile ölçülmüştür:

| Darboğaz Sorusu | 32K Ölçüm Bulgusu | Temel Metrik / Kanıt |
|---|---|---|
| **S1. CPU MoE RAM Yerleşikliği (`mincore(2)`)** | Toplam `18,662.00 MiB` (`4,777,592` adet 4 KiB sayfa) CPU MoE ağırlığının (`466.55 MiB/katman × 40`), 32K `P3` prefill hemen sonrasında **`%51.87`'si (`9,680.17 MiB`)** fiziksel RAM'dedir (`~11–13 GiB` masaüstü RAM baskısında `%46.28` / `8,637.75 MiB`; `>= 19 GiB` boş RAM olduğunda **`%100`**). | `ffn_{up,gate,down}_exps` üzerinde katman bazlı `mincore(2)` taraması |
| **S2. Katman Bazlı Major Page Fault (`blk.0..39`)** | `P3` katman-öncelikli prefill `blk.0 -> blk.39` sırasıyla aktığı için **`blk.30..39` prefill sonunda `%61.0–%92.5` yerleşiklikle sıcak kalır** (yalnızca `2,190` `majflt`, `8.55 MiB` okuma) ve **`blk.0..29`'u `%25.2–%50.3`'e iter**. Decode sırasında en yoğun fault bölgesi **`blk.18..29`** (`30,590` `majflt`, 8 token'da `119.33 MiB` okuma) ve ardından **`blk.0..7`** (`8,741` `majflt`, `34.85 MiB`) katmanlarıdır. | Katman bazlı `ru_majflt` + `/proc/self/io` `read_bytes` kancaları |
| **S3. `MADV_RANDOM` vs `MADV_NORMAL` Readahead I/O** | `blk.0..7` (`3,712 MiB`) `%0.01` yerleşikliğe boşaltılıp 8 token (`335` tekil uzman = `607.19 MiB`) decode edildiğinde: `MADV_NORMAL`, `blk.0..7` içine **`2,060.04 MiB`** çeker (**`+1,452.85 MiB` / `3.39x` gereksiz readahead şişmesi**), `1,684.73 KiB/majflt` okur, `blk.0..39` genelinde **`12,036.62 MiB` (`11.75 GiB`)** disk okuması tetikler ve `0.17 tok/s`'ye düşer. `MADV_RANDOM` yalnızca **`612.18 MiB`** çeker (**`+5.00 MiB` fark, `-%99.66` şişme azalması**), **`4.07 KiB/majflt`** okur, disk I/O'yu **`19.29x` (`623.92 MiB`)** azaltır ve soğuk decode'u **`2.45x` (`0.42 tok/s`)** hızlandırır. | `madvise(MADV_DONTNEED)` + `posix_fadvise(POSIX_FADV_DONTNEED)` soğuk test |
| **S4. `Multipole` K/V Bellek Okuma ve Kernel Süresi** | 32K'da 10 tam-attention katmanında (`637.97 MiB/tok` exact KV): `E1 Açık, Multipole Kapalı` **`317.11 MiB/tok` (GPU `S1`) / `313.95 MiB/tok` (CPU `K1`)** okur (`-%50.3 / -%50.8`); kernel süresini GPU `S1`'de `11.522 -> 10.122 ms/tok` (`-%12.15`), CPU `K1`'de `37.977 -> 25.659 ms/tok` (`-%32.44`) seviyesine indirir. `Multipole Açık` **`322.39 / 319.31 MiB/tok`** (`-%49.5 / -%49.9`, $\mu_V$ için `+6.56 MiB/tok`) okur ve GPU `S1` kernel süresini **`9.780 ms/tok` (`-%15.12`)** seviyesine çeker (CPU `K1`'de `26.368 ms/tok`, `-%30.57`). | `cudaEventElapsedTime` (`S1`) + `steady_clock` (`K1`) + tam bayt sayaçları |
| **S5. Graph-Split ve `P3` Decode Katkısı** | 32K sıcak RAM'de: `Stock (-nkvo)` **`3.13 tok/s`** (`282` split, `316.35 ms/tok`) ile çalışır. Yalnızca `S1+K1` **`9.00 tok/s` (`2.87x`, `-209.30 ms/tok`)** sağlar. Recurrent State'in GPU'da tutulması (`ARGUS_RS_GPU_OPT=1`) 30 Gated DeltaNet katmanında **`180` graph split'i (`282 -> 102`)** ortadan kaldırarak decode hızını ikiye katlar: **`18.23 tok/s` (`+2.03x`, `-53.25 ms/tok`, Stock'a göre `5.82x`)**. `MoE Split Fusion`, `ARGUS_MOE_CACHE=1` açıkken oluşabilecek `+80` split gerilemesini (`102 -> 182`) önler. | `ARGUS_RS_GPU_OPT` & `ARGUS_MOE_SPLIT_OPT` ablasyonu |

### 2. Uçtan Uca `ARGUS_AUTO=1`, C1 Karışık Hassasiyet ve E1 Atlama (8K Dolu Bağlam, `Qwen2.5-0.5B`)

`qwen2.5-0.5b-instruct-q4_k_m.gguf` üzerinde 8192 bağlamda (8096 prompt token'ı + 32
decode token'ı) hem dar 16 MiB bölünmüş ayna (8192 hücrenin 1280'i GPU'da `F16`
olarak aynalanmış; 6912 hücre host üzerinde) hem de varsayılan `ARGUS_AUTO=1` `auto`
aynası (8192 hücrenin tamamı GPU'da `F16` olarak aynalanmış ve GPU ayna prefill bypass
etkin) ile ölçülmüştür ([veri](docs/measurements/v080-libargus-2026-10-27/README.md)):

| Yapılandırma | Host KV Codec | GPU Ayna | Host KV Bellek Tasarrufu | Prefill | Decode Hızı | Stock `-nkvo`'ya Göre | Tepe RSS | Needle (`657446`) |
|---|---|---|---|---:|---:|---:|---:|:---:|
| `stock-nkvo-f16` | `f16` | Yok | %0 | 0.898 s (9,012 tok/s) | **23.18 tok/s** | 1.00x | 713.5 MiB | Bulundu |
| `argus-auto-f16` *(16 MiB bölünmüş)* | `f16` | `f16` (1280) | %0 | 0.888 s (9,122 tok/s) | **79.54 tok/s** | **3.43x** | 699.3 MiB | Bulundu |
| `argus-auto-c1-q8_0` *(16 MiB bölünmüş)* | `q8_0` | `f16` (1280) | **-%46.9 (1.88x)** | 0.891 s (9,084 tok/s) | **66.94 tok/s** | **2.89x** | 658.2 MiB (**-55.3 MiB**) | Bulundu |
| `argus-auto-c1-q8_0-e1-skip` *(16 MiB bölünmüş)* | `q8_0` | `f16` (1280) | **-%46.9 (1.88x)** | 0.889 s (9,108 tok/s) | **67.67 tok/s** | **2.92x** | 665.2 MiB (**-48.3 MiB**) | Bulundu |
| `argus-auto-c1-k_q8-v_q4-skip` *(16 MiB bölünmüş)* | **K=`q8_0`, V=`q4_0`** | `f16` (1280) | **-%59.4 (2.46x)** | 0.925 s (8,751 tok/s) | **69.60 tok/s** | **3.00x** | 659.7 MiB (**-53.8 MiB**) | Bulundu |
| `argus-auto-full-mirror-k_q8-v_q4` *(`auto` ayna)* | **K=`q8_0`, V=`q4_0`** | `f16` (8192) | **-%59.4 (2.46x)** | **0.815 s (9,936 tok/s)** | **175.45 tok/s** | **7.57x** | **638.5 MiB (-75.0 MiB)** | **Bulundu** |

### 3. Bağımsız CPU Decode Kernel'i: Stock GGML vs K1 Exact vs K1 + E1 Atlama (`Qwen3.6-35B-A3B` Geometrisi)

16K, 64K ve 128K hücrede tek token CPU decode gecikmesi (`d = 256`, `16/2` başlık, 4
iş parçacığı, `-march=native`, [kaynak ve veri](docs/measurements/v080-libargus-2026-10-27/README.md)):

| Bağlam | KV Codec | Katman Başına KV Boyutu | Stock GGML (ms) | `libargus` K1 Exact (ms) | `libargus` K1 + E1 Atlama (ms) | Stock'a Göre Hızlanma | Atlanan Blok (K / Toplam) | Maks Bağıl Hata |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| **16K** | `f16` | 32.0 MiB | 12.91 ms | 1.72 ms (7.5x) | **0.48 ms** | **26.7x** | 1002 / 1024 | `4.35e-07` |
| **16K** | `q8_0` | 17.0 MiB (-%47) | 10.12 ms | 2.41 ms (4.2x) | **0.33 ms** | **31.1x** *(Stock F16'ya göre 39.1x)* | 1002 / 1024 | `4.67e-07` |
| **16K** | `q4_0` | 9.0 MiB (-%72) | 7.84 ms | 2.38 ms (3.3x) | **0.45 ms** | **17.3x** *(Stock F16'ya göre 28.7x)* | 1002 / 1024 | `3.88e-07` |
| **64K** | `f16` | 128.0 MiB | 57.43 ms | 5.79 ms (9.9x) | **0.91 ms** | **62.8x** | 4074 / 4096 | `5.81e-07` |
| **64K** | `q8_0` | 68.0 MiB (-%47) | 34.32 ms | 9.60 ms (3.6x) | **1.33 ms** | **25.8x** *(Stock F16'ya göre 43.2x)* | 4074 / 4096 | `3.94e-07` |
| **64K** | `q4_0` | 36.0 MiB (-%72) | 37.43 ms | 9.40 ms (4.0x) | **1.34 ms** | **27.9x** *(Stock F16'ya göre 42.9x)* | 4074 / 4096 | `4.43e-07` |
| **128K** | `f16` | 256.0 MiB | 114.72 ms | 11.35 ms (10.1x) | **2.56 ms** | **44.9x** | 8170 / 8192 | `6.03e-07` |
| **128K** | `q8_0` | 136.0 MiB (-%47) | 81.50 ms | **10.95 ms (7.4x)** | **2.60 ms** | **31.4x** *(Stock F16'ya göre 44.1x)* | 8170 / 8192 | `3.28e-07` |
| **128K** | `q4_0` | 72.0 MiB (-%72) | 83.19 ms | 18.66 ms (4.5x) | **2.56 ms** | **32.5x** *(Stock F16'ya göre 44.8x)* | 8170 / 8192 | `4.10e-07` |

---

## v0.7 sonuçları: Doğrudan disk sayfalı KV

`Qwen2.5-0.5B-Instruct` Q4_K_M, `F16` KV, 4096 bağlam, 4016 token prompt, 16 üretilen
token, ubatch 64, RTX 3050 Ti Laptop (4 GB), `-nkvo`, GPU/pinned bütçeleri 64 MiB
([veri](docs/measurements/v070-e10-2026-09-26/)):

| Mod | Prefill | Stock'a Göre | Decode | Çıktı Hash'i |
|---|---:|---:|---:|---|
| stock llama.cpp, host KV (tensor-core FlashAttention) | 1.336 s (1.331–1.341) | 1.00x | 37.5 tok/s | `007ddc77` |
| ARGUS, GPU control (teşhis, tüm KV GPU'da) | 1.719 s (1.666–1.759) | 1.29x | 55.6 tok/s | `a152ed56` |
| ARGUS, policy on (GPU/pinned/RAM/disk) | 1.821 s (1.814–1.826) | 1.36x | 48.2 tok/s | `a152ed56` |

KV, GPU bütçesini aştığında (`48 MiB` KV için `2 MiB` GPU ve `2 MiB` pinned katmanı):
`47.6 s` prefill ve `1.0 tok/s` decode (istek başına diskten `2.1 GB` okuma).

---

## Stock karşılaştırmasını okumak

- **Aritmetik sözleşmeler.** Varsayılan exact modda (`ARGUS_KV_SKIP_EPS=0`), `K1` ve
  `S1` bir `float64` referansıyla `2e-6` içinde eşleşir (`fp32` birikimi) ve `K1` iş
  parçacığı sayıları arasında bit-özdeştir. `ARGUS_KV_SKIP_EPS > 0` (`E1`) olduğunda,
  `K1` katı bit-özdeşliği sertifikalı bir softmax payda sınırıyla (`< 1e-4`) takas
  eder. GPU üzerindeki stock `llama.cpp` ise kendi birikim sırasına sahip tensor-core
  FlashAttention kullanır.
- **Ölçülen donanım.** Bu depodaki her ölçüm tek bir makinede (Intel i5-11300H + RTX
  3050 Ti Laptop 4 GB, PCIe Gen3 x4) alınmıştır. Oranları orada ölçüm yapmadan çoklu
  GPU sunucu donanımlarına genellemeyin.

---

## HuggingFace ile kullanım (eski araştırma yolu)

Python paketi (`pip install --no-build-isolation -e .`), v0.4 HuggingFace
`PagedDynamicKVCache` araştırma yolunu sağlar:

```python
import torch
from transformers import AutoModelForCausalLM
from argus_cache import AdaptiveCachePolicy, patch_model_with_argus

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct", dtype=torch.float16,
).to("cuda")

model = patch_model_with_argus(
    model,
    page_size=1024,
    max_active_pages=2,
    max_fp8_pages=2,
    sink_tokens=4,
    activation_policy=AdaptiveCachePolicy(mode="balanced", expected_tokens=16_448),
    pipeline_profile="balanced",
)
```

---

## Ölçüm metodolojisi

```bash
# 1. Python birim ve depo hijyeni testleri
.venv/bin/python -m pytest -q --ignore=tests/test_native_llama_paged.py

# 2. Native llama.cpp C++/CUDA kernel ve yaşam döngüsü testleri
ARGUS_TEST_CUDA=1 \
ARGUS_LLAMA_CPP_DIR="$PWD/scratch/llama.cpp-v050" \
ARGUS_LLAMA_BUILD=build-cuda \
ARGUS_TEST_GGUF="$PWD/scratch/models/stories15M.gguf" \
ARGUS_TEST_QUANT_GGUF="$PWD/scratch/models/qwen2.5-0.5b-instruct-q4_k_m.gguf" \
  .venv/bin/python -m pytest -q tests/test_native_llama_paged.py

# 3. Uçtan uca v0.8 uzun bağlam ve C1/E1 benchmark'ı
.venv/bin/python benchmarks/bench_long_context.py \
  --server scratch/llama.cpp-v050/build-cuda/bin/llama-server \
  --model scratch/models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  --configs docs/measurements/v080-libargus-2026-10-27/bench_e2e_configs.json \
  --contexts 8192 --predict 32 --output result.json
```

---

## Bilinen sınırlar

- **Tek KV akışı (`-np 1`):** `libargus` tek dizili KV önbelleklerini yönetir; çoklu
  akış (`-np > 1`, birleşik önbellek olmadan) veya MLA/ALiBi/soft-capping kullanan
  modeller yükleme sırasında otomatik olarak `llama.cpp`'nin yerel KV önbelleğine
  düşer.
- **S1 GPU kernel'lerinde desteklenen başlık boyutları:** `d` $\in \{64, 128, 256, 512\}$;
  diğer boyutlar otomatik olarak CPU üzerindeki K1'e düşer.
- **Kısa turlarda prefill CPU'ya bağlı (expert'ler RAM'de):** P3 eşiğinin (128 token)
  altındaki promptlar MoE expert'lerini CPU'da çalıştırır; 4 çekirdekli test makinesinde
  token başına ~30–40 ms, yani decode ile aynı hız. Eşiğin üstünde P3 devralır.
- **Sabitlenmiş `llama.cpp` revizyonu:** Yamalar upstream
  `54315813269112dd0baed7112ec87ad93a8218ca` commit'ini hedefler.

---

## Önceki kanıtlar (v0.4–v0.6)

- **v0.5 KV sahipliği ve nicemlenmiş (quantized) CPU eşitliği:** `Qwen2.5-0.5B`
  üzerinde `f32`, `f16`, `q8_0` ve `q4_0` için (crop ve durum geri yükleme dahil) 19
  adım boyunca sıfır logit farkı
  ([artifact](docs/measurements/v050-llama-host-ownership-2026-09-15.json)).
- **v0.6 ilk çalışma noktası (4K):** Stock'a göre prefill `2.12x` (GPU control) ve
  `7.42x` (policy on); v0.7'de `1.29x` ve `1.36x`'e indirildi
  ([checkpoint](docs/measurements/v060-checkpoint-2026-09-18.md)).
- **v0.4 HuggingFace yolu (`Qwen2.5-0.5B`):** 16K'da `4.2x` TPOT karşılığında `-%7.7`
  VRAM; yalnızca `FP8` katmanına ulaşarak `-0.0176` perplexity farkı
  ([analiz](docs/findings-2026-08-14.md)).

### Kalite (HuggingFace yolu)

HuggingFace araştırma yolunda perplexity değerlendirmelerinde yalnızca `FP8`
katmanına ulaşılmıştır; `INT4`, `INT2`, `1-bit` ve `JL` katmanları uçtan uca model
kalitesi açısından **Ölçülmedi**.

---

## Yol haritası

- **v0.8 (mevcut):** `libargus` birleşik otonom yapılandırma (`ARGUS_AUTO=1`), `K1`
  gruplanmış SIMD CPU decode, `E1` sertifikalı blok atlama, `S1` bölünmüş GPU+CPU
  decode, `C1` karışık hassasiyetli (`Q8_0`/`Q4_0` host + `F16` GPU ayna) katmanlar,
  `P2` GPU prefill ve `P3` faz-paylaşımlı katman-öncelikli MoE prefill.
- **Sıradaki:** Uyarlanabilir VRAM bütçe planlayıcısı ile `Qwen3.6-35B-A3B` üzerinde
  canlı 128K `P3` katman-öncelikli prefill süresini kaydetmek ve `Gemma-4-26B-A4B`
  ile `UI-Mate-9B` modellerini uçtan uca ölçmek.

## Lisans

ARGUS, [Apache 2.0 Lisansı](LICENSE) altında lisanslanmıştır.
