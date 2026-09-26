# ARGUS

**LLM runtime'ları için bir KV-cache bellek yöneticisi. KV cache'i tek bir cihaza
sabitlenmiş tek bir tensör olarak değil, sayfalı bir bellek hiyerarşisi (GPU,
pinned RAM, pageable RAM, disk) olarak ele alır; böylece bir bağlam, normalde onu
tutacak VRAM'den daha uzun yaşayabilir.**

ARGUS bir inference sunucusu değildir. Bir sunucunun altındaki katmandır: her KV
sayfasının nerede duracağını, nasıl yazılacağını ve attention'ın onu nasıl
okuyacağını yönetir. En derin entegrasyon llama.cpp'dir. Orada KV ayırma, sayfa
yazma ve attention okumaları ARGUS'a aittir ve ARGUS kendi referans yoluyla
bit-exact kalmak zorundadır.

**Durum: v0.7.0.** Uçtan uca ölçülen tek iş yükünde (4K bağlam, Qwen2.5-0.5B,
RTX 3050 Ti Laptop):
- **Prefill:** tüm KV GPU'dayken stock llama.cpp süresinin **1.29 katı**,
  placement policy GPU/RAM/disk katmanlarını yönetirken **1.36 katı**.
- **Decode:** iki modda da **stock'tan hızlı**; 55.6 ve 48.2 tok/s, stock 37.5.
- **v0.7 başında:** aynı prefill 2.09x, policy-on modu 7.4x idi.

Bunlar tek model, tek GPU ve tek bağlam uzunluğunda alınmış ölçümlerdir; genel bir
hız iddiası değildir. Rakamları alıntılamadan önce
[karşılaştırma notunu](#stock-karşılaştırmasını-okumak) okuyun.

[![packaging](https://github.com/Zwannfrederick/ARGUS-Anchored-Random-Geometric-Unbiased-Storage/actions/workflows/packaging.yml/badge.svg)](https://github.com/Zwannfrederick/ARGUS-Anchored-Random-Geometric-Unbiased-Storage/actions/workflows/packaging.yml)

English: [README.md](README.md)

---

## İçindekiler

- [Bir bakışta durum](#bir-bakışta-durum)
- [ARGUS ne değildir](#argus-ne-değildir)
- [Kurulum](#kurulum)
- [llama.cpp ile kullanım](#llamacpp-ile-kullanım): build, modlar, bütün ayarlar
- [HuggingFace ile kullanım](#huggingface-ile-kullanım)
- [v0.7 sonuçları](#v07-sonuçları) ve [oraya nasıl gelindi](#v07-nasıl-geldi)
- [Stock karşılaştırmasını okumak](#stock-karşılaştırmasını-okumak)
- [Nasıl çalışır](#nasıl-çalışır)
- [Ölçüm metodolojisi](#ölçüm-metodolojisi)
- [Bilinen sınırlar](#bilinen-sınırlar)
- [Önceki kanıtlar](#önceki-kanıtlar-v04v06)
- [Yol haritası](#yol-haritası)

---

## Bir bakışta durum

Burada tahmin yok. Her sayı açıp bakabileceğiniz kayıtlı bir koşudur; her
davranışın arkasında, o davranış bozulursa kırılan bir test vardır; ölçülmemiş
olan her şey **Ölçülmedi** diye yazar. "Ölçülmedi" satırları yayın bekleyen işler
değil, açık konulardır.

| İddia | Durum | Kanıt |
|---|---|---|
| llama.cpp'de KV ayırma, yazma ve attention okumaları ARGUS'a ait | Kanıtlandı | [host ownership](docs/measurements/v050-llama-host-ownership-2026-09-15.json) |
| Her ARGUS attention yolu kendi staged referansıyla bit-exact | Kanıtlandı: float vektör eşitliği, adversarial mask'ler, mutation kontrolü; 4K çıktı hash'i her modda `a152ed56` | [yol eşitliği testleri](tests/cpp/test_ggml_cuda_mechanism.cpp), [hash'ler](docs/measurements/v070-e10-2026-09-26/final-baseline.json), [v0.7 planı](plans/argus-v0.7.0.md) |
| Çıktı stock llama.cpp ile byte-byte aynı | **Sadece aritmetik aynıyken.** v0.5 UI-Mate koşusunda kanıtlandı; stock'un tensor-core FlashAttention kullandığı v0.7 4K benchmark'ında **aynı değil** | [UI-Mate parity](docs/measurements/v050-ui-mate-reference-parity-2026-09-16.json), [not](#stock-karşılaştırmasını-okumak) |
| KV sayfaları GPU ↔ pinned ↔ pageable ↔ disk arasında kaynağı koruyarak taşınır | Kanıtlandı | [CUDA mechanism](docs/measurements/v050-cuda-mechanism-2026-09-16.json) |
| GPU'da yazılan KV satırları llama.cpp'nin CPU kodlamasıyla byte-byte aynı | Kanıtlandı: GGML'in kendi `from_float`'ı ile karşılaştırılır | [`check_gpu_set_rows`](tests/cpp/test_ggml_cuda_mechanism.cpp) |
| Bozulmuş disk sayfası reddedilir, asla sunulmaz | Kanıtlandı: hasarlı slot, byte'lar attention'a ulaşmadan okumayı düşürür | [disk buffer testi](tests/cpp/test_ggml_disk_buffer.cpp) |
| Arka plan write-back yalnızca doğrulanmış slotları yayımlar | Kanıtlandı: yazılır, geri okunur, checksum kontrol edilir; bu arada değişen sayfa dirty kalır | [`check_gpu_write_back`](tests/cpp/test_ggml_cuda_mechanism.cpp), [E7b](docs/measurements/v070-e7b-write-back-2026-09-25.md) |
| Attention kernel'leri kuyruktayken sayfalar taşınabilir | Test edilen sürücüde kanıtlandı: düşmanca test, sayfalar taşınırken kernel'leri 512 MiB'lık kopyanın arkasında kuyruğa sokar; `cudaFree`'nin kuyruktaki işi beklediği probe ile ölçüldü | [`check_policy_table`](tests/cpp/test_ggml_cuda_mechanism.cpp), [probe](docs/measurements/v070-e10-2026-09-26/cudafree-sync-probe.cu) |
| Placement policy kararları attention yolundan bağımsız, birebir aynı | Kanıtlandı: staged, resident ve sayfa tablosu yollarında; 4 MiB ve 256 KiB bütçelerde | [E6](docs/measurements/v070-e6-policy-logic-2026-09-25.md), [E10](plans/argus-v0.7.0.md) |
| 4K prefill, stock'a göre | **Daha yavaş:** 1.29x (GPU-control), 1.36x (policy-on) | [final baseline](docs/measurements/v070-e10-2026-09-26/) |
| 4K decode, stock'a göre | **Daha hızlı:** 55.6 / 48.2, stock 37.5 tok/s | [final baseline](docs/measurements/v070-e10-2026-09-26/) |
| GPU bütçesinden büyük KV | Çalışıyor ama diske bağlı: 48 MiB KV için 2 MiB GPU → 47.6 s prefill, 1.0 tok/s | [E6](docs/measurements/v070-e6-policy-logic-2026-09-25.md) |
| Desteklenmeyen modeller llama.cpp'nin kendi KV cache'ine düşer | Kanıtlandı: `-np 2` açılıyor ve cevap veriyor | [`test_unsupported_model_falls_back_to_stock_kv_at_load`](tests/test_native_llama_paged.py) |
| 262K'da uzun bağlam throughput'u | **Ölçülmedi** | — |
| Başka model, GPU ve bağlam uzunlukları | **Ölçülmedi** | — |
| INT4, INT2, 1-bit veya JL katmanlarında kalite (HuggingFace yolu) | **Ölçülmedi**; sadece FP8'e ulaşıldı | [Kalite](#kalite-huggingface-yolu) |

Bu sürüm için geliştirme makinesindeki yerel test sonuçları:
- Python: **382 geçti, 6 atlandı**
- native llama.cpp, CUDA build'i: **18 geçti** (quantized KV yaşam döngüsü
  kontrolleri dahil)
- native, CPU build'i: **11 geçti, 7 atlandı** (atlananlar sadece CUDA'ya özgü
  kontroller)

CI rozeti sadece paketlemeyi ve depo hijyenini kapsar. Hosted runner'larda CUDA
cihazı yoktur; yeşil rozet ARGUS'un çalıştığını değil, paketin doğru dosyaları
içerdiğini gösterir.

## ARGUS ne değildir

- **Bir inference sunucusu değildir.** Sampling, batching, API sunumu ve model
  yükleme runtime'da kalır; ARGUS KV belleğini yönetir.
- **Genel bir hızlandırıcı değildir.** Ölçülen tek iş yükünde ARGUS prefill'i
  stock'tan hâlâ yavaş, decode'u ise hızlı. İkisi de o iş yükünün ötesinde iddia
  edilmez.
- **Varsayılan olarak bir quantizer değildir.** llama.cpp yolunda KV, GGML
  codec'inde kalır; CUDA yolu için F16. Sayfa bazında karışık hassasiyet sadece
  HuggingFace araştırma yolunda vardır ve kalite kanıtı yalnızca FP8 içindir.
- **Kalıcı depolama değildir.** Disk kopyası unlink edilmiş bir `O_DIRECT`
  dosyasında durur; süreç kapanınca hiçbir şey kalmaz. "Doğrulanmış yedek", bir
  sayfanın tek bir süreç içinde demote edilip geri okunabildiği kopyadır.

---

## Kurulum

```bash
pip install torch                                        # önceden import edilebilir olmalı
pip install --no-build-isolation argus-cache             # Python runtime + native eklenti
pip install --no-build-isolation "argus-cache[gateway]"  # + Anthropic Messages gateway
```

ARGUS kaynak dağıtımı (sdist) olarak gelir ve CUDA eklentisini sizin makinenizde
derler. Bu yüzden CUDA destekli PyTorch, CUDA toolkit ve bir C++17 derleyicisi
önceden kurulu olmalıdır. Hazır wheel yoktur: tek bir PyTorch ABI'si ve CUDA
sürümüne göre derlenmiş bir binary çoğu kurulum için yanlış olur.

`--no-build-isolation` zorunludur. Build, eklentiyi yapılandırmak için kurulu
`torch`'unuzu okur; pip'in izole build'i onu gizler ve eklenti, runtime'ınızda
olmayan bir ABI için derlenir.

llama.cpp entegrasyonu bir pip özelliği **değildir**. Kaynakları
(`argus_cache/csrc/ggml_*`) ve patch'leri (`integrations/llama.cpp/`) hem depoda hem
kaynak dağıtımında (sdist) bulunur ve aşağıda anlatıldığı gibi llama.cpp'ye derlenir.
ARGUS üzerinde geliştirme yapmak için:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip && pip install -e . --no-build-isolation
python setup.py build_ext --inplace
```

---

## llama.cpp ile kullanım

### 1. llama.cpp'yi ARGUS ile derlemek

ARGUS, sabitlenmiş tek bir llama.cpp revizyonunu patch'ler. Revizyon ve tam
sözleşme için
[`integrations/llama.cpp/README.md`](integrations/llama.cpp/README.md)'ye bakın.
Revizyonu doğrulayın, iki patch'i sırayla uygulayın ve CMake'i ARGUS kaynaklarına
yönlendirin:

```bash
git -C /path/to/llama.cpp apply --check /path/to/ARGUS/integrations/llama.cpp/host-kv.patch
git -C /path/to/llama.cpp apply         /path/to/ARGUS/integrations/llama.cpp/host-kv.patch
git -C /path/to/llama.cpp apply --check /path/to/ARGUS/integrations/llama.cpp/cuda-kv.patch
git -C /path/to/llama.cpp apply         /path/to/ARGUS/integrations/llama.cpp/cuda-kv.patch
cmake -S /path/to/llama.cpp -B /path/to/llama.cpp/build \
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DARGUS_CORE_DIR=/path/to/ARGUS/argus_cache/csrc
cmake --build /path/to/llama.cpp/build --target llama-server -j
```

- `CMAKE_CUDA_ARCHITECTURES=86`, test edilen RTX 30 serisinin değeridir; kendi
  GPU'nuzunkini kullanın.
- Sadece CPU'lu bir build (`-DGGML_CUDA=OFF`), CUDA attention olmadan host ve
  doğrudan disk modlarını alır.
- Hiçbir `ARGUS_*` değişkeni ayarlı değilse patch'lenmiş binary aynen stock
  llama.cpp gibi davranır.

### 2. Mod seçimi

Her şey `llama-server`'ın (ya da patch'lenmiş `libllama`'yı kullanan herhangi bir
programın) okuduğu ortam değişkenleriyle ayarlanır. Bütçeler kesin sınırdır; aşılırsa
sessizce büyümek yerine açık bir hatayla durur.

| Mod | Ne için | Gerekli ayarlar | llama-server bayrakları |
|---|---|---|---|
| **Eşlenmiş host KV** | KV dosya destekli host belleğinde; attention llama.cpp'nin | `ARGUS_KV_DIR`, `ARGUS_KV_MAX_BYTES` | `-nkvo` |
| **CPU blok attention** | Eşlenmiş KV üzerinde ARGUS CPU attention | + `ARGUS_KV_RESIDENT_BYTES` | `-nkvo -fa on` |
| **Doğrudan disk KV** | KV sadece `O_DIRECT` diskte, doğrulanmış sayfalar, CPU attention | + `ARGUS_KV_STAGING_BYTES` | `-nkvo -fa on` |
| **CUDA, policy kapalı** | CUDA attention'lı referans disk yolu (her okuma diskten) | + `ARGUS_KV_GPU_BYTES`, `ARGUS_KV_PINNED_BYTES` | `-nkvo -fa on -ctk f16 -ctv f16` |
| **CUDA, policy açık** (ürün modu) | Sayfalar bütçeler içinde GPU / pinned / RAM / disk arasında yerleşir | + `ARGUS_KV_POLICY=on` (isteğe bağlı `ARGUS_KV_RAM_BYTES`) | aynı |
| **GPU-control** (teşhis) | Bütün KV GPU'da, hiç disk yok | + `ARGUS_KV_GPU_CONTROL=1`, policy kapalı | aynı |

v0.7 ölçümlerinde kullanılan eksiksiz bir policy-on örneği (4K bağlam,
Qwen2.5-0.5B):

```bash
export ARGUS_KV_DIR=/fiziksel/diskte/bir/dizin    # tmpfs değil: sayfalar tahliye edilebilmeli
export ARGUS_KV_MAX_BYTES=$((64 << 30))           # KV yedeği için disk baytı (KV'nin 2 katı ayrılır)
export ARGUS_KV_RESIDENT_BYTES=$((4 << 20))       # sayfa tanımlayıcı metadata bütçesi
export ARGUS_KV_STAGING_BYTES=$((4 << 20))        # ARGUS'un bütün host scratch'i, bounce buffer'lar, worker stack'leri
export ARGUS_KV_GPU_BYTES=$((64 << 20))           # GPU katmanı bütçesi (KV sayfaları + attention scratch)
export ARGUS_KV_PINNED_BYTES=$((64 << 20))        # pinned RAM katmanı bütçesi
export ARGUS_KV_POLICY=on
llama-server -m qwen2.5-0.5b-instruct-q4_k_m.gguf -c 4096 -np 1 -ngl 99 \
  -fa on -nkvo -ctk f16 -ctv f16 -ub 64
```

Rejimi GPU bütçesi belirler:
- **Bütün KV sığıyorsa** (burada 48 MiB), ARGUS en hızlı yolunu kullanır.
- **Sığmıyorsa**, sayfalar pinned RAM, RAM ya da diskte durur ve okumalar diske
  bağlı hale gelir ([sonuçlar](#v07-sonuçları)).

### 3. Bütün ayarlar

| Değişken | Değerler | Anlamı |
|---|---|---|
| `ARGUS_KV_DIR` | dizin | ARGUS'u açar. Yedek dosyalar burada oluşturulur (ve unlink edilir). Fiziksel bir dosya sistemi kullanın. |
| `ARGUS_KV_MAX_BYTES` | bayt | KV yedeği için disk / eşleme bütçesi. Doğrudan disk modu her sayfa için iki slot ayırır (KV'nin 2 katı). |
| `ARGUS_KV_RESIDENT_BYTES` | bayt | Eşlenmiş modda CPU blok attention'ı seçen resident hedefi. Disk modlarında sayfa tanımlayıcıları ve resident handle'lar için metadata bütçesi (KV verisi değil). |
| `ARGUS_KV_STAGING_BYTES` | bayt | Doğrudan disk KV'yi açar. Süreç genelinde ARGUS'un her host scratch buffer'ını, bounce sayfasını, prefetch/flusher stack'ini ve attention scratch'ini sınırlar. En az bir kodlanmış satır artı bir sayfa almalı. |
| `ARGUS_KV_BLOCK_CELLS` | hücre (varsayılan 256) | CPU blok attention'ın blok boyu. |
| `ARGUS_KV_GPU_BYTES` | bayt | Disk modlarında CUDA attention'ı açar. GPU KV sayfaları ve ARGUS'un GPU scratch'i için kesin bütçe; llama.cpp'nin kendi VRAM'i için bir sınır değil. |
| `ARGUS_KV_PINNED_BYTES` | bayt | GPU katmanıyla birlikte zorunlu. Pinned RAM katmanı bütçesi. |
| `ARGUS_KV_RAM_BYTES` | bayt (varsayılan: ayarsız = katman kapalı) | Policy için pageable RAM katmanı bütçesi. |
| `ARGUS_KV_POLICY` | `off` (varsayılan) / `on` | Placement policy. Bilinmeyen değer hata verir. |
| `ARGUS_KV_GPU_CONTROL` | `1` / ayarsız | Teşhis amaçlı, GPU'nun tek sahip olduğu store. Disk yedeği ve taşıma yok; policy kapalı olmalı. |
| `ARGUS_KV_STATS_PATH` | dosya | JSON istatistikler (bütçeler, tepe değerler, baytlar, policy sayaçları, profil scope'ları); attention çağrılarından sonra atomik olarak yeniden yazılır. |
| `ARGUS_KV_PROFILE` | `cpu` / `1` | `cpu`: CPU scope'ları ve sayaçlar; `1`: bunlara ek olarak CUDA event zamanlaması. Profil zamanlamayı değiştirir; sadece aynı binary'nin profilsiz koşularıyla karşılaştırın. |
| `ARGUS_KV_ATTENTION_PATH` | `cells-v2` (varsayılan), `cells-kc`, `cells-mlp`, `cells`, `batched`, `direct`, `staged` | A/B kontrolleri için referans attention yolları. Hepsi `staged` ile bit-exact. |
| `ARGUS_KV_CHECKSUM` | `crc32c` (varsayılan) / `fnv` | Sayfa özeti (digest). İkisi de kazara bozulmayı yakalar; hiçbiri bir kimlik doğrulama kontrolü değildir. |
| `ARGUS_KV_PAGE_COMMIT` | `run` (varsayılan) / `page` | GPU-control'de host yazımları: bütün sayfa dizileri ya da sayfa sayfa (referans). |
| `ARGUS_KV_NO_OVERLAP` | ayarlı / ayarsız | Staged attention: transfer ve hesaplamayı üst üste bindirmek yerine tile'ları sırayla işler. |

### 4. Çalışma sırasında ne olur

- **Yazma.** Yeni K/V satırları GPU'da, CPU kodlamasıyla byte-byte aynı yazılır
  (F32→F16, round-to-nearest-even).
  - **Policy-on** store'larda sayfa *dirty* olur. Arka plandaki flusher onu
    pasif disk slotuna yayınlar: yazar, geri okur, checksum ile doğrular, sonra
    yayınlar.
  - **GPU-control**'de yazım tamamen cihazda çözülür.
  - GPU bütçesi sayfaları alamazsa yazım değişmeden host yoluna düşer.
  - Policy-off host yolunda kalır, çünkü referans odur.
- **Okuma.** Bir K/V görünümündeki yazılmış bütün sayfalar GPU'daysa attention
  cihazdaki bir sayfa tablosunu okur ve beklemeden döner. Değilse çağrı sayfaları
  ödünç alır, cold sayfaları çağrıya özel scratch'e kopyalar ve kernel'in bitmesini
  bekler. İkisi de `staged` ile bit-exact.
- **Yerleşim (policy-on).** Her attention çağrısından sonra en az iki kez okunmuş
  sayfalar bütçeler içinde önce GPU'ya, sonra pinned'e, sonra RAM'e terfi eder.
  Bir katman dolduğunda daha az okunan sayfalar doğrulanmış disk kopyalarına
  demote edilir. Attention scratch payı boş tutulur; terfi onunla çatışıp churn
  üretmez.
- **Desteklenmeyen modeller.** Model MLA, attention sinks, KQ bias, soft-capping,
  ALiBi ya da Grok attention kullanıyorsa, veya bağlamda birden fazla KV stream
  varsa (unified cache olmadan `-np > 1`), ARGUS KV cache'i llama.cpp'ye bırakır ve
  `ARGUS KV disabled (<özellik> is unsupported)` loglar. Yanlış yapılandırmalar
  (eksik `-nkvo`, `-fa on` olmadan disk KV) açılmayı yine reddeder.
- **Çökme semantiği.** Yedek dosya unlink edildiği için KV süreçten uzun yaşamaz.
  Süreç içinde, dirty bir sayfanın disk slotu flusher (ya da önce flush yapan bir
  demotion) onu yayınlayana kadar GPU kopyasının gerisinde kalır. Yayınlanan her
  slot doğrulanır ve yenisi doğrulanana kadar önceki slot korunur.

### 5. Gözlemlenebilirlik

`ARGUS_KV_STATS_PATH` ayarlıysa istatistik dosyası şunları raporlar:
- **bütçeler ve tepe değerler:** `peak_gpu_bytes`, `peak_pinned_bytes`,
  `peak_staging_bytes` vb.;
- **disk trafiği:** `read_bytes`, `written_bytes`; ayrıca `committed_pages`;
- **policy sayaçları:** `policy_promotions`, `policy_demotions`,
  `policy_rejected`, `policy_nanoseconds`;
- **resident yol kararları:** her attention çağrısının neden resident yolu
  kullandığı ya da kullanmadığı (`resident_*_accepted`, `resident_*_reject_*`,
  `resident_*_cold_pages`).

`ARGUS_KV_PROFILE` ayarlıysa faz başına (prefill / decode) kapsayıcı ve hariç CPU
scope'ları eklenir; `1` ile ayrıca CUDA kernel ve kopya süreleri.

Benchmark aracı `benchmarks/bench_llama_paged_context.py` bütün modları
(`stock-host-kv`, `argus-cuda-off`, `argus-cuda-on`, `argus-cuda-control`, …)
çalıştırır. Bağlama gizlenmiş bir bilgiyi (needle) kontrol eder, her tekrarda
sunucuyu sıfırdan başlatır ve yukarıdaki istatistikleri toplar.

---

## HuggingFace ile kullanım

v0.4'ten kalan araştırma yolu: ARGUS, modelin KV cache'ini soğuk sayfaları
quantize edebilen sayfalı ve katmanlı bir cache ile değiştirir.

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
    activation_policy=AdaptiveCachePolicy(
        mode="balanced",
        expected_tokens=16_448,  # prompt + istenen en fazla çıktı
    ),
    pipeline_profile="balanced",
)
```

- `page_size=1024` test makinesindeki en iyi gecikme/bellek dengesiydi, evrensel
  bir öneri değil.
- Sayfa yaşam döngüsü olayları bellekte (sınırlı sayıda) tutulur;
  `ARGUS_TRACE_PATH=/yol/trace.jsonl` ile ayrıca JSON satırları olarak yazılır.
- `ARGUS_LOG_LEVEL` logger seviyesini ayarlar.

---

## v0.7 sonuçları

Ölçüm koşulları:
- **Model ve iş yükü:** Qwen2.5-0.5B-Instruct Q4_K_M, F16 KV, 4096 bağlam,
  4016 token'lık prompt, 16 üretilen token, ubatch 64.
- **Donanım ve yapılandırma:** RTX 3050 Ti Laptop (4 GB); her modda KV host'ta
  (`-nkvo`); GPU/pinned bütçeleri 64 MiB.
- **Yöntem:** profiler kapalı, her tekrarda yeni sunucu, bir ısınma; değerler
  median (min–max).

Final baseline `95c74fa` commit'inde alındı ([veri](docs/measurements/v070-e10-2026-09-26/)).

| mod | prefill | stock'a göre | decode | çıktı hash'i |
|---|---:|---:|---:|---|
| stock llama.cpp, host KV (tensor-core FlashAttention) | 1.336 s (1.331–1.341) | 1.00x | 37.5 tok/s | `007ddc77` |
| ARGUS, GPU-control (teşhis, bütün KV GPU'da) | 1.719 s (1.666–1.759) | 1.29x | 55.6 tok/s | `a152ed56` |
| ARGUS, policy-on (GPU/pinned/RAM/disk) | 1.821 s (1.814–1.826) | 1.36x | 48.2 tok/s | `a152ed56` |

Ölçülen request'teki yerleşim sayaçları:
- **Policy-on:** 0 terfi, 0 demote, 0 ret, 12,768 commit edilmiş sayfa, 0 cold
  sayfa.
- **GPU-control:** hiç policy etkinliği ve disk trafiği yok.

**KV, GPU bütçesine sığmadığında** (aynı iş yükü; 48 MiB KV için 2 MiB GPU ve 2 MiB
pinned katmanı):
- 47.6 s prefill, 1.0 tok/s decode,
- request başına 36 terfi, 34 demote ve diskten 2.1 GB okuma.

ARGUS'un var olma sebebi bu rejimdir: bağlam hayatta kalır. Ama diske bağlıdır ve
çok daha yavaştır.

## v0.7 nasıl geldi

GPU-control prefill; aynı iş yükü, her adım ayrı ölçülmüş bir deney
([plan](plans/argus-v0.7.0.md)):

| adım | değişiklik | prefill | decode |
|---|---|---:|---:|
| v0.6.0 başlangıcı | | 2.778 s (2.09x) | 8.0 tok/s |
| E1 | exact kernel'de birleştirilmiş (coalesced) K satırı yüklemeleri | 2.709 s | |
| E2 | donanım CRC32C özetleri, bütün sayfa commit'leri | 2.406 s | 8.1 |
| E4 | V'nin `half2` olarak yüklenmesi | 2.328 s | |
| E5 | tek token'lı decode'un resident yola alınması | | **39.3** |
| E7a | KV yazımlarının GPU'da yapılması | ~1.98–2.03 s | 43 |
| E8 | yazımların cihazda çözülmesi, host'a gidip gelme yok | −%3.5 | |
| E9 | attention'ın store sayfa tablosunu beklemeden okuması | **1.68 s** | 55 |

Policy-on'daki gelişme:
- 9.2 s'den (v0.6/E2) 2.0 s'ye indi: yazımlar GPU'ya taşındı ve disk yazımı arka
  plana alındı (E7b).
- Ardından 1.82 s'ye indi: attention sayfa tablosunu beklemeden okumaya başladı
  (E10).

Yolda denetim ve ölçümle bulunan iki placement-policy hatası düzeltildi (E6):
- yeniden yazılan bir sayfa artık terfisini kaybetmiyor,
- terfi artık attention scratch payını doldurmuyor.

Kazancın büyük kısmı kernel'den değil, attention'ın etrafındaki veri yolundan
geldi: beklemeler, host'a gidip gelmeler, CPU'da kodlama. Exact kernel'in kendisi
prefill'in hâlâ yaklaşık 1.1 s'sini alıyor; stock'un FlashAttention'ı için bu süre
yaklaşık 0.07 s.

## Stock karşılaştırmasını okumak

- **Farklı aritmetik sözleşmeler.** Buradaki stock llama.cpp, tensor core'larda
  kendi toplama sırası ve hassasiyetiyle FlashAttention kullanır. ARGUS'un
  kernel'leri, ARGUS'un staged referansındaki FP32 toplama sırasını bit bit korur:
  Kategori 1 exactness; yeniden sıralama ve tensor core yok.
  - İkisi farklı bitler üretir: 16 token'lık çıktıların hash'leri farklıdır
    (`007ddc77` / `a152ed56`), ama ikisi de needle sorusunu doğru cevaplar.
  - Hız oranları, aynı hesabın iki uygulamasını değil, iki farklı hesabı
    karşılaştırır.
- **Tek iş yükü.** Tek model, tek GPU, tek bağlam uzunluğu; her modda KV host'ta.
  Buradaki hiçbir şey 32K, 262K, başka bir model ya da başka bir GPU hakkında
  tahmin yürütmez.
- **Teşhis ve ürün.** GPU-control her sayfayı disk yedeği olmadan GPU'da tutar;
  veri yolunun yapabileceğinin üst sınırını çizer. Katmanları gerçekten yöneten
  mod policy-on'dur.

---

## Nasıl çalışır

```text
llama.cpp (patch'li)                      HuggingFace modeli
   |  KV ayırma, set_rows, attention          |  cache değişimi
   v                                          v
ARGUS disk store (C++/CUDA)              PagedDynamicKVCache (Python + C++)
   |- sayfa tanımlayıcıları: içerik / yerleşim revizyonları, checksum'lar
   |- katmanlar: GPU | pinned | pageable RAM | O_DIRECT disk (sayfa başına 2 doğrulanmış slot)
   |- placement policy (ggml_kv_policy.cpp), bütçeler store'a ait
   |- GPU yazımları + arka plan write-back flusher
   '- attention: cihaz sayfa tablosu (beklemesiz) | ödünç sayfalar | staged referans
```

- **Yerleşim ve içerik ayrı revizyon eksenleridir.** Bayt koruyan, doğrulanmış
  bir taşıma okumaları geçersiz kılmaz; yazma kılar.
- **Bütçeler store'a aittir.** Policy taşıma önerir; store doğrular, reddeder ve
  sayar.
- **Kuyrukta bekleyen GPU işi varken sayfa ömrü.**
  - Sayfa tablosu girdileri sadece stream sırasıyla yayınlanarak değişir.
  - Değiştirilen bir GPU sayfası `cudaFree` ile serbest bırakılır; test edilen
    sürücüde bu çağrı kuyruktaki bütün cihaz işini bekler.
  - Host'un yerinde yazımları son tablo okumasını bekler.

  Ayrıntı: [E10 invariant'ları](plans/argus-v0.7.0.md).
- **Kaynak düzeni** (`argus_cache/csrc/` altında):
  - `ggml_disk_store.h`: ortak store iç yapıları,
  - `ggml_disk_buffer.cpp`: store ömrü, GGML buffer, host IO,
  - `ggml_disk_gpu.cpp`: GPU yerleşimi, yazımlar, write-back, taşıma,
  - `ggml_cuda_attention.cu`: kernel'ler ve dispatch,
  - `ggml_kv_policy.cpp`: placement policy,
  - `ggml_paged_attention.cpp`: CPU attention,
  - `ggml_host_buffer.cpp`: eşlenmiş mod.

HuggingFace yolunun ilkesi şudur: mantıksal bir KV sayfası, fiziksel yerleşimi ve
fiziksel hassasiyeti üç ayrı şeydir; sıkıştırma katmanları yeteneğe göre seçilen
eklentilerdir. Ayrıntı: [`docs/architecture_TR.md`](docs/architecture_TR.md).

İki mimari sınır:
- `AttentionAdapter`, model sözleşmelerini çekirdeğin dışında tutar.
- `argus_cache/models/hybrid_cache.py`, hibrit modellerde sahipliği açıkça
  belirtir: tam attention'ın KV'si ARGUS'a aittir; lineer attention durumuna
  ARGUS asla dokunmaz.

### Runtime durumu

| runtime | durum | ARGUS'un yönettiği |
|---|---|---|
| llama.cpp | En derin entegrasyon; CPU ve CUDA attention; kendi referansıyla bit-exact | KV ayırma, bütçeler, yazma, yerleşim ve attention okumaları |
| HuggingFace Transformers | Araştırma yolu, v0.4'te ölçüldü | Modelin KV cache'i |
| Ollama | Harici adapter | Ollama içinde hiçbir şey; sadece yapılandırma ve zamanlama |
| vLLM | Güvenli şekilde reddeder | Hiçbir şey; gerçek entegrasyon vLLM'in KV connector'ını gerektirir ([notlar](docs/vllm-verification.md)) |
| SGLang | Uygulanmadı | Hiçbir şey |

---

## Ölçüm metodolojisi

- **Boş makine.** Başka bir GPU hesaplama süreci yok.
  - `codebase-memory-mcp` gibi arka plan yeniden indeksleyicileri her editör/agent
    oturumunda bir tane çalışır ve sürekli `--index-worker` süreçleri doğurur.
  - Her final ölçümde bunlar ölçüm boyunca `SIGSTOP` ile durduruldu, sonra devam
    ettirildi.
  - Bir release-gate denemesinde load'u 7'ye çıkarmışlar ve bir stock koşusunu
    1.33 s yerine 3.1 s'ye itmişlerdi. O deneme etiketlenerek
    [`v070-release-gate-2026-09-26/`](docs/measurements/v070-release-gate-2026-09-26/)
    içinde saklanıyor.
- **Aynı binary ile A/B.** Varyantlar tek bir `llama-server` binary'si altında
  karşılaştırılır: `libllama.so`, `LD_LIBRARY_PATH` ya da bir
  `ARGUS_KV_ATTENTION_PATH` referansıyla değiştirilir ve tekrarlar dönüşümlü
  sırayla yapılır.
- **Profiler kapalı.** Raporlanan her zamanlama profiler kapalıyken alındı;
  profilli koşular sadece darboğaz tespiti için kullanıldı.
- **Önce exactness.** Bir değişiklik ancak her ARGUS yolu `staged` ile bit-exact
  kalıyor ve çıktı hash'i değişmiyorsa kabul edilir.

```bash
# Python test suite'i (çoğu test için CUDA isteğe bağlı; canlı Ollama testleri ARGUS_TEST_LIVE=1 ister)
pytest tests/ -q

# Native llama.cpp + CUDA suite'i
ARGUS_LLAMA_CPP_DIR=/path/to/llama.cpp ARGUS_LLAMA_BUILD=build ARGUS_TEST_CUDA=1 \
ARGUS_TEST_GGUF=/path/to/stories15M.gguf pytest tests/test_native_llama_paged.py -q

# 4K karşılaştırması
python benchmarks/bench_llama_paged_context.py --server /path/to/llama-server \
  --model qwen2.5-0.5b-instruct-q4_k_m.gguf --kv-dir /fiziksel/disk/dizini --contexts 4096 \
  --modes stock-host-kv argus-cuda-control argus-cuda-on --kv-type f16 --ubatch 64 \
  --predict 16 --resident-bytes 4194304 --gpu-bytes 67108864 --pinned-bytes 67108864 \
  --warmups 1 --repeats 5 --output result.json
```

## Bilinen sınırlar

- **CUDA yolu:**
  - sadece F16 KV (Q8/Q4 KV CPU attention kullanır),
  - head boyutu ≤ 256; en hızlı kernel'ler D = 64 ister,
  - tek dizi (tek KV stream), CUDA cihaz 0.
- **Beklemesiz okumalar** yazılmış her K/V sayfasının GPU'da olmasını ister. Policy
  modunda sayfa tablosu, bütün store, tablo ve scratch payı birlikte GPU bütçesine
  sığdığında kurulur. Aksi halde bekleyen (ödünç alan) yol kullanılır.
- **Bir GPU sayfasının serbest bırakılması**, `cudaFree`'nin kuyruktaki cihaz işini
  beklemesine dayanır (test edilen sürücüde öyle). Stream sırasına bağlı bir
  allocator bu cihaz çapındaki beklemeyi kaldırırdı; uygulanmadı.
- **Policy-on yazımları** her çağrıda stream ile hâlâ senkronize olur; orada
  sadece attention beklemesizdir.
- **Yapılandırma** sadece ortam değişkenleriyle yapılır; bütçeler süreç geneli.
- **llama.cpp patch'leri** sabitlenmiş tek bir revizyonu hedefler.
- **İki ayrı store:** HuggingFace yolunun Python store'u ile llama.cpp store'u ayrı
  implementasyonlardır. Sayfa bazında karışık hassasiyet llama.cpp'de yok.
- **Ölçüm kapsamı:** yukarıdaki her şey tek GPU'da (RTX 3050 Ti Laptop), tek modelde
  ve 4K'da ölçüldü.

---

## Önceki kanıtlar (v0.4–v0.6)

**llama.cpp KV sahipliği (v0.5).** Sahiplik iddiasından önce şunlar şart koşuldu:
ayırma kayıtları, sayfa ID'leri, yazma/okuma sayaçları ve attention'ın bu sayfaları
tükettiğini gösteren bir iz.
- Gerçek bir CPU model koşusunda prefill ve durum geri yüklemesinden sonra sekiz
  decode adımı boyunca en büyük logit farkı 0
  ([artifact](docs/measurements/v050-llama-host-ownership-2026-09-15.json)).
- `q8_0` ve `q4_0` KV ile (4 MiB staging, 1 MiB resident bütçe): crop ve geri
  yükleme dahil 19 adımda attention ve logit farkı 0.

**v0.5: sadece mekanizma.** Placement policy olmadan her okuma diske gidiyordu. Tek
bir UI-Mate request'i 14.98 GB okudu ve stock'un 260 s'sine karşı 752 s sürdü.

**v0.6: ilk çalışma noktası (4K).**
- Prefill: stock 1.332 s, GPU-resident 2.821 s (2.12x), policy-on 9.888 s (7.42x).
- Decode: 8.0 / 6.4 tok/s.

Kayıt: [checkpoint](docs/measurements/v060-checkpoint-2026-09-18.md).

**HuggingFace araştırma yolu (v0.4).** Qwen2.5-0.5B, 1024 token'lık sayfalar.
- VRAM tasarrufu bağlamla birlikte büyüyor (16K'da −%7.7).
- TPOT baseline'ın 4.2 katına çıkıyor.
- Test edilen hiçbir satırda baseline OOM olurken ARGUS'un hayatta kaldığı görülmedi.

Kayıtlar: [artifact](docs/measurements/downstream-2026-08-14.json),
[analiz](docs/findings-2026-08-14.md).

**Hassasiyet ve gecikme, motor tek başına.** 32K'da q4_0 bağlamı FP16'nın
136.16 MiB'ine karşı 44.16 MiB'de tutuyor: 3.1 kat daha az bellek, 2.14 kat
gecikme. Motor HuggingFace decode yoluna bağlı değil
([artifact](docs/measurements/v040-fused-attention-benchmark.json)).

### Kalite (HuggingFace yolu)

Ölçülen perplexity farkı −0.0176, ama sadece neredeyse kayıpsız FP8 katmanına
ulaşıldı. INT4, INT2, 1-bit ve JL kalitesi doğrulanmadı.

### Bilerek saklanan bir negatif sonuç

Yerel bir llama-server'a karşı yapılan bir A/B taraması ARGUS'un kazandığını
gösteriyor gibiydi; denetim, ARGUS'un o sürece hiç yüklenmediğini gösterdi. Bundan
hiçbir iddia çıkarılmaz
([artifact](docs/measurements/argus-ab-cache-comparison-2026-09-04.json)).

---

## Yol haritası

- **v0.7 (bu sürüm):** ölçülen 4K farkı GPU-control'de 2.09x'ten 1.29x'e,
  policy-on'da 7.4x'ten 1.36x'e kapandı. Decode stock'tan hızlı. Exactness boyunca
  korundu. Kayıt: [`plans/argus-v0.7.0.md`](plans/argus-v0.7.0.md).
- **v0.8 (backlog):** ARGUS'u başka runtime'ların benimseyebileceği bir eklentiye
  dönüştürmek (paketleme, yapılandırma API'si, çoklu instance bütçeleri, çoklu
  dizi desteği) ve 4K'nın ötesinde ölçümler. Bkz.
  [`plans/argus-v0.8.0.md`](plans/argus-v0.8.0.md).

## Lisans

ARGUS [Apache 2.0 Lisansı](LICENSE) ile lisanslanmıştır.
