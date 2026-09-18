# ARGUS

**LLM runtime'ları için KV-cache bellek yöneticisi. KV cache'i tek bir cihaza
sabitlenmiş tek bir tensor olarak değil, sayfalı bir bellek hiyerarşisi olarak
ele alır — GPU, pinned RAM, pageable RAM, disk — böylece bir context, normalde
onu tutması gereken VRAM'den daha uzun yaşayabilir.**

ARGUS bir inference server değildir ve hızlandırıcı değildir. Bir server'ın
altındaki katmandır: her KV sayfasının nerede ve hangi hassasiyette durduğunun
sahibi. En derin entegrasyon llama.cpp'dir; orada KV tahsisi, sayfa yazımları ve
attention okumaları ARGUS'a aittir.

**Durum: v0.6.0. İsteğe bağlı bir placement policy llama.cpp KV sayfalarının
nerede duracağına karar veriyor; GPU'da duran sayfaları yerinde okuyan CUDA
attention yolu staged referansla bit düzeyinde aynı sonucu veriyor.** Şimdiye
kadar ölçülen tek iş yükünde (4K context, Qwen2.5-0.5B, RTX 3050 Ti Laptop)
ARGUS hâlâ **stock llama.cpp'den yavaş**: KV tamamen GPU'dayken prefill süresi
stock'un 2,12 katı, policy sayfaları GPU ile disk arasında yerleştirirken 7,42
katı. Aşağıdaki her performans sayısı bir maliyet ölçümüdür, kazanç değil.

[![packaging](https://github.com/Zwannfrederick/ARGUS-Anchored-Random-Geometric-Unbiased-Storage/actions/workflows/packaging.yml/badge.svg)](https://github.com/Zwannfrederick/ARGUS-Anchored-Random-Geometric-Unbiased-Storage/actions/workflows/packaging.yml)

English: [README.md](README.md)

## Tek bakışta durum

"Kanıtlandı" diyen her satır onu kanıtlayan dosyaya bağlanır. "Ölçülmedi" diyen
her satır açıktır — yayımlanmayı bekleyen bir şey değildir.

| İddia | Durum | Kanıt |
|---|---|---|
| llama.cpp KV tahsisi, yazımları ve attention okumaları ARGUS'a ait | Kanıtlandı | [host ownership](docs/measurements/v050-llama-host-ownership-2026-09-15.json) |
| Çıktı stock llama.cpp ile birebir aynı | Kanıtlandı | [UI-Mate parity](docs/measurements/v050-ui-mate-reference-parity-2026-09-16.json) |
| KV sayfaları GPU ↔ pinned ↔ pageable ↔ disk arasında, kaynağı koruyarak taşınır | Kanıtlandı | [CUDA mechanism](docs/measurements/v050-cuda-mechanism-2026-09-16.json) |
| HuggingFace yolunda VRAM tasarrufu context ile büyüyor | Kanıtlandı, v0.4 | [downstream](docs/measurements/downstream-2026-08-14.json) |
| Hassasiyet, belleği gecikmeyle monoton biçimde takas ediyor | Kanıtlandı, motor izole | [fused attention](docs/measurements/v040-fused-attention-benchmark.json) |
| ARGUS altındaki runtime'dan hızlıdır | **Hayır**, ölçülen tek iş yükünde: 4K prefill stock süresinin 2,12 katı (GPU-resident) ve 7,42 katı (policy açık) | [v0.6, 4K](#v06-ilk-çalışma-noktası-4k) |
| İsteğe bağlı placement policy sayfaları tier bütçeleri içinde terfi ettirir | 4K'da kanıtlandı; policy açıkken staged ve resident yollar aynı policy kararlarını verir | [residency census](docs/measurements/v060-residency-census-2026-09-18.md), [mixed resident](docs/measurements/v060-mixed-resident-2026-09-18.md) |
| GPU-resident CUDA attention staged referansla bit düzeyinde aynıdır | Kanıtlandı (float-vektör eşitliği, zorlayıcı maskeler, mutation kontrollü) | [lane-per-cell kernel](docs/measurements/v060-kernel-cells-2026-09-18.md) |
| 262K'da uzun bağlam throughput'u | **Ölçülmedi**, ertelendi | [v0.6 planı](plans/argus-v0.6.0.md) |
| INT4, INT2, 1-bit veya JL katmanlarında kalite | **Ölçülmedi** | yalnız FP8'e ulaşıldı, bkz. [Kalite](#kalite) |

Geliştirme makinesinde yerel suite: **401 passed, 3 skipped**. Yukarıdaki CI
rozeti yalnız paketleme ve depo hijyenini kapsar — barındırılan runner'larda
CUDA cihazı yoktur, dolayısıyla yeşil rozet "paket doğru dosyaları gönderiyor"
demektir, "ARGUS çalışıyor" değil.

## ARGUS ne değildir

- **Inference server değildir.** Sampling, batching, API serving ve model
  yükleme runtime'da kalır. ARGUS KV belleğini yönetir.
- **Hızlandırma değildir.** Bu depoda ARGUS'un altındaki runtime'dan hızlı
  decode ettiğini gösteren hiçbir ölçüm yoktur. v0.5 sayıları maliyeti gösterir.
- **Sıkıştırma benchmark'ı değildir.** Codec oranları depolama gerçeğidir, model
  kalitesini kanıtlamaz. Yalnız FP8 katmanının downstream kanıtı vardır.

## Kurulum

```bash
pip install torch                                        # önce import edilebilir olmalı
pip install --no-build-isolation argus-cache             # çekirdek çalışma zamanı
pip install --no-build-isolation "argus-cache[gateway]"  # + Anthropic Messages gateway
```

ARGUS kaynak dağıtımı olarak gelir ve CUDA eklentisini sizin makinenizde derler;
bu yüzden CUDA destekli bir PyTorch, CUDA toolkit ve C++17 derleyicisi önceden
kurulu olmalıdır. Hazır wheel yoktur: tek bir PyTorch ABI ve CUDA sürümüne göre
derlenmiş ikili, kurulumların çoğu için yanlış olurdu.

`--no-build-isolation` opsiyonel değil, zorunludur. Build, eklentiyi
yapılandırmak için kurulu `torch`'unuzu okur; pip'in izole build'i onu gizler.
pip'in geçici ortama indirdiği bir torch'a karşı derlemek hata vermekten daha
kötü olurdu — eklenti, runtime'ınızda olmayan bir ABI için derlenirdi.

ARGUS'un kendisi üzerinde çalışmak için:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip && pip install -e .
python setup.py build_ext --inplace
```

## llama.cpp ile kullanım

En derin entegrasyon ve v0.5'in üzerinde kapandığı yol. KV katmanları, zorunlu
disk, metadata ve staging bütçeleriyle ARGUS'a ait bir `O_DIRECT` store'da
tahsis edilir; sayfa yazımları descriptor taşınmadan önce checksum ile
doğrulanır; attention sayfaları blok blok, sınırlı tek derinlikli prefetch ile
okur.

```bash
export ARGUS_KV_DIR=/hizli/depolamada/bir/yol
export ARGUS_KV_MAX_BYTES=$((1 << 30))     # disk bütçesi
export ARGUS_KV_STAGING_BYTES=$((4 << 20)) # staging bütçesi, direct disk KV'yi açar
llama-server -m model.gguf -c 4096 -fa on -ctk f16 -ctv f16
```

Tanımlanmazsa ikili stock yolu izler. Bütçeler katıdır: aşım sessizce büyümek
yerine açık hata verir. Kurulum, patch ve tam sözleşme:
[`integrations/llama.cpp/README.md`](integrations/llama.cpp/README.md).

## HuggingFace ile kullanım

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
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
        expected_tokens=16_448,  # prompt + istenen maksimum çıktı
    ),
    pipeline_profile="balanced",
)
```

`page_size=1024` test makinesinde ölçülen en iyi gecikme/bellek dengesidir,
evrensel bir tavsiye değildir. Her model, GPU, context dağılımı ve servis
hedefi için yeniden ölçün.

---

# Kanıtlar

## llama.cpp KV sahipliği (v0.5)

**Sahiplik gerçektir, yüklenmiş bir kütüphane değil.** Sahiplik iddia
edilmeden önce tahsis ve page ID kayıtları, write/read sayaçları, attention'ın
bu sayfaları tükettiğine dair iz ve kontrollü backend kapatma testi gerekir;
süreç haritalarında bir kütüphanenin görünmesi hiçbir şey kanıtlamaz. Gerçek bir
CPU model koşusunda: altı attention katmanı, 32.000 vocabulary logiti, prefill
ve state restore sonrası sekiz decode adımında maksimum logit farkı **0**,
1.769.472 byte tahsis ve teardown sonrası 0.
Dosya: [`v050-llama-host-ownership-2026-09-15.json`](docs/measurements/v050-llama-host-ownership-2026-09-15.json).

**Çıktı stock ile birebir aynıdır.** `1cb9e1e4` revision'ındaki sabitlenmiş
upstream UI-Mate mesaj kurucusu ve parser'ıyla, hash'i aynı payload altında,
stock llama.cpp ve ARGUS 9B çok kipli bir modelde CUDA attention ile aynı
reasoning metnini, aynı koordinatı, aynı parse edilmiş eylemi ve aynı 119
completion token'ı üretti.
Dosya: [`v050-ui-mate-reference-parity-2026-09-16.json`](docs/measurements/v050-ui-mate-reference-parity-2026-09-16.json).

**Birebirlik sıkıştırma ve restore altında da korunur.** Qwen2.5-0.5B, `q8_0` ve
`q4_0` KV, 4 MiB staging, 1 MiB resident: crop ve state restore dahil 19 adımda
attention ve logit farkı 0, peak staging 1,27 MiB.

**Migration kaynağı korur.** GPU, pinned RAM, pageable RAM ve disk ayrı
bütçelidir. Taşıma, kaynağı bırakmadan önce hedefi ayırır ve doğrular;
başarısız transferler ve eski revision'lar yayımlanmış sayfayı korur. 48/64/256
head boyutları için, Q8 ve Q4 encoded byte'lar dört yerleşim arasında
değişmeden taşınarak doğrulandı.
Dosya: [`v050-cuda-mechanism-2026-09-16.json`](docs/measurements/v050-cuda-mechanism-2026-09-16.json).

## Neden hız sonucu yok

v0.5 **yalnız mekanizma** sağlar. Servis yolunda hiçbir şey sayfayı terfi
ettirmez — `argus_disk_move_page` yalnız testlerden çağrılır — dolayısıyla her
attention okuması diske gider. Yukarıdaki UI-Mate koşusunda:

| | stock | ARGUS |
|---|---:|---:|
| prefill | 89,34 ms/token | 151,31 ms/token |
| decode | 165,58 ms/token | 2915,31 ms/token |
| tek istek | 260,4 s | 752,0 s |

Tek istek diskten **14,98 GB** okudu. Bağlayıcı kısıt bütçeler değildi: 4 MiB
GPU ve 4 MiB pinned verildi, yalnız ~0,5 MiB kullanıldı, çünkü onları
kullanmaya karar veren bir şey yok. Ayrıca `ARGUS_KV_RESIDENT_BYTES` bir KV
çalışma kümesini değil sayfa descriptor tablosunu bütçeler — 256 MiB KV için
65.536 descriptor.

**Bu, placement policy'sinin yokluğunun maliyetidir.** Ayarlanmış bir
konfigürasyon değil, kısılmış bir konfigürasyon değil ve bir çalışma noktası
değil. v0.6 policy'yi ve resident attention yolunu ekledi; ilk çalışma noktası
ölçümü aşağıdadır.

## v0.6: ilk çalışma noktası (4K)

Qwen2.5-0.5B-Instruct Q4_K_M, F16 KV, 4096 context, 4016 token prompt, 16 üretilen
token, ubatch 64, RTX 3050 Ti Laptop (4 GB); her modda llama.cpp KV'si host'ta
(`-nkvo`). Profiler kapalı, dönüşümlü sırada üç tekrar; median (min–max). Çıktı
ARGUS modları ve yolları arasında aynıdır.

| mod | prefill | stock'a göre | decode |
|---|---:|---:|---:|
| stock llama.cpp, host KV | 1,332 s (1,326–1,335) | 1,00x | 37,3 tok/s |
| ARGUS, tüm KV GPU'da (tanısal kontrol) | 2,821 s (2,779–2,873) | 2,12x | 8,0 tok/s |
| ARGUS, policy açık (GPU + disk yerleşimi) | 9,888 s (9,779–9,929) | 7,42x | 6,4 tok/s |

v0.6 boyunca aynı iş yükündeki değişim (GPU-resident kontrol prefill'i): staged
skaler attention 40,9 s → çağrı başına tek batched launch 12,8 s → D=64
özelleştirmesi 5,5 s → lane-per-cell exact kernel 3,7 s → dalsız value döngüsü
2,8 s. Policy açıkken süre, yeni yazılmış tek bir cold sayfanın bütün çağrıyı
staged yola düşürmesi engellenince 48,3 s'den 9,9 s'ye indi. Her adım staged
kernel ile float-vektör eşitliğini korudu ve nedeni ölçümle doğrulandı (son iki
kernel adımı için Nsight Compute).

Bunun göstermedikleri: decode değişmedi (hâlâ staged yol, stock'tan 4,7 kat
yavaş); policy açıkken kalan farkın çoğu doğrulamalı disk write-through'dan
geliyor; 4K üstü, başka model veya GPU ölçülmedi. Ayrıntılar:
[checkpoint](docs/measurements/v060-checkpoint-2026-09-18.md),
[kernel adımları](docs/measurements/v060-kernel-cells-mlp-2026-09-18.md),
[Nsight Compute atfı](docs/measurements/v060-ncu-cells-2026-09-18.md).

## HuggingFace araştırma yolu (v0.4)

Projenin nereden geldiğine dair tarihsel bağlam. Qwen2.5-0.5B-Instruct,
RTX 3050 Ti Laptop (4 GB), FP16, batch 1, 64 üretilen token, üç ölçülen tekrar,
bir ısınma, 1024 token'lık sayfalar.

| context | baseline VRAM | ARGUS VRAM | VRAM değişimi | baseline TPOT | ARGUS TPOT |
|---:|---:|---:|---:|---:|---:|
| 512 | 982,2 MiB | 988,2 MiB | +%0,6 | 18,62 ms | 28,69 ms |
| 1.024 | 1007,1 MiB | 1007,1 MiB | %0,0 | 18,55 ms | 30,21 ms |
| 2.048 | 1056,8 MiB | 1056,9 MiB | %0,0 | 19,24 ms | 30,85 ms |
| 4.096 | 1149,4 MiB | 1137,4 MiB | -%1,0 | 19,14 ms | 40,21 ms |
| 8.192 | 1340,4 MiB | 1280,5 MiB | **-%4,5** | 17,85 ms | **52,51 ms** |
| 16.384 | 1722,5 MiB | 1590,6 MiB | **-%7,7** | 18,84 ms | **79,78 ms** |

16K'da TTFT baseline için 1,819 s, ARGUS için 3,992 s. Bunun gösterdiği ve
göstermediği:

- VRAM tasarrufu gerçektir ve bu deneyde context uzunluğuyla büyür.
- Çözülmemiş sorun decode gecikmesidir: 16K'da TPOT baseline'ın 4,2 katı.
- **Hiçbir test satırında baseline OOM olurken ARGUS ayakta kalmıyor.** OOM
  önleme ve daha geniş kullanılabilir context penceresi hipotezdir, sonuç değil.

Dosya: [`downstream-2026-08-14.json`](docs/measurements/downstream-2026-08-14.json).
Analiz: [`docs/findings-2026-08-14.md`](docs/findings-2026-08-14.md).

## Hassasiyet–gecikme takası, motor izole

Structure-of-Arrays sayfa tablosu, *hassasiyeti* (`ACTIVE_FP16`, `GGML_Q8_0`,
`GGML_Q4_0`) *yerleşimden* (`GPU_DEVICE`, `HOST_PINNED`, `HOST_PAGEABLE`) tek
bir bitişik descriptor tablosunda ayırır. `DirectPagedAttentionEngine` bunun
üzerinde tam bir tile-by-tile online-softmax özyinelemesi çalıştırır: context
boyutunda FP16 KV tensörü hiç oluşturulmaz ve yeniden kurulum tek bir sayfa
tile'ıyla sınırlanır.

RTX 3050 Ti Laptop üzerinde Qwen benzeri geometriyle ölçüldü (24 query head,
4 KV head, head_dim 256, sayfa 128):

| context | ACTIVE_FP16 | GGML_Q8_0 | GGML_Q4_0 |
|---:|---|---|---|
| 1.024 | 1,98 ms / 12,15 MiB | 3,15 ms / 10,28 MiB | 4,24 ms / 9,28 MiB |
| 4.096 | 7,51 ms / 24,15 MiB | 11,43 ms / 16,65 MiB | 15,86 ms / 12,65 MiB |
| 8.192 | 14,71 ms / 40,15 MiB | 22,55 ms / 25,15 MiB | 31,61 ms / 17,15 MiB |
| 16.384 | 29,26 ms / 72,15 MiB | 44,97 ms / 44,15 MiB | 62,77 ms / 26,15 MiB |
| 32.768 | 58,48 ms / 136,16 MiB | 90,15 ms / 76,16 MiB | 125,25 ms / 44,16 MiB |

Takas monoton ve diktir: 32K'da q4_0 context'i 44,16 MiB'de tutuyor, FP16'nın
136,16 MiB'ine karşı — 3,1 kat az bellek, 2,14 kat gecikme. Bu yalnız motorun
kendisidir. **HuggingFace decode yoluna bağlanmış değildir**, dolayısıyla bu
sayılar hiçbir uçtan uca sonuçta görünmez.

Dosya: [`v040-fused-attention-benchmark.json`](docs/measurements/v040-fused-attention-benchmark.json).

## Kalite

Ölçülen perplexity farkı **-0,0176**, ancak o pasaj uzunluğunda yalnız
kayıpsıza yakın FP8 katmanına ulaşıldı. Bu, INT4, INT2, 1-bit veya JL
kalitesini **doğrulamaz** — bu uyarı ölçüm dosyasının kendisinde de yazılıdır.
q4_0 retrieval probu 31k token'da tek ve bağışlayıcı bir görevdir; kuantize KV
altında akıl yürütme, kod üretimi ve uzun menzilli tutarlılık ölçülmemiştir.

## Bilerek saklanan bir olumsuz sonuç

Yerel bir llama-server'a karşı 4K/16K/32K/64K A/B taraması (Qwen3.6-35B-A3B,
q4_0 KV) ARGUS lehine bir kazanç gösterir gibi oldu. **Dosyadaki denetim
ARGUS'un sürece hiç yüklenmediğini gösteriyor**: `argus_in_llama_server_maps:
false`, `argus_maps_count: 0` ve her context'te iki kolda byte düzeyinde aynı
peak VRAM. Ortaya çıkan decode hızı farkı (16K'da 17,66'ya karşı 11,88 tok/s)
ARGUS'a değil, gateway'in prompt-prefix cache'ine dayanıyor.

Bu koşudan hiçbir ARGUS iddiası çıkarılmıyor. Depoda duruyor çünkü yanlış
atfedilmiş bir kazanç, tam olarak sorgulanmadan geçip gidecek türden sonuçtur.
Yanında yayımlanan taramalar (`load-mode-comparison`, `pmin-sweep`,
`speculative-sweep-n2-n3-n4`) llama.cpp runtime ayarıdır ve öyle etiketlenmiştir.

Dosya: [`argus-ab-cache-comparison-2026-09-04.json`](docs/measurements/argus-ab-cache-comparison-2026-09-04.json).

---

# Nasıl çalışır

## Mimari

```text
runtime (llama.cpp / HuggingFace)
       |
       v
ARGUS KV bellek katmanı
       |
       +-- page descriptor tablosu: hassasiyet ve yerleşim bağımsız
       |
       +-- bütçeli katmanlar: GPU, pinned RAM, pageable RAM, disk
       |
       +-- native C++ sayfa yaşam döngüsü, checksum doğrulamalı yazımlar
       |
       +-- sayfaları yerinde okuyan CUDA / CPU attention
```

Yönetici ilke şudur: **mantıksal bir KV sayfası, fiziksel yerleşimi ve fiziksel
hassasiyeti üç ayrı şeydir.** Bir sayfa GPU/FP16, pinned/Q8 veya disk/Q4
olabilir; GPU, FP16 anlamına gelmez. Sıkıştırma katmanları eklentidir; sabit
kodlanmış isimlerle değil, yetenekler ve sayısal codec metadata'sıyla seçilir.
Sahiplik ve genişletme noktaları:
[`docs/architecture.md`](docs/architecture.md).

## Model sözleşmeleri core'un dışında kalır

`AttentionAdapter`, `config.model_type` başına native uygunluğu, query
hazırlığını ve çıktı yerleşimini sahiplenir. Uygulamalar
`register_attention_adapter()` ile sözleşme ekler; kayıtlı olmayan bir model
yanlışlıkla native page attention'a girmez ve kendi attention implementasyonunu
korur.

Tam ve doğrusal attention'ı karıştıran modeller için (örneğin Qwen3.8 Gated
DeltaNet), `argus_cache/models/hybrid_cache.py` sahipliği açıkça belirtir:
büyüyen full-attention KV'si ARGUS'undur, doğrusal attention katmanlarının sabit
boyutlu recurrent ve conv state'i ARGUS'un dokunacağı şey değildir. Her cache
işleminin çevresinde deterministik state digest'leri doğrulanır ve desteklenmeyen
bir katman rolü sayfalanmak yerine kapanarak hata verir.

## Runtime durumu

| runtime | durum | ARGUS neyi yönetiyor |
|---|---|---|
| llama.cpp | En derin entegrasyon; host ve direct-disk KV, CPU ve CUDA attention, birebir parity | KV tahsisi, bütçeler, sayfa yazımları ve attention okumaları |
| HuggingFace Transformers | Araştırma yolu, ölçüldü | Modelin KV cache'i |
| Ollama | Dış adapter, canlı test edildi | Ollama içinde hiçbir şey; yalnız konfigürasyon ve zamanlama |
| vLLM | Kullanılamıyor, kapanarak reddediyor | Hiçbir şey; sahte monkey patch kurulmuyor |
| SGLang | Uygulanmadı | Hiçbir şey |

Eski vLLM entegrasyonu vLLM KV bloklarının sahibi değildi ve onları
sıkıştıramıyordu; bu yüzden mevcut adapter numara yapmak yerine aktivasyonu
reddediyor. Gerçek bir entegrasyon vLLM'in KV connector'ını veya özel
attention-backend arayüzlerini kullanmalıdır; bkz.
[`docs/vllm-verification.md`](docs/vllm-verification.md).

---

# Yol haritası

## v0.6 — placement policy (yayınlandı)

[`plans/argus-v0.6.0.md`](plans/argus-v0.6.0.md) içindeki sözleşmeye göre
uygulandı:

- **Policy, store'un üstünde bir modüldür** (`ARGUS_KV_POLICY=on`, varsayılan
  kapalı). Mekanizma store'da kalır; `off`, ikinci bir kod yoluyla değil hiç
  karar üretmeyerek v0.5 davranışını yeniden üretir.
- **İçerik ve yerleşim ayrı revision eksenleridir.** Byte'ları koruyan,
  checksum ile doğrulanmış bir taşıma artık uçuştaki prefetch'i iptal ettirmez;
  store'un başka bir yerindeki yazım migration'ı düşürmez.
- **Bütçelerin sahibi store'dur.** Policy önerir; store doğrular veya açıkça
  reddeder, reddedilenler yutulmak yerine sayılır.
- **Policy yalnız yerleşim seçer.** Precision tanımlıdır ama kapalıdır ve kuralı
  şudur: hassasiyet düşürme sayfa başına tek yönlüdür — q4 bir sayfayı f16'ya
  yükseltmek dequantize edilmiş q4 verir, orijinali değil; hiçbir policy bunu
  geri kazanılmış kalite diye raporlayamaz.
- **Attention yerleştirmez, yalnız kopyalar.** GPU'da olmayan sayfalar tek bir
  okuma için çağrıya özel scratch'e kopyalanır; bu yerleşimi, revision'ları ve
  erişim geçmişini hiç değiştirmez.

v0.6 ayrıca yukarıda ölçülen GPU-resident attention yolunu ekledi. 262K benchmark
baseline'ı, kalite toleransı ve başarı metriği hâlâ açık; çalıştırılmadı.

## v0.7 — ölçülen farkı kapatmak

Deneysel optimizasyon, her seferinde ölçülmüş tek bir nedensel faktör,
varsayılan olarak exact: önce GPU-resident 4K prefill'in stock'a olan farkı
(attention ve çevresindeki her şey), sonra policy açıkken çalışma zamanı
maliyeti, sonra decode.

## Bilinen sınırlar

- Native HuggingFace decode yalnız doğrulanmış Qwen2 full-attention
  sözleşmesini kapsar. Diğer modeller ve maskeli/yerel attention durumları K/V'yi
  yeniden kurar.
- Streaming attention tek bir fused kernel değil, bir ATen işlem dizisidir; bu
  yüzden sayfa başına launch maliyeti taşımaya devam eder.
- `DirectPagedAttentionEngine` yalnız izole ölçülmüştür ve HuggingFace decode
  yoluna bağlı değildir.
- Python `PageStore` ile native store ayrıdır.
- llama.cpp resident attention hızlı yolu F16 KV ile prefill'i (Q>1) kapsar; en
  hızlı kernel head dimension 64 ister. Decode staged yolu kullanır.
- Gerçek bellek baskısı altında CPU-spill gecikmesi ve çok kullanıcılı
  throughput ölçülmemiştir.
- Öngörücü sayfalama deneyseldir ve varsayılan olarak kapalıdır.
- [1M token sentetik disk testi](docs/measurements/v050-disk-capacity-smoke-2026-09-15.json)
  tek katman, tek KV head ve head boyutu 4 kullanır. Gerçek modelde 1M context
  üretimi değildir ve NVMe performans kanıtı değildir.

## Kanıtları yeniden üretme

```bash
pytest tests/ -q
python benchmarks/bench_native_runtime.py --json native.json
python benchmarks/bench_jl_fidelity.py --json jl.json --tokens 512
python benchmarks/bench_downstream.py \
  --json downstream.json \
  --contexts 512 1024 2048 4096 8192 16384 \
  --new-tokens 64 --repeats 3 --warmups 1 --page-size 1024
```

Benchmark sınıfları bilerek ayrı tutulur:

- `reconstruction`: tensor codec hatası, model doğruluğu değil;
- `runtime`: kernel ve cache gecikmesi ile belleği;
- `downstream`: uçtan uca TTFT, TPOT, peak VRAM ve model skorlaması.

## Lisans

ARGUS [Apache 2.0 Lisansı](LICENSE) ile lisanslanmıştır.
