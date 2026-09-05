# ARGUS: Heterojen KV önbellek bellek yönetimi

ARGUS, transformer KV önbellek sayfalarını farklı hassasiyet ve yerleşim
katmanlarında saklayan deneysel bir çalışma zamanıdır. Yeni sayfalar FP16'da
kalır; eski sayfalar yapılandırılabilir politikayla FP8, INT8, INT4, INT2,
1-bit, projeksiyon ve CPU katmanlarına indirilebilir.

ARGUS bir hızlandırma motoru değil, **kapasite odaklı bir araştırma
çalışma zamanıdır**. Güncel ağaç exact cache güvenli VRAM bütçesine sığıyorsa
ARGUS'u bypass eder. Zorunlu kapasite modu uzun bağlamlarda tepe VRAM'i
düşürebilir; HuggingFace decode gecikmesi ise düşük gecikmeli servise hazır
değildir.

İngilizce belge: [README.md](README.md)

## Bugün gerçekten kanıtlanan sonuç

Uçtan uca ölçüm Qwen2.5-0.5B-Instruct, 4 GB RTX 3050 Ti Laptop GPU, FP16,
batch 1, 64 üretilen token, üç tekrar, bir ısınma ve 1024 token ARGUS sayfası
ile yapıldı.

| bağlam | baseline VRAM | ARGUS VRAM | VRAM değişimi | baseline TPOT | ARGUS TPOT |
|---:|---:|---:|---:|---:|---:|
| 512 | 982.2 MiB | 988.2 MiB | +%0.6 | 18.62 ms | 28.69 ms |
| 1.024 | 1007.1 MiB | 1007.1 MiB | %0.0 | 18.55 ms | 30.21 ms |
| 2.048 | 1056.8 MiB | 1056.9 MiB | %0.0 | 19.24 ms | 30.85 ms |
| 4.096 | 1149.4 MiB | 1137.4 MiB | -%1.0 | 19.14 ms | 40.21 ms |
| 8.192 | 1340.4 MiB | 1280.5 MiB | **-%4.5** | 17.85 ms | **52.51 ms** |
| 16.384 | 1722.5 MiB | 1590.6 MiB | **-%7.7** | 18.84 ms | **79.78 ms** |

16K'da TTFT baseline için 1.819 saniye, ARGUS için 3.992 saniyedir. Sonuç:

- VRAM tasarrufu gerçektir ve bu deneyde bağlamla birlikte büyümektedir.
- Ana açık sorun decode gecikmesidir: 16K TPOT baseline'ın 4.2 katıdır.
- Baseline'ın OOM olup ARGUS'un çalıştığı bir satır henüz yoktur. “OOM'u
  önler” ve “daha uzun kullanılabilir bağlam sağlar” iddiaları kanıtlanmamıştır.
- Perplexity farkı -0.0176'dır; fakat yalnızca kayba çok yakın FP8 katmanı
  çalışmıştır. Bu sayı INT4, INT2, 1-bit veya JL kalitesini doğrulamaz.

Ham ölçüm, min/max süreler ve komut:
[`docs/measurements/downstream-2026-08-14.json`](docs/measurements/downstream-2026-08-14.json).
Ayrıntılı değerlendirme:
[`docs/findings-2026-08-14.md`](docs/findings-2026-08-14.md).
Düşük context bypass, dengeli aktivasyon politikası ve daha hafif decode
seçenekleri:
[`docs/optimization-notes-2026-08-14-tr.md`](docs/optimization-notes-2026-08-14-tr.md).

## Adaptif aktivasyon

`AdaptiveCachePolicy`, gözlenen batch/KV head/head dimension/dtype değerleri,
model katman sayısı ve beklenen token sayısından exact KV maliyetini hesaplar.
`balanced` modu yalnız hem minimum bellek kazancı hem yüksek VRAM su seviyesi
kapısı geçildiğinde ARGUS'u açar. Açılan bir istek decode ortasında geri
kapanmaz; istekler arasında ayrı açma/kapatma eşikleri histerezis sağlar.

| mod | davranış |
|---|---|
| `latency` | her zaman exact-cache bypass |
| `balanced` | kapasite ve baskı kapıları geçilene kadar exact bypass |
| `capacity` | kontrollü kapasite deneyi için ARGUS'u zorla |

Varsayılan dengeli servis hattı `ACTIVE → FP8`'dir. Eski derin zincir
`pipeline_profile="research"` ile kullanılabilir.

## Gecikme neden yüksek?

ARGUS depolamayı sıkıştırır; attention hesabını düşük bitte çalıştırmaz. Güncel
ağaç, doğrulanmış model sözleşmelerinde tam K/V reconstruction'ını atlamak için
Transformers'ın fonksiyonel `AttentionInterface` noktasını kullanır. İlk native
sözleşme Qwen2 full-attention, maskesiz, tek-token decode yoludur. Prefill,
padding/local maskeler ve training reconstruct+SDPA'ya fail-closed düşer.
Kayıtsız mimariler kendi model attention yolunu korur ve reconstruct edilmiş K/V alır.

Native cache artık sıkıştırılmış sayfaları tek tek tüketen exact online-softmax
prototipine sahiptir; FP16 çalışma alanı bir sayfayla sınırlıdır ve SDPA/GQA
eşitlik testleri vardır. Yol henüz tek CUDA/Triton kernelinde füze edilmedi;
bu yüzden uçtan uca hızlanma iddiası değildir.

Model farkları açık bir registry ile yönetilir: `AttentionAdapter`, her
`config.model_type` için native uygunluğu, query hazırlığını ve çıktı düzenini
tanımlar. Uygulamalar `register_attention_adapter()` ile yeni sözleşme
ekleyebilir; kayıtlı olmayan model native sayfalı attention'a yanlışlıkla girmez.

## Mimari

```text
HuggingFace model
       |
       v
PagedDynamicQuantizedCache
       |
       +-- Python: politika ve yapılandırma
       +-- C++: sayfa yaşam döngüsü ve katman geçişleri
       +-- CUDA/Triton: codec ve attention deneyleri
       +-- sabitlenmiş CPU belleğine arşivleme
```

Sıkıştırma katmanları eklentidir. Önbellek yöneticisi katman adlarına göre
hardcode edilmiş dallar yerine yetenek ve sayısal codec metadatası kullanır.
Ayrıntılar: [`docs/architecture.md`](docs/architecture.md).

## Doğrudan sayfalı attention

0.4.0 ile geldi. Structure-of-Arrays sayfa tablosu (`core/page_table.py`)
*hassasiyeti* (`ACTIVE_FP16`, `GGML_Q8_0`, `GGML_Q4_0`) *yerleşimden*
(`GPU_DEVICE`, `HOST_PINNED`, `HOST_PAGEABLE`) ayırıp tek bir bitişik
tanımlayıcı tablosunda tutuyor; böylece decode sıcak yolu Python nesneleri
yerine bir dizi üzerinde yürüyor. Aynı codec'e ait sayfalar sayfa başına değil,
bitişik bir blok havuzundan tahsis ediliyor. `DirectPagedAttentionEngine` ise
bu yapıların üzerinde doğrudan, karo karo online-softmax özyinelemesi
çalıştırıyor: bağlam boyutunda bir FP16 KV tensörü hiç oluşturulmuyor, yeniden
kurulum tek bir sayfa karosuyla sınırlı kalıyor.

RTX 3050 Ti Laptop üzerinde, Qwen benzeri geometriyle (24 sorgu başlığı,
4 KV başlığı, head_dim 256, sayfa 128) yalıtılmış ölçüm:

| bağlam | ACTIVE_FP16 | GGML_Q8_0 | GGML_Q4_0 |
|---:|---|---|---|
| 1.024 | 1,98 ms / 12,15 MiB | 3,15 ms / 10,28 MiB | 4,24 ms / 9,28 MiB |
| 4.096 | 7,51 ms / 24,15 MiB | 11,43 ms / 16,65 MiB | 15,86 ms / 12,65 MiB |
| 8.192 | 14,71 ms / 40,15 MiB | 22,55 ms / 25,15 MiB | 31,61 ms / 17,15 MiB |
| 16.384 | 29,26 ms / 72,15 MiB | 44,97 ms / 44,15 MiB | 62,77 ms / 26,15 MiB |
| 32.768 | 58,48 ms / 136,16 MiB | 90,15 ms / 76,16 MiB | 125,25 ms / 44,16 MiB |

Takas monoton ve dik. 32K'da q4_0 bağlamı 44,16 MiB'de tutuyor, FP16 ise
136,16 MiB'de — 3,1 kat az bellek, 2,14 kat gecikme. Bu **runtime** sınıfı bir
ölçüm; uçtan uca servis sonucu değil. Gecikme sütunu, ARGUS'un neden hâlâ bir
servis hızlandırıcısı olmadığının cevabıdır.

Kanıt: [`docs/measurements/v040-fused-attention-benchmark.json`](docs/measurements/v040-fused-attention-benchmark.json).

## Hibrit mimariler

`models/hybrid_cache.py`, tam ve doğrusal attention'ı karıştıran modellerde
(örneğin Qwen3.8 Gated DeltaNet) sahipliği açıkça beyan eder. Tam attention
katmanlarının büyüyen KV önbelleği ARGUS'undur; doğrusal attention
katmanlarının sabit boyutlu recurrent ve conv state'i ARGUS'un dokunacağı şey
değildir. Bu sınırın korunduğu, her önbellek işleminin etrafında deterministik
state özetleriyle kanıtlanır; desteklenmeyen bir katman rolü sayfalanmak yerine
güvenli biçimde reddedilir.

## Denenip olmayan şey

Yerel bir llama-server'a karşı (Qwen3.6-35B-A3B, q4_0 KV) 4K/16K/32K/64K'da bir
A/B taraması yapıldı; amaç ARGUS destekli önbelleği vanilla olanla
karşılaştırmaktı. **Kanıt dosyasındaki denetim, ARGUS'un sürece hiç
yüklenmediğini gösteriyor:** `argus_in_llama_server_maps: false`,
`argus_maps_count: 0`. Her bağlamda iki kolun tepe VRAM'i bayt bayt aynı
(3594 / 3596 / 3598 MiB) — yani iki kol da aynı llama-server'dı.

Ortaya çıkan decode farkı (16K'da 17,66'ya karşı 11,88 tok/s) ARGUS'tan değil,
gateway'in prompt önek önbelleğinden geliyor: vanilla kol tüm prompt'u yeniden
işlerken TTFT 99,88 ms'ye düşüyor. Bu koşudan hiçbir ARGUS iddiası
çıkarılmıyor. Depoda durmasının sebebi şu: yanlış atfedilmiş bir kazanç, tam da
sorgulanmadan hayatta kalan sonuç türüdür.

ARGUS'un **llama.cpp entegrasyonu yoktur**. Yanında yayımlanan taramalar
(`load-mode-comparison`, `pmin-sweep`, `speculative-sweep-n2-n3-n4`) llama.cpp
runtime ayarıdır ve öyle etiketlenmiştir.

Kanıt: [`docs/measurements/argus-ab-cache-comparison-2026-09-04.json`](docs/measurements/argus-ab-cache-comparison-2026-09-04.json).

## Çalışma zamanı durumu

| çalışma zamanı | durum | ARGUS neyi yönetiyor? |
|---|---|---|
| HuggingFace Transformers | Ölçülmüş araştırma yolu | Modelin KV önbelleğini |
| Ollama | Canlı test edilmiş dış adaptör | Ollama içini değil; yalnızca ayar ve süreleri |
| vLLM | Kullanılamıyor, güvenli biçimde reddediyor | Hiçbir şeyi |
| llama.cpp | Entegre değil, denetimle doğrulandı | Hiçbir şeyi; "Denenip olmayan şey" bölümüne bakın |
| SGLang | Uygulanmadı | Hiçbir şeyi |

Eski vLLM entegrasyonu vLLM'in KV bloklarına sahip değildi ve onları
sıkıştırmıyordu. Güncel adaptör bu yüzden aktivasyonu bilerek reddeder. Gerçek
entegrasyon KV connector ve/veya özel attention backend üzerinden kurulmalıdır:
[`docs/vllm-verification.md`](docs/vllm-verification.md).

## Kurulum

```bash
pip install torch                                        # önce kurulu olmalı
pip install --no-build-isolation argus-cache             # çekirdek çalışma zamanı
pip install --no-build-isolation "argus-cache[gateway]"  # + Anthropic gateway
```

ARGUS kaynak dağıtımı olarak yayımlanır: native CUDA eklentisi kurulum anında
sizin makinenizde derlenir. Bu yüzden CUDA uyumlu bir PyTorch kurulumu, CUDA
araç zinciri ve C++17 destekli bir derleyici önceden hazır olmalıdır. Hazır
wheel yoktur — tek bir PyTorch ABI ve CUDA sürümüne göre derlenmiş bir ikili,
kurulumların çoğunda yanlış olurdu.

`--no-build-isolation` isteğe bağlı değil, zorunludur. Derleme, CUDA eklentisini
yapılandırmak için kurulu `torch`'unuzu okur; pip'in varsayılan izole derlemesi
onu gizler ve `ModuleNotFoundError: No module named 'torch'` ile başarısız olur.
pip'in geçici bir ortama indirdiği bir torch'a karşı derlemek ise başarısız
olmaktan daha kötü olurdu: eklenti, çalışma zamanınızda bulunmayan bir ABI için
derlenmiş olurdu.

ARGUS'un kendisi üzerinde çalışmak için depodan kurun:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
python setup.py build_ext --inplace
```

## HuggingFace örneği

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from argus_cache import AdaptiveCachePolicy, patch_model_with_argus

model_id = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.float16,
).to("cuda")

model = patch_model_with_argus(
    model,
    page_size=1024,
    max_active_pages=2,
    max_fp8_pages=2,
    sink_tokens=4,
    activation_policy=AdaptiveCachePolicy(
        mode="balanced",
        expected_tokens=16_448,
    ),
    pipeline_profile="balanced",
)

inputs = tokenizer("KV önbellek için sanal bellek", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=64, use_cache=True)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

`page_size=1024`, yalnızca test makinesinde ölçülen en dengeli varyanttır;
evrensel tavsiye değildir. Her model, GPU, bağlam dağılımı ve gecikme hedefi
için yeniden ölçülmelidir.

## Kanıtı yeniden üretme

```bash
pytest tests/ -q
python benchmarks/bench_native_runtime.py --json native.json
python benchmarks/bench_jl_fidelity.py --json jl.json --tokens 512
python benchmarks/bench_downstream.py \
  --json downstream.json \
  --contexts 512 1024 2048 4096 8192 16384 \
  --new-tokens 64 --repeats 3 --warmups 1 --page-size 1024
```

Ölçüm sınıfları birbirine karıştırılmaz:

- `reconstruction`: tensor codec hatasıdır, model doğruluğu değildir;
- `runtime`: çekirdek/önbellek gecikmesi ve bellektir;
- `downstream`: uçtan uca TTFT, TPOT, tepe VRAM ve model skorudur.

## Açık sınırlar ve sonraki kilometre taşı

- Decode her token'da tüm sıkıştırılmış sayfaları yeniden oluşturuyor. Ana
  üretim engeli budur.
- Kayıplı arşiv katmanları için perplexity ve retrieval testi gerekiyor.
- Gerçek bellek baskısı altında CPU spill gecikmesi ve çok kullanıcılı
  throughput ölçülmedi.
- OOM üstünlüğü, baseline'ın gerçekten cihaz belleğini aştığı daha büyük model
  veya daha uzun bağlam deneyi gerektiriyor.
- Predictive paging deneysel ve varsayılan olarak kapalıdır.
- `DirectPagedAttentionEngine` yalnızca yalıtılmış olarak ölçüldü. HuggingFace
  decode yoluna bağlanmadı; dolayısıyla sayıları henüz hiçbir uçtan uca
  sonuçta görünmüyor.
- llama.cpp entegrasyonu yoktur; tek deneme yukarıda negatif sonuç olarak
  yayımlanmıştır.
- q4_0 retrieval probu 31k token'da tek ve bağışlayıcı bir görevdir. Kuantize
  KV altında akıl yürütme, kod üretimi ve uzun menzilli tutarlılık ölçülmedi.

Bir sonraki anlamlı hedef yeni bir teorik sıkıştırma oranı değil; kanıtlanan
VRAM eğrisini koruyup TPOT'u baseline'a yeterince yaklaştıran sayfalı attention
entegrasyonudur.

## Lisans

ARGUS [Apache 2.0](LICENSE) ile lisanslanmıştır.
