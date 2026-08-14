# ARGUS: Heterojen KV önbellek bellek yönetimi

ARGUS, transformer KV önbellek sayfalarını farklı hassasiyet ve yerleşim
katmanlarında saklayan deneysel bir çalışma zamanıdır. Yeni sayfalar FP16'da
kalır; eski sayfalar yapılandırılabilir politikayla FP8, INT8, INT4, INT2,
1-bit, projeksiyon ve CPU katmanlarına indirilebilir.

ARGUS bir hızlandırma motoru değil, **kapasite odaklı bir araştırma
çalışma zamanıdır**. Mevcut HuggingFace entegrasyonu uzun bağlamlarda tepe VRAM
kullanımını düşürüyor; ancak decode gecikmesi henüz düşük gecikmeli servis için
uygun değil.

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

## Gecikme neden yüksek?

ARGUS depolamayı sıkıştırır; attention hesabını düşük bitte çalıştırmaz.
Mevcut HuggingFace yolu, her üretilen token için sıkıştırılmış sayfaları tekrar
FP16'a açar ve modelin beklediği bitişik K/V tensörünü oluşturur. Tekrarlanan
açma ve birleştirme maliyeti sayfa sayısıyla büyür.

Kalıcı çözüm, sayfalı önbelleği doğrudan tüketen ve yalnızca gereken blokları
attention çekirdeğinin içinde açan bir backend'dir. Sayfa boyutu ayarı maliyeti
azaltır, fakat bitişik tensör oluşturma maliyetini ortadan kaldıramaz.

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

## Çalışma zamanı durumu

| çalışma zamanı | durum | ARGUS neyi yönetiyor? |
|---|---|---|
| HuggingFace Transformers | Ölçülmüş araştırma yolu | Modelin KV önbelleğini |
| Ollama | Canlı test edilmiş dış adaptör | Ollama içini değil; yalnızca ayar ve süreleri |
| vLLM | Kullanılamıyor, güvenli biçimde reddediyor | Hiçbir şeyi |
| SGLang | Uygulanmadı | Hiçbir şeyi |

Eski vLLM entegrasyonu vLLM'in KV bloklarına sahip değildi ve onları
sıkıştırmıyordu. Güncel adaptör bu yüzden aktivasyonu bilerek reddeder. Gerçek
entegrasyon KV connector ve/veya özel attention backend üzerinden kurulmalıdır:
[`docs/vllm-verification.md`](docs/vllm-verification.md).

## Kurulum

Bu çalışma ağacındaki stabilizasyon değişiklikleri yayımlanmadı. Kaynak koddan
kurun:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
python setup.py build_ext --inplace
```

Native eklenti için CUDA uyumlu PyTorch ve çalışan bir derleyici zinciri
gerekir.

## HuggingFace örneği

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from argus_cache import patch_model_with_argus

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

Bir sonraki anlamlı hedef yeni bir teorik sıkıştırma oranı değil; kanıtlanan
VRAM eğrisini koruyup TPOT'u baseline'a yeterince yaklaştıran sayfalı attention
entegrasyonudur.

## Lisans

ARGUS [Apache 2.0](LICENSE) ile lisanslanmıştır.
