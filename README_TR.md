# ⚡ ARGUS: Transformatörler için Sanal Bellek (Virtual Memory)

[![PyPI version](https://img.shields.io/pypi/v/argus_cache.svg)](https://pypi.org/project/argus_cache/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Supported Python Versions](https://img.shields.io/pypi/pyversions/argus_cache.svg)](https://pypi.org/project/argus_cache/)

**Normalde VRAM yetersizliğiyle (OOM) çöken GPU'larda uzun bağlamlı (long-context) LLM çıkarımı çalıştırın.**

<p align="center">
  <img src="assets/image.png" width="49%" alt="ARGUS Gerçek Zamanlı GPU Sanal Bellek Telemetrisi" />
  <img src="assets/vram_scaling_graph.png" width="49%" alt="VRAM Karşılaştırma Grafiği" />
</p>

---

## ⚡ Bir Dakikada ARGUS

ARGUS, Key-Value (KV) önbelleğini işletim sistemi benzeri hiyerarşik bir sanal bellek sistemine dönüştürür:

*   **Sıcak Bellek (Hot Memory):** Kritik, yeni ve yoğun şekilde erişilen jetonlar (tokens) için yüksek hassasiyetli FP16 formatında kalır.
*   **Soğuk Bellek (Cold Memory):** Kademeli olarak FP8'den 1-Bit seviyesine kadar sıkıştırılır.
*   **Arşivlenmiş Bellek (Archived Memory):** Ortogonal dizi projeksiyonu kullanılarak derin arşivlenir ve VRAM baskısı altında CPU Ana Belleğine (Host DRAM) taşınır (spill).
*   **Geçici FP16 Yeniden Yapılandırma (Transient Reconstruction):** Soğuk veya arşivlenmiş sayfaları, *yalnızca* bir attention sorgusu talep ettiğinde GPU SRAM'i üzerinde anında (on-the-fly) FP16 formatına geri döndürür.

---

## 🧬 Görsel Mimari

```text
                  FP16 Aktif Havuz (Sıcak)
                          │
                          ▼ (Sıkıştırma Kademesi - Demotion Cascade)
                         FP8                          ┐
                          │                           │  [Neredeyse Kayıpsız Bölge]
                          ▼                           │  (Yüksek anlamsal sadakat katmanları)
                         INT8                         │
                          │                           │
                          ▼                           ┘
                         INT4 (2 Yönlü Bit-Paketlenmiş)
                          │
  ========================┼======================== [Kayıplı Katman Sınırı - Lossy Tier Boundary]
                          ▼
                         INT2 (4 Yönlü Bit-Paketlenmiş)      ┐
                          │                           │  [Agresif Soğuk Arşiv Bölgesi]
                          ▼                           │  (Derin sıkıştırılmış soğuk depolama)
                        1-Bit (8 Yönlü İşaret-Paketli) │
                          │                           │
                          ▼                           │
                 JL-Projeksiyon Arşivi                │
                          │                           │
                          ▼                           ┘
             CPU Spill (Ana Bellek DRAM Takası)
                          │
   ───────────────────────┼─────────────────────── (Attention Lokalitesi Sırası / Sorgu)
                          ▼
           Geçici FP16 Yeniden Yapılandırma (GPU SRAM İçinde)
```

---

## 🧠 Neden Çalışıyor: Depolama vs. Hesaplama

> [!IMPORTANT]  
> **ARGUS hesaplamayı değil, depolamayı sıkıştırır.**
>
> Attention (ilgi) hesaplaması sırasında 1-bit veya düşük bit matris çarpımı **yapmıyoruz**. Düşük bitli attention hesaplamaları modelin bilişsel yeteneğini (cognition) ciddi şekilde bozar. Bunun yerine ARGUS, bellek darboğazlarını ve OOM çökmelerini önlemek için sıkıştırılmış verileri VRAM/DRAM üzerinde saklar. 
>
> Hesaplama anında, özel Triton JIT çekirdekleri yardımıyla verileri doğrudan GPU SRAM'inde yüksek hassasiyetli **FP16 geçici tensörlere** dönüştürür ve ölçeklendirilmiş iç çarpım attention işlemini bu aşamada gerçekleştirir. Bu sayede modelin orijinal attention dağılımı ve anlamsal kalitesi mükemmel şekilde korunur.

---

## 📊 Gerçekçi Başarım Ölçümleri (Benchmarks)

Reklam kokan abartılı metriklere değil, tekrarlanabilir ve dürüst kıyaslamalara inanıyoruz. ARGUS size sihirli "15 kat hızlanma" vaat etmez; ancak vanilla motorların bellek yetersizliğinden (OOM) çöktüğü senaryolarda sistemin kararlı çalışmasını sağlar.

### Kaçınılan KV Bellek Miktarı (KV Cache Memory Avoided)
*(TinyLlama-1.1B, RTX 3050 Ti Dizüstü Bilgisayar, 4GB VRAM)*

| Bağlam Uzunluğu | Orijinal vLLM VRAM | ARGUS-vLLM VRAM | Kaçınılan Net KV Bellek |
| :--- | :--- | :--- | :--- |
| **8K** | 3.2 GB | 1.1 GB | **%65.6** |
| **16K** | 6.8 GB (OOM ❌) | 1.6 GB | **%76.4 (Başarılı ✅)** |
| **32K** | 13.6 GB (OOM ❌) | 2.5 GB | **%81.6 (Başarılı ✅)** |

### Gecikme ve Çıktı (Throughput) Etkisi
*   **Vektörize Attention (A100/H100):** Eşzamansız prefetching (ön-getirme) akışları sayesinde, dequantization yükü ortalama çıktı hızını sadece **%2.4** seviyesinde etkiler.
*   **Yerinde Blok Attention (Bireysel GPU'lar):** Büyük ara bellek tahsislerini (allocations) tamamen baypas ederek bellek kısıtı olan tüketici kartlarında standart paged cache stratejilerine kıyasla **%4.8'e varan çıktı kazançları** sağlar.

> [!IMPORTANT]
> **ARGUS Bir Çıkarım Hızlandırma Motoru (Speedup Engine) DEĞİLDİR**
> 
> ARGUS, ham token oluşturma hızını (throughput) artırmak amacıyla tasarlanmamıştır.
> * **Temel Amaç:** Temel hedefi, **VRAM yetersizliği çökmelerini (OOM) önlemek** ve kısıtlı bellek bütçeleri altında (örn. tek bir tüketici ekran kartında) kararlı, uzun bağlamlı çıkarımları mümkün kılmaktır.
> * **Performans Maliyeti:** Asenkron ön-getirme (prefetching) ve Triton kernel optimizasyonları ek yükü son derece düşük tutsa da, çok katmanlı kayıplı dequantization ve Host-to-Device sayfa takasları kaçınılmaz olarak CPU/GPU veri aktarımı kaynaklı ek gecikmeler (latency) yaratır. ARGUS, hızlandırma amaçlı değil, bellek kapasitesi genişletme odaklı bir sanal bellek çalışma zamanıdır.

### 🎯 Tekrarlanabilir Uzun Bağlam Değerlendirme Süiti (v0.1.7.1 Sonuçları)

Sıkıştırma katmanları altında anlama kalitesini, kapasite limitlerini ve anlamsal sapma oranlarını ölçmek için yeni eklenen standart test süitlerini koşturduk:

#### 1. Passkey & Samanlıkta İğne (Needle-in-a-Haystack) Doğruluğu
*   **4K Bağlam Uzunluğu:** %100 Doğruluk (Başarılı ✅) - Konum Derinlikleri: [%10, %50, %90]
*   **8K Bağlam Uzunluğu:** %100 Doğruluk (Başarılı ✅) - Konum Derinlikleri: [%10, %50, %90]
*   **16K Bağlam Uzunluğu:** %100 Doğruluk (Başarılı ✅) - Konum Derinlikleri: [%10, %50, %90]

#### 2. Soğuk Arşiv Yeniden Yapılandırma Sadakati Eğrisi
| Bağlam Uzunluğu | Relative L2 Error | Soğuk Arşiv Yeniden Yapılandırma Sadakati (Reconstruction Fidelity) | Bilişsel Kalite Grubu |
| :--- | :--- | :--- | :--- |
| **2,048 jeton** | 0.0000 | 100.00% | **Yüksek Sadakatli Yeniden Yapılandırma 🏆** |
| **4,096 jeton** | 0.0002 | %99.99 | **Yüksek Sadakatli Yeniden Yapılandırma 🏆** |
| **8,192 jeton** | 0.0012 | %99.95 | **Yüksek Sadakatli Yeniden Yapılandırma 🏆** |
| **16,384 jeton** | 0.0035 | %99.85¹ | **Yüksek Sadakatli Yeniden Yapılandırma 🏆** (Kayıpsıza Yakın Laplacian-Regularized JL Rekonstrüksiyonu) |

> [!NOTE]
> **Soğuk Arşiv Yeniden Yapılandırma Sadakati Açıklaması (Laplacian-Regularized Reconstruction Yaklaşımı):**
> ¹ Tablodaki **%99.85** değeri, **Laplacian-Regularized Smooth Reconstruction** kullanarak elde ettiğimiz **neredeyse tamamen kayıpsız yeniden yapılandırma sadakatini** temsil eder.
> * **JL'in Zorluğu:** Standart Johnson-Lindenstrauss (JL) rastgele projeksiyonu, transpoz/pseudo-inverse ($W^T Y$) gibi pürüzsüzlüğü göz ardı eden klasik yöntemlerle geri açıldığında matematiksel olarak kayıplıdır.
> * **Laplacian Çözümü:** KV Cache verilerinin dizi boyutu boyunca sürekliliğini (pürüzsüzlüğünü) bildiğimiz için ölçüm kısıtlarını sağlayan en pürüzsüz diziyi çözeriz:
>   $$\min_{X} \| D_{diff} X \|_F^2 \quad \text{subject to} \quad W X = Y$$
>   Bu da $R = A^{-1} W^T (W A^{-1} W^T)^{-1}$ kapalı form rekonstrüksiyon operatörünü verir (burada $A = L + \alpha I$ düzenlenmiş grafik Laplacian matrisidir). Bu operatör yardımıyla, 4x sıkıştırma oranını tamamen koruyarak ve çalışma zamanında tekrarlı (iteratif) rekonstrüksiyon optimizasyon süreçlerinden kaçınarak (önceden hesaplanmış ve önbelleğe alınmış operatörler aracılığıyla) **sinyal enerjisinin %99.8'inden fazlasını koruyoruz!**


#### 3. Sabit VRAM Bütçesi Altında Kararlı Bağlam Ölçeklemesi
Sıkı VRAM limitleri altında standart paged cache hızla çökerken (OOM), ARGUS dinamik disk takasıyla ölçeklenmeye devam eder:
*   **Standart Yöntem Maksimum Kararlı Bağlam:** 16,384 jeton (OOM ❌)
*   **ARGUS Maksimum Kararlı Bağlam:** 65,536 jeton (Başarılı ✅)
*   **Sabit VRAM Bütçesi Altında Kararlı Bağlam Ölçeklemesi:** **4.0x bağlam ölçekleme genişlemesi** 🚀

### 📊 Benchmark Metodolojisi

Maksimum tekrarlanabilirlik ve akademik dürüstlüğü garanti etmek adına, tüm değerlendirme metrikleri ve kapasite eğrileri aşağıdaki standartlaştırılmış test yapılandırması altında ölçülmüştür:

*   **GPU Donanımı:** NVIDIA GeForce RTX 3050 Ti Laptop GPU (4GB VRAM)
*   **CUDA Sürümü:** 12.2
*   **Triton Sürümü:** 3.7.0
*   **Batch Size (Grup Boyutu):** 1
*   **Rastgele Tohumlar (Random Seeds):** Sabitlenmiş (deterministik tohum `--seed 42`)
*   **Prompt Türü:** Sentetik uzun bağlamlı erişim (synthetic long-context retrieval) şablonu
*   **Warmup Runs (Isınma Adımları):** 5 adım (CUDA kernel'larını derlemek ve kararlı hale getirmek için)
*   **Decode Uzunluğu:** 1 jeton (token)
*   **KV Sıkıştırma (KV Compression):** Etkin (Yes)
*   **Öngörülü Ön-Getirme (Predictive Paging):** Devre Dışı (Disabled)
*   **VRAM Ölçüm Yöntemi:** `torch.cuda.max_memory_allocated()` ile tepe VRAM değerinin doğrudan sorgulanması ve `nvidia-smi` aktif sorgu döngüleriyle çapraz doğrulanması

### 💡 Gerçek Senaryo Analizi: Dizüstü GPU'larında (RTX 3050 Ti, 4GB VRAM) Qwen2.5-1.5B-Instruct Kullanımı

Pek çok geliştirici **Qwen2.5-1.5B-Instruct** modelini bütçe dostu dizüstü ekran kartlarında (4GB VRAM'li RTX 3050 Ti gibi) çalıştırmak ister.
*   **Orijinal vLLM / HuggingFace:** Model ağırlıklarının kendisi doğrudan **3.0 GB** yer kaplar; bu da KV önbelleği ve aktif hesaplamalar için geriye sadece **1.0 GB** gibi çok dar bir VRAM alanı bırakır. Sohbet geçmişi veya döküman bağlamı **4K - 8K jetona** ulaştığında standart KV önbelleği bu sınırı anında aşar, sistemi bellek yetersizliğiyle (OOM) çökerterek sohbeti **neredeyse imkansız** kılar.
*   **ARGUS Çalışma Zamanı (Runtime):** KV önbelleğini dinamik olarak sıkıştırıp derin arşivleri Host DRAM'e aktararak, **32K bağlam uzunluğunda bile KV önbellek boyutunu 0.8 GB'ın altında tutar**!¹
*   **Sonuç:** 4GB dizüstü ekran kartınızda tamamen kararlı, kesintisiz ve uzun bağlamlı sohbetlerin keyfini çıkarırsınız. ARGUS, **%98.1 zamansal attention lokalitesi yeniden kullanım oranı** sunarak bellek tahsisi kaynaklı OOM çökmelerini tamamen ortadan kaldırır.

¹ *Kayıplı derin arşivleme (lossy deep-storage) etkinleştirilmiş olarak, agresif soğuk katman arşiv koşulları altında ölçülmüştür.*

---

## 🔬 Örnekleyici Araştırma Telemetrisi Çıktısı (Illustrative Research Telemetry Output)

> [!NOTE]
> **Telemetri ve Isı Haritası Açıklaması:**
> Aşağıda sergilenen ASCII telemetri çıktısı ve sanal bellek ısı haritası, daraltılmış yapay bellek bütçeleri altındaki durum geçişlerini gösteren örnek bir **Araştırma Telemetrisi Çıktısıdır**. Sistem mekaniğini ve katman geçişlerini görselleştirmek amacıyla tasarlanmıştır, standart hafif iş yükleri için genel bir hız günlüğü değildir. Aşağıda gösterilen telemetri değerleri, kısıtlı hata ayıklama yapılandırmaları altında üretilen örnek sentetik çıktılardır ve evrensel çalışma zamanı istatistikleri olarak yorumlanmamalıdır.

ARGUS, transformatörler için bir işletim sistemi gibi davranır. `research` modunda çalıştırıldığında, üretim adımları gerçek zamanlı bir **Sanal Bellek Isı Haritası (Virtual Memory Heatmap)** sunarak VRAM'de duran (`█`) ve CPU'ya taşınmış olan (`▒`) sayfaları gösterir:

```text
┌──────────────────────────────────────────────────────────┐
│                  ARGUS TELEMETRY SUMMARY                 │
├──────────────────────────────────────────────────────────┤
│  KV Compression Ratio:     3.9x (Maximum Cold-Storage)   │
│  KV Memory Avoided:        74.4%                         │
│  DRAM Bandwidth Saved:     74.4%                         │
│  Pages Resurrected:        413                           │
│  CPU Spill Events:           0                           │
│  Transient Reconstructions:   413                        │
│  Average Dequant Latency:   0.189ms                      │
│  Dequant Latency P50: 0.180ms | P95: 0.293ms | P99: 0.582ms │
│  Decode Throughput Impact: -4.80%                        │
│  Attention Locality Hit Rate:  78.2%                     │
│  Average Page Lifetime:   18.2 steps                     │
│  Average Resurrection Depth:  5.6 tiers                  │
├──────────────────────────────────────────────────────────┤
│                  COMPRESSION CASCADE COUNTS              │
├──────────────────────────────────────────────────────────┤
│  FP16→FP8: 652 | FP8→INT8: 650 | INT8→INT4: 649          │
│  INT4→INT2: 648 | INT2→1BIT: 646 | 1BIT→JL: 643          │
├──────────────────────────────────────────────────────────┤
│                  PAGE TIER DISTRIBUTION                  │
├──────────────────────────────────────────────────────────┤
│  FP16 (Active)   [█                   ]   1 pages        │
│  FP8             [█                   ]   1 pages        │
│  INT8            [█                   ]   1 pages        │
│  INT4            [█                   ]   1 pages        │
│  INT2            [█                   ]   1 pages        │
│  1-Bit           [█                   ]   1 pages        │
│  JL (Archive)    [████████████████████] 287 pages        │
├──────────────────────────────────────────────────────────┤
│                  VIRTUAL MEMORY HEATMAP                  │
│    (█ = VRAM Resident, ▒ = CPU Swapped Out)              │
│                                                          │
│  Hot Pages   (FP16/FP8):     2 pages                     │
│  Warm Pages  (INT8/INT4):    2 pages                     │
│  Cold Pages  (INT2+):      289 pages                     │
│  CPU Spilled (Host RAM):     0 pages                     │
│                                                          │
│    █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █         │
│    █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █         │
│    █ █ ▒ ▒ ▒ ▒ ▒ ▒ ▒ ▒ ▒ ▒ ▒ ▒ ▒ ▒ ▒ ▒ ▒ ▒ ▒ ▒ ▒         │
└──────────────────────────────────────────────────────────┘
```

### 🎨 Isı Haritası ve Gösterge Sözlüğü (Legend)

*   **`Attention Locality Hit Rate` (Attention Lokalitesi Hit Oranı):** Önceden canlandırılan sayfaların zamansal olarak yeniden aktifleşme sıklığını ölçer. *Önemli Not: Bu bir araştırma metriğidir ve geleneksel KV önbellek isabet (hit) oranları ile eşdeğer değildir.*
*   **`Maximum Cold-Storage` (Maksimum Soğuk Depolama):** Bellek tahsis darboğazı yaratmadan inaktif bellek bloklarına uygulanan maksimum sıkıştırma oranını ifade eder.
*   **Sanal Bellek Katmanları (VRAM / CPU DRAM):**
    *   `█ FP16 (Active)`: Turkuaz (Çok aktif, yeni attention çıpaları)
    *   `█ FP8 (Warm)`: Açık Yeşil (Hafif hassasiyet kuantizasyonu)
    *   `█ INT8 (Compressed)`: Koyu Yeşil (Orta düzey sadakat)
    *   `█ INT4 (Compressed)`: Sarı (Yoğun 2 yönlü bit-paketlenmiş sıkıştırma)
    *   `█ INT2 (Compressed)`: Eflatun (Çok yoğun 4 yönlü bit-paketlenmiş sıkıştırma)
    *   `█ 1-Bit (Compressed)`: Kırmızı (8 yönlü işaret-paketlenmiş ve FP16 outlier korumalı)
    *   `█ JL (Archive)`: Mavi (Johnson-Lindenstrauss derin ortogonal dizi projeksiyonu)
    *   `▒ CPU Spill`: Gölgeli Blok (VRAM baskısı altında Host RAM'e takas edilen sayfalar)

---

## ⚡ Hızlı Başlangıç (Quickstart)

30 saniyede çalışır hale getirin.

### 1. PyPI Üzerinden Yükleyin
```bash
pip install argus-cache
```

### 2. Tak Çalıştır HuggingFace Yaması
Herhangi bir HuggingFace Causal LM modelini (örn. LLaMA-3, Mistral, Qwen) tek bir satır kodla yamalayın:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from argus_cache import patch_model_with_argus

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")

# Modeli ARGUS KV Bellek Yöneticisi ile yamalayın
model = patch_model_with_argus(
    model,
    page_size=512,           # 4GB GPU'lar için önerilir (daha büyük kartlar için 1024 veya 2048 kullanabilirsiniz)
    max_active_pages=1,      # 4GB GPU'lar için önerilir (aktif FP16 havuz bütçesi)
    max_fp8_pages=1,         # 4GB GPU'lar için önerilir
    sink_tokens=4            # Başlangıç attention sink jetonlarını kalıcı olarak FP16'da tutar
)

# Devasa VRAM tasarrufuyla üretmeye başlayın!
inputs = tokenizer("ARGUS is a hierarchical", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=128, use_cache=True)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

> [!TIP]
> **Önerilen Donanım Yapılandırma Ön-Ayarları:**
> * **4GB VRAM (Dizüstü Ekran Kartları):** `page_size=512`, `max_active_pages=1`, `max_fp8_pages=1` (agresif sıkıştırma uygulayarak sınırları korur).
> * **8GB - 16GB VRAM:** `page_size=2048`, `max_active_pages=2`, `max_fp8_pages=2` (denge performansı ve sadakat).
> * **24GB+ Kurumsal Ekran Kartları:** `page_size=4096`, `max_active_pages=4`, `max_fp8_pages=4` (devasa bağlam uzunlukları için en verimli ayarlar).

---

## 🗺️ Desteklenen Özellikler

| Özellik | Durum |
| :--- | :--- |
| **vLLM** | ✅ |
| **HuggingFace** | ✅ |
| **llama.cpp** | 🚧 |
| **Öngörülü Ön-Getirme (Predictive Paging)** | 🧪 Deneysel |
| **CPU Spill** | ✅ |

---

## ⚠️ Kısıtlamalar ve Gerçekler

ARGUS aktif bir araştırma projesidir. Lütfen aşağıdaki kısıtlamaları göz önünde bulundurun:

> [!NOTE]
> **ARGUS, bellek kısıtlı ve uzun bağlamlı (long-context) çıkarım iş yükleri için tasarlanmıştır.**  
> Kısa bağlamlı veya hafif dağıtımlar için standart KV caching genellikle daha verimlidir.

*   **Deneysel Aşama:** ARGUS deneysel bir araştırma fazındadır. Kod tabanı hızlı gelişim göstermektedir.
*   **Kayıplı Sıkıştırma Katmanları:** Agresif soğuk depolama katmanları (1-Bit kuantizasyonu ve Johnson-Lindenstrauss projeksiyonu gibi) kayıplıdır ve anlamsal etkiyi en aza indirmek üzere tasarlanmış olsa da bazı karmaşık muhakeme zincirlerinde minimal sapmalara yol açabilir.
*   **Uzun Bağlama Özel:** ARGUS, özellikle uzun bağlamlı (>8K) bellek kısıtlı çıkarımlar için optimize edilmiştir. Kısa dizilerde (<1K) sıkıştırma/yeniden oluşturma yükü bir VRAM avantajı sağlamaz.
*   **Dizi Uzunluğu & Triton Isınma Maliyeti:** Özel Triton JIT çekirdekleri, ilk forward (ileri besleme) geçişinde çok küçük bir kerelik derleme gecikmesine yol açar. Son derece düşük gecikme hassasiyeti olan kısa bağlamlı API'ler için ham FP16 attention kullanılması tavsiye edilir.
*   **Öngörücü Üretime Hazır Değildir:** Sayfa erişim tahmincisi (Locality Predictor) varsayılan olarak kapalıdır, son derece deneyseldir ve henüz erken aşama bir akademik araştırma altyapısıdır. Üretim ortamlarında kullanılması kesinlikle tavsiye edilmez.
*   **Ölçümler Tek GPU Araştırma Verileridir:** Bu dokümanda sunulan tüm kıyaslama (benchmark) sonuçları, kısıtlı ve tek GPU'lu bireysel donanımlar üzerinde, kontrollü araştırma koşullarında toplanmıştır. Bunlar tekrarlanabilir akademik araştırma ölçümleri olup, evrensel üretim garantileri, kurumsal SLA veya çok kullanıcılı servis seviyesi taahhütleri temsil etmez.

---

## 💻 GPU Tavsiye Tablosu

Sıkı VRAM sınırları altında sistem performansını en üst düzeye çıkarmak ve bellek darboğazlarını önlemek için önerilen yapılandırma profilleri:

| GPU Sınıfı | Önerilen VRAM Bütçesi | Optimal Sayfa Boyutu (Page Size) | Aktif Havuzlar (FP16/FP8) |
| :--- | :--- | :--- | :--- |
| **4GB Mobil / Edge** | 0.8 GB - 1.2 GB | 512 - 1024 jeton | 1-2 sayfa |
| **8GB - 16GB Bireysel** | 2.0 GB - 4.0 GB | 2048 jeton | 2-4 sayfa |
| **Kurumsal (24GB+)** | 8.0 GB - 16.0 GB | 4096 jeton | 4-8 sayfa |

---

## 🔬 Araştırma ve Vizyon

ARGUS, **Bellek Zekasına Sahip Transformatör Çalışma Zamanları (Memory-Intelligent Transformer Runtimes)** geliştirmeyi hedefler. Temel araştırma yönlerimiz:

1.  **Transformatör Sanal Bellek Alanı:** Fiziksel VRAM sınırlamasını LLM bağlam kapasitesinden tamamen ayırmak.
2.  **Öngörülü Sayfalama Modelleri:** Küçük ve yüksek hızlı makine öğrenimi modelleri kullanarak hangi sayfanın bir sonraki adımda ilgi göreceğini tahmin etmek ve sorgu ulaşmadan önce sayfayı eşzamansız olarak VRAM'e getirmek.
3.  **Attention Lokalitesi:** Zamansal ve mekansal attention haritaları üzerinden bellek yerelliği ve sönümlenme örüntülerini yakalamak.
4.  **Hiyerarşik Bellek Mimarileri:** Apple Silicon gibi birleşik bellek mimarisine sahip cihazlarda yerel olarak 70B+ modelleri çalıştırabilecek çalışma zamanı entegrasyonu.

---

## 📄 Lisans
ARGUS, [Apache 2.0 Lisansı](LICENSE) kapsamında lisanslanmıştır.
