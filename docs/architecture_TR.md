# ARGUS Mimarisi (v0.7.0)

*English: [architecture.md](architecture.md)*

ARGUS bir KV-cache yöneticisidir. Her KV sayfasının nerede duracağına (GPU,
pinned RAM, pageable RAM, disk) katı bütçeler içinde karar verir ve attention
okumalarını sayfa neredeyse oradan karşılar; modelin aritmetiğini değiştirmez.
Bir quantization algoritması değil, cache belleği için bir kontrol katmanıdır.

Bu ilkeyi paylaşan ama kod paylaşmayan iki entegrasyon yolu vardır:

| yol | store | v0.7'deki durumu |
|---|---|---|
| **llama.cpp** (patch'li `libllama` / `ggml-cuda`) | `argus_cache/csrc/ggml_*` içindeki C++/CUDA disk store | Ürün yolu. CPU ve CUDA attention, kendi referansıyla bit-exact |
| **HuggingFace Transformers** | `PagedDynamicKVCache` (Python + `argus_cpp_backend`) | Araştırma yolu, en son v0.4'te ölçüldü |

Aşağıdaki her davranış iddiası, o davranış bozulursa kırılan testle biter; her sayı
kaydedildiği koşuya bağlanır.

Derleme, ayar ve çalıştırma: [README_TR](../README_TR.md). Her tasarım
kararının gerekçesi ve ölçümleri: [`plans/argus-v0.7.0.md`](../plans/argus-v0.7.0.md).

---

## 1. Harita

```text
┌───────────────────────── llama.cpp yolu ─────────────────────────┐
│ llama.cpp (host-kv.patch + cuda-kv.patch)                        │
│   KV buffer tipi │ set_rows │ attention op │ state kaydet/yükle  │
│        │             │           │                               │
│        v             v           v                               │
│ ggml_disk_buffer.cpp   store ömrü, GGML buffer, host I/O         │
│ ggml_disk_gpu.cpp      GPU'da tutma, yazımlar, write-back, taşıma│
│ ggml_kv_policy.cpp     bütçe içinde yerleşim kararları           │
│ ggml_cuda_attention.cu CUDA kernel'leri + dispatch               │
│ ggml_paged_attention.cpp  CPU blok attention                     │
│ ggml_host_buffer.cpp   map'lenmiş host KV (disk store yok)       │
│        │                                                         │
│   katmanlar: GPU │ pinned │ pageable RAM │ O_DIRECT disk (2 slot)│
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────── HuggingFace yolu ────────────────────────┐
│ Public API        argus_cache/__init__.py                        │
│ Model sözleşmeleri argus_cache/models/ (AttentionAdapter, hibrit)│
│ Politika          argus_cache/core/, plugins/                    │
│ Native motor      csrc/manager.cpp, tier_codec, zero_copy_pool   │
│ Kernel'ler        csrc/quantization_kernels.cu, Triton           │
└──────────────────────────────────────────────────────────────────┘
Runtime adapter'ları (kenarda, KV sahipliği yok): Ollama, vLLM (kapalı
başarısız olur), Anthropic-Messages ↔ OpenAI gateway.
```

---

## 2. llama.cpp yolu

### 2.1 ARGUS'un llama.cpp'ye bağlandığı yerler

Sabitlenmiş tek bir llama.cpp revizyonuna karşı iki patch
([`integrations/llama.cpp/`](../integrations/llama.cpp/README.md)):

| bağlantı noktası | patch | ARGUS'un devraldığı |
|---|---|---|
| KV buffer tipi (`llama_kv_cache` constructor) | host-kv | `ARGUS_KV_DIR` ayarlıysa K/V tensörleri ARGUS'un buffer tipinde ayrılır: disk store ya da map'lenmiş host buffer |
| `cpy_k` / `cpy_v` | host-kv | Yeni satırlar GGML'in `set_rows`'u yerine `argus_ggml_disk_set_rows`'tan geçer |
| attention grafı | host-kv | ARGUS tensörleri üzerinde flash attention yerine `argus_ggml_paged_attention` |
| state `write_tensor` / `read_tensor` | host-kv | Kaydet / yükle disk tensörlerini 4 KiB'lık staging bloğu ile akıtır, asla tam boy kopya yapmaz |
| CUDA custom op | cuda-kv | `ggml_backend_cuda_register_custom_op`: CUDA backend'inin device 0'da kabul ettiği, CUDA graph'a asla almadığı tek bir harici op (marker + compute + buffer tipi) |
| yüklemede geri dönüş | host-kv | ARGUS'un sunamayacağı modeller llama.cpp'nin kendi KV'sini kullanır (§2.9) |

ARGUS'un tüm CUDA işlemleri (attention, set_rows) tek bir kayıtlı marker'ı paylaşır;
op parametrelerindeki bir tür etiketiyle ayırt edilir.

Kanıt: [host ownership koşusu](measurements/v050-llama-host-ownership-2026-09-15.json) (logit farkı 0), [`tests/test_native_llama_paged.py`](../tests/test_native_llama_paged.py) içindeki uçtan uca sunucu testleri.

### 2.2 Store

K ya da V tensör buffer'ı başına bir store; **4 KiB'lık sayfalara** bölünür.

- **Arka dosya.** `ARGUS_KV_DIR` içinde oluşturulur, hemen unlink edilir ve
  `O_DIRECT` açılır: KV süreçten uzun yaşamaz ve page cache'ten geçmez.
- **Sayfa başına iki slot.** Yazım etkin olmayan slota gider, geri okunur,
  checksum ile doğrulanır (varsayılan CRC32C) ve ancak ondan sonra etkin olur. Yeni
  slot doğrulanana kadar önceki geçerli kalır. Direct disk modunun KV boyutunun 2
  katını ayırmasının sebebi budur.
- **Sayfa tanımlayıcısı.** `content_revision` (yazımda değişir), `placement_revision`
  (taşımada değişir), checksum, etkin slot, codec, erişim geçmişi ve CUDA
  derlemelerinde GPU'daki kopya ile `dirty` / `digest_pending` bayrakları.
- **İki revizyon ekseni.** Doğrulanmış, byte'ları koruyan bir taşıma okuyucuları
  geçersiz kılmaz; yazım kılar.
- **Bütçelerin sahibi store'dur.** Metadata (`ARGUS_KV_RESIDENT_BYTES`), host
  scratch (`ARGUS_KV_STAGING_BYTES`), GPU (`ARGUS_KV_GPU_BYTES`), pinned
  (`ARGUS_KV_PINNED_BYTES`), RAM (`ARGUS_KV_RAM_BYTES`). Aşım ya açık bir hatadır ya
  da reddedilen bir taşıma; asla sessizce büyüme değil.
- **Kilitler.** `mutex` tanımlayıcıları ve yerleşimi, `io_mutex` disk-slot I/O'sunu
  korur. Etkin slot yalnızca ikisi birden (bu sırayla) tutulurken değişir; böylece
  write-back flusher attention'ı bloklamadan slot yazabilir.

Paylaşılan iç yapı: `ggml_disk_store.h` (namespace `argus_disk`, API değildir).

Kanıt: slot yayımlama (başarısız yazım yayımlanmış slotu asla değiştirmez) ve bozulma reddi [`tests/cpp/test_ggml_disk_buffer.cpp`](../tests/cpp/test_ggml_disk_buffer.cpp) içinde; [`test_direct_disk_pages_preserve_data_and_enforce_budgets`](../tests/test_native_llama_paged.py).

### 2.3 Katmanlar ve yerleşim politikası

```text
GPU  ──  pinned RAM  ──  pageable RAM  ──  disk (doğrulanmış slot; yayımlanmış
 en hızlı                                  içerik için her zaman gerçek kaynak)
```

`ARGUS_KV_POLICY=on` ile `ggml_kv_policy.cpp` her attention çağrısının etrafında çalışır:

- **prepare**, çağrının okuyacağı sayfaları toplar;
- **observe**, erişimleri kaydeder (sayfa başına 32 hücrelik geçmiş) ve sayfaları
  taşır: en az iki kez okunan sayfalar önce GPU'ya, sonra pinned'a, sonra RAM'e
  terfi eder; bir katman dolunca daha soğuk sayfalar doğrulanmış disk kopyalarına
  indirilir.
- Terfi, attention scratch payını (`argus_kv_policy_headroom`) boş bırakır; böylece
  çağrının kendi scratch'iyle çekişmez.
- Politika önerir; store doğrular, reddeder ve sayar (`policy_promotions`,
  `policy_demotions`, `policy_rejected`).

`move_page` kopyalar, doğrular, tanımlayıcıyı değiştirir ve eski kopyayı ancak
ondan sonra serbest bırakır. Dirty bir sayfa indirilmeden önce flush edilir.

Kanıt: [`test_cuda_policy_off_on_preserves_outputs_and_budgets`, `test_cuda_mixed_resident_path_preserves_policy_semantics`](../tests/test_native_llama_paged.py); iki politika düzeltmesi [E6](measurements/v070-e6-policy-logic-2026-09-25.md)'da ölçüldü.

### 2.4 Yazım yolu (append)

`set_rows_compute` üç yolu sırayla dener:

1. **Device'ta çözülen** (GPU control, tüm sayfalar zaten GPU'da): encode kernel'i
   satır indekslerini store'un sayfa-adres tablosu üzerinden device'ta okur. Aralık
   dışı bir satır, bir sonraki append'in reddettiği pinned bir hata bayrağı kurar.
2. **Host'ta çözülen GPU append** (policy on, CUDA, F16, tek sayfaya sığan
   satırlar): dokunulan sayfalar bir GPU kopyası alır (yerinde güncellenir, yüklenir
   ya da sıfırlanır) ve **dirty** işaretlenir. Yalnızca GPU bütçesi sayfaları, satır
   tablosunu ve politikanın payını birlikte tutabiliyorsa seçilir.
3. **Host append** (policy off ya da yukarıdakilerin tutamadığı her şey): referans
   yol, disk slotları üzerinden yazar.

GPU'daki F32→F16 dönüşümü round-to-nearest-even kullanır; GGML'in CPU `from_float`'ı
ile byte-byte aynıdır.

Kanıt: [`check_gpu_set_rows`](../tests/cpp/test_ggml_cuda_mechanism.cpp) GPU'da kodlanan her satırı GGML'in `from_float`'ıyla karşılaştırır ve aralık dışı reddini kontrol eder; süreler [E7a](measurements/v070-e7a-gpu-set-rows-2026-09-25.md) ve [E8](measurements/v070-e8-2026-09-26/)'de.

### 2.5 Write-back

GPU append almış her diskli store'un bir **Flusher** thread'i vardır
(`ggml_disk_gpu.cpp`). Her dirty sayfa için:

1. GPU kopyasının anlık görüntüsünü `mutex` altında alır;
2. etkin olmayan slotu yalnızca `io_mutex` altında yazar, geri okur ve doğrular;
3. slotu iki kilit birden tutulurken yayımlar; sayfa bu arada değiştiyse dirty
   kalır ve bir sonraki turda yeniden işlenir.

GPU'da yazılmış bir sayfayı host'tan okuyan, bekleyen digest'i hesaplar; indirme
sayfayı önce senkron olarak flush eder, `argus_disk_flush` ise bir store'u
istendiğinde boşaltır.

Kanıt: [`check_gpu_write_back`](../tests/cpp/test_ggml_cuda_mechanism.cpp); [E7b](measurements/v070-e7b-write-back-2026-09-25.md), ayrı I/O kilidinin düzelttiği gerilemeyi kaydeder (flusher store kilidini tutarken attention 1.53 → 5.94 s).

### 2.6 Okuma yolu (attention)

CUDA dispatch; kabul eden ilk yol kazanır:

| yol | ne zaman | bekler mi? |
|---|---|---|
| `table_attention` | sayfa tablosu olan GPU-control store | hayır |
| `policy_table_attention` | tablosu var olan ve yazılmış tüm K/V sayfaları GPU'da olan policy store | hayır |
| `try_resident` (ödünç alma) | sayfalar bulundukları yerden ödünç alınır; soğuk sayfalar çağrı başına scratch'e kopyalanır | evet, kernel için |
| staged | referans: tile'lar host belleğinden geçirilir, transfer hesaplamayla örtüşür | evet |

Kernel'ler (`ARGUS_KV_ATTENTION_PATH`): varsayılan `cells-v2`'dir (lane-per-cell
D = 64, birleşik K yüklemeleri, `half2` V yüklemeleri) ve hem prefill'i hem tek
token'lık decode'u karşılar; `cells-kc`, `cells-mlp`, `cells`, `batched`, `direct`
ve `staged` referans olarak tutulur. **Her yol `staged` ile bit-exact'tir.**

Kanıt: [`tests/cpp/test_ggml_cuda_mechanism.cpp`](../tests/cpp/test_ggml_cuda_mechanism.cpp) `direct`, `batched`, `cells`, `cells-mlp`, `cells-kc` ve `cells-v2`'yi prefill, decode ve karışık yerleşimde `staged`'e karşı çalıştırır ve float vektörlerinin eşit olmasını şart koşar; 4K koşuları her ARGUS modunda `a152ed56` hash'ini kaydeder ([final baseline](measurements/v070-e10-2026-09-26/final-baseline.json)).

CPU yolu (`ggml_paged_attention.cpp`) kesin causal attention'ı hücre bloğu bloğu
hesaplar ve blokları log-sum-exp ile birleştirir; soğuk bloklar okunduktan sonra
depolamaya geri döner.

### 2.7 Kuyruktaki GPU işiyle sayfa ömrü

Beklemesiz okuma, host bir sayfayı taşırken kernel'lerin hâlâ kuyrukta olabileceği
anlamına gelir. Bunu güvenli tutan değişmezler (türetimi:
[E10](../plans/argus-v0.7.0.md)):

- **Stream sırasına bağlı yayımlama.** Bir policy sayfa-tablosu girdisi yalnızca
  `publish_entry` ile değişir: attention stream'inde çalışan tek thread'lik bir
  kernel. Sayfanın GPU yerleşimindeki her değişiklikten sonra çağrılır (terfi,
  indirme, append ile ekleme, başarısız host yerinde yazımı). Daha önce kuyruğa
  giren kernel'ler gördükleri girdiyi korur.
- **Tablo ancak her şey sığıyorsa var olur.** Tüm sayfalar, tablo ve scratch payı
  GPU bütçesine birlikte sığmalıdır; böylece tablo hiçbir sayfayı yerinden etmez ve
  yerleşim kararları ödünç alma yolununkiyle aynı kalır.
- **Serbest bırakma kuyruktaki işi bekler.** Yerine yenisi gelen GPU sayfaları
  `cudaFree` ile bırakılır; test edilen sürücüde bu çağrı kuyruktaki tüm device
  işini bekler. Stream sıralı bir allocator (`cudaFreeAsync`) bu cihaz çapındaki
  beklemeyi kaldırırdı; uygulanmadı.
- **Host yazımları okuyucuları bekler.** Bir GPU sayfasına host'tan yerinde yazım,
  tablonun `read_event`'ini bekler; GPU-control sayfalarının host okumaları
  `write_event`'i bekler.
- **Nesil, epoch ya da refcount yok.** `clear` ve kapanış, sayfaları ve tabloyu
  aynı şekilde serbest bırakır.

Kanıt: [`check_policy_table`](../tests/cpp/test_ggml_cuda_mechanism.cpp) kernel'leri 512 MiB'lık bir device kopyasının arkasında kuyruğa sokar, bu sırada sayfaları taşır ve sonuçların `staged` ile eşit olmasını şart koşar; [`cudafree-sync-probe.cu`](measurements/v070-e10-2026-09-26/cudafree-sync-probe.cu) `cudaFree`'nin beklediğini ölçer (1 GiB kopyanın arkasında yaklaşık 12 ms).

### 2.8 Modlar

| mod | store | attention |
|---|---|---|
| Map'lenmiş host KV | `ggml_host_buffer.cpp`, dosya destekli mapping | llama.cpp'nin kendisi |
| CPU blok attention | map'lenmiş | ARGUS CPU |
| Direct disk KV | disk store | ARGUS CPU |
| CUDA, policy off | disk store, her okuma diskten | ARGUS CUDA (referans) |
| **CUDA, policy on** | disk store + katmanlar | ARGUS CUDA — ürün modu |
| GPU control | GPU otoriteli, disk yok | ARGUS CUDA — tanısal tavan |

Her birinin ayarları: [README_TR](../README_TR.md).

### 2.9 Hata semantiği

- **Desteklenmeyen modeller yüklemede geri döner.** MLA, attention sink'leri, KQ
  bias, soft-capping, ALiBi, Grok attention ya da birden fazla KV stream'i
  (`-np > 1`, unified cache olmadan): llama.cpp kendi KV'sini kullanır ve
  `ARGUS KV disabled (<feature> is unsupported)` loglar.
- **Yanlış yapılandırma başlatmayı reddeder** (`-nkvo` eksik, `-fa on` olmadan
  disk KV, bilinmeyen ayar değerleri).
- **Bütçeler reddeder, büyümez.** Sığmayan taşıma reddedilir ve sayılır; GPU'ya
  sığmayan append host yoluna gider.
- **Bozulma tespit edilir, tolere edilmez.** Checksum'ı tutmayan slot asla
  yayımlanmaz; doğrulanamayan bir sayfanın okunması açık bir hatayla durur.
- **Çökme.** Arka dosya unlink edilmiştir; hiçbir şey kalıcı olmaz. Süreç içinde,
  dirty bir sayfanın disk slotu flush edilene kadar GPU kopyasının gerisinde kalır.

Kanıt: [`test_unsupported_model_falls_back_to_stock_kv_at_load`](../tests/test_native_llama_paged.py); bozulma reddi [`tests/cpp/test_ggml_disk_buffer.cpp`](../tests/cpp/test_ggml_disk_buffer.cpp) içinde.

### 2.10 Kaynak haritası

| dosya | satır | sorumluluk |
|---|---:|---|
| `ggml_disk_store.h` | 107 | Store ve sayfa yapıları, paylaşılan iç yapı |
| `ggml_disk_buffer.{h,cpp}` | 103 + 783 | Store ömrü, GGML buffer arayüzü, host okuma/yazma, slotlar, istatistik |
| `ggml_disk_gpu.cpp` | 707 | GPU yerleşimi, append, flusher, `move_page`, policy sayfa tablosu |
| `ggml_cuda_attention.{h,cu}` | 115 + 1010 | Kernel'ler, set_rows/attention dispatch, event'ler |
| `ggml_kv_policy.{h,cpp}` | 14 + 131 | Yerleşim politikası |
| `ggml_paged_attention.cpp` | 427 | CPU blok attention |
| `ggml_host_buffer.{h,cpp}` | 26 + 297 | Map'lenmiş host KV |
| `ggml_profile.h` | 152 | CPU scope'ları ve CUDA event profili |
| `integrations/llama.cpp/*.patch` | 259 + 144 | llama.cpp bağlantıları |

Gözlemlenebilirlik (`ARGUS_KV_STATS_PATH`, `ARGUS_KV_PROFILE`): [README_TR](../README_TR.md).

---

## 3. HuggingFace yolu

### 3.1 Katmanlar ve sahiplik

**Kararlar Python'dadır** (hangi sayfa atılır, sıradaki katman hangisi, ne zaman
spill edilir). **Mekanik C++'tadır** (byte'lar nerede durur, nasıl paketlenir,
kernel'ler ne zaman başlar). Sınır, GIL tutularak senkron geçilir; arka plandaki
prefetch worker'ın Python'a çağrı yapması yasaktır.

İlke: mantıksal bir KV sayfası, fiziksel yerleşimi ve fiziksel hassasiyeti üç ayrı
şeydir.

| modül (`argus_cache/core/`) | sorumluluk | satır |
|---|---|---:|
| `memory_manager.py` | Koordinatör: sayfa yaşam döngüsü, katman kaskadı, attention birleştirme | 1975 |
| `telemetry.py` | Sıkıştırma / bant genişliği muhasebesi, VRAM ve parçalanma raporları | 397 |
| `granularity.py` | Deneysel sayfa bölme / birleştirme | 334 |
| `activation.py` | Basınca duyarlı yönlendirme: ARGUS kâr ettirene kadar exact-cache bypass | 186 |
| `host_spill.py` | Pinned host belleğine kayıpsız spill | 185 |
| `pool_allocator.py` | Katmanın codec'inden şekillenen katman başına sıkıştırılmış havuzlar | 131 |
| `jl_operators.py` | Önbellekli JL izdüşüm / geri kurma operatörleri | 128 |
| `outliers.py` | Aykırı değer ayırma ve geri koyma | 85 |
| `page_table.py`, `backend_pool.py`, `direct_attention.py` | SoA tanımlayıcı tablosu, bitişik q8_0/q4_0 havuzları, bunlar üzerinde kesin online-softmax attention (izole ölçüldü, HF decode yolunda değil) | 220 / 130 / 197 |
| `page_store.py`, `disk_pool.py` | Sınırlı havuzlar arasında transaction'lı yerleşim; atomik değiştirmeli sıkıştırılmış disk sayfaları | 146 / 158 |

### 3.2 Katman codec'i

Bir katmanın depolama biçimi adıyla değil, sayısal olarak `argus::TierCodec`
(`csrc/tier_codec.h`) ile tanımlanır:

| alan | anlamı |
|---|---|
| `kind` | `SignedLinear` · `UnsignedAffine` · `SignPacked` · `Projection` · `Passthrough` |
| `bits` | saklanan eleman başına bit |
| `pack_factor` | türetilir: byte altı codec'lerde `8/bits` |
| `levels` | kind ve bits'ten türetilir |
| `compression_ratio` | fp16'ya göre depolama maliyeti |
| `lossy` | yetenek bilgisi |

Her indirme, geri getirme, peek ve prefetch `compress_page()` / `decompress_page()`
ve tek bir parametreli `dequantize_generic_kernel`'den geçer. **Katman eklemek yeni
bir dal değil, bir registry kaydıdır.**

### 3.3 Plugin'ler

Bir quantization backend'i dört metot (`compress`, `decompress`,
`decompress_batch`, `memory_bytes`) ve bir `BackendCapabilities` bildirimidir.
`native_codec` bildirmek, native motorun katmanı kendisinin sıkıştırmasını sağlar;
bildirmeyen plugin de çalışır ve kayıpsız spill edilir.

```python
from argus_cache import (
    PagedDynamicKVCache, PipelineConfig, TierSpec,
    BackendCapabilities, NativeCodecSpec,
    register_quantizer, unregister_quantizer,
)

class TernaryBackend:
    def compress(self, tensor, **kw):
        scale = tensor.abs().amax().clamp_min(1e-8)
        return {"q": torch.round(tensor / scale).clamp(-1, 1).to(torch.int8),
                "scales": scale}
    def decompress(self, c, **kw):
        return (c["q"].to(torch.float32) * c["scales"]).to(torch.float16)
    def decompress_batch(self, cs, **kw):
        return [self.decompress(c, **kw) for c in cs]
    def memory_bytes(self, c):
        return c["q"].nelement() * c["q"].element_size()

unregister_quantizer("one_bit")
register_quantizer("ternary", TernaryBackend,
    BackendCapabilities(name="ternary", effective_bits=2.0,
        native_codec=NativeCodecSpec(kind="unsigned_affine", bits=2)))

cache = PagedDynamicKVCache(pipeline=PipelineConfig(tiers=[
    TierSpec(name="fp8",     backend="fp8",     max_pages=1),
    TierSpec(name="ternary", backend="ternary", max_pages=8),
]))
```

Politika bir backend'in *adını* değil *maliyetini* sorar:
`available_quantizers(device="cuda", dtype=torch.float16, max_effective_bits=4.0)`.

> **Paketleme ekseni uyarısı.** `argus_cache/backends/quantization.py` içindeki
> Python backend'leri sekans ekseni boyunca paketler; native kernel'ler `head_dim`
> boyunca. Birinin sıkıştırdığı sayfa diğeriyle asla açılmamalıdır. Native katman
> sayfalarını `peek_decompress_page()` ile okuyun.

### 3.4 Model sözleşmeleri

- `models/hf_attention.py`, `argus`'u Transformers'ın `AttentionInterface`'i ile
  kaydeder. Modele duyarlı bir `AttentionAdapter` registry'si query düzenini,
  uygunluğu ve çıktı sözleşmesini belirler. Doğrulanmış native sözleşme: Qwen2
  full-attention, eval modu, maskesiz tek token decode. Geri kalan her şey K/V'yi
  geri kurar ya da modelin kendi attention'ını kullanır (kapalı başarısızlık).
- `models/hybrid_cache.py` hibrit modellerde sahipliği tanımlar: ARGUS
  full-attention KV'nin sahibidir, linear-attention recurrent state'in asla değil.

### 3.5 Runtime adapter'ları

Adapter'lar kenarda yaşar; çekirdek hiçbirini import etmez. Yaşam döngüsü açık ve
idempotenttir (`initialize → activate → deactivate → shutdown`), kapanış asla hata
fırlatmaz.

| adapter | tür | KV'yi yönetir mi? |
|---|---|---|
| Ollama | `EXTERNAL` | Hayır. Sadece ayar aktarımı ve zamanlama; canlı sunucuya karşı doğrulandı (2026-08-14) |
| vLLM | `IN_PROCESS` | Hayır, kapalı başarısız olur. Gerçek entegrasyon `KVConnectorBase_V1` ve/veya özel bir `AttentionBackend` ister ([notlar](vllm-verification.md)) |
| Messages gateway | HTTP | Hayır. Yerel bir `llama-server` için Anthropic Messages ↔ OpenAI chat completions çevirisi |
| SGLang | — | Uygulanmadı; bir `RuntimeAdapter` alt sınıfı ve bir registry satırı yeter |

---

## 4. Ölçümler

- **v0.7 (llama.cpp, 4K, Qwen2.5-0.5B, RTX 3050 Ti Laptop):** stock 1.336 s;
  GPU control 1.719 s (1.29x), decode 55.6 tok/s; policy on 1.821 s (1.36x),
  decode 48.2 tok/s; stock decode 37.5 tok/s. Yöntem ve uyarılar:
  [README_TR](../README_TR.md).
- **Deney kayıtları** (E1–E10): [`plans/argus-v0.7.0.md`](../plans/argus-v0.7.0.md)
  ve [`docs/measurements/`](measurements/README.md).

### 4.1 Tarihçe: HuggingFace yolu (v0.4)

2026-08-14'te kaydedildi; RTX 3050 Ti Laptop, CUDA 13.0, torch 2.12.0. HF yolu o
zamandan beri yeniden ölçülmediği için tutuluyor.

**Codec çalışma süresi ve sentetik doğruluk** (rastgele Gauss girdi; modeli değil
codec'i ölçer; [artifact](measurements/native-2026-08-14.json)):

| katman | oran | açma (ms) | rel. L2 | kosinüs |
|---|---:|---:|---:|---:|
| fp8 | 2.00× | 0.0729 | 0.0097 | 1.0000 |
| int8 | 2.00× | 0.0752 | 0.0097 | 1.0000 |
| int4 | 4.00× | 0.0523 | 0.1593 | 0.9876 |
| int2 | 8.00× | 0.0403 | 0.8332 | 0.8135 |
| one_bit | 16.00× | 0.0359 | 0.6023 | 0.7983 |
| jl | 4.00× | 0.1013 | 1.1733 | 0.2666 |

JL'nin beyaz gürültü üzerindeki sonucu yapısı gereği anlamsızdır. Gerçek
Qwen2.5-0.5B key'lerinde JL (düşük rank yöntemi değil, sekans ekseni boyunca bir
düzgünlük önseli) medyan rel. L2 0.413'e ulaştı; dağıtılan int2 0.565'te kaldı ve
JL 24 katmanın 20'sinde kazandı ([artifact](measurements/jl-2026-08-14.json)).

**Uçtan uca** (Qwen2.5-0.5B-Instruct, FP16, sayfa 1024;
[artifact](measurements/downstream-2026-08-14.json)):

| bağlam | baseline TTFT | ARGUS TTFT | baseline TPOT | ARGUS TPOT | baseline VRAM | ARGUS VRAM |
|---:|---:|---:|---:|---:|---:|---:|
| 4,096 | 0.316 s | 0.385 s | 19.14 ms | 40.21 ms | 1149.4 MiB | 1137.4 MiB |
| 16,384 | 1.819 s | 3.992 s | 18.84 ms | 79.78 ms | 1722.5 MiB | 1590.6 MiB |

VRAM tasarrufu bağlamla büyür (16K'da −7.7%), TPOT ise 4.2 katına çıkar: HF
arayüzü bitişik bir K/V tensörü istediği için saklanan sayfalar her adımda açılır.
Perplexity baseline 33.9169, ARGUS 33.8993 (yalnızca ACTIVE + FP8 dolu olduğu için
bu hiçbir kayıplı arşiv katmanını doğrulamaz).

**İzole direct paged attention** (24 query head, 4 KV head, head_dim 256;
[artifact](measurements/v040-fused-attention-benchmark.json)): 32K'da q4_0, FP16'nın
2.14 katı gecikmeye mal olur ve bağlamı 3.1 kat daha az bellekte tutar. Yalnızca
runtime sınıfı; uçtan uca bir sonuç değildir.

**Bilerek tutulan olumsuz sonuç.** 2026-09-04 llama.cpp A/B'si
([artifact](measurements/argus-ab-cache-comparison-2026-09-04.json)) ARGUS'u sunucuya
hiç yüklememişti (`argus_maps_count: 0`); görülen fark bir prompt-prefix
cache'inden geliyordu. llama.cpp entegrasyonundan (v0.5+) öncesine aittir ve hiçbir
iddiayı desteklemez.

---

## 5. Testler

```bash
# Python paketi (canlı Ollama testleri ARGUS_TEST_LIVE=1 ister)
pytest tests/ -q

# Native llama.cpp + CUDA paketi
ARGUS_LLAMA_CPP_DIR=/path/to/llama.cpp ARGUS_LLAMA_BUILD=build ARGUS_TEST_CUDA=1 \
ARGUS_TEST_GGUF=/path/to/stories15M.gguf pytest tests/test_native_llama_paged.py -q
```

- `tests/cpp/test_ggml_cuda_mechanism.cpp`: GPU set_rows eşitliği ve reddi,
  write-back, scratch payı ve sayfalar taşınırken kernel'leri 512 MiB'lık bir
  kopyanın arkasında kuyruğa sokan düşmanca bir policy-tablo testi.
- `tests/cpp/test_llama_paged_attention.cpp`: gerçek bir modelde ARGUS'a karşı
  stock llama.cpp; crop ve state kaydet / yükle dahil.
- `tests/test_native_llama_paged.py`: mod başına uçtan uca `llama-server`
  koşuları; yüklemede geri dönüş dahil.
- `tests/test_plugin_system.py`: kayıt, kaldırma, yetenek filtreleme, katman
  değiştirme ve native-codec aktarımı.
