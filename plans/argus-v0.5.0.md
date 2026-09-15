# ARGUS v0.5 — modelden bağımsız KV runtime

Tarih: 2026-09-15. Durum: **yeniden açıldı**. Temel teslim commit `9861e9c`;
v0.5.0 M2 ve M6 kapanınca yayımlanır.
Kapsam: kullanıcı tüm hedefleri, gerçek llama.cpp KV sahipliği dahil, onayladı.

Tarihçe için [v0.5 devir notu](../docs/handoffs/v050-handoff-2026-09-15.md).
Yerel CPU derlemesi `scratch/llama.cpp-v050` altında, sabit revision arşivinden.

## Sürüm kapsamı (kullanıcı kararı, 2026-09-15)

- **v0.5 kalanlar:** M2 GPU/pinned katmanı ve M6 UI-Mate/Neo normal–ARGUS karşılaştırması.
- **Sonraya not — uzun context ölçümü (M5):** uzun süren ladder testleri sonra
  yapılacak; ölçüm 262K basamağından başlanarak ele alınacak. 1M gerçek model
  bu makinede hedef değil (model sınırı 262K, CPU attention prefill saatler sürer).
- **v0.6:** ölçümlerle birlikte yeni ürün vizyonu görüşmesi; ölçüm stabilizasyonu, Python PageStore–native store birleşmesi (M1),
  hybrid/multimodal kalite, FP8/Q2 (M3), vLLM ve diğer entegrasyonlar.
- Ölçüm diski: `scratch/kv` SATA SSD (`/dev/sda`) üzerinde; NVMe ölçümü için KV
  dizini `nvme0n1` üzerinde seçilmeli ve kayıtta cihaz yazılmalı.

## Kabul durumu

| Kapı | Durum | Kalan |
|---|---|---|
| M1 — ortak native page yaşam döngüsü | Native store'da kapandı | Python–native birleşme v0.6 |
| M2 — GPU/RAM/disk bütçeleri | Disk/metadata/staging kesin bütçe kapandı | **v0.5:** GPU ve pinned RAM sıcak katmanı, bütçe ve parity |
| M3 — attention contract kapsamı | Kernel referansları + iki model ailesi | Hybrid/multimodal kalite, FP8, Q2 v0.6 |
| M4 — runtime sahipliği | llama.cpp allocation/write/read ARGUS'ta | GPU-resident KV M2 ile gelir; karışık hassasiyet, vLLM v0.6 |
| M5 — uzun context ölçümü | Ertelendi (kullanıcı notu) | 262K'dan başlayarak sonra |
| M6 — UI-Mate/Hermes/Neo | Hermes ARGUS modu ve Neo KV kartı var (`0df1033`) | **v0.5:** UI-Mate normal vs ARGUS workload karşılaştırması |

## Amaç ve sınır

Önce cache motoru, ardından runtime entegrasyonu, en son Neo/Hermes computer-use
doğrulaması. UI-Mate bir test workload'udur; core tasarımının bağımlılığı değildir.
Mevcut `argus_cache/core/` dizini korunur; sırf isim için `argus_core/` taşınması yok.
Core'da model adına göre davranış seçilmez. Model kayıtları ve konumsal dönüşümler
`models/`, runtime yaşam döngüsü ve entegrasyon kodu `adapters/` içinde kalır.

Bu plan eski v0.4 kabul kapılarının geçtiği anlamına gelmez. Paket sürümü 0.4.0;
önceki planın tamamlanmamış ölçümleri tarihsel kayıtta açık kalır.

## Depodaki gerçek başlangıç noktası

Bu bölüm uygulama öncesi tarihsel durumdur; güncel sonuçlar aşağıdaki uygulama
kaydında ve üstteki kabul tablosundadır.

- `models/hf_attention.py`: mevcut `AttentionAdapter` ve full-attention GQA
  single-token contract yeniden kullanılır. İkinci bir model registry kurulmaz.
- HF native çağrı zinciri: `argus_attention_forward` →
  `PagedDynamicKVCache.inplace_paged_attention` → C++ manager.
- `core/direct_attention.py`: FP16/q8_0/q4_0 page-tile attention mevcut;
  bu ayrı engine'in çağıranları test ve izole benchmark. HF manager ile bağlı değil.
- `core/page_table.py`: codec ve placement ayrı; generation counter mevcut.
  `core/backend_pool.py`: sabit slotlu codec pool'ları mevcut.
- Direct engine pool seçimini yalnız codec ile yapıyor; descriptor placement'ını
  kullanmıyor. Aynı codec'in GPU ve host pool'ları birlikte seçilemiyor.
- `core/host_spill.py`: toplu host spill mevcut. Native manager decode öncesinde
  toplu swap-in yapabiliyor. Bu, sınırlı staging belleğiyle diskten attention değildir.
- NVMe placement/store ve direct engine'e bağlı NVMe prefetch henüz yok.
- vLLM adapter aktivasyonu reddediyor. Ollama adapter dış süreç ölçümü yapıyor.
  Hiçbiri KV sahipliğinin kanıtı değil; llama.cpp entegrasyonu yok.

Kaynaklar: [Neo handoff](../docs/plans/neo-handoff.md),
[v0.4 handoff](../docs/handoffs/v040-handoff-2026-08-15.md),
[llama.cpp deneyi](../docs/plans/argus-llamacpp-kv-backend.md).
Son belgedeki runtime özellikleri tarihsel gözlemdir; yeni kaynak sürümü seçilirken
yeniden doğrulanmalıdır. Özellikle vLLM'in çalıştığı yönündeki eski ifade güncel
adapter implementasyonuyla çelişir.

## Uygulama sırası ve kabul kapıları

### M1 — Tek page yaşam döngüsü ve gerçek direct decode

Mevcut page table, pool ve native manager yaşam döngülerini birleştir.
Append, kısmi son sayfa, demotion, crop, sequence reset ve teardown aynı kaynak
sayfalarını güncellesin. Decode için her adımda ikinci bir KV kopyası üretme.
Pool seçimi codec ve placement'ı birlikte kullansın; descriptor fiziksel kaynağı
doğru tarif etsin. Geçersiz slot, yinelenen page ID ve stale generation reddedilsin.

Kabul: HF generate sırasında direct yolun çağrıldığı ölçülsün; full-context
reconstruction çağrısını hata veren bir testle kapatarak desteklenen decode'un
hâlâ çalıştığı gösterilsin. FP16, Q8 ve Q4 için decoded-byte referansıyla parity;
GQA/MQA, kısmi sayfa, page sırası, crop/reset ve teardown kapsansın.
Quantization quality ayrıca orijinal FP16 referansına karşı ölçülsün.

### M2 — Bütçeli GPU → RAM → disk yerleşimi

Sıkıştırılmış K/V byte'larını koruyan disk store ekle. GPU, pinned RAM, pageable
RAM, disk ve staging için ayrı byte limitleri olsun. Anchor/recent pinning toplam
bütçeyi aşarsa açık hata ver; sessizce sınırı aşma. Disk dolması, kısa okuma/yazma,
bozuk payload ve iptal durumunda kaynak sayfayı koru. Descriptor ancak yeni kopya
başarıyla yazılıp doğrulandıktan sonra taşınsın.

Önce senkron page okuma doğrulansın. Sonra sınırlı kuyruk ve sınırlı staging
buffer kullanan async prefetch eklensin. In-flight sayfanın slotu iş tamamlanmadan
yeniden kullanılamaz; generation doğrulaması iptal/reset sonrası eski işi reddeder.
Sınırsız future, tam cache swap-in ve tüm dosyayı RAM'e yükleme yok.

Kabul: küçük yapay RAM/disk bütçeleriyle round-trip, hata enjeksiyonu ve reset;
aynı codec'in farklı placement'larında aynı attention sonucu; gerçek process RSS,
pinned byte, staging byte ve disk kullanımının bütçe içinde kaldığı ölçüm.
Dosya sistemi testi NVMe performansı diye raporlanmaz; mmap tek başına RAM limiti
veya async I/O garantisi sayılmaz.

### M3 — Attention kabiliyetleri

Adapter core'a normalize K/V, query, mantıksal konumlar, layer/head geometrisi,
scale ve desteklenen attention metadata'sını verir. RoPE uygulanma sınırı açık
olsun; iki kez uygulanmasın. Sliding window, padding/causal mask ve multimodal
position semantiği ayrı parity testleri olmadan native olarak ilan edilmesin.
Hybrid recurrent/conv state runtime'da kalır; yalnız büyüyen attention KV taşınır.

FP16/BF16 ve FP8/Q8 farklı formatlardır; birinin testi diğerini doğrulamaz.
Q2 doğrudan decode ayrı kapıdır. 1-bit/JL deneysel kalır; özellikle JL'nin
yaklaşık skoru exact quantized-page attention gibi sunulmaz.

Kabul: aynı core ile en az iki farklı model ailesi, GQA/MQA ve supported-mask
referansları; unsupported contract için açık fallback veya aktivasyon reddi.

### M4 — Runtime sahipliği

Önce sabit revision'da vanilla llama.cpp kaynaktan derleme ve baseline parity.
Ardından tek full-attention layer'ın allocation, write ve attention read yolu
ARGUS'a bağlanır; sonra desteklenen tüm layer'lara yayılır. Sequence lifecycle,
state save/load ve teardown korunur. vLLM ayrı adapter ve ayrı kabul kapısıdır.

`argus_in_llama_server_maps=true` yalnız kütüphanenin yüklendiğini kanıtlar.
Sahiplik için ayrıca allocation/page ID kayıtları, write/read sayaçları,
attention'ın bu page'leri tükettiği iz ve kontrollü backend kapatma testi gerekir.
Statik linkte maps girdisi olmayabilir; sahiplik kriteri dosya adı değildir.
Proxy/prefix-cache etkisi eşitlenmeden ölçüm ARGUS kazancı olarak yayımlanmaz.

### M5 — Context ve kalite ölçümleri

8K → 32K → 64K → 128K → 256K → 512K → 1M.
K birimi 1,024; bu planın 1M kapısı **1,048,576 retained token**.
Her basamak gerçek doldurulmuş cache ve decode içerir; descriptor kapasitesi
veya yalnız allocation sonucu end-to-end context desteği değildir.

Her koşuda model/runtime revision, quantization, cihazlar, limitler, komut,
seed/input, warmup/repeat, min/median/max, peak VRAM/RSS/pinned/staging/disk,
TTFT, TPOT, throughput, I/O byte/latency ve quality drift kaydedilir.
Baseline OOM ise açıkça OOM yazılır; olmayan throughput için oran hesaplanmaz.
Modelin native context sınırının ötesi ayrı position-extension kalite kapısıdır.

Örnek bütçe: 16 full-attention layer, 4 KV head, head_dim 256 ve batch 1 için
1,048,576 token FP16 KV 64 GiB, q8_0 34 GiB, q4_0 18 GiB'dır.
Bunlar yalnız K/V block byte'larıdır; weights, metadata, recurrent state,
workspace, staging ve OS ayrıca bütçelenir.

Disk kapasite sağlar; full attention her token'da bütün geçmişi okuyorsa disk
bant genişliği decode hızını sınırlar. Async prefetch bu byte miktarını ortadan
kaldırmaz. Ölçülmemiş 1M düşük gecikme veya 4 GB computer-use başarısı vaat edilmez.

### M6 — Neo/Hermes doğrulama workload'u

Motor ve runtime kapıları sonrasında vanilla GUI modeli + projector baseline;
Türkçe, screenshot grounding, click doğruluğu, tool calling ve uçtan uca görevler.
Ardından aynı workload ARGUS ile çalıştırılır. Core'a model ismi eklenmez.
Neo handoff'taki alıcı doğrulama, gönderim sonrası kontrol ve döngü sınırları
korunur. Servisleri açmak veya mesaj göndermek benchmark hazırlığının parçası değildir.

## 2026-09-15 yerel doğrulama ve engeller

- Sandbox içinde NVIDIA aygıtları görünmüyor; bu sürücü arızası değildir.
  Sandbox dışında `nvidia-smi` RTX 3050 Ti, sürücü 610.57.04 gösteriyor;
  aynı `.venv` içinde PyTorch 2.12.0+cu130 ve CUDA erişimi doğrulandı.
- `/tmp/argus-llama.cpp`: önceki handoff kaynak ağacı artık bulunmuyor.
- `.venv/bin/python -m pytest -q tests/test_hf_attention_bridge.py tests/test_direct_paged_attention.py`:
  sandbox içinde **10 passed, 2 skipped**, GPU erişimiyle **12 passed**.
- Bu kontrol v0.5 uygulaması, tam suite veya sürüm onayı değildir.

İlk uygulama M1 olarak başlatıldı. GPU kontrolleri sandbox dışında çalıştırılmalı.
M4 için sabit revision ve çalışır CPU/CUDA build artık mevcut. Kabul kapıları
kapanmadan sürüm etiketi yükseltilmez.

## Uygulama kaydı — 2026-09-15

- `DiskBlockPool`: sınırlı slot kapasitesi, checksum, atomik dosya değiştirme,
  bir sayfalık prefetch, slot yeniden kullanmadan in-flight I/O tamamlama.
- `PageStore`: descriptor/slot sahipliği, generation kontrolüyle taşıma,
  başarısız disk yazımında kaynak koruma, oldest-position spill ve pin koruması.
- Direct engine: `(codec, placement)` pool seçimi, kısmi FP16 sayfa düzeltmesi,
  BF16 ve mantıksal konumla sliding-window maskesi. Model adapter aktivasyonu
  ayrıca doğrulanmalıdır; core testi otomatik model desteği sayılmaz.
- HF GGML profili artık mevcut C++ streaming attention yolunu seçiyor.
  Q8/Q4/Q2/1-bit decoded-reference parity testleri geçti.
- Son tam yerel GPU/core koşusu: **374 passed, 1 skipped, 5 deselected**.
  Atlanan vLLM probu kurulu runtime gerektiriyor. Beş canlı Ollama testi ayrı:
  önceki tam koşuda 35B model yükleme çağrısı 300 saniyede timeout verdi.
- Sentetik 1,048,576 token disk/attention smoke geçti; geometri 1 layer,
  1 KV head, head_dim 4. Gerçek modelin 1M kapısı açık kalır.
- llama.cpp `54315813269112dd0baed7112ec87ad93a8218ca` ayrı `/tmp/argus-llamacpp-v050`
  ağacına indirildi. `integrations/llama.cpp/host-kv.patch` ile opsiyonel native
  ARGUS host buffer ekleniyor. Standalone allocation/read/write/copy/view,
  bütçe reddi ve teardown testi geçti. Python PageStore bağlantısı değildir;
  mapped allocation sınırı RSS sınırı değildir. Gerçek model testi ayrı tutulur.
- Gerçek CPU llama.cpp testi de geçti: altı attention layer, 32,000 vocabulary
  logiti, prefill ve state restore sonrası sekiz decode adımında maksimum fark
  **0**; 1,769,472 byte ARGUS KV allocation teardown sonrası 0'a döndü.
  1 byte limit context oluşturmayı reddetti. Artefact:
  `docs/measurements/v050-llama-host-ownership-2026-09-15.json`.
- UI-Mate artefact revision/boyut/hash bilgileri
  `plans/ui-mate-v050-artifact-manifest.json` içinde sabitlendi; yerel indirme
  ve hash doğrulaması henüz yapılmadı.
- Geçici otomatik onay kullanım limiti sonraki devam turunda kalktı; yetkili
  GPU testleri yeniden çalıştı. Sandbox aygıt görünürlüğü ayrı bir kısıttır.
- CPU llama-server A/B geçti: prompt cache kapalı, 16 greedy token aynı;
  ARGUS dosya mapping'i yalnız ARGUS sürecinde mevcut. Allocation 1,769,472
  byte, teardown live byte 0. Tekrar çalıştırılabilir kontrol:
  `tests/cpp/check_llama_server_ownership.py`. CUDA server derlemesi tamamlandı.
- CUDA server testi de geçti: RTX 3050 Ti üzerinde 7/7 model katmanı offload,
  57.95 MiB model buffer; stock/ARGUS 16 greedy token aynı. ARGUS host KV
  mapping ve 1,769,472 byte tahsis doğrulandı, teardown live byte 0.
  Bu GPU model inference + host KV sahipliğidir, GPU-resident KV değildir.
- Hybrid topology artık nested text config ve explicit layer_types bilgisini
  esas alıyor; geçersiz geometri/layout reddediliyor. Beş hybrid testi geçti.

Kalan kritik işler: sayfa deposunu üretim native yaşam döngüsüne bağlamak,
resident/staging toplam bütçeleri, kernel ve model contract kapsamı,
llama.cpp heterojen sayfalı attention/tiering bağlantısı, hybrid lifecycle,
vLLM sahipliği, gerçek model context/kalite merdiveni ve Neo/Hermes workload'u.

## Devam oturumu — 2026-09-15

- Devir notundan sonra eklenmiş bloklu native attention, lifecycle testleri ve
  UI-Mate dosyaları bulundu. Önceki "dosyalar indirilmedi" kaydı artık güncel değil.
- UI-Mate Q4_K_M ve f16 projector boyutları ve SHA-256 değerleri manifestle
  birebir doğrulandı; manifestin yerel doğrulama alanı güncellendi.
- KV adresine bağlı kalıcı worker koordinasyonu, her attention çağrısına özel
  ortak sahiplikle değiştirildi. Önceki çağrıdan kalan sayaçlar ve koordinasyon
  kaydı sızıntısı giderildi; 1/4/16 worker için tekrar ve serbest bırakma testi var.
- FP32 V okuma yolu tamamlandı. FP32/FP16/BF16/Q8/Q4 kernel referansları ve
  aynı graph'ın farklı worker sayılarıyla kullanımı doğrulandı.
- Geçersiz blok boyutu graph oluşturulurken reddediliyor; staging blok uzunluğu
  gerçek KV hücre sayısıyla sınırlandırılıyor. Bu toplam RAM bütçesi değildir.
- Geniş GPU/core suite: **375 passed, 5 skipped, 5 deselected**. Dört native
  test ortam değişkenleri verilmediği için bu koşuda atlandı;
  diğer skip vLLM, hariç tutulan beş test canlı Ollama testleridir.
- Son native suite ayrıca **5 passed**: koordinasyon, yedi kernel senaryosu,
  FP32/FP16 model lifecycle ve CPU stock–host–paged sunucu parity. Q8/Q4 model
  denemeleri stories15M head_dim=48 uyumsuzluğundan stock context oluşturamadı;
  quantized gerçek model kapısı açık, kernel parity bunun yerine sayılmıyor.
- `ARGUS_KV_RESIDENT_BYTES` kesin RSS üst sınırı değildir. Örneklenmiş resident
  değerleri ve tahliye istek sayaçları gerçek disk I/O/peak kanıtı sayılmıyor.
  M2 kabul kapısı açık kalır; native yol Python PageStore ile birleşmiş değildir.

## Disk direct KV yolu — 2026-09-15

- `ARGUS_DISK` llama.cpp buffer'ı: K/V sayfaları diske yazılır, attention blok
  blok okur; bir sonraki görünür blok için tek derinlikli, sınırlı staging
  kullanan async prefetch var (dolu kuyruk ve eski nesil reddedilir, yıkıcı
  bekleyen işi join eder). Kısa yazma, bozuk sayfa ve bütçe aşımı testleri geçti.
- Kök neden: GGML özel op'u `n_tasks=1` ipucuna rağmen her graph worker'ında
  çağırıyordu; disk/stock attention ve disk `set_rows` artık yalnızca `ith==0`
  üzerinde çalışır. Staging bütçe hatası bununla kapandı.
- Qwen2.5-0.5B Q4_K_M, 4 MiB staging, 1 MiB resident: q8_0 ve q4_0 için 19 adımda
  attention/logit farkı 0, greedy mismatch 0, crop + state geri yükleme dahil.
  Peak staging ≈1,27 MiB, peak resident 16–28 KiB. 1 MiB staging Qwen'de açık
  hatayla reddedildi (beklenen).
- Native pytest: 7 passed (kernel, worker, bütçe, f32/f16 × paged/direct) ve
  `ARGUS_TEST_QUANT_GGUF` ile 2 passed (q8_0/q4_0 direct lifecycle).
- `integrations/llama.cpp/host-kv.patch` sabit revision arşivine uygulanıp çalışma
  ağacıyla altı dosyada birebir doğrulandı.
- `bench_llama_paged_context.py` `argus-direct` modu, `--staging-bytes` ve
  `--ubatch` aldı; bu modla gerçek cihaz ladder ölçümü henüz yapılmadı, M2 açık.
