# v0.5 M2 — GPU/pinned KV tasarım önerisi

Tarih: 2026-09-16. Durum: **onaylandı, uygulama ve doğrulama sürüyor**.
Kullanıcı kararı: **v0.5 = mechanism, v0.6 = policy**.
Sıra: UI-Mate kısa parity → bu tasarım → M2 uygulama → M6 tam workload.

## Ön kontrol sonucu

[Ölçüm](../docs/measurements/v050-ui-mate-short-parity-2026-09-16.json):
UI-Mate-9B Q4_K_M, CPU, q8_0 KV, context 256, ubatch 32, tek sequence.
Aynı Türkçe prompt'ta stock / ARGUS host / direct disk **32 token ID ve tam
metin eşitliği** geçti. Model SHA-256 manifestle eşleşti. Full-attention
katmanları 3/7/11/15/19/23/27/31 ARGUS buffer'ında; recurrent state runtime'da.
Host ve direct teardown sayaçları sıfırlandı. Direct staging peak 983.040 byte
(4 MiB limit), metadata peak 20.480 byte (1 MiB limit).

Bu sonuç text-only greedy parity'dir; logit eşitliği, hybrid crop/state restore,
multimodal kalite, GPU KV veya performans kabulü değildir. `ignore_eos=true`
ile 32 token zorlandığı için yanıtın uzaması ayrıca kalite sonucu sayılmaz.

## Mevcut bağlantı ve eksik parça

- `integrations/llama.cpp/host-kv.patch`, `llama_kv_cache` allocation ve
  `set_rows` yolunu ARGUS'a bağlıyor. Hybrid model aynı cache'i yalnız full
  attention katmanları için kuruyor.
- Direct store 4 KiB fiziksel sayfalar, iki disk slotu, checksum ve generation
  kullanıyor. Bunlar token/head/layer düzeyindeki logical KV page ile aynı şey değil.
- Patch'teki `llama-graph.cpp` bağlantısı ARGUS attention op'unu açıkça CPU'ya
  atıyor. Sadece CUDA allocation eklemek GPU attention sağlamaz.
- Python `PageDescriptor` placement/codec/generation ayrımına zaten sahip;
  sözleşme örnek alınır. Python store'u llama.cpp'ye taşımak M2'nin önkoşulu değildir.

## Önerilen ilk kapsam

Tek GPU, tek sequence, desteklenen full-attention KV ve sabit codec ile başla.
İlk kernel/parity kapısı FP16; Q8/Q4 ancak kendi kernel ve model testleri geçince
GPU yolunda açılır. Desteklenmeyen kombinasyon açıkça reddedilir; mevcut CPU
yolunun seçimi ayrı ve görünür kalır. Codec alanı placement'tan bağımsızdır;
sabit codec ilk uygulama sınırıdır, GPU=FP16 mimari kuralı değildir.

Attention KV dışındaki weights, sampling, batching, serving ve recurrent state
llama.cpp'de kalır. GPU/pinned/pageable/disk aynı logical page sahipliğine bağlanır.
v0.5 açık taşıma/yerleşim işlemleri sağlar; otomatik recent/anchor seçimi,
erişim tahmini ve codec dönüşüm politikası v0.6'da kalır.

## Sayfa ve bütçe sözleşmesi önerisi

Logical kimlik allocation/sequence epoch, layer, K/V tarafı ve token aralığını
ayırt eder. Descriptor codec, placement, slot/offset, generation, geçerli token
sayısı ve kullanım/in-flight bilgisini taşır. Diskin 4 KiB slotları bu aralığın
fiziksel backing'idir; K/V tensor geometri bilgisi adapter'dan gelir.

GPU, pinned RAM, pageable RAM, metadata ve disk için ayrı allocation limitleri;
host/device staging için ayrıca açık limitler tutulur. Pinned staging hem pinned
hem staging limitine, device staging hem GPU hem staging limitine dahildir;
fiziksel toplam raporda iki kez toplanmaz. Alignment, padding, attention scratch,
in-flight source+destination, worker stack ve descriptor alanları hesaba katılır.
Model/runtime allocation'ları ARGUS limitinin dışında ayrıca ölçülür; bu limitler
tüm process RSS veya toplam cihaz VRAM'i için garanti diye sunulmaz.

Pinned RAM gerçek page-locked allocation'dır; mmap/advice bunun yerine geçmez.
Bütçe rezervasyonu allocation'dan önce yapılır. Yer yoksa kaynak korunarak
açık hata verilir; hangi sayfanın indirileceğine çağıran taraf karar verir.
Zorunlu pin kümesinin bütçeyi aşması da hatadır.

## Write, migration ve attention akışı

1. Append geçerli logical aralığı günceller; kısmi sayfa ve her iki K/V tarafı
   runtime'ın token commit sınırıyla tutarlı kalır. Eski nesil kopyalar kullanılamaz.
2. Migration hedef kapasitesini ayırır; source'u ve generation'ı sabitler.
   Copy/I/O doğrulanıp CUDA event tamamlanmadan descriptor yayımlanmaz.
   Başarısızlıkta source geçerli kalır; geçici hedef serbest bırakılır.
3. GPU'daki sıcak blok doğrudan tüketilir. Host/disk blokları sınırlı pinned ve
   device staging üzerinden GPU'ya gelir; bütün context GPU'ya geri kurulmaz.
4. Blokların attention sonuçları tek online-softmax accumulator ile birleştirilir.
   Ayrı normalize edilmiş blok çıktıları ortalanamaz. Mevcut GGML CUDA flash
   attention kodundan yararlanılabilir; bloklar arası accumulator sözleşmesi
   ayrıca uygulanıp doğrulanmalıdır. CPU özel op'unun içine kontrolsüz CUDA
   çağrısı eklemek yerine GGML CUDA backend/scheduler yolu kullanılır.
5. İlk doğrulama senkron olur. Sonra sabit sayıda buffer ve event ile
   disk → pinned → GPU kopyaları compute ile overlap edilir. Bir slot son
   okuyucusunun event'i tamamlanmadan yeniden kullanılamaz.
6. Crop/reset/restore generation'ı değiştirir, eski işleri iptal eder veya bekler;
   teardown worker ve CUDA event'lerini tamamlayıp bütün rezervasyonları bırakır.
   Hybrid crop kısıtları runtime contract'ına göre korunur.

Policy olmadan referans davranış üretilebilir. Prefetch açık/kapalı aynı sonucu
vermelidir. Full attention'da eski sayfalar her decode'da gerekebilir; sıcak sayfa
seçiminin hız kazandıracağı varsayılmaz. Bunun ölçümü v0.6 policy seçimini besler.

## Uygulama ve kabul sırası

1. CUDA backend bağlantısı + sabit codec'li küçük full-attention model:
   stock ile attention/logit toleransı ve greedy parity; gerçek GPU op kanıtı.
2. Küçük GPU/pinned/pageable bütçeleriyle zorunlu spill: placement değişse de
   aynı sonuç; tüm sayaçlar limit içinde. Eksik bütçe ve allocation hatası kaynak
   sayfayı korur; CPU disk yolu regresyon testleri geçer.
3. Async pipeline: prefetch açık/kapalı parity; geciken event, kısa disk I/O,
   bozuk payload, stale generation, reset/crop/restore ve teardown testleri.
4. UI-Mate kısa parity'yi GPU tiering ile tekrarla; ardından M6 Türkçe,
   screenshot grounding, click ve tool-calling workload karşılaştırması.

## Kod öncesi görüşülecek karar

Öneri: **önce sabit codec ile doğru GPU attention ve bütçeli migration; sonra
bounded overlap**. İlk FP16 kapısı geçince Q8/Q4 kapsamını ayrı testlerle aç.
Sayfa başına precision dönüşümü ve gelişmiş policy v0.6'da kalsın. Bu kapsam
v0.6'daki bağımsız placement/codec sözleşmesini şimdiden korur.

## Uygulanan mekanizma ve doğrulama — 2026-09-16

Kullanıcı yukarıdaki kapsamı `v0.5 = mechanism, v0.6 = policy` ilkesiyle onayladı.
Native yol Python descriptor'ını taşımıyor: codec tensor geometrisinde, placement
4 KiB backing descriptor'ında; logical KV aralığı runtime adapter'ında kalıyor.
Şimdilik migration bir fiziksel sayfayı atomik yayımlar; çok sayfalı logical KV
transaction garantisi yoktur. Disk backing korunur ve yazmalar disk üzerinde
doğrulandıktan sonra resident kopya geçersizleşir.

- Gerçek `cudaMalloc`, `cudaHostAlloc`, pageable mmap ve direct disk arasında
  açık migration; hedef doğrulanana kadar source korunur. Otomatik policy yok.
- GGML CUDA scheduler üzerinden FP16 attention, D≤256, tek sequence/device 0;
  32-cell çift pinned/device staging ve bloklar arası online FP32 softmax.
  GPU sayfaları tile'a D2D taşınır; doğrudan resident tensor üzerinde fused
  attention değildir. Her çağrıda staging allocation yapılır; hız sınırı ölçülmedi.
- D=48/64/256 gerçek GPU kontrolleri: RAM/pinned/GPU/disk roundtrip,
  stale generation, kısa write, bozuk backing, bütçe reddi, overlap açık/kapalı
  birebir eşitlik ve teardown geçti. En büyük kernel referans hatası 3,81e-7.
- stories15M: 600-token prefill, decode, crop ve restore içeren 19 karşılaştırma;
  0 greedy farkı, attention hata ≤0,000344426, logit hata ≤0,0102015.
  126 gerçek CUDA attention çağrısı ve açıkça taşınan iki sayfa.
  Peak GPU 90.112, pinned 77.824, staging 163.840 byte; her limit 4 MiB.
- Host ve CUDA patch'leri ters/ileri uygulanarak sekiz dosyada birebir doğrulandı.

Bu ölçümler GPU mekanizmasının doğruluğunu gösterir; UI-Mate hybrid lifecycle,
262K hız hedefi veya Neo uçtan uca görev kabulü olarak yorumlanmaz.
