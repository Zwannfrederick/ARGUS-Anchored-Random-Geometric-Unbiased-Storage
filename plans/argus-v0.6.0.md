# ARGUS v0.6 — heterogeneous KV memory runtime

Tarih: 2026-09-18. Durum: **4K exact lane-per-cell kernel default (control 2,8×, policy-on 8,3× stock); decode ve 262K beklemede**.
Sürüm ayrımı (kullanıcı onayı): **v0.5 = mechanism, v0.6 = policy**. Bütçeli
allocation, açık migration ve attention v0.5; hangi sayfanın nereye ve hangi
codec ile taşınacağını seçen otomatik policy v0.6 sorumluluğudur.
v0.5 M2 ve M6 kapandı. Aşağıdaki policy sözleşmesi de sabitlendi; implementation
bu sözleşmeye göre başlayabilir. Aşağıdaki ilerleme bölümü doğrulanan uygulamayı kaydeder; sözleşmenin geri
kalanı henüz çalışan özellik iddiası değildir.

> ARGUS treats KV cache as a heterogeneous virtual-memory hierarchy rather than a tensor assigned to one device.

## Attention datapath ilerlemesi — 2026-09-18

`89eaf06` attribution baseline ve placement policy algoritması korunuyor.
`6f1ef8c` GPU-resident F16 KV için kilitlerle korunan direct pointer-table yolunu
getirdi: prefill payload D2D sıfırlandı, fakat scalar kernel yüzünden anlamlı
hızlanma görülmedi (41,24 → 40,05 s). Sonraki batched kernel aynı FP32 reduction
ve FMA sırasını koruyarak Q>1 işi attention başına tek CUDA invocation'a topladı:
profil açık prefill **12,51 s**, kernel çağrısı **101376 → 1512**, sync **1512**.
Decode ve non-resident KV staged fallback kullanıyor; yalnız zaten tamamlanmış
transferin teardown wait'i kaldırıldı. Correctness eşiği gevşetilmedi; 5 CPU ve
5 GPU testi, exact staged/batched parity ve aynı 4K completion hash'i doğrulandı.

Profiler kapalı üç tekrarda medyan prefill: stock-host **1,42 s**, GPU-control
**12,76 s**, policy-on **48,32 s**. GPU-control hâlâ stock'tan yaklaşık **9×**
yavaş; decode kazancı ve policy-on hızlanması iddia edilmiyor. **262K çalıştırılmadı.**
Detaylı süreler, transfer/sync sayıları ve sınırlar:
[4K datapath raporu](../docs/measurements/v060-datapath-2026-09-18.md).

### Checkpoint — 2026-09-18 (`7c51892`..`ff1ca2d`)

Census: policy-on'da her prefill çağrısını, mevcut ubatch'in yeniden yazdığı (write_page
residency'yi düşürür) tek bir küçük cold run reddediyordu; bütçe/alignment/codec hiç
reddetmedi. D=64 kernel specialization, satır başına sayfa çözümü ve cold sayfaları
tek çağrılık scratch'e kopyalayan mixed pointer table (placement/policy değişmeden)
sonrası profiler kapalı üç tekrarda medyan prefill: stock-host **1,33 s**, GPU-control
**5,22 s**, policy-on **12,27 s**. Exact staged parity ve policy promotion/demotion
eşdeğerliği korunuyor. Decode (Q=1 staged) değişmedi. Sonraki kernel adımı önerisi
(başlatılmadı): lane-per-cell exact tile kernel.
[Checkpoint raporu](../docs/measurements/v060-checkpoint-2026-09-18.md).

Lane-per-cell exact kernel (bit-exact, default; warp-per-cell `batched` kontrol olarak
kaldı): profiler kapalı üç tekrarda medyan prefill stock-host **1,33 s**, GPU-control
**3,69 s**, policy-on **10,99 s**.
[Kernel raporu](../docs/measurements/v060-kernel-cells-2026-09-18.md).

## Uygulama ilerlemesi — 2026-09-17

İlk adım uygulandı: native store ve page descriptor içerik revision'ı ile
placement revision'ını ayırıyor. Prefetch/CUDA attention store içerik
revision'ını izliyor; migration token'ı allocation, fiziksel sayfa ve o sayfanın
içerik revision'ına bağlı. Başka sayfaya yazmak veya byte'ları koruyan migration
bu token'ı geçersiz kılmıyor; hedef sayfaya yazmak ve reset geçersiz kılıyor.
Başarısız yazım revision'ları değiştirmiyor.

Descriptor snapshot API'si (`argus_disk_page_descriptor`) placement, codec,
revision'lar, yazılmışlık, `last_access_step` ve `access_count` döndürüyor.
Codec tensor allocation'ında kaydediliyor; view, yazım ve migration codec'i
değiştirmiyor. Precision dönüşümü hâlâ kapalı. Snapshot kısmi son sayfayı ve
view offset'ini destekliyor; migration'ın tam ve hizalı sayfa şartı korunuyor.

Erişim adımı store içindeki başarılı okuma işleminin sırası; token/decode
adımı veya store'lar arasında ortak saat değil. CPU get, prefetch ve CUDA
staging okumaları sayılıyor; migration doğrulaması ve read-modify-write
okumaları sayılmıyor. Her başarılı okumada dokunulan sayfa başına sayaç bir
artıyor; sayaçlar taşmak yerine UINT64_MAX'ta kalıyor. Yazımlar erişim geçmişini
koruyor; `clear(0)` codec'i koruyup erişim geçmişini sıfırlıyor. Ek metadata
mevcut `sizeof(Page)` tabanlı resident bütçe hesabına dahil.

Doğrulama: `tests/cpp/test_ggml_cuda_mechanism.cpp` gerçek GPU'da 48/64/256
head dimension için geçti: dört tier arasında taşıma sırasında prefetch,
ilgisiz sayfaya yazım, yanlış sayfa token'ı, stale içerik/reset reddi ve mevcut
attention/bütçe/rollback kontrolleri. CPU disk, paged attention referansı ve
worker koordinasyon testleri de geçti. Yeniden derlenen CUDA llama.cpp ile
native lifecycle testi de geçti (19 adım, sıfır greedy mismatch, iki sayfa
taşıması). Bu doğrulama performans ölçümü değildir.
Descriptor kontrolleri de geçti: F32/F16/Q8/Q4 codec korunumu, view/tail
snapshot, erişim sayımı, başarısız okumada sayaç korunumu, reset ve saturasyon.
CUDA-linked disk/prefetch, üç boyutlu migration ve native lifecycle yeniden geçti.

CUDA kütüphaneleriyle bağlı prefetch testinde eski 64 KiB worker stack'inin
yerel cuBLAS TLS verisine (~104 KiB) yetmediği görüldü. Worker stack'i 256 KiB
+ 4 KiB guard oldu; tamamı staging bütçesinde ve attention block hesabında.
Bu, önceki sürüme göre worker başına 192 KiB ek staging gerektiriyor.

İlk policy ayrı `ggml_kv_policy.cpp` modülünde CUDA attention yoluna bağlandı.
`ARGUS_KV_POLICY=off` (veya unset) karar üretmiyor; `on` iki okumadan sonra
sayfaları GPU → pinned → pageable RAM sırasıyla değerlendiriyor. Dolu tier'da
daha az okunmuş bir sayfa doğrulanmış disk kopyasına indirilebiliyor; eşit
frekanslı sayfalar yerinde kalıyor. Codec değişmiyor. Her attention öncesinde
tile/state için gereken GPU/pinned alanı hesaplanıp gerekirse bütün canlı
store'lardan resident sayfalar indiriliyor. Bütçe enforcement store/tier
allocator'da kalıyor; allocation kimliği taşıyan token'lar teardown sonrası
reddediliyor. Registry pointer'ı store metadata bütçesinde; ayrı sınırsız cache
veya policy kuyruğu yok.

`policy_promotions`, `policy_demotions`, `policy_rejected` sayaçları stats'a
eklendi. Migration reddi kaynak sayfayı koruyor ve failure sayacını artırıyor.
Bu ilk algoritma erişim frekansına dayanıyor; topology/predicted-access yok,
taşımalar senkron, eviction seçimi descriptor tarıyor. Olumsuz eviction kararı
aynı pass boyunca tekrar kullanılabiliyor; başarılı eviction başına hâlâ tüm
descriptor'lar taranabiliyor. Uzun context maliyeti ölçülmedi.

[Off/on smoke artefaktı](../docs/measurements/v060-policy-smoke-2026-09-17.json):
stories15M, 64-token prefill, 1024 context kapasitesi, F16 KV; her iki koşulda
19 lifecycle adımı ve sıfır greedy mismatch, manuel migration yok. 4 MiB GPU
bütçesinde disk okuması 20.004.864 → 12.877.824 byte; 256 KiB GPU bütçesinde
12.939.264 byte, 5 otomatik demotion ve bütçe içinde 262.144 byte peak GPU.
Bu sayılar küçük işlev/I/O kontrolüdür, throughput kazancı veya gerçekçi uzun
context çalışma noktası değildir. 3 CPU ve 4 GPU testi geçti; GPU mekanizma
testi D=48/64/256, tier fallback, admission, eviction, bozuk backing reddi ve
teardown'u kapsıyor.

Kullanıcı yönlendirmesi: 262K çalıştırılmayacak. Önce aynı 4K Qwen2.5-0.5B
workload'unda stock-host / policy off / policy on / disk erişimsiz GPU control
karşılaştırılacak. Policy, blocking disk I/O, set_rows/write, descriptor lookup/
scan, H2D/D2D staging, kernel ve synchronization ayrı ölçülecek; ölçümden önce
attention/placement optimizasyonu yapılmayacak. Küçük-context yol makul hale
gelmeden 262K'ya dönülmeyecek. 262K baseline ve kalite eşikleri hâlâ açık.

Bu yönlendirmeden önceki 1K/4K ölçümleri: [dar bütçe ladder](../docs/measurements/v060-policy-context-ladder-2026-09-17.json)
ve [64 MiB GPU/pinned karşılaştırması](../docs/measurements/v060-policy-context-resident-2026-09-17.json).
İlk koşul kapasite baskısı testidir; ikincisi ~48 MiB KV'yi tutabilir. İkisi de
küçük model tanısıdır; büyük hybrid model veya 262K doğrulaması değildir.
1K ön incelemede ölçülen tekrarlı 4 KiB row-write maliyeti, mevcut encoding
buffer'ında bitişik satırları gruplayarak azaltılmıştı. Yeni atıf koşuları bu
aynı başlangıç uygulamasını kullanır; kernel/tile/staging algoritması değişmez.

[4K maliyet atfı tamamlandı](../docs/measurements/v060-4k-attribution-2026-09-17.md):
15 ölçüm, aynı 4016-token giriş, policy off/on, 2/64 MiB bütçeler ve sıfır KV
disk I/O'lu GPU control. Profiler-kapalı control prefill ~39–40 s; kernel
event toplamı 37,33 s. Dar bütçeli on koşusunda decode disk-read beklemesi
10,98 s; policy tüm istekte 0,78 s (~%1,25). CPU-only control decode attention
1,76 s: copy submission 0,61 s, açık synchronization 0,52 s ve staging'in
exclusive CPU işi 0,44 s. Event overhead ve çalışma zamanı değişkenliği ayrı
raporlandı; örtüşen GPU/CPU süreleri toplanmadı. Öncelik attention/staging
yoludur. Bu atıf adımında optimizasyon yapılmadı; 262K hâlâ beklemede.

## Ürün ve mimari sınır

ARGUS, runtime-bağımsız bir KV memory-management layer/runtime'dır; kendi
inference server'ı değildir. llama.cpp ilk ve en derin entegrasyondur. İleride
vLLM/SGLang adapter veya connector ile bağlanabilir. Sampling, batching, API
serving ve model loading sorumlulukları ARGUS core'a taşınmaz.

## Temel prensip ve page sözleşmesi

**logical KV page ≠ physical placement ≠ physical precision**

Sayfanın placement ve codec'i bağımsızdır: GPU/FP16, GPU/Q8, pinned/Q8,
pageable RAM/Q4 veya NVMe/Q4 mümkündür. GPU → FP16 bağı mimari zorunluluk
olamaz; sıcak sayfalarda f16 yalnız başlangıç policy'si olabilir. Her kombinasyonun
runtime/kernel desteği ayrıca doğrulanır; bu örnekler mevcut destek iddiası değildir.

Page descriptor/policy katmanı en az placement, codec, generation ve
sıcaklık/erişim bilgisini taşır. Promotion/demotion ile precision dönüşümü,
aynı policy engine'in iki bağımsız eksenidir. v0.5 M2 GPU/pinned altyapısı temel
alınır. Python `PageStore`–native store birleşmesi yalnız bu sözleşme gerektirirse
yapılır; ayrı bir yeniden yazım hedefi değildir.

## Tier ve policy yönü

GPU, pinned RAM, pageable RAM ve disk ayrı bütçeli tier'lardır; SATA ile NVMe
ölçümleri ayrılır. Policy yalnız klasik LRU'ya dayanmaz: layer topology,
full/sliding attention, page age, predicted next access, migration/codec maliyeti
ve tier bandwidth bilgisinden yararlanabilir. Özellikle hybrid modellerde
“hot” yalnız `last_access` demek değildir. Hangi sinyallerin ilk policy'ye gireceği
ölçümle seçilir; hepsini ilk sürümde uygulamak zorunlu değildir.

Prefetch/migration pipeline compute ile I/O'yu overlap etmeyi hedefler: GPU
mevcut page üzerinde hesaplarken sonraki page pinned → GPU, daha sonraki page
disk → pinned taşınabilir. v0.5 bounded async disk prefetch başlangıç altyapısıdır;
bu çok aşamalı overlap'ın zaten çalıştığı anlamına gelmez.

## Resource ve failure invariants

- Hiçbir tier kendi bütçesini aşamaz; staging gizli ve sınırsız ek tier olamaz.
  Migration sırasında birlikte yaşayan source/destination ve in-flight buffer'lar
  bütçe hesabına dahildir. Kesin bütçenin kapsadığı allocation'lar ile process RSS
  ve OS page cache ölçümleri açıkça ayrılır.
- Migration doğrulanmadan source yok edilemez. Başarısız promotion/demotion
  logical page'i bozamaz; precision dönüşümü de aynı korumayı sağlar.
- Stale generation okunamaz; reset/iptal sonrası eski işler sayfayı değiştiremez.
- Prefetch yalnız performansı değiştirebilir, correctness'i değiştiremez.
- Policy kapatılarak referans davranış üretilebilmelidir.

## Performans problemi ve ölçüm sözleşmesi

Çıkış noktası olarak alınan çift: RTX 3050 Ti 4 GB + Qwen3.6-35B-A3B Q4_K_M,
32K/VRAM-KV **19,14 tok/s**, 262K/RAM-KV **8,78 tok/s**. Amaç 262K context'te
heterogeneous tiering ile bu farkı mümkün olduğunca kapatmaktır. Context'ler
farklı olduğundan bu iki sayı kontrollü placement A/B sonucu değildir.

> **ARTEFAKTSIZ — doğrulanmadan baseline olarak kullanılamaz (2026-09-16).**
> Bu çiftin `docs/measurements/` altında ölçüm dosyası yok. 19,14 yalnız
> `docs/plans/control-audit.md` içinde düz metin iddia olarak geçiyor; 8,78
> depoda hiçbir yerde geçmiyor. (Aramada çıkan `19.14 ms` eşleşmeleri farklı bir
> büyüklüktür — 4096 context'teki TPOT; `262144` eşleşmeleri ise
> `ARGUS_KV_PINNED_BYTES` byte değeridir.) Sayılar oturum notlarından geliyor.
> v0.6 benchmark sözleşmesi bu çiftin üzerine kurulu olduğundan, ölçüm koşuları
> başlamadan önce ya yeniden ölçülüp artefakt üretilmeli ya da artefaktı olan
> başka bir çiftle değiştirilmelidir. README'ye bilerek alınmadı.

19,14 tok/s'yi geçmek vaat veya zorunlu kabul kriteri değildir. Önce aynı kalite
koşulu altında VRAM referansına yaklaşma oranı ölçülür. Örnek ara hedef
**15–17 tok/s** olabilir: 17 tok/s, kayıtlı RAM referansının yaklaşık **1,94×**'ı
ve 32K VRAM referansının yaklaşık **%89**'udur. Bunlar hedef oranlarıdır,
ölçülmüş ARGUS kazancı değildir; nihai eşikler gerçek benchmark'lardan sonra seçilir.

Asıl hız kazancı aynı 262K workload'unda stock RAM-KV ile ARGUS arasında
ölçülür. 32K VRAM sonucu ayrı bir hız referansı olarak raporlanır. Aynı context'te
VRAM baseline sığmazsa OOM açıkça yazılır; olmayan throughput'a oran hesaplanmaz.
262K'nın tam token sayısı ve gerçekten doldurulmuş/retained KV miktarı sabitlenir.

Benchmark model/runtime revision, model ve KV quantization, cihaz, limitler,
input/seed, warmup/repeat, min/median/max, peak VRAM/RSS/pinned/staging/disk,
TTFT/TPOT, throughput ve I/O byte/latency kaydeder. Stock ve ARGUS kalite
karşılaştırması aynı prompt'larla yapılır; needle/RULER tarzı görevler ve eski
sayfaların quantization drift'i ayrıca ölçülür. v0.5'ten ertelenen uzun context
ladder'ı 262K basamağından ele alınır.

## Policy sözleşmesi — sabitlendi (kullanıcı kararı, 2026-09-16)

Aşağıdaki dört madde karara bağlandı ve implementation'ın sözleşmesidir.
Beşinci madde (262K benchmark baseline'ı) ölçüm koşuları sonraya bırakıldığı
için açık kalır; kullanıcı kararı: policy önce düzgünce sabitlenir, ölçüme
sonraki koşularda bakılır.

### 1. Policy motorunun yeri — store üstünde ayrı modül

Native store yalnız **mekanizmayı** sunar: descriptor okuma, bütçe sorgusu ve
`argus_disk_move_page`. Policy ayrı bir C++ modülüdür ve store'a yalnız karar
sonucunu bildirir. v0.5 mekanizma kodu bu iş için değiştirilmez.

`ARGUS_KV_POLICY=off` referans davranış üretir ve bu **v0.5 davranışının
aynısıdır** — ayrı bir kod yolu değil, policy modülünün hiç karar üretmemesidir.
Her policy ölçümünün yanında aynı build'in `off` koşusu bulunur; aksi halde
kazanç policy'ye atfedilmez.

### 2. Page descriptor — içerik ve yerleşim ayrı eksenler

v0.5 descriptor'ı (`argus_cache/csrc/ggml_disk_buffer.cpp`, `struct Page`)
bugün `generation`, `checksum`, `active` ve `resident` taşıyor. Policy için
üç ekleme yapılır:

- **Eksen ayrımı.** `content_revision` ve `placement_revision` ayrılır.
  Bugün tek `store.revision` hem içerik yazımında (`:148`, `:234`) hem placement
  taşımasında (`argus_disk_move_page:552`) artıyor. Sonucu iki yönlü bir hata:
  `ArgusDiskPrefetch::take()` (`:455`) tüm store revision'ını karşılaştırdığı
  için **byte'ları koruyan ve checksum'la doğrulanan** bir taşıma bile uçuştaki
  prefetch'i "stale" sayıp iptal ettiriyor; ters yönde `argus_disk_move_page`
  (`:531`) store'un herhangi bir yerindeki içerik yazımı yüzünden düşüyor.
  Sözleşme: **prefetch yalnız `content_revision`'a bakar; migration yalnız
  taşıdığı sayfanın içerik yazımına bakar.** Böylece taşıma ile attention aynı
  anda çalışabilir ve plandaki compute/IO overlap hedefi korunur.
- **Per-page codec alanı.** Sayfanın precision'ı descriptor'da taşınır.
  v0.5'te disk yolu byte-opak; codec llama.cpp seviyesinde (`-ctk/-ctv`)
  seçiliyor ve sayfa başına değişemiyor. Alan v0.6'da tanımlanır, ilk policy
  tarafından **değiştirilmez** (bkz. madde 4).
- **Erişim sinyali.** Descriptor en az son erişim adımını ve erişim sayısını
  taşır. Hangi sinyallerin ilk policy'ye gireceği ölçümle seçilir; descriptor
  bunları taşımadan policy seçemez.

Değişmez kurallar v0.5'ten devralınır: doğrulanmadan source yok edilmez,
stale content okunamaz, prefetch correctness'i değiştiremez.

### 3. Tier semantiği ve migration sahipliği

GPU, pinned RAM, pageable RAM ve disk ayrı bütçeli tier'lardır; bütçe sahipliği
store'da kalır ve policy bütçeyi **aşamaz, gevşetemez**. Policy bir taşıma
önerir; store bütçeyi ve revision'ı doğrulayıp uygular veya açık hatayla reddeder.
Reddedilen öneri sessizce yutulmaz, sayaca yazılır. Migration sırasında birlikte
yaşayan source/destination ve in-flight buffer'lar bütçe hesabına dahildir.

### 4. Mixed precision — tanımlı eksen, ilk policy'de kapalı

Placement ve precision bağımsız eksenlerdir; **ilk policy yalnız placement
seçer.** Gerekçe: ölçülen 14,98 GB/istek disk okumasının kaynağı yerleşimdir ve
tek değişkenle A/B temiz kalır. Precision policy'si açılmadan önce ayrıca
karara bağlanacak sözleşme noktası: **precision düşürme sayfa başına tek
yönlüdür.** q4'e indirilmiş bir sayfayı f16'ya "yükseltmek" orijinali geri
getirmez, yalnız dequantize edilmiş q4 verir; bir policy bunu kalite geri
kazanımı gibi raporlayamaz. Bu nedenle promotion (tier yükseltme) ile precision
geri dönüşü aynı işlem sayılmaz.

### 5. Benchmark baseline'ı — açık

262K baseline'ı, kalite toleransı ve başarı metriği sabitlenmedi; ölçüm
koşuları sonraki oturumlara bırakıldı. Yukarıdaki ölçüm sözleşmesi
(policy `off`/`on` çifti, kaydedilen alanlar, OOM'a oran hesaplanmaması)
geçerlidir. v0.5'in kapanış ölçümü policy'siz mekanizmanın maliyetini verir ve
policy'nin neden gerektiğini gösterir:
[UI-Mate reference parity](../docs/measurements/v050-ui-mate-reference-parity-2026-09-16.json)
— tek istekte 14,98 GB disk okuması, decode 165,58 → 2915,31 ms/token. Bu sayı
bir çalışma noktası değil: `argus_disk_move_page` yalnız testlerden çağrıldığı
için hiçbir sayfa terfi etmiyor, 4 MiB GPU + 4 MiB pinned bütçeden yalnız
~0,5 MiB kullanılıyor ve her okuma diske gidiyor.

**Depoda gerçekçi çalışma noktasında ARGUS ölçümü yok** ve policy gelmeden
üretilemez — terfi eden sayfa olmadan ölçülecek bir yerleşim davranışı yoktur.
Günlük kullanımı yansıtan referans olarak alınan stock çifti — 32K/VRAM-KV
**19,14 tok/s** ve 262K/RAM-KV **8,78 tok/s** — yukarıdaki uyarıya tabidir:
artefaktı yoktur ve doğrulanmadan baseline sayılamaz. v0.6'nın kapatmayı
hedeflediği açık budur, fakat açığın büyüklüğü henüz ölçülmüş değildir.

vLLM/SGLang, Hermes/Neo ürünleştirmesi, paketleme ve daha geniş codec/model
kapsamı bu sözleşmenin ardından ayrıca önceliklendirilir; vizyon bunların
tamamını v0.6 kabul kapısı yapmaz.
