# ARGUS v0.6 — heterogeneous KV memory runtime

Tarih: 2026-09-16. Durum: **policy sözleşmesi sabitlendi; benchmark baseline'ı açık**.
Sürüm ayrımı (kullanıcı onayı): **v0.5 = mechanism, v0.6 = policy**. Bütçeli
allocation, açık migration ve attention v0.5; hangi sayfanın nereye ve hangi
codec ile taşınacağını seçen otomatik policy v0.6 sorumluluğudur.
v0.5 M2 ve M6 kapandı. Aşağıdaki policy sözleşmesi de sabitlendi; implementation
bu sözleşmeye göre başlayabilir. Bu belge yön ve sözleşmedir; uygulanmış özellik
listesi değildir — sözleşmedeki hiçbir madde çalışan kod iddiası taşımaz.

> ARGUS treats KV cache as a heterogeneous virtual-memory hierarchy rather than a tensor assigned to one device.

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

Çıkış noktası kayıtlı ölçüm: RTX 3050 Ti 4 GB + Qwen3.6-35B-A3B Q4_K_M,
32K/VRAM-KV **19,14 tok/s**, 262K/RAM-KV **8,78 tok/s**. Amaç 262K context'te
heterogeneous tiering ile bu farkı mümkün olduğunca kapatmaktır. Context'ler
farklı olduğundan bu iki sayı kontrollü placement A/B sonucu değildir.

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
Günlük kullanımı yansıtan referans stock tarafındaki kayıtlı çifttir:
32K/VRAM-KV **19,14 tok/s** ve 262K/RAM-KV **8,78 tok/s**. v0.6'nın kapatmayı
hedeflediği açık budur.

vLLM/SGLang, Hermes/Neo ürünleştirmesi, paketleme ve daha geniş codec/model
kapsamı bu sözleşmenin ardından ayrıca önceliklendirilir; vizyon bunların
tamamını v0.6 kabul kapısı yapmaz.
