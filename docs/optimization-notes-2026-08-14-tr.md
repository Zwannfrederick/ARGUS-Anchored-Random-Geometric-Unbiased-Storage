# ARGUS optimizasyon notları — 2026-08-14

Bu not, stabilizasyon çalışmasının sonunda ortaya çıkan performans tablosunu ve
bundan sonra izlenebilecek dengeli optimizasyon yönünü kaydeder. Amaç “her
koşulda ARGUS'u açmak” veya tek bir metriği büyütmek değildir. Amaç, gerçekten
gerekli olduğu noktada bellek kapasitesi kazanırken kabul edilebilir bir
gecikme bütçesinde kalmaktır.

## 1. Projenin bugünkü amacı

ARGUS, transformer KV önbelleğini farklı maliyet ve hassasiyet katmanlarında
tutan deneysel bir bellek çalışma zamanıdır. Yeni ve önemli sayfalar FP16'da
kalabilir; eski sayfalar FP8 ve daha düşük bitli katmanlara veya CPU belleğine
indirilebilir.

Bu yaklaşımın hedefi ham token hızını artırmak değildir. Hedef, KV önbelleğin
kalıcı VRAM maliyetini azaltarak daha fazla bağlama yer açmaktır. Dolayısıyla
başarı ölçütü yalnız “kaç MiB düştü” olamaz; kazanılan bellek karşılığında TTFT,
TPOT ve model kalitesinin ne kadar değiştiği birlikte ölçülmelidir.

## 2. Neler yapıldı?

Stabilizasyon sırasında:

- sıkıştırılmış sayfaların yanında tutulan kalıcı FP16 kopyalar kaldırıldı;
- prefill sırasında promptun ikinci bir tam boy kopya olarak saklanması
  engellendi;
- native depolamanın yanında gereksiz oluşturulan Python tier pool'ları tembel
  tahsise geçirildi;
- benchmark kollarını bellekte tutan native callback referans döngüsü zayıf
  referanslarla kırıldı;
- JL projeksiyon operatörleri yalnızca bu katmana gerçekten ihtiyaç olduğunda
  oluşturulacak hâle getirildi;
- benchmark her kol öncesinde ısınıyor, her kol sonrasında cache'i kapatıyor ve
  prefill için yalnız son pozisyonun logit'ini üretiyor;
- gerçek olmayan vLLM monkey patch'i kaldırıldı ve entegrasyon güvenli biçimde
  kapalı duruma getirildi;
- eski OOM, throughput ve vLLM tabloları dokümantasyondan çıkarıldı.

## 3. Elde edilen bulgular

Qwen2.5-0.5B-Instruct ve 4 GB RTX 3050 Ti Laptop GPU ile elde edilen temiz
sonuç:

| bağlam | VRAM değişimi | TTFT oranı | TPOT oranı | yorum |
|---:|---:|---:|---:|---|
| 512 | +%0.6 | 3.33x | 1.54x | ARGUS gereksiz maliyet oluşturuyor |
| 1.024 | %0.0 | 2.37x | 1.63x | bellek kazancı yok |
| 2.048 | %0.0 | 1.80x | 1.60x | bellek kazancı yok |
| 4.096 | -%1.0 | 1.22x | 2.10x | ilk küçük bellek kazancı |
| 8.192 | -%4.5 | 1.23x | 2.94x | bellek kazancı anlamlılaşmaya başlıyor |
| 16.384 | -%7.7 | 2.19x | 4.23x | kapasite kazanılıyor, decode çok pahalı |

En önemli çıkarım: **düşük context'te tier sistemi kullanılmamalı veya tam
bypass edilmelidir.** 512–2K aralığında ölçülebilir bellek kazancı yokken hem
TTFT hem TPOT kötüleşiyor. Bu makinede bellek eğrisi yaklaşık 4K'da kesişiyor;
fakat gerçek eşik model, KV boyutu, GPU, batch ve boş VRAM miktarına göre
değişecektir. Sabit bir “4K kuralı” ürün davranışı olarak hardcode edilmemelidir.

16K'da 131.95 MiB tasarruf gerçektir. Buna karşılık token başına süre 18.84
ms'den 79.78 ms'ye çıkmaktadır. Bu nedenle mevcut uygulama kapasite deneyi
olarak değerlidir, düşük gecikmeli servis çözümü olarak hazır değildir.

## 4. Neden decode pahalı?

Mevcut HuggingFace sözleşmesi attention'a bitişik FP16 K/V tensörleri verir.
ARGUS kalıcı depolamayı sıkıştırsa da her yeni token için eski sayfaları tekrar
FP16'a açıp tek bir büyük tensörde birleştirir. Aynı, değişmeyen K/V sayfaları
decode boyunca tekrar tekrar işlenir.

Maliyet yalnız quantization çekirdeği değildir:

1. sayfa metadata'sını dolaşma;
2. sıkıştırılmış veriyi okuma;
3. FP16 reconstruction;
4. büyük geçici çıktı tahsisi veya doldurulması;
5. sayfaları bitişik tensörde birleştirme;
6. sonrasında normal attention hesabını yine tam bağlam üzerinde çalıştırma.

Sayfa boyutunu 512'den 1024'e çıkarmak 16K TPOT'u 122.08 ms'den 79.78 ms'ye
indirdi. Bu yararlı bir sabit maliyet optimizasyonudur, fakat her token'da tam
cache reconstruction yapılması sorununu çözmez.

## 5. Dengeli aktivasyon politikası

ARGUS isteğin başından itibaren koşulsuz açılmamalıdır. Daha doğru politika:

```text
istek başlar
   |
   +-- tahmini KV + model çalışma alanı güvenli bütçede mi?
   |       |
   |       +-- evet: native exact cache / tam bypass
   |       |
   |       +-- hayır: önce FP8 depolama
   |                    |
   |                    +-- baskı devam ediyor: sıradaki daha ucuz katman
   |                    +-- baskı yok: daha kayıplı katmanlara inme
   |
   +-- her model/GPU profili için gecikme bütçesini kontrol et
```

Uygulanabilecek kontrol ilkeleri:

- **İstek öncesi kapasite tahmini:** katman, KV head, head dimension, dtype,
  bağlam ve beklenen output token sayısından exact-cache maliyetini hesapla.
- **Gerçek yüksek su seviyesi:** ARGUS'u yalnız tahmini tepe kullanım güvenli
  VRAM bütçesini aşıyorsa etkinleştir.
- **Histerezis:** eşik çevresinde sürekli aç/kapa yapma. Etkinleştirme ve geri
  dönüş sınırları farklı olsun.
- **İstek boyunca kararlı mod:** mümkünse karar istek başında verilsin. Decode
  ortasında sık sık bütün cache formatını değiştirmek yeni gecikme sıçramaları
  yaratır.
- **Kademeli kalite:** ACTIVE → FP8 ile başla; INT4/INT2/1-bit/JL yalnız gerçek
  baskı sürdüğünde ve kalite bütçesi buna izin verdiğinde kullanılsın.
- **Kazanç kapısı:** beklenen VRAM tasarrufu belirli bir alt sınırın altındaysa
  optimizasyonu açma. Yüzde kadar mutlak MiB değeri de önemlidir.
- **Gecikme kapısı:** tahmini TPOT servis hedefini aşacaksa daha küçük model,
  daha kısa bağlam, CPU offload veya isteği reddetme gibi açık alternatifler
  sunulsun.

Bu politika “ARGUS açık/kapalı” ikiliğinden çok üç çalışma modu olarak
tasarlanabilir:

| mod | kullanım | cache davranışı |
|---|---|---|
| `latency` | kısa context, interaktif servis | exact cache, ARGUS data plane bypass |
| `balanced` | orta/uzun context, sınırlı VRAM | ACTIVE + FP8, seçici reconstruction |
| `capacity` | aksi hâlde OOM olacak batch/araştırma işi | daha derin tier ve gerekirse CPU spill |

## 6. Daha pratik ve hafif decode seçenekleri

### A. Sıkıştırılmış sayfalar üzerinde doğrudan attention

En güçlü çözüm, tüm K/V'yi önceden FP16 tensöre çevirmek yerine attention
çekirdeğinin sayfaları doğrudan okumasıdır. Her blok register/shared-memory
çalışma alanına açılır, o blok için QK ve online softmax güncellenir, sonra
geçici alan tekrar kullanılır. Böylece bağlam boyutunda ikinci bir FP16 K/V
tensörü oluşmaz.

Bu yol en çok geliştirme ister, fakat hem VRAM kazancını koruma hem TPOT'u
düşürme ihtimali en yüksek seçenektir.

### B. Hot/cold ayrımı ve seçici reconstruction

Son kullanılan ACTIVE/FP8 sayfalar hızlı yolda tutulabilir. Soğuk sayfalar
yalnız attention politikası gerçekten ihtiyaç gösterdiğinde açılabilir. Bunun
iki farklı doğruluk seviyesi vardır:

- exact attention için bütün bloklar yine işlenir, fakat açma kernel içinde ve
  küçük çalışma alanıyla yapılır;
- yaklaşık attention için yalnız seçilen cold bloklar işlenir. Bu daha hızlı
  olabilir, ancak retrieval ve perplexity testleri geçmeden varsayılan olamaz.

### C. Sınırlı reconstruction cache

K/V sayfaları tokenlar arasında değişmediğinden, sık kullanılan birkaç cold
sayfanın FP16 reconstruction'ı kısa süreli tutulabilir. Bu, tekrar hesaplamayı
azaltır fakat VRAM tasarrufunun bir bölümünü geri harcar. Cache boyutu sabit
bütçeli olmalı ve yalnız ölçülen tekrar kullanımına göre çalışmalıdır.

### D. Önceden ayrılmış çalışma alanı

Her token'da yeni büyük tensör tahsis etmek yerine katman başına veya akış
başına sabit ring/workspace kullanılabilir. Bu yöntem reconstruction miktarını
azaltmaz; allocator, kopyalama ve fragmentation maliyetini düşürür. Daha kolay
ve düşük riskli bir ara adımdır.

### E. Asenkron açma ve hesapla örtüştürme

Bir sonraki KV bloğu açılırken mevcut blok üzerinde attention hesaplanabilir.
CUDA stream/event düzeni ve çift buffer ile dequantization gecikmesinin bir
kısmı gizlenebilir. Kazanç, bellek bant genişliği ile attention hesabının ne
kadar örtüşebildiğine bağlıdır; ölçülmeden sabit oran iddia edilmemelidir.

### F. Donanıma uygun daha az tier

Çok sayıda küçük tier geçişi metadata ve kernel launch maliyetini büyütebilir.
Servis profili için ACTIVE + native FP8/INT8 + CPU gibi daha sade bir pipeline,
derin araştırma pipeline'ından hızlı olabilir. 1-bit veya JL'nin teorik oranı
yüksek olsa da reconstruction maliyeti toplam sistemde daha kötü sonuç
verebilir. Tier seçimi yalnız sıkıştırma oranına göre yapılmamalıdır.

### G. Prefill ve decode için farklı yollar

Prefill büyük paralel matris işlemleridir; decode ise küçük query ile büyüyen
cache'i tekrar tekrar okur. Aynı kernel ve sayfa politikası ikisi için optimal
olmak zorunda değildir:

- prefill sayfaları toplu ve büyük chunk'larla ingest edebilir;
- decode doğrudan paged kernel, sabit workspace ve hot-page cache kullanabilir;
- ilk token ve sonraki token metrikleri ayrı bütçelenmelidir.

## 7. Önerilen uygulama sırası

1. Düşük context ve yeterli boş VRAM için gerçek data-plane bypass ekle.
2. Model/GPU bazlı KV maliyet tahmini ve histerezisli aktivasyon kapısı ekle.
3. Decode tahsislerini profille; sabit workspace/ring buffer uygula.
4. ACTIVE + FP8 sade pipeline'ını mevcut çok-tier pipeline ile karşılaştır.
5. Kernel içinde blok blok dequantization + online-softmax prototipi geliştir.
6. Aynı VRAM noktasında baseline, mevcut ARGUS ve yeni decode yolunu karşılaştır.
7. Yaklaşık cold-page seçimi düşünülüyorsa önce perplexity ve retrieval kapıları
   koy.
8. Son olarak baseline OOM / ARGUS tamamlıyor deneyini daha büyük model veya
   bağlamla gerçekleştir.

## 8. Başarı ölçütü

Yeni bir yöntem yalnız aşağıdaki koşulları birlikte iyileştiriyorsa ilerlemiş
sayılmalıdır:

- aynı veya daha düşük tepe VRAM;
- mevcut ARGUS'tan anlamlı derecede düşük TPOT;
- kabul edilebilir TTFT;
- tanımlı perplexity ve retrieval hata bütçesi;
- temiz teardown ve tekrarlı isteklerde bellek drift'i olmaması;
- kısa context'te baseline yoluna ölçülebilir ek maliyet getirmemesi.

Özetle doğru yön, optimizasyonu coşkuyla her isteğe uygulamak değil;
**ihtiyaç olduğunda devreye giren, ölçülen kazanç kadar gecikme ve kalite
maliyetini de hesaba katan adaptif bir bellek sistemi** kurmaktır.
