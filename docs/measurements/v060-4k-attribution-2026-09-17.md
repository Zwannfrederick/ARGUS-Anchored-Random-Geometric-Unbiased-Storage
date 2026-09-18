# 4K Qwen2.5-0.5B maliyet atfı — 2026-09-17

262K çalıştırılmadı. Bu atıf adımında attention/staging veya placement algoritması optimize edilmedi.
Disk erişimsiz GPU kontrolü stock-host farkını kapatmıyor; prefill kernel yolu, decode ise disk kapasitesi ve kalan attention/staging maliyeti olarak ayrışıyor.

## Koşul

Qwen2.5-0.5B-Instruct Q4_K_M, F16 KV, context=4096; 4016 giriş, 16 üretim ve 4031 retained/cached token.
ubatch=64, threads=6, GPU layers=99, flash attention açık, stock-host ve ARGUS için -nkvo.
Her atıf koşulu fresh server, 1 workload warmup + 1 ölçüm; prompt cache kapalı, seed=42.
Önceki context ladder 3 tekrarlı; buradaki ayrıntılı atıf tek ölçümlüdür, release kabul testi değildir.
RTX 3050 Ti Laptop 4 GiB, i5-11300H, driver 610.57.04; KIOXIA-EXCERIA SATA SSD / btrfs.
GPU ve pinned için tabloda belirtilen bütçe ayrı ayrı uygulanır; staging ve metadata bütçeleri 4 MiB.
Tüm giriş hashleri aynı ve needle doğru. ARGUS off/on/control çıktıları aynı; stock doğru koddan sonraki devam metninde ayrışıyor. Tam çıktı kalite eşitliği iddiası yok.

## Profiler kapalı referans — event ölçümüyle aynı binary

| Koşul (64 MiB/tier) | Prefill s | Decode tok/s | İstek s | Disk okuma MiB | Disk yazma MiB |
|---|---:|---:|---:|---:|---:|
| stock-host-kv | 1.282 | 39.018 | 1.669 | — | — |
| argus-cuda-off | 47.828 | 0.927 | 64.008 | 2343.19 | 49.88 |
| argus-cuda-on | 46.919 | 6.382 | 49.272 | 149.62 | 49.88 |
| argus-cuda-control | 39.777 | 7.942 | 41.668 | 0.00 | 0.00 |

GPU control ayrı bir tanı modudur: yazılmış KV GPU-authoritative, backing dosyası ve payload pread/pwrite yok.
Aynı CPU set_rows dönüşümü, doğrulanmış/bütçeli GPU sayfa değiştirme, page lookup, staging, 32-cell kernel ve beklemeleri korunur.
Maskelenen yazılmamış padding hosttan sıfır taşınabilir. Sıfır I/O iddiası KV payload içindir; model/log/stats dosya erişimlerini kapsamaz.

## CPU-only süreler (saniye, inclusive)

CUDA timing event’leri kapalı. Alt kapsamlar üst kapsamların içindedir: set_rows/write_page/disk_write veya attention/staging/synchronization satırları toplanmaz.
Tam exclusive dağılım summary JSON’dadır; her fazda exclusive toplam = attention + set_rows eşitliği doğrulandı.
disk_read/write blocking O_DIRECT syscall geçen süresidir; saf disk cihazı servis süresi değildir.
page_lookup offset/indeks/erişim bookkeeping; descriptor_scan snapshot/lock ve victim taramasını kapsar.

### prefill — 63 ubatch

| Kapsam | off / 64 MiB | on / 64 MiB | GPU control / 64 MiB | on / 2 MiB |
|---|---:|---:|---:|---:|
| attention | 43.2634 | 39.9945 | 37.8341 | 40.3521 |
| set_rows | 5.9945 | 6.3721 | 0.6398 | 5.8375 |
| write_page | 5.6373 | 6.0400 | 0.4808 | 5.5765 |
| policy | 0.0000 | 0.9917 | 0.0000 | 0.5720 |
| disk_read | 33.2654 | 5.9460 | 0.0000 | 27.3346 |
| disk_write | 1.4329 | 1.5216 | 0.0000 | 1.2705 |
| page_lookup | 0.0500 | 0.0338 | 0.0248 | 0.0497 |
| descriptor_scan | 0.0000 | 0.0941 | 0.0000 | 0.2005 |
| staging | 37.9363 | 36.7671 | 35.6017 | 32.6664 |
| copy_enqueue | 1.9382 | 1.7318 | 1.6208 | 1.5525 |
| synchronization | 5.2986 | 33.5201 | 34.3114 | 10.0525 |
| checksum | 2.3647 | 0.4270 | 0.1315 | 2.2145 |
| allocation | 0.3286 | 0.1589 | 0.0953 | 0.0596 |
| release | 0.2627 | 0.2420 | 0.0935 | 0.0460 |
| tier_read | 0.0000 | 0.1958 | 0.1997 | 0.0444 |
| tier_write | 0.0000 | 0.0730 | 0.0570 | 0.0028 |

### decode — 15 değerlendirme

| Kapsam | off / 64 MiB | on / 64 MiB | GPU control / 64 MiB | on / 2 MiB |
|---|---:|---:|---:|---:|
| attention | 17.3732 | 2.2173 | 1.7555 | 14.5410 |
| set_rows | 0.4080 | 0.4703 | 0.0683 | 0.3737 |
| write_page | 0.3501 | 0.4103 | 0.0318 | 0.3308 |
| policy | 0.0000 | 0.1100 | 0.0000 | 0.2040 |
| disk_read | 13.2009 | 0.3219 | 0.0000 | 10.9794 |
| disk_write | 0.0869 | 0.1123 | 0.0000 | 0.0725 |
| page_lookup | 0.0217 | 0.0144 | 0.0100 | 0.0221 |
| descriptor_scan | 0.0000 | 0.0427 | 0.0000 | 0.0590 |
| staging | 16.7623 | 1.8558 | 1.5461 | 13.9709 |
| copy_enqueue | 0.8262 | 0.7188 | 0.6134 | 0.6651 |
| synchronization | 0.8020 | 0.5716 | 0.5218 | 0.7659 |
| checksum | 0.9997 | 0.0294 | 0.0119 | 0.9547 |
| allocation | 0.0722 | 0.0292 | 0.0150 | 0.0115 |
| release | 0.0666 | 0.0305 | 0.0155 | 0.0088 |
| tier_read | 0.0000 | 0.0340 | 0.0248 | 0.0079 |
| tier_write | 0.0000 | 0.0074 | 0.0038 | 0.0000 |

## CUDA event süreleri — ayrı cohort, saniye

Event aralıkları host submission boşluğu ve diğer stream ile cihaz kaynak yarışını içerebilir. H2D/D2D saf DMA veya bant genişliği ölçümü değildir.
GPU aralıkları CPU beklemesiyle ve birbirleriyle örtüşür; birbirlerine veya CPU tablosuna eklenmez. Event’ler mevcut beklemelerden sonra okunur.

| Koşul / faz | Kernel | H2D staging | D2D staging |
|---|---:|---:|---:|
| argus-cuda-control / prefill | 37.3275 | 0.0040 | 33.7829 |
| argus-cuda-control / decode | 1.4629 | 0.0200 | 0.8905 |
| argus-cuda-on / prefill | 37.4887 | 0.1040 | 32.9435 |
| argus-cuda-on / decode | 1.4675 | 0.0263 | 0.8480 |
| argus-cuda-off / prefill | 37.2795 | 2.8412 | 0.0000 |
| argus-cuda-off / decode | 1.8440 | 1.2977 | 0.0000 |

Control: prefill 101.376, decode 46.080 tile launch; toplam 147.456. Decode’da 184.320 staging memcpy, 92.160 staging çağrısı ve 138.600 açık CUDA bekleme çağrısı.

## Ölçüm etkisi ve açıklanmayan kalan süre

| Koşul | Event açık/kapalı prefill oranı | Event açık/kapalı TPOT oranı |
|---|---:|---:|
| argus-cuda-control | 1.036 | 1.564 |
| argus-cuda-on | 1.020 | 1.365 |
| argus-cuda-off | 1.051 | 1.122 |

CPU-only control / aynı son binary’nin profiler-kapalı control’ü: prefill 1.015, TPOT 1.182 oranı.
Son profiler-kapalı control 39.085 s / 8.642 tok/s; aynı tur stock-host 1,291 s / 29,932 tok/s.
Bu tek çift oranları çalışma zamanı gürültüsünü de içerir; saf profiler overhead tahmini veya istatistiksel güven aralığı değildir.
Stock decode da turlar arasında 29,93–39,28 tok/s değişti. Sonuçlar kesin yüzdeler için değil, büyük darboğazları ayırmak içindir.

| CPU-only koşul | ARGUS CPU kapsamı s | İstek kalan süresi s |
|---|---:|---:|
| off / 64 MiB | 67.0390 | 1.4104 |
| on / 64 MiB | 49.0542 | 1.4657 |
| GPU control / 64 MiB | 40.2977 | 1.4469 |
| on / 2 MiB | 61.1043 | 1.3478 |

Kalan süre model/scheduler, kapsam dışı transfer, stats publication ve HTTP işini içerir; ölçülmeden kernel/disk diye etiketlenmedi. Tier allocation/free örtük CUDA beklemesi içerebilir.

## Karar

- Dar bütçeli policy-on: 0.991 tok/s; decode disk-read beklemesi 10.979 s. Policy toplam 0.777 s, isteğin %1.24 kadarı.
- Disk kaldırıldığında prefill yaklaşık 39–40 s kalıyor; control kernel event toplamı 37,33 s. Prefill farkını policy/placement değişikliği kapatmıyor.
- Decode’da kapasite disk beklemesini azaltıyor; GPU control hâlâ stock’tan birkaç kat yavaş. CPU-only control: attention 1,756 s; copy submission 0,613 s, açık synchronization 0,522 s, staging’in exclusive CPU işi 0,436 s.
- Öncelik prefill attention kernel ve decode tile/copy/synchronization yolu. Policy/descriptor taraması ilk optimizasyon hedefi değil. Bu raporda optimizasyon uygulanmadı.
- 262K beklemede; küçük-context yol henüz makul kabul edilmedi.

Doğrulama: 5 CPU/harness + 5 GPU testi geçti; D=48/64/256, CPU-only event yokluğu, exclusive sayaç mutabakatı, control sıfır pread/pwrite, budget/rollback ve lifecycle greedy parity.
Normal set_rows batching önceki ölçülmüş row-write adımından gelir; atıf boyunca kernel/tile/policy algoritması sabit kaldı.

## Artefaktlar

- [unprofiled](v060-attribution-unprofiled-2026-09-17.json) — komut, ortam, model/binary/harness/source hashleri ve ham sayaçlar.
- [profiled](v060-attribution-profiled-2026-09-17.json) — komut, ortam, model/binary/harness/source hashleri ve ham sayaçlar.
- [cpu](v060-attribution-cpu-2026-09-17.json) — komut, ortam, model/binary/harness/source hashleri ve ham sayaçlar.
- [pressure-cpu](v060-attribution-pressure-cpu-2026-09-17.json) — komut, ortam, model/binary/harness/source hashleri ve ham sayaçlar.
- [control-reference](v060-attribution-control-reference-2026-09-17.json) — komut, ortam, model/binary/harness/source hashleri ve ham sayaçlar.
- [Makine okunur özet](v060-4k-attribution-summary-2026-09-17.json) — inclusive/exclusive dağılım, residual ve kaynak hashleri.
