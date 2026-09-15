# ARGUS v0.6 — "ara katman": çok katmanlı KV yönetimi (taslak)

Tarih: 2026-09-15. Durum: **taslak, vizyon görüşmesi yapılacak**. v0.5 M2 ve M6
kapanmadan başlanmaz. Kod yok; bu dosya yön ve ölçüm sözleşmesidir.

## Vizyon (kullanıcı onayı, 2026-09-15)

llama.cpp'de KV ya tamamen VRAM'de ya tamamen RAM'de. Arası yok. ARGUS o "ara"
olacak: GPU ve host/disk katmanlarını birlikte, düzgün yöneterek hızı
"KV tamamen VRAM'de" durumuna çok yaklaştırmak, mümkünse geçmek. Kapasite
ise disk/RAM kadar büyük olacak.

Çıkış noktası ölçüm (2026-08-16, RTX 3050 Ti 4 GB, Qwen3.6-35B-A3B Q4_K_M):
KV VRAM'e sığınca **19,14 tok/s** (32K), sığmayınca `-nkvo` ile **8,78 tok/s**
(262K). v0.6 hedefi: 262K'da bu açığı kapatmak ve bunu tekrarlanabilir ölçmek.
Bu hedef vaat değildir; ölçülmeden sonuç ilan edilmez.

## Başlıklar

1. **Çok katmanlı KV yönetimi (ana iş).** GPU (sıcak, f16) → pinned RAM →
   pageable RAM → disk. Politikayla terfi/indirme, bütçe başına kesin sınır.
   v0.5 M2 GPU/pinned katmanı temel alınır.
2. **Sayfa başına karışık hassasiyet.** Yeni sayfalar f16, eskiler q8_0/q4_0.
   ARGUS GGML codec'leri llama.cpp ile byte-uyumlu; eksik olan `llama_kv_cache`
   içinde sayfa başına tip. Kalite kaybı ayrıca ölçülür (başlık 4).
3. **Ölçüm stabilizasyonu.** Tek benchmark şeması: model/runtime revision, cihaz
   (SATA/NVMe ayrı), limitler, warmup/repeat, min/median/max, peak VRAM/RSS/
   pinned/staging/disk, TTFT/TPOT, I/O byte/latency. v0.5'ten ertelenen uzun context
   ladder'ı 262K basamağından başlar.
4. **Kalite kanıtı.** Aynı prompt'larla stock vs ARGUS: needle/RULER tarzı görevler,
   eski sayfalar quantize edildiğinde kalite kayması.
5. **Entegrasyon.** Hermes/Neo'da ARGUS varsayılanı, yamalı `llama-server`'ın
   kurulabilir paketi, Claude Code yerel oturum prefill takılması, vLLM KV connector.
6. **Birleşik store (gerekirse).** Python `PageStore` ile native store'un tek
   yaşam döngüsü; ancak 1–2 gerektirirse.

## Görüşmede karar verilecekler

- ARGUS'un kimliği: llama.cpp için KV katmanı mı, kendi yerel uzun-context sunucusu mu?
- Hedef model ve donanım matrisi (Qwen3.6-35B-A3B, UI-Mate-9B; SATA vs NVMe).
- Hız hedefi tanımı: hangi context'te, stock'un hangi moduna karşı.
