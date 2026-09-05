# ARGUS'u llama.cpp KV Backend'i Olarak Bağlama — Fizibilite ve Plan

**Tarih:** 2026-09-05
**Durum:** Adım 1 ve 2 UYGULANDI (2026-09-05). Adım 3 (ARGUS köprüsü) — ölçümlerle ERTELENDİ.
**Bağlam:** Hermes/Neo runtime'ında decode ~7 tok/s. Hedef: ARGUS'un sayfalı KV
yönetiminin (sıcak GPU / soğuk host RAM + tier sıkıştırma) bu runtime'a bağlanması.

---

## 1. Bugün ölçülenler (RTX 3050 Ti Laptop, 3770 MiB VRAM)

Hepsi aynı kanonik 5433-token Hermes prefix'i ile, `gemma-4-E4B-it-Q4_K_M` + mmproj +
MTP draft, `-fa on -ctk q4_0 -ctv q4_0`.

| Konfigürasyon | VRAM | Prefill | Decode |
|---|---|---|---|
| CUDA backend yüklenmiyor (eski hal) | 15 MiB | 27 tok/s | — |
| `-c 131072`, auto-fit **(şu anki üretim)** | 2461–2551 MiB | **368 tok/s** | **7.0 tok/s** |
| `-c 131072`, auto-fit, `-nkvo` (KV host RAM'de) | 2461 MiB | 383 tok/s | 7.3 tok/s |
| `-c 16384`, auto-fit | 1769 MiB | — | 5.9 tok/s |
| `-c 131072`, `-ngl 24` | — | abort | abort |
| `-c 131072`, `-nkvo -ngl 22` | — | abort | abort |

## 2. Ölçümlerin söyledikleri (planı değiştiren kısım)

**(a) KV cache beklendiği kadar VRAM yemiyor.** 128K → 16K düşürmek sadece ~780 MiB
kazandırdı (2551 → 1769), config yorumlarındaki ~2.6 GB değil. Yani ARGUS'un sayfalı
dinamik KV'sinden alınabilecek maksimum ödül bu runtime'da **~780 MiB** ile sınırlı.

**(b) Asıl kayıp orada değil: ~2 GB VRAM boşta duruyor.** 16K'da fitter kartın sadece
1769/3770 MiB'ını kullandı ve decode *düştü* (5.9 vs 7.0). llama.cpp'nin
`common_fit_params` tahmincisi bu modelde aşırı muhafazakâr; boşalan VRAM'i katman
eklemek için kullanmıyor.

**(c) Elle katman zorlanamıyor.** `-ngl 22` bile — KV tamamen host RAM'e alınmışken
dahi — `failed to fit params to free device memory ... abort` veriyor. Fitter'ın
Gemma-4 E4B (matformer + PLE) + MTP draft kombinasyonundaki tahmini hatalı.

**Sonuç: decode darboğazı KV yerleşimi değil, fitter'ın katman sayısını düşük tutması.**
ARGUS bağlansa ve KV'nin tamamını host'a taşısa bile, boşalan VRAM'i katmana çeviren
mekanizma llama.cpp'de bozuk olduğu için kazanç ~780 MiB'lık teorik tavanda kalır.

## 3. Gerçek engel: kaynak ağacı yok

Çalışan binary `/usr/lib/ollama/llama-server` — **ollama'nın ön-derlenmiş fork'u**
(`version: 0.3.0-dev, commit d222767c7`). Bu makinede llama.cpp kaynağı yok.

ARGUS'u KV backend'i olarak bağlamak = `llama_kv_cache` seviyesinde C++ yazmak =
kaynaktan derlemek = **ollama binary'sini terk etmek**. Bunun bedeli:

- `--spec-type draft-mtp,ngram-mod` ollama'ya özgü bir uzantı. Upstream llama.cpp'de
  MTP speculative decoding **yok**. Bırakmak zorunda kalırız.
- `--cache-ram`, fitter davranışı ve mmproj entegrasyonu da fork'a özgü.

Yani M0 (kaynaktan parity build) tek başına belirsiz ve büyük bir iş, ve muhtemelen
MTP kaybıyla **net yavaşlama** ile başlar.

## 4. ARGUS tarafındaki zemin (hazır olan kısım)

Bunlar duruyor ve doğru tasarlanmış:

- `argus_cache/core/page_table.py` (Stage S5) — precision (ACTIVE / q8_0 / q4_0)
  ile placement (GPU / PINNED_HOST / PAGEABLE_HOST) ayrımı, SoA descriptor tablosu,
  stale-eviction için generation counter.
- `argus_cache/core/backend_pool.py` — `CodecKind.GGML_Q8_0` / `GGML_Q4_0`, 32'lik
  blok düzeni. GGML ile byte düzeyinde uyumlu.
- `argus_cache/csrc/` — `manager.cpp`, `tier_codec.{h,cpp}`, `zero_copy_pool`,
  `quantization_kernels.cu`.

Eksik olan tek şey C++ köprüsü. Ama bkz. bölüm 3.

## 5. Neden ARGUS bugün Hermes'e dokunamıyor

`argus_cache/adapters/ollama.py` bunu kendi docstring'inde yazıyor:

> "Ollama runs as a separate server process wrapping llama.cpp, which owns its own KV
> cache in its own address space. ARGUS cannot manage that cache — there is no
> in-process attention path to intercept."

`adapters/base.py:13` ayrımı: vLLM "ARGUS can actually own the cache" sınıfında,
llama.cpp/Ollama değil.

## 6. Öneri: sırayı ters çevir

Haftalarca C++ yazmadan önce, ucuz olan ve ödülü daha büyük görünen adımlar:

**Adım 1 — Fitter'ı aş (saatler).** Amaç: boştaki ~2 GB VRAM'i katmana çevirmek.
`--no-host`, `--override-tensor`, `-ot` gibi yerleşim flag'lerini ve ollama fork'unun
fitter'ı devre dışı bırakan bir seçeneği olup olmadığını tara. Kazanç potansiyeli
decode'da 7 → 12-15 tok/s aralığında. Risk: düşük, geri alınabilir.

**Adım 2 — MTP'nin gerçekten çalıştığını doğrula (saatler).** `--spec-draft-n-max 2`
ile kabul oranını ölç. Çalışmıyorsa zaten upstream llama.cpp'ye geçmenin maliyeti
düşer ve Adım 3 kolaylaşır.

**Adım 3 — Ancak bundan sonra ARGUS köprüsü.** Adım 1 ve 2'den sonra hâlâ VRAM
darboğazsa ve ödül ölçülebilirse: upstream llama.cpp'yi kaynaktan derle (M0), KV
sayfalarını ARGUS page table'ına yönlendir (M1), attention öncesi host→GPU resurrect
ve async prefetch (M2), benchmark (M3).

**Alternatif rota:** modeli PyTorch/vLLM'e taşımak. ARGUS orada zaten çalışıyor
(`models/attention_wrapper.py`, `adapters/vllm.py`) ve köprü yazmaya gerek yok.
Bedeli: 4 GB VRAM'de GGUF Q4_K_M verimliliğini ve MTP'yi kaybetmek.

---

## 7. Adım 1 ve 2 sonuçları (2026-09-05, uygulandı)

### Adım 1 — Fitter aşıldı ✅

`llama-server --help` taramasında `-fit [on|off]`, `-fitt --fit-target`,
`-fitc --fit-ctx` bulundu. `--fit off -ngl 24` ile fitter'ın abort'u ortadan kalktı ve
kart gerçekten doldu.

| | auto-fit (önceki) | `--fit off -ngl 24` | kazanç |
|---|---|---|---|
| VRAM | 2461 MiB | 3565 MiB | kart kullanılıyor |
| Prefill | 368 tok/s | **599 tok/s** | +63% |
| 5433-token prefix | 14.8 s | **9.1 s** | -39% |
| Decode | 7.0 tok/s | **12.3–12.8 tok/s** | **+83%** |
| Gateway warmup | 27.9 s | **17.9 s** | -36% |

Tepe VRAM görüntü (vision) isteğinde 3627/3770 MiB — 143 MiB pay. Ekran iGPU'da
olduğu için GPU'da çekişme yok. `neo-model.service`'e `--fit off` + `-ngl 24` eklendi.

### Adım 2 — MTP speculative decoding gerçek ✅

Aynı `--fit off -ngl 24` tabanında, decode:

| Konfigürasyon | VRAM | Decode |
|---|---|---|
| MTP açık (üretim) | 3565 MiB | **12.3 tok/s** |
| MTP kapalı, `-ngl 24` | 2825 MiB | 9.0 tok/s |
| MTP kapalı, `-ngl 28` (draft'ın VRAM'i katmana) | 3201 MiB | 9.6 tok/s |

MTP **+37% decode** veriyor. Draft modelin yediği ~740 MiB'ı katmana çevirmek bunu
telafi etmiyor (9.6 < 12.3). Yani ollama fork'unun MTP uzantısı gerçek ve değerli.

## 8. Adım 3 kararı: ERTELENDİ

Ölçümler ARGUS köprüsünün gerekçesini zayıflattı:

- **Ödül tavanı düşük:** ARGUS'un sayfalı KV'sinden alınabilecek maksimum kazanç bu
  runtime'da ~780 MiB (bölüm 2a). Adım 1 tek satır flag ile bundan daha fazlasını
  zaten aldı (+1100 MiB kullanılabilir VRAM, decode +83%).
- **Peşin maliyet yüksek:** upstream llama.cpp'ye geçmek MTP'yi kaybettiriyor =
  **-37% decode** (bölüm 7). ARGUS köprüsü bu açığı kapatmadan önce net kayıpta başlar.
- **İş büyüklüğü:** kaynaktan parity build + `llama_kv_cache` seviyesinde C++ =
  haftalar.

ARGUS'un asıl değeri PyTorch/vLLM yolunda kalıyor (`models/attention_wrapper.py`,
`adapters/vllm.py`), llama.cpp tarafında değil. `adapters/ollama.py` kendi
docstring'inde bunu zaten dürüstçe yazıyor.

**Adım 3 yeniden değerlendirilir eğer:** (a) runtime PyTorch/vLLM'e taşınırsa,
(b) upstream llama.cpp MTP desteği eklerse, ya da (c) VRAM tekrar darboğaz olursa
(daha büyük model, çoklu slot `-np > 1`).
