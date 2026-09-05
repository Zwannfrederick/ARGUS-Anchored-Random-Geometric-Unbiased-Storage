# ARGUS Codebase Deep Control & Architecture Audit Report
**Tarih:** 2026-09-04  
**Kapsam:** `/control` protokolü kapsamında mimari graf, test suite derleme/çalıştırma, sistemik bug avı (P0), dosya boyutları/sorumluluk sınırları (P1), Ponytail & DRY sadeleştirmeleri (P2) ve ölü kod analizi (P3).

---

## 1. Mimari Graf ve Statik Doğrulama Özeti
- **AST Bilgi Grafı:** `codebase-memory-mcp` üzerinde 2,279 düğüm, 10,348 kenar, 101 Python ve 4 C++ dosyası haritalandı.
- **Test Suite Doğrulaması:** `.venv/bin/python -m pytest` ile toplam **363 testten 362'si BAŞARILI**, 1'i atlandı (`test_vllm_live_probe_fails_closed` - vLLM opsiyonel olduğu için).
- **Canlı Çalışma Durumu:** `test_ollama_live_generate_reports_real_timings` testi yerel Ollama üzerindeki `qwen3.6-35b-a3b:latest` (22 GB) modelini başarıyla yükleyip gerçek zamanlı token üretimi gerçekleştirdi.
- **Sözdizimi Derlemesi:** Projedeki tüm Python dosyaları `py_compile` ile hatasız derlendi.

---

## 2. P0 - Kritik Hatalar (Systemic Bugs & Race Conditions)

### [P0.1] Port 8000 Çakışması ve Yanıltıcı "ARGUS ● READY" Durumu (False-Positive Hijack)
- **Hedef Dosya:** [`scripts/argus_service.py:69-100, 130-164`](file:///home/zwannfrederick/Masaüstü/Sektor/Coding/mamba fix/scripts/argus_service.py#L69-L164)
- **Tespit Edilen Somut Risk:**
  - `check_gateway_health(port=8000)` fonksiyonu yalnızca `http://127.0.0.1:8000/health` adresinden HTTP 200 dönüp dönmediğine bakıyor.
  - Sistemde başka bir projeye ait (FastAPI/Uvicorn + Postgres/Redis) yabancı bir servis 8000 portunu dinliyor ve `/health` sorgusuna 200 dönüyor.
  - `gateway.pid` dosyası olmamasına ve ARGUS Gateway hiç çalışmamasına rağmen `argus_ctl status` sistemi **"ARGUS ● READY"** olarak raporluyor.
  - Ayrıca `check_backend_health()` 8080 portundaki llama-server kapalı olduğunda Ollama'ya (11434) fallback yaparak `backend_ok = True` döndürüyor; ancak durum çıktısında kapalı olan `http://127.0.0.1:8080/v1 (🟢 OK)` yazıyor! Bu durum gateway'e istek atıldığında anında çökme/bağlantı hatasına yol açar.
- **Önerilen Minimal Çözüm:**
  1. `check_gateway_health()` içinde dönen JSON yanıtında ARGUS Gateway imzası aranmalı (`resp.json().get("upstream") is not None` veya özel header).
  2. PID dosyası doğrulanmadan `READY` durumuna geçilmemeli.
  3. LLM backend sağlık kontrolü ile raporlanan upstream URL birebir eşleştirilmeli; Ollama aktifse URL açıkça `11434` olarak raporlanmalı.
  4. Port 8000 başka bir servis tarafından kullanılıyorsa `Port conflict: port 8000 occupied by foreign process` uyarısı verilerek alternatif porta (örn: 8008) geçiş seçeneği sunulmalı.

---

### [P0.2] Waybar / Polling Döngüsünde `import torch` Kaynaklı I/O ve CPU Fırtınası
- **Hedef Dosya:** [`scripts/argus_service.py:103-120`](file:///home/zwannfrederick/Masaüstü/Sektor/Coding/mamba fix/scripts/argus_service.py#L103-L120) ve [`scripts/argus_waybar.sh`](file:///home/zwannfrederick/Masaüstü/Sektor/Coding/mamba fix/scripts/argus_waybar.sh)
- **Tespit Edilen Somut Risk:**
  - Waybar sürekli olarak `argus_waybar.sh status` çağırarak `python argus_service.py status --json` çalıştırıyor.
  - `get_vram_metrics()` fonksiyonu VRAM miktarını öğrenmek için **her durum sorgusunda `import torch` yapıyor** ve CUDA çalışma zamanını başlatıyor!
  - 300+ MB PyTorch kütüphanesinin disktan/swap'tan tekrar tekrar yüklenmesi, sistemde %32'ye varan I/O wait (`bi: 92616`, CPU wait state) oluşturarak makineyi kilitliyor ve testlerin dakikalarca beklemesine sebep oluyor.
- **Önerilen Minimal Çözüm:**
  - Ponytail Merdiveni: CLI durum sorgusunda PyTorch import etmek yerine VRAM bilgisi `nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits` ile veya `/sys/class/drm` üzerinden <5ms içinde stdlib ile okunmalı.

---

### [P0.3] Online Softmax Recurrence İçinde `-inf` Kaynaklı NaN Zehirlenmesi
- **Hedef Dosya:** [`argus_cache/core/direct_attention.py:145-153`](file:///home/zwannfrederick/Masaüstü/Sektor/Coding/mamba fix/argus_cache/core/direct_attention.py#L145-L153)
- **Tespit Edilen Somut Risk:**
  - `DirectPagedAttentionEngine.decode_single_token` içinde `running_max` başlangıçta `-inf` olarak tanımlanır.
  - Eğer ilk bloktaki attention skorları maske veya sıfır token nedeniyle tamamen `-inf` gelirse, `next_max = -inf` olur.
  - Bu durumda `running_max - next_max` işlemi `(-inf) - (-inf) = NaN` üretir.
  - `torch.exp(NaN)` sonucu tüm `running_out` ve `running_sum` tensörleri NaN ile zehirlenir ve sonraki tüm token üretimleri bozulur.
- **Önerilen Minimal Çözüm:**
  - `next_max` `-inf` olduğunda veya `running_max` `-inf` iken `rescale` doğrudan `0.0` kabul edilmeli ve çıkarma işlemine girmeden guard konulmalı.

---

### [P0.4] SSE Streaming Gateway İçinde Sabit/Sahte `output_tokens: 10` Dönüşü
- **Hedef Dosya:** [`argus_cache/adapters/claude_gateway.py:747`](file:///home/zwannfrederick/Masaüstü/Sektor/Coding/mamba fix/argus_cache/adapters/claude_gateway.py#L747)
- **Tespit Edilen Somut Risk:**
  - Claude API streaming (`message_delta`) event'inde token kullanımı sabit kodlanmış: `"usage": {"output_tokens": 10}`.
  - Model ister 5 token ister 4000 token üretsin, Claude Code ve istemciler sürekli 10 token üretildiğini varsayarak context penceresi ve maliyet/limit hesaplamalarını yanlış yapar.
- **Önerilen Minimal Çözüm:**
  - Stream boyunca üretilen text/thinking/tool chunk'larının token sayısı (veya kelime/karakter bazlı tahmin/upstream delta) toplanarak gerçek değer `message_delta` içinde gönderilmeli.

---

### [P0.5] `get_all_keys_values` Fonksiyonunda Senkronizasyon Kilidi Eksikliği (Race Condition)
- **Hedef Dosya:** [`argus_cache/core/memory_manager.py:1620-1650`](file:///home/zwannfrederick/Masaüstü/Sektor/Coding/mamba fix/argus_cache/core/memory_manager.py#L1620-L1650)
- **Tespit Edilen Somut Risk:**
  - `push_new_tokens` ve `inplace_paged_attention` fonksiyonları `with self._attention_lock:` ile korunurken, `get_all_keys_values()` metodu kilitsiz çalışıyor.
  - Çok iş parçacıklı veya asenkron prefetch/spill senaryolarında `self.pages_by_tier` veya `self.active_pages` iterate edilirken liste üzerinde değişiklik yapılırsa `RuntimeError: dictionary/list changed size during iteration` fırlar veya eksik/bozuk tensor parçaları birleştirilir.
- **Önerilen Minimal Çözüm:**
  - `get_all_keys_values` fonksiyonunun tamamı `with self._attention_lock:` bloğu içine alınmalı.

---

## 3. P1 - Mimari & Dosya Boyutu Standartları (Hard Ceiling)

| Dosya Yolu | Satır Sayısı | Durum | Ayrıştırma / Modül Önerisi |
| :--- | :--- | :--- | :--- |
| [`argus_cache/core/memory_manager.py`](file:///home/zwannfrederick/Masaüstü/Sektor/Coding/mamba fix/argus_cache/core/memory_manager.py) | **1898** satır | **Kritik İhlal** (>1000 tavanı) | 4 modüle ayrılmalı: `compiled_attention.py`, `page_dict.py`, `eviction_controller.py`, `swap_engine.py` |
| [`argus_cache/csrc/manager.cpp`](file:///home/zwannfrederick/Masaüstü/Sektor/Coding/mamba fix/argus_cache/csrc/manager.cpp) | **1055** satır | **İhlal** (>1000 tavanı) | `ggml_codecs.cpp` ve `cuda_dispatch.cpp` olarak ikiye bölünmeli |
| [`scripts/argus_service.py`](file:///home/zwannfrederick/Masaüstü/Sektor/Coding/mamba fix/scripts/argus_service.py) | **961** satır | **Sınırda** (800+ limit) | CLI/TUI gösterimi `argus_service_cli.py` dosyasına alınmalı |
| [`argus_cache/adapters/claude_gateway.py`](file:///home/zwannfrederick/Masaüstü/Sektor/Coding/mamba fix/argus_cache/adapters/claude_gateway.py) | **878** satır | **Sınırda** (800+ limit) | Streaming SSE ayrıştırıcısı `claude_sse_stream.py` içine taşınmalı |

---

## 4. P2 - Pragmatik DRY & Ponytail Sadeleştirmeleri

1. **Tekrarlayan Ağır HTTP İstemcileri (`httpx.Client` Spawns):**
   - `argus_service.py` içinde her durum sorgusunda 4-5 kez `httpx.Client()` oluşturulup yok ediliyor. Stdlib `urllib.request` veya tek bir oturum (session) ile bağlantı havuzu kullanılmalı.
2. **Çift/Üçlü Endpoint Sorgulaması:**
   - Hem `check_backend_health` hem de `get_detailed_status` Ollama'nın `/api/tags` endpoint'ine arka arkaya iki kez istek gönderiyor. Tek bir istek yapılıp sonuç paylaşılmalı.
3. **HTTP Kütüphanesi Uyuşmazlığı:**
   - `ollama.py` stdlib `urllib.request` tabanlı `HttpTransport` kullanırken, `claude_gateway.py` harici `httpx` kullanıyor. Ponytail prensibine göre hafif olan stdlib transport standardına çekilmeli.

---

## 5. P3 - Ölü Kod ve Git Takibi

1. **İzlenmeyen (Untracked) Yeni Mimari Bileşenler:**
   - Aşağıdaki dosyalar başarıyla tamamlanmış ve tüm testleri (%100) geçmektedir; ancak henüz git takibine alınmamıştır:
     - `argus_cache/core/backend_pool.py` (Contiguous pool)
     - `argus_cache/core/direct_attention.py` (Direct attention engine)
     - `argus_cache/core/page_table.py` (Structure-of-Arrays page table)
     - `argus_cache/models/hybrid_cache.py` (Qwen hybrid DeltaNet/KV cache)
     - `tests/test_hybrid_qwen_cache.py`
     - `tests/test_direct_paged_attention.py`
     - `tests/test_claude_gateway.py`
     - `tests/test_argus_service.py`
2. **Temizlenecek Geçici / Artık Dosyalar:**
   - `.system_generated` altındaki geçici loglar ve test artifact'leri.

---

## 6. Kullanıcı Talebine Yönelik Eylem Planı: Qwen 35B A3B'nin Ollama ile Çalıştırılması

Kullanıcının RTX 3050 Ti Laptop GPU (4 GB VRAM) ve 32 GB RAM konfigürasyonunda Qwen 3.6 35B A3B modeli için durum:

1. **Model Durumu:**
   - `scratch/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` (21 GB) diskte mevcut.
   - Ollama içinde `qwen3.6-35b-a3b:latest` (22 GB) olarak zaten kayıtlı.
   - `Modelfile_qwen36_35b` context boyutu `num_ctx 65536` olarak yapılandırılmış.

2. **Ollama ile Doğrudan Çalıştırma (Önerilen Ortam Değişkenleri):**
   - Standart Ollama MoE katmanlarını CPU'ya sabitleyen `--cpu-moe` bayrağını desteklemez. Ancak KV önbelleğini küçültmek için şu değişkenler **şarttır**:
     ```bash
     export OLLAMA_KV_CACHE_TYPE=q4_0
     export OLLAMA_FLASH_ATTENTION=1
     ollama run qwen3.6-35b-a3b:latest
     ```
   - Bu sayede 4 GB VRAM aşımı engellenir ve bellek taşması minimuma iner.

3. **Maksimum Performans Alternatifi (llama-server ile 18-19 tok/s):**
   - Projedeki ölçümlere göre `llama-server --cpu-moe -ctk q4_0 -ctv q4_0 -fa on` kombinasyonu, uzmanları 32 GB RAM'de tutup dikkat ağırlıklarını 4 GB VRAM'de çalıştırarak **19.14 tok/s** üretim hızı sağlamaktadır (standart Ollama'ya göre ~1.6x daha hızlı).
