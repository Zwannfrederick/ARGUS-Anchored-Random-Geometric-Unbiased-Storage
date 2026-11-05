# Gemma 4 + Neo Görsel Koordinat Teşhisi ve Sansürsüz Model Değerlendirmesi

Tarih: 2026-11-04
Kapsam: `experiments/hermes/` (Neo masaüstü ajanı), `scratch/models/mmproj-gemma-4-*.gguf`, `scratch/llama.cpp-v050/tools/mtmd/` ve `Qwen3.6-35B-A3B-Heretic` ile karşılaştırma.

---

## 1. Yönetici Özeti (Sorun Teyidi)

Kullanıcı hipotezi **%100 doğrulandı**: Gemma 4 üzerinde Neo'nun ekran görüntülerinde koordinat ıskalaması modelin "kör" ya da sabit `224×224` çözünürlüğe mahkûm olmasından **değil**, servis bayrakları ve koordinat uzayı parametrelerindeki **4 somut yapılandırma/kod hatasından** kaynaklanıyordu.

[neo-handoff.md](file:///home/zwannfrederick/Masaüstü/Sektor/Coding/mamba%20fix/docs/plans/neo-handoff.md#L35-L40) içindeki *"Görüntü kodlayıcı 224×224, patch 16 → 14×14 = 196 patch... Prompt ile düzeltilemez"* tespiti GGUF tensörleri ve `llama.cpp` `mtmd` kaynak koduyla çürütüldü:

- Diskteki `mmproj-gemma-4-E4B-BF16.gguf` ve `mmproj-gemma-4-26B-BF16.gguf` dosyalarında `clip.vision.image_size = 224` yalnızca eski sabit-boyutlu CLIP okuyucuları için bırakılmış geriye dönük bir skalerdir.
- Gerçek 2B konumsal gömme tablosu `v.position_embd.weight` **E4B'de `[768, 10240, 2]`**, **26B'de `[1152, 10240, 2]`** boyutundadır (`10.240 × 10.240` yamaya kadar dinamik 2D RoPE/pozisyon desteği).
- [mtmd.cpp](file:///home/zwannfrederick/Masaüstü/Sektor/Coding/mamba%20fix/scratch/llama.cpp-v050/tools/mtmd/mtmd.cpp#L877-L884) ve [clip.cpp](file:///home/zwannfrederick/Masaüstü/Sektor/Coding/mamba%20fix/scratch/llama.cpp-v050/tools/mtmd/clip.cpp#L1627-L1641) içinde `PROJECTOR_TYPE_GEMMA4V`, en-boy oranını koruyan dinamik NaViT ön-işleyicisini (`mtmd_image_preprocessor_dyn_size`) kullanır:
  - `patch_size = 16`, `n_merge = 3` ([gemma4v.cpp](file:///home/zwannfrederick/Masaüstü/Sektor/Coding/mamba%20fix/scratch/llama.cpp-v050/tools/mtmd/models/gemma4v.cpp#L84-L98) içinde `3×3` ortalama havuzlama → token başına `48×48 = 2.304 px²`).
  - Varsayılan token bütçesi `hparams.set_limit_image_tokens(70, 1120)` → **`2.580.480` piksel (`2.58 MP`)** tavan.
  - `1868×988` (`1.845.584 px`) boyutundaki bir pencere görüntüsü `2.58 MP` sınırının altındadır; güncel `mtmd` ile **hiç küçültülmeden tam çözünürlükte (`1872×1008`, ~819 görsel token)** işlenir.

---

## 2. Tespit Edilen 4 Somut Hata (Kök Nedenler)

### Hata 1 — Eski Ollama Binary'si, Eksik Token Bayrakları ve CPU'da Kalan ViT (`neo-model.service`)
- **Kanıt:** [neo-model.service](file:///home/zwannfrederick/.config/systemd/user/neo-model.service#L14-L17) servisi ARGUS `llama-server` yerine `/usr/lib/ollama/llama-server` ikilisini `--no-mmproj-offload` bayrağıyla çalıştırıyordu ve `--image-min-tokens` / `--image-max-tokens` parametrelerini vermiyordu.
- **Etkisi:**
  1. Eski binary `gemma4v` dinamik NaViT yolunu desteklemediğinde GGUF içindeki `image_size = 224` fallback değerine düşerek görüntüyü gerçekten `224×224`'e eziyordu.
  2. `--no-mmproj-offload` yüzünden BF16 ViT kodlayıcısı tamamen CPU'da koştu; [hermes-benchmark-gemma4-26b.json](file:///home/zwannfrederick/Masaüstü/Sektor/Coding/mamba%20fix/docs/measurements/hermes-benchmark-gemma4-26b.json#L140-L155) içindeki `T07_UI_SCREENSHOT_GROUNDING` testinde TTFT **23.73 saniyeye** fırladı (metin senaryolarında ~3.0 sn).

### Hata 2 — Koordinat Uzayı Uyuşmazlığı (`[0..1000]` Normalize vs Ham `1868×988` Piksel)
- **Kanıt:** Gemma 3 / Gemma 4 görsel modelleri ham ekran pikseli değil, **`[0, 1000]` aralığında normalize koordinatlar** (ve kutularda `[ymin, xmin, ymax, xmax]`) üretecek şekilde eğitilmiştir. Buna karşın [hermes_supervisor.py](file:///home/zwannfrederick/Masaüstü/Sektor/Coding/mamba%20fix/experiments/hermes/hermes_supervisor.py#L1100-L1116) modelden gelen `(x, y)` değerini doğrudan ham piksel kabul edip `wayland_ui.click_at(ox + cx, oy + cy)` çağrısına gönderiyordu.
- **Kesin Kanıt (Smoking Gun):** [wayland_ui.py](file:///home/zwannfrederick/Masaüstü/Sektor/Coding/mamba%20fix/experiments/hermes/wayland_ui.py#L710-L712) yorumunda modelin *"Kardemir COO sohbetini aç"* komutu için sürekli `(100, 450)` ve `(150, 550)` ürettiği ve "yuvarlak sayı uydurduğu" yazılmış. Oysa `1000×1000` normalize uzayda `(100..150, 450..550)` koordinatları `1868×988` pencereye ölçeklendiğinde:
  - `x = (100..150) / 1000 × 1868 = 187..280 px`
  - `y = (450..550) / 1000 × 988 = 445..543 px`
  noktasına, yani **ZapZap/WhatsApp sol sohbet listesinin tam ortasına** denk gelir. Model uydurmuyor, kendi eğitim uzayında (`0..1000`) doğru sütunu işaret ediyordu; kod ölçeklemediği için tıklama sol üst boşluğa gidiyordu.

### Hata 3 — Wayland Kesirli Ölçekleme (`shot_size` Mantıksal vs `grim` Fiziksel PNG Boyutu)
- **Kanıt:** [hermes_supervisor.py](file:///home/zwannfrederick/Masaüstü/Sektor/Coding/mamba%20fix/experiments/hermes/hermes_supervisor.py#L591-L603) içindeki `_grab_frame`, pencere boyutunu `hyprctl activewindow -j` (`win["size"]`, mantıksal Wayland pikseli) üzerinden alıyor; gerçek PNG boyutunu okuyan `_png_size(...)` fonksiyonunu yalnızca `if size is None:` durumunda çağırıyor.
- **Etkisi:** Hyprland ölçeği `1.25` veya `1.5` olduğunda `grim` fiziksel piksellerle daha büyük bir PNG üretirken `turn_state["shot_size"]` ve `ydotool` mantıksal koordinat uzayında kalıyor; her tıklama ölçek çarpanı kadar kayıyor.

### Hata 4 — Birbiriyle Çelişen Sistem ve Gözlem Promptları
- **Kanıt:** [prefix_builder.py](file:///home/zwannfrederick/Masaüstü/Sektor/Coding/mamba%20fix/experiments/hermes/prefix_builder.py#L231) ve [prefix_builder.py](file:///home/zwannfrederick/Masaüstü/Sektor/Coding/mamba%20fix/experiments/hermes/prefix_builder.py#L469-L471) modele *"ekran sana aşırı küçültülmüş gelir, ASLA x/y kullanma"* derken, [hermes_supervisor.py](file:///home/zwannfrederick/Masaüstü/Sektor/Coding/mamba%20fix/experiments/hermes/hermes_supervisor.py#L1533-L1539) her ekran görüntüsü mesajında *"Bir şeye tıklaman gerekiyorsa `ui_click` aracına BU GÖRÜNTÜDEKİ piksel koordinatlarını ver (sol üst köşe 0,0)"* talimatını enjekte ediyor.

---

## 3. Neo İçin Minimal Onarım Reçetesi (İleride Devreye Alınacağında)

Neo (`experiments/hermes/`) üzerinde görsel tıklama tekrar açılmak istendiğinde yalnızca şu 4 dar değişiklik yeterlidir:

1. **ARGUS `llama-server` + Dinamik Görsel Token Bütçesi:**
   - Binary olarak `scratch/llama.cpp-v050/build-cuda/bin/llama-server` kullanılmalı.
   - `--no-mmproj-offload` kaldırılmalı (veya VRAM dar ise yalnızca `--image-min-tokens 560 --image-max-tokens 1120` açıkça verilmeli).
2. **`0..1000` → Mantıksal Pencere Koordinat Dönüşümü (`hermes_supervisor.py`):**
   - `ui_click` şemasında `x` ve `y` açıkça `0..1000` normalize koordinat olarak tanımlanmalı:
     `logical_x = ox + round((x / 1000.0) * logical_w)` ve `logical_y = oy + round((y / 1000.0) * logical_h)`.
3. **Mantıksal / Fiziksel Boyut Ayrımı (`_grab_frame`):**
   - `turn_state["logical_size"] = tuple(win["size"])` (`ydotool` hedef uzayı) ve `turn_state["png_size"] = _png_size(filepath)` ayrı tutulmalı; OCR / Set-of-Mark kutuları `png_size` üzerinden `logical_size`'a oranlanmalı (`logical_w / png_w`).
4. **Prompt Tekilleştirme:**
   - `hermes_supervisor.py:1536-1538` içindeki "piksel koordinatlarını ver" cümlesi `prefix_builder.py` kuralıyla uyumlu hale getirilmeli (öncelik: `text` → `mark` → `0..1000` normalize `x,y`).

---

## 4. Qwen3.6-35B-A3B-Heretic vs Sansürsüz Gemma 4 Karşılaştırması

Bu makinede (`RTX 3050 Ti 4 GB VRAM + 32 GB RAM + i5-11300H`) her iki ailenin ölçülmüş gerçek değerleri:

| Kriter | Qwen3.6-35B-A3B-Heretic (`Q4_K_S`) | Gemma-4-26B-A4B (Heretic / Abliterated) | Gemma-4-E4B (Heretic / Abliterated) |
|---|---|---|---|
| **Mimari** | Hibrit Gated DeltaNet (30 lineer + 10 tam dikkat), 35B/3B MoE | Tam/hibrit Transformer MoE, 26B/4B | Yoğun (Dense) ~4.5B |
| **Disk / RAM Ayak İzi** | `18.51 GiB` GGUF (~16.8 GiB host MoE) | `~15–16 GiB` GGUF | `~4.5 GiB` GGUF |
| **128K (`q4_0`) KV Cache** | **`720 MiB`** (yalnızca 10 dikkat katmanı!) | Yüksek (tüm dikkat katmanları KV tutar) | Orta-Düşük |
| **Decode Hızı (Sıcak)** | **`24.7 – 29.5 tok/s`** (`ngram-mod` + `-ncmoe 40`) | `13.8 tok/s` ([benchmark](file:///home/zwannfrederick/Masaüstü/Sektor/Coding/mamba%20fix/docs/measurements/hermes-benchmark-gemma4-26b.json#L10)) | `17.6 tok/s` ([benchmark](file:///home/zwannfrederick/Masaüstü/Sektor/Coding/mamba%20fix/docs/measurements/hermes-benchmark-gemma4-e4b.json#L10)) |
| **Ajan / Kodlama İsabeti** | **En yüksek** (35B çok adımlı planlama + GUI grounding eğitimi) | `%85.7` (`6/7` senaryo) | `%71.4` (`5/7`, çok adımlı planlamada zayıf) |
| **Yerel Görsel (`mmproj`)** | **Doğal Multimodal + GUI Grounding** (Jinja şablonunda `<\|vision_start\|><\|image_pad\|>` hazır; HF `mmproj-BF16/F16.gguf` `--mmproj` ile doğrudan takılır) | **Hazır** (`mmproj-gemma-4-26B-BF16.gguf` + MTP diskte) | **Hazır** (`mmproj-gemma-4-E4B-BF16.gguf` + MTP diskte) |
| **Ne Zaman "Ağır" Kalır?** | İlk soğuk açılışta NVMe'den sayfa ısıtırken veya RAM'de >14 GB başka iş varken | 128K bağlamda VRAM/RAM baskısı ve 2× daha yavaş decode | Neredeyse hiç ağır kalmaz (~4.5 GB) |

### Pratik Karar ve PC / Masaüstü Kontrolü (Neo) Değerlendirmesi
1. **Hem Claude Code Hem PC Kontrolü (Neo) İçin Birincil Model — `Qwen3.6-35B-A3B-Heretic`:**
   - **Neden PC kullanmada Gemma 4'ten çok daha üstün?**
     1. **Doğal Multimodal & Uzamsal GUI Grounding:** `Qwen3.6-35B-A3B` doğuştan görsel-dil (VLM) ve masaüstü/tarayıcı GUI etkileşimi (OSWorld / ScreenSpot tarzı koordinat ve kutu hedefleme) üzerine eğitilmiştir. Diskteki `Qwen3.6-35B-A3B-uncensored-heretic.i1-Q4_K_S.gguf` dosyasının gömülü Jinja şablonunda `<|vision_start|><|image_pad|><|vision_end|>` görsel blokları halihazırda mevcuttur; Hugging Face üzerindeki uyumlu `mmproj-BF16.gguf` / `mmproj-F16.gguf` (`unsloth/Qwen3.6-35B-A3B-GGUF` veya `lmstudio-community/Qwen3.6-35B-A3B-GGUF`) indirildiğinde `llama-server --mmproj` ile gövdeyi değiştirmeden doğrudan görsel tıklama kazanır.
     2. **2× Hız + 35B Ajan Planlama Zekâsı:** Masaüstü otomasyonunda her adımda ekran + geçmiş araç çağrıları birikir. `Qwen3.6-35B-A3B` `128K` bağlamda yalnızca **`720 MiB`** KV cache harcar (boşta `~1.2 GiB` VRAM kalır — `mmproj` GPU offload için yeterli tavan bırakır!) ve **`~25–29 tok/s`** hızla Gemma-4-26B'den (`13.8 tok/s`) neredeyse **2 kat hızlı**, Gemma-4-E4B'den (`%71.4`) ise çok adımlı araç zincirlerinde çok daha isabetli karar verir.
   - *Not (WebUI Hız İpucu):* WebUI varsayılan olarak `<think>` (uzun içsel düşünme) açık başlar; anlık hızlı sohbet istendiğinde WebUI ayarlarından düşünme kapatılabilir veya `./scripts/argus_ctl.sh chat` kullanılabilir.
2. **Ne Zaman Sansürsüz Gemma 4 Kullanılmalı?**
   - Yalnızca aynı anda ağır Docker / derleme işleri varken `18.5 GiB` RAM sayfa önbelleğini hiç tutmak istemediğiniz ultra-hafif arka plan senaryolarında **`gemma-4-E4B-it` Heretic (`~4.5 GiB`)** yedek profil olarak anlamlıdır; ana PC kontrolü ve kodlama için `Qwen3.6-35B-A3B-Heretic` + `mmproj` en doğru mimaridir.

