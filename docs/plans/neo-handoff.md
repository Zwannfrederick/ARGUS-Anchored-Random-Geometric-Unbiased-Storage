# Neo — Deney Sonu Notu ve Devir Belgesi

Tarih: 2026-09-05. Durum: **duraklatıldı**, servisler devre dışı.
Amaç: daha güçlü bir setup'ta devam etmek için ne öğrendiğimizi ve nerede
bıraktığımızı kaydetmek.

## Ne inşa edildi

Telefondan kontrol edilen yerel masaüstü orkestratörü:

```
neo-mobile/ (React Native)  →  hermes/ (Starlette gateway, :8765)  →  llama-server (:8080, Gemma 4 E4B)
                                        ↓
                               Hyprland / Wayland masaüstü
```

Dosyalar (`hermes/`, ~5.8k satır):

| Dosya | İş |
|---|---|
| `neo_mobile_gateway.py` | HTTP/SSE API, oturum, precompaction (60K'da) |
| `hermes_supervisor.py` | Araç döngüsü, guard'lar, `send_chat_message` bileşik akışı |
| `prefix_builder.py` | Byte-stable kanonik prefix + 14 araç şeması (KV cache için) |
| `wayland_ui.py` | Ekran görüntüsü, OCR, Set-of-Mark, tıklama, klavye |
| `agent_bridge.py` | hermes-agent kayıt defterine tek dispatcher üzerinden köprü |
| `cognitive_router.py` | Risk → thinking on/off, onay kapısı |
| `test_send_guard.py` | 12 test; guard'ların hepsi kapsanıyor |

Test durumu son tam koşuda: **22 passed**.

## Ölçülen sert kısıtlar

**Bunlar mimari kararları belirledi; yeni setupta yeniden ölçülmeli.**

1. **Görüntü kodlayıcı 224×224, patch 16 → 14×14 = 196 patch.**
   1868×988'lik bir kare hücre başına ~133×71 ekran pikseline düşüyor — bir sohbet
   satırından daha geniş. **Model ekran görüntüsünü okuyabiliyor ama içinde
   konum belirleyemiyor.** Bu, "AI baksın ve tıklasın" fikrinin neden çalışmadığının
   tek ve yeterli açıklaması. Prompt ile düzeltilemez.

2. **GPU: RTX 3050 Ti Laptop, 4096 MiB toplam.** Model yüklüyken ~400 MiB boştu.
   OmniParser (YOLOv8 + Florence-2, ≥8 GB) gibi hazır çözümler bu yüzden elendi.

3. **KV prefix caching %99.8 hit** — ama yalnızca prefix byte düzeyinde sabitse.
   Herhangi bir şema/prompt değişikliği cache'i düşürüyor (soğuk ~15-30 sn, sıcak ~1.5 sn).
   Bu yüzden `prefix_builder.py` araç sırasını sabit tutuyor.

4. **E4B planlama yapamıyor.** "Sohbeti aç → yaz → gönder" üç adımını güvenilir
   biçimde sıraya koyamadı. Araç çeşitliliği arttıkça seçim doğruluğu düşüyor
   (paralel araçları karıştırdı; 12 kez aynı aracı çağırdı).

## Çalışan çözümler (yeniden kullanılabilir)

- **Pano ile yazma** — `wl-copy` + Ctrl+V. `ydotool type` klavye düzenine bağlı
  (`trq` düzeninde tüm Türkçe karakterler düşüyordu). Pano yöntemi düzenden
  bağımsız ve Unicode-safe. Canlı doğrulandı: "Kezban durağına yürüyor... ŞĞÜİÖÇ" ✓
- **OCR ile tıklama** — tesseract `--psm 11` tsv, geometrik satır gruplama,
  katlanmış eşleşme + gevşek OCR-karışıklık geri düşüşü (ç↔g gibi).
  Eşleşmeler **soldan sağa** sıralanıyor: masaüstü uygulamaları master-detail,
  sol sütun tıklanacak liste, sağdaki aynı isim açık öğenin başlığı.
- **Set-of-Mark** — tıklanabilir öğeleri numaralandır, model piksel yerine
  numara seçsin. Canlı çalıştı (model `mark: 12` ile arama kutusunu ilk denemede buldu).
- **Küçük modelde araç birleştirme** — benzer araçları modlu tek araca indir
  (`ui_click`: text/mark/x-y; `hermes_skill`: search+run). Ayrı araçlar karıştırılıyor.
- **Akışı koda alma** — `send_chat_message` planlamayı modelden alıp koda koyuyor.
  Küçük modelle çalışmanın tek güvenilir yolu bu.
- **Guard'lar** — alıcı ekrandan OCR ile doğrulanmadan Return yok; taslak kutudan
  kaybolmadan "gönderildi" denmiyor; tur genelinde döngü sayacı. Bunlar yalan
  başarı raporlarını bitirdi.

## Bir sonraki setup için yön

Deneyin dürüst sonucu: **bu donanımda GUI otomasyonu doğru problem değildi.**
Kişisel asistan olarak konumlandırmak daha isabetli, çünkü:

- API üzerinden yönetilebilen işler (arama, mail, takvim, Telegram, dosya)
  metin girer/metin çıkar — 224×224 kodlayıcı sınırı hiç devreye girmiyor.
- Arka planda çalışan iş için yavaş decode (7-12 tok/s) önemsiz.
- Hata profili tersine dönüyor: kötü araştırma çıktısı **ucuz ve görünür**;
  yanlış kişiye giden mesaj **sessiz ve geri alınamaz**.

**Pratik en büyük kazanç: mesajlaşmayı WhatsApp'tan Telegram'a taşımak.**
Kişisel WhatsApp'ın API'si yok — bu yüzden OCR+tıklama zincirine mahkumduk.
Telegram Bot API ile aynı iş tek fonksiyon çağrısı; ne yanlış alıcı riski,
ne karakter derdi, ne gönderim doğrulaması gerekiyor.

Önerilen sıra (yapılmadı, yeni setupa bırakıldı):
1. Kalıcı proje/görev deposu — her koşu bir öncekinin üstüne binsin.
   **Bu olmazsa 4 saatte bir aynı aramayı yapan bir şey elde edilir.**
2. Zamanlayıcı (systemd timer) + araştırma turu: açık soruları oku → sadece
   onları araştır → kaynak referanslı ekle → açık soruları güncelle.
3. Telegram — hem çıktı kanalı hem asistanın konuştuğu yer.
4. Gmail + Takvim — okuma serbest, yazma onay arkasında.
5. Panel — en sona; ilk dördü değerini kanıtladıktan sonra.

UI otomasyonu silinmedi. Okuma tarafı (ekran görüntüsü, ekran OCR'ı) güvenli ve
faydalı. Yazma/tıklama tarafı yalnızca açık istekle açılmalı.

## Kapatma durumu

```
systemctl --user disable --now neo.target neo-gateway neo-model ydotoold
```
Hepsi durduruldu ve otomatik başlatmadan çıkarıldı. VRAM 3669 MiB → 15 MiB.
8765 ve 8080 portları boş.

Yeniden başlatmak için:
```
systemctl --user enable --now neo.target
```
Unit dosyaları duruyor, kod duruyor, testler geçiyor. Hiçbir şey silinmedi.

## Bitmemiş işler

- `send_chat_message` uçtan uca canlı testi tamamlanmadı (arka planda 180 sn'yi
  aştı, muhtemelen OCR turlarında bekliyordu). Guard'ların birim testleri geçiyor.
- `test_hermes.py::test_live_low_risk_fast_action` prefix değişimi sonrası soğuk
  KV cache yüzünden ara sıra düşüyor; sıcak koşuda geçiyor. Gerçek hata değil,
  ama test bir warmup beklemeli.
- `hermes/` git'e hiç eklenmedi (untracked). İki kez `prefix_builder.py`
  index tabanlı düzenlemeyle bozuldu ve geri dönülecek commit yoktu.
  **Yeni setupta ilk iş: commit et.**
