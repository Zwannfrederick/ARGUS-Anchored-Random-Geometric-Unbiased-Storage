# Neo Mobile & Hermes Autonomous Agent Loop Audit & Implementation Plan

**Tarih:** 2026-09-05  
**Revizyon:** 2.0 (Mimari ve Üretim Kısıtları Güncellemesi)  
**Durum:** Tasarım Onaylandı — Uygulama Öncesi Referans Spesifikasyonu

---

## Revision Summary (Yapılan Değişiklikler)

1. **Auto-Title & Production Slot Koruması:** Tek slotlu (`-np 1`) production llama-server üzerinde Fast-Path KV önbelleğini (`~4731` token, `%99.7` hit) bozacak yardımcı LLM çağrıları kesin olarak yasaklandı. Otomatik başlık üretimi deterministik/heuristik kural motoruna çekildi.
2. **Tool Parser Öncelik Hiyerarşisi:** Ayrıştırma sırası kesinleştirildi: `A) Native OpenAI tool_calls` → `B) <tool_call> fallback` → `C) Strict JSON fallback`. Güvenlik doğrulamaları (kanonik araç kontrolü, şema validasyonu, prose içi JSON izolasyonu) eklendi.
3. **Autonomous Loop & Native Tool Message Semantiği:** Çok adımlı döngüde `role: user` yerine native `role: tool` (`tool_call_id`) yapısı tanımlandı. `max_steps=5`, ardışık aynı araç çağrısını önleyen döngü muhafızı (`loop guard`) ve yapılandırılmış hata geri beslemesi eklendi.
4. **Internal Orchestration İzolasyonu:** Mobil arayüzde çiğ JSON, tool argümanları, iç muhakeme (`thinking/reasoning`) ve ara planlama metinleri gizlendi. Tek bir dinamik bekleyen balon (`pending container`) üzerinden sadece insani durum etiketleri (`public_label`) gösterilecek şekilde state machine kurgulandı.
5. **İşlem/İstek Bazlı Mesaj Tekilleştirme:** Çift mesajlaşma sorunu `client_msg_id` ve turn korelasyonu ile çözüldü; reconnect/replay durumlarında mükerrer balon oluşumu engellendi.
6. **Onay Mekanizmasının (Approval Gate) Döngüye Entegrasyonu:** Yüksek riskli araçlarda döngünün `future` ile askıya alınması, mobil karttan onay/ret kararı gelince oturumun yeni mesaj gerekmeksizin kaldığı yerden sürmesi sağlandı.
7. **Ölçülebilir Warmup ve Gecikme Hedefleri:** Gerçekçi olmayan ifadeler kaldırılarak Fast-Path için `%99+` önbellek yeniden kullanımı ve kısa/düşük riskli sorgularda `TTFT <= 1–2s` hedefi konuldu. Warmup esnasında gelen istekler için sınırlı kuyruk (`bounded queue`) tanımlandı.
8. **Fast Path Last İnvariantı:** `-np 1` slot yapısı gereği önce Reasoning Path, EN SON Fast Path çalıştırılarak Slot 0'da kalıcı resident cache tutulması kurala bağlandı.
9. **Mobil Tool Progress State Machine:** `tool_progress` olay şeması (`turn_id`, `step`, `tool_name`, `status`, `public_label`) ve kullanıcı dostu Türkçe rozet eşlemeleri belirlendi.
10. **Mobil Klavye ve Responsiveness:** Yatay WhatsApp/ChatGPT stili tek satır input, dinamik `useSafeAreaInsets`, `FlatList` dokunma sürekliliği ve ayarlar modalı `ScrollView` yapısı korundu.
11. **Kapsamlı Test Matrisi:** Uygulama sonrası koşulacak 10 zorunlu regresyon ve kabul testi (A-J) dokümante edildi.
12. **Non-Negotiable Invariants:** Sistemin temel güvenlik, bağlam (128K), sıkıştırma (60K), Tailscale ve donanım sınırlarını koruyan katı kurallar listesi eklendi.

---

## 1. Tespit Edilen Problemler ve Kök Neden Analizi

### Problem 1: Çift Mesajlaşma (Double Bubble / Duplicate Message)
- **Kök Neden:** `neo-mobile/src/screens/ChatScreen.tsx` içinde kullanıcı mesaj gönderdiğinde geçici bir asistan balonu oluşturuluyor:
  ```ts
  const pendingAssistantMessage = {
    id: `pending-${Date.now()}`,
    role: 'assistant',
    isPending: true,
    content: '',
  };
  ```
  Streaming deltaleri bu `pendingAssistantMessage` içerisine yazılıyor. İşlem tamamlandığında backend `type: "message"` ile nihai mesajı (`id: msg_1788566..._asst`) gönderiyor.
  `ChatScreen.tsx` içindeki `unsubMessage` dinleyicisi `m.id === newMsg.id` kontrolü yapıyor. Geçici ID (`pending-...`) ile sunucunun ürettiği ID (`msg_...`) eşleşmediği için bekleyen mesaj listeden silinmiyor; nihai mesaj da listenin sonuna yeni bir eleman olarak ekleniyor. Sonuç olarak kullanıcı ekranda aynı mesajı iki kere görüyor.

### Problem 2: Ham Tool JSON Çıktısının Sohbete Sızması (Tool Call Leak)
- **Kök Neden:** Model (Gemma-4 E4B), OpenAI native tool call yerine doğrudan metin formatında markdown JSON ürettiğinde (`{ "tool_name": "wayland_list_windows", "params": {} }`), `hermes_supervisor.py` sadece `<tool_call>{...}</tool_call>` etiketini aradığı için bu JSON'ı yakalayamıyor.
- Sonuç:
  1. Tool hiç çalıştırılmıyor.
  2. Ham JSON metin olarak kullanıcıya mesaj balonunda gösteriliyor.

### Problem 3: Tek Adımlı Yürütme vs Otonom Çok Adımlı Agent Döngüsü
- **Kök Neden:** Mevcut `execute_session_turn` tek seferlik çalışıyor (`single-pass`). Model bir tool çağrısı ürettiğinde tool çalışsa bile sonuç modele geri beslenip bir sonraki adım ("Adım 2: Zapzap açılıyor...") tetiklenmiyor. Kullanıcı araya girip "açılmadıysa şunu yap" diye tekrar yazmak zorunda kalıyor.

### Problem 4: Mobil UI Tool İlerleme Rozeti ve Dönen Çark
- **Kök Neden:** Tool tetiklendiğinde kullanıcı arayüzde çiğ parametreler yerine sade bir rozet, dönen çark ve kilitli bir girdi alanı görmek istiyor. Ara adımlar tamamlanana kadar kullanıcının akışı bozması engellenmeli.

### Problem 5: Yeni Oturum Açma Maliyeti & Warmup/Cache Sapması (Slot Eviction)
- **Kök Neden:** `journalctl` kayıtlarında `task 297` incelendiğinde yeni bir oturum ilk açıldığında LCP benzerliğinde sapma yaşandığı (`f_keep = 0.889`) ve 518 token'ın yeniden değerlendirildiği görüldü:
  ```text
  prompt eval time = 29930.16 ms / 518 tokens (17.31 tokens/s) | total time = 32.7s
  ```
  Prompt değerlendirme hızı CPU/offload katmanında ~17 token/s olduğu için, 500 token'lık bir önbellek kaybı ilk mesajda 30 saniyelik bir donmaya yol açıyor.

### Problem 6: Başlıksız Sohbetler ("New Session" Karmaşası)
- **Kök Neden:** Açılan tüm oturumlar "New Session" olarak kalıyor; ancak bunu çözmek için production LLM slotuna auxiliary istek atmak resident KV cache'i tahrip edebilir.

### Problem 7: Mobil Klavye ve Girdi Alanı Responsiveness
- **Kök Neden:** `ChatInput.tsx` dikey yerleşimli, `SafeAreaView` klavye açıkken de alt navigasyon boşluğunu korumaya çalışıyor ve `FlatList` üzerinde klavye süreklilik bayrakları eksik.

---

## 2. Uygulama ve Düzeltme Reçetesi (Eylem Planı)

### Adım 1: İstek/Korelasyon Bazlı Mesaj Tekilleştirme (`ChatScreen.tsx`)
Bekleyen mesaj takibi turn korelasyonu ile yapılacak:
1. Kullanıcı mesaj gönderdiğinde:
   ```ts
   const turnCorrelationId = clientMsgId || `turn_${Date.now()}`;
   const pendingAssistantMessage: ChatMessage = {
     id: `pending_${turnCorrelationId}`,
     session_id: session.session_id,
     role: 'assistant',
     content: '',
     isPending: true,
     turn_id: turnCorrelationId,
     tool_calls: [],
   };
   ```
2. `unsubMessage` içinde:
   - Eğer listede `turn_id` eşleşen veya `isPending: true` olan bir asistan mesajı varsa, o mesaj **doğrudan gelen nihai mesaj ile değiştirilecek** (`replace in-place`).
   - Listenin sonuna mükerrer kopya eklenmeyecek.
   - Reconnect veya replay olaylarında `id` kontrolü ile mutlak tekillik sağlanacak.

---

### Adım 2: Tool Çağrısı Ayrıştırma Hiyerarşisi ve Güvenlik Filtresi (`hermes_supervisor.py`)

Ayrıştırma sırası strictly şu şekilde işletilecek:
1. **A. Native OpenAI Tool Calls:** Streaming `delta.tool_calls` veya response `message.tool_calls` önceliklidir.
2. **B. `<tool_call>{...}</tool_call>` Fallback:** Metin içinde XML benzeri etiket varsa regex ile ayrıştırılır.
3. **C. Strict JSON Fallback (Uyumluluk Modu):** Yalnızca A ve B bulunamadıysa, metin içindeki ````json ... ```` veya ham JSON aranır.

**Güvenlik ve Doğrulama Kuralları:**
- `tool_name` kanonik araç listesinde (`CANONICAL_HERMES_TOOLS`) kayıtlı olmak **zorundadır**.
- Argümanlar JSON şemasına göre doğrulanır; hatalı veya bilinmeyen araç JSON'ları yürütmeye gönderilmez.
- Düz sohbet metni içindeki örnek kod veya rastgele JSON blokları araç çağrısı olarak algılanamaz.
- Araç çağrısı olarak ayrıştırılan JSON blokları, kullanıcıya gösterilecek nihai metinden (`full_content`) tamamen ayıklanır (`strip`); kullanıcıya çiğ JSON asla gösterilmez.

---

### Adım 3: Native Tool Message Semantiği ile Çok Adımlı Otonom Ajan Döngüsü

Model geçmişi strictly OpenAI formatını koruyacaktır:

```yaml
assistant:
  tool_calls:
    - id: "call_abc123"
      type: "function"
      function:
        name: "wayland_list_windows"
        arguments: "{}"

tool:
  tool_call_id: "call_abc123"
  content: '{"windows": [{"id": 1, "title": "Kitty"}]}'

assistant:
  content: "ZapZap açık değil, başlatıcı üzerinden açıyorum..."
  tool_calls:
    - id: "call_def456"
      type: "function"
      function:
        name: "wayland_focus_or_launch"
        arguments: '{"app_name": "zapzap"}'
```

**Döngü Kuralları:**
1. `max_steps = 5` (yapılandırılabilir).
2. Model tool çağırdığı sürece döngü kullanıcı "devam et" yazmadan kendi kendine ilerler.
3. **Loop Guard:** Eğer aynı araç aynı argümanlarla art arda 2 kez çağrılırsa döngü kırılır; hata structured tool result olarak modele verilir.
4. Tool çalışma hatası alırsa, hata exception fırlatmaz; JSON hata nesnesi olarak modele döner ve model telafi adımı dener.
5. `max_steps` aşılırsa kullanıcıya kontrollü bir kısmi tamamlanma/durum raporu sunulur.

---

### Adım 4: Mobil UI State Machine ve İlerleme Rozetleri

Kullanıcı normal modda iç orkestrasyonu, ham JSON'ı veya iç muhakemeyi görmez.

**State Machine:**
```text
Kullanıcı Mesajı Gönderdi
  │
  ├──► Input Alanı Kilitlenir (disabled = true)
  │
  ├──► TEK bir Pending Asistan Balonu Açılır
  │      │
  │      ├──► Tool 1 Çalışıyor: [⚙️ Pencereler kontrol ediliyor...]
  │      │
  │      ├──► Tool 1 Tamamlandı: [✓ Pencereler kontrol edildi]
  │      │
  │      ├──► Tool 2 Çalışıyor: [⚙️ ZapZap açılıyor...]
  │      │
  │      └──► Model Nihai Yanıtı Üretiyor (Streaming text)
  │
  ├──► Nihai Asistan Yanıtı Balona Yerleşir (isPending = false)
  │
  └──► Input Alanı Açılır (disabled = false)
```

**Event Formatı (`tool_progress`):**
```json
{
  "turn_id": "turn_1788567890",
  "step": 1,
  "tool_name": "wayland_list_windows",
  "status": "running",
  "public_label": "Masaüstü pencereleri kontrol ediliyor..."
}
```

**Rozet Eşlemeleri:**
- `wayland_list_windows` → `"Pencereler kontrol ediliyor..."`
- `wayland_focus_or_launch` → `"{app_name} açılıyor / odaklanılıyor..."`
- `wayland_trigger_shortcut` → `"Masaüstü kısayolu tetikleniyor ({action})..."`
- `capture_screenshot` → `"Ekran görüntüsü alınıyor..."`
- `execute_terminal_command` → `"Komut yürütülüyor..."`
- `route_coding_agent` → `"{agent} kodlama ajanına delege ediliyor..."`

---

### Adım 5: Servis Warmup ve Fast-Path-Last Yaşam Döngüsü

**Warmup Sıralaması (Single-Slot `-np 1` Kuralı):**
1. `llama-server` başlatılır ve `/health` beklenir.
2. **Aşama 1 (Önce):** Reasoning Path Warmup (`enable_thinking=True`).
3. **Aşama 2 (EN SON - FAST PATH LAST):** Fast Path Warmup (`enable_thinking=False`).
   - *Amaç:* Slot 0'da resident kalan KV önbelleğinin günlük hızlı turn'lerin kullandığı Fast-Path formatında kalmasıdır.
4. Gateway yalnızca bu iki aşama bittiğinde `ready = true` döner.
5. Warmup sırasında gelen kullanıcı istekleri düşürülmez; sınırlı bir kuyrukta (`bounded queue`) bekletilip warmup bittiğinde anında işletilir.
6. Gateway restart edildiğinde eğer model restart edilmediyse ve kanonik prefix hash'i değişmediyse mevcut önbellek korunur; soğuk warmup tekrarlanmaz.

**Hedeflenen Metrikler:**
- Kanonik önek: `~4731 token` sabit.
- Fast Path Önbellek Yeniden Kullanımı: `~%99+`.
- Düşük riskli kısa komutlarda TTFT: `<= 1–2s`.

---

### Adım 6: Deterministik / Heuristik Otomatik Sohbet Başlığı Oluşturma

> [!CAUTION]
> Production llama-server tek slottur (`-np 1`). Başlık üretimi için modele ek bir LLM isteği atılması, resident KV önbelleğini evict ederek turn yanıt sürelerini 30 saniyeye fırlatır. Bu nedenle başlıklandırma sıfır-LLM kuralıyla çalışacaktır.

**Akış:**
1. Oturumdaki ilk kullanıcı mesajı (`user_text`) alındığında deterministik temizleyici çalışır.
2. Doldurma sözcükleri (filler: "lütfen", "bana", "açıp bakar mısın", "eder misin", "selam", "merhaba") kırpılır.
3. 3–6 kelimelik anlamlı başlık çıkarılır:
   - *"Zapzapı açıp kardemir grubuna mesaj yazar mısın"* → **"ZapZap Kardemir Mesajı"**
   - *"selam neo nasılsın"* → **"Neo ile Sohbet"**
   - *"antigravity ile su projeyi refactor et"* → **"Antigravity Proje Refactor"**
4. Veritabanı güncellenir: `UPDATE sessions SET title = ? WHERE session_id = ?`.
5. WebSocket üzerinden `type: "session_updated"` olayı fırlatılır.
6. Mobil UI'daki başlık anında güncellenir (llama-server slotu asla meşgul edilmez).

---

### Adım 7: Kriptografik Onay Askıya Alma ve Sürdürme (Approval Resume)

Otonom döngü yüksek etkili bir işleme rastladığında:
1. `hermes_supervisor` döngüyü durdurur ve bir `asyncio.Future` oluşturur.
2. Mobil uygulamaya `approval_required` olayı gönderilir; UI'da `ApprovalCard` açılır.
3. Kullanıcı telefonundan **Onayla** veya **Reddet** butonuna basar.
4. **Onaylanırsa:** Gelecek çözülür (`future.set_result('approve')`), döngü yeni kullanıcı mesajı gerektirmeden aracın çalıştırılmasıyla kaldığı yerden sürer.
5. **Reddedilirse:** Gelecek `'reject'` döner, model reddedildi bilgisini alarak temiz bir bilgilendirme mesajı üretir ve döngüyü kapatır.

---

### Adım 8: Mobil Klavye ve Girdi Alanı Responsiveness İyileştirmesi

1. **Yatay Modern ChatInput (`ChatInput.tsx`):**
   - Dikey stacking yerine tek yatay eksen:
     `[ 🎙️ Ses ] [ Esnek Çok Satırlı TextInput (1-4 satır uzayan) ] [ ⬆️ Gönder Butonu ]`
   - Metin uzadıkça butonlar kaybolmaz veya aşağı itilmez.
2. **Dinamik Safe-Area:**
   - `useSafeAreaInsets()` entegrasyonu: Klavye açıkken `bottom padding = 0`, kapalıyken navigasyon çubuğu kadar boşluk.
3. **FlatList Optimizasyonu:**
   - `keyboardShouldPersistTaps="handled"`
   - `keyboardDismissMode="interactive"`
4. **Ayarlar Modalı (`ConversationListScreen.tsx`):**
   - `KeyboardAvoidingView` + `ScrollView` ile sarılarak klavye açıldığında tüm input ve butonların erişilebilir olması sağlanacak.

---

## 3. Regresyon ve Kabul Testleri (Acceptance Test Suite)

Uygulama sonrasında aşağıdaki 10 test eksiksiz doğrulanacaktır:

- [ ] **Test A - Duplicate Bubble Test:** Tek bir kullanıcı mesajı gönderildiğinde streaming deltaleri ve final mesajın tek bir asistan balonunda birleştiği, asla ikinci bir kopya balon oluşmadığı doğrulanacak.
- [ ] **Test B - Hidden Tool Call Test:** "ZapZapı aç" dendiğinde sohbet balonunda hiçbir ham JSON, `<tool_call>` etiketi veya argüman sızmadığı doğrulanacak.
- [ ] **Test C - Multi-Step Autonomous Test:** Modelin önce pencere listeleme, ardından uygulama açma adımlarını araya kullanıcı girmeden tek turn içinde otonom tamamladığı doğrulanacak.
- [ ] **Test D - Approval Resume Test:** Yüksek riskli bir aksiyonda onay kartının çıktığı, onay verildiğinde aynı turn'ün kaldığı yerden devam edip işlemi tamamladığı doğrulanacak.
- [ ] **Test E - Reject Test:** Onay kartı reddedildiğinde işlemin çalıştırılmadığı ve modelin nazikçe durduğu doğrulanacak.
- [ ] **Test F - Loop Guard Test:** Aynı aracın aynı parametrelerle döngüye girmesi durumunda guard mekanizmasının 2. tekrarda döngüyü güvenle kestiği doğrulanacak.
- [ ] **Test G - Warmup/Cache Regression:** Servis baştan başlatıldığında Fast Path'in en son ısıtıldığı, 3 ardışık istekte prefix hash'inin değişmediği ve önbellek yeniden kullanımının `%99+` olduğu doğrulanacak.
- [ ] **Test H - New Session Cache Test:** Yeni bir oturum açıldığında resident KV önbelleğin bozulmadığı, ilk turn yanıtının 30 saniyelik slot tahliyesine uğramadığı doğrulanacak.
- [ ] **Test I - Auto-Title Cache Safety:** Heuristik başlık üretiminin llama-server'a hiçbir ek istek atmadığı ve önbelleği sıfırlamadığı teyit edilecek.
- [ ] **Test J - Physical Mobile E2E:** Kullanıcının fiziksel telefonu üzerinden Tailscale/LAN ile bağlanıp klavye yazımı, dönen çark ve çok adımlı aksiyonları bizzat teyit etmesi sağlanacak.

---

## 4. Non-Negotiable Architectural Invariants (Değiştirilemez Kurallar)

1. **Production Slot Koruması:** Tek slotlu production llama-server'a başlık üretimi, telemetri açıklaması, özet veya kolaylık amacıyla turn dışı auxiliary LLM isteği atılamaz. Fast-Path resident önbelleği korunmak zorundadır.
2. **Sabit Önek İzolasyonu:** `session_id`, `timestamp`, `title`, UUID veya anlık pencere/ekran durumu gibi uçucu veriler kanonik stable prefix'e kesinlikle dahil edilemez.
3. **Fast Path Last Kuralı:** Startup warmup sıralamasında Fast Path her zaman EN SON çalıştırılır; Slot 0 Fast Path durumunda bırakılır.
4. **Native Tool Calls Önceliği:** Tool ayrıştırmada her zaman native OpenAI `tool_calls` önceliklidir; metin içi JSON sadece güvenlik doğrulamalarından geçen zorunlu bir fallback'tir.
5. **İç Detayların Gizliliği:** Çiğ tool JSON'ları, parametreler, ham sonuçlar ve model iç muhakemesi mobil son kullanıcı sohbetinde varsayılan olarak gösterilemez.
6. **Otonom Adım İlerlemesi:** Bir turn içerisinde model hedefi tamamlayana veya onaya ihtiyaç duyana kadar birden fazla tool adımını kendiliğinden yürütür; kullanıcı gereksiz yere araya sokulmaz.
7. **Onay Güvenliği:** Otonom döngü hiçbir koşulda onay politikasını (kriptografik gating) bypass edemez.
8. **Doğrulama Zorunluluğu:** Durum değiştiren nihai aksiyonlar (pencere açma, sistem değişikliği) yürütme sonucu doğrulanmadan kullanıcıya "başarılı" ilan edilemez.
9. **Bağlam ve Sıkıştırma Standartları:** 60K token precompaction eşiği ve en son 5 kullanıcı/asistan çiftinin harfiyen korunması kuralı değiştirilemez.
10. **128K Bağlam Tavanı:** Gemma-4 E4B 131,072 token bağlam tavanı korunacaktır.
11. **Güvenlik ve Ağ:** Tailscale, yerel LAN ve Bearer token ilk-frame handshake kimlik doğrulama mekanizması bozulamaz.
12. **Otomatik Kurulum Yasağı:** Üretilen APK'lar hiçbir fiziksel cihaza agent tarafından otomatik olarak kurulamaz veya başlatılamaz; kullanıcı fiziksel testi bizzat gerçekleştirir.
