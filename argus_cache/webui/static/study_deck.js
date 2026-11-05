(() => {
  "use strict";

  const el = (id) => document.getElementById(id);

  const LOOP_STEPS = [
    {
      key: "1_arastir",
      label: "1. Araştır",
      prompt:
        "Mevcut araştırma konusu için 1. Araştır adımını yürüt: paralel web_search(queries=[...]) ve read_webpage ile birincil kaynakları topla ve bulguları sentezle.",
    },
    {
      key: "2_hipotez",
      label: "2. Hipotez Kur",
      prompt:
        "Mevcut bulgulara dayanarak 2. Hipotez Kur adımını yürüt: test edilebilir, yanlışlanabilir bilimsel hipotezler ve deney planı üretip update_research_state ile kaydet.",
    },
    {
      key: "3_dogrula",
      label: "3. Doğrula (SymPy)",
      prompt:
        "Aktif hipotez için 3. Doğrula adımını yürüt: verify_with_python aracıyla SymPy/NumPy kullanarak bağımsız sembolik/sayısal hesaplamaları çalıştır ve sonuçları göster.",
    },
    {
      key: "4_elestir",
      label: "4. Eleştir",
      prompt:
        "4. Eleştir adımını yürüt: sonuçları sınır koşulları, hata payları ve alternatif açıklamalar açısından eleştir; audit_citations ile tüm kaynak URL'lerini denetle.",
    },
    {
      key: "5_kaydet",
      label: "5. Kaydet",
      prompt:
        "5. Kaydet adımını yürüt: doğrulanan/elenen hipotezleri, sayısal kanıtları ve sıradaki deneyi update_research_state ile kalıcı research_state.md dosyasına yaz.",
    },
  ];

  let currentQuiz = null;
  let quizAnswers = {};
  let currentDeck = null;
  let deckIndex = 0;
  let deckFlipped = false;
  let deckMastery = {}; // cardId -> "known" | "review"
  let filterReviewOnly = false;

  function renderResearchLoopBar(mode, sess, optsOrCb) {
    const bar = el("research-loop-bar");
    if (!bar) return;
    const opts = typeof optsOrCb === "function" ? { onOpenDock: () => optsOrCb() } : optsOrCb || {};
    const { escapeHtml } = window.ArgusMarkdown;
    const sendPrompt = opts.onSendPrompt || (() => {});
    const prefillPrompt = opts.onPrefillPrompt || (() => {});
    const openDock = opts.onOpenDock || (() => {});

    bar.hidden = false;
    bar.innerHTML = "";

    if (mode === "research") {
      const hasState = Boolean(sess && sess.research_state_md);
      const rawPhase = sess && sess.research_phase;
      const activePhase = rawPhase || "1_arastir";
      const activeIdx = Math.max(0, LOOP_STEPS.findIndex((s) => s.key === activePhase));
      const nextStep = rawPhase ? LOOP_STEPS[(activeIdx + 1) % LOOP_STEPS.length] : LOOP_STEPS[0];

      const stepsHtml = LOOP_STEPS.map((st, idx) => {
        const cls = st.key === activePhase ? "loop-step active" : idx < activeIdx ? "loop-step done" : "loop-step";
        const arrow = idx < LOOP_STEPS.length - 1 ? `<span class="loop-arrow" aria-hidden="true">→</span>` : "";
        return (
          `<button type="button" class="${cls}" data-step-idx="${idx}" title="Bu araştırma adımını doğrudan çalıştır">` +
          `${escapeHtml(st.label)}</button>${arrow}`
        );
      }).join("");

      bar.innerHTML =
        `<div class="loop-steps">${stepsHtml}</div>` +
        `<div class="workbench-right">` +
        `<button id="btn-run-next-phase" type="button" class="primary-btn mini-btn" title="Sıradaki bilimsel adımı otomatik başlat">` +
        `▶ ${escapeHtml(nextStep.label)}</button>` +
        `<button id="btn-open-research-state" type="button" class="ghost-btn mini-btn">` +
        `🗂️ research_state.md ${hasState ? "(Aktif)" : "(Boş)"}</button>` +
        `</div>`;

      bar.querySelectorAll("button[data-step-idx]").forEach((btn) => {
        btn.addEventListener("click", () => {
          const st = LOOP_STEPS[Number(btn.dataset.stepIdx)];
          if (st) sendPrompt(st.prompt);
        });
      });
      bar.querySelector("#btn-run-next-phase").addEventListener("click", () => sendPrompt(nextStep.prompt));
      bar.querySelector("#btn-open-research-state").addEventListener("click", () => openDock("research"));
      return;
    }

    if (mode === "study") {
      const quizCount = sess && sess.study_quiz && sess.study_quiz.questions ? sess.study_quiz.questions.length : 0;
      const cardCount = sess && sess.study_flashcards && sess.study_flashcards.cards ? sess.study_flashcards.cards.length : 0;
      const hasNotes = Boolean(sess && sess.canvas_md && sess.canvas_md.trim());

      bar.innerHTML =
        `<div class="loop-steps">` +
        `<span class="workbench-kicker">🎓 ETÜT ARAÇLARI:</span>` +
        `<button type="button" class="workbench-chip" data-act="quiz">🧪 5 Soruluk Test Üret</button>` +
        `<button type="button" class="workbench-chip" data-act="cards">🃏 8 Flashcard Üret</button>` +
        `<button type="button" class="workbench-chip" data-act="notes">📝 Ders Notu Çıkar (.md)</button>` +
        `<button type="button" class="workbench-chip" data-act="solve">🧮 Python Doğrulamalı Çöz</button>` +
        `</div>` +
        `<div class="workbench-right">` +
        (quizCount ? `<button type="button" class="ghost-btn mini-btn" data-dock-open="quiz">🧪 Test (${quizCount})</button>` : "") +
        (cardCount ? `<button type="button" class="ghost-btn mini-btn" data-dock-open="flashcards">🃏 Kartlar (${cardCount})</button>` : "") +
        `<button type="button" class="ghost-btn mini-btn" data-dock-open="notes">📝 Notlar${hasNotes ? " ✓" : ""}</button>` +
        `</div>`;

      const prompts = {
        quiz: "Bu oturumdaki konu veya ekli belgeden orta-zor seviyede, ipuçlu ve detaylı çözümlü 5 soruluk interaktif bir sınav hazırla ve publish_quiz aracıyla sağ panelde yayınla.",
        cards: "Bu oturumdaki konu veya ekli belgedeki en önemli kavram ve formüllerden hafıza ipuçlu (mnemonic) 8 kartlık bir Flashcards destesini publish_flashcards aracıyla oluştur.",
        notes: "Bu oturumdaki konu veya ekli belgeyi temel tanımlar, kritik formüller, sık yapılan hatalar ve çözümlü örneklerle yapılandırılmış bir Markdown çalışma kağıdı olarak update_canvas ile sağ panele yaz.",
      };
      bar.querySelectorAll("button[data-act]").forEach((btn) => {
        btn.addEventListener("click", () => {
          const act = btn.dataset.act;
          if (act === "solve") {
            prefillPrompt("Şu ödev/sınav problemini sezgisel mantığıyla adım adım anlat ve sayısal hesabını verify_with_python ile doğrula: ");
          } else if (prompts[act]) {
            sendPrompt(prompts[act]);
          }
        });
      });
      bar.querySelectorAll("button[data-dock-open]").forEach((btn) => {
        btn.addEventListener("click", () => openDock(btn.dataset.dockOpen));
      });
      return;
    }

    if (mode === "draft") {
      const charLen = sess && sess.canvas_md ? sess.canvas_md.length : 0;
      bar.innerHTML =
        `<div class="loop-steps">` +
        `<span class="workbench-kicker">📐 TASLAK ATÖLYESİ:</span>` +
        `<button type="button" class="workbench-chip" data-draft="init">✍️ Yeni Taslak Başlat</button>` +
        `<button type="button" class="workbench-chip" data-draft="critique">🔍 Eleştir &amp; Güçlendir</button>` +
        `<button type="button" class="workbench-chip" data-draft="roadmap">✅ Eylem Planına Çevir</button>` +
        `</div>` +
        `<div class="workbench-right">` +
        `<button type="button" class="ghost-btn mini-btn" id="btn-open-draft-canvas">📝 Canvas (${charLen} kr)</button>` +
        `</div>`;

      bar.querySelectorAll("button[data-draft]").forEach((btn) => {
        btn.addEventListener("click", () => {
          const k = btn.dataset.draft;
          if (k === "init") {
            prefillPrompt("Şu fikri yapılandırılmış bir Markdown taslağı olarak update_canvas ile sağdaki Canvas paneline yaz: ");
          } else if (k === "critique") {
            sendPrompt("Mevcut Canvas taslağındaki zayıf noktaları veya eksikleri belirle ve güçlendirilmiş sürümünü update_canvas ile güncelle.");
          } else if (k === "roadmap") {
            sendPrompt("Bu oturumdaki fikirleri ve mevcut taslağı önceliklendirilmiş adımlar ve kontrol listesi içeren somut bir uygulama planı olarak update_canvas ile güncelle.");
          }
        });
      });
      bar.querySelector("#btn-open-draft-canvas").addEventListener("click", () => openDock("notes"));
      return;
    }

    // Default: "assistant" (Asistan & Hermes PC)
    const pcMode = opts.pcMode || "auto";
    const pcBadgeMap = {
      auto: "👻 Hermes: Arka Plan Sanal Ekran (Oto)",
      background: "👻 Hermes: Arka Plan Sanal Ekran (Hep)",
      foreground: "🖥️ Hermes: Ön Plan (Görünür)",
      off: "🚫 Hermes PC: Kapalı",
    };
    bar.innerHTML =
      `<div class="loop-steps">` +
      `<button type="button" class="workbench-chip active" id="btn-wb-pc-mode" title="Tıkla: Arka Plan / Ön Plan / Kapalı değiştir">` +
      `${escapeHtml(pcBadgeMap[pcMode] || pcBadgeMap.auto)}</button>` +
      `<button type="button" class="workbench-chip" data-pc="wins">🖥️ Açık Pencereler</button>` +
      `<button type="button" class="workbench-chip" data-pc="app">🚀 Arka Planda Uygulama Kontrolü…</button>` +
      `<button type="button" class="workbench-chip" data-pc="ocr">📸 Ekranı Oku (OCR)</button>` +
      `</div>`;

    bar.querySelector("#btn-wb-pc-mode").addEventListener("click", () => {
      if (opts.onCyclePcMode) opts.onCyclePcMode();
    });
    bar.querySelectorAll("button[data-pc]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const action = btn.dataset.pc;
        if (action === "wins") {
          sendPrompt("Bilgisayarımda şu an açık olan masaüstü pencerelerini ve hangi workspace'te olduklarını pc_list_windows ile listele.");
        } else if (action === "ocr") {
          sendPrompt("Hedef veya aktif pencerenin ekran görüntüsünü pc_capture_screen ile alıp ekrandaki öğeleri özetle.");
        } else if (action === "app") {
          prefillPrompt("Arka planda ekranımı bozmadan şu uygulamayı açıp kontrol et: ");
        }
      });
    });
  }

  function renderResearchStatePane(sess) {
    const pane = el("dock-research-pane");
    if (!pane) return;
    const { escapeHtml, renderMarkdown } = window.ArgusMarkdown;
    const md = (sess && sess.research_state_md) || "";
    const path = (sess && sess.research_state_path) || "~/.argus_runtime/studio/research/.../research_state.md";
    if (!md.trim()) {
      pane.innerHTML =
        `<p class="empty-hint">Bu oturumda henüz kalıcı <code>research_state.md</code> kaydı yok. ` +
        `Araştırma modunda hipotez testi başlattığında ajan bulgularını otomatik olarak <code>${escapeHtml(path)}</code> dosyasına yazar.</p>`;
      return;
    }
    pane.innerHTML =
      `<div class="perm-path">📍 Kalıcı Durum Dosyası: ${escapeHtml(path)}</div>` +
      `<div class="md-prose">${renderMarkdown(md)}</div>`;
  }

  function renderQuizPane(quiz, sendFollowupPrompt) {
    currentQuiz = quiz || null;
    quizAnswers = {};
    drawQuiz(sendFollowupPrompt);
  }

  function drawQuiz(sendFollowupPrompt, onlyMissedIds = null) {
    const pane = el("dock-quiz-pane");
    if (!pane) return;
    const { escapeHtml, renderMarkdown } = window.ArgusMarkdown;

    if (!currentQuiz || !currentQuiz.questions || currentQuiz.questions.length === 0) {
      pane.innerHTML = `<p class="empty-hint">Henüz aktif bir test yok. Ders Çalışma modunda <strong>"Bu konudan 5 soruluk test hazırla"</strong> diyerek interaktif sınav başlatabilirsin.</p>`;
      return;
    }

    const questions = onlyMissedIds
      ? currentQuiz.questions.filter((q) => onlyMissedIds.includes(q.id))
      : currentQuiz.questions;

    let correctCount = 0;
    let answeredCount = 0;
    const missedQuestions = [];

    questions.forEach((q) => {
      if (quizAnswers[q.id] !== undefined) {
        answeredCount++;
        if (quizAnswers[q.id] === q.correct_index) correctCount++;
        else missedQuestions.push(q);
      }
    });

    const pct = answeredCount > 0 ? Math.round((correctCount / answeredCount) * 100) : 0;
    pane.innerHTML = "";

    const board = document.createElement("div");
    board.className = "quiz-scoreboard";
    board.innerHTML =
      `<div>` +
      `<div class="quiz-score-main">${escapeHtml(currentQuiz.title || "Konu Tarama Testi")}</div>` +
      `<div class="empty-hint tabular">Yanıtlanan: ${answeredCount}/${questions.length} · Doğru: ${correctCount} · Başarı: %${pct}</div>` +
      `</div>` +
      `<div class="perm-actions">` +
      `<button type="button" class="ghost-btn mini-btn" id="btn-quiz-reset">↻ Sıfırla</button>` +
      (missedQuestions.length > 0
        ? `<button type="button" class="ghost-btn mini-btn" id="btn-quiz-retry-missed">🎯 Yanlışları Çöz (${missedQuestions.length})</button>` +
          `<button type="button" class="primary-btn mini-btn" id="btn-quiz-ask-ai">🧠 Eksiklerimi Anlattır</button>`
        : "") +
      `</div>`;
    pane.appendChild(board);

    board.querySelector("#btn-quiz-reset").addEventListener("click", () => {
      quizAnswers = {};
      drawQuiz(sendFollowupPrompt, null);
    });
    const retryBtn = board.querySelector("#btn-quiz-retry-missed");
    if (retryBtn) {
      retryBtn.addEventListener("click", () => {
        const ids = missedQuestions.map((q) => q.id);
        ids.forEach((id) => { delete quizAnswers[id]; });
        drawQuiz(sendFollowupPrompt, ids);
      });
    }
    const askBtn = board.querySelector("#btn-quiz-ask-ai");
    if (askBtn && sendFollowupPrompt) {
      askBtn.addEventListener("click", () => {
        const summary = missedQuestions.map((q, i) => `${i + 1}) ${q.question}`).join("\n");
        sendFollowupPrompt(
          `Testte şu sorularda hata yaptım, temel mantığını Feynman tekniğiyle ve kısa örnekle açıklar mısın?\n${summary}`
        );
      });
    }

    const letters = ["A", "B", "C", "D", "E", "F"];
    questions.forEach((q, idx) => {
      const card = document.createElement("div");
      card.className = "quiz-card";
      const chosen = quizAnswers[q.id];
      const isAnswered = chosen !== undefined;

      const top = document.createElement("div");
      top.className = "quiz-card-top";
      top.innerHTML =
        `<span class="meta-pill tabular">Soru ${idx + 1} · ${escapeHtml(q.difficulty || "Orta")}</span>` +
        (q.hint && !isAnswered ? `<button type="button" class="ghost-btn mini-btn hint-btn">💡 İpucu</button>` : "");
      card.appendChild(top);

      const qTitle = document.createElement("div");
      qTitle.className = "quiz-q-title md-prose";
      qTitle.innerHTML = renderMarkdown(q.question);
      card.appendChild(qTitle);

      if (q.hint) {
        const hintBox = document.createElement("div");
        hintBox.className = "quiz-hint-box";
        hintBox.hidden = true;
        hintBox.textContent = `💡 İpucu: ${q.hint}`;
        card.appendChild(hintBox);
        const hb = top.querySelector(".hint-btn");
        if (hb) hb.addEventListener("click", () => { hintBox.hidden = !hintBox.hidden; });
      }

      const optsWrap = document.createElement("div");
      optsWrap.className = "quiz-options";
      (q.options || []).forEach((optText, oIdx) => {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "quiz-opt-btn";
        if (isAnswered) {
          btn.disabled = true;
          if (oIdx === q.correct_index) btn.classList.add("correct");
          else if (oIdx === chosen) btn.classList.add("wrong");
        }
        btn.innerHTML = `<strong>${letters[oIdx] || oIdx + 1})</strong> <span>${escapeHtml(optText)}</span>`;
        btn.addEventListener("click", () => {
          quizAnswers[q.id] = oIdx;
          drawQuiz(sendFollowupPrompt, onlyMissedIds);
        });
        optsWrap.appendChild(btn);
      });
      card.appendChild(optsWrap);

      if (isAnswered && q.explanation) {
        const exp = document.createElement("div");
        exp.className = "quiz-explain-box md-prose";
        const statusPrefix = chosen === q.correct_index ? "✅ **Doğru!** " : "❌ **Yanlış.** ";
        exp.innerHTML = renderMarkdown(statusPrefix + q.explanation);
        card.appendChild(exp);
      }

      pane.appendChild(card);
    });
  }

  function renderFlashcardsPane(deck) {
    currentDeck = deck ? { title: deck.title, cards: [...(deck.cards || [])] } : null;
    deckIndex = 0;
    deckFlipped = false;
    deckMastery = {};
    filterReviewOnly = false;
    drawFlashcards();
  }

  function activeDeckCards() {
    if (!currentDeck || !currentDeck.cards) return [];
    if (!filterReviewOnly) return currentDeck.cards;
    return currentDeck.cards.filter((c) => deckMastery[c.id] !== "known");
  }

  function drawFlashcards() {
    const pane = el("dock-flashcards-pane");
    if (!pane) return;
    const { escapeHtml, renderMarkdown } = window.ArgusMarkdown;

    if (!currentDeck || !currentDeck.cards || currentDeck.cards.length === 0) {
      pane.innerHTML = `<p class="empty-hint">Henüz bilgi kartı destesi yok. <strong>"Bu konudan 8 flashcard oluştur"</strong> diyerek aktif ezber/kavram kartları hazırlatabilirsin.</p>`;
      return;
    }

    const cards = activeDeckCards();
    const knownCount = currentDeck.cards.filter((c) => deckMastery[c.id] === "known").length;
    const reviewCount = currentDeck.cards.filter((c) => deckMastery[c.id] === "review").length;

    pane.innerHTML = "";
    const bar = document.createElement("div");
    bar.className = "fc-toolbar";
    bar.innerHTML =
      `<div><strong>${escapeHtml(currentDeck.title || "Bilgi Kartları")}</strong>` +
      `<div class="empty-hint tabular">✓ Öğrenildi: ${knownCount}/${currentDeck.cards.length} · ↻ Tekrar: ${reviewCount}</div></div>` +
      `<div class="perm-actions">` +
      `<button type="button" class="ghost-btn mini-btn" id="btn-fc-filter">${filterReviewOnly ? "Tüm Kartlar" : "Sadece Kalanlar"}</button>` +
      `<button type="button" class="ghost-btn mini-btn" id="btn-fc-shuffle">🔀 Karıştır</button>` +
      `</div>`;
    pane.appendChild(bar);

    bar.querySelector("#btn-fc-filter").addEventListener("click", () => {
      filterReviewOnly = !filterReviewOnly;
      deckIndex = 0;
      deckFlipped = false;
      drawFlashcards();
    });
    bar.querySelector("#btn-fc-shuffle").addEventListener("click", () => {
      currentDeck.cards.sort(() => Math.random() - 0.5);
      deckIndex = 0;
      deckFlipped = false;
      drawFlashcards();
    });

    if (cards.length === 0) {
      const doneMsg = document.createElement("div");
      doneMsg.className = "quiz-card";
      doneMsg.innerHTML = `<strong>🎉 Tüm kartları öğrendin!</strong><span class="empty-hint">"Tüm Kartlar" butonuna basarak desteyi baştan gözden geçirebilirsin.</span>`;
      pane.appendChild(doneMsg);
      return;
    }

    if (deckIndex >= cards.length) deckIndex = 0;
    const card = cards[deckIndex];

    const stage = document.createElement("div");
    stage.className = "fc-stage" + (deckFlipped ? " is-back" : "");
    stage.setAttribute("tabindex", "0");
    stage.setAttribute("role", "button");
    stage.setAttribute("aria-label", "Kartı çevirmek için tıkla veya Boşluk tuşuna bas");

    const sideLabel = deckFlipped ? "ARKA YÜZ · CEVAP & AÇIKLAMA" : "ÖN YÜZ · SORU / KAVRAM (Çevirmek için tıkla)";
    const mainBody = deckFlipped ? card.back : card.front;
    stage.innerHTML =
      `<div class="fc-side-kicker"><span>${sideLabel}</span><span class="meta-pill tabular">${deckIndex + 1} / ${cards.length} · ${escapeHtml(card.tag || "Kavram")}</span></div>` +
      `<div class="fc-main-text md-prose">${renderMarkdown(mainBody)}</div>` +
      (deckFlipped && card.mnemonic
        ? `<div class="fc-mnemonic">🧠 <strong>Hafıza İpucu:</strong> ${escapeHtml(card.mnemonic)}</div>`
        : `<div class="empty-hint">Çevir: Tıkla veya <code>Space</code></div>`);

    const flipCard = () => {
      stage.classList.add("flipping");
      setTimeout(() => {
        deckFlipped = !deckFlipped;
        drawFlashcards();
      }, 120);
    };
    stage.addEventListener("click", flipCard);
    stage.addEventListener("keydown", (ev) => {
      if (ev.key === " " || ev.key === "Enter") { ev.preventDefault(); flipCard(); }
      else if (ev.key === "ArrowRight") { deckIndex = (deckIndex + 1) % cards.length; deckFlipped = false; drawFlashcards(); }
      else if (ev.key === "ArrowLeft") { deckIndex = (deckIndex - 1 + cards.length) % cards.length; deckFlipped = false; drawFlashcards(); }
    });
    pane.appendChild(stage);

    const gradeRow = document.createElement("div");
    gradeRow.className = "fc-controls";
    gradeRow.innerHTML =
      `<button type="button" class="ghost-btn" id="btn-fc-review">↻ Tekrar Et</button>` +
      `<button type="button" class="primary-btn" id="btn-fc-known">✓ Öğrendim</button>`;
    pane.appendChild(gradeRow);

    gradeRow.querySelector("#btn-fc-review").addEventListener("click", () => {
      deckMastery[card.id] = "review";
      deckIndex = (deckIndex + 1) % cards.length;
      deckFlipped = false;
      drawFlashcards();
    });
    gradeRow.querySelector("#btn-fc-known").addEventListener("click", () => {
      deckMastery[card.id] = "known";
      deckIndex = (deckIndex + 1) % Math.max(1, activeDeckCards().length);
      deckFlipped = false;
      drawFlashcards();
    });

    const navRow = document.createElement("div");
    navRow.className = "fc-nav-row";
    navRow.innerHTML =
      `<button type="button" class="ghost-btn" id="btn-fc-prev">← Önceki</button>` +
      `<button type="button" class="ghost-btn" id="btn-fc-flip">🔄 Kartı Çevir</button>` +
      `<button type="button" class="ghost-btn" id="btn-fc-next">Sonraki →</button>`;
    pane.appendChild(navRow);

    navRow.querySelector("#btn-fc-prev").addEventListener("click", () => {
      deckIndex = (deckIndex - 1 + cards.length) % cards.length;
      deckFlipped = false;
      drawFlashcards();
    });
    navRow.querySelector("#btn-fc-flip").addEventListener("click", flipCard);
    navRow.querySelector("#btn-fc-next").addEventListener("click", () => {
      deckIndex = (deckIndex + 1) % cards.length;
      deckFlipped = false;
      drawFlashcards();
    });
  }

  function renderGroundingBadge(container, audit) {
    if (!container || !audit) return;
    const { escapeHtml } = window.ArgusMarkdown;
    const unverified = audit.unverified_urls || [];
    const div = document.createElement("div");
    div.className = "grounding-shield" + (unverified.length ? " warn" : "");
    const parts = ["🛡️ Bağımsız Doğrulama:"];
    if (audit.python_verified) parts.push("✓ Python/SymPy Hesaplamalı");
    if (audit.verified_source_count > 0) parts.push(`✓ ${audit.verified_source_count} Kaynak Doğrulandı`);
    if (unverified.length > 0) parts.push(`⚠️ ${unverified.length} Doğrulanmamış URL İşaretlendi`);
    if (parts.length === 1) parts.push("✓ Halüsinasyonlu URL Yok");
    div.textContent = parts.join(" · ");
    container.innerHTML = "";
    container.appendChild(div);
  }

  function appendPcScreenshot(traceStack, payload) {
    if (!traceStack || !payload || !payload.url) return;
    const { escapeHtml } = window.ArgusMarkdown;
    const box = document.createElement("details");
    box.className = "pc-shot-card";
    box.open = true;
    box.innerHTML =
      `<summary>🖥️ Masaüstü Görüntüsü: ${escapeHtml(payload.window || payload.scope || "Ekran")}</summary>` +
      `<a href="${escapeHtml(payload.url)}" target="_blank" rel="noopener noreferrer">` +
      `<img src="${escapeHtml(payload.url)}" alt="Hermes Masaüstü Ekran Görüntüsü" loading="lazy" /></a>`;
    traceStack.appendChild(box);
  }

  const MODE_CONFIG = {
    assistant: {
      label: "Asistan & PC",
      heading: "Asistan & Hermes Arka Plan Masaüstü Kontrolü",
      desc: "Hermes motoru fiziksel ekranını ve odağını bozmadan arka plan sanal ekranında uygulamaları açar, hızlı önbellekli OCR ile okur ve tıklar; paralel web araması ve PDF/Word analizi yapar.",
      starters: [
        { title: "👻 Arka Planda Uygulama Aç & Oku", sub: "Ekranımı bölmeden ZapZap/uygulama kontrol et", prompt: "Arka planda ekranımı bozmadan ZapZap uygulamasını aç ve son notlarımda ne yazdığını kontrol et." },
        { title: "🖥️ Açık Pencereleri & Masaüstünü Tara", sub: "Hyprland workspace ve çalışan uygulamaları listele", prompt: "Şu an bilgisayarımda açık olan masaüstü pencerelerini ve hangi workspace'te olduklarını listele." },
        { title: "🌐 Paralel Web Sentezi", sub: "Birden fazla sorguyu aynı anda tarayıp özetle", prompt: "2026 itibarıyla Linux Wayland masaüstü ekosistemi ve açık ağırlıklı dil modellerindeki son gelişmeleri paralel web aramalarıyla özetle." },
        { title: "📄 Belgeyi Markdown'a Çevir & İncele", sub: "PDF veya Word ekle, bölümler halinde analiz et", prompt: "Eklediğim belgeyi ana başlıkları, kritik verileri ve eylem maddeleriyle temiz bir Markdown özeti haline getir." },
      ],
    },
    research: {
      label: "Bilimsel AR-GE",
      heading: "Doğrulanmış Bilimsel Araştırma Laboratuvarı",
      desc: "5 adımlı kesintisiz döngü: Paralel literatür taraması → Hipotez → Python/SymPy hesap doğrulaması → audit_citations ve sınır testi → Kalıcı research_state.md hafızası.",
      starters: [
        { title: "⚗️ Kimya / Fizik Hipotez & SymPy Kanıtı", sub: "Denklemleri sembolik/sayısal doğrula, duruma kaydet", prompt: "İdeal gaz ile Van der Waals denklemlerinin yüksek basınçtaki (CO2, 310 K, 1..200 atm) sapmasını araştır: hipotez kur, verify_with_python ile sayısal sapmaları hesapla, sınır koşullarını eleştir ve research_state.md dosyasına kaydet." },
        { title: "🔬 5 Adımlı Literatür & Karşıt Kanıt Taraması", sub: "Paralel kaynak topla, audit_citations ile denetle", prompt: "Katı hal bataryalarda sülfür bazlı elektrolitlerin dendrit oluşumuna karşı sınırlarını paralel web aramasıyla araştır; test edilebilir hipotezleri, karşıt bulguları ve doğrulanmış kaynakları research_state.md'ye işleyerek sentezle." },
        { title: "📐 Matematiksel Modelleme & Duyarlılık", sub: "SymPy/NumPy ile türev, integral ve kararlılık analizi", prompt: "Sönümlü harmonik osilatörde kritik sönüm sınırını SymPy ile analitik olarak türet, 3 farklı sönüm oranı için enerji kaybını hesapla ve bulguları research_state.md'ye kaydet." },
        { title: "🗂️ Kaldığım Araştırmaya Devam Et", sub: "research_state.md içindeki sıradaki deneyi yürüt", prompt: "Mevcut research_state.md dosyasını oku; açık kalan soruyu ve sıradaki deneyi çalıştırıp hipotez durumlarını güncelle." },
      ],
    },
    study: {
      label: "Ders Çalışma",
      heading: "İnteraktif Ders, Test & Flashcard Stüdyosu",
      desc: "Ödev ve sınav hazırlığı için sağ panelde anında puanlanan interaktif Test (Quiz), çevrilebilir Flashcards destesi ve düzenli Markdown ders notları üret.",
      starters: [
        { title: "🧪 İnteraktif Test (Quiz) Hazırla", sub: "İpuçlu, çözümlü ve anında puanlanan sınav başlat", prompt: "Lineer Cebir (özdeğerler, özvektörler ve matris köşegenleştirme) konusundan orta-zor seviyede 5 soruluk interaktif bir test hazırla; hesapları doğru kurgula ve sağ panelde sınav olarak yayınla." },
        { title: "🃏 Flashcards (Bilgi Kartları) Oluştur", sub: "Ön/arka yüz çevrilebilir + hafıza ipuçlu kart destesi", prompt: "Termodinamiğin yasaları, entropi, entalpi ve Gibbs serbest enerjisi üzerine hafıza ipuçları (mnemonic) içeren 8 kartlık interaktif bir Flashcards destesi oluştur." },
        { title: "📝 Ders Notu & Formül Kağıdı (.md)", sub: "Ekli PDF/Word veya konudan çalışma notu çıkar", prompt: "Eklediğim ders belgesini (veya belirttiğim konuyu) temel kavramlar, kritik formüller, sık yapılan sınav hataları ve çözümlü örneklerle sağdaki Not paneline kapsamlı Markdown çalışma kağıdı olarak yaz." },
        { title: "🧮 Ödev Problemini Python Doğrulamalı Çöz", sub: "Sezgisel anlatım + hatasız adım adım hesap", prompt: "Bu fizik/matematik problemini ezberletmeden sezgisel mantığıyla adım adım anlat ve sayısal sonucu Python ile doğrulayarak göster." },
      ],
    },
    draft: {
      label: "Fikir Taslağı",
      heading: "Fikir & Taslak Stüdyosu",
      desc: "Ham fikirlerini at; birlikte tartışıp sağdaki canlı Markdown Canvas üzerinde makale, proje mimarisi veya teknik doküman haline getirelim.",
      starters: [
        { title: "📐 Proje / Ürün Taslağı (PRD)", sub: "Ham fikri mimari ve yol haritasına dönüştür", prompt: "Yeni bir yazılım fikrim var: hedefleri, temel özellikleri, teknik mimariyi ve ilk sürüm adımlarını içeren temiz bir Markdown taslağını Canvas'ta başlat." },
        { title: "✍️ Makale / Teknik Yazı Taslağı", sub: "Güçlü giriş, alt başlıklar ve akıcı kurgu", prompt: "Teknik bir blog yazısı için dikkat çekici başlık alternatifleri ve yapılandırılmış bir Markdown taslağını Canvas paneline oluştur." },
        { title: "🧭 Beyin Fırtınası → Eylem Planı", sub: "Dağınık notları önceliklendirilmiş plana çevir", prompt: "Aklımdaki dağınık fikirleri netleştirmeme yardım et; güçlü ve zayıf yönlerini sorgulayıp somut bir Markdown taslağına dönüştür." },
        { title: "💾 Taslağı Diske .md Kaydet", sub: "Onaylı diff önizlemesiyle klasöre yaz", prompt: "Sağdaki Canvas'ta oluşturduğumuz taslağı çalışma klasörüme 'taslak-notlar.md' adıyla kaydet." },
      ],
    },
  };

  window.ArgusStudyResearch = {
    MODE_CONFIG,
    renderResearchLoopBar,
    renderResearchStatePane,
    renderQuizPane,
    renderFlashcardsPane,
    renderGroundingBadge,
    appendPcScreenshot,
  };
})();

