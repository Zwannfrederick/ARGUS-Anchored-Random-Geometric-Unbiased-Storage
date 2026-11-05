(() => {
  "use strict";

  const el = (id) => document.getElementById(id);

  function switchDockTab(tabName, openPanel = false) {
    const valid = ["notes", "quiz", "flashcards", "research"];
    const target = valid.includes(tabName) ? tabName : "notes";
    document.querySelectorAll(".dock-tab").forEach((btn) => {
      btn.classList.toggle("active", btn.dataset.dock === target);
    });
    const isNotes = target === "notes";
    const isEditMode = el("btn-canvas-edit") && el("btn-canvas-edit").classList.contains("active");
    el("canvas-notes-actions").hidden = !isNotes;
    el("canvas-preview").hidden = !(isNotes && !isEditMode);
    el("canvas-editor-wrap").hidden = !(isNotes && isEditMode);
    el("dock-quiz-pane").hidden = target !== "quiz";
    el("dock-flashcards-pane").hidden = target !== "flashcards";
    el("dock-research-pane").hidden = target !== "research";
    if (openPanel) {
      el("canvas-panel").hidden = false;
    }
  }

  function setCanvasContent(title, markdown, openPanel = false) {
    const { renderMarkdown } = window.ArgusMarkdown;
    el("canvas-title-input").value = title || "Çalışma Taslağı";
    el("canvas-textarea").value = markdown || "";
    el("canvas-stats").textContent = `${(markdown || "").length} karakter`;
    el("canvas-preview").innerHTML = markdown
      ? renderMarkdown(markdown)
      : `<p class="empty-hint">Henüz içerik yok.</p>`;
    if (openPanel) {
      switchDockTab("notes", true);
    }
  }

  function renderCitationsInto(container, items) {
    container.innerHTML = "";
    (items || []).forEach((it) => {
      const a = document.createElement("a");
      a.className = "citation-chip";
      a.href = it.url;
      a.target = "_blank";
      a.rel = "noopener noreferrer";
      a.textContent = `🔗 ${it.domain || "kaynak"} — ${(it.title || "").slice(0, 42)}`;
      container.appendChild(a);
    });
  }

  function renderPermissionCard(container, payload, fetchJson, scrollToBottom) {
    const { escapeHtml, renderDiffHtml } = window.ArgusMarkdown;
    const card = document.createElement("div");
    card.className = "perm-card";
    const opMap = {
      write: "DOSYA YAZMA ONAYI",
      edit: "DOSYA DÜZENLEME ONAYI",
      read: "KLASÖR / DOSYA OKUMA ONAYI",
      exec: "TERMİNAL / MASAÜSTÜ İŞLEM ONAYI",
      exec_critical: "KRİTİK / YIKICI KOMUT ONAYI (HERMES)",
      desktop: "MASAÜSTÜ EYLEM ONAYI (PENCERE AÇAR)",
    };
    const allowDirBtn =
      payload.operation === "exec_critical" || payload.operation === "desktop"
        ? ""
        : `<button type="button" class="ghost-btn" data-dec="allow_dir">✓ Bu Klasöre Oturum Boyu İzin Ver</button>`;
    card.innerHTML =
      `<div class="perm-header"><span>⚠️ ${escapeHtml(opMap[payload.operation] || "İZİN GEREKLİ")}</span>` +
      `<span class="perm-status">Onay bekleniyor…</span></div>` +
      `<div class="perm-path">${escapeHtml(payload.summary)}</div>` +
      (payload.diff_preview ? `<div class="diff-block">${renderDiffHtml(payload.diff_preview)}</div>` : "") +
      `<div class="perm-actions">` +
      `<button type="button" class="primary-btn" data-dec="allow_once">✓ Onayla (1 Kez)</button>` +
      allowDirBtn +
      `<button type="button" class="danger-btn" data-dec="deny">✕ Reddet</button>` +
      `</div>`;

    card.querySelectorAll("button[data-dec]").forEach((btn) => {
      btn.addEventListener("click", async () => {
        const decision = btn.dataset.dec;
        card.querySelectorAll("button").forEach((b) => { b.disabled = true; });
        try {
          await fetchJson(`/api/permissions/${payload.req_id}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ decision }),
          });
          const labels = {
            allow_once: "✓ 1 kez onaylandı",
            allow_dir: "✓ Klasör oturum boyu onaylandı",
            deny: "✕ Reddedildi",
          };
          card.querySelector(".perm-status").textContent = labels[decision] || decision;
          card.querySelector(".perm-actions").remove();
        } catch (err) {
          card.querySelector(".perm-status").textContent = `Hata: ${err.message}`;
        }
      });
    });

    container.appendChild(card);
    scrollToBottom();
  }

  function renderCompactionBanner(sess) {
    const box = el("compaction-banner-wrap");
    if (!box) return;
    const summary = (sess && sess.compacted_summary) || "";
    const untilId = (sess && sess.compacted_until_msg_id) || 0;
    if (!summary || !untilId) {
      box.hidden = true;
      box.innerHTML = "";
      return;
    }
    const { renderMarkdown } = window.ArgusMarkdown;
    box.hidden = false;
    box.innerHTML =
      `<details class="compaction-banner">` +
      `<summary>📦 <strong>Sıkıştırılmış Bağlam Özeti</strong> ` +
      `<span class="tabular">(Mesaj #1–#${untilId} özetlendi · Tüm geçmiş mesajlar silinmeden korunuyor, sorgulanabilir)</span></summary>` +
      `<div class="compaction-body md-prose">${renderMarkdown(summary)}</div>` +
      `</details>`;
  }

  function wireCanvasEvents(getActiveSession, fetchJson) {
    const { renderMarkdown } = window.ArgusMarkdown;
    el("btn-toggle-canvas").addEventListener("click", () => {
      el("canvas-panel").hidden = !el("canvas-panel").hidden;
    });
    el("btn-close-canvas").addEventListener("click", () => {
      el("canvas-panel").hidden = true;
    });
    document.querySelectorAll(".dock-tab").forEach((btn) => {
      btn.addEventListener("click", () => switchDockTab(btn.dataset.dock, false));
    });
    el("btn-canvas-view").addEventListener("click", () => {
      el("btn-canvas-view").classList.add("active");
      el("btn-canvas-edit").classList.remove("active");
      el("canvas-preview").innerHTML = renderMarkdown(el("canvas-textarea").value);
      el("canvas-preview").hidden = false;
      el("canvas-editor-wrap").hidden = true;
    });
    el("btn-canvas-edit").addEventListener("click", () => {
      el("btn-canvas-edit").classList.add("active");
      el("btn-canvas-view").classList.remove("active");
      el("canvas-preview").hidden = true;
      el("canvas-editor-wrap").hidden = false;
    });
    el("canvas-textarea").addEventListener("input", () => {
      el("canvas-stats").textContent = `${el("canvas-textarea").value.length} karakter`;
    });
    el("btn-canvas-save").addEventListener("click", async () => {
      const title = el("canvas-title-input").value.trim() || "Çalışma Taslağı";
      const md = el("canvas-textarea").value;
      el("canvas-preview").innerHTML = renderMarkdown(md);
      const sess = getActiveSession();
      if (sess) {
        await fetchJson(`/api/sessions/${sess.id}`, {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ canvas_title: title, canvas_md: md }),
        });
      }
    });
    el("btn-canvas-copy").addEventListener("click", () => {
      navigator.clipboard.writeText(el("canvas-textarea").value);
    });
    el("btn-canvas-download").addEventListener("click", () => {
      const rawTitle = (el("canvas-title-input").value || "taslak").replace(/[^a-zA-Z0-9çğıöşüÇĞİÖŞÜ_-]+/g, "-");
      const blob = new Blob([el("canvas-textarea").value], { type: "text/markdown;charset=utf-8" });
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = rawTitle.endsWith(".md") ? rawTitle : `${rawTitle}.md`;
      a.click();
      URL.revokeObjectURL(a.href);
    });
  }

  const TOOL_LABELS = {
    web_search: "🌐 Paralel Web Araması",
    read_webpage: "🌍 Web Sayfası Okuma (.md)",
    search_chat_history: "🗂️ Sohbet Geçmişi Arşivi Sorgusu",
    read_uploaded_doc: "📄 Ekli Belge Bölümü Okuma",
    list_directory: "📂 Klasör Listeleme",
    search_local_files: "🔎 Yerel Dosya Arama",
    read_local_file: "📖 Yerel Dosya Okuma (.md)",
    write_local_file: "✍️ Yerel Dosya Yazma",
    edit_local_file: "🛠️ Yerel Dosya Düzenleme",
    update_canvas: "📝 Ders Notu / Canvas Güncelleme",
    verify_with_python: "⚗️ Python / SymPy Bağımsız Doğrulama",
    audit_citations: "🛡️ Kaynak & URL Doğrulama Denetimi",
    update_research_state: "🔬 Kalıcı research_state.md Kaydı",
    read_research_state: "📖 research_state.md Okuma",
    publish_quiz: "🧪 İnteraktif Test (Quiz) Yayınlama",
    publish_flashcards: "🃏 İnteraktif Flashcards Oluşturma",
    pc_list_windows: "🖥️ Masaüstü Pencerelerini Listeleme",
    pc_focus_or_launch: "🚀 Uygulama Odaklama / Başlatma (Arka Plan + OCR)",
    pc_shortcut: "⌨️ Masaüstü Kısayolu Çalıştırma",
    pc_capture_screen: "📸 Ekran Görüntüsü & OCR Okuma",
    pc_click: "🖱️ Ekranda Öğeye Tıklama",
    pc_type_text: "⌨️ Odaklı Uygulamaya Metin Yazma",
    pc_press_key: "⏎ Tuş / Kısayol Gönderme",
    pc_terminal_command: "💻 Terminal Komutu Yürütme",
    hermes_skill: "🧩 Hermes Yetenek Köprüsü",
  };

  function buildReasoningEl(roundNo, text, openByDefault = false) {
    const box = document.createElement("details");
    box.className = "reasoning-box";
    if (openByDefault) box.open = true;
    const estTok = Math.max(1, Math.round((text || "").length / 4));
    const stepPrefix = roundNo && roundNo > 1 ? `Adım ${roundNo} · ` : "";
    box.innerHTML = `<summary></summary><div class="trace-body"></div>`;
    box.querySelector("summary").textContent = `🧠 ${stepPrefix}Düşünce Süreci (~${estTok} tok · Sınırsız)`;
    box.querySelector(".trace-body").textContent = text || "";
    return box;
  }

  function buildToolPillEl(te, isRunning = false) {
    const { escapeHtml } = window.ArgusMarkdown;
    const pill = document.createElement("details");
    pill.className = "trace-pill" + (isRunning ? " is-running" : " is-done");
    const label = TOOL_LABELS[te.name] || te.name || "Araç";
    const stepTag = te.round && te.round > 1 ? `<span class="step-chip tabular">Adım ${te.round}</span> ` : "";
    const statusIcon = isRunning ? "⏳" : "✓";
    const suffix = isRunning ? " çalışıyor…" : "";
    pill.innerHTML =
      `<summary>${statusIcon} ${stepTag}<span class="pill-title">${escapeHtml(label)}${suffix}</span></summary>` +
      `<div class="trace-body">${escapeHtml(te.preview || te.arguments || "")}</div>`;
    return pill;
  }

  function buildScreenshotEl(payload) {
    const { escapeHtml } = window.ArgusMarkdown;
    const box = document.createElement("details");
    box.className = "pc-shot-card";
    box.open = true;
    box.innerHTML =
      `<summary>🖥️ Masaüstü Görüntüsü: ${escapeHtml(payload.window || payload.scope || "Ekran")}</summary>` +
      `<a href="${escapeHtml(payload.url)}" target="_blank" rel="noopener noreferrer">` +
      `<img src="${escapeHtml(payload.url)}" alt="Hermes Masaüstü Ekran Görüntüsü" loading="lazy" /></a>`;
    return box;
  }

  function createAssistantDom(initial = {}, onSaveToCanvas = null) {
    const { renderMarkdown } = window.ArgusMarkdown;
    const row = document.createElement("div");
    row.className = "msg-row assistant";

    const timelineEl = document.createElement("div");
    timelineEl.className = "assistant-timeline";

    const permContainer = document.createElement("div");
    permContainer.className = "perm-stack";

    const activityBar = document.createElement("div");
    activityBar.className = "live-activity-bar";
    activityBar.hidden = true;
    activityBar.innerHTML =
      `<span class="activity-dots" aria-hidden="true"><span></span><span></span><span></span></span>` +
      `<span class="activity-label">Hazırlanıyor…</span>` +
      `<span class="activity-timer tabular">0.0 sn</span>`;

    const groundingRow = document.createElement("div");
    const citationsRow = document.createElement("div");
    citationsRow.className = "citations-row";
    if (initial.citations && initial.citations.length) {
      renderCitationsInto(citationsRow, initial.citations);
    }

    const actionsRow = document.createElement("div");
    actionsRow.className = "msg-actions";

    const copyBtn = document.createElement("button");
    copyBtn.type = "button";
    copyBtn.className = "mini-action-btn";
    copyBtn.textContent = "⧉ Kopyala";

    const toCanvasBtn = document.createElement("button");
    toCanvasBtn.type = "button";
    toCanvasBtn.className = "mini-action-btn";
    toCanvasBtn.textContent = "📝 Notlara Aktar";

    let rawMarkdown = initial.content || "";
    const roundReasoningText = {};
    const roundReasoningEls = {};
    const roundContentText = {};
    const roundCardEls = {};
    const toolElsById = {};
    let phaseTimerId = null;
    let phaseStartedAt = 0;
    let stoppedMarked = false;

    function syncRawMarkdown() {
      const keys = Object.keys(roundContentText)
        .map((k) => Number(k))
        .sort((a, b) => a - b);
      const joined = keys
        .map((k) => (roundContentText[k] || "").trim())
        .filter(Boolean)
        .join("\n\n");
      if (joined) rawMarkdown = joined;
      actionsRow.hidden = !rawMarkdown.trim();
    }

    copyBtn.addEventListener("click", () => {
      navigator.clipboard.writeText(rawMarkdown);
      copyBtn.textContent = "✓ Kopyalandı";
      setTimeout(() => { copyBtn.textContent = "⧉ Kopyala"; }, 1400);
    });
    toCanvasBtn.addEventListener("click", () => {
      const title = el("active-session-title").textContent || "Çalışma Notu";
      setCanvasContent(title, rawMarkdown, true);
      if (typeof onSaveToCanvas === "function") onSaveToCanvas(title, rawMarkdown);
    });
    actionsRow.append(copyBtn, toCanvasBtn);

    // Hydrate saved message strictly in chronological order when `timeline` is present
    const savedTimeline = Array.isArray(initial.timeline) ? initial.timeline : [];
    if (savedTimeline.length > 0) {
      savedTimeline.forEach((item) => {
        if (!item || !item.type) return;
        if (item.type === "reasoning" && item.text) {
          timelineEl.appendChild(buildReasoningEl(item.round || 1, item.text, false));
        } else if (item.type === "text" && item.text) {
          const card = document.createElement("div");
          card.className = "assistant-card md-prose";
          card.innerHTML = renderMarkdown(item.text);
          timelineEl.appendChild(card);
        } else if (item.type === "tool") {
          timelineEl.appendChild(buildToolPillEl(item, false));
        } else if (item.type === "screenshot" && item.url) {
          timelineEl.appendChild(buildScreenshotEl(item));
        } else if (item.type === "stopped") {
          const sp = document.createElement("div");
          sp.className = "stopped-pill";
          sp.textContent = item.text || "⏹ İstek kullanıcı tarafından durduruldu.";
          timelineEl.appendChild(sp);
        }
      });
    } else {
      // Legacy fallback for older database messages without timeline_json
      if (initial.reasoning) {
        timelineEl.appendChild(buildReasoningEl(1, initial.reasoning, false));
      }
      if (initial.tool_events && initial.tool_events.length) {
        initial.tool_events.forEach((te) => {
          timelineEl.appendChild(buildToolPillEl(te, false));
        });
      }
      if (initial.content) {
        const card = document.createElement("div");
        card.className = "assistant-card md-prose";
        card.innerHTML = renderMarkdown(initial.content);
        timelineEl.appendChild(card);
      }
    }
    actionsRow.hidden = !rawMarkdown.trim();

    function ensureTimerRunning() {
      if (phaseTimerId) return;
      phaseStartedAt = performance.now();
      const timerSpan = activityBar.querySelector(".activity-timer");
      phaseTimerId = setInterval(() => {
        const sec = ((performance.now() - phaseStartedAt) / 1000).toFixed(1);
        timerSpan.textContent = `${sec} sn`;
      }, 100);
    }

    function setPhase(label, phaseKind = "working") {
      activityBar.hidden = false;
      activityBar.dataset.phase = phaseKind;
      activityBar.querySelector(".activity-label").textContent = label || "Çalışıyor…";
      ensureTimerRunning();
    }

    function clearPhase() {
      if (phaseTimerId) {
        clearInterval(phaseTimerId);
        phaseTimerId = null;
      }
      activityBar.hidden = true;
    }

    function appendReasoningDelta(roundNo, deltaText) {
      const rKey = roundNo || 1;
      roundReasoningText[rKey] = (roundReasoningText[rKey] || "") + (deltaText || "");
      let box = roundReasoningEls[rKey];
      if (!box) {
        box = buildReasoningEl(rKey, roundReasoningText[rKey], true);
        roundReasoningEls[rKey] = box;
        timelineEl.appendChild(box);
      } else {
        const estTok = Math.max(1, Math.round(roundReasoningText[rKey].length / 4));
        const stepPrefix = rKey > 1 ? `Adım ${rKey} · ` : "";
        box.querySelector("summary").textContent = `🧠 ${stepPrefix}Düşünce Süreci (~${estTok} tok · Sınırsız)`;
        const tb = box.querySelector(".trace-body");
        tb.textContent = roundReasoningText[rKey];
        tb.scrollTop = tb.scrollHeight;
      }
      const totalTok = Math.max(1, Math.round(roundReasoningText[rKey].length / 4));
      setPhase(`🧠 Düşünüyor… (~${totalTok} tok)`, "thinking");
    }

    function appendContentDelta(roundNo, deltaText) {
      const rKey = roundNo || 1;
      roundContentText[rKey] = (roundContentText[rKey] || "") + (deltaText || "");
      // Collapse current round's reasoning box once visible answer starts streaming
      if (roundReasoningEls[rKey] && roundReasoningEls[rKey].open) {
        roundReasoningEls[rKey].open = false;
      }
      const cleanVisible = roundContentText[rKey]
        .replace(/<think>[\s\S]*?<\/think>/g, "")
        .replace(/<think>[\s\S]*$/, "")
        .replace(/<tool_call>[\s\S]*?<\/tool_call>/g, "")
        .replace(/<tool_call>[\s\S]*$/, "")
        .trimStart();
      if (!cleanVisible) return;

      let card = roundCardEls[rKey];
      if (!card) {
        card = document.createElement("div");
        card.className = "assistant-card md-prose";
        roundCardEls[rKey] = card;
        timelineEl.appendChild(card);
      }
      card.innerHTML = renderMarkdown(cleanVisible);
      syncRawMarkdown();
      setPhase("✍️ Yanıt yazılıyor…", "writing");
    }

    function replaceRoundContent(roundNo, fullRoundText) {
      const rKey = roundNo || 1;
      roundContentText[rKey] = fullRoundText || "";
      const card = roundCardEls[rKey];
      if (!fullRoundText || !fullRoundText.trim()) {
        if (card) {
          card.remove();
          delete roundCardEls[rKey];
        }
      } else {
        const target = card || (() => {
          const c = document.createElement("div");
          c.className = "assistant-card md-prose";
          roundCardEls[rKey] = c;
          timelineEl.appendChild(c);
          return c;
        })();
        target.innerHTML = renderMarkdown(fullRoundText);
      }
      syncRawMarkdown();
    }

    function appendToolStart(payload) {
      if (roundReasoningEls[payload.round || 1]) {
        roundReasoningEls[payload.round || 1].open = false;
      }
      const pill = buildToolPillEl(payload, true);
      timelineEl.appendChild(pill);
      if (payload.id) toolElsById[payload.id] = pill;
      const lbl = TOOL_LABELS[payload.name] || payload.name || "Araç";
      setPhase(`🔧 ${lbl} çalışıyor…`, "tool");
    }

    function updateToolEnd(payload) {
      const pill = payload.id ? toolElsById[payload.id] : null;
      if (!pill) return;
      pill.classList.remove("is-running");
      pill.classList.add("is-done");
      const lbl = TOOL_LABELS[payload.name] || payload.name || "Araç";
      const pillTitle = pill.querySelector(".pill-title");
      if (pillTitle) pillTitle.textContent = lbl;
      const summary = pill.querySelector("summary");
      if (summary && summary.firstChild) {
        summary.firstChild.textContent = "✓ ";
      }
      pill.querySelector(".trace-body").textContent = payload.preview || payload.arguments || "";
    }

    function appendScreenshot(payload) {
      if (!payload || !payload.url) return;
      timelineEl.appendChild(buildScreenshotEl(payload));
    }

    function appendError(errMsg) {
      clearPhase();
      const errDiv = document.createElement("div");
      errDiv.className = "assistant-card md-prose error-card";
      errDiv.innerHTML = renderMarkdown(`> **Hata:** ${errMsg}`);
      timelineEl.appendChild(errDiv);
    }

    function markStopped() {
      clearPhase();
      if (stoppedMarked) return;
      stoppedMarked = true;
      Object.values(toolElsById).forEach((pill) => {
        if (pill.classList.contains("is-running")) {
          pill.classList.remove("is-running");
          const pt = pill.querySelector(".pill-title");
          if (pt) pt.textContent = pt.textContent.replace(" çalışıyor…", " (durduruldu)");
        }
      });
      const sp = document.createElement("div");
      sp.className = "stopped-pill";
      sp.textContent = "⏹ İstek kullanıcı tarafından durduruldu.";
      timelineEl.appendChild(sp);
      syncRawMarkdown();
    }

    row.append(timelineEl, permContainer, activityBar, groundingRow, citationsRow, actionsRow);
    return {
      row,
      timelineEl,
      traceStack: timelineEl,
      permContainer,
      activityBar,
      groundingRow,
      citationsRow,
      actionsRow,
      setPhase,
      clearPhase,
      appendReasoningDelta,
      appendContentDelta,
      replaceRoundContent,
      appendToolStart,
      updateToolEnd,
      appendScreenshot,
      appendError,
      markStopped,
      setRaw(md) {
        rawMarkdown = md || "";
        replaceRoundContent(1, rawMarkdown);
      },
    };
  }

  function appendUserMessageDom(content, attachments = [], msgId = null) {
    const { escapeHtml } = window.ArgusMarkdown;
    const row = document.createElement("div");
    row.className = "msg-row user";
    let attachHtml = "";
    if (attachments && attachments.length) {
      attachHtml =
        `<div class="pending-attachments">` +
        attachments.map((a) => `<span class="meta-pill">📄 ${escapeHtml(a.filename)} (.md)</span>`).join("") +
        `</div>`;
    }
    const idTitle = msgId ? ` title="Mesaj #${msgId}"` : "";
    row.innerHTML = `${attachHtml}<div class="user-bubble"${idTitle}>${escapeHtml(content)}</div>`;
    el("messages-list").appendChild(row);
  }

  window.ArgusPanels = {
    TOOL_LABELS,
    switchDockTab,
    setCanvasContent,
    renderCitationsInto,
    renderPermissionCard,
    renderCompactionBanner,
    wireCanvasEvents,
    createAssistantDom,
    appendUserMessageDom,
  };
})();
