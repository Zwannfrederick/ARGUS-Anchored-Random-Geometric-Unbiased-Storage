(() => {
  "use strict";

  const state = {
    sessions: [],
    activeSession: null,
    mode: "assistant",
    webMode: "auto",
    pcMode: "auto",
    autoCompact: true,
    enableThinking: false,
    workspaceDir: "",
    maxContextTokens: 131072,
    pendingUploads: [],
    abortController: null,
    activeAssistantDom: null,
    streamingSessionId: null,
  };

  const el = (id) => document.getElementById(id);
  const { escapeHtml } = window.ArgusMarkdown;
  const {
    switchDockTab,
    setCanvasContent,
    renderCitationsInto,
    renderPermissionCard,
    renderCompactionBanner,
    wireCanvasEvents,
    createAssistantDom,
    appendUserMessageDom,
  } = window.ArgusPanels;
  const {
    MODE_CONFIG,
    renderResearchLoopBar,
    renderResearchStatePane,
    renderQuizPane,
    renderFlashcardsPane,
    renderGroundingBadge,
  } = window.ArgusStudyResearch;

  async function fetchJson(url, opts = {}) {
    const res = await fetch(url, opts);
    if (!res.ok) {
      const errText = await res.text();
      throw new Error(errText || `HTTP ${res.status}`);
    }
    return res.json();
  }

  function updateContextBadge(activeEstTokens) {
    const maxK = Math.round((state.maxContextTokens || 131072) / 1024);
    const compactedId = state.activeSession && state.activeSession.compacted_until_msg_id;
    if (activeEstTokens && activeEstTokens > 0) {
      const actStr = activeEstTokens >= 1000 ? `~${(activeEstTokens / 1000).toFixed(1)}K` : `~${activeEstTokens}`;
      el("ctx-badge").textContent = compactedId
        ? `${actStr} / ${maxK}K (🗜️ #${compactedId})`
        : `${actStr} / ${maxK}K Bağlam`;
    } else {
      el("ctx-badge").textContent = compactedId ? `${maxK}K (🗜️ #${compactedId})` : `${maxK}K Bağlam`;
    }
  }

  async function refreshStatus() {
    try {
      const st = await fetchJson("/api/status");
      el("status-dot").classList.toggle("online", Boolean(st.backend_ok));
      el("model-subtitle").textContent = st.backend_ok ? st.model_name : "Motor çevrimdışı";
      if (st.context_length) {
        state.maxContextTokens = st.context_length;
        updateContextBadge();
      }
      if (!state.workspaceDir && st.default_workspace) {
        el("workspace-input").placeholder = st.default_workspace;
      }
    } catch (_) {
      el("status-dot").classList.remove("online");
      el("model-subtitle").textContent = "Bağlantı yok";
    }
  }

  function applyThinkingUI(enabled) {
    state.enableThinking = Boolean(enabled);
    const b = el("btn-think-toggle");
    if (!b) return;
    b.classList.toggle("active", state.enableThinking);
    b.setAttribute("aria-pressed", String(state.enableThinking));
    b.textContent = state.enableThinking ? "🧠 Düşünme: Açık (Sınırsız)" : "🧠 Düşünme: Kapalı";
  }

  function applyAutoCompactUI(enabled) {
    state.autoCompact = Boolean(enabled);
    const b = el("btn-autocompact-toggle");
    if (!b) return;
    b.classList.toggle("active", state.autoCompact);
    b.setAttribute("aria-pressed", String(state.autoCompact));
    b.textContent = `🔄 Oto Sıkıştırma: ${state.autoCompact ? "Açık" : "Kapalı"}`;
  }

  function applyWebModeUI(nextMode) {
    state.webMode = nextMode;
    const labels = { auto: "Otomatik", always: "Her Zaman", off: "Kapalı" };
    const b = el("btn-web-toggle");
    if (!b) return;
    const active = state.webMode !== "off";
    b.classList.toggle("active", active);
    b.setAttribute("aria-pressed", String(active));
    b.textContent = `🌐 Web: ${labels[state.webMode] || "Otomatik"}`;
  }

  function applyPcModeUI(nextMode) {
    state.pcMode = nextMode;
    const labels = {
      auto: "👻 PC: Arka Plan (Oto)",
      background: "👻 PC: Arka Plan (Hep)",
      foreground: "🖥️ PC: Ön Plan",
      off: "🚫 PC: Kapalı",
    };
    const b = el("btn-pc-toggle");
    if (b) {
      const active = state.pcMode !== "off";
      b.classList.toggle("active", active);
      b.setAttribute("aria-pressed", String(active));
      b.textContent = labels[state.pcMode] || labels.auto;
    }
    refreshResearchBar();
  }

  function cyclePcMode() {
    const order = ["auto", "background", "foreground", "off"];
    const next = order[(order.indexOf(state.pcMode) + 1) % order.length];
    applyPcModeUI(next);
  }

  function resizeComposer() {
    const inp = el("chat-input");
    if (!inp) return;
    inp.style.height = "auto";
    inp.style.height = `${Math.min(200, Math.max(54, inp.scrollHeight))}px`;
  }

  function prefillComposer(promptText) {
    const inp = el("chat-input");
    inp.value = promptText;
    resizeComposer();
    inp.focus();
    inp.setSelectionRange(inp.value.length, inp.value.length);
  }

  function triggerFollowupPrompt(promptText) {
    if (state.abortController) return;
    el("chat-input").value = promptText;
    resizeComposer();
    sendMessage();
  }

  function refreshResearchBar() {
    renderResearchLoopBar(state.mode, state.activeSession, {
      pcMode: state.pcMode,
      onCyclePcMode: cyclePcMode,
      onSendPrompt: triggerFollowupPrompt,
      onPrefillPrompt: prefillComposer,
      onOpenDock: (tabName) => {
        if (tabName === "research") renderResearchStatePane(state.activeSession);
        switchDockTab(tabName || "notes", true);
      },
    });
  }

  function applyModeUI(mode, fromUserSwitch = false) {
    state.mode = MODE_CONFIG[mode] ? mode : "assistant";
    const cfg = MODE_CONFIG[state.mode];
    document.querySelectorAll(".mode-tab").forEach((b) => {
      b.classList.toggle("active", b.dataset.mode === state.mode);
    });
    el("active-mode-badge").textContent = cfg.label;
    el("empty-heading").textContent = cfg.heading;
    el("empty-desc").textContent = cfg.desc;

    if (fromUserSwitch) {
      if (state.mode === "research") {
        applyThinkingUI(true);
        if (state.webMode === "auto") applyWebModeUI("always");
      } else if (state.mode === "draft") {
        switchDockTab("notes", true);
      } else if (state.mode === "study" && state.activeSession) {
        if (state.activeSession.study_quiz) switchDockTab("quiz", true);
        else if (state.activeSession.study_flashcards) switchDockTab("flashcards", true);
      }
    }
    refreshResearchBar();

    const grid = el("starter-grid");
    grid.innerHTML = "";
    cfg.starters.forEach((s) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "starter-card";
      btn.innerHTML = `<strong>${escapeHtml(s.title)}</strong><span>${escapeHtml(s.sub)}</span>`;
      btn.addEventListener("click", () => prefillComposer(s.prompt));
      grid.appendChild(btn);
    });
  }

  function updateWorkspaceUI(dir) {
    state.workspaceDir = (dir || "").trim();
    el("workspace-input").value = state.workspaceDir;
    if (state.workspaceDir) {
      const parts = state.workspaceDir.split("/").filter(Boolean);
      el("workspace-label").textContent = parts.slice(-2).join("/") || state.workspaceDir;
    } else {
      el("workspace-label").textContent = "Klasör Seçilmedi (İzin Sorulur)";
    }
  }

  function renderSidebarSessions() {
    const filter = el("session-filter").value.trim().toLowerCase();
    const nav = el("session-list");
    nav.innerHTML = "";
    state.sessions
      .filter((s) => !filter || s.title.toLowerCase().includes(filter))
      .forEach((s) => {
        const row = document.createElement("div");
        row.className = "session-item" + (state.activeSession && state.activeSession.id === s.id ? " active" : "");
        const tags = (s.research_phase ? " 🔬" : "") + (s.compacted_until_msg_id ? " 🗜️" : "");
        row.innerHTML =
          `<span class="session-item-title">${escapeHtml(s.title)}${tags}</span>` +
          `<button type="button" class="session-del-btn" title="Oturumu sil">✕</button>`;
        row.addEventListener("click", (ev) => {
          if (ev.target.closest(".session-del-btn")) return;
          loadSession(s.id);
        });
        row.querySelector(".session-del-btn").addEventListener("click", async (ev) => {
          ev.stopPropagation();
          await fetchJson(`/api/sessions/${s.id}`, { method: "DELETE" });
          await loadSessionsList();
          if (state.activeSession && state.activeSession.id === s.id) {
            if (state.sessions.length > 0) await loadSession(state.sessions[0].id);
            else await createNewSession();
          }
        });
        nav.appendChild(row);
      });
  }

  function renderSessionDocs(docs) {
    const wrap = el("session-docs");
    if (!docs || docs.length === 0) {
      wrap.innerHTML = `<div class="empty-hint">Henüz PDF/Word eklenmedi.</div>`;
      return;
    }
    wrap.innerHTML = "";
    docs.forEach((d) => {
      const item = document.createElement("div");
      item.className = "doc-item";
      item.innerHTML = `<span>📄 ${escapeHtml(d.filename)}</span><span class="tabular">~${d.est_tokens} tok</span>`;
      item.title = "Dönüştürülmüş Markdown içeriğini sağ panelde aç";
      item.addEventListener("click", () => openDocInCanvas(d.id));
      wrap.appendChild(item);
    });
  }

  async function openDocInCanvas(docId) {
    const doc = await fetchJson(`/api/docs/${docId}`);
    setCanvasContent(`${doc.filename} (.md)`, doc.markdown, true);
  }

  function renderPendingAttachments() {
    const box = el("pending-attachments");
    if (state.pendingUploads.length === 0) {
      box.hidden = true;
      box.innerHTML = "";
      return;
    }
    box.hidden = false;
    box.innerHTML = "";
    state.pendingUploads.forEach((doc, idx) => {
      const chip = document.createElement("div");
      chip.className = "attach-chip";
      chip.innerHTML =
        `<span>📄 <strong>${escapeHtml(doc.filename)}</strong> → .md (${doc.page_count} blm · ~${doc.est_tokens} tok)</span>` +
        `<button type="button" class="inspect-btn" title="Markdown önizle">👁</button>` +
        `<button type="button" class="remove-btn" title="Kaldır">✕</button>`;
      chip.querySelector(".inspect-btn").addEventListener("click", () => openDocInCanvas(doc.id));
      chip.querySelector(".remove-btn").addEventListener("click", () => {
        state.pendingUploads.splice(idx, 1);
        renderPendingAttachments();
      });
      box.appendChild(chip);
    });
  }

  async function persistCanvas(title, rawMarkdown) {
    if (!state.activeSession) return;
    await fetchJson(`/api/sessions/${state.activeSession.id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ canvas_title: title, canvas_md: rawMarkdown }),
    });
  }

  function scrollToBottom() {
    const sc = el("messages-scroll");
    sc.scrollTop = sc.scrollHeight;
  }

  async function loadSessionsList() {
    const data = await fetchJson("/api/sessions");
    state.sessions = data.sessions || [];
    renderSidebarSessions();
  }

  async function loadSession(sessionId) {
    const sess = await fetchJson(`/api/sessions/${sessionId}`);
    state.activeSession = sess;
    el("active-session-title").textContent = sess.title;
    applyModeUI(sess.mode || "assistant", false);
    applyAutoCompactUI(sess.auto_compact !== 0);
    updateWorkspaceUI(sess.workspace_dir || "");
    renderSessionDocs(sess.documents || []);
    renderCompactionBanner(sess);
    updateContextBadge();

    setCanvasContent(sess.canvas_title || "Çalışma Notu", sess.canvas_md || "", false);
    renderQuizPane(sess.study_quiz, triggerFollowupPrompt);
    renderFlashcardsPane(sess.study_flashcards);
    renderResearchStatePane(sess);
    refreshResearchBar();

    if (sess.study_quiz && sess.mode === "study") switchDockTab("quiz", true);
    else if (sess.study_flashcards && sess.mode === "study") switchDockTab("flashcards", true);
    else if (sess.research_state_md && sess.mode === "research") switchDockTab("research", true);
    else if (sess.canvas_md) switchDockTab("notes", true);
    else el("canvas-panel").hidden = true;

    const list = el("messages-list");
    list.innerHTML = "";
    const msgs = sess.messages || [];
    el("empty-state").hidden = msgs.length > 0;
    msgs.forEach((m) => {
      if (m.role === "user") {
        appendUserMessageDom(m.content, m.attachments, m.id);
      } else if (m.role === "assistant") {
        const dom = createAssistantDom(m, persistCanvas);
        list.appendChild(dom.row);
        if (m.timings && m.timings.decode_tps) {
          const sb = el("speed-badge");
          sb.hidden = false;
          sb.textContent = `⚡ ${m.timings.decode_tps} tok/s`;
        }
      }
    });
    renderSidebarSessions();
    scrollToBottom();
  }

  async function createNewSession() {
    const sess = await fetchJson("/api/sessions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title: "Yeni Sohbet", mode: state.mode, workspace_dir: state.workspaceDir }),
    });
    await loadSessionsList();
    await loadSession(sess.id);
    el("chat-input").focus();
  }

  async function triggerManualCompaction() {
    if (!state.activeSession || state.abortController) return;
    const btn = el("btn-compact-now");
    const orig = btn.innerHTML;
    btn.disabled = true;
    btn.innerHTML = `<span>⏳ Sıkıştırılıyor…</span>`;
    try {
      const res = await fetchJson(`/api/sessions/${state.activeSession.id}/compact`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ keep_recent: 4 }),
      });
      if (res.session) {
        state.activeSession = res.session;
        renderCompactionBanner(res.session);
        updateContextBadge(res.active_est_tokens);
        await loadSessionsList();
      }
      if (!res.compacted && res.reason) alert(res.reason);
    } catch (err) {
      alert(`Bağlam sıkıştırma hatası: ${err.message}`);
    } finally {
      btn.disabled = false;
      btn.innerHTML = orig;
    }
  }

  async function handleFilesUpload(fileList) {
    if (!fileList || fileList.length === 0) return;
    if (!state.activeSession) await createNewSession();

    const attachBtn = el("btn-attach");
    const origLabel = attachBtn.textContent;
    attachBtn.textContent = "⏳ .md'ye çevriliyor…";
    attachBtn.disabled = true;

    try {
      for (const file of Array.from(fileList)) {
        const fd = new FormData();
        fd.append("session_id", state.activeSession.id);
        fd.append("file", file);
        const saved = await fetchJson("/api/upload", { method: "POST", body: fd });
        state.pendingUploads.push(saved);
      }
      renderPendingAttachments();
      const refreshed = await fetchJson(`/api/sessions/${state.activeSession.id}`);
      renderSessionDocs(refreshed.documents || []);
    } catch (err) {
      alert(`Belge dönüştürme hatası: ${err.message}`);
    } finally {
      attachBtn.textContent = origLabel;
      attachBtn.disabled = false;
      el("file-input").value = "";
    }
  }

  async function stopActiveStream() {
    const sid = state.streamingSessionId || (state.activeSession && state.activeSession.id);
    const stopBtn = el("btn-stop");
    if (stopBtn) {
      stopBtn.disabled = true;
      stopBtn.textContent = "⏳ Durduruluyor…";
    }
    if (state.activeAssistantDom) {
      state.activeAssistantDom.markStopped();
    }
    if (sid) {
      fetch(`/api/sessions/${sid}/stop`, { method: "POST" }).catch(() => {});
    } else {
      fetch("/api/chat/stop", { method: "POST" }).catch(() => {});
    }
    if (state.abortController) {
      state.abortController.abort();
    }
  }

  async function sendMessage() {
    const input = el("chat-input");
    const text = input.value.trim();
    if (!text && state.pendingUploads.length === 0) return;
    if (state.abortController) return;

    if (!state.activeSession) await createNewSession();
    const promptText = text || "Eklediğim belgeyi Markdown yapısına göre özetle ve ana noktalarını çıkar.";
    const attachmentsToSend = [...state.pendingUploads];
    state.pendingUploads = [];
    renderPendingAttachments();
    input.value = "";
    resizeComposer();

    el("empty-state").hidden = true;
    appendUserMessageDom(promptText, attachmentsToSend);

    const assistantDom = createAssistantDom({}, persistCanvas);
    el("messages-list").appendChild(assistantDom.row);
    assistantDom.setPhase("⚡ Bağlam işleniyor…", "prefill");
    scrollToBottom();

    const ac = new AbortController();
    state.abortController = ac;
    state.activeAssistantDom = assistantDom;
    state.streamingSessionId = state.activeSession.id;

    const stopBtn = el("btn-stop");
    stopBtn.disabled = false;
    stopBtn.textContent = "■ Durdur";
    el("btn-send").hidden = true;
    stopBtn.hidden = false;

    let currentRound = 1;

    try {
      const resp = await fetch("/api/chat/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: ac.signal,
        body: JSON.stringify({
          session_id: state.activeSession.id,
          message: promptText,
          attachment_ids: attachmentsToSend.map((d) => d.id),
          mode: state.mode,
          enable_web: state.webMode !== "off",
          web_mode: state.webMode,
          pc_mode: state.pcMode,
          enable_thinking: state.enableThinking,
          workspace_dir: state.workspaceDir,
        }),
      });

      if (!resp.ok || !resp.body) throw new Error(await resp.text());

      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const frames = buffer.split("\n\n");
        buffer = frames.pop() || "";

        for (const frame of frames) {
          let evName = "message";
          let dataRaw = "";
          for (const ln of frame.split("\n")) {
            if (ln.startsWith("event: ")) evName = ln.slice(7).trim();
            else if (ln.startsWith("data: ")) dataRaw += ln.slice(6);
          }
          if (!dataRaw) continue;
          const payload = JSON.parse(dataRaw);

          if (evName === "session_meta") {
            el("active-session-title").textContent = payload.title;
            if (state.activeSession && (!payload.session_id || state.activeSession.id === payload.session_id)) {
              state.activeSession.title = payload.title;
            }
            loadSessionsList();
          } else if (evName === "round_start") {
            currentRound = payload.round || currentRound + 1;
          } else if (evName === "phase") {
            assistantDom.setPhase(payload.label || "Çalışıyor…", payload.phase || "working");
            scrollToBottom();
          } else if (evName === "compaction_notice") {
            if (state.activeSession) {
              state.activeSession.compacted_until_msg_id = payload.compacted_until_msg_id;
              state.activeSession.compacted_summary = payload.compacted_summary;
              renderCompactionBanner(state.activeSession);
              updateContextBadge();
            }
          } else if (evName === "reasoning_delta") {
            assistantDom.appendReasoningDelta(payload.round || currentRound, payload.text || "");
            scrollToBottom();
          } else if (evName === "content_delta") {
            assistantDom.appendContentDelta(payload.round || currentRound, payload.text || "");
            scrollToBottom();
          } else if (evName === "round_content_replace") {
            assistantDom.replaceRoundContent(payload.round || currentRound, payload.text || "");
            scrollToBottom();
          } else if (evName === "tool_start") {
            assistantDom.appendToolStart(payload);
            scrollToBottom();
          } else if (evName === "tool_end") {
            assistantDom.updateToolEnd(payload);
          } else if (evName === "pc_screenshot") {
            assistantDom.appendScreenshot(payload);
            scrollToBottom();
          } else if (evName === "permission_request") {
            assistantDom.setPhase("⚠️ Kullanıcı onayı bekleniyor…", "permission");
            renderPermissionCard(assistantDom.permContainer, payload, fetchJson, scrollToBottom);
          } else if (evName === "citations") {
            renderCitationsInto(assistantDom.citationsRow, payload.items || []);
          } else if (evName === "canvas_update") {
            if (state.activeSession) {
              state.activeSession.canvas_title = payload.title;
              state.activeSession.canvas_md = payload.markdown;
            }
            setCanvasContent(payload.title, payload.markdown, true);
            refreshResearchBar();
          } else if (evName === "study_quiz") {
            if (state.activeSession) state.activeSession.study_quiz = payload;
            el("canvas-title-input").value = payload.title || "İnteraktif Test";
            renderQuizPane(payload, triggerFollowupPrompt);
            refreshResearchBar();
            switchDockTab("quiz", true);
          } else if (evName === "study_flashcards") {
            if (state.activeSession) state.activeSession.study_flashcards = payload;
            el("canvas-title-input").value = payload.title || "Bilgi Kartları";
            renderFlashcardsPane(payload);
            refreshResearchBar();
            switchDockTab("flashcards", true);
          } else if (evName === "research_state") {
            if (state.activeSession) {
              state.activeSession.research_state_md = payload.markdown;
              state.activeSession.research_phase = payload.phase;
              state.activeSession.research_state_path = payload.path;
            }
            renderResearchStatePane(state.activeSession);
            refreshResearchBar();
            switchDockTab("research", true);
          } else if (evName === "grounding_audit") {
            renderGroundingBadge(assistantDom.groundingRow, payload);
          } else if (evName === "timings" && payload.decode_tps) {
            const sb = el("speed-badge");
            sb.hidden = false;
            sb.textContent = `⚡ ${payload.decode_tps} tok/s`;
          } else if (evName === "stopped") {
            assistantDom.markStopped();
          } else if (evName === "done") {
            assistantDom.clearPhase();
            if (payload.active_est_tokens) updateContextBadge(payload.active_est_tokens);
          } else if (evName === "error") {
            assistantDom.appendError(payload.message);
          }
        }
      }
    } catch (err) {
      if (err.name === "AbortError") {
        assistantDom.markStopped();
      } else {
        assistantDom.appendError(`Bağlantı hatası: ${err.message}`);
      }
    } finally {
      assistantDom.clearPhase();
      state.abortController = null;
      state.activeAssistantDom = null;
      state.streamingSessionId = null;
      stopBtn.disabled = false;
      stopBtn.textContent = "■ Durdur";
      stopBtn.hidden = true;
      el("btn-send").hidden = false;
      loadSessionsList();
    }
  }

  function safeGetTheme() {
    try { return localStorage.getItem("argus_theme") || "dark"; } catch (_) { return "dark"; }
  }
  function safeSetTheme(val) {
    try { localStorage.setItem("argus_theme", val); } catch (_) {}
  }

  let studioBooted = false;
  async function bootStudio() {
    if (studioBooted) return;
    studioBooted = true;
    document.documentElement.setAttribute("data-theme", safeGetTheme());

    el("btn-theme").addEventListener("click", () => {
      const next = document.documentElement.getAttribute("data-theme") === "dark" ? "paper" : "dark";
      document.documentElement.setAttribute("data-theme", next);
      safeSetTheme(next);
    });

    el("btn-toggle-sidebar").addEventListener("click", () => {
      el("sidebar").classList.toggle("collapsed");
    });

    el("btn-new-chat").addEventListener("click", createNewSession);
    el("active-session-title").title = " Yeniden adlandırmak için çift tıkla (boş bırakırsan yapay zeka adlandırır)";
    el("active-session-title").addEventListener("dblclick", async () => {
      if (!state.activeSession || state.abortController) return;
      const next = window.prompt("Sohbet başlığı (yapay zekanın otomatik isimlendirmesi için boş bırakın):", state.activeSession.title || "");
      if (next === null) return;
      const updated = await fetchJson(`/api/sessions/${state.activeSession.id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title: next }),
      });
      state.activeSession.title = updated.title;
      el("active-session-title").textContent = updated.title;
      loadSessionsList();
    });
    el("btn-compact-now").addEventListener("click", triggerManualCompaction);
    el("session-filter").addEventListener("input", renderSidebarSessions);

    document.querySelectorAll(".mode-tab").forEach((btn) => {
      btn.addEventListener("click", async () => {
        applyModeUI(btn.dataset.mode, true);
        if (state.activeSession) {
          await fetchJson(`/api/sessions/${state.activeSession.id}`, {
            method: "PATCH",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ mode: state.mode }),
          });
        }
      });
    });

    el("btn-workspace").addEventListener("click", () => {
      el("workspace-bar").hidden = !el("workspace-bar").hidden;
      if (!el("workspace-bar").hidden) el("workspace-input").focus();
    });
    el("btn-save-workspace").addEventListener("click", async () => {
      updateWorkspaceUI(el("workspace-input").value);
      el("workspace-bar").hidden = true;
      if (state.activeSession) {
        await fetchJson(`/api/sessions/${state.activeSession.id}`, {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ workspace_dir: state.workspaceDir }),
        });
      }
    });
    el("btn-clear-workspace").addEventListener("click", async () => {
      updateWorkspaceUI("");
      el("workspace-bar").hidden = true;
      if (state.activeSession) {
        await fetchJson(`/api/sessions/${state.activeSession.id}`, {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ workspace_dir: "" }),
        });
      }
    });

    el("btn-attach").addEventListener("click", () => el("file-input").click());
    el("file-input").addEventListener("change", (ev) => handleFilesUpload(ev.target.files));

    el("btn-web-toggle").addEventListener("click", () => {
      const order = ["auto", "always", "off"];
      applyWebModeUI(order[(order.indexOf(state.webMode) + 1) % order.length]);
    });

    el("btn-pc-toggle").addEventListener("click", cyclePcMode);

    el("btn-autocompact-toggle").addEventListener("click", async () => {
      applyAutoCompactUI(!state.autoCompact);
      if (state.activeSession) {
        state.activeSession = await fetchJson(`/api/sessions/${state.activeSession.id}`, {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ auto_compact: state.autoCompact }),
        });
      }
    });

    el("btn-think-toggle").addEventListener("click", () => {
      applyThinkingUI(!state.enableThinking);
    });

    el("btn-send").addEventListener("click", sendMessage);
    el("btn-stop").addEventListener("click", stopActiveStream);
    el("chat-input").addEventListener("input", resizeComposer);
    el("chat-input").addEventListener("keydown", (ev) => {
      if (ev.key === "Enter" && !ev.shiftKey) {
        ev.preventDefault();
        sendMessage();
      } else if (ev.key === "Escape" && state.abortController) {
        ev.preventDefault();
        stopActiveStream();
      }
    });
    window.addEventListener("keydown", (ev) => {
      if (ev.key === "Escape" && state.abortController) {
        stopActiveStream();
      }
    });

    wireCanvasEvents(() => state.activeSession, fetchJson);

    document.body.addEventListener("click", (ev) => {
      const copyBtn = ev.target.closest(".code-copy-btn");
      if (!copyBtn) return;
      navigator.clipboard.writeText(decodeURIComponent(copyBtn.dataset.copy || ""));
      copyBtn.textContent = "✓";
      setTimeout(() => { copyBtn.textContent = "Kopyala"; }, 1200);
    });

    let dragDepth = 0;
    window.addEventListener("dragenter", (ev) => {
      if (ev.dataTransfer && Array.from(ev.dataTransfer.types).includes("Files")) {
        dragDepth++;
        el("drop-overlay").hidden = false;
      }
    });
    window.addEventListener("dragleave", () => {
      dragDepth = Math.max(0, dragDepth - 1);
      if (dragDepth === 0) el("drop-overlay").hidden = true;
    });
    window.addEventListener("dragover", (ev) => ev.preventDefault());
    window.addEventListener("drop", (ev) => {
      ev.preventDefault();
      dragDepth = 0;
      el("drop-overlay").hidden = true;
      if (ev.dataTransfer && ev.dataTransfer.files.length) {
        handleFilesUpload(ev.dataTransfer.files);
      }
    });

    applyPcModeUI("auto");
    applyModeUI("assistant", false);
    await refreshStatus();
    setInterval(refreshStatus, 8000);
    try {
      await loadSessionsList();
      if (state.sessions.length > 0) {
        await loadSession(state.sessions[0].id);
      } else {
        await createNewSession();
      }
    } catch (err) {
      el("model-subtitle").textContent = `Oturum hatası: ${err.message}`;
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", bootStudio);
  } else {
    bootStudio();
  }
})();
