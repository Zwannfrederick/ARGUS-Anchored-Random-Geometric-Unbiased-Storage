(() => {
"use strict";

function escapeHtml(str) {
  return String(str || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function renderMarkdown(md) {
  if (!md) return "";
  const codeBlocks = [];
  let text = String(md).replace(/\r\n/g, "\n");

  text = text.replace(/```([a-zA-Z0-9_+-]*)\n([\s\S]*?)```/g, (_, lang, code) => {
    const idx = codeBlocks.length;
    const safeLang = escapeHtml(lang || "text");
    const safeCode = escapeHtml(code.replace(/\n$/, ""));
    codeBlocks.push(
      `<div class="code-wrap"><div class="code-bar"><span>${safeLang}</span>` +
      `<button type="button" class="code-copy-btn" data-copy="${encodeURIComponent(code.replace(/\n$/, ""))}">Kopyala</button></div>` +
      `<pre><code>${safeCode}</code></pre></div>`
    );
    return `\u0000CODEBLOCK_${idx}\u0000`;
  });

  const lines = text.split("\n");
  const out = [];
  let i = 0;

  const formatInline = (s) => {
    let h = escapeHtml(s);
    h = h.replace(/`([^`]+)`/g, "<code>$1</code>");
    h = h.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    h = h.replace(/\*([^*\n]+)\*/g, "<em>$1</em>");
    h = h.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
    return h;
  };

  while (i < lines.length) {
    const line = lines[i];
    if (line.includes("\u0000CODEBLOCK_")) { out.push(line); i++; continue; }
    if (!line.trim()) { i++; continue; }

    const hMatch = line.match(/^(#{1,4})\s+(.*)$/);
    if (hMatch) {
      const lvl = hMatch[1].length;
      out.push(`<h${lvl}>${formatInline(hMatch[2])}</h${lvl}>`);
      i++;
      continue;
    }
    if (line.startsWith("|") && i + 1 < lines.length && /^\|\s*[-:]+/.test(lines[i + 1])) {
      const parseRow = (r) => r.replace(/^\||\|$/g, "").split("|").map((c) => formatInline(c.trim()));
      const headers = parseRow(line);
      i += 2;
      const rows = [];
      while (i < lines.length && lines[i].trim().startsWith("|")) { rows.push(parseRow(lines[i])); i++; }
      out.push(
        "<table><thead><tr>" + headers.map((h) => `<th>${h}</th>`).join("") + "</tr></thead><tbody>" +
        rows.map((r) => "<tr>" + r.map((c) => `<td>${c}</td>`).join("") + "</tr>").join("") + "</tbody></table>"
      );
      continue;
    }
    if (/^>\s?/.test(line)) {
      const qLines = [];
      while (i < lines.length && /^>\s?/.test(lines[i])) { qLines.push(formatInline(lines[i].replace(/^>\s?/, ""))); i++; }
      out.push(`<blockquote>${qLines.join("<br/>")}</blockquote>`);
      continue;
    }
    if (/^\s*[-*]\s+/.test(line)) {
      const items = [];
      while (i < lines.length && /^\s*[-*]\s+/.test(lines[i])) { items.push(`<li>${formatInline(lines[i].replace(/^\s*[-*]\s+/, ""))}</li>`); i++; }
      out.push(`<ul>${items.join("")}</ul>`);
      continue;
    }
    if (/^\s*\d+\.\s+/.test(line)) {
      const items = [];
      while (i < lines.length && /^\s*\d+\.\s+/.test(lines[i])) { items.push(`<li>${formatInline(lines[i].replace(/^\s*\d+\.\s+/, ""))}</li>`); i++; }
      out.push(`<ol>${items.join("")}</ol>`);
      continue;
    }
    if (/^---+$/.test(line.trim())) { out.push("<hr/>"); i++; continue; }

    const pLines = [];
    while (
      i < lines.length && lines[i].trim() &&
      !lines[i].includes("\u0000CODEBLOCK_") &&
      !/^(#{1,4}\s|\||>\s?|[-*]\s+|\d+\.\s+|---+$)/.test(lines[i].trim())
    ) {
      pLines.push(formatInline(lines[i]));
      i++;
    }
    out.push(`<p>${pLines.join("<br/>")}</p>`);
  }

  return out.join("\n").replace(/\u0000CODEBLOCK_(\d+)\u0000/g, (_, idx) => codeBlocks[Number(idx)] || "");
}

function renderDiffHtml(diffText) {
  if (!diffText) return "";
  return diffText
    .split("\n")
    .map((ln) => {
      const safe = escapeHtml(ln);
      if (ln.startsWith("+") && !ln.startsWith("+++")) return `<span class="diff-add">${safe}</span>`;
      if (ln.startsWith("-") && !ln.startsWith("---")) return `<span class="diff-del">${safe}</span>`;
      return safe;
    })
    .join("\n");
}

window.ArgusMarkdown = { escapeHtml, renderMarkdown, renderDiffHtml };
})();

