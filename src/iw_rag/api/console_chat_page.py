# ruff: noqa: E501
"""Native, branded IW RAG Operator Console **Chat** view (F-00096, S15).

This module owns the rich Chat console page — a faithful port of the Chainlit
chat surface into the console shell, using the shared
:func:`iw_rag.api.console_shell.render_console_page` assembly contract.

Features
--------
* **Grounding toggle** — labelled switch (assist / strict mode).  Assist mode
  (default ON) sends ``allow_ungrounded=true``; strict mode sends
  ``allow_ungrounded=false`` (fail-closed, documents-only).
* **Conversation log** — scrollable in-session message log; user messages
  right-aligned, assistant left-aligned bubbles.  Textarea ``#q`` + Send button
  ``#send``; Enter sends, Shift+Enter inserts newline.  Send is disabled while a
  request is in-flight.
* **Trust badge** — four states: grounded (teal), fail-closed refusal (red),
  general-knowledge (amber), unverified / answer-ungrounded (amber).
* **Markdown rendering** — small, safe inline renderer (HTML-escape first, then
  limited transform set: fenced code blocks, inline code, bold, italic,
  headings, lists, links, paragraphs).  Never injects raw HTML from the model.
* **Citations list** — each ``[marker] source_uri`` + truncated quote ≤ 160
  chars.  Download link when ``download_url`` present (prefixed with API base).
* **Provenance panel** — collapsed ``<details>`` with groundedness verdict,
  mode, retrieval route, guard trail, PII-redaction note, sources, Phoenix
  trace link.
* **Timing panel** — collapsed ``<details>`` with per-step timeline bars.
* **Phoenix trace link** — built from the browser host at port 6006.

Shell contract
--------------
Calls ``render_console_page(title=…, extra_css=…, body=…, view_js=…)``.
``_CHAT_BODY`` fills ``#appView``; ``_CHAT_CSS`` is appended inside
``<style>``; ``_CHAT_JS`` is a ``<script>`` block that defines
``window.onConsoleReady``, called by the shell after sign-in.

Inside ``_CHAT_JS``, ``window.IWConsole`` provides:
``API``, ``authHeaders(extra)``, ``esc(s)``, ``toast(msg, isErr)``,
``token()``, ``$(id)``.

No external CDN / hosted fonts (Invariant 8 — local-only).  The placeholder
tokens ``__BODY__``, ``__VIEWJS__``, ``__EXTRA_CSS__``, ``__TITLE__`` are
handled by the shell and must NOT appear literally in the strings below.
"""

from __future__ import annotations

from iw_rag.api.console_shell import render_console_page

# ---------------------------------------------------------------------------
# Body markup — fills #appView after sign-in
# ---------------------------------------------------------------------------

_CHAT_BODY = r"""
<div class="chat-outer">

  <!-- ── Controls row ── -->
  <div class="chat-controls card">
    <div class="controls-row">
      <div class="toggle-group">
        <label class="switch" aria-label="Grounding mode toggle">
          <input type="checkbox" id="assistToggle" checked />
          <span class="slider"></span>
        </label>
        <span class="toggle-label" id="modeLabel">Answer from general knowledge when my documents don't cover it.</span>
      </div>
      <div id="modeHint" class="mode-hint mode-assist">Assist mode — uses general knowledge as a fallback.</div>
    </div>
  </div>

  <!-- ── Message log ── -->
  <div class="card chat-card">
    <div class="chat-log" id="chatLog" aria-live="polite" aria-label="Conversation"></div>

    <!-- Input row -->
    <div class="chat-input-row">
      <textarea
        id="q"
        class="chat-textarea"
        placeholder="Ask a question about your documents…"
        rows="2"
        aria-label="Question"
      ></textarea>
      <button class="chat-clear" id="clear" aria-label="Clear conversation and memory" title="Clear conversation and memory">Clear</button>
      <button class="btn-primary chat-send" id="send" aria-label="Send message">Send</button>
    </div>
  </div>

</div>
"""

# ---------------------------------------------------------------------------
# View-specific CSS — appended inside the shared <style>
# ---------------------------------------------------------------------------

_CHAT_CSS = r"""
  /* ── Layout ── */
  .chat-outer { display: flex; flex-direction: column; gap: 14px; }
  .chat-card { padding: 0; overflow: hidden; display: flex; flex-direction: column; }
  .chat-controls { padding: 14px 18px; margin-bottom: 0; }
  .controls-row { display: flex; align-items: center; gap: 18px; flex-wrap: wrap; }
  .toggle-group { display: flex; align-items: center; gap: 10px; }
  .toggle-label { font-size: 13px; color: var(--text); line-height: 1.4; }
  .mode-hint { font-size: 12px; padding: 3px 10px; border-radius: 999px; border: 1px solid transparent; }
  .mode-assist { color: var(--teal); border-color: var(--teal-deep); background: rgba(45,212,191,0.08); }
  .mode-strict { color: var(--warn); border-color: #5b4322; background: rgba(245,158,11,0.07); }

  /* ── Message log ── */
  .chat-log { display: flex; flex-direction: column; gap: 12px; overflow-y: auto; max-height: 54vh; padding: 18px 18px 6px; scroll-behavior: smooth; }
  .msg-wrap { display: flex; flex-direction: column; }
  .msg-wrap.user { align-items: flex-end; }
  .msg-wrap.bot { align-items: flex-start; }
  .bubble { padding: 12px 16px; border-radius: 14px; max-width: 86%; word-wrap: break-word; line-height: 1.6; font-size: 14px; }
  .bubble.user { background: var(--ink-3); border: 1px solid var(--line); border-bottom-right-radius: 4px; }
  .bubble.bot { background: rgba(24,29,41,0.75); border: 1px solid var(--line); border-bottom-left-radius: 4px; }
  .bubble .thinking { color: var(--muted); font-style: italic; font-size: 13px; }

  /* ── Trust badges (4 states) ── */
  .trust-badge { display: inline-block; font-size: 11px; font-weight: 600; padding: 3px 9px; border-radius: 999px; margin-bottom: 9px; border: 1px solid transparent; }
  .trust-badge.grounded  { background: rgba(45,212,191,0.14); color: var(--teal); border-color: var(--teal-deep); }
  .trust-badge.refused   { background: rgba(244,63,94,0.14); color: #ffb3c0; border-color: var(--danger); }
  .trust-badge.genknow   { background: rgba(245,158,11,0.14); color: #fde7c0; border-color: #5b4322; }
  .trust-badge.unverif   { background: rgba(245,158,11,0.10); color: #fde7c0; border-color: #5b4322; }

  /* ── Markdown rendered answer ── */
  .answer-body { font-size: 14px; line-height: 1.65; color: var(--text); }
  .answer-body h1,.answer-body h2,.answer-body h3 { font-family: "Space Grotesk", system-ui, sans-serif; margin: 14px 0 6px; color: var(--text); }
  .answer-body h1 { font-size: 17px; } .answer-body h2 { font-size: 15px; } .answer-body h3 { font-size: 13px; }
  .answer-body p { margin: 6px 0; }
  .answer-body ul,.answer-body ol { margin: 6px 0 6px 20px; padding: 0; }
  .answer-body li { margin: 3px 0; }
  .answer-body code { font-family: "JetBrains Mono", ui-monospace, monospace; font-size: 12px; background: rgba(45,212,191,0.10); border: 1px solid rgba(45,212,191,0.2); padding: 1px 5px; border-radius: 4px; }
  .answer-body pre { background: var(--ink-2); border: 1px solid var(--line); border-radius: 8px; padding: 12px 14px; overflow-x: auto; margin: 10px 0; }
  .answer-body pre code { background: none; border: none; padding: 0; font-size: 12px; }
  .answer-body a { color: var(--teal); text-decoration: underline; text-underline-offset: 2px; }
  .answer-body strong { color: var(--text); font-weight: 700; }
  .answer-body em { color: #c4c9d8; }

  /* ── Citations ── */
  .cites-section { margin-top: 12px; border-top: 1px dashed var(--line); padding-top: 10px; }
  .cites-heading { font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: .5px; color: var(--muted); margin-bottom: 6px; }
  .cite-row { font-size: 12px; color: var(--muted); margin: 4px 0; line-height: 1.5; }
  .cite-marker { font-weight: 700; color: var(--teal); }
  .cite-source { font-family: "JetBrains Mono", ui-monospace, monospace; font-size: 11px; }
  .cite-quote { font-style: italic; color: var(--muted); display: block; margin-top: 2px; }
  .cite-row a { color: var(--teal); }

  /* ── Provenance / timing panels ── */
  .panels-area { margin-top: 10px; display: flex; flex-direction: column; gap: 8px; }
  details.prov-panel, details.time-panel { background: rgba(24,29,41,0.5); border: 1px solid var(--line); border-radius: 10px; padding: 0; }
  details.prov-panel summary, details.time-panel summary { cursor: pointer; font-size: 12px; font-weight: 600; padding: 8px 12px; color: var(--muted); list-style: none; display: flex; align-items: center; gap: 6px; user-select: none; }
  details.prov-panel summary::before { content: "▶"; font-size: 10px; transition: .15s; }
  details.time-panel summary::before { content: "▶"; font-size: 10px; transition: .15s; }
  details[open].prov-panel summary::before, details[open].time-panel summary::before { content: "▼"; }
  details.prov-panel summary:hover, details.time-panel summary:hover { color: var(--text); }
  .prov-body, .time-body { padding: 0 12px 12px; font-size: 12px; color: var(--muted); display: flex; flex-direction: column; gap: 8px; }
  .prov-row { display: flex; gap: 6px; align-items: flex-start; }
  .prov-label { font-weight: 700; color: var(--text); white-space: nowrap; min-width: 90px; }
  .prov-row code { font-family: "JetBrains Mono", ui-monospace, monospace; font-size: 11px; background: rgba(45,212,191,0.08); border: 1px solid rgba(45,212,191,0.15); padding: 1px 4px; border-radius: 4px; }
  .guard-trail { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 4px; }
  .guard-trail li { display: flex; align-items: flex-start; gap: 6px; }
  .guard-trail li::before { content: "•"; color: var(--muted); }
  .pii-note { color: var(--warn); font-size: 11px; padding: 3px 8px; background: rgba(245,158,11,0.07); border: 1px solid #5b4322; border-radius: 6px; }
  .trace-row { font-size: 11px; }
  .trace-row a { color: var(--teal); }

  /* ── Timeline bars ── */
  .timeline-rows { display: flex; flex-direction: column; gap: 6px; }
  .tl-row { display: flex; align-items: center; gap: 10px; }
  .tl-label { min-width: 120px; font-size: 11px; color: var(--text); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .tl-bar-wrap { flex: 1; background: var(--ink-3); border-radius: 4px; height: 8px; overflow: hidden; }
  .tl-bar { height: 100%; background: linear-gradient(90deg, var(--teal-deep), var(--teal)); border-radius: 4px; transition: width .3s; }
  .tl-dur { min-width: 52px; font-size: 11px; color: var(--muted); text-align: right; }
  .tl-what { font-size: 11px; color: var(--muted); }

  /* ── Input row ── */
  .chat-input-row { display: flex; gap: 10px; align-items: flex-end; padding: 14px 18px 18px; border-top: 1px solid var(--line); }
  .chat-textarea { flex: 1; resize: none; min-height: 42px; max-height: 160px; overflow-y: auto; }
  .chat-send { flex-shrink: 0; align-self: flex-end; padding: 10px 20px; }
  .chat-clear { flex-shrink: 0; align-self: flex-end; padding: 10px 16px; background: transparent; color: var(--muted); border: 1px solid var(--line); border-radius: 8px; cursor: pointer; font-size: 13px; }
  .chat-clear:hover { color: #ffb3c0; border-color: var(--danger); }
  .chat-clear:disabled { opacity: .5; cursor: default; }
  .chat-empty { color: var(--muted); font-size: 14px; font-style: italic; padding: 24px 0; text-align: center; }
"""

# ---------------------------------------------------------------------------
# View JavaScript — defines window.onConsoleReady (called by shell after login)
# ---------------------------------------------------------------------------

_CHAT_JS = r"""<script>
(function () {
  "use strict";

  // ── Tiny safe Markdown renderer (~50 lines) ──────────────────────────────
  // HTML is escaped FIRST; then a limited, known-safe transform set is applied.
  // No raw model HTML is ever injected.
  function renderMd(raw) {
    if (!raw) return "";
    // 1. HTML-escape the full string before any transform.
    var s = raw
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");

    // 2. Fenced code blocks (``` lang\n…\n```).
    s = s.replace(/```[\w]*\n([\s\S]*?)```/g, function (_, code) {
      return "<pre><code>" + code + "</code></pre>";
    });

    // 3. Headings (### H3, ## H2, # H1) — full-line.
    s = s.replace(/^### ([^\n]+)/gm, "<h3>$1</h3>");
    s = s.replace(/^## ([^\n]+)/gm, "<h2>$1</h2>");
    s = s.replace(/^# ([^\n]+)/gm, "<h1>$1</h1>");

    // 4. Unordered lists (- item or * item).
    s = s.replace(/((?:^[\-\*] [^\n]+\n?)+)/gm, function (block) {
      var items = block.trim().split("\n").map(function (ln) {
        return "<li>" + ln.replace(/^[\-\*] /, "") + "</li>";
      }).join("");
      return "<ul>" + items + "</ul>";
    });

    // 5. Ordered lists (1. item).
    s = s.replace(/((?:^\d+\. [^\n]+\n?)+)/gm, function (block) {
      var items = block.trim().split("\n").map(function (ln) {
        return "<li>" + ln.replace(/^\d+\. /, "") + "</li>";
      }).join("");
      return "<ol>" + items + "</ol>";
    });

    // 6. Inline code (`code`).
    s = s.replace(/`([^`\n]+)`/g, "<code>$1</code>");

    // 7. Bold (**text**) and italic (*text*).
    s = s.replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>");
    s = s.replace(/\*([^*\n]+)\*/g, "<em>$1</em>");

    // 8. Links [text](url) — only allow http/https/relative (no javascript:).
    s = s.replace(/\[([^\]]+)\]\(((?:https?:\/\/|\/)[^)]+)\)/g,
      '<a href="$2" target="_blank" rel="noopener">$1</a>');

    // 9. Paragraphs — blank-line separated blocks become <p>.
    s = s.split(/\n{2,}/).map(function (blk) {
      blk = blk.trim();
      if (!blk) return "";
      // Don't wrap already-block-tagged content.
      if (/^<(h[1-3]|ul|ol|pre|li)/.test(blk)) return blk;
      return "<p>" + blk.replace(/\n/g, "<br>") + "</p>";
    }).join("\n");

    return s;
  }

  // ── Helpers ──────────────────────────────────────────────────────────────
  var C = window.IWConsole;
  var $ = C.$;

  // Stable per-page session id (scope ephemeral attachments server-side).
  var sessionId = (window.crypto && typeof crypto.randomUUID === "function")
    ? crypto.randomUUID()
    : ("s-" + Date.now() + "-" + Math.floor(Math.random() * 1e6));

  // Persisted conversation id (multi-turn chat memory). Survives page refresh
  // via localStorage so reloading the console keeps the thread; the server
  // re-validates ownership on every use, so a stale/foreign id is harmless.
  var CONV_KEY = "iwrag_conversation_id";
  var conversationId = "";
  try { conversationId = window.localStorage.getItem(CONV_KEY) || ""; } catch (_e) {}

  function setConversationId(id) {
    conversationId = id || "";
    try {
      if (conversationId) window.localStorage.setItem(CONV_KEY, conversationId);
      else window.localStorage.removeItem(CONV_KEY);
    } catch (_e) {}
  }

  function scrollLog() {
    var log = $("chatLog");
    if (log) log.scrollTop = log.scrollHeight;
  }

  // Create a message wrapper + bubble; returns the bubble element.
  function mkBubble(role) {
    var log = $("chatLog");
    var wrap = document.createElement("div");
    wrap.className = "msg-wrap " + role;
    var bub = document.createElement("div");
    bub.className = "bubble " + role;
    wrap.appendChild(bub);
    log.appendChild(wrap);
    scrollLog();
    return bub;
  }

  // ── Trust badge ──────────────────────────────────────────────────────────
  // Four states, most-specific first:
  //   refused   → grounded=false                          (red)
  //   genknow   → diagnostics.general_knowledge           (amber)
  //   unverif   → diagnostics.answer_ungrounded (but grounded=true) (amber)
  //   grounded  → grounded=true + neither of the above   (teal)
  function makeBadge(grounded, diag) {
    diag = diag || {};
    var cls, label;
    if (!grounded) {
      cls = "refused";
      label = "Fail-closed refusal · not enough grounded evidence";
    } else if (diag.general_knowledge) {
      cls = "genknow";
      label = "General knowledge · not from your documents";
    } else if (diag.answer_ungrounded) {
      cls = "unverif";
      label = "Unverified · not fully supported by your documents";
    } else {
      cls = "grounded";
      label = "Grounded · answered from your documents";
    }
    return '<span class="trust-badge ' + cls + '">' + C.esc(label) + "</span>";
  }

  // ── Citations HTML ────────────────────────────────────────────────────────
  function citesHtml(citations) {
    if (!citations || !citations.length) return "";
    var rows = citations.map(function (c) {
      var marker = c.marker != null ? "[" + c.marker + "]" : "[?]";
      var src = c.source_uri || c.document_id || "source";
      var dlHtml = "";
      if (c.download_url) {
        var href = C.API + c.download_url;
        dlHtml = ' <a href="' + C.esc(href) + '" download rel="noopener">download</a>';
      }
      var quote = c.quote ? c.quote.slice(0, 160) : "";
      var quoteHtml = quote ? ('<span class="cite-quote">“' + C.esc(quote) + (quote.length === 160 ? "…" : "") + '”</span>') : "";
      return (
        '<div class="cite-row">' +
          '<span class="cite-marker">' + C.esc(marker) + "</span> " +
          '<span class="cite-source">' + C.esc(src) + "</span>" +
          dlHtml +
          quoteHtml +
        "</div>"
      );
    }).join("");
    return '<div class="cites-section"><div class="cites-heading">Sources</div>' + rows + "</div>";
  }

  // ── Phoenix trace link HTML ───────────────────────────────────────────────
  function traceHtml(diag, traceId) {
    diag = diag || {};
    var ref = diag.trace_ref;
    var tid = (ref && typeof ref === "object" && ref.trace_id) ? ref.trace_id
            : (typeof ref === "string" && ref) ? ref
            : (traceId || "");
    if (!tid) return "";
    var base = location.protocol + "//" + location.hostname + ":6006";
    return (
      '<div class="trace-row">' +
        '<a href="' + C.esc(base) + '/projects" target="_blank" rel="noopener">View full trace in Phoenix</a>' +
        " · <code>trace_id=" + C.esc(tid) + "</code>" +
      "</div>"
    );
  }

  // ── Guard trail HTML ──────────────────────────────────────────────────────
  function guardTrailHtml(trail) {
    if (!trail || !trail.length) return "";
    var items = trail.map(function (g) {
      var score = (g.score != null) ? " (" + g.score + ")" : "";
      var reason = g.reason ? " — " + C.esc(g.reason) : "";
      return (
        "<li>• <code>" + C.esc(g.kind || "") + "</code>: <strong>" +
        C.esc(g.value || "") + "</strong>" + C.esc(score) + reason + "</li>"
      );
    }).join("");
    return '<ul class="guard-trail">' + items + "</ul>";
  }

  // ── Provenance panel HTML ─────────────────────────────────────────────────
  function provHtml(resp, allowUngrounded) {
    var diag = resp.diagnostics || {};
    var verdict = diag.output_guard_gen_verdict || "";
    var assistMode = diag.output_guard_assist_mode || allowUngrounded;
    var modeLabel = assistMode ? "assist (general knowledge fallback enabled)" : "strict (documents-only, fail-closed)";
    var retriever = C.esc(resp.retriever || "");
    var trail = guardTrailHtml(diag.guard_trail);
    var piiNote = diag.output_guard_pii_redacted
      ? '<div class="pii-note">PII redaction applied to this response.</div>'
      : "";
    var citeBlock = citesHtml(resp.citations);
    var trace = traceHtml(diag, resp.trace_id);

    return (
      '<details class="prov-panel">' +
        "<summary>🔎 How grounded is this?</summary>" +
        '<div class="prov-body">' +
          (verdict ? '<div class="prov-row"><span class="prov-label">Verdict</span><span>' + C.esc(verdict) + "</span></div>" : "") +
          '<div class="prov-row"><span class="prov-label">Mode</span><span>' + C.esc(modeLabel) + "</span></div>" +
          '<div class="prov-row"><span class="prov-label">Retrieval</span><span>' + retriever + "</span></div>" +
          (trail ? '<div class="prov-row"><span class="prov-label">Guard trail</span><div>' + trail + "</div></div>" : "") +
          piiNote +
          (citeBlock ? '<div>' + citeBlock + "</div>" : "") +
          trace +
        "</div>" +
      "</details>"
    );
  }

  // ── Timing panel HTML ─────────────────────────────────────────────────────
  function timingHtml(diag) {
    diag = diag || {};
    var total = diag.total_ms;
    var timeline = diag.timeline;
    if (!Array.isArray(timeline) || !timeline.length) return "";

    var totalLabel = (total != null)
      ? (total >= 1000 ? (total / 1000).toFixed(1) + " s" : Math.round(total) + " ms")
      : "?";

    var maxDur = 0;
    for (var i = 0; i < timeline.length; i++) {
      if ((timeline[i].duration_ms || 0) > maxDur) maxDur = timeline[i].duration_ms;
    }
    if (!maxDur) maxDur = 1;

    var rows = timeline.map(function (step) {
      var dur = step.duration_ms || 0;
      var pct = Math.round((dur / maxDur) * 100);
      var durLabel = dur >= 1000 ? (dur / 1000).toFixed(1) + " s" : Math.round(dur) + " ms";
      var whatHow = [step.what, step.how].filter(Boolean).map(C.esc).join(" · ");
      return (
        '<div class="tl-row">' +
          '<span class="tl-label" title="' + C.esc(step.label || step.key || "") + '">' + C.esc(step.label || step.key || "") + "</span>" +
          '<div class="tl-bar-wrap"><div class="tl-bar" style="width:' + pct + '%"></div></div>' +
          '<span class="tl-dur">' + C.esc(durLabel) + "</span>" +
        "</div>" +
        (whatHow ? '<div class="tl-what">' + whatHow + "</div>" : "")
      );
    }).join("");

    return (
      '<details class="time-panel">' +
        "<summary>⏱ " + C.esc(totalLabel) + "</summary>" +
        '<div class="time-body"><div class="timeline-rows">' + rows + "</div></div>" +
      "</details>"
    );
  }

  // ── Render a complete assistant bubble ────────────────────────────────────
  function renderBotBubble(bub, resp, allowUngrounded) {
    var diag = resp.diagnostics || {};
    var badge = makeBadge(resp.grounded, diag);
    var answerHtml = '<div class="answer-body">' + renderMd(resp.answer || "") + "</div>";
    var cites = citesHtml(resp.citations);
    var prov = provHtml(resp, allowUngrounded);
    var timing = timingHtml(diag);
    var panels = (prov || timing)
      ? '<div class="panels-area">' + prov + timing + "</div>"
      : "";
    bub.innerHTML = badge + answerHtml + cites + panels;
  }

  // ── Mode toggle ───────────────────────────────────────────────────────────
  function updateModeHint() {
    var tog = $("assistToggle");
    var hint = $("modeHint");
    if (!tog || !hint) return;
    if (tog.checked) {
      hint.className = "mode-hint mode-assist";
      hint.textContent = "Assist mode — uses general knowledge as a fallback.";
    } else {
      hint.className = "mode-hint mode-strict";
      hint.textContent = "Strict mode — documents-only, fails closed when evidence is insufficient.";
    }
  }

  // ── Send a query ──────────────────────────────────────────────────────────
  async function ask() {
    var qEl = $("q");
    var sendBtn = $("send");
    if (!qEl || !sendBtn) return;
    var q = qEl.value.trim();
    if (!q) return;
    if (!C.token()) { C.toast("Please sign in first", true); return; }

    var tog = $("assistToggle");
    var allowUngrounded = tog ? tog.checked : true;

    qEl.value = "";
    sendBtn.disabled = true;

    // User bubble.
    var ubub = mkBubble("user");
    ubub.textContent = q;

    // Bot placeholder bubble.
    var bbub = mkBubble("bot");
    bbub.innerHTML = '<span class="thinking">thinking…</span>';

    try {
      var resp = await fetch(C.API + "/query", {
        method: "POST",
        headers: C.authHeaders({ "Content-Type": "application/json" }),
        body: JSON.stringify({
          query: q,
          allow_ungrounded: allowUngrounded,
          session_id: sessionId,
          conversation_id: conversationId || null
        })
      });

      if (resp.status === 401 || resp.status === 403) {
        bbub.innerHTML = makeBadge(false, {}) + '<div class="answer-body"><p>Not authorized — please sign in again.</p></div>';
        return;
      }
      if (!resp.ok) {
        var errMsg = "Request failed (" + resp.status + ").";
        try { var ej = await resp.json(); if (ej && (ej.detail || ej.error)) errMsg = ej.detail || ej.error; } catch (_e) {}
        bbub.innerHTML = makeBadge(false, {}) + '<div class="answer-body"><p>' + C.esc(errMsg) + "</p></div>";
        return;
      }

      var data = await resp.json();
      // Persist the conversation id the server resolved/created so the next
      // turn carries it (multi-turn memory).
      if (data && data.conversation_id) setConversationId(data.conversation_id);
      renderBotBubble(bbub, data, allowUngrounded);

    } catch (e) {
      bbub.innerHTML = makeBadge(false, {}) + '<div class="answer-body"><p>Network error — ' + C.esc(String(e)) + "</p></div>";
    } finally {
      sendBtn.disabled = false;
      scrollLog();
    }
  }

  // ── Empty-state hint ──────────────────────────────────────────────────────
  function showEmptyHint() {
    var log = $("chatLog");
    if (!log || log.querySelector(".chat-empty") || log.children.length) return;
    var hint = document.createElement("div");
    hint.className = "chat-empty";
    hint.textContent = "Ask a question about your documents to get started.";
    log.appendChild(hint);
  }
  function removeEmptyHint() {
    var log = $("chatLog");
    var h = log ? log.querySelector(".chat-empty") : null;
    if (h) h.remove();
  }

  // ── History replay (multi-turn memory) ────────────────────────────────────
  // Render one persisted turn into the log on load.
  function renderStored(msg) {
    if (!msg || !msg.content) return;
    if (msg.role === "user") {
      mkBubble("user").textContent = msg.content;
    } else {
      renderBotBubble(mkBubble("bot"), {
        answer: msg.content,
        grounded: msg.disposition === "answered",
        citations: msg.citations || [],
        diagnostics: {}
      }, true);
    }
  }

  // Load + render the persisted turns of the current conversation.
  async function loadHistory() {
    if (!conversationId || !C.token()) return;
    try {
      var resp = await fetch(C.API + "/conversations/" + encodeURIComponent(conversationId), {
        headers: C.authHeaders({})
      });
      if (!resp.ok) return;
      var data = await resp.json();
      var msgs = (data && data.messages) || [];
      if (!msgs.length) return;
      removeEmptyHint();
      msgs.forEach(renderStored);
      scrollLog();
    } catch (_e) { /* memory is best-effort; ignore load errors */ }
  }

  // ── Clear: delete server-side memory + reset the view ─────────────────────
  async function clearConversation() {
    var clearBtn = $("clear");
    if (clearBtn) clearBtn.disabled = true;
    try {
      if (conversationId && C.token()) {
        await fetch(C.API + "/conversations/" + encodeURIComponent(conversationId), {
          method: "DELETE",
          headers: C.authHeaders({})
        });
      }
    } catch (_e) { /* idempotent; clear the UI regardless */ }
    finally {
      setConversationId("");
      var log = $("chatLog");
      if (log) log.innerHTML = "";
      showEmptyHint();
      if (clearBtn) clearBtn.disabled = false;
      C.toast("Conversation cleared");
    }
  }

  // ── onConsoleReady — shell calls this after sign-in ───────────────────────
  window.onConsoleReady = function () {
    var sendBtn = $("send");
    var qEl = $("q");
    var tog = $("assistToggle");
    var clearBtn = $("clear");

    showEmptyHint();

    if (sendBtn) {
      sendBtn.addEventListener("click", function () { removeEmptyHint(); ask(); });
    }

    if (qEl) {
      qEl.addEventListener("keydown", function (e) {
        if (e.key === "Enter" && !e.shiftKey) {
          e.preventDefault();
          removeEmptyHint();
          ask();
        }
      });
    }

    if (clearBtn) {
      clearBtn.addEventListener("click", clearConversation);
    }

    if (tog) {
      tog.addEventListener("change", updateModeHint);
      updateModeHint();
    }

    // Restore prior turns of the persisted conversation (multi-turn memory).
    loadHistory();
  };

})();
</script>"""


def console_chat_page_html() -> str:
    """Return the rich native Chat console view HTML.

    Assembles the shared console shell (brand chrome, login, nav, token
    plumbing) with the Chat-specific body markup, view CSS, and JavaScript
    via :func:`iw_rag.api.console_shell.render_console_page`.

    The returned page features:
    - Grounding toggle (assist / strict mode).
    - Scrollable in-session conversation log with user/bot bubbles.
    - Inline safe Markdown renderer (no raw model HTML injected).
    - Four-state trust badge (grounded / fail-closed / general-knowledge /
      unverified).
    - Citations list with server-minted download links.
    - Provenance panel (guard trail, PII note, Phoenix trace link).
    - Timing panel with proportional per-step bars.

    Returns:
        A complete, self-contained HTML document string suitable for serving
        directly from a FastAPI ``HTMLResponse``.
    """
    return render_console_page(
        title="Chat",
        extra_css=_CHAT_CSS,
        body=_CHAT_BODY,
        view_js=_CHAT_JS,
    )


__all__ = ["console_chat_page_html"]
