# ruff: noqa: E501
"""The reranker lab's markup, styles and script (rendered inside the reranking lesson).

A thin client over ``GET /learn/lab/rerank``: the order, the expansion rule and every
verdict come from :mod:`iw_rag.learn.rerank_lab`, which replays a recorded run of the
production spine.  Every server string goes through ``IWConsole.esc`` before it reaches
the DOM.
"""

from __future__ import annotations

from html import escape

from iw_rag.learn.rerank_corpus import QUERIES

LAB_CSS = """
  .rr-ctl-row { display: flex; flex-wrap: wrap; gap: 14px 28px; margin-bottom: 12px; align-items: flex-start; }
  .rr-ctl-row .grp > .lbl { display: block; font-size: 12px; text-transform: uppercase; letter-spacing: .08em; color: var(--muted); margin-bottom: 6px; }
  .rr-questions { display: flex; flex-direction: column; gap: 6px; max-width: 520px; }
  .rr-questions button { text-align: left; font-size: 13px; padding: 7px 12px; border-radius: 8px; background: var(--ink-2); }
  .rr-questions button[aria-pressed="true"] { border-color: var(--teal); color: var(--teal); }
  .rr-sliders { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 14px; min-width: 280px; max-width: 520px; flex: 1; }
  .rr-slider .name { display: flex; justify-content: space-between; font-size: 13px; margin-bottom: 4px; }
  .rr-slider .name b { color: var(--teal); }
  .rr-slider input[type=range] { width: 100%; accent-color: var(--teal); }
  .rr-verdict { border: 1px solid var(--line); border-left: 4px solid var(--teal); border-radius: 10px; padding: 12px 16px; margin-bottom: 14px; background: rgba(24,29,41,0.5); }
  .rr-verdict .q { font-weight: 600; margin-bottom: 2px; } .rr-verdict .l { color: var(--muted); font-size: 13px; margin-bottom: 8px; }
  .rr-chips { display: flex; flex-wrap: wrap; gap: 8px; margin: 6px 0; }
  .rr-chip { border: 1px solid var(--line); border-radius: 999px; padding: 3px 11px; font-size: 13px; } .rr-chip b { color: var(--teal); } .rr-chip.bad b { color: var(--warn); }
  .rr-verdict ul { margin: 8px 0 0; padding-left: 18px; font-size: 14px; line-height: 1.55; }
  .rr-cols { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }
  @media (max-width: 800px) { .rr-cols { grid-template-columns: 1fr; } }
  .rr-col h4 { margin: 0 0 2px; font-size: 14px; } .rr-col .sub { color: var(--muted); font-size: 12px; margin-bottom: 8px; line-height: 1.4; }
  .rr-hit { border: 1px solid var(--line); border-radius: 8px; padding: 7px 10px; margin-bottom: 6px; font-size: 12.5px; background: rgba(24,29,41,0.5); }
  .rr-hit.gold { border-color: var(--ok); box-shadow: 0 0 0 1px var(--ok) inset; } .rr-hit.cut { opacity: .5; }
  .rr-hit .top { display: flex; align-items: center; gap: 7px; color: var(--muted); font-size: 11.5px; margin-bottom: 3px; flex-wrap: wrap; }
  .rr-hit .rk { background: var(--teal-deep); color: #06231f; font-weight: 700; border-radius: 999px; min-width: 22px; height: 22px; display: inline-flex; align-items: center; justify-content: center; font-size: 12px; }
  .rr-hit .kind { border: 1px solid var(--line); border-radius: 999px; padding: 0 7px; font-size: 10.5px; } .rr-hit .kind.section { border-color: var(--warn); color: var(--warn); }
  .rr-hit .star { color: var(--ok); font-weight: 700; margin-left: auto; }
  .rr-hit .tx { line-height: 1.4; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }
  .rr-hit .sc { color: var(--muted); font-size: 11.5px; margin-top: 3px; }
  .rr-up { color: var(--ok); font-weight: 600; } .rr-down { color: var(--warn); font-weight: 600; }
  .rr-cutline { text-align: center; font-size: 11.5px; color: var(--danger); border-top: 2px dashed var(--danger); margin: 8px 0; padding-top: 3px; }
  .rr-sent { margin-top: 16px; }
  .rr-sent h3 { margin-bottom: 4px; }
  .rr-stats { display: flex; flex-wrap: wrap; gap: 10px; margin: 6px 0 10px; font-size: 13px; }
  .rr-stats span { border: 1px solid var(--line); border-radius: 8px; padding: 4px 10px; } .rr-stats b { color: var(--teal); } .rr-stats .warn b { color: var(--warn); }
  .rr-card { border: 1px solid var(--line); border-left: 4px solid var(--teal-deep); border-radius: 8px; padding: 10px 12px; margin-bottom: 8px; font-size: 13px; background: rgba(24,29,41,0.5); }
  .rr-card.section { border-left-color: var(--warn); }
  .rr-card .hd { color: var(--muted); font-size: 12px; display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 5px; }
  .rr-card .tx { white-space: pre-wrap; line-height: 1.5; }
  .rr-card mark { background: rgba(45,212,191,0.28); color: inherit; border-radius: 3px; padding: 0 1px; }
  .rr-card .nt { color: var(--muted); font-size: 12px; margin-top: 6px; line-height: 1.45; }
  .rr-card details summary { cursor: pointer; color: var(--teal); font-size: 12.5px; }
  .rr-foot { color: var(--muted); font-size: 12px; line-height: 1.55; margin-top: 12px; }
"""


def lab_html() -> str:
    """Return the reranker lab's markup, with the question picker rendered server-side."""
    questions = "".join(
        f'<button type="button" data-q="{escape(q.query_id)}" aria-pressed="{"true" if i == 0 else "false"}">{escape(q.text)}</button>'
        for i, q in enumerate(QUERIES)
    )
    return (
        '<div class="lab" id="rerankLab">'
        '<h2 class="lab-title">Try it: from 49 candidates to what the model reads</h2>'
        '<p class="lab-sub">This replays a real run of IW-RAG\'s own retrieval spine over a small library: the fused candidate list, '
        "the cross-encoder's score for every candidate, and the small-to-big expansion. Change how many chunks are kept, switch the "
        "reranker off (what the spine does when the reranker is down) or change the expansion, and see what reaches the generator. "
        "Every setting reproduces what the real spine returned.</p>"
        '<div class="rr-ctl-row">'
        f'<div class="grp"><span class="lbl">Question</span><div class="rr-questions" id="rrQuestions">{questions}</div></div>'
        '<div class="grp"><span class="lbl">Reranker</span><div class="seg-btn" id="rrRerank">'
        '<button type="button" data-v="on" aria-pressed="true">On</button><button type="button" data-v="off" aria-pressed="false">Off (down)</button></div>'
        '<span class="lbl" style="margin-top:12px">Generator receives</span><div class="seg-btn" id="rrExpand">'
        '<button type="button" data-v="parent" aria-pressed="true">Section (small-to-big)</button>'
        '<button type="button" data-v="leaf" aria-pressed="false">Matched leaf only</button></div>'
        '<span class="lbl" style="margin-top:12px">Sections can match on their own</span><div class="seg-btn" id="rrSections">'
        '<button type="button" data-v="on" aria-pressed="true">On</button><button type="button" data-v="off" aria-pressed="false">Off</button></div></div>'
        '<div class="rr-sliders">'
        '<div class="rr-slider"><div class="name"><span>Keep top-k</span><b id="rrKVal">8</b></div>'
        '<input type="range" id="rrK" min="1" max="10" step="1" value="8" aria-label="Chunks kept after reranking"></div>'
        '<div class="rr-slider"><div class="name"><span>Section budget (characters)</span><b id="rrBVal">4000</b></div>'
        '<input type="range" id="rrB" min="500" max="8000" step="500" value="4000" aria-label="Expansion budget in characters"></div>'
        "</div></div>"
        '<div id="rrVerdict"></div>'
        '<div class="rr-cols" id="rrCols"></div>'
        '<div class="rr-sent"><h3>What the generator receives</h3><div class="rr-stats" id="rrStats"></div><div id="rrSent"></div></div>'
        '<div class="rr-foot" id="rrFoot"></div>'
        '<div class="lab-err" id="rrErr" hidden></div>'
        "</div>"
    )


LAB_JS = """<script>
(function () {
  var q = null, reranker = "on", expansion = "parent", sections = "on", timer = null, seq = 0;
  function $(id) { return document.getElementById(id); }
  function esc(s) { return window.IWConsole.esc(s); }
  function fmt(n) { return Number(n).toLocaleString("en-US"); }
  function label(c) { return esc(c.chunk_id) + " · " + esc(c.document) + " › " + esc(c.section); }

  function hit(c, keptRank, topK, mode) {
    var inKept = mode === "after" ? c.rerank_rank_shown <= topK : c.fused_rank <= topK;
    var delta = "";
    if (mode === "after" && c.fused_rank !== c.rerank_rank) {
      var d = c.fused_rank - c.rerank_rank;
      delta = d > 0 ? ' · <span class="rr-up">↑ ' + d + " (was #" + c.fused_rank + ")</span>" : ' · <span class="rr-down">↓ ' + (-d) + " (was #" + c.fused_rank + ")</span>";
    }
    var rank = mode === "after" ? c.rerank_rank_shown : c.fused_rank;
    var score = mode === "after" ? "relevance " + Number(c.rerank_score).toFixed(3) + delta :
      "dense " + (c.dense_rank == null ? "—" : "#" + c.dense_rank) + " · BM25 " + (c.sparse_rank == null ? "—" : "#" + c.sparse_rank);
    return '<div class="rr-hit' + (c.is_answer ? " gold" : "") + (inKept ? "" : " cut") + '"><div class="top"><span class="rk">' + rank + '</span><span class="kind ' + c.kind + '">' +
      (c.kind === "section" ? "section" : "leaf") + "</span><span>" + label(c) + "</span>" + (c.is_answer ? '<span class="star">★ has the answer</span>' : "") +
      '</div><div class="tx">' + esc(c.text) + '</div><div class="sc">' + score + "</div></div>";
  }
  function column(list, mode, topK) {
    var out = "";
    list.forEach(function (c, i) {
      out += hit(c, i + 1, topK, mode);
      if (i + 1 === topK && list.length > topK) out += '<div class="rr-cutline">kept above · cut below (top-k = ' + topK + ")</div>";
    });
    return out;
  }
  function sentCard(s) {
    var text = s.sent_text, body;
    var i = s.expanded ? text.indexOf(s.matched_text) : -1;
    function render(t, from, upTo) { return i >= 0 ? esc(t.slice(0, i)) + "<mark>" + esc(t.slice(i, i + s.matched_text.length)) + "</mark>" + esc(t.slice(i + s.matched_text.length)) : esc(t); }
    body = text.length > 700 ? "<details><summary>Show the " + fmt(text.length) + " characters sent</summary><div class='tx'>" + render(text) + "</div></details>" :
      '<div class="tx">' + render(text) + "</div>";
    return '<div class="rr-card ' + s.kind + '"><div class="hd"><b>#' + s.rank + "</b><span>" + esc(s.chunk_id) + " · " + esc(s.document) + " › " + esc(s.section) + "</span><span>" +
      fmt(s.sent_chars) + " characters sent</span>" + (s.is_answer ? "<span>★ has the answer</span>" : "") + "</div>" + body + '<div class="nt">' + esc(s.note) +
      (s.expanded ? " The highlighted text is the matched leaf; the citation still quotes it." : "") + "</div></div>";
  }

  function render(d) {
    $("rrErr").hidden = true;
    var f = d.facts, g = f.gold_ranks, shownRank = d.settings.reranker_on ? g.reranked : g.fused;
    var cut = shownRank != null && shownRank > d.settings.top_k;
    var chips = '<div class="rr-chips"><span class="rr-chip">Answer leaf: fused <b>#' + (g.fused == null ? "—" : g.fused) + "</b> of " + d.candidates_total +
      (d.settings.reranker_on ? " → reranked <b>#" + (g.reranked == null ? "—" : g.reranked) + "</b>" : "") + "</span>" +
      '<span class="rr-chip' + (f.answer_in_context ? "" : " bad") + '">Answer reaches the generator: <b>' + (f.answer_in_context ? "yes" : "no") + "</b></span>" +
      (f.context_phrase ? '<span class="rr-chip' + (f.context_phrase_in_context ? "" : " bad") + '">“' + esc(f.context_phrase) + '…” reaches it: <b>' + (f.context_phrase_in_context ? "yes" : "no") + "</b></span>" : "") + "</div>";
    $("rrVerdict").innerHTML = '<div class="rr-verdict"><div class="q">' + esc(d.query.text) + '</div><div class="l">' + esc(d.query.lesson) + "</div>" + chips +
      (d.notes.length ? "<ul>" + d.notes.map(function (n) { return "<li>" + esc(n) + "</li>"; }).join("") + "</ul>" : "") + "</div>";
    d.reranked.forEach(function (c, i) { c.rerank_rank_shown = i + 1; });
    var shown = d.fused.length;
    $("rrCols").innerHTML =
      '<div class="rr-col"><h4>① Before: the fused order</h4><div class="sub">What dense + BM25 + RRF produced. Top ' + shown + " of " + d.candidates_total + ". Sections are searchable too (switch it off to compare).</div>" + column(d.fused, "before", d.settings.top_k) + "</div>" +
      '<div class="rr-col"><h4>② After: ' + (d.settings.reranker_on ? "the reranker's order" : "no reranker: the same order") + '</h4><div class="sub">' +
      (d.settings.reranker_on ? "The cross-encoder reads the question and each chunk together. Top " + shown + "." : "The reranker is down, so the spine keeps the fused order.") + "</div>" +
      column(d.reranked, "after", d.settings.top_k) + "</div>";
    $("rrStats").innerHTML = '<span><b>' + f.chunks_sent + "</b> chunks</span><span><b>" + fmt(f.chars_sent) + "</b> characters sent</span>" +
      '<span><b>' + f.repeats_merged + "</b> repeats merged</span>" +
      '<span class="' + (f.chars_duplicated ? "warn" : "") + '"><b>' + fmt(f.chars_duplicated) + "</b> characters repeated</span>";
    $("rrSent").innerHTML = d.selected.map(sentCard).join("");
    $("rrFoot").textContent = "Recorded " + d.meta.recorded + " with " + d.meta.embedder + " and " + d.meta.reranker + ". " + d.meta.embedding_note +
      " The spine's own defaults are top-k = " + d.meta.spine_top_k + " and a " + fmt(d.meta.spine_budget) + "-character budget. This library was chunked with the offline defaults (parent size uncapped); documents uploaded in the console cap each parent at the budget.";
  }

  async function run() {
    var mine = ++seq, k = $("rrK").value, b = $("rrB").value;
    $("rrKVal").textContent = k; $("rrBVal").textContent = fmt(b);
    try {
      var r = await fetch(window.IWConsole.API + "/learn/lab/rerank?query=" + encodeURIComponent(q) + "&top_k=" + k + "&reranker=" + reranker + "&expansion=" + expansion + "&budget=" + b + "&sections=" + sections,
        { headers: window.IWConsole.authHeaders({}) });
      if (!r.ok) throw new Error("HTTP " + r.status);
      var d = await r.json();
      if (mine === seq) render(d);
    } catch (e) {
      $("rrErr").hidden = false; $("rrErr").textContent = "Could not load the replay (" + e.message + "). Are you signed in?";
    }
  }
  function schedule() { clearTimeout(timer); timer = setTimeout(run, 120); }
  function group(sel, set) {
    var btns = document.querySelectorAll(sel);
    btns.forEach(function (b) {
      b.addEventListener("click", function () {
        set(b.getAttribute("data-q") || b.getAttribute("data-v"));
        btns.forEach(function (x) { x.setAttribute("aria-pressed", x === b ? "true" : "false"); });
        run();
      });
    });
    return btns[0].getAttribute("data-q") || btns[0].getAttribute("data-v");
  }

  window.onConsoleReady = function () {
    q = group("#rrQuestions button", function (v) { q = v; });
    group("#rrRerank button", function (v) { reranker = v; });
    group("#rrExpand button", function (v) { expansion = v; });
    group("#rrSections button", function (v) { sections = v; });
    $("rrK").addEventListener("input", schedule); $("rrB").addEventListener("input", schedule);
    run();
  };
})();
</script>"""

__all__ = ["LAB_CSS", "LAB_JS", "lab_html"]
