# ruff: noqa: E501
"""The hybrid-search lab's markup, styles and script (rendered inside the search lesson).

A thin client over ``GET /learn/lab/hybrid``: the arms, the re-fusion and the verdicts
all come from :mod:`iw_rag.learn.hybrid_lab`.  Every server string goes through
``IWConsole.esc`` before it reaches the DOM.
"""

from __future__ import annotations

from html import escape

from iw_rag.learn.lab_corpus import PRINCIPALS, QUERIES

LAB_CSS = """
  .hy-ctl-row .grp > .lbl { display: block; font-size: 12px; text-transform: uppercase; letter-spacing: .08em; color: var(--muted); margin-bottom: 6px; }
  .hy-ctl-row { display: flex; flex-wrap: wrap; gap: 14px 28px; margin-bottom: 12px; align-items: flex-start; }
  .hy-questions { display: flex; flex-direction: column; gap: 6px; max-width: 520px; }
  .hy-questions button { text-align: left; font-size: 13px; padding: 7px 12px; border-radius: 8px; background: var(--ink-2); }
  .hy-questions button[aria-pressed="true"] { border-color: var(--teal); color: var(--teal); }
  .hy-sliders { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 14px; flex: 1; min-width: 280px; max-width: 640px; }
  .hy-slider .name { display: flex; justify-content: space-between; font-size: 13px; margin-bottom: 4px; }
  .hy-slider .name b { color: var(--teal); }
  .hy-slider input[type=range] { width: 100%; accent-color: var(--teal); }
  .hy-banner { border: 1px solid var(--line); border-radius: 10px; padding: 10px 14px; margin: 6px 0 12px; font-size: 14px; background: rgba(24,29,41,0.5); line-height: 1.5; }
  .hy-banner b { color: var(--teal); }
  .hy-verdict { border: 1px solid var(--line); border-left: 4px solid var(--teal); border-radius: 10px; padding: 12px 16px; margin-bottom: 14px; background: rgba(24,29,41,0.5); }
  .hy-verdict.blocked { border-left-color: var(--warn); }
  .hy-verdict .q { font-weight: 600; margin-bottom: 2px; }
  .hy-verdict .l { color: var(--muted); font-size: 13px; margin-bottom: 8px; }
  .hy-chips { display: flex; flex-wrap: wrap; gap: 8px; margin: 6px 0; }
  .hy-chip { border: 1px solid var(--line); border-radius: 999px; padding: 3px 11px; font-size: 13px; }
  .hy-chip b { color: var(--teal); }
  .hy-chip.none b { color: var(--warn); }
  .hy-verdict ul { margin: 8px 0 0; padding-left: 18px; font-size: 14px; line-height: 1.55; }
  .hy-cols { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }
  @media (max-width: 1100px) { .hy-cols { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
  @media (max-width: 640px) { .hy-cols { grid-template-columns: 1fr; } }
  .hy-col h4 { margin: 0 0 2px; font-size: 14px; }
  .hy-col .sub { color: var(--muted); font-size: 12px; margin-bottom: 8px; min-height: 30px; line-height: 1.4; }
  .hy-col .empty { color: var(--muted); font-size: 13px; border: 1px dashed var(--line); border-radius: 8px; padding: 12px; line-height: 1.5; }
  .hy-hit { border: 1px solid var(--line); border-radius: 8px; padding: 8px 10px; margin-bottom: 8px; font-size: 12.5px; background: rgba(24,29,41,0.5); }
  .hy-hit.gold { border-color: var(--ok); box-shadow: 0 0 0 1px var(--ok) inset; }
  .hy-hit .top { display: flex; align-items: center; gap: 8px; margin-bottom: 4px; color: var(--muted); font-size: 11.5px; }
  .hy-hit .rk { background: var(--teal-deep); color: #06231f; font-weight: 700; border-radius: 999px; min-width: 22px; height: 22px; display: inline-flex; align-items: center; justify-content: center; font-size: 12px; }
  .hy-hit .star { color: var(--ok); font-weight: 700; margin-left: auto; }
  .hy-hit .tx { line-height: 1.45; display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden; }
  .hy-hit .sc { color: var(--muted); margin-top: 4px; font-size: 11.5px; }
  .hy-hit mark { background: rgba(245,158,11,0.3); color: inherit; border-radius: 3px; padding: 0 1px; }
  .hy-foot { color: var(--muted); font-size: 12px; line-height: 1.55; margin-top: 12px; }
"""


def lab_html() -> str:
    """Return the hybrid-search lab's markup, with the pickers rendered server-side."""
    questions = "".join(
        f'<button type="button" data-q="{escape(q.query_id)}" aria-pressed="{"true" if i == 0 else "false"}">{escape(q.text)}</button>'
        for i, q in enumerate(QUERIES)
    )
    users = "".join(
        f'<button type="button" data-p="{escape(p.principal_id)}" aria-pressed="{"true" if i == 0 else "false"}">{escape(p.label)}</button>'
        for i, p in enumerate(PRINCIPALS)
    )
    return (
        '<div class="lab" id="hybridLab">'
        '<h2 class="lab-title">Try it: one question, four stages</h2>'
        '<p class="lab-sub">This replays a real run of IW-RAG\'s own search components over a small corpus: '
        "the pgvector dense search, the ParadeDB BM25 search and the cross-encoder reranker, all recorded from the "
        "production code. The fusion step is re-run live with the real RRF function, so you can change how much "
        "each search counts. The answer chunk is marked with a star.</p>"
        '<div class="hy-ctl-row">'
        f'<div class="grp"><span class="lbl">Question</span><div class="hy-questions" id="hyQuestions">{questions}</div></div>'
        f'<div class="grp"><span class="lbl">Signed in as</span><div class="seg-btn" id="hyUsers">{users}</div></div>'
        "</div>"
        '<div class="hy-ctl-row"><div class="hy-sliders">'
        '<div class="hy-slider"><div class="name"><span>Dense weight</span><b id="hyWdVal">1</b></div>'
        '<input type="range" id="hyWd" min="0" max="2" step="0.25" value="1" aria-label="Dense search weight"></div>'
        '<div class="hy-slider"><div class="name"><span>BM25 weight</span><b id="hyWsVal">1</b></div>'
        '<input type="range" id="hyWs" min="0" max="2" step="0.25" value="1" aria-label="BM25 weight"></div>'
        '<div class="hy-slider"><div class="name"><span>RRF constant k</span><b id="hyKVal">60</b></div>'
        '<input type="range" id="hyK" min="1" max="200" step="1" value="60" aria-label="RRF constant"></div>'
        "</div>"
        '<div class="grp"><span class="lbl">Presets</span><div class="preset-row" id="hyPresets">'
        '<button type="button" class="btn-sm" data-wd="1" data-ws="0">Dense only</button>'
        '<button type="button" class="btn-sm" data-wd="0" data-ws="1">BM25 only</button>'
        '<button type="button" class="btn-sm" data-wd="1" data-ws="1">Hybrid (IW-RAG default)</button>'
        "</div></div></div>"
        '<div id="hyBanner" class="hy-banner"></div>'
        '<div id="hyVerdict"></div>'
        '<div class="hy-cols" id="hyCols"></div>'
        '<div class="hy-foot" id="hyFoot"></div>'
        '<div class="lab-err" id="hyErr" hidden></div>'
        "</div>"
    )


LAB_JS = """<script>
(function () {
  var q = null, p = null, timer = null, seq = 0;
  function $(id) { return document.getElementById(id); }
  function esc(s) { return window.IWConsole.esc(s); }
  function fmt(x, n) { return Number(x).toFixed(n); }

  function highlight(text, terms) {
    var t = esc(text);
    if (!terms || !terms.length) return t;
    return t.replace(new RegExp("\\\\b(" + terms.join("|") + ")\\\\b", "gi"), "<mark>$1</mark>");
  }
  function hit(h, goldId, withMarks, extra) {
    var gold = h.chunk_id === goldId;
    return '<div class="hy-hit' + (gold ? " gold" : "") + '"><div class="top"><span class="rk">' + h.rank + "</span><span>" +
      esc(h.chunk_id) + " · " + esc(h.document) + "</span>" + (gold ? '<span class="star">★ answer</span>' : "") + '</div><div class="tx">' +
      highlight(h.text, withMarks ? h.matched_terms : []) + '</div><div class="sc">' + extra + "</div></div>";
  }
  function col(title, sub, hits, total, goldId, render, emptyMsg) {
    var body = hits.length ? hits.map(render).join("") : '<div class="empty">' + esc(emptyMsg) + "</div>";
    var count = hits.length && total > hits.length ? " · top " + hits.length + " of " + total : (hits.length ? " · " + total + " result" + (total === 1 ? "" : "s") : "");
    return '<div class="hy-col"><h4>' + title + '</h4><div class="sub">' + sub + count + "</div>" + body + "</div>";
  }
  function rankLabel(r) { return r == null ? "—" : "#" + r; }
  function chip(name, r, off) {
    if (off) return '<span class="hy-chip">' + name + " <b>off</b></span>";
    return '<span class="hy-chip' + (r == null ? " none" : "") + '">' + name + " <b>" + (r == null ? "not found" : "#" + r) + "</b></span>";
  }

  function render(d) {
    $("hyErr").hidden = true;
    $("hyBanner").innerHTML = "Signed in as <b>" + esc(d.principal.label) + "</b> (groups: " + esc(d.principal.groups.join(", ")) + "). " +
      "<b>" + d.visible_count + "</b> of " + (d.visible_count + d.hidden_count) + " chunks are visible to this user" +
      (d.hidden_count ? "; the other " + d.hidden_count + " were removed by the access filter <b>before</b> any search ran." : ".");
    var g = d.gold, notes = (g.notes || []).map(function (n) { return "<li>" + esc(n) + "</li>"; }).join("");
    var chips = g.visible ? '<div class="hy-chips">' + chip("Dense", g.ranks.dense, d.weights.dense === 0) + chip("BM25", g.ranks.sparse, d.weights.sparse === 0) + chip("After fusion", g.ranks.fused) + chip("After rerank", g.ranks.reranked) + "</div>" : "";
    $("hyVerdict").innerHTML = '<div class="hy-verdict' + (g.visible ? "" : " blocked") + '"><div class="q">' + esc(d.query.text) + '</div><div class="l">' + esc(d.query.lesson) +
      "</div>" + chips + (notes ? "<ul>" + notes + "</ul>" : "") + "</div>";
    var gid = g.chunk_id;
    $("hyCols").innerHTML =
      col("① Dense search", "By meaning · pgvector, cosine distance", d.dense, d.totals.dense, gid,
        function (h) { return hit(h, gid, false, "distance " + fmt(h.score, 3) + " (lower is closer)"); },
        d.weights.dense === 0 ? "Switched off (weight 0)." : "Nothing visible to search.") +
      col("② BM25 search", "By exact words · ParadeDB. Matching words are highlighted", d.sparse, d.totals.sparse, gid,
        function (h) { return hit(h, gid, true, "BM25 score " + fmt(h.score, 2)); },
        d.weights.sparse === 0 ? "Switched off (weight 0)." : "No chunk shares a useful word with the question.") +
      col("③ After fusion", "RRF, k = " + d.rrf_k + " · weights " + d.weights.dense + " / " + d.weights.sparse, d.fused, d.totals.fused, gid,
        function (h) { return hit(h, gid, false, "RRF " + fmt(h.score, 4) + " · dense " + rankLabel(h.detail.dense_rank) + " · BM25 " + rankLabel(h.detail.sparse_rank)); },
        "Nothing to fuse.") +
      col("④ After rerank", "Cross-encoder reads question and chunk together", d.reranked, d.totals.fused, gid,
        function (h) { return hit(h, gid, false, "relevance " + fmt(h.score, 3)); },
        "Nothing reached the reranker.");
    $("hyFoot").textContent = "Recorded " + d.meta.recorded + " with " + d.meta.embedder + " and " + d.meta.reranker + ". " + d.meta.embedding_note +
      " With a corpus this small the reranker sees every visible chunk that survives fusion; the real system keeps the best 80 candidates and returns 8. " +
      "On a tiny corpus a strong embedder is often enough on its own: the hybrid gain shows at scale (see the measured ablation below).";
  }

  async function run() {
    var mine = ++seq;
    var wd = $("hyWd").value, ws = $("hyWs").value, k = $("hyK").value;
    $("hyWdVal").textContent = wd; $("hyWsVal").textContent = ws; $("hyKVal").textContent = k;
    try {
      var r = await fetch(window.IWConsole.API + "/learn/lab/hybrid?query=" + encodeURIComponent(q) + "&principal=" + encodeURIComponent(p) +
        "&w_dense=" + wd + "&w_sparse=" + ws + "&k=" + k, { headers: window.IWConsole.authHeaders({}) });
      if (!r.ok) throw new Error("HTTP " + r.status);
      var d = await r.json();
      if (mine === seq) render(d);
    } catch (e) {
      $("hyErr").hidden = false;
      $("hyErr").textContent = "Could not load the replay (" + e.message + "). Are you signed in?";
    }
  }
  function schedule() { clearTimeout(timer); timer = setTimeout(run, 120); }
  function group(sel, attr, set) {
    var btns = document.querySelectorAll(sel);
    btns.forEach(function (b) {
      b.addEventListener("click", function () {
        set(b.getAttribute(attr));
        btns.forEach(function (x) { x.setAttribute("aria-pressed", x === b ? "true" : "false"); });
        run();
      });
    });
    return btns[0].getAttribute(attr);
  }

  window.onConsoleReady = function () {
    q = group("#hyQuestions button", "data-q", function (v) { q = v; });
    p = group("#hyUsers button", "data-p", function (v) { p = v; });
    ["hyWd", "hyWs", "hyK"].forEach(function (id) { $(id).addEventListener("input", schedule); });
    document.querySelectorAll("#hyPresets button").forEach(function (b) {
      b.addEventListener("click", function () { $("hyWd").value = b.getAttribute("data-wd"); $("hyWs").value = b.getAttribute("data-ws"); run(); });
    });
    run();
  };
})();
</script>"""

__all__ = ["LAB_CSS", "LAB_JS", "lab_html"]
