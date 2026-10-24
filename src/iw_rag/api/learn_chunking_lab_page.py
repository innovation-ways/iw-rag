# ruff: noqa: E501
"""The chunking lab's markup, styles and script (rendered inside the chunking lesson).

A thin client: the chunking, the expansion rule and every verdict are computed
server-side by :mod:`iw_rag.learn.chunking_lab`; this module only draws them.  Every
server string goes through ``IWConsole.esc`` before it reaches the DOM.
"""

from __future__ import annotations

from html import escape

from iw_rag.learn.samples import SAMPLES

LAB_CSS = """
  .leaf-ctl { min-width: 280px; flex: 1; max-width: 460px; }
  .leaf-ctl input[type=range] { width: 100%; accent-color: var(--teal); }
  .leaf-ctl .val { font-size: 22px; font-weight: 700; color: var(--teal); }
  .lab-stats { display: grid; grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); gap: 10px; margin: 10px 0 14px; }
  .stat { border: 1px solid var(--line); border-radius: 10px; padding: 10px 12px; background: rgba(24,29,41,0.5); }
  .stat b { display: block; font-size: 20px; }
  .stat span { font-size: 12px; color: var(--muted); }
  .stat.warn b { color: var(--warn); }
  .lab-strip { display: flex; gap: 2px; height: 46px; margin: 6px 0 4px; }
  .lab-strip .seg { position: relative; min-width: 3px; background: var(--pc); opacity: .82; border-radius: 3px; cursor: pointer; transition: opacity .1s; }
  .lab-strip .seg:hover { opacity: 1; }
  .lab-strip .seg.cut::after { content: ""; position: absolute; right: -2px; top: -4px; bottom: -4px; width: 3px; background: var(--danger); border-radius: 2px; }
  .lab-strip .seg.hit-ok { outline: 2px solid var(--ok); outline-offset: 1px; opacity: 1; }
  .lab-strip .seg.hit-split { outline: 2px solid var(--warn); outline-offset: 1px; opacity: 1; }
  .strip-legend { font-size: 12px; color: var(--muted); margin-bottom: 12px; }
  .strip-legend i { display: inline-block; width: 3px; height: 12px; background: var(--danger); vertical-align: middle; margin: 0 4px 0 2px; }
  .probe { border: 1px solid var(--line); border-left-width: 4px; border-radius: 10px; padding: 12px 14px; margin-bottom: 10px; background: rgba(24,29,41,0.5); }
  .probe.ok { border-left-color: var(--ok); } .probe.bad { border-left-color: var(--warn); }
  .probe .q { font-weight: 600; margin-bottom: 4px; }
  .probe .s { color: var(--muted); font-style: italic; font-size: 13px; margin-bottom: 6px; }
  .probe .v { font-size: 14px; font-weight: 600; } .probe.ok .v { color: var(--ok); } .probe.bad .v { color: var(--warn); }
  .probe .d { font-size: 13px; color: var(--muted); margin-top: 3px; line-height: 1.5; }
  .parent-row { display: flex; align-items: center; gap: 10px; font-size: 13px; padding: 5px 0; flex-wrap: wrap; }
  .parent-row .dot { width: 12px; height: 12px; border-radius: 3px; background: var(--pc); flex: none; }
  .parent-row .tag { font-size: 11px; padding: 2px 8px; border-radius: 999px; border: 1px solid var(--line); color: var(--muted); }
  .parent-row .tag.bad { border-color: var(--warn); color: var(--warn); }
  .leaf-card { border: 1px solid var(--line); border-left: 4px solid var(--pc); border-radius: 8px; padding: 10px 12px; margin: 8px 0; font-size: 13px; background: rgba(24,29,41,0.5); scroll-margin-top: 90px; }
  .leaf-card .hd { color: var(--muted); font-size: 12px; margin-bottom: 4px; display: flex; gap: 10px; flex-wrap: wrap; }
  .leaf-card .hd .cutb { color: var(--danger); font-weight: 600; }
  .leaf-card .tx { white-space: pre-wrap; line-height: 1.5; }
  .lab-details summary { cursor: pointer; color: var(--teal); font-size: 14px; margin: 14px 0 4px; }
"""


def lab_html() -> str:
    """Return the chunking lab's markup, with the sample picker rendered server-side."""
    buttons = "".join(
        f'<button type="button" data-sample="{escape(s.sample_id)}" '
        f'aria-pressed="{"true" if i == 0 else "false"}" title="{escape(s.blurb)}">{escape(s.title)}</button>'
        for i, s in enumerate(SAMPLES)
    )
    blurbs = "".join(
        f'<span class="sample-blurb" data-for="{escape(s.sample_id)}"{"" if i == 0 else " hidden"}>{escape(s.blurb)}</span>'
        for i, s in enumerate(SAMPLES)
    )
    return (
        '<div class="lab" id="chunkLab">'
        '<h2 class="lab-title">Try it: the real chunker</h2>'
        '<p class="lab-sub">This runs IW-RAG\'s own chunker over a sample document. Drag the slider and watch '
        "where the cuts fall. The questions below ask for one sentence of the document: "
        "does the chunker keep it whole? Sizes are in characters, as for documents uploaded in this "
        "console (the offline corpus counts model tokens instead: 384 per leaf, roughly 1,500 characters).</p>"
        '<div class="lab-ctl">'
        f'<div class="grp"><span class="lbl">Document</span><div class="seg-btn" id="labSamples">{buttons}</div>'
        f'<div class="muted" style="margin-top:6px" id="labBlurb">{blurbs}</div></div>'
        '<div class="grp leaf-ctl"><span class="lbl">Leaf size</span>'
        '<div><span class="val" id="labLeafVal">500</span> <span class="muted">characters</span></div>'
        '<input type="range" id="labLeaf" min="50" max="2500" step="50" value="500" aria-label="Leaf size in characters">'
        '<div class="preset-row">'
        '<button type="button" class="btn-sm" data-leaf="150">150 · tiny</button>'
        '<button type="button" class="btn-sm" data-leaf="500">500</button>'
        '<button type="button" class="btn-sm" data-leaf="800">800</button>'
        '<button type="button" class="btn-sm" data-leaf="1600">1600 · upload default</button>'
        "</div></div></div>"
        '<div class="lab-stats" id="labStats"></div>'
        "<h3>The document, cut into leaves</h3>"
        '<div class="lab-strip" id="labStrip" role="img" aria-label="Leaves in document order, coloured by section"></div>'
        '<div class="strip-legend">One block per leaf, width = size, colour = section (parent). '
        "<i></i>= a leaf that ends mid-sentence. Click a block to jump to its text.</div>"
        '<h3>Does the answer survive?</h3><div id="labProbes"></div>'
        '<h3>Sections (parents): what the model reads</h3><div id="labParents"></div>'
        '<details class="lab-details"><summary>Show every leaf\'s text</summary><div id="labLeaves"></div></details>'
        '<div class="lab-err" id="labErr" hidden></div>'
        "</div>"
    )


LAB_JS = """<script>
(function () {
  var sample = null, timer = null, seq = 0;
  function $(id) { return document.getElementById(id); }
  function fmt(n) { return Number(n).toLocaleString("en-US"); }
  function color(i) { return "hsl(" + ((i * 57 + 170) % 360) + " 55% 52%)"; }
  function esc(s) { return window.IWConsole.esc(s); }

  function render(d) {
    var leaves = d.leaves, parents = d.parents;
    $("labErr").hidden = true;
    $("labStats").innerHTML =
      stat(leaves.length, "leaves") + stat(fmt(d.avg_leaf_chars), "average leaf, characters") +
      stat(parents.length, parents.length === 1 ? "section (parent)" : "sections (parents)") +
      '<div class="stat' + (d.mid_sentence_cuts ? " warn" : "") + '"><b>' + d.mid_sentence_cuts + "</b><span>leaves ending mid-sentence</span></div>" +
      '<div class="stat' + (d.oversized_parents ? " warn" : "") + '"><b>' + d.oversized_parents + "</b><span>sections over the " + fmt(d.parent_budget_chars) + "-character budget</span></div>";

    var hit = {};
    d.probes.forEach(function (p) { p.leaf_indexes.forEach(function (i) { hit[i] = p.status === "intact" ? "hit-ok" : "hit-split"; }); });
    $("labStrip").innerHTML = leaves.map(function (l) {
      return '<div class="seg' + (l.ends_mid_sentence ? " cut" : "") + (hit[l.index] ? " " + hit[l.index] : "") +
        '" style="flex:' + l.chars + ';--pc:' + color(l.parent_index) + '" data-i="' + l.index +
        '" title="Leaf ' + (l.index + 1) + " · " + esc(l.section) + " · " + l.chars + ' characters"></div>';
    }).join("");

    $("labProbes").innerHTML = d.probes.map(function (p) {
      var ok = p.status === "intact";
      var mark = ok ? "✓ " : (p.status === "bare" ? "⚠ " : "✗ ");
      return '<div class="probe ' + (ok ? "ok" : "bad") + '"><div class="q">' + esc(p.question) + '</div><div class="s">“' +
        esc(p.sentence) + '”</div><div class="v">' + mark + esc(p.verdict) + '</div><div class="d">' + esc(p.detail) + "</div></div>";
    }).join("");

    $("labParents").innerHTML = parents.map(function (p) {
      return '<div class="parent-row" style="--pc:' + color(p.index) + '"><span class="dot"></span><b>' + esc(p.section) +
        "</b><span class=\\"muted\\">" + p.leaf_count + (p.leaf_count === 1 ? " leaf" : " leaves") + " · " + fmt(p.chars) + " characters</span>" +
        (p.expanded ? '<span class="tag">model reads the whole section</span>'
                    : '<span class="tag bad">too large: model reads only the matched leaf</span>') + "</div>";
    }).join("");

    $("labLeaves").innerHTML = leaves.map(function (l) {
      return '<div class="leaf-card" id="leaf-' + l.index + '" style="--pc:' + color(l.parent_index) + '"><div class="hd"><span>Leaf ' +
        (l.index + 1) + "</span><span>" + esc(l.section) + "</span><span>" + l.chars + " characters</span>" +
        (l.ends_mid_sentence ? '<span class="cutb">ends mid-sentence</span>' : "") + '</div><div class="tx">' + esc(l.text) + "</div></div>";
    }).join("");
  }
  function stat(v, label) { return '<div class="stat"><b>' + v + "</b><span>" + label + "</span></div>"; }

  async function run() {
    var mine = ++seq, leaf = $("labLeaf").value;
    $("labLeafVal").textContent = leaf;
    try {
      var r = await fetch(window.IWConsole.API + "/learn/lab/chunking?sample=" + encodeURIComponent(sample) + "&leaf=" + encodeURIComponent(leaf),
                          { headers: window.IWConsole.authHeaders({}) });
      if (!r.ok) throw new Error("HTTP " + r.status);
      var d = await r.json();
      if (mine === seq) render(d);
    } catch (e) {
      $("labErr").hidden = false;
      $("labErr").textContent = "Could not run the chunker (" + e.message + "). Are you signed in?";
    }
  }
  function schedule() { clearTimeout(timer); timer = setTimeout(run, 120); }

  window.onConsoleReady = function () {
    var btns = document.querySelectorAll("#labSamples button");
    sample = btns[0].getAttribute("data-sample");
    btns.forEach(function (b) {
      b.addEventListener("click", function () {
        sample = b.getAttribute("data-sample");
        btns.forEach(function (x) { x.setAttribute("aria-pressed", x === b ? "true" : "false"); });
        document.querySelectorAll(".sample-blurb").forEach(function (s) { s.hidden = s.getAttribute("data-for") !== sample; });
        run();
      });
    });
    $("labLeaf").addEventListener("input", schedule);
    document.querySelectorAll(".preset-row button").forEach(function (b) {
      b.addEventListener("click", function () { $("labLeaf").value = b.getAttribute("data-leaf"); run(); });
    });
    $("labStrip").addEventListener("click", function (e) {
      var seg = e.target.closest(".seg"); if (!seg) return;
      var det = document.querySelector(".lab-details"); det.open = true;
      var card = $("leaf-" + seg.getAttribute("data-i"));
      if (card) card.scrollIntoView({ behavior: "smooth", block: "center" });
    });
    run();
  };
})();
</script>"""

__all__ = ["LAB_CSS", "LAB_JS", "lab_html"]
