# ruff: noqa: E501
"""The guards lab's markup, styles and script (rendered inside the fail-closed lesson).

A thin client over ``GET /learn/lab/guards``: every step, verdict and counter comes from
:mod:`iw_rag.learn.guards_lab`, which replays a recorded run of the production graph.
Every server string goes through ``IWConsole.esc`` before it reaches the DOM.
"""

from __future__ import annotations

from html import escape

from iw_rag.learn.guards_lab import SOURCE_LABELS, summaries

LAB_CSS = """
  .gd-legend { display: flex; flex-wrap: wrap; gap: 8px 16px; margin: 0 0 14px; font-size: 12.5px; color: var(--muted); }
  .gd-legend span { display: inline-flex; align-items: center; gap: 6px; }
  .gd-dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
  .gd-dot.recorded { background: var(--teal); } .gd-dot.scripted { background: var(--warn); } .gd-dot.logic { background: #8b93ad; }
  .gd-cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(210px, 1fr)); gap: 10px; margin-bottom: 14px; }
  .gd-card { text-align: left; border: 1px solid var(--line); border-radius: 10px; padding: 10px 12px; background: var(--ink-2); display: block; }
  .gd-card[aria-pressed="true"] { border-color: var(--teal); box-shadow: 0 0 0 1px var(--teal) inset; }
  .gd-card b { display: block; font-size: 13.5px; margin-bottom: 6px; }
  .gd-badge { font-size: 11px; padding: 2px 8px; border-radius: 999px; border: 1px solid var(--line); margin-right: 6px; }
  .gd-badge.answered { color: var(--ok); border-color: var(--ok); } .gd-badge.refused { color: var(--warn); border-color: var(--warn); }
  .gd-card small { color: var(--muted); font-size: 11.5px; }
  .gd-story { border: 1px solid var(--line); border-left: 4px solid var(--teal); border-radius: 10px; padding: 12px 16px; margin-bottom: 12px; background: rgba(24,29,41,0.5); }
  .gd-story .q { font-weight: 600; } .gd-story .s { color: var(--muted); font-size: 14px; margin-top: 4px; line-height: 1.5; }
  .gd-ctl { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; margin-bottom: 12px; }
  .gd-ctl .prog { color: var(--muted); font-size: 13px; margin-left: 8px; }
  .gd-path { display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 12px; }
  .gd-chip { font-size: 11px; padding: 2px 8px; border-radius: 999px; border: 1px solid var(--line); color: var(--muted); }
  .gd-chip.done { color: var(--text); border-color: #5b6382; } .gd-chip.cur { border-color: var(--teal); color: var(--teal); font-weight: 600; }
  .gd-layout { display: grid; grid-template-columns: minmax(0, 1fr) 250px; gap: 14px; align-items: start; }
  @media (max-width: 900px) { .gd-layout { grid-template-columns: 1fr; } }
  .gd-step { border: 1px solid var(--line); border-left: 4px solid #8b93ad; border-radius: 10px; padding: 10px 14px; margin-bottom: 8px; background: rgba(24,29,41,0.5); cursor: pointer; }
  .gd-step.pass { border-left-color: var(--ok); } .gd-step.stop { border-left-color: var(--danger); } .gd-step.retry { border-left-color: var(--warn); }
  .gd-step .hd { display: flex; gap: 8px; align-items: center; font-size: 12px; color: var(--muted); }
  .gd-step .hd b { color: var(--text); font-size: 13.5px; }
  .gd-step .hl { font-size: 14px; margin-top: 4px; line-height: 1.5; }
  .gd-step .dt { display: none; font-size: 13px; color: var(--muted); margin-top: 6px; line-height: 1.55; }
  .gd-step.open .dt { display: block; }
  .gd-step ul { margin: 6px 0 0; padding-left: 18px; font-size: 13px; }
  .gd-step .mk { margin-top: 6px; font-family: ui-monospace, monospace; font-size: 12px; background: var(--ink-2); border: 1px solid var(--line); border-radius: 6px; padding: 6px 8px; word-break: break-word; }
  .gd-step .src { margin-left: auto; font-size: 11px; display: inline-flex; align-items: center; gap: 5px; }
  .gd-side { border: 1px solid var(--line); border-radius: 10px; padding: 12px 14px; background: rgba(24,29,41,0.5); font-size: 13px; position: sticky; top: 90px; }
  .gd-side h4 { margin: 0 0 8px; font-size: 12px; text-transform: uppercase; letter-spacing: .08em; color: var(--muted); }
  .gd-side .row { margin-bottom: 10px; line-height: 1.45; } .gd-side .row b { display: block; color: var(--muted); font-size: 11.5px; font-weight: 500; }
  .gd-pips { display: flex; gap: 6px; margin-top: 4px; } .gd-pip { width: 22px; height: 10px; border-radius: 3px; background: var(--ink-3); border: 1px solid var(--line); }
  .gd-pip.used { background: var(--warn); border-color: var(--warn); }
  .gd-out { border-radius: 10px; padding: 14px 16px; margin-top: 10px; border: 1px solid var(--line); line-height: 1.55; }
  .gd-out.answered { border-color: var(--ok); background: rgba(45,212,191,0.07); } .gd-out.refused { border-color: var(--warn); background: rgba(245,158,11,0.07); }
  .gd-out b { display: block; margin-bottom: 4px; }
  .gd-foot { color: var(--muted); font-size: 12px; line-height: 1.55; margin-top: 12px; }
"""


def lab_html() -> str:
    """Return the guards lab's markup, with the scenario cards rendered server-side."""
    cards = "".join(
        f'<button type="button" class="gd-card" data-s="{escape(s.scenario_id)}" aria-pressed="{"true" if i == 0 else "false"}">'
        f"<b>{escape(s.title)}</b>"
        f'<span class="gd-badge {escape(s.disposition)}">{escape(s.disposition)}</span><small>decided by: {escape(s.decided_by)}</small></button>'
        for i, s in enumerate(summaries())
    )
    legend = "".join(
        f'<span><i class="gd-dot {escape(key)}"></i>{escape(label)}</span>'
        for key, label in SOURCE_LABELS.items()
    )
    return (
        '<div class="lab" id="guardsLab">'
        '<h2 class="lab-title">Try it: walk a question through the checkpoints</h2>'
        '<p class="lab-sub">Pick a scenario and step through it. This replays a real run of IW-RAG\'s own graph with the guards '
        "switched on: the routing, the retry budget, the sanitising, the citation checks and the refusal are the production code. "
        "Prompt Guard 2 scores come from the live classifier. The language-model steps (router, grader, writer, self-check) are "
        "scripted, so each scenario can force a failure on demand. The colour of each step tells you which is which.</p>"
        f'<div class="gd-legend">{legend}</div>'
        f'<div class="gd-cards" id="gdCards">{cards}</div>'
        '<div class="gd-story" id="gdStory"></div>'
        '<div class="gd-ctl"><button type="button" id="gdBack" class="btn-sm">◀ Back</button>'
        '<button type="button" id="gdNext" class="btn-sm btn-primary">Next step ▶</button>'
        '<button type="button" id="gdPlay" class="btn-sm">Play all</button>'
        '<button type="button" id="gdReset" class="btn-sm">Reset</button><span class="prog" id="gdProg"></span></div>'
        '<div class="gd-path" id="gdPath"></div>'
        '<div class="gd-layout"><div id="gdTimeline"></div>'
        '<div class="gd-side" id="gdSide"></div></div>'
        '<div id="gdOutcome"></div>'
        '<div class="gd-foot" id="gdFoot"></div>'
        '<div class="lab-err" id="gdErr" hidden></div>'
        "</div>"
    )


LAB_JS = """<script>
(function () {
  var data = null, cur = -1, scenario = null, timer = null, seq = 0;
  function $(id) { return document.getElementById(id); }
  function esc(s) { return window.IWConsole.esc(s); }

  function stepHtml(s, isCur) {
    return '<div class="gd-step ' + s.outcome + (isCur ? " open" : "") + '" data-i="' + s.index + '"><div class="hd"><b>' + esc(s.title) +
      '</b><span class="src"><i class="gd-dot ' + s.source + '"></i>' + esc(data.meta.source_labels[s.source]) + '</span></div><div class="hl">' + esc(s.headline) +
      '</div><div class="dt">' + esc(s.detail) +
      (s.items.length ? "<ul>" + s.items.map(function (i) { return "<li>" + esc(i) + "</li>"; }).join("") + "</ul>" : "") +
      (s.marked_sample ? '<div class="mk">' + esc(s.marked_sample) + " …</div>" : "") + "</div></div>";
  }
  function side() {
    if (cur < 0) { $("gdSide").innerHTML = "<h4>State</h4><div class='row'>Press <b style='display:inline;color:inherit'>Next step</b> to begin.</div>"; return; }
    var st = data.steps[cur].state, pips = "";
    for (var i = 0; i < data.max_retries; i++) pips += '<span class="gd-pip' + (i < st.retries ? " used" : "") + '"></span>';
    $("gdSide").innerHTML = "<h4>State after this step</h4>" +
      '<div class="row"><b>Retries used</b>' + st.retries + " of " + data.max_retries + '<div class="gd-pips">' + pips + "</div></div>" +
      '<div class="row"><b>Question now</b>' + esc(st.question) + "</div>" +
      '<div class="row"><b>Evidence kept</b>' + (st.kept.length ? esc(st.kept.join(", ")) : "none") + "</div>" +
      (st.dropped.length ? '<div class="row"><b>Dropped by sanitising</b>' + esc(st.dropped.join(", ")) + "</div>" : "") +
      '<div class="row"><b>Evidence verdict</b>' + esc(st.doc_verdict || "not yet") + "</div>" +
      '<div class="row"><b>Draft verdict</b>' + esc(st.gen_verdict || "not yet") + "</div>" +
      '<div class="row"><b>Outcome</b>' + esc(st.disposition || "in progress") + "</div>";
  }
  function paint() {
    var steps = data.steps;
    $("gdTimeline").innerHTML = steps.slice(0, cur + 1).map(function (s, i) { return stepHtml(s, i === cur); }).join("") ||
      '<div class="muted">Nothing has happened yet. The question has not even been checked.</div>';
    $("gdPath").innerHTML = steps.map(function (s, i) {
      return '<span class="gd-chip' + (i < cur ? " done" : (i === cur ? " cur" : "")) + '">' + esc(s.title) + "</span>";
    }).join("");
    $("gdProg").textContent = cur < 0 ? steps.length + " steps in this run" : "Step " + (cur + 1) + " of " + steps.length;
    $("gdBack").disabled = cur < 0; $("gdNext").disabled = cur >= steps.length - 1;
    side();
    var done = cur === steps.length - 1, f = data.final;
    $("gdOutcome").innerHTML = !done ? "" : (f.disposition === "answered"
      ? '<div class="gd-out answered"><b>The user gets an answer</b>' + esc(f.answer) + "</div>"
      : '<div class="gd-out refused"><b>The user gets an honest refusal</b>' + esc(f.refusal) + "</div>");
    var last = $("gdTimeline").lastElementChild; if (last && cur >= 0) last.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }
  function stop() { clearInterval(timer); timer = null; $("gdPlay").textContent = "Play all"; }
  function advance() { if (cur < data.steps.length - 1) { cur++; paint(); } else stop(); }

  async function load(id) {
    var mine = ++seq; stop();
    try {
      var r = await fetch(window.IWConsole.API + "/learn/lab/guards?scenario=" + encodeURIComponent(id), { headers: window.IWConsole.authHeaders({}) });
      if (!r.ok) throw new Error("HTTP " + r.status);
      var d = await r.json();
      if (mine !== seq) return;
      data = d; cur = -1; $("gdErr").hidden = true;
      $("gdStory").innerHTML = '<div class="q">' + esc(d.scenario.question) + '</div><div class="s">' + esc(d.scenario.story) + "</div>";
      $("gdFoot").textContent = "Recorded " + d.meta.recorded + ". Real: " + d.meta.provenance.real.join("; ") + ". Scripted: " + d.meta.provenance.scripted.join("; ") +
        ". Not run in this recording: " + d.meta.provenance.not_run.join("; ") + ".";
      paint();
    } catch (e) {
      $("gdErr").hidden = false; $("gdErr").textContent = "Could not load the scenario (" + e.message + "). Are you signed in?";
    }
  }

  window.onConsoleReady = function () {
    var cards = document.querySelectorAll("#gdCards .gd-card");
    cards.forEach(function (c) {
      c.addEventListener("click", function () {
        cards.forEach(function (x) { x.setAttribute("aria-pressed", x === c ? "true" : "false"); });
        load(c.getAttribute("data-s"));
      });
    });
    $("gdNext").addEventListener("click", function () { stop(); advance(); });
    $("gdBack").addEventListener("click", function () { stop(); if (cur >= 0) { cur--; paint(); } });
    $("gdReset").addEventListener("click", function () { stop(); cur = -1; paint(); });
    $("gdPlay").addEventListener("click", function () {
      if (timer) { stop(); return; }
      if (cur >= data.steps.length - 1) cur = -1;
      $("gdPlay").textContent = "Pause"; advance(); timer = setInterval(advance, 1100);
    });
    $("gdTimeline").addEventListener("click", function (e) {
      var s = e.target.closest(".gd-step"); if (s) s.classList.toggle("open");
    });
    load(cards[0].getAttribute("data-s"));
  };
})();
</script>"""

__all__ = ["LAB_CSS", "LAB_JS", "lab_html"]
