# ruff: noqa: E501
"""Self-contained, branded Benchmark Studio web page (F-00095, S03).

Chainlit is chat-first with no multi-page routing or data tables, so the
Benchmark Studio is a standalone page served by the API at
``GET /benchmark/ui`` (reached through Caddy at ``/api/benchmark/ui``).  It
is a single dependency-free HTML document: it logs in via ``/auth/login``,
lets an ``eval-operator`` pick per-step models (generator / embedder /
reranker / grader) from the F-00094 ``GET /models`` catalog, pick a golden
dataset + run params, click **Run Full Test** to stream progress over SSE
from ``POST /benchmark/run``, then renders the persisted report (metric
tiers + latency + VRAM + judge κ) and supports A/B comparing two reports
via ``GET /benchmark/compare``.  It carries the Innovation Ways visual
identity (teal ``#0D9488``/``#2DD4BF`` on ink ``#181D29``).

Kept inline (not a static asset) so it is always packaged with the wheel
and needs no extra MIME/static-mount wiring.  The page derives its API
base from its own URL so it works behind Caddy (``/api``) or when hit
directly (``/``).  No external CDN / Google Fonts link (Invariant 7,
local-only rule).

Mirrors :mod:`iw_rag.api.documents_page` — same header/footer chrome,
same login flow, same teal/ink palette, same sessionStorage-only bearer
token.  The S02 route ``GET /benchmark/ui`` is wired against
:func:`benchmark_page_html`.
"""

from __future__ import annotations

from iw_rag.api.benchmark_render import REPORT_CSS, REPORT_JS, TOOLTIP_CSS
from iw_rag.api.console_shell import add_favicon

_PAGE = r"""<!doctype html>
<html lang="en" data-theme="dark">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>IW - RAG</title>
<style>
  :root {
    --teal: #2DD4BF; --teal-deep: #0D9488; --ink: #181D29; --ink-2: #222631;
    --ink-3: #2c3142; --line: #353b4f; --text: #E8EAF0; --muted: #9aa3b8;
    --danger: #f43f5e; --ok: #2DD4BF; --warn: #f59e0b; --radius: 12px;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: radial-gradient(1200px 600px at 70% -10%, #1d2740 0%, var(--ink) 60%);
    color: var(--text); font-family: "Inter", system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
    min-height: 100vh;
  }
  header {
    display: flex; align-items: center; gap: 14px; padding: 16px 28px;
    border-bottom: 1px solid var(--line); background: rgba(24,29,41,0.7); backdrop-filter: blur(6px);
    position: sticky; top: 0; z-index: 5;
  }
  .logo { font-family: "Space Grotesk", system-ui, sans-serif; font-weight: 700; font-size: 20px; letter-spacing: .2px; }
  .logo .i { color: var(--teal); } .logo .w { color: var(--teal); }
  .spacer { flex: 1; }
  .who { color: var(--muted); font-size: 14px; }
  a.btn, button {
    font: inherit; cursor: pointer; border-radius: 10px; border: 1px solid var(--line);
    background: var(--ink-3); color: var(--text); padding: 9px 14px; text-decoration: none;
    transition: .15s ease; display: inline-flex; align-items: center; gap: 8px;
  }
  a.btn:hover, button:hover { border-color: var(--teal-deep); }
  .btn-primary { background: linear-gradient(180deg, var(--teal) 0%, var(--teal-deep) 100%); color: #06231f; border: none; font-weight: 600; }
  .btn-primary:disabled { opacity: .5; cursor: not-allowed; }
  .btn-ghost { background: transparent; }
  .report-row { display: flex; align-items: center; justify-content: space-between; gap: 12px; padding: 6px 0; border-bottom: 1px solid var(--ink-2); }
  .report-row:last-child { border-bottom: none; }
  .btn-danger { background: transparent; color: var(--danger); border-color: #5b2230; }
  .btn-danger:hover { background: #2a1620; border-color: var(--danger); }
  main { max-width: 1080px; margin: 0 auto; padding: 28px 20px 80px; }
  .card { background: rgba(34,38,49,0.72); border: 1px solid var(--line); border-radius: var(--radius); padding: 22px; margin-bottom: 22px; }
  h1 { font-family: "Space Grotesk", system-ui, sans-serif; font-size: 22px; margin: 0 0 4px; }
  h2 { font-family: "Space Grotesk", system-ui, sans-serif; font-size: 16px; margin: 0 0 14px; color: var(--text); }
  .sub { color: var(--muted); font-size: 14px; margin: 0 0 18px; }
  label { display: block; font-size: 13px; color: var(--muted); margin: 12px 0 6px; }
  input[type=text], input[type=password], input[type=number], select {
    width: 100%; padding: 10px 12px; border-radius: 10px; border: 1px solid var(--line);
    background: var(--ink-2); color: var(--text); font: inherit;
  }
  input:disabled, select:disabled, button:disabled { opacity: .6; cursor: not-allowed; }
  .picker-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  @media (max-width: 720px) { .picker-grid { grid-template-columns: 1fr; } }
  .row { display: flex; gap: 16px; flex-wrap: wrap; align-items: flex-end; }
  .row > div { flex: 1; min-width: 180px; }
  table { width: 100%; border-collapse: collapse; }
  th, td { text-align: left; padding: 10px 8px; border-bottom: 1px solid var(--line); font-size: 13px; vertical-align: top; }
  th { color: var(--muted); font-weight: 600; font-size: 11px; text-transform: uppercase; letter-spacing: .5px; }
  td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
  .chip { font-size: 11px; padding: 2px 8px; border-radius: 999px; border: 1px solid var(--line); white-space: nowrap; }
  .chip.ok { color: var(--ok); border-color: var(--teal-deep); }
  .chip.bad { color: var(--danger); border-color: #5b2230; }
  .chip.warn { color: var(--warn); border-color: #5b4322; }
  .chip.muted { color: var(--muted); }
  .toast { position: fixed; bottom: 22px; left: 50%; transform: translateX(-50%); background: var(--ink-3); border: 1px solid var(--line); padding: 12px 18px; border-radius: 10px; opacity: 0; transition: .25s; pointer-events: none; max-width: 90vw; z-index: 10; }
  .toast.show { opacity: 1; }
  .toast.err { border-color: var(--danger); color: #ffd7de; }
  .hidden { display: none !important; }
  .muted { color: var(--muted); font-size: 13px; }
  .loginwrap { max-width: 420px; margin: 8vh auto; }
  .catalog-warn { background: rgba(245,158,11,0.08); border: 1px solid #5b4322; color: #fde7c0; padding: 10px 12px; border-radius: 10px; font-size: 13px; margin: 6px 0 0; }
  .progress { display: flex; gap: 8px; flex-wrap: wrap; margin: 12px 0; }
  .stage { font-size: 12px; padding: 4px 10px; border-radius: 999px; border: 1px solid var(--line); color: var(--muted); background: var(--ink-2); transition: .15s; }
  .stage.active { color: #06231f; background: linear-gradient(180deg, var(--teal) 0%, var(--teal-deep) 100%); border-color: transparent; }
  .stage.done { color: var(--ok); border-color: var(--teal-deep); }
  .stage.error { color: var(--danger); border-color: #5b2230; background: #2a1620; }
  .progress-detail { font-size: 13px; color: var(--muted); min-height: 18px; margin-top: 6px; font-family: "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
  .spin { display: inline-block; width: 12px; height: 12px; border: 2px solid var(--line); border-top-color: var(--teal); border-radius: 50%; animation: spin .7s linear infinite; }
  @keyframes spin { to { transform: rotate(360deg); } }
  .metric-tier { margin: 14px 0; }
  .metric-tier h3 { font-family: "Space Grotesk", system-ui, sans-serif; font-size: 13px; margin: 0 0 8px; color: var(--muted); text-transform: uppercase; letter-spacing: .6px; }
  .latency-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; }
  .latency-grid > div { background: var(--ink-2); border: 1px solid var(--line); border-radius: 10px; padding: 10px 12px; }
  .latency-grid .v { font-size: 18px; font-weight: 600; font-family: "JetBrains Mono", ui-monospace, monospace; color: var(--teal); }
  .latency-grid .l { font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: .4px; }
  .compare-winner { display: inline-block; padding: 4px 10px; border-radius: 999px; font-size: 13px; font-weight: 600; }
  .compare-winner.A { background: linear-gradient(180deg, var(--teal) 0%, var(--teal-deep) 100%); color: #06231f; }
  .compare-winner.B { background: linear-gradient(180deg, #a78bfa 0%, #6d4dde 100%); color: #1a103a; }
  .compare-winner.tie { background: var(--ink-3); color: var(--muted); border: 1px solid var(--line); }
  .caveat-box { background: rgba(245,158,11,0.10); border: 1px solid #5b4322; color: #fde7c0; padding: 12px 14px; border-radius: 10px; margin: 12px 0; font-size: 13px; line-height: 1.5; }
  .caveat-box strong { color: var(--warn); }
  details { margin: 8px 0; }
  details summary { cursor: pointer; color: var(--muted); font-size: 13px; padding: 6px 0; }
  .compare-reasoning { font-size: 13px; color: var(--muted); line-height: 1.5; padding: 10px 12px; background: var(--ink-2); border: 1px solid var(--line); border-radius: 10px; margin: 10px 0; }
/* __IW_REPORT_CSS__ */
</style>
</head>
<body>
<header>
  <span class="logo"><span class="i">I</span>nnovation <span class="w">W</span>ays</span>
  <span class="muted">· Benchmark Studio</span>
  <span class="spacer"></span>
  <a class="btn btn-ghost" id="chatLink" href="/">&#128172; Chat</a>
  <a class="btn btn-ghost" id="docsLink" href="/documents/ui">&#128196; Documents</a>
  <span class="who" id="who"></span>
  <button class="btn-ghost hidden" id="logout">Sign out</button>
</header>

<main>
  <section id="loginView" class="hidden">
    <div class="card loginwrap">
      <h1>Sign in</h1>
      <p class="sub">Use your demo credentials. Benchmark Studio is gated to the <strong>eval-operator</strong> group.</p>
      <label for="u">Username</label>
      <input type="text" id="u" autocomplete="username" />
      <label for="p">Password</label>
      <input type="password" id="p" autocomplete="current-password" />
      <div style="margin-top:18px"><button class="btn-primary" id="loginBtn">Sign in</button></div>
    </div>
  </section>

  <section id="appView" class="hidden">

    <div class="card" id="runCard">
      <h2>Run configuration</h2>
      <p class="sub">Pick the per-step models, the dataset, and the run params. The generator is cold-swapped via F-00094; the run streams progress over SSE from <code>POST /benchmark/run</code>.</p>

      <div id="catalogWarn" class="catalog-warn hidden">Model catalog unavailable — populate the pickers manually or fix <code>GET /models</code>.</div>

      <div class="picker-grid">
        <div>
          <label for="pickGenerator">Generator</label>
          <select id="pickGenerator"><option value="">(loading…)</option></select>
        </div>
        <div>
          <label for="pickEmbedder">Embedder</label>
          <select id="pickEmbedder"><option value="">(loading…)</option></select>
        </div>
        <div>
          <label for="pickReranker">Reranker</label>
          <select id="pickReranker"><option value="">(loading…)</option></select>
        </div>
        <div>
          <label for="pickGrader">Grader (judge)</label>
          <select id="pickGrader"><option value="">(loading…)</option></select>
        </div>
      </div>

      <div class="row" style="margin-top: 14px;">
        <div>
          <label for="pickDataset">Dataset</label>
          <select id="pickDataset">
            <option value="benchmark_golden" selected>benchmark_golden (default)</option>
          </select>
        </div>
        <div>
          <label for="paramTopK">top_k</label>
          <input type="number" id="paramTopK" min="1" max="20" value="5" />
        </div>
        <div>
          <label for="paramTemp">temperature</label>
          <input type="number" id="paramTemp" min="0" max="2" step="0.05" value="0.0" />
        </div>
        <div>
          <label for="paramMaxTokens">max_tokens</label>
          <input type="number" id="paramMaxTokens" min="64" max="4096" value="512" />
        </div>
      </div>

      <div style="margin-top: 18px; display: flex; gap: 12px; align-items: center;">
        <button class="btn-primary" id="runBtn">Run Full Test</button>
        <span class="muted" id="runHint">Live stream — your bearer token is used; the run is gated to the <code>eval-operator</code> group server-side.</span>
      </div>
    </div>

    <div class="card" id="progressCard">
      <h2>Live progress</h2>
      <p class="sub">SSE stream from <code>POST /benchmark/run</code>. Stages: <code>queued → swapping → ready → running (query i / N) → scoring → done</code>.</p>
      <div class="progress" id="progressPanel">
        <span class="stage" data-stage="queued">Queued</span>
        <span class="stage" data-stage="swapping">Swapping</span>
        <span class="stage" data-stage="ready">Ready</span>
        <span class="stage" data-stage="running">Running</span>
        <span class="stage" data-stage="scoring">Scoring</span>
        <span class="stage" data-stage="done">Done</span>
        <span class="stage" data-stage="error">Error</span>
      </div>
      <div class="progress-detail" id="progressDetail">No run yet.</div>
    </div>

    <div class="card" id="reportsCard">
      <h2>Saved reports</h2>
      <p class="sub">Persisted benchmark reports, newest first. <strong>View</strong> opens the metric tiers below; the A/B compare pairs any two on the same dataset.</p>
      <div id="reportsList"><p class="muted">Loading&hellip;</p></div>
    </div>

    <div class="card" id="reportCard" hidden>
      <h2>Report <span class="muted" id="reportId"></span></h2>
      <p class="sub" id="reportSub"></p>
      <div id="reportBody"></div>
    </div>

    <div class="card" id="compareCard">
      <h2>A/B compare</h2>
      <p class="sub">Pick two persisted reports and inspect the per-metric delta table. The <strong>winner</strong> flag is set only when one config is better on <strong>both nDCG@5 AND faithfulness</strong>; the n&asymp;37 underpowered caveat is always surfaced.</p>
      <div class="row">
        <div>
          <label for="compareA">Report A</label>
          <select id="compareA"><option value="">(loading…)</option></select>
        </div>
        <div>
          <label for="compareB">Report B</label>
          <select id="compareB"><option value="">(loading…)</option></select>
        </div>
        <div>
          <button class="btn-primary" id="compareBtn">Compare</button>
        </div>
      </div>
      <div id="compareBody"></div>
    </div>
  </section>
</main>

<div class="toast" id="toast"></div>

<script>
(function () {
  // API base: strip "/benchmark/ui" from our own path → "/api" behind Caddy, or "" direct.
  var API = location.pathname.replace(/\/benchmark\/ui\/?$/, "");
  var tokenKey = "iw_token", whoKey = "iw_principal", groupsKey = "iw_groups";
  var $ = function (id) { return document.getElementById(id); };
  var esc = function (s) { var d = document.createElement("div"); d.textContent = s == null ? "" : s; return d.innerHTML; };
  var fmt = function (n, d) { if (n == null || isNaN(n)) return "&mdash;"; if (d == null) d = 3; return Number(n).toFixed(d); };

  function token() { return sessionStorage.getItem(tokenKey); }
  function groups() {
    try { return JSON.parse(sessionStorage.getItem(groupsKey) || "[]"); }
    catch (e) { return []; }
  }
  function isOperator() { return (groups() || []).indexOf("eval-operator") !== -1; }
  function setToken(t, who, grp) {
    if (t) {
      sessionStorage.setItem(tokenKey, t);
      sessionStorage.setItem(whoKey, who || "");
      sessionStorage.setItem(groupsKey, JSON.stringify(grp || []));
    } else {
      sessionStorage.removeItem(tokenKey);
      sessionStorage.removeItem(whoKey);
      sessionStorage.removeItem(groupsKey);
    }
  }
  function authHeaders(extra) {
    var h = extra || {}; var tk = token(); if (tk) h["Authorization"] = "Bearer " + tk; return h;
  }
  function toast(msg, isErr) {
    var t = $("toast"); t.textContent = msg; t.className = "toast show" + (isErr ? " err" : "");
    setTimeout(function () { t.className = "toast"; }, 3500);
  }

  /* __IW_REPORT_JS__ */

  function showLogin() {
    $("loginView").classList.remove("hidden");
    $("appView").classList.add("hidden");
    $("logout").classList.add("hidden");
    $("who").textContent = "";
  }
  function showApp(who, grp) {
    $("loginView").classList.add("hidden");
    $("appView").classList.remove("hidden");
    $("logout").classList.remove("hidden");
    var tag = who ? ("Signed in as " + who) : "";
    if (grp && grp.length) tag += " · " + grp.join(", ");
    $("who").textContent = tag;
    // UX-only gate: hide Run for non-eval-operators. The API is the
    // authoritative gate; the UI just keeps the operator honest.
    if (!isOperator()) {
      $("runBtn").disabled = true;
      $("runBtn").title = "eval-operator group required (server-enforced)";
      $("runHint").textContent = "Your account is not in the eval-operator group — Run Full Test is disabled (the API will 403 anyway).";
    } else {
      $("runBtn").disabled = false;
      $("runBtn").title = "";
      $("runHint").textContent = "Live stream — your bearer token is used; the run is gated to the eval-operator group server-side.";
    }
    loadModels();
    loadReports();
  }

  async function login() {
    var u = $("u").value.trim(), p = $("p").value;
    if (!u || !p) { toast("Enter username and password", true); return; }
    try {
      var r = await fetch(API + "/auth/login", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username: u, password: p })
      });
      if (!r.ok) { toast("Invalid username or password", true); return; }
      var d = await r.json();
      setToken(d.token, d.principal_id, d.groups || []);
      $("p").value = "";
      showApp(d.principal_id, d.groups || []);
    } catch (e) { toast("Login failed: " + e, true); }
  }

  // -------------------------------------------------------------------------
  // Models catalog (F-00094 GET /models) — populates the per-step pickers.
  // Degrades gracefully: any 4xx/5xx/timeout leaves the pickers in a
  // "catalog unavailable" state but the page keeps loading.
  // -------------------------------------------------------------------------
  async function loadModels() {
    var pickers = ["pickGenerator", "pickEmbedder", "pickReranker", "pickGrader"];
    try {
      var r = await fetch(API + "/models", { headers: authHeaders() });
      if (r.status === 401) { setToken(null); showLogin(); return; }
      if (!r.ok) throw new Error("status " + r.status);
      var d = await r.json();
      var rows = (d && d.models) || [];
      var active = d && d.active_model_id;
      // Group rows by inferred role from the served_name / id when the catalog
      // doesn't carry an explicit role; fall back to a single bucket so the
      // pickers are still populated.
      var byRole = { generator: [], embedder: [], reranker: [], grader: [] };
      rows.forEach(function (m) {
        var id = m.id || "", sn = (m.served_name || "").toLowerCase();
        if (/embed|bge|e5|minilm|m3/.test(id.toLowerCase()) || /embed/.test(sn)) byRole.embedder.push(m);
        else if (/rerank/.test(id.toLowerCase()) || /rerank/.test(sn)) byRole.reranker.push(m);
        else if (/judge|grader|lynx|selene/.test(id.toLowerCase()) || /judge|grader/.test(sn)) byRole.grader.push(m);
        else byRole.generator.push(m);
      });
      // If any role came back empty (the catalog is generator-only), mirror
      // the full list into the empty buckets so the operator still has choices.
      ["embedder", "reranker", "grader"].forEach(function (role) {
        if (byRole[role].length === 0 && byRole.generator.length > 0) byRole[role] = byRole.generator.slice();
      });
      Object.keys(byRole).forEach(function (role) {
        var sel = $("pick" + role.charAt(0).toUpperCase() + role.slice(1));
        if (!sel) return;
        sel.innerHTML = "";
        if (byRole[role].length === 0) {
          sel.innerHTML = '<option value="">(catalog unavailable)</option>';
        } else {
          byRole[role].forEach(function (m, i) {
            var label = m.id + " · " + (m.served_name || "") + " · " + (m.quantization || "");
            if (m.id === active) label += " · active";
            sel.innerHTML += '<option value="' + esc(m.id) + '"' + (i === 0 ? " selected" : "") + '>' + esc(label) + '</option>';
          });
        }
      });
    } catch (e) {
      pickers.forEach(function (id) {
        var sel = $(id); if (sel) sel.innerHTML = '<option value="">(catalog unavailable)</option>';
      });
      $("catalogWarn").classList.remove("hidden");
    }
  }

  // -------------------------------------------------------------------------
  // Reports list — populates the A/B compare dropdowns and the recent runs.
  // -------------------------------------------------------------------------
  async function loadReports() {
    try {
      var r = await fetch(API + "/benchmark/reports", { headers: authHeaders() });
      if (r.status === 401) { setToken(null); showLogin(); return; }
      if (!r.ok) { $("compareA").innerHTML = '<option value="">(unavailable)</option>'; $("compareB").innerHTML = '<option value="">(unavailable)</option>'; $("reportsList").innerHTML = '<p class="muted">Reports unavailable.</p>'; return; }
      var rows = await r.json();
      var opts = '<option value="">(select a report)</option>';
      var list = "";
      rows.forEach(function (rep) {
        var label = (rep.display_id || ("#" + (rep.id || "").slice(0, 8))) + " · " + (rep.report_label || (rep.dataset_name || rep.dataset_id || "report")) + " · " + (rep.created_at || "");
        opts += '<option value="' + esc(rep.id) + '">' + esc(label) + '</option>';
        list += '<div class="report-row"><span class="muted">' + esc(label) + '</span>' +
          '<button class="btn-ghost" data-report-id="' + esc(rep.id) + '">View</button></div>';
      });
      $("compareA").innerHTML = opts;
      $("compareB").innerHTML = opts;
      $("reportsList").innerHTML = list || '<p class="muted">No reports yet — run a benchmark to create one.</p>';
    } catch (e) {
      $("compareA").innerHTML = '<option value="">(unavailable)</option>';
      $("compareB").innerHTML = '<option value="">(unavailable)</option>';
      $("reportsList").innerHTML = '<p class="muted">Reports unavailable.</p>';
    }
  }

  // -------------------------------------------------------------------------
  // Live progress panel — drives stage chips and the detail line.
  // -------------------------------------------------------------------------
  var STAGES = ["queued", "swapping", "ready", "running", "scoring", "done"];
  function setStage(name, detail) {
    var stageOrder = STAGES.indexOf(name);
    var chips = document.querySelectorAll("#progressPanel .stage");
    chips.forEach(function (c) {
      var n = c.getAttribute("data-stage");
      c.classList.remove("active", "done", "error");
      if (n === "error") return;
      var idx = STAGES.indexOf(n);
      if (name === "error") { /* nothing — error chip handled below */ }
      else if (idx === -1) {}
      else if (idx < stageOrder) c.classList.add("done");
      else if (idx === stageOrder) c.classList.add("active");
    });
    if (name === "error") {
      var ec = document.querySelector('#progressPanel .stage[data-stage="error"]');
      if (ec) ec.classList.add("error");
    }
    $("progressDetail").innerHTML = detail || "";
  }
  function resetProgress() {
    var chips = document.querySelectorAll("#progressPanel .stage");
    chips.forEach(function (c) { c.classList.remove("active", "done", "error"); });
    $("progressDetail").textContent = "No run yet.";
  }

  // -------------------------------------------------------------------------
  // Run — POST /benchmark/run and consume the SSE stream.
  // -------------------------------------------------------------------------
  function assembleBody() {
    var params = {
      top_k: parseInt($("paramTopK").value, 10),
      temperature: parseFloat($("paramTemp").value),
      max_tokens: parseInt($("paramMaxTokens").value, 10)
    };
    return {
      generator: $("pickGenerator").value,
      embedder: $("pickEmbedder").value,
      reranker: $("pickReranker").value,
      grader: $("pickGrader").value,
      dataset_id: $("pickDataset").value,
      params: params
    };
  }

  function setRunButtonDisabled(disabled) {
    var b = $("runBtn");
    b.disabled = !!disabled;
  }

  async function runBenchmark() {
    if (!isOperator()) { toast("eval-operator group required", true); return; }
    var body = assembleBody();
    if (!body.generator || !body.embedder || !body.reranker || !body.grader) {
      toast("Pick a model for every step", true); return;
    }
    resetProgress();
    setRunButtonDisabled(true);
    setStage("queued", "queued — submitting…");
    var reportId = null;
    try {
      var r = await fetch(API + "/benchmark/run", {
        method: "POST",
        headers: authHeaders({ "Content-Type": "application/json" }),
        body: JSON.stringify(body)
      });
      if (r.status === 401) { setToken(null); showLogin(); return; }
      if (r.status === 403) { setStage("error", "403 — eval-operator group required (server-enforced)."); toast("Forbidden", true); setRunButtonDisabled(false); return; }
      if (!r.ok || !r.body) {
        var msg = "Run failed (" + r.status + ")";
        try { var ej = await r.json(); if (ej && (ej.error || ej.detail)) msg = ej.error || ej.detail; } catch (x) {}
        setStage("error", msg); toast(msg, true); setRunButtonDisabled(false); return;
      }
      var reader = r.body.getReader();
      var dec = new TextDecoder("utf-8");
      var buf = "";
      var lastEvent = "";
      // SSE: each event is "event: <type>\ndata: <json>\n\n" (per /query/stream).
      while (true) {
        var chunk = await reader.read();
        if (chunk.done) break;
        buf += dec.decode(chunk.value, { stream: true });
        var idx;
        while ((idx = buf.indexOf("\n\n")) !== -1) {
          var raw = buf.slice(0, idx); buf = buf.slice(idx + 2);
          if (raw.charAt(0) === ":") continue; // heartbeat / comment
          var ev = "", data = "";
          raw.split("\n").forEach(function (line) {
            if (line.indexOf("event: ") === 0) ev = line.slice(7);
            else if (line.indexOf("data: ") === 0) data = line.slice(6);
          });
          lastEvent = ev || lastEvent;
          if (!ev || !data) continue;
          var payload = {};
          try { payload = JSON.parse(data); } catch (x) { payload = {}; }
          handleRunEvent(ev, payload);
          if (ev === "done") { reportId = payload.report_id || null; }
          if (ev === "done" || ev === "error") { try { await reader.cancel(); } catch (x) {} break; }
        }
        if (lastEvent === "done" || lastEvent === "error") break;
      }
      if (reportId) {
        await loadReport(reportId);
        loadReports(); // refresh the compare dropdowns
      }
    } catch (e) {
      setStage("error", "Run failed: " + e);
      toast("Run failed: " + e, true);
    } finally {
      setRunButtonDisabled(false);
    }
  }

  function handleRunEvent(ev, p) {
    if (ev === "queued") { setStage("queued", "queued…"); return; }
    if (ev === "swapping") { setStage("swapping", "cold-swapping generator — model_load_ms will be reported in its own bucket."); return; }
    if (ev === "ready") {
      var ml = p.model_load_ms;
      setStage("ready", "ready · model_load_ms = " + (ml != null ? fmt(ml, 1) : "?") + " ms (separate from per-query latency)");
      return;
    }
    if (ev === "running") {
      var i = p.i, n = p.total, q = p.question;
      setStage("running", "running query " + (i + 1) + " / " + n + (q ? " — " + (q.length > 60 ? q.slice(0, 60) + "…" : q) : ""));
      return;
    }
    if (ev === "scoring") { setStage("scoring", "scoring · n_queries=" + (p.n_queries != null ? p.n_queries : "?")); return; }
    if (ev === "done") {
      setStage("done", "done · report_id = " + (p.report_id || ""));
      return;
    }
    if (ev === "error") {
      var msg = p.message || "benchmark run failed";
      var trace = p.trace_ref ? " · trace=" + p.trace_ref : "";
      setStage("error", msg + trace);
      toast(msg, true);
      return;
    }
  }

  // -------------------------------------------------------------------------
  // Report view — fetch GET /benchmark/reports/{id} and render the metric
  // tiers, latency, VRAM, and judge κ.  model_load_ms is shown SEPARATELY
  // from per-query latency (Invariant 4).
  // -------------------------------------------------------------------------
  async function loadReport(reportId) {
    var card = $("reportCard");
    try {
      var r = await fetch(API + "/benchmark/reports/" + encodeURIComponent(reportId), { headers: authHeaders() });
      if (r.status === 401) { setToken(null); showLogin(); return; }
      if (!r.ok) { card.hidden = true; toast("Report not available", true); return; }
      var rep = await r.json();
      card.hidden = false;
      $("reportId").textContent = "· " + (rep.display_id || ("#" + (rep.id || "").slice(0, 8)));
      $("reportSub").innerHTML = IW_reportMeta(rep);
      // Shared renderer (benchmark_render.py) — headline stat cards,
      // metric tiers with hover explanations + units, the REAL latency
      // fields (model_load_ms / per_stage_ms / vLLM p99s), GPU/VRAM, and
      // the config snapshot.  Identical to the Console Results view.
      $("reportBody").innerHTML = IW_renderReport(rep);
    } catch (e) {
      card.hidden = true;
      toast("Report load failed: " + e, true);
    }
  }
  // -------------------------------------------------------------------------
  // A/B compare — GET /benchmark/compare?a=..&b=.. + the winner + caveat.
  // -------------------------------------------------------------------------
  async function compare() {
    var a = $("compareA").value, b = $("compareB").value;
    var body = $("compareBody");
    body.innerHTML = "";
    if (!a || !b) { toast("Pick two reports to compare", true); return; }
    if (a === b) { toast("Pick two different reports", true); return; }
    try {
      var r = await fetch(API + "/benchmark/compare?a=" + encodeURIComponent(a) + "&b=" + encodeURIComponent(b), { headers: authHeaders() });
      if (r.status === 401) { setToken(null); showLogin(); return; }
      if (r.status === 404) { body.innerHTML = '<div class="caveat-box"><strong>Not found.</strong> One of the report ids is unknown.</div>'; return; }
      if (r.status === 400) { body.innerHTML = '<div class="caveat-box"><strong>Cannot compare.</strong> The two reports are on different datasets — paired stats require the same golden queries.</div>'; return; }
      if (!r.ok) { body.innerHTML = '<div class="caveat-box"><strong>Compare failed (' + r.status + ').</strong></div>'; return; }
      var d = await r.json();
      // Shared compare renderer — underpowered caveat, winner verdict
      // (only when better on BOTH nDCG@5 and faithfulness), and the
      // per-metric delta table with hover explanations.
      body.innerHTML = IW_renderCompare(d);
    } catch (e) {
      body.innerHTML = '<div class="caveat-box"><strong>Compare failed:</strong> ' + esc(String(e)) + '</div>';
    }
  }

  // -------------------------------------------------------------------------
  // Wire up
  // -------------------------------------------------------------------------
  $("loginBtn").addEventListener("click", login);
  $("p").addEventListener("keydown", function (e) { if (e.key === "Enter") login(); });
  $("logout").addEventListener("click", function () { setToken(null); showLogin(); });
  $("runBtn").addEventListener("click", runBenchmark);
  $("compareBtn").addEventListener("click", compare);
  $("reportsList").addEventListener("click", function (e) {
    var btn = e.target.closest ? e.target.closest("button[data-report-id]") : null;
    if (!btn) return;
    var id = btn.getAttribute("data-report-id");
    if (!id) return;
    loadReport(id).then(function () {
      var c = $("reportCard");
      if (c && c.scrollIntoView) c.scrollIntoView({ behavior: "smooth", block: "start" });
    });
  });

  async function bootstrap() {
    if (token()) { showApp(sessionStorage.getItem(whoKey), groups()); return; }
    showLogin();
  }
  bootstrap();
})();
</script>
</body>
</html>
"""


def benchmark_page_html() -> str:
    """Return the standalone Benchmark Studio HTML page."""
    page = _PAGE.replace("/* __IW_REPORT_CSS__ */", TOOLTIP_CSS + REPORT_CSS).replace(
        "/* __IW_REPORT_JS__ */", REPORT_JS
    )
    return add_favicon(page)


__all__ = ["benchmark_page_html"]
