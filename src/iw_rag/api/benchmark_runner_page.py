# ruff: noqa: E501
"""Self-contained, branded IW RAG Operator Console Benchmark Runner view (F-00096, S12).

The Runner view is the Operator Console page that drives the
``POST /benchmark/run`` end-to-end — per-step model pickers populated
from ``GET /models`` (the S08-extended catalog with the
server-computed VRAM budget triple), a **live VRAM budget bar** that
renders the API's ``{fits, budget_used_gb, headroom_gb}`` triple, and
the SSE progress stream from ``POST /benchmark/run``.  It is a **thin
API client** — Invariant 4: it enforces NO retrieval / ACL / grounding
/ guard logic.  The eval-operator gate is server-side (the API is the
authoritative gate; the UI disable is honesty only, mirroring
:mod:`iw_rag.api.benchmark_page`'s "UX-only gate" comment on the Run
button).

**Invariant 7 — the keystone.**  The VRAM math is computed
server-side in :func:`iw_rag.api.models_routes._compute_vram_budget`
(the two named constants ``RESIDENT_INFINITY_GB`` and
``VRAM_CEILING_GB`` live there).  The console NEVER recomputes the
budget locally — it reads the server's ``vram_budget.{fits,
budget_used_gb, headroom_gb}`` triple and renders it.  When
``fits === false`` the bar enters the RED zone and the Run button is
disabled (AC1 / Boundary row *"VRAM over budget"*); when the catalog
returns **no eligible candidates** the picker shows a **"no
candidates" empty state** and Run is disabled (Boundary row *"Catalog
has no eligible models"*); the terminal ``error`` event with
``reason_code:"swap_timeout"`` is surfaced as a swap-timeout terminal
state with no report rendered (Boundary row *"Swap timeout during a
run"* — the F-00095 behaviour, re-homed into the console view).

It is a **server-rendered FastAPI HTML page** in the
``benchmark_page.py`` style — NOT Chainlit, NOT a JS SPA, NO new
build toolchain (per the F-00096 design *Out-of-Scope* row "A
JavaScript SPA / new build toolchain").  The page derives its API
base from its own URL so it works behind Caddy (``/api``) or when
hit directly (``/``).  No external CDN / Google Fonts link
(Invariant 8, local-only rule).

Mirrors :mod:`iw_rag.api.benchmark_page`,
:mod:`iw_rag.api.console_page`, and :mod:`iw_rag.api.config_page` —
same header/footer chrome, same teal/ink palette, same
location-derived API base regex idiom with ``\\/`` (the
forward-slash escape that keeps the JS parser from closing the
literal at the first ``/``), same sessionStorage-only bearer token.
The S12 route ``GET /console/benchmark`` is wired against
:func:`benchmark_runner_page_html`.
"""

from __future__ import annotations

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
  .main-wrap { max-width: 1080px; margin: 0 auto; padding: 28px 20px 80px; }
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
  .toast { position: fixed; bottom: 22px; left: 50%; transform: translateX(-50%); background: var(--ink-3); border: 1px solid var(--line); padding: 12px 18px; border-radius: 10px; opacity: 0; transition: .25s; pointer-events: none; max-width: 90vw; z-index: 10; }
  .toast.show { opacity: 1; }
  .toast.err { border-color: var(--danger); color: #ffd7de; }
  .hidden { display: none !important; }
  .muted { color: var(--muted); font-size: 13px; }
  .loginwrap { max-width: 420px; margin: 8vh auto; }
  .catalog-warn { background: rgba(245,158,11,0.08); border: 1px solid #5b4322; color: #fde7c0; padding: 10px 12px; border-radius: 10px; font-size: 13px; margin: 6px 0 0; }
  .no-candidates { background: rgba(244,63,94,0.08); border: 1px solid #5b2230; color: #ffd7de; padding: 10px 12px; border-radius: 10px; font-size: 13px; margin: 6px 0 0; }
  .mode-badge { font-size: 11px; font-weight: 600; letter-spacing: .04em; padding: 3px 9px; border-radius: 999px; border: 1px solid var(--line); text-transform: uppercase; }
  .mode-badge.benchmark { color: #06231f; background: linear-gradient(180deg, var(--teal) 0%, var(--teal-deep) 100%); border-color: transparent; }
  .mode-badge.serving { color: #fde7c0; background: rgba(245,158,11,0.14); border-color: #5b4322; }
  .mode-panel { background: rgba(245,158,11,0.08); border: 1px solid #5b4322; color: #fde7c0; padding: 10px 12px; border-radius: 10px; font-size: 13px; margin: 6px 0 0; }
  .mode-panel code { background: rgba(0,0,0,0.3); padding: 1px 6px; border-radius: 6px; font-family: "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
  /* VRAM budget bar (AC1 / Invariant 7).  The fill width is driven by
     the server's budget_used_gb relative to the 32 GB ceiling (the
     ceiling itself lives server-side; the page renders the triple
     verbatim — no client-side math).  The RED class is keyed off the
     server's ``fits`` flag. */
  .vram-bar { position: relative; width: 100%; height: 18px; background: var(--ink-2); border: 1px solid var(--line); border-radius: 999px; overflow: hidden; margin: 8px 0 4px; }
  .vram-bar-fill { position: absolute; left: 0; top: 0; bottom: 0; width: 0%; background: linear-gradient(180deg, var(--teal) 0%, var(--teal-deep) 100%); transition: width .2s ease, background .2s ease; }
  .vram-bar-fill.vram-bar-red { background: linear-gradient(180deg, var(--danger) 0%, #be123c 100%); }
  .vram-bar-label { font-size: 12px; color: var(--muted); font-family: "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; margin-top: 4px; display: flex; justify-content: space-between; gap: 8px; }
  .vram-bar-label .fits-ok { color: var(--ok); }
  .vram-bar-label .fits-bad { color: var(--danger); }
  .progress { display: flex; gap: 8px; flex-wrap: wrap; margin: 12px 0; }
  .stage { font-size: 12px; padding: 4px 10px; border-radius: 999px; border: 1px solid var(--line); color: var(--muted); background: var(--ink-2); transition: .15s; }
  .stage.active { color: #06231f; background: linear-gradient(180deg, var(--teal) 0%, var(--teal-deep) 100%); border-color: transparent; }
  .stage.done { color: var(--ok); border-color: var(--teal-deep); }
  .stage.error { color: var(--danger); border-color: #5b2230; background: #2a1620; }
  .progress-detail { font-size: 13px; color: var(--muted); min-height: 18px; margin-top: 6px; font-family: "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
  .spin { display: inline-block; width: 12px; height: 12px; border: 2px solid var(--line); border-top-color: var(--teal); border-radius: 50%; animation: spin .7s linear infinite; }
  @keyframes spin { to { transform: rotate(360deg); } }
  nav.console-nav { display: flex; gap: 6px; flex-wrap: wrap; align-items: center; padding: 14px 28px; background: rgba(24,29,41,0.5); border-bottom: 1px solid var(--line); }
  nav.console-nav a.nav-item { font-size: 13px; padding: 6px 12px; border-radius: 8px; border: 1px solid transparent; color: var(--muted); }
  nav.console-nav a.nav-item:hover { color: var(--text); border-color: var(--line); }
  nav.console-nav a.nav-item.active { color: #06231f; background: linear-gradient(180deg, var(--teal) 0%, var(--teal-deep) 100%); border-color: transparent; font-weight: 600; }
</style>
</head>
<body>
<header>
  <span class="logo"><span class="i">I</span>nnovation <span class="w">W</span>ays</span>
  <span class="muted">· Benchmark Runner</span>
  <span class="mode-badge hidden" id="modeBadge"></span>
  <span class="spacer"></span>
  <span class="who" id="who"></span>
  <button class="btn-ghost hidden" id="logout">Sign out</button>
</header>

<nav class="console-nav hidden" id="consoleNav" aria-label="Console navigation">
  <a class="nav-item" id="navConfig" href="/console/config">Config</a>
  <a class="nav-item" id="navBenchmark" href="/console/benchmark">Benchmark</a>
  <a class="nav-item" id="navResults" href="/console/results">Results</a>
  <a class="nav-item" id="navCompare" href="/console/compare">Compare</a>
  <a class="nav-item" id="navScoreboard" href="/console/scoreboard">Scoreboard</a>
  <a class="nav-item" id="navChat" href="/console/chat">Chat</a>
  <a class="nav-item" id="navUpload" href="/console/upload">Upload</a>
  <a class="nav-item" id="navSystem" href="/console/system">System</a>
</nav>

<main>
  <section id="loginView" class="hidden">
    <div class="card loginwrap">
      <h1>Sign in</h1>
      <p class="sub">Use your demo credentials. The Benchmark Runner is gated to the <strong>eval-operator</strong> group for runs; read-only viewing is open to any authenticated principal.</p>
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
      <p class="sub">Pick the per-step models, the golden dataset, and the run params. The generator is cold-swapped via F-00094; the live <strong>VRAM budget bar</strong> below renders the server-computed triple from <code>GET /models</code> — the math is server-side (Invariant 7). The run streams progress over SSE from <code>POST /benchmark/run</code>.</p>

      <div id="catalogWarn" class="catalog-warn hidden">Model catalog unavailable — populate the pickers manually or fix <code>GET /models</code>.</div>
      <div id="noCandidatesWarn" class="no-candidates hidden">No candidates — the catalog returned no eligible models. Run is disabled until a model becomes available.</div>
      <div id="modePanel" class="mode-panel hidden"></div>

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
          <label for="pickTopK">top_k</label>
          <input type="number" id="pickTopK" min="1" max="20" value="5" />
        </div>
        <div>
          <label for="pickMode">mode</label>
          <select id="pickMode">
            <option value="auto" selected>auto</option>
            <option value="retrieve">retrieve</option>
            <option value="graph">graph</option>
            <option value="long_context">long_context</option>
            <option value="visual">visual</option>
          </select>
        </div>
      </div>

      <div style="margin-top: 18px;">
        <label>VRAM budget (server-computed, 32 GB ceiling)</label>
        <div class="vram-bar" id="vramBar" aria-label="VRAM budget bar">
          <div class="vram-bar-fill" id="vramBarFill"></div>
        </div>
        <div class="vram-bar-label" id="vramBarLabel">
          <span class="muted" id="vramBarUsed">budget: —</span>
          <span id="vramBarFits" class="muted">fits: —</span>
        </div>
      </div>

      <div style="margin-top: 18px; display: flex; gap: 12px; align-items: center;">
        <button class="btn-primary" id="runBtn">Run Full Test</button>
        <span class="muted" id="runHint">Live stream — your bearer token is used; the run is gated to the <code>eval-operator</code> group server-side.</span>
      </div>
    </div>

    <div class="card" id="progressCard">
      <h2>Live progress</h2>
      <p class="sub">SSE stream from <code>POST /benchmark/run</code>. Stages: <code>queued → swapping → ready → running (query i / N) → scoring → done</code>. A terminal <code>error</code> with <code>reason_code: "swap_timeout"</code> is surfaced as a swap-timeout terminal state (no report rendered).</p>
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
  </section>
</main>

<div class="toast" id="toast"></div>

<script>
(function () {
  // API base. The Operator Console is served at /console* behind the Caddy
  // edge (ADR-016 single entrypoint); the API lives under /api/*, so the base
  // is the fixed "/api" — fetch(API + "/api/config") -> /api/api/config, which
  // the edge's `handle_path /api/*` strips back to /api/config on the API.
  var API = "/api";
  var tokenKey = "iw_token", whoKey = "iw_principal", groupsKey = "iw_groups";
  var $ = function (id) { return document.getElementById(id); };
  var esc = function (s) { var d = document.createElement("div"); d.textContent = s == null ? "" : s; return d.innerHTML; };
  var fmt = function (n, d) { if (n == null || isNaN(n)) return "—"; if (d == null) d = 3; return Number(n).toFixed(d); };

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

  // The catalog rows, keyed by id, with the server-computed VRAM budget
  // triple attached.  Populated by loadModels; consumed by
  // updateVramBar when the generator picker changes.
  var CATALOG = { rows: [], rowsById: {} };

  function showLogin() {
    $("loginView").classList.remove("hidden");
    $("appView").classList.add("hidden");
    $("consoleNav").classList.add("hidden");
    $("logout").classList.add("hidden");
    $("who").textContent = "";
  }
  function showApp(who, grp) {
    $("loginView").classList.add("hidden");
    $("appView").classList.remove("hidden");
    $("consoleNav").classList.remove("hidden");
    $("logout").classList.remove("hidden");
    var tag = who ? ("Signed in as " + who) : "";
    if (grp && grp.length) tag += " · " + grp.join(", ");
    $("who").textContent = tag;
    // UX-only gate: hide Run for non-eval-operators.  The API is the
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
    highlightActiveNav();
    loadMode();
    // Restore the live-progress panel AFTER the picker gate is settled,
    // so attachExistingRun captures the correct Run-button state to
    // restore when a reconnected run finishes.
    loadModels().then(attachExistingRun, attachExistingRun);
  }
  // Read-only operating-mode badge + panel (serving vs benchmark). The actual
  // switch is a host-side `make` target (Option B — no privileged in-api flip),
  // so we surface the command rather than a button. See GET /benchmark/mode.
  async function loadMode() {
    try {
      var r = await fetch(API + "/benchmark/mode", { headers: authHeaders() });
      if (!r.ok) return;
      var m = await r.json();
      var badge = $("modeBadge");
      badge.textContent = (m.mode === "benchmark" ? "Benchmark mode" : "Serving mode");
      badge.className = "mode-badge " + (m.mode === "benchmark" ? "benchmark" : "serving");
      var panel = $("modePanel");
      if (m.mode === "benchmark") {
        panel.innerHTML = "Benchmark mode is active (guards off, generator via the swap proxy). "
          + "To restore normal chat/query, run <code>" + m.command + "</code> on the host.";
      } else {
        panel.innerHTML = "This box is in <strong>serving mode</strong>, so the model picker is empty by design — "
          + "the Benchmark Studio needs benchmark mode (swap sidecars up, GPU freed). "
          + "Run <code>" + m.command + "</code> on the host to enable it.";
      }
      panel.classList.remove("hidden");
    } catch (e) { /* badge is best-effort; never block the page */ }
  }
  function highlightActiveNav() {
    var path = location.pathname;
    var items = document.querySelectorAll("nav.console-nav a.nav-item");
    items.forEach(function (a) {
      var href = a.getAttribute("href") || "";
      if (href && (path === href || path.indexOf(href + "/") === 0)) {
        a.classList.add("active");
      } else {
        a.classList.remove("active");
      }
    });
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
  // Models catalog (F-00094 GET /models extended by S08 with
  // vram_budget.{fits, budget_used_gb, headroom_gb}).  Degrades
  // gracefully on 4xx/5xx/timeout — the page keeps loading and the
  // pickers fall into the "catalog unavailable" state.  Zero eligible
  // candidates surfaces the "no candidates" empty state and disables
  // Run (Boundary row).
  // -------------------------------------------------------------------------
  async function loadModels() {
    var pickers = ["pickGenerator", "pickEmbedder", "pickReranker"];
    try {
      var r = await fetch(API + "/models", { headers: authHeaders() });
      if (r.status === 401) { setToken(null); showLogin(); return; }
      if (!r.ok) throw new Error("status " + r.status);
      var d = await r.json();
      var rows = (d && d.models) || [];
      // Index rows by id so the VRAM-bar update can find the
      // currently-selected generator's vram_budget without re-issuing
      // the GET /models call.
      CATALOG.rows = rows.filter(function (m) { return m && m.id; });
      CATALOG.rowsById = {};
      CATALOG.rows.forEach(function (m) { CATALOG.rowsById[m.id] = m; });
      var active = d && d.active_model_id;
      // Group rows by inferred role from the served_name / id when the
      // catalog doesn't carry an explicit role; fall back to a single
      // bucket so the pickers are still populated.
      var byRole = { generator: [], embedder: [], reranker: [] };
      CATALOG.rows.forEach(function (m) {
        var id = m.id || "", sn = (m.served_name || "").toLowerCase();
        if (/embed|bge|e5|minilm|m3/.test(id.toLowerCase()) || /embed/.test(sn)) byRole.embedder.push(m);
        else if (/rerank/.test(id.toLowerCase()) || /rerank/.test(sn)) byRole.reranker.push(m);
        else byRole.generator.push(m);
      });
      // If any role came back empty (the catalog is generator-only), mirror
      // the full list into the empty buckets so the operator still has choices.
      ["embedder", "reranker"].forEach(function (role) {
        if (byRole[role].length === 0 && byRole.generator.length > 0) byRole[role] = byRole.generator.slice();
      });
      var anyEmpty = false;
      Object.keys(byRole).forEach(function (role) {
        var sel = $("pick" + role.charAt(0).toUpperCase() + role.slice(1));
        if (!sel) return;
        sel.innerHTML = "";
        if (byRole[role].length === 0) {
          sel.innerHTML = '<option value="">(no candidates)</option>';
          anyEmpty = true;
        } else {
          byRole[role].forEach(function (m, i) {
            var label = m.id + " · " + (m.served_name || "") + " · " + (m.quantization || "");
            if (m.id === active) label += " · active";
            sel.innerHTML += '<option value="' + esc(m.id) + '"' + (i === 0 ? " selected" : "") + '>' + esc(label) + '</option>';
          });
        }
      });
      // Boundary: zero eligible candidates → "no candidates" empty
      // state + Run disabled.  We disable Run BEFORE the vram-bar
      // update runs so the RED zone is not required to disable Run in
      // the empty-catalog case.
      if (anyEmpty) {
        $("noCandidatesWarn").classList.remove("hidden");
        $("runBtn").disabled = true;
        $("runBtn").title = "No eligible candidates in the catalog";
        $("runHint").textContent = "Run is disabled — the catalog returned no eligible candidates.";
      } else {
        $("noCandidatesWarn").classList.add("hidden");
        // Re-enable Run only when the operator is also in the
        // eval-operator group (the UX-only gate set in showApp).  When the
        // principal is NOT an operator the button stays disabled, so we must
        // preserve the not-in-group warning rather than clobbering it with the
        // happy "Live stream" hint — otherwise the button is disabled while the
        // hint misleadingly reassures the operator that a run is possible.
        if (isOperator()) {
          $("runBtn").disabled = false;
          $("runBtn").title = "";
          $("runHint").textContent = "Live stream — your bearer token is used; the run is gated to the eval-operator group server-side.";
        } else {
          $("runBtn").disabled = true;
          $("runBtn").title = "eval-operator group required (server-enforced)";
          $("runHint").textContent = "Your account is not in the eval-operator group — Run Full Test is disabled (the API will 403 anyway).";
        }
      }
      updateVramBar();
    } catch (e) {
      pickers.forEach(function (id) {
        var sel = $(id); if (sel) sel.innerHTML = '<option value="">(catalog unavailable)</option>';
      });
      $("catalogWarn").classList.remove("hidden");
    }
  }

  // -------------------------------------------------------------------------
  // VRAM budget bar — AC1 / Invariant 7.  The bar reads the
  // server-computed triple off the currently-selected generator's
  // catalog row.  The page MUST NOT recompute the budget locally; the
  // server's ``vram_budget.fits`` flag is the authoritative verdict.
  // -------------------------------------------------------------------------
  function updateVramBar() {
    var genId = $("pickGenerator").value;
    var row = genId ? CATALOG.rowsById[genId] : null;
    var b = (row && row.vram_budget) || null;
    var fill = $("vramBarFill");
    var used = $("vramBarUsed");
    var fitsLabel = $("vramBarFits");
    if (!b) {
      fill.style.width = "0%";
      fill.classList.remove("vram-bar-red");
      used.textContent = "budget: —";
      fitsLabel.textContent = "fits: —";
      fitsLabel.className = "muted";
      return;
    }
    // Drive the bar's width from the server's budget_used_gb
    // relative to the server's ceiling.  The ceiling itself is
    // NEVER hard-coded in the page; the page reads the server's
    // budget_used_gb and headroom_gb verbatim, and the bar scale
    // is ``budget_used_gb / (budget_used_gb + headroom_gb)`` —
    // the denominator is the server-computed ceiling, derived
    // purely from the two server fields (Invariant 7).  When
    // ``fits === false`` the bar is rendered at 100% (clamped, RED
    // zone) and a headroom_gb <= 0 is rendered as ``100%`` /
    // over-budget rather than a negative percentage.  The bar is a
    // render of the server triple, never a re-derivation of the
    // math.
    var bg = Number(b.budget_used_gb) || 0;
    var hg = Number(b.headroom_gb);
    var pct;
    if (b.fits === false) {
      pct = 100;
    } else if (!isNaN(hg) && (bg + hg) > 0) {
      pct = Math.max(0, Math.min(100, bg / (bg + hg) * 100));
    } else {
      pct = 0;
    }
    fill.style.width = pct.toFixed(1) + "%";
    // The RED class is keyed off the server's ``fits`` flag (the
    // keystone — NOT a client-side check on headroom_gb < 0).
    if (b.fits === false) {
      fill.classList.add("vram-bar-red");
      used.textContent = "budget: " + fmt(bg, 1) + " GB (over ceiling)";
      used.className = "fits-bad";
      fitsLabel.textContent = "fits: false (headroom " + fmt(hg, 1) + " GB)";
      fitsLabel.className = "fits-bad";
    } else {
      fill.classList.remove("vram-bar-red");
      used.textContent = "budget: " + fmt(bg, 1) + " GB / ceiling";
      used.className = "fits-ok";
      fitsLabel.textContent = "fits: true (headroom " + fmt(hg, 1) + " GB)";
      fitsLabel.className = "fits-ok";
    }
    // AC1: Run is disabled when the server says fits === false
    // (the RED zone), or when the operator is not in the
    // eval-operator group (the UX-only gate in showApp).  We never
    // disable Run on a client-side headroom check — the server's
    // ``fits`` flag is the authoritative verdict.
    if (b.fits === false) {
      $("runBtn").disabled = true;
      $("runBtn").title = "Generator exceeds the 32 GB VRAM ceiling (server-computed fits=false)";
      $("runHint").textContent = "Run is disabled — the selected generator exceeds the 32 GB VRAM ceiling (server-computed fits=false).";
    } else if (isOperator()) {
      $("runBtn").disabled = false;
      $("runBtn").title = "";
      $("runHint").textContent = "Live stream — your bearer token is used; the run is gated to the eval-operator group server-side.";
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
    return {
      generator: $("pickGenerator").value,
      embedder: $("pickEmbedder").value,
      reranker: $("pickReranker").value,
      grader: $("pickGenerator").value,  // F-00096 S12: the Runner view
                                          // uses the generator as the
                                          // judge (F-00095 / S01 default).
                                          // The S07 Config view owns the
                                          // grader's editable config key.
      dataset_id: "benchmark_golden",
      params: {
        top_k: parseInt($("pickTopK").value, 10),
        mode: $("pickMode").value
      }
    };
  }

  function setRunButtonDisabled(disabled) {
    var b = $("runBtn");
    b.disabled = !!disabled;
  }

  async function runBenchmark() {
    if (!isOperator()) { toast("eval-operator group required", true); return; }
    var body = assembleBody();
    if (!body.generator || !body.embedder || !body.reranker) {
      toast("Pick a model for every step", true); return;
    }
    // Re-check the server's ``fits`` flag at click time — the
    // operator may have edited the catalog between the picker
    // change and the click, or the picker change may not have
    // re-run updateVramBar (e.g. a stale value).  The server is
    // authoritative; this is a UX-only defence.
    var genRow = CATALOG.rowsById[body.generator];
    if (genRow && genRow.vram_budget && genRow.vram_budget.fits === false) {
      toast("Selected generator exceeds the 32 GB VRAM ceiling (server-computed fits=false).", true);
      return;
    }
    resetProgress();
    setRunButtonDisabled(true);
    setStage("queued", "queued — submitting…");
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
        // A 400 "already in progress" means a run is live server-side
        // (e.g. started in another tab) — attach to it instead of erroring.
        if (r.status === 400 && /already in progress/i.test(msg)) {
          toast("A run is already in progress — attaching to it.", false);
          setRunButtonDisabled(false);
          attachExistingRun();
          return;
        }
        setStage("error", msg); toast(msg, true); setRunButtonDisabled(false); return;
      }
      await consumeStream(r.body.getReader());
    } catch (e) {
      setStage("error", "Run failed: " + e);
      toast("Run failed: " + e, true);
    } finally {
      setRunButtonDisabled(false);
    }
  }

  // Consume one SSE stream (the POST live stream or the GET reconnect
  // stream). Each event is "event: <type>\ndata: <json>\n\n". onFirstEvent
  // (optional) fires once when the first real event arrives — used by the
  // reconnect path to disable Run only when a run actually exists.
  async function consumeStream(reader, onFirstEvent) {
    var dec = new TextDecoder("utf-8");
    var buf = "";
    var lastEvent = "";
    var sawEvent = false;
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
        if (!sawEvent) { sawEvent = true; if (onFirstEvent) onFirstEvent(); }
        handleRunEvent(ev, payload);
        if (ev === "done" || ev === "error") { try { await reader.cancel(); } catch (x) {} return; }
      }
      if (lastEvent === "done" || lastEvent === "error") return;
    }
  }

  // Reconnect: on (re)load, re-attach to the current/most-recent run so the
  // Live progress panel is restored (the console nav is full-page links, so
  // navigating away dropped the POST SSE stream — but the run keeps going
  // server-side). When no run exists the stream is just a heartbeat and we
  // leave the "No run yet." state and the Run-button gate untouched.
  async function attachExistingRun() {
    var prevDisabled = $("runBtn").disabled;
    var changed = false;
    try {
      var r = await fetch(API + "/benchmark/run/events", { headers: authHeaders() });
      if (r.status === 401) { setToken(null); showLogin(); return; }
      if (!r.ok || !r.body) return;
      await consumeStream(r.body.getReader(), function () {
        // A run exists — restore the panel context and lock Run while it streams.
        resetProgress();
        changed = true;
        setRunButtonDisabled(true);
      });
    } catch (e) { /* best-effort restore; never block the page */ }
    finally {
      // Restore whatever gate loadModels had settled (operator / VRAM /
      // candidates) — only if we actually took the button over.
      if (changed) $("runBtn").disabled = prevDisabled;
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
      // Boundary "Swap timeout during a run" — surface the
      // ``reason_code: "swap_timeout"`` terminal state verbatim.  The
      // run was a no-op; no report is rendered.  A non-timeout error
      // is surfaced as a generic error stage with the message +
      // trace id.
      var rc = p.reason_code || "";
      var msg = p.message || "benchmark run failed";
      var trace = p.trace_ref ? " · trace=" + p.trace_ref : "";
      if (rc === "swap_timeout") {
        setStage("error", "swap_timeout — the swap-controller never reported ready; no report rendered." + (p.trace_ref ? " · trace=" + p.trace_ref : ""));
        toast("Swap timeout — no report rendered.", true);
      } else {
        setStage("error", msg + trace);
        toast(msg, true);
      }
      return;
    }
  }

  // -------------------------------------------------------------------------
  // Wire up
  // -------------------------------------------------------------------------
  $("loginBtn").addEventListener("click", login);
  $("p").addEventListener("keydown", function (e) { if (e.key === "Enter") login(); });
  $("logout").addEventListener("click", function () { setToken(null); showLogin(); });
  $("runBtn").addEventListener("click", runBenchmark);
  // AC1: re-evaluate the VRAM bar + Run-disabled state whenever the
  // operator changes the generator pick.  The bar reads the
  // server's ``vram_budget.fits`` flag (Invariant 7) and disables
  // Run when the flag is ``false`` (the RED zone).
  $("pickGenerator").addEventListener("change", updateVramBar);

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


def benchmark_runner_page_html() -> str:
    """Return the standalone Benchmark Runner view HTML page.

    The view is a self-contained inline HTML+CSS+vanilla-JS page
    mirroring :mod:`iw_rag.api.benchmark_page` and
    :mod:`iw_rag.api.console_page`, re-homed as an Operator Console
    sub-view.  It carries the IW brand chrome (teal ``#0D9488`` /
    ``#2DD4BF`` on ink ``#181D29``), the location-derived API base,
    the sessionStorage-only bearer token, the ``/auth/login``
    self-handled sign-in, the per-step model pickers (generator /
    embedder / reranker / top_k / mode) populated from
    ``GET /models``, the live **VRAM budget bar** (reading the
    server-computed ``vram_budget.{fits, budget_used_gb,
    headroom_gb}`` triple — the page NEVER recomputes the math
    locally, per Invariant 7), the ``POST /benchmark/run`` SSE
    consumer with the full stage set (``queued → swapping → ready →
    running → scoring → done`` + ``error``), the swap-timeout
    terminal state for ``reason_code:"swap_timeout"`` Boundary
    row, the no-candidates empty state Boundary row, and the
    run-button disable / re-enable lifecycle.

    Returns:
        The raw, self-contained HTML document as a ``str`` (no MIME /
        static-mount wiring required).
    """
    return add_favicon(_PAGE)


__all__ = ["benchmark_runner_page_html"]
