# ruff: noqa: E501
"""Self-contained, branded IW RAG Operator Console multi-run Compare view.

The Compare view is the Operator Console page that compares **2..5**
persisted ``benchmark_reports`` rows on the **same dataset**, item by
item, against one designated **baseline** run.  Where the older A/B
panel on the Results page rendered a single aggregate delta table, this
view is built for the volume of an N-way, per-item comparison:

* a sticky **legend** identifying each run by its execution id + module
  composition, each assigned a stable palette **colour** reused across
  every chart in the view (and the matching PDF);
* an **aggregate** per-metric grouped bar chart (one coloured bar per
  run);
* **baseline-relative** per-run delta tables (the same paired Wilcoxon /
  BCa-CI / Bonferroni-Holm machinery the 2-way compare uses, run against
  the baseline) with the shared underpowered caveat;
* a **per-item** section: a metric-selectable heatmap (rows = items,
  columns = runs) over the whole golden set, with each item expandable
  to per-metric grouped bars **and** each run's generated answer +
  retrieved contexts for a qualitative diff;
* a **Generate Report (PDF)** button mirroring the single-report export.

It is a **thin API client** (Invariant 4) — all statistics + assembly
stay server-side (``GET /benchmark/compare/multi`` / the F-00093 stats
module).  Like every console view it is a server-rendered, self-contained
inline HTML+CSS+vanilla-JS page (NOT Chainlit, NOT a JS SPA, no new build
toolchain), self-handles sign-in via ``/auth/login``, derives its API
base from its own URL so it works behind Caddy (``/api``) or hit
directly, carries a sessionStorage-only bearer token, and links no
external CDN / font (Invariant 8, local-only).

Mirrors :mod:`iw_rag.api.results_page` for the head/footer chrome and
reuses the shared :mod:`iw_rag.api.benchmark_render` fragments
(``TOOLTIP_CSS`` / ``REPORT_CSS`` / ``REPORT_JS``) so the metric
glossary, escaping, number formatting, value bars and score-colour
helpers stay identical to the Results deep-dive.  The
``GET /console/compare`` route is wired against :func:`compare_page_html`.
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
  .main-wrap { max-width: 1180px; margin: 0 auto; padding: 28px 20px 80px; }
  .card { background: rgba(34,38,49,0.72); border: 1px solid var(--line); border-radius: var(--radius); padding: 22px; margin-bottom: 22px; }
  h1 { font-family: "Space Grotesk", system-ui, sans-serif; font-size: 22px; margin: 0 0 4px; }
  h2 { font-family: "Space Grotesk", system-ui, sans-serif; font-size: 16px; margin: 0 0 14px; color: var(--text); }
  .sub { color: var(--muted); font-size: 14px; margin: 0 0 18px; }
  label { display: block; font-size: 13px; color: var(--muted); margin: 12px 0 6px; }
  input[type=text], input[type=password], select {
    width: 100%; padding: 10px 12px; border-radius: 10px; border: 1px solid var(--line);
    background: var(--ink-2); color: var(--text); font: inherit;
  }
  input:disabled, select:disabled, button:disabled { opacity: .6; cursor: not-allowed; }
  .row { display: flex; gap: 16px; flex-wrap: wrap; align-items: flex-end; }
  .row > div { flex: 1; min-width: 180px; }
  table { width: 100%; border-collapse: collapse; }
  th, td { text-align: left; padding: 10px 8px; border-bottom: 1px solid var(--line); font-size: 13px; vertical-align: top; }
  th { color: var(--muted); font-weight: 600; font-size: 11px; text-transform: uppercase; letter-spacing: .5px; }
  td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
  .chip { font-size: 11px; padding: 2px 8px; border-radius: 999px; border: 1px solid var(--line); white-space: nowrap; }
  .chip.ok { color: var(--ok); border-color: var(--teal-deep); }
  .chip.bad { color: var(--danger); border-color: #5b2230; }
  .chip.muted { color: var(--muted); }
  .toast { position: fixed; bottom: 22px; left: 50%; transform: translateX(-50%); background: var(--ink-3); border: 1px solid var(--line); padding: 12px 18px; border-radius: 10px; opacity: 0; transition: .25s; pointer-events: none; max-width: 90vw; z-index: 10; }
  .toast.show { opacity: 1; }
  .toast.err { border-color: var(--danger); color: #ffd7de; }
  .hidden { display: none !important; }
  .muted { color: var(--muted); font-size: 13px; }
  .loginwrap { max-width: 420px; margin: 8vh auto; }
  .caveat-box { background: rgba(245,158,11,0.10); border: 1px solid #5b4322; color: #fde7c0; padding: 12px 14px; border-radius: 10px; margin: 12px 0; font-size: 13px; line-height: 1.5; }
  .caveat-box strong { color: var(--warn); }
  .caveat-box.err { background: rgba(244,63,94,0.10); border-color: #5b2230; color: #ffd7de; }
  .caveat-box.err strong { color: var(--danger); }
  details { margin: 8px 0; }
  details summary { cursor: pointer; color: var(--muted); font-size: 13px; padding: 6px 0; }
  nav.console-nav { display: flex; gap: 6px; flex-wrap: wrap; align-items: center; padding: 14px 28px; background: rgba(24,29,41,0.5); border-bottom: 1px solid var(--line); }
  nav.console-nav a.nav-item { font-size: 13px; padding: 6px 12px; border-radius: 8px; border: 1px solid transparent; color: var(--muted); }
  nav.console-nav a.nav-item:hover { color: var(--text); border-color: var(--line); }
  nav.console-nav a.nav-item.active { color: #06231f; background: linear-gradient(180deg, var(--teal) 0%, var(--teal-deep) 100%); border-color: transparent; font-weight: 600; }
  /* ---- Compare-specific ---- */
  .pick-table td, .pick-table th { vertical-align: middle; }
  .pick-table td.c { text-align: center; width: 36px; }
  /* ---- Runs picker: sortable headers (every data column) ---- */
  .pick-table th.sortable { cursor: pointer; -webkit-user-select: none; user-select: none; }
  .pick-table th.sortable:hover { color: var(--teal); }
  .pick-table th.sortable .sort-ind { display: inline-block; width: 12px; margin-left: 4px; color: var(--teal); font-size: 10px; }
  .pick-table th.sortable:not(.asc):not(.desc) .sort-ind { color: var(--muted); opacity: .55; }
  /* ---- Runs picker: pagination bar (fixed 20 rows / page) ---- */
  .pick-pager { display: flex; align-items: center; justify-content: flex-end; gap: 10px; margin-top: 14px; font-size: 13px; }
  .pick-page-info { color: var(--muted); font-variant-numeric: tabular-nums; }
  .pick-pager button { padding: 6px 10px; font-size: 12px; }
  .pick-pager button:disabled { opacity: .4; cursor: not-allowed; }
  .pick-pager button:disabled:hover { border-color: var(--line); }
  .runchip { display: inline-flex; align-items: center; gap: 8px; padding: 6px 12px; border: 1px solid var(--line); border-radius: 999px; background: var(--ink-2); margin: 0 8px 8px 0; font-size: 13px; }
  .runchip .sw { width: 12px; height: 12px; border-radius: 3px; flex: none; }
  .runchip .bl { font-size: 10px; text-transform: uppercase; letter-spacing: .5px; color: #06231f; background: var(--teal); border-radius: 4px; padding: 1px 5px; font-weight: 700; }
  .runchip .mods { color: var(--muted); font-size: 11px; font-family: "JetBrains Mono", ui-monospace, monospace; }
  .legend-wrap { display: flex; flex-wrap: wrap; align-items: center; }
  .agg-metric { margin: 0 0 14px; }
  .agg-metric .ttl { font-size: 13px; color: var(--text); display: flex; align-items: center; margin: 0 0 6px; }
  .agg-bar-row { display: grid; grid-template-columns: 120px 1fr 64px; align-items: center; gap: 10px; padding: 2px 0; }
  .agg-bar-name { font-size: 11px; color: var(--muted); font-family: "JetBrains Mono", ui-monospace, monospace; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .agg-bar-track { height: 13px; background: var(--ink-3); border-radius: 7px; overflow: hidden; }
  .agg-bar-fill { height: 100%; border-radius: 7px; }
  .agg-bar-val { font-size: 12px; color: var(--muted); text-align: right; font-variant-numeric: tabular-nums; font-family: "JetBrains Mono", ui-monospace, monospace; }
  .hm-scroll { overflow-x: auto; }
  table.heatmap td, table.heatmap th { text-align: center; padding: 6px 8px; border: 1px solid var(--line); font-size: 12px; }
  table.heatmap th.q { text-align: left; max-width: 320px; }
  table.heatmap td.q { text-align: left; max-width: 320px; color: var(--text); }
  table.heatmap .cellv { font-variant-numeric: tabular-nums; font-family: "JetBrains Mono", ui-monospace, monospace; }
  .col-sw { display: inline-block; width: 10px; height: 10px; border-radius: 3px; margin-right: 5px; vertical-align: middle; }
  .item-card { border: 1px solid var(--line); border-radius: 10px; padding: 12px 14px; margin: 10px 0; background: var(--ink-2); }
  .item-card > summary { color: var(--text); font-size: 13px; }
  .item-metric { display: grid; grid-template-columns: 120px 1fr 64px; align-items: center; gap: 10px; padding: 2px 0; }
  .item-metric .lab { font-size: 11px; color: var(--muted); }
  .ans-run { border-left: 3px solid var(--line); padding: 6px 10px; margin: 8px 0; }
  .ans-run .ans-head { font-size: 12px; font-weight: 600; margin-bottom: 4px; display: flex; align-items: center; gap: 6px; }
  .ans-run .ans-body { font-size: 13px; line-height: 1.5; white-space: pre-wrap; color: var(--text); }
  .ans-run .ctx { font-size: 12px; color: var(--muted); white-space: pre-wrap; }
  .metric-seg { display: inline-flex; border: 1px solid var(--line); border-radius: 10px; overflow: hidden; flex-wrap: wrap; }
  .metric-seg button { border: none; border-radius: 0; background: var(--ink-2); color: var(--muted); padding: 8px 12px; font-size: 12px; }
  .metric-seg button.active { color: #06231f; background: linear-gradient(180deg, var(--teal) 0%, var(--teal-deep) 100%); font-weight: 600; }
/* __IW_REPORT_CSS__ */
</style>
</head>
<body>
<header>
  <span class="logo"><span class="i">I</span>nnovation <span class="w">W</span>ays</span>
  <span class="muted">· Compare</span>
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
</nav>

<main class="main-wrap">
  <section id="loginView" class="hidden">
    <div class="card loginwrap">
      <h1>Sign in</h1>
      <p class="sub">Use your demo credentials. The Compare view is read-only and open to any authenticated principal.</p>
      <label for="u">Username</label>
      <input type="text" id="u" autocomplete="username" />
      <label for="p">Password</label>
      <input type="password" id="p" autocomplete="current-password" />
      <div style="margin-top:18px"><button class="btn-primary" id="loginBtn">Sign in</button></div>
    </div>
  </section>

  <section id="appView" class="hidden">

    <div class="card" id="pickCard">
      <h2>Compare runs</h2>
      <p class="sub">Pick <strong>2&ndash;5 runs on the same dataset</strong> and a <strong>baseline</strong>. Every other run is compared item-by-item against the baseline; each run keeps a colour used across every chart below and in the PDF. Picking a run on one dataset disables runs on the others (paired stats need the same golden queries).</p>
      <div id="pickBody"><p class="muted">Loading reports&hellip;</p></div>
      <div class="row" style="margin-top:14px">
        <div style="max-width:280px">
          <label for="baselineSel">Baseline</label>
          <select id="baselineSel"><option value="">(pick runs first)</option></select>
        </div>
        <div style="flex:0 0 auto">
          <button class="btn-primary" id="compareBtn" disabled>Compare</button>
        </div>
      </div>
    </div>

    <div id="compareOut"></div>
  </section>
</main>

<div class="toast" id="toast"></div>

<script>
(function () {
  // Fixed "/api" edge base (Caddy strips /api/*). Mirrors the other console pages.
  var API = "/api";
  var tokenKey = "iw_token", whoKey = "iw_principal", groupsKey = "iw_groups";
  var $ = function (id) { return document.getElementById(id); };
  var esc = function (s) { var d = document.createElement("div"); d.textContent = s == null ? "" : s; return d.innerHTML; };

  function token() { return sessionStorage.getItem(tokenKey); }
  function groups() {
    try { return JSON.parse(sessionStorage.getItem(groupsKey) || "[]"); }
    catch (e) { return []; }
  }
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
    highlightActiveNav();
    loadReports();
  }
  function highlightActiveNav() {
    var path = location.pathname;
    var items = document.querySelectorAll("nav.console-nav a.nav-item");
    items.forEach(function (a) {
      var href = a.getAttribute("href") || "";
      if (href && (path === href || path.indexOf(href + "/") === 0)) { a.classList.add("active"); }
      else { a.classList.remove("active"); }
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

  // ---- run-picker state ----------------------------------------------------
  var REPORTS = [];        // the GET /benchmark/reports rows
  var selected = [];       // selected ids, in selection order
  var MAX_RUNS = 5;
  // ---- runs table: client-side sort + pagination (S01) ----------------------
  var PAGE_SIZE = 20;          // fixed 20 rows per page
  var sortKey = "created_at";  // default sort column
  var sortDir = "desc";        // default sort direction (newest first)
  var page = 1;                // 1-indexed current page
  var DEFAULT_DIR = {          // default direction when first activating a column
    run: "asc",
    dataset: "asc",
    config: "asc",
    created_at: "desc"
  };
  var PICK_COLS = [            // ordered (sort_key, label) pairs for the runs picker header
    { key: "run",        label: "Run" },
    { key: "dataset",    label: "Dataset" },
    { key: "config",     label: "Configuration" },
    { key: "created_at", label: "Created" }
  ];

  function reportById(id) {
    for (var i = 0; i < REPORTS.length; i++) { if (REPORTS[i].id === id) return REPORTS[i]; }
    return null;
  }
  function modsSummary(rep) {
    var c = (rep && rep.config_snapshot) || {};
    var parts = [];
    ["generator", "embedder", "reranker", "grader"].forEach(function (k) {
      if (c[k] != null && c[k] !== "") parts.push(String(c[k]));
    });
    var ds = rep.dataset_name || rep.dataset_id;
    if (ds) parts.push(String(ds));
    return parts.join(" / ");
  }

  async function loadReports() {
    var body = $("pickBody");
    try {
      var r = await fetch(API + "/benchmark/reports", { headers: authHeaders() });
      if (r.status === 401) { setToken(null); showLogin(); return; }
      if (!r.ok) { body.innerHTML = '<div class="caveat-box err"><strong>Reports unavailable (' + r.status + ').</strong></div>'; return; }
      REPORTS = await r.json() || [];
      if (REPORTS.length === 0) {
        body.innerHTML = '<p class="muted">No reports yet — run a benchmark from the Benchmark Runner to create one.</p>';
        return;
      }
      renderPicker();
    } catch (e) {
      body.innerHTML = '<div class="caveat-box err"><strong>Reports load failed:</strong> ' + esc(String(e)) + '</div>';
    }
  }

  /**
   * Return a sorted copy of REPORTS using the current sortKey / sortDir.
   * Sort is applied across the FULL report set before any pagination
   * slicing (so changing page never changes the row order).
   *
   * Performance: the per-row sort key is pre-computed once (the config
   * key in particular calls modsSummary(), which would otherwise be
   * invoked 2× per comparison — O(n² × cost(modsSummary))).  After the
   * precompute the comparator is O(1) per call (S04 polish).
   *
   * @returns {Array<Object>} Sorted copy of REPORTS.
   */
  function sortedReports() {
    var dir = sortDir === "asc" ? 1 : -1;
    var rows = new Array(REPORTS.length);
    for (var i = 0; i < REPORTS.length; i++) {
      var r = REPORTS[i];
      var k = "";
      switch (sortKey) {
        case "run":
          k = String(r.display_id || ("#" + ((r.id || "").slice(0, 8)))).toLowerCase();
          break;
        case "dataset":
          k = String(r.dataset_name || r.dataset_id || "").toLowerCase();
          break;
        case "config":
          // modsSummary memoized once per row here (instead of being
          // recomputed on every a/b comparison — see the docstring above).
          k = modsSummary(r).toLowerCase();
          break;
        case "created_at": {
          var t = Date.parse(r.created_at);
          // "_t:<n>" marks a parseable timestamp (numeric compare in the
          // comparator); raw String(r.created_at || "") is used for invalid
          // timestamps so the string-fallback path matches the original
          // localeCompare-on-raw-strings semantics.
          k = isNaN(t) ? String(r.created_at || "") : "_t:" + t;
          break;
        }
      }
      rows[i] = { rep: r, key: k };
    }
    rows.sort(function (a, b) {
      // created_at: numeric compare when both keys carry a parseable
      // timestamp; otherwise fall back to string compare (matches the
      // original String(created_at || "").localeCompare behaviour).
      if (sortKey === "created_at") {
        if (a.key.indexOf("_t:") === 0 && b.key.indexOf("_t:") === 0) {
          return dir * (Number(a.key.slice(3)) - Number(b.key.slice(3)));
        }
      }
      return dir * a.key.localeCompare(b.key);
    });
    var out = new Array(rows.length);
    for (var j = 0; j < rows.length; j++) { out[j] = rows[j].rep; }
    return out;
  }

  /** Sort indicator glyph for the given column key (▾ default, ▲ asc, ▼ desc).
   *  Requires the source file to be UTF-8 encoded (the Python source emits
   *  these Unicode glyphs as literals — keep `fileencoding=utf-8` in
   *  pyproject.toml so the bytes round-trip through the build). */
  function sortInd(key) {
    if (key !== sortKey) { return "▾"; }
    return sortDir === "asc" ? "▲" : "▼";
  }

  /**
   * Switch to the given sort column; toggle direction if already active.
   * Always resets pagination to page 1 so the user sees the top of the
   * freshly-sorted set first.
   *
   * @param {string} key - The sort key (run, dataset, config, created_at).
   */
  function setSort(key) {
    if (sortKey === key) {
      sortDir = sortDir === "asc" ? "desc" : "asc";
    } else {
      sortKey = key;
      sortDir = DEFAULT_DIR[key] || "asc";
    }
    page = 1;
    renderPicker();
  }

  /**
   * Build the pagination bar markup for the runs picker.  Returns an empty
   * string when the row count is at or below PAGE_SIZE (no pager needed).
   *
   * @param {number} total - The total row count across all pages.
   * @returns {string} HTML string (may be empty).
   */
  function renderPickerPager(total) {
    if (total <= PAGE_SIZE) { return ''; }
    var totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE));
    var from = (page - 1) * PAGE_SIZE + 1;
    var to = Math.min(page * PAGE_SIZE, total);
    return '<div class="pick-pager">' +
      '<span class="pick-page-info">Page ' + page + ' of ' + totalPages + ' &middot; ' + from + '&ndash;' + to + ' of ' + total + '</span>' +
      '<button type="button" id="pickPrev">&#8249; Prev</button>' +
      '<button type="button" id="pickNext">Next &#8250;</button>' +
      '</div>';
  }

  function renderPicker() {
    // Sort the full REPORTS list first, then slice to the current page
    // (sort runs across the whole set — never just the visible page).
    var sorted = sortedReports();
    var total = sorted.length;
    var totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE));
    if (page < 1) { page = 1; }
    if (page > totalPages) { page = totalPages; }
    var start = (page - 1) * PAGE_SIZE;
    var slice = sorted.slice(start, start + PAGE_SIZE);

    // Build the sortable <thead>: every data column gets a .sortable th
    // with a data-sort key + a sort indicator glyph.
    var headCells = '<th class="c">pick</th>';
    PICK_COLS.forEach(function (c) {
      var cls = "sortable" + (sortKey === c.key ? " " + sortDir : "");
      headCells += '<th class="' + cls + '" data-sort="' + c.key + '">' + c.label +
        '<span class="sort-ind">' + sortInd(c.key) + '</span></th>';
    });

    var html = '<div class="hm-scroll"><table class="pick-table"><thead><tr>' +
      headCells + '</tr></thead><tbody>';
    slice.forEach(function (rep) {
      var ds = rep.dataset_name || rep.dataset_id || "—";
      var runId = rep.display_id || ("#" + (rep.id || "").slice(0, 8));
      var sel = selected.indexOf(rep.id) !== -1;
      html += '<tr>' +
        '<td class="c"><input type="checkbox" class="pickbox" data-id="' + esc(rep.id) + '" data-ds="' + esc(rep.dataset_id || "") + '"' + (sel ? ' checked' : '') + ' /></td>' +
        '<td><span style="font-family:JetBrains Mono,monospace">' + esc(runId) + '</span></td>' +
        '<td>' + esc(ds) + '</td>' +
        '<td><span class="mods" style="font-size:11px;color:var(--muted)">' + esc(modsSummary(rep)) + '</span></td>' +
        '<td>' + esc(rep.created_at || "—") + '</td>' +
        '</tr>';
    });
    html += '</tbody></table></div>';
    html += renderPickerPager(total);
    $("pickBody").innerHTML = html;

    // Wire up the per-row checkbox change handler (restore checked state
    // above so this only fires on user-initiated clicks).
    var boxes = document.querySelectorAll(".pickbox");
    boxes.forEach(function (b) { b.addEventListener("change", onPick); });

    // Wire up the sortable header click handlers.
    var ths = document.querySelectorAll(".pick-table th.sortable");
    ths.forEach(function (th) {
      th.addEventListener("click", function () { setSort(th.getAttribute("data-sort")); });
    });

    // Wire up the pager buttons (only present when the picker overflows one page).
    var prev = $("pickPrev"), next = $("pickNext");
    if (prev) {
      prev.disabled = page <= 1;
      prev.addEventListener("click", function () { if (page > 1) { page--; renderPicker(); } });
    }
    if (next) {
      next.disabled = page >= totalPages;
      next.addEventListener("click", function () { if (page < totalPages) { page++; renderPicker(); } });
    }

    syncPickerState();
  }

  function onPick(ev) {
    var id = ev.target.getAttribute("data-id");
    if (ev.target.checked) {
      if (selected.indexOf(id) === -1) selected.push(id);
    } else {
      selected = selected.filter(function (x) { return x !== id; });
    }
    syncPickerState();
  }

  function syncPickerState() {
    // Determine the locked dataset (the dataset of the first selected run).
    var lockDs = null;
    if (selected.length) {
      var first = reportById(selected[0]);
      lockDs = first ? (first.dataset_id || "") : null;
    }
    var boxes = document.querySelectorAll(".pickbox");
    boxes.forEach(function (b) {
      var id = b.getAttribute("data-id");
      var ds = b.getAttribute("data-ds");
      var isSel = selected.indexOf(id) !== -1;
      var disabled = false;
      if (!isSel) {
        if (lockDs !== null && ds !== lockDs) disabled = true;     // different dataset
        if (selected.length >= MAX_RUNS) disabled = true;          // cap reached
      }
      b.disabled = disabled;
    });
    // Baseline select — populated from the current selection (display ids).
    var sel = $("baselineSel");
    var prev = sel.value;
    if (!selected.length) {
      sel.innerHTML = '<option value="">(pick runs first)</option>';
    } else {
      var opts = "";
      selected.forEach(function (id) {
        var rep = reportById(id);
        var lbl = rep ? (rep.display_id || ("#" + id.slice(0, 8))) : id;
        opts += '<option value="' + esc(id) + '">' + esc(lbl) + '</option>';
      });
      sel.innerHTML = opts;
      if (selected.indexOf(prev) !== -1) sel.value = prev;
    }
    $("compareBtn").disabled = selected.length < 2;
  }

  // ---- compare + render ----------------------------------------------------
  async function runCompare() {
    if (selected.length < 2) { toast("Pick at least 2 runs", true); return; }
    var baseline = $("baselineSel").value || selected[0];
    var out = $("compareOut");
    out.innerHTML = '<div class="card"><p class="muted">Comparing&hellip;</p></div>';
    var url = API + "/benchmark/compare/multi?ids=" + encodeURIComponent(selected.join(",")) +
      "&baseline=" + encodeURIComponent(baseline);
    try {
      var r = await fetch(url, { headers: authHeaders() });
      if (r.status === 401) { setToken(null); showLogin(); return; }
      if (r.status === 404) { out.innerHTML = '<div class="card"><div class="caveat-box err"><strong>Not found.</strong> One of the selected reports is unknown.</div></div>'; return; }
      if (r.status === 400) {
        var ej = {}; try { ej = await r.json(); } catch (x) {}
        out.innerHTML = '<div class="card"><div class="caveat-box"><strong>Cannot compare.</strong> ' + esc(ej.detail || ej.error || "The runs are not comparable (same dataset + 2–5 runs required).") + '</div></div>';
        return;
      }
      if (!r.ok) { out.innerHTML = '<div class="card"><div class="caveat-box err"><strong>Compare failed (' + r.status + ').</strong></div></div>'; return; }
      var data = await r.json();
      IWC_renderAll(data);
    } catch (e) {
      out.innerHTML = '<div class="card"><div class="caveat-box err"><strong>Compare failed:</strong> ' + esc(String(e)) + '</div></div>';
    }
  }

  function IWC_hexToRgba(hex, alpha) {
    var h = String(hex || "#2DD4BF").replace("#", "");
    if (h.length === 3) { h = h[0] + h[0] + h[1] + h[1] + h[2] + h[2]; }
    var r = parseInt(h.slice(0, 2), 16), g = parseInt(h.slice(2, 4), 16), b = parseInt(h.slice(4, 6), 16);
    if (isNaN(r) || isNaN(g) || isNaN(b)) { r = 45; g = 212; b = 191; }
    var a = Math.max(0, Math.min(1, alpha == null ? 1 : alpha));
    return "rgba(" + r + "," + g + "," + b + "," + a.toFixed(3) + ")";
  }
  function IWC_metricLabel(m) {
    var g = (typeof IW_GLOSSARY !== "undefined") ? IW_GLOSSARY[m] : null;
    return g ? g.label : m;
  }
  function IWC_tip(m) {
    return (typeof IW_GLOSSARY !== "undefined" && IW_GLOSSARY[m] && typeof IW_tip === "function") ? IW_tip(m) : "";
  }

  var IWC_DATA = null;       // last compare payload (for the PDF button)
  var IWC_HM_METRIC = null;  // currently selected heatmap metric

  function IWC_renderAll(data) {
    IWC_DATA = data;
    IWC_HM_METRIC = (data.metrics && data.metrics.length) ? data.metrics[0] : null;
    var html = "";
    html += IWC_renderLegendCard(data);
    html += IWC_renderConclusionCard();
    if (data.underpowered || data.caveat) {
      html += '<div class="card"><div class="caveat-box"><strong>Underpowered.</strong> ' +
        esc(data.caveat || "Small sample — read deltas with care.") + '</div></div>';
    }
    html += IWC_renderAggregateCard(data);
    html += IWC_renderDeltasCard(data);
    html += IWC_renderItemsCard(data);
    $("compareOut").innerHTML = html;
    IWC_renderHeatmap();
    IWC_fetchConclusion(false);
  }

  // Local-LLM conclusion + reasoning — lazy-loaded after the tables render so a
  // slow judge call never blocks the comparison itself. Cached server-side, so
  // the PDF export shows the same narrative; Regenerate forces a fresh one.
  function IWC_renderConclusionCard() {
    return '<div class="card" id="conclCard"><h2>Conclusion &amp; reasoning</h2>' +
      '<p class="sub">A local-LLM verdict grounded in the numbers above — never a hosted API.</p>' +
      '<div id="conclBody"><p class="muted">Generating a local-LLM conclusion&hellip;</p></div>' +
      '<div style="margin-top:8px"><button class="btn-ghost" id="conclRegen">&#8635; Regenerate</button></div></div>';
  }

  async function IWC_fetchConclusion(refresh) {
    if (!IWC_DATA) return;
    var body = $("conclBody");
    if (!body) return;
    body.innerHTML = '<p class="muted">Generating a local-LLM conclusion&hellip;</p>';
    var baseline = $("baselineSel").value || selected[0];
    var url = API + "/benchmark/compare/multi/conclusion?ids=" + encodeURIComponent(selected.join(",")) +
      "&baseline=" + encodeURIComponent(baseline) + (refresh ? "&refresh=true" : "");
    try {
      var r = await fetch(url, { headers: authHeaders() });
      if (!r.ok) { body.innerHTML = '<p class="muted">Conclusion unavailable (' + r.status + ').</p>'; return; }
      var d = await r.json();
      if (d && d.conclusion) {
        body.innerHTML = '<div class="iw-concl">' + esc(d.conclusion) + '</div>';
      } else {
        body.innerHTML = '<p class="muted">No conclusion — the local judge model is unavailable in this deployment.</p>';
      }
    } catch (e) {
      body.innerHTML = '<p class="muted">Conclusion failed: ' + esc(String(e)) + '</p>';
    }
  }

  function IWC_renderLegendCard(data) {
    var runs = data.runs || [];
    var chips = "";
    runs.forEach(function (run) {
      chips += '<span class="runchip">' +
        '<span class="sw" style="background:' + esc(run.color) + '"></span>' +
        '<span>' + esc(run.display_id || ("#" + (run.id || "").slice(0, 8))) + '</span>' +
        (run.is_baseline ? '<span class="bl">baseline</span>' : '') +
        '<span class="mods">' + esc(IWC_runMods(run)) + '</span>' +
        '</span>';
    });
    var dsName = "";
    if (runs.length && (runs[0].dataset_name || data.dataset_id)) dsName = runs[0].dataset_name || data.dataset_id;
    return '<div class="card"><h2>Comparing ' + runs.length + ' runs' + (dsName ? ' · ' + esc(dsName) : '') + '</h2>' +
      '<p class="sub">n = ' + esc(data.n_queries) + ' queries. Colours are stable across every chart below and the PDF.</p>' +
      '<div class="legend-wrap">' + chips + '</div>' +
      '<div style="margin-top:10px"><button class="btn-primary" id="pdfBtn">&#128196; Generate Report (PDF)</button></div>' +
      '</div>';
  }
  function IWC_runMods(run) {
    var c = run.config_snapshot || {};
    var parts = [];
    ["generator", "embedder", "reranker", "grader"].forEach(function (k) {
      if (c[k] != null && c[k] !== "") parts.push(String(c[k]));
    });
    return parts.join(" / ");
  }

  function IWC_renderAggregateCard(data) {
    var runs = data.runs || [];
    var metrics = data.metrics || [];
    var body = "";
    metrics.forEach(function (m) {
      var rows = "";
      runs.forEach(function (run) {
        var v = (run.means || {})[m];
        var pct = (v == null || isNaN(v)) ? 0 : Math.max(0, Math.min(1, Number(v))) * 100;
        var valStr = (v == null || isNaN(v)) ? "—" : Number(v).toFixed(3);
        rows += '<div class="agg-bar-row">' +
          '<div class="agg-bar-name" title="' + esc(run.display_id) + '">' + esc(run.display_id || ("#" + (run.id || "").slice(0, 8))) + '</div>' +
          '<div class="agg-bar-track"><div class="agg-bar-fill" style="width:' + pct.toFixed(1) + '%;background:' + esc(run.color) + '"></div></div>' +
          '<div class="agg-bar-val">' + valStr + '</div>' +
          '</div>';
      });
      body += '<div class="agg-metric"><div class="ttl">' + esc(IWC_metricLabel(m)) + IWC_tip(m) + '</div>' + rows + '</div>';
    });
    return '<div class="card"><h2>Aggregate — mean per metric</h2>' +
      '<p class="sub">Each run\'s mean for every metric (0–1, higher is better). One coloured bar per run.</p>' + body + '</div>';
  }

  function IWC_renderDeltasCard(data) {
    var runs = data.runs || [];
    var baseId = data.baseline_id;
    var baseLabel = "";
    runs.forEach(function (run) { if (run.id === baseId) baseLabel = run.display_id || ("#" + run.id.slice(0, 8)); });
    var blocks = "";
    runs.forEach(function (run) {
      if (run.is_baseline) return;
      blocks += '<h3 style="font-size:13px;margin:14px 0 6px;display:flex;align-items:center">' +
        '<span class="col-sw" style="background:' + esc(run.color) + '"></span>' +
        esc(run.display_id || ("#" + run.id.slice(0, 8))) + ' vs baseline</h3>';
      var rows = run.deltas || [];
      if (!rows.length) { blocks += '<p class="muted">No shared per-query metrics with the baseline.</p>'; return; }
      blocks += '<table class="iw-metrics"><thead><tr><th>metric</th><th class="num">baseline</th><th class="num">this run</th><th class="num">delta</th><th class="num">p_corrected</th><th class="num">CI 95%</th><th>significant</th></tr></thead><tbody>';
      rows.forEach(function (row) {
        var chip = row.significant ? '<span class="chip ok">✓ significant</span>' : '<span class="chip muted">not significant</span>';
        var ci = row.ci && row.ci.length === 2 ? ("[" + IW_num(row.ci[0], 3) + ", " + IW_num(row.ci[1], 3) + "]") : "—";
        var gk = (typeof IW_GLOSSARY !== "undefined" && IW_GLOSSARY[row.metric]) ? IW_tip(row.metric) : "";
        var dcls = (row.delta == null) ? "" : (row.delta >= 0 ? "iw-v-good" : "iw-v-low");
        blocks += '<tr><td class="lab">' + esc(row.metric) + gk + '</td>' +
          '<td class="num">' + IW_num(row.val_a, 3) + '</td>' +
          '<td class="num">' + IW_num(row.val_b, 3) + '</td>' +
          '<td class="num ' + dcls + '">' + (row.delta == null ? "—" : (row.delta >= 0 ? "+" : "") + IW_num(row.delta, 3)) + '</td>' +
          '<td class="num">' + IW_num(row.p_corrected, 4) + '</td>' +
          '<td class="num">' + ci + '</td><td>' + chip + '</td></tr>';
      });
      blocks += '</tbody></table>';
    });
    return '<div class="card"><h2>Baseline-relative deltas</h2>' +
      '<p class="sub">Every other run vs the baseline <strong>' + esc(baseLabel) + '</strong>. Delta = run − baseline (positive ⇒ the run beats the baseline). Paired Wilcoxon + BCa 95% CI, Bonferroni-Holm corrected.</p>' + blocks + '</div>';
  }

  function IWC_renderItemsCard(data) {
    var metrics = data.metrics || [];
    var seg = "";
    metrics.forEach(function (m) {
      seg += '<button data-hm="' + esc(m) + '"' + (m === IWC_HM_METRIC ? ' class="active"' : '') + '>' + esc(IWC_metricLabel(m)) + '</button>';
    });
    return '<div class="card"><h2>Per-item comparison</h2>' +
      '<p class="sub">Every golden item across all runs. The heatmap colours each cell by the run\'s value (deeper = higher). Expand an item for per-metric bars and each run\'s answer + retrieved contexts.</p>' +
      '<div class="metric-seg" id="hmSeg">' + seg + '</div>' +
      '<div class="hm-scroll" id="hmWrap" style="margin-top:12px"></div>' +
      '<h3 style="font-size:13px;margin:18px 0 6px">Item detail</h3>' +
      '<div id="itemList"></div>' +
      '</div>';
  }

  function IWC_renderHeatmap() {
    var data = IWC_DATA; if (!data) return;
    var runs = data.runs || [];
    var items = data.items || [];
    var m = IWC_HM_METRIC;
    var head = '<table class="heatmap"><thead><tr><th class="q">item</th>';
    runs.forEach(function (run) {
      head += '<th><span class="col-sw" style="background:' + esc(run.color) + '"></span>' + esc(run.display_id || ("#" + run.id.slice(0, 8))) + '</th>';
    });
    head += '</tr></thead><tbody>';
    var rows = "";
    items.forEach(function (it) {
      var qlabel = it.question ? it.question : ("Item " + (it.index + 1));
      if (qlabel.length > 90) qlabel = qlabel.slice(0, 90) + "…";
      rows += '<tr><td class="q" title="' + esc(it.question || "") + '">' + esc(qlabel) + '</td>';
      var vals = (it.values && it.values[m]) || [];
      runs.forEach(function (run, ri) {
        var v = vals[ri];
        if (v == null || isNaN(v)) {
          rows += '<td class="muted">—</td>';
        } else {
          var bg = IWC_hexToRgba(run.color, 0.15 + 0.75 * Math.max(0, Math.min(1, Number(v))));
          rows += '<td class="cellv" style="background:' + bg + '">' + Number(v).toFixed(2) + '</td>';
        }
      });
      rows += '</tr>';
    });
    $("hmWrap").innerHTML = head + rows + '</tbody></table>';
    IWC_renderItemList();
  }

  function IWC_renderItemList() {
    var data = IWC_DATA; if (!data) return;
    var runs = data.runs || [];
    var metrics = data.metrics || [];
    var items = data.items || [];
    var html = "";
    items.forEach(function (it) {
      var qlabel = it.question ? it.question : ("Item " + (it.index + 1));
      var summ = "Item " + (it.index + 1) + " — " + (qlabel.length > 120 ? qlabel.slice(0, 120) + "…" : qlabel);
      var inner = "";
      // per-metric grouped bars
      metrics.forEach(function (m) {
        var vals = (it.values && it.values[m]) || [];
        var bars = "";
        runs.forEach(function (run, ri) {
          var v = vals[ri];
          var pct = (v == null || isNaN(v)) ? 0 : Math.max(0, Math.min(1, Number(v))) * 100;
          bars += '<div class="item-metric"><div class="lab">' + esc(run.display_id || ("#" + run.id.slice(0, 8))) + '</div>' +
            '<div class="agg-bar-track"><div class="agg-bar-fill" style="width:' + pct.toFixed(1) + '%;background:' + esc(run.color) + '"></div></div>' +
            '<div class="agg-bar-val">' + (v == null || isNaN(v) ? "—" : Number(v).toFixed(3)) + '</div></div>';
        });
        inner += '<div style="margin:8px 0"><div class="lab" style="font-size:12px;color:var(--text);margin-bottom:2px">' + esc(IWC_metricLabel(m)) + '</div>' + bars + '</div>';
      });
      // per-run answers + contexts
      var anyAns = false;
      var ans = "";
      runs.forEach(function (run, ri) {
        var s = (it.samples || [])[ri];
        if (!s) return;
        anyAns = true;
        var ctxHtml = "";
        if (s.contexts && s.contexts.length) {
          var ctxs = s.contexts.map(function (c, ci) { return "[" + (ci + 1) + "] " + c; }).join("\n\n");
          ctxHtml = '<details><summary>' + s.contexts.length + ' retrieved context' + (s.contexts.length === 1 ? '' : 's') + '</summary><div class="ctx">' + esc(ctxs) + '</div></details>';
        }
        ans += '<div class="ans-run" style="border-left-color:' + esc(run.color) + '">' +
          '<div class="ans-head"><span class="col-sw" style="background:' + esc(run.color) + '"></span>' + esc(run.display_id || ("#" + run.id.slice(0, 8))) + '</div>' +
          '<div class="ans-body">' + (s.answer ? esc(s.answer) : '<span class="muted">no answer recorded</span>') + '</div>' +
          ctxHtml + '</div>';
      });
      if (anyAns) {
        inner += '<h4 style="font-size:12px;color:var(--muted);margin:12px 0 2px;text-transform:uppercase;letter-spacing:.5px">Answers &amp; contexts</h4>' + ans;
      } else {
        inner += '<p class="muted">No per-run answers stored for this item (runs predate sample capture).</p>';
      }
      html += '<details class="item-card"><summary>' + esc(summ) + '</summary>' + inner + '</details>';
    });
    $("itemList").innerHTML = html;
  }

  function downloadPdf(btn) {
    if (!IWC_DATA) return;
    var baseline = $("baselineSel").value || selected[0];
    var url = API + "/benchmark/compare/multi/pdf?ids=" + encodeURIComponent(selected.join(",")) +
      "&baseline=" + encodeURIComponent(baseline);
    var orig = btn.innerHTML;
    btn.disabled = true; btn.innerHTML = "Generating PDF&hellip;";
    fetch(url, { headers: authHeaders() })
      .then(function (r) {
        if (!r.ok) return r.json().then(function (j) { throw new Error(j.detail || j.error || ("HTTP " + r.status)); });
        return r.blob();
      })
      .then(function (blob) {
        var u = URL.createObjectURL(blob);
        var a = document.createElement("a");
        a.href = u; a.download = "benchmark-comparison.pdf";
        document.body.appendChild(a); a.click(); a.remove();
        URL.revokeObjectURL(u);
        btn.disabled = false; btn.innerHTML = orig;
      })
      .catch(function (err) { toast("PDF failed: " + err, true); btn.disabled = false; btn.innerHTML = orig; });
  }

  // ---- wire up -------------------------------------------------------------
  $("loginBtn").addEventListener("click", login);
  $("p").addEventListener("keydown", function (e) { if (e.key === "Enter") login(); });
  $("logout").addEventListener("click", function () { setToken(null); showLogin(); });
  $("compareBtn").addEventListener("click", runCompare);

  // Delegated clicks inside the rendered output (PDF button + heatmap metric segs).
  $("compareOut").addEventListener("click", function (e) {
    var pdfBtn = e.target.closest ? e.target.closest("#pdfBtn") : null;
    if (pdfBtn) { downloadPdf(pdfBtn); return; }
    var regen = e.target.closest ? e.target.closest("#conclRegen") : null;
    if (regen) { IWC_fetchConclusion(true); return; }
    var seg = e.target.closest ? e.target.closest("[data-hm]") : null;
    if (seg) {
      IWC_HM_METRIC = seg.getAttribute("data-hm");
      var segs = document.querySelectorAll("#hmSeg [data-hm]");
      segs.forEach(function (s) { s.classList.toggle("active", s.getAttribute("data-hm") === IWC_HM_METRIC); });
      IWC_renderHeatmap();
    }
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


def compare_page_html() -> str:
    """Return the standalone multi-run Compare view HTML page.

    A self-contained inline HTML+CSS+vanilla-JS page mirroring
    :mod:`iw_rag.api.results_page`'s chrome and reusing the shared
    :mod:`iw_rag.api.benchmark_render` fragments (``TOOLTIP_CSS`` /
    ``REPORT_CSS`` / ``REPORT_JS`` — the metric glossary, escaping,
    number formatting, value bars and score-colour helpers).  It carries
    the IW brand chrome, the location-derived API base, the
    sessionStorage-only bearer token, the ``/auth/login`` self-handled
    sign-in, the shared console nav (with the new Compare tab), the run
    picker (2..5 same-dataset runs + baseline), and the comparison render
    (legend, aggregate bars, baseline-relative deltas, per-item heatmap +
    expandable answer/context diff) plus the Generate Report (PDF) button.

    Returns:
        The raw, self-contained HTML document as a ``str``.
    """
    page = _PAGE.replace("/* __IW_REPORT_CSS__ */", TOOLTIP_CSS + REPORT_CSS).replace(
        "/* __IW_REPORT_JS__ */", REPORT_JS
    )
    return add_favicon(page)


__all__ = ["compare_page_html"]
