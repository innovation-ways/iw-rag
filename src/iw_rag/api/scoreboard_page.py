# ruff: noqa: E501
"""Self-contained, branded IW RAG Operator Console Scoreboard view (F-00096, S14).

The Scoreboard view is the Operator Console page that ranks the
append-only ``benchmark_reports`` history into per-metric leaderboards
(``faithfulness`` / ``ndcg_at_5`` / ``tok_per_s`` / ``latency`` /
``vram_efficiency`` / ``composite``).  It is a **thin API client** —
Invariant 4: it contains NO ranking logic, NO retrieval, NO ACL, NO
grounding, NO guard logic.  All such logic stays server-side.  The
S05 :class:`~iw_rag.eval.benchmark.scoreboard.ScoreboardService` is
the **sole** authority on the per-row ordering (AC6: descending
faithfulness for the ``faithfulness`` board); the page only renders
what ``GET /benchmark/scoreboard`` returns.

**AC6** — when the operator opens the Scoreboard and picks the
``faithfulness`` board, reports are listed in descending faithfulness
order with ``model_config`` + run timestamp.  Each of the six
leaderboards surfaces the same shape (rank, model_config summary,
``created_at`` run timestamp, the board's metric value).

**Boundary row coverage** — empty report history (0
``benchmark_reports`` rows) renders the explicit *"No benchmark
reports yet — run a benchmark from the Benchmark Runner to populate
the leaderboards."* empty state, not an error.

It is a **server-rendered FastAPI HTML page** in the
``benchmark_page.py`` / ``console_page.py`` / ``config_page.py`` /
``benchmark_runner_page.py`` / ``results_page.py`` style — NOT
Chainlit, NOT a JS SPA, NO new build toolchain (per the F-00096
design *Out-of-Scope* row "A JavaScript SPA / new build toolchain").
The page derives its API base from its own URL so it works behind
Caddy (``/api``) or when hit directly (``/``).  No external CDN /
Google Fonts link (Invariant 8, local-only rule).

Mirrors :mod:`iw_rag.api.benchmark_page`,
:mod:`iw_rag.api.console_page`, :mod:`iw_rag.api.config_page`,
:mod:`iw_rag.api.benchmark_runner_page`, and
:mod:`iw_rag.api.results_page` — same header/footer chrome, same
teal/ink palette, same location-derived API base regex idiom with
``\\/`` (the forward-slash escape that keeps the JS parser from
closing the literal at the first ``/``), same sessionStorage-only
bearer token, same raw-string ``_PAGE`` whose inline ``<script>``
passes ``node --check``.  The S14 route ``GET /console/scoreboard``
is wired against :func:`scoreboard_page_html`.

**Sortable table pattern** — the board table reuses the
``documents_page.py`` ``th.sortable`` / ``setSort`` /
``refreshHeaderIndicators`` pattern verbatim.  Column-header
client-side sorting is permitted for **display convenience** over
the already-ranked rows (the server's ranking is the authoritative
default; the page never recomputes ranking from raw metrics —
Invariant 4).
"""

from __future__ import annotations

from iw_rag.api.benchmark_render import TOOLTIP_CSS
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
  .chip.warn { color: var(--warn); border-color: #5b4322; }
  .chip.muted { color: var(--muted); }
  .toast { position: fixed; bottom: 22px; left: 50%; transform: translateX(-50%); background: var(--ink-3); border: 1px solid var(--line); padding: 12px 18px; border-radius: 10px; opacity: 0; transition: .25s; pointer-events: none; max-width: 90vw; z-index: 10; }
  .toast.show { opacity: 1; }
  .toast.err { border-color: var(--danger); color: #ffd7de; }
  .hidden { display: none !important; }
  .muted { color: var(--muted); font-size: 13px; }
  .loginwrap { max-width: 420px; margin: 8vh auto; }
  /* Board picker: segmented control, mirrors the documents_page.py .seg
     idiom — one board active at a time, the active tab carries the
     teal-on-ink gradient. */
  .seg { display: inline-flex; border: 1px solid var(--line); border-radius: 10px; overflow: hidden; flex-wrap: wrap; }
  .seg label { margin: 0; padding: 9px 14px; cursor: pointer; color: var(--muted); background: var(--ink-2); font-size: 13px; }
  .seg input { display: none; }
  .seg input:checked + span { color: #06231f; font-weight: 600; }
  .seg label:has(input:checked) { background: linear-gradient(180deg, var(--teal) 0%, var(--teal-deep) 100%); }
  /* Sortable column headers: clickable, with a directional indicator on
     the active column. Mirrors the documents_page.py th.sortable
     pattern verbatim so a future change to one is mirrored on the
     other. */
  th.sortable { cursor: pointer; -webkit-user-select: none; user-select: none; }
  th.sortable:hover { color: var(--teal); }
  th.sortable .sort-ind { display: inline-block; width: 12px; margin-left: 4px; color: var(--teal); font-size: 10px; }
  th.sortable:not(.asc):not(.desc) .sort-ind { color: var(--muted); opacity: .55; }
  nav.console-nav { display: flex; gap: 6px; flex-wrap: wrap; align-items: center; padding: 14px 28px; background: rgba(24,29,41,0.5); border-bottom: 1px solid var(--line); }
  nav.console-nav a.nav-item { font-size: 13px; padding: 6px 12px; border-radius: 8px; border: 1px solid transparent; color: var(--muted); }
  nav.console-nav a.nav-item:hover { color: var(--text); border-color: var(--line); }
  nav.console-nav a.nav-item.active { color: #06231f; background: linear-gradient(180deg, var(--teal) 0%, var(--teal-deep) 100%); border-color: transparent; font-weight: 600; }
  .board-meta { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; margin: 6px 0 12px; }
  .board-meta .muted { font-size: 12px; }
  .caveat-box { background: rgba(245,158,11,0.10); border: 1px solid #5b4322; color: #fde7c0; padding: 12px 14px; border-radius: 10px; margin: 12px 0; font-size: 13px; line-height: 1.5; }
  .caveat-box.err { background: rgba(244,63,94,0.10); border-color: #5b2230; color: #ffd7de; }
  .model-config { font-size: 12px; color: var(--muted); font-family: "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
  .model-config .k { color: var(--teal); }
  .run-id { font-family: "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; color: var(--text); white-space: nowrap; }
  .vh-unit { color: var(--muted); font-weight: 400; text-transform: none; letter-spacing: 0; font-size: 11px; }
  .board-help { font-size: 13px; color: var(--muted); line-height: 1.55; margin: 0 0 6px; }
/* __IW_TOOLTIP_CSS__ */
</style>
</head>
<body>
<header>
  <span class="logo"><span class="i">I</span>nnovation <span class="w">W</span>ays</span>
  <span class="muted">· Scoreboard</span>
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

<main>
  <section id="loginView" class="hidden">
    <div class="card loginwrap">
      <h1>Sign in</h1>
      <p class="sub">Use your demo credentials. The Scoreboard ranks the append-only <code>benchmark_reports</code> history; the ranking is computed server-side (S05) and is the sole authority on the per-row ordering (AC6).</p>
      <label for="u">Username</label>
      <input type="text" id="u" autocomplete="username" />
      <label for="p">Password</label>
      <input type="password" id="p" autocomplete="current-password" />
      <div style="margin-top:18px"><button class="btn-primary" id="loginBtn">Sign in</button></div>
    </div>
  </section>

  <section id="appView" class="hidden">

    <div class="card" id="boardCard">
      <h2>Leaderboard</h2>
      <p class="sub">Pick a board. The default order is the server-provided ranking (S05) — the page only renders, never recomputes (Invariant 4). Column-header sorting is display convenience over the already-ranked rows.</p>

      <div class="seg" id="boardPicker" aria-label="Scoreboard boards">
        <label><input type="radio" name="board" value="faithfulness" checked data-board="faithfulness" /><span>faithfulness</span></label>
        <label><input type="radio" name="board" value="ndcg_at_5" data-board="ndcg_at_5" /><span>ndcg_at_5</span></label>
        <label><input type="radio" name="board" value="tok_per_s" data-board="tok_per_s" /><span>tok/s</span></label>
        <label><input type="radio" name="board" value="latency" data-board="latency" /><span>latency</span></label>
        <label><input type="radio" name="board" value="vram_efficiency" data-board="vram_efficiency" /><span>VRAM-efficiency</span></label>
        <label><input type="radio" name="board" value="composite" data-board="composite" /><span>composite</span></label>
      </div>

      <p class="board-help" id="boardHelp"></p>
      <div class="board-meta">
        <span class="muted" id="boardDesc">Ranking by <strong>faithfulness</strong> (descending — best first).</span>
        <span class="chip muted" id="boardCount">0 reports</span>
      </div>
    </div>

    <div class="card" id="boardTableCard">
      <h2 id="boardTitle">Faithfulness leaderboard</h2>
      <table>
        <thead><tr>
          <th class="sortable num" data-sort="rank" scope="col">Rank<span class="sort-ind">▾</span></th>
          <th scope="col">Run</th>
          <th class="sortable" data-sort="config_snapshot" scope="col">Configuration<span class="sort-ind">▾</span></th>
          <th class="sortable" data-sort="created_at" scope="col">Run timestamp<span class="sort-ind">▾</span></th>
          <th class="sortable num desc" data-sort="metric_value" scope="col" id="valueHeader">Value<span class="sort-ind">▼</span></th>
        </tr></thead>
        <tbody id="rows"></tbody>
      </table>
      <div id="empty" class="empty hidden">No benchmark reports yet &mdash; run a benchmark from the Benchmark Runner to populate the leaderboards.</div>
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

  // -------------------------------------------------------------------------
  // Board catalogue — the six S05 leaderboards (per-metric + composite).
  // The S05 service is the sole authority on the per-row ordering; the
  // page renders the server's order verbatim.  The "direction" hint is
  // informational only — it appears in the "Ranking by ..." line, so
  // the operator knows whether the board is "best first" (descending)
  // or "fastest first" (ascending).  Latency is the only ascending
  // board; the rest are descending (S05 service contract).
  // -------------------------------------------------------------------------
  var BOARDS = {
    faithfulness:     { label: "faithfulness",     direction: "desc", unit: "score · 0–1",        higher: true,  title: "Faithfulness leaderboard",
      what: "Ranks each run by mean faithfulness — of the claims an answer made, the fraction actually supported by the retrieved context. The single best signal for \"does this configuration hallucinate?\" Higher is better." },
    ndcg_at_5:        { label: "ndcg_at_5",        direction: "desc", unit: "score · 0–1",        higher: true,  title: "nDCG@5 leaderboard",
      what: "Ranks by retrieval ranking quality at cut-off 5 (nDCG@5) — how well the most relevant passages land near the top of the results. Higher is better." },
    tok_per_s:        { label: "tok/s",            direction: "desc", unit: "tokens / second",   higher: true,  title: "Throughput leaderboard",
      what: "Ranks by generation throughput: output tokens per second, derived from the vLLM token counters and the per-query retrieve+generate time. Higher is faster." },
    latency:          { label: "latency",          direction: "asc",  unit: "seconds (e2e p99)", higher: false, title: "Latency leaderboard",
      what: "Ranks by end-to-end p99 latency in seconds reported by the vLLM server — the worst-case round-trip. Lower is better, so the fastest run is listed first." },
    vram_efficiency:  { label: "VRAM-efficiency",  direction: "desc", unit: "nDCG@5 per GiB",     higher: true,  title: "VRAM-efficiency leaderboard",
      what: "Ranks by retrieval quality per gigabyte of resident GPU memory (nDCG@5 ÷ peak VRAM in GiB) — quality you get for the memory you spend. Higher is better." },
    composite:        { label: "composite",        direction: "desc", unit: "blended score · 0–1", higher: true,  title: "Composite leaderboard",
      what: "A weighted blend of all five boards — faithfulness 30%, nDCG@5 20%, throughput 20%, latency 15%, VRAM-efficiency 15% — each min-max normalised across the cohort. A single \"all-round best\" number. Higher is better." }
  };
  var DEFAULT_BOARD = "faithfulness";

  // -------------------------------------------------------------------------
  // Hover glossary helper (shares the .iw-tip styling injected from
  // benchmark_render.TOOLTIP_CSS).  The Scoreboard has no metric blob to
  // render, so it carries its own tiny tooltip builder rather than the
  // full report renderer.
  // -------------------------------------------------------------------------
  function IW_tip(title, unit, body) {
    return '<span class="iw-tip" tabindex="0" aria-label="' + esc(title) + '">i' +
      '<span class="iw-bub"><span class="t">' + esc(title) + '</span>' +
      (unit ? '<span class="u">' + esc(unit) + '</span>' : '') + esc(body) + '</span></span>';
  }

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
    loadBoard(currentBoard());
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
  // Empty-state anchor — Boundary row "Empty report history".  The same
  // copy is shown in the board table; the page is rendered (not an
  // error) so the operator sees the call-to-action.
  // -------------------------------------------------------------------------
  var EMPTY_HISTORY_HTML =
    '<p class="muted">No benchmark reports yet — run a benchmark from the Benchmark Runner to populate the leaderboards.</p>';

  // -------------------------------------------------------------------------
  // Active board — read from the segmented control.  The default is
  // "faithfulness" (AC6 — the Feature Design's primary acceptance
  // scenario).
  // -------------------------------------------------------------------------
  function currentBoard() {
    var el = document.querySelector('input[name="board"]:checked');
    return el ? el.value : DEFAULT_BOARD;
  }
  function setBoard(name) {
    if (!BOARDS[name]) name = DEFAULT_BOARD;
    var el = document.querySelector('input[name="board"][value="' + name + '"]');
    if (el) el.checked = true;
  }

  // -------------------------------------------------------------------------
  // Load a board — GET /benchmark/scoreboard?board=<name>.  The S05
  // service is the sole authority on the per-row ordering; the page
  // preserves the service's order verbatim.  Column-header
  // client-side sorting is a display convenience (see setSort
  // below); the DEFAULT sort is the server-provided order (the page
  // never recomputes ranking from raw metrics — Invariant 4).
  // -------------------------------------------------------------------------
  var allRows = [], sortKey = "rank", sortDir = "asc";

  // Sensible default direction per column.  "rank" is a numeric
  // server-assigned column (best = 1, ascending = best-first when
  // the server's order is "best first" for the active board).  The
  // server order is preserved as the default; client-side sort
  // re-orders the already-ranked rows for display convenience only.
  var DEFAULT_DIR = {
    rank: "asc",
    config_snapshot: "asc",
    created_at: "desc",
    metric_value: "desc"
  };

  function sortedRows() {
    var out = allRows.slice();
    out.sort(function (a, b) {
      var dir = sortDir === "asc" ? 1 : -1;
      switch (sortKey) {
        case "rank":
          return dir * ((Number(a.rank) || 0) - (Number(b.rank) || 0));
        case "config_snapshot": {
          // Display sort: compare a short text summary of the model_config
          // (the keys + values) so two reports with similar configs
          // cluster together when the operator sorts by config.
          var sa = configSummary(a.config_snapshot, a.dataset_name);
          var sb = configSummary(b.config_snapshot, b.dataset_name);
          return dir * sa.toLowerCase().localeCompare(sb.toLowerCase());
        }
        case "created_at": {
          var ta = Date.parse(a.created_at), tb = Date.parse(b.created_at);
          if (!isNaN(ta) && !isNaN(tb)) return dir * (ta - tb);
          return dir * String(a.created_at || "").localeCompare(String(b.created_at || ""));
        }
        case "metric_value": {
          // Nulls always sort last regardless of direction (the server
          // also ranks them last; we mirror the rule client-side).
          if (a.metric_value == null && b.metric_value == null) return 0;
          if (a.metric_value == null) return 1;
          if (b.metric_value == null) return -1;
          return dir * ((Number(a.metric_value) || 0) - (Number(b.metric_value) || 0));
        }
        default: return 0;
      }
    });
    return out;
  }

  function refreshHeaderIndicators() {
    var ths = document.querySelectorAll("th.sortable");
    for (var i = 0; i < ths.length; i++) {
      var th = ths[i];
      th.classList.remove("asc", "desc");
      var ind = th.querySelector(".sort-ind");
      if (th.dataset.sort === sortKey) {
        th.classList.add(sortDir);
        if (ind) ind.textContent = sortDir === "asc" ? "▲" : "▼";
      } else {
        if (ind) ind.textContent = "▾";
      }
    }
  }

  function configSummary(snap, datasetName) {
    if (!snap || typeof snap !== "object") return "";
    // Show the generator / embedder / reranker / grader / dataset as
    // a short slash-joined summary; the full JSON is in the
    // <details> below the table.  Mirrors the "summary" the
    // benchmark_page.py uses on the saved-reports list.
    var parts = [];
    ["generator", "embedder", "reranker", "grader"].forEach(function (k) {
      if (snap[k] != null && snap[k] !== "") parts.push(String(snap[k]));
    });
    // Prefer the resolved human dataset name; fall back to the snapshot's
    // raw uuid so the slot is never blank for legacy rows.
    var ds = datasetName || snap.dataset_id;
    if (ds != null && ds !== "") parts.push(String(ds));
    if (parts.length === 0) {
      // Fall back to the first 3 keys of the snapshot.
      var keys = Object.keys(snap).slice(0, 3);
      keys.forEach(function (k) { parts.push(k + "=" + String(snap[k])); });
    }
    return parts.join(" / ");
  }

  function renderBoardDesc(boardName) {
    var meta = BOARDS[boardName] || BOARDS[DEFAULT_BOARD];
    var dirLabel = meta.direction === "asc" ? "ascending (lower is better — fastest first)" : "descending (higher is better — best first)";
    $("boardDesc").innerHTML = "Ranking by <strong>" + esc(meta.label) + "</strong> (" + dirLabel + ").";
    $("boardTitle").textContent = meta.title;
    // Plain-language explanation of what this board ranks + how to read it.
    if ($("boardHelp")) $("boardHelp").textContent = meta.what || "";
    // The Value column header carries the board's unit + a hover tooltip so
    // the number is never a bare "0.234".  Keep a .sort-ind span so the
    // sortable-header indicator logic still finds it.
    var vh = $("valueHeader");
    if (vh) {
      vh.innerHTML = 'Value <span class="vh-unit">' + esc(meta.unit) + '</span>' +
        IW_tip(meta.title, meta.unit, meta.what || "") + '<span class="sort-ind">▼</span>';
    }
  }

  function renderEmpty(empty) {
    var emptyEl = $("empty");
    if (empty) emptyEl.classList.remove("hidden");
    else emptyEl.classList.add("hidden");
  }

  function renderRows() {
    var tb = $("rows"); tb.innerHTML = "";
    refreshHeaderIndicators();
    if (allRows.length === 0) { renderEmpty(true); $("boardCount").textContent = "0 reports"; return; }
    renderEmpty(false);
    $("boardCount").textContent = allRows.length + " report" + (allRows.length === 1 ? "" : "s");
    var sorted = sortedRows();
    sorted.forEach(function (e) {
      var tr = document.createElement("tr");
      var summary = configSummary(e.config_snapshot, e.dataset_name);
      var fullJson = JSON.stringify(e.config_snapshot || {}, null, 2);
      var runId = e.display_id || ("#" + (e.report_id || "").slice(0, 8));
      var trHtml =
        '<td class="num">' + (e.rank != null ? e.rank : "—") + '</td>' +
        '<td><span class="run-id">' + esc(runId) + '</span></td>' +
        '<td><div class="model-config" title="' + esc(summary) + '">' + esc(summary) + '</div>' +
        '<details style="margin:4px 0 0"><summary class="muted" style="cursor:pointer;font-size:11px">full config_snapshot</summary>' +
        '<pre style="white-space:pre-wrap;font-size:11px;color:var(--muted);margin:6px 0 0">' + esc(fullJson) + '</pre></details></td>' +
        '<td>' + esc(e.created_at || "—") + '</td>' +
        '<td class="num">' + (e.metric_value == null ? '<span class="muted">—</span>' : fmt(e.metric_value, 3)) + '</td>';
      tr.innerHTML = trHtml;
      tb.appendChild(tr);
    });
  }

  async function loadBoard(boardName) {
    if (!BOARDS[boardName]) boardName = DEFAULT_BOARD;
    setBoard(boardName);
    renderBoardDesc(boardName);
    // Default sort direction follows the board's direction (the
    // server's order IS the board's direction — so the default sort
    // is the server's order).  Reset on every board switch so a
    // "sort by created_at" on faithfulness does not carry over to
    // the latency board.
    sortKey = "rank";
    sortDir = "asc";
    allRows = [];
    try {
      var r = await fetch(API + "/benchmark/scoreboard?board=" + encodeURIComponent(boardName), { headers: authHeaders() });
      if (r.status === 401) { setToken(null); showLogin(); return; }
      if (!r.ok) {
        // Render the empty state on 4xx (the server is the
        // authoritative gate; we never compute ranking client-side).
        allRows = [];
        renderRows();
        var msg = "Scoreboard unavailable (" + r.status + ")";
        try { var ej = await r.json(); if (ej && (ej.error || ej.detail)) msg = ej.error || ej.detail; } catch (x) {}
        toast(msg, true);
        return;
      }
      var d = await r.json();
      // S08 returns {board, entries: [{report_id, rank, config_snapshot,
      // created_at, metric_value}, ...]}.  Preserve the service's
      // order verbatim (AC6 — the page never recomputes ranking from
      // raw metrics).  The route assigns the 1-based rank on the
      // server; the page keeps it.
      allRows = (d && d.entries) ? d.entries.slice() : [];
      renderRows();
    } catch (e) {
      allRows = [];
      renderRows();
      toast("Scoreboard load failed: " + e, true);
    }
  }

  function setSort(key) {
    if (sortKey === key) { sortDir = sortDir === "asc" ? "desc" : "asc"; }
    else { sortKey = key; sortDir = DEFAULT_DIR[key] || "asc"; }
    renderRows();
  }

  // -------------------------------------------------------------------------
  // Wire up
  // -------------------------------------------------------------------------
  $("loginBtn").addEventListener("click", login);
  $("p").addEventListener("keydown", function (e) { if (e.key === "Enter") login(); });
  $("logout").addEventListener("click", function () { setToken(null); showLogin(); });

  // Board picker — clicking a different board refetches the
  // server-ranked rows for that board (Invariant 4: the page never
  // re-derives ranking; every board-switch is a fresh GET).
  var boardPicker = $("boardPicker");
  if (boardPicker) {
    boardPicker.addEventListener("change", function (ev) {
      var t = ev.target;
      if (t && t.name === "board" && t.value) loadBoard(t.value);
    });
  }

  // Sortable headers — display convenience over the already-ranked
  // rows; the default sort follows the server's order.  Mirrors the
  // documents_page.py th.sortable click handler verbatim.
  document.addEventListener("click", function (ev) {
    var t = ev.target;
    var th = t.closest && t.closest("th.sortable");
    if (th && th.dataset && th.dataset.sort) { setSort(th.dataset.sort); }
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


def scoreboard_page_html() -> str:
    """Return the standalone Scoreboard view HTML page.

    The view is a self-contained inline HTML+CSS+vanilla-JS page
    mirroring :mod:`iw_rag.api.benchmark_page` and
    :mod:`iw_rag.api.console_page`, re-homed as an Operator Console
    sub-view.  It carries the IW brand chrome (teal ``#0D9488`` /
    ``#2DD4BF`` on ink ``#181D29``), the location-derived API base,
    the sessionStorage-only bearer token, the ``/auth/login``
    self-handled sign-in, the S10 shared six-item nav, the six-board
    segmented control (faithfulness / ndcg_at_5 / tok_per_s / latency
    / vram_efficiency / composite — the six S05 leaderboards), the
    sortable board table (preserving the server-provided ranking as
    the default; the ``documents_page.py`` ``th.sortable`` / ``setSort``
    / ``refreshHeaderIndicators`` pattern reused for display
    convenience), and the empty-history empty state.

    The view is a **thin API client** (Invariant 4): it renders
    exactly what ``GET /benchmark/scoreboard`` returns and never
    recomputes ranking from raw metrics.  The S05
    :class:`~iw_rag.eval.benchmark.scoreboard.ScoreboardService` is
    the **sole** authority on the per-row ordering (AC6: descending
    faithfulness for the ``faithfulness`` board).

    Returns:
        The raw, self-contained HTML document as a ``str`` (no MIME /
        static-mount wiring required).
    """
    return add_favicon(_PAGE.replace("/* __IW_TOOLTIP_CSS__ */", TOOLTIP_CSS))


__all__ = ["scoreboard_page_html"]
