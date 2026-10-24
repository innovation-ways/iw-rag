# ruff: noqa: E501
"""Self-contained, branded IW RAG Operator Console Results + Compare view (F-00096, S13).

The Results + Compare view is the Operator Console page that deep-dives
on a single persisted ``benchmark_reports`` row and supports A/B
comparing two reports on the same dataset.  It is a **thin API
client** — Invariant 4: it enforces NO retrieval, ACL, grounding, or
guard logic.  All such logic stays server-side (the F-00095
``GET /benchmark/reports`` / ``GET /benchmark/reports/{id}`` /
``GET /benchmark/compare`` routes; the F-00093 stats module).  The
view consumes those endpoints exactly as :mod:`iw_rag.api.benchmark_page`
already does and reuses the proven ``loadReport()`` and ``compare()``
render logic verbatim so the two views stay consistent.

**AC5** — the per-metric delta table, the corrected significance
column, the winner verdict (or ``winner=None`` when neither config
wins on both nDCG@5 AND faithfulness), and the underpowered caveat
are all rendered.  **Boundary row coverage** —

* Empty report history (0 ``benchmark_reports`` rows) → explicit
  "No reports yet — run a benchmark from the Benchmark Runner to
  create one." empty state in both the deep-dive picker and the A/B
  pickers (not an error).
* Compare across different datasets (Invariant 6) → ``GET
  /benchmark/compare`` returns 400; the panel renders the
  not-comparable message verbatim.
* Unknown report id (deep-dive) → 404; the panel renders the
  not-found state.
* Unknown report id (compare) → 404; the panel renders the
  "one of the report ids is unknown" message.

It is a **server-rendered FastAPI HTML page** in the
``benchmark_page.py`` / ``console_page.py`` / ``config_page.py`` /
``benchmark_runner_page.py`` style — NOT Chainlit, NOT a JS SPA, NO
new build toolchain (per the F-00096 design *Out-of-Scope* row "A
JavaScript SPA / new build toolchain").  The page derives its API
base from its own URL so it works behind Caddy (``/api``) or when
hit directly (``/``).  No external CDN / Google Fonts link
(Invariant 8, local-only rule).

Mirrors :mod:`iw_rag.api.benchmark_page`,
:mod:`iw_rag.api.console_page`, :mod:`iw_rag.api.config_page`, and
:mod:`iw_rag.api.benchmark_runner_page` — same header/footer chrome,
same teal/ink palette, same location-derived API base regex idiom
with ``\\/`` (the forward-slash escape that keeps the JS parser
from closing the literal at the first ``/``), same
sessionStorage-only bearer token, same raw-string ``_PAGE`` whose
inline ``<script>`` passes ``node --check``.  The S13 route
``GET /console/results`` is wired against
:func:`results_page_html`.
"""

from __future__ import annotations

from iw_rag.api.benchmark_primer import apply_benchmark_primer
from iw_rag.api.benchmark_render import REPORT_CSS, REPORT_JS, TOOLTIP_CSS
from iw_rag.api.console_nav import apply_console_nav
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
  .caveat-box.err { background: rgba(244,63,94,0.10); border-color: #5b2230; color: #ffd7de; }
  .caveat-box.err strong { color: var(--danger); }
  details { margin: 8px 0; }
  details summary { cursor: pointer; color: var(--muted); font-size: 13px; padding: 6px 0; }
  .compare-reasoning { font-size: 13px; color: var(--muted); line-height: 1.5; padding: 10px 12px; background: var(--ink-2); border: 1px solid var(--line); border-radius: 10px; margin: 10px 0; }
  nav.console-nav { display: flex; gap: 6px; flex-wrap: wrap; align-items: center; padding: 14px 28px; background: rgba(24,29,41,0.5); border-bottom: 1px solid var(--line); }
  nav.console-nav a.nav-item { font-size: 13px; padding: 6px 12px; border-radius: 8px; border: 1px solid transparent; color: var(--muted); }
  nav.console-nav a.nav-item:hover { color: var(--text); border-color: var(--line); }
  nav.console-nav a.nav-item.active { color: #06231f; background: linear-gradient(180deg, var(--teal) 0%, var(--teal-deep) 100%); border-color: transparent; font-weight: 600; }
/* __IW_REPORT_CSS__ */
</style>
</head>
<body>
<header>
  <span class="logo"><span class="i">I</span>nnovation <span class="w">W</span>ays</span>
  <span class="muted">· Results</span>
  <span class="spacer"></span>
  <span class="who" id="who"></span>
  <button class="btn-ghost hidden" id="logout">Sign out</button>
</header>

__CONSOLE_NAV__

<main>
  <section id="loginView" class="hidden">
    <div class="card loginwrap">
      <h1>Sign in</h1>
      <p class="sub">Use your demo credentials. The Results + Compare view is read-only and open to any authenticated principal.</p>
      <label for="u">Username</label>
      <input type="text" id="u" autocomplete="username" />
      <label for="p">Password</label>
      <input type="password" id="p" autocomplete="current-password" />
      <div style="margin-top:18px"><button class="btn-primary" id="loginBtn">Sign in</button></div>
    </div>
  </section>

  <section id="appView" class="hidden">
    __BENCH_PRIMER__

    <div class="card" id="reportsCard">
      <h2>Saved reports</h2>
      <p class="sub">Every saved benchmark run, newest first. Open one to see how it did: did it find the right passages, were its answers honest, how fast was it and how much GPU memory did it use. To compare two runs on the same dataset, use the A/B compare box below.</p>
      <div class="row">
        <div>
          <label for="reportPicker">Report (deep-dive)</label>
          <select id="reportPicker"><option value="">(select a report)</option></select>
        </div>
        <div>
          <button class="btn-primary" id="openReportBtn">Open</button>
        </div>
      </div>
      <div id="reportDetail"></div>
    </div>

    <div class="card" id="compareCard">
      <h2>A/B compare</h2>
      <p class="sub">Pick two saved runs on the same dataset to see which is better, and whether the difference is real. A <strong>winner</strong> is named only when one run is better at <strong>both search quality (nDCG@5) and honesty (faithfulness)</strong>. With about 37 questions small gaps can be luck, so the page always says when a result is underpowered. If something is wrong, for example the two runs used different datasets, the reason is shown right here.</p>
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
  // API base. The Operator Console is served at /console* behind the Caddy
  // edge (ADR-016 single entrypoint); the API lives under /api/*, so the base
  // is the fixed "/api" — fetch(API + "/api/config") -> /api/api/config, which
  // the edge's `handle_path /api/*` strips back to /api/config on the API.
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
  // Empty-state anchor — Boundary row "Empty report history".  The
  // same copy is shown in both the deep-dive picker and the compare
  // pickers; the page is rendered (not an error) so the operator
  // sees the call-to-action.
  // -------------------------------------------------------------------------
  var EMPTY_HISTORY_HTML =
    '<p class="muted">No reports yet — run a benchmark from the Benchmark Runner to create one.</p>';

  // -------------------------------------------------------------------------
  // Reports list — populates the deep-dive picker + the A/B compare
  // dropdowns.  Boundary: 0 reports → explicit empty state, not an
  // error.  4xx/5xx → "(unavailable)" placeholders.
  // -------------------------------------------------------------------------
  async function loadReports() {
    try {
      var r = await fetch(API + "/benchmark/reports", { headers: authHeaders() });
      if (r.status === 401) { setToken(null); showLogin(); return; }
      if (!r.ok) {
        $("compareA").innerHTML = '<option value="">(unavailable)</option>';
        $("compareB").innerHTML = '<option value="">(unavailable)</option>';
        $("reportPicker").innerHTML = '<option value="">(unavailable)</option>';
        return;
      }
      var rows = await r.json();
      if (!rows || rows.length === 0) {
        // Boundary: 0 reports → explicit empty state, not an error.
        $("reportPicker").innerHTML = '<option value="">(no reports yet)</option>';
        $("compareA").innerHTML = '<option value="">(no reports yet)</option>';
        $("compareB").innerHTML = '<option value="">(no reports yet)</option>';
        $("reportDetail").innerHTML = EMPTY_HISTORY_HTML;
        return;
      }
      var opts = '<option value="">(select a report)</option>';
      rows.forEach(function (rep) {
        var label = (rep.display_id || ("#" + (rep.id || "").slice(0, 8))) + " · " +
          (rep.report_label || (rep.dataset_name || rep.dataset_id || "report")) + " · " +
          (rep.created_at || "");
        opts += '<option value="' + esc(rep.id) + '">' + esc(label) + '</option>';
      });
      $("reportPicker").innerHTML = opts;
      $("compareA").innerHTML = opts;
      $("compareB").innerHTML = opts;
    } catch (e) {
      $("compareA").innerHTML = '<option value="">(unavailable)</option>';
      $("compareB").innerHTML = '<option value="">(unavailable)</option>';
      $("reportPicker").innerHTML = '<option value="">(unavailable)</option>';
    }
  }

  // -------------------------------------------------------------------------
  // Report deep-dive — fetch GET /benchmark/reports/{id} and render
  // the metric tiers, latency (model_load_ms in its own bucket),
  // VRAM, and config_snapshot.  Boundary: 404 → "not found" state in
  // the panel (never a toast that hides the cause).  Reuses the
  // exact loadReport() helper shape from benchmark_page.py so the
  // two views stay consistent.
  // -------------------------------------------------------------------------
  async function loadReport(reportId) {
    var body = $("reportDetail");
    body.innerHTML = "";
    if (!reportId) { return; }
    try {
      var r = await fetch(API + "/benchmark/reports/" + encodeURIComponent(reportId), { headers: authHeaders() });
      if (r.status === 401) { setToken(null); showLogin(); return; }
      if (r.status === 404) {
        // Boundary: 404 → "not found" state in the panel, never a
        // toast that hides the cause.  Mirrors the benchmark_page.py
        // caveat-box visual treatment.
        body.innerHTML = '<div class="caveat-box err"><strong>Not found.</strong> Report ' + esc(reportId) + ' is unknown — it may have been removed or the id is mistyped.</div>';
        return;
      }
      if (!r.ok) { body.innerHTML = '<div class="caveat-box err"><strong>Report not available (' + r.status + ').</strong></div>'; return; }
      var rep = await r.json();
      // Shared renderer (benchmark_render.py) — headline stat cards,
      // metric tiers with hover explanations + units, the REAL latency
      // fields (model_load_ms / per_stage_ms / vLLM p99s), GPU/VRAM, and
      // the config snapshot.  Identical to the standalone Studio view.
      var html = '<h2 style="margin-top:18px">Run ' + esc(rep.display_id || ("#" + (rep.id || "").slice(0, 8))) + '</h2>';
      html += '<p class="sub">' + IW_reportMeta(rep) + '</p>';
      html += IW_renderReport(rep);
      body.innerHTML = html;
    } catch (e) {
      body.innerHTML = '<div class="caveat-box err"><strong>Report load failed:</strong> ' + esc(String(e)) + '</div>';
    }
  }

  // -------------------------------------------------------------------------
  // A/B compare — GET /benchmark/compare?a=..&b=.. + the winner +
  // underpowered caveat.  Boundary rows: 400 (different datasets) +
  // 404 (unknown id) are rendered in the panel verbatim, not as a
  // toast that hides the cause.  Client-side pre-checks (A==B or
  // either empty) toast the operator but the server is
  // authoritative.
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
      // Boundary: 404 → "one of the report ids is unknown" message
      // in the panel (never a toast that hides the cause).  Mirrors
      // benchmark_page.py verbatim.
      if (r.status === 404) {
        body.innerHTML = '<div class="caveat-box err"><strong>Not found.</strong> One of the report ids is unknown.</div>';
        return;
      }
      // Boundary: 400 → "not comparable" message in the panel
      // (Invariant 6 — paired stats require the same golden
      // queries).  Mirrors benchmark_page.py verbatim.
      if (r.status === 400) {
        body.innerHTML = '<div class="caveat-box"><strong>Cannot compare.</strong> The two reports are on different datasets — paired stats require the same golden queries.</div>';
        return;
      }
      if (!r.ok) { body.innerHTML = '<div class="caveat-box err"><strong>Compare failed (' + r.status + ').</strong></div>'; return; }
      var d = await r.json();
      // Shared compare renderer — underpowered caveat, winner verdict
      // (only when better on BOTH nDCG@5 and faithfulness), and the
      // per-metric delta table with hover explanations.
      body.innerHTML = IW_renderCompare(d);
    } catch (e) {
      body.innerHTML = '<div class="caveat-box err"><strong>Compare failed:</strong> ' + esc(String(e)) + '</div>';
    }
  }

  // -------------------------------------------------------------------------
  // Wire up
  // -------------------------------------------------------------------------
  $("loginBtn").addEventListener("click", login);
  $("p").addEventListener("keydown", function (e) { if (e.key === "Enter") login(); });
  $("logout").addEventListener("click", function () { setToken(null); showLogin(); });
  $("openReportBtn").addEventListener("click", function () {
    var id = $("reportPicker").value;
    if (!id) { toast("Pick a report to deep-dive", true); return; }
    loadReport(id);
  });
  $("compareBtn").addEventListener("click", compare);

  // Regenerate the conclusion (delegated — the button is inside the rendered
  // report body). POST refreshes against the latest comparable runs, then we
  // re-load the report so the new conclusion shows.
  $("reportDetail").addEventListener("click", function (e) {
    // Generate Report → fetch the PDF WITH the bearer token (window.open can't
    // send it), then trigger a browser download of the blob.
    var pdfBtn = e.target.closest ? e.target.closest("[data-pdf-report]") : null;
    if (pdfBtn) {
      var pid = pdfBtn.getAttribute("data-pdf-report");
      if (!pid) return;
      var orig = pdfBtn.innerHTML;
      pdfBtn.disabled = true; pdfBtn.innerHTML = "Generating PDF&hellip;";
      fetch(API + "/benchmark/reports/" + encodeURIComponent(pid) + "/pdf", { headers: authHeaders() })
        .then(function (r) {
          if (!r.ok) return r.json().then(function (j) { throw new Error(j.detail || j.error || ("HTTP " + r.status)); });
          return r.blob();
        })
        .then(function (blob) {
          var url = URL.createObjectURL(blob);
          var a = document.createElement("a");
          a.href = url; a.download = pid + ".pdf";
          document.body.appendChild(a); a.click(); a.remove();
          URL.revokeObjectURL(url);
          pdfBtn.disabled = false; pdfBtn.innerHTML = orig;
        })
        .catch(function (err) {
          toast("PDF failed: " + err, true);
          pdfBtn.disabled = false; pdfBtn.innerHTML = orig;
        });
      return;
    }
    var btn = e.target.closest ? e.target.closest("[data-regen-conclusion]") : null;
    if (!btn) return;
    var id = btn.getAttribute("data-regen-conclusion");
    if (!id) return;
    btn.disabled = true;
    btn.innerHTML = "Regenerating&hellip;";
    fetch(API + "/benchmark/reports/" + encodeURIComponent(id) + "/conclusion",
      { method: "POST", headers: authHeaders() })
      .then(function (r) { if (!r.ok) throw new Error("HTTP " + r.status); return r.json(); })
      .then(function () { return loadReport(id); })
      .catch(function (err) {
        toast("Regenerate failed: " + err, true);
        btn.disabled = false; btn.innerHTML = "&#8635; Regenerate";
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


def results_page_html() -> str:
    """Return the standalone Results + Compare view HTML page.

    The view is a self-contained inline HTML+CSS+vanilla-JS page
    mirroring :mod:`iw_rag.api.benchmark_page` and
    :mod:`iw_rag.api.console_page`, re-homed as an Operator Console
    sub-view.  It carries the IW brand chrome (teal ``#0D9488`` /
    ``#2DD4BF`` on ink ``#181D29``), the location-derived API base,
    the sessionStorage-only bearer token, the ``/auth/login``
    self-handled sign-in, the S10 shared six-item nav, the report
    deep-dive panel (populated from ``GET /benchmark/reports/{id}``
    with the same metric-tier / latency / VRAM / config_snapshot
    renderers as :mod:`iw_rag.api.benchmark_page`), and the A/B
    compare panel (populated from ``GET /benchmark/compare`` with
    the same delta-table / winner / underpowered-caveat renderers
    as :mod:`iw_rag.api.benchmark_page`).

    Returns:
        The raw, self-contained HTML document as a ``str`` (no MIME /
        static-mount wiring required).
    """
    page = _PAGE.replace("/* __IW_REPORT_CSS__ */", TOOLTIP_CSS + REPORT_CSS).replace(
        "/* __IW_REPORT_JS__ */", REPORT_JS
    )
    return apply_console_nav(apply_benchmark_primer(add_favicon(page)))


__all__ = ["results_page_html"]
