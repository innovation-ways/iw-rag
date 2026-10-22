# ruff: noqa: E501
"""Self-contained, branded IW RAG Operator Console System view (VRAM power switch).

The System view is the Operator Console page that surfaces the
single-GPU power state (on / off / partial / transitioning), a live
VRAM gauge polled every 2 s, per-component start/stop controls, and
the idle-unload / block-and-warm info line.  It is a **thin API
client** — Invariant 4: it enforces NO power-control logic, NO
VRAM math, NO ACL decisions.  All such logic stays server-side.

The page consumes four backend endpoints that already exist:

* ``GET /api/system/status`` — any authenticated principal; returns
  the full system status blob (enabled, available, state, transition,
  components, vram, mode, power_control_active, mode_switch_enabled,
  idle_unload_*, …).
* ``GET /api/system/vram`` — vram-only subset; polled every 2 s for
  the live gauge.
* ``POST /api/system/power`` — requires power group; body
  ``{action: "on"|"off", components: string[]|null}``.  Returns the
  same status blob on success.  403 means read-only access.
* ``POST /api/system/mode`` — requires power group; body
  ``{target: "serving"|"benchmark"}``.  Returns the same status blob
  on success.  403 means read-only access.  Can take 30–120 s while
  the GPU-resident generator is loaded or unloaded.

It is a **server-rendered FastAPI HTML page** in the
``config_page.py`` / ``scoreboard_page.py`` style — NOT Chainlit,
NOT a JS SPA, NO new build toolchain (per the F-00096 design
*Out-of-Scope* row "A JavaScript SPA / new build toolchain").  The
page derives its API base from its own URL so it works behind Caddy
(``/api``) or when hit directly (``/``).  No external CDN / Google
Fonts link (Invariant 8, local-only rule).

Mirrors :mod:`iw_rag.api.config_page` and
:mod:`iw_rag.api.scoreboard_page` — same header/footer chrome, same
teal/ink palette, same location-derived API base with ``var API =
"/api"``, same sessionStorage-only bearer token, same
``node --check``-passing raw-string ``_PAGE``.  The new route ``GET
/console/system`` is wired against :func:`system_page_html`.
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
  .btn-danger { border-color: var(--danger); color: #ffd7de; }
  .btn-sm { padding: 5px 10px; font-size: 12px; border-radius: 8px; }
  .main-wrap { max-width: 1080px; margin: 0 auto; padding: 28px 20px 80px; }
  .card { background: rgba(34,38,49,0.72); border: 1px solid var(--line); border-radius: var(--radius); padding: 22px; margin-bottom: 22px; }
  h1 { font-family: "Space Grotesk", system-ui, sans-serif; font-size: 22px; margin: 0 0 4px; }
  h2 { font-family: "Space Grotesk", system-ui, sans-serif; font-size: 16px; margin: 0 0 14px; color: var(--text); }
  .sub { color: var(--muted); font-size: 14px; margin: 0 0 18px; }
  label { display: block; font-size: 13px; color: var(--muted); margin: 12px 0 6px; }
  input[type=text], input[type=password] {
    width: 100%; padding: 11px 12px; border-radius: 10px; border: 1px solid var(--line);
    background: var(--ink-2); color: var(--text); font: inherit;
  }
  input:disabled, button:disabled { opacity: .6; cursor: not-allowed; }
  table { width: 100%; border-collapse: collapse; }
  th, td { text-align: left; padding: 10px 8px; border-bottom: 1px solid var(--line); font-size: 13px; vertical-align: top; }
  th { color: var(--muted); font-weight: 600; font-size: 11px; text-transform: uppercase; letter-spacing: .5px; }
  td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
  .chip { font-size: 11px; padding: 2px 8px; border-radius: 999px; border: 1px solid var(--line); white-space: nowrap; display: inline-block; }
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
  nav.console-nav { display: flex; gap: 6px; flex-wrap: wrap; align-items: center; padding: 14px 28px; background: rgba(24,29,41,0.5); border-bottom: 1px solid var(--line); }
  nav.console-nav a.nav-item { font-size: 13px; padding: 6px 12px; border-radius: 8px; border: 1px solid transparent; color: var(--muted); }
  nav.console-nav a.nav-item:hover { color: var(--text); border-color: var(--line); }
  nav.console-nav a.nav-item.active { color: #06231f; background: linear-gradient(180deg, var(--teal) 0%, var(--teal-deep) 100%); border-color: transparent; font-weight: 600; }
  /* VRAM gauge bar */
  .vram-bar-outer { background: var(--ink-2); border: 1px solid var(--line); border-radius: 6px; height: 18px; overflow: hidden; margin: 10px 0 6px; }
  .vram-bar-inner { height: 100%; background: linear-gradient(90deg, var(--teal-deep) 0%, var(--teal) 100%); border-radius: 6px; transition: width .4s ease; width: 0%; }
  .vram-bar-inner.warn { background: linear-gradient(90deg, #b45309 0%, var(--warn) 100%); }
  .vram-bar-inner.bad { background: linear-gradient(90deg, #9b1c2e 0%, var(--danger) 100%); }
  /* Master switch card */
  .state-badge { display: inline-flex; align-items: center; gap: 6px; font-size: 14px; font-weight: 600; padding: 6px 14px; border-radius: 999px; border: 1px solid var(--line); }
  .state-badge.on { color: var(--ok); border-color: var(--teal-deep); background: rgba(13,148,136,0.12); }
  .state-badge.off { color: var(--muted); background: var(--ink-2); }
  .state-badge.partial { color: var(--warn); border-color: #5b4322; background: rgba(245,158,11,0.08); }
  .state-badge.transitioning { color: var(--teal); border-color: var(--teal-deep); background: rgba(45,212,191,0.08); }
  .state-badge.unknown { color: var(--muted); }
  .power-btns { display: flex; gap: 12px; flex-wrap: wrap; margin-top: 18px; }
  /* Spinner */
  .spinner { display: inline-block; width: 16px; height: 16px; border: 2px solid var(--line); border-top-color: var(--teal); border-radius: 50%; animation: spin .8s linear infinite; vertical-align: middle; }
  @keyframes spin { to { transform: rotate(360deg); } }
  .working-msg { color: var(--muted); font-size: 13px; margin-top: 10px; display: flex; align-items: center; gap: 8px; }
  /* Component rows (advanced) */
  details.adv-details > summary { cursor: pointer; font-size: 13px; color: var(--teal); padding: 10px 0; -webkit-user-select: none; user-select: none; }
  details.adv-details > summary:hover { color: var(--text); }
  .comp-row { display: grid; grid-template-columns: 1fr auto; gap: 12px; align-items: start; padding: 14px 0; border-bottom: 1px solid var(--ink-2); }
  .comp-row:last-child { border-bottom: none; }
  .comp-meta { display: flex; flex-direction: column; gap: 4px; }
  .comp-label { font-weight: 600; font-size: 14px; }
  .comp-desc { color: var(--muted); font-size: 12px; line-height: 1.4; }
  .comp-containers { color: var(--muted); font-size: 12px; margin-top: 4px; font-family: "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
  .comp-actions { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
  /* Idle info */
  .idle-info { color: var(--muted); font-size: 12px; margin-top: 4px; }
  /* Process table */
  .proc-table { font-size: 12px; }
  .proc-table td { padding: 5px 6px; font-family: "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
  /* Mode card */
  .mode-badge { display: inline-flex; align-items: center; gap: 8px; font-size: 15px; font-weight: 700; padding: 7px 16px; border-radius: 999px; border: 1px solid var(--line); margin-bottom: 14px; }
  .mode-badge.serving { color: var(--ok); border-color: var(--teal-deep); background: rgba(13,148,136,0.12); }
  .mode-badge.benchmark { color: var(--warn); border-color: #5b4322; background: rgba(245,158,11,0.08); }
  .mode-badge.unknown { color: var(--muted); background: var(--ink-2); }
  .mode-btns { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 4px; }
  .mode-helper { color: var(--muted); font-size: 12px; margin-top: 10px; }
  .mode-working { display: flex; align-items: center; gap: 10px; color: var(--warn); font-size: 13px; margin-top: 10px; border: 1px solid #5b4322; background: rgba(245,158,11,0.06); border-radius: 8px; padding: 10px 14px; }
</style>
</head>
<body>
<header>
  <span class="logo"><span class="i">I</span>nnovation <span class="w">W</span>ays</span>
  <span class="muted">· System</span>
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
      <p class="sub">Use your demo credentials. The System view surfaces the GPU power state and VRAM usage. Power-control actions require the power-control group; read-only views are open to any authenticated principal.</p>
      <label for="u">Username</label>
      <input type="text" id="u" autocomplete="username" />
      <label for="p">Password</label>
      <input type="password" id="p" autocomplete="current-password" />
      <div style="margin-top:18px"><button class="btn-primary" id="loginBtn">Sign in</button></div>
    </div>
  </section>

  <section id="appView" class="hidden">
    <div class="main-wrap">

      <!-- Power control not enabled for this deployment -->
      <div id="disabledCard" class="card hidden">
        <p class="muted">Power control is not enabled for this deployment.</p>
      </div>

      <!-- System-control sidecar unreachable -->
      <div id="unavailableCard" class="card hidden">
        <p class="muted" id="unavailableReason">system-control sidecar unreachable</p>
      </div>

      <!-- Mode card — shown when power control is enabled and sidecar is available -->
      <div class="card hidden" id="modeCard">
        <h2>Mode</h2>

        <!-- Current mode badge -->
        <div class="mode-badge unknown" id="modeBadge">unknown</div>

        <!-- Mode switch buttons — shown when mode_switch_enabled is true and no mode switch in flight -->
        <div class="mode-btns hidden" id="modeBtns">
          <button id="btnToServing">Switch to Running</button>
          <button id="btnToBenchmark">Switch to Benchmark</button>
        </div>

        <!-- Mode-switching in-flight indicator -->
        <div class="mode-working hidden" id="modeWorking" role="status" aria-live="polite">
          <span class="spinner"></span>
          <span>Switching mode&hellip; this can take up to ~2 minutes (loading/unloading the generator)</span>
        </div>

        <!-- mode_switch_enabled === false note -->
        <p class="mode-helper hidden" id="modeSwitchDisabledNote">Mode switching disabled by operator configuration.</p>

        <p class="mode-helper">Running = chat/query (generator pinned). Benchmark = Benchmark Studio swap stack.</p>
      </div>

      <!-- VRAM gauge -->
      <div class="card" id="vramCard">
        <h2>VRAM</h2>
        <div id="vramUnavail" class="hidden"><p class="muted">GPU telemetry unavailable</p></div>
        <div id="vramGaugeWrap">
          <div class="vram-bar-outer" aria-label="VRAM usage bar">
            <div class="vram-bar-inner" id="vramBar"></div>
          </div>
          <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;">
            <span id="vramLabel" class="muted"></span>
            <span id="vramDevice" class="chip muted"></span>
          </div>
          <!-- Per-process table (largest first) -->
          <div id="procWrap" class="hidden" style="margin-top:14px;">
            <h2 style="font-size:13px;color:var(--muted);margin-bottom:8px;">GPU processes</h2>
            <table class="proc-table">
              <thead>
                <tr>
                  <th scope="col">PID</th>
                  <th scope="col">Process</th>
                  <th class="num" scope="col">Used MB</th>
                </tr>
              </thead>
              <tbody id="procRows"></tbody>
            </table>
          </div>
        </div>
      </div>

      <!-- Master power switch -->
      <div class="card" id="masterCard">
        <h2>Power</h2>
        <p class="sub">Master switch for all GPU-resident services. "Power OFF" unloads all components and frees VRAM. "Power ON" loads all components (cold start may take 30&ndash;60 s).</p>

        <div style="margin-bottom:14px;">
          <span class="state-badge unknown" id="stateBadge">unknown</span>
        </div>

        <div id="workingMsg" class="working-msg hidden">
          <span class="spinner"></span>
          <span>Working&hellip; (cold start can take ~30&ndash;60 s)</span>
        </div>

        <div class="power-btns" id="powerBtns">
          <button class="btn-primary" id="btnOn">Power ON</button>
          <button class="btn-danger" id="btnOff">Power OFF (free all VRAM)</button>
        </div>

        <!-- Idle info line -->
        <p class="idle-info" id="idleInfo"></p>
      </div>

      <!-- Advanced: per-component rows -->
      <div class="card" id="advCard">
        <details class="adv-details" id="advDetails">
          <summary>Advanced — per-component control</summary>
          <div id="compRows"></div>
        </details>
      </div>

    </div>
  </section>
</main>

<div class="toast" id="toast"></div>

<script>
(function () {
  // API base. The Operator Console is served at /console* behind the Caddy
  // edge (ADR-016 single entrypoint); the API lives under /api/*, so the base
  // is the fixed "/api" — fetch(API + "/system/status") -> /api/system/status,
  // which the edge's `handle_path /api/*` strips back to /system/status on the API.
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

  // ---------------------------------------------------------------------------
  // Poll state
  // ---------------------------------------------------------------------------
  var vramPollId = null;   // setInterval id for the VRAM poller (cleared on logout)
  var inFlight = false;    // true while a POST /system/power request is in flight
  var modeInFlight = false; // true while a POST /system/mode request is in flight

  // ---------------------------------------------------------------------------
  // Nav chrome — mirrors config_page.py / scoreboard_page.py verbatim
  // ---------------------------------------------------------------------------
  function showLogin() {
    $("loginView").classList.remove("hidden");
    $("appView").classList.add("hidden");
    $("consoleNav").classList.add("hidden");
    $("logout").classList.add("hidden");
    $("who").textContent = "";
    stopVramPoll();
  }
  function showApp(who, grp) {
    $("loginView").classList.add("hidden");
    $("appView").classList.remove("hidden");
    $("consoleNav").classList.remove("hidden");
    $("logout").classList.remove("hidden");
    var tag = who ? ("Signed in as " + who) : "";
    if (grp && grp.length) tag += " \xb7 " + grp.join(", ");
    $("who").textContent = tag;
    highlightActiveNav();
    startVramPoll();
    loadStatus();
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

  // ---------------------------------------------------------------------------
  // Login
  // ---------------------------------------------------------------------------
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

  // ---------------------------------------------------------------------------
  // VRAM gauge — polls GET /system/vram every 2 s
  // ---------------------------------------------------------------------------
  function stopVramPoll() {
    if (vramPollId !== null) { clearInterval(vramPollId); vramPollId = null; }
  }
  function startVramPoll() {
    stopVramPoll();
    pollVram();
    vramPollId = setInterval(pollVram, 2000);
  }
  async function pollVram() {
    if (!token()) return;
    try {
      var r = await fetch(API + "/system/vram", { headers: authHeaders() });
      if (r.status === 401) { setToken(null); showLogin(); return; }
      if (!r.ok) { renderVramUnavail(); return; }
      var d = await r.json();
      renderVram(d);
    } catch (e) {
      renderVramUnavail();
    }
  }
  function renderVramUnavail() {
    $("vramUnavail").classList.remove("hidden");
    $("vramGaugeWrap").classList.add("hidden");
  }
  function fmtGb(mb) {
    if (mb == null || isNaN(mb)) return "?";
    return (mb / 1024).toFixed(1);
  }
  function renderVram(v) {
    // v: { available, used_mb, free_mb, total_mb, device_name, processes, vram_ceiling_gb }
    if (!v || !v.available) { renderVramUnavail(); return; }
    $("vramUnavail").classList.add("hidden");
    $("vramGaugeWrap").classList.remove("hidden");

    var totalMb = v.total_mb || (v.vram_ceiling_gb ? v.vram_ceiling_gb * 1024 : null);
    var usedMb = v.used_mb;
    var freeMb = v.free_mb != null ? v.free_mb : (totalMb != null && usedMb != null ? totalMb - usedMb : null);
    var pct = (totalMb && usedMb != null) ? Math.min(100, (usedMb / totalMb) * 100) : 0;

    var bar = $("vramBar");
    bar.style.width = pct.toFixed(1) + "%";
    // Colour the bar: >=90% bad, >=70% warn, else ok
    bar.classList.remove("warn", "bad");
    if (pct >= 90) bar.classList.add("bad");
    else if (pct >= 70) bar.classList.add("warn");

    var labelParts = [];
    if (usedMb != null && totalMb != null) {
      labelParts.push(fmtGb(usedMb) + " / " + fmtGb(totalMb) + " GB used");
    }
    if (freeMb != null) labelParts.push(fmtGb(freeMb) + " GB free");
    $("vramLabel").textContent = labelParts.join(" \xb7 ");
    $("vramDevice").textContent = v.device_name || "";

    // Per-process table (server sends largest-first)
    var procs = Array.isArray(v.processes) ? v.processes : [];
    var procWrap = $("procWrap");
    if (procs.length > 0) {
      procWrap.classList.remove("hidden");
      var tb = $("procRows"); tb.innerHTML = "";
      procs.forEach(function (p) {
        var tr = document.createElement("tr");
        tr.innerHTML = '<td>' + esc(p.pid != null ? String(p.pid) : "—") + '</td>' +
          '<td>' + esc(p.name || "—") + '</td>' +
          '<td class="num">' + esc(p.used_mb != null ? String(p.used_mb) : "—") + '</td>';
        tb.appendChild(tr);
      });
    } else {
      procWrap.classList.add("hidden");
    }
  }

  // ---------------------------------------------------------------------------
  // Status — GET /system/status; also called after each POST /system/power
  // ---------------------------------------------------------------------------
  async function loadStatus() {
    try {
      var r = await fetch(API + "/system/status", { headers: authHeaders() });
      if (r.status === 401) { setToken(null); showLogin(); return; }
      if (!r.ok) {
        toast("System status unavailable (" + r.status + ")", true);
        return;
      }
      var d = await r.json();
      renderStatus(d);
    } catch (e) {
      toast("System status load failed: " + e, true);
    }
  }

  // ---------------------------------------------------------------------------
  // Render the mode card
  // ---------------------------------------------------------------------------
  function renderModeCard(d) {
    var modeCard = $("modeCard");
    modeCard.classList.remove("hidden");

    // --- mode badge ---
    var badge = $("modeBadge");
    var mode = d.mode || "unknown";
    badge.className = "mode-badge " + (mode === "serving" ? "serving" : mode === "benchmark" ? "benchmark" : "unknown");
    badge.textContent = mode === "serving" ? "Mode: Running (serving)" : mode === "benchmark" ? "Mode: Benchmark" : "Mode: " + mode;

    // --- disabled note + buttons ---
    var switchEnabled = (d.mode_switch_enabled !== false); // treat missing as enabled
    var disabledNote = $("modeSwitchDisabledNote");
    var modeBtns = $("modeBtns");
    var modeWorking = $("modeWorking");

    if (modeInFlight) {
      // While a mode switch is in flight: hide buttons, show spinner
      modeBtns.classList.add("hidden");
      modeWorking.classList.remove("hidden");
      disabledNote.classList.add("hidden");
    } else if (!switchEnabled) {
      // mode_switch_enabled === false: hide buttons, show note
      modeBtns.classList.add("hidden");
      modeWorking.classList.add("hidden");
      disabledNote.classList.remove("hidden");
    } else {
      // Normal: show the appropriate switch button(s)
      modeWorking.classList.add("hidden");
      disabledNote.classList.add("hidden");
      modeBtns.classList.remove("hidden");
      // Show "Switch to Running" only when in benchmark; "Switch to Benchmark" only when in serving.
      // Show both when mode is unknown so the user can recover.
      var showToServing = (mode !== "serving");
      var showToBenchmark = (mode !== "benchmark");
      $("btnToServing").classList.toggle("hidden", !showToServing);
      $("btnToBenchmark").classList.toggle("hidden", !showToBenchmark);
    }
  }

  // ---------------------------------------------------------------------------
  // Render the full status blob
  // ---------------------------------------------------------------------------
  function renderStatus(d) {
    // --- enabled / available guards ---
    if (!d.enabled) {
      $("disabledCard").classList.remove("hidden");
      $("masterCard").classList.add("hidden");
      $("advCard").classList.add("hidden");
      $("vramCard").classList.add("hidden");
      $("modeCard").classList.add("hidden");
      return;
    }
    $("disabledCard").classList.add("hidden");
    if (!d.available) {
      $("unavailableCard").classList.remove("hidden");
      $("unavailableReason").textContent = d.reason || "system-control sidecar unreachable";
      $("masterCard").classList.add("hidden");
      $("advCard").classList.add("hidden");
      $("modeCard").classList.add("hidden");
      return;
    }
    $("unavailableCard").classList.add("hidden");
    $("masterCard").classList.remove("hidden");
    $("advCard").classList.remove("hidden");
    $("vramCard").classList.remove("hidden");

    // --- mode card ---
    renderModeCard(d);

    // --- state badge ---
    renderStateBadge(d.state);

    // --- power buttons: disabled only during a transition or mode-switch in flight ---
    var transitioning = (d.state === "powering_on" || d.state === "powering_off" || inFlight);
    var working = $("workingMsg");
    var btns = $("powerBtns");
    if (transitioning) {
      working.classList.remove("hidden");
      btns.classList.add("hidden");
    } else {
      working.classList.add("hidden");
      btns.classList.remove("hidden");
    }
    // Power buttons are also disabled while a mode switch is in flight
    var buttonsDisabled = transitioning || modeInFlight;
    $("btnOn").disabled = buttonsDisabled;
    $("btnOff").disabled = buttonsDisabled;
    // Remove any legacy disabled-card styling (mode no longer disables power control)
    $("masterCard").classList.remove("disabled-card");
    $("advCard").classList.remove("disabled-card");

    // --- idle info ---
    var idleInfo = $("idleInfo");
    var idleParts = [];
    if (d.idle_unload_enabled) {
      idleParts.push("Auto-unload: after " + (d.idle_unload_minutes || "?") + " min idle (components flagged auto-unload)");
    } else {
      idleParts.push("Auto-unload: disabled");
    }
    idleParts.push("Block-and-warm: " + (d.block_and_warm ? "on (first query after idle reloads the generator)" : "off"));
    idleInfo.textContent = idleParts.join(" \xb7 ");

    // --- per-component advanced rows ---
    var comps = Array.isArray(d.components) ? d.components : [];
    renderComponents(comps, buttonsDisabled);
  }

  function stateBadgeClass(state) {
    if (state === "on") return "on";
    if (state === "off") return "off";
    if (state === "partial") return "partial";
    if (state === "powering_on" || state === "powering_off") return "transitioning";
    return "unknown";
  }
  function renderStateBadge(state) {
    var el = $("stateBadge");
    el.className = "state-badge " + stateBadgeClass(state);
    el.textContent = state || "unknown";
  }

  // ---------------------------------------------------------------------------
  // Per-component rows (advanced)
  // ---------------------------------------------------------------------------
  function compStateChip(state) {
    var cls = "muted";
    if (state === "on") cls = "ok";
    else if (state === "off") cls = "muted";
    else if (state === "partial") cls = "warn";
    else if (state === "absent") cls = "bad";
    else if (state === "unreachable") cls = "bad";
    return '<span class="chip ' + cls + '">' + esc(state || "unknown") + '</span>';
  }
  function renderComponents(comps, buttonsDisabled) {
    var wrap = $("compRows"); wrap.innerHTML = "";
    if (comps.length === 0) {
      wrap.innerHTML = '<p class="muted" style="padding:10px 0">No components reported by the sidecar.</p>';
      return;
    }
    comps.forEach(function (comp) {
      var div = document.createElement("div");
      div.className = "comp-row";

      // Container list: show name, running, health (compact)
      var ctrs = Array.isArray(comp.containers) ? comp.containers : [];
      var ctrLines = ctrs.map(function (c) {
        var running = c.running ? "running" : "stopped";
        var health = c.health ? " \xb7 " + c.health : "";
        return esc(c.name || "?") + " [" + running + health + "]";
      }).join(", ");

      // Idle-unload badge
      var idleBadge = comp.idle_unload
        ? ' <span class="chip warn" title="auto-unloads when idle">auto-unload</span>'
        : "";

      // Per-row start/stop buttons
      var startBtn = '<button class="btn-sm btn-primary" data-comp-key="' + esc(comp.key || "") + '" data-comp-action="on"' + (buttonsDisabled ? " disabled" : "") + '>Start</button>';
      var stopBtn = '<button class="btn-sm btn-danger" data-comp-key="' + esc(comp.key || "") + '" data-comp-action="off"' + (buttonsDisabled ? " disabled" : "") + '>Stop</button>';

      div.innerHTML = (
        '<div class="comp-meta">' +
          '<div class="comp-label">' + esc(comp.label || comp.key || "?") + idleBadge + ' ' + compStateChip(comp.state) + '</div>' +
          '<div class="comp-desc">' + esc(comp.description || "") + '</div>' +
          (ctrLines ? '<div class="comp-containers">' + ctrLines + '</div>' : '') +
        '</div>' +
        '<div class="comp-actions">' + startBtn + stopBtn + '</div>'
      );
      wrap.appendChild(div);
    });

    // Wire per-component button clicks via event delegation on the container
    wrap.addEventListener("click", function (ev) {
      var btn = ev.target.closest && ev.target.closest("button[data-comp-key]");
      if (!btn) return;
      var key = btn.getAttribute("data-comp-key");
      var action = btn.getAttribute("data-comp-action");
      if (key && action) postPower(action, [key]);
    });
  }

  // ---------------------------------------------------------------------------
  // POST /system/power — master or per-component
  // ---------------------------------------------------------------------------
  async function postPower(action, components) {
    // components === null means master (all components)
    inFlight = true;
    $("btnOn").disabled = true;
    $("btnOff").disabled = true;
    $("workingMsg").classList.remove("hidden");
    $("powerBtns").classList.add("hidden");
    try {
      var body = { action: action };
      if (components !== null) body.components = components;
      // No aggressive timeout: block-and-warm can take 60 s or more.
      var r = await fetch(API + "/system/power", {
        method: "POST",
        headers: authHeaders({ "Content-Type": "application/json" }),
        body: JSON.stringify(body)
      });
      if (r.status === 401) { setToken(null); showLogin(); return; }
      if (r.status === 403) {
        toast("You need the power-control group to flip the switch (read-only access).", true);
        inFlight = false;
        loadStatus();
        return;
      }
      if (!r.ok) {
        var msg = "Power action failed (" + r.status + ")";
        try { var ej = await r.json(); if (ej && (ej.error || ej.detail)) msg = ej.error || ej.detail; } catch (x) {}
        toast(msg, true);
        inFlight = false;
        loadStatus();
        return;
      }
      var d = await r.json();
      inFlight = false;
      renderStatus(d);
      toast("Power " + action + " applied.");
    } catch (e) {
      toast("Power action failed: " + e, true);
      inFlight = false;
      loadStatus();
    }
  }

  // ---------------------------------------------------------------------------
  // POST /system/mode — switch between serving and benchmark stacks
  // ---------------------------------------------------------------------------
  async function postMode(target) {
    modeInFlight = true;
    // Disable power buttons and mode buttons immediately
    $("btnOn").disabled = true;
    $("btnOff").disabled = true;
    $("modeBtns").classList.add("hidden");
    $("modeWorking").classList.remove("hidden");
    try {
      // No timeout: switching to serving loads the generator and can take 30-120 s.
      var r = await fetch(API + "/system/mode", {
        method: "POST",
        headers: authHeaders({ "Content-Type": "application/json" }),
        body: JSON.stringify({ target: target })
      });
      if (r.status === 401) { setToken(null); showLogin(); return; }
      if (r.status === 403) {
        toast("You need the power-control group to switch modes.", true);
        modeInFlight = false;
        loadStatus();
        return;
      }
      if (!r.ok) {
        var errMsg = "Mode switch failed (" + r.status + ")";
        try { var ej2 = await r.json(); if (ej2 && (ej2.error || ej2.detail)) errMsg = ej2.error || ej2.detail; } catch (x) {}
        toast(errMsg, true);
        modeInFlight = false;
        loadStatus();
        return;
      }
      var d = await r.json();
      modeInFlight = false;
      // Re-render from returned status; also refresh VRAM gauge immediately
      renderStatus(d);
      pollVram();
      toast("Switched to " + target + " mode.");
    } catch (e) {
      toast("Mode switch failed: " + e, true);
      modeInFlight = false;
      loadStatus();
    }
  }

  // ---------------------------------------------------------------------------
  // Wire up
  // ---------------------------------------------------------------------------
  $("loginBtn").addEventListener("click", login);
  $("p").addEventListener("keydown", function (e) { if (e.key === "Enter") login(); });
  $("logout").addEventListener("click", function () {
    setToken(null);
    inFlight = false;
    modeInFlight = false;
    showLogin();
  });
  $("btnOn").addEventListener("click", function () { postPower("on", null); });
  $("btnOff").addEventListener("click", function () { postPower("off", null); });
  $("btnToServing").addEventListener("click", function () { postMode("serving"); });
  $("btnToBenchmark").addEventListener("click", function () { postMode("benchmark"); });

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


def system_page_html() -> str:
    """Return the standalone System / Management view HTML page.

    The view is a self-contained inline HTML+CSS+vanilla-JS page
    mirroring :mod:`iw_rag.api.config_page` and
    :mod:`iw_rag.api.scoreboard_page`.  It carries the IW brand chrome
    (teal ``#0D9488`` / ``#2DD4BF`` on ink ``#181D29``), the
    location-derived API base, the sessionStorage-only bearer token,
    the ``/auth/login`` self-handled sign-in, the unified eight-item
    nav (Config / Benchmark / Results / Compare / Scoreboard / Chat /
    Upload / System), the VRAM gauge (polled every 2 s via ``GET
    /system/vram``), a Mode card above the VRAM gauge (current mode
    badge, Switch to Running / Switch to Benchmark buttons that call
    ``POST /system/mode``, in-flight spinner with a 2-minute warning,
    ``mode_switch_enabled`` guard), the master power switch (``POST
    /system/power`` with ``{action, components:null}``), the
    per-component advanced expander (per-row start/stop with ``POST
    /system/power`` and ``{action, components:[key]}``), the idle-unload
    / block-and-warm info line, and the 403/degraded/unavailable
    states.

    Power control is active in both serving and benchmark modes
    (``power_control_active`` is always ``true`` from the backend);
    power buttons are only disabled while a transition or mode-switch
    is in flight, or on a 403.

    Returns:
        The raw, self-contained HTML document as a ``str`` (no MIME /
        static-mount wiring required).
    """
    return add_favicon(_PAGE)


__all__ = ["system_page_html"]
