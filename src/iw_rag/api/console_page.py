# ruff: noqa: E501
"""Self-contained, branded IW RAG Operator Console shell (F-00096, S10).

Chainlit is chat-first with no multi-page routing or data tables, so
the IW RAG Operator Console is a standalone, single-page htmx shell
served by the API at ``GET /console`` (reached through Caddy at
``/api/console``).  It is the **unified nav and shared chrome** every
console view (Config / Benchmark / Results / Scoreboard / Chat /
Upload) renders inside: the ``/auth/login`` self-handled sign-in, the
location-derived API base, the sessionStorage-only bearer token, the
Innovation Ways visual identity (teal ``#0D9488``/``#2DD4BF`` on ink
``#181D29``), the brand tokens resolved from
``frontend/public/theme.json``.

The shell is a **thin API client** (Invariant 4): it contains NO
retrieval, ACL, grounding, or guard logic — all of that stays
server-side.  The S11–S15 view pages load their own partials into the
``#viewContent`` region inside the shell; the shell itself never
re-implements any business rule.

Kept inline (not a static asset) so it is always packaged with the
wheel and needs no extra MIME/static-mount wiring.  The page derives
its API base from its own URL so it works behind Caddy (``/api``) or
when hit directly (``/``).  No external CDN / Google Fonts link
(Invariant 8, local-only rule).

Mirrors :mod:`iw_rag.api.benchmark_page` and
:mod:`iw_rag.api.documents_page` — same header/footer chrome, same
login flow, same teal/ink palette, same sessionStorage-only bearer
token, same ``node --check``-passing raw-string ``_PAGE``.  The S10
route ``GET /console`` is wired against :func:`console_page_html`.
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
  input[type=text], input[type=password] {
    width: 100%; padding: 11px 12px; border-radius: 10px; border: 1px solid var(--line);
    background: var(--ink-2); color: var(--text); font: inherit;
  }
  input:disabled, button:disabled { opacity: .6; cursor: not-allowed; }
  .toast { position: fixed; bottom: 22px; left: 50%; transform: translateX(-50%); background: var(--ink-3); border: 1px solid var(--line); padding: 12px 18px; border-radius: 10px; opacity: 0; transition: .25s; pointer-events: none; max-width: 90vw; z-index: 10; }
  .toast.show { opacity: 1; }
  .toast.err { border-color: var(--danger); color: #ffd7de; }
  .hidden { display: none !important; }
  .muted { color: var(--muted); font-size: 13px; }
  .loginwrap { max-width: 420px; margin: 8vh auto; }
  /* Unified nav.  Each item is a stable anchor (id + href) that the
     S11–S15 sub-views and the S23 browser verification can target
     deterministically.  The "active" item is highlighted via the
     .nav-item.active class — the page script sets it on load. */
  nav.console-nav { display: flex; gap: 6px; flex-wrap: wrap; align-items: center; padding: 14px 28px; background: rgba(24,29,41,0.5); border-bottom: 1px solid var(--line); }
  nav.console-nav a.nav-item { font-size: 13px; padding: 6px 12px; border-radius: 8px; border: 1px solid transparent; color: var(--muted); }
  nav.console-nav a.nav-item:hover { color: var(--text); border-color: var(--line); }
  nav.console-nav a.nav-item.active { color: #06231f; background: linear-gradient(180deg, var(--teal) 0%, var(--teal-deep) 100%); border-color: transparent; font-weight: 600; }
</style>
</head>
<body>
<header>
  <span class="logo"><span class="i">I</span>nnovation <span class="w">W</span>ays</span>
  <span class="muted">· Operator Console</span>
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
      <p class="sub">Use your demo credentials. The Operator Console surfaces Config, Benchmark, Results, Scoreboard, Chat, and Upload under one signed-in session.</p>
      <label for="u">Username</label>
      <input type="text" id="u" autocomplete="username" />
      <label for="p">Password</label>
      <input type="password" id="p" autocomplete="current-password" />
      <div style="margin-top:18px"><button class="btn-primary" id="loginBtn">Sign in</button></div>
    </div>
  </section>

  <section id="appView" class="hidden">
    <div class="card" id="welcomeCard">
      <h2>Welcome</h2>
      <p class="sub">Pick a destination above. The S11–S15 sub-views mount their own markup into the <code>#viewContent</code> region below; this shell is the single host page and the unified nav stays put.</p>
    </div>

    <div class="card" id="viewContent">
      <h2>View</h2>
      <p class="muted">No view loaded — pick a destination from the nav above. (The S11–S15 sub-view routes own the content of this region.)</p>
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
  }
  function highlightActiveNav() {
    // Mark the nav item whose href matches the current path.  Lets the
    // operator see which sub-view they are on at a glance.
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

  $("loginBtn").addEventListener("click", login);
  $("p").addEventListener("keydown", function (e) { if (e.key === "Enter") login(); });
  $("logout").addEventListener("click", function () { setToken(null); showLogin(); });

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


def console_page_html() -> str:
    """Return the standalone IW RAG Operator Console shell HTML page.

    The shell is the single host page every console view (Config,
    Benchmark, Results, Scoreboard, Chat, Upload) renders inside.  It
    carries the IW brand chrome, the unified six-item nav, the
    ``/auth/login`` self-handled sign-in, the location-derived API
    base, the sessionStorage-only bearer token, and the
    ``#viewContent`` region the S11–S15 sub-view partials load into.

    Returns:
        The raw, self-contained HTML document as a ``str`` (no MIME /
        static-mount wiring required).
    """
    return add_favicon(_PAGE)


__all__ = ["console_page_html"]
