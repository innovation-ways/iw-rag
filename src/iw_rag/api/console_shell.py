# ruff: noqa: E501
"""Shared chrome for the native Operator Console views (Chat, Upload, …).

Every console view is a self-contained, branded, single-page htmx/vanilla-JS
document served by the FastAPI app.  They all share the same head + brand
styling + sign-in + token plumbing + unified nav; only the body markup, a small
slice of view-specific CSS, and a view ``<script>`` differ.  This module owns
the shared parts so each view module is just its own body + CSS + JS.

Contract for a view module::

    from iw_rag.api.console_shell import render_console_page

    _BODY = "<div class='card'>…</div>"          # fills #appView (post sign-in)
    _CSS  = ".my-thing { … }"                     # appended into the <style>
    _JS   = "<script>(function(){ window.onConsoleReady = function(){ … }; })();</script>"

    def my_page_html() -> str:
        return render_console_page(title="Chat", extra_css=_CSS, body=_BODY, view_js=_JS)

The shell guarantees, for the view script, that by the time
``window.onConsoleReady`` is called:

* ``window.IWConsole`` exists with ``{ API, authHeaders, esc, toast, token, $ }``
  — ``API`` is the fixed ``"/api"`` edge base (the Caddy edge strips ``/api/*``),
  ``authHeaders(extra)`` returns headers with the bearer token, ``esc(s)`` is an
  HTML-escaper, ``toast(msg, isErr)`` shows a transient toast, ``token()`` reads
  the sessionStorage bearer, ``$(id)`` is ``getElementById``.
* The principal is signed in (the shell only calls ``onConsoleReady`` from
  ``showApp``), and the unified nav + ``#appView`` are visible.

The shell defers its first paint to ``DOMContentLoaded`` so the per-view
``<script>`` (which runs after the shell script during parsing) has already
defined ``window.onConsoleReady`` — otherwise a token-bearing reload would call
``showApp`` before the view's init hook exists and the view would never wire up.

No external CDN / hosted font (local-only, Invariant 8).  ``__TITLE__`` /
``__EXTRA_CSS__`` / ``__BODY__`` / ``__VIEWJS__`` are the only placeholder
tokens and appear exactly once each — keep them out of prose/comments so a
``str.replace`` never double-substitutes.
"""

from __future__ import annotations

# Innovation Ways mark as an inline SVG data-URI favicon, shared by every console
# document — the shared shell (Chat/Upload) *and* the standalone pages that build
# their own <head> (Console/Config/Benchmark/Runner/Results/Scoreboard/Documents).
# It is INLINED (a data URI, not a served static file or CDN asset) so the console
# stays fully local-only: the browser never makes an external request for it
# (Invariant 8). Without a favicon at all, the browser requests /favicon.ico, gets
# a 404, and the tab shows no icon — which is the bug this fixes. Injected after
# the <title> via ``add_favicon`` so there is a single source of truth for the mark.
FAVICON_LINK = (
    '<link rel="icon" type="image/svg+xml" href="data:image/svg+xml;base64,'
    "PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCA2MCA2MCI+PGRl"
    "ZnM+PGxpbmVhckdyYWRpZW50IGlkPSJnIiB4MT0iMCUiIHkxPSIwJSIgeDI9IjEwMCUiIHkyPSIxMDAlIj48"
    "c3RvcCBvZmZzZXQ9IjAlIiBzdG9wLWNvbG9yPSIjMkU4NkFCIi8+PHN0b3Agb2Zmc2V0PSIxMDAlIiBzdG9w"
    "LWNvbG9yPSIjMUIyQTRBIi8+PC9saW5lYXJHcmFkaWVudD48L2RlZnM+PGNpcmNsZSBjeD0iMjQiIGN5PSIx"
    "OCIgcj0iMTAiIGZpbGw9InVybCgjZykiLz48Y2lyY2xlIGN4PSI0NCIgY3k9IjM4IiByPSI4IiBmaWxsPSIj"
    "MkU4NkFCIiBvcGFjaXR5PSIwLjg1Ii8+PGNpcmNsZSBjeD0iMTQiIGN5PSI0NCIgcj0iNiIgZmlsbD0iI0Yx"
    "OEYwMSIgb3BhY2l0eT0iMC45Ii8+PGxpbmUgeDE9IjI0IiB5MT0iMTgiIHgyPSI0NCIgeTI9IjM4IiBzdHJv"
    "a2U9IiMyRTg2QUIiIHN0cm9rZS13aWR0aD0iMi41IiBvcGFjaXR5PSIwLjU1Ii8+PGxpbmUgeDE9IjI0IiB5"
    "MT0iMTgiIHgyPSIxNCIgeTI9IjQ0IiBzdHJva2U9IiMxQjJBNEEiIHN0cm9rZS13aWR0aD0iMi41IiBvcGFj"
    "aXR5PSIwLjQ1Ii8+PGxpbmUgeDE9IjQ0IiB5MT0iMzgiIHgyPSIxNCIgeTI9IjQ0IiBzdHJva2U9IiNGMThG"
    "MDEiIHN0cm9rZS13aWR0aD0iMiIgb3BhY2l0eT0iMC40NSIvPjwvc3ZnPg=="
    '" />'
)


def add_favicon(html: str) -> str:
    """Inject the shared IW favicon ``<link>`` after the document's ``</title>``.

    Used by the standalone console pages that assemble their own ``<head>`` (rather
    than going through :func:`render_console_page`) so they share one source of
    truth for the mark.  Idempotent-ish: only the first ``</title>`` is targeted,
    and pages call this exactly once on their final HTML.

    Args:
        html: A complete console HTML document containing exactly one ``</title>``.

    Returns:
        The same document with the favicon ``<link>`` inserted right after the
        closing ``</title>`` tag.
    """
    return html.replace("</title>", "</title>\n" + FAVICON_LINK, 1)


_SHELL = r"""<!doctype html>
<html lang="en" data-theme="dark">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>IW - RAG</title>
__FAVICON__
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
  input[type=text], input[type=password], textarea, select {
    width: 100%; padding: 11px 12px; border-radius: 10px; border: 1px solid var(--line);
    background: var(--ink-2); color: var(--text); font: inherit;
  }
  textarea { resize: vertical; min-height: 64px; }
  input:disabled, button:disabled, textarea:disabled { opacity: .6; cursor: not-allowed; }
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
  /* Switch (toggle) — shared control used by views. */
  .switch { position: relative; display: inline-block; width: 42px; height: 24px; vertical-align: middle; }
  .switch input { opacity: 0; width: 0; height: 0; }
  .switch .slider { position: absolute; cursor: pointer; inset: 0; background: var(--ink-3); border: 1px solid var(--line); border-radius: 999px; transition: .2s; }
  .switch .slider:before { content: ""; position: absolute; height: 16px; width: 16px; left: 3px; bottom: 3px; background: var(--muted); border-radius: 50%; transition: .2s; }
  .switch input:checked + .slider { background: linear-gradient(180deg, var(--teal) 0%, var(--teal-deep) 100%); border-color: transparent; }
  .switch input:checked + .slider:before { transform: translateX(18px); background: #06231f; }
__EXTRA_CSS__
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

<main class="main-wrap">
  <section id="loginView" class="hidden">
    <div class="card loginwrap">
      <h1>Sign in</h1>
      <p class="sub">Use your demo credentials. Every console view — Config, Benchmark, Results, Scoreboard, Chat, Upload — runs under one sign-in.</p>
      <label for="u">Username</label>
      <input type="text" id="u" autocomplete="username" />
      <label for="p">Password</label>
      <input type="password" id="p" autocomplete="current-password" />
      <div style="margin-top:18px"><button class="btn-primary" id="loginBtn">Sign in</button></div>
    </div>
  </section>

  <section id="appView" class="hidden">
__BODY__
  </section>
</main>

<div class="toast" id="toast"></div>

<script>
(function () {
  // Fixed "/api" edge base: behind the Caddy edge, "/api/*" is prefix-stripped
  // to the FastAPI app, so fetch(API + "/query") -> /api/query -> /query.
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
  function esc(s) {
    return String(s == null ? "" : s)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
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
    if (typeof window.onConsoleReady === "function") {
      try { window.onConsoleReady(); } catch (e) { /* a view init error must not break sign-in */ }
    }
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
  $("loginBtn").addEventListener("click", login);
  $("p").addEventListener("keydown", function (e) { if (e.key === "Enter") login(); });
  $("logout").addEventListener("click", function () { setToken(null); showLogin(); });

  window.IWConsole = { API: API, authHeaders: authHeaders, toast: toast, esc: esc, token: token, $: $ };

  // Defer first paint until the per-view <script> below has run and defined
  // window.onConsoleReady (the view init hook), so a token-bearing reload wires
  // the view up instead of calling showApp() before the hook exists.
  function bootstrap() {
    if (token()) { showApp(sessionStorage.getItem(whoKey), groups()); return; }
    showLogin();
  }
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", bootstrap);
  } else {
    bootstrap();
  }
})();
</script>
__VIEWJS__
</body>
</html>
"""


def render_console_page(*, title: str, extra_css: str, body: str, view_js: str) -> str:
    """Assemble a native console view from the shared shell + a view's parts.

    Args:
        title: The ``<title>`` suffix (cosmetic).
        extra_css: View-specific CSS appended inside the shared ``<style>``.
        body: The view markup that fills ``#appView`` (shown after sign-in).
        view_js: A ``<script>`` block that defines ``window.onConsoleReady`` and
            the view's handlers; it runs after the shared shell script.

    Returns:
        A complete, self-contained HTML document string.
    """
    return (
        _SHELL.replace("__TITLE__", title)
        .replace("__FAVICON__", FAVICON_LINK)
        .replace("__EXTRA_CSS__", extra_css)
        .replace("__BODY__", body)
        .replace("__VIEWJS__", view_js)
    )


__all__ = ["FAVICON_LINK", "add_favicon", "render_console_page"]
