# ruff: noqa: E501
"""Self-contained, branded IW RAG Operator Console Config view (F-00096, S11).

The Config view is the Operator Console page that renders the live
configuration with per-key explanations, an edit form for the
allowlisted keys (with bounds), a locked / pending-restart badge on
each row, and a change-log panel surfacing the recent
``config_change_log`` entries (who / when / key / old→new / hot-or-
pending) for the audit trail.  It is a **thin API client** —
Invariant 4: it enforces NO allowlist, bounds, or lock logic itself.
The UI's disabled-for-locked-keys is UX-only honesty; the API is the
authoritative gate (mirror :mod:`iw_rag.api.benchmark_page`'s
"UX-only gate" comment on the Run button).

It is a **server-rendered FastAPI HTML page** in the
``benchmark_page.py`` style — NOT Chainlit, NOT a JS SPA, NO new
build toolchain (per the F-00096 design *Out-of-Scope* row "A
JavaScript SPA / new build toolchain").  The page derives its API
base from its own URL so it works behind Caddy (``/api``) or when
hit directly (``/``).  No external CDN / Google Fonts link
(Invariant 8, local-only rule).

The page consumes the S07 endpoints:

* ``GET /api/config`` — the active values + per-key ``explanation``
  + ``editable`` / ``locked`` / ``pending_restart`` flags + bounds
  (``min``/``max`` for range keys, ``choices`` for enum keys).  The
  page renders the response verbatim; it does no allowlist / bounds /
  ACL math of its own (Invariant 4).  Locked keys' ``value`` carries
  the redacted placeholder the API sent — never a raw secret.
* ``PUT /api/config/{key}`` — the edit submission.  The body carries
  ONLY ``value`` (Invariant 3; mirrors ``ConfigUpdateRequest`` /
  ``QueryRequest`` / ``ActivateRequest`` / ``BenchmarkRunRequest``);
  the principal is sourced from the authenticated
  :class:`AccessContext` (auth-only) by the S07 route.

Mirrors :mod:`iw_rag.api.benchmark_page` and
:mod:`iw_rag.api.console_page` — same header/footer chrome, same
teal/ink palette, same location-derived API base regex idiom with
``\\/`` (the forward-slash escape that keeps the JS parser from
closing the literal at the first ``/``), same sessionStorage-only
bearer token.  The S11 route ``GET /console/config`` is wired
against :func:`config_page_html`.
"""

from __future__ import annotations

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
  input[type=text], input[type=number], select {
    width: 100%; padding: 10px 12px; border-radius: 10px; border: 1px solid var(--line);
    background: var(--ink-2); color: var(--text); font: inherit;
  }
  input:disabled, select:disabled, button:disabled { opacity: .6; cursor: not-allowed; }
  table { width: 100%; border-collapse: collapse; }
  th, td { text-align: left; padding: 10px 8px; border-bottom: 1px solid var(--line); font-size: 13px; vertical-align: top; }
  th { color: var(--muted); font-weight: 600; font-size: 11px; text-transform: uppercase; letter-spacing: .5px; }
  td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
  .chip { font-size: 11px; padding: 2px 8px; border-radius: 999px; border: 1px solid var(--line); white-space: nowrap; display: inline-block; }
  .chip.ok { color: var(--ok); border-color: var(--teal-deep); }
  .chip.bad { color: var(--danger); border-color: #5b2230; }
  .chip.warn { color: var(--warn); border-color: #5b4322; }
  .chip.muted { color: var(--muted); }
  .chip-locked { color: var(--danger); border-color: #5b2230; background: rgba(244,63,94,0.06); font-weight: 600; }
  .chip-pending-restart { color: var(--warn); border-color: #5b4322; background: rgba(245,158,11,0.06); font-weight: 600; }
  .row { display: flex; gap: 16px; flex-wrap: wrap; align-items: flex-end; }
  .row > div { flex: 1; min-width: 180px; }
  .toast { position: fixed; bottom: 22px; left: 50%; transform: translateX(-50%); background: var(--ink-3); border: 1px solid var(--line); padding: 12px 18px; border-radius: 10px; opacity: 0; transition: .25s; pointer-events: none; max-width: 90vw; z-index: 10; }
  .toast.show { opacity: 1; }
  .toast.err { border-color: var(--danger); color: #ffd7de; }
  .hidden { display: none !important; }
  .muted { color: var(--muted); font-size: 13px; }
  .catalog-warn { background: rgba(245,158,11,0.08); border: 1px solid #5b4322; color: #fde7c0; padding: 10px 12px; border-radius: 10px; font-size: 13px; margin: 6px 0 0; }
  .inline-err { background: rgba(244,63,94,0.10); border: 1px solid #5b2230; color: #ffd7de; padding: 8px 12px; border-radius: 8px; font-size: 12px; margin: 6px 0 0; }
  nav.console-nav { display: flex; gap: 6px; flex-wrap: wrap; align-items: center; padding: 14px 28px; background: rgba(24,29,41,0.5); border-bottom: 1px solid var(--line); }
  nav.console-nav a.nav-item { font-size: 13px; padding: 6px 12px; border-radius: 8px; border: 1px solid transparent; color: var(--muted); }
  nav.console-nav a.nav-item:hover { color: var(--text); border-color: var(--line); }
  nav.console-nav a.nav-item.active { color: #06231f; background: linear-gradient(180deg, var(--teal) 0%, var(--teal-deep) 100%); border-color: transparent; font-weight: 600; }
  .config-row { display: grid; grid-template-columns: 1.4fr 1fr 1.2fr auto; gap: 14px; align-items: center; padding: 14px 0; border-bottom: 1px solid var(--ink-2); }
  .config-row:last-child { border-bottom: none; }
  .config-row .key { font-family: "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; color: var(--text); font-size: 13px; }
  .config-row .explanation { color: var(--muted); font-size: 12px; line-height: 1.4; }
  .config-row .badges { display: flex; gap: 6px; align-items: center; flex-wrap: wrap; }
  @media (max-width: 720px) { .config-row { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<header>
  <span class="logo"><span class="i">I</span>nnovation <span class="w">W</span>ays</span>
  <span class="muted">· Config</span>
  <span class="spacer"></span>
  <span class="who" id="who"></span>
  <button class="btn-ghost hidden" id="logout">Sign out</button>
</header>

__CONSOLE_NAV__

<main>
  <section id="loginView" class="hidden">
    <div class="card loginwrap">
      <h1>Sign in</h1>
      <p class="sub">Use your demo credentials. The Config view is gated to the <strong>eval-operator</strong> group for edits; read-only viewing is open to any authenticated principal.</p>
      <label for="u">Username</label>
      <input type="text" id="u" autocomplete="username" />
      <label for="p">Password</label>
      <input type="password" id="p" autocomplete="current-password" />
      <div style="margin-top:18px"><button class="btn-primary" id="loginBtn">Sign in</button></div>
    </div>
  </section>

  <section id="appView" class="hidden">
    <div class="card" id="configCard">
      <h2>Active configuration</h2>
      <p class="sub">Per-key active value, one-line explanation, and (for the allowlisted editable set) the edit form. The edit gate is server-side: <code>PUT /api/config/{key}</code> is gated to the <code>eval-operator</code> group, the locked set is enforced in the S03 service, and the audit row carries the authenticated <code>principal_id</code> (never from the body). The badges below are UX-only honesty — the API is the authoritative gate.</p>

      <div id="configWarn" class="catalog-warn hidden">Config API unavailable — sign in again or check the <code>GET /api/config</code> route.</div>

      <div id="configList" class="config-list">
        <p class="muted">Loading&hellip;</p>
      </div>
    </div>

    <div class="card" id="changeLogCard">
      <h2>Change log</h2>
      <p class="sub">Recent <code>config_change_log</code> entries — who changed what, when, and whether the edit was applied hot or flagged pending restart. The read surface (S07) is not yet wired: S11 renders the panel as an explicit empty state with the dependency noted (Boundary row "no change-log endpoint wired").</p>
      <div id="changeLogPanel" class="config-list">
        <p class="muted" id="changeLogEmpty">No change-log endpoint wired &mdash; the S07 read surface for <code>config_change_log</code> is not yet exposed. (Boundary row: render an explicit empty state, not an error. The audit-write path <em>is</em> live &mdash; <code>PUT /api/config/{key}</code> writes one row per accepted edit.)</p>
      </div>
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
  // Config table — consumes GET /api/config verbatim.  Renders one row
  // per item: key, active value, per-key explanation, and (for editable
  // items) the edit form wired to the registry bounds.  Locked items
  // show the redacted placeholder the API sent + a locked badge.
  // Pending-restart items show a pending-restart badge.
  // -------------------------------------------------------------------------
  function badgeFor(item) {
    var parts = [];
    if (item.locked) parts.push('<span class="chip chip-locked" data-badge="locked">Locked</span>');
    if (item.pending_restart) parts.push('<span class="chip chip-pending-restart" data-badge="pending-restart">Pending restart</span>');
    return parts.length ? parts.join(" ") : '<span class="chip muted" data-badge="none">&mdash;</span>';
  }
  function renderValue(item) {
    var v = item.value;
    if (item.choices && Array.isArray(item.choices) && item.choices.length) {
      var opts = item.choices.map(function (c) {
        return '<option value="' + esc(c) + '"' + (String(v) === String(c) ? " selected" : "") + '>' + esc(c) + '</option>';
      }).join("");
      var disabled = (!item.editable || item.locked) ? " disabled" : "";
      return '<select data-input="value"' + disabled + '>' + opts + '</select>';
    }
    if (typeof v === "number" || (item.min != null && item.max != null)) {
      var step = "1";
      if (typeof v === "number" && v % 1 !== 0) step = "0.01";
      if (item.min != null && (typeof v !== "number" || v % 1 !== 0)) step = "0.01";
      var min = item.min != null ? ' min="' + esc(String(item.min)) + '"' : "";
      var max = item.max != null ? ' max="' + esc(String(item.max)) + '"' : "";
      var disabled = (!item.editable || item.locked) ? " disabled" : "";
      return '<input type="number" data-input="value" value="' + esc(String(v)) + '"' + min + max + ' step="' + step + '"' + disabled + ' />';
    }
    var disabled = (!item.editable || item.locked) ? " disabled" : "";
    return '<input type="text" data-input="value" value="' + esc(v == null ? "" : String(v)) + '"' + disabled + ' />';
  }
  function renderRow(item) {
    var editCol = "";
    if (item.editable && !item.locked) {
      editCol = (
        '<div class="badges">' + badgeFor(item) + '</div>' +
        '<div style="margin-top:8px;display:flex;gap:8px;align-items:center;">' +
        renderValue(item) +
        '<button class="btn-primary" data-save-key="' + esc(item.key) + '">Save</button>' +
        '</div>' +
        '<div class="inline-err hidden" data-err-key="' + esc(item.key) + '"></div>'
      );
    } else {
      editCol = (
        '<div class="badges">' + badgeFor(item) + '</div>' +
        '<div class="muted" style="margin-top:6px;">' + esc(item.value == null ? "" : String(item.value)) + '</div>'
      );
    }
    return (
      '<div class="config-row" data-row-key="' + esc(item.key) + '">' +
        '<div><div class="key">' + esc(item.key) + '</div>' +
          '<div class="explanation" style="margin-top:4px;">' + esc(item.explanation || "") + '</div></div>' +
        '<div>' + editCol + '</div>' +
        '<div class="muted">' + (item.choices ? "enum: " + esc(item.choices.join(", ")) : (item.min != null && item.max != null ? "range: [" + esc(String(item.min)) + ", " + esc(String(item.max)) + "]" : "")) + '</div>' +
        '<div class="badges">' + (item.editable ? '<span class="chip ok">editable</span>' : '<span class="chip muted">read-only</span>') + '</div>' +
      '</div>'
    );
  }
  async function loadConfig() {
    var list = $("configList");
    try {
      var r = await fetch(API + "/api/config", { headers: authHeaders() });
      if (r.status === 401) { setToken(null); showLogin(); return; }
      if (!r.ok) throw new Error("status " + r.status);
      var d = await r.json();
      var items = (d && d.items) || [];
      if (items.length === 0) {
        list.innerHTML = '<p class="muted">No config keys registered.</p>';
        return;
      }
      list.innerHTML = items.map(renderRow).join("");
      // Wire up Save buttons (UX-only gate: the API is the authoritative
      // gate; the disabled-for-locked-keys is honesty, not security).
      list.addEventListener("click", function (e) {
        var btn = e.target.closest ? e.target.closest("button[data-save-key]") : null;
        if (!btn) return;
        var key = btn.getAttribute("data-save-key");
        if (!key) return;
        var row = list.querySelector('[data-row-key="' + key.replace(/"/g, '\\"') + '"]');
        if (!row) return;
        var input = row.querySelector('[data-input="value"]');
        if (!input) return;
        // Parse the value (numeric vs string) so the API gets the right
        // type — never trust the input's string form for a number field.
        var raw = input.value;
        var value;
        if (input.tagName === "INPUT" && input.type === "number") {
          value = input.step && input.step.indexOf(".") !== -1 ? parseFloat(raw) : parseInt(raw, 10);
        } else {
          value = raw;
        }
        saveEdit(key, value, row);
      });
    } catch (e) {
      $("configWarn").classList.remove("hidden");
      list.innerHTML = '<p class="muted">Config load failed.</p>';
    }
  }
  async function saveEdit(key, value, row) {
    // UX-only gate (mirror benchmark_page's "Run Full Test" gate): the
    // PUT body carries ONLY `value` — NEVER principal_id / groups / acl
    // (Invariant 3).  The principal is sourced from the resolved
    // AccessContext by the S07 route.
    var err = row.querySelector('[data-err-key="' + key.replace(/"/g, '\\"') + '"]');
    if (err) { err.classList.add("hidden"); err.textContent = ""; }
    try {
      var r = await fetch(API + "/api/config/" + encodeURIComponent(key), {
        method: "PUT",
        headers: authHeaders({ "Content-Type": "application/json" }),
        body: JSON.stringify({ value: value })
      });
      if (r.status === 401) { setToken(null); showLogin(); return; }
      if (r.status === 403) { toast("eval-operator group required (server-enforced)", true); return; }
      if (!r.ok) {
        var msg = "Edit failed (" + r.status + ")";
        try { var ej = await r.json(); if (ej && (ej.error || ej.detail)) msg = ej.error || ej.detail; } catch (x) {}
        if (err) { err.textContent = msg; err.classList.remove("hidden"); }
        toast(msg, true);
        return;
      }
      var body = await r.json();
      var mode = body && body.applied_mode;
      if (mode === "hot") {
        toast("Saved (hot) — applied on the next request.");
      } else if (mode === "pending_restart") {
        toast("Saved (pending restart) — running request path is unchanged until restart.");
      } else {
        toast("Saved.");
      }
      loadConfig();
    } catch (e) {
      if (err) { err.textContent = String(e); err.classList.remove("hidden"); }
      toast("Edit failed: " + e, true);
    }
  }

  $("loginBtn").addEventListener("click", login);
  $("p").addEventListener("keydown", function (e) { if (e.key === "Enter") login(); });
  $("logout").addEventListener("click", function () { setToken(null); showLogin(); });

  async function bootstrap() {
    if (token()) { showApp(sessionStorage.getItem(whoKey), groups()); loadConfig(); return; }
    showLogin();
  }
  bootstrap();
})();
</script>
</body>
</html>
"""


def config_page_html() -> str:
    """Return the standalone Config view HTML page.

    The view is a self-contained inline HTML+CSS+vanilla-JS page
    mirroring :mod:`iw_rag.api.benchmark_page` and
    :mod:`iw_rag.api.console_page`.  It carries the IW brand chrome
    (teal ``#0D9488`` / ``#2DD4BF`` on ink ``#181D29``), the
    :class:`API <APIRouter>`-derived API base, the sessionStorage-only
    bearer token, the ``/auth/login`` self-handled sign-in, the
    per-key active-value table (with explanation, locked badge,
    pending-restart badge, edit form wired to bounds), and the
    change-log panel anchor.

    Returns:
        The raw, self-contained HTML document as a ``str`` (no MIME /
        static-mount wiring required).
    """
    return apply_console_nav(add_favicon(_PAGE))


__all__ = ["config_page_html"]
