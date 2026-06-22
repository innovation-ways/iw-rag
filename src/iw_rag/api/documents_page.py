# ruff: noqa: E501
"""Self-contained, branded document-management web page (WI-DOCMGMT).

Chainlit is chat-first with no multi-page routing or data tables, so the document
manager is a standalone page served by the API at ``GET /documents/ui`` (reached
through Caddy at ``/api/documents/ui``).  It is a single dependency-free HTML
document: it logs in via ``/auth/login``, lists/uploads/deletes/views/downloads
documents through the ``/documents`` API under the caller's own bearer token, and
carries the Innovation Ways visual identity (teal ``#0D9488``/``#2DD4BF`` on ink
``#181D29``).

Features: sortable column headers (Name / Scope / Chunks / Size / Added; default
Added desc), per-row Download (reuses ``/documents/{id}/raw`` and forces a save
to disk via an object-URL anchor), and 50-row client-side pagination with
Prev/Next controls.  Sorting and pagination are purely client-side over the
already-fetched list (the endpoint returns the full list).

Kept inline (not a static asset) so it is always packaged with the wheel and needs
no extra MIME/static-mount wiring.  The page derives its API base from its own URL
so it works behind Caddy (``/api``) or when hit directly (``/``).
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
    --danger: #f43f5e; --ok: #2DD4BF; --radius: 12px;
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
  .btn-ghost { background: transparent; }
  .btn-danger { background: transparent; color: var(--danger); border-color: #5b2230; }
  .btn-danger:hover { background: #2a1620; border-color: var(--danger); }
  main { max-width: 980px; margin: 0 auto; padding: 28px 20px 80px; }
  .card { background: rgba(34,38,49,0.72); border: 1px solid var(--line); border-radius: var(--radius); padding: 22px; margin-bottom: 22px; }
  h1 { font-family: "Space Grotesk", system-ui, sans-serif; font-size: 22px; margin: 0 0 4px; }
  h2 { font-family: "Space Grotesk", system-ui, sans-serif; font-size: 16px; margin: 0 0 14px; color: var(--text); }
  .sub { color: var(--muted); font-size: 14px; margin: 0 0 18px; }
  label { display: block; font-size: 13px; color: var(--muted); margin: 12px 0 6px; }
  input[type=text], input[type=password] {
    width: 100%; padding: 11px 12px; border-radius: 10px; border: 1px solid var(--line);
    background: var(--ink-2); color: var(--text); font: inherit;
  }
  .filepick { display: flex; align-items: center; gap: 10px; height: 44px; padding: 0 12px 0 6px;
    border: 1px solid var(--line); border-radius: 10px; background: var(--ink-2); }
  .filepick .pickbtn { padding: 7px 14px; border-radius: 8px; border: 1px solid var(--line);
    background: var(--ink-3); color: var(--text); font: inherit; cursor: pointer; white-space: nowrap; }
  .filepick .pickbtn:hover { border-color: var(--teal-deep); }
  .filepick .filename { color: var(--muted); font-size: 13px; overflow: hidden;
    text-overflow: ellipsis; white-space: nowrap; }
  .row { display: flex; gap: 16px; flex-wrap: wrap; align-items: flex-end; }
  .seg { display: inline-flex; border: 1px solid var(--line); border-radius: 10px; overflow: hidden; }
  .seg label { margin: 0; padding: 10px 14px; cursor: pointer; color: var(--muted); background: var(--ink-2); }
  .seg input { display: none; }
  .seg input:checked + span { color: #06231f; }
  .seg label:has(input:checked) { background: linear-gradient(180deg, var(--teal) 0%, var(--teal-deep) 100%); }
  table { width: 100%; border-collapse: collapse; table-layout: fixed; }
  th, td { text-align: left; padding: 12px 10px; border-bottom: 1px solid var(--line); font-size: 14px; vertical-align: middle; overflow: hidden; }
  th { color: var(--muted); font-weight: 600; font-size: 12px; text-transform: uppercase; letter-spacing: .5px; }
  td .fname { font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  td .meta { color: var(--muted); font-size: 12px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  col.c-scope { width: 96px; } col.c-chunks { width: 74px; } col.c-size { width: 88px; }
  col.c-added { width: 148px; } col.c-act { width: 152px; }
  th.center, td.center { text-align: center; }
  /* Sortable column headers: clickable, with a directional indicator on the
     active column. Matches the muted teal-on-ink brand. */
  th.sortable { cursor: pointer; -webkit-user-select: none; user-select: none; }
  th.sortable:hover { color: var(--teal); }
  th.sortable .sort-ind { display: inline-block; width: 12px; margin-left: 4px; color: var(--teal); font-size: 10px; }
  th.sortable:not(.asc):not(.desc) .sort-ind { color: var(--muted); opacity: .55; }
  .chip { font-size: 12px; padding: 3px 9px; border-radius: 999px; border: 1px solid var(--line); }
  .chip.common { color: var(--teal); border-color: var(--teal-deep); }
  .chip.specific { color: #c0a3ff; border-color: #4a3b6b; }
  .empty { color: var(--muted); text-align: center; padding: 36px 0; }
  .toast { position: fixed; bottom: 22px; left: 50%; transform: translateX(-50%); background: var(--ink-3); border: 1px solid var(--line); padding: 12px 18px; border-radius: 10px; opacity: 0; transition: .25s; pointer-events: none; max-width: 90vw; }
  .toast.show { opacity: 1; }
  .toast.err { border-color: var(--danger); color: #ffd7de; }
  .hidden { display: none !important; }
  .muted { color: var(--muted); font-size: 13px; }
  .actions { display: flex; gap: 8px; justify-content: flex-end; }
  .actions button { padding: 6px 12px; font-size: 13px; }
  /* Pager: Prev/Next + "Showing X–Y of N". Sits under the table, on-brand. */
  .pager {
    display: flex; align-items: center; justify-content: flex-end; gap: 12px;
    margin-top: 14px; color: var(--muted); font-size: 13px;
  }
  .pager .pageinfo { font-variant-numeric: tabular-nums; }
  .pager button { padding: 6px 14px; font-size: 13px; }
  .pager button[disabled] { opacity: .4; cursor: not-allowed; }
  .pager button[disabled]:hover { border-color: var(--line); }
  .loginwrap { max-width: 420px; margin: 8vh auto; }
  /* Drag & drop + upload queue (session-only, client-side) */
  .card.dragover { border-color: var(--teal); box-shadow: 0 0 0 2px rgba(45,212,191,0.25) inset; }
  .filepick.dragover { border-color: var(--teal); background: rgba(45,212,191,0.08); }
  .queue { margin-top: 18px; border-top: 1px solid var(--line); padding-top: 14px; }
  .queue-head { display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-bottom: 12px; }
  .queue-head .qsummary { color: var(--muted); font-size: 13px; }
  .queue-head button { padding: 6px 12px; font-size: 13px; }
  .queue-list { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 8px; }
  .qitem { display: flex; align-items: center; gap: 12px; padding: 10px 12px; border: 1px solid var(--line); border-radius: 10px; background: var(--ink-2); }
  .qitem .qname { flex: 1; min-width: 0; font-size: 14px; font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .qitem .qsize { color: var(--muted); font-size: 12px; white-space: nowrap; }
  .qitem .qstatus { font-size: 13px; white-space: nowrap; display: inline-flex; align-items: center; gap: 6px; }
  .qstatus.queued { color: var(--muted); }
  .qstatus.running { color: var(--teal); }
  .qstatus.done { color: var(--ok); }
  .qstatus.error { color: var(--danger); max-width: 280px; overflow: hidden; text-overflow: ellipsis; }
  .qretry { padding: 5px 10px !important; font-size: 12px !important; }
  .spin { display: inline-block; width: 12px; height: 12px; border: 2px solid var(--line); border-top-color: var(--teal); border-radius: 50%; animation: spin .7s linear infinite; }
  @keyframes spin { to { transform: rotate(360deg); } }
</style>
</head>
<body>
<header>
  <span class="logo"><span class="i">I</span>nnovation <span class="w">W</span>ays</span>
  <span class="muted">· Document Manager</span>
  <span class="spacer"></span>
  <a class="btn btn-ghost" id="chatLink" href="/">&#128172; Chat</a>
  <span class="who" id="who"></span>
  <button class="btn-ghost hidden" id="logout">Sign out</button>
</header>

<main>
  <section id="loginView" class="hidden">
    <div class="card loginwrap">
      <h1>Sign in</h1>
      <p class="sub">Use your demo credentials to manage your documents.</p>
      <label for="u">Username</label>
      <input type="text" id="u" autocomplete="username" />
      <label for="p">Password</label>
      <input type="password" id="p" autocomplete="current-password" />
      <div style="margin-top:18px"><button class="btn-primary" id="loginBtn">Sign in</button></div>
    </div>
  </section>

  <section id="appView" class="hidden">
    <div class="card" id="uploadCard">
      <h2>Upload a document</h2>
      <p class="sub">PDF, text, markdown, or an image (invoices, scans &mdash; text is extracted via OCR). <strong>Drag files anywhere onto this card</strong> &mdash; or choose them &mdash; and they upload &amp; index automatically. They become queryable by the assistant in chat.</p>
      <div class="row">
        <div style="flex:1; min-width:240px">
          <label for="file">Files</label>
          <input type="file" id="file" class="hidden" multiple />
          <div class="filepick" id="dropZone">
            <button type="button" class="pickbtn" id="filePickBtn">Choose files</button>
            <span class="filename" id="fileName">Drag &amp; drop files here, or click to choose</span>
          </div>
        </div>
        <div>
          <label>Visibility</label>
          <div class="seg">
            <label><input type="radio" name="vis" value="specific" checked /><span>&#128274; Only me</span></label>
            <label><input type="radio" name="vis" value="common" /><span>&#127758; Everyone</span></label>
          </div>
        </div>
      </div>
      <div class="queue hidden" id="queue">
        <div class="queue-head">
          <span class="qsummary" id="queueSummary">Upload queue</span>
          <button class="btn-ghost hidden" id="clearQueue">Clear</button>
        </div>
        <ul class="queue-list" id="queueList"></ul>
      </div>
    </div>

    <div class="card">
      <h2>Your documents</h2>
      <p class="sub">Documents you own plus ones shared with everyone. Deleting a document also removes its embeddings &mdash; the assistant can no longer reference it.</p>
      <table>
        <colgroup>
          <col /><col class="c-scope" /><col class="c-chunks" />
          <col class="c-size" /><col class="c-added" /><col class="c-act" />
        </colgroup>
        <thead><tr>
          <th class="sortable" data-sort="filename" scope="col">Name<span class="sort-ind">▾</span></th>
          <th class="sortable" data-sort="visibility" scope="col">Scope<span class="sort-ind">▾</span></th>
          <th class="sortable center" data-sort="chunk_count" scope="col">Chunks<span class="sort-ind">▾</span></th>
          <th class="sortable" data-sort="size_bytes" scope="col">Size<span class="sort-ind">▾</span></th>
          <th class="sortable desc" data-sort="created_at" scope="col">Added<span class="sort-ind">▼</span></th>
          <th scope="col"></th>
        </tr></thead>
        <tbody id="rows"></tbody>
      </table>
      <div class="pager hidden" id="pager">
        <span class="pageinfo" id="pageInfo"></span>
        <button class="btn-ghost" id="pagePrev" type="button">&#8249; Prev</button>
        <button class="btn-ghost" id="pageNext" type="button">Next &#8250;</button>
      </div>
      <div id="empty" class="empty hidden">No documents yet &mdash; upload one above.</div>
    </div>
  </section>
</main>

<div class="toast" id="toast"></div>

<script>
(function () {
  // API base: strip "/documents/ui" from our own path → "/api" behind Caddy, or "" direct.
  var API = location.pathname.replace(/\/documents\/ui\/?$/, "");
  var tokenKey = "iw_token", whoKey = "iw_principal";
  var $ = function (id) { return document.getElementById(id); };

  function token() { return sessionStorage.getItem(tokenKey); }
  function setToken(t, who) {
    if (t) { sessionStorage.setItem(tokenKey, t); sessionStorage.setItem(whoKey, who || ""); }
    else { sessionStorage.removeItem(tokenKey); sessionStorage.removeItem(whoKey); }
  }
  function toast(msg, isErr) {
    var t = $("toast"); t.textContent = msg; t.className = "toast show" + (isErr ? " err" : "");
    setTimeout(function () { t.className = "toast"; }, 3200);
  }
  function authHeaders(extra) {
    var h = extra || {}; var tk = token(); if (tk) h["Authorization"] = "Bearer " + tk; return h;
  }
  function humanSize(n) {
    if (!n) return "&mdash;"; var u = ["B", "KB", "MB", "GB"]; var i = 0;
    while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; }
    return n.toFixed(i ? 1 : 0) + " " + u[i];
  }
  function fmtDate(s) {
    if (!s) return "";
    var d = new Date(s); if (isNaN(d.getTime())) return s;
    var p = function (n) { return (n < 10 ? "0" : "") + n; };
    return d.getFullYear() + "-" + p(d.getMonth() + 1) + "-" + p(d.getDate()) +
      " " + p(d.getHours()) + ":" + p(d.getMinutes());
  }
  function esc(s) { var d = document.createElement("div"); d.textContent = s == null ? "" : s; return d.innerHTML; }

  function showLogin() { $("loginView").classList.remove("hidden"); $("appView").classList.add("hidden"); $("logout").classList.add("hidden"); $("who").textContent = ""; }
  function showApp(who) {
    $("loginView").classList.add("hidden"); $("appView").classList.remove("hidden");
    $("logout").classList.remove("hidden"); $("who").textContent = who ? ("Signed in as " + who) : "";
    loadDocs();
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
      var d = await r.json(); setToken(d.token, d.principal_id); $("p").value = ""; showApp(d.principal_id);
    } catch (e) { toast("Login failed: " + e, true); }
  }

  // ---- Sorting + pagination (client-side) ----------------------------------
  // The /documents endpoint returns the full owned+common list in one shot
  // (heaviest demo user has 67 rows today), so sorting and 50-row pagination
  // happen entirely in the browser over `allDocs` — no server round-trip, no
  // API contract change.  Defaults preserve today's newest-first ordering.
  var PAGE_SIZE = 50;
  var allDocs = [], sortKey = "created_at", sortDir = "desc", page = 1;

  // Sensible default direction per column: text columns ascend, numeric /
  // chronological columns descend (largest = most recent first).
  var DEFAULT_DIR = {
    filename: "asc", visibility: "asc", chunk_count: "desc",
    size_bytes: "desc", created_at: "desc",
  };

  function sortedDocs() {
    // Copy first so we never mutate the in-memory list.
    var out = allDocs.slice();
    out.sort(function (a, b) {
      var dir = sortDir === "asc" ? 1 : -1;
      switch (sortKey) {
        case "filename":
        case "visibility":
          return dir * String(a[sortKey] || "").toLowerCase()
            .localeCompare(String(b[sortKey] || "").toLowerCase());
        case "chunk_count":
        case "size_bytes":
          return dir * ((Number(a[sortKey]) || 0) - (Number(b[sortKey]) || 0));
        case "created_at": {
          var ta = Date.parse(a[sortKey]), tb = Date.parse(b[sortKey]);
          if (!isNaN(ta) && !isNaN(tb)) return dir * (ta - tb);
          // Fall back to lexicographic for unparseable timestamps.
          return dir * String(a[sortKey] || "").localeCompare(String(b[sortKey] || ""));
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

  async function loadDocs() {
    try {
      var r = await fetch(API + "/documents", { headers: authHeaders() });
      if (r.status === 401) { setToken(null); showLogin(); return; }
      if (!r.ok) { toast("Could not load documents", true); return; }
      allDocs = await r.json();
      // Clamp the page so a deletion on the last page doesn't strand the
      // user on an empty page; sort/dir are preserved across reloads.
      var totalPages = Math.max(1, Math.ceil(allDocs.length / PAGE_SIZE));
      if (page > totalPages) page = totalPages;
      render();
    } catch (e) { toast("Load failed: " + e, true); }
  }

  function render() {
    var tb = $("rows"); tb.innerHTML = "";
    refreshHeaderIndicators();
    // Empty state: only when there's truly nothing in the underlying list.
    $("empty").classList.toggle("hidden", allDocs.length > 0);
    if (allDocs.length === 0) { renderPager(0); return; }

    var sorted = sortedDocs();
    var total = sorted.length;
    var totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE));
    if (page < 1) page = 1;
    if (page > totalPages) page = totalPages;
    var start = (page - 1) * PAGE_SIZE;
    var slice = sorted.slice(start, start + PAGE_SIZE);

    slice.forEach(function (d) {
      var tr = document.createElement("tr");
      var scope = d.visibility === "common"
        ? '<span class="chip common">Everyone</span>'
        : '<span class="chip specific">Only me</span>';
      var del = d.is_owner
        ? '<button class="btn-danger" data-del="' + esc(d.id) + '">Delete</button>'
        : '<span class="muted">shared</span>';
      tr.innerHTML =
        '<td><div class="fname" title="' + esc(d.filename) + '">' + esc(d.filename) + '</div>' +
        '<div class="meta">' + esc(d.mime) + (d.chunk_count === 0 ? ' &middot; &#9888; no text extracted' : '') + '</div></td>' +
        '<td>' + scope + '</td>' +
        '<td class="center">' + (d.chunk_count || 0) + '</td>' +
        '<td>' + humanSize(d.size_bytes) + '</td>' +
        '<td>' + esc(fmtDate(d.created_at)) + '</td>' +
        '<td><div class="actions">' +
          '<button class="btn-ghost" data-view="' + esc(d.id) + '" data-name="' + esc(d.filename) + '">View</button>' +
          '<button class="btn-ghost" data-download="' + esc(d.id) + '" data-name="' + esc(d.filename) + '">Download</button>' +
          del + '</div></td>';
      tb.appendChild(tr);
    });

    renderPager(total);
  }

  function renderPager(total) {
    var pager = $("pager"), info = $("pageInfo"),
      prev = $("pagePrev"), next = $("pageNext");
    // Hide the pager entirely when the list fits on one page.
    if (total <= PAGE_SIZE) { pager.classList.add("hidden"); return; }
    pager.classList.remove("hidden");
    var totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE));
    var from = (page - 1) * PAGE_SIZE + 1;
    var to = Math.min(page * PAGE_SIZE, total);
    info.textContent = "Showing " + from + "\u2013" + to + " of " + total;
    prev.disabled = page <= 1;
    next.disabled = page >= totalPages;
  }

  // ---- Upload queue (session-only: lives in memory, gone on reload) ----------
  // Each item: { id, file, name, size, vis, status, detail }.
  // status ∈ "queued" | "running" | "done" | "error". Files upload one at a
  // time (sequential) to stay gentle on the single-GPU indexer.
  var queue = [], qseq = 0, running = false;

  function currentVis() {
    var el = document.querySelector('input[name=vis]:checked');
    return el ? el.value : "specific";
  }

  function enqueue(files) {
    for (var i = 0; i < files.length; i++) {
      var f = files[i];
      queue.push({ id: ++qseq, file: f, name: f.name, size: f.size, vis: currentVis(), status: "queued", detail: "" });
    }
    renderQueue();
    pump();
  }

  function renderQueue() {
    var box = $("queue"), list = $("queueList");
    if (queue.length === 0) { box.classList.add("hidden"); return; }
    box.classList.remove("hidden");
    list.innerHTML = "";
    queue.forEach(function (q) {
      var st;
      if (q.status === "queued") st = '<span class="qstatus queued">Queued</span>';
      else if (q.status === "running") st = '<span class="qstatus running"><span class="spin"></span>Indexing&hellip;</span>';
      else if (q.status === "done") st = '<span class="qstatus done">&#10003; Done &middot; ' + esc(q.detail) + '</span>';
      else st = '<span class="qstatus error" title="' + esc(q.detail) + '">&#10007; ' + esc(q.detail || "Failed") + '</span>';
      var retry = q.status === "error" ? '<button class="btn-ghost qretry" data-retry="' + q.id + '">Retry</button>' : "";
      var li = document.createElement("li"); li.className = "qitem";
      li.innerHTML =
        '<span class="qname" title="' + esc(q.name) + '">' + esc(q.name) + '</span>' +
        '<span class="qsize">' + humanSize(q.size) + '</span>' + st + retry;
      list.appendChild(li);
    });
    var done = 0, err = 0;
    queue.forEach(function (q) { if (q.status === "done") done++; else if (q.status === "error") err++; });
    var pend = queue.length - done - err;
    var parts = [done + " done"];
    if (err) parts.push(err + " failed");
    if (pend) parts.push(pend + " pending");
    $("queueSummary").textContent = "Upload queue · " + parts.join(", ");
    $("clearQueue").classList.toggle("hidden", done + err === 0);
  }

  async function processItem(item) {
    item.status = "running"; renderQueue();
    var fd = new FormData(); fd.append("file", item.file); fd.append("visibility", item.vis);
    try {
      var r = await fetch(API + "/documents", { method: "POST", headers: authHeaders(), body: fd });
      if (r.status === 401) { setToken(null); item.status = "error"; item.detail = "Session expired — sign in again"; renderQueue(); showLogin(); return; }
      if (!r.ok) {
        var msg = "Upload failed";
        try { var e = await r.json(); if (e.error) msg = e.error; else if (typeof e.detail === "string") msg = e.detail; } catch (x) {}
        item.status = "error"; item.detail = msg; renderQueue(); return;
      }
      var d = await r.json();
      item.status = "done"; item.detail = (d.chunk_count || 0) + " chunks indexed";
      renderQueue(); loadDocs();
    } catch (e) {
      item.status = "error"; item.detail = String(e); renderQueue();
    }
  }

  async function pump() {
    if (running) return;
    var item = null;
    for (var i = 0; i < queue.length; i++) { if (queue[i].status === "queued") { item = queue[i]; break; } }
    if (!item) return;
    running = true;
    await processItem(item);
    running = false;
    pump();
  }

  function retry(id) {
    var item = null;
    for (var i = 0; i < queue.length; i++) { if (queue[i].id === id) { item = queue[i]; break; } }
    if (!item) return;
    item.status = "queued"; item.detail = ""; renderQueue(); pump();
  }

  function clearQueue() {
    // Drop finished rows (done + failed); keep anything still in-flight.
    queue = queue.filter(function (q) { return q.status === "running" || q.status === "queued"; });
    renderQueue();
  }

  async function del(id) {
    if (!confirm("Delete this document and its embeddings? This cannot be undone.")) return;
    try {
      var r = await fetch(API + "/documents/" + encodeURIComponent(id), { method: "DELETE", headers: authHeaders() });
      if (r.status === 401) { setToken(null); showLogin(); return; }
      if (r.status !== 204 && !r.ok) { toast("Delete failed", true); return; }
      toast("Deleted"); loadDocs();
    } catch (e) { toast("Delete failed: " + e, true); }
  }

  async function view(id, name) {
    try {
      var r = await fetch(API + "/documents/" + encodeURIComponent(id) + "/raw", { headers: authHeaders() });
      if (!r.ok) { toast("Cannot open document", true); return; }
      var blob = await r.blob(); var url = URL.createObjectURL(blob);
      window.open(url, "_blank");
      setTimeout(function () { URL.revokeObjectURL(url); }, 60000);
    } catch (e) { toast("Open failed: " + e, true); }
  }

  // Force a save-to-disk via the existing /raw endpoint (its FileResponse
  // already carries attachment disposition + filename).  Distinct from
  // `view`, which opens inline in a new tab.
  async function download(id, name) {
    try {
      var r = await fetch(API + "/documents/" + encodeURIComponent(id) + "/raw", { headers: authHeaders() });
      if (r.status === 401) { setToken(null); showLogin(); return; }
      if (!r.ok) { toast("Cannot download document", true); return; }
      var blob = await r.blob();
      var url = URL.createObjectURL(blob);
      var a = document.createElement("a");
      a.href = url; a.download = name || "document";
      // Some browsers ignore programmatic clicks on detached anchors — append
      // to the body, click, then remove.  Same-origin keeps the download
      // attribute authoritative; we revoke the URL after the click lands.
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      setTimeout(function () { URL.revokeObjectURL(url); }, 60000);
    } catch (e) { toast("Download failed: " + e, true); }
  }

  function setSort(key) {
    if (sortKey === key) { sortDir = sortDir === "asc" ? "desc" : "asc"; }
    else { sortKey = key; sortDir = DEFAULT_DIR[key] || "asc"; }
    page = 1;  // changing sort always returns the user to the top result
    render();
  }

  document.addEventListener("click", function (ev) {
    var t = ev.target;
    // Header clicks bubble up — walk up to the nearest <th.sortable>.
    var th = t.closest && t.closest("th.sortable");
    if (th && th.dataset && th.dataset.sort) { setSort(th.dataset.sort); return; }
    if (t.dataset && t.dataset.del) del(t.dataset.del);
    else if (t.dataset && t.dataset.view) view(t.dataset.view, t.dataset.name);
    else if (t.dataset && t.dataset.download) download(t.dataset.download, t.dataset.name);
    else if (t.dataset && t.dataset.retry) retry(parseInt(t.dataset.retry, 10));
  });
  $("pagePrev").addEventListener("click", function () { if (page > 1) { page--; render(); } });
  $("pageNext").addEventListener("click", function () {
    var totalPages = Math.max(1, Math.ceil(allDocs.length / PAGE_SIZE));
    if (page < totalPages) { page++; render(); }
  });
  $("loginBtn").addEventListener("click", login);
  $("p").addEventListener("keydown", function (e) { if (e.key === "Enter") login(); });
  $("clearQueue").addEventListener("click", clearQueue);
  // Custom (English, on-brand) file picker over the native input — multi-select.
  $("filePickBtn").addEventListener("click", function () { $("file").click(); });
  $("file").addEventListener("change", function () {
    if ($("file").files.length) enqueue($("file").files);
    $("file").value = "";  // reset so the same file can be re-picked later
  });
  $("logout").addEventListener("click", function () { setToken(null); showLogin(); });

  // Drag & drop onto the upload card. Dropping anywhere on the card enqueues
  // every dropped file; the rest of the page swallows stray drops so the
  // browser never navigates away to an accidentally-dropped file.
  var card = $("uploadCard");
  function stop(e) { e.preventDefault(); e.stopPropagation(); }
  ["dragenter", "dragover"].forEach(function (evt) {
    card.addEventListener(evt, function (e) { stop(e); card.classList.add("dragover"); });
  });
  card.addEventListener("dragleave", function (e) {
    stop(e);
    // Only drop the highlight when the pointer truly leaves the card.
    if (!card.contains(e.relatedTarget)) card.classList.remove("dragover");
  });
  card.addEventListener("drop", function (e) {
    stop(e); card.classList.remove("dragover");
    var files = e.dataTransfer && e.dataTransfer.files;
    if (files && files.length) enqueue(files);
  });
  ["dragover", "drop"].forEach(function (evt) {
    window.addEventListener(evt, function (e) { e.preventDefault(); });
  });

  // Single sign-on: reuse the chat login. The chat (same origin) exposes the
  // current user's token at /iw/session-token, authorized by the Chainlit cookie.
  async function ssoLogin() {
    try {
      var r = await fetch("/iw/session-token", { credentials: "include" });
      if (!r.ok) return false;
      var d = await r.json();
      if (d && d.token) { setToken(d.token, d.username || ""); return true; }
    } catch (e) { /* fall through to the login form */ }
    return false;
  }

  async function bootstrap() {
    if (token()) { showApp(sessionStorage.getItem(whoKey)); return; }
    if (await ssoLogin()) { showApp(sessionStorage.getItem(whoKey)); return; }
    showLogin();
  }
  bootstrap();
})();
</script>
</body>
</html>
"""


def documents_page_html() -> str:
    """Return the standalone document-manager HTML page."""
    return add_favicon(_PAGE)
