# ruff: noqa: E501
"""Operator Console **Upload / Document Manager** view (F-00096, S15-upload).

Implements the native, branded, server-rendered Upload view for the Operator
Console.  The view is composed via :func:`iw_rag.api.console_shell.render_console_page`
so it inherits the shared chrome (head, brand palette, header, nav, sign-in,
``#appView``, ``#toast``) without duplicating it.  Only the view-specific
body markup (``_UP_BODY``), CSS (``_UP_CSS``), and JavaScript (``_UP_JS``)
are defined here.

Features
--------
* **Drag-and-drop multi-upload** — dropzone accepts files dragged onto the
  card or selected via the hidden ``<input type="file" multiple>`` browse
  button.  A visibility ``<select>`` (Only me = ``"specific"`` default /
  Everyone = ``"common"``) is captured *per-file at enqueue time*.
* **Upload queue** (client-side only, cleared on page reload) — each queued
  file shows a coloured status pill that progresses
  ``Queued → Uploading → Done`` (or ``Failed``).  Files are processed
  **one at a time, sequentially**, to stay gentle on the single-GPU indexer.
  A failed item shows a ``Retry`` button.  After each successful upload the
  documents table is refreshed.
* **Documents table** — ``GET /documents`` on init and after every
  upload / delete.  Columns: **Name**, **Visibility**, **Size**, **Chunks**,
  **Added**, and an actions column (Download always; Delete only when
  ``is_owner``).
* **Sortable headers** — every data column is clickable to toggle asc/desc;
  an indicator arrow (▲/▼) marks the active column.  Sort is client-side.
* **Client-side pagination** — 25 rows per page, Prev / Next, page counter.
  Changing sort resets to page 1.
* **Download** (fetch → blob → ``<a download>``) and **Delete** (with
  ``confirm()`` first; owner-only).

The ``window.onConsoleReady`` hook is called by the shell after sign-in /
on a token-bearing reload.  At that point ``window.IWConsole`` is guaranteed
to exist.

No external CDN / hosted font (local-only, Invariant 8).
"""

from __future__ import annotations

from iw_rag.api.console_shell import render_console_page

# ---------------------------------------------------------------------------
# Body markup — fills #appView (post sign-in)
# ---------------------------------------------------------------------------
_UP_BODY = """
    <div class="card up-dropcard" id="upDropCard">
      <h2>Upload documents</h2>
      <p class="sub">PDF, text, markdown, or image (OCR). Drag files onto this card or choose them. Files are queued and uploaded sequentially so the GPU indexer is not overloaded.</p>
      <div class="up-controls">
        <div class="up-dropzone" id="upDropZone" role="button" tabindex="0" aria-label="Drop zone — drag files here or press Enter to browse">
          <input type="file" id="upFileInput" class="hidden" multiple aria-hidden="true" />
          <span class="up-dz-icon" aria-hidden="true">&#8682;</span>
          <span class="up-dz-label">Drag &amp; drop files here, or <button type="button" class="btn-ghost up-browse-btn" id="upBrowseBtn">browse</button></span>
        </div>
        <div class="up-vis-wrap">
          <label for="upVis">Visibility</label>
          <select id="upVis">
            <option value="specific" selected>Only me</option>
            <option value="common">Everyone</option>
          </select>
        </div>
      </div>
      <div class="up-queue hidden" id="upQueue">
        <div class="up-queue-head">
          <span class="up-queue-summary muted" id="upQueueSummary">Upload queue</span>
          <button type="button" class="btn-ghost btn-sm hidden" id="upClearQueue">Clear finished</button>
        </div>
        <ul class="up-queue-list" id="upQueueList"></ul>
      </div>
    </div>

    <div class="card">
      <h2>Documents</h2>
      <p class="sub">Your documents plus ones shared with everyone. Deleting a document also removes its embeddings.</p>
      <div class="up-table-wrap">
        <table class="up-table">
          <colgroup>
            <col class="up-col-name" />
            <col class="up-col-vis" />
            <col class="up-col-size" />
            <col class="up-col-chunks" />
            <col class="up-col-added" />
            <col class="up-col-act" />
          </colgroup>
          <thead><tr>
            <th class="up-th sortable" data-sort="name" scope="col">Name<span class="sort-ind">&#9662;</span></th>
            <th class="up-th sortable" data-sort="visibility" scope="col">Visibility<span class="sort-ind">&#9662;</span></th>
            <th class="up-th sortable" data-sort="size_bytes" scope="col">Size<span class="sort-ind">&#9662;</span></th>
            <th class="up-th sortable" data-sort="chunk_count" scope="col">Chunks<span class="sort-ind">&#9662;</span></th>
            <th class="up-th sortable desc" data-sort="created_at" scope="col">Added<span class="sort-ind">&#9660;</span></th>
            <th class="up-th" scope="col"><span class="sr-only">Actions</span></th>
          </tr></thead>
          <tbody id="upRows"></tbody>
        </table>
      </div>
      <div class="up-pager hidden" id="upPager">
        <span class="up-page-info muted" id="upPageInfo"></span>
        <button type="button" class="btn-ghost btn-sm" id="upPrev">&#8249; Prev</button>
        <button type="button" class="btn-ghost btn-sm" id="upNext">Next &#8250;</button>
      </div>
      <div class="up-empty hidden" id="upEmpty">No documents yet &mdash; upload one above.</div>
    </div>
"""

# ---------------------------------------------------------------------------
# View-specific CSS — appended inside the shared <style>
# ---------------------------------------------------------------------------
_UP_CSS = """
  /* --- Upload dropzone --------------------------------------------------- */
  .up-dropcard { transition: border-color .15s, box-shadow .15s; }
  .up-dropcard.dragover { border-color: var(--teal); box-shadow: 0 0 0 2px rgba(45,212,191,0.2) inset; }
  .up-controls { display: flex; gap: 18px; align-items: flex-end; flex-wrap: wrap; margin-top: 4px; }
  .up-dropzone {
    flex: 1; min-width: 220px; display: flex; align-items: center; gap: 12px;
    padding: 18px 20px; border: 2px dashed var(--line); border-radius: 12px;
    background: var(--ink-2); cursor: pointer; transition: border-color .15s, background .15s;
  }
  .up-dropzone:hover, .up-dropzone.dz-active { border-color: var(--teal); background: rgba(45,212,191,0.06); }
  .up-dz-icon { font-size: 22px; color: var(--muted); flex-shrink: 0; }
  .up-dz-label { font-size: 14px; color: var(--muted); }
  .up-browse-btn { padding: 0; font-size: 14px; color: var(--teal); text-decoration: underline; border: none; background: transparent; cursor: pointer; }
  .up-browse-btn:hover { color: var(--teal-deep); border: none; }
  .up-vis-wrap { min-width: 140px; }
  .up-vis-wrap label { font-size: 13px; color: var(--muted); margin: 0 0 6px; display: block; }
  .up-vis-wrap select { padding: 10px 12px; border-radius: 10px; border: 1px solid var(--line); background: var(--ink-2); color: var(--text); font: inherit; width: 100%; }
  /* --- Upload queue ------------------------------------------------------ */
  .up-queue { margin-top: 18px; border-top: 1px solid var(--line); padding-top: 14px; }
  .up-queue-head { display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-bottom: 10px; }
  .up-queue-summary { font-size: 13px; color: var(--muted); }
  .up-queue-list { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 8px; }
  .up-qitem { display: flex; align-items: center; gap: 10px; padding: 10px 12px; border: 1px solid var(--line); border-radius: 10px; background: var(--ink-2); }
  .up-qname { flex: 1; min-width: 0; font-size: 14px; font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .up-qsize { color: var(--muted); font-size: 12px; white-space: nowrap; }
  .up-qpill { font-size: 12px; padding: 3px 10px; border-radius: 999px; border: 1px solid var(--line); white-space: nowrap; }
  .up-qpill.queued { color: var(--muted); border-color: var(--line); }
  .up-qpill.uploading { color: var(--teal); border-color: var(--teal-deep); background: rgba(45,212,191,0.08); }
  .up-qpill.done { color: var(--ok); border-color: var(--teal-deep); }
  .up-qpill.failed { color: #ffd7de; border-color: var(--danger); background: rgba(244,63,94,0.08); max-width: 260px; overflow: hidden; text-overflow: ellipsis; }
  .up-spin { display: inline-block; width: 11px; height: 11px; border: 2px solid rgba(45,212,191,0.3); border-top-color: var(--teal); border-radius: 50%; animation: up-spin .7s linear infinite; vertical-align: middle; margin-right: 4px; }
  @keyframes up-spin { to { transform: rotate(360deg); } }
  /* --- Documents table --------------------------------------------------- */
  .up-table-wrap { overflow-x: auto; }
  .up-table { width: 100%; border-collapse: collapse; table-layout: fixed; }
  .up-col-name  { width: auto; }
  .up-col-vis   { width: 110px; }
  .up-col-size  { width: 80px; }
  .up-col-chunks { width: 72px; }
  .up-col-added { width: 140px; }
  .up-col-act   { width: 148px; }
  .up-th { text-align: left; padding: 10px 8px; border-bottom: 1px solid var(--line); color: var(--muted); font-weight: 600; font-size: 12px; text-transform: uppercase; letter-spacing: .5px; }
  .up-th.sortable { cursor: pointer; -webkit-user-select: none; user-select: none; }
  .up-th.sortable:hover { color: var(--teal); }
  .up-th .sort-ind { display: inline-block; width: 12px; margin-left: 4px; color: var(--teal); font-size: 10px; }
  .up-th.sortable:not(.asc):not(.desc) .sort-ind { color: var(--muted); opacity: .55; }
  .up-td { text-align: left; padding: 11px 8px; border-bottom: 1px solid var(--line); font-size: 14px; vertical-align: middle; overflow: hidden; }
  .up-fname { font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; display: block; }
  .up-meta { color: var(--muted); font-size: 12px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; display: block; }
  /* Visibility pill — white-space:nowrap so "Only me" never wraps */
  .up-vis-pill { font-size: 12px; padding: 3px 10px; border-radius: 999px; border: 1px solid var(--line); white-space: nowrap; min-width: 72px; display: inline-block; }
  .up-vis-pill.common   { color: var(--teal); border-color: var(--teal-deep); }
  .up-vis-pill.specific { color: #c0a3ff; border-color: #4a3b6b; }
  .up-act { display: flex; gap: 6px; justify-content: flex-end; }
  .up-act button { padding: 5px 10px; font-size: 12px; border-radius: 8px; }
  .up-empty { color: var(--muted); text-align: center; padding: 36px 0; font-size: 14px; }
  /* --- Pagination bar ---------------------------------------------------- */
  .up-pager { display: flex; align-items: center; justify-content: flex-end; gap: 10px; margin-top: 14px; font-size: 13px; }
  .up-page-info { color: var(--muted); font-variant-numeric: tabular-nums; }
  .up-pager button:disabled { opacity: .4; cursor: not-allowed; }
  .up-pager button:disabled:hover { border-color: var(--line); }
  /* Screen-reader only */
  .sr-only { position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px; overflow: hidden; clip: rect(0,0,0,0); white-space: nowrap; border: 0; }
"""

# ---------------------------------------------------------------------------
# View JavaScript — a single <script> block that defines window.onConsoleReady
# ---------------------------------------------------------------------------
_UP_JS = r"""<script>
(function () {
  "use strict";

  // IWConsole is guaranteed to exist when onConsoleReady is called.
  // Shadow local references so the inner functions close over them safely.
  var _API, _ah, _esc, _toast, _$;

  // ---- Upload queue (client-side only; gone on page reload) ---------------
  // Each item: { id, file, name, size, vis, status, detail }
  // status in {"queued","uploading","done","failed"}
  var queue = [], qseq = 0, qrunning = false;

  // ---- Documents list + sort + pagination ---------------------------------
  var PAGE_SIZE = 25;
  var allDocs = [], sortKey = "created_at", sortDir = "desc", page = 1;

  // Default sort direction per column
  var DEFAULT_DIR = {
    name: "asc",
    visibility: "asc",
    size_bytes: "desc",
    chunk_count: "desc",
    created_at: "desc"
  };

  // ---- Helpers ------------------------------------------------------------

  /**
   * Format a byte count as a human-readable string (B / KB / MB / GB).
   *
   * @param {number|null|undefined} n - Raw byte count.
   * @returns {string} Formatted size string, e.g. "1.4 MB".
   */
  function humanSize(n) {
    if (n == null || isNaN(n)) { return "—"; }
    var units = ["B", "KB", "MB", "GB"];
    var i = 0;
    while (n >= 1024 && i < units.length - 1) { n /= 1024; i++; }
    return (i ? n.toFixed(1) : n.toFixed(0)) + " " + units[i];
  }

  /**
   * Format an ISO timestamp as "YYYY-MM-DD HH:MM" in local time.
   *
   * @param {string|null|undefined} s - ISO 8601 datetime string.
   * @returns {string} Formatted string or empty string on failure.
   */
  function fmtDate(s) {
    if (!s) { return ""; }
    var d = new Date(s);
    if (isNaN(d.getTime())) { return s; }
    var p = function (n) { return (n < 10 ? "0" : "") + n; };
    return d.getFullYear() + "-" + p(d.getMonth() + 1) + "-" + p(d.getDate()) +
      " " + p(d.getHours()) + ":" + p(d.getMinutes());
  }

  /**
   * Read the currently selected visibility from the upload select.
   *
   * @returns {string} "specific" or "common".
   */
  function currentVis() {
    var el = _$("upVis");
    return el ? el.value : "specific";
  }

  // ---- Document list loading & rendering ----------------------------------

  /**
   * Fetch the full document list from GET /documents and re-render the table.
   * On 401, clears the session (the shell will show the login view on next
   * interaction, but we cannot call showLogin directly; we toast instead).
   *
   * @returns {Promise<void>}
   */
  async function loadDocs() {
    try {
      var r = await fetch(_API + "/documents", { headers: _ah() });
      if (r.status === 401) { _toast("Session expired — please sign in again.", true); return; }
      if (!r.ok) { _toast("Could not load documents (" + r.status + ")", true); return; }
      allDocs = await r.json();
      // Clamp page so a deletion on the last page does not strand the user.
      var totalPages = Math.max(1, Math.ceil(allDocs.length / PAGE_SIZE));
      if (page > totalPages) { page = totalPages; }
      render();
    } catch (e) { _toast("Load failed: " + e, true); }
  }

  /**
   * Return a sorted copy of allDocs using the current sortKey / sortDir.
   *
   * @returns {Array<Object>} Sorted document array.
   */
  function sortedDocs() {
    var out = allDocs.slice();
    out.sort(function (a, b) {
      var dir = sortDir === "asc" ? 1 : -1;
      switch (sortKey) {
        case "name": {
          var na = String((a.title || a.filename) || "").toLowerCase();
          var nb = String((b.title || b.filename) || "").toLowerCase();
          return dir * na.localeCompare(nb);
        }
        case "visibility":
          return dir * String(a.visibility || "").localeCompare(String(b.visibility || ""));
        case "size_bytes":
          return dir * ((Number(a.size_bytes) || 0) - (Number(b.size_bytes) || 0));
        case "chunk_count":
          return dir * ((Number(a.chunk_count) || 0) - (Number(b.chunk_count) || 0));
        case "created_at": {
          var ta = Date.parse(a.created_at), tb = Date.parse(b.created_at);
          if (!isNaN(ta) && !isNaN(tb)) { return dir * (ta - tb); }
          return dir * String(a.created_at || "").localeCompare(String(b.created_at || ""));
        }
        default: return 0;
      }
    });
    return out;
  }

  /** Re-render the sortable header indicators (▲/▼/▾). */
  function refreshHeaderIndicators() {
    var ths = document.querySelectorAll("th.sortable");
    for (var i = 0; i < ths.length; i++) {
      var th = ths[i];
      th.classList.remove("asc", "desc");
      var ind = th.querySelector(".sort-ind");
      if (th.dataset.sort === sortKey) {
        th.classList.add(sortDir);
        if (ind) { ind.textContent = sortDir === "asc" ? "▲" : "▼"; }
      } else {
        if (ind) { ind.textContent = "▾"; }
      }
    }
  }

  /** Full table + pagination re-render from the current allDocs / page state. */
  function render() {
    var tb = _$("upRows");
    if (!tb) { return; }
    tb.innerHTML = "";
    refreshHeaderIndicators();

    // Empty state
    var emptyEl = _$("upEmpty");
    if (emptyEl) { emptyEl.classList.toggle("hidden", allDocs.length > 0); }
    if (allDocs.length === 0) { renderPager(0); return; }

    var sorted = sortedDocs();
    var total = sorted.length;
    var totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE));
    if (page < 1) { page = 1; }
    if (page > totalPages) { page = totalPages; }
    var start = (page - 1) * PAGE_SIZE;
    var slice = sorted.slice(start, start + PAGE_SIZE);

    slice.forEach(function (d) {
      var tr = document.createElement("tr");
      var displayName = d.title || d.filename || "(untitled)";
      var visCls = d.visibility === "common" ? "common" : "specific";
      var visLabel = d.visibility === "common" ? "Everyone" : "Only me";
      var visHtml = '<span class="up-vis-pill ' + visCls + '">' + _esc(visLabel) + '</span>';
      var delHtml = d.is_owner
        ? '<button type="button" class="btn-danger" data-del="' + _esc(d.id) + '" data-delname="' + _esc(displayName) + '">Delete</button>'
        : "";
      tr.innerHTML =
        '<td class="up-td"><span class="up-fname" title="' + _esc(displayName) + '">' + _esc(displayName) + '</span>' +
          '<span class="up-meta">' + _esc(d.mime || "") + (d.chunk_count === 0 ? ' &middot; &#9888; no text extracted' : '') + '</span></td>' +
        '<td class="up-td">' + visHtml + '</td>' +
        '<td class="up-td">' + _esc(humanSize(d.size_bytes)) + '</td>' +
        '<td class="up-td">' + _esc(String(d.chunk_count || 0)) + '</td>' +
        '<td class="up-td">' + _esc(fmtDate(d.created_at)) + '</td>' +
        '<td class="up-td"><div class="up-act">' +
          '<button type="button" class="btn-ghost" data-dl="' + _esc(d.id) + '" data-dlname="' + _esc(d.filename || displayName) + '">Download</button>' +
          delHtml +
        '</div></td>';
      tb.appendChild(tr);
    });

    renderPager(total);
  }

  /**
   * Render the pagination bar for a given total count.
   *
   * @param {number} total - Total number of rows in the sorted list.
   */
  function renderPager(total) {
    var pager = _$("upPager"), info = _$("upPageInfo");
    var prev = _$("upPrev"), next = _$("upNext");
    if (!pager) { return; }
    if (total <= PAGE_SIZE) { pager.classList.add("hidden"); return; }
    pager.classList.remove("hidden");
    var totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE));
    var from = (page - 1) * PAGE_SIZE + 1;
    var to = Math.min(page * PAGE_SIZE, total);
    if (info) { info.textContent = "Page " + page + " of " + totalPages + " · " + from + "–" + to + " of " + total; }
    if (prev) { prev.disabled = page <= 1; }
    if (next) { next.disabled = page >= totalPages; }
  }

  // ---- Sort ---------------------------------------------------------------

  /**
   * Switch to the given sort column; toggle direction if already active.
   * Resets pagination to page 1.
   *
   * @param {string} key - Column sort key (name, visibility, size_bytes, chunk_count, created_at).
   */
  function setSort(key) {
    if (sortKey === key) {
      sortDir = sortDir === "asc" ? "desc" : "asc";
    } else {
      sortKey = key;
      sortDir = DEFAULT_DIR[key] || "asc";
    }
    page = 1;
    render();
  }

  // ---- Upload queue -------------------------------------------------------

  /**
   * Add files to the client-side queue (capturing visibility at enqueue time)
   * and trigger the sequential upload pump.
   *
   * @param {FileList} files - The files to enqueue.
   */
  function enqueue(files) {
    for (var i = 0; i < files.length; i++) {
      var f = files[i];
      queue.push({
        id: ++qseq,
        file: f,
        name: f.name,
        size: f.size,
        vis: currentVis(),
        status: "queued",
        detail: ""
      });
    }
    renderQueue();
    pump();
  }

  /** Re-render the upload queue list. */
  function renderQueue() {
    var box = _$("upQueue"), list = _$("upQueueList"), summary = _$("upQueueSummary");
    var clear = _$("upClearQueue");
    if (!box || !list) { return; }
    if (queue.length === 0) { box.classList.add("hidden"); return; }
    box.classList.remove("hidden");
    list.innerHTML = "";
    queue.forEach(function (q) {
      var pillHtml;
      if (q.status === "queued") {
        pillHtml = '<span class="up-qpill queued">Queued</span>';
      } else if (q.status === "uploading") {
        pillHtml = '<span class="up-qpill uploading"><span class="up-spin" aria-hidden="true"></span>Uploading…</span>';
      } else if (q.status === "done") {
        pillHtml = '<span class="up-qpill done">&#10003; Done' + (q.detail ? " · " + _esc(q.detail) : "") + '</span>';
      } else {
        // failed
        pillHtml = '<span class="up-qpill failed" title="' + _esc(q.detail) + '">&#10007; ' + _esc(q.detail || "Failed") + '</span>';
      }
      var retryHtml = q.status === "failed"
        ? '<button type="button" class="btn-ghost btn-sm" data-retry="' + q.id + '">Retry</button>'
        : "";
      var li = document.createElement("li");
      li.className = "up-qitem";
      li.innerHTML =
        '<span class="up-qname" title="' + _esc(q.name) + '">' + _esc(q.name) + '</span>' +
        '<span class="up-qsize">' + _esc(humanSize(q.size)) + '</span>' +
        pillHtml + retryHtml;
      list.appendChild(li);
    });

    // Summary line
    var done = 0, failed = 0;
    queue.forEach(function (q) {
      if (q.status === "done") { done++; }
      else if (q.status === "failed") { failed++; }
    });
    var pend = queue.length - done - failed;
    var parts = [done + " done"];
    if (failed) { parts.push(failed + " failed"); }
    if (pend) { parts.push(pend + " pending"); }
    if (summary) { summary.textContent = "Upload queue · " + parts.join(", "); }
    if (clear) { clear.classList.toggle("hidden", done + failed === 0); }
  }

  /**
   * Upload a single queue item via POST /documents (multipart).
   *
   * @param {Object} item - Queue item with .file, .vis, .status, .detail.
   * @returns {Promise<void>}
   */
  async function processItem(item) {
    item.status = "uploading";
    renderQueue();
    var fd = new FormData();
    fd.append("file", item.file);
    fd.append("visibility", item.vis);
    try {
      // authHeaders() called with NO extra arg so the browser sets the
      // multipart boundary (do not set Content-Type manually).
      var r = await fetch(_API + "/documents", {
        method: "POST",
        headers: _ah(),
        body: fd
      });
      if (r.status === 401) {
        item.status = "failed";
        item.detail = "Session expired — sign in again";
        renderQueue();
        return;
      }
      if (!r.ok) {
        var msg = "Upload failed (" + r.status + ")";
        try {
          var ej = await r.json();
          if (ej && ej.error) { msg = ej.error; }
          else if (ej && typeof ej.detail === "string") { msg = ej.detail; }
        } catch (x) { /* ignore parse errors */ }
        item.status = "failed";
        item.detail = msg;
        renderQueue();
        return;
      }
      var d = await r.json();
      item.status = "done";
      item.detail = (d.chunk_count || 0) + " chunks";
      renderQueue();
      loadDocs();
    } catch (e) {
      item.status = "failed";
      item.detail = String(e);
      renderQueue();
    }
  }

  /** Drive the queue one item at a time sequentially. */
  async function pump() {
    if (qrunning) { return; }
    var item = null;
    for (var i = 0; i < queue.length; i++) {
      if (queue[i].status === "queued") { item = queue[i]; break; }
    }
    if (!item) { return; }
    qrunning = true;
    await processItem(item);
    qrunning = false;
    pump();
  }

  /**
   * Reset a failed item to "queued" and restart the pump.
   *
   * @param {number} id - The queue item id to retry.
   */
  function retryItem(id) {
    for (var i = 0; i < queue.length; i++) {
      if (queue[i].id === id) {
        queue[i].status = "queued";
        queue[i].detail = "";
        renderQueue();
        pump();
        return;
      }
    }
  }

  /** Remove all finished (done or failed) rows from the queue. */
  function clearQueue() {
    queue = queue.filter(function (q) { return q.status === "queued" || q.status === "uploading"; });
    renderQueue();
  }

  // ---- Download / Delete --------------------------------------------------

  /**
   * Download a document by fetching its raw bytes and triggering a save.
   *
   * @param {string} id - Document id.
   * @param {string} name - Filename hint for the save dialog.
   * @returns {Promise<void>}
   */
  async function downloadDoc(id, name) {
    try {
      var r = await fetch(_API + "/documents/" + encodeURIComponent(id) + "/raw", { headers: _ah() });
      if (r.status === 401) { _toast("Session expired — sign in again.", true); return; }
      if (!r.ok) { _toast("Download failed (" + r.status + ")", true); return; }
      var blob = await r.blob();
      var url = URL.createObjectURL(blob);
      var a = document.createElement("a");
      a.href = url;
      a.download = name || "document";
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      setTimeout(function () { URL.revokeObjectURL(url); }, 60000);
    } catch (e) { _toast("Download failed: " + e, true); }
  }

  /**
   * Delete a document after confirmation, then refresh the table.
   *
   * @param {string} id - Document id to delete.
   * @param {string} name - Display name for the confirm dialog.
   * @returns {Promise<void>}
   */
  async function deleteDoc(id, name) {
    if (!confirm("Delete “" + name + "” and its embeddings? This cannot be undone.")) { return; }
    try {
      var r = await fetch(_API + "/documents/" + encodeURIComponent(id), {
        method: "DELETE",
        headers: _ah()
      });
      if (r.status === 401) { _toast("Session expired — sign in again.", true); return; }
      if (r.status !== 204 && !r.ok) { _toast("Delete failed (" + r.status + ")", true); return; }
      _toast("Deleted.");
      loadDocs();
    } catch (e) { _toast("Delete failed: " + e, true); }
  }

  // ---- Wire-up (called by the shell after sign-in) ------------------------

  /**
   * Entry point called by the shell after the principal is authenticated.
   * Caches IWConsole references, wires all event handlers, and loads the
   * initial document list.
   */
  window.onConsoleReady = function () {
    _API   = window.IWConsole.API;
    _ah    = window.IWConsole.authHeaders;
    _esc   = window.IWConsole.esc;
    _toast = window.IWConsole.toast;
    _$     = window.IWConsole.$;

    // File input → enqueue
    var fi = _$("upFileInput");
    if (fi) {
      fi.addEventListener("change", function () {
        if (fi.files && fi.files.length) { enqueue(fi.files); }
        fi.value = "";  // reset so re-picking the same file triggers change again
      });
    }

    // Browse button → open file dialog
    var browseBtn = _$("upBrowseBtn");
    if (browseBtn) {
      browseBtn.addEventListener("click", function (e) {
        e.stopPropagation();  // do not trigger the dropzone click handler
        if (fi) { fi.click(); }
      });
    }

    // Dropzone keyboard: Enter/Space opens file dialog
    var dz = _$("upDropZone");
    if (dz) {
      dz.addEventListener("keydown", function (e) {
        if ((e.key === "Enter" || e.key === " ") && fi) { fi.click(); }
      });
      dz.addEventListener("click", function (e) {
        // Ignore clicks that originated on the browse button (handled above).
        if (e.target && e.target.id === "upBrowseBtn") { return; }
        if (fi) { fi.click(); }
      });
    }

    // Drag & drop on the upload card
    var card = _$("upDropCard");
    function stop(e) { e.preventDefault(); e.stopPropagation(); }
    if (card) {
      ["dragenter", "dragover"].forEach(function (evt) {
        card.addEventListener(evt, function (e) {
          stop(e);
          card.classList.add("dragover");
          if (dz) { dz.classList.add("dz-active"); }
        });
      });
      card.addEventListener("dragleave", function (e) {
        stop(e);
        // Only drop the highlight when the pointer truly leaves the card.
        if (!card.contains(e.relatedTarget)) {
          card.classList.remove("dragover");
          if (dz) { dz.classList.remove("dz-active"); }
        }
      });
      card.addEventListener("drop", function (e) {
        stop(e);
        card.classList.remove("dragover");
        if (dz) { dz.classList.remove("dz-active"); }
        var files = e.dataTransfer && e.dataTransfer.files;
        if (files && files.length) { enqueue(files); }
      });
    }
    // Swallow stray drops on the rest of the page so the browser never
    // navigates away from an accidentally-dropped file.
    ["dragover", "drop"].forEach(function (evt) {
      window.addEventListener(evt, function (e) { e.preventDefault(); });
    });

    // Clear finished queue rows
    var clearBtn = _$("upClearQueue");
    if (clearBtn) { clearBtn.addEventListener("click", clearQueue); }

    // Pagination buttons
    var prevBtn = _$("upPrev"), nextBtn = _$("upNext");
    if (prevBtn) {
      prevBtn.addEventListener("click", function () {
        if (page > 1) { page--; render(); }
      });
    }
    if (nextBtn) {
      nextBtn.addEventListener("click", function () {
        var totalPages = Math.max(1, Math.ceil(allDocs.length / PAGE_SIZE));
        if (page < totalPages) { page++; render(); }
      });
    }

    // Delegated click handler for table actions + queue retry
    document.addEventListener("click", function (ev) {
      var t = ev.target;
      // Sortable header click — walk up to the nearest th.sortable
      var th = t.closest && t.closest("th.sortable");
      if (th && th.dataset && th.dataset.sort) { setSort(th.dataset.sort); return; }
      // Download button
      if (t.dataset && t.dataset.dl) { downloadDoc(t.dataset.dl, t.dataset.dlname || ""); return; }
      // Delete button
      if (t.dataset && t.dataset.del) { deleteDoc(t.dataset.del, t.dataset.delname || t.dataset.del); return; }
      // Retry button
      if (t.dataset && t.dataset.retry) { retryItem(parseInt(t.dataset.retry, 10)); return; }
    });

    // Load the initial document list
    loadDocs();
  };
})();
</script>"""


def console_upload_page_html() -> str:
    """Return the complete Operator Console Upload / Document Manager HTML page.

    Assembles the shared shell chrome from
    :func:`iw_rag.api.console_shell.render_console_page` with the
    Upload-view-specific body markup, CSS, and JavaScript defined in this
    module.

    The returned document is a self-contained, dependency-free HTML page that:

    * Inherits the brand palette (teal ``#2DD4BF``/``#0D9488`` on ink
      ``#181D29``), header, six-item nav (``#navUpload`` active), login view,
      ``#appView``, and ``#toast`` from the shared shell.
    * Provides a drag-and-drop multi-upload card (dropzone + hidden file input
      + visibility select) with a client-side-only upload queue (Queued →
      Uploading → Done/Failed, sequential, with per-row Retry).
    * Provides a sortable (all five data columns) + paginated (25 rows/page)
      documents table backed by ``GET /documents``.
    * Implements Download (fetch → blob → ``<a download>``) and Delete (with
      ``confirm()``; owner-only) for each row.

    Returns:
        Raw, self-contained HTML document string (no static-mount or MIME
        wiring required).
    """
    return render_console_page(
        title="Upload",
        extra_css=_UP_CSS,
        body=_UP_BODY,
        view_js=_UP_JS,
    )


__all__ = ["console_upload_page_html"]
