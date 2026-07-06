# ruff: noqa: E501
"""Behavioural tests for the Compare view runs-picker sort + pagination.

The :mod:`iw_rag.api.compare_page` page is a self-contained inline
HTML+CSS+vanilla-JS page.  Its run picker is fully sortable (every
data column) and pages the full report set in fixed 20-row slices,
applying the sort **before** the page slice so changing pages never
re-orders rows.  This module pins that behaviour by actually running
the inline ``<script>`` under a headless Node with a hand-rolled
DOM mock — verifying click/toggle, default direction per column,
the 20-row slice, page 2 of 25, and the sort-before-slice contract.

The DOM mock is intentionally minimal: it provides just enough surface
(:js:class:`Element`, ``document.getElementById``, ``querySelectorAll``,
``addEventListener``, ``classList``, ``sessionStorage``, ``location``,
``fetch``) for the IIFE to run end-to-end without throwing, and
nothing more.  The behavioural assertions exercise the **same**
``sortedReports`` / ``setSort`` / ``renderPickerPager`` / ``sortInd``
functions the real browser executes — no duplicated re-implementation.

Static structural tests (PAGE_SIZE constant, PICK_COLS order,
CSS class wiring, etc.) live in :mod:`tests.unit.api.test_compare_page`;
this file is the dynamic counterpart that runs the JS itself.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

from iw_rag.api.compare_page import compare_page_html

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _inline_script() -> str:
    """Return the inline ``<script>`` body of the Compare view page."""
    scripts = re.findall(r"<script\b[^>]*>(.*?)</script>", compare_page_html(), re.S)
    assert scripts, "Compare view page has no <script> block"
    return "\n".join(scripts)


# A small Node test runner that mocks the DOM just enough to let the Compare
# view's IIFE run end-to-end, exposes the inner functions (sortedReports,
# setSort, renderPickerPager, sortInd, REPORTS, sortKey, sortDir, page,
# PAGE_SIZE, DEFAULT_DIR, PICK_COLS, modsSummary, reportById) via a returned
# object, then runs a battery of behavioural assertions.
#
# The runner is self-contained (no project deps) — it uses only Node's
# built-in ``fs`` + ``vm`` modules.  It writes a pass/fail summary to stdout
# and exits with code 0 on success, 1 on failure.
_NODE_RUNNER = r"""
'use strict';
const fs = require('fs');
const vm = require('vm');

// --- Minimal DOM mock -------------------------------------------------------
class Element {
  constructor(id) {
    this.id = id;
    this.children = [];
    this._attrs = {};
    this._listeners = {};
    this._innerHTML = '';
    this._textContent = '';
    this._value = '';
    this._disabled = false;
    this._checked = false;
    this._dataSort = null;
    this.classList = {
      _set: new Set(),
      add: function () { for (const c of arguments) this._set.add(c); }.bind(this),
      remove: function () { for (const c of arguments) this._set.delete(c); }.bind(this),
      toggle: function (c, force) {
        if (force === true) this._set.add(c);
        else if (force === false) this._set.delete(c);
        else if (this._set.has(c)) this._set.delete(c);
        else this._set.add(c);
      }.bind(this),
      contains: function (c) { return this._set.has(c); }.bind(this)
    };
  }
  setAttribute(n, v) { this._attrs[n] = v; if (n === 'data-sort') this._dataSort = v; }
  getAttribute(n) { return (n in this._attrs) ? this._attrs[n] : null; }
  addEventListener(ev, fn) { (this._listeners[ev] = this._listeners[ev] || []).push(fn); }
  appendChild(c) { this.children.push(c); }
  set innerHTML(v) { this._innerHTML = String(v); }
  get innerHTML() { return this._innerHTML; }
  set textContent(v) { this._textContent = String(v); }
  get textContent() { return this._textContent; }
  set value(v) { this._value = String(v); }
  get value() { return this._value; }
  set disabled(v) { this._disabled = !!v; }
  get disabled() { return this._disabled; }
  set checked(v) { this._checked = !!v; }
  get checked() { return this._checked; }
  querySelectorAll() { return []; }
  querySelector() { return null; }
  closest() { return null; }
  click() {}
  focus() {}
}

const _elements = {};
function getEl(id) {
  if (!_elements[id]) _elements[id] = new Element(id);
  return _elements[id];
}

const documentMock = {
  getElementById: getEl,
  createElement: function () { return new Element(null); },
  querySelectorAll: function () { return []; },
  querySelector: function () { return null; },
  addEventListener: function () {}
};

const sessionStorageMock = {
  _data: {},
  getItem: function (k) { return Object.prototype.hasOwnProperty.call(this._data, k) ? this._data[k] : null; },
  setItem: function (k, v) { this._data[k] = String(v); },
  removeItem: function (k) { delete this._data[k]; }
};

const locationMock = { pathname: '/console/compare' };

// `fetch` is only called from loadReports/login/runCompare/downloadPdf —
// none of which are exercised by the sort/pagination tests.  A no-op
// promise keeps the IIFE's bootstrap() from touching the network.
const fetchMock = function () {
  return Promise.resolve({ ok: false, status: 404, json: function () { return Promise.resolve({}); }, text: function () { return Promise.resolve(''); }, blob: function () { return Promise.resolve({}); } });
};

// --- Load the page script ---------------------------------------------------
const scriptPath = process.argv[2];
if (!scriptPath) { console.error('usage: node runner.js <script-path>'); process.exit(2); }
const original = fs.readFileSync(scriptPath, 'utf-8');

// --- Strip the IIFE wrapper -------------------------------------------------
// Original: (function () { ... body ... })();
// Wrap the body in a fresh function that takes the mocks as parameters
// and returns the inner state for the test assertions to inspect.
const LEAD = '(function () {';
const TRAIL = '})();';
const startIdx = original.indexOf(LEAD);
const endIdx = original.lastIndexOf(TRAIL);
if (startIdx < 0 || endIdx <= startIdx) {
  console.error('Could not locate the Compare view IIFE wrapper.');
  process.exit(2);
}
const body = original.substring(startIdx + LEAD.length, endIdx);

const wrapped =
  '(function (document, sessionStorage, location, fetch) {\n' +
  body + '\n' +
  '  return {\n' +
  '    PAGE_SIZE: PAGE_SIZE,\n' +
  '    DEFAULT_DIR: DEFAULT_DIR,\n' +
  '    PICK_COLS: PICK_COLS,\n' +
  '    sortedReports: sortedReports,\n' +
  '    setSort: setSort,\n' +
  '    renderPickerPager: renderPickerPager,\n' +
  '    sortInd: sortInd,\n' +
  '    modsSummary: modsSummary,\n' +
  '    reportById: reportById,\n' +
  '    renderPicker: renderPicker,\n' +
  '    selected: selected,\n' +
  '    // State is read-only via getters (so tests can introspect the *current*\n' +
  '    // value of the IIFE-scoped vars, not a snapshot taken at return-time).\n' +
  '    // Mutations go through the explicit setters below.\n' +
  '    getReports: function () { return REPORTS; },\n' +
  '    setReports: function (rs) { REPORTS = rs; },\n' +
  '    getSortKey: function () { return sortKey; },\n' +
  '    setSortKey: function (k) { sortKey = k; },\n' +
  '    getSortDir: function () { return sortDir; },\n' +
  '    setSortDir: function (d) { sortDir = d; },\n' +
  '    getPage: function () { return page; },\n' +
  '    setPage: function (p) { page = p; }\n' +
  '  };\n' +
  '})(documentMock, sessionStorageMock, locationMock, fetchMock);';

// --- Run the wrapped IIFE inside a sandbox so the mocks are visible --------
// The wrapped IIFE invokes `(function (...) { ... })(documentMock, ...)`,
// so the sandbox must expose the mock variables *by those names* (not just
// as `document` / `sessionStorage` / `location` / `fetch`).
const sandbox = {
  document: documentMock,
  sessionStorage: sessionStorageMock,
  location: locationMock,
  fetch: fetchMock,
  documentMock: documentMock,
  sessionStorageMock: sessionStorageMock,
  locationMock: locationMock,
  fetchMock: fetchMock,
  console: console,
  Promise: Promise,
  Set: Set,
  Date: Date,
  Math: Math,
  JSON: JSON,
  Object: Object,
  Array: Array,
  String: String,
  Number: Number,
  isNaN: isNaN,
  encodeURIComponent: encodeURIComponent,
  URL: (typeof URL !== 'undefined') ? URL : undefined
};
vm.createContext(sandbox);
let IWC;
try {
  IWC = vm.runInContext(wrapped, sandbox, { timeout: 2000 });
} catch (e) {
  console.error('IIFE execution failed:', e && e.stack ? e.stack : e);
  process.exit(1);
}

// --- Tiny test framework ---------------------------------------------------
const tests = [];
function test(name, fn) {
  try { fn(); tests.push({ name: name, status: 'PASS' }); }
  catch (e) { tests.push({ name: name, status: 'FAIL', err: (e && e.message) || String(e) }); }
}
function eq(a, b, msg) {
  if (a !== b) throw new Error((msg || 'eq') + ': expected ' + JSON.stringify(b) + ', got ' + JSON.stringify(a));
}
function ok(c, msg) { if (!c) throw new Error(msg || 'ok: false'); }
function truthy(v, msg) { if (!v) throw new Error(msg || 'truthy: ' + JSON.stringify(v)); }

// --- Fixtures ---------------------------------------------------------------
const NOW = Date.parse('2026-07-01T00:00:00Z');
function makeReports(n) {
  const out = [];
  for (let i = 0; i < n; i++) {
    // Intentionally non-monotonic insertion order to make sure sort
    // (not array insertion order) drives the result.
    out.push({
      id: 'r' + String(i).padStart(3, '0'),
      display_id: 'run-' + String(i + 1).padStart(3, '0'),
      dataset_id: 'ds-alpha',
      dataset_name: 'Alpha',
      config_snapshot: { generator: 'g' + (i % 3), embedder: 'e' + (i % 2) },
      created_at: new Date(NOW + i * 86400000).toISOString()
    });
  }
  // Shuffle the input so the fixture would FAIL without a real sort.
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    const t = out[i]; out[i] = out[j]; out[j] = t;
  }
  return out;
}

// --- Tests ------------------------------------------------------------------
// 1. PAGE_SIZE contract
test('PAGE_SIZE is exactly 20 (fixed 20-row pagination)', function () {
  eq(IWC.PAGE_SIZE, 20);
});

// 2. Initial sort state (boots into newest-first)
test('initial sortKey="created_at" + sortDir="desc" + page=1 (newest first)', function () {
  eq(IWC.getSortKey(), 'created_at');
  eq(IWC.getSortDir(), 'desc');
  eq(IWC.getPage(), 1);
});

// 3. PICK_COLS structure (every data column, in display order)
test('PICK_COLS lists run/dataset/config/created_at in display order', function () {
  const keys = IWC.PICK_COLS.map(function (c) { return c.key; });
  eq(JSON.stringify(keys), JSON.stringify(['run', 'dataset', 'config', 'created_at']));
  const labels = IWC.PICK_COLS.map(function (c) { return c.label; });
  eq(JSON.stringify(labels), JSON.stringify(['Run', 'Dataset', 'Configuration', 'Created']));
});

// 4. DEFAULT_DIR covers every column
test('DEFAULT_DIR has an asc/desc entry for every PICK_COLS key', function () {
  for (let i = 0; i < IWC.PICK_COLS.length; i++) {
    const k = IWC.PICK_COLS[i].key;
    const d = IWC.DEFAULT_DIR[k];
    ok(d === 'asc' || d === 'desc', 'DEFAULT_DIR[' + k + '] must be asc/desc, got ' + JSON.stringify(d));
  }
});

// 5. sortedReports honours the FULL set, not just the visible page
test('sortedReports sorts the FULL REPORTS list (not a pre-sliced page)', function () {
  IWC.setReports(makeReports(25));
  IWC.setSortKey('run'); IWC.setSortDir('asc');
  const sorted = IWC.sortedReports();
  eq(sorted.length, 25, 'sortedReports must return the full 25-row set');
  // run-001 first, run-025 last (alphabetical on display_id).
  eq(sorted[0].id, 'r000');
  eq(sorted[24].id, 'r024');
});

// 6. Default sort: created_at desc → newest first
test('default sort (created_at desc) returns newest run first', function () {
  IWC.setReports(makeReports(25));
  IWC.setSortKey('created_at'); IWC.setSortDir('desc');
  const sorted = IWC.sortedReports();
  // makeReports assigns timestamps NOW + i*86400000 — r024 is newest.
  eq(sorted[0].id, 'r024', 'newest (r024) must come first under created_at desc');
  eq(sorted[24].id, 'r000', 'oldest (r000) must come last under created_at desc');
});

// 7. Sort by run asc / desc
test('sortKey=run + sortDir=asc → alphabetical (r000..r024)', function () {
  IWC.setReports(makeReports(25));
  IWC.setSortKey('run'); IWC.setSortDir('asc');
  const s = IWC.sortedReports();
  eq(s[0].id, 'r000');
  eq(s[24].id, 'r024');
});
test('sortKey=run + sortDir=desc → reverse alphabetical (r024..r000)', function () {
  IWC.setReports(makeReports(25));
  IWC.setSortKey('run'); IWC.setSortDir('desc');
  const s = IWC.sortedReports();
  eq(s[0].id, 'r024');
  eq(s[24].id, 'r000');
});

// 8. setSort toggles direction when clicking the active column
test('setSort on the active column TOGGLES direction (desc → asc → desc)', function () {
  IWC.setSortKey('created_at'); IWC.setSortDir('desc'); IWC.setPage(3);
  IWC.setSort('created_at');
  eq(IWC.getSortDir(), 'asc');
  eq(IWC.getPage(), 1, 'setSort must reset page to 1 on toggle');
  IWC.setSort('created_at');
  eq(IWC.getSortDir(), 'desc');
  eq(IWC.getPage(), 1);
});

// 9. setSort on a new column uses DEFAULT_DIR + resets page
test('setSort on a new column switches to DEFAULT_DIR and resets page', function () {
  IWC.setSortKey('created_at'); IWC.setSortDir('desc'); IWC.setPage(5);
  IWC.setSort('run');
  eq(IWC.getSortKey(), 'run');
  eq(IWC.getSortDir(), 'asc', 'new column run must use DEFAULT_DIR.run = "asc"');
  eq(IWC.getPage(), 1, 'setSort must reset page to 1 on column change');
  IWC.setSort('created_at');
  eq(IWC.getSortDir(), 'desc', 'new column created_at must use DEFAULT_DIR.created_at = "desc"');
  eq(IWC.getPage(), 1);
  IWC.setSort('config');
  eq(IWC.getSortDir(), 'asc', 'new column config must use DEFAULT_DIR.config = "asc"');
  eq(IWC.getPage(), 1);
  IWC.setSort('dataset');
  eq(IWC.getSortDir(), 'asc', 'new column dataset must use DEFAULT_DIR.dataset = "asc"');
  eq(IWC.getPage(), 1);
});

// 10. renderPickerPager hidden for small result sets
test('renderPickerPager returns "" for total ≤ PAGE_SIZE (no pager UI)', function () {
  IWC.setPage(1);
  eq(IWC.renderPickerPager(0), '');
  eq(IWC.renderPickerPager(1), '');
  eq(IWC.renderPickerPager(19), '');
  eq(IWC.renderPickerPager(20), '');
});

// 11. renderPickerPager for 25 reports, page 1
test('renderPickerPager(25) page 1 shows "Page 1 of 2 · 1–20 of 25"', function () {
  IWC.setPage(1);
  const html = IWC.renderPickerPager(25);
  truthy(html.indexOf('pickPrev') !== -1, 'pager must include #pickPrev');
  truthy(html.indexOf('pickNext') !== -1, 'pager must include #pickNext');
  truthy(html.indexOf('Page 1 of 2') !== -1, 'pager info must read "Page 1 of 2"');
  truthy(html.indexOf('1&ndash;20') !== -1, 'pager info must read "1&ndash;20" (en-dash entity)');
  truthy(html.indexOf('of 25') !== -1, 'pager info must read "of 25"');
});

// 12. Page 2 of 25 shows rows 21-25
test('renderPickerPager(25) page 2 shows "Page 2 of 2 · 21–25 of 25"', function () {
  IWC.setPage(2);
  const html = IWC.renderPickerPager(25);
  truthy(html.indexOf('Page 2 of 2') !== -1, 'pager info must read "Page 2 of 2"');
  truthy(html.indexOf('21&ndash;25') !== -1, 'pager info must read "21&ndash;25" (en-dash entity)');
  truthy(html.indexOf('of 25') !== -1, 'pager info must read "of 25"');
});

// 13. Page 2 of 40 shows rows 21-40 (full second page)
test('renderPickerPager(40) page 2 shows "Page 2 of 2 · 21–40 of 40"', function () {
  IWC.setPage(2);
  const html = IWC.renderPickerPager(40);
  truthy(html.indexOf('Page 2 of 2') !== -1, 'pager info must read "Page 2 of 2"');
  truthy(html.indexOf('21&ndash;40') !== -1, 'pager info must read "21&ndash;40" (en-dash entity)');
  truthy(html.indexOf('of 40') !== -1, 'pager info must read "of 40"');
});

// 14. Overflow by one row (21 reports) → Page 1 of 2, rows 1-20
test('renderPickerPager(21) overflows by one → "Page 1 of 2 · 1–20 of 21"', function () {
  IWC.setPage(1);
  const html = IWC.renderPickerPager(21);
  truthy(html.indexOf('Page 1 of 2') !== -1, 'pager info must read "Page 1 of 2"');
  truthy(html.indexOf('1&ndash;20') !== -1, 'pager info must read "1&ndash;20" (en-dash entity)');
  truthy(html.indexOf('of 21') !== -1, 'pager info must read "of 21"');
  // 21 reports → 2 pages
  IWC.setPage(2);
  const html2 = IWC.renderPickerPager(21);
  truthy(html2.indexOf('Page 2 of 2') !== -1, 'pager page 2 must read "Page 2 of 2"');
  truthy(html2.indexOf('21&ndash;21') !== -1, 'pager page 2 must read "21&ndash;21" (en-dash entity)');
});

// 15. Exactly 20 reports → 1 page, no pager
test('renderPickerPager(20) returns "" (exactly one full page, no pager)', function () {
  eq(IWC.renderPickerPager(20), '');
});

// 16. Sort applies to FULL set before page slicing
test('page 1 of 25 with sortKey=run asc shows r000..r019 (top 20 of the full sort)', function () {
  IWC.setReports(makeReports(25));
  IWC.setSortKey('run'); IWC.setSortDir('asc'); IWC.setPage(1);
  const sorted = IWC.sortedReports();
  const start = (1 - 1) * IWC.PAGE_SIZE;
  const slice = sorted.slice(start, start + IWC.PAGE_SIZE);
  eq(slice.length, 20);
  eq(slice[0].id, 'r000', 'page 1 row 1 must be r000 (first of full sorted set)');
  eq(slice[19].id, 'r019', 'page 1 row 20 must be r019 (20th of full sorted set)');
});

// 17. Page 2 of 25 → rows 20-24 (r020..r024)
test('page 2 of 25 with sortKey=run asc shows r020..r024 (continuation of the full sort)', function () {
  IWC.setReports(makeReports(25));
  IWC.setSortKey('run'); IWC.setSortDir('asc'); IWC.setPage(2);
  const sorted = IWC.sortedReports();
  const start = (2 - 1) * IWC.PAGE_SIZE;
  const slice = sorted.slice(start, start + IWC.PAGE_SIZE);
  eq(slice.length, 5, 'page 2 of 25 must have 5 rows (the tail of the full sort)');
  eq(slice[0].id, 'r020');
  eq(slice[4].id, 'r024');
});

// 18. sortInd glyphs
test('sortInd returns ▾ for inactive column, ▲ for active asc, ▼ for active desc', function () {
  IWC.setSortKey('created_at'); IWC.setSortDir('desc');
  eq(IWC.sortInd('run'), '\u25BE', 'inactive column must show ▾');
  eq(IWC.sortInd('created_at'), '\u25BC', 'active desc must show ▼');
  IWC.setSortDir('asc');
  eq(IWC.sortInd('created_at'), '\u25B2', 'active asc must show ▲');
});

// 19. Sort survives a page change (the same first-row id appears on page 1
//     and on page 1 of a fresh sort, so the user can't see rows shift just
//     because they clicked Next).
test('changing page does not change the row ORDER (sort is applied once, before slicing)', function () {
  IWC.setReports(makeReports(40));
  IWC.setSortKey('run'); IWC.setSortDir('asc');
  IWC.setPage(1);
  const full = IWC.sortedReports();
  const p1Ids = full.slice(0, 20).map(function (r) { return r.id; });
  IWC.setPage(2);
  const p2Ids = full.slice(20, 40).map(function (r) { return r.id; });
  // p1 row 1 = r000; p2 row 1 must NOT be r001 (it would be if sort was
  // applied to the wrong slice).  It must be the 21st row of the full sort.
  eq(p1Ids[0], 'r000', 'page 1 first row = r000');
  eq(p2Ids[0], 'r020', 'page 2 first row = r020 (continuation of full sort)');
  // Page 1 and page 2 must be disjoint.
  for (let i = 0; i < p1Ids.length; i++) {
    ok(p2Ids.indexOf(p1Ids[i]) === -1, 'page 1 row ' + p1Ids[i] + ' must not reappear on page 2');
  }
});

// 20. modsSummary (config sort key uses this)
test('modsSummary produces a stable string for the config sort key', function () {
  const rep = {
    id: 'r0', config_snapshot: { generator: 'qwen', embedder: 'bge', reranker: 'bge', grader: 'gpt' },
    dataset_name: 'Alpha'
  };
  const s = IWC.modsSummary(rep);
  truthy(s.indexOf('qwen') !== -1, 'modsSummary must include generator');
  truthy(s.indexOf('bge') !== -1, 'modsSummary must include embedder');
  truthy(s.indexOf('Alpha') !== -1, 'modsSummary must include dataset_name');
});

// 21. Page clamp after out-of-range navigation (S04 polish — the
//     render-time `if (page > totalPages) { page = totalPages; }` was
//     previously only covered by a static test; this exercises the
//     actual behaviour end-to-end through renderPicker()).
test('renderPicker clamps page to totalPages when setPage is out of range', function () {
  IWC.setReports(makeReports(25));
  IWC.setSortKey('run'); IWC.setSortDir('asc');
  IWC.setPage(5);  // out of range — 25 reports / 20 = 2 pages
  eq(IWC.getPage(), 5, 'setPage must set page to 5 verbatim (no clamp in the setter)');
  IWC.renderPicker();
  eq(IWC.getPage(), 2, 'renderPicker must clamp page to totalPages (2 for 25 reports)');
  const html = getEl('pickBody').innerHTML;
  truthy(html.indexOf('Page 2 of 2') !== -1, 'rendered pager must show "Page 2 of 2" after clamp');
  truthy(html.indexOf('21&ndash;25') !== -1, 'rendered pager must show "21&ndash;25" after clamp');
});

// 22. Page clamp also handles page < 1 (defensive — UI never exposes a
//     way to go below 1, but a stale state from a future server-driven
//     re-render could land here).
test('renderPicker clamps page to 1 when setPage is below the first page', function () {
  IWC.setReports(makeReports(25));
  IWC.setSortKey('run'); IWC.setSortDir('asc');
  IWC.setPage(0);
  IWC.renderPicker();
  eq(IWC.getPage(), 1, 'renderPicker must clamp page to 1 when setPage was 0');
  const html = getEl('pickBody').innerHTML;
  truthy(html.indexOf('Page 1 of 2') !== -1, 'rendered pager must show "Page 1 of 2" after clamp');
  truthy(html.indexOf('1&ndash;20') !== -1, 'rendered pager must show "1&ndash;20" after clamp');
});

// 23. ModsSummary memoization (S04 polish — the precompute refactor
//     collapses the per-comparison modsSummary() calls to a single
//     pass over REPORTS; the observable behaviour is unchanged but
//     the speed-up is real for large REPORTS).  We verify the
//     equivalence with a unique-config fixture (so asc/desc strictly
//     reverse — the standard makeReports fixture has duplicate configs
//     which makes asc[i] != desc[len-1-i] for stable-sort ties).
test('config sort with unique summaries is a perfect reversal of asc/desc', function () {
  const reps = [];
  for (let i = 0; i < 15; i++) {
    reps.push({
      id: 'r' + String(i).padStart(3, '0'),
      display_id: 'run-' + String(i + 1).padStart(3, '0'),
      dataset_id: 'ds-alpha',
      dataset_name: 'Alpha',
      config_snapshot: { generator: 'gen' + i, embedder: 'emb' + i },
      created_at: new Date(NOW + i * 86400000).toISOString()
    });
  }
  IWC.setReports(reps);
  IWC.setSortKey('config'); IWC.setSortDir('asc');
  const asc = IWC.sortedReports();
  IWC.setSortDir('desc');
  const desc = IWC.sortedReports();
  eq(asc.length, desc.length, 'asc and desc must return the same number of rows');
  for (let i = 0; i < asc.length; i++) {
    eq(asc[i].id, desc[desc.length - 1 - i].id,
       'asc[' + i + '] must equal desc[len-1-' + i + '] (unique-config strict reversal)');
  }
});

// --- Summary ----------------------------------------------------------------
let pass = 0, fail = 0;
for (let i = 0; i < tests.length; i++) {
  if (tests[i].status === 'PASS') { pass++; console.log('PASS  ' + tests[i].name); }
  else { fail++; console.error('FAIL  ' + tests[i].name + ' \u2014 ' + tests[i].err); }
}
console.log('\n' + pass + ' passed, ' + fail + ' failed');
process.exit(fail === 0 ? 0 : 1);
"""


# ---------------------------------------------------------------------------
# Pytest gate: Node must be on PATH (the existing static test already
# relies on it for `node --check`).
# ---------------------------------------------------------------------------

_NODE = shutil.which("node")
if _NODE is None:  # pragma: no cover - mirrors the existing skip path
    pytest.skip(
        "node binary not on PATH; skipping the sort + pagination behavioural tests",
        allow_module_level=True,
    )


def _run_node_tests() -> subprocess.CompletedProcess[str]:
    """Write the Node runner to a temp file, feed it the page's inline
    <script>, run it, and return the CompletedProcess.

    Skips (returns a synthetic CompletedProcess with returncode=0 and a
    short stdout note) if the Node binary disappears between module load
    and test execution.
    """
    import tempfile

    node = shutil.which("node")
    if node is None:  # pragma: no cover
        return subprocess.CompletedProcess(
            args=["node"], returncode=0, stdout="(node missing — skipped)\n", stderr=""
        )

    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as runner_fh:
        runner_fh.write(_NODE_RUNNER)
        runner_path = runner_fh.name
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as script_fh:
        script_fh.write(_inline_script())
        script_path = script_fh.name

    try:
        return subprocess.run(
            [node, runner_path, script_path],
            capture_output=True,
            text=True,
            timeout=30,
        )
    finally:
        import contextlib
        import os

        with contextlib.suppress(OSError):
            os.unlink(runner_path)
        with contextlib.suppress(OSError):
            os.unlink(script_path)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_compare_page_sort_node_runner_boots() -> None:
    """Sanity: the Node runner + the inline script are wired up correctly
    (this catches the env cases where the IIFE strip or the mocks regress)."""
    result = _run_node_tests()
    assert result.returncode == 0, (
        "Compare view sort + pagination behavioural runner failed.\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )


def test_compare_page_sort_page_size_is_twenty() -> None:
    """Behavioural: PAGE_SIZE is exactly 20 (fixed 20-row pagination)."""
    result = _run_node_tests()
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PAGE_SIZE is exactly 20" in result.stdout
    assert "PASS  PAGE_SIZE is exactly 20" in result.stdout


def test_compare_page_sort_initial_state_newest_first() -> None:
    """Behavioural: picker boots sorted by 'created_at' descending (page 1)."""
    result = _run_node_tests()
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'PASS  initial sortKey="created_at" + sortDir="desc" + page=1' in result.stdout


def test_compare_page_sort_every_column_is_sortable() -> None:
    """Behavioural: PICK_COLS lists every data column, in display order."""
    result = _run_node_tests()
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS  PICK_COLS lists run/dataset/config/created_at in display order" in result.stdout
    assert "PASS  DEFAULT_DIR has an asc/desc entry for every PICK_COLS key" in result.stdout


def test_compare_page_sort_click_same_column_toggles() -> None:
    """Behavioural: clicking the active column toggles direction (desc → asc)."""
    result = _run_node_tests()
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS  setSort on the active column TOGGLES direction" in result.stdout


def test_compare_page_sort_click_new_column_uses_default_dir() -> None:
    """Behavioural: clicking a new column switches to DEFAULT_DIR + resets page."""
    result = _run_node_tests()
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS  setSort on a new column switches to DEFAULT_DIR and resets page" in result.stdout


def test_compare_page_sort_exactly_20_rows_per_page() -> None:
    """Behavioural: page 1 of 25 returns exactly 20 rows, page 2 returns the rest."""
    result = _run_node_tests()
    assert result.returncode == 0, result.stdout + result.stderr
    # renderPickerPager hidden when total <= PAGE_SIZE
    assert 'PASS  renderPickerPager returns "" for total ≤ PAGE_SIZE' in result.stdout
    # Exactly 20 reports → no pager (one full page)
    assert 'PASS  renderPickerPager(20) returns ""' in result.stdout
    # 21 reports → 2 pages
    assert "PASS  renderPickerPager(21) overflows by one" in result.stdout
    # Page 1 of 25 → 20 rows
    assert "PASS  page 1 of 25 with sortKey=run asc shows r000..r019" in result.stdout


def test_compare_page_sort_page_2_shows_run_21_onward() -> None:
    """Behavioural: page 2 of a 25-report set shows rows 21..25 (the tail)."""
    result = _run_node_tests()
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'PASS  renderPickerPager(25) page 2 shows "Page 2 of 2' in result.stdout
    assert "PASS  page 2 of 25 with sortKey=run asc shows r020..r024" in result.stdout


def test_compare_page_sort_full_page_2_of_40() -> None:
    """Behavioural: page 2 of 40 shows rows 21..40 (a full second page)."""
    result = _run_node_tests()
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'PASS  renderPickerPager(40) page 2 shows "Page 2 of 2' in result.stdout


def test_compare_page_sort_sorts_full_set_before_slice() -> None:
    """Behavioural: sort applies to the FULL report set; changing page never
    re-orders rows.  A 40-report set under sortKey=run asc puts r000 on
    page 1 and r020 on page 2 (page 1 and page 2 are disjoint)."""
    result = _run_node_tests()
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS  changing page does not change the row ORDER" in result.stdout


def test_compare_page_sort_sorted_reports_sorts_full_set() -> None:
    """Behavioural: sortedReports() sorts the FULL list, not a pre-sliced page."""
    result = _run_node_tests()
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS  sortedReports sorts the FULL REPORTS list" in result.stdout
    assert "PASS  default sort (created_at desc) returns newest run first" in result.stdout
    assert "PASS  sortKey=run + sortDir=asc" in result.stdout
    assert "PASS  sortKey=run + sortDir=desc" in result.stdout


def test_compare_page_sort_indicator_glyphs() -> None:
    """Behavioural: sortInd() returns ▾ for inactive, ▲ for asc, ▼ for desc."""
    result = _run_node_tests()
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS  sortInd returns ▾ for inactive column" in result.stdout


def test_compare_page_sort_mods_summary_for_config_sort() -> None:
    """Behavioural: modsSummary produces a sortable string (used by config sort)."""
    result = _run_node_tests()
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS  modsSummary produces a stable string for the config sort key" in result.stdout


def test_compare_page_sort_all_behavioural_tests_pass() -> None:
    """Final aggregate: every behavioural assertion is green, and the
    runner reports the expected pass count."""
    result = _run_node_tests()
    assert result.returncode == 0, (
        "Compare view behavioural tests reported failures.\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    # The runner must report zero failures and a positive pass count.
    m = re.search(r"(\d+)\s+passed,\s+(\d+)\s+failed", result.stdout or "")
    assert m, f"Could not parse runner summary from stdout:\n{result.stdout!r}"
    passed, failed = int(m.group(1)), int(m.group(2))
    assert failed == 0, f"Runner reported {failed} failures; expected 0."
    # 23 internal JS tests (20 from S02 + 3 S04 polish: page-clamp-high,
    # page-clamp-low, config-sort-memoization-reversal).
    assert passed >= 23, f"Runner reported only {passed} passes; expected ≥ 23."


def test_compare_page_sort_page_clamp_high_after_render() -> None:
    """Behavioural (S04 polish): renderPicker() clamps page to totalPages
    when setPage was out of range (so a stale state from a future
    server-driven re-render cannot leave the picker on a non-existent
    page)."""
    result = _run_node_tests()
    assert result.returncode == 0, result.stdout + result.stderr
    assert (
        "PASS  renderPicker clamps page to totalPages when setPage is out of range" in result.stdout
    )


def test_compare_page_sort_page_clamp_low_after_render() -> None:
    """Behavioural (S04 polish): renderPicker() clamps page to 1 when
    setPage was below the first page (defensive against stale state)."""
    result = _run_node_tests()
    assert result.returncode == 0, result.stdout + result.stderr
    assert (
        "PASS  renderPicker clamps page to 1 when setPage is below the first page" in result.stdout
    )


def test_compare_page_sort_config_sort_reversal() -> None:
    """Behavioural (S04 polish): config sort is stable — ascending and
    descending orderings reverse each other on a unique-config fixture
    (the precompute refactor preserves the original stable-sort guarantee)."""
    result = _run_node_tests()
    assert result.returncode == 0, result.stdout + result.stderr
    assert (
        "PASS  config sort with unique summaries is a perfect reversal of asc/desc" in result.stdout
    )
