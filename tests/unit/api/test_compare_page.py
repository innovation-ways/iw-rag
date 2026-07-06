# ruff: noqa: E501
"""Unit tests for the multi-run Compare view page module.

The Compare view is a self-contained inline HTML+CSS+vanilla-JS page
(mirroring :mod:`iw_rag.api.results_page`) that compares 2..5
``benchmark_reports`` rows on the same dataset against a baseline.  These
tests pin the contract verifiable from Python: the page string returned
by :func:`iw_rag.api.compare_page.compare_page_html` carries every
section anchor (the run picker, the legend, the aggregate bars, the
baseline-relative deltas, the per-item heatmap + answer/context diff, the
Generate Report (PDF) button), the IW brand chrome, the shared console
nav (now including the Compare tab), the ``/auth/login`` self-handled
sign-in, the fixed ``/api`` edge base, the sessionStorage-only bearer
token, and a syntactically valid inline ``<script>`` (``node --check``).
"""

from __future__ import annotations

import re
import subprocess
import tempfile

from iw_rag.api.compare_page import compare_page_html

# ---------------------------------------------------------------------------
# Page factory / brand / operational keystone
# ---------------------------------------------------------------------------


def test_compare_page_html_returns_non_empty_document() -> None:
    page = compare_page_html()
    assert isinstance(page, str)
    assert page.lstrip().lower().startswith("<!doctype html>")
    assert "</html>" in page


def test_compare_page_html_contains_brand_palette() -> None:
    """All three primary brand palette tokens appear in the CSS; teal appears multiple times."""
    page = compare_page_html()
    assert "#0D9488" in page  # teal-deep
    assert "#2DD4BF" in page  # teal
    assert "#181D29" in page  # ink
    # teal (#2DD4BF) is used more than once (root var + at least one direct reference).
    assert page.count("#2DD4BF") >= 2


def test_compare_page_html_has_no_external_cdn() -> None:
    page = compare_page_html().lower()
    for forbidden in (
        "fonts.googleapis.com",
        "fonts.gstatic.com",
        "cdn.jsdelivr.net",
        "unpkg.com",
        "cdnjs.cloudflare.com",
        "http://localhost",
        "http://127.0.0.1",
    ):
        assert forbidden not in page, (
            f"Compare view must not reference '{forbidden}' (Invariant 8)."
        )


def test_compare_page_html_uses_fixed_api_edge_base() -> None:
    page = compare_page_html()
    assert page.count('var API = "/api";') == 1


def test_compare_page_html_logs_in_via_auth_login() -> None:
    page = compare_page_html()
    assert page.count("/auth/login") == 1
    assert "sessionStorage" in page


def test_compare_page_html_has_data_theme_dark_and_wordmark() -> None:
    page = compare_page_html()
    assert page.count('data-theme="dark"') == 1
    assert '<span class="i">I</span>nnovation <span class="w">W</span>ays' in page
    assert "<title>IW - RAG</title>" in page


# ---------------------------------------------------------------------------
# Console nav — the new Compare tab sits between Results and Scoreboard
# ---------------------------------------------------------------------------


def test_compare_page_html_has_compare_nav_after_results() -> None:
    page = compare_page_html()
    for nav in (
        'id="navConfig"',
        'id="navBenchmark"',
        'id="navResults"',
        'id="navCompare"',
        'id="navScoreboard"',
        'id="navChat"',
        'id="navUpload"',
    ):
        assert nav in page, f"missing console-nav anchor {nav}"
    assert 'href="/console/compare"' in page
    # Ordering: Compare appears after Results and before Scoreboard.
    i_results = page.index('id="navResults"')
    i_compare = page.index('id="navCompare"')
    i_score = page.index('id="navScoreboard"')
    assert i_results < i_compare < i_score


# ---------------------------------------------------------------------------
# Endpoints the view consumes
# ---------------------------------------------------------------------------


def test_compare_page_html_references_endpoints() -> None:
    """All three compare endpoints are referenced; PDF endpoint appears after data endpoint."""
    page = compare_page_html()
    assert "/benchmark/reports" in page  # picker source
    assert "/benchmark/compare/multi" in page  # data endpoint
    assert "/benchmark/compare/multi/pdf" in page  # PDF endpoint
    # The data fetch encodes the ids + baseline query params.
    assert "ids=" in page
    assert "baseline=" in page
    # The data endpoint appears before the PDF endpoint in the JS source.
    assert page.index("/benchmark/compare/multi?") < page.index("/benchmark/compare/multi/pdf")


# ---------------------------------------------------------------------------
# Picker + render anchors
# ---------------------------------------------------------------------------


def test_compare_page_html_has_run_picker_and_baseline() -> None:
    """Picker, baseline selector, compare button, and MAX_RUNS cap all present in order."""
    page = compare_page_html()
    assert 'id="pickBody"' in page
    assert 'id="baselineSel"' in page
    assert 'id="compareBtn"' in page
    # 2..5 run cap is enforced client-side (UX honesty; API is authoritative).
    assert "MAX_RUNS" in page
    # The picker appears before the compare button in the markup.
    assert page.index('id="pickBody"') < page.index('id="compareBtn"')


def test_compare_page_html_has_render_anchors() -> None:
    page = compare_page_html()
    # legend (per-run colour chips), aggregate bars, baseline deltas, heatmap.
    assert "legend-wrap" in page
    assert "Aggregate" in page
    assert "Baseline-relative deltas" in page
    assert "heatmap" in page
    assert "Per-item comparison" in page
    # The qualitative diff: answers + contexts per run.
    assert "Answers" in page or "answer" in page.lower()


def test_compare_page_html_has_pdf_button() -> None:
    """PDF button id and label appear exactly once each."""
    page = compare_page_html()
    assert 'id="pdfBtn"' in page
    assert "Generate Report (PDF)" in page
    # The button id and label should each appear exactly once.
    assert page.count('id="pdfBtn"') == 1


def test_compare_page_html_renders_underpowered_caveat() -> None:
    """Underpowered caveat CSS class is defined and 'Underpowered' text is present."""
    page = compare_page_html()
    assert "caveat-box" in page
    assert "Underpowered" in page
    # The caveat-box class must be defined in the CSS (style block) before the JS uses it.
    assert page.index(".caveat-box") < page.index("Underpowered")


def test_compare_page_html_handles_400_and_404() -> None:
    page = compare_page_html()
    assert "400" in page and "404" in page
    assert "unknown" in page.lower()
    assert "not comparable" in page.lower() or "same dataset" in page.lower()


# ---------------------------------------------------------------------------
# Inline script parses
# ---------------------------------------------------------------------------


def test_compare_page_html_inline_script_parses() -> None:
    """The page's inline ``<script>`` must be syntactically valid JS."""
    scripts = re.findall(r"<script\b[^>]*>(.*?)</script>", compare_page_html(), re.S)
    assert scripts, "no <script> block found in the Compare view page"
    script = "\n".join(scripts)
    assert "\ndata: <json>\n" not in ("\n" + script + "\n"), (
        "rendered inline script contains a bare `data: <json>` line — raw-string regression."
    )
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as fh:
        fh.write(script)
        path = fh.name
    try:
        node = subprocess.run(["node", "--check", path], capture_output=True, text=True, timeout=10)
    except FileNotFoundError:
        return  # no node in this env — skip the JS-side assertion
    assert node.returncode == 0, (
        f"Compare view inline script failed `node --check`: stderr={node.stderr!r}"
    )


# ---------------------------------------------------------------------------
# Runs picker — sortable headers + fixed 20-row pagination (S01 / S02)
#
# These tests pin the JS-side contract that the runs picker table is
# fully sortable (every data column), sorts across the FULL report set
# (not just the visible page) and pages 20 rows at a time.  The
# behavioural verification (click toggles direction, page 2 shows
# run 21.., etc.) lives in test_compare_page_sort.py, which actually
# executes the inline <script> under Node with a minimal DOM mock.
# ---------------------------------------------------------------------------


def _compare_page_script() -> str:
    """Return the inline <script> body for the Compare view page."""
    scripts = re.findall(r"<script\b[^>]*>(.*?)</script>", compare_page_html(), re.S)
    assert scripts, "Compare view page has no <script> block"
    return "\n".join(scripts)


def _function_body(script: str, name: str) -> str:
    """Extract the body of ``function NAME(...) { ... }`` from ``script``.

    Uses a brace counter (not a regex) so nested blocks (e.g. an inline
    ``switch`` with ``case "x": { ... }`` blocks) do not prematurely
    close the function.  Returns the text between the opening ``{`` and
    the matching closing ``}`` (both excluded).  Raises ``AssertionError``
    if the function is not found.
    """
    sig = re.search(rf"function\s+{re.escape(name)}\s*\([^)]*\)\s*\{{", script)
    assert sig, f"function {name}() not found in the Compare view script"
    start = sig.end()  # position just after the opening '{'
    depth = 1
    i = start
    n = len(script)
    while i < n and depth > 0:
        c = script[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        i += 1
    assert depth == 0, f"function {name}() has unbalanced braces in the Compare view script"
    return script[start : i - 1]


def test_compare_page_html_pins_page_size_to_20() -> None:
    """PAGE_SIZE is exactly 20 (fixed 20-row pagination contract)."""
    script = _compare_page_script()
    # The constant declaration must exist with the value 20 (no magic number elsewhere).
    assert re.search(r"var\s+PAGE_SIZE\s*=\s*20\s*;", script), (
        "Runs picker must pin PAGE_SIZE = 20 (fixed 20-row pagination)."
    )


def test_compare_page_html_pick_cols_lists_every_data_column() -> None:
    """Every data column appears in PICK_COLS in display order."""
    script = _compare_page_script()
    # All four data columns must be declared in PICK_COLS, in display order.
    m = re.search(r"var\s+PICK_COLS\s*=\s*\[(.*?)\];", script, re.S)
    assert m, "PICK_COLS array is missing from the Compare view script"
    body = m.group(1)
    # The keys (case-sensitive) must appear in this exact order.
    keys = re.findall(r"key\s*:\s*\"([^\"]+)\"", body)
    assert keys == ["run", "dataset", "config", "created_at"], (
        f"PICK_COLS keys must be ['run','dataset','config','created_at'] in that order, got {keys!r}"
    )
    # The labels (case-sensitive) must match the headers the user sees.
    labels = re.findall(r"label\s*:\s*\"([^\"]+)\"", body)
    assert labels == ["Run", "Dataset", "Configuration", "Created"], (
        f"PICK_COLS labels must be ['Run','Dataset','Configuration','Created'] in that order, "
        f"got {labels!r}"
    )


def test_compare_page_html_default_dir_covers_every_column() -> None:
    """DEFAULT_DIR has an entry for every PICK_COLS key."""
    script = _compare_page_script()
    m_dir = re.search(r"var\s+DEFAULT_DIR\s*=\s*\{(.*?)\};", script, re.S)
    m_cols = re.search(r"var\s+PICK_COLS\s*=\s*\[(.*?)\];", script, re.S)
    assert m_dir and m_cols, "DEFAULT_DIR and PICK_COLS must both be present"
    dir_keys = set(re.findall(r"(\w+)\s*:\s*\"(asc|desc)\"", m_dir.group(1)))
    col_keys = set(re.findall(r"key\s*:\s*\"([^\"]+)\"", m_cols.group(1)))
    assert col_keys <= set(k for k, _ in dir_keys), (
        f"Every PICK_COLS key must have a DEFAULT_DIR entry; "
        f"missing: {col_keys - set(k for k, _ in dir_keys)}"
    )
    # The default sort column is 'created_at' newest-first, and the picker
    # boots with that direction — the run column defaults to 'asc' (oldest
    # of the alphabetical runs first), the config + dataset columns to
    # 'asc'.  This is the contract the picker boots in.
    assert ("created_at", "desc") in dir_keys
    assert ("run", "asc") in dir_keys
    assert ("dataset", "asc") in dir_keys
    assert ("config", "asc") in dir_keys


def test_compare_page_html_initial_sort_state_is_created_at_desc() -> None:
    """The picker boots sorted by 'created_at' descending (newest first)."""
    script = _compare_page_script()
    # sortKey starts at "created_at" and sortDir at "desc" (newest first).
    assert re.search(r"var\s+sortKey\s*=\s*\"created_at\"\s*;", script), (
        "Initial sortKey must be 'created_at' (newest run first)."
    )
    assert re.search(r"var\s+sortDir\s*=\s*\"desc\"\s*;", script), (
        "Initial sortDir must be 'desc' (newest first)."
    )
    assert re.search(r"var\s+page\s*=\s*1\s*;", script), (
        "Initial page must be 1 (top of the freshly-sorted set)."
    )


def test_compare_page_html_sorted_reports_implements_every_column() -> None:
    """sortedReports() handles every PICK_COLS key (run/dataset/config/created_at)."""
    script = _compare_page_script()
    body = _function_body(script, "sortedReports")
    for key in ("run", "dataset", "config", "created_at"):
        assert re.search(rf'case\s+"{key}"\s*:', body), (
            f"sortedReports() must have a case for sortKey={key!r}"
        )
    # Date sorting must use numeric compare (Date.parse) on the timestamp
    # — not a fragile string compare — so a sort by 'created_at' is stable
    # across timezones and the user's locale.
    assert "Date.parse" in body, (
        "sortedReports() must use Date.parse() for created_at (numeric compare)."
    )


def test_compare_page_html_set_sort_toggles_and_resets_page() -> None:
    """setSort toggles direction on the same column, sets DEFAULT_DIR on a new column,
    and always resets page = 1 so the user lands on the top of the new sort."""
    script = _compare_page_script()
    body = _function_body(script, "setSort")
    # Toggle branch: clicking the active column flips sortDir.
    assert re.search(r"sortKey\s*===\s*key", body), (
        "setSort must detect a click on the active column (toggle)."
    )
    assert "asc" in body and "desc" in body, "setSort must flip between asc and desc"
    # New-column branch: assign DEFAULT_DIR[key] and switch sortKey.
    assert "DEFAULT_DIR" in body, "setSort must consult DEFAULT_DIR for a new column"
    assert re.search(r"sortKey\s*=\s*key", body), "setSort must switch sortKey to the new column"
    # Pagination must reset to 1 on every sort change.
    assert re.search(r"page\s*=\s*1\s*;", body), "setSort must reset page to 1"


def test_compare_page_html_render_picker_sorts_before_slice() -> None:
    """renderPicker() sorts the FULL list first, then slices — so changing
    the active page never changes the row order."""
    script = _compare_page_script()
    body = _function_body(script, "renderPicker")
    # The sorted() call must come before the slice() call.
    idx_sort = body.index("sortedReports()")
    idx_slice = body.index(".slice(start, start + PAGE_SIZE)")
    assert idx_sort < idx_slice, (
        f"renderPicker() must call sortedReports() before .slice(...); "
        f"got sort at offset {idx_sort}, slice at offset {idx_slice}"
    )


def test_compare_page_html_render_picker_pager_contract() -> None:
    """renderPickerPager() returns the Prev/Next buttons + page info, and
    returns '' when the total is at or below PAGE_SIZE (no pager UI clutter)."""
    script = _compare_page_script()
    body = _function_body(script, "renderPickerPager")
    # Hidden when total <= PAGE_SIZE — no pager markup for small result sets.
    assert re.search(r"if\s*\(\s*total\s*<=\s*PAGE_SIZE\s*\)\s*\{\s*return\s*''\s*;\s*\}", body), (
        "renderPickerPager must return '' when total <= PAGE_SIZE (no pager for small sets)."
    )
    # The pager must carry Prev + Next buttons + a 'Page X of Y' indicator.
    assert 'id="pickPrev"' in body, "pager must include a Prev button (#pickPrev)"
    assert 'id="pickNext"' in body, "pager must include a Next button (#pickNext)"
    assert 'class="pick-page-info"' in body, "pager must include a page-info element"
    assert "Page " in body and " of " in body, "pager info must read 'Page X of Y'"


def test_compare_page_html_sortable_header_wiring() -> None:
    """Every PICK_COLS header is rendered as <th class="sortable" data-sort="...">
    and a click handler is wired to setSort(...)."""
    script = _compare_page_script()
    body = _function_body(script, "renderPicker")
    assert "PICK_COLS.forEach" in body, "renderPicker() must iterate PICK_COLS to build headers"
    # The class string is 'sortable' (+ ' asc' / ' desc' on the active col)
    # and the data-sort attribute carries the column key for the click handler.
    assert re.search(r'"sortable"\s*\+', body) or re.search(r"'sortable'\s*\+", body), (
        "renderPicker() must add the 'sortable' class to every header cell."
    )
    assert (
        re.search(r'data-sort="\'\s*\+\s*c\.key', body)
        or re.search(r"data-sort=\"\"\s*\+\s*c\.key", body)
        or re.search(r"data-sort='\s*'\s*\+\s*c\.key", body)
        or re.search(r"data-sort=['\"]\\?['\"]\s*\+\s*c\.key", body)
    ), 'renderPicker() must add data-sort="<key>" to every header cell.'
    # The click handler is wired from querySelectorAll('.pick-table th.sortable')
    # and calls setSort(th.getAttribute("data-sort")) on every click.
    assert re.search(r"querySelectorAll\(\"\.pick-table\s+th\.sortable\"\)", body), (
        "renderPicker() must querySelectorAll('.pick-table th.sortable') to wire clicks"
    )
    assert re.search(r"setSort\(th\.getAttribute\(\"data-sort\"\)\)", body), (
        "header click handler must call setSort(th.getAttribute('data-sort'))"
    )


def test_compare_page_html_sort_indicator_glyphs() -> None:
    """sortInd() returns ▾ for inactive columns, ▲ for asc, ▼ for desc."""
    script = _compare_page_script()
    body = _function_body(script, "sortInd")
    # Inactive column glyph: ▾ (default)
    assert "▾" in body, "sortInd() must return ▾ for inactive columns"
    # Active asc: ▲
    assert "▲" in body, "sortInd() must return ▲ for active asc"
    # Active desc: ▼
    assert "▼" in body, "sortInd() must return ▼ for active desc"


def test_compare_page_html_pager_buttons_have_disabled_boundary_logic() -> None:
    """Prev/Next are disabled at the boundaries (page 1 / last page) and
    click handlers drive the page state."""
    script = _compare_page_script()
    body = _function_body(script, "renderPicker")
    # Prev disabled when on the first page; Next disabled on the last page.
    assert re.search(r"prev\.disabled\s*=\s*page\s*<=\s*1", body), (
        "Prev button must be disabled on page 1 (boundary)."
    )
    assert re.search(r"next\.disabled\s*=\s*page\s*>=\s*totalPages", body), (
        "Next button must be disabled on the last page (boundary)."
    )
    # Click handlers decrement / increment page.
    assert re.search(r"page--", body) and re.search(r"page\+\+", body), (
        "Prev/Next click handlers must decrement/increment page."
    )


def test_compare_page_html_data_sort_wired_from_pick_cols_only() -> None:
    """Whitelist guardrail (S04 polish): the only data-sort values that
    can ever reach setSort() are the hard-coded PICK_COLS keys.  The
    header construction iterates PICK_COLS and writes ``c.key`` into
    the data-sort attribute; the click handler reads the value back via
    ``th.getAttribute("data-sort")`` and passes it straight to setSort().

    This test pins that contract structurally so a future change that
    makes PICK_COLS data-driven cannot silently widen the set of
    keys the click handler will accept.
    """
    script = _compare_page_script()
    body = _function_body(script, "renderPicker")

    # 1. The header construction must iterate PICK_COLS (not user input).
    assert "PICK_COLS.forEach" in body, (
        "renderPicker() must iterate PICK_COLS to build the sortable headers "
        "(data-sort whitelist guardrail: PICK_COLS is the source of truth)."
    )

    # 2. The data-sort attribute value must be derived from c.key only.
    #    The pattern is `data-sort="' + c.key + '"` (or the single-quoted
    #    equivalent).  Assert that nothing else writes to data-sort.
    assert re.search(r"data-sort=\"'\s*\+\s*c\.key\s*\+\s*'\"", body) or re.search(
        r"data-sort='\s*'\s*\+\s*c\.key\s*\+\s*\"'", body
    ), "renderPicker() must derive data-sort strictly from c.key (whitelist guardrail)."

    # 3. The click handler must read the value back via getAttribute and
    #    pass it straight to setSort() — no string interpolation, no
    #    eval, no other transformation that could widen the accepted set.
    assert re.search(r"setSort\(th\.getAttribute\(\"data-sort\"\)\)", body), (
        "renderPicker() must read data-sort via getAttribute and pass it "
        "to setSort() unchanged (whitelist guardrail)."
    )

    # 4. The PICK_COLS keys form the whitelist.  If anyone ever introduces
    #    a literal data-sort value in the source (e.g. via a future bug
    #    or a data-driven PICK_COLS), it MUST be one of these keys.
    pck_match = re.search(r"var\s+PICK_COLS\s*=\s*\[(.*?)\];", script, re.S)
    assert pck_match, "PICK_COLS declaration missing"
    whitelist = set(re.findall(r"key\s*:\s*\"([^\"]+)\"", pck_match.group(1)))
    assert whitelist == {"run", "dataset", "config", "created_at"}, (
        f"PICK_COLS whitelist drifted: {whitelist!r}"
    )

    # 5. Any literal data-sort value that exists in the script body
    #    (rare — usually the value is interpolated) must be either an
    #    opening/closing quote fragment or a whitelist member.  This is
    #    the catch-all: if a contributor accidentally writes
    #    ``data-sort="hardcoded_user_input"``, this test fires.
    for match in re.finditer(r"data-sort=(['\"])([^'\"]*?)\1", script):
        literal = match.group(2)
        if literal in {"'", '"', ""}:
            # Opening/closing fragment of a string-concat template
            # (the script writes ``data-sort="' + c.key + '"`` and only
            # the opening/closing quotes appear as literals).
            continue
        assert literal in whitelist, (
            f"Literal data-sort={match.group(0)!r} found in source but "
            f"{literal!r} is not in the PICK_COLS whitelist ({sorted(whitelist)})."
        )
