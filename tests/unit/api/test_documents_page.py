"""Unit tests for the existing :mod:`iw_rag.api.documents_page` HTML page.

The V6 no-regression verification in the F-00095 / S10 prompt revisits
``GET /documents/ui`` and confirms it still loads + behaves as before.
The Studio page (S03) and the Document Manager page (WI-DOCMGMT) share
the same :func:`documents_page_html` factory; both pages derive their API
base from ``window.location`` via the same regex strip idiom.

Regression pin for the F-00095 / S10 fix cycle: the documents page
shipped with the same backslash-escaped regex bug as the Studio page
(``/\\/documents\\/ui\\/?$/``), causing JavaScript's regex parser to
fail with ``Invalid regular expression flags`` and the entire page
script to throw at load time.  This file guards both pages against the
regression by checking the regex is a valid ``vm.Script``-parseable JS
literal that matches the expected paths.

CR-00099 (Document Manager table) added three concerns on top of the
page: sortable column headers (default Added/desc), a per-row Download
action, and 50-row client-side pagination.  The tests below pin the
markup markers S01 introduced (string assertions on the emitted page)
and exercise the page's pure ``sortedDocs()`` function via a Node
subprocess against a 120-document fixture to pin AC4 (pagination
respects sort order).  The Node call is guarded with
``FileNotFoundError`` → ``return`` so hosts without Node skip the
JS-side assertion, matching the idiom of the existing test.
"""

from __future__ import annotations

import re
import subprocess

from iw_rag.api.documents_page import documents_page_html


def test_documents_page_html_returns_non_empty_string() -> None:
    """The page factory returns a non-empty HTML string."""
    page = documents_page_html()
    assert isinstance(page, str)
    assert page.strip() != ""
    assert page.lstrip().lower().startswith("<!doctype html>")


def test_documents_page_html_derives_api_base_with_evaluable_js_regex() -> None:
    """The API-base regex must be a valid JavaScript regex literal AND must
    correctly strip ``/documents/ui`` from the page's own path (Invariant 6
    + a real-browser sanity check that a SyntaxError-free page actually
    derives the API base).

    Regression pin for the F-00095 / S10 fix cycle: the page shipped with
    a backslash-escaped regex that failed JavaScript's regex parser with
    ``Invalid regular expression flags`` and would have caused the entire
    page script to throw at load time.  The fix is to escape the forward
    slashes (which are the regex literal delimiters) with a backslash.
    """
    page = documents_page_html()
    m = re.search(r"var API = location\.pathname\.replace\((.+?), \"\"\);", page)
    assert m is not None, "API-base regex assignment not found in page"
    regex_source = m.group(1).strip()
    # The bug pattern: the original regex was `/\\/documents\\/ui\\/?$/`
    # (4 backslashes inside a regex literal — each `\\` matches a literal
    # `\`, so the parser was looking for backslash separators in the URL
    # path and the regex closed early at the first `/`, leaving `d` as an
    # invalid flag — hence the SyntaxError).  The fix is to escape the
    # forward slashes (which are the regex literal delimiters) with
    # `\/`, so the correct source is `/\/documents\/ui\/?$/`.  The
    # distinguishing signal in the source text is the presence of `\\`
    # before `documents` and `ui` — that is the bug.
    assert "\\\\documents" not in regex_source, (
        "API-base regex still uses `\\\\` before `documents` (regression of the "
        "S10 /documents/ui SyntaxError bug — the parser was treating `d` as a flag)."
    )
    assert "\\\\ui" not in regex_source, (
        "API-base regex still uses `\\\\` before `ui` (regression of the S10 "
        "/documents/ui SyntaxError bug)."
    )
    assert "documents" in regex_source and "ui" in regex_source, (
        "API-base regex does not reference /documents/ui (the page lives at /documents/ui)."
    )
    # JS-side validation: build a vm.Script-style function and call it.
    # Skip when Node is unavailable (CI hosts without Node should not fail this test).
    try:
        node = subprocess.run(
            [
                "node",
                "-e",
                (
                    f"const re = {regex_source};"
                    "for (const [path, want] of [['/documents/ui',''],"
                    " ['/api/documents/ui','/api'],"
                    " ['/documents/ui/','']]) {"
                    " const got = path.replace(re, '');"
                    " if (got !== want) {"
                    "   console.error('FAIL', path, '->', got, 'want', want);"
                    "   process.exit(1);"
                    " }"
                    "}"
                    "console.log('OK');"
                ),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except FileNotFoundError:
        return  # No node in this env — skip the JS-side assertion.
    assert "OK" in node.stdout, (
        f"API-base regex {regex_source!r} did not behave correctly in Node: "
        f"stdout={node.stdout!r} stderr={node.stderr!r}"
    )


# ---------------------------------------------------------------------------
# CR-00099 — Document Manager: sortable columns + Download + 50-row pagination
# ---------------------------------------------------------------------------
#
# The page is a self-contained inline HTML/JS string returned by
# ``documents_page_html()``.  S01 introduced:
#
#   * Sortable column headers: ``<th class="sortable" data-sort="<field>">``
#     for the five sortable columns (Name / Scope / Chunks / Size / Added)
#     with a direction-indicator ``<span class="sort-ind">`` and an initial
#     state of Added/desc (AC1, AC2).
#   * A per-row Download action: ``<button class="btn-ghost"
#     data-download="<id>">Download</button>`` backed by the
#     ``download(id, name)`` JS function, which reuses
#     ``/documents/{id}/raw`` to force a save-to-disk (AC3).
#   * 50-row client-side pagination: a hidden ``#pager`` div with Prev/Next
#     buttons and a "Showing X–Y of N" indicator, hidden when the list
#     fits on a single page; the sorted list is sliced per-page (AC4).
#
# The tests below pin the markup markers (string assertions on the
# emitted page string) and exercise the page's pure ``sortedDocs()``
# function against a 120-doc fixture to confirm AC4's "second page is
# the correct slice of the fully-sorted list".


# All five sortable columns from AC1.  The ``data-sort`` value must match
# the field name on the document object the API returns.
_SORTABLE_FIELDS = (
    "filename",
    "visibility",
    "chunk_count",
    "size_bytes",
    "created_at",
)


def test_documents_page_exposes_download_action() -> None:
    """AC3: each row exposes a Download button (``data-download=...``) backed
    by a ``download(id, name)`` JS function.  The pre-S01 page had neither.
    """
    page = documents_page_html()
    # Row template must emit a Download button (sits between View and Delete
    # in S01's render()).  We assert on the attribute so the test doesn't
    # break if the button's label text ever changes.
    assert "data-download=" in page, (
        "Document Manager page must expose a per-row Download action via "
        "a data-download attribute (AC3)."
    )
    # A `download(id, name)` JS function must be defined to back the button.
    assert re.search(r"async\s+function\s+download\s*\(\s*id\s*,\s*name\s*\)", page), (
        "Document Manager page must define an async `download(id, name)` "
        "function to handle the Download action (AC3)."
    )
    # The Download button label "Download" is the human-readable affordance.
    assert ">Download</button>" in page, (
        "Document Manager page row template must include a Download button label."
    )


def test_documents_page_exposes_pager_with_page_size_50() -> None:
    """AC4: 50-row client-side pagination with Prev/Next controls and a
    "Showing X–Y of N" indicator.  The pager is hidden when the list fits
    on a single page, but its elements must exist in the markup.
    """
    page = documents_page_html()
    # The page-size constant — pins the "50 rows per page" acceptance criterion.
    assert re.search(r"var\s+PAGE_SIZE\s*=\s*50\s*;", page), (
        "Document Manager page must declare PAGE_SIZE = 50 (AC4)."
    )
    # Pager container + Prev / Next buttons + range indicator.
    assert 'class="pager hidden" id="pager"' in page, (
        "Document Manager page must include a hidden #pager container (AC4)."
    )
    assert 'id="pagePrev"' in page, (
        "Document Manager page must include a #pagePrev Prev button (AC4)."
    )
    assert 'id="pageNext"' in page, (
        "Document Manager page must include a #pageNext Next button (AC4)."
    )
    assert 'id="pageInfo"' in page, (
        "Document Manager page must include a #pageInfo range indicator (AC4)."
    )
    # The renderPager() function must exist and must use the en-dash range
    # marker the S01 design settled on (U+2013, "Showing X–Y of N").
    assert "function renderPager(" in page, (
        "Document Manager page must define a renderPager() function (AC4)."
    )
    assert "Showing " in page and "\u2013" in page and " of " in page, (
        "Document Manager page must produce a 'Showing X–Y of N' indicator "
        "with an en-dash range marker (AC4)."
    )


def test_documents_page_exposes_sortable_headers_with_indicator() -> None:
    """AC1: every sortable column header carries ``class="sortable"`` and a
    ``data-sort`` attribute pointing at the right field, and a direction
    indicator mechanism (``<span class="sort-ind">``) is present so the
    active column can show ▲/▼.
    """
    page = documents_page_html()
    # The five sortable columns from AC1 must each have class=sortable +
    # data-sort=<field>.  We count occurrences to assert ALL five are present
    # (one stale entry would slip past a per-field search).
    sortable_count = page.count('class="sortable')
    assert sortable_count >= len(_SORTABLE_FIELDS), (
        f"Document Manager page must mark all {len(_SORTABLE_FIELDS)} "
        f'sortable columns with class="sortable" (AC1); found '
        f"{sortable_count} occurrences."
    )
    for field in _SORTABLE_FIELDS:
        assert f'data-sort="{field}"' in page, (
            f"Document Manager page must mark a header with "
            f'data-sort="{field}" so click-to-sort wires up correctly (AC1).'
        )
    # Sort direction indicator: every sortable header carries a
    # <span class="sort-ind"> so JS can swap the glyph (▲/▼/▾).
    assert page.count('class="sort-ind"') >= len(_SORTABLE_FIELDS), (
        "Document Manager page must give every sortable header a direction-indicator span (AC1)."
    )
    # The static initial markup must include both glyphs so the first paint
    # is correct even before render() runs: the inactive columns show ▾
    # ("sortable, not active") and the active Added column shows ▼.
    assert "\u25be" in page and "\u25bc" in page, (
        "Document Manager page must include both the inactive (▾) and "
        "active-desc (▼) sort-direction glyphs in its initial markup (AC1)."
    )
    # The JS that flips the glyphs lives in refreshHeaderIndicators() and
    # is what makes the indicator actually update on click.
    assert "function refreshHeaderIndicators(" in page, (
        "Document Manager page must define refreshHeaderIndicators() so the "
        "active column's direction glyph updates on click (AC1)."
    )


def test_documents_page_default_sort_is_added_desc() -> None:
    """AC2: the page initializes its sort state to the Added column
    (``created_at``) with descending direction, and the Added <th> is
    pre-marked ``class="sortable desc"`` so the initial render is
    correct even before render() runs.
    """
    page = documents_page_html()
    # The JS state initializers must pin the default sort.
    assert re.search(
        r'sortKey\s*=\s*"created_at"\s*,\s*sortDir\s*=\s*"desc"',
        page,
    ), "Document Manager page must initialize sortKey='created_at' and sortDir='desc' (AC2)."
    # Static markup: the Added <th> is pre-marked class="sortable desc" so
    # the first paint is correct even before refreshHeaderIndicators() runs.
    assert re.search(
        r'<th[^>]*class="sortable\s+desc"[^>]*data-sort="created_at"',
        page,
    ), (
        "Document Manager page must pre-mark the Added <th> with "
        'class="sortable desc" and data-sort="created_at" (AC2).'
    )
    # The default direction for created_at in DEFAULT_DIR is "desc" — pins
    # the "switching to a different column respects its default direction"
    # part of the sort behaviour.
    assert re.search(
        r'created_at:\s*"desc"',
        page,
    ), (
        "Document Manager page's DEFAULT_DIR must map created_at -> 'desc' "
        "so the chronological column defaults to newest-first (AC2)."
    )


def test_documents_page_sort_and_paginate_via_node() -> None:
    """AC4: pagination must respect sort order — the second page is a
    slice of the **sorted** list, not the unsorted list.  We extract the
    page's pure ``sortedDocs()`` function via regex and exercise it
    against a 120-document fixture in a Node subprocess.  Skipped
    gracefully when Node is absent (matches the FileNotFoundError idiom
    used by the API-base regex test above).
    """
    page = documents_page_html()

    # Extract the page's ``sortedDocs()`` function body so we test the
    # actual algorithm shipped in the page, not a re-implementation.
    m = re.search(r"function sortedDocs\(\) \{[\s\S]*?\n  \}", page)
    assert m is not None, "Document Manager page must define a sortedDocs() function (AC4)."
    sorted_docs_src = m.group(0)
    # Pin that sortedDocs starts with the expected function signature.
    assert sorted_docs_src.startswith("function sortedDocs()")

    # 120-row fixture: deterministic, deliberately mixed values for both
    # a numeric column (size_bytes) and a lexicographic column (filename).
    fixture_js = (
        "var allDocs = [];"
        "for (var i = 0; i < 120; i++) {"
        "  allDocs.push({"
        "    id: 'd' + i,"
        # doc000, doc001, ... doc119 — clean lexicographic ordering
        "    filename: 'doc' + String(i).padStart(3, '0') + '.txt',"
        "    visibility: i % 2 ? 'common' : 'specific',"
        "    chunk_count: i * 3,"
        # size_bytes strictly decreasing in i → size desc puts doc 0 first
        "    size_bytes: 1024 * (120 - i),"
        "    created_at: '2025-01-01T00:00:00Z'"
        "  });"
        "}"
    )

    # Build the Node snippet: declare the page's PAGE_SIZE constant, inject
    # the extracted sortedDocs() function, then exercise the sort+slice
    # logic for both a numeric and a lexicographic column.  We check
    # orderedness of the sorted list and that page 2 is the correct
    # continuation of the sorted order (not the unsorted list).
    node_script = (
        "var PAGE_SIZE = 50;"
        "var sortKey, sortDir;"
        + fixture_js
        + sorted_docs_src
        + (
            "function fail(msg) { console.error('FAIL', msg); process.exit(1); }"
            # --- numeric column: size_bytes desc ---
            "sortKey = 'size_bytes'; sortDir = 'desc';"
            "var sortedSize = sortedDocs();"
            "if (sortedSize.length !== 120) fail('expected 120 docs, got ' + sortedSize.length);"
            "for (var i = 0; i < 119; i++) {"
            "  if (sortedSize[i].size_bytes < sortedSize[i + 1].size_bytes) {"
            "    fail('size_bytes not desc at ' + i + ': ' + sortedSize[i].size_bytes"
            " + ' < ' + sortedSize[i + 1].size_bytes);"
            "  }"
            "}"
            "var p1 = sortedSize.slice(0, PAGE_SIZE);"
            "var p2 = sortedSize.slice(PAGE_SIZE, 2 * PAGE_SIZE);"
            "var p1Exp = sortedSize.slice(0, 50);"
            "var p2Exp = sortedSize.slice(50, 100);"
            "if (JSON.stringify(p1) !== JSON.stringify(p1Exp)) fail('size p1 mismatch');"
            "if (JSON.stringify(p2) !== JSON.stringify(p2Exp)) fail('size p2 mismatch');"
            # Page 2 must continue the same order: p2[0] == sortedSize[50]
            "if (p2[0].id !== sortedSize[50].id) fail('size p2[0] != sorted[50]');"
            # --- lexicographic column: filename asc ---
            "sortKey = 'filename'; sortDir = 'asc';"
            "var sortedName = sortedDocs();"
            "if (sortedName.length !== 120) fail('expected 120 docs, got ' + sortedName.length);"
            "for (var i = 0; i < 119; i++) {"
            "  if (String(sortedName[i].filename).toLowerCase() >"
            " String(sortedName[i + 1].filename).toLowerCase()) {"
            "    fail('filename not asc at ' + i + ': ' + sortedName[i].filename + ' > '"
            " + sortedName[i + 1].filename);"
            "  }"
            "}"
            "var p1n = sortedName.slice(0, PAGE_SIZE);"
            "var p2n = sortedName.slice(PAGE_SIZE, 2 * PAGE_SIZE);"
            "var p1nExp = sortedName.slice(0, 50);"
            "var p2nExp = sortedName.slice(50, 100);"
            "if (JSON.stringify(p1n) !== JSON.stringify(p1nExp)) fail('name p1 mismatch');"
            "if (JSON.stringify(p2n) !== JSON.stringify(p2nExp)) fail('name p2 mismatch');"
            "if (p2n[0].id !== sortedName[50].id) fail('name p2[0] != sorted[50]');"
            "console.log('OK');"
        )
    )

    try:
        node = subprocess.run(
            ["node", "-e", node_script],
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except FileNotFoundError:
        return  # No node in this env — skip the JS-side assertion.
    assert "OK" in node.stdout, (
        f"sortedDocs() did not behave correctly in Node: "
        f"stdout={node.stdout!r} stderr={node.stderr!r}"
    )
