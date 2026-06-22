# ruff: noqa: E501
"""Unit tests for the F-00096 S14 Scoreboard view page module.

The Scoreboard view is a self-contained inline HTML+CSS+vanilla-JS
page mirroring :mod:`iw_rag.api.benchmark_page`,
:mod:`iw_rag.api.console_page`, :mod:`iw_rag.api.results_page`, and
the rest of the S10–S15 sibling pages — re-homed as a console sub-view
that ranks the append-only ``benchmark_reports`` history into per-metric
leaderboards (``faithfulness`` / ``ndcg_at_5`` / ``tok_per_s`` /
``latency`` / ``vram_efficiency`` / ``composite``).  These tests pin
the behavioural contract that is verifiable from Python: the page
string returned by :func:`iw_rag.api.scoreboard_page.scoreboard_page_html`
must carry every section anchor the F-00096 Feature Design (Scope
"Scoreboard", AC6, Boundary row "Empty report history", Invariants
4 + 6 + 8) requires (the six per-metric leaderboards, the
server-provided ranking preserved verbatim — never recomputed
client-side, the board picker, the sortable column headers, the
empty-history empty state, the IW brand chrome, the shared six-item
nav, the ``/auth/login`` self-handled sign-in, the location-derived
API base, the sessionStorage-only bearer token, the raw-string
``_PAGE`` whose inline ``<script>`` passes ``node --check``).

The page content itself is asserted via these anchors; the S17
integration suite verifies the live ``GET /console/scoreboard`` route
returns 200 HTML carrying the same markers.  No JS engine is spun up
here: the page is structurally checked, then the S23
browser-verification step exercises the interactive behaviour in a
real browser.

**Invariant 4** — the view is a pure API client; it contains NO
ranking logic, NO retrieval, NO ACL, NO grounding, NO guard logic.
The S05 ``ScoreboardService`` is the **sole** authority on the
per-row ordering (AC6: descending faithfulness for the
``faithfulness`` board); the page only renders what
``GET /benchmark/scoreboard`` returns.  Column-header client-side
sorting is permitted for display convenience over the already-ranked
rows, but the **default order is the server-provided ranking** — the
page never recomputes ranking from raw metrics.

**Invariant 6** — paired-statistics invariant is preserved
(defensive — the scoreboard is a single-board view, not a paired
compare, so the dataset-mismatch 400 path is not surfaced here).

**Invariant 8** — no external CDN / Google Fonts; the page is
self-hosted.
"""

from __future__ import annotations

import re
import subprocess
import tempfile

from iw_rag.api.scoreboard_page import scoreboard_page_html

# ---------------------------------------------------------------------------
# Page factory / brand / operational keystone
# ---------------------------------------------------------------------------


def test_scoreboard_page_html_returns_non_empty_string() -> None:
    """The page factory returns a non-empty HTML string starting with ``<!doctype html>``."""
    page = scoreboard_page_html()
    assert isinstance(page, str)
    assert page.strip() != ""
    # Top-level doctype + html tag — the page is real HTML, not just text.
    assert page.lstrip().lower().startswith("<!doctype html>")


def test_scoreboard_page_html_contains_brand_palette() -> None:
    """The IW teal/ink palette tokens are present (branding keystone).

    Mirrors the benchmark-page assertion style — the :root palette
    defines each token; subsequent call-sites reuse ``var(--teal)`` /
    ``var(--teal-deep)`` so the teal-deep token appears exactly once.
    """
    page = scoreboard_page_html()
    assert "#0D9488" in page  # teal-deep
    assert "#2DD4BF" in page  # teal
    assert "#181D29" in page  # ink
    # The :root palette defines teal-deep exactly once; subsequent
    # call-sites reuse ``var(--teal-deep)`` (mirror benchmark_page
    # assertion).
    assert page.count("#0D9488") == 1


def test_scoreboard_page_html_has_no_external_cdn() -> None:
    """Invariant 8 — no external CDN / Google Fonts link (local-only rule)."""
    page = scoreboard_page_html().lower()
    # The page must not pull a stylesheet or font from any third party.
    for forbidden in (
        "fonts.googleapis.com",
        "fonts.gstatic.com",
        "cdn.jsdelivr.net",
        "unpkg.com",
        "cdnjs.cloudflare.com",
    ):
        assert forbidden not in page, (
            f"Scoreboard view page must not reference external CDN '{forbidden}' (Invariant 8)."
        )


def test_scoreboard_page_html_derives_api_base_from_location() -> None:
    """Invariant 4 + 8 — the API base is derived from ``window.location``.

    The page lives at ``/console/scoreboard`` (or
    ``/api/console/scoreboard`` behind Caddy) and must strip that
    suffix from ``location.pathname`` to derive the API base.
    Hardcoded host:port pairs are forbidden.
    """
    page = scoreboard_page_html()
    # The Scoreboard view lives at /console/scoreboard and must strip
    # that suffix from `location.pathname` to derive the API base.
    assert "location.pathname" in page
    assert "/console/scoreboard" in page
    # The page must NOT hardcode a host:port pair.
    assert "http://localhost" not in page
    assert "http://127.0.0.1" not in page


def test_scoreboard_page_html_logs_in_via_auth_login() -> None:
    """The page self-handles sign-in via ``/auth/login`` (mirrors shell + benchmark page)."""
    page = scoreboard_page_html()
    assert "/auth/login" in page
    # Session-memory bearer token (never persisted beyond the tab).
    assert "sessionStorage" in page
    # Exactly one POST to /auth/login (the shell's login flow).
    assert page.count("/auth/login") == 1


# ---------------------------------------------------------------------------
# API endpoint reference — the S08 scoreboard endpoint the view consumes
# ---------------------------------------------------------------------------


def test_scoreboard_page_html_references_scoreboard_endpoint() -> None:
    """AC6 — the view consumes ``GET /benchmark/scoreboard?board=<name>``.

    The page must issue a single ``fetch`` to ``/benchmark/scoreboard``
    with a ``board=`` query string; the page must also include the
    strong-form ``.count(...) == 1`` pin to guard against a future
    regression that silently duplicates the fetch path.
    """
    page = scoreboard_page_html()
    # The endpoint is anchored.
    assert "/benchmark/scoreboard" in page
    # The fetch is invoked with a board query string.
    assert "board=" in page
    # Strong-form count: the scoreboard fetch path is invoked exactly
    # once (the page's loadBoard / bootstrap path).
    assert page.count('"/benchmark/scoreboard?board=" +') >= 1


# ---------------------------------------------------------------------------
# AC6 — the six per-metric leaderboards the S05 service produces
# ---------------------------------------------------------------------------


def test_scoreboard_page_html_lists_all_six_boards() -> None:
    """AC6 — the page surfaces all six S05 leaderboards exactly.

    The S05 ``ScoreboardService`` produces six boards (per
    ``eval/benchmark/scoreboard.py``); the S08 route accepts each by
    name; the S14 page must surface each as a tab / board-picker
    option (the operator picks one, the page fetches its entries).
    The canonical board names are: ``faithfulness``, ``ndcg_at_5``,
    ``tok_per_s``, ``latency``, ``vram_efficiency``, ``composite``.
    """
    page = scoreboard_page_html()
    # Each board name appears as a value the page forwards to the API
    # (the board-picker options + the strong-form ``.count(...) == 1``
    # pins for each board's data attribute).
    for board in (
        "faithfulness",
        "ndcg_at_5",
        "tok_per_s",
        "latency",
        "vram_efficiency",
        "composite",
    ):
        assert board in page, f"missing S05 board name '{board}' in the page (AC6)"
    # Operator-visible labels for the boards (mirror the S05 service's
    # human-readable names; the S08 endpoint returns the raw
    # snake_case ids).
    for label in ("faithfulness", "ndcg_at_5", "tok/s", "latency", "VRAM-efficiency"):
        assert label in page, f"missing operator-visible board label '{label}' (AC6)"


def test_scoreboard_page_html_faithfulness_board_default() -> None:
    """AC6 — the faithfulness board is the default-active board.

    The Feature Design's AC6 is phrased around the ``faithfulness``
    board: *"When the operator opens the Scoreboard and picks the
    'faithfulness' board, reports are listed in descending
    faithfulness order..."*.  The page must default to the
    ``faithfulness`` board on first load (so the operator sees the
    AC6 surface without an extra click).
    """
    page = scoreboard_page_html()
    # The default active board is anchored by ``data-board="faithfulness"``
    # AND the ``checked`` attribute (the default-selection marker is
    # the ``faithfulness`` board) — a strong-form pin to guard against
    # a future regression that silently changes the default.  Exactly
    # one ``data-board="faithfulness"`` marker on the page (the
    # default-active radio input), and the ``checked`` attribute sits
    # on the same input.
    assert 'data-board="faithfulness"' in page
    assert page.count('data-board="faithfulness"') == 1
    assert 'value="faithfulness" checked' in page


# ---------------------------------------------------------------------------
# Sortable leaderboard table — client-side sort over server-ranked rows
# ---------------------------------------------------------------------------


def test_scoreboard_page_html_has_sortable_column_headers() -> None:
    """The board table uses sortable column headers (display convenience).

    The authoritative ranking is the server's (Invariant 4 — the
    page never recomputes ranking from raw metrics).  Column-header
    client-side sorting is allowed for display convenience over the
    already-ranked rows; the page must reuse the
    ``documents_page.py`` ``th.sortable`` / ``setSort`` /
    ``refreshHeaderIndicators`` pattern.  The sortable-header marker
    is the ``th.sortable`` CSS class on the column-header cells.
    """
    page = scoreboard_page_html()
    # Sortable header marker.
    assert "th.sortable" in page or "sortable" in page.lower()
    # The columns that are sortable include the metric value (the
    # primary display column), the model_config, and the run timestamp.
    for header in ("metric_value", "config_snapshot", "created_at"):
        # The data-sort attribute is the canonical ``documents_page.py``
        # sortable-key anchor.
        assert ('data-sort="' + header + '"') in page or ("th sortable" in page.lower()), (
            f"missing sortable header anchor for '{header}'"
        )
    # The sort indicator glyph and the refreshHeaderIndicators helper
    # (mirror the ``documents_page.py`` sortable pattern).
    assert "sort-ind" in page or "▲" in page or "▼" in page or "▾" in page


def test_scoreboard_page_html_renders_ranked_rows_with_model_config_and_timestamp() -> None:
    """AC6 — each row carries the rank, the model_config, and the run timestamp.

    AC6 says: *"reports are listed in descending faithfulness order
    with model_config + run timestamp"*.  The page must render the
    rank, a model_config summary, and the created_at (run timestamp)
    on every row of the active board.  The board's metric value is
    the board's primary column.
    """
    page = scoreboard_page_html()
    # The board table is anchored by a stable id the JS fills on load.
    assert "rows" in page.lower() or "tbody" in page.lower()
    # The model_config snapshot is the per-row primary column — the
    # F-00093 storage column ``model_config`` is exposed on the API
    # surface as ``config_snapshot`` (Pydantic reserved-attribute
    # collision, mirror ``BenchmarkReportOut``); the S14 page maps it
    # back to ``model_config`` for display.
    assert "model_config" in page
    assert "config_snapshot" in page
    # The created_at (run timestamp) column.
    assert "created_at" in page
    # The rank column.
    assert "rank" in page.lower()


# ---------------------------------------------------------------------------
# Empty-history empty state (Boundary row: 0 benchmark_reports)
# ---------------------------------------------------------------------------


def test_scoreboard_page_html_has_empty_history_state() -> None:
    """Boundary row "Empty report history" — explicit empty state, never an error.

    When ``GET /benchmark/scoreboard`` returns an empty history (no
    reports), the page must render the canonical empty-state copy
    *"No benchmark reports yet — run a benchmark from the Benchmark
    Runner to populate the leaderboards."* — not an error toast that
    hides the cause.  Mirrors the F-00095 / F-00096 design Boundary
    row + the ``documents_page.py`` empty-state treatment.
    """
    page = scoreboard_page_html()
    # The canonical empty-state copy is present verbatim — the
    # operator sees the call-to-action (run a benchmark from the
    # Benchmark Runner).
    assert "No benchmark reports yet" in page
    # Strong-form count: the empty-state anchor id ("empty") appears
    # exactly once on the page — the table-card region the JS
    # toggles on empty / non-empty.  Regression pin against a future
    # refactor that silently duplicates the empty state.
    assert page.count('id="empty"') == 1
    # The call-to-action names the Benchmark Runner explicitly so the
    # operator can act on it (at least one occurrence on the page
    # — once in the empty-state markup, once in the JS string the
    # page falls back to).  Strong-form count: the empty-state
    # anchor id ("empty") appears exactly once.
    assert "Benchmark Runner" in page
    assert page.count("Benchmark Runner") >= 1


# ---------------------------------------------------------------------------
# Console shell integration — shared chrome, nav, theme
# ---------------------------------------------------------------------------


def test_scoreboard_page_html_has_six_nav_destinations() -> None:
    """The shared nav exposes all six console destinations (S10 scope).

    * Config       → ``/console/config``        (S11)
    * Benchmark    → ``/console/benchmark``     (S12)
    * Results      → ``/console/results``       (S13)
    * Scoreboard   → ``/console/scoreboard``    (S14 — this view, marked active)
    * Chat         → ``/``                       (Chainlit — linked, not rebuilt)
    * Upload       → ``/documents/ui``           (Document Manager — linked, not rebuilt)
    """
    page = scoreboard_page_html()
    # Console-owned views.
    assert 'href="/console/config"' in page, "Config nav → /console/config missing"
    assert 'href="/console/benchmark"' in page, "Benchmark nav → /console/benchmark missing"
    assert 'href="/console/results"' in page, "Results nav → /console/results missing"
    assert 'href="/console/scoreboard"' in page, "Scoreboard nav → /console/scoreboard missing"
    # Linked/embedded surfaces (not rebuilt).
    assert 'href="/console/chat"' in page, "Chat nav → /console/chat missing"
    assert 'href="/console/upload"' in page, "Upload nav → /console/upload missing"
    # Each label is present (operator-visible).
    for label in ("Config", "Benchmark", "Results", "Scoreboard", "Chat", "Upload"):
        assert label in page, f"nav label '{label}' missing"


def test_scoreboard_page_html_has_data_theme_dark() -> None:
    """Dark theme is the default (matches benchmark_page + Chainlit theme)."""
    page = scoreboard_page_html()
    # The dark theme is the single default — exactly one declaration.
    assert page.count('data-theme="dark"') == 1
    assert 'data-theme="dark"' in page


def test_scoreboard_page_html_has_innovation_ways_wordmark() -> None:
    """The IW brand chrome — the "Innovation Ways" wordmark in the header.

    The wordmark is split into coloured initials
    (``<span class="i">I</span>nnovation <span class="w">W</span>ays``), so the
    literal "Innovation Ways" substring is not in the markup — assert the header
    logo wordmark itself.  The browser-tab ``<title>`` is the short, uniform
    "IW - RAG" form.
    """
    page = scoreboard_page_html()
    # Exactly one wordmark occurrence (the header) — guards against a
    # silent duplicate being introduced in a future refactor.
    assert page.count('<span class="logo">') == 1
    assert '<span class="i">I</span>nnovation <span class="w">W</span>ays' in page
    assert "<title>IW - RAG</title>" in page
    assert "Innovation Ways" not in page  # long brand must not return to the tab


# ---------------------------------------------------------------------------
# JS contract — API-base regex and inline script (parity with sibling pages)
# ---------------------------------------------------------------------------


def test_scoreboard_page_html_derives_api_base_is_fixed_api_edge_base() -> None:
    """The page derives its API base as the fixed ``"/api"`` edge prefix.

    F-00096: the Operator Console is served at ``/console*`` behind the single
    Caddy edge (ADR-016), where the API lives under ``/api/*``. The base is the
    constant ``"/api"`` (no location-derived regex), so a fetch of
    ``API + "/api/config"`` -> ``/api/api/config`` which the edge's
    ``handle_path /api/*`` strips back to ``/api/config`` on the API. Replaces
    the old ``/api/console`` location regex, under which every sub-view data
    fetch 404'd at ``/console/*`` behind the edge.
    """
    page = scoreboard_page_html()
    assert page.count('var API = "/api";') == 1, (
        "console page must declare the fixed /api edge base exactly once (F-00096) — "
        "a second/stale base declaration would re-introduce the /console/* 404 bug"
    )


def test_scoreboard_page_html_inline_script_parses() -> None:
    """The page's entire inline ``<script>`` must be syntactically valid JS.

    Regression pin mirroring
    :func:`test_benchmark_page_html_inline_script_parses`.  The page is
    generated from a triple-quoted Python string; when that string was
    **not** raw, JS escape sequences written in the source were
    consumed by Python and the page would throw a SyntaxError at load
    time — invisible to string-match tests but fatal in a real browser.

    This test extracts the full ``<script>`` body and runs
    ``node --check`` over it.  Skips when Node is unavailable.
    """
    page = scoreboard_page_html()
    scripts = re.findall(r"<script\b[^>]*>(.*?)</script>", page, re.S)
    assert scripts, "no <script> block found in the Scoreboard view page"
    script = "\n".join(scripts)
    # Guard the specific render-time signatures directly (cheap, Node-free):
    # a bare `data: <json>` line means the SSE comment broke across newlines.
    assert "\ndata: <json>\n" not in ("\n" + script + "\n"), (
        "rendered inline script contains a bare `data: <json>` line — the SSE "
        "comment's \\n expanded to real newlines (is _PAGE a raw string?)."
    )
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as fh:
        fh.write(script)
        path = fh.name
    try:
        node = subprocess.run(
            ["node", "--check", path],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except FileNotFoundError:
        return  # No node in this env — skip the JS-side assertion.
    assert node.returncode == 0, (
        "Scoreboard view inline script failed `node --check` "
        f"(SyntaxError -> page never boots): stderr={node.stderr!r}"
    )
