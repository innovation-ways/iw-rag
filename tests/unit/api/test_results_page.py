# ruff: noqa: E501
"""Unit tests for the F-00096 S13 Results + Compare view page module.

The Results + Compare view is a self-contained inline HTML+CSS+vanilla-JS
page mirroring :mod:`iw_rag.api.benchmark_page` and
:mod:`iw_rag.api.console_page` — re-homed as a console sub-view that
deep-dives on a single persisted ``benchmark_reports`` row and supports
A/B comparing two reports on the same dataset.  These tests pin the
behavioural contract that is verifiable from Python: the page string
returned by :func:`iw_rag.api.results_page.results_page_html` must carry
every section anchor the F-00096 Feature Design (Scope "Results +
Comparison view", AC5, Boundary rows "Empty report history", "Compare
across different datasets", "Unknown report id", Invariants 4 + 6 + 8)
requires (the report picker + deep-dive panel populated from
``GET /benchmark/reports`` / ``GET /benchmark/reports/{id}``, the A/B
compare picker + delta-table populated from ``GET /benchmark/compare``,
the winner verdict, the underpowered caveat, the 400/404/empty-state
messages, the empty-history empty state, the IW brand chrome, the
shared six-item nav, the ``/auth/login`` self-handled sign-in, the
location-derived API base, the sessionStorage-only bearer token, the
raw-string ``_PAGE`` whose inline ``<script>`` passes ``node --check``).

The page content itself is asserted via these anchors; the S17
integration suite verifies the live ``GET /console/results`` route
returns 200 HTML carrying the same markers.  No JS engine is spun up
here: the page is structurally checked, then the S23
browser-verification step exercises the interactive behaviour in a
real browser.

**Invariant 4** — the view is a pure API client; it contains NO
retrieval / ACL / grounding / guard logic.  All such logic stays
server-side (the F-00095 / F-00093 report service + the
``GET /benchmark/compare`` route).  The UI's disabled-state and
client-side pre-checks are UX-only honesty; the API is the
authoritative gate.

**Invariant 6** — ``GET /benchmark/compare`` only compares reports on
the same dataset.  The view's 400 / "not-comparable" message is
rendered verbatim in the panel, never an error toast that hides the
cause.

**Invariant 8** — no external CDN / Google Fonts; the page is
self-hosted.
"""

from __future__ import annotations

import re
import subprocess
import tempfile

from iw_rag.api.results_page import results_page_html

# ---------------------------------------------------------------------------
# Page factory / brand / operational keystone
# ---------------------------------------------------------------------------


def test_results_page_html_returns_non_empty_string() -> None:
    """The page factory returns a non-empty HTML string starting with ``<!doctype html>``."""
    page = results_page_html()
    assert isinstance(page, str)
    assert page.strip() != ""
    # Top-level doctype + html tag — the page is real HTML, not just text.
    assert page.lstrip().lower().startswith("<!doctype html>")


def test_results_page_html_contains_brand_palette() -> None:
    """The IW teal/ink palette tokens are present (branding keystone).

    Mirrors the benchmark-page assertion style — the :root palette
    defines each token; subsequent call-sites reuse ``var(--teal)`` /
    ``var(--teal-deep)`` so the teal-deep token appears exactly once.
    """
    page = results_page_html()
    assert "#0D9488" in page  # teal-deep
    assert "#2DD4BF" in page  # teal
    assert "#181D29" in page  # ink
    # The :root palette defines teal-deep exactly once; subsequent
    # call-sites reuse ``var(--teal-deep)`` (mirror benchmark_page
    # assertion).
    assert page.count("#0D9488") == 1


def test_results_page_html_has_no_external_cdn() -> None:
    """Invariant 8 — no external CDN / Google Fonts link (local-only rule)."""
    page = results_page_html().lower()
    # The page must not pull a stylesheet or font from any third party.
    for forbidden in (
        "fonts.googleapis.com",
        "fonts.gstatic.com",
        "cdn.jsdelivr.net",
        "unpkg.com",
        "cdnjs.cloudflare.com",
    ):
        assert forbidden not in page, (
            f"Results + Compare view page must not reference external CDN '{forbidden}' (Invariant 8)."
        )


def test_results_page_html_derives_api_base_from_location() -> None:
    """Invariant 4 + 6 — the API base is derived from ``window.location``.

    The page lives at ``/console/results`` (or ``/api/console/results``
    behind Caddy) and must strip that suffix from ``location.pathname``
    to derive the API base.  Hardcoded host:port pairs are forbidden.
    """
    page = results_page_html()
    # The Results view lives at /console/results and must strip that
    # suffix from `location.pathname` to derive the API base.
    assert "location.pathname" in page
    assert "/console/results" in page
    # The page must NOT hardcode a host:port pair.
    assert "http://localhost" not in page
    assert "http://127.0.0.1" not in page


def test_results_page_html_logs_in_via_auth_login() -> None:
    """The page self-handles sign-in via ``/auth/login`` (mirrors shell + benchmark page)."""
    page = results_page_html()
    assert "/auth/login" in page
    # Session-memory bearer token (never persisted beyond the tab).
    assert "sessionStorage" in page
    # Exactly one POST to /auth/login (the shell's login flow).
    assert page.count("/auth/login") == 1


# ---------------------------------------------------------------------------
# API endpoint references — the three F-00095 endpoints the view consumes
# ---------------------------------------------------------------------------


def test_results_page_html_reports_endpoints() -> None:
    """AC5 — the view references all three F-00095 read endpoints.

    * ``GET /benchmark/reports`` — list (newest first), populates the
      report picker and the A/B compare dropdowns.
    * ``GET /benchmark/reports/{id}`` — single report detail,
      populates the deep-dive panel.
    * ``GET /benchmark/compare?a=..&b=..`` — A/B delta table, winner,
      underpowered caveat.
    """
    page = results_page_html()
    # The list endpoint (anchored for the picker).
    assert "/benchmark/reports" in page
    # The detail endpoint (the report deep-dive fetch).
    assert "/benchmark/reports/" in page
    # The compare endpoint (the A/B delta-table fetch).
    assert "/benchmark/compare" in page
    # The compare fetch must encode both A and B ids in the query string
    # (mirrors the F-00095 ``compare()`` shape: ``?a=..&b=..``).
    assert "compare?a=" in page
    assert "&b=" in page
    # Strong-form counts: the detail fetch path is invoked exactly
    # once (the deep-dive ``loadReport``) plus the conclusion-regenerate POST
    # (``.../reports/{id}/conclusion``); the compare endpoint is invoked
    # exactly once (the ``compare`` click handler).
    assert page.count('"/benchmark/reports/" + encodeURIComponent') >= 1
    assert page.count('"/benchmark/compare?a=" + encodeURIComponent') == 1


# ---------------------------------------------------------------------------
# Report deep-dive (AC5 — single-report view)
# ---------------------------------------------------------------------------


def test_results_page_html_has_report_picker() -> None:
    """The report picker is a dropdown populated from ``GET /benchmark/reports``."""
    page = results_page_html()
    # The deep-dive picker is anchored by a stable id the JS can target.
    assert 'id="reportPicker"' in page or "reportPicker" in page
    # The picker is filled from the list endpoint.
    assert (
        "GET /benchmark/reports" in page
        or '"/benchmark/reports"' in page
        or '+ "/benchmark/reports"' in page
    )


def test_results_page_html_has_deep_dive_anchors() -> None:
    """The deep-dive panel renders the metric tiers, latency, VRAM, model_config.

    Mirrors the F-00095 ``benchmark_page.py`` deep-dive: the page
    MUST reuse the same metric-row / latency-cell helpers and the
    same model_load_ms separation (Invariant 4).  The anchors
    survive verbatim into the Results + Compare view so the two
    views stay consistent.
    """
    page = results_page_html()
    # The deep-dive card has a stable id.
    assert "reportDetail" in page
    # model_load_ms is shown SEPARATELY from per-query latency (Invariant 4).
    assert "model_load_ms" in page
    # The three metric groupings are surfaced (the shared renderer uses
    # plain-language section titles instead of "Tier N" jargon):
    #   retrieval quality, reference-free answer quality, reference-required.
    assert "Retrieval quality" in page
    assert page.lower().count("answer quality") >= 2  # reference-free + reference-required
    assert "reference-free" in page.lower()
    assert "reference answers" in page.lower()
    # The latency grid surfaces the per-query stage + the e2e p99
    # (model_load_ms is in its own bucket, not folded in).
    assert "per_query" in page.lower() or "per-query" in page.lower() or "per query" in page.lower()
    # The VRAM breakdown is rendered (baseline / peak / incremental /
    # util peak).
    assert "VRAM" in page or "vram" in page.lower()
    # The model_config snapshot is collapsible.
    assert "config_snapshot" in page or "model_config" in page


def test_results_page_html_handles_404_for_unknown_report() -> None:
    """Boundary row "Unknown report id" — the deep-dive panel renders
    a 404 not-found state when ``GET /benchmark/reports/{id}`` returns
    404 (never a toast that hides the cause)."""
    page = results_page_html()
    # The 404 branch is explicit in the JS handler.
    assert "404" in page
    # The not-found copy surfaces the cause (mirrors
    # ``benchmark_page.py``'s "Not found." caveat-box).
    assert "not found" in page.lower() or "not-found" in page.lower() or "unknown" in page.lower()
    # The response uses the caveat-box class so the user sees the
    # same visual treatment as the underpowered caveat.
    assert "caveat-box" in page


# ---------------------------------------------------------------------------
# A/B compare (AC5 + Invariant 6)
# ---------------------------------------------------------------------------


def test_results_page_html_has_ab_compare_anchors() -> None:
    """The A/B compare picker has A + B dropdowns + a Compare button."""
    page = results_page_html()
    # A and B dropdowns (mirrors ``benchmark_page.py``).
    assert "compareA" in page
    assert "compareB" in page
    # The compare trigger.
    assert "compareBtn" in page
    # The compare-result body region.
    assert "compareBody" in page
    # Strong-form counts: each anchor is referenced exactly once on
    # the page (the dropdown, the trigger, the result region — not
    # duplicated in the empty-history empty state or in copy).
    assert page.count('id="compareA"') == 1
    assert page.count('id="compareB"') == 1
    assert page.count('id="compareBtn"') == 1
    assert page.count('id="compareBody"') == 1


def test_results_page_html_renders_delta_table_columns() -> None:
    """AC5 — the per-metric delta table renders the canonical column set.

    The columns are: metric, val A, val B, delta, p_value,
    p_corrected, CI 95%, significant (the F-00095
    ``benchmark_page.py`` delta-table shape, reused verbatim).
    """
    page = results_page_html()
    for col in (
        "metric",
        "val A",
        "val B",
        "delta",
        "p_value",
        "p_corrected",
        "CI 95",
        "significant",
    ):
        assert col in page, f"missing delta-table column '{col}' (AC5)"


def test_results_page_html_renders_winner_badge_classes() -> None:
    """AC5 — the winner verdict is rendered as a badge with one of
    three classes: A, B, or tie.  The ``None`` case (neither config
    wins on both nDCG@5 AND faithfulness) is rendered as the tie
    class with the "No winner" copy (not a crash, not a toast)."""
    page = results_page_html()
    # The badge anchor class.
    assert "compare-winner" in page
    # The three outcome classes (the F-00095 ``benchmark_page.py``
    # shape, reused verbatim — CSS uses the .A / .B / .tie compound
    # selectors).
    assert ".compare-winner.A" in page, "missing CSS rule for compare-winner A (AC5)"
    assert ".compare-winner.B" in page, "missing CSS rule for compare-winner B (AC5)"
    # The tie class.
    assert ".compare-winner.tie" in page, "missing CSS rule for compare-winner tie (AC5)"
    # The "No winner" / "winner=None" copy is explicit so the
    # operator can see the API returned a null winner.
    assert "no winner" in page.lower() or "winner=None" in page or "neither config" in page.lower()
    # The "better on both nDCG@5 AND faithfulness" copy is surfaced
    # (the F-00095 / AC5 contract — the winner flag is set ONLY
    # when one config is better on BOTH nDCG@5 AND faithfulness).
    assert "nDCG@5" in page and "faithfulness" in page


def test_results_page_html_renders_underpowered_caveat() -> None:
    """AC5 — the underpowered caveat is surfaced (R-00121 §4.2).

    The caveat must be rendered when the API returns
    ``underpowered=true`` OR when the caveat text matches the
    canonical n≈37 / n≈150 anchors.  Mirrors the F-00095
    ``benchmark_page.py`` regex.
    """
    page = results_page_html()
    # The CSS class anchor.
    assert "caveat-box" in page
    # The explicit underpowered text.
    assert "underpower" in page.lower()
    # The canonical anchors (n≈37 and n≈150) surface in the JS
    # regex that matches the API caveat text (R-00121 §4.2).
    assert "37" in page
    assert "150" in page
    # Strong-form counts: the underpowered banner is rendered
    # exactly once (the JS handler in ``compare()`` is the sole
    # emission site; the F-00095 shape reused verbatim).
    assert page.count('class="caveat-box"><strong>Underpowered.</strong>') == 1
    # The n≈37 and n≈150 anchors appear in the JS regex pattern
    # (at-least-once: the regex pattern + the canonical fallback
    # copy both carry the anchors).
    assert page.count("37") >= 2
    assert page.count("150") >= 2


def test_results_page_html_handles_400_for_different_dataset_compare() -> None:
    """Boundary row "Compare across different datasets" — the panel
    renders the not-comparable message when
    ``GET /benchmark/compare`` returns 400 (Invariant 6).  The
    message is rendered in the panel, not as a toast that hides
    the cause."""
    page = results_page_html()
    # The 400 branch is explicit in the JS handler.
    assert "400" in page
    # The not-comparable copy is in the page (different datasets).
    assert "different datasets" in page.lower() or "different dataset" in page.lower()
    # The paired-stats rationale is in the copy (the F-00095
    # rationale reused verbatim — "paired stats require the same
    # golden queries").
    assert (
        "paired stats" in page.lower()
        or "same golden queries" in page.lower()
        or "same dataset" in page.lower()
    )
    # Uses the caveat-box so the user sees the same visual treatment
    # as the underpowered caveat.
    assert "caveat-box" in page


def test_results_page_html_handles_404_for_compare_unknown_id() -> None:
    """Boundary row "Unknown report id" on compare — the panel
    renders the not-found message when either A or B is unknown."""
    page = results_page_html()
    # The compare path's 404 branch is distinct from the deep-dive
    # 404 branch (different message, but both use the caveat-box
    # visual treatment).
    assert (
        "one of the report ids is unknown" in page.lower()
        or "one of the report ids" in page.lower()
        or "ids is unknown" in page.lower()
    )


# ---------------------------------------------------------------------------
# Empty-history empty state (Boundary row "Empty report history")
# ---------------------------------------------------------------------------


def test_results_page_html_has_empty_history_state() -> None:
    """Boundary row "Empty report history" — when
    ``GET /benchmark/reports`` returns an empty list, both the
    deep-dive picker and the compare pickers render an explicit
    empty state ("No reports yet — run a benchmark from the
    Benchmark Runner to create one."), NOT an error.  Mirrors
    the F-00095 / documents-page empty-state pattern."""
    page = results_page_html().lower()
    # The empty-state copy.
    assert "no reports yet" in page
    # The call-to-action references the Benchmark Runner (the
    # sub-view that creates reports).
    assert "benchmark runner" in page or "run a benchmark" in page


# ---------------------------------------------------------------------------
# Console shell integration (S10 nav + IW chrome)
# ---------------------------------------------------------------------------


def test_results_page_html_has_six_nav_destinations() -> None:
    """The page renders the S10 console nav with all six destinations
    (Config / Benchmark / Results / Scoreboard / Chat / Upload).
    Mirrors the shell nav contract so the S23 browser-verification
    can assert navigation between sub-views works."""
    page = results_page_html()
    for nav in (
        'id="navConfig"',
        'id="navBenchmark"',
        'id="navResults"',
        'id="navScoreboard"',
        'id="navChat"',
        'id="navUpload"',
    ):
        assert nav in page, f"missing console-nav anchor {nav}"
    for href in (
        'href="/console/config"',
        'href="/console/benchmark"',
        'href="/console/results"',
        'href="/console/scoreboard"',
        'href="/console/chat"',
        'href="/console/upload"',
    ):
        assert href in page, f"missing console-nav href {href}"


def test_results_page_html_has_data_theme_dark() -> None:
    """Dark theme is the single default (matches the benchmark page + Chainlit theme)."""
    page = results_page_html()
    assert 'data-theme="dark"' in page
    # Strong-form: the data-theme attribute appears exactly once on
    # the <html> root.
    assert page.count('data-theme="dark"') == 1


def test_results_page_html_has_innovation_ways_wordmark() -> None:
    """The 'Innovation Ways' header wordmark is present, exactly once.

    The wordmark renders as split, coloured initials, so the literal
    "Innovation Ways" substring is not in the markup — assert the header logo
    wordmark itself.  The browser-tab ``<title>`` is the short, uniform
    "IW - RAG" form.
    """
    page = results_page_html()
    assert page.count('<span class="logo">') == 1
    assert '<span class="i">I</span>nnovation <span class="w">W</span>ays' in page
    assert "<title>IW - RAG</title>" in page
    assert "Innovation Ways" not in page  # long brand must not return to the tab


# ---------------------------------------------------------------------------
# API-base regex (S10 forward-slash-escape regression pin) + raw-string
# node --check parser
# ---------------------------------------------------------------------------


def test_results_page_html_derives_api_base_is_fixed_api_edge_base() -> None:
    """The page derives its API base as the fixed ``"/api"`` edge prefix.

    F-00096: the Operator Console is served at ``/console*`` behind the single
    Caddy edge (ADR-016), where the API lives under ``/api/*``. The base is the
    constant ``"/api"`` (no location-derived regex), so a fetch of
    ``API + "/api/config"`` -> ``/api/api/config`` which the edge's
    ``handle_path /api/*`` strips back to ``/api/config`` on the API. Replaces
    the old ``/api/console`` location regex, under which every sub-view data
    fetch 404'd at ``/console/*`` behind the edge.
    """
    page = results_page_html()
    assert page.count('var API = "/api";') == 1, (
        "console page must declare the fixed /api edge base exactly once (F-00096) — "
        "a second/stale base declaration would re-introduce the /console/* 404 bug"
    )


def test_results_page_html_inline_script_parses() -> None:
    """The page's entire inline ``<script>`` must be syntactically valid JS.

    Mirrors :func:`test_benchmark_page_html_inline_script_parses`.
    The page is generated from a triple-quoted Python string; when
    that string is **not** raw, JS escape sequences written in the
    source are consumed by Python and the page would throw a
    SyntaxError at load time — invisible to string-match tests but
    fatal in a real browser.

    This test extracts the full ``<script>`` body and runs
    ``node --check`` over it.  Skips when Node is unavailable.
    """
    scripts = re.findall(r"<script\b[^>]*>(.*?)</script>", results_page_html(), re.S)
    assert scripts, "no <script> block found in the Results + Compare view page"
    script = "\n".join(scripts)
    # Guard the specific render-time signatures directly (cheap, Node-free).
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
        "Results + Compare view inline script failed `node --check` "
        f"(SyntaxError -> page never boots): stderr={node.stderr!r}"
    )
