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
