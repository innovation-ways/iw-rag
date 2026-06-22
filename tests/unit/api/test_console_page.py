"""Unit tests for the F-00096 Console shell page module.

The Console shell is the unified nav and shared chrome that every
console view (Config / Benchmark / Results / Scoreboard / Chat / Upload)
renders inside.  It mirrors :mod:`iw_rag.api.benchmark_page` and
:mod:`iw_rag.api.documents_page`: a self-contained inline HTML+CSS+vanilla-
JS page whose only collaborator is the existing ``GET /auth/login`` +
``GET /api/...`` HTTP surface — the console contains NO retrieval, ACL,
grounding, or guard logic (Invariant 4).

These tests pin the behavioural contract that is verifiable from
Python: the page string returned by
:func:`iw_rag.api.console_page.console_page_html` must carry every
section anchor the F-00096 Feature Design (Scope "Console shell",
Invariants 4 + 8) requires (the IW brand chrome, the unified nav with
six destinations, the ``/auth/login`` self-handled sign-in, the
location-derived API base, the sessionStorage-only bearer token), and
must NOT introduce any external CDN or hosted font.

The page content itself is asserted via these anchors; the S17
integration suite verifies the live ``GET /console`` route returns 200
HTML carrying the same markers.  No JS engine is spun up here: the
page is structurally checked, then the S23 browser-verification step
exercises the interactive behaviour in a real browser.
"""

from __future__ import annotations

from iw_rag.api.console_page import console_page_html


def test_console_page_html_returns_non_empty_string() -> None:
    """The page factory returns a non-empty HTML string."""
    page = console_page_html()
    assert isinstance(page, str)
    assert page.strip() != ""
    # Top-level doctype + html tag — the page is real HTML, not just text.
    assert page.lstrip().lower().startswith("<!doctype html>")


def test_console_page_html_contains_brand_palette() -> None:
    """The IW teal/ink palette tokens are present (branding keystone).

    Mirrors the benchmark-page assertion style — the :root palette
    defines each token; subsequent call-sites reuse ``var(--teal)`` /
    ``var(--teal-deep)`` so the teal-deep token appears exactly once.
    """
    page = console_page_html()
    assert "#0D9488" in page  # teal-deep
    assert "#2DD4BF" in page  # teal
    assert "#181D29" in page  # ink
    # The :root palette defines teal-deep exactly once; the brand chrome
    # and the gradient call-sites reuse var(--teal) / var(--teal-deep).
    assert page.count("#0D9488") == 1


def test_console_page_html_has_no_external_cdn() -> None:
    """Invariant 8 — no external CDN / Google Fonts link (local-only rule).

    Mirrors :func:`test_benchmark_page_html_has_no_external_cdn`.  The
    Console is a thin API client; even the *fonts* must be self-hosted.
    """
    page = console_page_html().lower()
    # The page must not pull a stylesheet or font from any third party.
    for forbidden in (
        "fonts.googleapis.com",
        "fonts.gstatic.com",
        "cdn.jsdelivr.net",
        "unpkg.com",
        "cdnjs.cloudflare.com",
    ):
        assert forbidden not in page, (
            f"Console page must not reference external CDN '{forbidden}' (Invariant 8)."
        )


def test_console_page_html_derives_api_base_from_location() -> None:
    """Invariant 8 — the API base is derived from ``window.location`` (not hardcoded).

    Mirrors :func:`test_benchmark_page_html_derives_api_base_from_location`.
    The Console lives at ``/console`` (or ``/api/console`` behind Caddy)
    and must strip that suffix from ``location.pathname`` to derive the
    API base.  Hardcoded host:port pairs are forbidden.
    """
    page = console_page_html()
    assert "location.pathname" in page
    assert "/console" in page
    # The page must NOT hardcode a host:port pair.
    assert "http://localhost" not in page
    assert "http://127.0.0.1" not in page


def test_console_page_html_logs_in_via_auth_login() -> None:
    """The page self-handles sign-in via ``/auth/login``.

    Mirrors the documents_page + benchmark_page login flow.
    """
    page = console_page_html()
    assert "/auth/login" in page
    # Session-memory bearer token (never persisted beyond the tab).
    assert "sessionStorage" in page
    # Exactly one POST to /auth/login (one login button).
    assert page.count("/auth/login") == 1


def test_console_page_html_has_innovation_ways_wordmark() -> None:
    """The IW brand chrome — the "Innovation Ways" wordmark in the header.

    The wordmark renders as split, coloured initials
    (``<span class="i">I</span>nnovation <span class="w">W</span>ays``), so the
    literal "Innovation Ways" substring is not present in the markup — assert the
    header logo wordmark itself.  The browser-tab ``<title>`` is the short
    "IW · …" form (pinned by :func:`test_console_page_html_has_short_iw_title`).
    """
    page = console_page_html()
    # The header logo wordmark — exactly once (guards against a duplicate).
    assert page.count('<span class="logo">') == 1
    assert '<span class="i">I</span>nnovation <span class="w">W</span>ays' in page


def test_console_page_html_has_short_iw_title() -> None:
    """The browser-tab ``<title>`` is the short, uniform "IW - RAG" form.

    The long "Innovation Ways · …" title overflowed the tab; the title is the
    short "IW - RAG" for every console view, while the on-page header keeps the
    full wordmark.
    """
    page = console_page_html()
    assert "<title>IW - RAG</title>" in page
    assert "Innovation Ways" not in page  # the long brand must not return to the tab


def test_console_page_html_has_data_theme_dark() -> None:
    """Dark theme is the default (matches benchmark_page + Chainlit theme)."""
    page = console_page_html()
    # The dark theme is the single default — exactly one declaration.
    assert page.count('data-theme="dark"') == 1
    assert 'data-theme="dark"' in page


def test_console_page_html_has_six_nav_destinations() -> None:
    """The unified nav exposes all six console destinations (S10 scope).

    * Config       → ``/console/config``        (S11)
    * Benchmark    → ``/console/benchmark``     (S12)
    * Results      → ``/console/results``       (S13)
    * Scoreboard   → ``/console/scoreboard``    (S14)
    * Chat         → ``/``                       (Chainlit — linked, not rebuilt)
    * Upload       → ``/documents/ui``           (Document Manager — linked, not rebuilt)

    Each nav item is anchored by a stable id/href so tests + the S11–S15
    views can target it deterministically.
    """
    page = console_page_html()
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


def test_console_page_html_derives_api_base_is_fixed_api_edge_base() -> None:
    """The page derives its API base as the fixed ``"/api"`` edge prefix.

    F-00096: the Operator Console is served at ``/console*`` behind the single
    Caddy edge (ADR-016), where the API lives under ``/api/*``. The base is the
    constant ``"/api"`` (no location-derived regex), so a fetch of
    ``API + "/api/config"`` -> ``/api/api/config`` which the edge's
    ``handle_path /api/*`` strips back to ``/api/config`` on the API. Replaces
    the old ``/api/console`` location regex, under which every sub-view data
    fetch 404'd at ``/console/*`` behind the edge.
    """
    page = console_page_html()
    assert page.count('var API = "/api";') == 1, (
        "console page must declare the fixed /api edge base exactly once (F-00096) — "
        "a second/stale base declaration would re-introduce the /console/* 404 bug"
    )


def test_console_page_html_inline_script_parses() -> None:
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
    import re
    import subprocess
    import tempfile

    page = console_page_html()
    scripts = re.findall(r"<script\b[^>]*>(.*?)</script>", page, re.S)
    assert scripts, "no <script> block found in the Console shell page"
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
        "Console shell inline script failed `node --check` "
        f"(SyntaxError -> page never boots): stderr={node.stderr!r}"
    )


def test_console_page_html_has_console_shell_marker() -> None:
    """The shell exposes a content region the S11–S15 views swap into.

    The shell is a single host page; the S11–S15 sub-views load their
    own partials into a stable content region (``#viewContent``).  The
    ``#viewContent`` id is the contract the S11–S15 pages use to mount
    their own markup.  The shell must also expose a Sign-out affordance
    that mirrors :mod:`iw_rag.api.benchmark_page`.
    """
    page = console_page_html()
    # Exactly one viewContent anchor — the S11–S15 contract.
    assert page.count('id="viewContent"') == 1
    # Exactly one logout affordance (the header).
    assert page.count('id="logout"') == 1
    assert 'id="viewContent"' in page, (
        "Console shell must expose a stable #viewContent region for the S11–S15 views to swap into."
    )
    assert 'id="logout"' in page, "Sign-out affordance (logout) missing"
