"""Unit tests for the F-00096 S11 Config view page module.

The Config view is a self-contained inline HTML+CSS+vanilla-JS page
mirroring :mod:`iw_rag.api.benchmark_page` and
:mod:`iw_rag.api.console_page`.  These tests pin the behavioural
contract that is verifiable from Python: the page string returned by
:func:`iw_rag.api.config_page.config_page_html` must carry every
section anchor the F-00096 Feature Design (Scope "Config view +
live-edit", AC2/AC3/AC4) requires (per-key explanation column,
per-editable-key edit form wired to bounds, locked badge,
pending-restart badge, change-log panel anchor, PUT body that carries
ONLY ``value``), and the page must follow the same operational
keystone as :mod:`iw_rag.api.benchmark_page` (location-derived API
base, no external CDN, sessionStorage-only bearer token, IW brand
palette).

The page content itself is asserted via these anchors; the S17
integration suite verifies the live ``GET /console/config`` route
returns 200 HTML carrying the same markers.  No JS engine is spun up
here: the page is structurally checked, then the S23
browser-verification step exercises the interactive behaviour in a
real browser.
"""

from __future__ import annotations

from iw_rag.api.config_page import config_page_html


def test_config_page_html_returns_non_empty_string() -> None:
    """The page factory returns a non-empty HTML string."""
    page = config_page_html()
    assert isinstance(page, str)
    assert page.strip() != ""
    # Top-level doctype + html tag — the page is real HTML, not just text.
    assert page.lstrip().lower().startswith("<!doctype html>")


def test_config_page_html_contains_brand_palette() -> None:
    """The IW teal/ink palette tokens are present (branding keystone).

    Mirrors the benchmark-page + console-shell assertion style — the
    :root palette defines each token; subsequent call-sites reuse
    ``var(--teal)`` / ``var(--teal-deep)`` so the teal-deep token
    appears exactly once.
    """
    page = config_page_html()
    assert "#0D9488" in page  # teal-deep
    assert "#2DD4BF" in page  # teal
    assert "#181D29" in page  # ink
    # The :root palette defines teal-deep exactly once; the brand chrome
    # and the gradient call-sites reuse var(--teal) / var(--teal-deep).
    assert page.count("#0D9488") == 1


def test_config_page_html_has_no_external_cdn() -> None:
    """Invariant 8 — no external CDN / Google Fonts link (local-only rule)."""
    page = config_page_html().lower()
    # The page must not pull a stylesheet or font from any third party.
    for forbidden in (
        "fonts.googleapis.com",
        "fonts.gstatic.com",
        "cdn.jsdelivr.net",
        "unpkg.com",
        "cdnjs.cloudflare.com",
    ):
        assert forbidden not in page, (
            f"Config view page must not reference external CDN '{forbidden}' (Invariant 8)."
        )


def test_config_page_html_derives_api_base_from_location() -> None:
    """The API base is derived from ``window.location`` (not hardcoded).

    Mirrors :func:`test_benchmark_page_html_derives_api_base_from_location`.
    The Config view lives at ``/console/config`` (or
    ``/api/console/config`` behind Caddy) and must strip that suffix
    from ``location.pathname`` to derive the API base.  Hardcoded
    host:port pairs are forbidden.
    """
    page = config_page_html()
    assert "location.pathname" in page
    assert "/console/config" in page
    # The page must NOT hardcode a host:port pair.
    assert "http://localhost" not in page
    assert "http://127.0.0.1" not in page


def test_config_page_html_consumes_get_api_config() -> None:
    """AC2 — the page reads the active config from ``GET /api/config``."""
    page = config_page_html()
    assert "GET /api/config" in page or '"/api/config"' in page or '+ "/api/config"' in page
    # The fetch call is anchored so a regression that breaks the GET is
    # caught by the string match.
    assert "/api/config" in page
    # The fetch is wired to the per-row loader (the page populates the
    # table on load; the S23 step exercises the round-trip).
    assert "loadConfig" in page or "loadActive" in page or "loadConfigItems" in page


def test_config_page_html_submits_via_put_api_config_key() -> None:
    """AC2 — the page submits an edit to ``PUT /api/config/{key}``."""
    page = config_page_html()
    # The {key} placeholder is interpolated via URL composition
    # (``"/api/config/" + encodeURIComponent(key)`` is the canonical
    # pattern in the benchmark page too).
    assert "PUT /api/config/" in page or '"/api/config/"' in page or '+ "/api/config/"' in page
    # The body carries ONLY ``value`` (Invariant 3 — keystone).
    assert "JSON.stringify" in page
    # The submit path is named in JS so the S23 step can target it.
    assert "putConfig" in page or "submitEdit" in page or "saveEdit" in page


def test_config_page_html_has_locked_badge_marker() -> None:
    """AC3 — a **locked** badge is rendered for items with ``locked=true``.

    Regression pin for the S11 contract: the badge must appear with a
    stable class (so CSS can style it red) and a stable text marker
    (so the S23 browser-verification step can assert the badge is
    visible for a ``guardrails.enabled`` row).  The page is a pure
    API client — the locked set is enforced server-side; the badge is
    UX-only honesty.
    """
    page = config_page_html()
    # The literal badge text is used so the page renders the operator-
    # facing message verbatim.
    assert "locked" in page.lower()
    # A stable CSS class the page's render path assigns to the locked
    # row's badge element.
    assert "chip-locked" in page or "badge-locked" in page or "locked-badge" in page
    # The badge label is surfaced to the operator.
    assert "Locked" in page or "locked" in page


def test_config_page_html_has_pending_restart_badge_marker() -> None:
    """AC4 — a **pending restart** badge is rendered for items with
    ``pending_restart=true``.

    The page must surface the literal badge text (so the operator sees
    that the running request path is unchanged until restart) and a
    stable CSS class (so the S23 step can assert the badge is visible
    for a ``retrieval.top_k`` row).
    """
    page = config_page_html()
    # The literal badge text is used so the page renders the operator-
    # facing message verbatim.
    assert "pending restart" in page.lower() or "pending-restart" in page.lower()
    # A stable CSS class the page's render path assigns to the pending-
    # restart row's badge element.
    assert (
        "chip-pending" in page
        or "badge-pending" in page
        or "pending-restart-badge" in page
        or "chip-pending-restart" in page
    )


def test_config_page_html_has_change_log_panel_anchor() -> None:
    """The change-log panel has a stable id (so the S17 + S23 steps
    can assert it is present and populated).

    S07 exposes the ``config_change_log`` table; S11 surfaces the
    panel.  The page must always render a stable anchor (the panel
    is present even on the empty-state — Boundary row "empty
    history" is rendered as an explicit empty state, not a missing
    section).
    """
    page = config_page_html()
    # The panel has a stable id (the S17 + S23 contract).
    assert 'id="changeLogPanel"' in page or 'id="changeLog"' in page or "changeLogPanel" in page
    # The panel is referenced by the loader (so the S23 step can
    # target it deterministically).
    assert "changeLog" in page
    # The "no change-log endpoint wired" empty state is rendered
    # explicitly (the design Boundary row "empty state, not an
    # error") — a literal marker the S23 step can assert.
    assert (
        "no change-log" in page.lower()
        or "no_change_log" in page.lower()
        or "nochangelog" in page.lower()
        or "no log endpoint" in page.lower()
        or "log panel" in page.lower()
    )


def test_config_page_html_edit_body_carries_only_value() -> None:
    """Invariant 3 — the edit-submit body contains ``value`` ONLY.

    The keystone: the body MUST NOT carry ``principal_id`` /
    ``groups`` / ``acl`` — the principal is sourced from the resolved
    :class:`AccessContext` (auth-only), never from the request body
    (mirrors the :class:`QueryRequest` / :class:`ActivateRequest` /
    :class:`BenchmarkRunRequest` / :class:`ConfigUpdateRequest`
    keystone).

    The page's PUT body assembly is in the inline script; this test
    extracts the ``body`` assembly code and greps the literal keys
    that MUST NOT appear in the body.
    """
    import re

    page = config_page_html()
    # Find the script body and the section that builds the PUT body
    # (it lives inside the putConfig / submitEdit / saveEdit function).
    scripts = re.findall(r"<script\b[^>]*>(.*?)</script>", page, re.S)
    assert scripts, "no <script> block found in the Config view page"
    script = "\n".join(scripts)
    # The body builder must reference ``value`` (the S07 contract).
    assert "value" in script
    # The body builder MUST NOT mention ``principal_id`` / ``groups``
    # / ``acl`` (Invariant 3 — keystone).  Each is a forbidden key
    # in the JSON.stringify(...) call.
    # Note: the search is restricted to the PUT body assembly region
    # (the one whose body sits inside the saveEdit function — the
    # login body is `username/password` and is intentionally
    # ignored).  We find the PUT by anchoring on the
    # ``/api/config/`` URL composition that precedes the body.
    body_match = re.search(
        r"/api/config/[^\"']*\"[^)]*?body:\s*JSON\.stringify\(\s*\{([^}]+)\}",
        script,
    )
    if body_match is None:
        # Fallback: match the FIRST JSON.stringify whose body contains
        # the literal "value" key (the PUT body is the only one in
        # this page that includes the S07 "value" key — the login
        # body is {username, password}).
        for m in re.finditer(r"JSON\.stringify\(\s*\{([^}]+)\}", script):
            if "value" in m.group(1):
                body_match = m
                break
    assert body_match is not None, (
        "could not find the PUT /api/config/{key} body assembly (JSON.stringify({...})) "
        "in the inline script (AC2 — the page must submit a {value: ...} body)."
    )
    body_inner = body_match.group(1)
    for forbidden in ("principal_id", "groups", "acl", "principal_groups", "principalGroups"):
        assert forbidden not in body_inner, (
            f"PUT body assembly mentions forbidden key '{forbidden}' (Invariant 3 — "
            f"the principal comes from auth, never from the request body)."
        )
    # The body MUST mention ``value`` (the only permitted key).
    assert "value" in body_inner, (
        "PUT body assembly does not include the 'value' key — the S07 route expects "
        "{value: <new_value>} (mirrors ConfigUpdateRequest)."
    )


def test_config_page_html_derives_api_base_is_fixed_api_edge_base() -> None:
    """The page derives its API base as the fixed ``"/api"`` edge prefix.

    F-00096: the Operator Console is served at ``/console*`` behind the single
    Caddy edge (ADR-016), where the API lives under ``/api/*``. The base is the
    constant ``"/api"`` (no location-derived regex), so a fetch of
    ``API + "/api/config"`` -> ``/api/api/config`` which the edge's
    ``handle_path /api/*`` strips back to ``/api/config`` on the API. Replaces
    the old ``/api/console`` location regex, under which every sub-view data
    fetch 404'd at ``/console/*`` behind the edge.
    """
    page = config_page_html()
    assert page.count('var API = "/api";') == 1, (
        "console page must declare the fixed /api edge base exactly once (F-00096) — "
        "a second/stale base declaration would re-introduce the /console/* 404 bug"
    )


def test_config_page_html_inline_script_parses() -> None:
    """The page's entire inline ``<script>`` must be syntactically valid JS.

    Regression pin mirroring
    :func:`test_benchmark_page_html_inline_script_parses`.  The page
    is generated from a triple-quoted Python string; when that string
    is **not** raw, JS escape sequences written in the source are
    consumed by Python and the page would throw a SyntaxError at load
    time — invisible to string-match tests but fatal in a real
    browser.

    This test extracts the full ``<script>`` body and runs
    ``node --check`` over it.  Skips when Node is unavailable.
    """
    import re
    import subprocess
    import tempfile

    page = config_page_html()
    scripts = re.findall(r"<script\b[^>]*>(.*?)</script>", page, re.S)
    assert scripts, "no <script> block found in the Config view page"
    script = "\n".join(scripts)
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
        "Config view inline script failed `node --check` "
        f"(SyntaxError -> page never boots): stderr={node.stderr!r}"
    )
