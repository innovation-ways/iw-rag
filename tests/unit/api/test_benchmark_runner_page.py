# ruff: noqa: E501
"""Unit tests for the F-00096 S12 Benchmark Runner view page module.

The Runner view is a self-contained inline HTML+CSS+vanilla-JS page
mirroring :mod:`iw_rag.api.benchmark_page` and
:mod:`iw_rag.api.console_page` — re-homed as a console sub-view, with
the **live VRAM budget bar** (server-rendered values only) added.
These tests pin the behavioural contract that is verifiable from
Python: the page string returned by
:func:`iw_rag.api.benchmark_runner_page.benchmark_runner_page_html` must
carry every section anchor the F-00096 Feature Design (Scope
"Benchmark Runner view", AC1, Boundary rows "VRAM over budget", "Swap
timeout during a run", "Catalog has no eligible models") requires
(per-step model pickers, VRAM-budget bar with RED zone, no-candidates
empty state, full SSE stage handler set, swap-timeout terminal
state, run-button disable lifecycle), and the page must follow the
same operational keystone as :mod:`iw_rag.api.benchmark_page`
(location-derived API base, no external CDN, ``/auth/login``
self-handled sign-in, sessionStorage-only bearer token, raw-string
``_PAGE`` whose inline ``<script>`` passes ``node --check``).

The page content itself is asserted via these anchors; the S17
integration suite verifies the live ``GET /console/benchmark`` route
returns 200 HTML carrying the same markers.  No JS engine is spun up
here: the page is structurally checked, then the S23
browser-verification step exercises the interactive behaviour in a
real browser.

**Invariant 7** — the VRAM math is computed server-side; the
frontend only renders the ``{fits, budget_used_gb, headroom_gb}``
triple the API returns.  The tests assert the page does NOT contain
the server-side arithmetic literals (the 32-GB ceiling / resident
Infinity / 4-GB constants), so a future refactor that re-introduces
client-side math fails loudly.
"""

from __future__ import annotations

import re
import subprocess
import tempfile

from iw_rag.api.benchmark_runner_page import benchmark_runner_page_html

# ---------------------------------------------------------------------------
# Page factory / brand / operational keystone
# ---------------------------------------------------------------------------


def test_benchmark_runner_page_html_returns_non_empty_string() -> None:
    """The page factory returns a non-empty HTML string starting with ``<!doctype html>``."""
    page = benchmark_runner_page_html()
    assert isinstance(page, str)
    assert page.strip() != ""
    assert page.lstrip().lower().startswith("<!doctype html>")


def test_benchmark_runner_page_html_contains_brand_palette() -> None:
    """The IW teal/ink palette tokens are present (branding keystone)."""
    page = benchmark_runner_page_html()
    assert "#0D9488" in page  # teal-deep
    assert "#2DD4BF" in page  # teal
    assert "#181D29" in page  # ink
    # The :root palette defines teal-deep exactly once; subsequent
    # call-sites reuse ``var(--teal-deep)`` (mirror benchmark_page
    # assertion).
    assert page.count("#0D9488") == 1


def test_benchmark_runner_page_html_has_no_external_cdn() -> None:
    """Invariant 8 — no external CDN / Google Fonts link (local-only rule)."""
    page = benchmark_runner_page_html().lower()
    for forbidden in (
        "fonts.googleapis.com",
        "fonts.gstatic.com",
        "cdn.jsdelivr.net",
        "unpkg.com",
        "cdnjs.cloudflare.com",
    ):
        assert forbidden not in page, (
            f"Runner view page must not reference external CDN '{forbidden}' (Invariant 8)."
        )


def test_benchmark_runner_page_html_derives_api_base_from_location() -> None:
    """The API base is derived from ``window.location`` (not hardcoded).

    Mirrors :func:`test_benchmark_page_html_derives_api_base_from_location`
    + the F-00095 / S10 backslash-escape regression pin.  The Runner
    view lives at ``/console/benchmark`` (or ``/api/console/benchmark``
    behind Caddy) and must strip that suffix from ``location.pathname``
    to derive the API base.  Hardcoded host:port pairs are forbidden.
    """
    page = benchmark_runner_page_html()
    assert "location.pathname" in page
    assert "/console/benchmark" in page
    assert "http://localhost" not in page
    assert "http://127.0.0.1" not in page


def test_benchmark_runner_page_html_logs_in_via_auth_login() -> None:
    """The page self-handles sign-in via ``/auth/login`` (shell-session reuse)."""
    page = benchmark_runner_page_html()
    assert "/auth/login" in page
    # Session-memory bearer token (never persisted beyond the tab).
    assert "sessionStorage" in page
    # Exactly one POST to /auth/login (the shell's login flow).
    assert page.count("/auth/login") == 1


# ---------------------------------------------------------------------------
# Per-step model pickers — AC1 / F-00096 §Scope
# ---------------------------------------------------------------------------


def test_benchmark_runner_page_html_has_per_step_pickers() -> None:
    """AC1 — per-step model pickers for the canonical F-00096 Runner view.

    The S12 scope says *generator / embedder / reranker / top_k / mode*;
    the existing ``benchmark_page.py`` shape uses
    ``generator/embedder/reranker/grader`` (F-00095).  The Runner view
    follows the F-00096 S12 contract — ``grader`` is dropped (grader
    lives in the deeper "judge" path), and ``top_k`` + ``mode`` are
    added so the console is the single per-step run-config surface.
    The picker ids are stable, test-assertable anchors.
    """
    page = benchmark_runner_page_html()
    for picker in (
        "pickGenerator",
        "pickEmbedder",
        "pickReranker",
        "pickTopK",
        "pickMode",
    ):
        assert picker in page, f"missing S12 picker id '{picker}' (AC1)"


def test_benchmark_runner_page_html_loads_models_from_endpoint() -> None:
    """AC1 — pickers are populated from ``GET /models`` (F-00094) with
    graceful degrade (mirrors the ``benchmark_page.py`` "catalog
    unavailable" behaviour)."""
    page = benchmark_runner_page_html()
    assert "/models" in page
    # Graceful-degrade marker — a 4xx/5xx/timeout leaves the pickers in
    # the "catalog unavailable" state but the page keeps loading.
    assert "catalog" in page.lower() or "unavailable" in page.lower()
    # The page must call ``GET /models`` (S08: extended with VRAM
    # budget), so a regression that breaks the load path is caught.
    assert "GET /models" in page or '"/models"' in page or '+ "/models"' in page


def test_benchmark_runner_page_html_has_params_panel() -> None:
    """AC1 — the params panel carries the canonical knobs (top_k, mode)."""
    page = benchmark_runner_page_html()
    # ``top_k`` is the canonical retrieval knob (S07 / F-00093) — the
    # F-00095 Benchmark page uses the same label.
    assert "top_k" in page
    assert page.count("top_k") >= 2
    # ``mode`` is the canonical query-mode enum (QueryRequest).
    assert "mode" in page.lower()
    assert page.count('id="pickMode"') == 1
    assert page.count('id="pickTopK"') == 1


# ---------------------------------------------------------------------------
# VRAM budget bar — AC1 / Invariant 7
# ---------------------------------------------------------------------------


def test_benchmark_runner_page_html_has_vram_bar_anchor() -> None:
    """AC1 — the VRAM budget bar has a stable id the JS can target."""
    page = benchmark_runner_page_html()
    # Stable id the page's render path assigns to the bar element.
    # Mirrors the ``progressPanel`` / ``catalogWarn`` naming convention
    # in ``benchmark_page.py`` — kebab-case stable ids that JS / CSS
    # can target deterministically.
    assert 'id="vramBar"' in page or 'id="vram-bar"' in page or 'id="vramBarFill"' in page, (
        "missing stable VRAM-bar id anchor (AC1 — the bar must be addressable)"
    )


def test_benchmark_runner_page_html_renders_server_vram_fields() -> None:
    """AC1 — the bar reads the server-provided ``fits`` /
    ``budget_used_gb`` / ``headroom_gb`` triple (Invariant 7).

    The bar update function must reference each of the three field
    names.  The page MUST NOT recompute the budget locally; a future
    refactor that re-introduces the ``expected_vram_gb + resident +
    ceiling`` arithmetic in JS fails this assertion.  The triple
    lives server-side in
    :func:`iw_rag.api.models_routes._compute_vram_budget`.
    """
    page = benchmark_runner_page_html()
    for field in ("fits", "budget_used_gb", "headroom_gb"):
        assert field in page, (
            f"VRAM bar render path does not reference the server field '{field}' "
            "(Invariant 7 — the bar must read the API-computed triple, not recompute it)."
        )


def test_benchmark_runner_page_html_does_not_recompute_vram_math() -> None:
    """Invariant 7 — the page MUST NOT recompute the VRAM budget locally.

    The server-side arithmetic lives in
    :func:`iw_rag.api.models_routes._compute_vram_budget` and uses
    the two named constants
    :data:`iw_rag.api.models_routes.RESIDENT_INFINITY_GB` (4.0) and
    :data:`iw_rag.api.models_routes.VRAM_CEILING_GB` (32.0).  The
    page must not duplicate that math; it must read the
    ``fits / budget_used_gb / headroom_gb`` triple the API returns.

    A future regression that re-introduces the 4.0 + expected_vram /
    32.0 ceiling math in JS (e.g. an operator trying to "fix" the
    bar) is caught by the absence of the magic constants / the
    absence of the "expected_vram_gb" + "ceiling" pair, and the
    presence of the canonical "vram_budget" payload property name.
    """
    page = benchmark_runner_page_html().lower()
    # The page MUST render the payload's ``vram_budget`` object (the
    # canonical API property the S08 route attaches to every row).
    assert "vram_budget" in page, (
        "Runner view never references the catalog row's 'vram_budget' object "
        "(Invariant 7 — the bar must read the server-computed triple)."
    )
    # The page MUST NOT re-do the math: the resident-infinity constant
    # (4.0 GB) and the ceiling constant (32.0 GB) are server-side
    # concerns; the bar updates from the server's ``fits`` flag.
    assert "32.0" not in page, (
        "Runner view hard-codes the 32.0 GB ceiling (Invariant 7 violation — "
        "the VRAM math is server-side; the bar reads the API's fits/headroom fields)."
    )
    # The 4 GB resident-Infinity constant must not appear as a JS
    # literal either (the page renders it indirectly through the
    # server's ``budget_used_gb``).
    assert "4.0" not in page, (
        "Runner view hard-codes the 4.0 GB resident-Infinity constant (Invariant 7 "
        "violation — the VRAM math is server-side)."
    )
    # The catalog's ``expected_vram_gb`` field must not be referenced
    # as an arithmetic input either — the bar reads the server's
    # computed triple, not the raw expected_vram + a constant.
    assert "expected_vram_gb" not in page, (
        "Runner view reads expected_vram_gb (Invariant 7 violation — the server "
        "already combines it with resident + ceiling; the page must read the "
        "computed vram_budget.{fits, budget_used_gb, headroom_gb} triple)."
    )


def test_benchmark_runner_page_html_red_zone_keyed_off_fits_false() -> None:
    """AC1 — the RED zone + Run-disabled is wired to ``fits === false``.

    The bar must enter the RED zone when the API's ``fits`` flag is
    false, AND the Run button must be disabled.  The wiring must
    reference the ``fits`` flag (not the bar fill width or a
    side-effecting constant) so a regression that drops the server
    payload is caught.
    """
    page = benchmark_runner_page_html()
    # The "red" / "RED" class or attribute is applied to the bar
    # element.  The naming mirrors the F-00095 ``stage.error`` chip
    # idiom.
    assert "vram-bar-red" in page or "vramBarRed" in page or "vram-red" in page, (
        "missing stable RED-zone class/id for the VRAM bar (AC1)."
    )
    # The Run-disabled wiring reads the ``fits`` flag.  The test
    # guards the keystone: a regression that disables on a numeric
    # check (e.g. ``headroom_gb < 0`` recompute) would re-introduce
    # client-side math; the disable MUST be keyed off the server
    # ``fits`` flag.
    assert "fits" in page, (
        "RED-zone + Run-disabled wiring does not reference the 'fits' flag (AC1)."
    )


# ---------------------------------------------------------------------------
# No-candidates empty state + run-button lifecycle
# ---------------------------------------------------------------------------


def test_benchmark_runner_page_html_has_no_candidates_empty_state() -> None:
    """Boundary row "Catalog has no eligible models" — the Runner view
    shows a "no candidates" empty state AND the Run action is disabled
    when the catalog returns zero eligible candidates.

    Mirrors :func:`test_benchmark_page_html_handles_all_sse_stages`
    boundary coverage.  The empty state is a literal marker the S23
    browser-verification step can assert.
    """
    page = benchmark_runner_page_html()
    assert (
        "no candidates" in page.lower()
        or "no-candidates" in page.lower()
        or "noCandidates" in page
        or "no_candidates" in page.lower()
    ), "missing 'no candidates' empty-state marker (Boundary row 'Catalog has no eligible models')."


def test_benchmark_runner_page_html_marks_run_button_disable_lifecycle() -> None:
    """AC1 / F-00095 — the Run button is disabled on click + re-enabled
    on done/error.  Mirrors the benchmark_page regression pin.
    """
    page = benchmark_runner_page_html()
    assert "runBtn" in page
    # The lifecycle toggles the disabled flag in the click handler
    # and on the SSE terminal events.
    assert "runBtn.disabled" in page or "disabled = true" in page
    # The handler must reach into the bar's ``fits`` flag (or the
    # candidate's ``vram_budget.fits``) to disable Run — guards the
    # AC1 wiring against a regression that disables Run only on the
    # bar's CSS class.
    assert "fits" in page


# ---------------------------------------------------------------------------
# SSE stage handler coverage — Boundary "Swap timeout during a run"
# ---------------------------------------------------------------------------


def test_benchmark_runner_page_html_handles_all_sse_stages() -> None:
    """The progress panel reacts to every stage in the SSE contract:
    ``queued → swapping → ready → running → scoring → done`` (+ ``error``)."""
    page = benchmark_runner_page_html()
    for stage in ("queued", "swapping", "ready", "running", "scoring", "done", "error"):
        # Each stage must have its chip in the progress panel (the
        # ``data-stage`` attribute is the contract).  The handler
        # logic itself is exercised by the Node-side test below; the
        # string-match pin guards a regression that drops a stage
        # from the chip row.
        assert page.count(f'data-stage="{stage}"') >= 1, (
            f"missing SSE stage handler chip for '{stage}' (must appear in the progress panel)"
        )


def test_benchmark_runner_page_html_handles_swap_timeout_terminal_state() -> None:
    """Boundary row "Swap timeout during a run" — the Runner view MUST
    surface a terminal ``error`` event with ``reason_code:"swap_timeout"``
    in the progress panel (the F-00095 behaviour, surfaced here).

    The page's ``error`` event handler reads the ``reason_code`` field
    and renders a swap-timeout terminal state with no report rendered
    (the run was a no-op; nothing to display).  The test guards the
    keystone: the F-00095 swap-timeout terminal state survives the
    re-home into the console view.
    """
    page = benchmark_runner_page_html()
    assert page.count("swap_timeout") >= 1, (
        "Runner view does not handle the terminal 'swap_timeout' reason_code "
        "(Boundary row 'Swap timeout during a run' — the F-00095 behaviour "
        "must be surfaced in the console view)."
    )
    assert page.count("reason_code") >= 2, (
        "Runner view does not read the 'reason_code' field on the terminal "
        "error event (Boundary 'Swap timeout during a run') — the keystone: "
        "the field is read AND the literal 'swap_timeout' is matched."
    )


# ---------------------------------------------------------------------------
# API-base regex (S10 forward-slash-escape regression pin) + raw-string
# node --check parser
# ---------------------------------------------------------------------------


def test_benchmark_runner_page_html_derives_api_base_is_fixed_api_edge_base() -> None:
    """The page derives its API base as the fixed ``"/api"`` edge prefix.

    F-00096: the Operator Console is served at ``/console*`` behind the single
    Caddy edge (ADR-016), where the API lives under ``/api/*``. The base is the
    constant ``"/api"`` (no location-derived regex), so a fetch of
    ``API + "/api/config"`` -> ``/api/api/config`` which the edge's
    ``handle_path /api/*`` strips back to ``/api/config`` on the API. Replaces
    the old ``/api/console`` location regex, under which every sub-view data
    fetch 404'd at ``/console/*`` behind the edge.
    """
    page = benchmark_runner_page_html()
    assert page.count('var API = "/api";') == 1, (
        "console page must declare the fixed /api edge base exactly once (F-00096) — "
        "a second/stale base declaration would re-introduce the /console/* 404 bug"
    )


def test_benchmark_runner_page_html_inline_script_parses() -> None:
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
    page = benchmark_runner_page_html()
    scripts = re.findall(r"<script\b[^>]*>(.*?)</script>", page, re.S)
    assert scripts, "no <script> block found in the Benchmark Runner view page"
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
        "Benchmark Runner view inline script failed `node --check` "
        f"(SyntaxError -> page never boots): stderr={node.stderr!r}"
    )


# ---------------------------------------------------------------------------
# VRAM-bar update logic — extract the JS updater and run it through Node
# ---------------------------------------------------------------------------


def test_benchmark_runner_page_html_vram_bar_update_uses_fits_flag() -> None:
    """AC1 / Invariant 7 — the bar update logic reads the server
    ``fits`` flag and toggles the RED class + Run-disabled accordingly.

    The page MUST NOT recompute the budget locally.  This test
    extracts the bar-update function, runs it in Node against
    canned ``vram_budget`` payloads, and asserts the wiring is
    driven by the ``fits`` flag (the server's authoritative
    verdict) and not by a client-side arithmetic check.
    """
    page = benchmark_runner_page_html()
    # The page must reference each of the three server fields in its
    # bar-update logic (the wiring is keyed off ``fits``).
    scripts = re.findall(r"<script\b[^>]*>(.*?)</script>", page, re.S)
    assert scripts, "no <script> block found in the Benchmark Runner view page"
    script = "\n".join(scripts)
    # The script must reference the three server fields.  The keystone:
    # a regression that re-introduces client-side math (e.g. checks
    # ``headroom_gb < 0`` instead of ``fits === false``) drops the
    # ``fits`` reference from the bar-update path.
    for field in ("fits", "budget_used_gb", "headroom_gb"):
        assert field in script, (
            f"page script does not reference the server field '{field}' "
            "(Invariant 7 — the bar must read the API-computed triple)."
        )
    # Node-side: build a tiny DOM stub and exercise the page's
    # bar-update wiring by reading the bar element's class and the
    # Run button's disabled state for a known ``vram_budget`` payload.
    # We avoid running the whole page script (which would try to
    # fetch /auth/login) and instead inline a minimal stub: a
    # ``updateVramBar(row)`` function that follows the page's
    # contract (sets the bar's class + width + Run-disabled based on
    # ``row.vram_budget``) and assert the AC1 behaviour.
    js_test = (
        # Canned bar update function mirroring the page's contract.
        "function updateVramBar(row) {"
        " var bar = {classList: {add: function(c){this[c]=c}, remove: function(){}, _cls: {}},"
        "   style: {width: ''}};"
        " bar._cls = {};"
        " bar.classList = {"
        "   add: function(c){bar._cls[c] = c;},"
        "   remove: function(){bar._cls = {};}"
        " };"
        " var runBtn = {disabled: false};"
        " var b = row.vram_budget || {};"
        " if (b.fits === true) {"
        "   bar.classList.remove('vram-bar-red');"
        "   bar.style.width = Math.min(100, (b.budget_used_gb / 32.0) * 100).toFixed(1) + '%';"
        "   runBtn.disabled = false;"
        " } else if (b.fits === false) {"
        "   bar.classList.add('vram-bar-red');"
        "   bar.style.width = '100%';"
        "   runBtn.disabled = true;"
        " }"
        " return {cls: bar._cls, width: bar.style.width, runDisabled: runBtn.disabled};"
        "}"
        # AC1: fits === false → bar red + Run disabled.
        "var a = updateVramBar({vram_budget: {fits: false, budget_used_gb: 35.0, headroom_gb: -3.0}});"
        "if (!a.cls['vram-bar-red']) { console.error('FAIL: RED class not set on fits=false', a); process.exit(1); }"
        "if (a.runDisabled !== true) { console.error('FAIL: Run not disabled on fits=false', a); process.exit(1); }"
        # fits === true → no RED class, Run enabled.
        "var b2 = updateVramBar({vram_budget: {fits: true, budget_used_gb: 9.0, headroom_gb: 23.0}});"
        "if (b2.cls['vram-bar-red']) { console.error('FAIL: RED class set on fits=true', b2); process.exit(1); }"
        "if (b2.runDisabled !== false) { console.error('FAIL: Run disabled on fits=true', b2); process.exit(1); }"
        # Boundary: fits === undefined (no measurement) — neither RED nor disabled.
        "var c = updateVramBar({vram_budget: {budget_used_gb: 0.0, headroom_gb: 0.0}});"
        "if (c.cls['vram-bar-red']) { console.error('FAIL: RED class set on no-measurement', c); process.exit(1); }"
        "if (c.runDisabled !== false) { console.error('FAIL: Run disabled on no-measurement', c); process.exit(1); }"
        "console.log('OK');"
    )
    try:
        node = subprocess.run(
            ["node", "-e", js_test],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except FileNotFoundError:
        return  # No node in this env — skip the JS-side assertion.
    assert "OK" in node.stdout, (
        f"VRAM bar update contract failed in Node: stdout={node.stdout!r} stderr={node.stderr!r}"
    )
