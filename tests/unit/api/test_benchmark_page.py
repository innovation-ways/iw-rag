"""Unit tests for the F-00095 Benchmark Studio page module.

The Studio page is a self-contained inline HTML+CSS+vanilla-JS page
mirroring :mod:`iw_rag.api.documents_page`.  These tests pin the
behavioural contract that is verifiable from Python: the page string
returned by :func:`iw_rag.api.benchmark_page.benchmark_page_html` must
carry every section anchor the F-00095 Feature Design and S02/S03
contract require (per-step model pickers, params panel, dataset picker,
Run Full Test button, live SSE progress panel, report view, A/B compare
view), and the page must follow the same operational keystone as
:mod:`iw_rag.api.documents_page` (location-derived API base, no
external CDN, ``/auth/login`` self-handled sign-in, sessionStorage-only
bearer token).

The page content itself is asserted via these anchors; the S05
integration suite verifies the live ``GET /benchmark/ui`` route returns
200 HTML containing the same markers.  No JS engine is spun up here:
the page is structurally checked, then the S10 browser-verification
step exercises the interactive behaviour in a real browser.
"""

from __future__ import annotations

from iw_rag.api.benchmark_page import benchmark_page_html


def test_benchmark_page_html_returns_non_empty_string() -> None:
    """The page factory returns a non-empty HTML string."""
    page = benchmark_page_html()
    assert isinstance(page, str)
    assert page.strip() != ""
    # Top-level doctype + html tag — the page is real HTML, not just text.
    assert page.lstrip().lower().startswith("<!doctype html>")


def test_benchmark_page_html_contains_brand_palette() -> None:
    """The IW teal/ink palette tokens are present (branding keystone)."""
    page = benchmark_page_html()
    assert "#0D9488" in page  # teal-deep
    assert "#2DD4BF" in page  # teal
    assert "#181D29" in page  # ink
    assert page.count("#0D9488") == 1


def test_benchmark_page_html_has_no_external_cdn() -> None:
    """Invariant 7 — no external CDN / Google Fonts link (local-only rule)."""
    page = benchmark_page_html().lower()
    # The page must not pull a stylesheet or font from any third party.
    for forbidden in (
        "fonts.googleapis.com",
        "fonts.gstatic.com",
        "cdn.jsdelivr.net",
        "unpkg.com",
        "cdnjs.cloudflare.com",
    ):
        assert forbidden not in page, (
            f"Studio page must not reference external CDN '{forbidden}' (Invariant 7)."
        )


def test_benchmark_page_html_derives_api_base_from_location() -> None:
    """Invariant 6 — the API base is derived from ``window.location`` (not hardcoded)."""
    page = benchmark_page_html()
    # The Studio lives at /benchmark/ui (or /api/benchmark/ui behind Caddy) and
    # must strip that suffix from `location.pathname` to derive the API base.
    assert "location.pathname" in page
    assert "/benchmark/ui" in page
    # The page must NOT hardcode a host:port pair.
    assert "http://localhost" not in page
    assert "http://127.0.0.1" not in page


def test_benchmark_page_html_logs_in_via_auth_login() -> None:
    """The page self-handles sign-in via ``/auth/login`` (mirrors documents_page)."""
    page = benchmark_page_html()
    assert "/auth/login" in page
    # Session-memory bearer token (never persisted beyond the tab).
    assert "sessionStorage" in page
    assert page.count("/auth/login") == 1


def test_benchmark_page_html_has_per_step_model_pickers() -> None:
    """AC4 — four per-step model pickers (generator/embedder/reranker/grader)."""
    page = benchmark_page_html()
    for picker in ("pickGenerator", "pickEmbedder", "pickReranker", "pickGrader"):
        assert picker in page, f"missing picker id '{picker}' (AC4)"


def test_benchmark_page_html_loads_models_from_endpoint() -> None:
    """AC4 — the pickers are populated from ``GET /models`` (F-00094)."""
    page = benchmark_page_html()
    assert "/models" in page
    # Must tolerate a 503/404 (F-00094 not yet wired) without crashing — the
    # S03 prompt requires a "catalog unavailable" graceful-degrade state.
    assert "catalog" in page.lower() or "unavailable" in page.lower()


def test_benchmark_page_html_has_params_panel() -> None:
    """AC4 — params panel with the canonical knobs (top_k, temperature)."""
    page = benchmark_page_html()
    assert "top_k" in page
    assert "temperature" in page
    # The params panel carries the top_k input + label; the shared report
    # renderer (benchmark_render) also reads params.top_k for the intro
    # paragraph, so the token now appears more than twice — assert the panel
    # is present rather than pinning an exact count.
    assert page.count("top_k") >= 2


def test_benchmark_page_html_has_dataset_picker_defaulting_to_golden() -> None:
    """AC4 — dataset picker defaulting to ``benchmark_golden``."""
    page = benchmark_page_html()
    assert "pickDataset" in page
    assert "benchmark_golden" in page
    assert page.count("benchmark_golden") == 2


def test_benchmark_page_html_has_run_full_test_button() -> None:
    """AC5 — the Run Full Test button exists and is identified."""
    page = benchmark_page_html()
    assert "runBtn" in page
    assert "Run Full Test" in page
    assert page.count("Run Full Test") == 2


def test_benchmark_page_html_has_sse_progress_panel() -> None:
    """AC5 — live progress panel consuming the SSE stream from ``POST /benchmark/run``."""
    page = benchmark_page_html()
    assert "/benchmark/run" in page
    # The progress panel is anchored by a stable id so tests can assert it.
    assert "progressPanel" in page
    # The button is disabled during the run and re-enabled on done/error —
    # the implementation must touch ``disabled`` in the click handler.
    assert "disabled" in page
    assert page.count("/benchmark/run") == 4


def test_benchmark_page_html_handles_all_sse_stages() -> None:
    """The progress panel must react to every stage in the SSE contract:
    ``queued → swapping → ready → running → scoring → done`` (+ ``error``)."""
    page = benchmark_page_html()
    for stage in ("queued", "swapping", "ready", "running", "scoring", "done", "error"):
        assert stage in page, f"missing SSE stage handler for '{stage}'"


def test_benchmark_page_html_has_report_view() -> None:
    """The report view fetches ``GET /benchmark/reports/{id}`` and renders the metric tiers."""
    page = benchmark_page_html()
    assert "/benchmark/reports" in page
    # The three metric tiers are surfaced (when present).
    assert "tier" in page.lower() or "Tier" in page
    # model_load_ms is shown SEPARATELY from per-query latency (Invariant 4).
    assert "model_load_ms" in page
    # Judge κ is reported.
    assert "\u03ba" in page or "kappa" in page.lower() or "judge_kappa" in page


def test_benchmark_page_html_has_saved_reports_list_with_view_affordance() -> None:
    """A saved-reports list lets the operator open any persisted report's detail.

    Regression pin for F-00095 / S10: the report view (``#reportCard``) was
    originally reachable ONLY immediately after a run, so a seeded/historical
    report could never be opened in the UI — V4 (report view renders with
    fixture data) was unsatisfiable. The page must expose a reports list
    (``#reportsList``) whose per-row ``View`` control (``data-report-id``)
    drives the same ``loadReport`` render path.
    """
    page = benchmark_page_html()
    assert 'id="reportsList"' in page, "no saved-reports list container on the page"
    assert "data-report-id" in page, "no per-report View affordance (data-report-id) on the page"
    # The View affordance must drive the single-report render path.
    assert "loadReport(" in page
    assert page.count('id="reportsList"') == 1


def test_benchmark_page_html_has_ab_compare_view() -> None:
    """The A/B compare view picks two reports and renders the delta table + caveat."""
    page = benchmark_page_html()
    assert "/benchmark/compare" in page
    # Significance flag + winner + the n≈37 underpowered caveat are surfaced.
    assert "p_value" in page or "p-value" in page
    assert "winner" in page.lower()
    # The underpowered caveat is prominent (R-00121 §4.2) — surfaced verbatim
    # in the rendered template.
    assert "underpower" in page.lower() or "n\u224837" in page or "n\u2272" in page


def test_benchmark_page_html_marks_run_button_disable_lifecycle() -> None:
    """AC5 — the button is disabled immediately on click and re-enabled on done/error.

    The literal "disabled" attribute is set on the run button when the user
    clicks; the click handler must be wired to the ``runBtn`` id.
    """
    page = benchmark_page_html()
    # The button element with id runBtn must be present and a click handler must
    # exist that toggles the disabled flag.
    assert "runBtn" in page
    assert "runBtn.disabled" in page or "disabled = true" in page


def test_benchmark_page_html_derives_api_base_with_evaluable_js_regex() -> None:
    """The API-base regex must be a valid JavaScript regex literal AND must
    correctly strip ``/benchmark/ui`` from the page's own path (Invariant 6
    + a real-browser sanity check that a SyntaxError-free page actually
    derives the API base).

    Regression pin for the F-00095 / S10 fix cycle: the page shipped with
    a backslash-escaped regex (``/\\\\/benchmark\\\\/ui\\\\/?$/``) that
    failed JavaScript's regex parser with ``Invalid regular expression
    flags`` (the parser saw ``b`` as a flag char) and would have caused
    the entire page script to throw at load time, blocking V1..V6.  The
    fix is to escape the forward slashes (``\\/benchmark\\/ui\\/?$/``).
    This test guards against the regression by asserting the regex source
    is a valid ``vm.Script``-parseable JS literal that matches the
    expected paths.
    """
    import re
    import subprocess

    page = benchmark_page_html()
    m = re.search(r"var API = location\.pathname\.replace\((.+?), \"\"\);", page)
    assert m is not None, "API-base regex assignment not found in page"
    regex_source = m.group(1).strip()
    # The bug pattern: the original regex was `/\\/benchmark\\/ui\\/?$/`
    # (4 backslashes inside a regex literal — each `\\` matches a literal
    # `\`, so the parser was looking for backslash separators in the URL
    # path and the regex closed early at the first `/`, leaving `b` as an
    # invalid flag — hence the SyntaxError).  The fix is to escape the
    # forward slashes (which are the regex literal delimiters) with
    # `\/`, so the correct source is `/\/benchmark\/ui\/?$/`.  The
    # distinguishing signal in the source text is the presence of `\\`
    # before `benchmark` and `ui` — that is the bug.  After the fix,
    # only `\/` (a single backslash followed by a forward slash —
    # escaping the delimiter) appears.
    assert "\\\\benchmark" not in regex_source, (
        "API-base regex still uses `\\\\` before `benchmark` (regression of the "
        "S10 /benchmark/ui SyntaxError bug — the parser was treating `b` as a flag)."
    )
    assert "\\\\ui" not in regex_source, (
        "API-base regex still uses `\\\\` before `ui` (regression of the S10 "
        "/benchmark/ui SyntaxError bug)."
    )
    assert "benchmark" in regex_source and "ui" in regex_source, (
        "API-base regex does not reference /benchmark/ui (the page lives at /benchmark/ui)."
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
                    "for (const [path, want] of [['/benchmark/ui',''],"
                    " ['/api/benchmark/ui','/api'],"
                    " ['/benchmark/ui/','']]) {"
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


def test_benchmark_page_html_renders_scalar_metric_values_from_report() -> None:
    """V4 — the report view must render real metric values (not em-dash).

    Regression pin for the F-00095 / S10 fix cycle: the page previously
    read all metric values from ``rep.metrics`` (the F-00093
    per-query-vector JSONB: ``{"ndcg_at_5": [0.5, 0.6, 0.7], ...}``)
    and passed the list to ``fmt``, which returned the em-dash
    placeholder because ``Number([...])`` is ``NaN``.  The report
    view rendered every metric as ``—`` and V4 silently failed (the
    previous qv-browser agents misdiagnosed the failure as a missing
    E2E stack).

    The fix surfaces the F-00093 promoted scalar columns as
    top-level API fields and the page's ``pick()`` helper prefers
    them (and mean-reduces a vector blob as a fallback).  This test
    extracts the pick() helper and runs it through Node.js to confirm:

    * a scalar top-level field renders the scalar;
    * a list-valued ``metrics.X`` field mean-reduces to the expected mean;
    * a missing field (Tier-3 omitted) returns ``null`` (em-dash).
    """
    import re
    import subprocess

    page = benchmark_page_html()
    # Find the shared function IW_pick(rep, keys) { ... } — match the
    # opening brace and then count braces to find the matching close.  The
    # renderer moved to ``benchmark_render.py`` (shared with the Console
    # Results view); ``IW_pick`` is the scalar-first / vector-fallback
    # helper that replaced the old per-page ``pick(name)``.
    open_match = re.search(r"function IW_pick\(rep, keys\) \{", page)
    assert open_match is not None, "IW_pick() helper not found in page (V4 rendering broken)"
    start = open_match.end()
    depth = 1
    i = start
    while i < len(page) and depth > 0:
        c = page[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        i += 1
    assert depth == 0, "could not find end of IW_pick() function"
    pick_source = page[open_match.start() : i]

    js_setup = (
        "const rep = {"
        " recall_at_5: 0.62, recall_at_10: 0.74, ndcg_at_5: 0.55,"
        " ndcg_at_10: 0.66, mrr: 0.58, faithfulness_mean: 0.78,"
        " context_recall: null, factual_correctness: null,"
        # ``ndcg_at_5`` is also in the blob as a list (the per-query
        # vector the F-00093 stats module uses for paired tests).  The
        # top-level scalar must take precedence; IW_pick must return the
        # scalar, not mean-reduce the list.  The blob now lives on
        # ``rep.metrics`` (IW_pick reads it from there).
        " metrics: {"
        "  ndcg_at_5: [0.5, 0.6, 0.7],"
        "  recall_at_5: [0.7, 0.8, 0.9],"
        "  mrr: [0.5, 0.6],"
        "  faithfulness: [0.8, 0.85, 0.9],"
        "  answer_relevancy: [0.7, 0.8, 0.9]"
        " }"
        "};"
        f"{pick_source}"
        # Top-level scalar (rep.X) MUST win over the blob's vector.
        "const out = {"
        " ndcg_at_5: IW_pick(rep, ['ndcg_at_5']),"
        " recall_at_5: IW_pick(rep, ['recall_at_5']),"
        " mrr: IW_pick(rep, ['mrr']),"
        " faithfulness: IW_pick(rep, ['faithfulness_mean', 'faithfulness']),"
        " context_recall: IW_pick(rep, ['context_recall']),"
        " missing: IW_pick(rep, ['totally_missing'])"
        "};"
    )
    js_asserts = [
        # The previous bug: pick() returned the LIST itself (not a
        # scalar), so fmt() rendered an em-dash.  Guard the regression
        # explicitly — pick() must return a scalar or null, never a list.
        "if (Array.isArray(out.ndcg_at_5))"
        "  { console.error('FAIL ndcg_at_5 is a list (V4 bug regression)');"
        "    process.exit(1); }",
        "if (out.ndcg_at_5 !== 0.55)"
        "  { console.error('FAIL ndcg_at_5 =', out.ndcg_at_5); process.exit(1); }",
        "if (out.recall_at_5 !== 0.62)"
        "  { console.error('FAIL recall_at_5 =', out.recall_at_5); process.exit(1); }",
        "if (out.mrr !== 0.58)  { console.error('FAIL mrr =', out.mrr); process.exit(1); }",
        "if (out.faithfulness !== 0.78)"
        "  { console.error('FAIL faithfulness =', out.faithfulness); process.exit(1); }",
        "if (out.context_recall !== null)"
        "  { console.error('FAIL context_recall =', out.context_recall); process.exit(1); }",
        "if (out.missing !== null)"
        "  { console.error('FAIL missing =', out.missing); process.exit(1); }",
        # When the top-level scalar is absent, IW_pick must mean-reduce
        # the vector blob.  ``answer_relevancy`` lives only in the blob.
        "const blobOnly = IW_pick(rep, ['answer_relevancy']);",
        "if (blobOnly === null || Array.isArray(blobOnly))"
        "  { console.error('FAIL blobOnly fallback =', blobOnly); process.exit(1); }",
        "if (Math.abs(blobOnly - 0.8) > 1e-9)"
        "  { console.error('FAIL blobOnly mean =', blobOnly, 'want 0.8'); process.exit(1); }",
        # Stub by removing the top-level field and re-evaluating ndcg_at_5.
        "delete rep.ndcg_at_5;",
        "const ndcgFromBlob = IW_pick(rep, ['ndcg_at_5']);",
        "if (Math.abs(ndcgFromBlob - 0.6) > 1e-9)"
        "  { console.error('FAIL ndcg_at_5 from blob =', ndcgFromBlob, 'want 0.6');"
        "    process.exit(1); }",
        "console.log('OK');",
    ]
    js_check = js_setup + "".join(js_asserts)

    try:
        node = subprocess.run(
            ["node", "-e", js_check],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except FileNotFoundError:
        return  # No node in this env — skip the JS-side assertion.

    assert "OK" in node.stdout, (
        f"pick() helper did not behave correctly in Node: "
        f"stdout={node.stdout!r} stderr={node.stderr!r}"
    )


def test_benchmark_page_html_inline_script_parses() -> None:
    """The page's entire inline ``<script>`` must be syntactically valid JS.

    Regression pin for the F-00095 / S10 fix cycle. The page is generated
    from a triple-quoted Python string; when that string was **not** raw,
    JS escape sequences written in the source were consumed by Python:

    * ``"event: <type>\\ndata: <json>\\n\\n"`` in a ``//`` comment expanded
      to real newlines, pushing ``data: <json>`` onto its own line as
      executable JS -> ``SyntaxError: Unexpected token '<'``.
    * ``"<span class=\\"muted\\">"`` collapsed its ``\\"`` to ``"``, ending
      the JS string early -> ``SyntaxError: Unexpected identifier 'muted'``.

    Either error makes the *whole* inline script fail to parse, so no page
    JS runs (the login view never shows, ``<main>`` is empty) — invisible
    to the string-match tests above but fatal in a real browser. This test
    extracts the full ``<script>`` body and runs ``node --check`` over it,
    which would have caught both defects. Skips when Node is unavailable.
    """
    import re
    import subprocess
    import tempfile

    page = benchmark_page_html()
    scripts = re.findall(r"<script\b[^>]*>(.*?)</script>", page, re.S)
    assert scripts, "no <script> block found in the Benchmark Studio page"
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
        "Benchmark Studio inline script failed `node --check` "
        f"(SyntaxError -> page never boots): stderr={node.stderr!r}"
    )
