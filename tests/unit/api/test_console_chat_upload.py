"""Unit tests for the native Chat + Upload console views.

Chat (`console_chat_page.py`) and Upload (`console_upload_page.py`) are native
console views built on the shared `console_shell.render_console_page` chrome:
one `/auth/login` sessionStorage bearer token, the unified nav, no iframe, no
second frontend.  They talk directly to the existing HTTP APIs:

* **Chat**   → ``POST /query`` (non-streaming, for the full ``diagnostics``),
  with a grounding (assist/strict) toggle, markdown answer, 4-state trust badge,
  clickable citation downloads, a collapsible provenance panel (verdict / mode /
  guard trail / sources), a timing timeline, and a Phoenix trace link.
* **Upload** → ``GET/POST /documents`` + ``DELETE /documents/{id}`` +
  ``GET /documents/{id}/raw``, with drag-and-drop multi-upload, a sortable +
  client-paginated (25/page) documents table.

Keystone invariants pinned here: shared console chrome, single login (one
``/auth/login``, ``iw_token``), no iframe / no Chainlit SSO bridge, local-only
(no external CDN), and valid inline JS (the page must boot).
"""

from __future__ import annotations

import re
import subprocess
import tempfile

from iw_rag.api.console_chat_page import console_chat_page_html
from iw_rag.api.console_upload_page import console_upload_page_html

_FORBIDDEN_HOSTS = (
    "fonts.googleapis.com",
    "fonts.gstatic.com",
    "cdn.jsdelivr.net",
    "unpkg.com",
    "cdnjs.cloudflare.com",
)

_NAV_HREFS = (
    "/console/config",
    "/console/benchmark",
    "/console/results",
    "/console/scoreboard",
    "/console/chat",
    "/console/upload",
)


def _assert_console_chrome(page: str) -> None:
    """Assert the shared console chrome contract (brand, nav, single login)."""
    assert page.lstrip().lower().startswith("<!doctype html>")
    # Brand palette + dark theme + the header logo wordmark. The wordmark renders
    # as split, coloured initials, so the literal "Innovation Ways" substring is
    # not present; assert the header logo span. The tab <title> is the short,
    # uniform "IW - RAG" form (the long brand must not return to the tab).
    assert "#0D9488" in page and "#2DD4BF" in page and "#181D29" in page
    assert page.count('data-theme="dark"') == 1
    assert '<span class="i">I</span>nnovation <span class="w">W</span>ays' in page
    assert "<title>IW - RAG</title>" in page
    assert "Innovation Ways" not in page
    # No external CDN / hosted font (Invariant 8).
    lowered = page.lower()
    for host in _FORBIDDEN_HOSTS:
        assert host not in lowered, f"forbidden external host {host!r} present"
    assert "http://localhost" not in page
    assert "http://127.0.0.1" not in page
    # Fixed "/api" edge base; single sign-in (one /auth/login), sessionStorage token.
    assert 'var API = "/api"' in page
    assert page.count("/auth/login") == 1
    assert "sessionStorage" in page and "iw_token" in page
    # The unified six-destination nav.
    for href in _NAV_HREFS:
        assert f'href="{href}"' in page, f"nav must include href='{href}'"
    assert "highlightActiveNav" in page
    # No second-frontend bridges.
    assert "<iframe" not in page, "native console views must not embed an <iframe>"
    assert "/iw/session-token" not in page, "single login: no Chainlit SSO bridge"


def _assert_inline_script_parses(page: str) -> None:
    """Assert every inline <script> block is syntactically valid JS."""
    scripts = re.findall(r"<script\b[^>]*>(.*?)</script>", page, re.S)
    assert scripts, "no <script> block found"
    script = "\n".join(scripts)
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as fh:
        fh.write(script)
        path = fh.name
    try:
        node = subprocess.run(["node", "--check", path], capture_output=True, text=True, timeout=10)
    except FileNotFoundError:
        return  # No node in this env — skip the JS-side assertion.
    assert node.returncode == 0, (
        f"inline script failed `node --check` (SyntaxError -> page never boots): "
        f"stderr={node.stderr!r}"
    )


# ---------------------------------------------------------------------------
# Chat view (`/console/chat`)
# ---------------------------------------------------------------------------


def test_console_chat_page_html_returns_non_empty_string() -> None:
    """The chat view factory returns a non-empty HTML string."""
    page = console_chat_page_html()
    assert isinstance(page, str)
    assert page.lstrip().lower().startswith("<!doctype html>")


def test_console_chat_page_html_has_console_chrome() -> None:
    """The chat view carries the shared console chrome + single login + nav."""
    page = console_chat_page_html()
    _assert_console_chrome(page)
    assert page.count('id="navChat"') == 1, "the Chat nav item must be marked active exactly once"


def test_console_chat_page_html_uses_non_streaming_query_for_diagnostics() -> None:
    """Chat calls ``POST /query`` (non-streaming) so the full diagnostics are
    available for the provenance + timing panels (the SSE final event drops them)."""
    page = console_chat_page_html()
    assert "/query" in page, "chat must call the /query endpoint"
    assert "/query/stream" not in page, (
        "chat must use non-streaming /query (the stream final event lacks diagnostics)"
    )


def test_console_chat_page_html_has_grounding_toggle() -> None:
    """Chat exposes an assist/strict grounding toggle wired to allow_ungrounded."""
    page = console_chat_page_html()
    assert page.count('id="assistToggle"') == 1, (
        "chat must have exactly one grounding (assist/strict) toggle element"
    )
    assert "allow_ungrounded" in page, "the toggle must drive allow_ungrounded"


def test_console_chat_page_html_ports_old_chat_features() -> None:
    """Chat ports the rich features: citation downloads, provenance (guard trail),
    timing timeline, and the Phoenix trace link."""
    page = console_chat_page_html()
    assert "download_url" in page, "citations must render the server-minted download link"
    assert "guard_trail" in page, "provenance panel must render the guard trail"
    assert "timeline" in page, "timing panel must render the per-step timeline"
    # Phoenix trace link → <browser-host>:6006/projects (built exactly once).
    assert page.count(":6006") == 1, "must link to Phoenix at the :6006 host exactly once"
    assert "/projects" in page, "Phoenix link must target the /projects view"


def test_console_chat_page_html_inline_script_parses() -> None:
    """The chat view's inline JS must be syntactically valid (boots the page)."""
    page = console_chat_page_html()
    _assert_inline_script_parses(page)
    # A bootable view defines exactly one onConsoleReady entrypoint for the shell
    # to invoke post-login (the shell only references/calls it, never defines it).
    assert page.count("window.onConsoleReady = function") == 1, (
        "the chat view must define exactly one onConsoleReady entrypoint"
    )


# ---------------------------------------------------------------------------
# Upload view (`/console/upload`)
# ---------------------------------------------------------------------------


def test_console_upload_page_html_returns_non_empty_string() -> None:
    """The upload view factory returns a non-empty HTML string."""
    page = console_upload_page_html()
    assert isinstance(page, str)
    assert page.lstrip().lower().startswith("<!doctype html>")


def test_console_upload_page_html_has_console_chrome() -> None:
    """The upload view carries the shared console chrome + single login + nav."""
    page = console_upload_page_html()
    _assert_console_chrome(page)
    assert page.count('id="navUpload"') == 1, (
        "the Upload nav item must be marked active exactly once"
    )


def test_console_upload_page_html_is_native_document_manager() -> None:
    """Upload talks to the ``/documents`` API directly (list/upload/delete/raw),
    multipart upload, no iframe embed."""
    page = console_upload_page_html()
    assert page.count("/documents") >= 2, "upload must call the /documents API"
    assert "/raw" in page, "upload must offer a raw download"
    assert "FormData" in page, "upload must POST multipart/form-data"
    assert "specific" in page and "common" in page, "visibility choices present"


def test_console_upload_page_html_has_dragdrop_sort_pagination() -> None:
    """Upload ports the new affordances: drag-and-drop multi-upload, sortable
    headers, client-side pagination, and a non-wrapping Visibility column."""
    page = console_upload_page_html()
    assert "multiple" in page, "the file input must accept multiple files"
    assert "drop" in page.lower(), "must have a drag-and-drop dropzone"
    assert "sortable" in page, "table headers must be sortable"
    assert page.count("var PAGE_SIZE = 25;") == 1, "client-side pagination must be 25 per page"
    assert "nowrap" in page, "the Visibility column must not wrap ('Only me')"


def test_console_upload_page_html_inline_script_parses() -> None:
    """The upload view's inline JS must be syntactically valid (boots the page)."""
    page = console_upload_page_html()
    _assert_inline_script_parses(page)
    # A bootable view defines exactly one onConsoleReady entrypoint for the shell
    # to invoke post-login (the shell only references/calls it, never defines it).
    assert page.count("window.onConsoleReady = function") == 1, (
        "the upload view must define exactly one onConsoleReady entrypoint"
    )


def test_console_upload_page_html_has_view_button_for_browser_renderable_types() -> None:
    html = console_upload_page_html()
    assert 'data-view="' in html
    assert "function viewDoc(" in html or "async function viewDoc(" in html
    # Opened in a new tab from a typed blob, not by navigating the console.
    assert 'window.open("", "_blank")' in html


def _extract_preview_type_fn() -> str:
    pattern = r"function previewType\(mime, filename\) \{.*?\n  \}\n"
    m = re.search(pattern, console_upload_page_html(), re.S)
    assert m, "previewType() not found in upload page script"
    return m.group(0)


def test_preview_type_allows_only_browser_safe_types() -> None:
    import json
    import shutil

    node = shutil.which("node")
    if node is None:
        return  # No node in this env — skip the JS-side assertion.
    cases = {
        ("application/pdf", "a.pdf"): "application/pdf",
        ("image/png", "a.png"): "image/png",
        ("text/plain", "a.txt"): "text/plain;charset=utf-8",
        ("text/markdown", "a.md"): "text/plain;charset=utf-8",
        ("application/octet-stream", "a.csv"): "text/plain;charset=utf-8",
        ("text/html", "a.html"): "",
        ("image/svg+xml", "a.svg"): "",
        ("application/vnd.openxmlformats-officedocument.wordprocessingml.document", "a.docx"): "",
        ("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "a.xlsx"): "",
        ("", ""): "",
    }
    script = (
        _extract_preview_type_fn()
        + f"const cases = {json.dumps([[list(k), v] for k, v in cases.items()])};"
        + "const bad = cases.filter(([a, want]) => previewType(a[0], a[1]) !== want);"
        + "console.log(JSON.stringify(bad));"
    )
    out = subprocess.run([node, "-e", script], capture_output=True, text=True, timeout=10)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]", f"previewType mismatches: {out.stdout}"


def _run_view_doc(scenario: str) -> dict[str, object]:
    """Run the page's real ``viewDoc`` in node against stubbed window/fetch/toast."""
    import json
    import shutil

    node = shutil.which("node")
    if node is None:
        return {}
    html = console_upload_page_html()
    m = re.search(r"async function viewDoc\(id, type\) \{.*?\n  \}\n", html, re.S)
    assert m, "viewDoc() not found in upload page script"
    harness = f"""
    var _API = "/api", _ah = function () {{ return {{ Authorization: "Bearer t" }}; }};
    var log = {{ toasts: [], closed: false, href: null, fetched: null, blobType: null }};
    var _toast = function (msg, err) {{ log.toasts.push([msg, !!err]); }};
    var scenario = {json.dumps(scenario)};
    var tab = {{ close: function () {{ log.closed = true; }}, location: {{}} }};
    Object.defineProperty(tab.location, "href", {{ set: function (v) {{ log.href = v; }} }});
    var window = {{ open: function () {{ return scenario === "blocked" ? null : tab; }} }};
    var fetch = async function (url, opts) {{
      log.fetched = [url, opts.headers.Authorization];
      var status = scenario === "401" ? 401 : scenario === "404" ? 404 : 200;
      var blob = async function () {{ return new Blob(["x"]); }};
      return {{ status: status, ok: status === 200, blob: blob }};
    }};
    URL.createObjectURL = function (b) {{ log.blobType = b.type; return "blob:stub"; }};
    URL.revokeObjectURL = function () {{}};
    var setTimeout = function () {{}};  // the real 60 s revoke timer would keep node alive
    {m.group(0)}
    viewDoc("abc 1", "application/pdf").then(function () {{ console.log(JSON.stringify(log)); }});
    """
    out = subprocess.run([node, "-e", harness], capture_output=True, text=True, timeout=10)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout.strip().splitlines()[-1])  # type: ignore[no-any-return]


def test_view_doc_opens_typed_blob_in_new_tab_with_bearer() -> None:
    log = _run_view_doc("ok")
    if not log:
        return  # No node in this env.
    assert log["fetched"] == ["/api/documents/abc%201/raw", "Bearer t"]
    assert log["blobType"] == "application/pdf"
    assert log["href"] == "blob:stub"
    assert log["closed"] is False


def test_view_doc_failures_close_the_tab_and_toast() -> None:
    for scenario, fragment in (("401", "Session expired"), ("404", "Cannot open document (404)")):
        log = _run_view_doc(scenario)
        if not log:
            return
        assert log["closed"] is True
        assert log["href"] is None
        assert fragment in str(log["toasts"]), (scenario, log)


def test_view_doc_popup_blocked_does_not_fetch() -> None:
    log = _run_view_doc("blocked")
    if not log:
        return
    assert log["fetched"] is None
    assert "Popup blocked" in str(log["toasts"])


def test_upload_actions_column_fits_view_download_delete() -> None:
    html = console_upload_page_html()
    m = re.search(r"\.up-col-act\s*\{\s*width:\s*(\d+)px", html)
    assert m, "actions column width not found"
    # View + Download + Delete need ~210px; 148px (two buttons) overlapped the Added column.
    assert int(m.group(1)) >= 210
    assert re.search(r"\.up-act\s*\{[^}]*flex-wrap:\s*wrap", html)
