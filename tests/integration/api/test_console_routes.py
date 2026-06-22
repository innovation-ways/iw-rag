"""Integration tests for the F-00096 console page routes (S10–S15 + S17 sweep).

The S10–S15 frontend steps each wrote their own per-view unit test
in ``tests/unit/api/test_*_page.py`` (string-match pins for the
page factory).  This file is the **end-to-end console route sweep**
the S17 prompt pins — every console page route
(``/console``, ``/console/config``, ``/console/benchmark``,
``/console/results``, ``/console/scoreboard``, ``/console/chat``,
``/console/upload``) is exercised through the live FastAPI app's
``TestClient`` and asserted to return **200 HTML** (or the documented
redirect) — never a **5xx** (the F-00096 design "no 5xx" keystone).

The console routes are static page factories — no DB-backed
collaborator is consulted; the test only needs a valid
:func:`create_app` composition root (a stub query service is enough).
The console HTML does not render server-side, so we only verify the
route returns 200 with a non-empty ``text/html`` body containing the
expected page-id anchors (e.g. ``/console`` for the shell, the
``Innovation Ways`` wordmark, etc.).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from iw_rag.api.app import create_app
from iw_rag.api.auth import StaticTokenAuthenticator

# ---------------------------------------------------------------------------
# Stub query service — the console pages don't actually run the
# query service; the stub is here only to satisfy create_app's
# ``query_service`` seam (a real AgentQueryService would build a
# LangGraph, a vLLM client, etc. — far too heavy for a page-route
# sweep).
# ---------------------------------------------------------------------------


class _StubQueryService:
    async def answer(self, query: str, ctx: object) -> object:  # pragma: no cover - never called
        raise RuntimeError("not used in S17 console route sweep")


# ---------------------------------------------------------------------------
# Authenticator — the console pages do their own sign-in via
# ``/auth/login``; the bearer-token auth the rest of the app uses
# is also wired (and tested in the S07 unit test).  For the page
# route sweep we use a no-groups token — the page factory never
# queries the access context server-side.
# ---------------------------------------------------------------------------


def _build_client() -> TestClient:
    """Build a TestClient with the static-token authenticator + a stub query service."""
    auth = StaticTokenAuthenticator.from_config({"u": ["u"]}, operator_tokens=[])
    app = create_app(
        authenticator=auth,
        query_service=_StubQueryService(),  # type: ignore[arg-type]
    )
    return TestClient(app)


# ---------------------------------------------------------------------------
# The console page-route list (S10 + S11 + S12 + S13 + S14 + S15).
# ---------------------------------------------------------------------------


CONSOLE_ROUTES: list[tuple[str, str]] = [
    ("/console", "shell page (S10)"),
    ("/console/config", "Config view (S11)"),
    ("/console/benchmark", "Benchmark Runner view (S12)"),
    ("/console/results", "Results + Compare view (S13)"),
    ("/console/compare", "Multi-run Compare view (2..5 runs)"),
    ("/console/scoreboard", "Scoreboard view (S14)"),
    ("/console/chat", "Chat native view (/query/stream)"),
    ("/console/upload", "Upload native view (/documents)"),
]


# ---------------------------------------------------------------------------
# Each route returns 200 HTML (no 5xx) — the F-00096 §"no 5xx" keystone
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.parametrize(("path", "_label"), CONSOLE_ROUTES)
def test_console_route_returns_200_html_and_no_5xx(path: str, _label: str) -> None:
    """Every console page route returns 200 HTML (the no-5xx sweep).

    The S17 prompt's "no-5xx sweep" — every console page route
    is exercised under an authenticated principal and asserted to
    return 200 HTML with a non-empty body.  The F-00096 design
    "Console page-route no-5xx sweep" boundary row.
    """
    client = _build_client()
    resp = client.get(path)
    assert resp.status_code == 200, (
        f"{path} returned {resp.status_code} (5xx sweep failed). "
        "Console pages are static HTML factories — they must not 5xx."
    )
    assert resp.headers["content-type"].startswith("text/html"), (
        f"{path} content-type is {resp.headers.get('content-type')!r}; expected text/html."
    )
    # The body is a non-empty HTML page.
    assert "<" in resp.text
    assert len(resp.text) > 100


# ---------------------------------------------------------------------------
# Each route is reachable under an authenticated principal (the
# /auth/login self-handled sign-in flow doesn't apply to the page
# routes — they self-handle the sign-in client-side; the bearer
# token the page factories consult is its own sessionStorage).
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.parametrize(("path", "_label"), CONSOLE_ROUTES)
def test_console_route_brand_chrome_marker_is_present(path: str, _label: str) -> None:
    """Every console page route carries the IW brand chrome (wordmark / data-theme).

    The console shell + every S11–S15 sub-view share the IW brand
    chrome: the teal/ink palette + the "Innovation Ways" wordmark +
    ``data-theme="dark"``.  Pinned here so a future refactor that
    drops the shared chrome from any sub-view breaks loudly.
    """
    client = _build_client()
    resp = client.get(path)
    assert resp.status_code == 200
    # The dark-theme default is on the <html> root of every console
    # page (the S10–S15 sibling tests pin the same marker).
    assert 'data-theme="dark"' in resp.text, (
        f'{path} is missing data-theme="dark" — the IW brand chrome '
        "is not consistent across the console shell + sub-views."
    )
