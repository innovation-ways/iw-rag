"""HTTP routes for the IW RAG Operator Console shell + sub-views (F-00096).

The S10 step owns the **shell** host page (``GET /console``); the
S11–S15 frontend-impl steps each add their own sub-view route
(``/console/config``, ``/console/benchmark``, ``/console/results``,
``/console/scoreboard``).  This module is the **single router** every
console HTML page mounts through (the S10 prompt directed
"extend an existing router OR create a small ``console_routes.py``;
follow whatever the S07/S08 reports established" — S07 established
the one-router-per-concern pattern, so all console pages live here).

The shell + every sub-view is a self-contained inline HTML+CSS+vanilla-
JS page that self-handles sign-in via ``/auth/login`` and is a thin
API client — all retrieval / ACL / grounding / guard logic stays
server-side (Invariant 4).  Each route is a thin wrapper that wires
the page factory into an :class:`HTMLResponse` (mirrors
:func:`iw_rag.api.benchmark_routes.benchmark_ui`).
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse

from iw_rag.api.benchmark_runner_page import benchmark_runner_page_html
from iw_rag.api.compare_page import compare_page_html
from iw_rag.api.config_page import config_page_html
from iw_rag.api.console_chat_page import console_chat_page_html
from iw_rag.api.console_upload_page import console_upload_page_html
from iw_rag.api.learn_page import learn_glossary_html, learn_hub_html, learn_lesson_html
from iw_rag.api.results_page import results_page_html
from iw_rag.api.scoreboard_page import scoreboard_page_html
from iw_rag.api.system_page import system_page_html
from iw_rag.learn.lessons import get_lesson

console_router = APIRouter()


@console_router.get("/console", include_in_schema=False)
async def console_ui() -> RedirectResponse:
    """Send ``/console`` to the Learn hub, the console's front door."""
    return RedirectResponse(url="/console/learn", status_code=307)


@console_router.get("/console/learn", response_class=HTMLResponse, include_in_schema=False)
async def console_learn_ui() -> HTMLResponse:
    """Serve the Learn hub: a beginner's map of the RAG pipeline."""
    return HTMLResponse(content=learn_hub_html())


@console_router.get("/console/learn/glossary", response_class=HTMLResponse, include_in_schema=False)
async def console_learn_glossary_ui() -> HTMLResponse:
    """Serve the Learn glossary."""
    return HTMLResponse(content=learn_glossary_html())


@console_router.get("/console/learn/{slug}", response_class=HTMLResponse, include_in_schema=False)
async def console_learn_lesson_ui(slug: str) -> HTMLResponse:
    """Serve one Learn lesson; 404 for an unknown step."""
    lesson = get_lesson(slug)
    if lesson is None:
        raise HTTPException(status_code=404, detail="Unknown lesson")
    return HTMLResponse(content=learn_lesson_html(lesson))


@console_router.get("/console/config", response_class=HTMLResponse, include_in_schema=False)
async def console_config_ui() -> HTMLResponse:
    """Serve the Config view page (S11 — live-edit config form).

    The S11 page owns the body — the per-key active-value table
    (with explanation, locked badge, pending-restart badge, edit
    form wired to bounds), the change-log panel anchor, the IW
    brand chrome, the same ``/auth/login`` self-handled sign-in,
    the location-derived API base, the sessionStorage-only bearer
    token.  This route just wires the response (mirrors
    :func:`iw_rag.api.benchmark_routes.benchmark_ui`).
    """
    return HTMLResponse(content=config_page_html())


@console_router.get("/console/benchmark", response_class=HTMLResponse, include_in_schema=False)
async def console_benchmark_ui() -> HTMLResponse:
    """Serve the Benchmark Runner view page (S12 — per-step pickers + VRAM bar + SSE).

    The S12 page owns the body — per-step model pickers (generator
    / embedder / reranker / top_k / mode) populated from
    ``GET /models``, the live **VRAM budget bar** (reading the
    server-computed ``vram_budget.{fits, budget_used_gb,
    headroom_gb}`` triple — the page NEVER recomputes the math
    locally, per Invariant 7), the ``POST /benchmark/run`` SSE
    consumer with the full stage set + the swap-timeout terminal
    state, the no-candidates empty state, the run-button disable
    lifecycle, the IW brand chrome, the same ``/auth/login``
    self-handled sign-in, the location-derived API base, the
    sessionStorage-only bearer token.  This route just wires the
    response (mirrors :func:`iw_rag.api.benchmark_routes.benchmark_ui`).
    """
    return HTMLResponse(content=benchmark_runner_page_html())


@console_router.get("/console/results", response_class=HTMLResponse, include_in_schema=False)
async def console_results_ui() -> HTMLResponse:
    """Serve the Results + Compare view page (S13 — report deep-dive + A/B compare).

    The S13 page owns the body — the report picker + deep-dive
    panel populated from ``GET /benchmark/reports`` /
    ``GET /benchmark/reports/{id}`` (with the same metric-tier /
    latency / VRAM / ``config_snapshot`` renderers as
    :mod:`iw_rag.api.benchmark_page`), the A/B compare picker +
    delta-table / winner / underpowered-caveat panel populated
    from ``GET /benchmark/compare`` (the F-00095
    ``benchmark_page.py`` shape reused verbatim so the two views
    stay consistent), the Boundary row coverage (empty-history
    empty state, 400 not-comparable for different-dataset
    compare, 404 not-found for unknown ids — all rendered in the
    panel, never as a toast that hides the cause), the IW brand
    chrome, the same ``/auth/login`` self-handled sign-in, the
    location-derived API base, the sessionStorage-only bearer
    token.  This route just wires the response (mirrors
    :func:`iw_rag.api.benchmark_routes.benchmark_ui`).
    """
    return HTMLResponse(content=results_page_html())


@console_router.get("/console/compare", response_class=HTMLResponse, include_in_schema=False)
async def console_compare_ui() -> HTMLResponse:
    """Serve the multi-run Compare view page (2..5 same-dataset runs).

    The page owns the body — the run picker (2..5 same-dataset runs +
    baseline), the comparison render populated from
    ``GET /benchmark/compare/multi`` (the legend with per-run palette
    colours, the aggregate per-metric bars, the baseline-relative delta
    tables, and the per-item heatmap + expandable answer/context diff),
    and the Generate Report (PDF) button hitting
    ``GET /benchmark/compare/multi/pdf``.  Same IW brand chrome,
    ``/auth/login`` self-handled sign-in, location-derived API base, and
    sessionStorage-only bearer token as the other console views.  This
    route just wires the response (mirrors :func:`console_results_ui`).
    """
    return HTMLResponse(content=compare_page_html())


@console_router.get("/console/scoreboard", response_class=HTMLResponse, include_in_schema=False)
async def console_scoreboard_ui() -> HTMLResponse:
    """Serve the Scoreboard view page (S14 — sortable per-metric leaderboards).

    The S14 page owns the body — the six-board segmented control
    (faithfulness / ndcg_at_5 / tok_per_s / latency /
    vram_efficiency / composite — the six S05 leaderboards), the
    sortable board table populated from
    ``GET /benchmark/scoreboard?board=<name>`` (the page preserves
    the S05 service's per-row order verbatim — Invariant 4: the
    page never recomputes ranking from raw metrics), the column-
    header client-side sort (the ``documents_page.py``
    ``th.sortable`` / ``setSort`` / ``refreshHeaderIndicators``
    pattern reused verbatim as display convenience over the
    already-ranked rows), the empty-history empty state
    (Boundary row "Empty report history"), the IW brand chrome,
    the same ``/auth/login`` self-handled sign-in, the
    location-derived API base, the sessionStorage-only bearer
    token.  This route just wires the response (mirrors
    :func:`iw_rag.api.benchmark_routes.benchmark_ui`).
    """
    return HTMLResponse(content=scoreboard_page_html())


@console_router.get("/console/chat", response_class=HTMLResponse, include_in_schema=False)
async def console_chat_ui() -> HTMLResponse:
    """Serve the **native** Chat view (streams ``/query/stream``).

    Chat is a native console view — NOT an iframe embed of a separate
    Chainlit frontend — so it shares the console's single ``/auth/login``
    sessionStorage bearer token (no second login).  The page owns the IW
    brand chrome, the six-destination nav, and a chat surface that POSTs to
    ``/query/stream`` and renders the SSE ``final`` / ``refusal`` / ``error``
    events (grounded badge + server-minted citation download links).  All
    retrieval / ACL / grounding / guard logic stays server-side
    (Invariant 4).
    """
    return HTMLResponse(content=console_chat_page_html())


@console_router.get("/console/upload", response_class=HTMLResponse, include_in_schema=False)
async def console_upload_ui() -> HTMLResponse:
    """Serve the **native** Upload / Document Manager view.

    Upload is a native console view — NOT an iframe embed — so it shares the
    console's single ``/auth/login`` sessionStorage bearer token (no second
    login).  The page owns the IW brand chrome, the six-destination nav, an
    upload form (file + visibility) that POSTs ``multipart/form-data`` to
    ``/documents``, and a table of the caller's own + common documents
    (``GET /documents``) with bearer-authenticated download
    (``/documents/{id}/raw``) and owner-only delete (``DELETE /documents/{id}``).
    The per-user document ACL is enforced server-side (WI-DOCMGMT, Invariant 4).
    """
    return HTMLResponse(content=console_upload_page_html())


@console_router.get("/console/system", response_class=HTMLResponse, include_in_schema=False)
async def console_system_ui() -> HTMLResponse:
    """Serve the System / Management view page (VRAM power switch).

    The System page owns the body — the live VRAM gauge (polled every
    2 s from ``GET /system/vram``), the mode banner (amber when
    benchmark mode or ``power_control_active`` is ``False``), the
    master ON/OFF power switch (``POST /system/power`` with
    ``{action, components:null}``), the per-component advanced
    expander with per-row start/stop buttons (``POST /system/power``
    with ``{action, components:[key]}``), the idle-unload / block-and-
    warm info line, and the 403 / degraded / sidecar-unreachable
    states.  All power-control and VRAM math stays server-side
    (Invariant 4).  The page is a thin API client: it never enforces
    ACL, VRAM budgeting, or component ordering locally.
    """
    return HTMLResponse(content=system_page_html())


__all__ = ["console_router"]
