"""HTTP routes for the Benchmark Studio (F-00095, S02).

Drives the run, lists and fetches reports, compares two reports on the
same dataset, and serves the Studio HTML page.  All routes share the
security keystone: groups come from the bearer token only (auth-only
ACL), never from the request body or query string (Invariant 3).

* ``POST /benchmark/run`` — runs the S01 orchestrator under the
  caller's :class:`~iw_rag.domain.acl.AccessContext`; returns an SSE
  stream of progress events whose terminal ``done`` carries the
  persisted ``report_id`` (AC1).  Gated to the ``eval-operator``
  group (AC2, read from auth only).
* ``GET /benchmark/reports`` — lists the persisted reports (newest
  first) via the F-00093 service on ``app.state``.
* ``GET /benchmark/reports/{report_id}`` — single report detail;
  404 when the id is unknown (Boundary row).
* ``GET /benchmark/compare?a=..&b=..`` — A/B delta table from the
  F-00093 stats module, with the winner flag (BOTH nDCG@5 AND
  faithfulness) and the n≈37 underpowered caveat (AC3).  400 across
  different datasets, 404 when an id is unknown.
* ``GET /benchmark/ui`` — serves the Studio HTML page (S03 owns the
  body; this route just wires the response).

The runner + report service resolve from ``app.state`` exactly like
``documents_routes.py`` resolves :class:`DocumentService`; the S02
composition root in :func:`iw_rag.api.app.create_app` binds the real
F-00093 / F-00094 collaborators (or test fakes) onto the state.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import HTMLResponse, StreamingResponse

from iw_rag.api.benchmark_page import benchmark_page_html
from iw_rag.api.routes import AccessCtx
from iw_rag.api.schemas import (
    BenchmarkCompareDeltaOut,
    BenchmarkCompareItemOut,
    BenchmarkCompareResponse,
    BenchmarkCompareRunOut,
    BenchmarkCompareSampleOut,
    BenchmarkMultiCompareResponse,
    BenchmarkReportOut,
    BenchmarkRunRequest,
)
from iw_rag.domain.acl import AccessContext
from iw_rag.domain.errors import AuthorizationError, NotFoundError, ValidationError

logger = logging.getLogger(__name__)

benchmark_router = APIRouter()


# ---------------------------------------------------------------------------
# ACL — require eval-operator (auth-only) — Invariant 3 / AC2
# ---------------------------------------------------------------------------


async def require_eval_operator(ctx: AccessCtx) -> AccessContext:
    """Require the ``eval-operator`` group in the caller's ACL (→ 403 otherwise).

    Mirrors :func:`iw_rag.api.routes.require_operator` — a thin ACL gate
    that reads groups from the resolved :class:`AccessContext` only
    (the keystone — never from the request body or query string).  The
    group is hard-coded to ``"eval-operator"`` (the F-00095 spec) and
    does NOT consult :attr:`Settings.serving.activation_group` (which
    governs ``POST /models/activate``): the Benchmark Studio gate is
    a fixed, narrower privilege; the activation gate is configurable.

    Args:
        ctx: The authenticated :class:`AccessContext` (from auth only).

    Returns:
        The same :class:`AccessContext` on success.

    Raises:
        AuthorizationError: When ``"eval-operator"`` is not present
            in ``ctx.principal_groups`` (→ 403).
    """
    if "eval-operator" not in ctx.principal_groups:
        raise AuthorizationError("'eval-operator' group required to run a benchmark")
    return ctx


EvalOperatorCtx = Annotated[AccessContext, Depends(require_eval_operator)]


# ---------------------------------------------------------------------------
# Collaborator resolution (mirrors documents_routes._service)
# ---------------------------------------------------------------------------


def _benchmark_runner(request: Request) -> Any:
    """Resolve the S01 runner from ``app.state`` (503 if unwired)."""
    runner: Any = getattr(request.app.state, "benchmark_runner", None)
    if runner is None:
        raise ValidationError(
            "benchmark runs are not enabled for this deployment "
            "(benchmark_runner is not wired on app.state)."
        )
    return runner


def _benchmark_service(request: Request) -> Any:
    """Resolve the F-00093 report service from ``app.state`` (503 if unwired)."""
    service: Any = getattr(request.app.state, "benchmark_service", None)
    if service is None:
        raise ValidationError(
            "benchmark reports are not enabled for this deployment "
            "(benchmark_service is not wired on app.state)."
        )
    return service


def _run_manager(request: Request) -> Any:
    """Resolve the singleton :class:`BenchmarkRunManager` from ``app.state``.

    The manager holds the current/most-recent live run so the console's
    progress panel survives a navigation (see
    :mod:`iw_rag.api.benchmark_run_state`).  ``create_app`` binds it
    unconditionally; this lazily creates one as a defensive fallback for
    app instances built before the binding existed (e.g. hand-rolled
    test apps).
    """
    manager: Any = getattr(request.app.state, "benchmark_run_manager", None)
    if manager is None:
        from iw_rag.api.benchmark_run_state import BenchmarkRunManager

        manager = BenchmarkRunManager()
        request.app.state.benchmark_run_manager = manager
    return manager


# ---------------------------------------------------------------------------
# SSE generator — mirrors iw_rag.api.routes._sse_generator
# ---------------------------------------------------------------------------


def _format_sse(event_type: str, data: Any) -> str:
    """Format one structured event into the canonical SSE chunk.

    The single source of the ``event: <type>\\ndata: <json>\\n\\n`` shape
    that :func:`iw_rag.api.routes._sse_generator` uses for
    ``/query/stream`` — used by both the live POST stream and the
    reconnect GET stream so the two never drift.
    """
    return f"event: {event_type}\ndata: {json.dumps(data, default=str)}\n\n"


async def _runner_events(
    runner: Any,
    model_config: dict[str, Any],
    ctx: AccessContext,
    trace_id: str,
) -> AsyncIterator[dict[str, Any]]:
    """Yield the runner's structured ``{"event", "data"}`` dicts, defensively.

    The defensive ``try/except`` around the runner's iterator catches any
    escape and yields a canonical terminal ``error`` event with ONLY
    ``{message, trace_ref}`` (never exception text, retrieved content, or
    PII — lld-api §error mapping).  The runner's own terminal ``error``
    event (e.g. with ``reason_code: "swap_timeout"``) passes through
    verbatim.  This is the structured source consumed by BOTH
    :func:`_run_sse_generator` (the formatted live stream) and the
    background run driver (:mod:`iw_rag.api.benchmark_run_state`).

    Args:
        runner: The S01 :class:`BenchmarkRunner` (real or fake).
        model_config: The submitted per-step model config (already
            validated by :class:`BenchmarkRunRequest`).
        ctx: The authenticated :class:`AccessContext` (eval-operator).
        trace_id: The request trace id (Phoenix correlation).

    Yields:
        Structured ``{"event": str, "data": dict}`` progress events.
    """
    try:
        async for event in runner.run(model_config, trace_id=trace_id, access_context=ctx):
            yield {
                "event": str(event.get("event", "data")),
                "data": event.get("data", {}),
            }
    except Exception:
        logger.exception("benchmark SSE stream failed (trace_id=%s)", trace_id)
        yield {
            "event": "error",
            "data": {
                "message": "benchmark run failed; see trace for details",
                "trace_ref": trace_id,
            },
        }


async def _run_sse_generator(
    runner: Any,
    model_config: dict[str, Any],
    ctx: AccessContext,
    trace_id: str,
) -> AsyncIterator[str]:
    """Yield SSE-formatted strings wrapping the S01 runner's event dicts.

    Mirrors :func:`iw_rag.api.routes._sse_generator` byte-for-byte:
    a ``: ping`` heartbeat first, then each runner event is wrapped as
    ``event: <type>\\ndata: <json>\\n\\n`` via :func:`_format_sse`, with
    the defensive error-shaping handled by :func:`_runner_events`.
    """
    yield ": ping\n\n"
    async for event in _runner_events(runner, model_config, ctx, trace_id):
        yield _format_sse(event["event"], event["data"])


async def _subscribe_sse(run: Any) -> AsyncIterator[str]:
    """Yield the canonical SSE stream for a :class:`LiveBenchmarkRun`.

    A ``: ping`` heartbeat first (mirroring :func:`_run_sse_generator`),
    then the run's replay buffer followed by its live events, each
    formatted via :func:`_format_sse`.  Used by both the live POST stream
    and the ``GET /benchmark/run/events`` reconnect stream — so a client
    that navigated away and back replays the full progress history and
    then continues live to the terminal event.
    """
    yield ": ping\n\n"
    async for event in run.subscribe():
        yield _format_sse(event["event"], event["data"])


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@benchmark_router.post("/benchmark/run")
async def run_benchmark(
    body: BenchmarkRunRequest,
    request: Request,
    ctx: EvalOperatorCtx,
) -> StreamingResponse:
    """Run one full benchmark and stream progress as SSE (eval-operator only).

    The route is gated to the ``eval-operator`` group (Invariant 3 /
    AC2).  The S01 runner drives activation → readiness → golden-set
    execution → scoring → persistence; each progress event is wrapped
    into the canonical SSE shape.  The terminal ``done`` event carries
    the persisted ``report_id`` (AC1); the terminal ``error`` event
    carries only ``{message, trace_ref}`` (mirroring
    :func:`iw_rag.api.routes._sse_generator`).

    Args:
        body: The per-step model config + params + dataset id.
        request: The raw FastAPI request (for app state + trace id).
        ctx: The authenticated :class:`AccessContext` (must contain
            ``eval-operator``, verified by the :data:`EvalOperatorCtx`
            dependency).

    Returns:
        A :class:`StreamingResponse` with
        ``media_type="text/event-stream"`` whose body is the SSE
        stream of the runner's progress events.
    """
    runner = _benchmark_runner(request)
    trace_id = getattr(request.state, "trace_id", "")
    # Resolve the logical dataset id ("benchmark_golden") to a real
    # eval_datasets uuid: the runner threads dataset_id into
    # assemble_report → uuid.UUID(...), so a name would raise. No-op when a
    # benchmark_service-less / DB-less deployment leaves the database unwired.
    dataset_id = body.dataset_id
    database = getattr(request.app.state, "database", None)
    if database is not None:
        from iw_rag.api.benchmark_composition import resolve_dataset_uuid

        dataset_id = await resolve_dataset_uuid(database, body.dataset_id)
    model_config: dict[str, Any] = {
        "generator": body.generator,
        "embedder": body.embedder,
        "reranker": body.reranker,
        "grader": body.grader,
        "dataset_id": dataset_id,
        "params": dict(body.params),
    }
    # Preserve the human dataset id the operator submitted (e.g.
    # "benchmark_golden") so the report label reads it instead of the resolved
    # uuid.  Only set when the submitted value was a name (it differs from the
    # resolved uuid); a uuid-in / uuid-out submission carries no human name.
    if body.dataset_id and body.dataset_id != dataset_id:
        model_config["dataset_name"] = body.dataset_id
    # Single-GPU: one run at a time.  Refuse a second start (400) rather
    # than racing two swaps; the operator can watch the in-flight run via
    # ``GET /benchmark/run/events``.
    manager = _run_manager(request)
    if manager.is_active():
        raise ValidationError(
            "a benchmark run is already in progress; watch it via the live stream"
        )
    # Drive the run in a background task decoupled from this request, so
    # a client disconnect / console navigation does NOT cancel the run.
    # This response subscribes to the run's replay buffer + live stream,
    # exactly like the ``GET /benchmark/run/events`` reconnect endpoint.
    run = await manager.start(
        trace_id=trace_id,
        event_source=_runner_events(runner, model_config, ctx, trace_id),
    )
    return StreamingResponse(
        _subscribe_sse(run),
        media_type="text/event-stream",
    )


@benchmark_router.get("/benchmark/run/events")
async def benchmark_run_events(request: Request, ctx: AccessCtx) -> StreamingResponse:
    """Re-subscribe to the current/most-recent benchmark run's SSE stream.

    The keystone of the "live progress survives navigation" fix: the
    Operator Console nav is full-page ``<a href>`` links, so leaving the
    Benchmark view tears down the page (and its ``POST /benchmark/run``
    SSE ``fetch``).  Because the run is driven by a background task and
    every event is buffered server-side (:mod:`benchmark_run_state`), a
    client returning to the view re-attaches here: the stream replays the
    full progress history (restoring the stage chips + detail line) and
    then continues live to the terminal event.

    Open to any authenticated principal (read-only viewing — the run gate
    on ``POST /benchmark/run`` is the operator-only one).  When no run has
    ever started, the stream is just the heartbeat and closes, so the
    page keeps its ``No run yet.`` state.

    Args:
        request: The raw FastAPI request (for app state).
        ctx: The authenticated :class:`AccessContext` (any principal).

    Returns:
        A :class:`StreamingResponse` (``text/event-stream``): the replay
        buffer + live tail of the current run, or just a heartbeat when
        there is no run.
    """
    del ctx  # auth-only gate; identity is not used to pick the run
    manager = _run_manager(request)
    run = manager.current
    if run is None:
        return StreamingResponse(_heartbeat_only_sse(), media_type="text/event-stream")
    return StreamingResponse(_subscribe_sse(run), media_type="text/event-stream")


async def _heartbeat_only_sse() -> AsyncIterator[str]:
    """A one-shot SSE body (just the heartbeat) for the no-run-yet case."""
    yield ": ping\n\n"


@benchmark_router.get("/benchmark/reports", response_model=list[BenchmarkReportOut])
async def list_benchmark_reports(
    request: Request,
    ctx: AccessCtx,
) -> list[BenchmarkReportOut]:
    """List persisted benchmark reports (newest first).

    Open to any authenticated principal — the report id list reveals
    nothing sensitive (the metric + label data is only visible after
    detail fetch, which the same auth keystone gates).  Reads through
    the F-00093 report service on ``app.state``.

    Args:
        request: The raw FastAPI request.
        ctx: The authenticated :class:`AccessContext` (any principal).

    Returns:
        A list of :class:`BenchmarkReportOut` rows.
    """
    service = _benchmark_service(request)
    rows = await service.list_reports()
    return [_to_report_out(r) for r in rows]


@benchmark_router.get("/benchmark/reports/{report_id}", response_model=BenchmarkReportOut)
async def get_benchmark_report(
    request: Request,
    ctx: AccessCtx,
    report_id: str,
) -> BenchmarkReportOut:
    """Return one benchmark report by id; 404 when the id is unknown.

    Args:
        request: The raw FastAPI request.
        ctx: The authenticated :class:`AccessContext` (any principal).
        report_id: The report's id (uuid-stringified).

    Returns:
        The :class:`BenchmarkReportOut` for the id.

    Raises:
        NotFoundError: When the id is not present in the F-00093
            repository (→ 404).
    """
    service = _benchmark_service(request)
    row = await service.get_report(report_id)
    if row is None:
        raise NotFoundError(f"benchmark report '{report_id}' not found")
    out = _to_report_out(row)
    # A regenerated conclusion (POST .../conclusion) overrides the immutable
    # run-time one for display.  The report row itself is never mutated.
    cache: dict[str, str] = getattr(request.app.state, "benchmark_conclusion_cache", {})
    if report_id in cache:
        metrics = dict(out.metrics)
        metrics["conclusion"] = {"text": cache[report_id], "basis": "regenerated"}
        out.metrics = metrics
    return out


@benchmark_router.post("/benchmark/reports/{report_id}/conclusion")
async def regenerate_conclusion(
    request: Request,
    ctx: AccessCtx,
    report_id: str,
) -> dict[str, Any]:
    """Regenerate the report's conclusion against the LATEST comparable runs.

    The report row is append-only, so the fresh conclusion is held in an
    in-memory override (``app.state.benchmark_conclusion_cache``) that the
    detail route prefers — the durable run-time conclusion in the report stays
    untouched.  Uses the same local judge writer as the run-time path.

    Args:
        request: The raw FastAPI request.
        ctx: The authenticated :class:`AccessContext` (any principal).
        report_id: The report to refresh.

    Returns:
        ``{"report_id": ..., "conclusion": <text>}``.

    Raises:
        NotFoundError: When the id is unknown.
        ValidationError: When the conclusion writer is unavailable or the LLM
            produced nothing.
    """
    del ctx
    service = _benchmark_service(request)
    row = await service.get_report(report_id)
    if row is None:
        raise NotFoundError(f"benchmark report '{report_id}' not found")
    from iw_rag.api.benchmark_composition import build_conclusion_fn

    settings = getattr(request.app.state, "settings", None)
    database = getattr(request.app.state, "database", None)
    fn = build_conclusion_fn(settings, database) if (settings and database) else None
    if fn is None:
        raise ValidationError("conclusion writer is not available in this deployment")
    text = await fn(row, str(row.get("dataset_id") or ""))
    if not text:
        raise ValidationError("conclusion generation produced no text (judge unavailable?)")
    cache = getattr(request.app.state, "benchmark_conclusion_cache", None)
    if cache is None:
        cache = {}
        request.app.state.benchmark_conclusion_cache = cache
    cache[report_id] = text
    return {"report_id": report_id, "conclusion": text}


@benchmark_router.get("/benchmark/compare", response_model=BenchmarkCompareResponse)
async def compare_benchmark_reports(
    request: Request,
    ctx: AccessCtx,
    a: str,
    b: str,
) -> BenchmarkCompareResponse:
    """A/B compare two persisted reports on the same dataset (AC3).

    Returns the per-metric delta table from the F-00093 stats module,
    a ``winner`` flag set only when one config is better on BOTH
    nDCG@5 AND faithfulness, and the n≪150 underpowered caveat.

    Args:
        request: The raw FastAPI request.
        ctx: The authenticated :class:`AccessContext` (any principal).
        a: The A-side report's id.
        b: The B-side report's id.

    Returns:
        A :class:`BenchmarkCompareResponse` with the delta table, the
        winner flag, and the underpowered caveat.

    Raises:
        NotFoundError: When either id is unknown (→ 404, Boundary row).
        ValidationError: When ``a`` and ``b`` reference different
            datasets (→ 400, paired stats require the same golden
            queries, Boundary row).
    """
    service = _benchmark_service(request)
    result = await service.compare(a, b)
    if result is None:
        raise NotFoundError("one or both report ids are unknown")
    return _to_compare_out(a, b, result)


def _parse_ids(ids: str) -> list[str]:
    """Split a comma-separated ``ids=`` query param into a clean id list."""
    return [s.strip() for s in (ids or "").split(",") if s.strip()]


@benchmark_router.get("/benchmark/compare/multi", response_model=BenchmarkMultiCompareResponse)
async def compare_benchmark_reports_multi(
    request: Request,
    ctx: AccessCtx,
    ids: str,
    baseline: str | None = None,
) -> BenchmarkMultiCompareResponse:
    """N-way (2..5) baseline-relative compare of reports on one dataset.

    The Compare view's data endpoint: returns the per-run identity +
    palette colours, the per-run baseline-relative delta tables, the
    per-item value matrix, and the per-item qualitative samples.

    Args:
        request: The raw FastAPI request.
        ctx: The authenticated :class:`AccessContext` (any principal).
        ids: Comma-separated report ids (2..5 after de-dup).
        baseline: The id to treat as the baseline (defaults to the first).

    Returns:
        A :class:`BenchmarkMultiCompareResponse`.

    Raises:
        NotFoundError: When any id is unknown (→ 404).
        ValidationError: When the run count is out of bounds, the reports
            span more than one dataset, or ``baseline`` is not among the
            ids (→ 400).
    """
    service = _benchmark_service(request)
    result = await service.compare_multi(_parse_ids(ids), baseline_id=baseline)
    if result is None:
        raise NotFoundError("one or more report ids are unknown")
    return _to_multi_compare_out(result)


def _compare_conclusion_cache(request: Request) -> dict[str, str]:
    """Resolve (lazily create) the per-comparison conclusion cache on app.state.

    Keyed by the comparison identity (sorted ids + baseline) so the
    conclusion the Compare view generated is reused verbatim by the PDF
    export (mirrors the single-report ``benchmark_conclusion_cache``).
    """
    cache: dict[str, str] | None = getattr(
        request.app.state, "benchmark_compare_conclusion_cache", None
    )
    if cache is None:
        cache = {}
        request.app.state.benchmark_compare_conclusion_cache = cache
    return cache


def _compare_cache_key(ids: list[str], baseline_id: str | None) -> str:
    """A stable cache key for a comparison (order-independent run set + baseline)."""
    return "|".join(sorted(set(ids))) + "::" + str(baseline_id or "")


async def _get_or_write_compare_conclusion(
    request: Request,
    payload: dict[str, Any],
    ids: list[str],
    baseline_id: str | None,
    *,
    refresh: bool = False,
) -> str | None:
    """Return the cached comparison conclusion, generating it on a miss.

    Uses the LOCAL judge (no hosted API) via the composition factory;
    returns ``None`` when the writer is unavailable in this deployment.
    """
    cache = _compare_conclusion_cache(request)
    key = _compare_cache_key(ids, baseline_id)
    if not refresh and key in cache:
        return cache[key]
    settings = getattr(request.app.state, "settings", None)
    if settings is None:
        return None
    from iw_rag.api.benchmark_composition import build_compare_conclusion_fn

    fn = build_compare_conclusion_fn(settings)
    if fn is None:
        return None
    text: str | None = await fn(payload)
    if text:
        cache[key] = text
    return text


@benchmark_router.get("/benchmark/compare/multi/conclusion")
async def compare_benchmark_reports_multi_conclusion(
    request: Request,
    ctx: AccessCtx,
    ids: str,
    baseline: str | None = None,
    refresh: bool = False,
) -> dict[str, Any]:
    """Local-LLM conclusion + reasoning for an N-way comparison.

    Grounded ONLY in the comparison's own numbers (per-run means +
    baseline-relative deltas + significance), written by the LOCAL judge
    model (never a hosted API).  Cached per comparison identity so the
    Compare view and the PDF export show the same text; ``refresh=true``
    forces a regeneration (the view's Regenerate button).

    Args:
        request: The raw FastAPI request.
        ctx: The authenticated :class:`AccessContext` (any principal).
        ids: Comma-separated report ids (2..5 after de-dup).
        baseline: The baseline id (defaults to the first).
        refresh: Force a regeneration instead of reading the cache.

    Returns:
        ``{"conclusion": <text or null>}`` — ``null`` when the local
        judge is unavailable in this deployment.

    Raises:
        NotFoundError: When any id is unknown (→ 404).
        ValidationError: When the run set is invalid (→ 400).
    """
    del ctx
    service = _benchmark_service(request)
    result = await service.compare_multi(_parse_ids(ids), baseline_id=baseline)
    if result is None:
        raise NotFoundError("one or more report ids are unknown")
    out = _to_multi_compare_out(result)
    text = await _get_or_write_compare_conclusion(
        request, out.model_dump(), _parse_ids(ids), out.baseline_id, refresh=refresh
    )
    return {"conclusion": text}


@benchmark_router.get("/benchmark/compare/multi/pdf", include_in_schema=False)
async def compare_benchmark_reports_multi_pdf(
    request: Request,
    ctx: AccessCtx,
    ids: str,
    baseline: str | None = None,
) -> Response:
    """Render a downloadable, branded PDF of the N-way comparison.

    Builds the self-contained comparison HTML
    (:mod:`iw_rag.api.benchmark_compare_pdf`) from the same payload the
    Compare view consumes and renders it to PDF via headless Chromium.

    Args:
        request: The raw FastAPI request.
        ctx: The authenticated :class:`AccessContext` (any principal).
        ids: Comma-separated report ids (2..5 after de-dup).
        baseline: The id to treat as the baseline (defaults to the first).

    Returns:
        A ``application/pdf`` response with a download filename.
    """
    del ctx
    service = _benchmark_service(request)
    result = await service.compare_multi(_parse_ids(ids), baseline_id=baseline)
    if result is None:
        raise NotFoundError("one or more report ids are unknown")
    out = _to_multi_compare_out(result)
    payload = out.model_dump()
    # Reuse the conclusion the Compare view generated (cache hit); otherwise
    # write it now so the PDF carries the same grounded narrative.
    payload["conclusion"] = await _get_or_write_compare_conclusion(
        request, payload, _parse_ids(ids), out.baseline_id
    )
    from iw_rag.api.benchmark_compare_pdf import build_compare_html

    pdf_bytes = await _render_pdf(build_compare_html(payload))
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": 'attachment; filename="benchmark-comparison.pdf"'},
    )


async def _render_pdf(html: str) -> bytes:
    """Render a self-contained HTML document to PDF bytes via headless Chromium.

    Uses Playwright (the ``pdf`` dependency group + ``playwright install
    chromium``).  Raises :class:`ValidationError` with an actionable message
    when Playwright/Chromium are not installed, so the route returns a clean
    4xx rather than a 500.
    """
    try:
        from playwright.async_api import async_playwright
    except ImportError as exc:  # pragma: no cover - depends on image build
        raise ValidationError(
            "PDF export needs Playwright — install the 'pdf' dependency group "
            "and run 'playwright install chromium' in the api image."
        ) from exc
    try:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(args=["--no-sandbox"])
            try:
                page = await browser.new_page()
                await page.set_content(html, wait_until="load")
                return await page.pdf(format="A4", print_background=True)
            finally:
                await browser.close()
    except ValidationError:
        raise
    except Exception as exc:  # noqa: BLE001 — surface a clean message
        raise ValidationError(
            "PDF render failed — the Chromium browser may not be installed "
            "(run 'playwright install chromium' in the api image)."
        ) from exc


@benchmark_router.get(
    "/benchmark/reports/{report_id}/pdf",
    include_in_schema=False,
)
async def benchmark_report_pdf(
    request: Request,
    ctx: AccessCtx,
    report_id: str,
) -> Response:
    """Render a downloadable, branded PDF of a benchmark report (AC: Generate Report).

    Builds the self-contained HTML (:mod:`iw_rag.api.benchmark_pdf_report`)
    from the persisted row — preferring a regenerated conclusion when one is
    cached — and renders it to PDF via headless Chromium.

    Args:
        request: The raw FastAPI request.
        ctx: The authenticated :class:`AccessContext` (any principal).
        report_id: The report to export.

    Returns:
        A ``application/pdf`` response with a download filename.
    """
    del ctx
    service = _benchmark_service(request)
    row = await service.get_report(report_id)
    if row is None:
        raise NotFoundError(f"benchmark report '{report_id}' not found")
    cache: dict[str, str] = getattr(request.app.state, "benchmark_conclusion_cache", {})
    if report_id in cache:
        row = dict(row)
        metrics = dict(row.get("metrics") or {})
        metrics["conclusion"] = {"text": cache[report_id], "basis": "regenerated"}
        row["metrics"] = metrics
    from iw_rag.api.benchmark_pdf_report import build_report_html

    pdf_bytes = await _render_pdf(build_report_html(row))
    fname = str(row.get("display_id") or f"benchmark-{report_id[:8]}") + ".pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )


# ---------------------------------------------------------------------------
# Operating mode (read-only)
# ---------------------------------------------------------------------------


@benchmark_router.get("/benchmark/mode")
async def benchmark_mode(request: Request, ctx: AccessCtx) -> dict[str, Any]:
    """Report the box's current operating mode (serving vs benchmark).

    The mode is inferred from the api's own generator endpoint: in benchmark
    mode the agent graph points at the ``llama-swap`` swap proxy; in serving
    mode it points at the pinned ``vllm`` container. Read-only — the actual
    switch is a host-side ``make`` target (no privileged in-api flip), so the
    response carries the command to run. Open to any authenticated principal.
    """
    settings = getattr(request.app.state, "settings", None)
    generator_endpoint = ""
    guards_enabled: bool | None = None
    if settings is not None:
        try:
            generator_endpoint = settings.models.get(settings.generation.model).endpoint
        except Exception:  # noqa: BLE001 — best-effort introspection for a status badge
            generator_endpoint = ""
        guards_enabled = getattr(getattr(settings, "guardrails", None), "enabled", None)
    mode = "benchmark" if "llama-swap" in (generator_endpoint or "") else "serving"
    return {
        "mode": mode,
        "generator_endpoint": generator_endpoint,
        "guards_enabled": guards_enabled,
        "switch_to": "serving" if mode == "benchmark" else "benchmark",
        "command": "make serving-mode" if mode == "benchmark" else "make benchmark-mode",
    }


# ---------------------------------------------------------------------------
# Studio page
# ---------------------------------------------------------------------------


@benchmark_router.get("/benchmark/ui", response_class=HTMLResponse, include_in_schema=False)
async def benchmark_ui() -> HTMLResponse:
    """Serve the standalone Benchmark Studio page (handles its own login).

    The S03 page owns the body — per-step model pickers populated from
    ``GET /models``, params panel, dataset picker, Run Full Test
    button, live SSE progress panel, report view, and A/B compare
    view.  This route just wires the response.
    """
    return HTMLResponse(content=benchmark_page_html())


# ---------------------------------------------------------------------------
# Mapping helpers
# ---------------------------------------------------------------------------


def _to_report_out(row: dict[str, Any]) -> BenchmarkReportOut:
    """Map a F-00093 repository row dict to the API response model.

    Args:
        row: The repository's row dict (or the F-00093 service's
            projection; any dict with the canonical keys).

    Returns:
        A :class:`BenchmarkReportOut` carrying the same fields.  JSONB
        columns are passed through verbatim; ``created_at`` is rendered
        as an ISO-8601 string when the row carries a ``datetime``.  The
        promoted scalar columns (``recall_at_5``, ``ndcg_at_5``,
        ``faithfulness_mean``, etc.) are surfaced as top-level fields
        so the S03 page can render the metric tiers without
        re-implementing vector→mean reduction client-side.  When the
        row carries the promoted column as ``None`` (dataset without
        reference answers, Tier-3 metrics) the API field is ``None``
        — never silently coerced to ``0.0`` (Boundary row 2).
    """
    created_at = row.get("created_at", "")
    if created_at and not isinstance(created_at, str):
        try:
            created_at = created_at.isoformat()
        except AttributeError:
            created_at = str(created_at)

    def _scalar(key: str) -> float | None:
        """Read a promoted scalar column, returning ``None`` when absent
        or non-numeric (defensive — the F-00093 stats module never
        coerces ``None`` to ``0.0``)."""
        val = row.get(key)
        if val is None:
            return None
        try:
            return float(val)
        except (TypeError, ValueError):
            return None

    return BenchmarkReportOut(
        id=str(row.get("id", "")),
        seq=row.get("seq"),
        display_id=str(row.get("display_id") or ""),
        run_id=(str(row["run_id"]) if row.get("run_id") else None),
        dataset_id=(str(row["dataset_id"]) if row.get("dataset_id") else None),
        dataset_name=(str(row["dataset_name"]) if row.get("dataset_name") else None),
        dataset_version=(str(row["dataset_version"]) if row.get("dataset_version") else None),
        report_label=str(row.get("report_label", "")),
        created_by=row.get("created_by"),
        config_snapshot=dict(row.get("model_config", {})),
        n_queries=int(row.get("n_queries", 0) or 0),
        n_judged=row.get("n_judged"),
        metrics=dict(row.get("metrics", {}) or {}),
        recall_at_5=_scalar("recall_at_5"),
        recall_at_10=_scalar("recall_at_10"),
        ndcg_at_5=_scalar("ndcg_at_5"),
        ndcg_at_10=_scalar("ndcg_at_10"),
        mrr=_scalar("mrr"),
        faithfulness_mean=_scalar("faithfulness_mean"),
        faithfulness_kappa=_scalar("faithfulness_kappa"),
        context_precision=_scalar("context_precision"),
        context_recall=_scalar("context_recall"),
        factual_correctness=_scalar("factual_correctness"),
        latency=row.get("latency"),
        resource=row.get("resource"),
        ci_95=row.get("ci_95"),
        created_at=created_at or "",
    )


def _to_compare_out(
    a: str,
    b: str,
    result: dict[str, Any],
) -> BenchmarkCompareResponse:
    """Map the F-00093 stats module's compare payload to the API model.

    The F-00093 ``compare`` returns a :class:`ComparisonResult` dataclass
    with ``.rows`` (a tuple of :class:`MetricDelta`).  The S02 layer
    accepts either the dataclass (preferred) or a dict projection
    (``{"rows": {...}, "winner": ..., "caveat": ...}``) so a slim fake
    or the real module both serialise cleanly.

    Args:
        a: The A report's id (echoed in the response).
        b: The B report's id (echoed in the response).
        result: The F-00093 stats module's output (dataclass or dict).

    Returns:
        A :class:`BenchmarkCompareResponse` with the per-metric rows,
        the ``winner`` flag, the underpowered caveat, and a reasoning
        string.
    """
    raw_rows: Any = getattr(result, "rows", None)
    if raw_rows is None and isinstance(result, dict):
        raw_rows = result.get("rows", ())
    out_rows: list[BenchmarkCompareDeltaOut] = []
    # ``result.rows`` may be a list/tuple of MetricDelta dataclasses or
    # a dict keyed by metric name.  Normalise both shapes.
    if isinstance(raw_rows, dict):
        for metric, delta in raw_rows.items():
            out_rows.append(_delta_from_dict(metric, delta))
    else:
        for entry in raw_rows or ():
            if isinstance(entry, dict):
                out_rows.append(_delta_from_dict(str(entry.get("metric", "")), entry))
            else:
                out_rows.append(_delta_from_dataclass(entry))
    winner: str | None = getattr(result, "winner", None)
    if winner is None and isinstance(result, dict):
        winner = result.get("winner")
    caveat: str | None = getattr(result, "caveat", None)
    if caveat is None and isinstance(result, dict):
        caveat = result.get("caveat")
    underpowered: bool = bool(getattr(result, "underpowered", False))
    if not underpowered and isinstance(result, dict):
        underpowered = bool(result.get("underpowered", False))
    reasoning: str = str(getattr(result, "reasoning", "") or "")
    if not reasoning and isinstance(result, dict):
        reasoning = str(result.get("reasoning", "") or "")
    dataset_id: str | None = None
    a_dataset = getattr(result, "dataset_id", None)
    if a_dataset is None and isinstance(result, dict):
        a_dataset = result.get("dataset_id")
    if a_dataset is not None:
        dataset_id = str(a_dataset)
    return BenchmarkCompareResponse(
        a=a,
        b=b,
        dataset_id=dataset_id,
        rows=out_rows,
        winner=winner,
        underpowered=underpowered,
        caveat=caveat,
        reasoning=reasoning,
    )


def _delta_rows_out(delta: Any) -> list[BenchmarkCompareDeltaOut]:
    """Map a :class:`ComparisonResult` (or dict / ``None``) to delta rows.

    Reused by the multi-compare mapper for each non-baseline run.  A
    ``None`` delta (the baseline run, or a run with no shared metrics)
    maps to an empty list.

    Args:
        delta: A ``ComparisonResult`` dataclass, a dict projection, or
            ``None``.

    Returns:
        The per-metric :class:`BenchmarkCompareDeltaOut` rows.
    """
    if delta is None:
        return []
    raw: Any = getattr(delta, "rows", None)
    if raw is None and isinstance(delta, dict):
        raw = delta.get("rows", ())
    out: list[BenchmarkCompareDeltaOut] = []
    if isinstance(raw, dict):
        for metric, entry in raw.items():
            out.append(_delta_from_dict(str(metric), entry))
    else:
        for entry in raw or ():
            if isinstance(entry, dict):
                out.append(_delta_from_dict(str(entry.get("metric", "")), entry))
            else:
                out.append(_delta_from_dataclass(entry))
    return out


def _to_multi_compare_out(result: dict[str, Any]) -> BenchmarkMultiCompareResponse:
    """Map the service's multi-compare payload to the API model.

    Args:
        result: The :meth:`BenchmarkReportService.compare_multi` dict.

    Returns:
        A :class:`BenchmarkMultiCompareResponse`.
    """
    runs: list[BenchmarkCompareRunOut] = []
    for r in result.get("runs", []) or []:
        runs.append(
            BenchmarkCompareRunOut(
                id=str(r.get("id") or ""),
                display_id=str(r.get("display_id") or ""),
                label=str(r.get("label") or ""),
                is_baseline=bool(r.get("is_baseline")),
                color=str(r.get("color") or ""),
                config_snapshot=dict(r.get("config_snapshot") or {}),
                dataset_name=r.get("dataset_name"),
                created_at=str(r.get("created_at") or ""),
                n_queries=int(r.get("n_queries") or 0),
                means=dict(r.get("means") or {}),
                deltas=_delta_rows_out(r.get("delta")),
            )
        )
    items: list[BenchmarkCompareItemOut] = []
    for it in result.get("items", []) or []:
        samples: list[BenchmarkCompareSampleOut | None] = []
        for s in it.get("samples", []) or []:
            if s is None:
                samples.append(None)
            else:
                samples.append(
                    BenchmarkCompareSampleOut(
                        answer=s.get("answer"),
                        contexts=list(s.get("contexts") or []),
                    )
                )
        items.append(
            BenchmarkCompareItemOut(
                index=int(it.get("index", 0)),
                question=it.get("question"),
                values={k: list(v) for k, v in (it.get("values") or {}).items()},
                samples=samples,
            )
        )
    return BenchmarkMultiCompareResponse(
        dataset_id=result.get("dataset_id"),
        baseline_id=str(result.get("baseline_id") or ""),
        n_queries=int(result.get("n_queries") or 0),
        underpowered=bool(result.get("underpowered")),
        caveat=result.get("caveat"),
        metrics=list(result.get("metrics") or []),
        runs=runs,
        items=items,
    )


def _delta_from_dataclass(delta: Any) -> BenchmarkCompareDeltaOut:
    """Map a F-00093 :class:`MetricDelta` dataclass to the API model.

    Args:
        delta: A F-00093 ``MetricDelta`` (frozen dataclass).

    Returns:
        The matching :class:`BenchmarkCompareDeltaOut`.
    """
    ci = getattr(delta, "ci_95", None)
    return BenchmarkCompareDeltaOut(
        metric=str(getattr(delta, "metric", "")),
        val_a=getattr(delta, "val_a", None),
        val_b=getattr(delta, "val_b", None),
        delta=getattr(delta, "delta", None),
        p_value=getattr(delta, "p_value", None),
        p_corrected=getattr(delta, "p_corrected", None),
        ci=(tuple(ci) if ci is not None else None),
        significant=bool(getattr(delta, "significant", False)),
    )


def _delta_from_dict(
    metric: str,
    delta: dict[str, Any],
) -> BenchmarkCompareDeltaOut:
    """Map a dict-shaped delta row to the API model.

    Args:
        metric: The metric name (when the row is keyed by name).
        delta: A ``{"val_a": ..., "val_b": ..., "delta": ...,
            "p_value": ..., "p_corrected": ..., "ci": (lo, hi),
            "significant": bool}`` dict.

    Returns:
        The matching :class:`BenchmarkCompareDeltaOut`.
    """
    ci: Any = delta.get("ci")
    return BenchmarkCompareDeltaOut(
        metric=str(delta.get("metric") or metric),
        val_a=delta.get("val_a"),
        val_b=delta.get("val_b"),
        delta=delta.get("delta"),
        p_value=delta.get("p_value"),
        p_corrected=delta.get("p_corrected"),
        ci=(tuple(ci) if ci is not None else None),
        significant=bool(delta.get("significant", False)),
    )


__all__ = [
    "EvalOperatorCtx",
    "benchmark_router",
    "require_eval_operator",
]
