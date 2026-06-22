"""HTTP routes for the F-00096 Operator Console Scoreboard (S08).

The Console's Scoreboard view is a **thin HTTP facade** over the S05
:class:`~iw_rag.eval.benchmark.scoreboard.ScoreboardService`.  The
S05 service is the **sole authority** on the per-row ordering (AC6:
descending faithfulness for the ``faithfulness`` board); the route
preserves the service's order verbatim and serialises the matching
per-board list to the API surface.

Routes:

* ``GET /benchmark/scoreboard?board=<name>`` — return one of the six
  S05 leaderboards (``faithfulness`` / ``ndcg_at_5`` / ``tok_per_s``
  / ``latency`` / ``vram_efficiency`` / ``composite``).  Open to any
  authenticated principal (mirrors
  :func:`list_benchmark_reports`).  With **0 persisted reports** the
  service returns empty per-board lists; the route surfaces them
  verbatim (Boundary row *"Empty report history → explicit empty
  state, not an error"*).

**Domain-error → HTTP mapping** is delegated to the existing
``_ERROR_STATUS`` table in :mod:`iw_rag.api.app`: an unwired
service raises :class:`ValidationError` → **400** (the route MUST
NOT catch-and-swallow).
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Query, Request

from iw_rag.api.routes import AccessCtx
from iw_rag.api.schemas import ScoreboardEntryOut, ScoreboardResponse
from iw_rag.domain.errors import ValidationError

logger = logging.getLogger(__name__)

scoreboard_router = APIRouter()


# ---------------------------------------------------------------------------
# Collaborator resolution (mirrors benchmark_routes._benchmark_service)
# ---------------------------------------------------------------------------


def _scoreboard_service(request: Request) -> Any:
    """Resolve the S05 :class:`ScoreboardService` from ``app.state``.

    Raises:
        ValidationError: When the service is not wired onto
            ``app.state.scoreboard_service`` (e.g. an unwired test
            deployment) so the existing ``_ERROR_STATUS`` table maps
            the error to 400 instead of a 5xx.  Mirrors
            :func:`iw_rag.api.benchmark_routes._benchmark_service`.
    """
    service: Any = getattr(request.app.state, "scoreboard_service", None)
    if service is None:
        raise ValidationError(
            "scoreboard service is not wired on app.state "
            "(the Console's Scoreboard view is not enabled for this deployment)."
        )
    return service


# ---------------------------------------------------------------------------
# Mapping helper — S05 ScoreboardEntry → API ScoreboardEntryOut
# ---------------------------------------------------------------------------


def _to_entry_out(
    *,
    rank: int,
    entry: Any,
) -> ScoreboardEntryOut:
    """Map an S05 :class:`ScoreboardEntry` to the API response model.

    The S05 service emits frozen-dataclass rows carrying
    ``report_id`` / ``model_config`` / ``created_at`` /
    ``metric_value``; the route assigns a 1-based ``rank`` in the
    order the service returned (the service is the sole authority on
    the ordering — AC6).

    The F-00093 storage column ``model_config`` is exposed on the API
    as ``config_snapshot`` to avoid colliding with Pydantic's reserved
    ``model_config`` class attribute (which would break field
    detection).  The S14 page maps it back to ``model_config`` for
    display (mirrors :class:`~iw_rag.api.schemas.BenchmarkReportOut`).

    Args:
        rank: The 1-based rank to assign (caller is responsible for
            passing the correct rank in the order the service emitted).
        entry: The S05 :class:`ScoreboardEntry` (or any duck-typed
            object with the same field shape — the fake service used
            by the unit tests has its own row type).

    Returns:
        A :class:`ScoreboardEntryOut` ready to serialise.
    """
    created_at = entry.created_at
    if not isinstance(created_at, str):
        # The S05 service round-trips the row's value verbatim
        # (string or datetime — both shapes observed in tests +
        # production).  Normalise to ISO-8601 string for the wire.
        try:
            created_at = created_at.isoformat()
        except AttributeError:
            created_at = str(created_at)
    return ScoreboardEntryOut(
        report_id=str(entry.report_id),
        seq=getattr(entry, "seq", None),
        display_id=str(getattr(entry, "display_id", "") or ""),
        rank=int(rank),
        config_snapshot=dict(entry.model_config or {}),
        dataset_name=str(getattr(entry, "dataset_name", "") or ""),
        created_at=str(created_at) if created_at is not None else "",
        metric_value=(float(entry.metric_value) if entry.metric_value is not None else None),
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@scoreboard_router.get("/benchmark/scoreboard", response_model=ScoreboardResponse)
async def get_scoreboard(
    request: Request,
    ctx: AccessCtx,
    board: str = Query(
        ...,
        description=(
            "The leaderboard to return.  One of ``faithfulness``, "
            "``ndcg_at_5``, ``tok_per_s``, ``latency``, ``vram_efficiency``, "
            "``composite`` (S05).  Forwarded verbatim to the S05 service."
        ),
    ),
) -> ScoreboardResponse:
    """Return one S05 scoreboard leaderboard for the operator console.

    The route is a **thin HTTP facade** over the S05
    :class:`ScoreboardService` — the service is the sole authority on
    the per-row ordering (AC6: descending faithfulness for the
    ``faithfulness`` board).  The route preserves the service's order
    verbatim and assigns a 1-based ``rank`` per row as it serialises.

    Open to any authenticated principal — the scoreboard reveals no
    sensitive content (ranked metric values + public model config
    snapshots).  Mirrors :func:`list_benchmark_reports`.

    With **0 persisted reports** the S05 service returns empty per-
    board lists; the route surfaces them verbatim (Boundary row
    *"Empty report history → explicit empty state, not an error"*).

    Args:
        request: The raw FastAPI request.
        ctx: The authenticated :class:`AccessContext` (any principal).
        board: The leaderboard name forwarded to the S05 service
            (``faithfulness`` / ``ndcg_at_5`` / ``tok_per_s`` /
            ``latency`` / ``vram_efficiency`` / ``composite``).

    Returns:
        A :class:`ScoreboardResponse` with the requested board's
        ranked entries (possibly empty).

    Raises:
        ValidationError: When ``scoreboard_service`` is not wired on
            ``app.state`` (→ 400 via the existing ``_ERROR_STATUS``
            table in :mod:`iw_rag.api.app`).
    """
    service = _scoreboard_service(request)
    full = await service.score()
    # Select the matching per-board list.  An unknown board name
    # produces an empty list — the S05 service does not validate the
    # name (it always returns the full Scoreboard with all six boards
    # populated); the route's job is to pick the right one.  This
    # matches the S05 report's "the route forwards the board string
    # to the service" contract.
    entries = getattr(full, board, None) or []
    return ScoreboardResponse(
        board=str(board),
        entries=[_to_entry_out(rank=i + 1, entry=e) for i, e in enumerate(entries)],
    )


__all__ = [
    "scoreboard_router",
]
