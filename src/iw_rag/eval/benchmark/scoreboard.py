"""Scoreboard ranking service for the Operator Console (F-00096, S05).

Ranks the append-only ``benchmark_reports`` history into per-metric
leaderboards plus a deterministic composite.  This is the offline
authoritative source for the **AC6 faithfulness ordering** — the S08
route (``GET /benchmark/scoreboard``) will call into it.

The service mirrors :class:`BenchmarkReportService` — constructor-
injected :class:`Database`, opens a short-lived async session per
call, reads through :class:`BenchmarkReportRepository` (F-00093 S06).
The S08 HTTP layer is the only consumer; this module is **offline
only** (two-clocks rule, ``eval/CLAUDE.md``) and MUST NOT be
imported by any hot-path module (``api/``, ``agent/``,
``retrieval/``, ``generation/``).

Board definitions (per F-00096 §Scope → Scoreboard and AC6):

* **``faithfulness``** — by ``benchmark_reports.faithfulness_mean``
  **descending** (higher is better).  AC6 names ``model_config`` +
  run timestamp as the per-row fields; the entry carries both.
* **``ndcg_at_5``** — by ``benchmark_reports.ndcg_at_5`` descending.
* **``tok_per_s``** (throughput) — by the derived value
  ``resource.vllm_server.generation_tokens_delta /``
  ``(latency.per_stage_ms.retrieve_generate_ms * n_queries / 1000)``,
  descending.  The two source keys are the only ones the perf
  collector writes (R-00122 §"Recommendation" JSON shape, see
  :mod:`iw_rag.eval.benchmark.assembly`).
* **``latency``** — by ``latency.e2e_p99_s`` **ascending** (lower
  end-to-end p99 wins).  ``e2e_p99_s`` is the vLLM-reported p99
  end-to-end request latency in seconds — the natural single-number
  latency for a leaderboard.
* **``vram_efficiency``** — by the derived value
  ``ndcg_at_5 / (resource.gpu.vram_peak_mb / 1024)`` descending
  (retrieval quality per resident GiB of VRAM).  When
  ``resource.gpu.available`` is ``False`` (no GPU) the entry has no
  VRAM signal and is ranked last.
* **``composite``** — deterministic weighted sum of the min-max-
  normalised per-metric scores.  The exact formula is in the
  :meth:`ScoreboardService._composite_for` docstring; the test
  suite pins the ranking for a fixed input.

**NULL handling** — a row whose source value is ``None`` (column
NULL, missing JSONB key, or absent nested block) is **ranked last**
on its board; the entry's ``metric_value`` is ``None`` so the UI can
render the "no measurement" state.  NULL rows are ordered amongst
themselves by ``created_at`` then ``id`` (stable).  A board whose
**every** row has a NULL value is the empty list (matches the
empty-history boundary — the unmeasured-only tail would add no
signal to the view).  No row ever crashes the ranker.

**Empty history** — 0 reports → every board is ``[]`` (boundary row:
*"Empty report history → explicit empty state, not an error"*).

**Tie-break** — equal metric values are ordered by ``created_at``
ascending, then ``id`` ascending.  Stable across runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy import select

from iw_rag.eval.benchmark.dataset_labels import attach_dataset_labels
from iw_rag.eval.benchmark.sequence import assign_sequence_numbers
from iw_rag.storage import schema
from iw_rag.storage.db import Database

# ---------------------------------------------------------------------------
# Typed result shapes
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ScoreboardEntry:
    """A single row in a scoreboard leaderboard.

    The Console renders one row per entry; ``metric_value`` is
    ``None`` when the source column / JSONB key was NULL — the UI
    uses the sentinel to render the "no measurement" state.

    Attributes:
        report_id: The report's id (str at the boundary).
        model_config: The static model configuration snapshot
            (JSONB from the row).
        created_at: The run timestamp (round-tripped from the row —
            the test suite exercises both string and datetime forms).
        metric_value: The numeric value of the board's metric, or
            ``None`` when the source was NULL.
        seq: The 1-based execution number derived from the append-only
            history (``None`` when the row carried no ``seq`` — e.g. a
            hand-built test fixture).
        display_id: The operator-facing ``BENCH-000007`` label, or ``""``
            when not derived.
        dataset_name: The human dataset name resolved from ``eval_datasets``
            (e.g. ``benchmark_golden``), or ``""`` when the FK is unresolved —
            so the board renders the name instead of the opaque uuid.
    """

    report_id: str
    model_config: dict[str, object]
    created_at: object
    metric_value: float | None
    seq: int | None = None
    display_id: str = ""
    dataset_name: str = ""


@dataclass(frozen=True, slots=True)
class Scoreboard:
    """The full leaderboard payload returned by :meth:`ScoreboardService.score`.

    Each board is a list of :class:`ScoreboardEntry` ordered best
    first (except ``latency`` which is ordered best=lowest first;
    both directions are documented on the service).

    Attributes:
        faithfulness: Reports ranked by ``faithfulness_mean`` descending.
        ndcg_at_5: Reports ranked by ``ndcg_at_5`` descending.
        tok_per_s: Reports ranked by throughput descending.
        latency: Reports ranked by ``e2e_p99_s`` ascending.
        vram_efficiency: Reports ranked by ``ndcg_at_5 / vram_peak_gb`` descending.
        composite: Reports ranked by the documented composite formula descending.
    """

    faithfulness: list[ScoreboardEntry]
    ndcg_at_5: list[ScoreboardEntry]
    tok_per_s: list[ScoreboardEntry]
    latency: list[ScoreboardEntry]
    vram_efficiency: list[ScoreboardEntry]
    composite: list[ScoreboardEntry]


# ---------------------------------------------------------------------------
# Composite weighting
# ---------------------------------------------------------------------------

# Per-metric weights for the composite score.  Sum to 1.0.  The
# weights are intentionally simple and documented so the test
# suite can pin the resulting ranking for a fixed input.
_COMPOSITE_WEIGHTS: dict[str, float] = {
    "faithfulness": 0.30,
    "ndcg_at_5": 0.20,
    "tok_per_s": 0.20,
    "latency_inverse": 0.15,
    "vram_efficiency": 0.15,
}


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class ScoreboardService:
    """Offline ranking service for the ``benchmark_reports`` history.

    Mirrors the :class:`BenchmarkReportService` DI pattern: the
    constructor takes a :class:`Database` (session factory), opens a
    short-lived async session per call, and reads through
    :class:`BenchmarkReportRepository` (the S06 surface, used in
    read-only mode).

    The S08 HTTP route (``GET /benchmark/scoreboard``) will inject
    the long-lived service instance and call :meth:`score` per
    request.
    """

    def __init__(self, database: Database) -> None:
        """Bind the service to the shared :class:`Database`.

        Args:
            database: The application's :class:`Database` (session
                factory).  Each :meth:`score` call opens its own
                short-lived :class:`AsyncSession`.
        """
        self._database = database

    async def score(self) -> Scoreboard:
        """Compute all six leaderboards and return a :class:`Scoreboard`.

        Reads the full ``benchmark_reports`` history (every dataset,
        newest first — matches :meth:`BenchmarkReportService.list_reports`),
        extracts the per-metric / per-JSONB value for each row, and
        produces:

        * five per-metric boards (``faithfulness``, ``ndcg_at_5``,
          ``tok_per_s``, ``latency``, ``vram_efficiency``) ordered
          per the direction documented in the module docstring; and
        * one ``composite`` board ordered by the documented
          weighted-sum-of-normalised-values formula.

        Empty history → every board is ``[]`` (no exception).
        A row with a NULL source value is **ranked last** on its
        board with ``metric_value=None``.

        Returns:
            A :class:`Scoreboard` with all six lists populated.  Each
            list may be empty (zero rows, or zero rows with a
            measured value for the board's metric).
        """
        rows = await self._fetch_history()
        if not rows:
            return Scoreboard(
                faithfulness=[],
                ndcg_at_5=[],
                tok_per_s=[],
                latency=[],
                vram_efficiency=[],
                composite=[],
            )

        # Per-metric boards.
        faithfulness = self._rank_faithfulness(rows)
        ndcg_at_5 = self._rank_ndcg_at_5(rows)
        tok_per_s = self._rank_tok_per_s(rows)
        latency = self._rank_latency(rows)
        vram_efficiency = self._rank_vram_efficiency(rows)

        # Composite — weighted sum of min-max-normalised per-metric
        # scores.  Built off the per-row signals, not off the per-
        # board orderings, so the composite ranking is independent
        # of the board-specific tie-break keys.
        composite = self._rank_composite(rows)

        return Scoreboard(
            faithfulness=faithfulness,
            ndcg_at_5=ndcg_at_5,
            tok_per_s=tok_per_s,
            latency=latency,
            vram_efficiency=vram_efficiency,
            composite=composite,
        )

    # -------------------------------------------------------------------
    # History read
    # -------------------------------------------------------------------

    async def _fetch_history(self) -> list[dict[str, Any]]:
        """Read the full ``benchmark_reports`` history (newest first).

        Reuses the same SELECT path as
        :meth:`BenchmarkReportService.list_reports` — direct read on
        the table (the F-00093 repository's ``list_reports`` is
        per-dataset, but the scoreboard ranks the *whole* history,
        so the read is unscoped).

        Returns:
            A list of row dicts, possibly empty.
        """
        stmt = select(schema.benchmark_reports).order_by(
            schema.benchmark_reports.c.created_at.desc()
        )
        async with self._database.session() as session:
            rows = [dict(r) for r in (await session.execute(stmt)).mappings()]
            # Resolve the opaque ``dataset_id`` uuid to the human dataset name
            # so the board's config summary reads ``benchmark_golden`` not a uuid.
            await attach_dataset_labels(session, rows)
        # Decorate with the stable execution number so every leaderboard
        # row can be traced back to its ``BENCH-000007`` report.
        assign_sequence_numbers(rows)
        return rows

    # -------------------------------------------------------------------
    # Per-metric board builders
    # -------------------------------------------------------------------

    @staticmethod
    def _sort_key_for_value(direction: str) -> Any:
        """Return the sort key for a per-board ranker.

        The :meth:`_rank_*` helpers all share the same "measured
        values first (in the metric's direction), NULLs last, ties
        broken by ``created_at`` then ``id``" rule.  This helper
        returns the metric-direction-aware primary key.
        """
        if direction == "desc":
            # Higher is better.  Measured values come first, sorted
            # DESCENDING.  NULL values sort last (Python ``None``
            # is already a stable "less than any float", so we
            # negate measured values to get DESC semantics).
            return lambda v, ts, rid: (
                v is None,  # False (=0) sorts before True (=1) → measured first
                -float(v) if v is not None else 0.0,
                ts,
                rid,
            )
        if direction == "asc":
            # Lower is better.  Measured values come first, sorted
            # ASCENDING.  NULL values sort last.
            return lambda v, ts, rid: (
                v is None,
                float(v) if v is not None else 0.0,
                ts,
                rid,
            )
        raise ValueError(f"unknown sort direction: {direction!r}")

    @classmethod
    def _rank(
        cls,
        rows: list[dict[str, Any]],
        *,
        value: Any,
        direction: str,
    ) -> list[ScoreboardEntry]:
        """Generic per-metric ranker.

        NULL handling — a row whose source value is ``None`` is
        **ranked last** on its board with ``metric_value=None``
        (the UI renders the "no measurement" state on the entry).
        A board whose **every** row has a NULL value is the empty
        list (consistent with the empty-history boundary; the
        "no measurement" entries would add no signal to the view).

        Args:
            rows: The history rows.
            value: A callable ``(row) -> float | None`` extracting the
                per-row metric value (or ``None`` for NULL).
            direction: ``"desc"`` (higher is better) or ``"asc"``
                (lower is better).

        Returns:
            A list of :class:`ScoreboardEntry` ordered per the
            direction; NULLs last; ties broken by
            ``created_at`` then ``id``.  Returns ``[]`` when no row
            has a measured value.
        """
        if not rows:
            return []
        # Partition: measured rows first (will be sorted by direction),
        # NULL rows last (sorted amongst themselves by created_at +
        # id for stable ordering of the unmeasured tail).
        measured = [r for r in rows if value(r) is not None]
        nulls = [r for r in rows if value(r) is None]
        if not measured:
            return []
        sort_key = cls._sort_key_for_value(direction)
        measured_ordered = sorted(
            measured,
            key=lambda r: sort_key(value(r), r.get("created_at"), r.get("id", "")),
        )
        # NULLs: stable order by created_at then id (matches the
        # tie-break rule for measured rows).
        nulls_ordered = sorted(
            nulls,
            key=lambda r: (r.get("created_at"), r.get("id", "")),
        )
        ordered = measured_ordered + nulls_ordered
        return [
            ScoreboardEntry(
                report_id=str(r.get("id", "")),
                model_config=dict(r.get("model_config") or {}),
                created_at=r.get("created_at"),
                metric_value=value(r),
                seq=r.get("seq"),
                display_id=str(r.get("display_id") or ""),
                dataset_name=str(r.get("dataset_name") or ""),
            )
            for r in ordered
        ]

    @classmethod
    def _rank_faithfulness(cls, rows: list[dict[str, Any]]) -> list[ScoreboardEntry]:
        """Rank by ``faithfulness_mean`` descending."""
        return cls._rank(
            rows,
            value=lambda r: _as_float(r.get("faithfulness_mean")),
            direction="desc",
        )

    @classmethod
    def _rank_ndcg_at_5(cls, rows: list[dict[str, Any]]) -> list[ScoreboardEntry]:
        """Rank by ``ndcg_at_5`` descending."""
        return cls._rank(
            rows,
            value=lambda r: _as_float(r.get("ndcg_at_5")),
            direction="desc",
        )

    @classmethod
    def _rank_latency(cls, rows: list[dict[str, Any]]) -> list[ScoreboardEntry]:
        """Rank by ``latency.e2e_p99_s`` ascending (lower wins)."""
        return cls._rank(
            rows,
            value=lambda r: _e2e_p99_s(r),
            direction="asc",
        )

    @classmethod
    def _rank_tok_per_s(cls, rows: list[dict[str, Any]]) -> list[ScoreboardEntry]:
        """Rank by ``generation_tokens_delta / (per_stage_ms * n_queries / 1000)`` descending.

        The full per-stage key is ``latency.per_stage_ms.retrieve_generate_ms``;
        the docstring is abbreviated for line length — see the module
        docstring's ``tok_per_s`` section for the full formula.
        """
        return cls._rank(
            rows,
            value=lambda r: _tok_per_s(r),
            direction="desc",
        )

    @classmethod
    def _rank_vram_efficiency(cls, rows: list[dict[str, Any]]) -> list[ScoreboardEntry]:
        """Rank by ``ndcg_at_5 / (vram_peak_mb / 1024)`` descending."""
        return cls._rank(
            rows,
            value=lambda r: _vram_efficiency(r),
            direction="desc",
        )

    # -------------------------------------------------------------------
    # Composite
    # -------------------------------------------------------------------

    @classmethod
    def _rank_composite(cls, rows: list[dict[str, Any]]) -> list[ScoreboardEntry]:
        """Rank by a weighted sum of min-max-normalised per-metric values.

        The formula (per the module docstring):

        1. For each metric, compute the value per row (or ``None``
           when the source is NULL).
        2. Min-max normalise each metric across the cohort to
           ``[0, 1]`` (rank-based: best row → 1.0, worst row → 0.0;
           all-NULL → 0.0 for every row).
        3. Multiply each normalised value by the documented weight
           (:data:`_COMPOSITE_WEIGHTS`) and sum.  For ``latency``
           (lower-is-better), the normalised value is ``1 - norm`` so
           the weight rewards a low latency.
        4. Order the resulting scores descending; ties broken by
           ``created_at`` then ``id``.

        Rows with **no measured value for any of the weighted
        metrics** score ``0.0`` and are ordered amongst themselves
        by ``created_at`` then ``id``.  A row with a NULL on one
        metric (but measured on others) still scores a positive
        sum — only the NULL metric contributes 0.0.
        """
        if not rows:
            return []

        metric_defs: list[tuple[str, str]] = [
            ("faithfulness", "desc"),
            ("ndcg_at_5", "desc"),
            ("tok_per_s", "desc"),
            ("latency", "asc"),
            ("vram_efficiency", "desc"),
        ]
        extractors: dict[str, Any] = {
            "faithfulness": lambda r: _as_float(r.get("faithfulness_mean")),
            "ndcg_at_5": lambda r: _as_float(r.get("ndcg_at_5")),
            "tok_per_s": lambda r: _tok_per_s(r),
            "latency": lambda r: _e2e_p99_s(r),
            "vram_efficiency": lambda r: _vram_efficiency(r),
        }

        scores: list[float] = []
        for r in rows:
            row_score = 0.0
            for metric_name, _direction in metric_defs:
                value = extractors[metric_name](r)
                if value is None:
                    continue
                cohort_values = [
                    extractors[metric_name](rr)
                    for rr in rows
                    if extractors[metric_name](rr) is not None
                ]
                if not cohort_values:
                    continue
                mn = min(cohort_values)
                mx = max(cohort_values)
                # All measured values are equal — no discrimination
                # possible; treat the row as 0.5 for this metric.
                norm = 0.5 if mx == mn else (value - mn) / (mx - mn)
                weight = _COMPOSITE_WEIGHTS.get(metric_name, 0.0)
                # For lower-is-better metrics, invert so the weight
                # rewards a low value.
                if metric_name == "latency":
                    norm = 1.0 - norm
                row_score += weight * norm
            scores.append(row_score)

        # Order: score DESC, then created_at ASC, then id ASC.
        indexed = list(enumerate(rows))
        indexed.sort(
            key=lambda ix: (
                -scores[ix[0]],
                ix[1].get("created_at"),
                ix[1].get("id", ""),
            )
        )
        return [
            ScoreboardEntry(
                report_id=str(r.get("id", "")),
                model_config=dict(r.get("model_config") or {}),
                created_at=r.get("created_at"),
                metric_value=scores[ix],
                seq=r.get("seq"),
                display_id=str(r.get("display_id") or ""),
                dataset_name=str(r.get("dataset_name") or ""),
            )
            for ix, r in indexed
        ]


# ---------------------------------------------------------------------------
# JSONB value extractors
# ---------------------------------------------------------------------------


def _as_float(v: Any) -> float | None:
    """Coerce a row's column value to ``float`` (or ``None`` for NULL)."""
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _e2e_p99_s(row: dict[str, Any]) -> float | None:
    """Read ``latency.e2e_p99_s`` from a row's latency JSONB.

    Returns ``None`` when the JSONB is absent, the key is missing,
    or the value is not a number.
    """
    latency = row.get("latency")
    if not isinstance(latency, dict):
        return None
    return _as_float(latency.get("e2e_p99_s"))


def _tok_per_s(row: dict[str, Any]) -> float | None:
    """Compute throughput in tokens per second for a row.

    Formula:
        ``resource.vllm_server.generation_tokens_delta``
        ``/ (latency.per_stage_ms.retrieve_generate_ms * n_queries / 1000)``

    The numerator is the vLLM counter delta over the run (perf
    collector writes this when both before+after ``/metrics`` are
    scraped — see :class:`VllmMetricsScraper` and
    :func:`assemble_report`).  The denominator is the total wall-
    clock time of the per-query ``retrieve + generate`` stage across
    the run (``per_stage_ms.retrieve_generate_ms`` is the per-query
    mean in ms; multiplied by ``n_queries`` and divided by 1000 to
    get seconds).

    Returns ``None`` when any of the required inputs is missing or
    non-numeric.  Returns ``None`` (NOT a division-by-zero crash)
    when the per-stage time is zero.
    """
    latency = row.get("latency")
    resource = row.get("resource")
    if not isinstance(latency, dict) or not isinstance(resource, dict):
        return None
    per_stage = latency.get("per_stage_ms")
    if not isinstance(per_stage, dict):
        return None
    per_stage_ms = _as_float(per_stage.get("retrieve_generate_ms"))
    if per_stage_ms is None or per_stage_ms <= 0.0:
        return None
    n_queries = row.get("n_queries")
    if not isinstance(n_queries, int) or n_queries <= 0:
        return None
    vllm = resource.get("vllm_server")
    if not isinstance(vllm, dict):
        return None
    gen_tokens_delta = _as_float(vllm.get("generation_tokens_delta"))
    if gen_tokens_delta is None:
        return None
    total_seconds = (per_stage_ms * n_queries) / 1000.0
    if total_seconds <= 0.0:
        return None
    return float(gen_tokens_delta) / total_seconds


def _vram_efficiency(row: dict[str, Any]) -> float | None:
    """Compute VRAM efficiency: ``ndcg_at_5 / (vram_peak_mb / 1024)``.

    Quality per resident GiB of VRAM.  Returns ``None`` when the
    source row has no ``ndcg_at_5`` (T1), no ``resource.gpu`` block,
    no ``vram_peak_mb``, or when the GPU sampler was unavailable
    (``resource.gpu.available`` is ``False``).
    """
    ndcg = _as_float(row.get("ndcg_at_5"))
    if ndcg is None:
        return None
    resource = row.get("resource")
    if not isinstance(resource, dict):
        return None
    gpu = resource.get("gpu")
    if not isinstance(gpu, dict):
        return None
    if gpu.get("available") is False:
        return None
    vram_peak_mb = _as_float(gpu.get("vram_peak_mb"))
    if vram_peak_mb is None or vram_peak_mb <= 0:
        return None
    vram_peak_gb = float(vram_peak_mb) / 1024.0
    if vram_peak_gb <= 0:
        return None
    return ndcg / vram_peak_gb


__all__ = [
    "Scoreboard",
    "ScoreboardEntry",
    "ScoreboardService",
]
