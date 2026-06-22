"""Production read-service for the Benchmark Studio (F-00095 S02 seam).

The S02 benchmark routes (`benchmark_routes.py`) consume an injected
``benchmark_service`` on ``app.state`` exposing three async methods —
``list_reports()`` / ``get_report(id)`` / ``compare(a, b)``. The S02 layer
deliberately left the *live* binding to a follow-up so tests could wire a
fake; this module is that live binding. It composes the append-only
:class:`~iw_rag.eval.benchmark.repository.BenchmarkReportRepository` (F-00093
S06) and the :mod:`~iw_rag.eval.benchmark.stats` compare module behind the
narrow interface the routes expect.

Read-only by construction: this service never creates, mutates, or deletes
a report (Invariant 2 — reports are append-only history). Run orchestration
is the :class:`BenchmarkRunner`'s surface (S01), not this service's.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import select

from iw_rag.domain.errors import ValidationError
from iw_rag.eval.benchmark import stats
from iw_rag.eval.benchmark.dataset_labels import attach_dataset_labels
from iw_rag.eval.benchmark.repository import BenchmarkReportRepository
from iw_rag.eval.benchmark.sequence import assign_sequence_numbers
from iw_rag.storage import schema
from iw_rag.storage.db import Database

# Metrics eligible for A/B compare, in display order. Only the metrics
# present as per-query vectors in BOTH report ``metrics`` blobs are tested;
# the winner rule (R-00121 §5.4) needs ``ndcg_at_5`` AND ``faithfulness``.
_COMPARE_METRICS: tuple[str, ...] = (
    "ndcg_at_5",
    "faithfulness",
    "context_recall",
    "factual_correctness",
)

# Metrics surfaced in the N-way multi-compare view, in display order.  A
# superset of ``_COMPARE_METRICS`` (the 2-way winner set) — the multi view
# shows the broader per-item picture (retrieval recall + context precision)
# while the baseline-relative significance still leans on the same paired
# machinery.  Only those present as a per-query vector in ≥1 run are returned.
_MULTI_COMPARE_METRICS: tuple[str, ...] = (
    "ndcg_at_5",
    "recall_at_5",
    "faithfulness",
    "context_precision",
    "context_recall",
    "factual_correctness",
)

# Per-run colour palette for the multi-compare view, assigned deterministically
# by selection order.  Distinct hues legible on the console's dark ink
# background; the run keeps its colour across every chart in the view AND the
# PDF (per-comparison colour identity — Feature: A/B → N-way compare).
_RUN_PALETTE: tuple[str, ...] = (
    "#2DD4BF",  # teal
    "#a78bfa",  # purple
    "#f59e0b",  # amber
    "#f43f5e",  # rose
    "#38bdf8",  # sky
)

#: Multi-compare run-count bounds (the picker + route enforce these too).
MIN_COMPARE_RUNS = 2
MAX_COMPARE_RUNS = 5


class BenchmarkReportService:
    """Live ``benchmark_service`` binding over the F-00093 repository + stats.

    Each call opens its own short-lived :class:`AsyncSession` from the shared
    :class:`~iw_rag.storage.db.Database` so the service is safe to hold as a
    long-lived singleton on ``app.state``.
    """

    def __init__(self, database: Database) -> None:
        """Bind the service to the shared database.

        Args:
            database: The application's :class:`Database` (session factory).
        """
        self._database = database

    async def list_reports(self) -> list[dict[str, Any]]:
        """Return every persisted report across all datasets, newest first.

        The F-00093 repository only lists *per dataset*; the Studio's report
        picker needs the full history, so this reads the table directly
        (read-only ``SELECT`` — no mutation surface is added to the repo).

        Each row is decorated with a stable ``seq`` + ``display_id``
        (``BENCH-000001`` …) derived from the full append-only history, so
        the operator sees a sequential execution number rather than the
        random uuid (see :mod:`iw_rag.eval.benchmark.sequence`).

        Returns:
            A list of row dicts ordered by ``created_at`` descending, each
            carrying ``seq`` + ``display_id``.
        """
        stmt = select(schema.benchmark_reports).order_by(
            schema.benchmark_reports.c.created_at.desc()
        )
        async with self._database.session() as session:
            rows = [dict(row) for row in (await session.execute(stmt)).mappings()]
            await attach_dataset_labels(session, rows)
        # The full history is in hand → number every row (the display order
        # stays newest-first; only the seq/display_id keys are attached).
        assign_sequence_numbers(rows)
        return rows

    async def get_report(self, report_id: str) -> dict[str, Any] | None:
        """Return one report by id, or ``None`` when the id is unknown.

        The returned row is decorated with the same ``seq`` + ``display_id``
        the report carries in :meth:`list_reports` — computed from the full
        history so the deep-dive header shows the same ``BENCH-000007``
        execution number as the picker.

        Args:
            report_id: The report's uuid-stringified id.

        Returns:
            The row dict (with ``seq`` + ``display_id``), or ``None`` when
            no row matches.
        """
        async with self._database.session() as session:
            row = await BenchmarkReportRepository(session).get_report(report_id)
            if row is None:
                return None
            # Number the row against the full history.  We only need the
            # ordering keys (id + created_at) of every report, not the
            # heavy JSONB blobs, so this stays a cheap two-column scan.
            meta_stmt = select(
                schema.benchmark_reports.c.id,
                schema.benchmark_reports.c.created_at,
            )
            history = [dict(m) for m in (await session.execute(meta_stmt)).mappings()]
            await attach_dataset_labels(session, [row])
        assign_sequence_numbers(history)
        seq_by_id = {str(m["id"]): m for m in history}
        meta = seq_by_id.get(str(row.get("id", "")))
        if meta is not None:
            row["seq"] = meta["seq"]
            row["display_id"] = meta["display_id"]
        return row

    async def compare(
        self,
        report_a_id: str,
        report_b_id: str,
    ) -> stats.ComparisonResult | None:
        """A/B compare two reports on the same dataset (F-00093 stats contract).

        Fetches both reports, enforces the same-dataset invariant (paired
        stats require the same golden queries), extracts the per-query metric
        vectors from each report's ``metrics`` blob, and runs the F-00093
        paired comparison.

        Args:
            report_a_id: The A-side report's id.
            report_b_id: The B-side report's id.

        Returns:
            A :class:`~iw_rag.eval.benchmark.stats.ComparisonResult`, or
            ``None`` when either id is unknown (the route maps that to 404).

        Raises:
            ValidationError: When the two reports reference different
                datasets (the route maps that to 400).
        """
        async with self._database.session() as session:
            repo = BenchmarkReportRepository(session)
            a = await repo.get_report(report_a_id)
            b = await repo.get_report(report_b_id)

        if a is None or b is None:
            return None
        if a.get("dataset_id") != b.get("dataset_id"):
            raise ValidationError(
                "compare across different datasets is not allowed "
                f"(a.dataset_id={a.get('dataset_id')}, b.dataset_id={b.get('dataset_id')}); "
                "paired stats require the same golden queries."
            )

        a_blob: dict[str, Any] = a.get("metrics") or {}
        b_blob: dict[str, Any] = b.get("metrics") or {}
        metrics = [
            m
            for m in _COMPARE_METRICS
            if isinstance(a_blob.get(m), list) and isinstance(b_blob.get(m), list)
        ]
        n_queries = int(a.get("n_queries") or 0)
        a_vectors = {m: a_blob[m] for m in metrics}
        b_vectors = {m: b_blob[m] for m in metrics}
        return stats.compare(
            a_vectors,
            b_vectors,
            metrics=metrics,
            n_queries=n_queries,
        )

    async def compare_multi(
        self,
        report_ids: list[str],
        baseline_id: str | None = None,
    ) -> dict[str, Any] | None:
        """N-way (2..5) baseline-relative compare of reports on one dataset.

        Fetches every requested report (de-duplicated, order preserved),
        enforces the run-count bounds + the same-dataset invariant
        (paired stats require the same golden queries), designates a
        baseline, and runs the :func:`stats.compare_multi` baseline-
        relative comparison.  On top of the statistics it assembles the
        per-run identity (display id, module composition, a stable
        palette colour) and the per-item matrix (per-metric value per
        run + the persisted question / answer / contexts for the
        expandable qualitative diff).

        Args:
            report_ids: The report ids to compare (2..5 after de-dup).
            baseline_id: The id to treat as the baseline; defaults to
                the first id when ``None``.

        Returns:
            A plain-dict payload (mapped to the API model by the route),
            or ``None`` when any id is unknown (the route maps that to
            404).

        Raises:
            ValidationError: When the run count is out of bounds, the
                reports span more than one dataset, or ``baseline_id``
                is not among the compared reports (each → 400).
        """
        # De-duplicate while preserving the caller's order.
        ids = list(dict.fromkeys(report_ids))
        if len(ids) < MIN_COMPARE_RUNS:
            raise ValidationError(
                f"multi-compare needs at least {MIN_COMPARE_RUNS} distinct reports; got {len(ids)}."
            )
        if len(ids) > MAX_COMPARE_RUNS:
            raise ValidationError(
                f"multi-compare supports at most {MAX_COMPARE_RUNS} reports; got {len(ids)}."
            )

        rows: list[dict[str, Any]] = []
        for rid in ids:
            row = await self.get_report(rid)
            if row is None:
                return None
            rows.append(row)

        dataset_ids = {str(r.get("dataset_id")) for r in rows}
        if len(dataset_ids) > 1:
            raise ValidationError(
                "multi-compare across different datasets is not allowed "
                f"(datasets={sorted(dataset_ids)}); paired stats require the same "
                "golden queries."
            )

        baseline_index = 0
        if baseline_id is not None:
            try:
                baseline_index = ids.index(baseline_id)
            except ValueError as exc:
                raise ValidationError(
                    f"baseline id '{baseline_id}' is not among the compared reports."
                ) from exc

        blobs: list[dict[str, Any]] = [dict(r.get("metrics") or {}) for r in rows]
        # Metrics present as a per-query vector in at least one run.
        present: list[str] = [
            m
            for m in _MULTI_COMPARE_METRICS
            if any(isinstance(b.get(m), list) and b.get(m) for b in blobs)
        ]
        n_queries = int(rows[baseline_index].get("n_queries") or 0)
        multi = stats.compare_multi(
            blobs,
            metrics=present,
            baseline_index=baseline_index,
            n_queries=n_queries,
        )

        runs = [
            {
                "id": str(rows[i].get("id") or ""),
                "display_id": str(rows[i].get("display_id") or ""),
                "label": str(rows[i].get("report_label") or ""),
                "is_baseline": i == baseline_index,
                "color": _RUN_PALETTE[i % len(_RUN_PALETTE)],
                "config_snapshot": dict(rows[i].get("model_config") or {}),
                "dataset_name": rows[i].get("dataset_name"),
                "created_at": str(rows[i].get("created_at") or ""),
                "n_queries": int(rows[i].get("n_queries") or 0),
                "means": multi.means[i],
                "delta": multi.deltas[i],
            }
            for i in range(len(rows))
        ]

        return {
            "dataset_id": next(iter(dataset_ids)) if dataset_ids else None,
            "baseline_id": ids[baseline_index],
            "n_queries": n_queries,
            "underpowered": multi.underpowered,
            "caveat": multi.caveat,
            "metrics": list(present),
            "runs": runs,
            "items": _build_item_matrix(blobs, present),
        }


def _build_item_matrix(
    blobs: list[dict[str, Any]],
    metrics: list[str],
) -> list[dict[str, Any]]:
    """Reshape per-run metric vectors + samples into per-item rows.

    Item ``idx`` aligns by query index across all runs (the same pairing
    the stats module assumes for same-dataset reports).  Each item row
    carries, per metric, the per-run value list (aligned to the run
    order) and the per-run persisted sample (answer + contexts) for the
    expandable qualitative diff.  Runs from before the sample-persistence
    change simply contribute ``None`` samples (graceful degradation).

    Args:
        blobs: The per-run ``metrics`` JSONB blobs (run order preserved).
        metrics: The metric names to surface per item.

    Returns:
        A list of per-item dicts ``{index, question, values, samples}``.
    """
    samples_per_run: list[list[Any] | None] = [
        b.get("samples") if isinstance(b.get("samples"), list) else None for b in blobs
    ]
    n_items = 0
    for b in blobs:
        for m in metrics:
            v = b.get(m)
            if isinstance(v, list):
                n_items = max(n_items, len(v))
    for s in samples_per_run:
        if s is not None:
            n_items = max(n_items, len(s))

    items: list[dict[str, Any]] = []
    for idx in range(n_items):
        values: dict[str, list[float | None]] = {}
        for m in metrics:
            row_vals: list[float | None] = []
            for b in blobs:
                v = b.get(m)
                if isinstance(v, list) and idx < len(v):
                    cell = v[idx]
                    row_vals.append(float(cell) if isinstance(cell, int | float) else None)
                else:
                    row_vals.append(None)
            values[m] = row_vals

        question: str | None = None
        samples_out: list[dict[str, Any] | None] = []
        for s in samples_per_run:
            if s is not None and idx < len(s) and isinstance(s[idx], dict):
                samp = s[idx]
                if question is None and samp.get("question"):
                    question = str(samp.get("question"))
                samples_out.append(
                    {
                        "answer": samp.get("answer"),
                        "contexts": list(samp.get("contexts") or []),
                    }
                )
            else:
                samples_out.append(None)
        items.append({"index": idx, "question": question, "values": values, "samples": samples_out})
    return items


__all__ = ["BenchmarkReportService"]
