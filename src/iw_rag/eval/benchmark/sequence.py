"""Stable, human-readable execution numbers for benchmark reports.

Benchmark reports are persisted with a random ``uuid`` primary key — fine
for joins, useless for an operator who wants to say "look at run 7".  This
module derives a **sequential execution number** (``BENCH-000001``,
``BENCH-000002``, …) from the append-only ``benchmark_reports`` history.

Why *derive* it rather than store a column?  Reports are append-only by
construction (Invariant 2 — :class:`BenchmarkReportRepository` exposes no
update / delete surface and the FKs do not cascade-delete), so the
position of a row in the history ordered by ``(created_at, id)`` never
changes once inserted.  The Nth report ever written is therefore always
``BENCH-{N:06d}`` — a stored counter would carry the same information with
a migration and an extra column.  The ordering key includes ``id`` as a
deterministic tie-break so two rows sharing a ``created_at`` (a clock
collision the single-GPU box makes vanishingly unlikely) still get a
total, stable order.

The helper mutates row dicts in place (attaching ``seq`` + ``display_id``)
so the read services can decorate the rows they already fetched without a
second query.  It is pure (no DB, no I/O) and **offline-only** by
package convention.
"""

from __future__ import annotations

from typing import Any


def format_bench_id(seq: int) -> str:
    """Render a 1-based execution number as the operator-facing id.

    Args:
        seq: The 1-based position of the report in the append-only
            history (1 = the first benchmark ever run).

    Returns:
        The zero-padded label, e.g. ``"BENCH-000007"`` for ``seq=7``.
        Six digits comfortably covers any realistic run count and keeps
        the labels the same width for tidy column alignment.
    """
    return f"BENCH-{seq:06d}"


def _order_key(row: dict[str, Any]) -> tuple[Any, str]:
    """Total, stable ordering key: ``created_at`` then ``id`` (both ascending).

    ``created_at`` is ``NOT NULL`` in the schema (server default ``now()``),
    so the primary key is always present; ``id`` stringified is the
    deterministic tie-break.
    """
    return (row.get("created_at"), str(row.get("id", "")))


def assign_sequence_numbers(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach ``seq`` + ``display_id`` to every row, in place.

    The number reflects each row's position in the **full** history (the
    caller must pass every report, not a filtered subset, for the numbers
    to be the canonical execution numbers).  Ordering is by
    ``(created_at, id)`` ascending — so the oldest run is ``BENCH-000001``
    and the number grows by one per run.

    Args:
        rows: The report row dicts (as returned by a ``SELECT *`` on
            ``benchmark_reports``).  Mutated in place.

    Returns:
        The same ``rows`` list (returned for call-site convenience); the
        order of the list itself is left untouched — only the ``seq`` /
        ``display_id`` keys are added to each element.
    """
    for i, row in enumerate(sorted(rows, key=_order_key), start=1):
        row["seq"] = i
        row["display_id"] = format_bench_id(i)
    return rows


__all__ = ["assign_sequence_numbers", "format_bench_id"]
