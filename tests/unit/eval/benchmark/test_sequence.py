"""Unit tests for the derived benchmark execution number (BENCH-000001).

The execution number is derived (not stored) from the append-only history;
these pins guard the two properties that make that safe:

* the number is the row's 1-based position ordered by ``(created_at, id)``
  ascending — so the oldest run is ``BENCH-000001`` and it grows by one
  per run; and
* it is **append-stable**: inserting a newer report never renumbers an
  existing one (the property that lets us derive instead of store).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from iw_rag.eval.benchmark.sequence import assign_sequence_numbers, format_bench_id


def _row(id_: str, created_at: datetime) -> dict[str, object]:
    return {"id": id_, "created_at": created_at}


def test_format_bench_id_zero_pads_to_six_digits() -> None:
    assert format_bench_id(1) == "BENCH-000001"
    assert format_bench_id(7) == "BENCH-000007"
    assert format_bench_id(123456) == "BENCH-123456"


def test_assign_numbers_by_created_at_ascending() -> None:
    """Oldest run is BENCH-000001 regardless of the list's own order."""
    t0 = datetime(2026, 6, 17, 12, 0, tzinfo=UTC)
    # Pass them newest-first (as list_reports does) — the number must
    # still follow created_at ascending, not list position.
    rows = [
        _row("c", t0 + timedelta(minutes=2)),
        _row("b", t0 + timedelta(minutes=1)),
        _row("a", t0),
    ]
    assign_sequence_numbers(rows)
    by_id = {r["id"]: r for r in rows}
    assert (by_id["a"]["seq"], by_id["a"]["display_id"]) == (1, "BENCH-000001")
    assert (by_id["b"]["seq"], by_id["b"]["display_id"]) == (2, "BENCH-000002")
    assert (by_id["c"]["seq"], by_id["c"]["display_id"]) == (3, "BENCH-000003")
    # The list order itself is left untouched — only keys were added.
    assert [r["id"] for r in rows] == ["c", "b", "a"]


def test_tie_broken_by_id_so_ordering_is_total() -> None:
    """Two rows sharing a created_at still get a deterministic order."""
    t0 = datetime(2026, 6, 17, 12, 0, tzinfo=UTC)
    rows = [_row("y", t0), _row("x", t0)]
    assign_sequence_numbers(rows)
    by_id = {r["id"]: r for r in rows}
    # "x" sorts before "y" → BENCH-000001.
    assert by_id["x"]["seq"] == 1
    assert by_id["y"]["seq"] == 2


def test_append_is_stable_existing_rows_keep_their_number() -> None:
    """Appending a newer report must not renumber the existing ones."""
    t0 = datetime(2026, 6, 17, 12, 0, tzinfo=UTC)
    first = [_row("a", t0), _row("b", t0 + timedelta(minutes=1))]
    assign_sequence_numbers(first)
    before = {r["id"]: r["seq"] for r in first}

    # A third, newer run arrives.
    later = [
        _row("a", t0),
        _row("b", t0 + timedelta(minutes=1)),
        _row("c", t0 + timedelta(minutes=2)),
    ]
    assign_sequence_numbers(later)
    after = {r["id"]: r["seq"] for r in later}

    assert after["a"] == before["a"] == 1
    assert after["b"] == before["b"] == 2
    assert after["c"] == 3  # the new run takes the next number, nothing shifts
