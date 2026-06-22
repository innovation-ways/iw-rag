"""Unit tests for the N-way multi-compare response serialization.

:func:`iw_rag.api.benchmark_routes._to_multi_compare_out` maps the
service's ``compare_multi`` dict payload (per-run identity + a
``ComparisonResult`` delta per non-baseline run + the per-item matrix +
samples) onto :class:`iw_rag.api.schemas.BenchmarkMultiCompareResponse`.
These tests pin the wire format the Compare page + the PDF read.
"""

from __future__ import annotations

from typing import Any

import pytest

from iw_rag.api.benchmark_routes import _parse_ids, _to_multi_compare_out
from iw_rag.api.schemas import (
    BenchmarkCompareItemOut,
    BenchmarkCompareRunOut,
    BenchmarkMultiCompareResponse,
)
from iw_rag.eval.benchmark.stats import ComparisonResult, MetricDelta


def _delta(metric: str, val_a: float, val_b: float, *, significant: bool = True) -> MetricDelta:
    return MetricDelta(
        metric=metric,
        val_a=val_a,
        val_b=val_b,
        delta=val_b - val_a,
        p_value=0.01,
        p_corrected=0.02,
        ci_95=(0.01, 0.20),
        significant=significant,
        test="wilcoxon",
    )


def _payload() -> dict[str, Any]:
    """A two-run compare payload as the service emits it."""
    base_delta = None
    run_delta = ComparisonResult(
        rows=(_delta("ndcg_at_5", 0.5, 0.7), _delta("faithfulness", 0.8, 0.9)),
        winner="B",
        underpowered=True,
        caveat="n=3 below min_queries_for_significance=150",
        reasoning="run beats baseline",
    )
    return {
        "dataset_id": "ds-1",
        "baseline_id": "a",
        "n_queries": 3,
        "underpowered": True,
        "caveat": "n=3 below min_queries_for_significance=150",
        "metrics": ["ndcg_at_5", "faithfulness"],
        "runs": [
            {
                "id": "a",
                "display_id": "BENCH-a",
                "label": "lbl-a",
                "is_baseline": True,
                "color": "#2DD4BF",
                "config_snapshot": {"generator": "gen-a"},
                "dataset_name": "benchmark_golden",
                "created_at": "2026-06-11",
                "n_queries": 3,
                "means": {"ndcg_at_5": 0.5, "faithfulness": 0.8},
                "delta": base_delta,
            },
            {
                "id": "b",
                "display_id": "BENCH-b",
                "label": "lbl-b",
                "is_baseline": False,
                "color": "#a78bfa",
                "config_snapshot": {"generator": "gen-b"},
                "dataset_name": "benchmark_golden",
                "created_at": "2026-06-12",
                "n_queries": 3,
                "means": {"ndcg_at_5": 0.7, "faithfulness": 0.9},
                "delta": run_delta,
            },
        ],
        "items": [
            {
                "index": 0,
                "question": "Q1",
                "values": {"ndcg_at_5": [0.5, 0.7], "faithfulness": [0.8, 0.9]},
                "samples": [
                    {"answer": "A1", "contexts": ["c1"]},
                    None,
                ],
            }
        ],
    }


def test_to_multi_compare_out_returns_response_model() -> None:
    out = _to_multi_compare_out(_payload())
    assert isinstance(out, BenchmarkMultiCompareResponse)
    assert out.baseline_id == "a"
    assert out.dataset_id == "ds-1"
    assert out.metrics == ["ndcg_at_5", "faithfulness"]


def test_to_multi_compare_out_maps_run_identity_and_colour() -> None:
    out = _to_multi_compare_out(_payload())
    assert len(out.runs) == 2
    a, b = out.runs
    assert isinstance(a, BenchmarkCompareRunOut)
    assert a.is_baseline is True and b.is_baseline is False
    assert a.color == "#2DD4BF" and b.color == "#a78bfa"
    assert a.display_id == "BENCH-a"
    assert b.config_snapshot == {"generator": "gen-b"}


def test_to_multi_compare_out_baseline_has_empty_deltas() -> None:
    out = _to_multi_compare_out(_payload())
    assert out.runs[0].deltas == []  # baseline: no self-delta
    assert len(out.runs[1].deltas) == 2  # run vs baseline


def test_to_multi_compare_out_delta_rows_carry_stats() -> None:
    out = _to_multi_compare_out(_payload())
    row = out.runs[1].deltas[0]
    assert row.metric == "ndcg_at_5"
    assert row.val_a == 0.5 and row.val_b == 0.7
    assert row.delta == pytest.approx(0.2)
    assert row.ci == (0.01, 0.20)
    assert row.significant is True


def test_to_multi_compare_out_item_matrix_and_samples() -> None:
    out = _to_multi_compare_out(_payload())
    assert len(out.items) == 1
    item = out.items[0]
    assert isinstance(item, BenchmarkCompareItemOut)
    assert item.question == "Q1"
    assert item.values["ndcg_at_5"] == [0.5, 0.7]
    assert item.samples[0] is not None
    assert item.samples[0].answer == "A1"
    assert item.samples[0].contexts == ["c1"]
    assert item.samples[1] is None


def test_to_multi_compare_out_json_wire_field_names() -> None:
    """The JSON the page reads carries the expected field names."""
    dumped = _to_multi_compare_out(_payload()).model_dump()
    for key in (
        "dataset_id",
        "baseline_id",
        "n_queries",
        "underpowered",
        "caveat",
        "metrics",
        "runs",
        "items",
    ):
        assert key in dumped
    run0 = dumped["runs"][0]
    for key in ("id", "display_id", "is_baseline", "color", "config_snapshot", "means", "deltas"):
        assert key in run0
    item0 = dumped["items"][0]
    for key in ("index", "question", "values", "samples"):
        assert key in item0


def test_to_multi_compare_out_round_trips() -> None:
    out = _to_multi_compare_out(_payload())
    rebuilt = BenchmarkMultiCompareResponse.model_validate(out.model_dump())
    assert rebuilt.baseline_id == "a"
    assert rebuilt.underpowered is True
    assert len(rebuilt.runs) == 2


def test_parse_ids_splits_and_strips() -> None:
    assert _parse_ids("a, b ,c,") == ["a", "b", "c"]
    assert _parse_ids("") == []
    assert _parse_ids("  ") == []
