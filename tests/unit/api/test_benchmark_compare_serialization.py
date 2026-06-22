"""Unit tests for the F-00095 / S02 A/B compare response serialization.

The S02 :func:`iw_rag.api.benchmark_routes._to_compare_out` helper maps
the F-00093 stats module's :class:`~iw_rag.eval.benchmark.stats.ComparisonResult`
(``rows``: tuple of :class:`MetricDelta`, ``winner``, ``underpowered``,
``caveat``, ``reasoning``) onto the API response model
:class:`iw_rag.api.schemas.BenchmarkCompareResponse`.  These tests pin
the wire format the S03 page renders:

* the **delta table** maps each metric to
  ``{val_a, val_b, delta, p_value, p_corrected, ci, significant}``;
* the **winner** flag obeys the *BOTH nDCG@5 AND faithfulness* rule
  (R-00121 §5.4 — a single-metric win returns ``winner=None``);
* the **n≈37 underpowered caveat** is present when n is small
  (F-00093 §Boundary Behavior row 5 / AC3);
* the response serialises to JSON with the S03-page-expected field names
  (no surprise renames).

The stats module's :class:`ComparisonResult` is the **preferred** input
shape; a dict projection (``{"rows": {…}, "winner": …, "caveat": …}``)
is also accepted (per the S02 layer's tolerance for slim fakes).
"""

from __future__ import annotations

from typing import Any

from iw_rag.api.benchmark_routes import _to_compare_out
from iw_rag.api.schemas import (
    BenchmarkCompareDeltaOut,
    BenchmarkCompareResponse,
)
from iw_rag.eval.benchmark.stats import ComparisonResult, MetricDelta

# ---------------------------------------------------------------------------
# Helpers — canned ``ComparisonResult`` fixtures
# ---------------------------------------------------------------------------


def _delta(
    metric: str,
    val_a: float,
    val_b: float,
    p_value: float = 0.01,
    p_corrected: float = 0.02,
    ci: tuple[float, float] = (0.05, 0.20),
    significant: bool = True,
    test: str = "wilcoxon",
) -> MetricDelta:
    """Build a :class:`MetricDelta` row with sensible defaults."""
    return MetricDelta(
        metric=metric,
        val_a=val_a,
        val_b=val_b,
        delta=val_b - val_a,
        p_value=p_value,
        p_corrected=p_corrected,
        ci_95=ci,
        significant=significant,
        test=test,
    )


def _well_powered_result() -> ComparisonResult:
    """A well-powered comparison: B is better on BOTH nDCG@5 AND faithfulness."""
    return ComparisonResult(
        rows=(
            _delta("ndcg_at_5", 0.70, 0.80, p_corrected=0.001, significant=True),
            _delta("faithfulness", 0.85, 0.92, p_corrected=0.005, significant=True),
            _delta("recall_at_10", 0.75, 0.78, p_corrected=0.04, significant=True),
        ),
        winner="B",
        underpowered=False,
        caveat=None,
        reasoning=(
            "ndcg_at_5: A=0.700 B=0.800 delta=+0.100 p_corrected=0.001 significant=True; "
            "faithfulness: A=0.850 B=0.920 delta=+0.070 p_corrected=0.005 significant=True; "
            "recall_at_10: A=0.750 B=0.780 delta=+0.030 p_corrected=0.04 significant=True; "
            "winner=B (BOTH ndcg_at_5 AND faithfulness significant and B-better)"
        ),
    )


def _underpowered_result() -> ComparisonResult:
    """A small-n (n=37) comparison: underpowered + the n≈37 caveat."""
    return ComparisonResult(
        rows=(
            _delta(
                "ndcg_at_5",
                0.60,
                0.65,
                p_value=0.42,
                p_corrected=0.42,
                ci=(-0.10, 0.20),
                significant=False,
            ),
            _delta(
                "faithfulness",
                0.80,
                0.82,
                p_value=0.55,
                p_corrected=0.55,
                ci=(-0.05, 0.10),
                significant=False,
            ),
        ),
        winner=None,
        underpowered=True,
        caveat=(
            "n=37 below min_queries_for_significance=150: the paired test "
            "detects only ~0.08-0.12 retrieval / ~0.06-0.10 faithfulness "
            "differences. Increase n to ~150 for stronger conclusions."
        ),
        reasoning=(
            "ndcg_at_5: A=0.600 B=0.650 delta=+0.050 p_corrected=0.42 significant=False; "
            "faithfulness: A=0.800 B=0.820 delta=+0.020 p_corrected=0.55 significant=False; "
            "winner=None (dual-metric rule not satisfied)"
        ),
    )


def _single_metric_win_result() -> ComparisonResult:
    """B is better on nDCG@5 only — winner MUST be ``None`` (dual-metric rule)."""
    return ComparisonResult(
        rows=(
            _delta("ndcg_at_5", 0.70, 0.80, p_corrected=0.005, significant=True),
            _delta("faithfulness", 0.92, 0.88, p_corrected=0.20, significant=False),
        ),
        winner=None,  # dual-metric rule not satisfied
        underpowered=False,
        caveat=None,
        reasoning=(
            "ndcg_at_5: A=0.700 B=0.800 delta=+0.100 p_corrected=0.005 significant=True; "
            "faithfulness: A=0.920 B=0.880 delta=-0.040 p_corrected=0.2 significant=False; "
            "winner=None (dual-metric rule not satisfied)"
        ),
    )


# ---------------------------------------------------------------------------
# Dataclass input — the preferred shape
# ---------------------------------------------------------------------------


def test_compare_serialization_accepts_comparison_result_dataclass() -> None:
    """The S02 layer accepts the F-00093 ``ComparisonResult`` dataclass
    (the production input shape) without modification."""
    response = _to_compare_out("a-id", "b-id", _well_powered_result())
    assert isinstance(response, BenchmarkCompareResponse)
    assert response.a == "a-id"
    assert response.b == "b-id"


def test_compare_serialization_emits_per_metric_delta_table() -> None:
    """AC3 — the response carries a per-metric delta table whose shape is
    ``{metric: (val_a, val_b, delta, p_value, ci, significant)}``."""
    response = _to_compare_out("a-id", "b-id", _well_powered_result())
    assert len(response.rows) == 3
    for row in response.rows:
        assert isinstance(row, BenchmarkCompareDeltaOut)
        assert row.metric
        assert row.val_a is not None
        assert row.val_b is not None
        assert row.delta is not None
        # delta == val_b - val_a
        assert row.delta == pytest_approx(row.val_b - row.val_a)
        assert row.p_value is not None
        assert row.p_corrected is not None
        assert row.ci is not None
        assert len(row.ci) == 2
        assert isinstance(row.significant, bool)


def test_compare_serialization_preserves_metric_names() -> None:
    """The delta table preserves the requested metric order (nDCG@5 first,
    then faithfulness, then recall@10)."""
    response = _to_compare_out("a-id", "b-id", _well_powered_result())
    metrics = [row.metric for row in response.rows]
    assert metrics == ["ndcg_at_5", "faithfulness", "recall_at_10"]


def test_compare_serialization_emits_winner_when_both_metrics_significant() -> None:
    """AC3 — the ``winner`` flag is set ONLY when one config is better on
    BOTH nDCG@5 AND faithfulness (R-00121 §5.4)."""
    response = _to_compare_out("a-id", "b-id", _well_powered_result())
    assert response.winner == "B"


def test_compare_serialization_emits_no_winner_on_single_metric_win() -> None:
    """R-00121 §5.4 — a single-metric win (better nDCG@5, equal/lower
    faithfulness) returns ``winner=None``."""
    response = _to_compare_out("a-id", "b-id", _single_metric_win_result())
    assert response.winner is None


def test_compare_serialization_emits_winner_a_when_a_wins_both() -> None:
    """R-00121 §5.4 — A wins (both ndcg_at_5 and faithfulness with
    negative delta AND significant) → ``winner="A"``."""
    result = ComparisonResult(
        rows=(
            _delta("ndcg_at_5", 0.80, 0.70, p_corrected=0.001, significant=True),
            _delta("faithfulness", 0.92, 0.85, p_corrected=0.005, significant=True),
        ),
        winner="A",
        underpowered=False,
        caveat=None,
        reasoning="winner=A (BOTH ndcg_at_5 AND faithfulness significant and B-worse)",
    )
    response = _to_compare_out("a-id", "b-id", result)
    assert response.winner == "A"


# ---------------------------------------------------------------------------
# Underpowered caveat (F-00093 §Boundary Behavior row 5 / AC3)
# ---------------------------------------------------------------------------


def test_compare_serialization_emits_underpowered_flag_for_small_n() -> None:
    """AC3 — the response surfaces the ``underpowered`` flag when n is below
    ``min_queries_for_significance``."""
    response = _to_compare_out("a-id", "b-id", _underpowered_result())
    assert response.underpowered is True


def test_compare_serialization_emits_n37_caveat_text() -> None:
    """The caveat string MUST include the underpowered / n≈37 / detectable
    effect-size language (the S03 page renders this verbatim in a
    ``.caveat-box``)."""
    response = _to_compare_out("a-id", "b-id", _underpowered_result())
    assert response.caveat is not None
    caveat_lc = response.caveat.lower()
    # The caveat must surface one of the documented anchors: the
    # underpower flag, the n<150 count, or the detectable effect size.
    assert (
        "underpower" in caveat_lc
        or "n=" in response.caveat
        or "detectable" in caveat_lc
        or "min_queries" in caveat_lc
    ), f"caveat missing underpower / n= / detectable anchor: {response.caveat!r}"


def test_compare_serialization_omits_caveat_when_well_powered() -> None:
    """A well-powered comparison (``underpowered=False``) has no caveat
    (the S03 page hides the caveat-box in that case)."""
    response = _to_compare_out("a-id", "b-id", _well_powered_result())
    assert response.underpowered is False
    assert response.caveat is None


# ---------------------------------------------------------------------------
# Dict input — the slim-fake / projection shape
# ---------------------------------------------------------------------------


def test_compare_serialization_accepts_dict_projection() -> None:
    """The S02 layer accepts a dict projection (e.g. ``{"rows": {...},
    "winner": "B", "caveat": "..."}``) — useful for slim fakes that
    don't carry the full ``ComparisonResult`` dataclass."""
    slim = {
        "rows": {
            "ndcg_at_5": {
                "val_a": 0.70,
                "val_b": 0.80,
                "delta": 0.10,
                "p_value": 0.01,
                "p_corrected": 0.02,
                "ci": (0.05, 0.20),
                "significant": True,
            },
            "faithfulness": {
                "val_a": 0.85,
                "val_b": 0.92,
                "delta": 0.07,
                "p_value": 0.005,
                "p_corrected": 0.01,
                "ci": (0.02, 0.10),
                "significant": True,
            },
        },
        "winner": "B",
        "underpowered": False,
        "caveat": None,
        "reasoning": "B wins both",
    }
    response = _to_compare_out("a-id", "b-id", slim)
    assert response.winner == "B"
    assert response.underpowered is False
    assert len(response.rows) == 2


def test_compare_serialization_dict_projection_preserves_p_value_ci_significant() -> None:
    """The dict-projection path also surfaces the per-metric ``p_value``,
    ``ci`` and ``significant`` flag the S03 page renders in the delta table."""
    slim = {
        "rows": {
            "ndcg_at_5": {
                "val_a": 0.60,
                "val_b": 0.65,
                "delta": 0.05,
                "p_value": 0.04,
                "p_corrected": 0.04,
                "ci": (-0.02, 0.12),
                "significant": True,
            },
        },
        "winner": None,
        "underpowered": True,
        "caveat": "n=37 below min_queries_for_significance=150",
        "reasoning": "single row",
    }
    response = _to_compare_out("a-id", "b-id", slim)
    assert len(response.rows) == 1
    row = response.rows[0]
    assert row.metric == "ndcg_at_5"
    assert row.p_value == 0.04
    assert row.p_corrected == 0.04
    assert row.ci == (-0.02, 0.12)
    assert row.significant is True


def test_compare_serialization_dict_projection_with_list_of_dicts() -> None:
    """The dict projection can also carry ``rows`` as a list of dicts
    (F-00093 ``MetricDelta`` row shape), not just a dict keyed by metric
    name.  The S02 layer must handle both."""
    slim = {
        "rows": [
            {
                "metric": "ndcg_at_5",
                "val_a": 0.70,
                "val_b": 0.80,
                "delta": 0.10,
                "p_value": 0.01,
                "p_corrected": 0.02,
                "ci": (0.05, 0.20),
                "significant": True,
            },
            {
                "metric": "faithfulness",
                "val_a": 0.85,
                "val_b": 0.92,
                "delta": 0.07,
                "p_value": 0.005,
                "p_corrected": 0.01,
                "ci": (0.02, 0.10),
                "significant": True,
            },
        ],
        "winner": "B",
        "underpowered": False,
        "caveat": None,
        "reasoning": "B wins both",
    }
    response = _to_compare_out("a-id", "b-id", slim)
    metrics = [row.metric for row in response.rows]
    assert metrics == ["ndcg_at_5", "faithfulness"]
    assert response.winner == "B"


# ---------------------------------------------------------------------------
# JSON wire format — the S03 page reads these field names verbatim
# ---------------------------------------------------------------------------


def test_compare_serialization_json_uses_s03_field_names() -> None:
    """The JSON wire format MUST use the field names the S03 page reads:
    ``val_a`` / ``val_b`` / ``delta`` / ``p_value`` / ``p_corrected`` /
    ``ci`` / ``significant`` / ``winner`` / ``underpowered`` / ``caveat``.

    The S03 page's renderer reads ``row.val_a`` / ``row.p_value`` /
    ``row.ci`` etc. directly.  A Pydantic renamer would silently break
    the page; this test pins the field names.
    """
    response = _to_compare_out("a-id", "b-id", _well_powered_result())
    dumped = response.model_dump()
    assert "a" in dumped
    assert "b" in dumped
    assert "rows" in dumped
    assert "winner" in dumped
    assert "underpowered" in dumped
    assert "caveat" in dumped
    assert "reasoning" in dumped
    # And per-row fields:
    row0 = dumped["rows"][0]
    for field in (
        "metric",
        "val_a",
        "val_b",
        "delta",
        "p_value",
        "p_corrected",
        "ci",
        "significant",
    ):
        assert field in row0, f"row field {field!r} missing from compare response wire format"


def test_compare_serialization_json_round_trip_preserves_winner_and_caveat() -> None:
    """``model_dump()`` / ``model_validate()`` round-trips preserve the
    ``winner`` flag and the underpowered caveat verbatim."""
    response = _to_compare_out("a-id", "b-id", _underpowered_result())
    dumped = response.model_dump()
    rebuilt = BenchmarkCompareResponse.model_validate(dumped)
    assert rebuilt.winner is None
    assert rebuilt.underpowered is True
    assert rebuilt.caveat == response.caveat


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_compare_serialization_handles_none_underpowered_with_dict() -> None:
    """A dict input with ``underpowered`` missing / falsy is coerced to ``False``
    (the default), not raised as a KeyError."""
    slim: dict[str, Any] = {
        "rows": {},
        "winner": None,
        "caveat": None,
        "reasoning": "empty comparison",
    }
    response = _to_compare_out("a-id", "b-id", slim)
    assert response.underpowered is False
    assert response.rows == []


def test_compare_serialization_handles_missing_p_corrected_in_dict() -> None:
    """A dict row without ``p_corrected`` defaults to ``None`` (the API
    model is ``float | None``).  This is the slim-fake path: a fake that
    only knows the raw p-value can still drive the response."""
    slim = {
        "rows": {
            "ndcg_at_5": {
                "val_a": 0.5,
                "val_b": 0.6,
                "delta": 0.1,
                "p_value": 0.04,
                "ci": None,
                "significant": True,
            },
        },
        "winner": None,
        "underpowered": False,
        "caveat": None,
    }
    response = _to_compare_out("a-id", "b-id", slim)
    assert len(response.rows) == 1
    row = response.rows[0]
    assert row.p_value == 0.04
    assert row.p_corrected is None
    assert row.ci is None


# ---------------------------------------------------------------------------
# Tiny pytest-style helper (avoids a second import in this module)
# ---------------------------------------------------------------------------


def pytest_approx(value: float) -> Any:  # type: ignore[no-redef]
    """A minimal pytest.approx replacement for non-pytest-aware callers.

    Imports :func:`pytest.approx` lazily so this test module does not
    import pytest at module load (it already does, transitively, via
    the test runner; this helper is here for clarity).
    """
    import pytest as _pytest

    return _pytest.approx(value)
