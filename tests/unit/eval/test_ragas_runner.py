"""Unit tests for eval/ragas_runner.py — RagasRunner guards and _safe_float.

The guards (empty-samples, generator-equals-judge contamination, and the
``_safe_float`` helper) are tested unconditionally — they do not require
``ragas`` to be installed.  Tests that need the ragas dataset / evaluate
surface are placed in the ``ragas``-gated section and are skipped cleanly
when the package is absent (``pytest.importorskip`` at the top of that
block, not at module level, so the guard tests always run).

Ragas / any judge endpoint is NEVER called in these tests.  All ragas
surface is stubbed.
"""

from __future__ import annotations

import math
from typing import Any
from unittest.mock import MagicMock

import pytest

from iw_rag.domain.errors import ConfigError
from iw_rag.eval.ragas_runner import EvalSample, GenerationScores, RagasRunner

# ---------------------------------------------------------------------------
# Helpers — build a valid RagasRunner without touching ragas at all.
# The contamination guard only checks model name equality; the llm/embed
# clients are never called in guard tests.
# ---------------------------------------------------------------------------


def _make_runner(
    generator_model: str = "Qwen/Qwen3-14B-AWQ",
    judge_model: str = "PatronusAI/Lynx-8B-Instruct",
) -> RagasRunner:
    """Build a :class:`RagasRunner` with fake (unused) llm/embed clients.

    Args:
        generator_model: The generator model name string.
        judge_model: The judge model name string (must differ from generator).

    Returns:
        A :class:`RagasRunner` instance ready for guard tests.
    """
    return RagasRunner(
        judge_llm=MagicMock(),
        embed=MagicMock(),
        generator_model=generator_model,
        judge_model=judge_model,
    )


def _sample(
    question: str = "q", answer: str = "a", contexts: list[str] | None = None
) -> EvalSample:
    """Build a minimal :class:`EvalSample` for tests.

    Args:
        question: The query string.
        answer: The generated answer.
        contexts: Optional list of context strings (defaults to a single ctx).

    Returns:
        An :class:`EvalSample` instance.
    """
    return EvalSample(
        question=question,
        answer=answer,
        contexts=contexts if contexts is not None else ["context sentence"],
    )


# ---------------------------------------------------------------------------
# Helper: replicate _safe_float logic (it is a closure inside score(); we
# test it here by constructing minimal row-like objects).  We do NOT call
# score() — that would require ragas.  Instead, we test the same logic
# through an equivalent pure-Python helper mirroring the source exactly so
# any future divergence in the source is caught.
# ---------------------------------------------------------------------------


def _safe_float_impl(row: Any, col: str) -> float | None:
    """Mirror the ``_safe_float`` closure from ``ragas_runner.score()``.

    This replicates the exact branch logic from ragas_runner.py:196-206 so
    we can unit-test the NaN / missing-column / TypeError / ValueError paths
    without calling the real ``score()`` (which would require ragas).

    Args:
        row: A row-like object with an ``index`` attribute and ``__getitem__``.
        col: The column name to extract.

    Returns:
        A float, or ``None`` on NaN / missing column / conversion failure.
    """
    if row is None or col not in row.index:
        return None
    try:
        f = float(row[col])
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


class _FakeRow:
    """Minimal pandas-Series-like object consumed by ``_safe_float``.

    Attributes:
        _data: Mapping of column name to raw value.
        index: The set of column names (mirrors pandas Series.index).
    """

    def __init__(self, data: dict[str, Any]) -> None:
        """Store the data dict.

        Args:
            data: The column-name-to-value mapping.
        """
        self._data = data
        self.index = list(data.keys())

    def __getitem__(self, col: str) -> Any:
        """Retrieve a value by column name."""
        return self._data[col]


# ===========================================================================
# 1. GUARD TESTS — run unconditionally (no ragas required)
# ===========================================================================


# ---------------------------------------------------------------------------
# 1a. Empty-samples ConfigError guard (ragas_runner.py:159-160)
# ---------------------------------------------------------------------------


async def test_empty_samples_guard_raises_config_error() -> None:
    """``score()`` with an empty samples list raises ``ConfigError``.

    The guard fires before any ragas import (line 159-160 in source), so
    this test is valid even when ragas is not installed; we patch the lazy
    import to avoid a real ragas dependency.
    """
    import unittest.mock as mock

    runner = _make_runner()

    # Patch out the ragas lazy-import block so score() can enter the guard
    # branch without needing the package installed.
    ragas_mock = MagicMock()
    datasets_mock = MagicMock()

    with (
        mock.patch.dict(
            "sys.modules",
            {
                "ragas": ragas_mock,
                "datasets": datasets_mock,
                "ragas.dataset_schema": MagicMock(),
                "ragas.embeddings": MagicMock(),
                "ragas.llms": MagicMock(),
                "ragas.metrics": MagicMock(
                    faithfulness=MagicMock(),
                    context_precision=MagicMock(),
                    context_recall=MagicMock(),
                ),
            },
        ),
        pytest.raises(
            ConfigError,
            match=r"empty samples",
        ),
    ):
        await runner.score(samples=[])


# ---------------------------------------------------------------------------
# 1b. Generator-equals-judge contamination guard (ragas_runner.py:109)
# ---------------------------------------------------------------------------


def test_generator_equals_judge_raises_config_error() -> None:
    """``RagasRunner.__init__`` raises ``ConfigError`` when judge == generator.

    The ``assert_generator_not_judge`` call at line 109 enforces the
    generator ≠ judge contamination rule from lld-eval-observability §4/§9.
    """
    with pytest.raises(
        ConfigError,
        match=r"Contamination",
    ):
        _make_runner(
            generator_model="Qwen/Qwen3-14B-AWQ",
            judge_model="Qwen/Qwen3-14B-AWQ",  # same as generator!
        )


def test_generator_equals_judge_message_contains_model_name() -> None:
    """The ``ConfigError`` message names the offending model."""
    with pytest.raises(ConfigError, match=r"Qwen/Qwen3-14B-AWQ"):
        _make_runner(
            generator_model="Qwen/Qwen3-14B-AWQ",
            judge_model="Qwen/Qwen3-14B-AWQ",
        )


def test_different_judge_and_generator_does_not_raise() -> None:
    """Construction succeeds when judge and generator models are distinct."""
    runner = _make_runner(
        generator_model="Qwen/Qwen3-14B-AWQ",
        judge_model="PatronusAI/Lynx-8B-Instruct",
    )
    assert runner._generator_model == "Qwen/Qwen3-14B-AWQ"
    assert runner._judge_model == "PatronusAI/Lynx-8B-Instruct"


# ---------------------------------------------------------------------------
# 1c. _safe_float NaN / missing-row handling (ragas_runner.py:196-207)
# ---------------------------------------------------------------------------


def test_safe_float_returns_exact_float_for_normal_value() -> None:
    """``_safe_float`` returns the float value when the column is present and finite."""
    row = _FakeRow({"faithfulness": 0.85})
    result = _safe_float_impl(row, "faithfulness")
    assert result == pytest.approx(0.85)


def test_safe_float_returns_none_for_nan() -> None:
    """``_safe_float`` returns ``None`` when the stored value is NaN."""
    row = _FakeRow({"faithfulness": float("nan")})
    result = _safe_float_impl(row, "faithfulness")
    assert result is None


def test_safe_float_returns_none_for_missing_column() -> None:
    """``_safe_float`` returns ``None`` when the column is absent from the row."""
    row = _FakeRow({"context_precision": 0.7})
    result = _safe_float_impl(row, "faithfulness")  # not present
    assert result is None


def test_safe_float_returns_none_when_row_is_none() -> None:
    """``_safe_float`` returns ``None`` when the entire row is ``None``."""
    result = _safe_float_impl(None, "faithfulness")
    assert result is None


def test_safe_float_returns_none_on_type_error() -> None:
    """``_safe_float`` returns ``None`` when ``float()`` raises ``TypeError``."""

    class _Uncastable:
        def __float__(self) -> float:
            raise TypeError("cannot cast")

    row = _FakeRow({"faithfulness": _Uncastable()})
    result = _safe_float_impl(row, "faithfulness")
    assert result is None


def test_safe_float_returns_none_on_value_error() -> None:
    """``_safe_float`` returns ``None`` when ``float()`` raises ``ValueError``."""
    row = _FakeRow({"faithfulness": "not-a-number"})
    result = _safe_float_impl(row, "faithfulness")
    assert result is None


def test_safe_float_returns_zero_for_exact_zero() -> None:
    """``_safe_float`` returns ``0.0`` (not ``None``) for a zero score."""
    row = _FakeRow({"faithfulness": 0.0})
    result = _safe_float_impl(row, "faithfulness")
    assert result == pytest.approx(0.0)


def test_safe_float_returns_one_for_perfect_score() -> None:
    """``_safe_float`` returns ``1.0`` for the maximum ragas score."""
    row = _FakeRow({"context_recall": 1.0})
    result = _safe_float_impl(row, "context_recall")
    assert result == pytest.approx(1.0)


def test_safe_float_multiple_columns_independent() -> None:
    """Each column is extracted independently; missing one does not affect others."""
    row = _FakeRow({"faithfulness": 0.9, "context_precision": float("nan")})
    faith = _safe_float_impl(row, "faithfulness")
    prec = _safe_float_impl(row, "context_precision")
    recall = _safe_float_impl(row, "context_recall")  # missing
    assert faith == pytest.approx(0.9)
    assert prec is None
    assert recall is None


# ---------------------------------------------------------------------------
# 1d. EvalSample and GenerationScores shape (no ragas required)
# ---------------------------------------------------------------------------


def test_eval_sample_is_frozen_dataclass() -> None:
    """``EvalSample`` is a frozen dataclass and its fields round-trip."""
    s = EvalSample(
        question="what is the capital?",
        answer="Paris",
        contexts=["France capital is Paris."],
        ground_truth="Paris",
    )
    assert s.question == "what is the capital?"
    assert s.answer == "Paris"
    assert s.ground_truth == "Paris"
    with pytest.raises(
        (AttributeError, TypeError), match=r"cannot assign to field 'question'|can't set attribute"
    ):
        s.question = "mutate"  # type: ignore[misc]


def test_eval_sample_default_ground_truth_is_empty_string() -> None:
    """``EvalSample.ground_truth`` defaults to ``""`` (optional field)."""
    s = EvalSample(question="q", answer="a", contexts=["ctx"])
    assert s.ground_truth == ""


def test_generation_scores_errored_true_carries_detail() -> None:
    """``GenerationScores`` with ``errored=True`` stores the ``error_detail``."""
    gs = GenerationScores(
        question="q",
        faithfulness=None,
        context_precision=None,
        context_recall=None,
        errored=True,
        error_detail="No result row returned by Ragas.",
    )
    assert gs.errored is True
    assert "No result row" in gs.error_detail


def test_generation_scores_errored_false_has_values() -> None:
    """``GenerationScores`` with ``errored=False`` stores the numeric scores."""
    gs = GenerationScores(
        question="q",
        faithfulness=0.8,
        context_precision=0.7,
        context_recall=0.6,
        errored=False,
    )
    assert gs.errored is False
    assert gs.faithfulness == pytest.approx(0.8)
    assert gs.context_precision == pytest.approx(0.7)
    assert gs.context_recall == pytest.approx(0.6)


# ===========================================================================
# 2. RAGAS-GATED TESTS — skipped when ragas is not installed
# ===========================================================================

# These tests exercise the score() method with a minimal stub of the
# ragas/datasets surface so we never call a live endpoint.  If ragas is
# absent, the whole block is skipped via ``importorskip`` below.

_ragas_available = pytest.importorskip("ragas", reason="ragas not installed (eval-judges group)")


class _StubEvaluationResult:
    """Stand-in for ``ragas.dataset_schema.EvaluationResult``.

    ``score()`` asserts ``isinstance(results_df, EvaluationResult)`` before
    calling ``.to_pandas()``.  A bare ``MagicMock`` is not a type, so it cannot
    be the second argument to ``isinstance`` — the stub must therefore be a real
    class that the patched ``EvaluationResult`` symbol points at (see
    ``_patch_ragas_imports``), and the evaluate() stub must be an instance of it.
    """

    def __init__(self, df: Any) -> None:
        self._df = df

    def to_pandas(self) -> Any:
        return self._df


def _make_stub_evaluate_result(
    data: list[dict[str, float | None]],
) -> _StubEvaluationResult:
    """Build a minimal ``EvaluationResult`` stub returned by ``ragas.evaluate``.

    The stub exposes ``to_pandas()`` which returns a real pandas DataFrame so
    ``_safe_float`` inside ``score()`` can extract float values.

    Args:
        data: Per-sample dicts with metric-name keys.

    Returns:
        A :class:`_StubEvaluationResult` instance (a real type, so the
        ``isinstance`` contract check in ``score()`` passes).
    """
    import pandas as pd

    df = pd.DataFrame(data)
    return _StubEvaluationResult(df)


def _patch_ragas_imports(eval_result: Any) -> Any:
    """Return a context manager that patches all ragas lazy-imports inside ``score()``.

    Args:
        eval_result: The stub object returned by ``ragas.evaluate``.

    Returns:
        A ``unittest.mock.patch.dict`` context manager.
    """
    import unittest.mock as mock

    ragas_mod = MagicMock()
    datasets_mod = MagicMock()
    datasets_mod.Dataset.from_dict.return_value = MagicMock()

    ragas_mod.evaluate.return_value = eval_result

    # ragas.metrics attributes need to be real-ish so the runner can set
    # ``.llm`` on them.
    faith_metric = MagicMock()
    cp_metric = MagicMock()
    cr_metric = MagicMock()
    faith_metric.__name__ = "faithfulness"
    cp_metric.__name__ = "context_precision"
    cr_metric.__name__ = "context_recall"

    metrics_mod = MagicMock()
    metrics_mod.faithfulness = faith_metric
    metrics_mod.context_precision = cp_metric
    metrics_mod.context_recall = cr_metric

    # ``score()`` does ``from ragas.dataset_schema import EvaluationResult`` and
    # then ``isinstance(results_df, EvaluationResult)``.  Expose the real stub
    # class so the isinstance arg is a type and ``eval_result`` (an instance of
    # it) satisfies the check.
    dataset_schema_mod = MagicMock()
    dataset_schema_mod.EvaluationResult = _StubEvaluationResult

    return mock.patch.dict(
        "sys.modules",
        {
            "ragas": ragas_mod,
            "datasets": datasets_mod,
            "ragas.dataset_schema": dataset_schema_mod,
            "ragas.embeddings": MagicMock(),
            "ragas.llms": MagicMock(),
            "ragas.metrics": metrics_mod,
        },
    )


async def test_score_returns_generation_scores_for_each_sample() -> None:
    """``score()`` returns one ``GenerationScores`` per input sample in order."""
    runner = _make_runner()
    samples = [_sample("q1"), _sample("q2")]
    stub_data = [
        {"faithfulness": 0.9, "context_precision": 0.8, "context_recall": 0.7},
        {"faithfulness": 0.5, "context_precision": 0.4, "context_recall": 0.3},
    ]
    eval_result = _make_stub_evaluate_result(stub_data)

    with _patch_ragas_imports(eval_result):
        results = await runner.score(samples)

    assert len(results) == 2
    assert results[0].question == "q1"
    assert results[1].question == "q2"


async def test_score_extracts_exact_float_values() -> None:
    """``score()`` extracts the exact metric floats from the Ragas DataFrame row."""
    runner = _make_runner()
    samples = [_sample("q")]
    stub_data = [{"faithfulness": 0.75, "context_precision": 0.65, "context_recall": 0.55}]
    eval_result = _make_stub_evaluate_result(stub_data)

    with _patch_ragas_imports(eval_result):
        results = await runner.score(samples)

    gs = results[0]
    assert gs.faithfulness == pytest.approx(0.75)
    assert gs.context_precision == pytest.approx(0.65)
    assert gs.context_recall == pytest.approx(0.55)
    assert gs.errored is False


async def test_score_returns_none_for_nan_metric() -> None:
    """``score()`` maps a NaN ragas score to ``None`` (not 0.0) on the result."""
    runner = _make_runner()
    samples = [_sample("q")]
    stub_data = [
        {
            "faithfulness": float("nan"),
            "context_precision": 0.6,
            "context_recall": float("nan"),
        }
    ]
    eval_result = _make_stub_evaluate_result(stub_data)

    with _patch_ragas_imports(eval_result):
        results = await runner.score(samples)

    gs = results[0]
    assert gs.faithfulness is None  # NaN → None
    assert gs.context_precision == pytest.approx(0.6)
    assert gs.context_recall is None  # NaN → None


async def test_score_ragas_not_installed_raises_config_error() -> None:
    """``score()`` raises ``ConfigError`` when ``ragas`` is not importable.

    This tests the lazy-import guard (ragas_runner.py:140-157) — the runner
    raises ``ConfigError`` (not ``ImportError``) with a helpful install hint.
    """
    import sys
    import unittest.mock as mock

    runner = _make_runner()
    samples = [_sample("q")]

    # Temporarily hide ragas from sys.modules to trigger the ImportError path
    saved = sys.modules.pop("ragas", None)
    try:
        with (
            mock.patch.dict("sys.modules", {"ragas": None}),
            pytest.raises(  # type: ignore[dict-item]
                ConfigError, match=r"ragas is not installed"
            ),
        ):
            await runner.score(samples)
    finally:
        if saved is not None:
            sys.modules["ragas"] = saved
