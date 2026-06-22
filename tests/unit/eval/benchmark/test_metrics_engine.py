"""Unit tests for the metrics engine (eval/benchmark/metrics_engine.py).

TDD RED-first: these tests are written before the engine implementation
(S02) and exercise its public contract — the typed per-run result and the
failure/edge handling required by F-00093 AC1/AC4/AC5 and the Boundary
Behavior table (empty golden set, judge down, no reference answers,
per-sample NaN).

The tests use fake judges (no live model, no network) — the engine MUST
inject its judge/runner wiring so unit tests stay hermetic.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from iw_rag.domain.errors import ConfigError, JudgeUnavailable
from iw_rag.domain.models import RetrievalResult, RetrievedChunk
from iw_rag.eval.repository import GoldenQuery

# ---------------------------------------------------------------------------
# Helpers / fakes
# ---------------------------------------------------------------------------


def _result(ids: list[str], *, retriever: str = "spine") -> RetrievalResult:
    """Build a best-first RetrievalResult from a list of chunk ids."""
    return RetrievalResult(
        chunks=[
            RetrievedChunk(
                document_id="doc-chunk", chunk_id=i, content="", score=1.0 - n * 0.1, source_uri="u"
            )
            for n, i in enumerate(ids)
        ],
        retriever=retriever,
        acl_applied=True,
    )


def _make_faithfulness_judge(verdicts: list[tuple[bool, float]]) -> Any:
    """Build a fake FaithfulnessJudge that replays (passed, score) verdicts.

    Args:
        verdicts: Sequence of (passed, score) tuples — one per query.

    Returns:
        A MagicMock with an async ``judge`` method that replays the verdicts
        in order and raises ``JudgeUnavailable`` once the list is exhausted.
    """
    copy = list(verdicts)
    calls = {"n": 0}

    async def _judge(question: str, context: str, answer: str) -> Any:
        idx = calls["n"]
        calls["n"] += 1
        if idx >= len(copy):
            raise JudgeUnavailable("fake judge exhausted")
        passed, score = copy[idx]
        return MagicMock(passed=passed, score=score, reason=f"verdict-{idx}")

    j = MagicMock()
    j.judge = AsyncMock(side_effect=_judge)
    j.call_count = lambda: calls["n"]  # type: ignore[assignment]
    return j


def _make_faithfulness_judge_erroring() -> Any:
    """Build a fake FaithfulnessJudge whose ``judge`` always raises JudgeUnavailable."""
    j = MagicMock()

    async def _boom(question: str, context: str, answer: str) -> Any:
        raise JudgeUnavailable("judge endpoint down")

    j.judge = AsyncMock(side_effect=_boom)
    return j


def _make_grading_judge(scores: list[float]) -> Any:
    """Build a fake GradingJudge that replays numeric scores for each grade call."""
    copy = list(scores)
    calls = {"n": 0}

    async def _grade(rubric: Any, sample: Any, *, position_swap: bool = True) -> Any:
        idx = calls["n"]
        calls["n"] += 1
        if idx >= len(copy):
            raise JudgeUnavailable("fake grading judge exhausted")
        return MagicMock(score=copy[idx], reason=f"grade-{idx}", stable=True)

    j = MagicMock()
    j.grade = AsyncMock(side_effect=_grade)
    j.call_count = lambda: calls["n"]  # type: ignore[assignment]
    return j


def _make_ragas_runner(scores: list[dict[str, float | None]] | None) -> Any:
    """Build a fake RagasRunner.

    Args:
        scores: A list of per-sample score dicts (keys: ``faithfulness``,
            ``context_precision``, ``context_recall``, optionally
            ``factual_correctness``).  Each entry maps to a ``GenerationScores``
            with ``errored`` always False.  ``None`` means the runner is not
            available (i.e. ragas is not installed); in that case the
            ``score`` method is not expected to be called.
    """
    runner = MagicMock()

    if scores is None:
        # Not-available: ``score`` should never be called by the engine
        async def _not_called(samples: Sequence[Any], metrics: Any = ()) -> list[Any]:
            raise AssertionError("RagasRunner.score must not be called when not injected")

        runner.score = AsyncMock(side_effect=_not_called)
        return runner

    copy = list(scores)
    calls = {"n": 0}

    async def _score(samples: Sequence[Any], metrics: Any = ()) -> list[Any]:
        calls["n"] += 1
        out: list[Any] = []
        for i, s in enumerate(samples):
            entry = copy[i] if i < len(copy) else {}
            gs = MagicMock()
            gs.question = s.question
            gs.faithfulness = entry.get("faithfulness")
            gs.context_precision = entry.get("context_precision")
            gs.context_recall = entry.get("context_recall")
            gs.factual_correctness = entry.get("factual_correctness")
            gs.noise_sensitivity = entry.get("noise_sensitivity")
            gs.errored = False
            gs.error_detail = ""
            out.append(gs)
        return out

    runner.score = AsyncMock(side_effect=_score)
    runner.call_count = lambda: calls["n"]  # type: ignore[assignment]
    return runner


def _golden_query(
    question: str,
    gold: list[str],
    *,
    reference: str | None = None,
) -> GoldenQuery:
    """Build a GoldenQuery for tests."""
    return GoldenQuery(
        question=question,
        relevant_chunk_ids=gold,
        reference_answer=reference,
    )


def _make_inputs(
    queries: list[GoldenQuery],
    retrievals: list[RetrievalResult],
    answers: list[str] | None = None,
    contexts: list[list[str]] | None = None,
) -> tuple[Sequence[GoldenQuery], Sequence[RetrievalResult], Sequence[str], Sequence[list[str]]]:
    """Build a 4-tuple of (queries, retrievals, answers, contexts) for the engine.

    ``answers`` and ``contexts`` default to a 1-element list of placeholder per
    query, so callers only have to specify them when they want a custom value.
    """
    n = len(queries)
    if answers is None:
        answers = [f"answer-for-{q.question[:8]}" for q in queries]
    if contexts is None:
        contexts = [["ctx"] for _ in range(n)]
    assert len(retrievals) == n
    assert len(answers) == n
    assert len(contexts) == n
    return queries, retrievals, answers, contexts


# ---------------------------------------------------------------------------
# Public API surface (compilation-only — drives the imports the engine exposes)
# ---------------------------------------------------------------------------


def test_engine_module_exposes_typed_result_and_engine() -> None:
    """The engine module MUST expose ``BenchmarkMetrics`` and ``MetricsEngine``."""
    from iw_rag.eval.benchmark import metrics_engine as me

    assert hasattr(me, "BenchmarkMetrics")
    assert hasattr(me, "MetricsEngine")


def test_benchmark_metrics_is_frozen_dataclass() -> None:
    """``BenchmarkMetrics`` is a frozen dataclass (immutable per-run result)."""
    from iw_rag.eval.benchmark.metrics_engine import BenchmarkMetrics

    bm = BenchmarkMetrics(
        recall_at_5=0.5,
        recall_at_10=0.6,
        ndcg_at_5=0.4,
        ndcg_at_10=0.5,
        mrr=0.3,
        faithfulness_mean=0.9,
        faithfulness_kappa=None,
        context_precision=0.7,
        context_recall=0.6,
        factual_correctness=None,
        n_queries=3,
        n_judged=3,
        errored=0,
        metrics={},
    )
    with pytest.raises((AttributeError, Exception), match="cannot assign to field"):
        bm.recall_at_5 = 0.0  # type: ignore[misc]
    assert bm.n_queries == 3
    assert bm.factual_correctness is None


# ---------------------------------------------------------------------------
# 1. Empty golden set → raises (Boundary: empty golden set)
# ---------------------------------------------------------------------------


async def test_empty_golden_set_raises() -> None:
    """An empty queries list MUST raise — never produce an all-zero report."""
    from iw_rag.eval.benchmark.metrics_engine import MetricsEngine

    engine = MetricsEngine(
        faithfulness_judge=_make_faithfulness_judge([]),
        grading_judge=_make_grading_judge([]),
        ragas_runner=_make_ragas_runner([]),
    )
    with pytest.raises(
        (ConfigError, ValueError), match="MetricsEngine.compute called with an empty queries list"
    ):
        await engine.compute(queries=[], retrievals=[], answers=[], contexts=[])


# ---------------------------------------------------------------------------
# 2. Tier unification with fake judges (AC1, AC4, AC5, Invariant 1/5/6)
# ---------------------------------------------------------------------------


async def test_compute_unifies_t1_t2_t3_with_reference_answers() -> None:
    """With reference answers, T1 + T2 (faithfulness + context precision) + T3
    (context recall) all populate, and per-query vectors are paired by index.
    """
    from iw_rag.eval.benchmark.metrics_engine import MetricsEngine

    queries = [
        _golden_query("q1", ["a", "b"], reference="ref-1"),
        _golden_query("q2", ["c"], reference="ref-2"),
        _golden_query("q3", ["d", "e", "f"], reference="ref-3"),
    ]
    retrievals = [
        _result(["a", "b", "x"]),
        _result(["x", "y", "c"]),
        _result(["d", "z", "e", "f"]),
    ]
    args = _make_inputs(queries, retrievals)

    # Faithfulness: 1.0, 0.5, 0.0
    faith = _make_faithfulness_judge([(True, 1.0), (True, 0.5), (False, 0.0)])
    # Grading (relevancy/rubric): 4, 3, 5
    grade = _make_grading_judge([4.0, 3.0, 5.0])
    # Ragas (per-sample T2/T3): 3 samples, all metrics.
    ragas = _make_ragas_runner(
        [
            {
                "faithfulness": 1.0,
                "context_precision": 0.9,
                "context_recall": 0.8,
            },
            {
                "faithfulness": 0.5,
                "context_precision": 0.6,
                "context_recall": 0.7,
            },
            {
                "faithfulness": 0.0,
                "context_precision": 0.2,
                "context_recall": 0.1,
            },
        ]
    )

    engine = MetricsEngine(
        faithfulness_judge=faith,
        grading_judge=grade,
        ragas_runner=ragas,
    )
    bm = await engine.compute(*args, heavy_ragas=True)  # exercise the full Ragas suite

    # T1 means
    assert bm.recall_at_5 is not None and bm.recall_at_5 > 0.0
    assert bm.recall_at_10 is not None
    assert bm.ndcg_at_5 is not None
    assert bm.ndcg_at_10 is not None
    assert bm.mrr is not None

    # T2 — faithfulness and context_precision populated
    assert bm.faithfulness_mean == pytest.approx((1.0 + 0.5 + 0.0) / 3.0)
    assert bm.context_precision == pytest.approx((0.9 + 0.6 + 0.2) / 3.0)
    # T3 — context_recall populated when reference answers present
    assert bm.context_recall == pytest.approx((0.8 + 0.7 + 0.1) / 3.0)

    # Counts
    assert bm.n_queries == 3
    assert bm.n_judged == 3
    assert bm.errored == 0

    # Per-query vectors are paired by query index (keyed by metric name)
    assert "ndcg_at_5" in bm.metrics
    assert "faithfulness" in bm.metrics
    assert "context_recall" in bm.metrics
    assert len(bm.metrics["ndcg_at_5"]) == 3
    assert len(bm.metrics["faithfulness"]) == 3


# ---------------------------------------------------------------------------
# 3. Reference-required gating (AC4, Invariant 5)
# ---------------------------------------------------------------------------


async def test_reference_required_metrics_are_none_without_reference_answers() -> None:
    """When NO reference_answer is present, context_recall / factual_correctness
    MUST be None and the engine MUST NOT make a judge call for them.
    """
    from iw_rag.eval.benchmark.metrics_engine import MetricsEngine

    queries = [
        _golden_query("q1", ["a", "b"], reference=None),
        _golden_query("q2", ["c"], reference=None),
    ]
    retrievals = [_result(["a", "b", "x"]), _result(["c", "x", "y"])]
    args = _make_inputs(queries, retrievals)

    faith = _make_faithfulness_judge([(True, 1.0), (True, 1.0)])
    grade = _make_grading_judge([4.0, 4.0])
    # Ragas IS injected, but the engine must skip T3 calls when no references exist.
    ragas = _make_ragas_runner(
        [
            {
                "faithfulness": 1.0,
                "context_precision": 0.9,
                "context_recall": 0.8,
            },
            {
                "faithfulness": 1.0,
                "context_precision": 0.7,
                "context_recall": 0.6,
            },
        ]
    )

    engine = MetricsEngine(
        faithfulness_judge=faith,
        grading_judge=grade,
        ragas_runner=ragas,
    )
    # heavy_ragas=True so Ragas runs at all (the whole suite is opt-in now); the
    # point of this test is the REFERENCE gating within Ragas, not the toggle.
    bm = await engine.compute(*args, heavy_ragas=True)

    # T3 reference-required metrics are None
    assert bm.context_recall is None
    assert bm.factual_correctness is None

    # T1 + T2 still computed
    assert bm.faithfulness_mean == pytest.approx(1.0)
    assert bm.context_precision is not None
    assert bm.n_queries == 2
    assert bm.n_judged == 2

    # The ragas runner SHOULD have been called for T2 metrics only (faithfulness
    # + context_precision), not for context_recall / factual_correctness.
    # We assert the metric argument passed to ``score`` excludes T3 names.
    if ragas.score.await_args is not None:
        kwargs = ragas.score.await_args.kwargs
        args_ = ragas.score.await_args.args
        metrics = kwargs.get("metrics", args_[1] if len(args_) > 1 else None)
        assert metrics is not None
        assert "context_recall" not in metrics
        assert "factual_correctness" not in metrics
        # faithfulness is NOT requested from Ragas — it comes from the dedicated
        # FaithfulnessJudge (Lynx); asking Ragas for it too only burns redundant
        # judge calls (the engine never reads Ragas's faithfulness back).
        assert "faithfulness" not in metrics
        assert "context_precision" in metrics


# ---------------------------------------------------------------------------
# 4. Judge endpoint down → errored/None, T1 still computed (Boundary: judge down)
# ---------------------------------------------------------------------------


async def test_judge_down_records_errored_and_still_computes_t1() -> None:
    """When the faithfulness judge raises JudgeUnavailable, T2 metrics are
    None (errored) and T1 retrieval metrics + report still produce.
    """
    from iw_rag.eval.benchmark.metrics_engine import MetricsEngine

    queries = [
        _golden_query("q1", ["a"], reference="ref-1"),
        _golden_query("q2", ["c"], reference="ref-2"),
    ]
    retrievals = [_result(["a", "x"]), _result(["c", "x", "y"])]
    args = _make_inputs(queries, retrievals)

    # Faithfulness judge always raises (endpoint down).
    faith = _make_faithfulness_judge_erroring()
    # Grading judge also raises.
    grade = _make_grading_judge_erroring()  # type: ignore[arg-type]
    # Ragas runner is not injected (None) — T2/T3 still degrade gracefully.
    ragas = None

    engine = MetricsEngine(
        faithfulness_judge=faith,
        grading_judge=grade,
        ragas_runner=ragas,
    )
    bm = await engine.compute(*args)

    # T1 still computed
    assert bm.recall_at_5 is not None
    assert bm.ndcg_at_5 is not None
    assert bm.mrr is not None
    # T2 judged metrics: None (errored) — not zero — so the report honestly
    # says the judge didn't run.
    assert bm.faithfulness_mean is None
    assert bm.context_precision is None
    # T3 — still None when references exist AND the judge never ran
    assert bm.context_recall is None
    assert bm.factual_correctness is None
    # ``errored`` count > 0
    assert bm.errored >= 1
    # The number of queries is preserved
    assert bm.n_queries == 2


def _make_grading_judge_erroring() -> Any:
    """Build a fake GradingJudge whose ``grade`` always raises JudgeUnavailable."""
    j = MagicMock()

    async def _boom(rubric: Any, sample: Any, *, position_swap: bool = True) -> Any:
        raise JudgeUnavailable("grading judge endpoint down")

    j.grade = AsyncMock(side_effect=_boom)
    return j


# ---------------------------------------------------------------------------
# 5. Per-sample NaN/parse failure — mirrors ragas_runner ``errored`` handling
# ---------------------------------------------------------------------------


async def test_per_sample_nan_records_as_errored_and_keeps_run() -> None:
    """A per-sample NaN in the ragas output is recorded as ``errored`` —
    the run does NOT crash, the rest of the report is still produced.
    """
    from iw_rag.eval.benchmark.metrics_engine import MetricsEngine

    queries = [
        _golden_query("q1", ["a"], reference="ref-1"),
        _golden_query("q2", ["b"], reference="ref-2"),
    ]
    retrievals = [_result(["a", "x"]), _result(["b", "x", "y"])]
    args = _make_inputs(queries, retrievals)

    faith = _make_faithfulness_judge([(True, 1.0), (True, 1.0)])
    grade = _make_grading_judge([4.0, 4.0])

    # Build a ragas fake that returns one ``errored=True`` GenerationScores.
    runner = MagicMock()
    call_args: list[Any] = []

    async def _score(samples: Sequence[Any], metrics: Any = ()) -> list[Any]:
        call_args.append(list(samples))
        out: list[Any] = []
        for i, s in enumerate(samples):
            gs = MagicMock()
            gs.question = s.question
            if i == 0:
                gs.faithfulness = None
                gs.context_precision = None
                gs.context_recall = None
                gs.factual_correctness = None
                gs.errored = True
                gs.error_detail = "NaN parse failure"
            else:
                gs.faithfulness = 0.7
                gs.context_precision = 0.6
                gs.context_recall = 0.5
                gs.factual_correctness = 0.4
                gs.errored = False
                gs.error_detail = ""
            out.append(gs)
        return out

    runner.score = AsyncMock(side_effect=_score)

    engine = MetricsEngine(
        faithfulness_judge=faith,
        grading_judge=grade,
        ragas_runner=runner,
    )
    bm = await engine.compute(*args, heavy_ragas=True)  # exercise the Ragas NaN path

    # Run did not crash, T1 + n_judged + partial T2/T3 all still produced.
    assert bm.n_queries == 2
    assert bm.faithfulness_mean is not None
    assert bm.context_precision is not None
    assert bm.context_recall is not None
    assert bm.factual_correctness is not None
    # The errored sample is reflected in the errored count.
    assert bm.errored >= 1
    # ragas was called once with the available samples
    assert len(call_args) == 1
    assert len(call_args[0]) == 2


# ---------------------------------------------------------------------------
# 6. Per-tier blob is a superset (AC1) — per-query vectors paired by query idx
# ---------------------------------------------------------------------------


async def test_metrics_blob_contains_paired_per_query_vectors() -> None:
    """The ``metrics`` blob holds per-query vectors for every continuous
    metric, paired by query index (so the stats module can run paired tests).
    """
    from iw_rag.eval.benchmark.metrics_engine import MetricsEngine

    queries = [
        _golden_query("q1", ["a"], reference="r1"),
        _golden_query("q2", ["b"], reference="r2"),
    ]
    retrievals = [_result(["a", "x"]), _result(["b", "x"])]
    args = _make_inputs(queries, retrievals)

    faith = _make_faithfulness_judge([(True, 0.8), (True, 0.6)])
    grade = _make_grading_judge([4.0, 3.0])
    ragas = _make_ragas_runner(
        [
            {"faithfulness": 0.8, "context_precision": 0.7, "context_recall": 0.5},
            {"faithfulness": 0.6, "context_precision": 0.4, "context_recall": 0.3},
        ]
    )

    bm = await MetricsEngine(
        faithfulness_judge=faith,
        grading_judge=grade,
        ragas_runner=ragas,
    ).compute(*args)

    for key in (
        "recall_at_5",
        "recall_at_10",
        "ndcg_at_5",
        "ndcg_at_10",
        "mrr",
        "faithfulness",
        "context_precision",
        "context_recall",
    ):
        assert key in bm.metrics, f"per-query vector missing: {key}"
        assert len(bm.metrics[key]) == 2
    # Per-query vectors are floats (None for individual samples is permitted)
    for v in bm.metrics["faithfulness"]:
        assert v is None or isinstance(v, float)


# ---------------------------------------------------------------------------
# 7. Mismatched input lengths → raises (defensive guard)
# ---------------------------------------------------------------------------


async def test_compute_mismatched_input_lengths_raises() -> None:
    """If ``queries``, ``retrievals``, ``answers``, ``contexts`` have
    different lengths, the engine MUST raise (defensive — the
    orchestrator is responsible for keeping the four sequences aligned,
    but a future bug that drifts them is caught at the boundary)."""
    from iw_rag.eval.benchmark.metrics_engine import MetricsEngine

    queries = [_golden_query("q1", ["a"]), _golden_query("q2", ["b"])]
    retrievals = [_result(["a", "x"])]  # length 1, queries length 2
    answers = ["ans-1", "ans-2"]
    contexts = [["ctx-1"], ["ctx-2"]]
    engine = MetricsEngine(
        faithfulness_judge=_make_faithfulness_judge([]),
        grading_judge=_make_grading_judge([]),
        ragas_runner=_make_ragas_runner([]),
    )
    with pytest.raises(
        (ConfigError, ValueError),
        match="MetricsEngine.compute received mismatched input lengths",
    ):
        await engine.compute(queries, retrievals, answers, contexts)


# ---------------------------------------------------------------------------
# 8. Ragas not injected → T2/T3 judged metrics are None, run still produces
# (mirrors Boundary Behavior "RagasRunner not installed" — the engine
# degrades gracefully and never raises).
# ---------------------------------------------------------------------------


async def test_ragas_not_injected_records_errored_and_run_succeeds() -> None:
    """When ``ragas_runner=None`` (ragas is not installed in this
    deployment), the engine still produces T1 + the report.  T2/T3
    judged metrics are ``None`` and the per-sample ``errored`` count
    reflects the unjudged queries.  No fallback to the generator as
    judge (Invariant 6)."""
    from iw_rag.eval.benchmark.metrics_engine import MetricsEngine

    queries = [
        _golden_query("q1", ["a"], reference="ref-1"),
        _golden_query("q2", ["b"], reference="ref-2"),
    ]
    retrievals = [_result(["a", "x"]), _result(["b", "x", "y"])]
    args = _make_inputs(queries, retrievals)

    # Use the (faith, grade) judges that always succeed; ragas is None.
    faith = _make_faithfulness_judge([(True, 0.9), (True, 0.8)])
    grade = _make_grading_judge([4.0, 3.0])
    engine = MetricsEngine(
        faithfulness_judge=faith,
        grading_judge=grade,
        ragas_runner=None,  # ragas not installed
    )
    bm = await engine.compute(*args)
    # T1 still populated
    assert bm.recall_at_5 is not None
    assert bm.mrr is not None
    # T2 faithfulness still populated (pointwise judge path)
    assert bm.faithfulness_mean is not None
    # T2 context_precision is None (ragas was the only path that produces it)
    assert bm.context_precision is None
    # T3 stays None
    assert bm.context_recall is None
    assert bm.factual_correctness is None
    # n_queries preserved
    assert bm.n_queries == 2
    # errored reflects the unjudged slots
    assert bm.errored >= 2


async def test_ragas_suite_is_opt_in_via_heavy_ragas() -> None:
    """The WHOLE Ragas suite is opt-in: the default run skips Ragas entirely
    (faithfulness-only, fast); ``heavy_ragas=True`` runs the full suite.
    """
    from iw_rag.eval.benchmark.metrics_engine import MetricsEngine

    queries = [_golden_query("q1", ["a"], reference="r1")]
    retrievals = [_result(["a", "x"])]
    args = _make_inputs(queries, retrievals)
    faith = _make_faithfulness_judge([(True, 1.0)])
    grade = _make_grading_judge([4.0])
    ragas = _make_ragas_runner([{"context_precision": 0.9, "context_recall": 0.8}])
    engine = MetricsEngine(faithfulness_judge=faith, grading_judge=grade, ragas_runner=ragas)

    # Default: Ragas is NOT called at all (fast path) — faithfulness still scores.
    bm = await engine.compute(*args, heavy_ragas=False)
    assert ragas.score.await_count == 0
    assert bm.faithfulness_mean == pytest.approx(1.0)
    assert bm.context_precision is None  # Ragas skipped → no context metrics

    # Opt-in: the full Ragas suite runs (with references → all four metrics).
    ragas.score.reset_mock()
    await engine.compute(*args, heavy_ragas=True)
    call = ragas.score.await_args
    m = call.kwargs.get("metrics", call.args[1] if len(call.args) > 1 else ())
    assert {
        "context_precision",
        "context_recall",
        "factual_correctness",
        "noise_sensitivity",
    } <= set(m)
