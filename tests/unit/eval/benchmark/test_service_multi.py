"""Unit tests for ``BenchmarkReportService.compare_multi`` (N-way compare).

Reuses the ``test_service`` fakes (``_make_service`` patches the
``BenchmarkReportRepository`` away and replays scripted rows).  These
tests exercise the service-level assembly that sits on top of
:func:`stats.compare_multi`:

* run-count bounds (2..5) + same-dataset invariant → ``ValidationError``;
* unknown id → ``None`` (route maps to 404);
* baseline designation (default first / explicit / not-in-set guard);
* per-run identity + palette colours + baseline flag;
* the per-item matrix (values aligned to run order) + samples
  (answer/contexts; ``None`` for runs that predate sample capture).
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from iw_rag.domain.errors import ValidationError
from tests.unit.eval.benchmark.test_service import _make_service


def _report(
    rid: str,
    ds: str,
    *,
    metrics: dict[str, Any] | None = None,
    n_queries: int = 3,
    generator: str = "gen",
) -> dict[str, Any]:
    """A report row with the fields ``compare_multi`` reads."""
    return {
        "id": rid,
        "dataset_id": ds,
        "n_queries": n_queries,
        "metrics": metrics or {},
        "model_config": {
            "generator": generator,
            "embedder": "bge-m3",
            "reranker": "rr",
            "grader": "judge",
        },
        "report_label": "lbl-" + rid,
        "display_id": "BENCH-" + rid,
        "dataset_name": "benchmark_golden",
        "created_at": "2026-06-1" + rid,
    }


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


def test_compare_multi_rejects_fewer_than_two_runs() -> None:
    """A single id raises ``ValidationError`` (need 2..5)."""
    store = {"a": _report("a", "ds")}
    svc, patcher = _make_service(repo_store=store)
    try:
        with pytest.raises(ValidationError, match=r"at least 2"):
            _run(svc.compare_multi(["a"]))
    finally:
        patcher.stop()


def test_compare_multi_rejects_more_than_five_runs() -> None:
    """Six ids raise ``ValidationError`` (cap is 5)."""
    store = {c: _report(c, "ds") for c in "abcdef"}
    svc, patcher = _make_service(repo_store=store)
    try:
        with pytest.raises(ValidationError, match=r"at most 5"):
            _run(svc.compare_multi(list("abcdef")))
    finally:
        patcher.stop()


def test_compare_multi_dedupes_ids() -> None:
    """Duplicate ids collapse; ``[a, a]`` is a single run → too few."""
    store = {"a": _report("a", "ds"), "b": _report("b", "ds")}
    svc, patcher = _make_service(repo_store=store)
    try:
        with pytest.raises(ValidationError, match=r"at least 2"):
            _run(svc.compare_multi(["a", "a"]))
    finally:
        patcher.stop()


def test_compare_multi_returns_none_when_an_id_is_unknown() -> None:
    """An unknown id makes the whole compare ``None`` (→ 404 at the route)."""
    store = {"a": _report("a", "ds")}
    svc, patcher = _make_service(repo_store=store)
    try:
        assert _run(svc.compare_multi(["a", "ghost"])) is None
    finally:
        patcher.stop()


def test_compare_multi_rejects_mixed_datasets() -> None:
    """Reports on different datasets raise ``ValidationError``."""
    store = {"a": _report("a", "ds-1"), "b": _report("b", "ds-2")}
    svc, patcher = _make_service(repo_store=store)
    try:
        with pytest.raises(ValidationError, match=r"different datasets"):
            _run(svc.compare_multi(["a", "b"]))
    finally:
        patcher.stop()


def test_compare_multi_unknown_baseline_id_raises() -> None:
    """A ``baseline_id`` not among the compared reports raises."""
    store = {"a": _report("a", "ds"), "b": _report("b", "ds")}
    svc, patcher = _make_service(repo_store=store)
    try:
        with pytest.raises(ValidationError, match=r"not among the compared"):
            _run(svc.compare_multi(["a", "b"], baseline_id="zzz"))
    finally:
        patcher.stop()


def test_compare_multi_default_baseline_is_first_id() -> None:
    """With no explicit baseline, the first id is the baseline."""
    store = {
        "a": _report("a", "ds", metrics={"ndcg_at_5": [0.4, 0.5, 0.6]}),
        "b": _report("b", "ds", metrics={"ndcg_at_5": [0.5, 0.6, 0.7]}),
    }
    svc, patcher = _make_service(repo_store=store)
    try:
        out = _run(svc.compare_multi(["a", "b"]))
    finally:
        patcher.stop()
    assert out["baseline_id"] == "a"
    assert out["runs"][0]["is_baseline"] is True
    assert out["runs"][1]["is_baseline"] is False


def test_compare_multi_assigns_distinct_palette_colours() -> None:
    """Each run gets a stable palette colour by selection order."""
    store = {
        "a": _report("a", "ds", metrics={"ndcg_at_5": [0.4, 0.5, 0.6]}),
        "b": _report("b", "ds", metrics={"ndcg_at_5": [0.5, 0.6, 0.7]}),
        "c": _report("c", "ds", metrics={"ndcg_at_5": [0.3, 0.4, 0.5]}),
    }
    svc, patcher = _make_service(repo_store=store)
    try:
        out = _run(svc.compare_multi(["a", "b", "c"]))
    finally:
        patcher.stop()
    colours = [r["color"] for r in out["runs"]]
    assert len(set(colours)) == 3
    assert all(c.startswith("#") for c in colours)


def test_compare_multi_surfaces_only_present_metrics() -> None:
    """Only metrics present as a vector in ≥1 run appear in ``metrics``."""
    store = {
        "a": _report("a", "ds", metrics={"ndcg_at_5": [0.4, 0.5, 0.6]}),
        "b": _report("b", "ds", metrics={"ndcg_at_5": [0.5, 0.6, 0.7]}),
    }
    svc, patcher = _make_service(repo_store=store)
    try:
        out = _run(svc.compare_multi(["a", "b"]))
    finally:
        patcher.stop()
    assert out["metrics"] == ["ndcg_at_5"]


def test_compare_multi_builds_per_item_matrix_aligned_to_runs() -> None:
    """Per-item values list is aligned to the run order, ``None`` where absent."""
    store = {
        "a": _report("a", "ds", metrics={"ndcg_at_5": [0.4, 0.5, 0.6]}),
        "b": _report("b", "ds", metrics={"ndcg_at_5": [0.7, 0.8, 0.9]}),
    }
    svc, patcher = _make_service(repo_store=store)
    try:
        out = _run(svc.compare_multi(["a", "b"]))
    finally:
        patcher.stop()
    assert len(out["items"]) == 3
    assert out["items"][0]["values"]["ndcg_at_5"] == [0.4, 0.7]
    assert out["items"][2]["values"]["ndcg_at_5"] == [0.6, 0.9]


def test_compare_multi_includes_samples_and_question() -> None:
    """Per-item samples carry question/answer/contexts; absent runs give ``None``."""
    samples = [
        {"question": "Q1", "answer": "A1", "contexts": ["c1"]},
        {"question": "Q2", "answer": "A2", "contexts": []},
        {"question": "Q3", "answer": "A3", "contexts": ["c3a", "c3b"]},
    ]
    store = {
        "a": _report("a", "ds", metrics={"ndcg_at_5": [0.4, 0.5, 0.6], "samples": samples}),
        "b": _report("b", "ds", metrics={"ndcg_at_5": [0.7, 0.8, 0.9]}),  # no samples
    }
    svc, patcher = _make_service(repo_store=store)
    try:
        out = _run(svc.compare_multi(["a", "b"], baseline_id="a"))
    finally:
        patcher.stop()
    item0 = out["items"][0]
    assert item0["question"] == "Q1"
    assert item0["samples"][0] == {"answer": "A1", "contexts": ["c1"]}
    # Run b predates sample capture → None sample for it.
    assert item0["samples"][1] is None


def test_compare_multi_constants_present() -> None:
    """The service exposes the run-count bounds + a 5-colour palette."""
    from iw_rag.eval.benchmark import service as svc_mod

    assert svc_mod.MIN_COMPARE_RUNS == 2
    assert svc_mod.MAX_COMPARE_RUNS == 5
    assert len(svc_mod._RUN_PALETTE) == 5
