"""Exact-value unit tests for the judge-free retrieval metrics."""

from __future__ import annotations

from math import log2

import pytest

from iw_rag.domain.models import RetrievalResult, RetrievedChunk
from iw_rag.eval.retrieval_metrics import (
    mrr,
    ndcg_at_k,
    recall_at_k,
    score_retrieval,
)


def test_recall_partial() -> None:
    # 2 of 3 gold in top-5
    assert recall_at_k(["a", "x", "b", "y", "z"], ["a", "b", "c"], 5) == pytest.approx(2 / 3)


def test_recall_all_gold_beyond_k_is_zero() -> None:
    assert recall_at_k(["x", "y", "a"], ["a"], 2) == 0.0


def test_recall_empty_gold_is_zero() -> None:
    assert recall_at_k(["a"], [], 5) == 0.0


def test_ndcg_single_gold_rank1_is_one() -> None:
    assert ndcg_at_k(["a", "x"], ["a"], 5) == pytest.approx(1.0)


def test_ndcg_single_gold_rank2() -> None:
    assert ndcg_at_k(["x", "a"], ["a"], 5) == pytest.approx(1.0 / log2(3))


def test_ndcg_two_gold_ranks_1_and_3() -> None:
    # DCG = 1/log2(2) + 1/log2(4) = 1 + 0.5 ; IDCG = 1/log2(2) + 1/log2(3)
    dcg = 1.0 + 1.0 / log2(4)
    idcg = 1.0 + 1.0 / log2(3)
    assert ndcg_at_k(["a", "x", "b"], ["a", "b"], 5) == pytest.approx(dcg / idcg, abs=1e-9)


def test_ndcg_empty_gold_is_zero() -> None:
    assert ndcg_at_k(["a"], [], 5) == 0.0


def test_mrr_first_hit_rank1() -> None:
    assert mrr(["a", "b"], ["a"]) == pytest.approx(1.0)


def test_mrr_first_hit_rank3() -> None:
    assert mrr(["x", "y", "a"], ["a"]) == pytest.approx(1 / 3)


def test_mrr_no_hit_is_zero() -> None:
    assert mrr(["x", "y"], ["a"]) == 0.0


def test_k_larger_than_list_is_safe() -> None:
    assert recall_at_k(["a"], ["a"], 10) == pytest.approx(1.0)


def test_determinism() -> None:
    r = ["a", "b", "c", "d"]
    g = ["c", "a"]
    assert ndcg_at_k(r, g, 10) == ndcg_at_k(r, g, 10)
    assert recall_at_k(r, g, 10) == recall_at_k(r, g, 10)


def _result(ids: list[str]) -> RetrievalResult:
    return RetrievalResult(
        chunks=[
            RetrievedChunk(
                document_id="doc-chunk", chunk_id=i, content="", score=1.0 - n * 0.1, source_uri="u"
            )
            for n, i in enumerate(ids)
        ],
        retriever="spine",
        acl_applied=True,
    )


def test_score_retrieval_does_not_resort() -> None:
    result = _result(["a", "x", "b"])  # best-first per spine invariant
    scores = score_retrieval(result, ["a", "b"], ks=(5, 10))
    assert scores.recall[5] == pytest.approx(1.0)
    assert scores.mrr == pytest.approx(1.0)
    assert scores.k_values == (5, 10)
    # nDCG matches the primitive on the same order (no re-sort)
    assert scores.ndcg[10] == pytest.approx(ndcg_at_k(["a", "x", "b"], ["a", "b"], 10))


def _merged_result() -> RetrievalResult:
    """Rank 1 is a section that absorbed leaf ``L1``; rank 2 is an unrelated chunk."""
    section = RetrievedChunk(
        chunk_id="S1",
        document_id="d",
        content="section",
        score=1.0,
        source_uri="x",
        metadata={"merged_chunk_ids": ["L0", "L1"]},
    )
    other = RetrievedChunk(chunk_id="x", document_id="d", content="o", score=0.5, source_uri="x")
    return RetrievalResult(chunks=[section, other], retriever="spine", acl_applied=True)


def test_a_chunk_that_absorbed_repeats_counts_as_each_id_it_covers() -> None:
    # Expanded ranking: S1, L0, L1, x.
    scores = score_retrieval(_merged_result(), ["S1", "L0", "L1"], ks=(3, 5))
    assert scores.recall[3] == pytest.approx(1.0)  # all three ids were delivered in one chunk
    assert scores.mrr == pytest.approx(1.0)
    assert scores.ndcg[5] == pytest.approx(
        ndcg_at_k(["S1", "L0", "L1", "x"], ["S1", "L0", "L1"], 5)
    )


def test_merged_ids_follow_their_representative_and_do_not_invent_hits() -> None:
    result = _merged_result()
    assert score_retrieval(result, ["L1"], ks=(5,)).mrr == pytest.approx(1 / 3)  # S1, L0, then L1
    assert score_retrieval(result, ["x"], ks=(5,)).mrr == pytest.approx(1 / 4)
    assert score_retrieval(result, ["nowhere"], ks=(5,)).recall[5] == 0.0


def test_a_result_with_no_merges_is_scored_exactly_as_before() -> None:
    plain = _result(["a", "x", "b"])
    scores = score_retrieval(plain, ["a", "b"], ks=(5,))
    assert scores.recall[5] == pytest.approx(recall_at_k(["a", "x", "b"], ["a", "b"], 5))
    assert scores.ndcg[5] == pytest.approx(ndcg_at_k(["a", "x", "b"], ["a", "b"], 5))
