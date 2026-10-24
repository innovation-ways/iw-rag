"""Hybrid-search lab: the recorded real run, replayed through the production RRF."""

from __future__ import annotations

import json
from dataclasses import asdict

import pytest

from iw_rag.learn.hybrid_lab import get_principal, get_query, load_trace, run_hybrid_lab
from iw_rag.learn.lab_corpus import CHUNKS, PRINCIPALS, QUERIES
from iw_rag.retrieval.fusion import RankedHit, RRFConfig, reciprocal_rank_fusion

CHUNK_IDS = {c.chunk_id for c in CHUNKS}


def _lab(query_id: str, principal_id: str = "analyst", **kw: float | int):
    query, principal = get_query(query_id), get_principal(principal_id)
    assert query is not None and principal is not None
    return run_hybrid_lab(query, principal, **kw)  # type: ignore[arg-type]


def test_trace_covers_every_query_and_principal() -> None:
    trace = load_trace()
    assert {q["id"] for q in trace["queries"]} == {q.query_id for q in QUERIES}
    for q in trace["queries"]:
        assert set(q["runs"]) == {p.principal_id for p in PRINCIPALS}


@pytest.mark.parametrize("principal", PRINCIPALS, ids=lambda p: p.principal_id)
def test_recorded_visibility_matches_the_access_tags(principal) -> None:  # type: ignore[no-untyped-def]
    # An independent recomputation of the pre-filter, not a read-back of the recording.
    expected = {c.chunk_id for c in CHUNKS if set(c.acl_tags) & set(principal.groups)}
    for q in load_trace()["queries"]:
        run = q["runs"][principal.principal_id]
        assert set(run["visible"]) == expected
        seen = {h["id"] for key in ("dense", "sparse", "fused", "reranked") for h in run[key]}
        assert seen <= expected <= CHUNK_IDS


def test_default_replay_reproduces_the_recorded_production_fusion() -> None:
    for q in load_trace()["queries"]:
        run = q["runs"]["analyst"]
        arms = [[RankedHit(h["id"], h["rank"]) for h in run[k]] for k in ("dense", "sparse")]
        production = reciprocal_rank_fusion(arms, RRFConfig(k=60))
        assert [f.chunk_id for f in production] == [f["id"] for f in run["fused"]]
        replay = _lab(q["id"])
        assert [h.chunk_id for h in replay.fused] == [f["id"] for f in run["fused"]][
            : len(replay.fused)
        ]


def test_hr_content_never_reaches_the_finance_analyst() -> None:
    hr_ids = {c.chunk_id for c in CHUNKS if c.acl_tags == ("hr",)}
    hr_texts = [c.text for c in CHUNKS if c.acl_tags == ("hr",)]
    for q in QUERIES:
        payload = json.dumps(asdict(_lab(q.query_id)))
        assert not any(i in payload for i in (f'"{x}"' for x in hr_ids)), q.query_id
        assert not any(t in payload for t in hr_texts), q.query_id


def test_blocked_answer_is_explained_as_the_access_filter() -> None:
    result = _lab("sick", "analyst")
    assert result.gold["visible"] is False
    assert "access filter" in result.gold["notes"][0]
    assert result.hidden_count == 4
    assert result.gold["document"] is None
    assert result.gold["chunk_id"] is None


def test_blocked_answer_points_at_the_near_zero_reranker_scores() -> None:
    result = _lab("sick", "analyst")
    assert result.reranked[0].score < 0.05
    assert any("close to zero" in n for n in result.gold["notes"])
    # When the answer IS visible, that refusal hint must not appear.
    assert not any("close to zero" in n for n in _lab("sick", "hr").gold["notes"])


def test_same_question_is_answerable_by_the_group_that_may_read_it() -> None:
    result = _lab("sick", "hr")
    assert result.gold["visible"] is True
    assert result.gold["ranks"]["reranked"] == 1


def test_dense_only_fuses_to_the_dense_order_and_bm25_only_to_the_bm25_order() -> None:
    dense_only = _lab("invoice-term", w_sparse=0)
    assert [h.chunk_id for h in dense_only.fused] == [h.chunk_id for h in dense_only.dense]
    assert dense_only.sparse == ()
    bm25_only = _lab("invoice-term", w_dense=0)
    assert [h.chunk_id for h in bm25_only.fused] == [h.chunk_id for h in bm25_only.sparse]
    assert bm25_only.dense == ()


def test_fusion_can_demote_the_answer_and_the_reranker_can_restore_it() -> None:
    gold = _lab("paraphrase").gold
    assert gold["ranks"]["dense"] == 1
    assert gold["ranks"]["sparse"] is None
    assert gold["ranks"]["fused"] == 2
    assert gold["ranks"]["reranked"] == 1
    assert any("below its best single search" in n for n in gold["notes"])
    assert any("moved it up from #2 to #1" in n for n in gold["notes"])


def test_reranked_list_is_sorted_by_relevance_descending() -> None:
    for q in QUERIES:
        scores = [h.score for h in _lab(q.query_id, "hr").reranked]
        assert scores == sorted(scores, reverse=True)


def test_switching_both_arms_off_leaves_nothing_to_rank() -> None:
    empty = _lab("invoice-id", w_dense=0, w_sparse=0)
    assert empty.fused == () and empty.reranked == ()
    assert "never reaches the fused list" in empty.gold["notes"][-1]


def test_weights_and_k_are_clamped() -> None:
    result = _lab("invoice-id", w_dense=99, w_sparse=-5, rrf_k=10**6)
    assert result.weights == {"dense": 2.0, "sparse": 0.0}
    assert result.rrf_k == 200


def test_bm25_highlights_only_words_the_chunk_really_contains() -> None:
    first = _lab("error-code").sparse[0]
    assert first.chunk_id == "p01"
    assert "e4421" in first.matched_terms
    for hit in _lab("error-code").sparse:
        assert set(hit.matched_terms) <= {w.lower() for w in hit.text.replace(".", " ").split()}
