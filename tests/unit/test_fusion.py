"""Unit tests for Reciprocal Rank Fusion (RRF, k=60 explicit)."""

from __future__ import annotations

import pytest

from iw_rag.retrieval.fusion import RankedHit, RRFConfig, reciprocal_rank_fusion


def test_both_arms_rank1_beats_single_arm_rank1() -> None:
    dense = [RankedHit("A", 1), RankedHit("B", 2)]
    sparse = [RankedHit("A", 1), RankedHit("C", 2)]
    fused = reciprocal_rank_fusion([dense, sparse], RRFConfig(k=60))
    assert fused[0].chunk_id == "A"  # rank-1 in both arms
    a = next(h.score for h in fused if h.chunk_id == "A")
    b = next(h.score for h in fused if h.chunk_id == "B")
    assert a == pytest.approx(1 / 61 + 1 / 61)
    assert b == pytest.approx(1 / 62)
    assert a > b


def test_exact_rrf_values_k60() -> None:
    fused = reciprocal_rank_fusion([[RankedHit("X", 1)], [RankedHit("X", 3)]], RRFConfig(k=60))
    assert fused[0].score == pytest.approx(1 / 61 + 1 / 63)


def test_weighted_rrf_biases_arm() -> None:
    dense = [RankedHit("D", 1)]
    sparse = [RankedHit("S", 1)]
    fused = reciprocal_rank_fusion([dense, sparse], RRFConfig(k=60, weights=(2.0, 1.0)))
    d = next(h.score for h in fused if h.chunk_id == "D")
    s = next(h.score for h in fused if h.chunk_id == "S")
    assert d == pytest.approx(2.0 / 61)
    assert s == pytest.approx(1.0 / 61)
    assert fused[0].chunk_id == "D"


def test_deterministic_tie_ordering_by_chunk_id() -> None:
    # Two docs with identical scores → tie broken by chunk_id ascending.
    fused = reciprocal_rank_fusion([[RankedHit("z", 1), RankedHit("a", 1)]], RRFConfig(k=60))
    assert [h.chunk_id for h in fused] == ["a", "z"]


def test_empty_arms_yield_empty() -> None:
    assert reciprocal_rank_fusion([[], []], RRFConfig(k=60)) == []
