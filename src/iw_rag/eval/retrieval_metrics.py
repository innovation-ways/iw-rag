"""Judge-free retrieval metrics — the contamination-proof backbone.

Pure / sync / CPU (no LLM, DB, or network). recall@k, nDCG@k, MRR with exact,
hand-verifiable arithmetic (binary relevance; ``log2(rank+1)`` discount). These
four functions are "the one number nobody can argue with" (lld-eval-observability
§1/§2/§7) — definitions are pinned here and locked by fixtures.

Note: ids are ``str`` (the canonical ``RetrievedChunk.chunk_id`` type); callers
that hold gold uuids str-ify them at the boundary. The LLD sketches ``UUID``;
``str`` is the project's chunk-id form (CONVENTIONS §1).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import log2

from iw_rag.domain.models import RetrievalResult


def recall_at_k(retrieved: Sequence[str], gold: Sequence[str], k: int) -> float:
    """Fraction of gold ids found in the top-k retrieved (0.0 if no gold)."""
    if not gold:
        return 0.0
    goldset = set(gold)
    hits = sum(1 for cid in retrieved[:k] if cid in goldset)
    return hits / len(goldset)


def ndcg_at_k(retrieved: Sequence[str], gold: Sequence[str], k: int) -> float:
    """Binary-relevance nDCG@k (0.0 if no gold).

    DCG uses the ``1/log2(rank+1)`` discount; IDCG places ``min(len(gold), k)``
    relevant items ideally.
    """
    if not gold:
        return 0.0
    goldset = set(gold)
    dcg = sum(1.0 / log2(i + 2) for i, cid in enumerate(retrieved[:k]) if cid in goldset)
    ideal_hits = min(len(goldset), k)
    idcg = sum(1.0 / log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg else 0.0


def mrr(retrieved: Sequence[str], gold: Sequence[str]) -> float:
    """Reciprocal rank of the first relevant hit (1-indexed); 0.0 if none."""
    goldset = set(gold)
    for i, cid in enumerate(retrieved):
        if cid in goldset:
            return 1.0 / (i + 1)
    return 0.0


@dataclass(frozen=True, slots=True)
class RetrievalScores:
    """Per-k retrieval scores for one query.

    Attributes:
        recall: ``{k: recall@k}``.
        ndcg: ``{k: nDCG@k}``.
        mrr: Mean reciprocal rank (single value).
        k_values: The k cutoffs scored.
    """

    recall: dict[int, float]
    ndcg: dict[int, float]
    mrr: float
    k_values: tuple[int, ...]


def score_retrieval(
    result: RetrievalResult, gold: Sequence[str], ks: tuple[int, ...] = (5, 10)
) -> RetrievalScores:
    """Score a RetrievalResult against gold ids at each k.

    Trusts the spine contract that ``result.chunks`` is sorted best-first
    (lld-retrieval-spine §4) — does NOT re-sort.

    Args:
        result: The retrieval result to score.
        gold: Gold-relevant chunk ids (str).
        ks: The k cutoffs.

    Returns:
        A :class:`RetrievalScores`.
    """
    ranked: list[str] = []
    for chunk in result.chunks:
        # A chunk that absorbed repeats (CR-00103) delivered their text too, so it counts as each
        # id it covers, listed right after it: the same ids the pre-merge result would have held.
        ranked.append(chunk.chunk_id)
        ranked.extend(chunk.metadata.get("merged_chunk_ids", ()))
    return RetrievalScores(
        recall={k: recall_at_k(ranked, gold, k) for k in ks},
        ndcg={k: ndcg_at_k(ranked, gold, k) for k in ks},
        mrr=mrr(ranked, gold),
        k_values=ks,
    )
