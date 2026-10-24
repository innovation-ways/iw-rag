"""Hybrid-search lab: replay a recorded real run, re-fusing with the production RRF.

``traces/hybrid_search.json`` holds what the production dense arm, BM25 arm and
cross-encoder reranker returned for each lab query and principal (see
``scripts/learn_record_hybrid_trace.py``).  The two arms and the reranker are
independent of the fusion settings, so the lab lets the reader change the RRF
constant and the per-arm weights and re-runs the *real*
:func:`~iw_rag.retrieval.fusion.reciprocal_rank_fusion` over the recorded arm ranks.
The reranker scores each chunk on its own, so re-ranking the new shortlist is a sort
over the recorded scores.  Nothing here needs a GPU or a database.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from iw_rag.learn.lab_corpus import CHUNKS, PRINCIPALS, QUERIES, LabChunk, LabPrincipal, LabQuery
from iw_rag.retrieval.fusion import RankedHit, RRFConfig, reciprocal_rank_fusion

TRACE_PATH = Path(__file__).parent / "traces" / "hybrid_search.json"
SHOWN = 6
MAX_WEIGHT = 2.0
MAX_RRF_K = 200
NOTHING_RELEVANT_BELOW = 0.05
_TERM = re.compile(r"[A-Za-z0-9]+")
_CHUNKS: dict[str, LabChunk] = {c.chunk_id: c for c in CHUNKS}


@dataclass(frozen=True, slots=True)
class Hit:
    """One chunk in one stage of the replay."""

    chunk_id: str
    document: str
    text: str
    rank: int
    score_label: str
    score: float
    matched_terms: tuple[str, ...] = ()
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class HybridLabResult:
    """What the hybrid-search lab shows for one question, user and fusion setting."""

    query: dict[str, str]
    principal: dict[str, Any]
    meta: dict[str, Any]
    visible_count: int
    hidden_count: int
    weights: dict[str, float]
    rrf_k: int
    totals: dict[str, int]
    dense: tuple[Hit, ...]
    sparse: tuple[Hit, ...]
    fused: tuple[Hit, ...]
    reranked: tuple[Hit, ...]
    gold: dict[str, Any]


@lru_cache(maxsize=1)
def load_trace() -> dict[str, Any]:
    """Load the recorded trace (cached); raises ``FileNotFoundError`` if not recorded."""
    data: dict[str, Any] = json.loads(TRACE_PATH.read_text(encoding="utf-8"))
    return data


def get_query(query_id: str) -> LabQuery | None:
    """Return the lab query for ``query_id``, or ``None``."""
    return next((q for q in QUERIES if q.query_id == query_id), None)


def get_principal(principal_id: str) -> LabPrincipal | None:
    """Return the lab principal for ``principal_id``, or ``None``."""
    return next((p for p in PRINCIPALS if p.principal_id == principal_id), None)


def _terms(text: str) -> set[str]:
    return {t.lower() for t in _TERM.findall(text)}


def _hit(chunk_id: str, rank: int, label: str, score: float, query: str, **detail: Any) -> Hit:
    chunk = _CHUNKS[chunk_id]
    return Hit(
        chunk_id=chunk_id,
        document=chunk.document,
        text=chunk.text,
        rank=rank,
        score_label=label,
        score=score,
        matched_terms=tuple(sorted(_terms(query) & _terms(chunk.text))),
        detail=detail,
    )


def _rank_of(hits: tuple[Hit, ...], chunk_id: str) -> int | None:
    return next((h.rank for h in hits if h.chunk_id == chunk_id), None)


def _gold_summary(
    gold_id: str,
    visible: bool,
    arms: dict[str, tuple[Hit, ...]],
    weights: tuple[float, float],
) -> dict[str, Any]:
    chunk = _CHUNKS[gold_id]
    if not visible:
        return {
            "chunk_id": None,  # even the id of a chunk the user may not read stays hidden
            "visible": False,
            "document": None,
            "ranks": {},
            "notes": [
                "The chunk that answers this question belongs to a group this user is not in. "
                "The access filter removed it before any search ran, so no stage can find it, "
                "and the user never learns it exists."
            ],
        }
    ranks = {name: _rank_of(hits, gold_id) for name, hits in arms.items()}
    notes: list[str] = []
    dense, sparse, fused, reranked = (ranks[k] for k in ("dense", "sparse", "fused", "reranked"))
    if weights[1] > 0 and sparse is None:
        notes.append("BM25 did not find the answer: the question shares no useful words with it.")
    if weights[0] > 0 and dense is None:
        notes.append("The dense search did not return the answer.")
    best_arm = min([r for r in (dense, sparse) if r is not None], default=None)
    if fused is not None and best_arm is not None and fused > best_arm:
        notes.append(
            f"Fusion placed it at #{fused}, below its best single search (#{best_arm}): "
            "the other search ranked a different chunk first and outweighed it."
        )
    if reranked is not None and fused is not None and reranked < fused:
        notes.append(f"The reranker moved it up from #{fused} to #{reranked}.")
    if reranked is not None and fused is not None and reranked > fused:
        notes.append(f"The reranker moved it down from #{fused} to #{reranked}.")
    if fused is None:
        notes.append("With these settings the answer never reaches the fused list.")
    return {
        "chunk_id": gold_id,
        "visible": True,
        "document": chunk.document,
        "ranks": ranks,
        "notes": notes,
    }


def run_hybrid_lab(
    query: LabQuery,
    principal: LabPrincipal,
    *,
    w_dense: float = 1.0,
    w_sparse: float = 1.0,
    rrf_k: int = 60,
) -> HybridLabResult:
    """Replay the recorded run for ``query`` as ``principal`` with the given fusion settings.

    Args:
        query: The lab question.
        principal: The signed-in user to replay as.
        w_dense: RRF weight of the dense arm (0 switches the arm off), clamped to 0..2.
        w_sparse: RRF weight of the BM25 arm (0 switches the arm off), clamped to 0..2.
        rrf_k: The RRF constant, clamped to 1..200.

    Returns:
        The top hits of each stage, the visibility numbers and a verdict on the answer chunk.
    """
    trace = load_trace()
    run = next(q for q in trace["queries"] if q["id"] == query.query_id)["runs"][
        principal.principal_id
    ]
    w_dense = max(0.0, min(MAX_WEIGHT, w_dense))
    w_sparse = max(0.0, min(MAX_WEIGHT, w_sparse))
    rrf_k = max(1, min(MAX_RRF_K, rrf_k))

    dense_rec = run["dense"]
    sparse_rec = run["sparse"]
    arms = [
        [RankedHit(h["id"], h["rank"]) for h in dense_rec] if w_dense > 0 else [],
        [RankedHit(h["id"], h["rank"]) for h in sparse_rec] if w_sparse > 0 else [],
    ]
    fused_raw = reciprocal_rank_fusion(arms, RRFConfig(k=rrf_k, weights=(w_dense, w_sparse)))
    dense_rank = {h["id"]: h["rank"] for h in dense_rec}
    sparse_rank = {h["id"]: h["rank"] for h in sparse_rec}
    rerank_score = {h["id"]: h["score"] for h in run["reranked"]}

    dense = tuple(
        _hit(h["id"], h["rank"], "distance", h["distance"], query.text) for h in dense_rec[:SHOWN]
    )
    sparse = tuple(
        _hit(h["id"], h["rank"], "BM25", h["bm25"], query.text) for h in sparse_rec[:SHOWN]
    )
    fused_all = tuple(
        _hit(
            f.chunk_id,
            i,
            "RRF",
            f.score,
            query.text,
            dense_rank=dense_rank.get(f.chunk_id) if w_dense > 0 else None,
            sparse_rank=sparse_rank.get(f.chunk_id) if w_sparse > 0 else None,
        )
        for i, f in enumerate(fused_raw, start=1)
    )
    reranked_all = tuple(
        _hit(cid, i, "rerank", rerank_score[cid], query.text)
        for i, cid in enumerate(
            sorted((f.chunk_id for f in fused_raw), key=lambda c: -rerank_score[c]), 1
        )
    )
    visible_ids = set(run["visible"])
    gold_visible = query.gold in visible_ids
    gold = _gold_summary(
        query.gold,
        gold_visible,
        {
            "dense": tuple(_hit(h["id"], h["rank"], "d", 0.0, "") for h in dense_rec)
            if w_dense > 0
            else (),
            "sparse": tuple(_hit(h["id"], h["rank"], "s", 0.0, "") for h in sparse_rec)
            if w_sparse > 0
            else (),
            "fused": fused_all,
            "reranked": reranked_all,
        },
        (w_dense, w_sparse),
    )
    if not gold_visible and reranked_all and reranked_all[0].score < NOTHING_RELEVANT_BELOW:
        gold["notes"].append(
            "Look at the reranker column: every candidate scores close to zero, because nothing "
            "visible to this user answers the question. That is the signal that lets the system "
            "refuse with 'not enough grounded evidence' instead of guessing."
        )
    return HybridLabResult(
        query={"id": query.query_id, "text": query.text, "lesson": query.lesson},
        principal={
            "id": principal.principal_id,
            "label": principal.label,
            "groups": list(principal.groups),
        },
        meta={
            "recorded": trace["recorded"],
            "embedder": trace["embedder"],
            "reranker": trace["reranker"],
            "embedding_note": trace["embedding_note"],
            "default_rrf_k": trace["rrf_k"],
        },
        visible_count=len(visible_ids),
        hidden_count=len(CHUNKS) - len(visible_ids),
        weights={"dense": w_dense, "sparse": w_sparse},
        rrf_k=rrf_k,
        totals={
            "dense": len(dense_rec) if w_dense > 0 else 0,
            "sparse": len(sparse_rec) if w_sparse > 0 else 0,
            "fused": len(fused_all),
        },
        dense=dense if w_dense > 0 else (),
        sparse=sparse if w_sparse > 0 else (),
        fused=fused_all[:SHOWN],
        reranked=reranked_all[:SHOWN],
        gold=gold,
    )


__all__ = [
    "SHOWN",
    "Hit",
    "HybridLabResult",
    "get_principal",
    "get_query",
    "load_trace",
    "run_hybrid_lab",
]
