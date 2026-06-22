"""Reciprocal Rank Fusion (RRF) — the pure, unit-tested fusion shim.

Fuses ranked arms by **rank**, not raw score, so it is robust to score-scale
incompatibility between dense (cosine) and sparse (BM25) arms. ``k`` is set
EXPLICITLY to 60 (ADR-003) — never a library default (the Weaviate RRF→RSF
default-flip is the cautionary tale). This Python fold is the correctness
reference; a single-CTE SQL realisation is an equivalent optimisation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class RankedHit:
    """One arm's ranked hit.

    Attributes:
        chunk_id: The chunk id (string form of the uuid).
        rank: 1-based rank within the arm (1 = best).
    """

    chunk_id: str
    rank: int


@dataclass(frozen=True, slots=True)
class FusedHit:
    """A fused hit after RRF.

    Attributes:
        chunk_id: The chunk id.
        score: The summed RRF score (higher = better).
    """

    chunk_id: str
    score: float


@dataclass(frozen=True, slots=True)
class RRFConfig:
    """RRF configuration.

    Attributes:
        k: The RRF constant (explicit 60; never a library default).
        weights: Per-arm weights ``(dense, sparse)``.
    """

    k: int = 60
    weights: tuple[float, ...] = field(default_factory=lambda: (1.0, 1.0))


def reciprocal_rank_fusion(arms: Sequence[Sequence[RankedHit]], cfg: RRFConfig) -> list[FusedHit]:
    """Fuse ranked arms by ``Σ_arm w_arm / (k + rank)``.

    Args:
        arms: Per-arm ranked hit lists (e.g. ``[dense, sparse]``).
        cfg: RRF config (k + per-arm weights).

    Returns:
        Fused hits sorted by score desc, ties broken by ``chunk_id`` for
        determinism.
    """
    scores: dict[str, float] = {}
    for arm_idx, arm in enumerate(arms):
        weight = cfg.weights[arm_idx] if arm_idx < len(cfg.weights) else 1.0
        for hit in arm:
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + weight / (cfg.k + hit.rank)
    fused = [FusedHit(chunk_id=cid, score=score) for cid, score in scores.items()]
    fused.sort(key=lambda h: (-h.score, h.chunk_id))
    return fused
