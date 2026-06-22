"""Hypothesis property-based tests for ``iw_rag.retrieval.fusion.reciprocal_rank_fusion``.

Invariants verified:
- **Sorted order**: the returned list is non-increasing in score (ties broken
  by ``chunk_id`` ascending — deterministic).
- **Monotonicity**: a doc present in strictly more arms (or at a higher rank
  in the same arm) never scores lower than one present in fewer arms.
- **Single-arm single-doc formula**: ``score == weight / (k + rank)`` exactly.
- **Identity / no dupes**: output ``chunk_id`` set equals the union of input ids,
  with no duplicates in the output.
"""

from __future__ import annotations

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from iw_rag.retrieval.fusion import FusedHit, RankedHit, RRFConfig, reciprocal_rank_fusion

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_CHUNK_ID = st.text(
    alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters="-_"),
    min_size=1,
    max_size=20,
)

_K = st.integers(min_value=1, max_value=200)
_WEIGHT = st.floats(min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False)


def _ranked_arm(chunk_ids: list[str]) -> list[RankedHit]:
    """Build a ranked arm from a list of chunk ids (1-based ranks).

    Args:
        chunk_ids: Ordered chunk ids for the arm (rank assigned by position).

    Returns:
        A list of ``RankedHit`` objects with 1-based rank.
    """
    return [RankedHit(chunk_id=cid, rank=i + 1) for i, cid in enumerate(chunk_ids)]


@st.composite
def _arms_strategy(
    draw: st.DrawFn,
) -> tuple[list[list[RankedHit]], RRFConfig]:
    """Draw a random multi-arm RRF input together with its config.

    Args:
        draw: Hypothesis draw function.

    Returns:
        A ``(arms, cfg)`` pair ready for ``reciprocal_rank_fusion``.
    """
    n_arms = draw(st.integers(min_value=1, max_value=4))
    all_ids = draw(st.lists(_CHUNK_ID, min_size=1, max_size=15, unique=True))
    arms: list[list[RankedHit]] = []
    for _ in range(n_arms):
        # Each arm gets a random subset of the global pool, in random order
        subset = draw(
            st.lists(st.sampled_from(all_ids), min_size=0, max_size=len(all_ids), unique=True)
        )
        arms.append(_ranked_arm(subset))
    k = draw(_K)
    weights = tuple(draw(_WEIGHT) for _ in range(n_arms))
    cfg = RRFConfig(k=k, weights=weights)
    return arms, cfg


# ---------------------------------------------------------------------------
# Property 1 — Sorted (non-increasing) output
# ---------------------------------------------------------------------------


@given(input_=_arms_strategy())
def test_fused_scores_are_non_increasing(
    input_: tuple[list[list[RankedHit]], RRFConfig],
) -> None:
    """Fused output is sorted by score descending (ties by chunk_id ascending).

    The list must be monotonically non-increasing in score so callers can safely
    take a prefix as the top-k result.
    """
    arms, cfg = input_
    result = reciprocal_rank_fusion(arms, cfg)

    for i in range(len(result) - 1):
        hi = result[i]
        lo = result[i + 1]
        # Primary: score non-increasing
        assert hi.score >= lo.score, (
            f"score at rank {i} ({hi.score}) < score at rank {i + 1} ({lo.score})"
        )
        # Tie-break: equal scores → chunk_id ascending
        if hi.score == lo.score:
            assert hi.chunk_id <= lo.chunk_id, (
                f"tie at score {hi.score}: chunk_id {hi.chunk_id!r} > {lo.chunk_id!r} (not sorted)"
            )


# ---------------------------------------------------------------------------
# Property 2 — Monotonicity (more arms / higher rank → higher score)
# ---------------------------------------------------------------------------


@given(
    k=_K,
    weight=_WEIGHT,
    rank_a=st.integers(min_value=1, max_value=50),
    rank_b=st.integers(min_value=1, max_value=50),
)
def test_higher_rank_yields_higher_score_in_single_arm(
    k: int,
    weight: float,
    rank_a: int,
    rank_b: int,
) -> None:
    """A better (lower-numbered) rank always yields a higher RRF score.

    Within a single arm, ``rank_a < rank_b`` implies ``score_a > score_b``
    because ``weight/(k + rank_a) > weight/(k + rank_b)`` when rank_a < rank_b.
    """
    assume(rank_a != rank_b)
    cfg = RRFConfig(k=k, weights=(weight,))
    result_a = reciprocal_rank_fusion([[RankedHit("doc_a", rank_a)]], cfg)
    result_b = reciprocal_rank_fusion([[RankedHit("doc_b", rank_b)]], cfg)
    assert len(result_a) == 1
    assert len(result_b) == 1
    score_a = result_a[0].score
    score_b = result_b[0].score
    if rank_a < rank_b:
        assert score_a > score_b, (
            f"rank {rank_a} should score higher than rank {rank_b}: {score_a} <= {score_b}"
        )
    else:
        assert score_a < score_b, (
            f"rank {rank_a} should score lower than rank {rank_b}: {score_a} >= {score_b}"
        )


@given(k=_K, weight=_WEIGHT, rank=st.integers(min_value=1, max_value=50))
def test_two_arms_beat_one_arm_same_rank(
    k: int,
    weight: float,
    rank: int,
) -> None:
    """A doc appearing in two arms at the same rank scores strictly higher than one arm.

    ``2 * weight/(k+rank) > weight/(k+rank)`` — monotonicity over arm count.
    """
    cfg = RRFConfig(k=k, weights=(weight, weight))
    two_arms = reciprocal_rank_fusion([[RankedHit("doc", rank)], [RankedHit("doc", rank)]], cfg)
    one_arm = reciprocal_rank_fusion([[RankedHit("doc", rank)]], RRFConfig(k=k, weights=(weight,)))
    assert len(two_arms) == 1
    assert len(one_arm) == 1
    assert two_arms[0].score > one_arm[0].score, (
        f"two-arm score {two_arms[0].score} should exceed one-arm score {one_arm[0].score}"
    )


# ---------------------------------------------------------------------------
# Property 3 — Formula pin (single arm, single doc)
# ---------------------------------------------------------------------------


@given(
    chunk_id=_CHUNK_ID,
    k=_K,
    weight=_WEIGHT,
    rank=st.integers(min_value=1, max_value=100),
)
def test_single_arm_single_doc_formula(
    chunk_id: str,
    k: int,
    weight: float,
    rank: int,
) -> None:
    """Score for a single-arm, single-doc input equals ``weight / (k + rank)`` exactly.

    This pins the RRF formula and guards against accidental constant changes
    (e.g. Weaviate's k-default flip from 60 to something else).
    """
    cfg = RRFConfig(k=k, weights=(weight,))
    result = reciprocal_rank_fusion([[RankedHit(chunk_id, rank)]], cfg)
    assert len(result) == 1, f"expected 1 result, got {len(result)}"
    expected_score = weight / (k + rank)
    assert result[0].score == pytest.approx(expected_score), (
        f"score {result[0].score} != weight/(k+rank) = {expected_score}"
    )
    assert result[0].chunk_id == chunk_id, f"chunk_id {result[0].chunk_id!r} != {chunk_id!r}"


# ---------------------------------------------------------------------------
# Property 4 — Identity: output ids == union of input ids, no dupes
# ---------------------------------------------------------------------------


@given(input_=_arms_strategy())
def test_output_ids_are_exactly_input_union(
    input_: tuple[list[list[RankedHit]], RRFConfig],
) -> None:
    """Output ``chunk_id`` set equals the union of all arm input ids, no more, no fewer.

    No document may be silently dropped and no spurious document may appear that
    was not present in any arm.  Output must also contain no duplicates.
    """
    arms, cfg = input_
    result: list[FusedHit] = reciprocal_rank_fusion(arms, cfg)

    expected_ids = {hit.chunk_id for arm in arms for hit in arm}
    output_ids = [h.chunk_id for h in result]
    output_id_set = set(output_ids)

    assert output_id_set == expected_ids, (
        f"Output ids {output_id_set} != union of input ids {expected_ids}"
    )
    # No duplicates in the output list
    assert len(output_ids) == len(output_id_set), f"Duplicate chunk_ids in output: {output_ids}"
