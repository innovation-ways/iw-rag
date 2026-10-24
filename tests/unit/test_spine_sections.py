"""CR-00103: section rows in the spine: which are candidates, and how repeats are merged."""

from __future__ import annotations

import pytest

from iw_rag.domain.models import RetrievedChunk
from iw_rag.retrieval.spine import keep_candidate, merge_repeated_sections

SECTIONS = frozenset({"S1", "S2"})
BUDGET = 100


def _keep(chunk_id: str, content: str, **kw: object) -> str | None:
    args: dict[str, object] = {
        "expansion_mode": "parent",
        "section_hits": True,
        "budget_chars": BUDGET,
    }
    args.update(kw)
    return keep_candidate(chunk_id, content, SECTIONS, **args)  # type: ignore[arg-type]


def test_a_leaf_is_always_a_candidate_whatever_the_settings() -> None:
    for mode in ("leaf", "parent", "window"):
        for hits in (True, False):
            assert _keep("L1", "x" * 10_000, expansion_mode=mode, section_hits=hits) is None


def test_a_section_that_fits_the_budget_is_a_candidate_in_parent_mode() -> None:
    assert _keep("S1", "x" * BUDGET) is None  # exactly at the budget still fits
    assert _keep("S1", "x" * (BUDGET + 1)) == "oversize"


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"expansion_mode": "leaf"}, "leaf_only"),
        ({"section_hits": False}, "section_hits_off"),
        ({"section_hits": False, "expansion_mode": "leaf"}, "section_hits_off"),
    ],
)
def test_sections_are_set_aside_when_leaves_only_or_sections_are_switched_off(
    kwargs: dict[str, object], reason: str
) -> None:
    assert _keep("S2", "short", **kwargs) == reason


def _chunk(chunk_id: str, content: str, *, parent: str | None = None, leaf: str | None = None):  # type: ignore[no-untyped-def]
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id="d",
        content=content,
        score=1.0,
        source_uri="x",
        metadata={"parent_id": parent} if parent else {},
        leaf_content=leaf,
    )


def test_leaves_expanded_to_the_same_section_are_merged_into_the_best_ranked() -> None:
    text = "the whole section"
    ranked = [
        _chunk("L2", text, parent="S1", leaf="second leaf"),
        _chunk("L1", text, parent="S1", leaf="first leaf"),
        _chunk("L3", text, parent="S1", leaf="third leaf"),
    ]
    out, merged = merge_repeated_sections(ranked, SECTIONS, top_k=8)
    assert [c.chunk_id for c in out] == ["L2"] and merged == 2
    assert out[0].content == text and out[0].leaf_content == "second leaf"
    assert out[0].metadata["merged_chunk_ids"] == ["L1", "L3"]


def test_a_section_row_that_ranks_first_adopts_the_leaf_text_for_precise_citations() -> None:
    text = "the whole section"
    out, merged = merge_repeated_sections(
        [_chunk("S1", text), _chunk("L1", text, parent="S1", leaf="the matched leaf")], SECTIONS, 8
    )
    assert [c.chunk_id for c in out] == ["S1"] and merged == 1
    assert out[0].leaf_content == "the matched leaf" and out[0].metadata["merged_chunk_ids"] == [
        "L1"
    ]


def test_merging_happens_before_the_cut_so_freed_slots_go_to_the_next_chunks() -> None:
    text = "section text"
    ranked = [
        _chunk("L1", text, parent="S1", leaf="a"),
        _chunk("L2", text, parent="S1", leaf="b"),
        _chunk("L3", text, parent="S1", leaf="c"),
        _chunk("X1", "another passage", parent="S2"),
        _chunk("X2", "a third passage"),
    ]
    out, merged = merge_repeated_sections(ranked, SECTIONS, top_k=3)
    assert [c.chunk_id for c in out] == ["L1", "X1", "X2"] and merged == 2


def test_chunks_with_different_text_or_different_sections_are_never_merged() -> None:
    ranked = [
        _chunk("L1", "leaf one", parent="S1"),  # oversize section: the leaf stays itself
        _chunk("L2", "leaf two", parent="S1"),
        _chunk("L3", "same text", parent="S1"),
        _chunk("L4", "same text", parent="S2"),  # equal text, other section
        _chunk("N1", "same text"),  # no section at all
    ]
    out, merged = merge_repeated_sections(ranked, SECTIONS, top_k=10)
    assert [c.chunk_id for c in out] == ["L1", "L2", "L3", "L4", "N1"] and merged == 0


def test_top_k_zero_and_empty_input_are_handled() -> None:
    assert merge_repeated_sections([], SECTIONS, 5) == ([], 0)
    assert merge_repeated_sections([_chunk("L1", "t", parent="S1")], SECTIONS, 0) == ([], 0)
