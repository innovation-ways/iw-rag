"""The small-to-big rule the retrieval spine applies to each reranked leaf."""

from __future__ import annotations

from iw_rag.domain.models import RetrievedChunk
from iw_rag.retrieval.spine import expand_to_parent


def _leaf(content: str = "leaf text") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="l1",
        document_id="d1",
        content=content,
        score=0.9,
        source_uri="file:///x.md",
        metadata={"parent_id": "p1"},
    )


def test_a_parent_within_budget_replaces_the_leaf_and_remembers_it() -> None:
    out = expand_to_parent(_leaf(), "the whole section", 100)
    assert out.content == "the whole section"
    assert out.leaf_content == "leaf text"


def test_a_parent_exactly_at_the_budget_still_expands() -> None:
    parent = "x" * 100
    assert expand_to_parent(_leaf(), parent, 100).content == parent


def test_an_oversized_parent_keeps_the_leaf_but_records_it_as_the_matched_text() -> None:
    out = expand_to_parent(_leaf(), "x" * 101, 100)
    assert out.content == "leaf text"
    assert out.leaf_content == "leaf text"


def test_a_chunk_with_no_parent_is_returned_unchanged() -> None:
    chunk = _leaf()
    assert expand_to_parent(chunk, None, 100) is chunk
