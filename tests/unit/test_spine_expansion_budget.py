"""Unit tests for parent-expansion context budget in ``retrieval.spine``.

The retrieval spine's small-to-big parent expansion replaces each leaf's
``content`` with its parent's full ``content``. For uploaded PDFs without
heading structure, the parent is the entire document (~29k tokens) which
overflows the generator's 16,384-token window (see WI I-00131).

The fix introduces a typed context budget
(``RetrievalSettings.parent_expansion_max_chars``) and gates parent
substitution on it.  These tests assert the runtime invariant: **after
``_expand_parents`` returns, no candidate's content exceeds the budget**.

The RED test below fails against the pre-fix code because the pre-fix
implementation always substitutes the parent's full content regardless
of size; the GREEN phase (matching S01 requirements 2-3) makes the
budget gate a contract.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest

from iw_rag.domain.models import RetrievedChunk
from iw_rag.retrieval.spine import RetrievalConfig, RetrievalSpine

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeRepo:
    """Minimal ChunkRepository fake exposing only ``get_retrievable``.

    The real repository joins chunk → document; ``get_retrievable`` returns
    dicts with at least ``id``, ``content``, ``source_uri``, ``metadata``,
    and ``parent_id``.  We honour that contract here so ``_expand_parents``
    can index parent rows by their id.
    """

    def __init__(self, parents: dict[str, dict[str, Any]]) -> None:
        self._parents = parents

    async def get_retrievable(self, chunk_ids: list[uuid.UUID]) -> list[dict[str, Any]]:
        return [self._parents[str(cid)] for cid in chunk_ids if str(cid) in self._parents]


def _make_spine(parent_expansion_max_chars: int) -> RetrievalSpine:
    """Build a RetrievalSpine whose collaborators are unreachable.

    ``_expand_parents`` only needs the parent repo (passed as argument),
    so the constructor collaborators are placeholders; we override them
    with objects that raise on access, then call ``_expand_parents``
    directly with our fake repo.

    Args:
        parent_expansion_max_chars: The budget to thread into
            ``RetrievalConfig``.

    Returns:
        A RetrievalSpine configured with the given budget.
    """
    spine = RetrievalSpine(  # type: ignore[arg-type]
        db=object(),  # never touched by _expand_parents
        encoder=object(),  # ditto
        dense=object(),  # ditto
        sparse=object(),  # ditto
        reranker=object(),  # ditto
        cfg=RetrievalConfig(parent_expansion_max_chars=parent_expansion_max_chars),
    )
    return spine


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_expand_parents_respects_budget_for_oversized_parent() -> None:
    """An oversized parent must NOT be substituted — the leaf content is retained.

    RED: pre-fix code replaces leaf.content with parent["content"] verbatim,
    so expanded.content has length == 50_000, which is > budget (4_000).

    GREEN: the budget gate is added in S01; the leaf is kept (matched text
    is always present) and ``leaf_content`` is also set to the leaf content
    (preserves the existing field's contract for downstream citation).
    """
    budget = 4_000
    parent_id = "11111111-1111-1111-1111-111111111111"
    leaf = RetrievedChunk(
        chunk_id="l1",
        document_id="doc-l1",
        content="leaf text that the user matched against",
        score=1.0,
        source_uri="upload://x",
        metadata={"parent_id": parent_id},
    )
    fake_repo = _FakeRepo(
        parents={
            parent_id: {
                "id": uuid.UUID(parent_id),
                "content": "X" * 50_000,
                "source_uri": "upload://x",
                "metadata": {},
                "parent_id": None,
            }
        }
    )
    spine = _make_spine(parent_expansion_max_chars=budget)

    [expanded] = await spine._expand_parents(fake_repo, [leaf])

    assert len(expanded.content) <= budget, (
        f"Oversized parent must NOT be substituted; expected len(expanded.content) "
        f"<= {budget}, got {len(expanded.content)}"
    )
    assert expanded.content == "leaf text that the user matched against", (
        f"Oversized parent → keep leaf content; got {expanded.content!r}"
    )
    assert expanded.leaf_content == "leaf text that the user matched against", (
        f"leaf_content field must be set to leaf text; got {expanded.leaf_content!r}"
    )


@pytest.mark.asyncio
async def test_expand_parents_substitutes_small_parent() -> None:
    """A small (within-budget) parent must still be substituted (AC3 — no regression).

    This is the common case for the bench corpus and the upload path once
    the S03 ingestion cap lands; the substitution behaviour must remain
    unchanged for small parents.
    """
    budget = 4_000
    parent_id = "22222222-2222-2222-2222-222222222222"
    parent_content = "parent text that fits well within the budget"
    leaf = RetrievedChunk(
        chunk_id="l2",
        document_id="doc-l2",
        content="leaf text",
        score=0.9,
        source_uri="upload://y",
        metadata={"parent_id": parent_id},
    )
    fake_repo = _FakeRepo(
        parents={
            parent_id: {
                "id": uuid.UUID(parent_id),
                "content": parent_content,
                "source_uri": "upload://y",
                "metadata": {},
                "parent_id": None,
            }
        }
    )
    spine = _make_spine(parent_expansion_max_chars=budget)

    [expanded] = await spine._expand_parents(fake_repo, [leaf])

    assert expanded.content == parent_content, (
        f"Small parent (within budget) must be substituted; got {expanded.content!r}"
    )
    assert expanded.leaf_content == "leaf text", (
        f"leaf_content must preserve the original leaf text; got {expanded.leaf_content!r}"
    )


@pytest.mark.asyncio
async def test_expand_parents_keeps_leaf_when_parent_missing() -> None:
    """When the parent row is missing, fall back to the leaf unchanged.

    Defensive: orphan leaves (e.g. parent deleted / not yet visible) must
    not blow up the spine; just keep the leaf as-is.
    """
    leaf = RetrievedChunk(
        chunk_id="l3",
        document_id="doc-l3",
        content="orphan leaf",
        score=0.8,
        source_uri="upload://z",
        metadata={"parent_id": "99999999-9999-9999-9999-999999999999"},
    )
    fake_repo = _FakeRepo(parents={})  # no parent rows
    spine = _make_spine(parent_expansion_max_chars=4_000)

    [expanded] = await spine._expand_parents(fake_repo, [leaf])

    assert expanded.content == "orphan leaf"
    assert expanded.leaf_content is None, (
        "leaf_content must be None when no parent expansion was attempted; "
        f"got {expanded.leaf_content!r}"
    )


@pytest.mark.asyncio
async def test_expand_parents_keeps_leaf_when_no_parent_id_metadata() -> None:
    """A leaf with no parent_id in metadata is returned unchanged."""
    leaf = RetrievedChunk(
        chunk_id="l4",
        document_id="doc-l4",
        content="bare leaf",
        score=0.7,
        source_uri="upload://w",
        metadata={},  # no parent_id
    )
    fake_repo = _FakeRepo(parents={})
    spine = _make_spine(parent_expansion_max_chars=4_000)

    [expanded] = await spine._expand_parents(fake_repo, [leaf])

    assert expanded.content == "bare leaf"
    assert expanded.leaf_content is None


@pytest.mark.asyncio
async def test_expand_parents_mixed_small_and_oversized_parents() -> None:
    """Per-leaf gating: small parents expand, oversized parents do not.

    Real-world rerank output often mixes large and small parents (e.g. the
    same query matches a leaf in a 29k-token whole-doc parent and a leaf
    in a small section). Each leaf must be gated individually.
    """
    budget = 4_000
    small_parent_id = "33333333-3333-3333-3333-333333333333"
    big_parent_id = "44444444-4444-4444-4444-444444444444"

    leaves = [
        RetrievedChunk(
            chunk_id="ls",
            document_id="doc-ls",
            content="small leaf text",
            score=0.95,
            source_uri="upload://s",
            metadata={"parent_id": small_parent_id},
        ),
        RetrievedChunk(
            chunk_id="lb",
            document_id="doc-lb",
            content="big leaf text",
            score=0.90,
            source_uri="upload://b",
            metadata={"parent_id": big_parent_id},
        ),
    ]
    fake_repo = _FakeRepo(
        parents={
            small_parent_id: {
                "id": uuid.UUID(small_parent_id),
                "content": "small parent content",
                "source_uri": "upload://s",
                "metadata": {},
                "parent_id": None,
            },
            big_parent_id: {
                "id": uuid.UUID(big_parent_id),
                "content": "Y" * 50_000,
                "source_uri": "upload://b",
                "metadata": {},
                "parent_id": None,
            },
        }
    )
    spine = _make_spine(parent_expansion_max_chars=budget)

    expanded = await spine._expand_parents(fake_repo, leaves)

    assert len(expanded) == 2
    # Small parent → expanded
    assert expanded[0].content == "small parent content"
    assert expanded[0].leaf_content == "small leaf text"
    # Big parent → kept as leaf
    assert expanded[1].content == "big leaf text"
    assert expanded[1].leaf_content == "big leaf text"
    assert len(expanded[1].content) <= budget
