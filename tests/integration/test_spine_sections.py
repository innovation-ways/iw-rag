"""CR-00103 against a real Postgres: sections as candidates, the budget, and merged repeats."""

from __future__ import annotations

import math
import uuid
from collections.abc import Sequence

import pytest

from iw_rag.domain.acl import AccessContext
from iw_rag.retrieval.dense import DenseRetriever
from iw_rag.retrieval.rerank import RerankCandidate, ScoredChunk
from iw_rag.retrieval.sparse import SparseRetriever
from iw_rag.retrieval.spine import RetrievalConfig, RetrievalSpine
from iw_rag.settings import DatabaseSettings
from iw_rag.storage.db import Database
from iw_rag.storage.repositories import ChunkRepository, DocumentRepository, NewChunk

DIMS = 1024
CTX = AccessContext(principal_id="u", principal_groups=("public",))


def _vec(cos: float) -> list[float]:
    """A unit vector whose cosine with the query vector (axis 0) is ``cos``."""
    v = [0.0] * DIMS
    v[0], v[1] = cos, math.sqrt(1 - cos * cos)
    return v


class AxisEncoder:
    async def encode(self, query: str) -> list[float]:  # noqa: ARG002
        return _vec(1.0)


class InOrderReranker:
    """Keeps the fused order, so the test controls ranking through the vectors alone."""

    async def rerank(
        self, query: str, candidates: Sequence[RerankCandidate], top_n: int
    ) -> list[ScoredChunk]:  # noqa: ARG002
        return [
            ScoredChunk(chunk_id=c.chunk_id, score=1.0 / (i + 1))
            for i, c in enumerate(candidates[:top_n])
        ]

    async def health(self) -> bool:
        return True


SMALL_LEAVES = ["leaf one of the small section", "leaf two of the small section", "leaf three"]
SMALL_SECTION = "\n".join(SMALL_LEAVES)
BIG_LEAVES = ["big section leaf alpha", "big section leaf beta"]
BIG_SECTION = "x" * 5000  # over the 4,000-character budget


async def _seed(dsn: str) -> dict[str, uuid.UUID]:
    ids: dict[str, uuid.UUID] = {}
    db = Database(DatabaseSettings(dsn=dsn))
    try:
        async with db.session() as session:
            docs, chunks = DocumentRepository(session), ChunkRepository(session)
            doc = await docs.upsert_document(source_uri="corpus://public/a.md", checksum="a")
            other = await docs.upsert_document(source_uri="corpus://public/b.md", checksum="b")

            async def add(
                name: str, doc_id: uuid.UUID, text: str, cos: float, parent: str | None
            ) -> None:
                (ids[name],) = await chunks.bulk_upsert_chunks(
                    [
                        NewChunk(
                            document_id=doc_id,
                            ordinal=len(ids),
                            content=text,
                            dense_embedding=_vec(cos),
                            parent_id=ids[parent] if parent else None,
                            acl_tags=["public"],
                        )
                    ]
                )

            # Nearest first: small section, its leaves, the big section, its leaves, a stray leaf.
            await add("small", doc, SMALL_SECTION, 0.99, None)
            for i, text in enumerate(SMALL_LEAVES):
                await add(f"s{i}", doc, text, 0.98 - i * 0.01, "small")
            await add("big", other, BIG_SECTION, 0.90, None)
            for i, text in enumerate(BIG_LEAVES):
                await add(f"b{i}", other, text, 0.89 - i * 0.01, "big")
            await add("stray", other, "an unrelated stray passage", 0.50, None)
            await session.commit()
    finally:
        await db.dispose()
    return ids


async def _retrieve(dsn: str, **cfg: object):  # type: ignore[no-untyped-def]
    params: dict[str, object] = {
        "candidate_k": 50,
        "fused_k": 50,
        "top_k": 6,
        "expansion_mode": "parent",
    }
    params.update(cfg)
    spine = RetrievalSpine(
        db=Database(DatabaseSettings(dsn=dsn)),
        encoder=AxisEncoder(),
        dense=DenseRetriever(),
        sparse=SparseRetriever(),
        reranker=InOrderReranker(),
        cfg=RetrievalConfig(**params),  # type: ignore[arg-type]
    )
    return await spine.retrieve("zzzunmatched", CTX)


@pytest.mark.integration
async def test_a_section_and_its_leaves_reach_the_generator_once_and_top_k_is_refilled(
    clean_pg: str,
) -> None:
    ids = await _seed(clean_pg)
    result = await _retrieve(clean_pg, top_k=3)
    texts = [c.content for c in result.chunks]
    assert texts.count(SMALL_SECTION) == 1  # not once per leaf, and not again for the section row
    merged = next(c for c in result.chunks if c.content == SMALL_SECTION)
    assert merged.chunk_id == str(ids["small"])  # the section row ranked first and kept its slot
    assert merged.leaf_content == SMALL_LEAVES[0]  # ...and adopted the best leaf for citations
    assert merged.metadata["merged_chunk_ids"] == [str(ids["s0"]), str(ids["s1"]), str(ids["s2"])]
    assert result.diagnostics["repeats_merged"] == 3
    # Before the merge-then-cut, top_k = 3 returned the same text three times. Now the freed slots
    # go to the next distinct chunks: the two leaves of the (oversized) big section.
    assert {c.chunk_id for c in result.chunks} == {str(ids[k]) for k in ("small", "b0", "b1")}


@pytest.mark.integration
async def test_an_oversized_section_is_not_sent_whole_and_its_leaves_stay_unexpanded(
    clean_pg: str,
) -> None:
    ids = await _seed(clean_pg)
    result = await _retrieve(clean_pg)
    returned = {c.chunk_id: c for c in result.chunks}
    assert str(ids["big"]) not in returned
    assert result.diagnostics["sections_set_aside"] == {"oversize": 1}
    for i, text in enumerate(BIG_LEAVES):
        leaf = returned[str(ids[f"b{i}"])]
        assert leaf.content == text  # the section is too big, so the leaf is what is sent
        assert leaf.leaf_content == text


@pytest.mark.integration
async def test_leaf_mode_returns_leaves_only(clean_pg: str) -> None:
    ids = await _seed(clean_pg)
    result = await _retrieve(clean_pg, expansion_mode="leaf")
    returned = {c.chunk_id for c in result.chunks}
    assert returned.isdisjoint({str(ids["small"]), str(ids["big"])})
    assert {str(ids["s0"]), str(ids["s1"]), str(ids["s2"])} <= returned
    assert result.diagnostics["sections_set_aside"] == {"leaf_only": 2}
    assert [c.content for c in result.chunks if c.chunk_id == str(ids["s0"])] == [SMALL_LEAVES[0]]


@pytest.mark.integration
async def test_switching_section_hits_off_leaves_only_leaves_but_leaves_still_expand(
    clean_pg: str,
) -> None:
    ids = await _seed(clean_pg)
    result = await _retrieve(clean_pg, section_hits=False)
    returned = {c.chunk_id for c in result.chunks}
    assert returned.isdisjoint({str(ids["small"]), str(ids["big"])})
    assert result.diagnostics["sections_set_aside"] == {"section_hits_off": 2}
    # The leaves of the small section are still expanded to it, and merged into one chunk.
    assert [c.content for c in result.chunks].count(SMALL_SECTION) == 1


@pytest.mark.integration
async def test_section_ids_are_the_rows_that_have_children(clean_pg: str) -> None:
    ids = await _seed(clean_pg)
    db = Database(DatabaseSettings(dsn=clean_pg))
    try:
        async with db.session() as session:
            found = await ChunkRepository(session).get_section_ids(list(ids.values()))
            none = await ChunkRepository(session).get_section_ids([])
    finally:
        await db.dispose()
    assert found == {str(ids["small"]), str(ids["big"])}
    assert none == frozenset()
