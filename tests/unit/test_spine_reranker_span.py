"""Unit tests for the reranker span tagging in the retrieval spine.

AC2: a dedicated ``RERANKER``-kinded child span wraps the reranker call, with
its own duration distinct from the ``iw_rag.retrieval.retrieve`` /
``.encode`` / ``dense`` / ``sparse`` spans.  Verified via the offline
timeline (``observability.timeline``) which records every span regardless of
whether OTel is configured — the pipeline contract is "always emitted".

The tests use fully-faked collaborators (no DB, no encoder, no model server)
so they run hermetically.  ``ChunkRepository`` is monkey-patched in the
``retrieval.spine`` module namespace so the spine's real ``repo = ChunkRepository(session)``
call resolves to a fake.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from typing import Any

import pytest

from iw_rag.domain.acl import AccessContext
from iw_rag.observability import tracing
from iw_rag.observability.timeline import start_timeline
from iw_rag.retrieval import spine as spine_mod
from iw_rag.retrieval.fusion import RankedHit
from iw_rag.retrieval.rerank import RerankCandidate, ScoredChunk
from iw_rag.retrieval.spine import RetrievalSpine

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


_ROW = {
    "id": uuid.UUID("11111111-1111-1111-1111-111111111111"),
    "document_id": uuid.UUID("22222222-2222-2222-2222-222222222222"),
    "content": "fake content",
    "source_uri": "upload://x",
    "metadata": {},
    "parent_id": None,
}


class _FakeEncoder:
    async def encode(self, query: str) -> list[float]:
        return [0.0] * 4  # 4-dim stub; never inspected


class _FakeDense:
    async def retrieve(
        self, session: Any, query_vec: list[float], acl: Any, k: int
    ) -> list[RankedHit]:
        return [RankedHit(chunk_id=str(_ROW["id"]), rank=1)]


class _FakeSparse:
    async def retrieve(self, session: Any, query: str, acl: Any, k: int) -> list[RankedHit]:
        return []


class _CapturingReranker:
    """Records the call and returns canned scored chunks."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    async def rerank(
        self, query: str, candidates: Sequence[RerankCandidate], top_n: int
    ) -> list[ScoredChunk]:
        self.calls.append((query, top_n))
        return [ScoredChunk(chunk_id=c.chunk_id, score=0.9) for c in candidates[:top_n]]


class _FakeChunkRepository:
    """In-memory stand-in for the real ``ChunkRepository`` (avoids needing a DB)."""

    def __init__(self, session: Any) -> None:
        self._session = session

    async def get_retrievable(self, chunk_ids: list[uuid.UUID]) -> list[dict[str, Any]]:
        ids = {str(cid) for cid in chunk_ids}
        return [_ROW] if str(_ROW["id"]) in ids else []


class _FakeSession:
    """Minimal AsyncSession stand-in (the fake repo doesn't actually use it)."""

    pass


class _FakeDB:
    """Minimal ``Database`` stand-in exposing ``session()`` as an async ctx mgr."""

    def __init__(self) -> None:
        self._session = _FakeSession()

    @asynccontextmanager
    async def session(self) -> AsyncIterator[_FakeSession]:
        yield self._session


@pytest.fixture(autouse=True)
def _patch_chunk_repo(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the ``ChunkRepository`` the spine instantiates with a fake."""
    monkeypatch.setattr(spine_mod, "ChunkRepository", _FakeChunkRepository)


@pytest.fixture(autouse=True)
def _reset_tracing() -> None:
    tracing.reset_tracing_for_tests()
    yield
    tracing.reset_tracing_for_tests()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_spine_emits_reranker_span_with_openinference_kind() -> None:
    """A run through the spine produces an ``iw_rag.retrieval.rerank`` ``StepRecord``
    with ``openinference.span.kind == "RERANKER"`` (AC2)."""
    reranker = _CapturingReranker()
    spine = RetrievalSpine(  # type: ignore[arg-type]
        db=_FakeDB(),
        encoder=_FakeEncoder(),
        dense=_FakeDense(),
        sparse=_FakeSparse(),
        reranker=reranker,
    )

    with start_timeline() as timeline:
        result = await spine.retrieve(
            "what is the policy?", AccessContext(principal_id="u", principal_groups=("hr",))
        )

    # Retrieval completed successfully with the fake reranker.
    assert result.chunks, "reranker returned at least one chunk"
    assert reranker.calls, "reranker was actually invoked"

    rerank_records = [r for r in timeline.records if r.name == "iw_rag.retrieval.rerank"]
    assert len(rerank_records) == 1, (
        f"expected exactly one rerank span, got {[r.name for r in timeline.records]}"
    )
    assert rerank_records[0].attributes.get("openinference.span.kind") == "RERANKER", (
        f"rerank span must be tagged RERANKER; got {rerank_records[0].attributes!r}"
    )


@pytest.mark.asyncio
async def test_reranker_span_is_distinct_from_other_stage_spans() -> None:
    """The ``RERANKER`` span is its own record, not folded into the encode/dense/sparse
    spans (AC2: per-stage latency separability)."""
    spine = RetrievalSpine(  # type: ignore[arg-type]
        db=_FakeDB(),
        encoder=_FakeEncoder(),
        dense=_FakeDense(),
        sparse=_FakeSparse(),
        reranker=_CapturingReranker(),
    )

    with start_timeline() as timeline:
        await spine.retrieve("q?", AccessContext(principal_id="u", principal_groups=("hr",)))

    names = [r.name for r in timeline.records]
    # All the per-stage spans the spine is expected to record on a normal run.
    for required in (
        "iw_rag.retrieval.acl_filter",
        "iw_rag.retrieval.encode",
        "iw_rag.retrieval.dense",
        "iw_rag.retrieval.sparse",
        "iw_rag.retrieval.fusion",
        "iw_rag.retrieval.rerank",
    ):
        assert required in names, f"missing {required!r} in spans {names!r}"

    rerank_records = [r for r in timeline.records if r.name == "iw_rag.retrieval.rerank"]
    assert len(rerank_records) == 1


@pytest.mark.asyncio
async def test_reranker_failure_degrade_still_records_reranker_span() -> None:
    """Degraded-on-failure (the default policy) must still produce a ``RERANKER`` span —
    the failure happens INSIDE the span, so the span's own duration is what we want.

    The span's attributes continue to carry the kind marker; the failure
    surfaces via ``result.diagnostics['rerank'] = 'degraded'``.
    """

    class _FailingReranker:
        async def rerank(
            self, query: str, candidates: Sequence[RerankCandidate], top_n: int
        ) -> list[ScoredChunk]:
            from iw_rag.domain.errors import RerankError

            raise RerankError("boom")

    spine = RetrievalSpine(  # type: ignore[arg-type]
        db=_FakeDB(),
        encoder=_FakeEncoder(),
        dense=_FakeDense(),
        sparse=_FakeSparse(),
        reranker=_FailingReranker(),
    )

    with start_timeline() as timeline:
        result = await spine.retrieve(
            "q?", AccessContext(principal_id="u", principal_groups=("hr",))
        )

    # The default rerank_on_failure="degrade" means the result still has chunks.
    assert result.chunks, "degrade-on-failure policy must keep chunks (fused order)"
    assert result.diagnostics.get("rerank") == "degraded"

    rerank_records = [r for r in timeline.records if r.name == "iw_rag.retrieval.rerank"]
    assert len(rerank_records) == 1
    assert rerank_records[0].attributes.get("openinference.span.kind") == "RERANKER"
