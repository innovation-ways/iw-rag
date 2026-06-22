"""Unit tests for spine invariants that need no DB (empty-ACL fail-closed path)."""

from __future__ import annotations

from collections.abc import Sequence

from iw_rag.domain.acl import AccessContext
from iw_rag.retrieval.rerank import RerankCandidate, ScoredChunk
from iw_rag.retrieval.spine import RetrievalSpine


class _Unused:
    """A collaborator that must not be called on the empty-ACL path."""

    def __getattr__(self, name: str) -> object:  # pragma: no cover - guard
        raise AssertionError(f"collaborator unexpectedly used: {name}")


class _NoopReranker:
    async def rerank(
        self, query: str, candidates: Sequence[RerankCandidate], top_n: int
    ) -> list[ScoredChunk]:  # pragma: no cover
        return []

    async def health(self) -> bool:  # pragma: no cover
        return True


async def test_empty_acl_returns_empty_result_without_touching_db() -> None:
    # No groups → fail-closed: empty result, acl_applied True, no encoder/db calls.
    spine = RetrievalSpine(
        db=_Unused(),  # type: ignore[arg-type]
        encoder=_Unused(),  # type: ignore[arg-type]
        dense=_Unused(),  # type: ignore[arg-type]
        sparse=_Unused(),  # type: ignore[arg-type]
        reranker=_NoopReranker(),
    )
    result = await spine.retrieve("anything", AccessContext(principal_id="u", principal_groups=()))
    assert result.chunks == []
    assert result.acl_applied is True
    assert result.retriever == "spine"
    assert result.diagnostics.get("acl") == "empty"
