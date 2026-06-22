"""Cross-encoder reranking — bge-reranker-v2-m3 (served by infinity on Blackwell).

The highest-ROI precision stage: rescore the fused candidates against the query
and keep the top-n. **Degrade-not-fail**: a reranker outage returns the fused
order unchanged with a ``degraded`` diagnostic, rather than failing the query
(``rerank_on_failure=degrade`` default). Endpoint/model come from config.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import httpx

from iw_rag.domain.errors import RerankError


@dataclass(frozen=True, slots=True)
class RerankCandidate:
    """A candidate passed to the reranker.

    Attributes:
        chunk_id: The chunk id.
        content: The text scored against the query.
    """

    chunk_id: str
    content: str


@dataclass(frozen=True, slots=True)
class ScoredChunk:
    """A reranked chunk.

    Attributes:
        chunk_id: The chunk id.
        score: The reranker relevance score (higher = better).
    """

    chunk_id: str
    score: float


@runtime_checkable
class Reranker(Protocol):
    """Rescores query-candidate pairs and returns the top-n."""

    async def rerank(
        self, query: str, candidates: Sequence[RerankCandidate], top_n: int
    ) -> list[ScoredChunk]:
        """Return the top-n candidates by relevance to the query."""
        ...

    async def health(self) -> bool:
        """Whether the reranker endpoint is reachable."""
        ...


class InfinityReranker:
    """bge-reranker-v2-m3 via the infinity ``/rerank`` endpoint (OpenAI-style)."""

    def __init__(
        self,
        endpoint: str,
        model: str,
        timeout_s: float = 30.0,
        *,
        max_doc_chars: int = 2048,
    ) -> None:
        """Configure the reranker endpoint + model.

        Args:
            endpoint: Base URL of the infinity server (e.g. ``http://localhost:7997``).
            model: Served reranker model id (e.g. ``BAAI/bge-reranker-v2-m3``).
            timeout_s: Per-request timeout.
            max_doc_chars: Truncate each candidate's text to this many chars
                before scoring (I-CTXOVERFLOW perf).  The cross-encoder judges
                query-relevance from the candidate's head; sending a whole
                ~50,000-token document (e.g. the long-context corpus chunks)
                forces bge-reranker-v2-m3 to chew through content far past its
                ~8k window and was costing a flat ~30s per query.  A ~2k-char
                head keeps scoring representative while bounding rerank latency
                to sub-second.  ``<= 0`` disables truncation.
        """
        self._endpoint = endpoint.rstrip("/")
        self._model = model
        self._timeout_s = timeout_s
        self._max_doc_chars = max_doc_chars

    async def rerank(
        self, query: str, candidates: Sequence[RerankCandidate], top_n: int
    ) -> list[ScoredChunk]:
        """Rescore via infinity and return the top-n scored chunks.

        Args:
            query: The user query.
            candidates: The fused candidates to rescore.
            top_n: How many to keep.

        Returns:
            The top-n scored chunks, sorted by score desc.

        Raises:
            RerankError: On any transport/decoding failure (the spine decides
                whether to degrade).
        """
        if not candidates:
            return []
        # Cap each candidate's length: the cross-encoder scores relevance from
        # the head, and an oversized candidate (whole-doc chunk) otherwise blows
        # rerank latency to ~30s (I-CTXOVERFLOW perf).
        if self._max_doc_chars > 0:
            docs = [c.content[: self._max_doc_chars] for c in candidates]
        else:
            docs = [c.content for c in candidates]
        try:
            async with httpx.AsyncClient(timeout=self._timeout_s) as client:
                resp = await client.post(
                    f"{self._endpoint}/rerank",
                    json={"model": self._model, "query": query, "documents": docs},
                )
                resp.raise_for_status()
                payload = resp.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise RerankError(f"reranker request failed: {exc}") from exc

        # The infinity /rerank endpoint may return {"results": [...]} or a bare
        # JSON list; only dicts carry the "results" envelope.
        results = payload.get("results", payload) if isinstance(payload, dict) else payload
        scored: list[ScoredChunk] = []
        for item in results:
            idx = item["index"]
            score = float(item.get("relevance_score", item.get("score", 0.0)))
            scored.append(ScoredChunk(chunk_id=candidates[idx].chunk_id, score=score))
        scored.sort(key=lambda s: -s.score)
        return scored[:top_n]

    async def health(self) -> bool:
        """Ping the infinity health endpoint."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self._endpoint}/health")
                return resp.status_code == 200
        except httpx.HTTPError:
            return False
