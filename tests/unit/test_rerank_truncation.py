"""Unit test for the reranker candidate-length cap (perf, I-CTXOVERFLOW).

bge-reranker-v2-m3 scores query-relevance from a candidate's head; sending a
whole ~50k-token long-document chunk pushed rerank latency to a flat ~30s/query.
The client truncates each candidate to ``max_doc_chars`` before scoring.  This
test patches the HTTP layer to capture exactly what is sent.
"""

from __future__ import annotations

from typing import Any

import pytest

from iw_rag.retrieval import rerank as rerank_mod
from iw_rag.retrieval.rerank import InfinityReranker, RerankCandidate


class _FakeResponse:
    def __init__(self, n: int) -> None:
        self._n = n

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return {
            "results": [{"index": i, "relevance_score": 1.0 - i * 0.01} for i in range(self._n)]
        }


class _FakeClient:
    """Captures the JSON body of the single ``/rerank`` POST."""

    last_body: dict[str, Any] = {}

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    async def __aenter__(self) -> _FakeClient:
        return self

    async def __aexit__(self, *args: Any) -> bool:
        return False

    async def post(self, url: str, json: dict[str, Any]) -> _FakeResponse:
        _FakeClient.last_body = json
        return _FakeResponse(len(json["documents"]))


@pytest.mark.asyncio
async def test_rerank_truncates_oversized_candidates(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rerank_mod.httpx, "AsyncClient", _FakeClient)
    reranker = InfinityReranker(
        "http://infinity:7997", "BAAI/bge-reranker-v2-m3", max_doc_chars=2048
    )
    candidates = [
        RerankCandidate(chunk_id="giant", content="A" * 196_160),  # whole-doc chunk
        RerankCandidate(chunk_id="small", content="short doc"),
    ]

    scored = await reranker.rerank("q?", candidates, top_n=2)

    sent = _FakeClient.last_body["documents"]
    assert len(sent[0]) == 2048, "oversized candidate must be capped to max_doc_chars"
    assert sent[1] == "short doc", "within-budget candidate passed through verbatim"
    # Scoring still maps back to the original candidate ids.
    assert {s.chunk_id for s in scored} == {"giant", "small"}


@pytest.mark.asyncio
async def test_rerank_zero_cap_disables_truncation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rerank_mod.httpx, "AsyncClient", _FakeClient)
    reranker = InfinityReranker("http://infinity:7997", "m", max_doc_chars=0)
    await reranker.rerank("q?", [RerankCandidate(chunk_id="c", content="B" * 5000)], top_n=1)
    assert len(_FakeClient.last_body["documents"][0]) == 5000
