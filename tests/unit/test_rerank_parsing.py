"""Unit tests for InfinityReranker response-parsing logic (rerank.py).

These tests exercise the parsing and ordering paths that are NOT covered by
``test_rerank_truncation.py`` (which only verifies what is *sent*, not what
is *returned*).  All tests patch ``httpx.AsyncClient`` — no live calls.

Branches covered:
  - Out-of-order ``index`` remap: results arrive in scrambled order; each
    ScoredChunk must map back to the correct candidate by index.
  - Descending score sort + ``top_n`` trim: exact ordered chunk_id list with
    scores.
  - ``relevance_score``→``score``→``0.0`` key fallback priority.
  - ``results = payload.get("results", payload)`` bare-list fallback.
  - ``if not candidates: return []`` short-circuit.
  - ``health()`` returning True on 200 and False on HTTPError.
  - ``RerankError`` raised on transport failure with ``match=``.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from iw_rag.domain.errors import RerankError
from iw_rag.retrieval import rerank as rerank_mod
from iw_rag.retrieval.rerank import InfinityReranker, RerankCandidate, ScoredChunk

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_ENDPOINT = "http://infinity:7997"
_MODEL = "BAAI/bge-reranker-v2-m3"


def _make_post_client(payload: Any) -> type:
    """Build a fake AsyncClient whose POST always returns the given payload.

    Args:
        payload: The dict/list returned by ``response.json()``.

    Returns:
        A class compatible with ``async with httpx.AsyncClient(...) as c``.
    """

    class _FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> Any:
            return payload

    class _FakeClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def __aenter__(self) -> _FakeClient:
            return self

        async def __aexit__(self, *args: Any) -> bool:
            return False

        async def post(self, url: str, json: dict[str, Any]) -> _FakeResponse:
            return _FakeResponse()

        async def get(self, url: str) -> _FakeResponse:  # pragma: no cover
            return _FakeResponse()

    return _FakeClient


def _make_get_client(status_code: int) -> type:
    """Build a fake AsyncClient whose GET returns the given HTTP status code.

    Args:
        status_code: The HTTP status code to simulate.

    Returns:
        A class compatible with ``async with httpx.AsyncClient(...) as c``.
    """

    class _FakeResponse:
        def __init__(self) -> None:
            self.status_code = status_code

        def raise_for_status(self) -> None:
            return None

    class _FakeClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def __aenter__(self) -> _FakeClient:
            return self

        async def __aexit__(self, *args: Any) -> bool:
            return False

        async def get(self, url: str) -> _FakeResponse:
            return _FakeResponse()

    return _FakeClient


def _make_error_client() -> type:
    """Build a fake AsyncClient whose POST raises an httpx.ConnectError.

    Returns:
        A class compatible with ``async with httpx.AsyncClient(...) as c``.
    """

    class _FakeClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def __aenter__(self) -> _FakeClient:
            return self

        async def __aexit__(self, *args: Any) -> bool:
            return False

        async def post(self, url: str, json: dict[str, Any]) -> Any:
            raise httpx.ConnectError("connection refused")

        async def get(self, url: str) -> Any:
            raise httpx.ConnectError("connection refused")

    return _FakeClient


# ---------------------------------------------------------------------------
# Empty-candidates short-circuit
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_candidates_returns_empty_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``if not candidates: return []`` fires before any HTTP call is made."""
    # If the HTTP layer is hit the test would fail with an error — the
    # _make_error_client ensures that would be noisy rather than silent.
    monkeypatch.setattr(rerank_mod.httpx, "AsyncClient", _make_error_client())
    reranker = InfinityReranker(_ENDPOINT, _MODEL)
    result = await reranker.rerank("q", [], top_n=5)
    assert result == []


# ---------------------------------------------------------------------------
# Out-of-order index remap
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_out_of_order_index_maps_to_correct_chunk_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Results in reversed index order must still map to the correct chunk_ids.

    Candidates are [chunk-A (index 0), chunk-B (index 1), chunk-C (index 2)].
    The server returns results for index 2, 0, 1 (reversed / scrambled).
    Each ScoredChunk must carry the chunk_id of the candidate at that index.
    """
    candidates = [
        RerankCandidate(chunk_id="chunk-A", content="alpha"),
        RerankCandidate(chunk_id="chunk-B", content="beta"),
        RerankCandidate(chunk_id="chunk-C", content="gamma"),
    ]
    # Server returns results in scrambled index order
    payload = {
        "results": [
            {"index": 2, "relevance_score": 0.9},  # chunk-C
            {"index": 0, "relevance_score": 0.7},  # chunk-A
            {"index": 1, "relevance_score": 0.5},  # chunk-B
        ]
    }
    monkeypatch.setattr(rerank_mod.httpx, "AsyncClient", _make_post_client(payload))
    reranker = InfinityReranker(_ENDPOINT, _MODEL, max_doc_chars=0)
    scored = await reranker.rerank("q", candidates, top_n=3)

    # After sorting by score desc: chunk-C(0.9), chunk-A(0.7), chunk-B(0.5)
    assert [s.chunk_id for s in scored] == ["chunk-C", "chunk-A", "chunk-B"]
    assert scored[0] == ScoredChunk(chunk_id="chunk-C", score=pytest.approx(0.9))
    assert scored[1] == ScoredChunk(chunk_id="chunk-A", score=pytest.approx(0.7))
    assert scored[2] == ScoredChunk(chunk_id="chunk-B", score=pytest.approx(0.5))


# ---------------------------------------------------------------------------
# Descending score sort + top_n trim
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_results_sorted_descending_and_trimmed_to_top_n(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Results are sorted by score descending and trimmed to top_n exactly."""
    candidates = [RerankCandidate(chunk_id=f"c{i}", content=f"doc {i}") for i in range(5)]
    # Deliberately provide scores out of order
    payload = {
        "results": [
            {"index": 0, "relevance_score": 0.3},
            {"index": 1, "relevance_score": 0.9},
            {"index": 2, "relevance_score": 0.1},
            {"index": 3, "relevance_score": 0.7},
            {"index": 4, "relevance_score": 0.5},
        ]
    }
    monkeypatch.setattr(rerank_mod.httpx, "AsyncClient", _make_post_client(payload))
    reranker = InfinityReranker(_ENDPOINT, _MODEL, max_doc_chars=0)
    scored = await reranker.rerank("q", candidates, top_n=3)

    assert len(scored) == 3
    assert [s.chunk_id for s in scored] == ["c1", "c3", "c4"]
    assert scored[0].score == pytest.approx(0.9)
    assert scored[1].score == pytest.approx(0.7)
    assert scored[2].score == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_top_n_larger_than_results_returns_all(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When top_n exceeds available results, all results are returned."""
    candidates = [
        RerankCandidate(chunk_id="x", content="doc x"),
        RerankCandidate(chunk_id="y", content="doc y"),
    ]
    payload = {
        "results": [
            {"index": 0, "relevance_score": 0.8},
            {"index": 1, "relevance_score": 0.6},
        ]
    }
    monkeypatch.setattr(rerank_mod.httpx, "AsyncClient", _make_post_client(payload))
    reranker = InfinityReranker(_ENDPOINT, _MODEL, max_doc_chars=0)
    scored = await reranker.rerank("q", candidates, top_n=10)
    assert len(scored) == 2
    assert [s.chunk_id for s in scored] == ["x", "y"]


# ---------------------------------------------------------------------------
# Key fallback: relevance_score → score → 0.0
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_score_key_fallback_used_when_relevance_score_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When 'relevance_score' is absent, 'score' is used as the fallback."""
    candidates = [
        RerankCandidate(chunk_id="a", content="doc a"),
        RerankCandidate(chunk_id="b", content="doc b"),
    ]
    payload = {
        "results": [
            {"index": 0, "score": 0.55},
            {"index": 1, "score": 0.75},
        ]
    }
    monkeypatch.setattr(rerank_mod.httpx, "AsyncClient", _make_post_client(payload))
    reranker = InfinityReranker(_ENDPOINT, _MODEL, max_doc_chars=0)
    scored = await reranker.rerank("q", candidates, top_n=2)

    assert [s.chunk_id for s in scored] == ["b", "a"]
    assert scored[0].score == pytest.approx(0.75)
    assert scored[1].score == pytest.approx(0.55)


@pytest.mark.asyncio
async def test_zero_fallback_used_when_both_score_keys_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When both 'relevance_score' and 'score' are absent, 0.0 is used."""
    candidates = [RerankCandidate(chunk_id="z", content="doc z")]
    payload = {"results": [{"index": 0}]}
    monkeypatch.setattr(rerank_mod.httpx, "AsyncClient", _make_post_client(payload))
    reranker = InfinityReranker(_ENDPOINT, _MODEL, max_doc_chars=0)
    scored = await reranker.rerank("q", candidates, top_n=1)
    assert scored == [ScoredChunk(chunk_id="z", score=0.0)]


@pytest.mark.asyncio
async def test_relevance_score_takes_priority_over_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When both keys are present, 'relevance_score' wins over 'score'."""
    candidates = [RerankCandidate(chunk_id="c", content="doc c")]
    payload = {"results": [{"index": 0, "relevance_score": 0.88, "score": 0.11}]}
    monkeypatch.setattr(rerank_mod.httpx, "AsyncClient", _make_post_client(payload))
    reranker = InfinityReranker(_ENDPOINT, _MODEL, max_doc_chars=0)
    scored = await reranker.rerank("q", candidates, top_n=1)
    assert scored[0].score == pytest.approx(0.88)


# ---------------------------------------------------------------------------
# Bare-list payload fallback (payload.get("results", payload))
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bare_list_payload_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the response is a bare list (no 'results' key), it is used directly.

    The infinity /rerank endpoint may return either ``{"results": [...]}`` or a
    bare JSON list; the parser handles both (the ``isinstance(payload, dict)``
    guard in rerank.py avoids calling ``.get()`` on a list).
    """
    candidates = [
        RerankCandidate(chunk_id="p", content="doc p"),
        RerankCandidate(chunk_id="q", content="doc q"),
    ]
    # Bare list — no wrapping dict with 'results'
    bare_payload = [
        {"index": 1, "relevance_score": 0.95},
        {"index": 0, "relevance_score": 0.40},
    ]
    monkeypatch.setattr(rerank_mod.httpx, "AsyncClient", _make_post_client(bare_payload))
    reranker = InfinityReranker(_ENDPOINT, _MODEL, max_doc_chars=0)
    scored = await reranker.rerank("query", candidates, top_n=2)

    assert [s.chunk_id for s in scored] == ["q", "p"]
    assert scored[0].score == pytest.approx(0.95)
    assert scored[1].score == pytest.approx(0.40)


# ---------------------------------------------------------------------------
# health()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_health_returns_true_on_200(monkeypatch: pytest.MonkeyPatch) -> None:
    """health() returns True when the /health endpoint responds with 200."""
    monkeypatch.setattr(rerank_mod.httpx, "AsyncClient", _make_get_client(200))
    reranker = InfinityReranker(_ENDPOINT, _MODEL)
    assert await reranker.health() is True


@pytest.mark.asyncio
async def test_health_returns_false_on_non_200(monkeypatch: pytest.MonkeyPatch) -> None:
    """health() returns False when the /health endpoint responds with a non-200 code."""
    monkeypatch.setattr(rerank_mod.httpx, "AsyncClient", _make_get_client(503))
    reranker = InfinityReranker(_ENDPOINT, _MODEL)
    assert await reranker.health() is False


@pytest.mark.asyncio
async def test_health_returns_false_on_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """health() returns False (not raises) when a transport error occurs."""
    monkeypatch.setattr(rerank_mod.httpx, "AsyncClient", _make_error_client())
    reranker = InfinityReranker(_ENDPOINT, _MODEL)
    assert await reranker.health() is False


# ---------------------------------------------------------------------------
# RerankError raised on transport failure
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rerank_raises_rerank_error_on_transport_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RerankError is raised (not swallowed) when a transport error occurs."""
    monkeypatch.setattr(rerank_mod.httpx, "AsyncClient", _make_error_client())
    reranker = InfinityReranker(_ENDPOINT, _MODEL)
    candidates = [RerankCandidate(chunk_id="c", content="doc")]
    with pytest.raises(RerankError, match="reranker request failed"):
        await reranker.rerank("q", candidates, top_n=1)


@pytest.mark.asyncio
async def test_rerank_error_chains_original_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RerankError.__cause__ is set (raise … from exc), preserving the cause."""
    monkeypatch.setattr(rerank_mod.httpx, "AsyncClient", _make_error_client())
    reranker = InfinityReranker(_ENDPOINT, _MODEL)
    candidates = [RerankCandidate(chunk_id="c", content="doc")]
    with pytest.raises(RerankError, match="reranker request failed") as exc_info:
        await reranker.rerank("q", candidates, top_n=1)
    # The transport error must be preserved as the chained cause.
    assert type(exc_info.value.__cause__) is rerank_mod.httpx.ConnectError


@pytest.mark.asyncio
async def test_rerank_raises_rerank_error_on_http_status_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RerankError wraps an httpx.HTTPStatusError from raise_for_status()."""

    class _BadStatusResponse:
        def raise_for_status(self) -> None:
            req = httpx.Request("POST", _ENDPOINT + "/rerank")
            resp = httpx.Response(500, request=req)
            raise httpx.HTTPStatusError("internal server error", request=req, response=resp)

        def json(self) -> Any:  # pragma: no cover
            return {}

    class _BadStatusClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def __aenter__(self) -> _BadStatusClient:
            return self

        async def __aexit__(self, *args: Any) -> bool:
            return False

        async def post(self, url: str, json: dict[str, Any]) -> _BadStatusResponse:
            return _BadStatusResponse()

    monkeypatch.setattr(rerank_mod.httpx, "AsyncClient", _BadStatusClient)
    reranker = InfinityReranker(_ENDPOINT, _MODEL)
    candidates = [RerankCandidate(chunk_id="c", content="doc")]
    with pytest.raises(RerankError, match="reranker request failed"):
        await reranker.rerank("q", candidates, top_n=1)
