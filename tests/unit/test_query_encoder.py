"""Unit tests for InfinityQueryEncoder.encode (encode.py).

All tests patch ``httpx.AsyncClient`` at the module level — no live network
calls are made.  The suite verifies:
  - Happy-path: the embedding vector is extracted from the correct JSON path.
  - Hard-error contract: every listed exception type (httpx.HTTPError, KeyError,
    IndexError, ValueError) is wrapped into RetrievalError with a message that
    matches the documented pattern.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from iw_rag.domain.errors import RetrievalError
from iw_rag.retrieval import encode as encode_mod
from iw_rag.retrieval.encode import InfinityQueryEncoder

# ---------------------------------------------------------------------------
# Shared fake-transport helpers
# ---------------------------------------------------------------------------

_ENDPOINT = "http://infinity:7997"
_MODEL = "BAAI/bge-m3"
_VECTOR = [0.1, 0.2, 0.3]


class _OkResponse:
    """Fake successful response carrying a single embedding."""

    def __init__(self, embedding: list[float]) -> None:
        self._embedding = embedding

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return {"data": [{"embedding": self._embedding}]}


class _ErrorResponse:
    """Fake response whose raise_for_status() raises an httpx.HTTPStatusError."""

    def raise_for_status(self) -> None:
        req = httpx.Request("POST", _ENDPOINT + "/embeddings")
        resp = httpx.Response(503, request=req)
        raise httpx.HTTPStatusError("service unavailable", request=req, response=resp)

    def json(self) -> dict[str, Any]:  # pragma: no cover
        return {}


def _make_client(response: Any) -> type:
    """Build a fake AsyncClient class that always returns the given response.

    Args:
        response: The response object returned by ``client.post()``.

    Returns:
        A class compatible with ``async with httpx.AsyncClient(...) as client``.
    """

    class _FakeClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def __aenter__(self) -> _FakeClient:
            return self

        async def __aexit__(self, *args: Any) -> bool:
            return False

        async def post(self, url: str, json: dict[str, Any]) -> Any:
            return response

    return _FakeClient


# ---------------------------------------------------------------------------
# Happy-path: correct extraction + exact vector
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_encode_returns_exact_embedding_vector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Happy path: the embedding at data[0]['embedding'] is returned verbatim."""
    monkeypatch.setattr(encode_mod.httpx, "AsyncClient", _make_client(_OkResponse(_VECTOR)))
    encoder = InfinityQueryEncoder(_ENDPOINT, _MODEL)
    result = await encoder.encode("find invoices")
    assert result == pytest.approx(_VECTOR)


@pytest.mark.asyncio
async def test_encode_returns_float_list(monkeypatch: pytest.MonkeyPatch) -> None:
    """The returned value is a list of floats (not a numpy array or other type)."""
    embedding = [0.5, -0.25, 1.0, 0.0]
    monkeypatch.setattr(encode_mod.httpx, "AsyncClient", _make_client(_OkResponse(embedding)))
    encoder = InfinityQueryEncoder(_ENDPOINT, _MODEL)
    result = await encoder.encode("q")
    assert isinstance(result, list)
    assert result == pytest.approx(embedding)


@pytest.mark.asyncio
async def test_encode_strips_trailing_slash_from_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Trailing slash on the endpoint is normalised away (no double slash in URL)."""
    captured_urls: list[str] = []

    class _CapturingClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def __aenter__(self) -> _CapturingClient:
            return self

        async def __aexit__(self, *args: Any) -> bool:
            return False

        async def post(self, url: str, json: dict[str, Any]) -> _OkResponse:
            captured_urls.append(url)
            return _OkResponse(_VECTOR)

    monkeypatch.setattr(encode_mod.httpx, "AsyncClient", _CapturingClient)
    encoder = InfinityQueryEncoder(_ENDPOINT + "/", _MODEL)
    await encoder.encode("q")
    assert captured_urls == [_ENDPOINT + "/embeddings"]


# ---------------------------------------------------------------------------
# Hard-error contract: RetrievalError wrapping
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_encode_wraps_http_status_error_as_retrieval_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """httpx.HTTPStatusError (subclass of HTTPError) → RetrievalError."""
    monkeypatch.setattr(encode_mod.httpx, "AsyncClient", _make_client(_ErrorResponse()))
    encoder = InfinityQueryEncoder(_ENDPOINT, _MODEL)
    with pytest.raises(RetrievalError, match="query embedding failed"):
        await encoder.encode("q")


@pytest.mark.asyncio
async def test_encode_wraps_connect_error_as_retrieval_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """httpx.ConnectError (subclass of HTTPError) during transport → RetrievalError."""

    class _ConnectErrorClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def __aenter__(self) -> _ConnectErrorClient:
            return self

        async def __aexit__(self, *args: Any) -> bool:
            return False

        async def post(self, url: str, json: dict[str, Any]) -> Any:
            raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(encode_mod.httpx, "AsyncClient", _ConnectErrorClient)
    encoder = InfinityQueryEncoder(_ENDPOINT, _MODEL)
    with pytest.raises(RetrievalError, match="query embedding failed"):
        await encoder.encode("q")


@pytest.mark.asyncio
async def test_encode_wraps_key_error_as_retrieval_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """KeyError on missing 'data' key → RetrievalError (embedding down contract)."""

    class _BadJsonResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {"object": "list"}  # missing 'data'

    monkeypatch.setattr(encode_mod.httpx, "AsyncClient", _make_client(_BadJsonResponse()))
    encoder = InfinityQueryEncoder(_ENDPOINT, _MODEL)
    with pytest.raises(RetrievalError, match="query embedding failed"):
        await encoder.encode("q")


@pytest.mark.asyncio
async def test_encode_wraps_index_error_as_retrieval_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """IndexError on empty 'data' list → RetrievalError."""

    class _EmptyDataResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {"data": []}  # data[0] → IndexError

    monkeypatch.setattr(encode_mod.httpx, "AsyncClient", _make_client(_EmptyDataResponse()))
    encoder = InfinityQueryEncoder(_ENDPOINT, _MODEL)
    with pytest.raises(RetrievalError, match="query embedding failed"):
        await encoder.encode("q")


@pytest.mark.asyncio
async def test_encode_wraps_value_error_as_retrieval_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ValueError (e.g. from json() on a non-JSON body) → RetrievalError."""

    class _ValueErrorResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            raise ValueError("not JSON")

    monkeypatch.setattr(encode_mod.httpx, "AsyncClient", _make_client(_ValueErrorResponse()))
    encoder = InfinityQueryEncoder(_ENDPOINT, _MODEL)
    with pytest.raises(RetrievalError, match="query embedding failed"):
        await encoder.encode("q")


@pytest.mark.asyncio
async def test_encode_wraps_missing_embedding_key_as_retrieval_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """KeyError on missing 'embedding' in the first data item → RetrievalError."""

    class _NoEmbeddingKeyResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {"data": [{"object": "embedding"}]}  # 'embedding' key absent

    monkeypatch.setattr(encode_mod.httpx, "AsyncClient", _make_client(_NoEmbeddingKeyResponse()))
    encoder = InfinityQueryEncoder(_ENDPOINT, _MODEL)
    with pytest.raises(RetrievalError, match="query embedding failed"):
        await encoder.encode("q")


@pytest.mark.asyncio
async def test_encode_retrieval_error_chains_original_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RetrievalError.__cause__ must be set (raise … from exc)."""

    class _BadJsonResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {}  # KeyError on 'data'

    monkeypatch.setattr(encode_mod.httpx, "AsyncClient", _make_client(_BadJsonResponse()))
    encoder = InfinityQueryEncoder(_ENDPOINT, _MODEL)
    with pytest.raises(RetrievalError, match="query embedding failed") as exc_info:
        await encoder.encode("q")
    # The wrapped cause must be the exact underlying error (missing 'data' key).
    assert type(exc_info.value.__cause__) is KeyError
