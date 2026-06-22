"""Unit tests for the online-upload pooled HTTP embedder (ingestion/http_embed.py).

Covers:
- ``InfinityDocumentEmbedder.embed_document``: L2-normalization of returned vectors
  (exact values via pytest.approx), batching boundaries, count-mismatch and
  dim-mismatch EmbedErrors (with match=), empty-leaves early-return.
- ``InfinityDocumentEmbedder.health()``: True on HTTP 200, False on transport error.

All ``httpx.AsyncClient`` calls are mocked — no live Infinity endpoint, no network.
"""

from __future__ import annotations

import math
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from iw_rag.domain.errors import EmbedError
from iw_rag.ingestion.chunk import ChunkNode, TokenSpan
from iw_rag.ingestion.embed import EMBEDDING_DIMS
from iw_rag.ingestion.http_embed import InfinityDocumentEmbedder

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _leaf(local_id: str, content: str = "some text") -> ChunkNode:
    """Build a minimal leaf ChunkNode for tests.

    Args:
        local_id: The local identifier for the chunk.
        content: The text content of the chunk.

    Returns:
        A ChunkNode suitable for passing to embed_document.
    """
    return ChunkNode(
        local_id=local_id,
        parent_local_id="p0",
        ordinal=0,
        content=content,
        content_type="text",
        section_path=None,
        page=None,
        token_span=TokenSpan(0, 1),
    )


def _canned_vec(dims: int = EMBEDDING_DIMS, value: float = 1.0) -> list[float]:
    """Build a vector of the given dimensionality filled with a constant.

    Args:
        dims: The number of dimensions.
        value: The fill value.

    Returns:
        A list of floats with the given length and fill value.
    """
    return [value] * dims


def _mock_httpx(
    payloads: list[dict[str, Any]],
    *,
    status_code: int = 200,
) -> MagicMock:
    """Build a mock httpx.AsyncClient whose ``post`` method returns canned payloads.

    Each successive ``post`` call returns the next payload in the list.  If only
    one payload is given it is repeated for all calls.

    Args:
        payloads: Ordered list of JSON payloads to return.
        status_code: HTTP status code for every response.

    Returns:
        A MagicMock matching the ``async with httpx.AsyncClient() as client:`` pattern.
    """
    responses = []
    for payload in payloads:
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = status_code
        resp.json = MagicMock(return_value=payload)
        if status_code >= 400:
            resp.raise_for_status = MagicMock(
                side_effect=httpx.HTTPStatusError(
                    f"HTTP {status_code}",
                    request=MagicMock(),
                    response=resp,
                )
            )
        else:
            resp.raise_for_status = MagicMock(return_value=None)
        responses.append(resp)

    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    if len(responses) == 1:
        mock_client.post = AsyncMock(return_value=responses[0])
    else:
        mock_client.post = AsyncMock(side_effect=responses)
    return mock_client


def _make_payload(vectors: list[list[float]]) -> dict[str, Any]:
    """Wrap a list of vectors in the Infinity ``/embeddings`` response envelope.

    Args:
        vectors: The embedding vectors to wrap.

    Returns:
        A dict matching the Infinity response schema.
    """
    return {"data": [{"embedding": v} for v in vectors]}


def _embedder(
    *,
    batch_size: int = 32,
    normalize: bool = True,
) -> InfinityDocumentEmbedder:
    """Build an InfinityDocumentEmbedder pointed at a fake endpoint.

    Args:
        batch_size: Max leaves per embeddings request.
        normalize: Whether to L2-normalize returned vectors.

    Returns:
        A configured InfinityDocumentEmbedder.
    """
    return InfinityDocumentEmbedder(
        endpoint="http://infinity:7997",
        model="BAAI/bge-m3",
        batch_size=batch_size,
        normalize=normalize,
    )


# ---------------------------------------------------------------------------
# 1. empty-leaves early return
# ---------------------------------------------------------------------------


async def test_embed_document_empty_leaves_returns_empty() -> None:
    """embed_document returns [] immediately when leaves is empty (no HTTP call)."""
    embedder = _embedder()
    with patch("httpx.AsyncClient") as mock_cls:
        result = await embedder.embed_document([], [])
    mock_cls.assert_not_called()
    assert result == []


# ---------------------------------------------------------------------------
# 2. L2-normalization — exact values via pytest.approx
# ---------------------------------------------------------------------------


async def test_embed_document_normalizes_returned_vector() -> None:
    """embed_document L2-normalizes each vector to unit length."""
    leaf = _leaf("l0", "hello world")
    # Raw vector: all 3.0 in 1024 dims — norm = 3.0 * sqrt(1024) = 3 * 32 = 96.0
    raw = [3.0] * EMBEDDING_DIMS
    payload = _make_payload([raw])
    mock_client = _mock_httpx([payload])

    embedder = _embedder(normalize=True)
    with patch("httpx.AsyncClient", return_value=mock_client):
        results = await embedder.embed_document([0], [leaf])

    assert len(results) == 1
    # Expected: each component = 3.0 / (3.0 * sqrt(1024)) = 1/32
    expected_component = 1.0 / math.sqrt(EMBEDDING_DIMS)
    assert results[0].dense == pytest.approx([expected_component] * EMBEDDING_DIMS)
    assert results[0].local_id == "l0"


async def test_embed_document_normalize_false_returns_raw_vector() -> None:
    """embed_document with normalize=False returns vectors unchanged."""
    leaf = _leaf("l1")
    # Raw vector with distinct values so we can verify no normalization happened
    raw = [float(i % 10) for i in range(EMBEDDING_DIMS)]
    payload = _make_payload([raw])
    mock_client = _mock_httpx([payload])

    embedder = _embedder(normalize=False)
    with patch("httpx.AsyncClient", return_value=mock_client):
        results = await embedder.embed_document([0], [leaf])

    assert results[0].dense == pytest.approx(raw)


async def test_embed_document_normalized_vector_is_unit_length() -> None:
    """Normalized vectors from embed_document have L2-norm == 1.0."""
    leaf = _leaf("l2")
    # Use a vector where components are not uniform (more meaningful check)
    raw = [float(i + 1) for i in range(EMBEDDING_DIMS)]
    payload = _make_payload([raw])
    mock_client = _mock_httpx([payload])

    embedder = _embedder(normalize=True)
    with patch("httpx.AsyncClient", return_value=mock_client):
        results = await embedder.embed_document([0], [leaf])

    norm = math.sqrt(sum(v**2 for v in results[0].dense))
    assert norm == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 3. Batching — two batches are issued when leaves exceeds batch_size
# ---------------------------------------------------------------------------


async def test_embed_document_single_batch_when_leaves_fit() -> None:
    """embed_document issues a single POST when leaves <= batch_size."""
    leaves = [_leaf(f"l{i}") for i in range(3)]
    raw_vecs = [_canned_vec() for _ in leaves]
    payload = _make_payload(raw_vecs)
    mock_client = _mock_httpx([payload])

    embedder = _embedder(batch_size=10, normalize=False)
    with patch("httpx.AsyncClient", return_value=mock_client):
        results = await embedder.embed_document([0, 1, 2], leaves)

    assert mock_client.post.call_count == 1
    assert len(results) == 3


async def test_embed_document_splits_into_two_batches() -> None:
    """embed_document issues two POST calls when leaves > batch_size."""
    leaves = [_leaf(f"l{i}") for i in range(5)]
    # Batch 1: leaves 0-2, batch 2: leaves 3-4
    batch1_vecs = [_canned_vec() for _ in range(3)]
    batch2_vecs = [_canned_vec() for _ in range(2)]
    mock_client = _mock_httpx([_make_payload(batch1_vecs), _make_payload(batch2_vecs)])

    embedder = _embedder(batch_size=3, normalize=False)
    with patch("httpx.AsyncClient", return_value=mock_client):
        results = await embedder.embed_document([], leaves)

    assert mock_client.post.call_count == 2
    assert len(results) == 5


async def test_embed_document_batch_boundary_all_leaves_embedded() -> None:
    """embed_document produces exactly one ChunkEmbedding per leaf across batches."""
    n_leaves = 7
    batch_size = 3
    leaves = [_leaf(f"l{i}") for i in range(n_leaves)]

    # Build per-batch payloads
    payloads = []
    start = 0
    while start < n_leaves:
        end = min(start + batch_size, n_leaves)
        payloads.append(_make_payload([_canned_vec() for _ in range(end - start)]))
        start += batch_size

    mock_client = _mock_httpx(payloads)
    embedder = _embedder(batch_size=batch_size, normalize=False)
    with patch("httpx.AsyncClient", return_value=mock_client):
        results = await embedder.embed_document([], leaves)

    assert len(results) == n_leaves
    result_ids = {r.local_id for r in results}
    assert result_ids == {f"l{i}" for i in range(n_leaves)}


async def test_embed_document_batch_texts_match_leaf_content() -> None:
    """embed_document posts the correct leaf content strings in each batch."""
    leaves = [_leaf("a", "text alpha"), _leaf("b", "text beta")]
    raw_vecs = [_canned_vec() for _ in leaves]
    payload = _make_payload(raw_vecs)
    mock_client = _mock_httpx([payload])

    embedder = _embedder(batch_size=10, normalize=False)
    with patch("httpx.AsyncClient", return_value=mock_client):
        await embedder.embed_document([], leaves)

    call_kwargs = mock_client.post.call_args
    body = call_kwargs.kwargs.get("json") or call_kwargs.args[1]
    assert body["input"] == ["text alpha", "text beta"]


# ---------------------------------------------------------------------------
# 4. Count-mismatch EmbedError
# ---------------------------------------------------------------------------


async def test_embed_document_count_mismatch_raises_embed_error() -> None:
    """embed_document raises EmbedError when server returns fewer vectors than inputs."""
    leaves = [_leaf("a"), _leaf("b"), _leaf("c")]
    # Only 2 vectors for 3 inputs
    payload = _make_payload([_canned_vec(), _canned_vec()])
    mock_client = _mock_httpx([payload])

    embedder = _embedder(normalize=False)
    with (
        patch("httpx.AsyncClient", return_value=mock_client),
        pytest.raises(
            EmbedError,
            match=r"embedder returned 2 vectors for 3 inputs",
        ),
    ):
        await embedder.embed_document([], leaves)


async def test_embed_document_count_mismatch_more_vectors_raises() -> None:
    """embed_document raises EmbedError when server returns more vectors than inputs."""
    leaves = [_leaf("a")]
    # 2 vectors for 1 input
    payload = _make_payload([_canned_vec(), _canned_vec()])
    mock_client = _mock_httpx([payload])

    embedder = _embedder(normalize=False)
    with (
        patch("httpx.AsyncClient", return_value=mock_client),
        pytest.raises(
            EmbedError,
            match=r"embedder returned 2 vectors for 1 inputs",
        ),
    ):
        await embedder.embed_document([], leaves)


# ---------------------------------------------------------------------------
# 5. Dimension-mismatch EmbedError
# ---------------------------------------------------------------------------


async def test_embed_document_dim_mismatch_raises_embed_error() -> None:
    """embed_document raises EmbedError when a vector has wrong dimensionality."""
    leaf = _leaf("l0")
    wrong_dim = EMBEDDING_DIMS + 1
    raw = [1.0] * wrong_dim
    payload = _make_payload([raw])
    mock_client = _mock_httpx([payload])

    embedder = _embedder(normalize=False)
    with (
        patch("httpx.AsyncClient", return_value=mock_client),
        pytest.raises(
            EmbedError,
            match=rf"embedder returned dim {wrong_dim} \(expected {EMBEDDING_DIMS}\)",
        ),
    ):
        await embedder.embed_document([], [leaf])


async def test_embed_document_dim_mismatch_too_small_raises() -> None:
    """embed_document raises EmbedError when vector dimension is too small."""
    leaf = _leaf("l0")
    wrong_dim = 512
    raw = [1.0] * wrong_dim
    payload = _make_payload([raw])
    mock_client = _mock_httpx([payload])

    embedder = _embedder(normalize=False)
    with (
        patch("httpx.AsyncClient", return_value=mock_client),
        pytest.raises(
            EmbedError,
            match=rf"embedder returned dim {wrong_dim} \(expected {EMBEDDING_DIMS}\)",
        ),
    ):
        await embedder.embed_document([], [leaf])


# ---------------------------------------------------------------------------
# 6. Transport / HTTP errors map to EmbedError
# ---------------------------------------------------------------------------


async def test_embed_document_transport_error_raises_embed_error() -> None:
    """embed_document wraps httpx transport errors in EmbedError."""
    leaf = _leaf("l0")
    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.post = AsyncMock(side_effect=httpx.ConnectError("connection refused"))

    embedder = _embedder()
    with (
        patch("httpx.AsyncClient", return_value=mock_client),
        pytest.raises(EmbedError, match="document embedding failed"),
    ):
        await embedder.embed_document([], [leaf])


async def test_embed_document_http_status_error_raises_embed_error() -> None:
    """embed_document wraps HTTP 5xx status errors in EmbedError."""
    leaf = _leaf("l0")
    # Payload doesn't matter since raise_for_status raises first
    payload = _make_payload([_canned_vec()])
    mock_client = _mock_httpx([payload], status_code=503)

    embedder = _embedder()
    with (
        patch("httpx.AsyncClient", return_value=mock_client),
        pytest.raises(EmbedError, match="document embedding failed"),
    ):
        await embedder.embed_document([], [leaf])


async def test_embed_document_malformed_payload_raises_embed_error() -> None:
    """embed_document wraps malformed server JSON (missing 'data' key) in EmbedError."""
    leaf = _leaf("l0")
    bad_payload = {"result": "oops"}  # no 'data' key
    resp = MagicMock(spec=httpx.Response)
    resp.raise_for_status = MagicMock(return_value=None)
    resp.json = MagicMock(return_value=bad_payload)
    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.post = AsyncMock(return_value=resp)

    embedder = _embedder()
    with (
        patch("httpx.AsyncClient", return_value=mock_client),
        pytest.raises(EmbedError, match="document embedding failed"),
    ):
        await embedder.embed_document([], [leaf])


# ---------------------------------------------------------------------------
# 7. health() — reachable and unreachable
# ---------------------------------------------------------------------------


async def test_health_returns_true_on_200() -> None:
    """health() returns True when the Infinity /models endpoint returns HTTP 200."""
    resp = MagicMock(spec=httpx.Response)
    resp.raise_for_status = MagicMock(return_value=None)
    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.get = AsyncMock(return_value=resp)

    embedder = _embedder()
    with patch("httpx.AsyncClient", return_value=mock_client):
        result = await embedder.health()

    assert result is True


async def test_health_returns_false_on_transport_error() -> None:
    """health() returns False when the Infinity endpoint is unreachable."""
    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.get = AsyncMock(side_effect=httpx.ConnectError("refused"))

    embedder = _embedder()
    with patch("httpx.AsyncClient", return_value=mock_client):
        result = await embedder.health()

    assert result is False


async def test_health_returns_false_on_http_status_error() -> None:
    """health() returns False when the /models endpoint returns a non-2xx status."""
    resp = MagicMock(spec=httpx.Response)
    resp.raise_for_status = MagicMock(
        side_effect=httpx.HTTPStatusError("503", request=MagicMock(), response=resp)
    )
    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.get = AsyncMock(return_value=resp)

    embedder = _embedder()
    with patch("httpx.AsyncClient", return_value=mock_client):
        result = await embedder.health()

    assert result is False


# ---------------------------------------------------------------------------
# 8. ChunkEmbedding metadata preservation
# ---------------------------------------------------------------------------


async def test_embed_document_local_id_preserved() -> None:
    """embed_document assigns the correct local_id from each leaf to its embedding."""
    leaves = [_leaf("chunk-42"), _leaf("chunk-99")]
    raw_vecs = [_canned_vec(), _canned_vec()]
    payload = _make_payload(raw_vecs)
    mock_client = _mock_httpx([payload])

    embedder = _embedder(normalize=False)
    with patch("httpx.AsyncClient", return_value=mock_client):
        results = await embedder.embed_document([], leaves)

    assert results[0].local_id == "chunk-42"
    assert results[1].local_id == "chunk-99"


async def test_embed_document_sparse_weights_always_empty() -> None:
    """embed_document always returns empty sparse_weights (Phase 0 contract)."""
    leaf = _leaf("l0")
    payload = _make_payload([_canned_vec()])
    mock_client = _mock_httpx([payload])

    embedder = _embedder(normalize=False)
    with patch("httpx.AsyncClient", return_value=mock_client):
        results = await embedder.embed_document([], [leaf])

    assert results[0].sparse_weights == {}


# ---------------------------------------------------------------------------
# 9. Endpoint URL construction
# ---------------------------------------------------------------------------


async def test_embed_document_posts_to_correct_endpoint() -> None:
    """embed_document POSTs to <endpoint>/embeddings (trailing slash stripped)."""
    leaf = _leaf("l0")
    payload = _make_payload([_canned_vec()])
    mock_client = _mock_httpx([payload])

    # Endpoint with trailing slash — should be stripped
    embedder = InfinityDocumentEmbedder(
        endpoint="http://infinity:7997/",
        normalize=False,
    )
    with patch("httpx.AsyncClient", return_value=mock_client):
        await embedder.embed_document([], [leaf])

    call_args = mock_client.post.call_args
    url = call_args.args[0] if call_args.args else call_args.kwargs.get("url", "")
    assert url == "http://infinity:7997/embeddings"


async def test_embed_document_posts_correct_model_name() -> None:
    """embed_document includes the configured model name in the request body."""
    leaf = _leaf("l0")
    payload = _make_payload([_canned_vec()])
    mock_client = _mock_httpx([payload])

    embedder = InfinityDocumentEmbedder(
        endpoint="http://infinity:7997",
        model="custom/my-model",
        normalize=False,
    )
    with patch("httpx.AsyncClient", return_value=mock_client):
        await embedder.embed_document([], [leaf])

    call_kwargs = mock_client.post.call_args
    body = call_kwargs.kwargs.get("json") or call_kwargs.args[1]
    assert body["model"] == "custom/my-model"
