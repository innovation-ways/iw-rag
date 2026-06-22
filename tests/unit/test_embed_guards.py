"""Unit tests for LateChunkingEmbedder error guards (ingestion/embed.py).

Covers the four untested error paths in ``LateChunkingEmbedder.embed_document``
and the ``health()`` delegation:

- leaf with ``token_span is None`` raises ``EmbedError`` (line ~186).
- token-vector count mismatch guard raises ``EmbedError`` (lines ~193-196).
- empty local span raises ``EmbedError`` (line ~202).
- ``health()`` delegates to and returns the token embedder's readiness.

All tests inject a ``FakeTokenEmbedder`` — no torch, no transformers, no network.
``BgeM3*`` classes are never imported.
"""

from __future__ import annotations

import pytest

from iw_rag.domain.errors import EmbedError
from iw_rag.ingestion.chunk import ChunkNode, TokenSpan
from iw_rag.ingestion.embed import LateChunkingEmbedder

# ---------------------------------------------------------------------------
# Fake helpers
# ---------------------------------------------------------------------------


class _ConstantTokenEmbedder:
    """Fake token embedder that returns a constant 2-d vector per token.

    Args:
        health_value: The boolean to return from ``health()``.
    """

    def __init__(self, *, health_value: bool = True) -> None:
        """Store the canned health flag."""
        self._health_value = health_value

    async def embed_window(self, token_ids: list[int]) -> list[list[float]]:
        """Return ``[1.0, 0.0]`` for every token id."""
        return [[1.0, 0.0] for _ in token_ids]

    async def health(self) -> bool:
        """Return the preconfigured health flag."""
        return self._health_value


class _MismatchTokenEmbedder:
    """Fake token embedder that deliberately returns the wrong number of vectors.

    Args:
        extra: Number of extra vectors to append beyond the correct count.
    """

    def __init__(self, extra: int = 1) -> None:
        """Store the extra-vector count."""
        self._extra = extra

    async def embed_window(self, token_ids: list[int]) -> list[list[float]]:
        """Return ``len(token_ids) + extra`` vectors (simulates a buggy backend)."""
        return [[1.0, 0.0] for _ in range(len(token_ids) + self._extra)]

    async def health(self) -> bool:
        """Always healthy."""
        return True


# ---------------------------------------------------------------------------
# Leaf builder helpers
# ---------------------------------------------------------------------------


def _leaf_with_span(local_id: str, start: int, end: int) -> ChunkNode:
    """Build a leaf ChunkNode with a concrete TokenSpan.

    Args:
        local_id: The leaf identifier string.
        start: The span start (inclusive).
        end: The span end (exclusive).

    Returns:
        A ChunkNode with the given span.
    """
    return ChunkNode(
        local_id=local_id,
        parent_local_id="p0",
        ordinal=0,
        content="x",
        content_type="text",
        section_path=None,
        page=None,
        token_span=TokenSpan(start, end),
    )


def _leaf_without_span(local_id: str) -> ChunkNode:
    """Build a leaf ChunkNode with ``token_span=None`` (simulates un-spanned leaf).

    Args:
        local_id: The leaf identifier string.

    Returns:
        A ChunkNode with ``token_span`` set to ``None``.
    """
    return ChunkNode(
        local_id=local_id,
        parent_local_id="p0",
        ordinal=0,
        content="x",
        content_type="text",
        section_path=None,
        page=None,
        token_span=None,
    )


# ---------------------------------------------------------------------------
# 1. None token_span guard
# ---------------------------------------------------------------------------


async def test_none_token_span_raises_embed_error() -> None:
    """embed_document raises EmbedError naming the leaf when token_span is None."""
    doc = [10, 20, 30]
    leaves = [_leaf_without_span("leaf_no_span")]
    embedder = LateChunkingEmbedder(_ConstantTokenEmbedder())
    with pytest.raises(EmbedError, match="leaf_no_span has no token_span"):
        await embedder.embed_document(doc, leaves)


async def test_none_token_span_error_mentions_local_id() -> None:
    """EmbedError message for None token_span includes the specific leaf local_id."""
    doc = [1, 2, 3, 4, 5]
    leaves = [_leaf_without_span("my_special_leaf")]
    embedder = LateChunkingEmbedder(_ConstantTokenEmbedder())
    with pytest.raises(EmbedError, match="my_special_leaf"):
        await embedder.embed_document(doc, leaves)


async def test_none_token_span_on_second_leaf_raises() -> None:
    """embed_document raises EmbedError when the second leaf (not the first) has None span."""
    doc = list(range(10))
    leaves = [
        _leaf_with_span("good_leaf", 0, 3),
        _leaf_without_span("bad_leaf"),
    ]
    embedder = LateChunkingEmbedder(_ConstantTokenEmbedder())
    with pytest.raises(EmbedError, match="bad_leaf has no token_span"):
        await embedder.embed_document(doc, leaves)


# ---------------------------------------------------------------------------
# 2. Token-vector count mismatch guard
# ---------------------------------------------------------------------------


async def test_mismatch_token_vec_count_raises_embed_error() -> None:
    """embed_document raises EmbedError when the embedder returns the wrong vector count."""
    doc = list(range(5))  # 5 tokens
    leaves = [_leaf_with_span("l0", 0, 3)]
    # The _MismatchTokenEmbedder returns len(window)+1 vectors, triggering the guard.
    embedder = LateChunkingEmbedder(_MismatchTokenEmbedder(extra=1))
    with pytest.raises(
        EmbedError,
        match=r"token embedder returned \d+ vecs for \d+ tokens",
    ):
        await embedder.embed_document(doc, leaves)


async def test_mismatch_error_message_shows_actual_and_expected_counts() -> None:
    """EmbedError for count mismatch includes both the actual and expected token counts."""
    doc = list(range(4))  # 4 tokens → window [0,4) → expects 4 vecs; fake returns 6
    leaves = [_leaf_with_span("l0", 0, 2)]
    embedder = LateChunkingEmbedder(_MismatchTokenEmbedder(extra=2))
    with pytest.raises(EmbedError, match="6 vecs for 4 tokens"):
        await embedder.embed_document(doc, leaves)


# ---------------------------------------------------------------------------
# 3. Empty local span guard
# ---------------------------------------------------------------------------


class _ZeroVecWindowEmbedder:
    """Fake token embedder that returns vectors but whose output slice is empty.

    Produces one vector per token, but always returns the correct count so the
    mismatch guard passes.  The empty-span guard fires when the leaf span slices
    to an empty list (start == end after window offset subtraction).
    """

    async def embed_window(self, token_ids: list[int]) -> list[list[float]]:
        """Return one ``[0.0, 0.0]`` vector per token id."""
        return [[0.0, 0.0] for _ in token_ids]

    async def health(self) -> bool:
        """Always healthy."""
        return True


async def test_empty_span_raises_embed_error() -> None:
    """embed_document raises EmbedError when a leaf's span slices to an empty list.

    A leaf whose TokenSpan has start == end produces an empty ``local`` slice
    after ``token_vecs[span.start - w_start : span.end - w_start]``, which the
    ``if not local`` guard catches.
    """
    doc = list(range(6))  # 6 tokens, fits in one window
    # A zero-length span (start == end) produces an empty slice.
    leaves = [_leaf_with_span("empty_span_leaf", 2, 2)]
    embedder = LateChunkingEmbedder(_ZeroVecWindowEmbedder())
    with pytest.raises(EmbedError, match="empty span for leaf empty_span_leaf"):
        await embedder.embed_document(doc, leaves)


async def test_empty_span_error_message_contains_local_id() -> None:
    """EmbedError for empty span includes the leaf's local_id in the message."""
    doc = list(range(8))
    leaves = [_leaf_with_span("target_leaf", 4, 4)]
    embedder = LateChunkingEmbedder(_ZeroVecWindowEmbedder())
    with pytest.raises(EmbedError, match="target_leaf"):
        await embedder.embed_document(doc, leaves)


# ---------------------------------------------------------------------------
# 4. health() delegation
# ---------------------------------------------------------------------------


async def test_health_returns_true_when_embedder_healthy() -> None:
    """health() returns True when the injected token embedder reports healthy."""
    embedder = LateChunkingEmbedder(_ConstantTokenEmbedder(health_value=True))
    result = await embedder.health()
    assert result is True


async def test_health_returns_false_when_embedder_unhealthy() -> None:
    """health() returns False when the injected token embedder reports unhealthy."""
    embedder = LateChunkingEmbedder(_ConstantTokenEmbedder(health_value=False))
    result = await embedder.health()
    assert result is False


async def test_health_delegates_to_token_embedder() -> None:
    """health() returns exactly the value from the token embedder, not a constant."""

    class _ToggleEmbedder:
        """Token embedder whose health flips on each call."""

        def __init__(self) -> None:
            """Initialise the call counter."""
            self._call_count = 0

        async def embed_window(self, token_ids: list[int]) -> list[list[float]]:
            """Return one vector per token id."""
            return [[0.0] for _ in token_ids]

        async def health(self) -> bool:
            """Toggle True/False on successive calls."""
            self._call_count += 1
            return self._call_count % 2 == 1

    toggle = _ToggleEmbedder()
    embedder = LateChunkingEmbedder(toggle)

    first = await embedder.health()
    second = await embedder.health()
    assert first is True
    assert second is False
