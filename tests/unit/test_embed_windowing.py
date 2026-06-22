"""Unit tests for Late-Chunking windowing + pooling (fake token embedder)."""

from __future__ import annotations

import math

import pytest

from iw_rag.domain.errors import EmbedError
from iw_rag.ingestion.chunk import ChunkNode, TokenSpan
from iw_rag.ingestion.embed import (
    EmbedConfig,
    LateChunkingEmbedder,
    plan_windows,
)


class FakeTokenEmbedder:
    """Returns a deterministic 2-d vector per token and records window sizes."""

    def __init__(self) -> None:
        self.calls: list[int] = []

    async def embed_window(self, token_ids: list[int]) -> list[list[float]]:
        self.calls.append(len(token_ids))
        # token id -> [id, 1.0]
        return [[float(tid), 1.0] for tid in token_ids]

    async def health(self) -> bool:
        return True


def _leaf(local_id: str, start: int, end: int) -> ChunkNode:
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


def test_plan_windows_single_when_fits() -> None:
    assert plan_windows(100, window=8192, overlap=512) == [(0, 100)]


def test_plan_windows_overlapping_cover_all() -> None:
    windows = plan_windows(25, window=10, overlap=4)  # step 6
    assert windows[0] == (0, 10)
    assert windows[-1][1] == 25
    # consecutive windows overlap by `overlap`
    assert windows[1][0] == 6


async def test_pooling_is_mean_then_normalized() -> None:
    # one window; leaf spans tokens [0,2) → ids 10,20 → mean [15,1] → normalized
    doc = [10, 20, 30, 40]
    leaves = [_leaf("l0", 0, 2)]
    out = await LateChunkingEmbedder(FakeTokenEmbedder()).embed_document(doc, leaves)
    assert len(out) == 1
    norm = math.sqrt(15 * 15 + 1 * 1)
    assert out[0].dense == pytest.approx([15 / norm, 1 / norm])
    assert out[0].local_id == "l0"


async def test_each_leaf_embedded_once_across_windows() -> None:
    # doc of 25 tokens; window 10 overlap 4 → windows (0,10),(6,16),(12,22),(18,25)
    doc = list(range(25))
    leaves = [_leaf("a", 0, 3), _leaf("b", 7, 9), _leaf("c", 19, 22)]
    fake = FakeTokenEmbedder()
    out = await LateChunkingEmbedder(
        fake,
        EmbedConfig(window_tokens=10, overlap_tokens=4, normalize=False),
    ).embed_document(doc, leaves)
    assert {e.local_id for e in out} == {"a", "b", "c"}
    # leaf 'a' pooled from window0: mean of ids 0,1,2 = 1.0
    a = next(e for e in out if e.local_id == "a")
    assert a.dense == pytest.approx([1.0, 1.0])
    # only windows that actually contain assigned leaves are embedded (≤ leaves)
    assert len(fake.calls) <= 3


async def test_uncontainable_span_raises() -> None:
    # overlap smaller than leaf size → a boundary-straddling leaf can't fit.
    # windows (window=10, overlap=2, step=8): (0,10),(8,18),(16,20).
    # span (5,12) fits none: (0,10) ends too early, (8,18) starts too late.
    doc = list(range(20))
    leaves = [_leaf("big", 5, 12)]
    with pytest.raises(EmbedError, match="not fully contained in any window"):
        await LateChunkingEmbedder(
            FakeTokenEmbedder(),
            EmbedConfig(window_tokens=10, overlap_tokens=2, normalize=False),
        ).embed_document(doc, leaves)
