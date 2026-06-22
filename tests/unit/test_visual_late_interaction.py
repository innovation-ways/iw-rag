"""Unit tests for the visual late-interaction path (WI-031 / Task 4b, steps 3–8).

Tests the pure deterministic helpers:
- pool_to_n: pooling reduction (~1000 → ~38)
- binary_quantize: round-trip
- maxsim: scoring over a deterministic tiny matrix
- RetrievalResult adapter: page → RetrievedChunk with correct fields
- ACL post-filter predicate
- VisualIndexReport structure

These are all pure (no I/O) and exhaustively unit-tested per the TDD
requirement (RED → GREEN → REFACTOR).

Two-clocks rule checks:
- The indexer (VisualIndexer) is OFFLINE-only.
- The encoder (ColVisionEncoder) is used by the retriever on the hot path
  for encode_query BUT the indexer is offline.
- We test the retriever contract independently (it DOES use encoder).
"""

from __future__ import annotations

import pytest

from iw_rag.domain.models import RetrievalResult, RetrievedChunk

# ---------------------------------------------------------------------------
# Helpers imported after the module exists (RED: these will fail until
# the module is created)
# ---------------------------------------------------------------------------


def test_pool_to_n_reduces_rows_to_target() -> None:
    """pool_to_n maps a tall matrix to exactly pooled_vectors_per_page rows."""
    # A tall matrix: 1000 patch vectors of 128 dims
    import random

    from iw_rag.modes.visual.math_utils import pool_to_n

    random.seed(42)
    big = [[random.random() for _ in range(128)] for _ in range(1000)]
    result = pool_to_n(big, n=38)
    assert len(result) == 38, f"Expected 38 rows, got {len(result)}"
    assert all(len(row) == 128 for row in result), "All pooled rows must have 128 dims"


def test_pool_to_n_handles_small_input_exactly() -> None:
    """pool_to_n returns the input unchanged when len(input) <= n."""
    from iw_rag.modes.visual.math_utils import pool_to_n

    small = [[1.0, 2.0, 3.0]] * 10
    result = pool_to_n(small, n=38)
    assert len(result) == 10  # fewer than n → returned as-is (no padding)


def test_pool_to_n_returns_exactly_n_when_equal() -> None:
    """pool_to_n with len(input)==n returns all rows unchanged."""
    from iw_rag.modes.visual.math_utils import pool_to_n

    exact = [[float(i)] * 4 for i in range(38)]
    result = pool_to_n(exact, n=38)
    assert len(result) == 38


def test_binary_quantize_produces_bits_list() -> None:
    """binary_quantize converts float matrix to list of bit-packed ints."""
    from iw_rag.modes.visual.math_utils import binary_quantize

    vecs = [[0.5, -0.1, 0.9, -0.3]] * 5
    result = binary_quantize(vecs)
    # result should have same number of rows
    assert len(result) == 5, f"Expected 5 rows, got {len(result)}"
    # Each row should be a list of integers
    for row in result:
        assert isinstance(row, list), "Each quantized row must be a list"
        for val in row:
            assert isinstance(val, int), f"Each quantized value must be int, got {type(val)}"


def test_binary_quantize_round_trip() -> None:
    """binary_quantize sign-encodes: positive → 1, negative → 0."""
    from iw_rag.modes.visual.math_utils import binary_quantize

    # Positive values → 1 bits; negative → 0 bits
    vecs = [[1.0, -1.0, 2.0, -2.0]]  # dims=4: [1, 0, 1, 0]
    result = binary_quantize(vecs)
    # 4 dims packed into bits: 1010 in MSB-first = 0b1010 = 10
    # Alternatively if dims packed per byte: first byte = 10100000 = 160 or 0b1010 = 10
    # The implementation should be consistent — we just verify round-trip sign consistency
    assert len(result) == 1
    # Reconstruct signs from the quantized output via dequantize
    from iw_rag.modes.visual.math_utils import dequantize_signs

    signs = dequantize_signs(result[0], n_dims=4)
    expected = [1, 0, 1, 0]
    assert signs == expected, f"Round-trip sign mismatch: {signs} != {expected}"


def test_maxsim_over_deterministic_matrix() -> None:
    """MaxSim over a 2×3 query × 4-patch page matrix is computed correctly.

    MaxSim = sum over query tokens of max over patch tokens of dot-product.
    """
    from iw_rag.modes.visual.math_utils import maxsim

    # query_vecs: 2 tokens × 3 dims
    # patch_vecs: 4 patches × 3 dims
    query_vecs = [
        [1.0, 0.0, 0.0],  # token 0 → aligns with patch 0
        [0.0, 1.0, 0.0],  # token 1 → aligns with patch 1
    ]
    patch_vecs = [
        [1.0, 0.0, 0.0],  # patch 0: dot q0=1, q1=0 → max=1
        [0.0, 1.0, 0.0],  # patch 1: dot q0=0, q1=1 → max=1
        [0.5, 0.5, 0.0],  # patch 2: dot q0=0.5, q1=0.5
        [0.0, 0.0, 1.0],  # patch 3: dot q0=0, q1=0
    ]
    # MaxSim = max-over-patches-for-each-token, then sum
    # token 0: max(1, 0, 0.5, 0) = 1.0
    # token 1: max(0, 1, 0.5, 0) = 1.0
    # total = 2.0
    score = maxsim(query_vecs, patch_vecs)
    assert abs(score - 2.0) < 1e-6, f"Expected MaxSim=2.0, got {score}"


def test_maxsim_zero_when_orthogonal() -> None:
    """MaxSim is 0 when query and patch vectors are all orthogonal."""
    from iw_rag.modes.visual.math_utils import maxsim

    query_vecs = [[1.0, 0.0]]
    patch_vecs = [[0.0, 1.0]]
    score = maxsim(query_vecs, patch_vecs)
    assert abs(score) < 1e-6, f"Expected MaxSim=0, got {score}"


def test_maxsim_with_heatmap_returns_matrix() -> None:
    """maxsim with return_heatmap=True returns (score, heatmap_matrix)."""
    from iw_rag.modes.visual.math_utils import maxsim

    query_vecs = [[1.0, 0.0], [0.0, 1.0]]
    patch_vecs = [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]
    result = maxsim(query_vecs, patch_vecs, return_heatmap=True)
    assert isinstance(result, tuple), "With return_heatmap, maxsim must return (score, matrix)"
    score, heatmap = result
    assert isinstance(score, float)
    # heatmap: n_query × n_patch matrix
    assert len(heatmap) == 2  # 2 query tokens
    assert len(heatmap[0]) == 3  # 3 patches


def test_mean_pool_reduces_to_single_vector() -> None:
    """mean_pool over N rows yields one row of the same dims."""
    from iw_rag.modes.visual.math_utils import mean_pool

    vecs = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    result = mean_pool(vecs)
    assert len(result) == 2  # same dims
    assert abs(result[0] - 3.0) < 1e-6  # mean of [1,3,5]=3
    assert abs(result[1] - 4.0) < 1e-6  # mean of [2,4,6]=4


# ---------------------------------------------------------------------------
# RetrievalResult adapter tests
# ---------------------------------------------------------------------------


def _make_page_record(
    *,
    chunk_id: str = "cid-1",
    content: str = "page text content",
    source_uri: str = "file:///doc.pdf",
    page: int = 1,
    acl_tags: list[str] | None = None,
    score: float = 0.85,
    bbox: dict[str, float] | None = None,
) -> dict[str, object]:
    """Build a fake page record as the store would return."""
    return {
        "chunk_id": chunk_id,
        "content": content,
        "source_uri": source_uri,
        "page": page,
        "acl_tags": acl_tags if acl_tags is not None else ["public"],
        "score": score,
        "bbox": bbox or {"x0": 0.0, "y0": 0.0, "x1": 1.0, "y1": 1.0},
        "page_image_ref": f"{source_uri}#page={page}",
    }


def test_retrieval_result_adapter_basic_fields() -> None:
    """Page record → RetrievedChunk has correct content, source_uri, score."""
    from iw_rag.modes.visual.retriever import _adapt_page_to_chunk

    page = _make_page_record(content="page text", source_uri="file:///a.pdf", score=0.7)
    chunk = _adapt_page_to_chunk(page, with_heatmap=False, heatmap=None)

    assert isinstance(chunk, RetrievedChunk)
    assert chunk.content == "page text"
    assert chunk.source_uri == "file:///a.pdf"
    assert abs(chunk.score - 0.7) < 1e-6
    assert chunk.retriever == "visual"  # type: ignore[attr-defined]
    assert chunk.acl_applied is True  # type: ignore[attr-defined]


def test_retrieval_result_adapter_has_page_image_ref_in_metadata() -> None:
    """The page_image_ref is carried in metadata with the correct URI#page= format."""
    from iw_rag.modes.visual.retriever import _adapt_page_to_chunk

    page = _make_page_record(source_uri="file:///doc.pdf", page=3)
    chunk = _adapt_page_to_chunk(page, with_heatmap=False, heatmap=None)
    assert "page_image_ref" in chunk.metadata, "page_image_ref must be present in chunk metadata"
    assert chunk.metadata["page_image_ref"] == "file:///doc.pdf#page=3", (
        f"page_image_ref must be 'file:///doc.pdf#page=3', got {chunk.metadata['page_image_ref']!r}"
    )


def test_retrieval_result_adapter_bbox_in_metadata() -> None:
    """Bounding box is carried in metadata."""
    from iw_rag.modes.visual.retriever import _adapt_page_to_chunk

    bbox = {"x0": 0.1, "y0": 0.2, "x1": 0.9, "y1": 0.8}
    page = _make_page_record(bbox=bbox)
    chunk = _adapt_page_to_chunk(page, with_heatmap=False, heatmap=None)
    assert chunk.metadata.get("bbox") == bbox


def test_retrieval_result_adapter_no_heatmap_when_not_requested() -> None:
    """heatmap is not in metadata when with_heatmap=False."""
    from iw_rag.modes.visual.retriever import _adapt_page_to_chunk

    page = _make_page_record()
    chunk = _adapt_page_to_chunk(page, with_heatmap=False, heatmap=[[0.5]])
    assert "maxsim_heatmap" not in chunk.metadata


def test_retrieval_result_adapter_heatmap_in_metadata_when_requested() -> None:
    """heatmap IS in metadata when with_heatmap=True."""
    from iw_rag.modes.visual.retriever import _adapt_page_to_chunk

    page = _make_page_record()
    heatmap = [[0.9, 0.1], [0.3, 0.7]]
    chunk = _adapt_page_to_chunk(page, with_heatmap=True, heatmap=heatmap)
    assert chunk.metadata.get("maxsim_heatmap") == heatmap


def test_retrieval_result_adapter_retriever_field() -> None:
    """The retriever discriminator is 'visual' on every adapted chunk."""
    from iw_rag.modes.visual.retriever import _adapt_page_to_chunk

    page = _make_page_record()
    chunk = _adapt_page_to_chunk(page, with_heatmap=False, heatmap=None)
    assert chunk.retriever == "visual"  # type: ignore[attr-defined]


def test_retrieval_result_adapter_no_text_flag_when_content_empty() -> None:
    """When the page has no text, no_text_for_generation flag is set in metadata."""
    from iw_rag.modes.visual.retriever import _adapt_page_to_chunk

    page = _make_page_record(content="")
    chunk = _adapt_page_to_chunk(page, with_heatmap=False, heatmap=None)
    assert chunk.metadata.get("no_text_for_generation") is True


# ---------------------------------------------------------------------------
# ACL post-filter predicate tests
# ---------------------------------------------------------------------------


def test_acl_postfilter_allows_when_tag_in_principal_groups() -> None:
    """ACL post-filter passes when page acl_tags ∩ principal_groups is non-empty."""
    from iw_rag.modes.visual.retriever import _acl_postfilter

    principal_groups = ["finance", "all-staff"]
    page_acl_tags = ["finance"]
    assert _acl_postfilter(page_acl_tags, principal_groups) is True


def test_acl_postfilter_blocks_when_no_overlap() -> None:
    """ACL post-filter drops when no tag matches."""
    from iw_rag.modes.visual.retriever import _acl_postfilter

    principal_groups = ["hr"]
    page_acl_tags = ["finance"]
    assert _acl_postfilter(page_acl_tags, principal_groups) is False


def test_acl_postfilter_allows_when_acl_tags_empty() -> None:
    """An empty acl_tags list means the page is publicly readable — allow."""
    from iw_rag.modes.visual.retriever import _acl_postfilter

    principal_groups = ["some-group"]
    page_acl_tags: list[str] = []
    assert _acl_postfilter(page_acl_tags, principal_groups) is True


def test_acl_postfilter_blocks_when_principal_groups_empty_and_tags_set() -> None:
    """No matching groups when principal_groups is empty and page has restricted tags."""
    from iw_rag.modes.visual.retriever import _acl_postfilter

    principal_groups: list[str] = []
    page_acl_tags = ["restricted"]
    assert _acl_postfilter(page_acl_tags, principal_groups) is False


# ---------------------------------------------------------------------------
# Contract test: VisualRetriever produces a valid RetrievalResult
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_visual_retriever_contract_returns_retrieval_result() -> None:
    """VisualRetriever.retrieve returns a structurally valid RetrievalResult.

    The generator + grade_generation accept it unchanged (mode-agnostic).
    Uses a fully mocked encoder and store so no GPU/network is needed.
    """
    from unittest.mock import AsyncMock, MagicMock

    from iw_rag.domain.acl import AccessContext
    from iw_rag.modes.visual.retriever import VisualRetriever, VisualRetrieverConfig

    # Fake encoder: encode_query returns a list of 3 vectors of dim 4
    fake_encoder = MagicMock()
    fake_encoder.encode_query = AsyncMock(
        return_value=[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
    )

    # Fake store: stage1_prefetch returns 2 candidate pages; stage2_maxsim returns the same
    fake_page_1 = {
        "chunk_id": "c1",
        "content": "page one text",
        "source_uri": "file:///doc.pdf",
        "page": 1,
        "acl_tags": ["public"],
        "score": 0.9,
        "bbox": {"x0": 0.0, "y0": 0.0, "x1": 1.0, "y1": 1.0},
        "page_image_ref": "file:///doc.pdf#page=1",
        "pooled_vectors": [[1.0, 0.0, 0.0, 0.0]],
    }
    fake_page_2 = {
        "chunk_id": "c2",
        "content": "page two text",
        "source_uri": "file:///doc.pdf",
        "page": 2,
        "acl_tags": ["public"],
        "score": 0.7,
        "bbox": {"x0": 0.0, "y0": 0.0, "x1": 1.0, "y1": 1.0},
        "page_image_ref": "file:///doc.pdf#page=2",
        "pooled_vectors": [[0.0, 1.0, 0.0, 0.0]],
    }
    fake_store = MagicMock()
    fake_store.stage1_prefetch = AsyncMock(return_value=[fake_page_1, fake_page_2])

    cfg = VisualRetrieverConfig(top_k=5, prefetch_n=200, pooled_vectors_per_page=38)
    retriever = VisualRetriever(encoder=fake_encoder, store=fake_store, cfg=cfg)

    ctx = AccessContext(principal_id="user1", principal_groups=["public"])
    result = await retriever.retrieve("what is in the chart?", ctx)

    # Must be a valid RetrievalResult
    assert isinstance(result, RetrievalResult)
    assert result.retriever == "visual"
    assert result.acl_applied is True
    assert isinstance(result.chunks, list)
    for chunk in result.chunks:
        assert isinstance(chunk, RetrievedChunk)
        assert chunk.retriever == "visual"  # type: ignore[attr-defined]
        assert chunk.acl_applied is True  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_visual_retriever_acl_drops_restricted_pages() -> None:
    """ACL-restricted pages are dropped from the result; no leak."""
    from unittest.mock import AsyncMock, MagicMock

    from iw_rag.domain.acl import AccessContext
    from iw_rag.modes.visual.retriever import VisualRetriever, VisualRetrieverConfig

    fake_encoder = MagicMock()
    fake_encoder.encode_query = AsyncMock(return_value=[[1.0, 0.0]])

    # One public page, one finance-only page
    public_page = {
        "chunk_id": "pub",
        "content": "public content",
        "source_uri": "file:///pub.pdf",
        "page": 1,
        "acl_tags": ["public"],
        "score": 0.8,
        "bbox": {},
        "page_image_ref": "file:///pub.pdf#1",
        "pooled_vectors": [[1.0, 0.0]],
    }
    restricted_page = {
        "chunk_id": "fin",
        "content": "finance only content",
        "source_uri": "file:///finance.pdf",
        "page": 1,
        "acl_tags": ["finance"],
        "score": 0.9,
        "bbox": {},
        "page_image_ref": "file:///finance.pdf#1",
        "pooled_vectors": [[0.0, 1.0]],
    }
    fake_store = MagicMock()
    fake_store.stage1_prefetch = AsyncMock(return_value=[public_page, restricted_page])

    cfg = VisualRetrieverConfig(top_k=5, prefetch_n=200, pooled_vectors_per_page=38)
    retriever = VisualRetriever(encoder=fake_encoder, store=fake_store, cfg=cfg)

    # User only has "public" access
    ctx = AccessContext(principal_id="user1", principal_groups=["public"])
    result = await retriever.retrieve("show me finance", ctx)

    # Only the public page should survive
    chunk_ids = [c.chunk_id for c in result.chunks]
    assert "fin" not in chunk_ids, "Finance-restricted page must be dropped"
    assert "pub" in chunk_ids, "Public page must remain"


@pytest.mark.asyncio
async def test_visual_retriever_fallback_on_encoder_error() -> None:
    """Encoder OOM/error → VisualEncodeUnavailable is raised."""
    from unittest.mock import AsyncMock, MagicMock

    from iw_rag.domain.acl import AccessContext
    from iw_rag.modes.visual.errors import VisualEncodeUnavailable
    from iw_rag.modes.visual.retriever import VisualRetriever, VisualRetrieverConfig

    fake_encoder = MagicMock()
    fake_encoder.encode_query = AsyncMock(side_effect=RuntimeError("CUDA OOM"))
    fake_store = MagicMock()

    cfg = VisualRetrieverConfig(top_k=5, prefetch_n=200, pooled_vectors_per_page=38)
    retriever = VisualRetriever(encoder=fake_encoder, store=fake_store, cfg=cfg)

    ctx = AccessContext(principal_id="u", principal_groups=["public"])
    with pytest.raises(VisualEncodeUnavailable, match="encode_query failed"):
        await retriever.retrieve("test", ctx)


# ---------------------------------------------------------------------------
# VisualIndexReport structure
# ---------------------------------------------------------------------------


def test_visual_index_report_has_required_fields() -> None:
    """VisualIndexReport carries pages_indexed, pages_failed, bytes_stored."""
    from iw_rag.modes.visual.indexer import VisualIndexReport

    report = VisualIndexReport(pages_indexed=10, pages_failed=1, bytes_stored=4096)
    assert report.pages_indexed == 10
    assert report.pages_failed == 1
    assert report.bytes_stored == 4096
