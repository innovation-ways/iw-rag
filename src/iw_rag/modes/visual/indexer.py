"""VisualIndexer — offline batch indexer for the visual late-interaction path (WI-031).

OFFLINE / batch only.  This module MUST NOT be imported by or called from any
hot-path module (api/, agent/, retrieval/, generation/).  The two-clocks rule
(CLAUDE.md, lld-multimodal §3) places all ColVision encoding and visual
indexing in the offline/batch path.

The offline guard test (tests/unit/test_visual_offline_guard.py) statically
asserts that no hot-path module imports this module.

Pipeline (per page, OFFLINE):
    1. ColVisionEncoder.encode_pages([image]) → raw patch vectors (~1000).
    2. pool_to_n(raw, n=pooled_vectors_per_page) → pooled (~38 vectors).
    3. binary_quantize(pooled) → quantized (bit-packed ints).
    4. mean_pool(pooled) → single-vector proxy for stage-1 HNSW.
    5. Store.upsert(...) → writes to visual_pages (no raw multi-vectors stored).

References:
- task-4b-visual-rag-route §4 (VisualIndexer)
- lld-multimodal §4 (ColVision encoder + transient loading)
- ADR-010 (no raw multi-vectors)
- R-00107 F3 (pooled_vectors_per_page≈38 + binary quantization)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from iw_rag.modes.visual.encoder import ColVisionEncoder
from iw_rag.modes.visual.math_utils import binary_quantize, mean_pool, pool_to_n
from iw_rag.modes.visual.store import MultiVectorStore
from iw_rag.modes.visual.types import VisualIndexConfig, VisualPageRecord

_log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class VisualIndexReport:
    """The outcome of a VisualIndexer.index_pages call.

    Attributes:
        pages_indexed: Number of pages successfully encoded and stored.
        pages_failed: Number of pages that failed encoding (skipped with log).
        bytes_stored: Estimated bytes written to the multi-vector store.
    """

    pages_indexed: int
    pages_failed: int
    bytes_stored: int


class VisualIndexer:
    """Offline encoder → pool → quantize → upsert pipeline.

    OFFLINE / batch only — called by ``scripts/index_visual.py`` (late-interaction
    mode).  Never imported by hot-path modules (api/, agent/, retrieval/).

    The encoder is transient: loaded per-batch and released.  The store is
    injected (so tests supply a fake).

    Attributes:
        encoder: ColVisionEncoder (or a test fake).
        store: MultiVectorStore (or a test fake).
        cfg: VisualIndexConfig controlling pooling depth and backend.
    """

    def __init__(
        self, encoder: ColVisionEncoder, store: MultiVectorStore, cfg: VisualIndexConfig
    ) -> None:
        """Wire the indexer with an encoder, store, and config.

        Args:
            encoder: ColVisionEncoder (or a test fake) with ``encode_pages``.
            store: MultiVectorStore (or a test fake) with ``upsert``.
            cfg: VisualIndexConfig controlling ``pooled_vectors_per_page``, etc.
        """
        self.encoder = encoder
        self.store = store
        self.cfg = cfg

    async def index_pages(self, pages: list[VisualPageRecord]) -> VisualIndexReport:
        """Encode, pool, quantize, and store a list of visual pages.

        For each page:
        1. Encode with ColVisionEncoder.encode_pages (transient; ~1000 patches).
        2. Pool to ~38 vectors via pool_to_n.
        3. Binary-quantize the pooled vectors.
        4. Compute mean-pool proxy for stage-1 HNSW.
        5. Upsert into the multi-vector store (no raw vectors stored).

        On encoder failure for a single page: log a warning and skip (skip-and-log
        pattern; the page is absent from the visual index but does not block others).

        Args:
            pages: The visual pages to index (produced by the ingestion pipeline).

        Returns:
            A :class:`VisualIndexReport` with counts and estimated bytes stored.
        """
        pages_indexed = 0
        pages_failed = 0
        bytes_stored_estimate = 0

        for page in pages:
            try:
                # Step 1: encode (transient encoder call)
                raw_vecs: list[list[list[float]]] = await self.encoder.encode_pages([page.image])
                if not raw_vecs or not raw_vecs[0]:
                    _log.warning(
                        "VisualIndexer: encode_pages returned empty for page=%d uri=%r — skipping",
                        page.page,
                        page.source_uri,
                    )
                    pages_failed += 1
                    continue

                patch_vecs = raw_vecs[0]  # first (and only) image's patches

                # Step 2: pool
                pooled = pool_to_n(patch_vecs, self.cfg.pooled_vectors_per_page)

                # Step 3: binary-quantize
                quantized = binary_quantize(pooled)

                # Step 4: mean-pool proxy
                proxy = mean_pool(pooled)

                # Step 5: upsert (no raw multi-vectors stored — ADR-010)
                page_image_ref = f"{page.source_uri}#page={page.page}"
                await self.store.upsert(
                    chunk_id=page.source_chunk_id,
                    source_uri=page.source_uri,
                    page=page.page,
                    content=page.text,
                    acl_tags=list(page.acl_tags),
                    pooled_vectors=pooled,
                    quantized_vectors=quantized,
                    proxy_vector=proxy,
                    bbox=dict(page.bbox),
                    page_image_ref=page_image_ref,
                    metadata=dict(page.metadata),
                )

                # Rough size estimate: pooled × dims × 4 bytes (float32)
                if pooled:
                    dims = len(pooled[0])
                    bytes_stored_estimate += len(pooled) * dims * 4

                pages_indexed += 1
                _log.debug(
                    "VisualIndexer: indexed page=%d uri=%r (pooled=%d vectors)",
                    page.page,
                    page.source_uri,
                    len(pooled),
                )

            except Exception as exc:  # noqa: BLE001
                _log.warning(
                    "VisualIndexer: failed to index page=%d uri=%r: %s — skipping",
                    page.page,
                    page.source_uri,
                    exc,
                )
                pages_failed += 1

        return VisualIndexReport(
            pages_indexed=pages_indexed,
            pages_failed=pages_failed,
            bytes_stored=bytes_stored_estimate,
        )
