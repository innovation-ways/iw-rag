"""Multi-vector store abstraction for the visual late-interaction path (WI-031).

Provides the ``MultiVectorStore`` protocol and two concrete implementations:

- ``PostgresCandidateStore`` (DEFAULT): stage-1 candidates via mean-pooled
  single-vector pgvector HNSW proxy; stage-2 MaxSim computed in-process.
  Preserves the single-store/single-backup audit story (ADR-007 §1).

- ``QdrantMultiVectorStore`` (SEAM/STUB): native multi-vector + quantization +
  pooled-prefetch.  Added ONLY when the visual demo is promoted to headline
  (the T1 trigger, ADR-007/ADR-010).  Guarded import: if ``qdrant_client`` is
  not installed, the class still exists as a seam that fails fast at startup
  with a clear ``ImportError``.

Two-clocks rule: the store is used by BOTH the offline VisualIndexer (upsert)
and the online VisualRetriever (prefetch/query).  The same PostgresCandidateStore
instance serves both clocks; the session factory is injected so tests remain
database-free.

D18 [DECISION]: The default backend is ``postgres_candidates`` because it
preserves the single-store invariant.  ``backend=qdrant`` requires the T1
trigger (headline promotion) and fails fast at startup if Qdrant is unreachable.

Open item (task-4b Risks §2): ``visual_pages.acl_tags`` enables future stage-1
pre-filtering in Postgres.  Until that index is used, the ACL post-filter in
VisualRetriever is the enforcement seam (late-interaction path is restricted
to uniformly-readable demo corpora until pre-filter is wired).

References:
- task-4b-visual-rag-route §5 (store abstraction, T1 trigger)
- lld-data-model §8 (visual_pages table, Qdrant T1 trigger)
- ADR-007 (Postgres-first + Qdrant scoped trigger)
- ADR-010 (no raw multi-vectors, pooled+binary storage)
"""

from __future__ import annotations

import logging
from typing import Any, Protocol, runtime_checkable

_log = logging.getLogger(__name__)


@runtime_checkable
class MultiVectorStore(Protocol):
    """Protocol for the visual multi-vector store.

    Both ``PostgresCandidateStore`` and ``QdrantMultiVectorStore`` implement
    this protocol; ``VisualRetriever`` is backend-agnostic (CONVENTIONS.md §4).
    """

    async def upsert(
        self,
        *,
        chunk_id: str,
        source_uri: str,
        page: int,
        content: str,
        acl_tags: list[str],
        pooled_vectors: list[list[float]],
        quantized_vectors: list[list[int]],
        proxy_vector: list[float],
        bbox: dict[str, float],
        page_image_ref: str,
        metadata: dict[str, Any],
    ) -> None:
        """Upsert a page's pooled+quantized multi-vectors into the store.

        Args:
            chunk_id: The source ``chunks.id`` back-reference.
            source_uri: The source document URI.
            page: 1-based page number within the document.
            content: The extracted page text (used as RetrievedChunk.content).
            acl_tags: ACL entitlement tags (from source manifest, never LLM).
            pooled_vectors: Pooled multi-vectors (shape ``[~38, dims]``).
            quantized_vectors: Binary-quantized pooled vectors (shape ``[~38, n_bytes]``).
            proxy_vector: Mean-pooled single-vector proxy for stage-1 HNSW.
            bbox: Page bounding box in PDF coordinate system.
            page_image_ref: Reference to the rendered page image.
            metadata: Free-form per-page metadata.
        """
        ...

    async def stage1_prefetch(
        self,
        query_proxy: list[float],
        *,
        prefetch_n: int = 200,
    ) -> list[dict[str, Any]]:
        """Stage-1 cheap prefetch: ANN over the mean-pooled proxy.

        Returns the top-``prefetch_n`` candidate page records (including their
        stored ``pooled_vectors`` for stage-2 MaxSim).

        Args:
            query_proxy: The mean-pooled single-vector proxy for the query.
            prefetch_n: Number of candidates to prefetch.

        Returns:
            List of candidate page dicts, each containing at least:
            ``chunk_id``, ``content``, ``source_uri``, ``page``, ``acl_tags``,
            ``score`` (proxy similarity), ``pooled_vectors``, ``bbox``,
            ``page_image_ref``.
        """
        ...


class PostgresCandidateStore:
    """Default multi-vector store: pgvector HNSW proxy + in-process MaxSim.

    Stage-1 uses a mean-pooled single-vector HNSW index over ``visual_pages``
    to cheaply retrieve ``prefetch_n`` candidates.  Stage-2 MaxSim is computed
    in-process over the candidates' stored pooled multi-vectors.

    This preserves the single-store/single-backup story while paying the cost
    of in-process MaxSim (acceptable at showcase scale per ADR-010 §2).

    D18 [DECISION — Postgres]: No native MaxSim index in Postgres. The in-process
    MaxSim over the prefetch set (~200 pages) is CPU-bound and fast at demo scale.
    At headline scale, the T1 trigger fires (QdrantMultiVectorStore).

    Attributes:
        session_factory: An async context-manager factory yielding an AsyncSession.
    """

    def __init__(self, session_factory: Any) -> None:
        """Wire the store with a session factory.

        Args:
            session_factory: An async context-manager factory (e.g.
                ``db.session``) yielding an ``AsyncSession``.
        """
        self._session_factory = session_factory

    async def upsert(
        self,
        *,
        chunk_id: str,
        source_uri: str,
        page: int,
        content: str,
        acl_tags: list[str],
        pooled_vectors: list[list[float]],
        quantized_vectors: list[list[int]],
        proxy_vector: list[float],
        bbox: dict[str, float],
        page_image_ref: str,
        metadata: dict[str, Any],
    ) -> None:
        """Upsert a page record into the ``visual_pages`` table.

        Uses ``ON CONFLICT (chunk_id, page) DO UPDATE`` for idempotency.

        Args:
            chunk_id: The source ``chunks.id`` back-reference.
            source_uri: The source document URI.
            page: 1-based page number within the document.
            content: The extracted page text.
            acl_tags: ACL entitlement tags.
            pooled_vectors: Pooled multi-vectors (stored as JSONB).
            quantized_vectors: Binary-quantized pooled vectors (stored as JSONB).
            proxy_vector: Mean-pooled proxy vector (stored as pgvector).
            bbox: Page bounding box.
            page_image_ref: Page image reference.
            metadata: Free-form per-page metadata.
        """
        from sqlalchemy import text as sql_text

        # Stored as JSONB since pgvector multi-vector is not natively supported.
        # The proxy_vector is stored as pgvector for HNSW indexing.
        proxy_str = "[" + ",".join(str(v) for v in proxy_vector) + "]"

        sql = sql_text(
            """
            INSERT INTO iw_rag.visual_pages
                (chunk_id, source_uri, page, content, acl_tags,
                 pooled_vectors, quantized_vectors, proxy_vector,
                 bbox, page_image_ref, metadata)
            VALUES
                (:chunk_id, :source_uri, :page, :content, :acl_tags,
                 CAST(:pooled_vectors AS jsonb), CAST(:quantized_vectors AS jsonb),
                 CAST(:proxy_vector AS vector),
                 CAST(:bbox AS jsonb), :page_image_ref, CAST(:metadata AS jsonb))
            ON CONFLICT (chunk_id, page) DO UPDATE SET
                content = EXCLUDED.content,
                acl_tags = EXCLUDED.acl_tags,
                pooled_vectors = EXCLUDED.pooled_vectors,
                quantized_vectors = EXCLUDED.quantized_vectors,
                proxy_vector = EXCLUDED.proxy_vector,
                bbox = EXCLUDED.bbox,
                page_image_ref = EXCLUDED.page_image_ref,
                metadata = EXCLUDED.metadata
            """
        )

        import json

        async with self._session_factory() as session:
            await session.execute(
                sql,
                {
                    "chunk_id": chunk_id,
                    "source_uri": source_uri,
                    "page": page,
                    "content": content,
                    "acl_tags": acl_tags,
                    "pooled_vectors": json.dumps(pooled_vectors),
                    "quantized_vectors": json.dumps(quantized_vectors),
                    "proxy_vector": proxy_str,
                    "bbox": json.dumps(bbox),
                    "page_image_ref": page_image_ref,
                    "metadata": json.dumps(metadata),
                },
            )
            await session.commit()

        _log.debug(
            "PostgresCandidateStore.upsert: chunk_id=%r page=%d source_uri=%r",
            chunk_id,
            page,
            source_uri,
        )

    async def stage1_prefetch(
        self,
        query_proxy: list[float],
        *,
        prefetch_n: int = 200,
    ) -> list[dict[str, Any]]:
        """Stage-1 ANN over the mean-pooled proxy vector (pgvector HNSW).

        Retrieves up to ``prefetch_n`` candidate pages using cosine similarity
        over the proxy vector column.

        Args:
            query_proxy: Mean-pooled query proxy vector.
            prefetch_n: Maximum candidates to return.

        Returns:
            List of candidate dicts (see protocol docstring).
        """
        import json

        from sqlalchemy import text as sql_text

        proxy_str = "[" + ",".join(str(v) for v in query_proxy) + "]"
        sql = sql_text(
            """
            SELECT
                chunk_id,
                source_uri,
                page,
                content,
                acl_tags,
                pooled_vectors,
                bbox,
                page_image_ref,
                1 - (proxy_vector <=> CAST(:query_vector AS vector)) AS score
            FROM iw_rag.visual_pages
            ORDER BY proxy_vector <=> CAST(:query_vector AS vector)
            LIMIT :n
            """
        )
        candidates: list[dict[str, Any]] = []
        async with self._session_factory() as session:
            result = await session.execute(sql, {"query_vector": proxy_str, "n": prefetch_n})
            for row in result.all():
                pooled = (
                    json.loads(row.pooled_vectors)
                    if isinstance(row.pooled_vectors, str)
                    else row.pooled_vectors
                )
                candidates.append(
                    {
                        "chunk_id": row.chunk_id,
                        "source_uri": row.source_uri,
                        "page": row.page,
                        "content": row.content,
                        "acl_tags": list(row.acl_tags),
                        "score": float(row.score),
                        "pooled_vectors": pooled,
                        "bbox": json.loads(row.bbox) if isinstance(row.bbox, str) else row.bbox,
                        "page_image_ref": row.page_image_ref,
                    }
                )
        return candidates


class QdrantMultiVectorStore:
    """Qdrant native multi-vector store — the headline T1 trigger (ADR-007).

    SEAM/STUB: Only instantiated when the visual demo is promoted to headline.
    Fails fast at startup if ``qdrant_client`` is not installed or Qdrant is
    unreachable.

    D18 [DECISION — Qdrant]: Adding Qdrant is the ONLY sanctioned reason for a
    second store in this architecture.  It is NEVER added as a convenience
    feature — only when the visual retrieval demo is headline and the in-process
    MaxSim in PostgresCandidateStore is insufficient (R-00104/T1, ADR-007/ADR-010).

    The import is guarded so the qdrant_client package remains optional — it is
    NOT a hard dependency of iw-rag.
    """

    def __init__(self, url: str, collection_name: str = "visual_pages") -> None:
        """Initialise the Qdrant store.

        Attempts to connect to the Qdrant server at ``url``.  Raises
        ``VisualStoreUnavailable`` if the server is unreachable or the package
        is not installed (fail-fast-at-startup semantics, ADR-007).

        Args:
            url: The Qdrant server URL.
            collection_name: The Qdrant collection name.

        Raises:
            VisualStoreUnavailable: If Qdrant is unreachable or the
                ``qdrant_client`` package is not installed.
        """
        from iw_rag.modes.visual.errors import VisualStoreUnavailable

        try:
            # Guarded import — qdrant_client is NOT a hard dependency.
            import qdrant_client  # noqa: F401
        except ImportError as exc:
            raise VisualStoreUnavailable(
                "qdrant_client is not installed. Install it or switch to "
                "backend=postgres_candidates (the default). "
                "Qdrant is the headline T1 trigger only (ADR-007)."
            ) from exc

        self._url = url
        self._collection_name = collection_name
        # TODO (T1 trigger): wire real QdrantClient when headline-promotion fires.
        # For now, this stub surfaces the seam clearly.
        raise VisualStoreUnavailable(
            f"QdrantMultiVectorStore at {url!r} is a stub/seam — not yet wired. "
            "The Qdrant path is the T1 headline trigger (ADR-007); promote the "
            "visual demo to headline and implement the full client here."
        )

    async def upsert(self, **kwargs: Any) -> None:
        """Not implemented — stub seam."""
        raise NotImplementedError("QdrantMultiVectorStore is a stub/seam — see ADR-007")

    async def stage1_prefetch(
        self,
        query_proxy: list[float],
        *,
        prefetch_n: int = 200,
    ) -> list[dict[str, Any]]:
        """Not implemented — stub seam."""
        raise NotImplementedError("QdrantMultiVectorStore is a stub/seam — see ADR-007")


def make_store(
    backend: str,
    *,
    session_factory: Any = None,
    qdrant_url: str = "",
) -> MultiVectorStore:
    """Factory: create the appropriate MultiVectorStore for the configured backend.

    Args:
        backend: ``"postgres_candidates"`` or ``"qdrant"``.
        session_factory: Required for ``postgres_candidates``; yields
            ``AsyncSession``.
        qdrant_url: Required for ``qdrant``; URL of the Qdrant server.

    Returns:
        A concrete ``MultiVectorStore`` implementation.

    Raises:
        ValueError: If an unknown backend string is given.
        VisualStoreUnavailable: If ``backend=qdrant`` but Qdrant is unreachable.
    """
    if backend == "postgres_candidates":
        return PostgresCandidateStore(session_factory)
    if backend == "qdrant":
        return QdrantMultiVectorStore(qdrant_url)
    raise ValueError(
        f"Unknown visual store backend: {backend!r}. Must be 'postgres_candidates' or 'qdrant'."
    )
