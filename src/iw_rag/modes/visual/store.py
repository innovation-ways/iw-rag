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

ACL posture (F-00097): ``stage1_prefetch`` enforces the ACL pre-filter at the
SQL boundary using the existing ``visual_pages_acl_gin`` index.  The
``WHERE acl_tags && :principal_groups`` clause is bound from the
authenticated ``AccessContext`` only — never from the query text or model
output (lld-guardrails §3.1).  An empty ``principal_groups`` is fail-closed
(mirrors :class:`~iw_rag.retrieval.acl_filter.AclPredicate.is_empty`): no
unfiltered query is ever issued.  The visual late-interaction path therefore
honors the headline security invariant "ACL is a pre-filter, NEVER a post-
filter" on multi-tenant corpora; the per-page ``_acl_postfilter`` in
:class:`~iw_rag.modes.visual.retriever.VisualRetriever` remains as a
defense-in-depth check.

References:
- task-4b-visual-rag-route §5 (store abstraction, T1 trigger)
- lld-data-model §8 (visual_pages table, Qdrant T1 trigger)
- ADR-007 (Postgres-first + Qdrant scoped trigger)
- ADR-010 (no raw multi-vectors, pooled+binary storage)
"""

from __future__ import annotations

import logging
from typing import Any, Protocol, runtime_checkable

from iw_rag.retrieval.acl_filter import AclPredicate

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
        principal_groups: list[str],
    ) -> list[dict[str, Any]]:
        """Stage-1 cheap prefetch: ANN over the mean-pooled proxy, ACL pre-filtered.

        Returns the top-``prefetch_n`` candidate page records (including their
        stored ``pooled_vectors`` for stage-2 MaxSim).  ACL is enforced at the
        SQL boundary via ``WHERE acl_tags && :principal_groups`` (GIN-accelerated
        by ``visual_pages_acl_gin``), so no unreadable page ever enters the
        ranked set — matching the spine's pre-filter invariant
        (F-00097, ``AclPredicate.sql_clause()``).

        Args:
            query_proxy: The mean-pooled single-vector proxy for the query.
            prefetch_n: Number of candidates to prefetch.
            principal_groups: The authenticated caller's ACL groups
                (from ``AccessContext``, never from query text).  An empty list
                is fail-closed: the SQL ``acl_tags && :principal_groups`` is
                false for every row, so no candidates are returned (no
                unfiltered query is issued).  Mirrors
                :class:`~iw_rag.retrieval.acl_filter.AclPredicate.is_empty`.

        Returns:
            List of candidate page dicts, each containing at least:
            ``chunk_id``, ``content``, ``source_uri``, ``page``, ``acl_tags``,
            ``score`` (proxy similarity), ``pooled_vectors``, ``bbox``,
            ``page_image_ref``.  Empty when ``principal_groups`` is empty.
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
        principal_groups: list[str],
    ) -> list[dict[str, Any]]:
        """Stage-1 ANN over the mean-pooled proxy vector (pgvector HNSW), ACL pre-filtered.

        Retrieves up to ``prefetch_n`` candidate pages using cosine similarity
        over the proxy vector column.  ACL is enforced at the SQL boundary via
        ``WHERE acl_tags && :principal_groups`` (GIN-accelerated by the existing
        ``visual_pages_acl_gin`` index), so a page the principal cannot see
        never enters the candidate set — matching the spine's pre-filter
        invariant (F-00097).

        The pre-filter ``&&`` reproduces the spine's empty-``acl_tags``
        semantics with no special-case: an item with ``acl_tags = []`` is
        invisible to every principal (the SQL ``[] && :principal_groups``
        is false for any non-empty ``principal_groups``, and false for an
        empty ``principal_groups`` too — fail-closed).  This matches the
        ingestion contract in :mod:`iw_rag.ingestion.enrich` ("an empty
        resolver result yields ``acl_tags = []`` → the document is invisible
        to every principal; fail-closed, never world-readable").  Broadly-
        readable documents carry an explicit shared tag (e.g. ``"public"``)
        that principals also hold; ``[]`` is never treated as public.

        Args:
            query_proxy: Mean-pooled query proxy vector.
            prefetch_n: Maximum candidates to return.
            principal_groups: The authenticated caller's ACL groups (from
                ``AccessContext``, never from query text).  An empty list
                short-circuits to ``[]`` (fail-closed); no unfiltered query
                is ever issued.  Mirrors
                :class:`~iw_rag.retrieval.acl_filter.AclPredicate.is_empty`.

        Returns:
            List of candidate dicts (see protocol docstring).  Empty when
            ``principal_groups`` is empty or the query has no readable
            matches in the prefetch budget.
        """
        import json

        from sqlalchemy import text as sql_text

        # Bind the ACL pre-filter at the SQL boundary (F-00097).  We reuse the
        # spine's AclPredicate so the visual pre-filter SQL is textually
        # identical to ``acl_tags && :principal_groups`` (Invariant 5).
        acl = AclPredicate(principal_groups=tuple(principal_groups))
        if acl.is_empty:
            return []

        proxy_str = "[" + ",".join(str(v) for v in query_proxy) + "]"
        sql = sql_text(
            f"""
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
            WHERE {acl.sql_clause()}
            ORDER BY proxy_vector <=> CAST(:query_vector AS vector)
            LIMIT :n
            """
        )
        candidates: list[dict[str, Any]] = []
        async with self._session_factory() as session:
            result = await session.execute(
                sql,
                {"query_vector": proxy_str, "n": prefetch_n, **acl.params()},
            )
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
        principal_groups: list[str],
    ) -> list[dict[str, Any]]:
        """Not implemented — stub seam.

        Args:
            query_proxy: Mean-pooled query proxy vector.
            prefetch_n: Number of candidates to prefetch.
            principal_groups: The authenticated caller's ACL groups (must be
                applied at the Qdrant-side filter when this seam is wired;
                F-00097 spine parity).
        """
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
