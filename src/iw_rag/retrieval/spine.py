"""The retrieval spine — the hot-path callable every consumer reuses unchanged.

ACL pre-filter → dense (pgvector) + sparse (BM25) → RRF(60) → rerank → top-k →
optional small-to-big parent expansion → :class:`RetrievalResult`. Becomes the
LangGraph tool in Phase 2. Invariants (test-asserted): ``acl_applied is True``;
chunks sorted by score desc; ``len(chunks) <= top_k``; every ``source_uri``
non-empty. Reranker outage *degrades* (fused order) — embedding outage is a hard
error. An empty ACL set legitimately yields an empty result (caller fails closed).
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field, replace
from typing import Any

from iw_rag.domain.acl import AccessContext
from iw_rag.domain.errors import RerankError
from iw_rag.domain.models import RetrievalResult, RetrievedChunk
from iw_rag.observability.tracing import span
from iw_rag.retrieval.acl_filter import acl_predicate
from iw_rag.retrieval.dense import DenseRetriever
from iw_rag.retrieval.encode import QueryEncoder
from iw_rag.retrieval.fusion import RRFConfig, reciprocal_rank_fusion
from iw_rag.retrieval.rerank import RerankCandidate, Reranker, ScoredChunk
from iw_rag.retrieval.sparse import SparseRetriever
from iw_rag.storage.db import Database
from iw_rag.storage.repositories import ChunkRepository

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    """Spine budgets + policies (from RetrievalSettings).

    Attributes:
        candidate_k: Per-arm candidate budget.
        fused_k: Candidates kept after fusion (fed to the reranker).
        top_k: Final results returned.
        rrf_k: RRF constant (explicit 60).
        rrf_weights: Per-arm RRF weights (dense, sparse).
        expansion_mode: ``leaf|parent|window`` (parent = small-to-big).
        parent_expansion_max_chars: Per-candidate char budget enforced during
            small-to-big parent expansion.  A leaf is expanded to its parent's
            full content only when the parent fits within this budget;
            otherwise the leaf's own content is retained (matched text is
            always present, no candidate ever exceeds the budget).  Sized so
            that ``top_k * (parent_expansion_max_chars / 4 chars-per-token) +
            ~2k system/question/JSON-schema overhead`` stays well under the
            generator's ``max_model_len`` (see I-00131 design §"Budget sizing
            constraint").  Mirrors :attr:`RetrievalSettings.parent_expansion_max_chars`.
        rerank_on_failure: ``degrade`` (fused order) or ``fail``.
    """

    candidate_k: int = 40
    fused_k: int = 80
    top_k: int = 8
    rrf_k: int = 60
    rrf_weights: tuple[float, ...] = field(default_factory=lambda: (1.0, 1.0))
    expansion_mode: str = "parent"
    parent_expansion_max_chars: int = 4000
    rerank_on_failure: str = "degrade"


class RetrievalSpine:
    """Assembles the ACL-filtered hybrid → RRF → rerank → expand pipeline."""

    def __init__(
        self,
        db: Database,
        encoder: QueryEncoder,
        dense: DenseRetriever,
        sparse: SparseRetriever,
        reranker: Reranker,
        cfg: RetrievalConfig | None = None,
    ) -> None:
        """Wire the spine collaborators.

        Args:
            db: Database handle (opens one session per retrieve call).
            encoder: Query embedder.
            dense: Dense (pgvector) arm.
            sparse: Sparse (BM25) arm.
            reranker: Cross-encoder reranker.
            cfg: Budgets + policies.
        """
        self._db = db
        self._encoder = encoder
        self._dense = dense
        self._sparse = sparse
        self._reranker = reranker
        self._cfg = cfg or RetrievalConfig()

    async def retrieve(self, query: str, ctx: AccessContext) -> RetrievalResult:
        """Run the spine for one query under one access context.

        Args:
            query: The user query.
            ctx: The authenticated access context (the sole ACL source).

        Returns:
            A ranked, ACL-respecting :class:`RetrievalResult` (``acl_applied=True``).
        """
        # ACL pre-filter (the keystone) — wrapped for trace visibility even though
        # it is a pure, instant predicate build (never raw groups in attributes).
        with span("iw_rag.retrieval.acl_filter", {"n_groups": len(ctx.principal_groups)}):
            acl = acl_predicate(ctx)
        diagnostics: dict[str, Any] = {}
        if acl.is_empty:
            return RetrievalResult(
                chunks=[],
                retriever="spine",
                acl_applied=True,
                query=query,
                expansion_mode=self._cfg.expansion_mode,
                candidate_count=0,
                diagnostics={"acl": "empty"},
            )

        with span("iw_rag.retrieval.encode"):
            qvec = await self._encoder.encode(query)
        async with self._db.session() as session:
            # Arms run sequentially: a single AsyncSession is not concurrency-safe.
            # (Parallel arms via separate connections is a Phase-1 perf item.)
            with span("iw_rag.retrieval.dense", {"candidate_k": self._cfg.candidate_k}):
                dense_hits = await self._dense.retrieve(session, qvec, acl, self._cfg.candidate_k)
            with span("iw_rag.retrieval.sparse", {"candidate_k": self._cfg.candidate_k}):
                sparse_hits = await self._sparse.retrieve(
                    session, query, acl, self._cfg.candidate_k
                )
            with span(
                "iw_rag.retrieval.fusion",
                {
                    "rrf_k": self._cfg.rrf_k,
                    "dense_n": len(dense_hits),
                    "sparse_n": len(sparse_hits),
                },
            ):
                fused = reciprocal_rank_fusion(
                    [dense_hits, sparse_hits],
                    RRFConfig(k=self._cfg.rrf_k, weights=self._cfg.rrf_weights),
                )
                top_fused = fused[: self._cfg.fused_k]
            repo = ChunkRepository(session)
            cand_rows = await repo.get_retrievable([uuid.UUID(h.chunk_id) for h in top_fused])
            candidates = [
                RerankCandidate(chunk_id=str(r["id"]), content=r["content"]) for r in cand_rows
            ]

            with span(
                "iw_rag.retrieval.rerank",
                {"candidates": len(candidates), "top_k": self._cfg.top_k},
                span_kind="RERANKER",
            ):
                try:
                    scored = await self._reranker.rerank(query, candidates, self._cfg.top_k)
                except RerankError:
                    if self._cfg.rerank_on_failure == "fail":
                        raise
                    diagnostics["rerank"] = "degraded"
                    scored = [
                        ScoredChunk(chunk_id=c.chunk_id, score=1.0 / (i + 1))
                        for i, c in enumerate(candidates[: self._cfg.top_k])
                    ]

            chunks = await self._materialize(repo, scored)
            if self._cfg.expansion_mode != "leaf":
                chunks = await self._expand_parents(repo, chunks)

        return RetrievalResult(
            chunks=chunks,
            retriever="spine",
            acl_applied=True,
            query=query,
            expansion_mode=self._cfg.expansion_mode,
            candidate_count=len(fused),
            diagnostics=diagnostics,
        )

    async def _materialize(
        self, repo: ChunkRepository, scored: list[ScoredChunk]
    ) -> list[RetrievedChunk]:
        """Fetch content/source_uri for scored ids and build RetrievedChunks (order kept).

        Threads the owning ``documents.id`` (selected by
        :meth:`ChunkRepository.get_retrievable` as ``document_id``) onto each
        :class:`RetrievedChunk`. The UUID embedded in ``source_uri`` is the
        *storage* uuid, not the PK the download endpoint keys on; only
        ``document_id`` is the addressable identifier for the download flow
        (CR-00098). The UUID is cast to ``str`` here so downstream layers
        (Citation, CitationOut, download-token minting) deal exclusively in
        string ids.
        """
        rows = await repo.get_retrievable([uuid.UUID(s.chunk_id) for s in scored])
        by_id = {str(r["id"]): r for r in rows}
        out: list[RetrievedChunk] = []
        for s in scored:
            row = by_id.get(s.chunk_id)
            if row is None:
                continue
            out.append(
                RetrievedChunk(
                    chunk_id=s.chunk_id,
                    document_id=str(row["document_id"]),
                    content=row["content"],
                    score=s.score,
                    source_uri=row["source_uri"],
                    metadata={
                        **(row["metadata"] or {}),
                        "parent_id": str(row["parent_id"]) if row["parent_id"] else None,
                    },
                )
            )
        return out

    async def _expand_parents(
        self, repo: ChunkRepository, chunks: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        """Small-to-big parent expansion, gated on a per-candidate char budget.

        For each leaf whose ``metadata["parent_id"]`` is set, replace the
        leaf's ``content`` with the parent's full content **only when** the
        parent fits within ``self._cfg.parent_expansion_max_chars`` chars.
        When the parent exceeds the budget (e.g. an ingested whole-document
        parent for an uploaded PDF without heading structure) the leaf is
        returned unchanged: the matched text is always present and no
        candidate ever overflows the generator's ``max_model_len`` window.

        A ``logger.debug`` is emitted when a parent is skipped so operators
        can spot documents whose parents should be split at ingestion
        (S03 / WI I-00131).

        Args:
            repo: The chunk repository used to fetch parent rows.
            chunks: The reranked leaves to expand.

        Returns:
            A new list of :class:`RetrievedChunk` with the same length and
            order as ``chunks``; each entry is either the parent's content
            (``leaf_content`` set to the original leaf text) or the leaf
            unchanged.
        """
        parent_ids = [
            uuid.UUID(c.metadata["parent_id"]) for c in chunks if c.metadata.get("parent_id")
        ]
        if not parent_ids:
            return chunks
        parent_rows = {str(r["id"]): r for r in await repo.get_retrievable(parent_ids)}
        budget = self._cfg.parent_expansion_max_chars
        expanded: list[RetrievedChunk] = []
        for c in chunks:
            pid = c.metadata.get("parent_id")
            parent = parent_rows.get(pid) if pid else None
            if parent is None:
                expanded.append(c)
                continue
            parent_content = parent["content"]
            if len(parent_content) > budget:
                # Oversized parent (e.g. whole-document parent for an uploaded
                # PDF without headings) — keep the leaf so the matched text
                # is preserved and the candidate stays within the budget.
                logger.debug(
                    "spine._expand_parents: skipping oversized parent "
                    "(parent_id=%s, parent_chars=%d > budget=%d); keeping leaf content",
                    pid,
                    len(parent_content),
                    budget,
                )
                expanded.append(replace(c, leaf_content=c.content))
                continue
            expanded.append(replace(c, content=parent_content, leaf_content=c.content))
        return expanded
