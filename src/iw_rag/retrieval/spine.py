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
        section_hits: Whether a section (parent) row may match on its own and be a
            candidate.  ``False`` leaves only leaves; ``expansion_mode="leaf"`` does the same.
    """

    candidate_k: int = 40
    fused_k: int = 80
    top_k: int = 8
    rrf_k: int = 60
    rrf_weights: tuple[float, ...] = field(default_factory=lambda: (1.0, 1.0))
    expansion_mode: str = "parent"
    parent_expansion_max_chars: int = 4000
    rerank_on_failure: str = "degrade"
    section_hits: bool = True


def expand_to_parent(
    chunk: RetrievedChunk, parent_content: str | None, budget_chars: int
) -> RetrievedChunk:
    """Apply the small-to-big rule to one reranked leaf.

    The leaf's ``content`` becomes its parent's full text (the original leaf text is
    kept in ``leaf_content``) **only when** the parent fits ``budget_chars``.  An
    oversized parent (e.g. a whole-document parent for a PDF without heading
    structure) leaves the chunk unchanged apart from ``leaf_content``, so the matched
    text is always present and no candidate overflows the generator's window.  A chunk
    with no parent is returned as is.

    Args:
        chunk: The reranked leaf.
        parent_content: The parent's text, or ``None`` when the leaf has no parent.
        budget_chars: Per-candidate character budget
            (:attr:`RetrievalConfig.parent_expansion_max_chars`).

    Returns:
        The chunk to hand to the generator.
    """
    if parent_content is None:
        return chunk
    if len(parent_content) > budget_chars:
        logger.debug(
            "spine.expand_to_parent: skipping oversized parent (parent_chars=%d > budget=%d); "
            "keeping leaf content",
            len(parent_content),
            budget_chars,
        )
        return replace(chunk, leaf_content=chunk.content)
    return replace(chunk, content=parent_content, leaf_content=chunk.content)


def keep_candidate(
    chunk_id: str,
    content: str,
    section_ids: frozenset[str],
    *,
    expansion_mode: str,
    section_hits: bool,
    budget_chars: int,
) -> str | None:
    """Decide whether a fused candidate may go on to the reranker.

    Only section (parent) rows can be set aside; a leaf is always eligible.  A section row is
    set aside when sections are switched off, when the spine returns leaves only
    (``expansion_mode="leaf"``), or when the section is longer than the per-candidate budget:
    it could never be handed to the generator whole, and its leaves are still candidates.

    Args:
        chunk_id: The candidate's id.
        content: The candidate's text.
        section_ids: The ids, among the candidates, of rows that have children (sections).
        expansion_mode: ``leaf``, ``parent`` or ``window``.
        section_hits: The ``retrieval.section_hits`` switch.
        budget_chars: The per-candidate character budget.

    Returns:
        ``None`` to keep the candidate, otherwise why it is set aside: ``"section_hits_off"``,
        ``"leaf_only"`` or ``"oversize"``.
    """
    if chunk_id not in section_ids:
        return None
    if not section_hits:
        return "section_hits_off"
    if expansion_mode == "leaf":
        return "leaf_only"
    if len(content) > budget_chars:
        return "oversize"
    return None


def merge_repeated_sections(
    chunks: list[RetrievedChunk], section_ids: frozenset[str], top_k: int
) -> tuple[list[RetrievedChunk], int]:
    """Collapse chunks of one section that carry identical text, then keep ``top_k``.

    After small-to-big expansion every selected leaf of a section holds the section's text, and
    the section row itself may be selected too.  Chunks are walked in rank order; a chunk
    whose final text equals an earlier chunk of the same section is merged into it: the
    earlier chunk keeps its rank and score, adopts the leaf text as ``leaf_content`` when it
    has none (so a citation still quotes the matched leaf), and lists the merged ids in
    ``metadata["merged_chunk_ids"]``.  Because merging happens before the cut, the freed
    slots go to the next distinct chunks.

    Args:
        chunks: Reranked (and expanded) chunks, best first.
        section_ids: The ids, among the candidates, of rows that have children (sections).
        top_k: How many distinct chunks to return.

    Returns:
        ``(chunks, merged)``: at most ``top_k`` distinct chunks in rank order, and how many
        repeats were merged away.
    """
    out: list[RetrievedChunk] = []
    seen: dict[tuple[str, str], int] = {}
    merged = 0
    for chunk in chunks:
        parent = chunk.metadata.get("parent_id")
        section = (
            str(parent) if parent else (chunk.chunk_id if chunk.chunk_id in section_ids else None)
        )
        slot = (section, chunk.content) if section is not None else None
        if slot is not None and slot in seen:
            kept = out[seen[slot]]
            ids = [*kept.metadata.get("merged_chunk_ids", []), chunk.chunk_id]
            out[seen[slot]] = replace(
                kept,
                leaf_content=kept.leaf_content
                if kept.leaf_content is not None
                else chunk.leaf_content,
                metadata={**kept.metadata, "merged_chunk_ids": ids},
            )
            merged += 1
            continue
        if len(out) >= top_k:
            break
        if slot is not None:
            seen[slot] = len(out)
        out.append(chunk)
    return out, merged


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
            section_ids = await repo.get_section_ids([r["id"] for r in cand_rows])
            set_aside: dict[str, int] = {}
            candidates = []
            for r in cand_rows:
                why = keep_candidate(
                    str(r["id"]),
                    r["content"],
                    section_ids,
                    expansion_mode=self._cfg.expansion_mode,
                    section_hits=self._cfg.section_hits,
                    budget_chars=self._cfg.parent_expansion_max_chars,
                )
                if why is None:
                    candidates.append(RerankCandidate(chunk_id=str(r["id"]), content=r["content"]))
                else:
                    set_aside[why] = set_aside.get(why, 0) + 1
            if set_aside:
                diagnostics["sections_set_aside"] = set_aside

            # Every candidate is scored (the reranker reads them all anyway); the cut to top_k
            # happens after repeats are merged, so a merged slot is refilled by the next chunk.
            with span(
                "iw_rag.retrieval.rerank",
                {"candidates": len(candidates), "top_k": self._cfg.top_k},
                span_kind="RERANKER",
            ):
                try:
                    scored = await self._reranker.rerank(query, candidates, len(candidates))
                except RerankError:
                    if self._cfg.rerank_on_failure == "fail":
                        raise
                    diagnostics["rerank"] = "degraded"
                    scored = [
                        ScoredChunk(chunk_id=c.chunk_id, score=1.0 / (i + 1))
                        for i, c in enumerate(candidates)
                    ]

            chunks = await self._materialize(repo, scored)
            if self._cfg.expansion_mode != "leaf":
                chunks = await self._expand_parents(repo, chunks)
            chunks, merged = merge_repeated_sections(chunks, section_ids, self._cfg.top_k)
            if merged:
                diagnostics["repeats_merged"] = merged

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
            expanded.append(expand_to_parent(c, parent["content"] if parent else None, budget))
        return expanded
