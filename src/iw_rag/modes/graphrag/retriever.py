"""GraphRetriever — hot-path query branch for the GraphRAG mode (WI-029 / Task 4a).

HOT-PATH: this module IS reachable from /query.  It calls ``AgeGraphStore``
**only for read operations** (``query_subgraph`` + ``resolve_to_chunks``) — never
the indexer, never the extraction model.  The two-clocks rule is enforced by the
offline guard test (``test_graphrag_offline_guard.py``).

ACL posture (open item, task-4a Risks §1):
This path uses an ACL POST-filter on the resolved chunks' ``acl_tags`` before any
chunk reaches the generator.  The spine enforces ACL as a pre-filter (WHERE
acl_tags && :principal_groups *before* retrieval); the graph route cannot pre-filter
during graph traversal without per-edge ``acl_tags`` on AGE nodes/edges.

Until the graph traversal itself is ACL-scoped, the graph route is restricted to
uniformly-readable demo corpora (task-4a Risks §1 — conservative rule).  The
post-filter ensures the generator NEVER sees a chunk whose ``acl_tags`` do not
overlap ``access.principal_groups``.

D22 [DECISION — LightRAG vs store-traversal]:
LightRAG's query engine is heavy and requires a running LLM endpoint (the
extraction model).  On the hot path, only AGE read queries are acceptable.  The
retriever therefore uses ``AgeGraphStore.query_subgraph`` (OpenCypher traversal)
to find entities matching the query seeds, then ``resolve_to_chunks`` to map
entity names → source chunk_ids.  For a simple ``MIX``-mode degrade, when no
entities are found the result is an empty ``RetrievalResult`` (the router falls
back to the spine).  This approach is model-agnostic and fully fakeable in tests.

References:
- task-4a-graphrag-route §7–§9 (retriever, adapter, ACL post-filter)
- lld-graphrag §3 (GraphRetriever public interface)
- lld-retrieval-spine §4 (RetrievalResult contract)
- CONVENTIONS.md §1 (chunks: list[RetrievedChunk], retriever, acl_applied)
"""

from __future__ import annotations

import logging
from typing import Any

from iw_rag.domain.acl import AccessContext
from iw_rag.domain.models import RetrievalResult, RetrievedChunk
from iw_rag.modes.graphrag.store import GraphRouteTimeout, GraphStoreUnavailable
from iw_rag.modes.graphrag.types import GraphQueryMode, GraphRetrieverConfig

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure helpers (unit-testable without services)
# ---------------------------------------------------------------------------


def _acl_postfilter(chunk_acl_tags: list[str], principal_groups: list[str]) -> bool:
    """ACL post-filter predicate: allow iff chunk tags ∩ principal_groups non-empty.

    An empty ``chunk_acl_tags`` means the chunk is publicly readable — always
    allow.  A non-empty ``chunk_acl_tags`` with zero overlap against
    ``principal_groups`` (including when ``principal_groups`` is empty) → deny
    (fail-closed).

    OPEN ITEM (task-4a Risks §1): This is a post-filter on the resolved-chunk
    boundary.  The model never sees a dropped chunk, but the pre-filter invariant
    (CONVENTIONS.md §6 / lld-data-model §5) requires pre-filter at the traversal
    level once per-edge ``acl_tags`` are added to AGE nodes/edges.  Until then
    the graph route is restricted to uniformly-readable demo corpora.

    Args:
        chunk_acl_tags: The chunk's ACL entitlement tags.
        principal_groups: The authenticated user's principal groups.

    Returns:
        ``True`` if the chunk is readable by the principal; ``False`` otherwise.
    """
    if not chunk_acl_tags:
        # Empty tags → publicly readable → always allow.
        return True
    return bool(set(chunk_acl_tags) & set(principal_groups))


def _adapt_graph_chunk(row: dict[str, Any], *, score: float = 0.0) -> RetrievedChunk:
    """Adapt a graph-resolved chunk row dict to a :class:`RetrievedChunk`.

    The ``content`` field carries the chunk text so the text generator answers
    from the stored chunk, not from graph entity metadata.  When ``chunk_id`` is
    empty (unresolved entity → no stored chunk), the adapter flags a
    ``doc_level_citation=True`` sentinel in metadata (task-4a §9
    [ASSUMPTION]: citation falls back to document-level).

    Args:
        row: Chunk row dict with keys ``chunk_id``, ``content``, ``source_uri``,
            ``acl_tags``, and optional ``metadata``.
        score: Ranking score for this chunk (provided by caller; defaults to 0.0).

    Returns:
        A :class:`RetrievedChunk` with ``retriever="graphrag"`` and
        ``acl_applied=True``.
    """
    chunk_id: str = str(row.get("chunk_id", ""))
    content: str = str(row.get("content", ""))
    source_uri: str = str(row.get("source_uri", ""))
    raw_meta: dict[str, Any] = dict(row.get("metadata") or {})

    # Doc-level citation fallback: when chunk_id is empty, the entity could not
    # be resolved to a stored iw_rag.chunks row.  Flag this so eval can score it
    # separately (task-4a §9).
    if not chunk_id:
        raw_meta["doc_level_citation"] = True

    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=str(row.get("document_id", "") or ""),
        content=content,
        score=float(score),
        source_uri=source_uri,
        metadata=raw_meta,
        leaf_content=None,
        retriever="graphrag",
        acl_applied=True,
    )


# ---------------------------------------------------------------------------
# GraphRetriever — the hot-path retriever
# ---------------------------------------------------------------------------


class GraphRetriever:
    """Hot-path GraphRAG retriever: graph traversal → chunk resolution → RetrievalResult.

    Implements the retrieve-and-adapt loop:
    1. ``AgeGraphStore.query_subgraph`` — OpenCypher traversal from query seeds.
    2. ``AgeGraphStore.resolve_to_chunks`` — entity names → source chunk_ids.
    3. Fetch chunk rows via ``chunk_repo.get_by_str_ids``.
    4. ACL post-filter (drop chunks whose ``acl_tags`` ∩ ``principal_groups`` = ∅).
    5. Adapt to ``RetrievedChunk``s ordered by score; set ``retriever="graphrag"``,
       ``acl_applied=True``.

    D22 [DECISION — query-seed extraction]:
    The seeds for the graph traversal come from a keyword-extraction step on the
    query string (simple tokenization; no model call on the hot path).  For the
    ``MIX`` mode, this is a graceful-degrade approximation: misrouted simple
    queries get an empty subgraph → empty result → router spine fallback.  Full
    LightRAG ``mix`` semantics (graph + vector blend with the extraction model)
    are deferred to the live run.

    Error handling (degrade, never block — lld-graphrag §5):
    - ``GraphRouteTimeout`` → empty ``RetrievalResult`` (caller/node falls back to spine).
    - ``GraphStoreUnavailable`` → empty ``RetrievalResult``.
    - Any other unexpected error → empty ``RetrievalResult`` + WARNING log.
    - Empty entity list → empty ``RetrievalResult`` (graph not yet indexed).
    - ACL drops all → empty ``RetrievalResult`` (the loop's weak-support path handles it).

    Attributes:
        store: The AgeGraphStore (read-only on the hot path).
        chunk_repo: A repository exposing ``get_by_str_ids(chunk_ids) -> list[dict]``.
        cfg: GraphRetrieverConfig controlling defaults, timeouts, and ACL behaviour.
    """

    def __init__(
        self,
        store: Any,
        chunk_repo: Any,
        cfg: GraphRetrieverConfig | None = None,
    ) -> None:
        """Wire the retriever.

        Args:
            store: AgeGraphStore (or a test fake) with ``query_subgraph`` and
                ``resolve_to_chunks`` async methods.
            chunk_repo: A repository (or test fake) with ``get_by_str_ids`` that
                accepts a list of chunk_id strings and returns a list of chunk
                row dicts.
            cfg: Optional retriever configuration.  Defaults to ``GraphRetrieverConfig()``.
        """
        self.store = store
        self.chunk_repo = chunk_repo
        self.cfg: GraphRetrieverConfig = cfg or GraphRetrieverConfig()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_seeds(self, query: str) -> list[str]:
        """Extract entity-name seeds from a query string (keyword extraction).

        Simple tokenization for the hot path (no model call).  This is a
        graceful-degrade approximation: real LightRAG NER seeds would require
        the extraction model which must not run on the hot path (two-clocks rule).

        Args:
            query: The user query string.

        Returns:
            A list of candidate entity name tokens (deduplicated, non-empty).
        """
        # Tokenise on whitespace; strip common stop-words and punctuation.
        # Keep tokens ≥ 3 chars (entity names are typically longer).
        stop_words = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "to",
            "of",
            "in",
            "and",
            "or",
            "but",
            "how",
            "what",
            "who",
            "which",
            "does",
            "do",
            "it",
            "its",
            "for",
            "at",
            "by",
            "with",
            "from",
            "that",
            "this",
        }
        tokens = query.strip("?!.,;:\"'").split()
        seen: set[str] = set()
        seeds: list[str] = []
        for tok in tokens:
            clean = tok.strip("?!.,;:\"'").strip()
            if len(clean) >= 3 and clean.lower() not in stop_words and clean not in seen:
                seen.add(clean)
                seeds.append(clean)
        return seeds

    def _determine_mode(self, mode: GraphQueryMode | None) -> GraphQueryMode:
        """Determine the effective query mode.

        Applies the ``enable_global`` guard: ``GLOBAL`` is only allowed when
        ``cfg.enable_global`` is ``True`` (requires community summaries at index
        time, which are expensive).

        Args:
            mode: Caller-provided mode override; ``None`` means use the default.

        Returns:
            The effective :class:`GraphQueryMode`.
        """
        effective = mode if mode is not None else self.cfg.default_mode
        if effective == GraphQueryMode.GLOBAL and not self.cfg.enable_global:
            _log.debug(
                "GraphRetriever: GLOBAL mode requested but enable_global=False — "
                "falling back to MIX"
            )
            effective = GraphQueryMode.MIX
        return effective

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def retrieve(
        self,
        query: str,
        access: AccessContext,
        *,
        mode: GraphQueryMode | None = None,
        top_k: int | None = None,
    ) -> RetrievalResult:
        """Execute the graph-based retrieval and return a RetrievalResult.

        The returned object is structurally identical to the spine's output
        (same ``RetrievalResult`` contract) so the generator and graders are
        mode-agnostic.

        Args:
            query: The user query string.
            access: The authenticated AccessContext (``principal_groups`` for ACL).
            mode: Optional query mode override (defaults to ``cfg.default_mode``).
            top_k: Optional top-k override (defaults to ``cfg.top_k``).

        Returns:
            A ``RetrievalResult`` with ``retriever="graphrag"`` and
            ``acl_applied=True``.  An empty chunk list is returned on any error
            (degrade, never 500); the router node falls back to the spine.

        Note:
            ACL post-filter drops chunks whose ``acl_tags`` do not overlap
            ``access.principal_groups`` before any chunk reaches the generator.
            The model NEVER sees a dropped chunk's content.
        """
        effective_top_k = top_k if top_k is not None else self.cfg.top_k
        effective_mode = self._determine_mode(mode)
        principal_groups = list(access.principal_groups)

        def _empty_result(reason: str = "") -> RetrievalResult:
            """Build an empty RetrievalResult with diagnostics.

            Args:
                reason: Optional reason tag for diagnostics.

            Returns:
                An empty but structurally valid RetrievalResult.
            """
            diag: dict[str, Any] = {"graph_mode": str(effective_mode)}
            if reason:
                diag["degrade_reason"] = reason
            return RetrievalResult(
                chunks=[],
                retriever="graphrag",
                acl_applied=True,
                query=query,
                candidate_count=0,
                diagnostics=diag,
            )

        # ----------------------------------------------------------------
        # Step 1: Extract seeds from query (keyword-only, no model call)
        # ----------------------------------------------------------------
        seeds = self._extract_seeds(query)
        if not seeds:
            _log.debug(
                "GraphRetriever: no seeds extracted from query %r → empty result", query[:80]
            )
            return _empty_result("no_seeds")

        # ----------------------------------------------------------------
        # Step 2: Graph traversal (read-only AGE query)
        # ----------------------------------------------------------------
        try:
            raw_ctx = await self.store.query_subgraph(seeds, hops=2)
        except GraphRouteTimeout:
            _log.warning(
                "GraphRetriever: query_subgraph timed out for query %r (seeds=%r)",
                query[:80],
                seeds,
            )
            return _empty_result("timeout")
        except GraphStoreUnavailable as exc:
            _log.warning(
                "GraphRetriever: AGE store unavailable: %s — returning empty result",
                exc,
            )
            return _empty_result("store_unavailable")
        except Exception as exc:
            _log.warning(
                "GraphRetriever: unexpected error from query_subgraph: %s — returning empty result",
                exc,
            )
            return _empty_result("unexpected_error")

        entity_ids = raw_ctx.entity_ids if raw_ctx else []
        if not entity_ids:
            _log.debug(
                "GraphRetriever: subgraph traversal returned 0 entity_ids for "
                "query %r (seeds=%r) — graph may not be indexed",
                query[:80],
                seeds,
            )
            return _empty_result("empty_graph")

        # ----------------------------------------------------------------
        # Step 3: Resolve entity IDs → chunk_id strings
        # ----------------------------------------------------------------
        try:
            chunk_id_strs = await self.store.resolve_to_chunks(entity_ids)
        except (GraphRouteTimeout, GraphStoreUnavailable) as exc:
            _log.warning(
                "GraphRetriever: resolve_to_chunks error: %s — returning empty result", exc
            )
            return _empty_result("resolve_error")
        except Exception as exc:
            _log.warning("GraphRetriever: unexpected error from resolve_to_chunks: %s", exc)
            return _empty_result("resolve_unexpected_error")

        if not chunk_id_strs:
            _log.debug(
                "GraphRetriever: resolve_to_chunks returned 0 chunk_ids for entity_ids=%r",
                entity_ids,
            )
            return _empty_result("no_chunks_resolved")

        # ----------------------------------------------------------------
        # Step 4: Fetch chunk rows from the repository
        # ----------------------------------------------------------------
        try:
            chunk_rows = await self.chunk_repo.get_by_str_ids(chunk_id_strs)
        except Exception as exc:
            _log.warning("GraphRetriever: get_by_str_ids failed: %s — returning empty result", exc)
            return _empty_result("chunk_fetch_error")

        candidate_count = len(chunk_rows)

        # ----------------------------------------------------------------
        # Step 5: ACL post-filter
        # Drop chunks whose acl_tags do not overlap principal_groups.
        # ----------------------------------------------------------------
        visible: list[dict[str, Any]] = []
        for row in chunk_rows:
            row_acl = list(row.get("acl_tags") or [])
            if _acl_postfilter(row_acl, principal_groups):
                visible.append(row)
            else:
                _log.debug(
                    "GraphRetriever: ACL post-filter dropped chunk_id=%r "
                    "(acl_tags=%r ∩ principal_groups=%r = ∅)",
                    row.get("chunk_id"),
                    row_acl,
                    principal_groups,
                )

        # ----------------------------------------------------------------
        # Step 6: Adapt to RetrievedChunks ordered by score
        # Score is uniform (graph traversal hop distance not computed; ordering
        # preserves resolution order which reflects entity centrality).
        # ----------------------------------------------------------------
        # Assign descending scores based on resolution order (first resolved =
        # most central entity → highest score).  A small epsilon step keeps
        # scores distinct and avoids ties.
        n = len(visible)
        chunks: list[RetrievedChunk] = []
        for i, row in enumerate(visible):
            # Score in (0, 1]: first entity gets 1.0, last gets 1/n
            score = (n - i) / max(n, 1)
            chunk = _adapt_graph_chunk(row, score=score)
            chunks.append(chunk)

        # Apply top_k cap
        chunks = chunks[:effective_top_k]

        _log.debug(
            "GraphRetriever: query=%r mode=%r seeds=%r entity_ids=%d chunk_ids=%d "
            "visible=%d (after ACL) top_k=%d",
            query[:80],
            str(effective_mode),
            seeds,
            len(entity_ids),
            len(chunk_id_strs),
            len(visible),
            effective_top_k,
        )

        return RetrievalResult(
            chunks=chunks,
            retriever="graphrag",
            acl_applied=True,
            query=query,
            candidate_count=candidate_count,
            diagnostics={
                "graph_mode": str(effective_mode),
                "entity_ids": entity_ids,
                "seeds": seeds,
            },
        )
