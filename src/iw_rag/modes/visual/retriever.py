"""VisualRetriever — two-stage visual late-interaction retriever (WI-031 / Task 4b).

HOT-PATH: this module is reachable from /query.  It imports ColVisionEncoder
(encode_query only) and MultiVectorStore, but NEVER imports VisualIndexer,
VlmCaptioner, or scripts.index_visual.  See two-clocks enforcement in
tests/unit/test_visual_offline_guard.py.

ACL posture (open item, task-4b Risks §2):
This path uses an ACL POST-filter on the matched page's source ``chunks.acl_tags``
before the page reaches the generator.  The spine uses a pre-filter (NEVER
retrieve-then-drop), but the visual late-interaction path has no native stage-1
pre-filter until ``visual_pages.acl_tags`` has a GIN index and the query is
rewritten to filter before the HNSW step.  Until that pre-filter is wired, the
late-interaction path is restricted to uniformly-readable demo corpora, OR the
parse-then-caption fallback should be used (which inherits the spine's pre-filter
ACL natively).

D18 [DECISION — router]: The visual route is added to the router as a new
``"visual"`` label.  When ``modes.visual.enabled=false`` (default), the edge
collapses to ``"retrieve"`` (spine) — same collapse pattern as graph/long_context.
When enabled AND the encoder is unavailable (OOM/timeout), the retriever raises
VisualEncodeUnavailable and the edge collapses to ``"retrieve"`` (spine fallback).

References:
- task-4b-visual-rag-route §6–§8 (retriever, adapter, heatmap)
- lld-retrieval-spine §4 (RetrievalResult contract)
- CONVENTIONS.md §1 (chunks: list[RetrievedChunk], retriever, acl_applied)
"""

from __future__ import annotations

import logging
from typing import Any

from iw_rag.domain.acl import AccessContext
from iw_rag.domain.models import RetrievalResult, RetrievedChunk
from iw_rag.modes.visual.errors import VisualEncodeUnavailable
from iw_rag.modes.visual.math_utils import maxsim, mean_pool
from iw_rag.modes.visual.types import VisualRetrieverConfig

_log = logging.getLogger(__name__)


def _acl_postfilter(page_acl_tags: list[str], principal_groups: list[str]) -> bool:
    """ACL post-filter predicate: allow iff page tags ∩ principal_groups is non-empty.

    An empty ``page_acl_tags`` means the page is publicly readable — always allow.
    An empty ``principal_groups`` with restricted tags → deny (fail-closed).

    OPEN ITEM (task-4b Risks §2): This is a post-filter on the matched page
    boundary.  The model never sees a dropped page, but the pre-filter invariant
    (CONVENTIONS.md §6 / lld-data-model §5) requires the stage-1 prefetch to
    also pre-filter on acl_tags once the GIN index on visual_pages.acl_tags is
    added.  Until then, the late-interaction path is restricted to uniformly-
    readable demo corpora.

    Args:
        page_acl_tags: The page's ACL entitlement tags.
        principal_groups: The authenticated user's principal groups.

    Returns:
        ``True`` if the page is readable by the principal; ``False`` otherwise.
    """
    if not page_acl_tags:
        # Empty tags → publicly readable → allow.
        return True
    return bool(set(page_acl_tags) & set(principal_groups))


def _adapt_page_to_chunk(
    page: dict[str, Any],
    *,
    with_heatmap: bool,
    heatmap: list[list[float]] | None,
) -> RetrievedChunk:
    """Adapt a visual page candidate dict to a RetrievedChunk.

    The ``content`` field carries the extracted page text (R-00102 extraction)
    so the text generator answers from text, not the image.  The ``metadata``
    carries ``page_image_ref``, ``bbox``, and optionally ``maxsim_heatmap``
    for the interpretability hook (R-00107/F6).

    When the page has no extracted text, the ``no_text_for_generation`` flag
    is set in metadata (the vision-native path — beyond a single 32 GB card,
    needs a swap/second GPU — is out of scope here).

    Args:
        page: The candidate page dict from the store (must contain ``chunk_id``,
            ``content``, ``source_uri``, ``page``, ``score``, ``bbox``,
            ``page_image_ref``).
        with_heatmap: Whether to include the MaxSim heatmap in metadata.
        heatmap: The per-query-token × per-patch similarity matrix (included
            only when ``with_heatmap=True``).

    Returns:
        A :class:`~iw_rag.domain.models.RetrievedChunk` with ``retriever="visual"``
        and ``acl_applied=True``.
    """
    content: str = page.get("content", "")
    meta: dict[str, Any] = {
        "page_image_ref": page.get("page_image_ref", ""),
        "bbox": page.get("bbox", {}),
        "page": page.get("page"),
    }
    if not content:
        meta["no_text_for_generation"] = True
    if with_heatmap and heatmap is not None:
        meta["maxsim_heatmap"] = heatmap

    return RetrievedChunk(
        chunk_id=str(page.get("chunk_id", "")),
        document_id=str(page.get("document_id", "") or ""),
        content=content,
        score=float(page.get("score", 0.0)),
        source_uri=str(page.get("source_uri", "")),
        metadata=meta,
        leaf_content=None,
        retriever="visual",
        acl_applied=True,
    )


class VisualRetriever:
    """Two-stage visual late-interaction retriever (hot-path).

    Implements the ColPali/ColQwen2.5 two-stage retrieve:
    1. ``encode_query`` → mean-pooled proxy.
    2. Stage-1 (prefetch): cheap HNSW ANN over the proxy → ``prefetch_n`` candidates.
    3. Stage-2 (MaxSim rerank): full-resolution MaxSim over candidates → top-``top_k``.
    4. ACL post-filter on the matched page's source ``chunks.acl_tags``.
    5. Adapt to ``RetrievalResult`` (page text as ``content``, bbox, optional heatmap).

    Error handling (degrade, never block — lld-multimodal §5):
    - Encoder OOM / GPU busy → ``VisualEncodeUnavailable`` (caller collapses to spine).
    - Empty candidate set → empty ``RetrievalResult`` (loop's weak-support path).
    - ACL drops all → empty ``RetrievalResult`` (no leak).

    Attributes:
        encoder: The ColVisionEncoder (transient; encode_query per-query).
        store: The MultiVectorStore (postgres_candidates default).
        cfg: The VisualRetrieverConfig.
    """

    def __init__(
        self,
        encoder: Any,
        store: Any,
        cfg: VisualRetrieverConfig,
    ) -> None:
        """Wire the retriever.

        Args:
            encoder: ColVisionEncoder (or a test fake) with ``encode_query``.
            store: MultiVectorStore (or a test fake) with ``stage1_prefetch``.
            cfg: Retriever configuration.
        """
        self.encoder = encoder
        self.store = store
        self.cfg = cfg

    async def retrieve(
        self,
        query: str,
        access: AccessContext,
        *,
        top_k: int | None = None,
        with_heatmap: bool = False,
    ) -> RetrievalResult:
        """Execute the two-stage visual retrieve and return a RetrievalResult.

        The returned object is structurally identical to the spine's output
        (same ``RetrievalResult`` contract) so the generator and graders are
        mode-agnostic.

        Args:
            query: The user query string.
            access: The authenticated AccessContext (principal_groups for ACL).
            top_k: Final top-k pages to return (overrides config default).
            with_heatmap: Whether to attach the MaxSim heatmap to each chunk's
                metadata (interpretability hook, R-00107/F6).

        Returns:
            A ``RetrievalResult`` with ``retriever="visual"`` and ``acl_applied=True``.
            An empty candidate set (e.g. all ACL-dropped) → empty ``chunks`` list
            (the loop's weak-support path handles this gracefully).

        Raises:
            VisualEncodeUnavailable: If the encoder is unavailable (OOM, GPU
                busy, timeout).  The caller (router edge) falls back to the spine.
        """
        effective_top_k = top_k if top_k is not None else self.cfg.top_k

        # ----------------------------------------------------------------
        # Step 1: encode_query → multi-vector query tokens (transient)
        # ----------------------------------------------------------------
        try:
            query_vecs: list[list[float]] = await self.encoder.encode_query(query)
        except VisualEncodeUnavailable:
            raise
        except Exception as exc:
            # Any unexpected error from the encoder → VisualEncodeUnavailable.
            # The router/node edge catches this and collapses to the spine.
            raise VisualEncodeUnavailable(f"encode_query failed: {exc}") from exc
        # VisualEncodeUnavailable propagates up; the edge collapses to spine.

        # Build the mean-pooled proxy for stage-1 ANN.
        query_proxy = mean_pool(query_vecs) if query_vecs else [0.0]

        # ----------------------------------------------------------------
        # Step 2: Stage-1 prefetch — cheap ANN over mean-pooled proxy
        # ----------------------------------------------------------------
        candidates = await self.store.stage1_prefetch(query_proxy, prefetch_n=self.cfg.prefetch_n)

        if not candidates:
            _log.debug("VisualRetriever: stage-1 returned 0 candidates for query %r", query[:80])
            return RetrievalResult(
                chunks=[],
                retriever="visual",
                acl_applied=True,
                query=query,
                candidate_count=0,
            )

        # ----------------------------------------------------------------
        # Step 3: Stage-2 — full-resolution MaxSim rerank over candidates
        # ----------------------------------------------------------------
        scored: list[tuple[float, dict[str, Any], list[list[float]] | None]] = []
        for candidate in candidates:
            patch_vecs: list[list[float]] = candidate.get("pooled_vectors", [])
            if not patch_vecs:
                # No stored vectors → skip (shouldn't happen for well-formed rows)
                continue
            if with_heatmap:
                raw_score, heatmap_matrix = maxsim(  # type: ignore[misc]
                    query_vecs, patch_vecs, return_heatmap=True
                )
                scored.append((float(raw_score), candidate, heatmap_matrix))
            else:
                raw_score = maxsim(query_vecs, patch_vecs)
                scored.append((float(raw_score), candidate, None))  # type: ignore[arg-type]

        # Sort by MaxSim score descending
        scored.sort(key=lambda x: x[0], reverse=True)

        # ----------------------------------------------------------------
        # Step 4: ACL post-filter
        # (OPEN ITEM: promote to stage-1 pre-filter when acl_tags GIN index added)
        # ----------------------------------------------------------------
        principal_groups = list(access.principal_groups)
        surviving: list[tuple[float, dict[str, Any], list[list[float]] | None]] = []
        for score_val, cand, heatmap_matrix in scored:
            page_acl = list(cand.get("acl_tags", []))
            if _acl_postfilter(page_acl, principal_groups):
                surviving.append((score_val, cand, heatmap_matrix))
            else:
                _log.debug(
                    "VisualRetriever: ACL post-filter dropped chunk_id=%r (page acl_tags=%r)",
                    cand.get("chunk_id"),
                    page_acl,
                )

        # Take top-k
        surviving = surviving[:effective_top_k]

        # ----------------------------------------------------------------
        # Step 5: Adapt to RetrievalResult
        # ----------------------------------------------------------------
        chunks: list[RetrievedChunk] = []
        for score_val, cand, heatmap_matrix in surviving:
            # Inject the MaxSim score into the candidate dict for the adapter
            page_with_score = {**cand, "score": score_val}
            chunk = _adapt_page_to_chunk(
                page_with_score,
                with_heatmap=with_heatmap,
                heatmap=heatmap_matrix,
            )
            chunks.append(chunk)

        _log.debug(
            "VisualRetriever: query=%r → %d candidates (stage-1=%d, ACL-surviving=%d, top_k=%d)",
            query[:80],
            len(candidates),
            len(scored),
            len(surviving),
            effective_top_k,
        )

        return RetrievalResult(
            chunks=chunks,
            retriever="visual",
            acl_applied=True,
            query=query,
            candidate_count=len(candidates),
        )
