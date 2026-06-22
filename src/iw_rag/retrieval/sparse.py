"""Sparse retrieval arm — ParadeDB pg_search BM25, ACL pre-filtered.

The sanctioned raw-SQL shim (repo-structure §2 exception). Returns rank-ordered
hits for RRF. The BM25 query syntax is version-sensitive (pinned to the installed
ParadeDB tag); the query string is sanitized to bare alphanumeric terms so the
ParadeDB query parser does not choke on punctuation (e.g. invoice ids).
"""

from __future__ import annotations

import re

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from iw_rag.retrieval.acl_filter import AclPredicate
from iw_rag.retrieval.fusion import RankedHit

_TERM = re.compile(r"[A-Za-z0-9]+")


def _bm25_query(query_text: str) -> str:
    """Turn free text into a ParadeDB ``content:term`` OR query over indexed text.

    Args:
        query_text: The raw user query.

    Returns:
        A pg_search query string like ``content:invoice OR content:total``;
        empty string if the query has no usable terms.
    """
    terms = _TERM.findall(query_text.lower())
    return " OR ".join(f"content:{t}" for t in terms)


class SparseRetriever:
    """ParadeDB BM25 retriever over the ``chunks_bm25`` index."""

    async def retrieve(
        self,
        session: AsyncSession,
        query_text: str,
        acl: AclPredicate,
        k: int,
    ) -> list[RankedHit]:
        """Return up to ``k`` ACL-filtered BM25 hits ranked by score.

        Args:
            session: Open async session.
            query_text: The raw user query (sanitized to BM25 terms internally).
            acl: The bound ACL predicate (empty → no results).
            k: Per-arm candidate budget.

        Returns:
            Rank-ordered hits (rank 1 = best BM25 score); empty when the
            principal has no groups or the query has no usable terms.
        """
        if acl.is_empty:
            return []
        bm25 = _bm25_query(query_text)
        if not bm25:
            return []
        stmt = text(
            "SELECT id::text AS id FROM iw_rag.chunks "
            "WHERE id @@@ :bm25 AND acl_tags && :principal_groups "
            "ORDER BY paradedb.score(id) DESC "
            "LIMIT :k"
        )
        result = await session.execute(
            stmt,
            {"bm25": bm25, "principal_groups": list(acl.principal_groups), "k": k},
        )
        return [RankedHit(chunk_id=row[0], rank=i + 1) for i, row in enumerate(result)]
