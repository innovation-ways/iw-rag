"""Dense retrieval arm — pgvector HNSW cosine search, ACL pre-filtered.

Returns rank-ordered hits (not raw scores) for RRF. The ACL predicate is bound
into the WHERE clause so an inaccessible chunk never enters the ranked list.
"""

from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from iw_rag.retrieval.acl_filter import AclPredicate
from iw_rag.retrieval.fusion import RankedHit


def _vec_literal(vec: list[float]) -> str:
    """Render a float vector as a pgvector literal ``[a,b,c]``."""
    return "[" + ",".join(repr(float(x)) for x in vec) + "]"


class DenseRetriever:
    """pgvector HNSW retriever over ``iw_rag.chunks.dense_embedding`` (cosine)."""

    async def retrieve(
        self,
        session: AsyncSession,
        query_vec: list[float],
        acl: AclPredicate,
        k: int,
    ) -> list[RankedHit]:
        """Return up to ``k`` ACL-filtered hits ranked by cosine distance.

        Args:
            session: Open async session.
            query_vec: The 1024-d query embedding.
            acl: The bound ACL predicate (empty → no results).
            k: Per-arm candidate budget.

        Returns:
            Rank-ordered hits (rank 1 = nearest); empty if the principal has no groups.
        """
        if acl.is_empty:
            return []
        stmt = text(
            "SELECT id::text AS id FROM iw_rag.chunks "
            "WHERE acl_tags && :principal_groups "
            "ORDER BY dense_embedding <=> CAST(:qvec AS vector) "
            "LIMIT :k"
        )
        result = await session.execute(
            stmt,
            {
                "principal_groups": list(acl.principal_groups),
                "qvec": _vec_literal(query_vec),
                "k": k,
            },
        )
        return [RankedHit(chunk_id=row[0], rank=i + 1) for i, row in enumerate(result)]
