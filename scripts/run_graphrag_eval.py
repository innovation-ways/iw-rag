"""Headline experiment #6 (GraphRAG route vs hybrid spine) — LIVE.

Runs both arms over two golden subsets from ``eval/golden/graphrag_golden.yaml``:
  - multi_hop  : the answer doc shares no anchor with the question (graph SHOULD win)
  - procedural : single-doc policy/how-to (graph should NOT regress)

Both arms hit the SAME (AGE-enabled) corpus:
  - graph arm  : GraphRetriever → AgeGraphStore.query_subgraph + back-ref resolution
  - spine arm  : the Phase-0 hybrid spine (dense+BM25+RRF+rerank)

Requires an **AGE-enabled** Postgres (the production paradedb is PG18, AGE is
PG18-RC only — see eval/reports/graphrag_experiment.md for the isolated-copy build)
plus a graph indexed by ``scripts/index_graph.py``. Point both at the isolated store:

    IW_RAG__DATABASE__DSN="postgresql+psycopg://iw_rag:change-me-locally@localhost:5434/iw_rag" \
    IW_RAG__MODES__GRAPHRAG__ENABLED=true \
      uv run python scripts/run_graphrag_eval.py

Judge-free (recall/nDCG@k). The gate: graph WINS nDCG@5 on multi_hop AND does not
regress >5% on procedural → the route may be enabled. A negative result is a valid,
publishable finding (the route stays off by config).
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml
from sqlalchemy import text

from iw_rag.domain.acl import AccessContext
from iw_rag.domain.models import RetrievalResult
from iw_rag.eval.experiments import run_experiment_6_graphrag
from iw_rag.eval.repository import EvalRepository, GoldenQuery
from iw_rag.modes.graphrag.retriever import GraphRetriever
from iw_rag.modes.graphrag.store import AgeGraphStore
from iw_rag.retrieval.acl_filter import AclPredicate
from iw_rag.retrieval.dense import DenseRetriever
from iw_rag.retrieval.encode import InfinityQueryEncoder
from iw_rag.retrieval.rerank import InfinityReranker
from iw_rag.retrieval.sparse import SparseRetriever
from iw_rag.retrieval.spine import RetrievalConfig, RetrievalSpine
from iw_rag.settings import get_settings
from iw_rag.storage.db import Database

_GOLDEN = Path(__file__).resolve().parent.parent / "eval" / "golden" / "graphrag_golden.yaml"


class _StrChunkRepo:
    """Adapter: get_by_str_ids → chunk rows shaped for the graph retriever.

    Accepts ``(ids, principal_groups)`` and applies the ACL pre-filter at the
    SQL boundary (F-00097, Option A).  See the class docstring below for
    the full contract.

    The retriever's ``_adapt_graph_chunk`` reads ``chunk_id`` / ``content`` /
    ``source_uri`` / ``acl_tags``; ``ChunkRepository.get_by_ids`` keys on uuid and
    returns ``id`` (no source_uri), so resolve the shape directly here.

    F-00097 (Option A — SQL-boundary pre-filter):
    The fetch SQL applies the ACL pre-filter at the SQL boundary via
    ``acl_tags && :principal_groups`` (GIN-accelerated by the existing
    ``chunks_acl_gin`` index, matches the spine's pre-filter invariant).
    No row outside the principal's ACL is ever fetched, ranked, or fed to
    the generator.  ``principal_groups`` is sourced solely from the
    authenticated ``AccessContext`` and is never derived from the query
    text, the graph content, or the model output (lld-guardrails §3.1).

    Empty-``acl_tags`` semantics: a chunk with ``acl_tags = []`` is invisible
    to every principal — the SQL ``[] && :principal_groups`` is false for
    any non-empty ``principal_groups`` and false for an empty one too
    (fail-closed).  This matches the ingestion contract in
    :mod:`iw_rag.ingestion.enrich` ("an empty resolver result yields
    ``acl_tags = []`` → the document is invisible to every principal; fail-
    closed, never world-readable") and the spine's pre-filter.  There is no
    special-case for ``[]`` — broadly-readable documents carry an explicit
    shared tag (e.g. ``"public"``) that principals also hold.
    """

    def __init__(self, db: Database) -> None:
        self._db = db

    async def get_by_str_ids(
        self,
        ids: Sequence[str],
        principal_groups: Sequence[str],
    ) -> list[dict[str, Any]]:
        """Fetch chunk rows by id, ACL-pre-filtered at the SQL boundary.

        The SQL emits ``WHERE c.id = ANY(:ids) AND acl_tags && :principal_groups``,
        bound to the supplied ``principal_groups``.  An empty
        ``principal_groups`` short-circuits to ``[]`` (fail-closed) and never
        issues an unfiltered query — mirrors
        :class:`~iw_rag.retrieval.acl_filter.AclPredicate.is_empty`.

        Args:
            ids: Chunk-id strings to resolve (non-UUID entries are silently
                dropped — only valid UUIDs are queried).
            principal_groups: The authenticated caller's ACL groups (from
                ``AccessContext``, never from query text or model output).
                An empty list is fail-closed: returns ``[]`` without issuing
                an unfiltered query.

        Returns:
            List of chunk row dicts shaped for ``GraphRetriever._adapt_graph_chunk``:
            ``chunk_id``, ``content``, ``source_uri``, ``acl_tags``.  Empty
            when ``principal_groups`` is empty or none of the requested ids
            are readable.
        """
        # Fail-closed short-circuit: an empty principal_groups means the
        # principal sees nothing (mirrors AclPredicate.is_empty).  Never
        # issue an unfiltered query.
        acl = AclPredicate(principal_groups=tuple(principal_groups))
        if acl.is_empty:
            return []

        uuids: list[uuid.UUID] = []
        for s in ids:
            try:
                uuids.append(uuid.UUID(s))
            except ValueError:
                continue
        if not uuids:
            return []
        # Reuse the spine's AclPredicate so the pre-filter SQL is textually
        # identical to `acl_tags && :principal_groups` (Invariant 5).
        sql = text(
            f"SELECT c.id, c.content, c.acl_tags, d.source_uri "
            f"FROM iw_rag.chunks c JOIN iw_rag.documents d ON d.id = c.document_id "
            f"WHERE c.id = ANY(:ids) AND {acl.sql_clause()}"
        )
        async with self._db.session() as session:
            rows = await session.execute(sql, {"ids": uuids, **acl.params()})
            return [
                {
                    "chunk_id": str(r["id"]),
                    "content": r["content"],
                    "source_uri": r["source_uri"],
                    "acl_tags": list(r["acl_tags"] or []),
                }
                for r in rows.mappings()
            ]


async def _resolve(db: Database, source: str | None, keyword: str | None) -> list[str]:
    if not source or not keyword:
        return []
    sql = text(
        "SELECT c.id FROM iw_rag.chunks c JOIN iw_rag.documents d ON d.id = c.document_id "
        "WHERE d.source_uri = :src AND c.content ILIKE :kw"
    )
    async with db.session() as session:
        rows = await session.execute(sql, {"src": source, "kw": f"%{keyword}%"})
        return [str(r[0]) for r in rows.all()]


async def _golden_subset(db: Database, entries: list[dict]) -> list[GoldenQuery]:
    out: list[GoldenQuery] = []
    for q in entries:
        ids = await _resolve(db, q.get("source"), q.get("keyword"))
        out.append(
            GoldenQuery(
                question=q["question"], relevant_chunk_ids=ids, query_type=q.get("query_type")
            )
        )
    return out


def _make_spine(db: Database) -> RetrievalSpine:
    cfg = get_settings()
    emb = cfg.models.get(cfg.retrieval.embed_model)
    rer = cfg.models.get(cfg.retrieval.rerank_model)
    return RetrievalSpine(
        db=db,
        encoder=InfinityQueryEncoder(emb.endpoint, emb.served_name),
        dense=DenseRetriever(),
        sparse=SparseRetriever(),
        reranker=InfinityReranker(rer.endpoint, rer.served_name),
        cfg=RetrievalConfig(),
    )


async def main() -> None:
    spec = yaml.safe_load(_GOLDEN.read_text())
    groups = tuple(spec.get("principal_groups") or ("eng", "finance", "hr", "public"))
    ctx = AccessContext(principal_id="eval", principal_groups=groups)
    cfg = get_settings()
    db = Database(cfg.database)
    store = AgeGraphStore(dsn=cfg.database.dsn)
    chunk_repo = _StrChunkRepo(db)
    graph = GraphRetriever(store=store, chunk_repo=chunk_repo)
    spine = _make_spine(db)

    async def graph_retrieve(question: str) -> RetrievalResult:
        return await graph.retrieve(question, ctx)

    async def spine_retrieve(question: str) -> RetrievalResult:
        return await spine.retrieve(question, ctx)

    try:
        multihop = await _golden_subset(db, spec.get("multi_hop", []))
        procedural = await _golden_subset(db, spec.get("procedural", []))
        print(
            f"Golden: {_GOLDEN.name} | multi_hop={len(multihop)} procedural={len(procedural)} "
            f"| groups={groups}\n"
        )
        async with db.session() as session:
            repo = EvalRepository(session)
            report = await run_experiment_6_graphrag(
                repo,
                multihop_queries=multihop,
                procedural_queries=procedural,
                graph_retrieve=graph_retrieve,
                spine_retrieve=spine_retrieve,
                config={"corpus": "benchmark", "store": "isolated-age-pg18"},
            )
            await session.commit()
    finally:
        await store.aclose()
        await db.dispose()

    def _g(d: dict[int, float], k: int) -> float:
        return d.get(k, 0.0)

    print("=== Experiment #6 — GraphRAG vs hybrid spine (benchmark corpus) ===")
    print(f"{'subset':<12}{'arm':<8}{'recall@5':>10}{'recall@10':>11}{'nDCG@5':>9}")
    print(
        f"{'multi_hop':<12}{'graph':<8}{_g(report.graph_multihop_recall, 5):>10.3f}"
        f"{_g(report.graph_multihop_recall, 10):>11.3f}{_g(report.graph_multihop_ndcg, 5):>9.3f}"
    )
    print(
        f"{'multi_hop':<12}{'spine':<8}{_g(report.spine_multihop_recall, 5):>10.3f}"
        f"{_g(report.spine_multihop_recall, 10):>11.3f}{_g(report.spine_multihop_ndcg, 5):>9.3f}"
    )
    print(
        f"{'procedural':<12}{'graph':<8}{_g(report.graph_procedural_recall, 5):>10.3f}"
        f"{_g(report.graph_procedural_recall, 10):>11.3f}"
    )
    print(
        f"{'procedural':<12}{'spine':<8}{_g(report.spine_procedural_recall, 5):>10.3f}"
        f"{_g(report.spine_procedural_recall, 10):>11.3f}"
    )
    print(
        f"\nmulti-hop win: {report.multihop_win} | procedural no-loss: "
        f"{report.procedural_no_loss} | GATE PASSED: {report.gate_passed}"
    )
    if report.notes:
        print(f"notes: {report.notes}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        _GOLDEN = Path(sys.argv[1])
    asyncio.run(main())
