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
from iw_rag.retrieval.dense import DenseRetriever
from iw_rag.retrieval.encode import InfinityQueryEncoder
from iw_rag.retrieval.rerank import InfinityReranker
from iw_rag.retrieval.sparse import SparseRetriever
from iw_rag.retrieval.spine import RetrievalConfig, RetrievalSpine
from iw_rag.settings import get_settings
from iw_rag.storage.db import Database

_GOLDEN = Path(__file__).resolve().parent.parent / "eval" / "golden" / "graphrag_golden.yaml"


class _StrChunkRepo:
    """Adapter: get_by_str_ids(list[str]) → chunk rows shaped for the graph retriever.

    The retriever's ``_adapt_graph_chunk`` reads ``chunk_id`` / ``content`` /
    ``source_uri`` / ``acl_tags``; ``ChunkRepository.get_by_ids`` keys on uuid and
    returns ``id`` (no source_uri), so resolve the shape directly here.
    """

    def __init__(self, db: Database) -> None:
        self._db = db

    async def get_by_str_ids(self, ids: Sequence[str]) -> list[dict[str, Any]]:
        uuids: list[uuid.UUID] = []
        for s in ids:
            try:
                uuids.append(uuid.UUID(s))
            except ValueError:
                continue
        if not uuids:
            return []
        sql = text(
            "SELECT c.id, c.content, c.acl_tags, d.source_uri "
            "FROM iw_rag.chunks c JOIN iw_rag.documents d ON d.id = c.document_id "
            "WHERE c.id = ANY(:ids)"
        )
        async with self._db.session() as session:
            rows = await session.execute(sql, {"ids": uuids})
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
