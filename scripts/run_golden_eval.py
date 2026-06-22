"""Headline experiment #1 (retrieval ablation) over the demo golden set — LIVE.

Resolves the curated golden set (eval/golden/demo_golden.yaml) to live ``chunks.id``
values, builds three retrieval configurations against the live spine, runs each
through the judge-free metric harness (recall@k / nDCG@k / MRR), persists an
``eval_runs`` row per arm, and prints the ablation table.

Arms:
    dense          — dense-only (RRF weights (1,0)) + no-op reranker
    hybrid         — dense + sparse (RRF (1,1)) + no-op reranker
    hybrid_rerank  — production: dense + sparse + bge-reranker-v2-m3

Run (with the live stack up + corpus ingested)::

    set -a; . ./.env; set +a
    uv run --group ingest python scripts/run_golden_eval.py

Requires only the live Postgres + infinity (embedder/reranker). No judge models
(this experiment is judge-free). Faithfulness (exp #3) is a separate, judge-gated
run.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml
from sqlalchemy import text

from iw_rag.domain.acl import AccessContext
from iw_rag.domain.models import RetrievalResult
from iw_rag.eval.experiments import RunSummary, run_experiment
from iw_rag.eval.repository import EvalRepository, GoldenQuery
from iw_rag.retrieval.dense import DenseRetriever
from iw_rag.retrieval.encode import InfinityQueryEncoder
from iw_rag.retrieval.rerank import InfinityReranker, RerankCandidate, ScoredChunk
from iw_rag.retrieval.sparse import SparseRetriever
from iw_rag.retrieval.spine import RetrievalConfig, RetrievalSpine
from iw_rag.settings import get_settings
from iw_rag.storage.db import Database

_DEFAULT_GOLDEN = Path(__file__).resolve().parent.parent / "eval" / "golden" / "demo_golden.yaml"
_DEFAULT_GROUPS = ("finance", "hr", "public")


class _NoOpReranker:
    """Identity reranker: preserves the fused order (isolates the rerank arm)."""

    async def rerank(
        self, query: str, candidates: Sequence[RerankCandidate], top_n: int
    ) -> list[ScoredChunk]:
        """Return the first ``top_n`` candidates in fused order, descending score."""
        n = len(candidates)
        return [
            ScoredChunk(chunk_id=c.chunk_id, score=float(n - i))
            for i, c in enumerate(candidates[:top_n])
        ]

    async def health(self) -> bool:
        return True


async def _resolve_relevant(db: Database, source: str | None, keyword: str | None) -> list[str]:
    """Resolve (source_uri, keyword) → the matching live chunk ids (str)."""
    if not source or not keyword:
        return []
    sql = text(
        "SELECT c.id FROM iw_rag.chunks c JOIN iw_rag.documents d ON d.id = c.document_id "
        "WHERE d.source_uri = :src AND c.content ILIKE :kw"
    )
    async with db.session() as session:
        rows = await session.execute(sql, {"src": source, "kw": f"%{keyword}%"})
        return [str(r[0]) for r in rows.all()]


def _make_spine(db: Database, *, rrf_weights: tuple[float, float], rerank: bool) -> RetrievalSpine:
    cfg = get_settings()
    embedder = cfg.models.get(cfg.retrieval.embed_model)
    reranker_cfg = cfg.models.get(cfg.retrieval.rerank_model)
    reranker: Any = (
        InfinityReranker(reranker_cfg.endpoint, reranker_cfg.served_name)
        if rerank
        else _NoOpReranker()
    )
    return RetrievalSpine(
        db=db,
        encoder=InfinityQueryEncoder(embedder.endpoint, embedder.served_name),
        dense=DenseRetriever(),
        sparse=SparseRetriever(),
        reranker=reranker,
        cfg=RetrievalConfig(rrf_weights=rrf_weights),
    )


def _retrieve_fn(spine: RetrievalSpine, ctx: AccessContext) -> Any:
    async def _retrieve(question: str) -> RetrievalResult:
        return await spine.retrieve(question, ctx)

    return _retrieve


async def main(spec: dict, golden_name: str) -> None:
    # Eval runs measure pure retrieval quality, so the principal sees every group
    # present in the corpus (read from the golden set, default to the demo groups).
    groups = tuple(spec.get("principal_groups") or _DEFAULT_GROUPS)
    ctx = AccessContext(principal_id="eval", principal_groups=groups)
    print(f"Golden: {golden_name} | principal groups: {groups}")
    cfg = get_settings()
    db = Database(cfg.database)
    try:
        # Resolve the golden set against the live corpus.
        golden: list[GoldenQuery] = []
        for q in spec["queries"]:
            ids = await _resolve_relevant(db, q.get("source"), q.get("keyword"))
            golden.append(
                GoldenQuery(
                    question=q["question"], relevant_chunk_ids=ids, query_type=q.get("query_type")
                )
            )
        n_gold = sum(1 for g in golden if g.relevant_chunk_ids)
        n_zero = len(golden) - n_gold
        print(f"Golden set: {len(golden)} queries ({n_gold} with gold, {n_zero} zero-gold)")

        arms = [
            ("exp1_dense", (1.0, 0.0), False),
            ("exp1_hybrid", (1.0, 1.0), False),
            ("exp1_hybrid_rerank", (1.0, 1.0), True),
        ]
        summaries: list[RunSummary] = []
        async with db.session() as session:
            repo = EvalRepository(session)
            ds = await repo.create_dataset(spec["dataset"]["name"], spec["dataset"]["version"])
            await repo.add_queries(ds, golden)
            await session.commit()

        for name, weights, rerank in arms:
            spine = _make_spine(db, rrf_weights=weights, rerank=rerank)
            async with db.session() as session:
                repo = EvalRepository(session)
                summary = await run_experiment(
                    repo,
                    experiment=name,
                    retrieve=_retrieve_fn(spine, ctx),
                    queries=golden,
                    ks=(5, 10),
                    config={"arm": name, "rrf_weights": list(weights), "rerank": rerank},
                )
                await session.commit()
            summaries.append(summary)

        print("\n=== Headline experiment #1 — retrieval ablation (live, demo corpus) ===")
        print(
            f"{'arm':<20} {'recall@5':>9} {'recall@10':>10} {'nDCG@5':>8} {'nDCG@10':>8} {'MRR':>6}"
        )
        for s in summaries:
            r5, r10 = s.mean_recall.get(5, 0.0), s.mean_recall.get(10, 0.0)
            n5, n10 = s.mean_ndcg.get(5, 0.0), s.mean_ndcg.get(10, 0.0)
            print(
                f"{s.experiment:<20} {r5:>9.3f} {r10:>10.3f} "
                f"{n5:>8.3f} {n10:>8.3f} {s.mean_mrr:>6.3f}"
            )
        floor = cfg.eval.retrieval.recall_at_10_min
        prod = next(s for s in summaries if s.experiment == "exp1_hybrid_rerank")
        prod_r10 = prod.mean_recall.get(10, 0.0)
        verdict = "PASS" if prod_r10 >= floor else "FAIL"
        print(f"\nRecall@10 floor {floor}: production={prod_r10:.3f} -> {verdict}")
    finally:
        await db.dispose()


if __name__ == "__main__":
    _path = Path(sys.argv[1]) if len(sys.argv) > 1 else _DEFAULT_GOLDEN
    _spec = yaml.safe_load(_path.read_text())
    asyncio.run(main(_spec, _path.name))
