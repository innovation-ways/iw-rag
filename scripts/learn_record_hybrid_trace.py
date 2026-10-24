"""Record the Learn section's hybrid-search lab from the REAL retrieval components.

Starts a throwaway ParadeDB container (never the live database), seeds the lab corpus
embedded by the served BGE-M3, then for every query and principal runs the production
``DenseRetriever`` / ``SparseRetriever`` / ``reciprocal_rank_fusion`` and the served
cross-encoder reranker, and writes everything to
``src/iw_rag/learn/traces/hybrid_search.json``.

The lab replays that file, so it needs no GPU at view time.  Re-run this script after
changing ``learn/lab_corpus.py`` or the retrieval stack::

    INFINITY_URL=http://localhost:7997 uv run python scripts/learn_record_hybrid_trace.py

Documents are embedded as pooled vectors through the served endpoint (not with late
chunking), which is recorded in the file so the lab can say so.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import uuid
from datetime import date
from pathlib import Path
from typing import Any

from sqlalchemy import text

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # lets this script reuse the integration-test DB bootstrap

from iw_rag.domain.acl import AccessContext  # noqa: E402
from iw_rag.learn.lab_corpus import CHUNKS, PRINCIPALS, QUERIES  # noqa: E402
from iw_rag.retrieval.acl_filter import acl_predicate  # noqa: E402
from iw_rag.retrieval.dense import DenseRetriever, _vec_literal  # noqa: E402
from iw_rag.retrieval.encode import InfinityQueryEncoder  # noqa: E402
from iw_rag.retrieval.fusion import RankedHit, RRFConfig, reciprocal_rank_fusion  # noqa: E402
from iw_rag.retrieval.rerank import InfinityReranker, RerankCandidate  # noqa: E402
from iw_rag.retrieval.sparse import SparseRetriever, _bm25_query  # noqa: E402
from iw_rag.settings import DatabaseSettings  # noqa: E402
from iw_rag.storage.db import Database  # noqa: E402
from iw_rag.storage.repositories import ChunkRepository, DocumentRepository, NewChunk  # noqa: E402
from tests.integration._alembic_rebase import rebased_alembic_config  # noqa: E402
from tests.integration.conftest import PARADEDB_IMAGE  # noqa: E402

OUT = ROOT / "src/iw_rag/learn/traces/hybrid_search.json"
EMBED_MODEL = "BAAI/bge-m3"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
CANDIDATE_K = 40
RRF_K = 60


async def _seed(db: Database, encoder: InfinityQueryEncoder) -> dict[str, str]:
    """Insert the corpus; return ``{db uuid: lab chunk id}``."""
    uuid_to_lab: dict[str, str] = {}
    async with db.session() as session:
        docs, chunks = DocumentRepository(session), ChunkRepository(session)
        doc_ids: dict[str, uuid.UUID] = {}
        for ordinal, chunk in enumerate(CHUNKS):
            if chunk.document not in doc_ids:
                doc_ids[chunk.document] = await docs.upsert_document(
                    source_uri=f"learn://{chunk.document}", checksum=chunk.document
                )
            (new_id,) = await chunks.bulk_upsert_chunks(
                [
                    NewChunk(
                        document_id=doc_ids[chunk.document],
                        ordinal=ordinal,
                        content=chunk.text,
                        dense_embedding=await encoder.encode(chunk.text),
                        acl_tags=list(chunk.acl_tags),
                    )
                ]
            )
            uuid_to_lab[str(new_id)] = chunk.chunk_id
        await session.commit()
    return uuid_to_lab


async def _record_run(
    db: Database,
    encoder: InfinityQueryEncoder,
    reranker: InfinityReranker,
    uuid_to_lab: dict[str, str],
    query: str,
    groups: tuple[str, ...],
) -> dict[str, Any]:
    ctx = AccessContext(principal_id="lab", principal_groups=groups)
    acl = acl_predicate(ctx)
    qvec = await encoder.encode(query)
    async with db.session() as session:
        dense = await DenseRetriever().retrieve(session, qvec, acl, CANDIDATE_K)
        sparse = await SparseRetriever().retrieve(session, query, acl, CANDIDATE_K)
        dist = {
            r[0]: float(r[1])
            for r in await session.execute(
                text(
                    "SELECT id::text, dense_embedding <=> CAST(:q AS vector) FROM iw_rag.chunks "
                    "WHERE acl_tags && :g"
                ),
                {"q": _vec_literal(qvec), "g": list(groups)},
            )
        }
        bm25 = _bm25_query(query)
        score = (
            {
                r[0]: float(r[1])
                for r in await session.execute(
                    text(
                        "SELECT id::text, paradedb.score(id) FROM iw_rag.chunks "
                        "WHERE id @@@ :b AND acl_tags && :g"
                    ),
                    {"b": bm25, "g": list(groups)},
                )
            }
            if bm25
            else {}
        )
        visible = [
            str(r[0])
            for r in await session.execute(
                text("SELECT id::text FROM iw_rag.chunks WHERE acl_tags && :g"), {"g": list(groups)}
            )
        ]
    # Fuse under the lab's stable ids: RRF breaks exact ties on chunk_id, and the database
    # uuids differ on every recording, which would make tied chunks swap places run to run.
    lab = uuid_to_lab
    fused = reciprocal_rank_fusion(
        [
            [RankedHit(lab[h.chunk_id], h.rank) for h in dense],
            [RankedHit(lab[h.chunk_id], h.rank) for h in sparse],
        ],
        RRFConfig(k=RRF_K),
    )
    cands = [RerankCandidate(chunk_id=cid, content=_text(uuid_to_lab[cid])) for cid in visible]
    scored = await reranker.rerank(query, cands, len(cands)) if cands else []
    return {
        "groups": list(groups),
        "visible": sorted(lab[c] for c in visible),
        "dense": [
            {"id": lab[h.chunk_id], "rank": h.rank, "distance": dist[h.chunk_id]} for h in dense
        ],
        "sparse": [
            {"id": lab[h.chunk_id], "rank": h.rank, "bm25": score[h.chunk_id]} for h in sparse
        ],
        "fused": [{"id": h.chunk_id, "rrf": h.score} for h in fused],
        "reranked": [{"id": lab[s.chunk_id], "score": s.score} for s in scored],
    }


def _git_sha() -> str:
    out = subprocess.run(  # noqa: S603 - fixed argv, no shell
        ["git", "rev-parse", "--short", "HEAD"],  # noqa: S607
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=False,
    )
    return out.stdout.strip()


def _text(lab_id: str) -> str:
    return next(c.text for c in CHUNKS if c.chunk_id == lab_id)


async def _main() -> None:
    from testcontainers.postgres import PostgresContainer

    from alembic import command

    infinity = os.environ.get("INFINITY_URL", "http://localhost:7997")
    encoder = InfinityQueryEncoder(infinity, EMBED_MODEL)
    reranker = InfinityReranker(infinity, RERANK_MODEL)
    with PostgresContainer(image=PARADEDB_IMAGE, username="t", password="t", dbname="t") as pg:
        dsn = pg.get_connection_url().replace("postgresql+psycopg2://", "postgresql+psycopg://")
        with rebased_alembic_config(dsn) as rebased:
            command.upgrade(rebased.cfg, "head")
        db = Database(DatabaseSettings(dsn=dsn))
        try:
            uuid_to_lab = await _seed(db, encoder)
            queries = []
            for q in QUERIES:
                runs = {
                    p.principal_id: await _record_run(
                        db, encoder, reranker, uuid_to_lab, q.text, p.groups
                    )
                    for p in PRINCIPALS
                }
                queries.append({"id": q.query_id, "runs": runs})
        finally:
            await db.dispose()
    sha = await asyncio.to_thread(_git_sha)
    trace = {
        "recorded": date.today().isoformat(),
        "git_commit": sha,
        "embedder": EMBED_MODEL,
        "reranker": RERANK_MODEL,
        "candidate_k": CANDIDATE_K,
        "rrf_k": RRF_K,
        "embedding_note": (
            "Chunks were embedded as pooled vectors via the served endpoint, without late chunking."
        ),
        "queries": queries,
    }
    OUT.write_text(json.dumps(trace, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} ({len(queries)} queries x {len(PRINCIPALS)} principals)")


if __name__ == "__main__":
    asyncio.run(_main())
