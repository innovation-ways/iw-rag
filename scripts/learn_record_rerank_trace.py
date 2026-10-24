"""Record the Learn section's reranker lab from the REAL retrieval spine.

Starts a throwaway ParadeDB container (never the live database), ingests the lab documents
with the production chunker (leaf and parent chunks, parents embedded as the mean of their
leaves, as the ingestion pipeline does), then for every lab question:

* runs the production dense and BM25 arms and the production RRF, and scores every fused
  candidate with the served cross-encoder reranker;
* runs the production ``RetrievalSpine`` end to end in four configurations (reranker up with
  parent expansion, reranker up with leaf only, reranker down, sections not searchable) to
  record what it really returns.

The lab replays that file, and its tests check that the replay reproduces the recorded
spine output exactly.  Chunks get deterministic ids so RRF ties break the same way on every
recording.  Re-run after changing ``learn/rerank_corpus.py`` or the retrieval stack::

    INFINITY_URL=http://localhost:7997 uv run python scripts/learn_record_rerank_trace.py
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import subprocess
import sys
import uuid
from datetime import date
from pathlib import Path
from typing import Any

from sqlalchemy import insert, text

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # lets this script reuse the integration-test DB bootstrap

from iw_rag.domain.acl import AccessContext  # noqa: E402
from iw_rag.domain.errors import RerankError  # noqa: E402
from iw_rag.ingestion.chunk import ChunkConfig, HybridChunker  # noqa: E402
from iw_rag.ingestion.light_parse import CharTokenizer  # noqa: E402
from iw_rag.ingestion.parse import parse_markdown_text  # noqa: E402
from iw_rag.learn.rerank_corpus import DOCS, LEAF_CHARS, QUERIES  # noqa: E402
from iw_rag.retrieval.acl_filter import acl_predicate  # noqa: E402
from iw_rag.retrieval.dense import DenseRetriever  # noqa: E402
from iw_rag.retrieval.encode import InfinityQueryEncoder  # noqa: E402
from iw_rag.retrieval.fusion import RRFConfig, reciprocal_rank_fusion  # noqa: E402
from iw_rag.retrieval.rerank import InfinityReranker, RerankCandidate  # noqa: E402
from iw_rag.retrieval.sparse import SparseRetriever  # noqa: E402
from iw_rag.retrieval.spine import RetrievalConfig, RetrievalSpine  # noqa: E402
from iw_rag.settings import DatabaseSettings  # noqa: E402
from iw_rag.storage import schema  # noqa: E402
from iw_rag.storage.db import Database  # noqa: E402
from tests.integration._alembic_rebase import rebased_alembic_config  # noqa: E402
from tests.integration.conftest import PARADEDB_IMAGE  # noqa: E402

OUT = ROOT / "src/iw_rag/learn/traces/rerank.json"
EMBED_MODEL = "BAAI/bge-m3"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
NAMESPACE = uuid.UUID("5d1f0f6e-6f4e-4d0b-9a55-1d3b6f0c7e11")
GROUPS = ("public",)
CANDIDATE_K, FUSED_K, TOP_K, RRF_K = 40, 80, 8, 60


class DownReranker:
    """A reranker that is always unavailable, to record the spine's degraded path."""

    async def rerank(self, query: str, candidates: Any, top_n: int) -> Any:  # noqa: ARG002
        raise RerankError("recorded outage")

    async def health(self) -> bool:
        return False


def _unit_mean(vectors: list[list[float]]) -> list[float]:
    mean = [sum(col) / len(vectors) for col in zip(*vectors, strict=True)]
    norm = math.sqrt(sum(x * x for x in mean)) or 1.0
    return [x / norm for x in mean]


async def _seed(db: Database, encoder: InfinityQueryEncoder) -> dict[str, Any]:
    """Ingest the lab documents; return the chunk structure keyed by deterministic uuid."""
    structure: dict[str, Any] = {"parents": [], "leaves": []}
    leaf_n = 0
    async with db.session() as session:
        for doc_n, doc in enumerate(DOCS):
            tree = HybridChunker(ChunkConfig(leaf_tokens=LEAF_CHARS), CharTokenizer()).chunk(
                parse_markdown_text(doc.text, f"learn://{doc.doc_id}")
            )
            doc_uuid = uuid.uuid5(NAMESPACE, f"doc:{doc.doc_id}")
            await session.execute(
                text(
                    "INSERT INTO iw_rag.documents "
                    "(id, source_uri, checksum, content_type, metadata) "
                    "VALUES (:id, :uri, :sum, 'document', '{}'::jsonb)"
                ),
                {"id": doc_uuid, "uri": f"learn://{doc.doc_id}", "sum": doc.doc_id},
            )
            leaf_vecs: dict[str, list[float]] = {}
            for leaf in tree.leaves:
                leaf_vecs[leaf.local_id] = await encoder.encode(leaf.content)
            parent_ids: dict[str, uuid.UUID] = {}
            rows: list[dict[str, Any]] = []
            for p_ord, parent in enumerate(tree.parents):
                kids = [
                    leaf_vecs[k.local_id]
                    for k in tree.leaves
                    if k.parent_local_id == parent.local_id
                ]
                pid = uuid.uuid5(NAMESPACE, f"{doc.doc_id}:{parent.local_id}")
                parent_ids[parent.local_id] = pid
                lab = f"S{len(structure['parents']) + 1}"
                structure["parents"].append(
                    {
                        "id": lab,
                        "uuid": str(pid),
                        "doc": doc.doc_id,
                        "title": doc.title,
                        "section": parent.section_path or "(no heading)",
                        "text": parent.content,
                    }
                )
                rows.append(
                    _row(
                        pid,
                        doc_uuid,
                        None,
                        p_ord,
                        parent.content,
                        _unit_mean(kids),
                        parent.section_path,
                    )
                )
            for leaf in tree.leaves:
                leaf_n += 1
                lid = uuid.uuid5(NAMESPACE, f"{doc.doc_id}:{leaf.local_id}")
                lab_parent = next(
                    p["id"]
                    for p in structure["parents"]
                    if p["uuid"] == str(parent_ids[leaf.parent_local_id or ""])
                )
                structure["leaves"].append(
                    {
                        "id": f"L{leaf_n:02d}",
                        "uuid": str(lid),
                        "parent": lab_parent,
                        "doc": doc.doc_id,
                        "section": leaf.section_path or "(no heading)",
                        "text": leaf.content,
                    }
                )
                rows.append(
                    _row(
                        lid,
                        doc_uuid,
                        parent_ids[leaf.parent_local_id or ""],
                        leaf.ordinal,
                        leaf.content,
                        leaf_vecs[leaf.local_id],
                        leaf.section_path,
                    )
                )
            _ = doc_n
            await session.execute(insert(schema.chunks), rows)
        await session.commit()
    return structure


def _row(
    cid: uuid.UUID,
    doc: uuid.UUID,
    parent: uuid.UUID | None,
    ordinal: int,
    content: str,
    vec: list[float],
    section: str | None,
) -> dict[str, Any]:
    return {
        "id": cid,
        "document_id": doc,
        "parent_id": parent,
        "ordinal": ordinal,
        "content": content,
        "context_prefix": None,
        "dense_embedding": vec,
        "sparse_weights": None,
        "content_type": "text",
        "section_path": section,
        "acl_tags": list(GROUPS),
        "metadata": {},
    }


async def _record_query(
    db: Database,
    encoder: InfinityQueryEncoder,
    reranker: InfinityReranker,
    lab: dict[str, str],
    contents: dict[str, str],
    query: str,
) -> dict[str, Any]:
    acl = acl_predicate(AccessContext(principal_id="lab", principal_groups=GROUPS))
    qvec = await encoder.encode(query)
    async with db.session() as session:
        dense = await DenseRetriever().retrieve(session, qvec, acl, CANDIDATE_K)
        sparse = await SparseRetriever().retrieve(session, query, acl, CANDIDATE_K)
    fused = reciprocal_rank_fusion([dense, sparse], RRFConfig(k=RRF_K))[:FUSED_K]
    dense_rank = {h.chunk_id: h.rank for h in dense}
    sparse_rank = {h.chunk_id: h.rank for h in sparse}
    scored = await reranker.rerank(
        query,
        [RerankCandidate(chunk_id=f.chunk_id, content=contents[f.chunk_id]) for f in fused],
        len(fused),
    )
    score = {s.chunk_id: s.score for s in scored}
    candidates = [
        {
            "uuid": f.chunk_id,
            "id": lab[f.chunk_id],
            "rrf": f.score,
            "dense_rank": dense_rank.get(f.chunk_id),
            "sparse_rank": sparse_rank.get(f.chunk_id),
            "rerank": score[f.chunk_id],
        }
        for f in fused
    ]
    spine: dict[str, Any] = {}
    runs = (
        ("parent_up", reranker, {"expansion_mode": "parent"}),
        ("leaf_up", reranker, {"expansion_mode": "leaf"}),
        ("parent_down", DownReranker(), {"expansion_mode": "parent"}),
        ("sections_off", reranker, {"expansion_mode": "parent", "section_hits": False}),
    )
    for name, rer, extra in runs:
        engine = RetrievalSpine(
            db,
            encoder,
            DenseRetriever(),
            SparseRetriever(),
            rer,  # type: ignore[arg-type]
            RetrievalConfig(
                candidate_k=CANDIDATE_K, fused_k=FUSED_K, top_k=TOP_K, rrf_k=RRF_K, **extra
            ),
        )
        result = await engine.retrieve(
            query, AccessContext(principal_id="lab", principal_groups=GROUPS)
        )
        spine[name] = {
            "degraded": result.diagnostics.get("rerank") == "degraded",
            "diagnostics": {
                k: v
                for k, v in result.diagnostics.items()
                if k in ("sections_set_aside", "repeats_merged")
            },
            "chunks": [
                {
                    "id": lab[c.chunk_id],
                    "chars": len(c.content),
                    "expanded": c.leaf_content is not None and c.content != c.leaf_content,
                    "merged": [lab[m] for m in c.metadata.get("merged_chunk_ids", [])],
                }
                for c in result.chunks
            ],
        }
    return {"candidates": candidates, "spine": spine}


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
            structure = await _seed(db, encoder)
            lab = {c["uuid"]: c["id"] for c in structure["parents"] + structure["leaves"]}
            contents = {c["uuid"]: c["text"] for c in structure["parents"] + structure["leaves"]}
            queries = [
                {
                    "id": q.query_id,
                    **await _record_query(db, encoder, reranker, lab, contents, q.text),
                }
                for q in QUERIES
            ]
        finally:
            await db.dispose()
    sha = await asyncio.to_thread(_git_sha)
    OUT.write_text(
        json.dumps(
            {
                "recorded": date.today().isoformat(),
                "git_commit": sha,
                "embedder": EMBED_MODEL,
                "reranker": RERANK_MODEL,
                "leaf_chars": LEAF_CHARS,
                "candidate_k": CANDIDATE_K,
                "fused_k": FUSED_K,
                "top_k": TOP_K,
                "rrf_k": RRF_K,
                "parent_budget_chars": 4000,
                "embedding_note": (
                    "Leaves were embedded one by one through the served endpoint, without late "
                    "chunking; each parent's vector is the normalised mean of its leaves."
                ),
                **structure,
                "queries": queries,
            },
            indent=1,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"wrote {OUT.relative_to(ROOT)} "
        f"({len(structure['leaves'])} leaves, {len(structure['parents'])} parents)"
    )


def _git_sha() -> str:
    out = subprocess.run(  # noqa: S603 - fixed argv, no shell
        ["git", "rev-parse", "--short", "HEAD"],  # noqa: S607
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=False,
    )
    return out.stdout.strip()


if __name__ == "__main__":
    asyncio.run(_main())
