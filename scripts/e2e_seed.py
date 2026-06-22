"""Seed the iw-rag E2E browser-verification database.

Run inside the E2E ``app`` container (``docker compose exec app uv run python
scripts/e2e_seed.py``) after migrations. Two phases:

1. **Base seed (always)** — one public document + chunk whose dense embedding
   is the constant vector the model stub (``e2e_model_stub.py``) returns for
   every query, so cosine distance is 0 and *any* query retrieves it. This is
   what lets a browser-driven question produce a grounded, cited answer with
   no GPU. ``acl_tags=["public"]`` + ``visibility="public"`` so any demo user
   can both retrieve and download it.

2. **Per-item fixtures (optional)** — when ``IW_ITEM_ID`` is set, load every
   ``ai-dev/active/<ITEM>/e2e_fixtures/*.py`` in lexical order and call its
   documented sync ``seed(db: Session)`` entrypoint, so a qv-browser agent can
   add item-specific rows (e.g. benchmark reports) that this stack will serve.

Idempotent: safe to re-run (the prompt re-runs it after writing fixtures).
"""

from __future__ import annotations

import asyncio
import importlib.util
import os
import sys
from pathlib import Path

# Constant 1024-d embedding — MUST equal CONST_EMBEDDING in e2e_model_stub.py.
EMBED_DIM = 1024
E2E_SEED_VECTOR = [0.1] * EMBED_DIM

# A document whose chunk content answers "What is iw-rag?" — quoted closely by
# the stub generator's canned answer so the grounded path holds end to end.
_DOC_URI = "corpus://public/iw-rag-overview.md"
_DOC_CHECKSUM = "e2e-iw-rag-overview-v1"
_CHUNK_TEXT = (
    "iw-rag is a fully-local, single-GPU, open-weight agentic RAG system with "
    "citations, ACL, and a verification layer that fails closed. It runs entirely "
    "on one RTX 5090 and never calls a hosted or cloud API."
)


async def _base_seed(dsn: str) -> None:
    """Insert the public iw-rag overview document + chunk (idempotent)."""
    from sqlalchemy import update

    from iw_rag.settings import DatabaseSettings
    from iw_rag.storage import schema
    from iw_rag.storage.db import Database
    from iw_rag.storage.repositories import ChunkRepository, DocumentRepository, NewChunk

    database = Database(DatabaseSettings(dsn=dsn))
    try:
        async with database.session() as session:
            docs = DocumentRepository(session)
            chunks = ChunkRepository(session)
            doc_id = await docs.upsert_document(
                source_uri=_DOC_URI,
                checksum=_DOC_CHECKSUM,
                title="iw-rag overview",
            )
            # Make the document publicly downloadable (CR-00098 get_raw ACL).
            await session.execute(
                update(schema.documents)
                .where(schema.documents.c.id == doc_id)
                .values(owner_id="public", visibility="public")
            )
            await chunks.bulk_upsert_chunks(
                [
                    NewChunk(
                        document_id=doc_id,
                        ordinal=0,
                        content=_CHUNK_TEXT,
                        dense_embedding=E2E_SEED_VECTOR,
                        acl_tags=["public"],
                    )
                ]
            )
            await session.commit()
        print(f"[e2e_seed] base: document {doc_id} + 1 public chunk seeded")
    finally:
        await database.dispose()


def _load_item_fixtures(dsn: str, item_id: str) -> None:
    """Load and run each ``e2e_fixtures/*.py``'s sync ``seed(db)`` for the item."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session

    fixtures_dir = Path("ai-dev/active") / item_id / "e2e_fixtures"
    if not fixtures_dir.is_dir():
        print(f"[e2e_seed] no fixtures dir for {item_id} ({fixtures_dir}) — skipping")
        return

    fixture_files = sorted(p for p in fixtures_dir.glob("*.py") if p.name != "__init__.py")
    if not fixture_files:
        print(f"[e2e_seed] no *.py fixtures in {fixtures_dir} — skipping")
        return

    engine = create_engine(dsn, pool_pre_ping=True)
    try:
        with Session(engine) as db:
            for path in fixture_files:
                spec = importlib.util.spec_from_file_location(f"e2e_fixture_{path.stem}", path)
                if spec is None or spec.loader is None:
                    print(f"[e2e_seed] WARN: cannot load {path} — skipping")
                    continue
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                seed_fn = getattr(module, "seed", None)
                if seed_fn is None:
                    print(f"[e2e_seed] WARN: {path.name} has no seed(db) — skipping")
                    continue
                seed_fn(db)
                print(f"[e2e_seed] fixture {path.name} applied")
            db.commit()
    finally:
        engine.dispose()


def main() -> None:
    dsn = os.environ.get("IW_RAG__DATABASE__DSN", "").strip()
    if not dsn:
        print("[e2e_seed] ERROR: IW_RAG__DATABASE__DSN is not set", file=sys.stderr)
        raise SystemExit(1)

    asyncio.run(_base_seed(dsn))

    item_id = os.environ.get("IW_ITEM_ID", "").strip()
    if item_id:
        _load_item_fixtures(dsn, item_id)
    else:
        print("[e2e_seed] IW_ITEM_ID unset — base seed only (no item fixtures)")


if __name__ == "__main__":
    main()
