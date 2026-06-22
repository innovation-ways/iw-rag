"""Offline graph indexer entrypoint — NEVER reachable from /query.

OFFLINE / batch only.  This script is the **only** way graph indexing runs.
It must be scheduled when the generator (Qwen3-14B) is idle — never triggered
by a ``/query`` request (GPU-contention rule, R-00106/F2, CLAUDE.md two-clocks).

Usage::

    uv run python scripts/index_graph.py [--corpus LABEL] [--incremental]

Environment:
    IW_RAG__DATABASE__DSN            — Postgres DSN (required).
    IW_RAG__MODELS__*__ENDPOINT      — Model endpoint vars (required for hot path,
                                       optional here — extraction uses batch endpoint).
    IW_RAG__MODES__GRAPHRAG__ENABLED — Must be true to run indexing.

Pipeline:
    1. Load Settings (models.yaml → graphrag-extract → Qwen3-4B batch endpoint).
    2. Fetch all chunks from ``iw_rag.chunks`` (or a corpus-label-filtered subset).
    3. Run GraphIndexer.index_chunks (incremental dedup on source_uri+checksum).
    4. Report summary (chunks indexed / skipped / failed).

References:
- task-4a-graphrag-route §6 (offline entrypoint)
- lld-graphrag §1 (two-clocks rule)
- CLAUDE.md (GPU-contention rule)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

_log = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for index_graph.

    Returns:
        Configured :class:`argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        prog="index_graph",
        description="Offline GraphRAG indexer — extracts entities/relations into AGE. "
        "NEVER run from /query (two-clocks rule).",
    )
    parser.add_argument(
        "--corpus",
        default=None,
        help="Corpus label to filter (default: index all chunks).",
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        default=True,
        help="Skip chunks already indexed in this session (default: True).",
    )
    parser.add_argument(
        "--no-incremental",
        dest="incremental",
        action="store_false",
        help="Force re-extraction of all chunks (ignores dedup).",
    )
    parser.add_argument(
        "--graph-name",
        default="iw_rag_graph",
        help="AGE graph name (default: iw_rag_graph).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level (default: INFO).",
    )
    return parser


async def _run(args: argparse.Namespace) -> int:
    """Main async entry point — loads config, fetches chunks, runs indexer.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Exit code: 0 on success, 1 on configuration error.
    """
    # Lazy imports — keeps the script importable without the full dep tree
    from iw_rag.modes.graphrag.indexer import GraphIndexer
    from iw_rag.modes.graphrag.store import AgeGraphStore, GraphStoreUnavailable
    from iw_rag.modes.graphrag.types import GraphIndexConfig
    from iw_rag.settings import get_settings

    try:
        cfg = get_settings()
    except Exception as exc:
        _log.error("Failed to load settings: %s", exc)
        return 1

    # Build the extraction model client (batch endpoint, NOT the hot-path generator)
    try:
        graphrag_entry = cfg.models.get("graphrag-extract")
    except Exception as exc:
        _log.error("Model 'graphrag-extract' not found in config/models.yaml: %s", exc)
        return 1

    # Build a minimal LLM client for the batch extraction endpoint
    from iw_rag.generation.llm import VllmClient

    llm_client = VllmClient(endpoint=graphrag_entry.endpoint)

    index_cfg = GraphIndexConfig(
        extraction_model=graphrag_entry.served_name,
        skip_community_summaries=True,  # always skip (cost, R-00106/F5)
    )

    store = AgeGraphStore(dsn=cfg.database.dsn, graph_name=args.graph_name)

    try:
        indexer = GraphIndexer(store=store, cfg=index_cfg, llm_client=llm_client)
    except GraphStoreUnavailable as exc:
        _log.error("GraphRAG store unavailable (AGE missing?): %s", exc)
        await store.aclose()
        return 1

    # Fetch chunks from Postgres
    from iw_rag.settings import DatabaseSettings
    from iw_rag.storage.db import Database

    db = Database(DatabaseSettings(dsn=cfg.database.dsn))
    try:
        async with db.session() as session:
            from sqlalchemy import select

            from iw_rag.storage import schema

            stmt = select(
                schema.chunks.c.id,
                schema.chunks.c.content,
                schema.chunks.c.acl_tags,
                schema.documents.c.source_uri,
                schema.documents.c.checksum,
            ).select_from(
                schema.chunks.join(
                    schema.documents,
                    schema.documents.c.id == schema.chunks.c.document_id,
                )
            )
            if args.corpus:
                stmt = stmt.where(schema.ingestion_runs.c.corpus_label == args.corpus)

            result = await session.execute(stmt)
            chunks = [
                {
                    "chunk_id": str(row[0]),
                    "content": row[1],
                    "acl_tags": row[2] or [],
                    "source_uri": row[3],
                    "checksum": row[4],
                }
                for row in result
            ]

        _log.info(
            "index_graph: fetched %d chunks from Postgres (corpus=%s)",
            len(chunks),
            args.corpus or "all",
        )

        report = await indexer.index_chunks(chunks, incremental=args.incremental)
        _log.info(
            "index_graph: done — indexed=%d skipped=%d failed=%d",
            report.chunks_indexed,
            report.chunks_skipped,
            report.chunks_failed,
        )
    except GraphStoreUnavailable as exc:
        _log.error("GraphRAG store unavailable during indexing: %s", exc)
        return 1
    finally:
        await db.dispose()
        await store.aclose()

    return 0


def main() -> None:
    """CLI entry point — parses args and runs the async indexer."""
    parser = _build_parser()
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    sys.exit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
