"""`make ingest` entry point — manifest-driven offline corpus ingestion.

Reads a corpus manifest (the source-system metadata that drives ACLs — never
content-inferred), builds the real BGE-M3 Late-Chunking pipeline, and ingests
every document with bounded concurrency, printing per-document results. The same
``IngestionPipeline.ingest_corpus`` the ``/ingest`` route will call.

Usage:
    uv run --group ingest python scripts/ingest_corpus.py [MANIFEST]
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from iw_rag.ingestion.chunk import ChunkConfig, HybridChunker
from iw_rag.ingestion.embed import BgeM3Tokenizer, EmbedConfig, make_bge_m3_embedder
from iw_rag.ingestion.enrich import IngestContext
from iw_rag.ingestion.parse import ParseConfig, ParseHints, SourceRef, select_parser
from iw_rag.ingestion.pipeline import IngestionConfig, IngestionPipeline
from iw_rag.settings import get_settings
from iw_rag.storage.db import Database

DEFAULT_MANIFEST = Path("data/corpus/manifest.yaml")


@dataclass(frozen=True, slots=True)
class ManifestEntry:
    """One manifest document record (source-system metadata).

    Attributes:
        uri: Stable logical id.
        path: File path relative to the manifest directory.
        mime: MIME type for parser routing.
        acl_groups: Entitlement groups (the ACL authority — not content-derived).
        hard_doc: Force escalation parser when True.
    """

    uri: str
    path: str
    mime: str
    acl_groups: list[str]
    hard_doc: bool = False


def load_manifest(manifest_path: Path) -> list[ManifestEntry]:
    """Load and validate a corpus manifest into typed entries.

    Args:
        manifest_path: Path to ``manifest.yaml``.

    Returns:
        The manifest's document entries.
    """
    raw: Any = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    docs = raw.get("documents", []) if isinstance(raw, dict) else []
    return [
        ManifestEntry(
            uri=d["uri"],
            path=d["path"],
            mime=d.get("mime", "text/markdown"),
            acl_groups=list(d.get("acl_groups", [])),
            hard_doc=bool(d.get("hard_doc", False)),
        )
        for d in docs
    ]


class ManifestAclResolver:
    """ACL resolver backed by a manifest (uri → group tags). Source-derived only."""

    def __init__(self, entries: list[ManifestEntry]) -> None:
        """Index manifest entries by uri."""
        self._by_uri = {e.uri: e.acl_groups for e in entries}

    def resolve(self, source: SourceRef) -> list[str]:
        """Return the source's group tags from the manifest (empty if unknown)."""
        return list(self._by_uri.get(source.uri, []))


def build_sources(entries: list[ManifestEntry], base_dir: Path) -> list[SourceRef]:
    """Build ``SourceRef``s from manifest entries, resolving paths under base_dir."""
    return [
        SourceRef(
            uri=e.uri,
            mime=e.mime,
            hints=ParseHints(hard_doc=e.hard_doc or None),
            path=base_dir / e.path,
        )
        for e in entries
    ]


async def main(manifest_path: Path) -> int:
    """Ingest the corpus described by ``manifest_path``; print results."""
    settings = get_settings()
    entries = load_manifest(manifest_path)
    sources = build_sources(entries, manifest_path.parent)
    resolver = ManifestAclResolver(entries)

    tokenizer = BgeM3Tokenizer()
    chunker = HybridChunker(ChunkConfig(leaf_tokens=settings.ingestion.leaf_tokens), tokenizer)
    embedder = make_bge_m3_embedder(
        EmbedConfig(window_tokens=settings.ingestion.embed_window_tokens)
    )
    db = Database(settings.database)
    parse_cfg = ParseConfig(force_parser=settings.ingestion.force_parser)

    pipeline = IngestionPipeline(
        parser_factory=lambda s: select_parser(s, parse_cfg),
        chunker=chunker,
        embedder=embedder,
        db=db,
        cfg=IngestionConfig(
            concurrency=settings.ingestion.concurrency,
            supersede=settings.ingestion.supersede,
        ),
    )
    ctx = IngestContext(acl_resolver=resolver, corpus_label="demo")
    try:
        result = await pipeline.ingest_corpus(sources, ctx)
    finally:
        await db.dispose()

    failures = 0
    for src, res in zip(sources, result.results, strict=True):
        print(f"  {res.status:18} {src.uri}  (chunks={res.chunk_count}) {res.error or ''}")
        failures += res.status == "failed"
    print(f"run {result.ingestion_run_id}: {len(result.results)} docs, {failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_MANIFEST
    raise SystemExit(asyncio.run(main(path)))
