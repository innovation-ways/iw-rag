"""Integration tests for the ingestion pipeline (testcontainers + faked embedder).

The heavy BGE-M3 model is faked with a deterministic 1024-d embedder; parsing
uses the real TextParser over the committed markdown fixture corpus; chunking
uses a deterministic char tokenizer. Verifies persistence shape, ACL tags,
parent links, idempotency, changed-checksum/supersede, and atomic rollback.
"""

from __future__ import annotations

from collections.abc import Generator, Sequence
from pathlib import Path

import pytest
import sqlalchemy
from sqlalchemy import text

from iw_rag.domain.errors import EmbedError
from iw_rag.ingestion.chunk import ChunkConfig, ChunkNode, HybridChunker
from iw_rag.ingestion.embed import ChunkEmbedding
from iw_rag.ingestion.enrich import IngestContext
from iw_rag.ingestion.parse import ParseConfig, ParseHints, SourceRef, select_parser
from iw_rag.ingestion.pipeline import IngestionConfig, IngestionPipeline
from iw_rag.settings import DatabaseSettings
from iw_rag.storage.db import Database
from scripts.ingest_corpus import ManifestAclResolver, build_sources, load_manifest

PARADEDB_IMAGE = "paradedb/paradedb:latest"
FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "corpus"
FIXTURE_MANIFEST = FIXTURE_DIR / "manifest.yaml"


class CharTokenizer:
    """Deterministic char-level tokenizer (exact round-trip)."""

    def encode(self, text_: str) -> list[int]:
        return [ord(c) for c in text_]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(i) for i in ids)


class FakeEmbedder:
    """Deterministic 1024-d embedder; optionally fails to test rollback."""

    def __init__(self, *, fail: bool = False) -> None:
        self._fail = fail

    async def embed_document(
        self, doc_token_ids: list[int], leaves: Sequence[ChunkNode]
    ) -> list[ChunkEmbedding]:
        if self._fail:
            raise EmbedError("injected embed failure")
        return [ChunkEmbedding(leaf.local_id, [0.1] * 1024, {}) for leaf in leaves]

    async def health(self) -> bool:
        return True


class FixedResolver:
    """ACL resolver returning a fixed tag set (source-derived stand-in)."""

    def __init__(self, tags: list[str]) -> None:
        self._tags = tags

    def resolve(self, source: SourceRef) -> list[str]:  # noqa: ARG002
        return list(self._tags)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def migrated_dsn() -> Generator[str, None, None]:
    """A migrated paradedb container; yields its psycopg DSN."""
    from testcontainers.postgres import PostgresContainer

    from alembic import command
    from tests.integration._alembic_rebase import rebased_alembic_config

    with PostgresContainer(image=PARADEDB_IMAGE, username="t", password="t", dbname="t") as pg:
        dsn = pg.get_connection_url().replace("postgresql+psycopg2://", "postgresql+psycopg://")
        with rebased_alembic_config(dsn) as rebased:
            command.upgrade(rebased.cfg, "head")
        yield dsn


@pytest.fixture
def clean_dsn(migrated_dsn: str) -> str:
    """Truncate iw_rag tables before each test for isolation."""
    engine = sqlalchemy.create_engine(migrated_dsn)
    with engine.begin() as conn:
        conn.execute(
            text(
                "TRUNCATE iw_rag.documents, iw_rag.chunks, iw_rag.ingestion_runs "
                "RESTART IDENTITY CASCADE"
            )
        )
    engine.dispose()
    return migrated_dsn


def _pipeline(dsn: str, *, fail_embed: bool = False, supersede: bool = False) -> IngestionPipeline:
    return IngestionPipeline(
        parser_factory=lambda s: select_parser(s, ParseConfig()),
        chunker=HybridChunker(ChunkConfig(leaf_tokens=64), CharTokenizer()),
        embedder=FakeEmbedder(fail=fail_embed),
        db=Database(DatabaseSettings(dsn=dsn)),
        cfg=IngestionConfig(concurrency=2, supersede=supersede),
    )


def _count(dsn: str, sql: str) -> int:
    engine = sqlalchemy.create_engine(dsn)
    try:
        with engine.connect() as conn:
            return int(conn.execute(text(sql)).scalar_one())
    finally:
        engine.dispose()


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


@pytest.mark.integration
async def test_end_to_end_ingest(clean_dsn: str) -> None:
    entries = load_manifest(FIXTURE_MANIFEST)
    sources = build_sources(entries, FIXTURE_DIR)
    pipe = _pipeline(clean_dsn)
    ctx = IngestContext(acl_resolver=ManifestAclResolver(entries), corpus_label="test")

    result = await pipe.ingest_corpus(sources, ctx)

    assert [r.status for r in result.results] == ["ingested"] * 3
    # every chunk has a 1024-d vector
    assert _count(clean_dsn, "SELECT count(*) FROM iw_rag.chunks") > 0
    assert (
        _count(
            clean_dsn,
            "SELECT count(*) FROM iw_rag.chunks WHERE vector_dims(dense_embedding) <> 1024",
        )
        == 0
    )
    # there are both parents (parent_id IS NULL) and leaves (parent_id NOT NULL)
    assert _count(clean_dsn, "SELECT count(*) FROM iw_rag.chunks WHERE parent_id IS NULL") >= 3
    assert _count(clean_dsn, "SELECT count(*) FROM iw_rag.chunks WHERE parent_id IS NOT NULL") > 0
    # every leaf's parent_id resolves to a real chunk row
    orphan_leaves = _count(
        clean_dsn,
        "SELECT count(*) FROM iw_rag.chunks c WHERE c.parent_id IS NOT NULL "
        "AND NOT EXISTS (SELECT 1 FROM iw_rag.chunks p WHERE p.id = c.parent_id)",
    )
    assert orphan_leaves == 0


@pytest.mark.integration
async def test_acl_tags_match_manifest(clean_dsn: str) -> None:
    entries = load_manifest(FIXTURE_MANIFEST)
    sources = build_sources(entries, FIXTURE_DIR)
    pipe = _pipeline(clean_dsn)
    await pipe.ingest_corpus(
        sources, IngestContext(acl_resolver=ManifestAclResolver(entries), corpus_label="t")
    )
    engine = sqlalchemy.create_engine(clean_dsn)
    try:
        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT d.source_uri, array_agg(DISTINCT t) "
                    "FROM iw_rag.documents d JOIN iw_rag.chunks c ON c.document_id = d.id "
                    "LEFT JOIN LATERAL unnest(c.acl_tags) t ON true GROUP BY d.source_uri"
                )
            ).all()
    finally:
        engine.dispose()
    tags_by_uri = {uri: set(tags) for uri, tags in rows}
    assert tags_by_uri["corpus://hr/leave_policy.md"] == {"hr"}
    assert tags_by_uri["corpus://finance/invoice_q3.md"] == {"finance"}
    assert tags_by_uri["corpus://public/intro.md"] == {"public"}


@pytest.mark.integration
async def test_idempotent_reingest(clean_dsn: str) -> None:
    entries = load_manifest(FIXTURE_MANIFEST)
    sources = build_sources(entries, FIXTURE_DIR)
    ctx = IngestContext(acl_resolver=ManifestAclResolver(entries), corpus_label="t")

    await _pipeline(clean_dsn).ingest_corpus(sources, ctx)
    chunks_after_first = _count(clean_dsn, "SELECT count(*) FROM iw_rag.chunks")
    second = await _pipeline(clean_dsn).ingest_corpus(sources, ctx)

    assert [r.status for r in second.results] == ["skipped_unchanged"] * 3
    assert _count(clean_dsn, "SELECT count(*) FROM iw_rag.chunks") == chunks_after_first


@pytest.mark.integration
async def test_changed_checksum_creates_new_doc(clean_dsn: str, tmp_path: Path) -> None:
    doc = tmp_path / "note.md"
    doc.write_text("# Note\n\noriginal body\n", encoding="utf-8")
    src = SourceRef(uri="corpus://note.md", mime="text/markdown", hints=ParseHints(), path=doc)
    ctx = IngestContext(acl_resolver=FixedResolver(["x"]), corpus_label="t")

    await _pipeline(clean_dsn).ingest_document(src, ctx)
    doc.write_text("# Note\n\nCHANGED body\n", encoding="utf-8")
    res = await _pipeline(clean_dsn).ingest_document(src, ctx)

    assert res.status == "ingested"
    # default supersede=false → both versions retained (two docs, same uri)
    assert (
        _count(
            clean_dsn, "SELECT count(*) FROM iw_rag.documents WHERE source_uri='corpus://note.md'"
        )
        == 2
    )

    # supersede=true → re-ingesting a third change leaves only the newest doc
    doc.write_text("# Note\n\nTHIRD body\n", encoding="utf-8")
    await _pipeline(clean_dsn, supersede=True).ingest_document(src, ctx)
    assert (
        _count(
            clean_dsn, "SELECT count(*) FROM iw_rag.documents WHERE source_uri='corpus://note.md'"
        )
        == 1
    )


@pytest.mark.integration
async def test_embed_failure_rolls_back(clean_dsn: str, tmp_path: Path) -> None:
    doc = tmp_path / "x.md"
    doc.write_text("# X\n\nbody text here\n", encoding="utf-8")
    src = SourceRef(uri="corpus://x.md", mime="text/markdown", hints=ParseHints(), path=doc)
    ctx = IngestContext(acl_resolver=FixedResolver(["x"]), corpus_label="t")

    res = await _pipeline(clean_dsn, fail_embed=True).ingest_document(src, ctx)

    assert res.status == "failed"
    assert "embed" in (res.error or "").lower()
    # zero rows persisted for the document (atomic: embed precedes any write)
    assert (
        _count(clean_dsn, "SELECT count(*) FROM iw_rag.documents WHERE source_uri='corpus://x.md'")
        == 0
    )
    assert _count(clean_dsn, "SELECT count(*) FROM iw_rag.chunks") == 0


@pytest.mark.integration
async def test_parse_failure_maps_to_failed(clean_dsn: str) -> None:
    src = SourceRef(
        uri="corpus://missing.md",
        mime="text/markdown",
        hints=ParseHints(),
        path=Path("/no/such/file.md"),
    )
    ctx = IngestContext(acl_resolver=FixedResolver(["x"]), corpus_label="t")
    res = await _pipeline(clean_dsn).ingest_document(src, ctx)
    assert res.status == "failed"
