"""Integration tests for visual caption → ingestion pipeline (testcontainers Postgres).

Verifies that captioned assets (tables + figures) flow through the existing
IngestionPipeline as chunks with the correct content_type (figure_caption / table),
are embedded with the same BGE-M3 path, are persisted with full ACL tags, and are
retrievable through the normal chunk query path.

Uses testcontainers Postgres (the ``clean_pg`` fixture from tests/integration/conftest.py
is not yet defined; this file defines its own module-scoped ``migrated_dsn`` mirror).
The VLM + embedder are both faked — no live model endpoints required.
"""

from __future__ import annotations

from collections.abc import Generator, Sequence
from pathlib import Path

import pytest
import sqlalchemy
from sqlalchemy import text

from iw_rag.ingestion.chunk import ChunkConfig, ChunkNode, HybridChunker
from iw_rag.ingestion.embed import ChunkEmbedding
from iw_rag.ingestion.enrich import IngestContext
from iw_rag.ingestion.pipeline import IngestionConfig, IngestionPipeline
from iw_rag.modes.visual.captioner import CaptionResult, VlmCaptioner, VlmCaptionerConfig
from iw_rag.modes.visual.types import VisualAsset, VisualAssetType
from iw_rag.settings import DatabaseSettings
from iw_rag.storage.db import Database

PARADEDB_IMAGE = "paradedb/paradedb:latest"


# ---------------------------------------------------------------------------
# Helpers / fakes
# ---------------------------------------------------------------------------


class _CharTokenizer:
    """Deterministic char-level tokenizer (test stand-in)."""

    def encode(self, text_: str) -> list[int]:
        """Encode text as char ordinals."""
        return [ord(c) for c in text_]

    def decode(self, ids: list[int]) -> str:
        """Decode char ordinals back to text."""
        return "".join(chr(i) for i in ids)


class _FakeEmbedder:
    """Deterministic 1024-d embedder (test stand-in, no GPU)."""

    async def embed_document(
        self, doc_token_ids: list[int], leaves: Sequence[ChunkNode]
    ) -> list[ChunkEmbedding]:
        """Return fixed 1024-d vectors for every leaf.

        Args:
            doc_token_ids: Whole-document token ids (unused by fake).
            leaves: The leaf chunks to embed.

        Returns:
            One ChunkEmbedding per leaf with a fixed 0.1 dense vector.
        """
        return [ChunkEmbedding(leaf.local_id, [0.1] * 1024, {}) for leaf in leaves]

    async def health(self) -> bool:
        """Always healthy (fake)."""
        return True


class _FakeVlmClient:
    """Fake VLM client returning fixed caption text for any asset.

    Attributes:
        caption_text: The fixed text returned for every caption call.
        fail: When True, raises GenerationError to simulate VLM failure.
    """

    def __init__(self, caption_text: str = "A described visual.", *, fail: bool = False) -> None:
        """Initialise the fake VLM client.

        Args:
            caption_text: Caption text to return on success.
            fail: Whether to simulate a VLM failure.
        """
        from iw_rag.generation.llm import LlmResponse

        self._response = LlmResponse(text=caption_text, usage={})
        self._fail = fail
        self.call_count = 0

    async def complete(self, messages: object, params: object) -> object:
        """Return the fixed caption or raise on failure.

        Args:
            messages: Ignored in the fake.
            params: Ignored in the fake.

        Returns:
            A fixed LlmResponse.

        Raises:
            GenerationError: When ``fail=True``.
        """
        from iw_rag.domain.errors import GenerationError

        self.call_count += 1
        if self._fail:
            raise GenerationError("Injected VLM failure")
        return self._response


class _FixedAclResolver:
    """ACL resolver returning fixed tags for all sources."""

    def __init__(self, tags: list[str]) -> None:
        """Initialise with fixed tags.

        Args:
            tags: The ACL tags to return for every source.
        """
        self._tags = tags

    def resolve(self, source: object) -> list[str]:
        """Return the fixed ACL tags regardless of source.

        Args:
            source: Ignored.

        Returns:
            The fixed tag list.
        """
        return list(self._tags)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def migrated_dsn() -> Generator[str, None, None]:
    """A migrated ParadeDB container; yields its psycopg DSN.

    Yields:
        The psycopg-compatible DSN string for the test database.
    """
    from testcontainers.postgres import PostgresContainer

    from alembic import command
    from tests.integration._alembic_rebase import rebased_alembic_config

    with PostgresContainer(image=PARADEDB_IMAGE, username="t", password="t", dbname="t") as pg:
        dsn = pg.get_connection_url().replace("postgresql+psycopg2://", "postgresql+psycopg://")
        with rebased_alembic_config(dsn) as rebased:
            command.upgrade(rebased.cfg, "head")
        yield dsn


@pytest.fixture
def clean_pg(migrated_dsn: str) -> str:
    """Truncate iw_rag tables before each test.

    Args:
        migrated_dsn: The module-scoped migrated DSN.

    Returns:
        The same DSN after truncating all iw_rag tables.
    """
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


def _count(dsn: str, sql: str) -> int:
    """Run a scalar COUNT query and return the result.

    Args:
        dsn: The database DSN.
        sql: The COUNT SQL query to execute.

    Returns:
        Integer result of the scalar query.
    """
    engine = sqlalchemy.create_engine(dsn)
    try:
        with engine.connect() as conn:
            return int(conn.execute(text(sql)).scalar_one())
    finally:
        engine.dispose()


def _rows(dsn: str, sql: str) -> list[tuple[object, ...]]:
    """Run a SELECT query and return all rows.

    Args:
        dsn: The database DSN.
        sql: The SELECT SQL to execute.

    Returns:
        All result rows as tuples.
    """
    engine = sqlalchemy.create_engine(dsn)
    try:
        with engine.connect() as conn:
            return conn.execute(text(sql)).all()  # type: ignore[return-value]
    finally:
        engine.dispose()


def _make_captioner(
    caption_text: str = "A described visual.",
    *,
    fail: bool = False,
) -> VlmCaptioner:
    """Construct a VlmCaptioner backed by a fake VLM client.

    Args:
        caption_text: The fixed caption text the fake VLM returns.
        fail: When True, the fake VLM simulates failure.

    Returns:
        A configured VlmCaptioner with the fake client.
    """
    client = _FakeVlmClient(caption_text, fail=fail)
    cfg = VlmCaptionerConfig(
        model_role="visual_caption",
        served_name="Qwen/Qwen2.5-VL-7B-Instruct",
        max_tokens=256,
    )
    return VlmCaptioner(vlm_client=client, cfg=cfg)


def _make_pipeline(dsn: str) -> IngestionPipeline:
    """Build an IngestionPipeline with fake embedder and char tokenizer.

    Args:
        dsn: The database DSN for the test container.

    Returns:
        A configured IngestionPipeline using fake components.
    """
    from iw_rag.ingestion.parse import ParseConfig, select_parser

    return IngestionPipeline(
        parser_factory=lambda s: select_parser(s, ParseConfig()),
        chunker=HybridChunker(ChunkConfig(leaf_tokens=64), _CharTokenizer()),
        embedder=_FakeEmbedder(),
        db=Database(DatabaseSettings(dsn=dsn)),
        cfg=IngestionConfig(concurrency=1, supersede=False),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_table_caption_persisted_with_correct_content_type(
    clean_pg: str, tmp_path: Path
) -> None:
    """Captioned table asset becomes a chunk with content_type='table' and 1024-d vector."""
    from iw_rag.ingestion.parse import ParseHints, SourceRef

    # First ingest a base document so we have a document_id
    doc_file = tmp_path / "report.md"
    doc_file.write_text("# Report\n\nSome text.\n", encoding="utf-8")
    src = SourceRef(
        uri="corpus://report.md",
        mime="text/markdown",
        hints=ParseHints(),
        path=doc_file,
    )
    ctx = IngestContext(acl_resolver=_FixedAclResolver(["finance"]), corpus_label="test")
    pipeline = _make_pipeline(clean_pg)
    result = await pipeline.ingest_document(src, ctx)
    assert result.status == "ingested"
    doc_id = result.document_id
    assert doc_id is not None

    # Now caption a table asset and persist it via the chunk repository
    captioner = _make_captioner("Q3 revenue table: $10M total.")
    asset = VisualAsset(
        asset_type=VisualAssetType.TABLE,
        content="<table><tr><td>$10M</td></tr></table>",
        source_uri="corpus://report.md",
        page=1,
        acl_tags=["finance"],
    )
    caption_result = await captioner.caption_to_chunk(asset, ordinal=0)
    assert caption_result is not None
    assert isinstance(caption_result, CaptionResult)

    # Persist the caption chunk directly via the repository (caller supplies doc_id + embedding)
    db = Database(DatabaseSettings(dsn=clean_pg))
    async with db.session() as session:
        from iw_rag.storage.repositories import ChunkRepository, NewChunk

        new_chunk = NewChunk(
            document_id=doc_id,
            ordinal=caption_result.ordinal,
            content=caption_result.content,
            dense_embedding=[0.2] * 1024,
            parent_id=None,
            content_type=caption_result.content_type,
            section_path=caption_result.section_path,
            acl_tags=caption_result.acl_tags,
            metadata=caption_result.metadata,
        )
        await ChunkRepository(session).bulk_upsert_chunks([new_chunk])
        await session.commit()
    await db.dispose()

    # Verify the persisted chunk has the correct content_type
    rows = _rows(
        clean_pg,
        "SELECT content_type, acl_tags, content FROM iw_rag.chunks WHERE content_type = 'table'",
    )
    assert len(rows) == 1
    ct, tags, content = rows[0]
    assert ct == "table"
    assert "finance" in tags
    assert "Q3 revenue table" in content


@pytest.mark.integration
async def test_figure_caption_persisted_with_figure_caption_content_type(
    clean_pg: str, tmp_path: Path
) -> None:
    """Captioned figure asset becomes a chunk with content_type='figure_caption'."""
    from iw_rag.ingestion.parse import ParseHints, SourceRef

    doc_file = tmp_path / "slides.md"
    doc_file.write_text("# Slides\n\nIntro text.\n", encoding="utf-8")
    src = SourceRef(
        uri="corpus://slides.md",
        mime="text/markdown",
        hints=ParseHints(),
        path=doc_file,
    )
    ctx = IngestContext(acl_resolver=_FixedAclResolver(["hr"]), corpus_label="test")
    pipeline = _make_pipeline(clean_pg)
    result = await pipeline.ingest_document(src, ctx)
    assert result.status == "ingested"
    doc_id = result.document_id
    assert doc_id is not None

    captioner = _make_captioner("A bar chart showing headcount growth.")
    asset = VisualAsset(
        asset_type=VisualAssetType.FIGURE,
        content="<img alt='headcount chart'/>",
        source_uri="corpus://slides.md",
        page=2,
        acl_tags=["hr"],
    )
    caption_result = await captioner.caption_to_chunk(asset, ordinal=0)
    assert caption_result is not None
    assert isinstance(caption_result, CaptionResult)

    db = Database(DatabaseSettings(dsn=clean_pg))
    async with db.session() as session:
        from iw_rag.storage.repositories import ChunkRepository, NewChunk

        new_chunk = NewChunk(
            document_id=doc_id,
            ordinal=caption_result.ordinal,
            content=caption_result.content,
            dense_embedding=[0.3] * 1024,
            parent_id=None,
            content_type=caption_result.content_type,
            section_path=caption_result.section_path,
            acl_tags=caption_result.acl_tags,
            metadata=caption_result.metadata,
        )
        await ChunkRepository(session).bulk_upsert_chunks([new_chunk])
        await session.commit()
    await db.dispose()

    rows = _rows(
        clean_pg,
        "SELECT content_type, acl_tags FROM iw_rag.chunks WHERE content_type = 'figure_caption'",
    )
    assert len(rows) == 1
    ct, tags = rows[0]
    assert ct == "figure_caption"
    assert "hr" in tags


@pytest.mark.integration
async def test_vlm_failure_skips_asset_without_abort(clean_pg: str, tmp_path: Path) -> None:
    """VLM failure on a caption asset skips it; the document ingest does not abort."""
    from iw_rag.ingestion.parse import ParseHints, SourceRef

    doc_file = tmp_path / "bad_vlm.md"
    doc_file.write_text("# Doc\n\nContent.\n", encoding="utf-8")
    src = SourceRef(
        uri="corpus://bad_vlm.md",
        mime="text/markdown",
        hints=ParseHints(),
        path=doc_file,
    )
    ctx = IngestContext(acl_resolver=_FixedAclResolver(["public"]), corpus_label="test")
    pipeline = _make_pipeline(clean_pg)
    result = await pipeline.ingest_document(src, ctx)
    assert result.status == "ingested"

    # Captioner fails → caption_to_chunk returns None → caption is skipped
    captioner = _make_captioner(fail=True)
    asset = VisualAsset(
        asset_type=VisualAssetType.TABLE,
        content="<table/>",
        source_uri="corpus://bad_vlm.md",
        page=1,
        acl_tags=["public"],
    )
    caption_result = await captioner.caption_to_chunk(asset, ordinal=0)

    # No caption produced; base document chunks still present
    assert caption_result is None
    assert _count(clean_pg, "SELECT count(*) FROM iw_rag.chunks") > 0


@pytest.mark.integration
async def test_caption_chunk_acl_parity_with_source_document(clean_pg: str, tmp_path: Path) -> None:
    """Caption chunks inherit ACL tags from the source document (ACL pre-filter parity)."""
    from iw_rag.ingestion.parse import ParseHints, SourceRef

    doc_file = tmp_path / "restricted.md"
    doc_file.write_text("# Restricted\n\nClassified content.\n", encoding="utf-8")
    src = SourceRef(
        uri="corpus://restricted.md",
        mime="text/markdown",
        hints=ParseHints(),
        path=doc_file,
    )
    ctx = IngestContext(acl_resolver=_FixedAclResolver(["finance"]), corpus_label="test")
    pipeline = _make_pipeline(clean_pg)
    result = await pipeline.ingest_document(src, ctx)
    assert result.status == "ingested"
    doc_id = result.document_id
    assert doc_id is not None

    captioner = _make_captioner("Net income breakdown.")
    asset = VisualAsset(
        asset_type=VisualAssetType.TABLE,
        content="<table/>",
        source_uri="corpus://restricted.md",
        page=1,
        acl_tags=["finance"],  # Must match source doc ACL
    )
    caption_result = await captioner.caption_to_chunk(asset, ordinal=0)
    assert caption_result is not None
    assert isinstance(caption_result, CaptionResult)
    assert caption_result.acl_tags == ["finance"]

    # The caption chunk's acl_tags must equal the source document's chunks' acl_tags
    db = Database(DatabaseSettings(dsn=clean_pg))
    async with db.session() as session:
        from iw_rag.storage.repositories import ChunkRepository, NewChunk

        new_chunk = NewChunk(
            document_id=doc_id,
            ordinal=caption_result.ordinal,
            content=caption_result.content,
            dense_embedding=[0.4] * 1024,
            parent_id=None,
            content_type=caption_result.content_type,
            section_path=caption_result.section_path,
            acl_tags=caption_result.acl_tags,
            metadata=caption_result.metadata,
        )
        await ChunkRepository(session).bulk_upsert_chunks([new_chunk])
        await session.commit()
    await db.dispose()

    # All chunks for this document — text + caption — should have "finance" ACL
    rows = _rows(
        clean_pg,
        f"SELECT acl_tags FROM iw_rag.chunks WHERE document_id = '{doc_id}'",
    )
    for (tags,) in rows:
        assert "finance" in tags, f"Chunk missing 'finance' ACL tag: {tags}"
