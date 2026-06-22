"""Unit tests for the GraphRAG offline indexer and tolerant JSON parser — TDD RED-first.

Tests cover:
- Tolerant JSON parser: valid JSON, malformed JSON (recover), retry-then-skip-and-log.
- Entity/relation dedup and set-merge logic.
- skip_community_summaries default enforcement.
- Extraction model failure → skip-and-log (no crash, batch continues).
- GraphIndexConfig field defaults.
- GraphIndexReport produced correctly.

All tests use faked LLM clients — no live models, no real Postgres required.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from iw_rag.modes.graphrag.indexer import (
    GraphIndexer,
    GraphIndexReport,
    _parse_extraction_json,
)
from iw_rag.modes.graphrag.types import Entity, GraphIndexConfig, Relation

# ---------------------------------------------------------------------------
# Helpers / fakes
# ---------------------------------------------------------------------------


def _make_fake_store(
    upserted_entities: list[Entity] | None = None,
    upserted_relations: list[Relation] | None = None,
) -> Any:
    """Build a fake AgeGraphStore that records upserted entities/relations.

    Args:
        upserted_entities: List to accumulate upserted entities into.
        upserted_relations: List to accumulate upserted relations into.

    Returns:
        A MagicMock with async upsert_entities/upsert_relations methods.
    """
    entities_sink: list[Entity] = upserted_entities if upserted_entities is not None else []
    relations_sink: list[Relation] = upserted_relations if upserted_relations is not None else []

    store = MagicMock()

    async def _upsert_entities(entities: list[Entity]) -> None:
        entities_sink.extend(entities)

    async def _upsert_relations(relations: list[Relation]) -> None:
        relations_sink.extend(relations)

    store.upsert_entities = AsyncMock(side_effect=_upsert_entities)
    store.upsert_relations = AsyncMock(side_effect=_upsert_relations)
    return store


def _chunk_row(
    chunk_id: str = "c1",
    source_uri: str = "file:///doc.md",
    checksum: str = "sha-001",
    content: str = "The system uses a cache layer.",
    acl_tags: list[str] | None = None,
) -> dict[str, Any]:
    """Produce a fake chunk row dictionary as returned by ChunkRepository.

    Args:
        chunk_id: Chunk identifier string.
        source_uri: Source document URI.
        checksum: Content checksum.
        content: Chunk text content.
        acl_tags: ACL entitlement tags.

    Returns:
        A dict simulating a DB chunk row.
    """
    return {
        "id": uuid.UUID(int=0) if chunk_id == "c1" else uuid.uuid4(),
        "chunk_id": chunk_id,
        "source_uri": source_uri,
        "checksum": checksum,
        "content": content,
        "acl_tags": acl_tags or [],
    }


def _make_fake_llm_client(responses: list[str], raise_on: set[int] | None = None) -> Any:
    """Build a fake LLM client that returns a sequence of string responses.

    Args:
        responses: List of raw text responses (indexed by call order).
        raise_on: Set of 0-based call indices on which to raise an exception.

    Returns:
        A MagicMock with an async ``complete`` method.
    """
    call_count_holder = [0]
    raise_on = raise_on or set()
    from iw_rag.generation.llm import LlmResponse

    async def _complete(messages: Any, params: Any) -> LlmResponse:
        idx = call_count_holder[0]
        call_count_holder[0] += 1
        if idx in raise_on:
            raise RuntimeError(f"Simulated LLM failure at call {idx}")
        resp_text = responses[idx % len(responses)]
        return LlmResponse(text=resp_text, usage={})

    client = MagicMock()
    client.complete = AsyncMock(side_effect=_complete)
    return client, call_count_holder


# ---------------------------------------------------------------------------
# Tests: tolerant JSON parser
# ---------------------------------------------------------------------------


def test_tolerant_parse_valid_json() -> None:
    """Valid JSON with entities and relations is parsed correctly."""
    raw = (
        '{"entities": [{"name": "Cache", "type": "Component", "chunk_id": "c1"}], '
        '"relations": [{"src": "Cache", "dst": "DB", "type": "USES", "chunk_id": "c1"}]}'
    )
    result = _parse_extraction_json(raw)
    assert result is not None
    assert len(result["entities"]) == 1
    assert result["entities"][0]["name"] == "Cache"
    assert len(result["relations"]) == 1


def test_tolerant_parse_json_wrapped_in_markdown_fences() -> None:
    """JSON wrapped in triple-backtick markdown fences is parsed correctly."""
    raw = (
        "```json\n"
        '{"entities": [{"name": "API", "type": "Component", "chunk_id": "c2"}], '
        '"relations": []}\n```'
    )
    result = _parse_extraction_json(raw)
    assert result is not None
    assert result["entities"][0]["name"] == "API"


def test_tolerant_parse_json_with_leading_prose() -> None:
    """JSON embedded in prose (after text prefix) is extracted and parsed."""
    raw = (
        "Here are the entities: "
        '{"entities": [{"name": "X", "type": "Y", "chunk_id": "c3"}], "relations": []}'
    )
    result = _parse_extraction_json(raw)
    assert result is not None
    assert result["entities"][0]["name"] == "X", f"Entity name must be 'X'; got: {result!r}"
    assert result["relations"] == [], f"Relations must be empty; got: {result!r}"


def test_tolerant_parse_completely_malformed_returns_none() -> None:
    """Completely non-JSON text returns None (skip-and-log behaviour)."""
    raw = "Sorry, I cannot extract entities from this text."
    result = _parse_extraction_json(raw)
    assert result is None


def test_tolerant_parse_partial_json_returns_none() -> None:
    """Truncated JSON that cannot be recovered returns None."""
    raw = '{"entities": [{"name": "X"'
    result = _parse_extraction_json(raw)
    assert result is None


def test_tolerant_parse_empty_entities_and_relations() -> None:
    """Empty-list JSON parses correctly (valid no-op extraction)."""
    raw = '{"entities": [], "relations": []}'
    result = _parse_extraction_json(raw)
    assert result is not None
    assert result["entities"] == []
    assert result["relations"] == []


# ---------------------------------------------------------------------------
# Tests: GraphIndexConfig defaults
# ---------------------------------------------------------------------------


def test_graph_index_config_defaults() -> None:
    """GraphIndexConfig defaults match the design spec (skip_community_summaries=True)."""
    cfg = GraphIndexConfig()
    assert cfg.skip_community_summaries is True, (
        "skip_community_summaries must default to True (R-00106/F5 cost control)"
    )
    assert cfg.max_gleanings == 1
    assert cfg.extraction_model == "graphrag-extract"
    assert cfg.enabled is False


# ---------------------------------------------------------------------------
# Tests: entity/relation dedup + set-merge
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_set_merge_skips_duplicate_chunk_by_checksum() -> None:
    """Incremental indexing skips a chunk already indexed (dedup on source_uri+checksum)."""
    entities_sink: list[Entity] = []
    relations_sink: list[Relation] = []
    store = _make_fake_store(entities_sink, relations_sink)

    valid_json = (
        '{"entities": [{"name": "Cache", "type": "Component", "chunk_id": "c1"}], "relations": []}'
    )
    llm_client, call_count = _make_fake_llm_client([valid_json])

    cfg = GraphIndexConfig()
    indexer = GraphIndexer(store=store, cfg=cfg, llm_client=llm_client)

    chunk = _chunk_row()

    # First pass — chunk is new, should be indexed
    report1 = await indexer.index_chunks([chunk], incremental=True)
    assert report1.chunks_indexed == 1
    assert report1.chunks_skipped == 0

    # Second pass — same (source_uri, checksum) — should be skipped
    report2 = await indexer.index_chunks([chunk], incremental=True)
    assert report2.chunks_indexed == 0
    assert report2.chunks_skipped == 1
    # LLM must NOT be called a second time (skipped)
    assert call_count[0] == 1


@pytest.mark.asyncio
async def test_entities_upserted_with_chunk_back_refs() -> None:
    """Entities produced by extraction carry the source chunk_id back-reference."""
    entities_sink: list[Entity] = []
    store = _make_fake_store(entities_sink)

    valid_json = (
        '{"entities": [{"name": "AuthService", "type": "Service", "chunk_id": "c42"}], '
        '"relations": []}'
    )
    llm_client, _ = _make_fake_llm_client([valid_json])

    cfg = GraphIndexConfig()
    indexer = GraphIndexer(store=store, cfg=cfg, llm_client=llm_client)

    chunk = _chunk_row(chunk_id="c42")
    await indexer.index_chunks([chunk])

    assert len(entities_sink) == 1
    assert entities_sink[0].chunk_id == "c42"
    assert entities_sink[0].name == "AuthService"


@pytest.mark.asyncio
async def test_relations_upserted_with_chunk_back_refs() -> None:
    """Relations produced by extraction carry the source chunk_id back-reference."""
    relations_sink: list[Relation] = []
    store = _make_fake_store(upserted_relations=relations_sink)

    valid_json = (
        '{"entities": [], '
        '"relations": [{"src": "A", "dst": "B", "type": "CALLS", "chunk_id": "c99"}]}'
    )
    llm_client, _ = _make_fake_llm_client([valid_json])

    cfg = GraphIndexConfig()
    indexer = GraphIndexer(store=store, cfg=cfg, llm_client=llm_client)

    chunk = _chunk_row(chunk_id="c99")
    await indexer.index_chunks([chunk])

    assert len(relations_sink) == 1
    assert relations_sink[0].chunk_id == "c99"
    assert relations_sink[0].src == "A"


# ---------------------------------------------------------------------------
# Tests: extraction failure → skip-and-log (no crash)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extraction_model_failure_skips_chunk_and_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """LLM failure on extraction is caught, logged at WARNING, chunk skipped — batch continues."""
    store = _make_fake_store()
    llm_client, _ = _make_fake_llm_client([""], raise_on={0})

    cfg = GraphIndexConfig()
    indexer = GraphIndexer(store=store, cfg=cfg, llm_client=llm_client)

    chunk = _chunk_row()
    with caplog.at_level(logging.WARNING, logger="iw_rag.modes.graphrag.indexer"):
        report = await indexer.index_chunks([chunk])

    assert report.chunks_failed == 1
    assert report.chunks_indexed == 0
    # Batch did not crash — report returned
    assert isinstance(report, GraphIndexReport)


@pytest.mark.asyncio
async def test_malformed_json_retry_then_skip_and_log(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Malformed JSON triggers one retry; if retry also fails, chunk is skipped and logged."""
    store = _make_fake_store()
    # Both calls return malformed JSON
    llm_client, call_count = _make_fake_llm_client(["not json at all", "still not json"])

    cfg = GraphIndexConfig()
    indexer = GraphIndexer(store=store, cfg=cfg, llm_client=llm_client)

    chunk = _chunk_row()
    with caplog.at_level(logging.WARNING, logger="iw_rag.modes.graphrag.indexer"):
        report = await indexer.index_chunks([chunk])

    # Exactly two LLM calls: first attempt + one bounded retry
    assert call_count[0] == 2
    assert report.chunks_failed == 1
    assert report.chunks_indexed == 0


@pytest.mark.asyncio
async def test_malformed_json_first_attempt_then_valid_retry_succeeds() -> None:
    """If first LLM response is malformed but retry returns valid JSON, chunk is indexed."""
    entities_sink: list[Entity] = []
    store = _make_fake_store(entities_sink)

    valid_json = (
        '{"entities": [{"name": "Retry", "type": "Test", "chunk_id": "c1"}], "relations": []}'
    )
    llm_client, call_count = _make_fake_llm_client(["not json", valid_json])

    cfg = GraphIndexConfig()
    indexer = GraphIndexer(store=store, cfg=cfg, llm_client=llm_client)

    chunk = _chunk_row()
    report = await indexer.index_chunks([chunk])

    assert call_count[0] == 2  # two calls: first attempt + retry
    assert report.chunks_indexed == 1
    assert report.chunks_failed == 0
    assert len(entities_sink) == 1


# ---------------------------------------------------------------------------
# Tests: skip_community_summaries default
# ---------------------------------------------------------------------------


def test_graph_index_config_skip_community_summaries_default() -> None:
    """skip_community_summaries defaults to True (R-00106/F5 — cost control)."""
    cfg = GraphIndexConfig()
    assert cfg.skip_community_summaries is True


# ---------------------------------------------------------------------------
# Tests: GraphIndexReport
# ---------------------------------------------------------------------------


def test_graph_index_report_is_frozen_dataclass() -> None:
    """GraphIndexReport is a frozen dataclass with expected fields."""
    report = GraphIndexReport(chunks_indexed=3, chunks_skipped=1, chunks_failed=0)
    assert report.chunks_indexed == 3
    assert report.chunks_skipped == 1
    assert report.chunks_failed == 0
    # Immutable
    with pytest.raises((AttributeError, TypeError), match="cannot assign to field"):
        report.chunks_indexed = 99  # type: ignore[misc]
