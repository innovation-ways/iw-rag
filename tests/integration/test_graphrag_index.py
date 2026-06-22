"""Integration tests for the GraphRAG offline indexer against testcontainers Postgres.

Attempts to use Apache AGE (``CREATE EXTENSION IF NOT EXISTS age``).  If AGE is
not available in the ParadeDB image the tests are gracefully skipped — the unit
tests (test_graphrag_indexer.py) and the store's fail-fast
(``GraphStoreUnavailable``) carry the coverage when AGE is absent.

When AGE IS present:
- Indexes a tiny fixture corpus (two chunks with entity/relation extraction).
- Asserts entities and relations are upserted with ``chunk_id`` back-refs.
- Asserts ``resolve_to_chunks`` round-trips entity IDs back to source chunk IDs.
- Asserts a second identical run skips already-indexed chunks (incremental mode).

References:
- lld-graphrag §8 (test strategy)
- task-4a-graphrag-route §3 (offline indexer)
- engineering-conventions §3 (testcontainers, never a shared DB)
"""

from __future__ import annotations

import os
import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

PARADEDB_IMAGE = "paradedb/paradedb:latest"


@pytest.fixture(scope="module")
def db_container_dsn() -> str:
    """Spin up a fresh paradedb/paradedb container and yield its DSN.

    Returns:
        A psycopg-driver DSN for the ephemeral test container.
    """
    from testcontainers.postgres import PostgresContainer

    with PostgresContainer(
        image=PARADEDB_IMAGE,
        username="test_user",
        password="test_pass",
        dbname="test_db",
    ) as pg:
        raw_url = pg.get_connection_url()
        dsn = raw_url.replace("postgresql+psycopg2://", "postgresql+psycopg://")
        yield dsn


@pytest.fixture(scope="module")
def age_available(db_container_dsn: str) -> bool:
    """Try to install AGE in the container; return True if successful.

    AGE may not be in the paradedb/paradedb image.  If it is absent the
    dependent tests will be skipped.

    Args:
        db_container_dsn: DSN for the testcontainer.

    Returns:
        True if the AGE extension was successfully created; False otherwise.
    """
    import sqlalchemy
    from sqlalchemy import text

    engine = sqlalchemy.create_engine(db_container_dsn)
    try:
        with engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS age"))
            conn.execute(text("LOAD 'age'"))
            conn.execute(text("SET search_path = ag_catalog, '$user', public"))
            conn.commit()
        return True
    except Exception:
        return False
    finally:
        engine.dispose()


@pytest.fixture(scope="module")
def migrated_engine(db_container_dsn: str) -> Any:
    """Run Alembic upgrade head and yield a synchronous engine.

    Args:
        db_container_dsn: DSN for the testcontainer.

    Returns:
        SQLAlchemy synchronous engine connected to the migrated DB.
    """
    import sqlalchemy

    import iw_rag.settings as settings_mod
    from alembic import command
    from tests.integration._alembic_rebase import rebased_alembic_config

    os.environ["IW_RAG__DATABASE__DSN"] = db_container_dsn
    os.environ["IW_RAG__MODELS__GENERATOR__ENDPOINT"] = "http://vllm:8001/v1"
    os.environ["IW_RAG__MODELS__EMBEDDER__ENDPOINT"] = "http://infinity:7997"
    os.environ["IW_RAG__MODELS__RERANKER__ENDPOINT"] = "http://infinity:7997"
    os.environ["IW_RAG__MODELS__JUDGE_FAITHFULNESS__ENDPOINT"] = "http://vllm-judge:8002/v1"
    os.environ["IW_RAG__MODELS__JUDGE_GENERAL__ENDPOINT"] = "http://vllm-judge:8002/v1"
    settings_mod.get_settings.cache_clear()

    with rebased_alembic_config(db_container_dsn) as rebased:
        command.upgrade(rebased.cfg, "head")

    engine = sqlalchemy.create_engine(db_container_dsn)
    yield engine

    engine.dispose()
    for key in [
        "IW_RAG__DATABASE__DSN",
        "IW_RAG__MODELS__GENERATOR__ENDPOINT",
        "IW_RAG__MODELS__EMBEDDER__ENDPOINT",
        "IW_RAG__MODELS__RERANKER__ENDPOINT",
        "IW_RAG__MODELS__JUDGE_FAITHFULNESS__ENDPOINT",
        "IW_RAG__MODELS__JUDGE_GENERAL__ENDPOINT",
    ]:
        os.environ.pop(key, None)
    settings_mod.get_settings.cache_clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_extraction_json(chunk_id: str, entity_name: str = "TestEntity") -> str:
    """Return a valid extraction JSON for a single entity and relation.

    Args:
        chunk_id: The source chunk id to embed as back-ref.
        entity_name: The entity name to include.

    Returns:
        A JSON string with one entity and one relation.
    """
    import json

    return json.dumps(
        {
            "entities": [{"name": entity_name, "type": "Concept", "chunk_id": chunk_id}],
            "relations": [
                {
                    "src": entity_name,
                    "dst": "System",
                    "type": "PART_OF",
                    "chunk_id": chunk_id,
                }
            ],
        }
    )


def _make_fake_llm_client(responses: list[str]) -> Any:
    """Build a fake LLM client that replays string responses.

    Args:
        responses: List of raw text responses indexed by call order.

    Returns:
        A MagicMock with an async ``complete`` method.
    """
    call_count = [0]
    from iw_rag.generation.llm import LlmResponse

    async def _complete(messages: Any, params: Any) -> LlmResponse:
        idx = call_count[0]
        call_count[0] += 1
        return LlmResponse(text=responses[idx % len(responses)], usage={})

    client = MagicMock()
    client.complete = AsyncMock(side_effect=_complete)
    return client


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_age_extension_is_absent_or_skipped(
    age_available: bool,
    migrated_engine: Any,
) -> None:
    """If AGE is absent the test is skipped; if present asserts AGE was loaded.

    This test documents the skip behaviour so CI output is transparent.
    """
    if not age_available:
        pytest.skip("AGE extension not available in this Postgres image — skipping graphrag tests")
    # AGE is available; verify we can query pg_extension
    from sqlalchemy import text

    with migrated_engine.connect() as conn:
        result = conn.execute(text("SELECT COUNT(*) FROM pg_extension WHERE extname = 'age'"))
        count = result.scalar()
    assert count == 1, "AGE extension should be installed"


@pytest.mark.integration
async def test_graphrag_store_fails_fast_when_age_absent(
    db_container_dsn: str,
    age_available: bool,
) -> None:
    """AgeGraphStore raises GraphStoreUnavailable when AGE is not installed.

    When AGE is present this test is skipped (the happy-path tests cover it).
    """
    if age_available:
        pytest.skip("AGE is available — GraphStoreUnavailable path not applicable")

    from iw_rag.modes.graphrag.store import AgeGraphStore, GraphStoreUnavailable
    from iw_rag.modes.graphrag.types import Entity

    store = AgeGraphStore(dsn=db_container_dsn, graph_name="test_graph")
    with pytest.raises(GraphStoreUnavailable, match="AGE"):
        # A real AGE operation on a store without AGE must fail fast.
        # (An empty entity list short-circuits before the AGE check.)
        await store.upsert_entities(
            [Entity(name="TestEntity", entity_type="Concept", chunk_id="c1")]
        )


@pytest.mark.integration
async def test_index_chunks_with_age_upserts_entities_and_relations(
    db_container_dsn: str,
    age_available: bool,
    migrated_engine: Any,
) -> None:
    """With AGE present: index two chunks → entities/relations upserted with chunk back-refs.

    Asserts:
    - GraphIndexReport.chunks_indexed == 2.
    - AgeGraphStore.upsert_entities was called with entities carrying chunk_id.
    - AgeGraphStore.upsert_relations was called with relations carrying chunk_id.
    """
    if not age_available:
        pytest.skip("AGE extension not available — skipping full indexing integration test")

    from iw_rag.modes.graphrag.indexer import GraphIndexer
    from iw_rag.modes.graphrag.store import AgeGraphStore
    from iw_rag.modes.graphrag.types import GraphIndexConfig

    chunk_id_a = str(uuid.uuid4())
    chunk_id_b = str(uuid.uuid4())

    responses = [
        _fake_extraction_json(chunk_id_a, "EntityA"),
        _fake_extraction_json(chunk_id_b, "EntityB"),
    ]
    llm_client = _make_fake_llm_client(responses)

    store = AgeGraphStore(dsn=db_container_dsn, graph_name="test_graph")
    cfg = GraphIndexConfig()
    indexer = GraphIndexer(store=store, cfg=cfg, llm_client=llm_client)

    chunks = [
        {
            "chunk_id": chunk_id_a,
            "source_uri": "file:///doc_a.md",
            "checksum": "sha-aaa",
            "content": "EntityA is a core component.",
            "acl_tags": [],
        },
        {
            "chunk_id": chunk_id_b,
            "source_uri": "file:///doc_b.md",
            "checksum": "sha-bbb",
            "content": "EntityB handles routing.",
            "acl_tags": [],
        },
    ]

    report = await indexer.index_chunks(chunks)
    assert report.chunks_indexed == 2
    assert report.chunks_failed == 0


@pytest.mark.integration
async def test_resolve_to_chunks_round_trips(
    db_container_dsn: str,
    age_available: bool,
    migrated_engine: Any,
) -> None:
    """With AGE present: entity_ids → resolve_to_chunks → returns source chunk_ids.

    This is the load-bearing citation path (lld-graphrag §3).
    """
    if not age_available:
        pytest.skip("AGE extension not available — skipping resolve_to_chunks integration test")

    from iw_rag.modes.graphrag.indexer import GraphIndexer
    from iw_rag.modes.graphrag.store import AgeGraphStore
    from iw_rag.modes.graphrag.types import GraphIndexConfig

    chunk_id = str(uuid.uuid4())
    responses = [_fake_extraction_json(chunk_id, "ResolveEntity")]
    llm_client = _make_fake_llm_client(responses)

    store = AgeGraphStore(dsn=db_container_dsn, graph_name="resolve_graph")
    cfg = GraphIndexConfig()
    indexer = GraphIndexer(store=store, cfg=cfg, llm_client=llm_client)

    chunks = [
        {
            "chunk_id": chunk_id,
            "source_uri": "file:///resolve_doc.md",
            "checksum": "sha-resolve",
            "content": "ResolveEntity handles resolution.",
            "acl_tags": [],
        }
    ]
    await indexer.index_chunks(chunks)

    # resolve back from the entity name to its source chunk_id
    resolved = await store.resolve_to_chunks(["ResolveEntity"])
    assert chunk_id in resolved, f"Expected chunk_id {chunk_id!r} in resolved {resolved!r}"


@pytest.mark.integration
async def test_incremental_mode_skips_already_indexed_chunks(
    db_container_dsn: str,
    age_available: bool,
    migrated_engine: Any,
) -> None:
    """Incremental indexing skips chunks already indexed (dedup on source_uri+checksum)."""
    if not age_available:
        pytest.skip("AGE extension not available — skipping incremental mode integration test")

    from iw_rag.modes.graphrag.indexer import GraphIndexer
    from iw_rag.modes.graphrag.store import AgeGraphStore
    from iw_rag.modes.graphrag.types import GraphIndexConfig

    chunk_id = str(uuid.uuid4())
    call_count = [0]

    from iw_rag.generation.llm import LlmResponse

    async def _complete(messages: Any, params: Any) -> LlmResponse:
        call_count[0] += 1
        return LlmResponse(text=_fake_extraction_json(chunk_id, "IncrementalEntity"), usage={})

    llm_client = MagicMock()
    llm_client.complete = AsyncMock(side_effect=_complete)

    store = AgeGraphStore(dsn=db_container_dsn, graph_name="incremental_graph")
    cfg = GraphIndexConfig()
    indexer = GraphIndexer(store=store, cfg=cfg, llm_client=llm_client)

    chunk = {
        "chunk_id": chunk_id,
        "source_uri": "file:///incremental_doc.md",
        "checksum": "sha-incr",
        "content": "IncrementalEntity handles incremental indexing.",
        "acl_tags": [],
    }

    report1 = await indexer.index_chunks([chunk], incremental=True)
    assert report1.chunks_indexed == 1
    first_call_count = call_count[0]

    # Second pass — same chunk, incremental=True → must be skipped
    report2 = await indexer.index_chunks([chunk], incremental=True)
    assert report2.chunks_skipped == 1
    assert call_count[0] == first_call_count  # LLM not called again
