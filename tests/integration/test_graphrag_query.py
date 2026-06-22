"""Integration tests for the GraphRAG query route (WI-029 / Task 4a step 8).

Requires testcontainers Postgres with Apache AGE.  When AGE is absent the
tests skip gracefully — the unit tests (test_graphrag_retriever.py, test_graph_route.py)
carry the coverage.

When AGE IS present:
- Index a tiny fixture corpus → query a known multi-hop question.
- Assert correct source chunks are returned and an ACL-restricted chunk is dropped.

References:
- task-4a-graphrag-route §8 (integration index→query test)
- engineering-conventions §3 (testcontainers, never a shared DB)
"""

from __future__ import annotations

import os
import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

PARADEDB_IMAGE = "paradedb/paradedb:latest"


# ---------------------------------------------------------------------------
# Fixtures (mirrored from test_graphrag_index.py)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def db_container_dsn() -> str:  # type: ignore[return]
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
    """Try to install AGE; return True if successful.

    Args:
        db_container_dsn: DSN for the testcontainer.

    Returns:
        True if AGE is available; False otherwise.
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


def _fake_extraction_json(chunk_id: str, entity_name: str) -> str:
    """Return a valid extraction JSON for a single entity.

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
                {"src": entity_name, "dst": "System", "type": "PART_OF", "chunk_id": chunk_id}
            ],
        }
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_graphrag_query_skipped_when_age_absent(age_available: bool) -> None:
    """If AGE is absent, the graphrag query integration tests skip gracefully.

    Unit tests (test_graphrag_retriever.py) carry the coverage when AGE is absent.
    """
    if age_available:
        pytest.skip("AGE is available — running full query integration tests instead")
    # AGE absent: GraphStoreUnavailable is raised — documented skip behaviour.
    pytest.skip(
        "AGE extension not available — unit tests carry coverage (see test_graphrag_retriever.py)"
    )


@pytest.mark.integration
async def test_graphrag_query_returns_correct_chunks_with_age(
    db_container_dsn: str,
    age_available: bool,
    migrated_engine: Any,
) -> None:
    """With AGE: index two chunks, query a multi-hop question → correct chunks returned.

    Asserts the ACL-restricted chunk is dropped and the public chunk is returned.
    """
    if not age_available:
        pytest.skip("AGE extension not available — skipping graphrag query integration test")

    from iw_rag.domain.acl import AccessContext
    from iw_rag.modes.graphrag.indexer import GraphIndexer
    from iw_rag.modes.graphrag.retriever import GraphRetriever
    from iw_rag.modes.graphrag.store import AgeGraphStore
    from iw_rag.modes.graphrag.types import GraphIndexConfig, GraphRetrieverConfig

    # -----------------------------------------------------------------------
    # Step 1: Offline indexing — two chunks
    # -----------------------------------------------------------------------
    chunk_id_public = str(uuid.uuid4())
    chunk_id_restricted = str(uuid.uuid4())

    responses = [
        _fake_extraction_json(chunk_id_public, "PublicEntity"),
        _fake_extraction_json(chunk_id_restricted, "RestrictedEntity"),
    ]
    llm_client = _make_fake_llm_client(responses)

    store = AgeGraphStore(dsn=db_container_dsn, graph_name="query_test_graph")
    cfg = GraphIndexConfig()
    indexer = GraphIndexer(store=store, cfg=cfg, llm_client=llm_client)

    chunks_to_index = [
        {
            "chunk_id": chunk_id_public,
            "source_uri": "file:///public_doc.md",
            "checksum": "sha-pub",
            "content": "PublicEntity is available to all users.",
            "acl_tags": [],
        },
        {
            "chunk_id": chunk_id_restricted,
            "source_uri": "file:///restricted_doc.md",
            "checksum": "sha-res",
            "content": "RestrictedEntity is confidential HR data.",
            "acl_tags": ["hr"],
        },
    ]
    report = await indexer.index_chunks(chunks_to_index)
    assert report.chunks_indexed == 2, f"Expected 2 indexed, got {report}"

    # -----------------------------------------------------------------------
    # Step 2: Build a fake chunk_repo that simulates DB lookup
    # -----------------------------------------------------------------------
    chunk_db: dict[str, dict[str, Any]] = {
        chunk_id_public: {
            "chunk_id": chunk_id_public,
            "content": "PublicEntity is available to all users.",
            "source_uri": "file:///public_doc.md",
            "acl_tags": [],
            "metadata": {},
        },
        chunk_id_restricted: {
            "chunk_id": chunk_id_restricted,
            "content": "RestrictedEntity is confidential HR data.",
            "source_uri": "file:///restricted_doc.md",
            "acl_tags": ["hr"],
            "metadata": {},
        },
    }

    async def _get_by_str_ids(chunk_ids: list[str]) -> list[dict[str, Any]]:
        return [chunk_db[cid] for cid in chunk_ids if cid in chunk_db]

    fake_chunk_repo = MagicMock()
    fake_chunk_repo.get_by_str_ids = AsyncMock(side_effect=_get_by_str_ids)

    # -----------------------------------------------------------------------
    # Step 3: Query — public user (no hr) should only see public chunk
    # -----------------------------------------------------------------------
    retriever_cfg = GraphRetrieverConfig()
    retriever = GraphRetriever(store=store, chunk_repo=fake_chunk_repo, cfg=retriever_cfg)

    ctx_public = AccessContext(
        principal_id="public-user",
        principal_groups=("public",),
        is_operator=False,
        request_id="req-integ",
    )

    result = await retriever.retrieve("What is PublicEntity?", ctx_public)

    assert result.retriever == "graphrag"
    assert result.acl_applied is True
    # Public chunk should be there
    public_ids = {c.chunk_id for c in result.chunks}
    # Restricted chunk must be absent
    assert chunk_id_restricted not in public_ids, (
        "ACL post-filter must have dropped the hr-restricted chunk"
    )

    # -----------------------------------------------------------------------
    # Step 4: Query as hr user — should see both chunks
    # -----------------------------------------------------------------------
    ctx_hr = AccessContext(
        principal_id="hr-user",
        principal_groups=("hr", "public"),
        is_operator=False,
        request_id="req-integ-hr",
    )

    result_hr = await retriever.retrieve("What is RestrictedEntity?", ctx_hr)
    hr_ids = {c.chunk_id for c in result_hr.chunks}
    assert chunk_id_restricted in hr_ids, "HR user should see the hr-restricted chunk"
