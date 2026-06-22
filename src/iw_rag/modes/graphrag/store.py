"""AgeGraphStore — the ONLY module issuing graph Cypher/SQL over Apache AGE.

OFFLINE / batch only.  This module MUST NOT be imported by any hot-path module
(api/, agent/, retrieval/, generation/).  The two-clocks rule (CLAUDE.md,
lld-graphrag §3) places all graph writes in the offline/batch path.

The graph lives in the **same Postgres instance** as the spine (one instance,
one backup — ADR-007, lld-graphrag §1).  openCypher queries are issued via the
AGE ``cypher()`` function:

    SELECT * FROM cypher('graph_name', $$ MATCH ... $$) AS (...)

All parameter casts use ``CAST(:p AS type)`` — NOT ``:p::type`` — because
SQLAlchemy ``text()`` does not parse ``::`` casts as bindparams (this bug
cost time in WI-031; do NOT regress it).

When AGE is unavailable at startup the store raises :class:`GraphStoreUnavailable`
on any operation (fail-fast; never crash the spine).

References:
- lld-graphrag §2 (store interface), §3 (algorithms)
- lld-data-model §8 (AGE storage + back-refs)
- task-4a-graphrag-route §1 (store boundary)
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence

import sqlalchemy
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine

from iw_rag.modes.graphrag.types import Entity, RawGraphContext, Relation

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Typed error hierarchy for the graphrag store
# ---------------------------------------------------------------------------


class GraphStoreUnavailable(Exception):
    """Raised when AGE is not available or the graph store cannot be reached.

    The route stays disabled; the hybrid spine is unaffected (lld-graphrag §5).
    """


class GraphRouteTimeout(Exception):
    """Raised when a graph query exceeds the configured timeout_s budget.

    The router falls back to the hybrid spine on this error (lld-graphrag §5).
    """


# ---------------------------------------------------------------------------
# AgeGraphStore
# ---------------------------------------------------------------------------


class AgeGraphStore:
    """Sole owner of openCypher over Apache AGE in the shared Postgres.

    All graph CRUD (entity/relation upsert, subgraph traversal, chunk
    resolution) is issued through this class — no other module may issue
    AGE SQL directly.  The store uses the same Postgres instance as
    ``storage/`` (one backup story, ADR-007).

    Async-first: every public method is a coroutine.  The DSN is passed at
    construction; the async engine is created lazily on first use.

    Attributes:
        _dsn: The psycopg DSN for the shared Postgres.
        _graph_name: The AGE graph name (created if absent on first use).
        _engine: Lazily created async SQLAlchemy engine.
        _age_checked: Whether the AGE extension availability has been verified.
    """

    def __init__(self, dsn: str, graph_name: str = "iw_rag_graph") -> None:
        """Initialise the store with a Postgres DSN and AGE graph name.

        Args:
            dsn: Full psycopg DSN for the shared Postgres instance.
            graph_name: The AGE graph identifier (created if absent on first use).
        """
        # AGE's cypher() requires the graph name as a LITERAL name constant (not a
        # bound parameter — `cypher(CAST(:gname AS text), ...)` raises "a name
        # constant is expected"). The name is therefore inlined into the SQL, so
        # validate it strictly here (identifier charset only) to keep that safe.
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", graph_name):
            raise ValueError(f"invalid AGE graph_name {graph_name!r} (identifier chars only)")
        self._dsn = dsn
        self._graph_name = graph_name
        self._engine: AsyncEngine | None = None
        self._age_checked = False
        self._age_available = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_engine(self) -> AsyncEngine:
        """Return (or lazily create) the async SQLAlchemy engine.

        Returns:
            The async engine connected to the shared Postgres.
        """
        if self._engine is None:
            # Convert sync DSN (postgresql+psycopg://...) for async use
            async_dsn = self._dsn
            # Ensure psycopg3 driver prefix for the async engine
            if (
                async_dsn.startswith("postgresql://")
                or async_dsn.startswith("postgresql+psycopg://")
            ) and "postgresql+psycopg://" not in async_dsn:
                async_dsn = async_dsn.replace("postgresql://", "postgresql+psycopg://")
            self._engine = create_async_engine(async_dsn, pool_pre_ping=True)
        return self._engine

    async def _ensure_age(self, session: AsyncSession) -> None:
        """Verify AGE is available; raise GraphStoreUnavailable if not.

        Also loads the AGE extension into the session's search_path so that
        openCypher functions are resolvable.

        Args:
            session: The current async session.

        Raises:
            GraphStoreUnavailable: If the AGE extension is not installed.
        """
        # The extension-existence check is instance-level (cacheable); but LOAD 'age'
        # and SET search_path are CONNECTION-scoped and MUST run for every session —
        # caching them away (as a prior version did) leaves a fresh pooled connection
        # without AGE, so `cypher(...)` fails with "function cypher(unknown, unknown)
        # does not exist" on every call after the first. Check once, load every time.
        if not self._age_checked:
            try:
                result = await session.execute(
                    text("SELECT COUNT(*) FROM pg_extension WHERE extname = 'age'")
                )
                self._age_available = bool(result.scalar())
            except Exception as exc:
                self._age_checked = True
                self._age_available = False
                raise GraphStoreUnavailable(
                    f"Failed to verify AGE extension availability: {exc}"
                ) from exc
            self._age_checked = True

        if not self._age_available:
            raise GraphStoreUnavailable(
                "Apache AGE extension is not installed in this Postgres instance. "
                "Install AGE and re-enable the graphrag mode (modes.graphrag.enabled=true)."
            )

        # Per-session load + search_path (connection-scoped — run every time).
        try:
            await session.execute(text("LOAD 'age'"))
            await session.execute(text("SET search_path = ag_catalog, '$user', public"))
        except Exception as exc:
            raise GraphStoreUnavailable(f"Failed to load AGE into the session: {exc}") from exc

    async def _ensure_graph_exists(self, session: AsyncSession) -> None:
        """Create the AGE graph if it does not already exist.

        Args:
            session: The current async session (AGE must be loaded).

        Raises:
            GraphStoreUnavailable: If graph creation fails.
        """
        # create_graph raises if the graph already exists — and a raised error
        # ABORTS the surrounding Postgres transaction (suppressing the Python
        # exception does NOT un-poison it, so every later statement in the batch
        # then fails with InFailedSqlTransaction). Check the catalog first so the
        # error is never triggered (AGE is already loaded by _ensure_age).
        exists = await session.execute(
            text("SELECT 1 FROM ag_catalog.ag_graph WHERE name = :gname").bindparams(
                sqlalchemy.bindparam("gname", value=self._graph_name, type_=sqlalchemy.Text)
            )
        )
        if exists.first() is None:
            await session.execute(
                text("SELECT create_graph(:gname)").bindparams(
                    sqlalchemy.bindparam("gname", value=self._graph_name, type_=sqlalchemy.Text)
                )
            )

    # ------------------------------------------------------------------
    # Back-ref table helpers (iw_rag schema, plain SQL — no AGE)
    # ------------------------------------------------------------------

    async def _ensure_backref_table(self, session: AsyncSession) -> None:
        """Create the graph_chunk_backrefs table if it does not exist.

        This table stores the mapping from entity/relation names to source
        chunk_id values so that resolve_to_chunks can work without full
        AGE traversal.

        Args:
            session: The current async session.
        """
        await session.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS iw_rag.graph_chunk_backrefs (
                    entity_name TEXT NOT NULL,
                    chunk_id    TEXT NOT NULL,
                    graph_name  TEXT NOT NULL DEFAULT 'iw_rag_graph',
                    PRIMARY KEY (entity_name, chunk_id, graph_name)
                )
                """
            )
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def upsert_entities(self, entities: Sequence[Entity]) -> None:
        """Upsert entities into the AGE graph and record chunk back-refs.

        Each entity node is created (or merged on name) in the AGE graph.
        A back-reference row is inserted into ``iw_rag.graph_chunk_backrefs``
        so that :meth:`resolve_to_chunks` can map entity names → chunk IDs
        without requiring a full AGE traversal.

        Args:
            entities: The entities to upsert (from extraction output).

        Raises:
            GraphStoreUnavailable: If AGE is not installed.
        """
        if not entities:
            return

        engine = self._get_engine()
        async with AsyncSession(engine) as session, session.begin():
            await self._ensure_age(session)
            await self._ensure_graph_exists(session)
            await self._ensure_backref_table(session)

            for entity in entities:
                # Upsert entity into AGE graph (MERGE by name)
                cypher_q = (
                    f"MERGE (e:Entity {{name: '{_escape_cypher(entity.name)}', "
                    f"entity_type: '{_escape_cypher(entity.entity_type)}', "
                    f"chunk_id: '{_escape_cypher(entity.chunk_id)}'}}) "
                    f"RETURN e"
                )
                try:
                    await session.execute(
                        text(
                            f"SELECT * FROM cypher('{self._graph_name}', $$"
                            f" {cypher_q} $$) AS (e agtype)"
                        )
                    )
                except Exception as exc:
                    _log.warning(
                        "AgeGraphStore: failed to upsert entity '%s': %s — skipping",
                        entity.name,
                        exc,
                    )
                    continue

                # Back-ref: entity_name → chunk_id
                await session.execute(
                    text(
                        "INSERT INTO iw_rag.graph_chunk_backrefs "
                        "(entity_name, chunk_id, graph_name) "
                        "VALUES (:name, :chunk_id, :gname) "
                        "ON CONFLICT DO NOTHING"
                    ).bindparams(
                        name=entity.name,
                        chunk_id=entity.chunk_id,
                        gname=self._graph_name,
                    )
                )

    async def upsert_relations(self, relations: Sequence[Relation]) -> None:
        """Upsert relations into the AGE graph and record chunk back-refs.

        Each relation edge is created (or merged on src/dst/type) in the AGE
        graph.  Back-references are recorded for both endpoints.

        Args:
            relations: The relations to upsert (from extraction output).

        Raises:
            GraphStoreUnavailable: If AGE is not installed.
        """
        if not relations:
            return

        engine = self._get_engine()
        async with AsyncSession(engine) as session, session.begin():
            await self._ensure_age(session)
            await self._ensure_graph_exists(session)
            await self._ensure_backref_table(session)

            for relation in relations:
                cypher_q = (
                    f"MATCH (a:Entity {{name: '{_escape_cypher(relation.src)}'}}), "
                    f"(b:Entity {{name: '{_escape_cypher(relation.dst)}'}}) "
                    f"MERGE (a)-[r:{_cypher_rel_type(relation.relation_type)} "
                    f"{{chunk_id: '{_escape_cypher(relation.chunk_id)}'}}]->(b) "
                    f"RETURN r"
                )
                try:
                    await session.execute(
                        text(
                            f"SELECT * FROM cypher('{self._graph_name}', $$"
                            f" {cypher_q} $$) AS (r agtype)"
                        )
                    )
                except Exception as exc:
                    _log.warning(
                        "AgeGraphStore: failed to upsert relation '%s→%s': %s — skipping",
                        relation.src,
                        relation.dst,
                        exc,
                    )
                    continue

                # Back-ref for src entity
                await session.execute(
                    text(
                        "INSERT INTO iw_rag.graph_chunk_backrefs "
                        "(entity_name, chunk_id, graph_name) "
                        "VALUES (:name, :chunk_id, :gname) "
                        "ON CONFLICT DO NOTHING"
                    ).bindparams(
                        name=relation.src,
                        chunk_id=relation.chunk_id,
                        gname=self._graph_name,
                    )
                )

    async def _resolve_seeds(self, session: AsyncSession, seeds: Sequence[str]) -> list[str]:
        """Resolve raw query-keyword seeds to actual entity names (WI-031, hot-path).

        The query seeds are keyword tokens (``_extract_seeds`` tokenises the query — no
        model call on the hot path, per the two-clocks rule). They rarely *exactly*
        match the stored entity names (``ada`` vs ``Ada Lovelace``, ``embedder`` vs
        ``embedder OOM``), so an exact Cypher ``MATCH {name: seed}`` returns nothing.
        This bridges that gap with a **fuzzy substring lookup against the known entity
        names** in ``graph_chunk_backrefs`` — still pure SQL, no model. Falls back to the
        raw seeds when nothing resolves (preserves the prior exact-match behaviour).

        NOTE: this recovers *entity-centric* queries (the entity is named in the query).
        Multi-hop queries that name only the *answer* (the bridge entity is unnamed)
        remain out of reach on the hot path — inferring the bridge needs the extraction
        model, which the two-clocks rule keeps offline.

        Args:
            session: The current async session (AGE loaded).
            seeds: Raw keyword seeds from the query.

        Returns:
            Resolved entity names (deduped), or the raw seeds if none resolved.
        """
        resolved: list[str] = []
        for seed in seeds:
            if len(seed) < 3:
                continue
            rows = await session.execute(
                text(
                    "SELECT entity_name FROM ("
                    "  SELECT DISTINCT entity_name FROM iw_rag.graph_chunk_backrefs "
                    "  WHERE graph_name = :g AND entity_name ILIKE '%' || :s || '%'"
                    ") sub ORDER BY length(entity_name) LIMIT 3"
                ).bindparams(g=self._graph_name, s=seed)
            )
            for (name,) in rows:
                if name not in resolved:
                    resolved.append(name)
        return resolved or list(seeds)

    async def query_subgraph(self, seeds: Sequence[str], hops: int = 2) -> RawGraphContext:
        """Traverse the AGE graph from seed entities up to ``hops`` hops.

        Returns a :class:`RawGraphContext` with entities, relations, and a
        flat ``entity_ids`` list (entity names) for downstream resolution.

        Seeds are first resolved from raw query keywords to actual entity names via
        :meth:`_resolve_seeds` (WI-031) — without this, the keyword-vs-entity-name
        mismatch makes the exact Cypher match return an empty subgraph.

        Args:
            seeds: Starting entity-name seeds (raw query keywords) for the traversal.
            hops: Maximum traversal depth (default 2).

        Returns:
            A :class:`RawGraphContext` with the traversed subgraph.

        Raises:
            GraphStoreUnavailable: If AGE is not installed.
        """
        engine = self._get_engine()
        async with AsyncSession(engine) as session:
            await self._ensure_age(session)

            entities: list[Entity] = []
            relations: list[Relation] = []
            entity_ids: list[str] = []

            resolved_seeds = await self._resolve_seeds(session, seeds)
            for seed in resolved_seeds:
                cypher_q = (
                    f"MATCH p=(e:Entity {{name: '{_escape_cypher(seed)}'}}) "
                    f"-[*0..{hops}]->() "
                    f"RETURN e"
                )
                try:
                    result = await session.execute(
                        text(
                            f"SELECT * FROM cypher('{self._graph_name}', $$"
                            f" {cypher_q} $$) AS (e agtype)"
                        )
                    )
                    for row in result:
                        # Parse AGE agtype into entity fields. AGE serialises a vertex as
                        # `{...}::vertex` (likewise `::edge` / `::path`); the type suffix
                        # makes a plain json.loads fail with "Extra data", so strip it.
                        import json

                        raw = str(row[0]).strip()
                        for _suffix in ("::vertex", "::edge", "::path"):
                            if raw.endswith(_suffix):
                                raw = raw[: -len(_suffix)]
                                break
                        node_data = json.loads(raw)
                        props = node_data.get("properties", {})
                        name = str(props.get("name", ""))
                        etype = str(props.get("entity_type", ""))
                        chunk_id = str(props.get("chunk_id", ""))
                        if name and name not in entity_ids:
                            entity_ids.append(name)
                            entities.append(
                                Entity(
                                    name=name,
                                    entity_type=etype,
                                    chunk_id=chunk_id,
                                )
                            )
                except Exception as exc:
                    _log.warning(
                        "AgeGraphStore: subgraph query failed for seed '%s': %s",
                        seed,
                        exc,
                    )

            return RawGraphContext(
                entities=entities,
                relations=relations,
                entity_ids=entity_ids,
            )

    async def resolve_to_chunks(self, entity_ids: Sequence[str]) -> list[str]:
        """Map entity names to their source chunk_id strings.

        Uses the ``iw_rag.graph_chunk_backrefs`` table (populated at upsert time)
        for efficient lookup without requiring a full AGE traversal.  This is the
        load-bearing citation/ACL path (lld-graphrag §3).

        Args:
            entity_ids: Entity names to resolve to chunk IDs.

        Returns:
            Deduplicated list of chunk_id strings sourced by any of the entities.

        Raises:
            GraphStoreUnavailable: If AGE is not installed.
        """
        if not entity_ids:
            return []

        engine = self._get_engine()
        async with AsyncSession(engine) as session:
            await self._ensure_age(session)
            await self._ensure_backref_table(session)

            result = await session.execute(
                text(
                    "SELECT DISTINCT chunk_id FROM iw_rag.graph_chunk_backrefs "
                    "WHERE entity_name = ANY(CAST(:names AS text[])) "
                    "  AND graph_name = CAST(:gname AS text)"
                ).bindparams(
                    names=list(entity_ids),
                    gname=self._graph_name,
                )
            )
            return [str(row[0]) for row in result]

    async def aclose(self) -> None:
        """Close the async engine and release pool connections.

        Should be called when the store is no longer needed (e.g. at script exit).
        """
        if self._engine is not None:
            await self._engine.dispose()
            self._engine = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _escape_cypher(value: str) -> str:
    """Escape a string value for safe embedding in an openCypher literal.

    Replaces single quotes with escaped single quotes.  This is a basic defence;
    for production use, parameterised Cypher queries are preferred wherever AGE
    supports them.

    Args:
        value: The raw string to escape.

    Returns:
        The escaped string safe for embedding in a Cypher string literal.
    """
    # Escape single quotes (the AGE Cypher literal delimiter)
    return value.replace("'", "\\'")


def _cypher_rel_type(value: str) -> str:
    """Normalize an extracted relation type into a valid Cypher relationship type.

    Cypher relationship types are *identifiers*, not string literals: ``:WORKS_AT``
    is valid, ``:works at`` is a syntax error. Extraction models emit free-text
    types ("request through", "is a") that must be folded to an identifier
    (``[^A-Za-z0-9_] → _``, upper-cased, leading digit prefixed) or the whole
    MERGE fails and the edge is dropped — leaving an edge-less graph that defeats
    multi-hop traversal.

    Args:
        value: The raw relation type string from extraction.

    Returns:
        An upper-snake-case identifier safe as a Cypher relationship type;
        ``RELATED_TO`` when the input has no identifier characters.
    """
    ident = re.sub(r"[^A-Za-z0-9_]", "_", value.strip()).strip("_").upper()
    if not ident:
        return "RELATED_TO"
    if ident[0].isdigit():
        ident = f"R_{ident}"
    return ident
