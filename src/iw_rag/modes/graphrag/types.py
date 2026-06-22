"""GraphRAG domain types — enums, config, and data containers.

Provides the canonical type vocabulary for the graphrag mode:
- :class:`GraphQueryMode` — the five LightRAG query modes.
- :class:`GraphIndexConfig` — offline indexer configuration.
- :class:`GraphRetrieverConfig` — hot-path retriever configuration (WI-029 target).
- :class:`Entity` — an extracted graph entity with source back-ref.
- :class:`Relation` — an extracted graph relation with source back-ref.
- :class:`RawGraphContext` — raw graph subgraph returned by AgeGraphStore.

These types are OFFLINE-safe (no import of hot-path modules); the retriever
config stub is defined here so WI-029 can import it without circular deps.

References:
- lld-graphrag §2 (public interface + type signatures)
- task-4a-graphrag-route §1 (store boundary + types)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

# ---------------------------------------------------------------------------
# GraphQueryMode
# ---------------------------------------------------------------------------


class GraphQueryMode(StrEnum):
    """The five LightRAG query modes (lld-graphrag §2).

    Attributes:
        LOCAL: Entity-neighbourhood retrieval (local graph context).
        GLOBAL: Community/global-summary retrieval (costlier; off by default).
        HYBRID: Local + global combined.
        NAIVE: Plain vector fallback — graceful degrade when graph is empty.
        MIX: LightRAG default: graph + vector blended (bounds misroute downside).
    """

    LOCAL = "local"
    GLOBAL = "global"
    HYBRID = "hybrid"
    NAIVE = "naive"
    MIX = "mix"


# ---------------------------------------------------------------------------
# Entity / Relation — graph data containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Entity:
    """A graph entity extracted from a chunk (lld-graphrag §3).

    Each entity carries a back-reference to its source ``chunk_id`` so that
    the retriever can resolve citations and apply ACL post-filtering.

    Attributes:
        name: Canonical entity name (normalised, deduplicated key).
        entity_type: Entity type label (e.g. ``"Component"``, ``"Person"``).
        chunk_id: Back-reference to the source ``iw_rag.chunks.id`` (str, per D5).
        metadata: Optional additional fields from extraction (description, …).
    """

    name: str
    entity_type: str
    chunk_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Relation:
    """A typed relation between two entities extracted from a chunk (lld-graphrag §3).

    Carries a back-reference to its source ``chunk_id`` for citation resolution
    and ACL post-filtering.

    Attributes:
        src: Source entity name.
        dst: Destination entity name.
        relation_type: Relation type label (e.g. ``"USES"``, ``"CALLS"``).
        chunk_id: Back-reference to the source ``iw_rag.chunks.id`` (str, per D5).
        metadata: Optional additional fields from extraction (weight, …).
    """

    src: str
    dst: str
    relation_type: str
    chunk_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# RawGraphContext — output of AgeGraphStore.query_subgraph
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RawGraphContext:
    """Raw graph subgraph returned by AgeGraphStore.query_subgraph.

    Used by the retriever (WI-029) to adapt graph results into a RetrievalResult.

    Attributes:
        entities: Entities in the traversed subgraph.
        relations: Relations in the traversed subgraph.
        entity_ids: Flat list of entity name keys (convenience for resolve calls).
    """

    entities: list[Entity] = field(default_factory=list)
    relations: list[Relation] = field(default_factory=list)
    entity_ids: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# GraphIndexConfig — offline indexer configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GraphIndexConfig:
    """Offline indexer configuration (lld-graphrag §7, task-4a §2).

    Attributes:
        extraction_model: Logical model role in ``config/models.yaml``
            (default ``"graphrag-extract"`` → Qwen3-4B, batch endpoint).
        max_gleanings: Number of bounded LLM re-prompts per chunk during
            extraction (default 1 to control cost, R-00106/F5).
        chunk_token_size: Target token size for extraction context window.
        skip_community_summaries: Skip community-summary generation (default
            ``True`` — dominant cost; only needed for global-summary demos,
            R-00106/F5).
        enabled: Master switch for the graphrag mode.  ``False`` by default until
            experiment #6 proves the multi-hop win (task-4a acceptance criteria).
    """

    extraction_model: str = "graphrag-extract"
    max_gleanings: int = 1
    chunk_token_size: int = 1200
    skip_community_summaries: bool = True
    enabled: bool = False


# ---------------------------------------------------------------------------
# GraphRetrieverConfig — hot-path retriever configuration (WI-029 target)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GraphRetrieverConfig:
    """Hot-path retriever configuration stub — WI-029 will fill this in.

    Defined here so WI-029 can import the config without modifying this module.

    Attributes:
        default_mode: Default LightRAG query mode (``MIX`` bounds misroute
            downside on simple queries, R-00106/F6).
        top_k: Final top-k chunks returned to the generator.
        enable_global: Whether the ``global`` community-summary mode is allowed
            (off by default; requires ``skip_community_summaries=False`` at index
            time and significant extra cost).
        acl_postfilter: Whether to post-filter resolved chunks by ACL tags before
            returning (always ``True`` — the ACL caveat, lld-graphrag §6).
        timeout_s: Per-query timeout in seconds; exceeded → GraphRouteTimeout and
            router fallback to the spine.
    """

    default_mode: GraphQueryMode = GraphQueryMode.MIX
    top_k: int = 10
    enable_global: bool = False
    acl_postfilter: bool = True
    timeout_s: float = 20.0
