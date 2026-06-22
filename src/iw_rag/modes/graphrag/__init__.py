"""iw_rag.modes.graphrag — optional router-gated GraphRAG retrieval mode (Phase 4).

Two-clocks split (CLAUDE.md, lld-graphrag §1):

OFFLINE / batch only (WI-028):
- :class:`~iw_rag.modes.graphrag.indexer.GraphIndexer`: extracts entities/relations
  from already-ingested chunks using the small extraction model (Qwen3-4B) and
  upserts them into the AGE graph.
- :class:`~iw_rag.modes.graphrag.store.AgeGraphStore`: the ONLY module issuing
  graph Cypher/SQL against the shared Postgres (AGE).
- :class:`~iw_rag.modes.graphrag.store.GraphStoreUnavailable`: raised when AGE
  is not installed; the spine is unaffected.

HOT-PATH allowed (WI-029):
- :class:`~iw_rag.modes.graphrag.retriever.GraphRetriever`: the hot-path query
  branch.  Uses AgeGraphStore for READ-ONLY operations only (query_subgraph +
  resolve_to_chunks).  Never imports the indexer, never calls the extraction model.

Config types (OFFLINE-safe; imported by both WI-028 and WI-029):
- :class:`~iw_rag.modes.graphrag.types.GraphQueryMode`
- :class:`~iw_rag.modes.graphrag.types.GraphIndexConfig`
- :class:`~iw_rag.modes.graphrag.types.GraphRetrieverConfig`
- :class:`~iw_rag.modes.graphrag.types.Entity`
- :class:`~iw_rag.modes.graphrag.types.Relation`
- :class:`~iw_rag.modes.graphrag.types.RawGraphContext`

Two-clocks rule:
The top-level ``__init__`` intentionally imports the offline-only components so
that ``from iw_rag.modes.graphrag import …`` in hot-path modules is caught by the
offline guard test (test_graphrag_offline_guard.py).

IMPORTANT: ``GraphRetriever`` is NOT imported here because this top-level package
is itself on the forbidden list in the offline guard.  Hot-path code that needs
the retriever must import it directly::

    from iw_rag.modes.graphrag.retriever import GraphRetriever
"""

from iw_rag.modes.graphrag.indexer import GraphIndexer, GraphIndexReport
from iw_rag.modes.graphrag.store import AgeGraphStore, GraphStoreUnavailable
from iw_rag.modes.graphrag.types import (
    Entity,
    GraphIndexConfig,
    GraphQueryMode,
    GraphRetrieverConfig,
    RawGraphContext,
    Relation,
)

__all__ = [
    # OFFLINE / batch (WI-028)
    "GraphIndexer",
    "GraphIndexReport",
    "AgeGraphStore",
    "GraphStoreUnavailable",
    # Config types (shared)
    "GraphQueryMode",
    "GraphIndexConfig",
    "GraphRetrieverConfig",
    "Entity",
    "Relation",
    "RawGraphContext",
    # NOTE: GraphRetriever is intentionally NOT in __all__ here — see module docstring.
    # Import it directly: from iw_rag.modes.graphrag.retriever import GraphRetriever
]
