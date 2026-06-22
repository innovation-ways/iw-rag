"""Enrichment stage — deterministic, source-derived metadata (ACL-critical).

The only place ``acl_tags`` are assigned. They come *solely* from the
``AclResolver`` (source-system metadata, e.g. a corpus manifest) — never from
parsed text or any model output (lld-ingestion §2.3, §6). An empty resolver
result yields ``acl_tags = []`` → the document is invisible to every principal
(fail-closed; never world-readable). ``content_type``/``section_path``/
``parent_id`` are already set by the chunker; this stage layers ACL on top.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, replace
from typing import Protocol, runtime_checkable

from iw_rag.ingestion.chunk import ChunkTree
from iw_rag.ingestion.parse import SourceRef


@runtime_checkable
class AclResolver(Protocol):
    """Maps a source document to its entitlement group tags (deterministic)."""

    def resolve(self, source: SourceRef) -> list[str]:
        """Return the source's group tags from source-system metadata.

        Args:
            source: The document being ingested.

        Returns:
            The entitlement group tags (e.g. ``["hr", "finance"]``); empty when
            the source has no known entitlements (→ invisible, fail-closed).
        """
        ...


@dataclass(frozen=True, slots=True)
class IngestContext:
    """Per-run ingestion context.

    Attributes:
        acl_resolver: The (source-derived) ACL authority.
        corpus_label: Human label for the corpus/run.
        ingestion_run_id: The owning ingestion-run id (set once the run row exists).
    """

    acl_resolver: AclResolver
    corpus_label: str
    ingestion_run_id: uuid.UUID | None = None


def enrich(tree: ChunkTree, source: SourceRef, ctx: IngestContext) -> ChunkTree:
    """Attach ``acl_tags`` (and only ACL) to every node from the resolver.

    Security invariant: ``acl_tags`` derive solely from
    ``ctx.acl_resolver.resolve(source)``. No parsed text or model output may
    influence them. An empty result → ``[]`` (fail-closed), never a wildcard.

    Args:
        tree: The chunked tree (content_type/section_path/parent already set).
        source: The source document (the resolver's only ACL input).
        ctx: The ingestion context carrying the resolver.

    Returns:
        A new ``ChunkTree`` with every parent and leaf carrying ``acl_tags``.
    """
    tags = list(ctx.acl_resolver.resolve(source))
    parents = [replace(node, acl_tags=list(tags)) for node in tree.parents]
    leaves = [replace(node, acl_tags=list(tags)) for node in tree.leaves]
    return ChunkTree(parents=parents, leaves=leaves, doc_token_ids=tree.doc_token_ids)
