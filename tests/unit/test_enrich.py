"""Unit tests for enrich — the security-critical ACL invariant.

acl_tags MUST come solely from the AclResolver (source-derived); never from
parsed content or model output. Empty resolver → empty tags (fail-closed),
never a wildcard.
"""

from __future__ import annotations

from iw_rag.ingestion.chunk import ChunkNode, ChunkTree, TokenSpan
from iw_rag.ingestion.enrich import IngestContext, enrich
from iw_rag.ingestion.parse import SourceRef


class StubResolver:
    """Resolver returning fixed tags, ignoring the source entirely."""

    def __init__(self, tags: list[str]) -> None:
        self._tags = tags

    def resolve(self, source: SourceRef) -> list[str]:  # noqa: ARG002 - tags are source-independent here
        return list(self._tags)


class ContentSniffingResolver:
    """A resolver that (wrongly) tried to read content would still only get a
    SourceRef — it has no access to chunk text. This proves the type-level
    guarantee: enrich passes only the SourceRef to resolve()."""

    def resolve(self, source: SourceRef) -> list[str]:
        # Can only see source metadata; no chunk content is reachable here.
        assert isinstance(source, SourceRef)
        return ["public"] if "public" in source.uri else []


def _tree() -> ChunkTree:
    leaf = ChunkNode(
        local_id="l0",
        parent_local_id="p0",
        ordinal=0,
        content="secret salary data",  # content must NOT influence acl_tags
        content_type="text",
        section_path="HR",
        page=None,
        token_span=TokenSpan(0, 3),
    )
    parent = ChunkNode(
        local_id="p0",
        parent_local_id=None,
        ordinal=0,
        content="secret salary data",
        content_type="text",
        section_path="HR",
        page=None,
        token_span=None,
    )
    return ChunkTree(parents=[parent], leaves=[leaf], doc_token_ids=[1, 2, 3])


def _ctx(resolver: object) -> IngestContext:
    return IngestContext(acl_resolver=resolver, corpus_label="test")  # type: ignore[arg-type]


def test_acl_tags_come_from_resolver() -> None:
    src = SourceRef(uri="file:///hr/policy.md", mime="text/markdown")
    out = enrich(_tree(), src, _ctx(StubResolver(["hr", "finance"])))
    for node in [*out.parents, *out.leaves]:
        assert node.acl_tags == ["hr", "finance"]


def test_empty_resolver_yields_empty_tags_never_wildcard() -> None:
    src = SourceRef(uri="file:///hr/policy.md", mime="text/markdown")
    out = enrich(_tree(), src, _ctx(StubResolver([])))
    for node in [*out.parents, *out.leaves]:
        assert node.acl_tags == []
        assert "*" not in node.acl_tags
        assert "public" not in node.acl_tags


def test_tags_independent_of_chunk_content() -> None:
    # Content says "secret salary data" but the source uri has no 'public' → [].
    src = SourceRef(uri="file:///hr/policy.md", mime="text/markdown")
    out = enrich(_tree(), src, _ctx(ContentSniffingResolver()))
    assert all(node.acl_tags == [] for node in out.leaves)
    # A public uri → ['public'], driven by source metadata, not content.
    src_pub = SourceRef(uri="file:///public/readme.md", mime="text/markdown")
    out_pub = enrich(_tree(), src_pub, _ctx(ContentSniffingResolver()))
    assert all(node.acl_tags == ["public"] for node in out_pub.leaves)


def test_per_node_tag_lists_are_independent_objects() -> None:
    src = SourceRef(uri="file:///hr/policy.md", mime="text/markdown")
    out = enrich(_tree(), src, _ctx(StubResolver(["hr"])))
    nodes = [*out.parents, *out.leaves]
    # Mutating one node's list (frozen dataclass forbids reassignment, but the
    # list object itself) must not bleed into others.
    nodes[0].acl_tags.append("leak")
    assert nodes[1].acl_tags == ["hr"]
