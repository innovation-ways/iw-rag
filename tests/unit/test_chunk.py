"""Unit tests for HybridChunker — tree validity + Late-Chunking span alignment."""

from __future__ import annotations

from iw_rag.ingestion.chunk import ChunkConfig, HybridChunker
from iw_rag.ingestion.parse import Block, ParsedDocument


class CharTokenizer:
    """Deterministic char-level tokenizer for tests (exact round-trip)."""

    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(i) for i in ids)


def _doc() -> ParsedDocument:
    blocks = [
        Block(text="Leave Policy", block_type="heading", section_path="Leave Policy"),
        Block(text="abcdefghij", block_type="text", section_path="Leave Policy"),  # 10 chars
        Block(text="Rates", block_type="heading", section_path="Leave Policy > Rates"),
        Block(text="| A | 20 |", block_type="table", section_path="Leave Policy > Rates"),
    ]
    full = "\n".join(b.text for b in blocks)
    return ParsedDocument(source_uri="file:///d", blocks=blocks, full_text=full)


def test_leaves_respect_leaf_tokens() -> None:
    tree = HybridChunker(ChunkConfig(leaf_tokens=4), CharTokenizer()).chunk(_doc())
    assert tree.leaves, "expected leaves"
    assert all(len(leaf.token_span) <= 4 for leaf in tree.leaves if leaf.token_span)
    # 10-char block at leaf_tokens=4 → 3 leaves (4+4+2)
    text_leaves = [leaf for leaf in tree.leaves if leaf.content_type == "text"]
    assert len(text_leaves) == 3


def test_tree_is_valid_parent_child() -> None:
    tree = HybridChunker(ChunkConfig(leaf_tokens=4), CharTokenizer()).chunk(_doc())
    parent_ids = {p.local_id for p in tree.parents}
    # every leaf resolves to a real parent
    assert all(leaf.parent_local_id in parent_ids for leaf in tree.leaves)
    # ordinals are monotonic within each parent
    for pid in parent_ids:
        ordinals = [leaf.ordinal for leaf in tree.leaves if leaf.parent_local_id == pid]
        assert ordinals == sorted(ordinals)
        assert ordinals == list(range(len(ordinals)))


def test_spans_index_doc_token_ids_exactly() -> None:
    tok = CharTokenizer()
    tree = HybridChunker(ChunkConfig(leaf_tokens=4), tok).chunk(_doc())
    # Each leaf's span, sliced from doc_token_ids and decoded, equals its content.
    for leaf in tree.leaves:
        assert leaf.token_span is not None
        sliced = tree.doc_token_ids[leaf.token_span.start : leaf.token_span.end]
        assert tok.decode(sliced) == leaf.content


def test_doc_token_ids_cover_all_block_text() -> None:
    tok = CharTokenizer()
    doc = _doc()
    tree = HybridChunker(ChunkConfig(leaf_tokens=4), tok).chunk(doc)
    expected = sum(len(tok.encode(b.text)) for b in doc.blocks)
    assert len(tree.doc_token_ids) == expected


def test_parents_grouped_by_section() -> None:
    tree = HybridChunker(ChunkConfig(leaf_tokens=64), CharTokenizer()).chunk(_doc())
    sections = {p.section_path for p in tree.parents}
    assert sections == {"Leave Policy", "Leave Policy > Rates"}
