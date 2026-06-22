"""Hypothesis property-based tests for ``iw_rag.ingestion.chunk.HybridChunker``.

Invariants verified:
- **Partition**: the leaf token spans collectively cover every CONTENT
  (non-heading) token in ``doc_token_ids`` exactly once — no gaps, no overlaps.
- **Round-trip stability**: ``decode(doc_token_ids[leaf.token_span])`` reproduces
  ``leaf.content`` for every leaf, across arbitrary block text and config values.

The ``CharTokenizer`` (ord/chr, exact round-trip) is borrowed from the unit tests
in ``tests/unit/test_chunk.py`` — it guarantees lossless tokenisation so the
encode/decode cycle can be asserted to exact equality.
"""

from __future__ import annotations

from hypothesis import assume, given
from hypothesis import strategies as st

from iw_rag.ingestion.chunk import ChunkConfig, HybridChunker
from iw_rag.ingestion.parse import Block, ParsedDocument

# ---------------------------------------------------------------------------
# Test-local helpers (mirroring the pattern in tests/unit/test_chunk.py)
# ---------------------------------------------------------------------------


class CharTokenizer:
    """Deterministic char-level tokenizer for property tests (exact round-trip).

    Maps every character to its Unicode code-point and back, so
    ``decode(encode(text)) == text`` always holds.

    Returns:
        An instance that satisfies the ``Tokenizer`` protocol.
    """

    def encode(self, text: str) -> list[int]:
        """Encode text to per-character code-points.

        Args:
            text: Input string.

        Returns:
            List of Unicode code-points, one per character.
        """
        return [ord(c) for c in text]

    def decode(self, ids: list[int]) -> str:
        """Decode code-points back to text.

        Args:
            ids: Unicode code-points produced by ``encode``.

        Returns:
            The original string.
        """
        return "".join(chr(i) for i in ids)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Printable ASCII minus tab/newline — avoids surrogates and codec surprises
_PRINTABLE = st.text(
    alphabet=st.characters(
        whitelist_categories=("Lu", "Ll", "Nd", "Zs"),
        whitelist_characters="!@#$%^&*()-_=+[]{}|;:',.<>?/`~",
        blacklist_characters="\x00",
    ),
    min_size=1,
    max_size=200,
)

_BLOCK_TYPE = st.sampled_from(["text", "table", "figure_caption"])

_CONTENT_BLOCK = st.builds(
    Block,
    text=_PRINTABLE,
    block_type=_BLOCK_TYPE,
    section_path=st.one_of(st.none(), st.text(min_size=1, max_size=30)),
)

_HEADING_BLOCK = st.builds(
    Block,
    text=_PRINTABLE,
    block_type=st.just("heading"),
    section_path=st.one_of(st.none(), st.text(min_size=1, max_size=30)),
)

# Interspersed mix of headings and content blocks, at least one content block
_MIXED_BLOCKS = st.lists(
    st.one_of(_HEADING_BLOCK, _CONTENT_BLOCK),
    min_size=1,
    max_size=20,
)

_LEAF_TOKENS = st.integers(min_value=1, max_value=64)
_PARENT_MAX = st.one_of(st.none(), st.integers(min_value=1, max_value=256))


def _make_doc(blocks: list[Block]) -> ParsedDocument:
    """Build a minimal ``ParsedDocument`` from a block list.

    Args:
        blocks: Ordered content + heading blocks.

    Returns:
        A ``ParsedDocument`` ready for ``HybridChunker.chunk``.
    """
    full = "\n".join(b.text for b in blocks)
    return ParsedDocument(source_uri="file:///prop_test", blocks=blocks, full_text=full)


# ---------------------------------------------------------------------------
# Property 1 — Partition
# ---------------------------------------------------------------------------


@given(blocks=_MIXED_BLOCKS, leaf_tokens=_LEAF_TOKENS, parent_max=_PARENT_MAX)
def test_leaf_spans_partition_content_tokens_exactly(
    blocks: list[Block],
    leaf_tokens: int,
    parent_max: int | None,
) -> None:
    """Leaf token spans tile the content portion of ``doc_token_ids`` exactly once.

    Concatenating all leaf spans in leaf order must reproduce exactly the
    subsequence of ``doc_token_ids`` that corresponds to CONTENT (non-heading)
    blocks — no token skipped, no token covered twice.
    """
    # Skip if there are no content blocks (would produce no leaves; vacuous)
    content_blocks = [b for b in blocks if b.block_type != "heading"]
    assume(len(content_blocks) >= 1)

    tok = CharTokenizer()
    cfg = ChunkConfig(leaf_tokens=leaf_tokens, parent_max_tokens=parent_max)
    tree = HybridChunker(cfg, tok).chunk(_make_doc(blocks))

    # Build the expected content-only token stream in block order
    expected_tokens: list[int] = []
    for block in blocks:
        if block.block_type != "heading":
            expected_tokens.extend(tok.encode(block.text))

    # Collect actual tokens via the leaf spans
    actual_tokens: list[int] = []
    for leaf in tree.leaves:
        assert leaf.token_span is not None, f"leaf {leaf.local_id} has no token_span"
        actual_tokens.extend(tree.doc_token_ids[leaf.token_span.start : leaf.token_span.end])

    assert actual_tokens == expected_tokens, (
        f"Leaf spans do not tile content tokens exactly: "
        f"expected {len(expected_tokens)} tokens, got {len(actual_tokens)}"
    )


# ---------------------------------------------------------------------------
# Property 2 — Round-trip stability
# ---------------------------------------------------------------------------


@given(blocks=_MIXED_BLOCKS, leaf_tokens=_LEAF_TOKENS, parent_max=_PARENT_MAX)
def test_leaf_content_round_trips_via_doc_token_ids(
    blocks: list[Block],
    leaf_tokens: int,
    parent_max: int | None,
) -> None:
    """``decode(doc_token_ids[leaf.token_span])`` equals ``leaf.content`` for every leaf.

    This pins the Late-Chunking alignment guarantee: the span stored on each leaf
    is exactly the span the embedder's mean-pooling window will consume from the
    whole-document token stream.
    """
    content_blocks = [b for b in blocks if b.block_type != "heading"]
    assume(len(content_blocks) >= 1)

    tok = CharTokenizer()
    cfg = ChunkConfig(leaf_tokens=leaf_tokens, parent_max_tokens=parent_max)
    tree = HybridChunker(cfg, tok).chunk(_make_doc(blocks))

    for leaf in tree.leaves:
        assert leaf.token_span is not None, f"leaf {leaf.local_id} missing token_span"
        sliced = tree.doc_token_ids[leaf.token_span.start : leaf.token_span.end]
        decoded = tok.decode(sliced)
        assert decoded == leaf.content, (
            f"leaf {leaf.local_id}: decode(doc_token_ids[span]) == {decoded!r} "
            f"but leaf.content == {leaf.content!r}"
        )
