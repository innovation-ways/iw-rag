"""Unit tests for ``ChunkConfig.parent_max_tokens`` (I-00131, S03).

The bug being closed: ``HybridChunker._build_parents`` forms one parent per
section with no size cap, so a PDF with no heading structure yields a single
parent containing the ENTIRE document.  When small-to-big expansion
substitutes such a parent in front of a 16,384-token generator the prompt
overflows the context window (vLLM 400 → API 503).

The fix is a structural cap: when ``parent_max_tokens`` is set, a section
whose accumulated leaves would exceed the cap is split into multiple parents,
each <= cap.  The default ``parent_max_tokens=None`` preserves the original
unbounded behaviour so the offline corpus path is unchanged (and the
inherited tests in ``test_chunk.py`` keep passing byte-for-byte).
"""

from __future__ import annotations

from iw_rag.api.documents import make_online_chunker, online_chunk_config
from iw_rag.ingestion.chunk import ChunkConfig, HybridChunker
from iw_rag.ingestion.parse import Block, ParsedDocument
from iw_rag.settings import Settings


class CharTokenizer:
    """Deterministic char-level tokenizer for tests (exact round-trip)."""

    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(i) for i in ids)


def _single_section_doc(total_chars: int) -> ParsedDocument:
    """One heading-less text block of ``total_chars`` characters (no headings)."""
    body = "a" * total_chars
    blocks = [Block(text=body, block_type="text", section_path="")]
    return ParsedDocument(source_uri="file:///d", blocks=blocks, full_text=body)


def test_build_parents_splits_oversized_section() -> None:
    """RED before fix: one parent holds the whole section.

    GREEN after: a section larger than ``parent_max_tokens`` is split into
    multiple bounded parents, each with accumulated leaf tokens <= cap.

    The cap is measured in the same tokenizer units as ``leaf_tokens`` (the
    design §"Budget sizing constraint").  With ``CharTokenizer`` (1 token ==
    1 char) and ``leaf_tokens=100``, a parent holds at most four 100-char
    leaves; the joined content can be slightly longer (by the ``"\\n"``
    join delimiters) but the *token sum* stays within the cap.
    """
    cfg = ChunkConfig(leaf_tokens=100, parent_max_tokens=400)
    doc = _single_section_doc(total_chars=2000)
    tree = HybridChunker(cfg, CharTokenizer()).chunk(doc)

    assert len(tree.parents) > 1
    for parent in tree.parents:
        kids = [leaf for leaf in tree.leaves if leaf.parent_local_id == parent.local_id]
        token_sum = sum(len(leaf.content) for leaf in kids)
        assert token_sum <= 400, f"parent {parent.local_id} holds {token_sum} leaf tokens > 400"


def test_parent_max_tokens_none_preserves_unbounded_default() -> None:
    """REFACTOR invariant: ``parent_max_tokens=None`` is byte-for-byte the old behaviour.

    A single-section, heading-less doc with the cap disabled must yield exactly
    one parent whose content equals the original code's ``"\\n".join(leaves)``
    output — the very same content the pre-fix chunker produced.
    """
    cfg = ChunkConfig(leaf_tokens=100)  # parent_max_tokens defaults to None
    doc = _single_section_doc(total_chars=2000)
    tree = HybridChunker(cfg, CharTokenizer()).chunk(doc)

    assert len(tree.parents) == 1
    expected_content = "\n".join(leaf.content for leaf in tree.leaves)
    assert tree.parents[0].content == expected_content


def test_split_parents_keep_leaf_parent_local_id_consistent() -> None:
    """Each leaf must reference a real parent (split or original) by ``parent_local_id``."""
    cfg = ChunkConfig(leaf_tokens=100, parent_max_tokens=400)
    doc = _single_section_doc(total_chars=2000)
    tree = HybridChunker(cfg, CharTokenizer()).chunk(doc)

    parent_ids = {p.local_id for p in tree.parents}
    assert all(leaf.parent_local_id in parent_ids for leaf in tree.leaves)
    for parent in tree.parents:
        kids = [leaf for leaf in tree.leaves if leaf.parent_local_id == parent.local_id]
        assert kids, "split parent must own at least one leaf"
        token_sum = sum(len(leaf.content) for leaf in kids)
        assert token_sum <= 400


def test_split_parents_ordinals_are_per_parent() -> None:
    """Leaf ordinals must be monotonic per parent and contiguous from 0."""
    cfg = ChunkConfig(leaf_tokens=100, parent_max_tokens=400)
    doc = _single_section_doc(total_chars=2000)
    tree = HybridChunker(cfg, CharTokenizer()).chunk(doc)

    for parent in tree.parents:
        ordinals = [leaf.ordinal for leaf in tree.leaves if leaf.parent_local_id == parent.local_id]
        assert ordinals == sorted(ordinals)
        assert ordinals == list(range(len(ordinals)))


def test_oversized_single_leaf_becomes_its_own_parent() -> None:
    """A single leaf larger than the cap must still become its own parent (never drop content)."""
    cfg = ChunkConfig(leaf_tokens=100, parent_max_tokens=50)
    doc = _single_section_doc(total_chars=2000)
    tree = HybridChunker(cfg, CharTokenizer()).chunk(doc)

    assert tree.leaves, "leaves should never be dropped"
    for leaf in tree.leaves:
        assert leaf.parent_local_id is not None
    parent_ids = {p.local_id for p in tree.parents}
    assert all(leaf.parent_local_id in parent_ids for leaf in tree.leaves)


def test_online_chunk_config_wires_parent_cap_from_retrieval_settings() -> None:
    """The online ChunkConfig must source ``parent_max_tokens`` from
    ``RetrievalSettings.parent_expansion_max_chars`` so bounded parents always
    fit the runtime expansion budget (single source of truth — no second
    literal to drift).
    """
    cfg = online_chunk_config(Settings())
    assert cfg.parent_max_tokens == Settings().retrieval.parent_expansion_max_chars
    assert cfg.leaf_tokens == Settings().ingestion.online_chunk_leaf_tokens


def test_online_chunker_keeps_oversized_parents_within_budget() -> None:
    """The online chunker (CharTokenizer, 1 token == 1 char) must produce
    parents whose accumulated leaf tokens fit the runtime expansion budget.
    """
    chunker = make_online_chunker(Settings())
    doc = _single_section_doc(total_chars=50_000)  # ~ 12 parent-sized parents
    tree = chunker.chunk(doc)

    budget = Settings().retrieval.parent_expansion_max_chars
    for parent in tree.parents:
        kids = [leaf for leaf in tree.leaves if leaf.parent_local_id == parent.local_id]
        token_sum = sum(len(leaf.content) for leaf in kids)
        assert token_sum <= budget
