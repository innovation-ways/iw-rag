"""Chunking stage — parsed blocks → a parent→child tree with leaf token spans.

Implements tokenizer-aware chunking (lld-ingestion §2.2). The tokenizer is
injected and MUST match the embedding model (BGE-M3) so leaf token spans index
the *same* token stream the Late-Chunking embedder runs its whole-document
forward pass over. The chunker therefore tokenizes the document once and emits
``ChunkTree.doc_token_ids`` plus, per leaf, a contiguous ``TokenSpan`` into it —
guaranteeing exact span alignment for mean-pooling (lld-ingestion §2.4).

Leaves are the embedded/searched unit; parents are the small-to-big return unit
(grouped by section). DTOs are ingestion-local.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

from iw_rag.ingestion.parse import ParsedDocument

LeafType = Literal["text", "table", "figure_caption"]


@runtime_checkable
class Tokenizer(Protocol):
    """Minimal tokenizer contract; the same instance feeds chunking + embedding."""

    def encode(self, text: str) -> list[int]:
        """Encode text to token ids (no special tokens)."""
        ...

    def decode(self, ids: list[int]) -> str:
        """Decode token ids back to text."""
        ...


@dataclass(frozen=True, slots=True)
class TokenSpan:
    """A half-open ``[start, end)`` token range into ``ChunkTree.doc_token_ids``."""

    start: int
    end: int

    def __len__(self) -> int:
        """Number of tokens in the span."""
        return self.end - self.start


@dataclass(frozen=True, slots=True)
class ChunkNode:
    """A parent or leaf node (pre-persistence).

    Attributes:
        local_id: Stable within-document id (e.g. ``"l3"``/``"p1"``).
        parent_local_id: The owning parent's ``local_id`` (None for parents).
        ordinal: Monotonic position within the parent (leaves) / document (parents).
        content: The node's text.
        content_type: ``text|table|figure_caption``.
        section_path: Heading breadcrumb.
        page: Source page when known.
        token_span: Leaf token range into ``doc_token_ids`` (None for parents).
        payload: Preserved structured payload (table/figure).
        acl_tags: Group entitlements — set ONLY by ``enrich`` from the
            ``AclResolver`` (never from content/model). Empty until enriched.
    """

    local_id: str
    parent_local_id: str | None
    ordinal: int
    content: str
    content_type: LeafType
    section_path: str | None
    page: int | None
    token_span: TokenSpan | None
    payload: dict[str, Any] = field(default_factory=dict)
    acl_tags: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class ChunkTree:
    """The chunker output.

    Attributes:
        parents: Section-level merge units (small-to-big return targets).
        leaves: Embedded/searched units; reference a parent via ``parent_local_id``.
        doc_token_ids: The whole-document token stream leaf spans index into.
    """

    parents: list[ChunkNode]
    leaves: list[ChunkNode]
    doc_token_ids: list[int]

    def leaf_spans(self) -> list[TokenSpan]:
        """Return the leaf token spans in leaf order (for the embedder)."""
        return [leaf.token_span for leaf in self.leaves if leaf.token_span is not None]


@dataclass(frozen=True, slots=True)
class ChunkConfig:
    """Chunking configuration.

    Attributes:
        leaf_tokens: Target leaf size in tokens (256–512; default 384).
        parent_strategy: How parents are formed (``section`` groups by heading path).
        parent_max_tokens: Optional cap on accumulated leaf tokens per parent
            (same tokenizer units as ``leaf_tokens``).  ``None`` preserves the
            pre-fix unbounded behaviour: one parent per section, regardless of
            size (this is the default and what the offline corpus path uses).
            When set, a section whose accumulated leaves would exceed the cap
            is split into multiple bounded parents — each leaf references the
            one parent that owns it, ordinals stay per-parent and contiguous
            from 0, and a single leaf larger than the cap still becomes its
            own parent (content is never dropped).  Sized to match the runtime
            ``RetrievalSettings.parent_expansion_max_chars`` so bounded parents
            always fit the small-to-big expansion budget and expand normally
            (I-00131).
    """

    leaf_tokens: int = 384
    parent_strategy: Literal["section", "window"] = "section"
    parent_max_tokens: int | None = None


class HybridChunker:
    """Tokenizer-aware parent→child chunker with Late-Chunking token spans."""

    def __init__(self, cfg: ChunkConfig, tokenizer: Tokenizer) -> None:
        """Bind chunking config + the (BGE-M3-matched) tokenizer.

        Args:
            cfg: Chunking configuration.
            tokenizer: Tokenizer shared with the embedder for span alignment.
        """
        self._cfg = cfg
        self._tok = tokenizer

    def chunk(self, doc: ParsedDocument) -> ChunkTree:
        """Build the parent→child tree and the aligned document token stream.

        Heading blocks update the running section path and contribute their tokens
        to ``doc_token_ids`` (context) but produce no leaf. Content blocks are
        packed into leaves of at most ``leaf_tokens`` tokens; an oversized block is
        split at token boundaries. Parents group consecutive leaves by section;
        when ``parent_max_tokens`` is set, a section is split into multiple
        bounded parents (each <= cap) so a heading-less document can never
        produce a whole-document parent (I-00131).

        Args:
            doc: The parsed document.

        Returns:
            A ``ChunkTree`` with parents, leaves (each with a ``token_span``), and
            the whole-document ``doc_token_ids``.
        """
        leaf_tokens = max(1, self._cfg.leaf_tokens)
        parent_cap = max(1, self._cfg.parent_max_tokens) if self._cfg.parent_max_tokens else None
        doc_ids: list[int] = []
        leaves: list[ChunkNode] = []
        # section_path -> the parent's local_id currently being filled for that
        # section.  When ``parent_cap`` is set, a leaf that would push the
        # parent's accumulated tokens past the cap triggers a new parent in
        # the same section.
        current_parent_of_section: dict[str, str] = {}
        current_parent_tokens: dict[str, int] = {}
        # Per-parent leaf ordinal counter (resets to 0 for each new parent).
        # In the unbounded default (one parent per section) this is identical
        # to the pre-fix per-section counter, preserving byte-for-byte output.
        parent_leaf_ordinal: dict[str, int] = {}
        # Ordered list of (section, parent_local_id) — emitted in the order
        # the parents are *opened* (the order a section first appears ×
        # subsequent splits).
        parent_order: list[tuple[str, str]] = []

        def _open_parent(section: str) -> str:
            """Open a fresh parent for ``section``; record its id in order."""
            parent_local = f"p{len(parent_order)}"
            parent_order.append((section, parent_local))
            current_parent_of_section[section] = parent_local
            current_parent_tokens[section] = 0
            parent_leaf_ordinal[parent_local] = 0
            return parent_local

        for block in doc.blocks:
            ids = self._tok.encode(block.text)
            if block.block_type == "heading":
                doc_ids.extend(ids)  # context only; no leaf
                continue

            section = block.section_path or ""
            if section not in current_parent_of_section:
                _open_parent(section)

            base = len(doc_ids)
            doc_ids.extend(ids)
            pos = 0
            while pos < len(ids):
                window = ids[pos : pos + leaf_tokens]
                window_len = len(window)
                if (
                    parent_cap is not None
                    and current_parent_tokens[section] > 0
                    and current_parent_tokens[section] + window_len > parent_cap
                ):
                    _open_parent(section)
                parent_local = current_parent_of_section[section]
                span = TokenSpan(base + pos, base + pos + window_len)
                ordinal = parent_leaf_ordinal[parent_local]
                parent_leaf_ordinal[parent_local] = ordinal + 1
                leaves.append(
                    ChunkNode(
                        local_id=f"l{len(leaves)}",
                        parent_local_id=parent_local,
                        ordinal=ordinal,
                        content=self._tok.decode(window),
                        content_type=block.block_type,
                        section_path=block.section_path,
                        page=block.page,
                        token_span=span,
                        payload=block.payload,
                    )
                )
                current_parent_tokens[section] = current_parent_tokens[section] + window_len
                pos += leaf_tokens

        parents = self._build_parents(parent_order, leaves)
        return ChunkTree(parents=parents, leaves=leaves, doc_token_ids=doc_ids)

    @staticmethod
    def _build_parents(
        parent_order: list[tuple[str, str]],
        leaves: list[ChunkNode],
    ) -> list[ChunkNode]:
        """Form one parent per (section, parent_slot) from its leaves.

        ``parent_order`` is the ordered list of (section, parent_local_id)
        pairs emitted by :meth:`chunk` in the order their parents were
        *opened*.  Content is the ``"\\n"`` join of the parent's leaves — the
        same composition rule as the pre-fix single-parent-per-section path,
        so ``parent_max_tokens=None`` is byte-for-byte the original output
        (REFACTOR invariant).
        """
        children: dict[str, list[ChunkNode]] = {
            parent_local: [] for _section, parent_local in parent_order
        }
        for leaf in leaves:
            pid = leaf.parent_local_id
            if pid is not None and pid in children:
                children[pid].append(leaf)
        parents: list[ChunkNode] = []
        for ordinal, (section, parent_local) in enumerate(parent_order):
            kids = children[parent_local]
            parents.append(
                ChunkNode(
                    local_id=parent_local,
                    parent_local_id=None,
                    ordinal=ordinal,
                    content="\n".join(k.content for k in kids),
                    content_type="text",
                    section_path=section or None,
                    page=kids[0].page if kids else None,
                    token_span=None,
                )
            )
        return parents
