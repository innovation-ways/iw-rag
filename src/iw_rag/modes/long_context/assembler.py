"""Whole-document assembler for the long-context route (WI-032 / Task 4c).

Assembles document(s) into a prompt context that is:
- **Prefix-cache-friendly**: stable system/instruction prefix FIRST (byte-
  identical across calls for the same doc set), then per-doc blocks, query
  appended last by the caller.  This ordering maximises vLLM automatic prefix
  caching (APC) so repeated queries against the same document reuse the cached
  KV of the shared prefix (lld-long-context §3, R-00109/F4).
- **YaRN-free**: the assembler never references or enables the 131K YaRN
  window.  Native 32K only (lld-long-context §1 invariant).

**No store dependency**: in test/CI mode the assembler uses placeholder content
from UUIDs.  A production subclass or mock can override ``_load_doc`` to pull
from Postgres via ``ChunkRepository`` or ``DocumentRepository``.

References:
- lld-long-context.md §2 (public interface), §3 (algorithms)
- task-4c-long-context-route §3 (assembler step)
- R-00109/F4 (prefix-cache ordering)
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field

from iw_rag.domain.models import RetrievedChunk

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# AssembledContext — the output of WholeDocAssembler.assemble
# ---------------------------------------------------------------------------

#: The stable system/instruction prefix that MUST appear byte-identical at the
#: start of every AssembledContext.full_text for the same doc set.
#: This is the APC hit-rate precondition (R-00109/F4).
_STABLE_SYSTEM_PREFIX: str = (
    "SYSTEM: You are an expert document analyst. "
    "Answer the user's question based ONLY on the document content below. "
    "Cite documents at the document level using [Doc N] markers.\n\n"
)


@dataclass
class AssembledContext:
    """The assembled prompt context for the long-context route.

    Attributes:
        full_text: The full assembled text (system prefix + doc blocks).
            Query is appended by the caller so the prefix remains stable.
        token_count: Estimated token count for the full_text.
        system_prefix: The stable system/instruction prefix that starts
            ``full_text``.  Must be byte-identical across calls for the
            same doc set (APC precondition).
        chunks: Document-level :class:`~iw_rag.domain.models.RetrievedChunk`
            objects representing each loaded document.
    """

    full_text: str
    token_count: int
    system_prefix: str
    chunks: list[RetrievedChunk] = field(default_factory=list)

    def as_doc_level_chunks(self) -> list[RetrievedChunk]:
        """Return the document-level chunks for inclusion in RetrievalResult.

        Returns:
            The list of document-level RetrievedChunks with document_id
            in metadata and acl_applied=False (the sanctioned carve-out).
        """
        return list(self.chunks)


# ---------------------------------------------------------------------------
# WholeDocAssembler
# ---------------------------------------------------------------------------


class WholeDocAssembler:
    """Assemble whole document(s) into a prefix-cache-friendly prompt context.

    Ordering (lld-long-context §3, R-00109/F4):
    1. Stable ``_STABLE_SYSTEM_PREFIX`` (byte-identical across calls for the
       same doc set — the APC hit-rate precondition).
    2. Per-document blocks (stable content: the doc changes only on re-ingest).
    3. Query is appended by the caller LAST (not included here).

    YaRN is never enabled (``enable_yarn = False`` invariant).

    In this implementation, document loading is in-memory (UUIDs as content
    placeholders). A production subclass overrides ``_load_doc`` to pull from
    the :class:`~iw_rag.storage.repositories.DocumentRepository`.

    Attributes:
        enable_yarn: Always ``False`` (YaRN invariant; lld-long-context §1).
    """

    enable_yarn: bool = False  # INVARIANT: never True for this route

    async def estimate_tokens(self, doc_ids: Sequence[uuid.UUID]) -> int:
        """Estimate the number of tokens required to assemble the given docs.

        This is a fast path called by the entry predicate BEFORE full assembly.
        Uses a word-count heuristic (1 token ≈ 0.75 words) over placeholder
        content; a production implementation calls a tokenizer or uses
        stored chunk token counts from the database.

        Args:
            doc_ids: The document UUIDs whose tokens to estimate.

        Returns:
            An integer token estimate ≥ 0.
        """
        if not doc_ids:
            return 0
        # Heuristic: each doc contributes ~200 tokens of placeholder content
        # plus the system prefix. Production uses stored chunk.token_count sums.
        prefix_tokens = self._estimate_text_tokens(_STABLE_SYSTEM_PREFIX)
        doc_tokens = sum(self._estimate_doc_tokens(doc_id) for doc_id in doc_ids)
        total = prefix_tokens + doc_tokens
        logger.debug(
            "WholeDocAssembler.estimate_tokens: %d doc(s) → %d tokens",
            len(list(doc_ids)),
            total,
        )
        return total

    async def assemble(self, doc_ids: Sequence[uuid.UUID]) -> AssembledContext:
        """Assemble document content into a prefix-cache-friendly context.

        Ordering (APC-optimised, R-00109/F4):
        1. Stable system/instruction prefix (byte-identical across calls).
        2. Per-document blocks (stable; content changes only on re-ingest).
        3. Query is NOT included — it is appended by the generator so the
           stable prefix occupies maximum KV-cache headroom.

        Args:
            doc_ids: The document UUIDs to load and assemble.

        Returns:
            An :class:`AssembledContext` with ``full_text``, ``token_count``,
            ``system_prefix``, and document-level ``chunks``.
        """
        # Step 1: stable system prefix (byte-identical across calls)
        system_prefix = _STABLE_SYSTEM_PREFIX

        # Step 2: per-document blocks in stable order (deterministic by doc_id)
        doc_blocks: list[str] = []
        chunks: list[RetrievedChunk] = []
        for i, doc_id in enumerate(doc_ids):
            doc_content = await self._load_doc(doc_id)
            block_header = f"[Doc {i + 1}] Document ID: {doc_id}\n"
            block = block_header + doc_content
            doc_blocks.append(block)
            chunks.append(
                RetrievedChunk(
                    chunk_id=str(doc_id),
                    document_id=str(doc_id),
                    content=doc_content,
                    score=1.0,
                    source_uri=f"file:///doc_{doc_id}.md",
                    metadata={
                        "document_id": str(doc_id),
                        "doc_index": i,
                        "route": "long_context",
                    },
                    retriever="long_context",
                    acl_applied=False,  # sanctioned carve-out (CONVENTIONS §6)
                )
            )

        # Assemble full text: prefix + docs (query appended last by caller)
        full_text = system_prefix + "\n\n".join(doc_blocks)
        token_count = self._estimate_text_tokens(full_text)

        logger.debug(
            "WholeDocAssembler.assemble: %d doc(s) → %d tokens (prefix_first=True, yarn=False)",
            len(list(doc_ids)),
            token_count,
        )

        return AssembledContext(
            full_text=full_text,
            token_count=token_count,
            system_prefix=system_prefix,
            chunks=chunks,
        )

    async def _load_doc(self, doc_id: uuid.UUID) -> str:
        """Load a document's text content by ID.

        This implementation returns a placeholder string.  A production
        subclass overrides this method to query
        :class:`~iw_rag.storage.repositories.ChunkRepository` and concatenate
        the chunk texts for the document.

        Args:
            doc_id: The document UUID to load.

        Returns:
            The document text content.
        """
        # Placeholder content for testing; overridden in production.
        return f"[Placeholder content for document {doc_id}]"

    @staticmethod
    def _estimate_text_tokens(text: str) -> int:
        """Estimate token count using a word-count heuristic.

        Heuristic: 1 token ≈ 0.75 words (conservative; actual tokenizer
        is per-model and requires a live client; real numbers deferred to
        experiment #8 with the real vLLM tokenizer endpoint).

        Args:
            text: The text to estimate.

        Returns:
            Estimated token count as an integer.
        """
        word_count = len(text.split())
        # 1 token ≈ 0.75 words → word_count / 0.75 → word_count * 4 / 3
        return max(0, int(word_count * 4 / 3))

    @staticmethod
    def _estimate_doc_tokens(doc_id: uuid.UUID) -> int:
        """Estimate token count for a single document (placeholder).

        In production this sums chunk.token_count for all chunks belonging
        to the document (from the database).  Here we use a fixed estimate
        per document for CI / harness use.

        Args:
            doc_id: The document UUID.

        Returns:
            Estimated token count for the document.
        """
        # Conservative placeholder: ~200 tokens per doc header + content.
        _ = doc_id  # unused in placeholder; production queries DB
        return 200
