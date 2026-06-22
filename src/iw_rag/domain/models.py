"""Canonical domain types and the retrieval contract.

These names are mandatory everywhere (CONVENTIONS.md §1). The retriever's return
object is :class:`RetrievalResult` with ``.chunks: list[RetrievedChunk]`` — never
``Candidate``/``.candidates``/``passages``. Per-item identity is ``source_uri`` +
``metadata`` (never ``source_metadata``). The same contract is returned by the
Phase-0 spine **and** every Phase-4 mode, so the generator and eval stay
mode-agnostic.

These are scaffold stubs (types only); fields are fleshed out per task. The
shapes here are the authoritative contract the later tasks build against.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

# ---------------------------------------------------------------------------
# Guard types — lld-guardrails §2
# ---------------------------------------------------------------------------

#: The four possible guard decision labels (canonical; lld-guardrails §2).
Decision = Literal["allow", "block", "redact", "flag"]

#: ``RetrievedChunk.metadata["kind"]`` marker for a synthetic chat-memory chunk
#: (a prior conversation turn replayed as citable evidence).  Such chunks are
#: principal-scoped by construction and carry the user's / assistant's own
#: words, so the sanitize node exempts them from injection-scan / PII-redaction /
#: datamarking — redacting "Sergio" out of the user's own "my name is Sergio"
#: would defeat the memory.  The retrieve node tags them via this key.
CONVERSATION_CHUNK_KIND = "conversation"


@dataclass(frozen=True)
class GuardVerdict:
    """A typed verdict returned by any guard layer (input, sanitize, output).

    Defined in ``domain/`` so ``guardrails/`` and ``agent/`` share the type
    without creating an import cycle (lld-guardrails §2).

    Attributes:
        layer: Which guard layer produced this verdict: ``"input"``,
            ``"sanitize"``, or ``"output"``.
        decision: The guard decision — one of ``"allow"``, ``"block"``,
            ``"redact"``, or ``"flag"`` (:data:`Decision`).
        score: Numeric guard confidence / risk score in ``[0, 1]``.
        reason: Machine tag + human-readable text, e.g. ``"injection:0.91"``
            or ``"clean"``.
        detail: Optional per-guard detail dict (e.g.
            ``{"prompt_guard": 0.9, "granite": 0.4}``); defaults to an empty
            dict so callers can safely ``v.detail.get("key")``.
    """

    layer: str
    decision: Decision
    score: float
    reason: str
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """One retrieved unit returned inside a :class:`RetrievalResult`.

    Attributes:
        chunk_id: Stable id of the chunk.
        document_id: The owning ``documents.id`` primary key (UUID-as-string).
            Distinct from any UUID embedded in ``source_uri`` (which is a
            *storage* uuid, not the PK the download endpoint keys on). The
            download-token flow (CR-00098) uses this to mint
            ``GET /documents/{document_id}/raw`` links.
        content: The chunk text used for grounding/citation.
        score: Final ranking score (post-fusion/rerank).
        source_uri: Where the chunk came from (document URI).
        metadata: Free-form per-item metadata (section path, content type, ...).
        leaf_content: The original leaf span when a parent was expanded
            (small-to-big); ``None`` when the chunk is itself a leaf.
        retriever: Discriminator naming the retriever/mode that produced this
            chunk (e.g. ``"spine"``, ``"visual"``).  Optional; used by Phase-4
            modes to tag each chunk with its origin.
        acl_applied: Whether ACL filtering was applied when this chunk was
            retrieved.  Optional; visual mode sets to ``True`` (post-filter).
    """

    chunk_id: str
    document_id: str
    content: str
    score: float
    source_uri: str
    metadata: dict[str, Any] = field(default_factory=dict)
    leaf_content: str | None = None
    retriever: str = ""
    acl_applied: bool = False


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """The retriever's return object — the contract every consumer reuses.

    Attributes:
        chunks: Ranked retrieved units (never ``candidates``/``passages``).
        retriever: Discriminator naming the retriever/mode that produced it
            (e.g. ``"spine"``, ``"graphrag"``, ``"visual"``).
        acl_applied: Whether the ACL pre-filter was applied. MUST be ``True`` on
            the hot path; the long-context route is the single sanctioned
            ``False`` carve-out (CONVENTIONS.md §6).
    """

    chunks: list[RetrievedChunk]
    retriever: str
    acl_applied: bool
    query: str = ""
    expansion_mode: str = "leaf"
    candidate_count: int = 0
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SanitizedContext:
    """The result of sanitizing a list of retrieved chunks (lld-guardrails §3.3).

    Returned by ``guardrails.sanitize.sanitize_context``.  Contains only
    surviving chunks (injection-scanned, PII-redacted, and datamarked) plus
    per-chunk verdicts and the ids of any dropped chunks.

    Defined in ``domain/`` so ``guardrails/``, ``agent/``, and ``api/`` can
    share the type without import cycles (CONVENTIONS §1).

    Attributes:
        chunks: Surviving chunks — datamarked + PII-redacted.  Poisoned chunks
            are absent (``on_poison=drop``) or present with
            ``metadata["poisoned"]=True`` (``on_poison=flag``).
        verdicts: Per-chunk ``GuardVerdict`` objects for injection-scan and
            PII-redact results.
        dropped_chunk_ids: Ids of chunks excluded because they were flagged as
            injections under the ``drop`` policy.
    """

    chunks: list[RetrievedChunk]
    verdicts: list[GuardVerdict]
    dropped_chunk_ids: list[str]


@dataclass(frozen=True, slots=True)
class Citation:
    """A resolved citation linking answer text back to a source chunk.

    Attributes:
        marker: The ``[n]`` marker emitted in the answer.
        chunk_id: The chunk the marker resolves to.
        document_id: The owning document's ``documents.id`` primary key
            (UUID-as-string). Carried so the API layer can mint a per-citation
            ``download_url`` addressing ``GET /documents/{document_id}/raw``
            (CR-00098).  Distinct from any UUID embedded in ``source_uri``.
        source_uri: The chunk's document URI.
        quote: The supporting span (retains the matched leaf on expansion).
    """

    marker: int
    chunk_id: str
    document_id: str
    source_uri: str
    quote: str = ""


@dataclass(frozen=True, slots=True)
class GeneratedAnswer:
    """A generated answer plus its grounding/citation provenance.

    Attributes:
        answer: The answer text (citation markers preserved).
        grounded: Whether the answer is grounded in retrieved evidence; an
            ungrounded/unanswerable query yields ``False`` (honest refusal),
            never a fabrication.
        citations: Resolved citations; hallucinated markers are dropped.
        used_chunk_ids: Ids of the chunks the answer actually used.
    """

    answer: str
    grounded: bool
    citations: list[Citation] = field(default_factory=list)
    used_chunk_ids: list[str] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
