"""GraphState and supporting types for the LangGraph agentic loop.

The ``GraphState`` TypedDict is the single shared mutable contract that all
Phase-2 nodes read and write.  Using ``total=False`` lets each node return a
**partial** dict; LangGraph merges the partial into the accumulated state.

The ``verdicts`` field uses ``Annotated[list[Verdict], operator.add]`` so every
grader node can *append* its verdict(s) without a full-state read.

See: docs/build/02-lld/lld-agentic-loop.md §2.1 for the authoritative spec.
CONVENTIONS.md §1: the retrieved-item field is ``chunks: list[RetrievedChunk]``
— never ``Candidate``/``candidates``/``passages``.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Annotated, Any, Literal, TypedDict

from iw_rag.domain.acl import AccessContext
from iw_rag.domain.models import Citation, RetrievedChunk

# ---------------------------------------------------------------------------
# Enumerated string types
# ---------------------------------------------------------------------------

Route = Literal["skip", "retrieve", "graph", "long_context", "visual"]
"""Which retrieval branch the router selected."""

DocVerdict = Literal["correct", "incorrect", "ambiguous"]
"""CRAG 3-action aggregate verdict over retrieved chunks (ISREL):
correct → generate · incorrect → rewrite+re-retrieve · ambiguous → generate."""

GenVerdict = Literal["grounded", "unsupported", "not_answering"]
"""Self-RAG externalized grade for a generated draft (ISSUP + ISUSE)."""


def _merge_diagnostics(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """Shallow-merge two ``diagnostics`` contributions (right wins on key clash).

    The ``diagnostics`` channel accumulates per-node contributions (retrieval,
    sanitize ``sanitize_dropped_ids`` for the Task-3.4 audit, generation, the
    terminal ``trace_ref``). Without a reducer LangGraph would keep only the
    last writer, silently dropping earlier keys; this preserves them all.
    """
    return {**left, **right}


# ---------------------------------------------------------------------------
# Verdict dataclass — the append-only audit trail
# ---------------------------------------------------------------------------


@dataclass
class Verdict:
    """A single grader verdict appended to the ``verdicts`` trail.

    Each grader node appends one :class:`Verdict` per invocation; the
    ``Annotated[list[Verdict], operator.add]`` reducer prevents overwrites so
    the full decision trail is preserved for audit and Phoenix/OTEL tracing.

    Attributes:
        kind: Which node emitted the verdict (e.g. ``"router"``,
            ``"grade_documents"``, ``"grade_generation"``).
        value: The string decision (e.g. ``"retrieve"``, ``"relevant"``,
            ``"grounded"``).
        score: Optional numeric confidence in ``[0, 1]``.
        reason: Short free-text reason or model output snippet.
        attempt: Zero-based retry pass this verdict belongs to.
    """

    kind: str
    value: str
    score: float | None = None
    reason: str = ""
    attempt: int = 0


# ---------------------------------------------------------------------------
# GraphState — the spine of the agentic loop
# ---------------------------------------------------------------------------


class GraphState(TypedDict, total=False):
    """LangGraph state shared across all Phase-2 nodes.

    ``total=False`` means every key is optional — nodes write **partial**
    dicts and LangGraph merges them.  Do not mutate and return the whole state
    from a node; return only the keys you update.

    Attributes:
        question: Current (possibly rewritten) query text.
        original_question: The user's original phrasing — never modified.
        access_context: Authenticated principal + groups (from auth, never LLM).

        chunks: Raw ``RetrievalResult.chunks`` from the spine (current pass).
            CONVENTIONS §1: must be ``list[RetrievedChunk]``, never candidates.
        graded: Subset of ``chunks`` kept after ``grade_documents`` (relevant +
            ambiguous).
        verdicts: Append-only trail of every grade; uses the ``operator.add``
            reducer so each node *appends* rather than overwrites.
        retries: Correction passes consumed so far (shared budget).
        route: Which retrieval branch the router selected.

        input_decision: The guard decision from ``input_guard_node``:
            ``"allow"`` or ``"block"``.  Read by ``route_after_input_guard``
            to determine the next node (Task 3.1).
        answer: Final answer text on success (``disposition="answered"``).
        citations: Chunk-ID citations parsed from the draft.
        used_chunk_ids: IDs of chunks the answer actually used.
        disposition: Terminal state: ``"answered"`` or ``"refused"``.
        diagnostics: Merged retrieval + generation diagnostics (operator-only).
    """

    # --- inputs (set once at invocation) ---
    question: str
    original_question: str
    access_context: AccessContext

    # --- chat memory (set once at invocation) ---
    # Prior conversation turns of the SAME thread, pre-built into citable,
    # principal-scoped synthetic chunks (``metadata["kind"]=="conversation"``).
    # The retrieve node prepends these to the spine's chunks so the agent can
    # ground a follow-up answer in what was said earlier ("what is my name?"
    # after "my name is Sergio") and CITE the originating turn — preserving the
    # fail-closed grounding contract rather than bypassing it. Absent/empty on
    # the first turn or when memory is off (then behaviour is unchanged).
    history_chunks: list[RetrievedChunk]

    # --- routing ---
    query_type: str | None
    route: Route | None

    # --- retrieval / grading ---
    chunks: list[RetrievedChunk]
    graded: list[RetrievedChunk]
    doc_verdict: DocVerdict | None

    # --- generation / verification ---
    draft: str | None
    citations: list[Citation]
    used_chunk_ids: list[str]
    gen_verdict: GenVerdict | None

    # --- append-only verdict trail (operator.add reducer) ---
    verdicts: Annotated[list[Verdict], operator.add]

    # --- budget ---
    retries: int
    max_retries: int

    # --- assist mode (set once at invocation) ---
    # When True, the output guard treats groundedness as *informational* and
    # returns an ungrounded answer (marked) instead of fail-closing. Every other
    # guard stays enforced. Defaults absent/False (strict fail-closed).
    allow_ungrounded: bool

    # --- guard decisions ---
    # input_guard_node → route_after_input_guard
    input_decision: str | None
    # output_guard_node → route_after_output_guard (Task 3.3)
    output_guard_decision: str | None

    # --- terminal ---
    answer: str | None
    refusal: str | None
    disposition: Literal["answered", "refused"] | None

    # --- observability (merged across nodes; see _merge_diagnostics) ---
    diagnostics: Annotated[dict[str, Any], _merge_diagnostics]
