"""Per-request step timeline — the data behind the UI's timing breakdown.

Every agent node already wraps its body in an ``observability.tracing.span``;
this module piggybacks on those spans to record, per request, how long each
pipeline step took — independently of OpenTelemetry/Phoenix (it works even when
the collector is down or tracing is disabled).  ``answer_question`` opens a
:func:`start_timeline` for the duration of one graph invocation; ``span``
appends a :class:`StepRecord` to whichever timeline is active (a no-op when
none is).  :func:`build_breakdown` then turns the raw records into the
human-readable, per-step breakdown the frontend renders in its "how long did
this take?" popup.

Why not just read Phoenix?  The breakdown must ship *inside* the ``/query``
response (the frontend is a pure API client and never queries Phoenix), must
work with tracing turned off, and needs plain-language descriptions of what each
step *did* — not just a span name.  This module owns that mapping.

Contains NO sensitive data: only span names, durations, and an allowlist of
non-identifying attributes (counts, verdict labels) — never raw content,
questions, or principal groups.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class StepRecord:
    """One timed span occurrence within a request.

    Attributes:
        name: The span name (e.g. ``iw_rag.agent.grade_documents``).
        start_ms: Milliseconds from the timeline's start to this span's start
            (used to order steps chronologically across retries).
        duration_ms: Wall-clock duration of the span in milliseconds.
        attributes: The span attributes (filtered to a safe allowlist before
            they ever leave the process).
    """

    name: str
    start_ms: float
    duration_ms: float
    attributes: dict[str, Any]


@dataclass(slots=True)
class Timeline:
    """A request's accumulating list of step records."""

    t0: float
    records: list[StepRecord] = field(default_factory=list)


_TIMELINE: ContextVar[Timeline | None] = ContextVar("iw_rag_timeline", default=None)


@contextmanager
def start_timeline() -> Iterator[Timeline]:
    """Activate a fresh timeline for the duration of one graph invocation.

    Spans entered while this is active append their timing to the returned
    :class:`Timeline`.  Nested invocations are isolated by the contextvar token.

    Yields:
        The active :class:`Timeline` (read its ``records`` after the block).
    """
    timeline = Timeline(t0=time.perf_counter())
    token = _TIMELINE.set(timeline)
    try:
        yield timeline
    finally:
        _TIMELINE.reset(token)


def record_span(name: str, start_perf: float, end_perf: float, attributes: Any) -> None:
    """Append a completed span to the active timeline (no-op if none active).

    Called by :func:`iw_rag.observability.tracing.span` on span exit.  Cheap and
    exception-free so it can sit on the hot path of every node.

    Args:
        name: The span name.
        start_perf: ``time.perf_counter()`` captured at span entry.
        end_perf: ``time.perf_counter()`` captured at span exit.
        attributes: The span attributes dict (or ``None``).
    """
    timeline = _TIMELINE.get()
    if timeline is None:
        return
    timeline.records.append(
        StepRecord(
            name=name,
            start_ms=(start_perf - timeline.t0) * 1000.0,
            duration_ms=(end_perf - start_perf) * 1000.0,
            attributes=dict(attributes) if attributes else {},
        )
    )


# ---------------------------------------------------------------------------
# Step catalog — span name → (label, what it does functionally, how technically).
# Only these "primary" pipeline steps are surfaced in the breakdown; finer-grain
# spans (retrieval sub-stages, the grade-batch wrapper) are recorded but folded
# into their parent step's wall time and omitted from the user-facing list.
# ---------------------------------------------------------------------------

_STEP_CATALOG: dict[str, tuple[str, str, str]] = {
    "iw_rag.agent.input_guard": (
        "Input guard",
        "Screens your question for prompt-injection and unsafe content before any "
        "retrieval or generation runs.",
        "Runs the Prompt-Guard / safety classifier over the raw question; a block "
        "verdict short-circuits the request to a refusal (fail-closed).",
    ),
    "iw_rag.agent.route": (
        "Route",
        "Decides how to answer — look in your documents, answer directly, or use a "
        "specialised retrieval mode.",
        "Adaptive-RAG classifier: one constrained structured-output call returning a "
        "query_type + route (skip / retrieve / graph / long_context / visual).",
    ),
    "iw_rag.agent.retrieve": (
        "Retrieve",
        "Searches your authorised documents for the passages most relevant to the question.",
        "ACL pre-filtered hybrid search: BGE-M3 dense + sparse candidates fused by "
        "RRF, reranked (bge-reranker-v2-m3), small-to-big parent expansion → top-k.",
    ),
    "iw_rag.guardrails.sanitize": (
        "Sanitize context",
        "Cleans the retrieved passages — strips embedded instructions, redacts PII, "
        "and marks the text as untrusted data.",
        "Per-chunk injection scan, Presidio PII redaction, then datamark "
        "spotlighting (U+2063 markers) so injected instructions can't steer the "
        "generator.",
    ),
    "iw_rag.agent.grade_documents": (
        "Grade documents",
        "Judges which retrieved passages are actually relevant and drops the rest.",
        "Batched CRAG/ISREL relevance grader (one structured-output call); token-"
        "budgeted to the model window; aggregates a correct/ambiguous/incorrect "
        "verdict that drives the corrective-retrieval loop.",
    ),
    "iw_rag.agent.rewrite_query": (
        "Rewrite query",
        "Reformulates the question to search again when the first passages weren't "
        "relevant enough.",
        "Single-hop reformulation anchored on the original question + the failure "
        "signal; consumes one unit of the shared retry budget.",
    ),
    "iw_rag.generation.answer": (
        "Generate answer",
        "Writes the answer using only the graded passages, citing each claim.",
        "Citation-required generation (Qwen3-14B-AWQ via vLLM); sources token-"
        "budgeted to the window; reasoning + datamarks stripped from the output.",
    ),
    "iw_rag.agent.grade_generation": (
        "Grade answer",
        "Checks the drafted answer is supported by the cited passages and actually "
        "answers the question.",
        "Two externalised graders — ISSUP (groundedness) + ISUSE (usefulness) — run "
        "concurrently as structured-output calls; the aggregate gen_verdict gates "
        "the output guard.",
    ),
    "iw_rag.guardrails.output_guard": (
        "Output guard",
        "Final safety pass: verifies every citation resolves to a real source and "
        "scrubs any residual PII before the answer is shown.",
        "Deterministic citation validation against the graded chunk-id set + the "
        "one groundedness authority (gen_verdict) + Presidio redaction; a hard "
        "failure routes to fail-closed.",
    ),
    "iw_rag.agent.done": (
        "Finalize",
        "Promotes the verified draft to the final answer and records the outcome.",
        "Terminal finaliser: sets disposition=answered on a grounded verdict (or a "
        "marked general-knowledge answer in assist mode) and stamps the trace ref.",
    ),
}

#: Span attributes that are safe to surface in the breakdown (counts + verdict
#: labels only — NEVER content, questions, or principal groups).
_SAFE_ATTRS: frozenset[str] = frozenset(
    {
        "attempt",
        "chunk_count",
        "graded_count",
        "has_citations",
        "gen_verdict",
        "node",
    }
)


def _safe_attrs(attributes: dict[str, Any]) -> dict[str, Any]:
    """Filter span attributes to the non-identifying allowlist."""
    return {k: v for k, v in attributes.items() if k in _SAFE_ATTRS and k != "node"}


def build_breakdown(timeline: Timeline | None) -> list[dict[str, Any]]:
    """Turn raw step records into the ordered, described breakdown for the UI.

    Only cataloged "primary" steps are surfaced; each occurrence (including
    corrective-loop repeats) becomes its own row, ordered by start time, with a
    plain-language ``what`` (functional) and ``how`` (technical) description and
    a safe subset of its attributes.

    Args:
        timeline: The request's timeline (``None`` → empty breakdown).

    Returns:
        A list of step dicts: ``{key, label, duration_ms, what, how, attributes}``.
    """
    if timeline is None:
        return []
    steps: list[dict[str, Any]] = []
    for record in sorted(timeline.records, key=lambda r: r.start_ms):
        meta = _STEP_CATALOG.get(record.name)
        if meta is None:
            continue
        label, what, how = meta
        steps.append(
            {
                "key": record.name,
                "label": label,
                "duration_ms": round(record.duration_ms, 1),
                "what": what,
                "how": how,
                "attributes": _safe_attrs(record.attributes),
            }
        )
    return steps
