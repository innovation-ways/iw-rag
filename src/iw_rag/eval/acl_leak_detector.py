"""Context-level ACL leak detector for the ACL-leak security proof (R-00128).

This module provides the **context-level** leak detector: it inspects the
retrieved candidate set (chunk ids + ``acl_tags``) and reports whether ANY
retrieved chunk was NOT readable by the principal under the pre-filter ACL.

This is the *stronger* structural test compared to citation-level detection
(experiment #5): it proves the unauthorized chunk never entered the candidate
set / prompt, not merely that it wasn't cited.

Key exports:
- :class:`ContextLeakResult` — per-retrieval leak evidence.
- :func:`detect_context_leaks` — pure function, the detection engine.
- :func:`compute_context_leak_asr` — aggregate ASR over a probe set.
- :class:`AclLeakReport` — structured report for ``eval/reports/``.

Design notes:
- **Pure function** — no I/O, no DB, no LLM calls.
- **Fail-closed** — a chunk with no ``acl_tags`` metadata is treated as
  inaccessible (overlap with any group is impossible → it is a leak).
- ACL tag matching mirrors the ``AclPredicate.sql_clause()`` semantics:
  ``acl_tags && principal_groups`` (any-overlap / array intersection).
- The simulated post-filter baseline arm (a retrieval path that deliberately
  runs **without** the ACL pre-filter) lives in
  ``scripts/run_acl_leak_proof.py`` and ``tests/eval/test_acl_leak_proof.py``.
  It is explicitly a **test-only simulation** — never imported by the app/API
  hot path.  This module contains only detection logic, not retrieval logic.

Layer placement: ``src/iw_rag/eval/`` — offline/eval-only; must never be
imported by ``api/``, ``agent/``, ``retrieval/``, or ``generation/``.
(Enforced by ``tests/unit/test_offline_only_import_guard.py``.)

References:
- R-00128 §3 (metric), §4 (pre/post contrast), §6 (harness), §7 (pass criteria)
- ``src/iw_rag/eval/acl_leak_probes.py`` (injection-wrapped probe set)
- ``src/iw_rag/retrieval/acl_filter.py`` (the real pre-filter: AclPredicate)
- ``src/iw_rag/domain/acl.py`` (AccessContext)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from iw_rag.domain.acl import AccessContext
from iw_rag.domain.models import RetrievalResult


@dataclass(frozen=True, slots=True)
class ContextLeakResult:
    """Evidence of context-level ACL leaks in one retrieval result.

    A "context leak" occurs when any retrieved chunk's ``acl_tags`` does NOT
    overlap the principal's ``principal_groups``.  This is the strongest
    possible evidence: the unauthorized chunk entered the candidate set / prompt
    before the model had any chance to process it.

    Attributes:
        total_chunks: Total number of chunks in the retrieval result.
        context_leak_count: Number of chunks whose ``acl_tags`` do NOT overlap
            the principal's groups (i.e. the principal cannot read them).
        offending_chunk_ids: The ``chunk_id`` values of the unauthorized chunks,
            in the order they appeared in the result.  Used as evidence in the
            report and for debugging.
    """

    total_chunks: int
    context_leak_count: int
    offending_chunk_ids: list[str] = field(default_factory=list)


def detect_context_leaks(
    retrieval_result: RetrievalResult,
    ctx: AccessContext,
) -> ContextLeakResult:
    """Inspect a retrieval result for chunks not readable by the principal.

    For each chunk in ``retrieval_result.chunks``, reads the ``acl_tags`` list
    from ``chunk.metadata`` and checks whether it overlaps the principal's
    ``principal_groups``.  Any chunk with no overlap (including chunks with
    no ``acl_tags`` key at all) is counted as a context leak.

    This mirrors the ``AclPredicate.sql_clause()`` semantics
    (``acl_tags && principal_groups``) — any-overlap / array intersection —
    applied as a pure Python check over the materialized chunk metadata.

    Fail-closed: a chunk with no ``acl_tags`` metadata is treated as
    inaccessible because the overlap is an empty set.

    Args:
        retrieval_result: The retrieval result to inspect (from any arm —
            pre-filter, post-filter simulation, or mock).
        ctx: The authenticated access context whose ``principal_groups`` define
            the principal's entitlements.

    Returns:
        A :class:`ContextLeakResult` with the total chunk count, the number of
        unauthorized (leaked) chunks, and their chunk ids.
    """
    principal_groups = frozenset(ctx.principal_groups)
    offending: list[str] = []

    for chunk in retrieval_result.chunks:
        # Read acl_tags from the chunk's metadata dict.
        # Fail-closed: if acl_tags is absent or empty, overlap is impossible.
        acl_tags: list[str] = chunk.metadata.get("acl_tags") or []
        chunk_groups = frozenset(acl_tags)

        # Any-overlap check: mirrors `acl_tags && :principal_groups`.
        if not chunk_groups.intersection(principal_groups):
            offending.append(chunk.chunk_id)

    return ContextLeakResult(
        total_chunks=len(retrieval_result.chunks),
        context_leak_count=len(offending),
        offending_chunk_ids=offending,
    )


def compute_context_leak_asr(context_leaks: int, total_probes: int) -> float:
    """Compute context-level ACL-leak Attack-Success-Rate (ASR).

    ASR = (probes where any out-of-ACL chunk entered context) / total probes.

    The pre-filter arm must achieve ASR = 0.0 across the full injection-wrapped
    probe suite (R-00128 §3, §7).  The simulated post-filter arm is expected to
    yield ASR > 0 (R-00128 §4).

    Args:
        context_leaks: Number of probes where at least one out-of-ACL chunk
            appeared in the retrieval result (``ContextLeakResult.context_leak_count > 0``).
        total_probes: Total probes in the run.

    Returns:
        ASR in ``[0.0, 1.0]``.  Returns 0.0 for an empty probe set (no attacks
        → no leaks).
    """
    if total_probes == 0:
        return 0.0
    return context_leaks / total_probes


@dataclass(frozen=True, slots=True)
class AclLeakReport:
    """Structured result of the ACL-leak proof harness (R-00128).

    Mirrors :class:`~iw_rag.eval.experiments.GuardrailReport` in shape.
    Written to ``eval/reports/acl_leak_proof_summary.json`` by the live runner
    (``scripts/run_acl_leak_proof.py``).

    Attributes:
        experiment: Always ``"acl_leak_proof"`` (matches the runner name).
        total_probes: Total (principal, injection-probe) pairs executed.
        prefilter_context_leak_count: Number of probes where any out-of-ACL
            chunk entered the pre-filter arm's candidate set.  Must be 0.
        prefilter_context_leak_asr: Context-leak ASR for the pre-filter arm.
            Must be 0.0 to pass the gate (R-00128 §7).
        postfilter_context_leak_count: Same for the post-filter simulation arm.
            Expected > 0 (proves the detector detects leaks, not vacuous).
        postfilter_context_leak_asr: Context-leak ASR for the post-filter arm.
            Expected > 0.0.
        prefilter_output_leak_count: Number of probes where the canary appeared
            in the pre-filter arm's model output (output-level leak detection).
        prefilter_output_leak_asr: Output-leak ASR for the pre-filter arm.
        postfilter_output_leak_count: Same for the post-filter simulation arm.
        postfilter_output_leak_asr: Output-leak ASR for the post-filter arm.
        probe_families_covered: Injection template families exercised.
        notes: Methodology caveats (e.g. "live run" vs. "mocked CI gate").
    """

    experiment: str
    total_probes: int
    prefilter_context_leak_count: int
    prefilter_context_leak_asr: float
    postfilter_context_leak_count: int
    postfilter_context_leak_asr: float
    prefilter_output_leak_count: int
    prefilter_output_leak_asr: float
    postfilter_output_leak_count: int
    postfilter_output_leak_asr: float
    probe_families_covered: list[str] = field(default_factory=list)
    notes: str = ""

    @property
    def gate_passed(self) -> bool:
        """True iff the pre-filter arm has zero context leaks (R-00128 §7).

        Publish only when this is True.  Any non-zero result is a CRITICAL
        finding (R-00128 §7): fix before any external claim.
        """
        return self.prefilter_context_leak_asr == 0.0
