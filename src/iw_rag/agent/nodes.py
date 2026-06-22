"""LangGraph node factory functions for the Phase-2/3 agentic loop.

All node *functions* are ``async`` (they issue model I/O); factory functions
(``make_*``) are sync constructors that close over the injected clients and
settings.

Phase-2.1 delivers:
- ``make_router`` — the Adaptive-RAG complexity-classifier node.
- ``make_answer_direct`` — the ``skip``-branch parametric-answer node.

Phase-2.3 adds:
- ``make_grade_documents`` — CRAG/ISREL batched relevance grader.
- ``_aggregate_doc_verdict`` — pure 3-action aggregation helper (unit-tested).
- ``make_rewrite_query`` — single-hop query reformulation from original intent.
- ``consume_retry`` — the single authority that increments ``state["retries"]``.

Phase-2.4 adds:
- ``make_generate`` — invoke citation-required assembler over ``state["graded"]``.
- ``make_grade_generation`` — ISSUP + ISUSE batched graders; ``gen_verdict``.
- ``_aggregate_gen_verdict`` — pure 4-combo aggregation (unit-tested).
- ``done`` — terminal finaliser: promotes draft → answer, sets disposition.

Phase-2.5 adds:
- ``fail_closed`` — terminal refusal node: ``disposition="refused"``, honest
  message, no ungrounded answer.  This is the enforcement seam for budget
  exhaustion and the guardrails entry point (Phase 3).

Phase-3.1 adds:
- ``input_guard_node`` — calls ``guardrails.input_guard.scan_input``, appends
  the verdict dict to the trail, and sets ``state["input_decision"]`` for the
  ``route_after_input_guard`` edge.
- ``reject`` — terminal node for blocked input turns: sets
  ``disposition="refused"``, ``refusal="request rejected by input guard"``,
  appends a reject verdict (Task 3.4 audit node will follow this node in 3.4).

Phase-3.2 adds:
- ``sanitize_node`` — calls ``guardrails.sanitize.sanitize_context`` on
  ``state["chunks"]``, writes sanitized chunks back, appends per-chunk
  verdicts mapped to ``Verdict(kind="sanitize", ...)`` and records
  ``dropped_chunk_ids`` in the diagnostics.  Inserted between ``retrieve``
  and ``grade_documents``; respects the ``guardrails.enabled`` master switch.

Phase-3.3 adds:
- ``output_guard_node`` — runs deterministic citation validation + PII redaction
  (Presidio, reusing Task 3.2's path) + reads the loop's existing groundedness
  verdict from ``state["gen_verdict"]`` (the ISSUP result from
  ``grade_generation``) — the ONE groundedness authority, no double model call
  (lld-agentic-loop §3.5).  Decides allow/block; on block routes to
  ``fail_closed``.  Respects the ``guardrails.enabled`` master switch.

Phase-3.4 adds:
- ``make_audit_node`` — factory that returns an async node function which
  assembles the :class:`~iw_rag.guardrails.audit.AuditRecord` from the terminal
  state and calls :func:`~iw_rag.guardrails.audit.write_audit`.  Wired as the
  LAST node before END on every terminal path: ``done → audit → END``,
  ``fail_closed → audit → END``, ``reject → audit → END``.

  D11 [DECISION]: When no session factory / Postgres is wired (unit loop tests,
  no DB), ``build_graph`` passes ``audit_repo=None``.  ``make_audit_node``
  detects this and returns a no-op pass-through node so the loop mechanics unit
  tests (``agent/conftest.py``) stay green without requiring a DB.  The audit
  trail is not gated by ``guardrails.enabled`` — it runs for all requests when a
  DB is wired (regardless of the master switch) because the trail is the
  differentiator, not a guard. The no-op behaviour is the test-scaffolding
  carve-out only.

References:
- task-2.1-router-node §2, §3
- task-2.3-grade-documents-and-rewrite §2, §3, §4, §6
- task-2.4-generate-and-grade-generation §1–§7
- task-2.5-budgets-and-fail-closed §2, §3, §4, §5
- task-3.1-input-guard §4 (input_guard_node, reject node)
- task-3.2-context-sanitization §5 (sanitize_node)
- task-3.3-output-guard-grounding §4 (output_guard_node)
- task-3.4-audit-log §4 (audit_node assembly), §5 (terminal routing),
  §6 (audit-write failure = fail-closed)
- lld-agentic-loop §2.2, §3.1–3.5, §3.6, §6.1, §7 (invariant)
- lld-guardrails §3.2, §3.3, §3.4, §3.5, §5 (fail-closed on model unavailability)
- CONVENTIONS.md §1 (chunks: list[RetrievedChunk], never Candidate), §4
- R-00105 F1/F2/F3/F4/F5/F8 (corrective retrieval, externalized graders, batching)
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Callable, Coroutine, Iterator
from typing import Any

from iw_rag.agent.prompts import (
    ANSWER_DIRECT_SYSTEM,
    DOC_GRADE_MAX_TOKENS,
    DOC_GRADE_SCHEMA,
    DOC_GRADE_SYSTEM,
    DOC_GRADE_TEMPERATURE,
    ISSUP_MAX_TOKENS,
    ISSUP_SCHEMA,
    ISSUP_SYSTEM,
    ISSUP_TEMPERATURE,
    ISUSE_MAX_TOKENS,
    ISUSE_SCHEMA,
    ISUSE_SYSTEM,
    ISUSE_TEMPERATURE,
    REWRITE_MAX_TOKENS,
    REWRITE_SYSTEM,
    REWRITE_TEMPERATURE,
    ROUTER_MAX_TOKENS,
    ROUTER_SCHEMA,
    ROUTER_SYSTEM,
    ROUTER_TEMPERATURE,
)
from iw_rag.agent.state import GenVerdict, GraphState, Verdict
from iw_rag.domain.models import CONVERSATION_CHUNK_KIND, RetrievalResult
from iw_rag.generation.answer import strip_datamarks, strip_reasoning
from iw_rag.generation.context_budget import (
    ContextBudget,
    estimate_tokens,
    fit_items_to_budget,
)
from iw_rag.generation.llm import GenerationParams, LlmClient
from iw_rag.generation.prompts import REFUSAL_TEXT as _GENERATION_REFUSAL_TEXT
from iw_rag.guardrails.output_guard import validate_citations_from_list
from iw_rag.guardrails.sanitize import _presidio_redact
from iw_rag.observability.tracing import current_trace_ref, hash_groups, span
from iw_rag.settings import AgentSettings, get_settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

#: All route labels that the router schema may emit.
_VALID_ROUTES = frozenset({"skip", "retrieve", "graph", "long_context", "visual"})

#: Safe default route applied whenever the router output is missing, an unknown
#: label, or unparseable.  ``retrieve`` is the vector/hybrid retrieval path —
#: the cheapest grounded route, and the one the classifier is itself biased
#: toward (see ROUTER_SYSTEM).  Falling back here keeps a corrupt or truncated
#: classification from dispatching a cheap lookup onto a heavy mode (graph /
#: long_context / visual).
_FALLBACK_ROUTE = "retrieve"

#: Qwen3-class models emit ``<think>…</think>`` reasoning that can leak around
#: the structured-output JSON; strip it before parsing.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

#: Markdown code fences the model sometimes wraps the JSON in.
_FENCE_RE = re.compile(r"```(?:json)?", re.IGNORECASE)

#: Phase-4 routes whose collapse the router NODE applies unconditionally.
#: Empty since WI-028/029/031/032: graph, visual, and long_context are now all
#: implemented routes whose dispatch-vs-collapse decision is made in
#: ``route_after_router`` (the edge), which can read ``modes.*.enabled`` from
#: ``get_settings()`` — the router node only has ``AgentSettings`` and so cannot
#: gate on the mode flags.  The node therefore PRESERVES the classified route
#: (recording it verbatim in the verdict for router-accuracy metrics) and the
#: edge collapses it to ``retrieve`` when the mode is disabled / the backend is
#: unavailable.  A pre-collapse here would prevent the edge from ever seeing the
#: mode route, breaking dispatch.
_PHASE4_ROUTES: frozenset[str] = frozenset()


def _extract_json_object(text: str) -> str | None:
    """Return the first balanced top-level ``{…}`` object substring, or ``None``.

    Brace matching is string-literal aware (braces inside JSON strings do not
    affect depth), so it tolerates leading/trailing prose around a complete
    object.  Returns ``None`` when no *complete* object is present — in
    particular when the JSON was truncated mid-emit (e.g. an over-long
    ``reason`` field exhausted ``max_tokens`` before the closing brace).  That
    deliberately routes the truncated case to the safe vector fallback rather
    than honouring a partially-emitted ``route``.

    Args:
        text: Text that may contain a JSON object somewhere within it.

    Returns:
        The matched ``{…}`` substring, or ``None`` if no balanced object exists.
    """
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None  # unbalanced → truncated / incomplete


def _router_json_candidates(text: str) -> Iterator[str]:
    """Yield progressively-cleaned candidate JSON strings to attempt parsing.

    Order matters: the cheapest (already-clean) candidate is yielded first so
    the common case stays a single ``json.loads``.  Later candidates strip
    ``<think>`` reasoning leakage and markdown fences, then fall back to
    extracting a balanced object from within surrounding prose.
    """
    raw = text.strip()
    if raw:
        yield raw
    cleaned = _FENCE_RE.sub("", _THINK_RE.sub("", text)).strip()
    if cleaned and cleaned != raw:
        yield cleaned
    obj = _extract_json_object(cleaned or text)
    if obj is not None:
        yield obj


def _parse_router_response(text: str) -> dict[str, Any]:
    """Tolerantly parse the router structured-output response.

    Recovers a *complete-but-dirty* JSON object — ``<think>`` reasoning
    leakage, markdown code fences, or surrounding prose — and returns the
    decoded dict.  Does **not** scrape fields out of an *incomplete* object: a
    truncated response (no closing brace) yields an empty dict so the caller
    applies the safe vector fallback (``_FALLBACK_ROUTE``) instead of trusting a
    partially-emitted ``route``.

    Args:
        text: The raw text from ``LlmClient.complete_structured``.

    Returns:
        A dict (possibly empty) with ``query_type``, ``route``, and
        optionally ``reason`` from the model output.  Empty on any failure.
    """
    for candidate in _router_json_candidates(text):
        try:
            result = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(result, dict):
            return result
    logger.warning(
        "Router: could not parse LLM response as JSON; using vector fallback. text=%r", text
    )
    return {}


# ---------------------------------------------------------------------------
# make_router
# ---------------------------------------------------------------------------


def make_router(
    llm: LlmClient,
    settings: AgentSettings,
) -> Callable[[GraphState], Coroutine[Any, Any, dict[str, Any]]]:
    """Create the Adaptive-RAG router node bound to the given LLM client + settings.

    The router is a short constrained structured-output classifier that:
    1. Checks whether routing is enabled (``settings.router_enabled``).
    2. Calls ``llm.complete_structured`` with ``ROUTER_SCHEMA`` and a tight
       param set (low temp, small tokens).
    3. Parses the JSON, validates the label, applies config gating
       (``skip_enabled``), and collapses Phase-4 routes to ``retrieve``.
    4. Records the *intended* route in ``verdicts`` even when collapsed, so
       router-accuracy metrics can score the classifier independently of the
       Phase-2 collapse.
    5. Appends exactly one ``Verdict(kind="router", ...)`` per invocation.

    Args:
        llm: The ``LlmClient`` used for the constrained structured-output call.
            In Phase 2 this is the same Qwen3-14B vLLM instance as the generator
            (CONVENTIONS §4: ``grader_model="generator"``).
        settings: ``AgentSettings`` instance controlling ``router_enabled`` and
            ``skip_enabled``.

    Returns:
        An async node function with signature
        ``(state: GraphState) -> dict[str, Any]``.
    """
    # Resolve the model name from settings.  The grader uses the same endpoint
    # as the generator (Phase-2 assumption; Phase-4 may swap to a smaller model).
    model_name: str = settings.grader_model

    async def router(state: GraphState) -> dict[str, Any]:
        """Adaptive-RAG complexity classification node.

        Classifies the current question into a query type + route, applies
        config gating, and appends a router verdict to the state.  Emits
        an ``iw_rag.agent.route`` span (no-op when tracing is disabled).

        Args:
            state: Current graph state; reads ``state["question"]``.

        Returns:
            Partial state dict with ``route``, ``query_type``, and
            ``verdicts`` (exactly one ``Verdict`` appended).
        """
        ctx = state.get("access_context")
        # NEVER include raw principal groups — hash them for span attributes.
        groups_hash = hash_groups(ctx.principal_groups) if ctx is not None else ""
        attempt: int = state.get("retries", 0)

        with span(
            "iw_rag.agent.route",
            {"node": "router", "attempt": attempt, "groups_hash": groups_hash},
        ):
            # ------------------------------------------------------------------
            # Fast-path: routing disabled → always retrieve
            # ------------------------------------------------------------------
            if not settings.router_enabled:
                verdict = Verdict(
                    kind="router",
                    value="retrieve",
                    reason="router disabled",
                )
                return {
                    "route": "retrieve",
                    "query_type": "factual",
                    "verdicts": [verdict],
                }

            # ------------------------------------------------------------------
            # Call the structured-output classifier
            # ------------------------------------------------------------------
            question: str = state.get("question", "")
            messages = [
                {"role": "system", "content": ROUTER_SYSTEM},
                {"role": "user", "content": question},
            ]
            params = GenerationParams(
                model=model_name,
                temperature=ROUTER_TEMPERATURE,
                max_tokens=ROUTER_MAX_TOKENS,
            )

            try:
                resp = await llm.complete_structured(messages, ROUTER_SCHEMA, params)
                parsed = _parse_router_response(resp.text)
            except Exception:
                logger.exception("Router: LLM call failed; defaulting to retrieve")
                parsed = {}

            # ------------------------------------------------------------------
            # Extract + validate the route label
            # ------------------------------------------------------------------
            raw_route: str = parsed.get("route", _FALLBACK_ROUTE)
            query_type: str | None = parsed.get("query_type")
            llm_reason: str = parsed.get("reason") or ""

            # Unknown/unsupported label → safe fallback
            if raw_route not in _VALID_ROUTES:
                logger.warning(
                    "Router: unknown label %r → defaulting to %s", raw_route, _FALLBACK_ROUTE
                )
                raw_route = _FALLBACK_ROUTE

            # Phase-4 routes collapse to retrieve; record the intended route
            intended_route = raw_route
            if raw_route in _PHASE4_ROUTES:
                # Record the intended route in the reason for audit/metrics
                reason = f"intended:{intended_route}" + (f"; {llm_reason}" if llm_reason else "")
                actual_route = "retrieve"
            else:
                reason = llm_reason
                actual_route = raw_route

            # ------------------------------------------------------------------
            # Apply skip gating
            # ------------------------------------------------------------------
            if actual_route == "skip" and not settings.skip_enabled:
                actual_route = "retrieve"
                reason = "skip disabled" + (f"; {reason}" if reason else "")

            # ------------------------------------------------------------------
            # Append exactly one verdict
            # ------------------------------------------------------------------
            verdict = Verdict(
                kind="router",
                value=actual_route,
                reason=reason,
            )

            logger.debug(
                "Router: question=%r → intended=%r actual=%r query_type=%r",
                question[:80],
                intended_route,
                actual_route,
                query_type,
            )

            return {
                "route": actual_route,
                "query_type": query_type,
                "verdicts": [verdict],
            }

    return router


# ---------------------------------------------------------------------------
# make_answer_direct
# ---------------------------------------------------------------------------


def make_answer_direct(
    llm: LlmClient,
    settings: AgentSettings,
) -> Callable[[GraphState], Coroutine[Any, Any, dict[str, Any]]]:
    """Create the ``answer_direct`` node for the ``skip`` branch.

    The ``skip`` route answers from parametric memory (no retrieval corpus
    is consulted).  The draft still flows to ``grade_generation`` (Task 2.4)
    for quality grading; when ``graded`` is empty ISSUP is bypassed — there
    is nothing to ground against.

    Args:
        llm: The ``LlmClient`` for the direct-answer call (same generator
            endpoint as the router in Phase 2).
        settings: ``AgentSettings`` (reserved for future per-branch params).

    Returns:
        An async node function with signature
        ``(state: GraphState) -> dict[str, Any]``.
    """
    model_name: str = settings.grader_model

    async def answer_direct(state: GraphState) -> dict[str, Any]:
        """Generate a draft from parametric memory — no retrieval.

        Calls the generator with a brief direct-answer system prompt.
        Citations are empty (no evidence to cite).  The draft will be sent to
        ``grade_generation`` (Task 2.4) once that node exists; until then the
        node sets ``answer`` and ``disposition`` so the graph can terminate.

        Args:
            state: Current graph state; reads ``state["original_question"]``.

        Returns:
            Partial state dict with ``answer``, ``citations``, ``disposition``,
            and one ``Verdict(kind="answer_direct", ...)``.
        """
        original_question: str = state.get("original_question", state.get("question", ""))

        messages = [
            {"role": "system", "content": ANSWER_DIRECT_SYSTEM},
            {"role": "user", "content": original_question},
        ]
        # Use generation-quality params (not grader-tight params) for the answer
        params = GenerationParams(
            model=model_name,
            temperature=0.1,
            max_tokens=512,
        )

        try:
            resp = await llm.complete(messages, params)
            # Strip the reasoning scratchpad (Qwen3 <think>…</think>, incl. an
            # unterminated one) so chain-of-thought never leaks to the user — the
            # assembler path does this too; the skip path must not be an exception.
            draft_text = strip_datamarks(strip_reasoning(resp.text or ""))
        except Exception:
            logger.exception("answer_direct: LLM call failed; returning empty draft")
            draft_text = ""

        verdict = Verdict(
            kind="answer_direct",
            value="skip",
            reason="parametric answer, no retrieval",
        )

        # Task 2.4: answer_direct now flows to grade_generation (not directly to END).
        # Set draft + graded=[] so grade_generation receives a skip-path state.
        # ISSUP is bypassed when graded is empty (no evidence to ground against,
        # so supported=False deterministically → unsupported unless overridden).
        # The done finaliser sets answer + disposition after grading.
        return {
            "draft": draft_text,
            "citations": [],
            "used_chunk_ids": [],
            "graded": [],
            "verdicts": [verdict],
        }

    return answer_direct


# ---------------------------------------------------------------------------
# _aggregate_doc_verdict — pure CRAG 3-action aggregation (unit-testable)
# ---------------------------------------------------------------------------


def _aggregate_doc_verdict(grades: list[bool]) -> str:
    """Aggregate per-chunk boolean grades into a CRAG 3-action verdict.

    This is the authoritative aggregation rule (lld-agentic-loop §3.2).

    CRAG thresholds / aggregation table:
    - ``correct``   — ALL chunks are relevant (full agreement → high confidence).
    - ``ambiguous`` — SOME (≥1) chunks relevant, but not all (mixed confidence).
    - ``incorrect`` — NONE relevant, OR empty candidate list.

    Routing semantics (CONVENTIONS §4 / lld-agentic-loop §3.2 [ASSUMPTION]):
    - ``correct`` and ``ambiguous`` → ``generate`` (ambiguous bounds latency;
      uncertainty is still recorded in ``verdicts`` for audit + experiment #4).
    - ``incorrect`` → ``consume_retry → rewrite_query → retrieve`` (if budget)
      or ``generate`` (fail-closed seam at budget exhaustion, Task 2.4/2.5).

    The ``ambiguous → generate`` choice is a latency call; to flip it to
    ``ambiguous → rewrite``, expose a config flag
    ``agent.grade.ambiguous_strategy = generate|rewrite`` (OPEN-QUESTIONS).

    Args:
        grades: A list of per-chunk booleans where ``True`` = relevant.
            Empty list → ``"incorrect"`` (nothing to grade).

    Returns:
        One of ``"correct"``, ``"ambiguous"``, or ``"incorrect"``.
    """
    if not grades:
        return "incorrect"
    n_relevant = sum(1 for g in grades if g)
    n_total = len(grades)
    if n_relevant == n_total:
        # All chunks relevant — full agreement, highest confidence.
        return "correct"
    if n_relevant > 0:
        # Some relevant but not all — mixed signal.
        return "ambiguous"
    # No relevant chunks at all.
    return "incorrect"


# ---------------------------------------------------------------------------
# make_grade_documents — CRAG/ISREL batched grader (Task 2.3)
# ---------------------------------------------------------------------------


def make_grade_documents(
    llm: LlmClient,
    settings: AgentSettings,
    *,
    parent_expansion_max_chars: int = 4000,
    max_model_len: int = 16384,
    context_headroom_tokens: int = 2048,
) -> Callable[[GraphState], Coroutine[Any, Any, dict[str, Any]]]:
    """Create the CRAG/ISREL grade_documents node bound to the grader LLM.

    Grades all retrieved chunks in a single batched structured-output request
    so vLLM continuous-batching can amortise the grader calls (R-00105 F8).
    All candidates are serialised into one user message; the model returns a
    parallel ``grades`` array under ``DOC_GRADE_SCHEMA``.

    The relevant subset is kept in ``state["graded"]``; the aggregate
    ``doc_verdict`` (correct/ambiguous/incorrect) is computed by the pure
    ``_aggregate_doc_verdict`` helper and stored in ``state["doc_verdict"]``.

    **Context-budget fit** (I-CTXOVERFLOW, supersedes the I-00131 char clamp):
    Each candidate's ``content`` is first char-clamped to
    ``parent_expansion_max_chars`` (a cheap per-candidate ceiling), then the
    candidate *list* is token-budgeted via
    :func:`~iw_rag.generation.context_budget.fit_items_to_budget`: the
    highest-ranked candidates that fit ``max_model_len`` minus the grader's
    output ceiling and ``context_headroom_tokens`` are kept, the lowest-ranked
    overflow tail is dropped, and only a lone oversized candidate is truncated.
    A char clamp ALONE cannot bound the prompt: N candidates each within the
    char budget still summed past the window, and guards-on datamarking inflates
    each candidate's *token* count ~12x (markers tokenize at ~1.5 chars/token vs
    ~6.8 for prose), so eight clamped candidates produced a 34k-token grader
    prompt that 400'd the generator.  The downstream ``graded`` subset is drawn
    from the fitted candidates, so generation never sees the dropped tail either.

    Node contract:
    - Exactly one ``Verdict(kind="grade_documents", ...)`` appended per call.
    - ``state["retries"]`` is NEVER modified here (single authority =
      ``consume_retry``).
    - ``state["graded"]`` contains only relevant chunks (relevant + ambiguous
      means the *relevant subset* is kept; irrelevant chunks are dropped).
    - The fail-open-to-relevant behaviour on grader exceptions is preserved
      (defense in depth never alters the relevance-decision policy).

    Args:
        llm: The ``LlmClient`` for the constrained structured-output grader call.
            In Phase 2 this is the same Qwen3-14B endpoint as the router
            (CONVENTIONS §4: ``grader_model="generator"``).
        settings: ``AgentSettings`` instance (used for ``grader_model``).
        parent_expansion_max_chars: Per-candidate char budget.  The grader
            prompt truncates each candidate's ``content`` to at most this
            many chars (deterministic first-N slice on a character
            boundary) before joining.  Mirrors
            :attr:`RetrievalSettings.parent_expansion_max_chars` — pass the
            same value the spine was built with.  Default 4000.

    Returns:
        An async node function with signature
        ``(state: GraphState) -> dict[str, Any]``.
    """
    model_name: str = settings.grader_model
    budget = max(0, int(parent_expansion_max_chars))
    # Token budget for the grader prompt: fit the candidates into the window
    # minus the grader's output ceiling and the fixed-overhead headroom, so the
    # batched grade call can never overflow max_model_len (I-CTXOVERFLOW).  A
    # per-candidate CHAR clamp cannot bound this: N datamarked candidates (~12x
    # token inflation under guards-on) summed past the window even when each was
    # individually clamped.
    token_budget = ContextBudget(
        max_model_len=max_model_len,
        reserved_output_tokens=DOC_GRADE_MAX_TOKENS,
        headroom_tokens=context_headroom_tokens,
    )

    async def grade_documents(state: GraphState) -> dict[str, Any]:
        """Grade retrieved chunks for relevance (CRAG/ISREL, batched).

        Submits all candidates in a single structured-output call, parses the
        ``grades`` array, filters to the relevant subset, aggregates to a
        3-action ``doc_verdict``, and appends one ``Verdict``.  Emits an
        ``iw_rag.agent.grade_documents`` span (no-op when tracing disabled).

        Args:
            state: Current graph state; reads ``state["chunks"]``,
                ``state["question"]``, and ``state["retries"]``.

        Returns:
            Partial state dict with ``graded`` (relevant chunks only),
            ``doc_verdict`` (CRAG 3-action), and one ``Verdict`` appended to
            ``verdicts``.
        """
        chunks = state.get("chunks") or []
        question: str = state.get("question", "")
        attempt: int = state.get("retries", 0)

        with span(
            "iw_rag.agent.grade_documents",
            {"attempt": attempt, "chunk_count": len(chunks)},
        ):
            # Fast path: nothing to grade.
            if not chunks:
                verdict = Verdict(
                    kind="grade_documents",
                    value="incorrect",
                    reason="no chunks to grade",
                    attempt=attempt,
                )
                return {
                    "graded": [],
                    "doc_verdict": "incorrect",
                    "verdicts": [verdict],
                }

            # Fit the candidates to the grader's token budget (I-CTXOVERFLOW).
            # Each candidate's content is first char-clamped (per-candidate
            # ceiling, I-00131), then the LIST is token-budgeted: the
            # highest-ranked candidates that fit the window are kept, the
            # lowest-ranked overflow tail is dropped, and only a lone oversized
            # candidate is truncated.  Grading (and the downstream graded subset)
            # therefore operates on exactly the candidates that fit — never a
            # 34k-token prompt that 400s the generator.
            # Strip datamark spotlighting before grading (perf, I-CTXOVERFLOW):
            # datamarking defends the GENERATOR against injected instructions in
            # retrieved content; the relevance grader is internal and emits only a
            # constrained boolean, so it gains nothing from the markers but pays
            # their ~12x token cost (a ~13k-token grader prompt → ~2k).  PII
            # redaction / poisoned-chunk drops from sanitize are preserved — only
            # the U+2063 markers are removed.
            fitted = fit_items_to_budget(
                chunks,
                lambda c: _clamp_to_budget(strip_datamarks(c.content), budget),
                available_tokens=token_budget.available_for_sources(
                    extra_fixed_tokens=estimate_tokens(question)
                ),
                separator_tokens=8,
            )
            graded_inputs = [c for c, _ in fitted]
            if len(graded_inputs) < len(chunks):
                logger.info(
                    "grade_documents: token budget fit %d/%d candidates "
                    "(dropped %d lowest-ranked to stay under max_model_len=%d)",
                    len(graded_inputs),
                    len(chunks),
                    len(chunks) - len(graded_inputs),
                    max_model_len,
                )
            numbered_docs = "\n\n".join(
                f"[{i + 1}] {content}" for i, (_, content) in enumerate(fitted)
            )
            user_message = (
                f"Question: {question}\n\nDocuments to grade (one per number):\n{numbered_docs}"
            )

            messages: list[dict[str, str]] = [
                {"role": "system", "content": DOC_GRADE_SYSTEM},
                {"role": "user", "content": user_message},
            ]
            params = GenerationParams(
                model=model_name,
                temperature=DOC_GRADE_TEMPERATURE,
                max_tokens=DOC_GRADE_MAX_TOKENS,
            )

            # Single batched structured call — one request for the fitted set.
            try:
                resp = await llm.complete_structured(messages, DOC_GRADE_SCHEMA, params)
                parsed = _parse_grade_response(resp.text, expected_count=len(graded_inputs))
            except Exception:
                logger.exception(
                    "grade_documents: grader call failed; treating all chunks as relevant"
                )
                # On failure, treat all as relevant to avoid silent fail-closed on a
                # transient error.  The grader failure is recorded in the verdict.
                parsed = [True] * len(graded_inputs)

            # Keep only relevant chunks in graded (from the fitted subset).
            relevant_chunks = [c for c, g in zip(graded_inputs, parsed, strict=False) if g]
            doc_verdict = _aggregate_doc_verdict(parsed)

            logger.debug(
                "grade_documents: %d/%d relevant → doc_verdict=%r (attempt=%d)",
                len(relevant_chunks),
                len(graded_inputs),
                doc_verdict,
                attempt,
            )

            verdict = Verdict(
                kind="grade_documents",
                value=doc_verdict,
                reason=(f"relevant={len(relevant_chunks)}/{len(graded_inputs)}"),
                attempt=attempt,
            )

            return {
                "graded": relevant_chunks,
                "doc_verdict": doc_verdict,
                "verdicts": [verdict],
            }

    return grade_documents


def _clamp_to_budget(content: str, budget_chars: int) -> str:
    """Truncate ``content`` to at most ``budget_chars`` chars (defensive clamp).

    Used by :func:`make_grade_documents` to guarantee the per-candidate
    contribution to the grader prompt never exceeds the context budget,
    even if a candidate slips through oversize.  The slice is a
    deterministic first-N-chars prefix (no word-boundary search) so the
    grader always sees the same prefix; the grader only needs enough
    text to judge relevance, not the full chunk.

    Args:
        content: The candidate content to clamp.
        budget_chars: Maximum allowed length in chars.  Values ``<= 0``
            return an empty string (the clamp is then useless; the spine
            should have filtered the candidate out).

    Returns:
        ``content`` unchanged when it is within budget, otherwise the
        first ``budget_chars`` chars of ``content``.
    """
    if budget_chars <= 0:
        return ""
    if len(content) <= budget_chars:
        return content
    return content[:budget_chars]


def _parse_grade_response(text: str, *, expected_count: int) -> list[bool]:
    """Parse the batched ISREL grader response into a list of booleans.

    Expects the grader to return a JSON object with a ``grades`` array.  If
    parsing fails or the array length does not match ``expected_count``, the
    function returns a safe fallback (all True — treat as relevant) to avoid
    silently discarding chunks due to a model formatting quirk.

    Args:
        text: The raw text from ``LlmClient.complete_structured``.
        expected_count: The number of chunks submitted for grading.

    Returns:
        A list of booleans of length ``expected_count`` where ``True`` =
        relevant.  Falls back to all-True on parse errors.
    """
    try:
        data = json.loads(text)
        if not isinstance(data, dict):
            logger.warning(
                "grade_documents: grader returned non-dict JSON; defaulting all-relevant"
            )
            return [True] * expected_count
        grades_raw = data.get("grades", [])
        if not isinstance(grades_raw, list):
            logger.warning("grade_documents: 'grades' field is not a list; defaulting all-relevant")
            return [True] * expected_count
        result: list[bool] = []
        for item in grades_raw:
            if isinstance(item, dict):
                result.append(bool(item.get("relevant", True)))
            else:
                result.append(True)
        # Pad or truncate to expected_count if the model misbehaved.
        if len(result) < expected_count:
            result.extend([True] * (expected_count - len(result)))
        return result[:expected_count]
    except (json.JSONDecodeError, ValueError):
        logger.warning(
            "grade_documents: could not parse grader response as JSON; defaulting all-relevant. "
            "text=%r",
            text,
        )
        return [True] * expected_count


# ---------------------------------------------------------------------------
# make_rewrite_query — single-hop query reformulation (Task 2.3)
# ---------------------------------------------------------------------------


def make_rewrite_query(
    llm: LlmClient,
    settings: AgentSettings,
) -> Callable[[GraphState], Coroutine[Any, Any, dict[str, Any]]]:
    """Create the rewrite_query node bound to the given LLM client.

    Reformulates the query from ``original_question`` (never from the current,
    already-rewritten ``question``) so successive rewrites do not drift off the
    user's original intent (R-00105 F4).  The failure signal (doc grade or gen
    grade) is included to guide the reformulation.

    The budget is NOT consumed here — ``consume_retry`` (a separate node on the
    rewrite edge) is the single authority that increments ``state["retries"]``
    (lld-agentic-loop §3.6, task-2.3 §6).

    Args:
        llm: The ``LlmClient`` for the rewrite completion call.  Uses the same
            generator endpoint as other graders in Phase 2.
        settings: ``AgentSettings`` (used for ``grader_model``).

    Returns:
        An async node function with signature
        ``(state: GraphState) -> dict[str, Any]``.
    """
    model_name: str = settings.grader_model

    async def rewrite_query(state: GraphState) -> dict[str, Any]:
        """Reformulate the query from original_question + failure signal.

        Reads ``state["original_question"]`` (immutable anchor) and
        ``state["doc_verdict"]`` (or ``state["gen_verdict"]``) to build the
        failure signal.  Writes the rewritten text to ``state["question"]``;
        ``original_question`` is never modified.

        Args:
            state: Current graph state; reads ``original_question``,
                ``doc_verdict``, and ``gen_verdict``.

        Returns:
            Partial state dict with ``question`` (rewritten) and one
            ``Verdict(kind="rewrite_query", ...)`` appended.
        """
        original_question: str = state.get("original_question", state.get("question", ""))
        doc_verdict = state.get("doc_verdict")
        gen_verdict = state.get("gen_verdict")
        attempt: int = state.get("retries", 0)

        with span("iw_rag.agent.rewrite_query", {"attempt": attempt}):
            # Build a concise failure signal to guide the rewrite.
            if doc_verdict == "incorrect":
                failure_signal = "The retrieved documents were not relevant to the question."
            elif gen_verdict is not None:
                failure_signal = f"The previous answer was graded as '{gen_verdict}'."
            else:
                failure_signal = "The previous retrieval attempt was unsuccessful."

            messages: list[dict[str, str]] = [
                {"role": "system", "content": REWRITE_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"Original question: {original_question}\n"
                        f"Failure reason: {failure_signal}\n"
                        "Produce a better search query:"
                    ),
                },
            ]
            params = GenerationParams(
                model=model_name,
                temperature=REWRITE_TEMPERATURE,
                max_tokens=REWRITE_MAX_TOKENS,
            )

            try:
                resp = await llm.complete(messages, params)
                rewritten = resp.text.strip() or original_question
            except Exception:
                logger.exception("rewrite_query: LLM call failed; keeping original question")
                rewritten = original_question

            logger.debug(
                "rewrite_query: %r → %r (signal=%r)",
                original_question[:80],
                rewritten[:80],
                failure_signal,
            )

            verdict = Verdict(
                kind="rewrite_query",
                value="rewritten",
                reason=f"from={original_question[:60]!r} to={rewritten[:60]!r}",
            )

            # Only write state["question"] — original_question must remain immutable.
            return {
                "question": rewritten,
                "verdicts": [verdict],
            }

    return rewrite_query


# ---------------------------------------------------------------------------
# consume_retry — the single authority on state["retries"] (Task 2.3)
# ---------------------------------------------------------------------------


async def consume_retry(state: GraphState) -> dict[str, Any]:
    """Increment the retry counter by exactly 1 — the ONLY place retries changes.

    This node is inserted between the corrective edge and ``rewrite_query`` so
    the budget increment is explicit, isolated, and unit-testable.  No other
    node body may write to ``state["retries"]`` (single authority invariant,
    lld-agentic-loop §3.6).

    The corrective cycle is::

        grade_documents --(incorrect & budget_left)--> consume_retry
            → rewrite_query → retrieve

    Args:
        state: Current graph state; reads ``state["retries"]``.

    Returns:
        Partial state dict with only ``{"retries": retries + 1}``.
    """
    current: int = state.get("retries", 0)
    return {"retries": current + 1}


# ---------------------------------------------------------------------------
# _aggregate_gen_verdict — pure Self-RAG aggregation (unit-testable, Task 2.4)
# ---------------------------------------------------------------------------


def _aggregate_gen_verdict(*, supported: bool, answers: bool) -> GenVerdict:
    """Aggregate ISSUP + ISUSE boolean results into a GenVerdict.

    This is the authoritative generation-grade aggregation rule
    (lld-agentic-loop §3.5, task-2.4-generate-and-grade-generation §4).

    Aggregation table (all 4 combinations):
    - ``(supported=True,  answers=True)``  → ``"grounded"``     (both pass → done)
    - ``(supported=False, answers=True)``  → ``"unsupported"``  (recycle to retrieval)
    - ``(supported=False, answers=False)`` → ``"unsupported"``  (recycle to retrieval)
    - ``(supported=True,  answers=False)`` → ``"not_answering"`` (rewrite if budget)

    Design note (task-2.4 §4 [ASSUMPTION]): both ``unsupported`` and
    ``not_answering`` recycle through ``consume_retry → rewrite_query → retrieve``
    sharing the one budget counter.  The distinction (``unsupported`` wants better
    *evidence*, ``not_answering`` wants a better *query*) is carried in the failure
    signal that ``rewrite_query`` reads from ``state["gen_verdict"]``.

    Args:
        supported: ISSUP result — True if the answer is supported by the evidence.
        answers: ISUSE result — True if the answer addresses the original question.

    Returns:
        One of ``"grounded"``, ``"unsupported"``, or ``"not_answering"``.
    """
    if not supported:
        # Not grounded by evidence → unsupported (regardless of answers flag).
        return "unsupported"
    if not answers:
        # Grounded but off-target → not_answering.
        return "not_answering"
    # Both pass → grounded.
    return "grounded"


# ---------------------------------------------------------------------------
# make_generate — Task 2.4 generate_answer node
# ---------------------------------------------------------------------------


def make_generate(
    assembler: Any,
) -> Callable[[GraphState], Coroutine[Any, Any, dict[str, Any]]]:
    """Create the generate_answer node bound to the answer assembler.

    The generator is citation-required (lld-generation §2.3) and is
    ``off-the-shelf Qwen3-14B`` — it does **not** self-grade.  All reflection
    is externalized to the separate ISSUP + ISUSE grader prompts in
    ``grade_generation`` (R-00105 F2).

    The node consumes ``state["graded"]`` (the relevant subset kept after
    ``grade_documents``), wraps them in a ``RetrievalResult`` to satisfy the
    assembler's typed contract, and calls ``assembler.generate``.

    The assembler is called with ``state["original_question"]`` (the immutable
    user phrasing), not the possibly rewritten ``state["question"]``, so the
    answer is aligned with the user's intent.

    This node writes only ``draft``, ``citations``, and ``used_chunk_ids`` — it
    does **not** set ``answer`` or ``disposition`` (that is the ``done`` finaliser's
    responsibility, enforcing the no-answer-without-grade invariant).

    Args:
        assembler: The answer assembler — any duck-type with
            ``async generate(query: str, result: RetrievalResult) -> GeneratedAnswer``.

    Returns:
        An async node function with signature
        ``(state: GraphState) -> dict[str, Any]``.
    """

    async def generate_answer(state: GraphState) -> dict[str, Any]:
        """Generate the answer from graded chunks; write draft + citations.

        Reads ``state["graded"]`` and ``state["original_question"]``; writes
        ``draft``, ``citations``, ``used_chunk_ids``, and ``diagnostics``.
        Does NOT set ``answer`` or ``disposition`` — those require passing
        ``grade_generation`` first (no-answer-without-grade invariant,
        lld-agentic-loop §7).

        Args:
            state: Current graph state.

        Returns:
            Partial state dict with ``draft``, ``citations``, ``used_chunk_ids``,
            ``diagnostics``, and one verdict appended.
        """
        # Use original_question so the answer aligns with the user's intent even
        # after one or more query rewrites.
        original_question: str = state.get("original_question", state.get("question", ""))

        # Prefer graded; empty graded → assembler returns refusal (no LLM call).
        graded = state.get("graded") or []
        attempt: int = state.get("retries", 0)

        with span(
            "iw_rag.generation.answer",
            {"attempt": attempt, "graded_count": len(graded)},
        ):
            if assembler is None:
                verdict = Verdict(kind="generate", value="refused", reason="no assembler wired")
                return {
                    "draft": None,
                    "citations": [],
                    "used_chunk_ids": [],
                    "diagnostics": {},
                    "verdicts": [verdict],
                }

            # Reconstruct a RetrievalResult from the graded subset so the assembler's
            # typed interface is met (lld-agentic-loop §3.4 [ASSUMPTION]).
            # acl_applied=True is safe: graded ⊆ candidates which were ACL-filtered.
            result = RetrievalResult(
                chunks=graded,
                retriever="spine",
                acl_applied=True,
            )
            # Assist mode (allow_ungrounded): when the graded evidence cannot
            # ground an answer, the assembler falls back to a marked
            # general-knowledge answer instead of a refusal. Strict mode (default)
            # is unchanged — empty/unsupported still yields a refusal here.
            allow_ungrounded: bool = bool(state.get("allow_ungrounded"))
            if allow_ungrounded:
                generated = await assembler.generate(
                    original_question, result, allow_ungrounded=True
                )
            else:
                generated = await assembler.generate(original_question, result)

            verdict = Verdict(
                kind="generate",
                value="generated",
                score=1.0 if generated.grounded else 0.0,
                reason=f"citations={len(generated.citations)}",
            )

            logger.debug(
                "generate_answer: %d graded chunks → citations=%d grounded=%r",
                len(graded),
                len(generated.citations),
                generated.grounded,
            )

            return {
                "draft": generated.answer,
                "citations": generated.citations,
                "used_chunk_ids": generated.used_chunk_ids,
                "diagnostics": dict(generated.diagnostics),
                "verdicts": [verdict],
            }

    return generate_answer


# ---------------------------------------------------------------------------
# make_grade_generation — Task 2.4 ISSUP + ISUSE batched grader
# ---------------------------------------------------------------------------


def make_grade_generation(
    llm: LlmClient,
    settings: AgentSettings,
    *,
    max_model_len: int = 16384,
    context_headroom_tokens: int = 2048,
) -> Callable[[GraphState], Coroutine[Any, Any, dict[str, Any]]]:
    """Create the grade_generation node bound to the grader LLM.

    Runs ISSUP (groundedness/support) and ISUSE (answer-quality/usefulness) as
    two **separate, batched** structured-output grader calls so vLLM continuous-
    batching can amortise them (R-00105 F5/F8).  These graders are NOT the
    generator's own tokens — all reflection is externalized (R-00105 F2).

    **No/invalid citations → ``unsupported`` deterministically** (before any LLM
    call for ISSUP): if there are no citations, the answer cannot be grounded,
    so ``supported=False`` without invoking the grader (lld-agentic-loop §7).

    Aggregation via ``_aggregate_gen_verdict``:
    - ``(supported & answers)`` → ``"grounded"``
    - ``(not supported)``       → ``"unsupported"`` (recycle to retrieval)
    - ``(supported & !answers)``→ ``"not_answering"`` (rewrite if budget)

    Exactly one ``Verdict(kind="grade_generation", ...)`` is appended per call.
    The node never writes to ``state["retries"]`` (single authority invariant).

    Args:
        llm: The ``LlmClient`` for the ISSUP + ISUSE structured-output calls.
            Phase 2: same Qwen3-14B endpoint as the generator.
            Phase 3: can be swapped to Granite Guardian via config (no code change).
        settings: ``AgentSettings`` instance (used for ``grader_model``).

    Returns:
        An async node function with signature
        ``(state: GraphState) -> dict[str, Any]``.
    """
    model_name: str = settings.grader_model
    # Token budget for the ISSUP evidence (I-CTXOVERFLOW): the support grader
    # joins every graded chunk's content into one prompt alongside the draft
    # answer.  Unbudgeted, the same datamarked whole-doc chunks that overflowed
    # grade_documents overflow ISSUP too.  Reserve the (small) ISSUP output
    # ceiling and the headroom; the draft answer is subtracted per-call.
    issup_budget = ContextBudget(
        max_model_len=max_model_len,
        reserved_output_tokens=ISSUP_MAX_TOKENS,
        headroom_tokens=context_headroom_tokens,
    )

    async def grade_generation(state: GraphState) -> dict[str, Any]:
        """Run ISSUP + ISUSE batched graders; aggregate to gen_verdict.

        Reads ``state["graded"]``, ``state["draft"]``, ``state["citations"]``,
        and ``state["original_question"]``.  Writes ``gen_verdict`` and appends
        one ``Verdict(kind="grade_generation", ...)``.

        Args:
            state: Current graph state.

        Returns:
            Partial state dict with ``gen_verdict`` and one verdict appended.
        """
        draft: str = state.get("draft") or ""
        graded = state.get("graded") or []
        citations = state.get("citations") or []
        original_question: str = state.get("original_question", state.get("question", ""))
        attempt: int = state.get("retries", 0)

        with span(
            "iw_rag.agent.grade_generation",
            {"attempt": attempt, "has_citations": bool(citations)},
        ):
            # ------------------------------------------------------------------
            # Deterministic ISSUP short-circuit: no citations → unsupported.
            # An answer with no resolvable citations cannot be grounded in evidence
            # (lld-agentic-loop §7; task-2.4-generate-and-grade-generation §3).
            # Skip the LLM call for ISSUP; still run ISUSE for completeness.
            # ------------------------------------------------------------------
            if not citations:
                supported = False
                # Run ISUSE only (one call instead of two)
                isuse_params = GenerationParams(
                    model=model_name,
                    temperature=ISUSE_TEMPERATURE,
                    max_tokens=ISUSE_MAX_TOKENS,
                )
                isuse_messages: list[dict[str, str]] = [
                    {"role": "system", "content": ISUSE_SYSTEM},
                    {
                        "role": "user",
                        "content": f"QUESTION:\n{original_question}\n\nANSWER:\n{draft}",
                    },
                ]
                try:
                    isuse_resp = await llm.complete_structured(
                        isuse_messages, ISUSE_SCHEMA, isuse_params
                    )
                    answers = _parse_bool(isuse_resp.text, "answers")
                except Exception:
                    logger.exception(
                        "grade_generation: ISUSE call failed (no-citations path); "
                        "defaulting answers=False"
                    )
                    answers = False
            else:
                # ------------------------------------------------------------------
                # Normal path: batch ISSUP + ISUSE together via asyncio.gather.
                # Both calls queue on the same vLLM instance so continuous-batching
                # folds them (R-00105 F5/F8).
                # A manual iw_rag.agent.grade_batch span wraps the gather so per-pass
                # call-count and latency are captured in Phoenix (task-2.7 §1).
                # ------------------------------------------------------------------
                # Token-budget the evidence so EVIDENCE + ANSWER never overflow
                # the window (I-CTXOVERFLOW).  The draft answer is fixed prompt
                # text the grader must also see, so it is subtracted from the
                # budget before fitting the graded chunks.
                fitted_evidence = fit_items_to_budget(
                    graded,
                    lambda c: c.content,
                    available_tokens=issup_budget.available_for_sources(
                        extra_fixed_tokens=estimate_tokens(draft)
                    ),
                    separator_tokens=2,
                )
                evidence: str = "\n\n".join(content for _, content in fitted_evidence)

                issup_params = GenerationParams(
                    model=model_name,
                    temperature=ISSUP_TEMPERATURE,
                    max_tokens=ISSUP_MAX_TOKENS,
                )
                isuse_params = GenerationParams(
                    model=model_name,
                    temperature=ISUSE_TEMPERATURE,
                    max_tokens=ISUSE_MAX_TOKENS,
                )

                issup_messages: list[dict[str, str]] = [
                    {"role": "system", "content": ISSUP_SYSTEM},
                    {
                        "role": "user",
                        "content": f"EVIDENCE:\n{evidence}\n\nANSWER:\n{draft}",
                    },
                ]
                isuse_messages_normal: list[dict[str, str]] = [
                    {"role": "system", "content": ISUSE_SYSTEM},
                    {
                        "role": "user",
                        "content": f"QUESTION:\n{original_question}\n\nANSWER:\n{draft}",
                    },
                ]

                with span("iw_rag.agent.grade_batch", {"attempt": attempt, "call_count": 2}):
                    try:
                        issup_resp, isuse_resp = await asyncio.gather(
                            llm.complete_structured(issup_messages, ISSUP_SCHEMA, issup_params),
                            llm.complete_structured(
                                isuse_messages_normal, ISUSE_SCHEMA, isuse_params
                            ),
                        )
                        supported = _parse_bool(issup_resp.text, "supported")
                        answers = _parse_bool(isuse_resp.text, "answers")
                    except Exception:
                        logger.exception(
                            "grade_generation: grader calls failed; defaulting to unsupported"
                        )
                        supported = False
                        answers = False

            verdict_value = _aggregate_gen_verdict(supported=supported, answers=answers)

            logger.debug(
                "grade_generation: supported=%r answers=%r → gen_verdict=%r (attempt=%d)",
                supported,
                answers,
                verdict_value,
                attempt,
            )

            verdict = Verdict(
                kind="grade_generation",
                value=verdict_value,
                reason=f"supported={supported} answers={answers}",
                attempt=attempt,
            )

            return {
                "gen_verdict": verdict_value,
                "verdicts": [verdict],
            }

    return grade_generation


def _parse_bool(text: str, field: str) -> bool:
    """Tolerantly parse a boolean field from a JSON grader response.

    Returns ``False`` on any parse failure (safe fail-closed default).

    Args:
        text: The raw text from ``LlmClient.complete_structured``.
        field: The JSON field name to extract (e.g. ``"supported"``).

    Returns:
        The boolean value of the field, or ``False`` on parse failure.
    """
    try:
        data = json.loads(text)
        if not isinstance(data, dict):
            logger.warning("grade_generation: grader returned non-dict JSON for field=%r", field)
            return False
        value = data.get(field)
        if isinstance(value, bool):
            return value
        # Tolerate string "true"/"false" from badly constrained models.
        if isinstance(value, str):
            return value.lower() == "true"
        logger.warning(
            "grade_generation: unexpected type for field=%r: %r; defaulting False", field, value
        )
        return False
    except (json.JSONDecodeError, ValueError):
        logger.warning(
            "grade_generation: could not parse grader JSON for field=%r; text=%r", field, text
        )
        return False


# ---------------------------------------------------------------------------
# done — terminal finaliser (Task 2.4)
# ---------------------------------------------------------------------------


async def done(state: GraphState) -> dict[str, Any]:
    """Terminal finaliser: promote verified draft to the final answer.

    This node is reachable ONLY via ``route_after_grade_generation`` when
    ``gen_verdict == "grounded"`` — enforcing the no-answer-without-grade
    invariant (lld-agentic-loop §7).

    When the budget is exhausted without a grounded answer, the edge also routes
    to ``done`` as a temporary seam (Task 2.4); Task 2.5 will repoint that
    budget-exhausted edge to the dedicated ``fail_closed`` node.  In the
    budget-exhausted case ``gen_verdict`` is ``"unsupported"`` or
    ``"not_answering"``, so the node sets ``disposition="answered"`` correctly
    only when ``gen_verdict == "grounded"``; otherwise it produces a refusal.

    Emits an ``iw_rag.agent.done`` span (no-op when tracing disabled).

    Args:
        state: Current graph state; reads ``draft`` and ``gen_verdict``.

    Returns:
        Partial state dict with ``answer`` (or ``refusal``) and ``disposition``.
    """
    gen_verdict = state.get("gen_verdict")
    draft: str = state.get("draft") or ""
    attempt: int = state.get("retries", 0)
    allow_ungrounded: bool = bool(state.get("allow_ungrounded"))
    guard_allowed: bool = state.get("output_guard_decision") == "allow"

    with span("iw_rag.agent.done", {"gen_verdict": str(gen_verdict), "attempt": attempt}):
        if gen_verdict == "grounded":
            # The invariant path: grade_generation passed → promote draft to answer.
            return {
                "answer": draft,
                "disposition": "answered",
                "diagnostics": {"trace_ref": current_trace_ref()},
            }

        if allow_ungrounded and guard_allowed and draft:
            # Assist mode: the output guard allowed an ungrounded answer (e.g. a
            # general-knowledge answer). Emit it — clearly marked as ungrounded via
            # diagnostics so the API/UI can badge it. Every other guard already ran.
            return {
                "answer": draft,
                "disposition": "answered",
                "diagnostics": {
                    "trace_ref": current_trace_ref(),
                    "answer_ungrounded": True,
                },
            }

        # Unexpected path: done should only receive grounded verdicts in Phase 2.5+.
        # This branch is retained defensively — the budget-exhausted edge now routes
        # to fail_closed directly (Task 2.5).  If somehow reached, refuse safely.
        refusal_reason = (
            "insufficient grounded evidence"
            if gen_verdict == "unsupported"
            else "answer did not address the question after exhausting retries"
        )
        logger.debug("done: unexpected non-grounded verdict=%r → defensive refuse", gen_verdict)
        return {
            "answer": None,
            "refusal": refusal_reason,
            "disposition": "refused",
        }


# ---------------------------------------------------------------------------
# fail_closed — terminal refusal node (Task 2.5)
# ---------------------------------------------------------------------------

#: The canonical refusal message emitted by the ``fail_closed`` node.
#: Exported so ``graph.py`` (GraphRecursionError handler) and tests can import
#: the identical string rather than duplicating it.
FAIL_CLOSED_MESSAGE: str = _GENERATION_REFUSAL_TEXT


async def fail_closed(state: GraphState) -> dict[str, Any]:
    """Terminal refusal node — the program's primary fail-closed enforcement.

    Reached when the shared budget counter is exhausted and the graph has been
    unable to produce a grounded answer.  This node is the enforcement seam
    (lld-agentic-loop §6.1) that Phase-3 guardrails also route into; it NEVER
    emits an ungrounded answer as if it were grounded.

    The ``disposition="refused"`` outcome is a **first-class result** (not an
    error).  The full ``verdicts`` trail up to this point is preserved; the
    node appends one ``Verdict(kind="fail_closed", ...)`` for auditability and
    for Phoenix/OTEL consumption.

    **Invariant:** This node never sets ``answer`` to a non-``None`` value and
    never sets ``disposition="answered"``.

    Args:
        state: Current graph state; reads ``retries``, ``doc_verdict``, and
            ``gen_verdict`` for the audit Verdict.

    Returns:
        Partial state dict with:
        - ``disposition="refused"``
        - ``answer=None``
        - ``refusal=FAIL_CLOSED_MESSAGE``
        - ``verdicts`` — a one-element list with the fail_closed Verdict
          (appended by the ``operator.add`` reducer, not overwriting prior verdicts).
    """
    retries: int = state.get("retries", 0)
    doc_verdict = state.get("doc_verdict")
    gen_verdict = state.get("gen_verdict")

    reason = (
        f"budget_exhausted retries={retries}; "
        f"last_doc_verdict={doc_verdict!r}; "
        f"last_gen_verdict={gen_verdict!r}"
    )

    verdict = Verdict(
        kind="fail_closed",
        value="refused",
        reason=reason,
        attempt=retries,
    )

    logger.info(
        "fail_closed: budget exhausted — refusing (retries=%d, doc_verdict=%r, gen_verdict=%r)",
        retries,
        doc_verdict,
        gen_verdict,
    )

    return {
        "disposition": "refused",
        "answer": None,
        "refusal": FAIL_CLOSED_MESSAGE,
        "verdicts": [verdict],
    }


# ---------------------------------------------------------------------------
# input_guard_node — Phase 3.1 guard entrypoint
# ---------------------------------------------------------------------------

#: The canonical refusal message emitted by the ``reject`` node when the
#: input guard blocks a turn (task-3.1-input-guard §5).
INPUT_GUARD_REFUSAL_MESSAGE: str = "request rejected by input guard"


async def input_guard_node(state: GraphState) -> dict[str, Any]:
    """Call the input guard and record the verdict in the state trail.

    Invokes ``guardrails.input_guard.scan_input`` over ``state["question"]``.
    The guard result is appended to the append-only ``verdicts`` trail as a
    dict (so it is serialisable by the checkpointer regardless of the
    ``GuardVerdict`` dataclass).  The ``input_decision`` field is set to
    ``v.decision`` so ``route_after_input_guard`` can route deterministically
    (without inspecting the verdict list).

    Fail-closed (lld-guardrails §5): if ``scan_input`` raises (the retry
    logic exhausted or something unexpected), this node returns a ``block``
    verdict with ``reason="guard_unavailable"`` — never allows silently.

    Args:
        state: Current graph state; reads ``state["question"]`` and
            ``state["access_context"]``.

    Returns:
        Partial state dict with:
        - ``input_decision``: ``"allow"`` or ``"block"``.
        - ``verdicts``: a one-element list with the guard verdict dict
          (merged by the ``operator.add`` reducer, not overwriting prior
          entries).
    """
    from iw_rag.domain.models import GuardVerdict
    from iw_rag.guardrails.input_guard import scan_input
    from iw_rag.settings import get_settings

    question: str = state.get("question", "")
    ctx = state.get("access_context")

    # Wrap in a span so the input guard shows as a step in the timing breakdown.
    with span("iw_rag.agent.input_guard", {"node": "input_guard"}):
        # Master switch: if guardrails are disabled, allow unconditionally.
        cfg = get_settings()
        if not cfg.guardrails.enabled:
            guard_verdict = GuardVerdict(
                layer="input",
                decision="allow",
                score=0.0,
                reason="guardrails_disabled",
            )
        else:
            try:
                guard_verdict = await scan_input(question, ctx)  # type: ignore[arg-type]
            except Exception:
                # Defensive belt-and-suspenders: scan_input already fails closed, but
                # if something truly unexpected escapes (e.g. import error), we catch
                # it here too and enforce the closed invariant.
                logger.exception(
                    "input_guard_node: unexpected error from scan_input — failing closed"
                )
                guard_verdict = GuardVerdict(
                    layer="input",
                    decision="block",
                    score=1.0,
                    reason="guard_unavailable",
                )

        # Map the layer-specific GuardVerdict onto the canonical loop ``Verdict``
        # so the append-only trail stays a homogeneous ``list[Verdict]`` (the rich
        # per-signal ``detail`` lives in logs / the Task 3.4 audit record, not the
        # loop trail). ``value`` carries the allow/block decision.
        verdict = Verdict(
            kind="input_guard",
            value=guard_verdict.decision,
            score=guard_verdict.score,
            reason=guard_verdict.reason,
        )

        logger.info(
            "input_guard_node: decision=%r score=%.3f reason=%r",
            guard_verdict.decision,
            guard_verdict.score,
            guard_verdict.reason,
        )

        return {
            "input_decision": guard_verdict.decision,
            "verdicts": [verdict],
        }


async def reject(state: GraphState) -> dict[str, Any]:
    """Terminal node for input-guard-blocked turns.

    Sets ``disposition="refused"`` and a canonical refusal message without
    invoking any spine or LLM call.  The ``verdicts`` trail is preserved; this
    node appends one verdict so the audit layer (Task 3.4) has a terminal
    marker.  The ``disposition`` is identical to ``fail_closed`` so downstream
    audit / SSE handling treats it uniformly.

    Task 3.4 will insert an audit node between ``reject`` and ``END`` — the
    seam is left clean (``reject → END`` for now; swap to ``reject → audit →
    END`` in Task 3.4 without touching ``reject``'s body).

    Args:
        state: Current graph state; reads ``input_decision`` for the verdict
            reason.

    Returns:
        Partial state dict with:
        - ``disposition="refused"``
        - ``answer=None``
        - ``refusal=INPUT_GUARD_REFUSAL_MESSAGE``
        - ``verdicts``: one reject sentinel verdict appended.
    """
    input_decision = state.get("input_decision", "block")

    verdict = Verdict(
        kind="reject",
        value="refused",
        reason=f"input_guard_blocked:{input_decision}",
    )

    logger.info("reject: input guard blocked the turn — setting disposition=refused")

    return {
        "disposition": "refused",
        "answer": None,
        "refusal": INPUT_GUARD_REFUSAL_MESSAGE,
        "verdicts": [verdict],
    }


# ---------------------------------------------------------------------------
# make_audit_node — Phase 3.4 terminal audit sink
# ---------------------------------------------------------------------------


def make_audit_node(
    session_factory: Any,
) -> Callable[[GraphState], Coroutine[Any, Any, dict[str, Any]]]:
    """Create the audit node bound to the given async session factory.

    The audit node is the LAST node before END on every terminal path
    (``done → audit → END``, ``fail_closed → audit → END``,
    ``reject → audit → END``).  It:

    1. Determines the final ``disposition`` from ``state["disposition"]`` and
       the terminal path that reached it (``reject`` → ``"rejected"``).
    2. Assembles an :class:`~iw_rag.guardrails.audit.AuditRecord` via
       :func:`~iw_rag.guardrails.audit.assemble_audit_record`.
    3. Opens a session, creates an
       :class:`~iw_rag.storage.repositories.AuditRepository`, calls
       :func:`~iw_rag.guardrails.audit.write_audit`, and commits.
    4. **On write failure (any exception from write_audit):** the node logs
       an ALERT-level message, clears ``state["answer"]`` (withholding the
       already-computed answer), and sets ``disposition="refused"`` — an
       unaudited answer is NEVER returned (lld-guardrails §5).

    **D11 [DECISION] No-DB pass-through:**  When ``session_factory`` is
    ``None`` (unit loop tests, no Postgres wired), the returned node is a
    **no-op pass-through** that returns an empty dict.  This keeps the
    loop-mechanics unit tests (``tests/unit/agent/conftest.py`` autouse) green
    without requiring a DB connection.  The production wiring (``build_graph``
    with a real ``pg_dsn``) always provides a session factory.

    Args:
        session_factory: An async context-manager factory that yields an
            ``AsyncSession``.  Typically the :meth:`~iw_rag.storage.db.Database.session`
            method.  Pass ``None`` to get a no-op pass-through (unit tests).

    Returns:
        An async node function with signature
        ``(state: GraphState) -> dict[str, Any]``.
    """
    if session_factory is None:
        # D11: no-op pass-through for unit loop tests (no DB wired).
        async def _noop_audit(state: GraphState) -> dict[str, Any]:
            """No-op audit node — used when no DB session factory is wired.

            Returns an empty dict (no state mutation) so the graph can
            proceed to END without error.

            Args:
                state: Current graph state (ignored).

            Returns:
                Empty dict — no state mutation.
            """
            logger.debug(
                "audit_node: no session factory wired — skipping audit (test pass-through)"
            )
            return {}

        return _noop_audit

    async def audit_node(state: GraphState) -> dict[str, Any]:
        """Assemble and persist the AuditRecord for the terminal disposition.

        Reads all provenance from state, assembles an
        :class:`~iw_rag.guardrails.audit.AuditRecord`, and inserts it via the
        :class:`~iw_rag.storage.repositories.AuditRepository`.

        The ``disposition`` mapping:
        - ``state["disposition"] == "answered"`` → ``"answered"``
        - ``state["disposition"] == "refused"`` and the last verdict kind is
          ``"reject"`` → ``"rejected"`` (input-guard-blocked turns)
        - ``state["disposition"] == "refused"`` otherwise → ``"refused"``

        On write failure: clears ``answer``, sets ``disposition="refused"``,
        logs CRITICAL-level alert, returns the modified state — the answer
        is withheld (lld-guardrails §5 fail-closed invariant).

        Args:
            state: The terminal graph state from the preceding node.

        Returns:
            Partial state dict — empty on success (no state mutation needed);
            ``{"answer": None, "disposition": "refused"}`` on write failure.
        """
        from iw_rag.guardrails.audit import assemble_audit_record, write_audit
        from iw_rag.storage.repositories import AuditRepository

        # Determine the canonical audit disposition.
        raw_disposition = state.get("disposition") or "refused"
        if raw_disposition == "answered":
            audit_disposition: Any = "answered"
        else:
            # Distinguish input-guard rejection from fail-closed refusal.
            # The ``reject`` terminal appends a Verdict(kind="reject", ...).
            verdicts = state.get("verdicts") or []
            # The trail is a homogeneous list[Verdict] (enforced since Task 3.1).
            has_reject_verdict = any(v.kind == "reject" for v in verdicts)
            audit_disposition = "rejected" if has_reject_verdict else "refused"

        record = assemble_audit_record(state, disposition=audit_disposition)

        try:
            async with session_factory() as session:
                repo = AuditRepository(session)
                await write_audit(record, repo)
                await session.commit()
            logger.info(
                "audit_node: persisted audit record request_id=%r disposition=%r",
                record.request_id,
                record.disposition,
            )
            return {}
        except Exception:
            # Audit-write failure = fail-closed critical error.
            # The already-computed answer must NOT be returned — an unauditable
            # answer is never emitted (lld-guardrails §5, task-3.4 §6).
            logger.critical(
                "audit_node: AUDIT WRITE FAILED — failing closed (answer withheld). "
                "request_id=%r disposition=%r. DB alert required.",
                record.request_id,
                record.disposition,
                exc_info=True,
            )
            return {
                "answer": None,
                "disposition": "refused",
                "refusal": "audit write failure — request refused for safety",
            }

    return audit_node


# ---------------------------------------------------------------------------
# sanitize_node — Phase 3.2 context sanitization entrypoint
# ---------------------------------------------------------------------------


async def sanitize_node(state: GraphState) -> dict[str, Any]:
    """Sanitize retrieved chunks before grade_documents (Task 3.2).

    Calls ``guardrails.sanitize.sanitize_context`` on ``state["chunks"]``,
    implementing the fixed pipeline (lld-guardrails §3.3):
    1. Per-chunk injection scan (Prompt Guard 2) — drop or flag poisoned chunks.
    2. PII redact (Presidio, CPU-only, lazy import).
    3. Datamark (session-randomized marker, string ops).

    The sanitized chunk list is written back to ``state["chunks"]`` so that
    ``grade_documents`` receives only clean, marked, redacted chunks.  Per-chunk
    ``GuardVerdict`` objects are mapped to the canonical ``Verdict`` dataclass
    (``kind="sanitize"``; ``value=<decision>``; ``score``; ``reason``) and
    appended to the loop trail.  ``dropped_chunk_ids`` are recorded in
    ``state["diagnostics"]`` for the audit layer.

    ACL is already applied upstream (Task 3.0).  This node must NOT filter by
    entitlement; ``principal_groups`` from ``state["access_context"]`` are used
    only for clearance-scoped PII policy and audit attribution.

    Guardrails disabled (``IW_RAG__GUARDRAILS__ENABLED=false``): the node
    respects the master switch and returns chunks unchanged with no verdicts —
    loop-mechanics unit tests use this to bypass the guard stack (agent/conftest.py).

    Fail-closed: if ``sanitize_context`` raises, the node logs a warning and
    returns an empty chunk list (not a 500) — a broken sanitizer must never
    allow unscanned chunks through to generation.

    Args:
        state: Current graph state; reads ``state["chunks"]`` and
            ``state["access_context"]``.

    Returns:
        Partial state dict with:
        - ``chunks``: sanitized (injection-scanned, PII-redacted, datamarked).
        - ``verdicts``: per-chunk ``Verdict`` objects (mapped from
          ``GuardVerdict`` via ``kind="sanitize"``).
        - ``diagnostics``: dict with ``"sanitize_dropped_ids"`` for audit.
    """
    from iw_rag.guardrails.sanitize import sanitize_context
    from iw_rag.settings import get_settings

    all_chunks = state.get("chunks") or []
    ctx = state.get("access_context")
    attempt: int = state.get("retries", 0)

    # Chat-memory chunks (prior conversation turns) are exempt: they are
    # principal-scoped by construction and carry the user's / assistant's OWN
    # words.  Injection-scanning them is pointless (the user turn already passed
    # input_guard when first sent; the assistant turn is our own output), and PII
    # redaction would scrub "Sergio" out of "my name is Sergio" — defeating the
    # memory.  Partition them out, sanitize only the corpus chunks, then
    # re-prepend the exempt chunks verbatim so the merged order is preserved.
    history_chunks = [c for c in all_chunks if c.metadata.get("kind") == CONVERSATION_CHUNK_KIND]
    chunks = [c for c in all_chunks if c.metadata.get("kind") != CONVERSATION_CHUNK_KIND]

    # Read on_poison from settings.
    cfg = get_settings()
    on_poison = cfg.guardrails.sanitize.on_poison

    with span(
        "iw_rag.guardrails.sanitize",
        {"node": "sanitize", "attempt": attempt, "chunk_count": len(chunks)},
    ):
        try:
            sanitized = await sanitize_context(
                chunks,
                ctx,  # type: ignore[arg-type]
                on_poison=on_poison,
            )
        except Exception:
            # Fail-closed: broken sanitizer → drop all chunks.
            logger.warning("sanitize_node: sanitize_context raised — failing closed (empty chunks)")
            # Keep the exempt chat-memory chunks (trusted, never sanitized); drop
            # every corpus chunk on a broken sanitizer (fail-closed).
            return {
                "chunks": list(history_chunks),
                "verdicts": [
                    Verdict(
                        kind="sanitize",
                        value="block",
                        score=1.0,
                        reason="sanitize_unavailable",
                        attempt=attempt,
                    )
                ],
                "diagnostics": {"sanitize_dropped_ids": [c.chunk_id for c in chunks]},
            }

        # ------------------------------------------------------------------
        # Map each GuardVerdict → canonical Verdict for the loop trail.
        # The rich per-signal ``detail`` lives in logs / the Task 3.4 audit
        # record, not the loop trail.  The trail is a homogeneous list[Verdict]
        # with kind/value/score/reason/attempt (lld-agentic-loop §2.1).
        # ------------------------------------------------------------------
        loop_verdicts: list[Verdict] = []
        for gv in sanitized.verdicts:
            loop_verdicts.append(
                Verdict(
                    kind="sanitize",
                    value=gv.decision,
                    score=gv.score,
                    reason=gv.reason,
                    attempt=attempt,
                )
            )

        existing_diag: dict[str, Any] = dict(state.get("diagnostics") or {})
        existing_diag["sanitize_dropped_ids"] = sanitized.dropped_chunk_ids

        logger.debug(
            "sanitize_node: %d/%d chunks survived (dropped=%d, on_poison=%r, attempt=%d)",
            len(sanitized.chunks),
            len(chunks),
            len(sanitized.dropped_chunk_ids),
            on_poison,
            attempt,
        )

        # Re-prepend the exempt chat-memory chunks verbatim (kept their original
        # position ahead of the corpus chunks; never sanitized).
        return {
            "chunks": [*history_chunks, *sanitized.chunks],
            "verdicts": loop_verdicts,
            "diagnostics": existing_diag,
        }


# ---------------------------------------------------------------------------
# output_guard_node — Phase 3.3 output verification entrypoint
# ---------------------------------------------------------------------------


async def output_guard_node(state: GraphState) -> dict[str, Any]:
    """Verify the generated answer before the ``done`` terminal (Task 3.3).

    Runs three checks — once on the output turn (not per chunk) to bound GPU
    cost (lld-guardrails §3.6):

    1. **Deterministic citation validation** — every ``Citation.chunk_id`` in
       ``state["citations"]`` must be in the ``state["graded"]`` chunk-ID set.
       An uncited answer (empty citations) is also invalid.  A failed check is a
       hard ``block`` → ``fail_closed`` (non-negotiable).

    2. **Groundedness — ONE authority, no double model call** (lld-agentic-loop
       §3.5):  The groundedness verdict is consumed from ``state["gen_verdict"]``
       (the ISSUP result already computed by ``grade_generation``), NOT by
       re-invoking Granite Guardian or any other model.  This is the key
       constraint: one groundedness authority, one model call per turn.

       *Config seam (CONVENTIONS §4):*
       ``agent.grader_model="generator"`` (Phase 2 default) = Qwen3-14B ISSUP.
       ``agent.grader_model="granite_guardian"`` (Phase 3 guards-on profile) =
       the dedicated ``guard_safety`` endpoint serves ISSUP — a one-line config
       change, no code change.  Swap is always in ``grade_generation``; the
       output guard node always reads from state, never calls a model.

    3. **PII redaction** — ``guardrails.sanitize._presidio_redact`` (Task 3.2,
       CPU-only, lazy Presidio import).  Residual PII is scrubbed from the final
       answer before it reaches ``done``.

    **Allow path:** ``gen_verdict == "grounded"`` AND ``citations_valid`` → write
    the PII-redacted answer back to ``state["draft"]`` and proceed to ``done``
    via ``route_after_output_guard``.

    **Block path:** ``not grounded`` OR ``not citations_valid`` → route to
    ``fail_closed`` (no answer emitted, ``disposition="refused"``).  This node
    does **not** emit its own refusal text — ``fail_closed`` owns the refusal.

    **Guardrails disabled** (``IW_RAG__GUARDRAILS__ENABLED=false``):  the node
    passes through unchanged (same pattern as ``sanitize_node``) so the
    loop-mechanics unit tests (``agent/conftest.py`` autouse fixture) still pass.

    Args:
        state: Current graph state; reads ``draft``, ``citations``, ``graded``,
            ``gen_verdict``, and ``access_context``.

    Returns:
        Partial state dict with:
        - ``output_guard_decision``: ``"allow"`` or ``"block"``.
        - ``draft``: PII-redacted answer (on allow path) or unchanged (on block).
        - ``verdicts``: one ``Verdict(kind="output_guard", ...)`` appended.
        - ``diagnostics``: merged output-guard diagnostics.
    """
    draft: str = state.get("draft") or ""
    citations = state.get("citations") or []
    graded = state.get("graded") or []
    gen_verdict = state.get("gen_verdict")
    ctx = state.get("access_context")
    attempt: int = state.get("retries", 0)
    allow_ungrounded: bool = bool(state.get("allow_ungrounded"))

    # Master switch: guardrails disabled → pass through unchanged.
    cfg = get_settings()
    if not cfg.guardrails.enabled:
        verdict = Verdict(
            kind="output_guard",
            value="allow",
            score=1.0,
            reason="guardrails_disabled",
            attempt=attempt,
        )
        return {
            "output_guard_decision": "allow",
            "verdicts": [verdict],
        }

    with span(
        "iw_rag.guardrails.output_guard",
        {
            "node": "output_guard",
            "attempt": attempt,
            "has_citations": bool(citations),
            "gen_verdict": str(gen_verdict),
        },
    ):
        # ------------------------------------------------------------------
        # Step 1: Deterministic citation validation (no model)
        # Build the retrieved chunk-ID set from the graded evidence.
        # ------------------------------------------------------------------
        retrieved_ids: set[str] = {chunk.chunk_id for chunk in graded}
        citations_valid, invalid_ids = await validate_citations_from_list(citations, retrieved_ids)

        # ------------------------------------------------------------------
        # Step 2: Groundedness — read from state["gen_verdict"] (ONE authority).
        # The ISSUP result from grade_generation is the canonical groundedness
        # signal.  No model is called here (lld-agentic-loop §3.5).
        # ------------------------------------------------------------------
        # "grounded" gen_verdict means ISSUP=True (supported by evidence).
        grounded: bool = gen_verdict == "grounded"

        # ------------------------------------------------------------------
        # Step 3: PII redaction (Presidio, CPU-only, lazy import, Task 3.2)
        # Only runs when allow path is likely (still run on block to ensure
        # PII is captured in diagnostics, but the redacted answer is used only
        # on the allow path).
        #
        # Provenance-aware carve-out (chat memory): PII the user supplied earlier
        # in THIS conversation — recalled into the answer via conversation-as-
        # source — must not be redacted, or "Sergio" would be scrubbed out of
        # "your name is Sergio".  The allow text is built ONLY from the graded
        # conversation chunks (the user's own turns, principal-scoped); the
        # corpus never contributes.  The kwarg is passed only when such memory is
        # present, so the strict (no-memory) path + every existing caller / mock
        # is byte-for-byte unchanged.
        # ------------------------------------------------------------------
        memory_text = " ".join(
            c.content for c in graded if c.metadata.get("kind") == CONVERSATION_CHUNK_KIND
        )
        try:
            if memory_text:
                redacted_draft = await _presidio_redact(draft, ctx, allow_text=memory_text)  # type: ignore[arg-type]
            else:
                redacted_draft = await _presidio_redact(draft, ctx)  # type: ignore[arg-type]
        except Exception:
            logger.warning("output_guard_node: _presidio_redact raised — using original draft")
            redacted_draft = draft

        pii_redacted = redacted_draft != draft

        # ------------------------------------------------------------------
        # Aggregate the block decision.
        #
        # Strict mode (default): block if EITHER groundedness or citation
        #   validity fails. Both are hard gates (fail-closed).
        #
        # Assist mode (allow_ungrounded): groundedness is INFORMATIONAL — an
        #   ungrounded answer (incl. a general-knowledge answer with no citations)
        #   is allowed. The ONLY remaining hard block is a *fabricated* citation
        #   (a marker pointing at a chunk that was never retrieved) — an
        #   anti-hallucination invariant that holds in every mode. ``invalid_ids``
        #   is non-empty only for fabricated citations; an uncited answer has
        #   ``citations_valid=False`` but ``invalid_ids==[]`` and is allowed here.
        # ------------------------------------------------------------------
        fabricated_citation = bool(invalid_ids)
        if allow_ungrounded:
            should_block = fabricated_citation
        else:
            should_block = not grounded or not citations_valid

        decision_value: str = "block" if should_block else "allow"
        reason_parts: list[str] = []
        if not grounded:
            # In assist mode this is recorded but does not block.
            tag = "ungrounded_allowed_assist" if allow_ungrounded else "ungrounded"
            reason_parts.append(f"{tag}:{gen_verdict!r}")
        if fabricated_citation:
            reason_parts.append(f"invalid_citations:{invalid_ids!r}")
        elif not citations_valid and not allow_ungrounded:
            reason_parts.append("uncited")
        if not reason_parts:
            reason_parts.append("grounded_and_cited")
        reason_str = ";".join(reason_parts)

        verdict = Verdict(
            kind="output_guard",
            value=decision_value,
            score=0.0 if should_block else 1.0,
            reason=reason_str,
            attempt=attempt,
        )

        # Build diagnostics for the audit layer.
        diag_update: dict[str, Any] = {
            "output_guard_grounded": grounded,
            "output_guard_gen_verdict": gen_verdict,
            "output_guard_citations_valid": citations_valid,
            "output_guard_invalid_citation_ids": invalid_ids,
            "output_guard_pii_redacted": pii_redacted,
            "output_guard_decision": decision_value,
            # Assist mode: groundedness was informational, not a gate.
            "output_guard_assist_mode": allow_ungrounded,
        }

        logger.info(
            "output_guard_node: decision=%r grounded=%r citations_valid=%r "
            "pii_redacted=%r invalid_ids=%r (attempt=%d)",
            decision_value,
            grounded,
            citations_valid,
            pii_redacted,
            invalid_ids,
            attempt,
        )

        result: dict[str, Any] = {
            "output_guard_decision": decision_value,
            "verdicts": [verdict],
            "diagnostics": diag_update,
        }

        # On allow: promote the PII-redacted draft (done finaliser will copy it to answer).
        if not should_block and pii_redacted:
            result["draft"] = redacted_draft

        return result


# ---------------------------------------------------------------------------
# make_visual_retrieve_node — Phase 4 (WI-031) visual retrieve node
# ---------------------------------------------------------------------------


def make_visual_retrieve_node(
    visual_retriever: Any,
    spine: Any,
) -> Callable[[GraphState], Coroutine[Any, Any, dict[str, Any]]]:
    """Create the ``visual_retrieve`` node bound to the VisualRetriever.

    Runs ``VisualRetriever.retrieve`` → writes ``state["chunks"]`` just like the
    spine's retrieve node.  On ``VisualEncodeUnavailable`` or any other error,
    falls back to the spine retriever (degrade, never block — lld-multimodal §5).

    The node writes ``state["chunks"]`` so the downstream ``sanitize →
    grade_documents → generate`` pipeline is unchanged.  A verdict with
    ``kind="visual_retrieve"`` is appended (``value="visual"`` on success,
    ``value="fallback"`` on spine-fallback).

    Args:
        visual_retriever: A VisualRetriever (or test fake) with
            ``async retrieve(query, access) -> RetrievalResult``.
        spine: The fallback retrieval spine (used on encoder error).

    Returns:
        An async node function with signature
        ``(state: GraphState) -> dict[str, Any]``.
    """

    async def visual_retrieve(state: GraphState) -> dict[str, Any]:
        """Run the two-stage visual late-interaction retrieve.

        Falls back to the spine on VisualEncodeUnavailable or any other
        encoder / store error.  Appends exactly one Verdict.

        Args:
            state: Current graph state; reads ``question`` and ``access_context``.

        Returns:
            Partial state dict with ``chunks`` (the retrieved pages as
            RetrievedChunks) and one ``Verdict(kind="visual_retrieve", ...)``.
        """
        question: str = state.get("question", "")
        ctx = state.get("access_context")

        with span(
            "iw_rag.agent.visual_retrieve",
            {"node": "visual_retrieve", "attempt": state.get("retries", 0)},
        ):
            try:
                result = await visual_retriever.retrieve(question, ctx)
                chunks = result.chunks
                verdict = Verdict(
                    kind="visual_retrieve",
                    value="visual",
                    reason=f"candidate_count={result.candidate_count}",
                )
                logger.debug("visual_retrieve: visual retriever returned %d chunks", len(chunks))
            except Exception:
                # VisualEncodeUnavailable, VisualRouteTimeout, or any unexpected error.
                # Degrade gracefully: fall back to the spine retriever.
                logger.warning("visual_retrieve: visual retriever failed — falling back to spine")
                if spine is not None:
                    try:
                        result = await spine.retrieve(question, ctx)
                        chunks = result.chunks
                    except Exception:
                        logger.exception("visual_retrieve: spine fallback also failed")
                        chunks = []
                else:
                    chunks = []
                verdict = Verdict(
                    kind="visual_retrieve",
                    value="fallback",
                    reason="visual_encoder_unavailable; spine_fallback",
                )

        return {
            "chunks": chunks,
            "verdicts": [verdict],
        }

    return visual_retrieve


# ---------------------------------------------------------------------------
# make_graph_retrieve_node — Phase 4 (WI-029) graph retrieve node
# ---------------------------------------------------------------------------


def make_graph_retrieve_node(
    graph_retriever: Any,
    spine: Any,
) -> Callable[[GraphState], Coroutine[Any, Any, dict[str, Any]]]:
    """Create the ``graph_retrieve`` node bound to the GraphRetriever.

    Runs ``GraphRetriever.retrieve`` → writes ``state["chunks"]`` just like the
    spine's retrieve node and the visual node.  On ``GraphRouteTimeout``,
    ``GraphStoreUnavailable``, or any other error, falls back to the spine
    retriever (degrade, never block — lld-graphrag §5).

    The node writes ``state["chunks"]`` so the downstream ``sanitize →
    grade_documents → generate`` pipeline is unchanged.  A verdict with
    ``kind="graph_retrieve"`` is appended (``value="graph"`` on success,
    ``value="fallback"`` on spine-fallback).

    D22 [DECISION — graph node fallback]:  The fallback pattern mirrors
    ``make_visual_retrieve_node`` exactly (WI-031):  any exception from the
    graph retriever → spine; any exception from the spine → empty chunks.
    The router never sees a 500 from this node.

    Args:
        graph_retriever: A GraphRetriever (or test fake) with
            ``async retrieve(query, access) -> RetrievalResult``.
        spine: The fallback retrieval spine (used on graph retriever error).

    Returns:
        An async node function with signature
        ``(state: GraphState) -> dict[str, Any]``.
    """

    async def graph_retrieve(state: GraphState) -> dict[str, Any]:
        """Run the GraphRAG graph-traversal retrieve.

        Falls back to the spine on GraphRouteTimeout, GraphStoreUnavailable,
        or any other graph retriever error.  Appends exactly one Verdict.

        Args:
            state: Current graph state; reads ``question`` and ``access_context``.

        Returns:
            Partial state dict with ``chunks`` (the retrieved chunks as
            RetrievedChunks) and one ``Verdict(kind="graph_retrieve", ...)``.
        """
        question: str = state.get("question", "")
        ctx = state.get("access_context")

        with span(
            "iw_rag.agent.graph_retrieve",
            {"node": "graph_retrieve", "attempt": state.get("retries", 0)},
        ):
            try:
                result = await graph_retriever.retrieve(question, ctx)
                chunks = result.chunks
                verdict = Verdict(
                    kind="graph_retrieve",
                    value="graph",
                    reason=f"candidate_count={result.candidate_count}",
                )
                logger.debug("graph_retrieve: graph retriever returned %d chunks", len(chunks))
            except Exception:
                # GraphRouteTimeout, GraphStoreUnavailable, or any unexpected error.
                # Degrade gracefully: fall back to the spine retriever.
                logger.warning("graph_retrieve: graph retriever failed — falling back to spine")
                if spine is not None:
                    try:
                        result = await spine.retrieve(question, ctx)
                        chunks = result.chunks
                    except Exception:
                        logger.exception("graph_retrieve: spine fallback also failed")
                        chunks = []
                else:
                    chunks = []
                verdict = Verdict(
                    kind="graph_retrieve",
                    value="fallback",
                    reason="graph_retriever_error; spine_fallback",
                )

        return {
            "chunks": chunks,
            "verdicts": [verdict],
        }

    return graph_retrieve


# ---------------------------------------------------------------------------
# make_long_context_node — Phase 4 (WI-032) long-context retrieve node
# ---------------------------------------------------------------------------


def make_long_context_node(
    long_context_route: Any,
    spine: Any,
    assembler: Any,
) -> Callable[[GraphState], Coroutine[Any, Any, dict[str, Any]]]:
    """Create the ``long_context`` node bound to the LongContextRoute.

    **Different wiring from graph/visual (spec §6, lld-long-context §4):**
    - Graph and visual nodes flow to ``sanitize → grade_documents → generate``.
    - The long-context node **bypasses ``grade_documents`` and ``sanitize``**
      (nothing was retrieved from an index; the whole doc is stuffed into the
      context, so per-chunk relevance grading is meaningless and the sanitize
      ACL assertion path must not be hit).
    - The node writes ``chunks`` AND ``graded`` directly (both set to the
      assembled document-level chunks), then flows straight to ``generate``
      (and from there to ``grade_generation`` via the existing graph edge).

    **Entry failure / ineligible path:**
    When ``LongContextRoute.evaluate_entry`` returns ``eligible=False`` OR
    ``build_context`` raises ``LongContextOverBudget``, the node falls back to
    the spine retriever so the downstream ``generate → grade_generation`` path
    receives normal retrieved chunks.  The verdict records ``value="fallback"``.

    **ACL invariant:** the emitted ``RetrievalResult`` has ``acl_applied=False``
    (the single sanctioned carve-out — CONVENTIONS.md §6).  The long-context node
    does NOT pass through ``sanitize`` (which would assert ``acl_applied=True``),
    so the loop's ACL invariant is not violated.

    **Exit / fallback from grade_generation:**
    After ``grade_generation`` runs, ``route_after_grade_generation`` emits the
    ``gen_verdict``.  The long-context node injects ``should_fallback`` as a
    verdict so the graph can inspect it — but the actual fallback routing is the
    caller's responsibility (graph edges handle it via the standard
    ``consume_retry → rewrite_query → retrieve`` corrective cycle).

    Args:
        long_context_route: A :class:`~iw_rag.modes.long_context.route.LongContextRoute`
            (or test fake) with async ``evaluate_entry`` and ``build_context``.
        spine: The fallback retrieval spine (used on entry failure or OOM).
        assembler: The answer assembler — passed here so the node can set up
            the generate node's input (``graded`` set to assembled chunks).

    Returns:
        An async node function with signature
        ``(state: GraphState) -> dict[str, Any]``.
    """
    from iw_rag.modes.long_context.route import LongContextOverBudget

    async def long_context_retrieve(state: GraphState) -> dict[str, Any]:
        """Run the long-context route: evaluate entry, assemble, or fall back.

        Evaluates the four entry predicates.  On success, assembles the whole
        doc(s) and writes ``chunks`` + ``graded`` (the assembled document-level
        chunks) directly, bypassing ``sanitize`` and ``grade_documents``.  On
        entry failure or ``LongContextOverBudget``, falls back to the spine and
        writes ``chunks`` for the normal downstream path.

        **Key difference from graph/visual nodes**: the node writes BOTH
        ``chunks`` AND ``graded`` directly so the ``generate`` node does not
        need to re-route through ``grade_documents`` (which would be a no-op at
        best and an ACL assertion failure at worst).

        Args:
            state: Current graph state; reads ``question``, ``query_type``,
                ``access_context``, and ``candidate_doc_ids`` from diagnostics.

        Returns:
            Partial state dict with ``chunks``, ``graded``, and one
            ``Verdict(kind="long_context_retrieve", ...)``.
        """
        question: str = state.get("question", "")
        original_question: str = state.get("original_question", question)
        ctx = state.get("access_context")
        query_type: str | None = state.get("query_type")

        # Candidate doc IDs may be injected via diagnostics (from the router)
        # or the state's candidate field.  Fallback: None → entry fails with
        # no_bounded_corpus and the spine is used.
        diag = state.get("diagnostics") or {}
        raw_candidate_ids = diag.get("candidate_doc_ids")
        candidate_doc_ids = raw_candidate_ids if raw_candidate_ids else None

        attempt: int = state.get("retries", 0)

        with span(
            "iw_rag.agent.long_context_retrieve",
            {"node": "long_context_retrieve", "attempt": attempt},
        ):
            try:
                decision = await long_context_route.evaluate_entry(
                    question,
                    ctx,
                    candidate_doc_ids=candidate_doc_ids,
                    query_type=query_type,
                )

                if not decision.eligible:
                    # Entry failed — fall back to spine (record the reason)
                    logger.debug(
                        "long_context_retrieve: entry ineligible reason=%r — spine fallback",
                        decision.reason,
                    )
                    raise ValueError(f"ineligible:{decision.reason}")

                result = await long_context_route.build_context(original_question, decision, ctx)
                chunks = result.chunks
                verdict = Verdict(
                    kind="long_context_retrieve",
                    value="long_context",
                    reason=(
                        f"eligible; tokens={decision.estimated_input_tokens}; "
                        f"doc_count={len(decision.candidate_doc_ids)}"
                    ),
                    attempt=attempt,
                )
                logger.debug(
                    "long_context_retrieve: assembled %d doc-level chunks, tokens=%d",
                    len(chunks),
                    decision.estimated_input_tokens,
                )

            except (LongContextOverBudget, ValueError, Exception) as exc:
                # Entry failure, budget overrun, or any other error → spine fallback.
                exc_name = type(exc).__name__
                logger.warning("long_context_retrieve: %s — falling back to spine", exc_name)
                if spine is not None:
                    try:
                        spine_result = await spine.retrieve(question, ctx)
                        chunks = spine_result.chunks
                    except Exception:
                        logger.exception("long_context_retrieve: spine fallback also failed")
                        chunks = []
                else:
                    chunks = []
                verdict = Verdict(
                    kind="long_context_retrieve",
                    value="fallback",
                    reason=f"long_context_unavailable:{exc_name}; spine_fallback",
                    attempt=attempt,
                )

        # Key difference: write BOTH chunks and graded so the generate node
        # skips grade_documents (which is not wired on the long_context path).
        return {
            "chunks": chunks,
            "graded": chunks,
            "verdicts": [verdict],
        }

    return long_context_retrieve
