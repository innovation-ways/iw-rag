"""QueryService and AgentQueryService — orchestration behind ``/query``.

Phase 0: ``QueryService`` = ``spine.retrieve(ctx) → assembler.generate`` (no
agentic loop).

Phase 2: ``AgentQueryService`` delegates to the compiled LangGraph graph via
``answer_question`` while keeping the same ``.answer(query, ctx) -> QueryOutcome``
signature.  A ``?loop=off`` escape hatch calls spine+assembler directly (same as
Phase-0 path) so the loop-on-vs-off experiment (#4) runs through the same endpoint.

Both are injectable so tests can substitute fakes without live wiring.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from iw_rag.domain.acl import AccessContext
from iw_rag.domain.models import GeneratedAnswer
from iw_rag.generation.answer import AnswerAssembler
from iw_rag.retrieval.spine import RetrievalSpine


@dataclass(frozen=True, slots=True)
class QueryOutcome:
    """The service result: the answer plus retrieval provenance.

    Attributes:
        answer: The grounded/refused answer.
        retriever: Which retriever produced the evidence (``"spine"`` in Phase 0).
        acl_applied: Whether the ACL pre-filter ran (always True on the spine).
        diagnostics: Merged retrieval + generation diagnostics (operator-only).
    """

    answer: GeneratedAnswer
    retriever: str
    acl_applied: bool
    diagnostics: dict[str, object]


class QueryService:
    """Wraps the retrieval spine + answer assembler."""

    def __init__(self, spine: RetrievalSpine, assembler: AnswerAssembler) -> None:
        """Wire the spine + assembler."""
        self._spine = spine
        self._assembler = assembler

    async def answer(self, query: str, ctx: AccessContext) -> QueryOutcome:
        """Retrieve under the ACL context, then assemble a cited answer.

        Args:
            query: The user question.
            ctx: The authenticated access context (sole ACL source).

        Returns:
            A :class:`QueryOutcome`.
        """
        result = await self._spine.retrieve(query, ctx)
        answer = await self._assembler.generate(query, result)
        diagnostics = {**result.diagnostics, **answer.diagnostics}
        return QueryOutcome(
            answer=answer,
            retriever=result.retriever,
            acl_applied=result.acl_applied,
            diagnostics=diagnostics,
        )


class AgentQueryService:
    """Phase-2 query service that routes through the LangGraph agent loop.

    Keeps the same ``async answer(query, ctx) -> QueryOutcome`` signature as
    :class:`QueryService` so the injectable seam in ``create_app`` is unchanged.
    A ``loop=False`` escape hatch bypasses the graph and calls spine+assembler
    directly (the Phase-0 path), enabling the loop-on-vs-off experiment (#4).

    Attributes:
        _spine: The retrieval spine.
        _assembler: The answer assembler.
        _graph: The compiled LangGraph graph (built once at app startup).
    """

    def __init__(
        self,
        spine: Any,
        assembler: Any,
        graph: Any,
    ) -> None:
        """Wire the spine, assembler, and compiled graph.

        Args:
            spine: The retrieval spine.
            assembler: The answer assembler.
            graph: The compiled LangGraph ``CompiledStateGraph``.
        """
        self._spine = spine
        self._assembler = assembler
        self._graph = graph

    async def answer(
        self,
        query: str,
        ctx: AccessContext,
        *,
        loop: bool = True,
        allow_ungrounded: bool = False,
        history_chunks: list[Any] | None = None,
    ) -> QueryOutcome:
        """Answer a query through the agent graph or directly via spine+assembler.

        When ``loop=True`` (default) the query is routed through the compiled
        LangGraph graph (``answer_question``).  When ``loop=False`` the graph is
        bypassed and spine+assembler are called directly — identical to the
        Phase-0 :class:`QueryService`.

        Args:
            query: The user question.
            ctx: The authenticated access context (sole ACL source).
            loop: Whether to use the agent graph (True) or the direct path
                (False, ``?loop=off`` escape hatch).

        Returns:
            A :class:`QueryOutcome`.
        """
        import uuid

        if not loop:
            # Escape hatch: direct spine → assembler (Phase-0 path, no graph).
            result = await self._spine.retrieve(query, ctx)
            generated = await self._assembler.generate(query, result)
            diagnostics: dict[str, Any] = {**result.diagnostics, **generated.diagnostics}
            return QueryOutcome(
                answer=generated,
                retriever=result.retriever,
                acl_applied=result.acl_applied,
                diagnostics=diagnostics,
            )

        # Agent loop path — run through LangGraph.
        from iw_rag.agent.graph import answer_question

        thread_id = str(uuid.uuid4())
        state = await answer_question(
            self._graph,
            query,
            ctx,
            thread_id=thread_id,
            allow_ungrounded=allow_ungrounded,
            history_chunks=history_chunks,
        )

        # Map terminal GraphState → QueryOutcome.
        # The answer node writes a GeneratedAnswer-compatible shape.
        answer_text: str = state.get("answer") or state.get("refusal") or ""
        disposition = state.get("disposition")
        answered: bool = disposition == "answered"
        citations = state.get("citations") or []
        used_chunk_ids = state.get("used_chunk_ids") or []
        diag: dict[str, Any] = dict(state.get("diagnostics") or {})
        diag["thread_id"] = thread_id
        diag["disposition"] = disposition

        # `grounded` reflects TRUE groundedness, NOT merely "we returned an answer".
        # In assist mode an answer can be returned (disposition="answered") yet be
        # ungrounded — the output guard's groundedness verdict is the authority.
        # When the guard did not record a verdict (guardrails disabled), fall back
        # to disposition so strict-mode/legacy behaviour is unchanged.
        og_grounded = diag.get("output_guard_grounded")
        grounded: bool = answered if og_grounded is None else (answered and bool(og_grounded))

        # Curated provenance trail for the UI's per-answer "how grounded was this?"
        # popup: the latest verdict per guard/grade kind (input guard, injection,
        # safety, grounding, output guard, router, …), compacted to plain dicts.
        verdicts = state.get("verdicts") or []
        guard_trail: dict[str, dict[str, Any]] = {}
        for v in verdicts:
            guard_trail[v.kind] = {
                "kind": v.kind,
                "value": v.value,
                "score": v.score,
                "reason": v.reason,
                "attempt": v.attempt,
            }
        diag["guard_trail"] = list(guard_trail.values())

        # Full append-only trajectory (every attempt, in order) for trajectory
        # evaluation (experiment #9) — distinct from `guard_trail`, which keeps
        # only the latest verdict per kind for the UI popup. The eval harness
        # scores this list; see docs/build/03-tasks/trajectory-eval/.
        # `reason` is omitted on purpose: it can carry model-output snippets the
        # scorer never needs, and keeping it out keeps the trace attribute-safe
        # (guard_trail still carries `reason` for the UI).
        diag["trajectory"] = [
            {"kind": v.kind, "value": v.value, "score": v.score, "attempt": v.attempt}
            for v in verdicts
        ]
        diag["route"] = state.get("route")
        diag["retries"] = state.get("retries", 0)
        diag["max_retries"] = state.get("max_retries", 0)

        generated_answer = GeneratedAnswer(
            answer=answer_text,
            grounded=grounded,
            citations=citations,
            used_chunk_ids=used_chunk_ids,
            diagnostics=diag,
        )
        return QueryOutcome(
            answer=generated_answer,
            retriever="agent",
            acl_applied=True,
            diagnostics=diag,
        )

    async def answer_question_stream(
        self,
        query: str,
        ctx: AccessContext,
        *,
        thread_id: str = "default",
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream Q-SSE-SCHEMA events from the agent graph.

        The presence of this method is what makes ``/query/stream`` take the
        native streaming path (per-node ``step``/``source``/``verdict`` events
        and a root-spanned ``final``/``refusal``) instead of the synthetic
        single-event fallback. Delegates verbatim to the module-level
        :func:`iw_rag.agent.graph.answer_question_stream`, which wraps the
        astream loop in the ``iw_rag.query`` root span so the final event's
        ``trace_ref`` carries the real OpenTelemetry trace id.

        Args:
            query: The user question.
            ctx: The authenticated access context (sole ACL source).
            thread_id: LangGraph thread id (also the Phoenix trace correlation id).

        Yields:
            SSE-shaped event dicts (``{"event": ..., "data": ...}``).
        """
        from iw_rag.agent.graph import answer_question_stream

        async for event in answer_question_stream(self._graph, query, ctx, thread_id=thread_id):
            yield event
