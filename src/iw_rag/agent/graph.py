"""LangGraph graph assembly and public entrypoints.

Phase 2.0 ships a single ``answer`` node that wires the retrieval spine and the
answer assembler together.  Phase 2.1 adds the router node + ``answer_direct``
node (skip branch) and wires the conditional edge
``START → router → (route_after_router) → {skip: answer_direct → END,
                                             retrieve: answer → END}``.

Phase 2.2 decomposes the monolithic ``answer`` node on the retrieve branch into
three dedicated nodes: ``retrieve → grade_documents → generate``.

Phase 2.3 replaces the ``grade_documents`` stub with the real CRAG/ISREL node
and wires the corrective cycle::

    retrieve → grade_documents → (route_after_grade_documents) → {
        generate → END,
        consume_retry → rewrite_query → retrieve,  # corrective cycle
    }

Phase 2.4 adds the generation verification half and wires the full loop::

    retrieve → grade_documents → (route_after_grade_documents) → {
        generate → grade_generation → (route_after_grade_generation) → {
            done → END,
            consume_retry → rewrite_query → retrieve,  # shared budget cycle
        },
        consume_retry → rewrite_query → retrieve,
    }
    answer_direct → grade_generation  # skip path also gets verified

Phase 2.5 adds the ``fail_closed`` terminal node and wires budget-exhausted
edges to it (instead of the ``done`` seam)::

    grade_generation → (route_after_grade_generation) → {
        done → END,                  # grounded verdict
        consume_retry → ...,         # budget left, not grounded
        fail_closed → END,           # budget exhausted, not grounded
    }

Phase 2.6 adds checkpointer injection and optional HITL (human-in-the-loop)
support.  HITL is wired via ``interrupt_before=["done"]`` at compile time when
``settings.hitl.enabled`` is ``True`` — **off by default** so the latency
benchmark (task-2.7) is not blocked.  When interrupted, an operator inspects
``draft + citations + verdicts`` via ``graph.aget_state(config)`` and resumes
with ``graph.ainvoke(None, config)`` (approve) or routes to ``fail_closed``
(reject).  See :func:`~iw_rag.agent.checkpoint.make_checkpointer` for the
saver factory (InMemorySaver dev / AsyncPostgresSaver durable).

Phase 3.1 inserts ``input_guard`` as the new graph entrypoint (before the
router), with a ``reject`` terminal for blocked turns::

    START → input_guard → (route_after_input_guard) → {
        reject → END,     # input guard blocked the turn
        router → (route_after_router) → {
            skip: answer_direct → grade_generation → ...,
            retrieve: retrieve → sanitize → grade_documents → ...,
        },
    }

Phase 3.2 inserts ``sanitize`` between ``retrieve`` and ``grade_documents``::

    retrieve → sanitize → grade_documents → ...

The corrective cycle (``rewrite_query → retrieve``) also flows through
``sanitize`` on re-retrieval so every retrieval pass is sanitized.

Phase 3.3 inserts ``output_guard`` between ``grade_generation`` and ``done``::

    grade_generation → (route_after_grade_generation) → {
        output_guard → (route_after_output_guard) → {done, fail_closed},
        consume_retry → ...,   # budget left, not grounded
        fail_closed → END,     # budget exhausted, not grounded
    }

The output_guard reads ``state["gen_verdict"]`` for groundedness — the ONE
authority (lld-agentic-loop §3.5, no double model call).

Phase 3.4 routes every terminal through ``audit`` as the last node before END::

    done      → audit → END   (disposition="answered")
    fail_closed → audit → END (disposition="refused")
    reject    → audit → END   (disposition="rejected")

Phase 4 (WI-031) adds the ``visual`` node::

    router → (route_after_router) → {
        ...
        visual: visual_retrieve → sanitize → grade_documents → ...
            (on VisualEncodeUnavailable: falls back to retrieve)
    }

The ``visual`` node is only wired when ``visual_retriever`` is injected into
:func:`build_graph`.  When ``visual_retriever`` is ``None`` (default), the
``"visual"`` route from the router collapses to ``"retrieve"`` in
``route_after_router`` (modes.visual.enabled=false by default).

Phase 4 (WI-029) adds the ``graph`` node for GraphRAG retrieval::

    router → (route_after_router) → {
        ...
        graph: graph_retrieve → sanitize → grade_documents → ...
            (on GraphRouteTimeout/GraphStoreUnavailable: falls back to spine)
    }

The ``graph`` node is only wired when ``graph_retriever`` is injected into
:func:`build_graph`.  When ``graph_retriever`` is ``None`` (default), the
``"graph"`` route from the router collapses to ``"retrieve"`` in
``route_after_router`` (modes.graphrag.enabled=false by default).

Phase 4 (WI-032) adds the ``long_context`` node::

    router → (route_after_router) → {
        ...
        long_context: long_context_retrieve → generate → grade_generation → ...
            (BYPASSES sanitize AND grade_documents — whole-doc stuffing,
            no per-chunk index retrieval; falls back to spine on any failure)
    }

The ``long_context`` node is only wired when ``long_context_route`` is injected
into :func:`build_graph`.  When ``None`` (default), the ``"long_context"`` route
from the router collapses to ``"retrieve"`` in ``route_after_router``
(modes.long_context.enabled=false by default).

**Key wiring difference from graph/visual:** the long_context node bypasses
``sanitize`` and ``grade_documents`` (nothing retrieved from index; the assembled
doc is stuffed directly; the sanitize ACL assertion must not be hit for a path
with ``acl_applied=False`` — the sanctioned carve-out, CONVENTIONS.md §6).
The node writes BOTH ``chunks`` and ``graded`` directly so ``generate`` uses
the assembled doc-level chunks without the grade_documents filter.

``audit`` assembles the :class:`~iw_rag.guardrails.audit.AuditRecord` from the
terminal state and persists it.  On write failure the audit node fails closed
(withholds the answer, sets ``disposition="refused"``).  When no session factory
is wired (unit tests, no DB) the audit node is a no-op pass-through (D11).

The Phase-2.0 fallback (``llm=None``) is also gated by the input guard::

    START → input_guard → (route_after_input_guard_degraded) → {
        reject → audit → END,
        answer → audit → END,     # Phase-2.0 monolithic node
    }

``GraphRecursionError`` (an escape past the counter) is caught in
``answer_question`` and converted to the same fail-closed result (never a 500).

Public API::

    graph = build_graph(spine, assembler, llm)
    state = await answer_question(graph, question, ctx)
    async for event in answer_question_stream(graph, question, ctx):
        ...   # SSE-shaped dicts

References:
- lld-agentic-loop §2.5 (signatures), §1 (responsibilities), §3.1 (router)
- lld-agentic-loop §5 (checkpointing/resume/HITL/time-travel)
- task-2.2-retrieve-toolnode §4 (graph wiring)
- task-2.3-grade-documents-and-rewrite §7 (corrective cycle wiring)
- task-2.4-generate-and-grade-generation §6 (grade_generation wiring)
- task-2.5-budgets-and-fail-closed §3, §4 (fail_closed wiring; recursion handling)
- task-2.6-checkpointing-and-hitl §2, §3 (HITL wiring; checkpointer injection)
- task-3.1-input-guard §5 (input_guard entrypoint + reject terminal wiring)
- task-3.2-context-sanitization §5 (sanitize node inserted between retrieve → grade_documents)
- task-3.4-audit-log §5 (all terminals route through audit before END)
- CONVENTIONS.md §1 (RetrievedChunk / no Candidate), §8 (checkpointer)
- Q-SSE-SCHEMA (canonical SSE event set)
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

from langgraph.graph import END, START, StateGraph

from iw_rag.agent.budgets import DEFAULT_MAX_RETRIES, RECURSION_LIMIT
from iw_rag.agent.nodes import FAIL_CLOSED_MESSAGE
from iw_rag.agent.state import GraphState, Verdict
from iw_rag.domain.acl import AccessContext

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Node implementations (Phase 2.0: single answer node — kept for the
# llm=None fallback path)
# ---------------------------------------------------------------------------


def _make_answer_node(spine: Any, assembler: Any) -> Any:
    """Create the single Phase-2.0 answer node bound to spine + assembler.

    This node is retained for the ``llm=None`` fallback path (Phase-2.0
    compatibility).  When ``llm`` is provided the retrieve branch uses the
    decomposed ``retrieve → grade_documents → generate`` topology instead.

    Args:
        spine: The retrieval spine (``RetrievalSpine`` or any duck-type with
            ``async retrieve(query, ctx) -> RetrievalResult``).
        assembler: The answer assembler (``AnswerAssembler`` or duck-type with
            ``async generate(query, result) -> GeneratedAnswer``).

    Returns:
        An async node function conforming to the LangGraph node signature.
    """

    async def _answer_node(state: GraphState) -> dict[str, Any]:
        """Retrieve and generate — the Phase-2.0 single-node implementation.

        Used only on the Phase-2.0 fallback path (``llm=None``).  The full
        Phase-2.2 path decomposes this into ``retrieve → grade_documents →
        generate`` nodes.

        Args:
            state: Current graph state.

        Returns:
            Partial state dict with answer, citations, disposition, etc.
        """
        question: str = state.get("question", "")
        ctx: AccessContext = state["access_context"]

        if spine is None or assembler is None:
            # Stub path: no real spine/assembler wired (test scaffolding).
            return {
                "answer": None,
                "refusal": "no spine or assembler wired",
                "disposition": "refused",
                "chunks": [],
                "graded": [],
                "citations": [],
                "used_chunk_ids": [],
                "diagnostics": {},
                "verdicts": [Verdict(kind="answer", value="refused", reason="stub")],
            }

        result = await spine.retrieve(question, ctx)
        generated = await assembler.generate(question, result)

        disposition = "answered" if generated.grounded else "refused"
        verdict = Verdict(
            kind="answer",
            value=disposition,
            score=1.0 if generated.grounded else 0.0,
            reason="phase-2.0 single-node",
            attempt=0,
        )
        merged_diag: dict[str, Any] = {
            **result.diagnostics,
            **generated.diagnostics,
        }

        return {
            "answer": generated.answer if generated.grounded else None,
            "refusal": None if generated.grounded else generated.answer,
            "disposition": disposition,
            "chunks": result.chunks,
            "graded": result.chunks,  # no grading yet — pass all through
            "citations": generated.citations,
            "used_chunk_ids": generated.used_chunk_ids,
            "diagnostics": merged_diag,
            "verdicts": [verdict],
        }

    return _answer_node


# ---------------------------------------------------------------------------
# Phase 2.2: decomposed node factories
# ---------------------------------------------------------------------------


def _make_grade_documents_stub() -> Any:
    """Create the Phase-2.2 grade_documents stub node.

    This is a pass-through stub: it copies ``state["chunks"]`` directly into
    ``state["graded"]`` without any actual relevance grading.  Real CRAG/ISREL
    grading is implemented in Task 2.3 (WI-017).

    The stub exists so the ``retrieve → grade_documents → generate`` topology
    is correct and the downstream ``generate`` node reads from ``graded`` (not
    raw ``chunks``), keeping the seam ready for Task 2.3 to replace without a
    graph-wiring change.

    Returns:
        An async node function with signature
        ``(state: GraphState) -> dict[str, Any]``.
    """

    async def grade_documents(state: GraphState) -> dict[str, Any]:
        """Pass-through stub: copies chunks → graded unchanged.

        Real CRAG/ISREL relevance grading is wired in Task 2.3.  Until then
        all chunks are treated as relevant so the downstream generate node
        receives a non-empty graded list.

        Args:
            state: Current graph state; reads ``state["chunks"]``.

        Returns:
            Partial state dict with ``graded`` set to the same list as
            ``chunks``, and one stub verdict appended.
        """
        chunks = state.get("chunks") or []
        verdict = Verdict(
            kind="grade_documents",
            value="stub",
            reason="pass-through stub; real CRAG/ISREL grading in Task 2.3",
        )
        return {
            "graded": list(chunks),
            "doc_verdict": "relevant",  # stub: all chunks treated as relevant
            "verdicts": [verdict],
        }

    return grade_documents


def _make_generate_node(assembler: Any) -> Any:
    """Create the Phase-2.2 generate node bound to the answer assembler.

    Consumes ``state["graded"]`` (or falls back to ``state["chunks"]`` if
    ``graded`` is empty/absent) by rebuilding a ``RetrievalResult`` and
    calling ``assembler.generate(question, result)``.

    Args:
        assembler: The answer assembler (``AnswerAssembler`` or duck-type with
            ``async generate(query, result) -> GeneratedAnswer``).

    Returns:
        An async node function with signature
        ``(state: GraphState) -> dict[str, Any]``.
    """
    from iw_rag.domain.models import RetrievalResult

    async def generate(state: GraphState) -> dict[str, Any]:
        """Generate the answer from graded (or raw) chunks.

        Reads ``state["graded"]`` as the primary evidence set (the subset that
        passed ``grade_documents``), falling back to ``state["chunks"]`` if
        ``graded`` is absent or empty.  Builds a ``RetrievalResult`` to satisfy
        the assembler's typed contract and calls ``assembler.generate``.

        Args:
            state: Current graph state.

        Returns:
            Partial state dict with ``answer``, ``citations``, ``disposition``,
            ``used_chunk_ids``, ``diagnostics``, and one answer verdict.
        """
        question: str = state.get("question", "")

        # Prefer graded; fall back to raw chunks if grade_documents is a stub.
        graded = state.get("graded") or state.get("chunks") or []

        if assembler is None:
            verdict = Verdict(kind="generate", value="refused", reason="no assembler wired")
            return {
                "answer": None,
                "refusal": "no assembler wired",
                "disposition": "refused",
                "citations": [],
                "used_chunk_ids": [],
                "diagnostics": {},
                "verdicts": [verdict],
            }

        # Reconstruct a RetrievalResult so the assembler's typed interface is met.
        result = RetrievalResult(
            chunks=graded,
            retriever="spine",
            acl_applied=True,  # ACL was asserted in the retrieve node
        )
        generated = await assembler.generate(question, result)

        disposition = "answered" if generated.grounded else "refused"
        verdict = Verdict(
            kind="generate",
            value=disposition,
            score=1.0 if generated.grounded else 0.0,
            reason="phase-2.2 generate node",
        )
        merged_diag: dict[str, Any] = dict(generated.diagnostics)

        return {
            "answer": generated.answer if generated.grounded else None,
            "refusal": None if generated.grounded else generated.answer,
            "disposition": disposition,
            "citations": generated.citations,
            "used_chunk_ids": generated.used_chunk_ids,
            "diagnostics": merged_diag,
            "verdicts": [verdict],
        }

    return generate


# ---------------------------------------------------------------------------
# Graph factory
# ---------------------------------------------------------------------------


def build_graph(
    spine: Any,
    assembler: Any,
    llm: Any = None,
    *,
    settings: Any = None,
    checkpointer: Any = None,
    budgets: Any = None,
    audit_repo: Any = None,
    pg_dsn: str | None = None,
    visual_retriever: Any = None,
    graph_retriever: Any = None,
    long_context_route: Any = None,
    parent_expansion_max_chars: int = 4000,
    generation_max_model_len: int = 16384,
    context_headroom_tokens: int = 2048,
) -> Any:
    """Compile the Phase-2/3/4 LangGraph graph.

    Phase 3.1 inserts ``input_guard`` as the graph entrypoint, and Phase 3.2
    inserts ``sanitize`` between ``retrieve`` and ``grade_documents``.

    Phase 3.4 routes every terminal through ``audit`` as the last node before
    END.  The audit session factory is built from ``pg_dsn`` when provided.
    When neither ``pg_dsn`` nor ``audit_repo`` is given, the audit node is a
    no-op pass-through (D11 decision — keeps unit loop tests green without DB).

    Phase 2.6 adds:
    - **Checkpointer injection**: pass an ``InMemorySaver`` (dev) or
      ``AsyncPostgresSaver`` (durable) to enable persistent state snapshots per
      super-step under a ``thread_id``.  See
      :func:`~iw_rag.agent.checkpoint.make_checkpointer`.
    - **HITL (human-in-the-loop)**: when ``settings.hitl.enabled`` is ``True``
      the graph is compiled with ``interrupt_before=["done"]`` so execution
      pauses before the final emit.  **Off by default** to keep the latency
      benchmark clean (lld-agentic-loop §5).

    When ``llm`` is ``None`` (or ``settings.router_enabled is False``), the
    graph degrades to ``START → input_guard → {reject, answer} → audit → END``
    (Phase-2.0 behaviour gated by the input guard) so existing tests and
    scripts that pass no LLM remain valid.

    Phase 4 (WI-029) adds the ``graph`` node for GraphRAG retrieval.  When
    ``graph_retriever`` is provided AND ``modes.graphrag.enabled=True``, the
    ``"graph"`` router route dispatches to the graph node, which flows into
    ``sanitize → grade_documents`` exactly like the spine.

    Args:
        spine: Retrieval spine used by the retrieve node.
        assembler: Answer assembler used by the generate node.
        llm: The ``LlmClient`` for the router + answer_direct nodes.  When
            ``None`` the router is skipped and the graph runs Phase-2.0 style
            (monolithic answer node).
        settings: Optional ``AgentSettings`` controlling router/skip/hitl
            behaviour.  Defaults to ``AgentSettings()`` (router enabled, skip
            enabled, hitl disabled).
        checkpointer: Optional LangGraph checkpointer for durable resume /
            time-travel / HITL.  ``None`` = no persistence; pass
            ``InMemorySaver()`` or an ``AsyncPostgresSaver`` built via
            :func:`~iw_rag.agent.checkpoint.make_checkpointer`.
        budgets: Optional ``AgentBudgets`` instance (reserved for 2.3).
        audit_repo: Reserved for future direct-injection testing (unused; the
            session factory is built from ``pg_dsn``).
        pg_dsn: Postgres DSN used to build the async session factory for the
            audit node.  When ``None``, the audit node is a no-op pass-through
            (D11 — unit loop tests with no DB).
        visual_retriever: Optional VisualRetriever for Phase-4 WI-031.  When
            provided AND ``modes.visual.enabled=True``, the ``"visual"`` router
            route dispatches to the visual node.  When ``None`` (default), the
            route_after_router edge collapses ``"visual"`` to ``"retrieve"``.
        graph_retriever: Optional GraphRetriever for Phase-4 WI-029.  When
            provided AND ``modes.graphrag.enabled=True``, the ``"graph"`` router
            route dispatches to the graph node.  When ``None`` (default), the
            route_after_router edge collapses ``"graph"`` to ``"retrieve"``.
        long_context_route: Optional LongContextRoute for Phase-4 WI-032.  When
            provided AND ``modes.long_context.enabled=True``, the
            ``"long_context"`` router route dispatches to the long_context node.
            The long_context node BYPASSES sanitize and grade_documents (whole-doc
            stuffing; no per-chunk index retrieval), flowing directly to generate.
            When ``None`` (default), the route_after_router edge collapses
            ``"long_context"`` to ``"retrieve"``.
        parent_expansion_max_chars: Per-candidate char budget threaded into
            ``make_grade_documents`` so the grader prompt can never exceed
            the generator's ``max_model_len`` window.  Mirrors
            :attr:`RetrievalSettings.parent_expansion_max_chars` — pass the
            same value the spine was built with.  Default 4000 (I-00131).
        generation_max_model_len: The served generator's context window in
            tokens.  Threaded into the grader nodes so they token-budget their
            prompts (I-CTXOVERFLOW) — a char clamp alone cannot bound tokens
            once guards-on datamarking inflates content ~12x.  Default 16384.
        context_headroom_tokens: Tokens reserved in every budgeted prompt for
            the system prompt + question + chat-template scaffolding + safety
            margin (mirrors ``GenerationSettings.context_headroom_tokens``).
            Default 2048.

    Returns:
        A compiled ``CompiledStateGraph`` ready for ``ainvoke`` / ``astream``.
        When HITL is enabled the graph will pause before the ``done`` node and
        must be resumed via ``graph.ainvoke(None, config)``.
    """
    from iw_rag.settings import AgentSettings

    effective_settings: AgentSettings = settings if settings is not None else AgentSettings()

    # HITL: pause before the final ``done`` node when enabled.
    # Off by default — do NOT set interrupt_before unless explicitly opted in.
    # A checkpointer is required for interrupt to work; LangGraph will raise if
    # interrupt_before is set without one.
    hitl_enabled: bool = effective_settings.hitl.enabled
    interrupt_before: list[str] | None = ["done"] if hitl_enabled else None

    # --- Task 3.4: build the audit session factory from pg_dsn ---
    # When no pg_dsn is provided (unit tests), the audit node is a no-op (D11).
    _audit_session_factory: Any = None
    if pg_dsn is not None:
        from iw_rag.settings import DatabaseSettings
        from iw_rag.storage.db import Database

        _audit_db = Database(DatabaseSettings(dsn=pg_dsn))
        _audit_session_factory = _audit_db.session

    from iw_rag.agent.nodes import make_audit_node

    audit_node_fn = make_audit_node(_audit_session_factory)

    g: Any = StateGraph(GraphState)

    # --- Phase 3.1: input_guard is the new entrypoint (before router) ---
    # Import the guard node and reject terminal regardless of llm presence.
    from iw_rag.agent.nodes import input_guard_node, reject

    g.add_node("input_guard", input_guard_node)
    g.add_node("reject", reject)
    # Task 3.4: audit is the last node before END on every terminal.
    g.add_node("audit", audit_node_fn)
    g.add_edge(START, "input_guard")
    # reject → audit → END (not reject → END directly)
    g.add_edge("reject", "audit")
    g.add_edge("audit", END)

    if llm is not None:
        # Phase 2.4/3.1: wire input_guard → router → ... (full loop).
        # Full topology (task-2.4 §6 + task-3.1 §5 + task-3.4):
        #   START → input_guard → {reject → audit → END, router → {answer_direct, retrieve}}
        #   grade_documents → {generate → grade_generation, consume_retry → rewrite → retrieve}
        #   grade_generation → {output_guard → {done → audit → END, fail_closed → audit → END}, ...}
        from iw_rag.agent.edges import (
            route_after_grade_documents,
            route_after_grade_generation,
            route_after_input_guard,
            route_after_output_guard,
            route_after_router,
        )
        from iw_rag.agent.nodes import (
            consume_retry,
            done,
            fail_closed,
            make_answer_direct,
            make_generate,
            make_grade_documents,
            make_grade_generation,
            make_rewrite_query,
            make_router,
            output_guard_node,
            sanitize_node,
        )
        from iw_rag.agent.tools import make_retrieve_node

        router_node = make_router(llm, effective_settings)
        answer_direct_node = make_answer_direct(llm, effective_settings)
        retrieve_node = make_retrieve_node(spine)
        grade_documents_node = make_grade_documents(
            llm,
            effective_settings,
            parent_expansion_max_chars=parent_expansion_max_chars,
            max_model_len=generation_max_model_len,
            context_headroom_tokens=context_headroom_tokens,
        )
        rewrite_query_node = make_rewrite_query(llm, effective_settings)
        generate_node = make_generate(assembler)
        grade_generation_node = make_grade_generation(
            llm,
            effective_settings,
            max_model_len=generation_max_model_len,
            context_headroom_tokens=context_headroom_tokens,
        )

        g.add_node("router", router_node)
        g.add_node("answer_direct", answer_direct_node)
        g.add_node("retrieve", retrieve_node)
        g.add_node("sanitize", sanitize_node)  # Task 3.2: sanitize between retrieve → grade_docs
        g.add_node("grade_documents", grade_documents_node)
        g.add_node("consume_retry", consume_retry)
        g.add_node("rewrite_query", rewrite_query_node)
        g.add_node("generate", generate_node)
        g.add_node("grade_generation", grade_generation_node)
        g.add_node("output_guard", output_guard_node)  # Task 3.3: between grade_generation → done
        g.add_node("done", done)
        g.add_node("fail_closed", fail_closed)

        # input_guard → {reject (block), router (allow)} — Task 3.1 entrypoint.
        g.add_conditional_edges(
            "input_guard",
            route_after_input_guard,
            {"reject": "reject", "router": "router"},
        )
        # Phase 4 (WI-031): wire the visual node when visual_retriever is provided.
        # Phase 4 (WI-029): wire the graph node when graph_retriever is provided.
        # Phase 4 (WI-032): wire the long_context node when long_context_route is provided.
        # The route_after_router edge handles the collapse when modes.*.enabled=false.
        router_edge_map: dict[str, str] = {
            "answer_direct": "answer_direct",
            "retrieve": "retrieve",
        }

        if visual_retriever is not None:
            from iw_rag.agent.nodes import make_visual_retrieve_node

            visual_retrieve_node = make_visual_retrieve_node(visual_retriever, spine)
            g.add_node("visual", visual_retrieve_node)
            # visual node → sanitize → grade_documents (same path as retrieve)
            g.add_edge("visual", "sanitize")
            router_edge_map["visual"] = "visual"

        if graph_retriever is not None:
            from iw_rag.agent.nodes import make_graph_retrieve_node

            graph_retrieve_node = make_graph_retrieve_node(graph_retriever, spine)
            g.add_node("graph", graph_retrieve_node)
            # graph node → sanitize → grade_documents (same path as retrieve/visual)
            g.add_edge("graph", "sanitize")
            router_edge_map["graph"] = "graph"

        if long_context_route is not None:
            from iw_rag.agent.nodes import make_long_context_node

            long_context_node = make_long_context_node(long_context_route, spine, assembler)
            g.add_node("long_context", long_context_node)
            # D23 [DECISION — long_context wiring]:
            # long_context node → generate DIRECTLY (bypasses sanitize + grade_documents).
            # Rationale:
            # 1. Nothing was retrieved from a chunk index — the whole doc is assembled;
            #    per-chunk relevance grading (grade_documents) is meaningless.
            # 2. sanitize asserts acl_applied=True on chunks; long_context emits
            #    acl_applied=False (the single sanctioned carve-out, CONVENTIONS §6).
            #    Running sanitize would trip that assertion.
            # 3. The node already writes BOTH state["chunks"] AND state["graded"]
            #    directly (assembled doc-level chunks), so generate can consume
            #    state["graded"] without grade_documents.
            g.add_edge("long_context", "generate")
            router_edge_map["long_context"] = "long_context"

        g.add_conditional_edges(
            "router",
            route_after_router,
            router_edge_map,
        )

        # Skip path: answer_direct → grade_generation (Task 2.4 wiring).
        # ISSUP is bypassed deterministically when graded/citations are empty.
        g.add_edge("answer_direct", "grade_generation")

        # Task 3.2: retrieve → sanitize → grade_documents.
        # The corrective cycle (rewrite_query → retrieve) also flows through
        # sanitize on re-retrieval so every retrieval pass is sanitized.
        g.add_edge("retrieve", "sanitize")
        g.add_edge("sanitize", "grade_documents")

        # Corrective cycle wiring (Task 2.3):
        #   grade_documents → {generate; consume_retry → rewrite_query → retrieve}
        g.add_conditional_edges(
            "grade_documents",
            route_after_grade_documents,
            {"generate": "generate", "consume_retry": "consume_retry"},
        )
        g.add_edge("consume_retry", "rewrite_query")
        g.add_edge("rewrite_query", "retrieve")  # corrective cycle back-edge

        # Generation verification + output guard wiring (Task 2.4/2.5/3.3/3.4):
        #   generate → grade_generation → {
        #       output_guard → {done → audit → END, fail_closed → audit → END},
        #       consume_retry → ...,
        #       fail_closed → audit → END,
        #   }
        # Task 3.4: done/fail_closed → audit → END (not directly to END).
        g.add_edge("generate", "grade_generation")
        g.add_conditional_edges(
            "grade_generation",
            route_after_grade_generation,
            {
                "output_guard": "output_guard",
                "consume_retry": "consume_retry",
                "fail_closed": "fail_closed",
            },
        )
        # Task 3.3: output_guard → {done (allow), fail_closed (block)}.
        # Task 3.4: done/fail_closed now route to audit before END.
        # HITL interrupt_before=["done"] semantics preserved: done is still the
        # pre-emit pause point; output_guard runs before it; audit runs after.
        g.add_conditional_edges(
            "output_guard",
            route_after_output_guard,
            {"done": "done", "fail_closed": "fail_closed"},
        )
        # Task 3.4: terminal nodes → audit → END (was → END directly).
        g.add_edge("done", "audit")
        g.add_edge("fail_closed", "audit")
    else:
        # Phase 2.0 fallback + Phase 3.1 guard + Task 3.4 audit:
        # input_guard → {reject, answer} → audit → END.
        from iw_rag.agent.edges import route_after_input_guard_degraded

        g.add_node("answer", _make_answer_node(spine, assembler))
        g.add_conditional_edges(
            "input_guard",
            route_after_input_guard_degraded,
            {"reject": "reject", "answer": "answer"},
        )
        # Task 3.4: answer → audit → END (not answer → END directly).
        g.add_edge("answer", "audit")

    if interrupt_before is not None:
        logger.debug("HITL enabled: interrupt_before=%r", interrupt_before)

    return g.compile(checkpointer=checkpointer, interrupt_before=interrupt_before)


# ---------------------------------------------------------------------------
# Public entrypoint: single-turn invoke
# ---------------------------------------------------------------------------


async def answer_question(
    graph: Any,
    question: str,
    ctx: AccessContext,
    *,
    thread_id: str = "default",
    recursion_limit: int = RECURSION_LIMIT,
    max_retries: int = DEFAULT_MAX_RETRIES,
    allow_ungrounded: bool = False,
    history_chunks: list[Any] | None = None,
) -> GraphState:
    """Invoke the compiled graph and return the terminal ``GraphState``.

    Initialises the minimal required state (``original_question``, ``retries``,
    ``verdicts``) and passes ``thread_id`` + ``recursion_limit`` in the LangGraph
    config dict.  Any ``GraphRecursionError`` is caught and converted to a
    fail-closed refusal so the caller never sees a raw framework exception.

    Args:
        graph: The compiled LangGraph graph.
        question: The user's question.
        ctx: The authenticated ``AccessContext`` (from auth, never the LLM).
        thread_id: LangGraph thread id for checkpointing / resume.
        recursion_limit: LangGraph super-step safety net.
        max_retries: Maximum correction passes (budget counter); overrides the
            module-level default for this invocation.
        allow_ungrounded: Assist mode. When True, an ungrounded answer is
            returned (marked) rather than fail-closed refused; groundedness
            becomes informational. All other guards stay enforced. Defaults
            False (strict fail-closed).

    Returns:
        The terminal ``GraphState`` dict.
    """
    import time

    from langgraph.errors import GraphRecursionError

    from iw_rag.observability.timeline import build_breakdown, start_timeline
    from iw_rag.observability.tracing import hash_groups, span

    init: GraphState = {
        "question": question,
        "original_question": question,
        "access_context": ctx,
        "retries": 0,
        "max_retries": max_retries,
        "allow_ungrounded": allow_ungrounded,
        "history_chunks": list(history_chunks or []),
        "verdicts": [],
    }
    config: dict[str, Any] = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": recursion_limit,
    }
    # A per-request timeline collects each node's wall-time (via the same spans
    # that feed Phoenix) so the breakdown can ship INSIDE the /query response —
    # the frontend renders it as the per-answer "how long did this take?" popup.
    # It works even when OTel/Phoenix is disabled.
    with start_timeline() as timeline:
        wall_start = time.perf_counter()
        # Root span so every per-node span (router/grade/generate/guards/…) nests
        # into ONE Phoenix trace per request rather than scattering as separate
        # single-span traces. OTel contextvars propagate into the awaited nodes.
        with span(
            "iw_rag.query",
            {"thread_id": thread_id, "principal_groups_hash": hash_groups(ctx.principal_groups)},
        ):
            try:
                result: GraphState = await graph.ainvoke(init, config=config)
            except GraphRecursionError:
                # Safety net: the state counter should have tripped first; log as anomaly
                # (a missing consume_retry on some edge is the likely root cause).
                logger.warning(
                    "recursion_limit hit — counter failed to trip first (anomaly); "
                    "thread_id=%r — converting to fail-closed refusal",
                    thread_id,
                )
                result = {
                    **init,
                    "answer": None,
                    "refusal": FAIL_CLOSED_MESSAGE,
                    "disposition": "refused",
                    "diagnostics": {"anomaly": "GraphRecursionError"},
                }
        total_ms = round((time.perf_counter() - wall_start) * 1000.0, 1)

    # Merge the timing breakdown into diagnostics (the diagnostics channel is a
    # plain dict on the terminal state here — merge, never clobber prior keys).
    diag = dict(result.get("diagnostics") or {})
    diag["timeline"] = build_breakdown(timeline)
    diag["total_ms"] = total_ms
    result["diagnostics"] = diag
    return result


# ---------------------------------------------------------------------------
# Public entrypoint: streaming (SSE-shaped events)
# ---------------------------------------------------------------------------


async def answer_question_stream(
    graph: Any,
    question: str,
    ctx: AccessContext,
    *,
    thread_id: str = "default",
    recursion_limit: int = RECURSION_LIMIT,
    allow_ungrounded: bool = False,
    history_chunks: list[Any] | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Stream SSE-shaped events from the graph invocation.

    Yields one dict per event following the Q-SSE-SCHEMA canonical taxonomy
    (``token`` · ``step`` · ``source`` · ``verdict`` · ``final`` · ``refusal`` ·
    ``error`` + ``: ping``). This generator emits the subset it can produce from
    the graph's super-step stream:
    - ``step``    — node lifecycle (start/end).
    - ``source``  — per retrieved chunk.
    - ``verdict`` — each grader verdict appended.
    - ``final``   — terminal grounded answer.
    - ``refusal`` — terminal fail-closed refusal.

    The ``error`` event (``{message, trace_ref}``) is emitted by the SSE
    formatting layer (``api.routes._sse_generator``) if this generator raises,
    so the stream always ends with an explicit frame. ``token`` (incremental
    answer text) is reserved in the canonical set but not yet emitted — the
    generator yields whole answers in the ``final`` event; a consumer must
    tolerate the absence of ``token`` events.

    Args:
        graph: The compiled LangGraph graph.
        question: The user's question.
        ctx: The authenticated ``AccessContext``.
        thread_id: LangGraph thread id.
        recursion_limit: LangGraph super-step safety net.

    Yields:
        SSE-shaped event dicts with ``"event"`` and ``"data"`` keys.
    """
    init: GraphState = {
        "question": question,
        "original_question": question,
        "access_context": ctx,
        "retries": 0,
        "max_retries": DEFAULT_MAX_RETRIES,
        "allow_ungrounded": allow_ungrounded,
        "history_chunks": list(history_chunks or []),
        "verdicts": [],
    }
    config: dict[str, Any] = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": recursion_limit,
    }

    # Accumulate node deltas into one merged view. ``astream`` yields per-node
    # state *updates* (stream_mode="updates"), not the full state — so the
    # terminal ``done`` node's delta carries ``disposition="answered"`` but NOT
    # ``citations`` (it only promotes the verified draft). Tracking the last
    # disposition-bearing delta alone would therefore drop the citations the
    # ``generate`` node emitted earlier. Merging deltas mirrors the reducer the
    # non-streaming ``ainvoke`` path applies, so the ``final`` event matches.
    final_state: dict[str, Any] = {}
    step_attempt = 0

    from iw_rag.observability.tracing import current_trace_ref, hash_groups, span

    # Root span so per-node spans nest into one Phoenix trace (parallels
    # answer_question). Explicit enter/exit avoids re-indenting the stream loop.
    root_span = span(
        "iw_rag.query",
        {"thread_id": thread_id, "principal_groups_hash": hash_groups(ctx.principal_groups)},
    )
    root_span.__enter__()
    trace_ref_id = current_trace_ref().get("trace_id") or thread_id

    try:
        async for chunk in graph.astream(init, config=config):
            # Each chunk is {node_name: partial_state_dict}
            for node_name, node_output in chunk.items():
                if not isinstance(node_output, dict):
                    continue

                # --- step event: node completed ---
                yield {
                    "event": "step",
                    "data": {
                        "node": node_name,
                        "status": "complete",
                        "attempt": step_attempt,
                        "detail": None,
                    },
                }

                # --- source events: one per retrieved chunk ---
                chunks_out = node_output.get("chunks") or []
                for rc in chunks_out:
                    yield {
                        "event": "source",
                        "data": {
                            "chunk_id": rc.chunk_id,
                            "document_id": rc.metadata.get("document_id", ""),
                            "source_uri": rc.source_uri,
                            "score": rc.score,
                            "snippet": (rc.content or "")[:200],
                        },
                    }

                # --- verdict events: one per Verdict appended ---
                for v in node_output.get("verdicts") or []:
                    yield {
                        "event": "verdict",
                        "data": {
                            "kind": v.kind,
                            "value": v.value,
                            "score": v.score,
                            "reason": v.reason,
                        },
                    }

                # Merge this delta so the final/refusal event sees the fully
                # accumulated state (citations from generate + disposition/answer
                # from done), not just the last disposition-bearing delta.
                final_state.update(node_output)

                step_attempt += 1

    except Exception:
        # Re-raise non-recursion errors; recursion is handled by answer_question
        raise
    finally:
        root_span.__exit__(None, None, None)

    # --- terminal event ---
    if final_state.get("disposition") == "answered":
        citations_raw = final_state.get("citations") or []
        yield {
            "event": "final",
            "data": {
                "answer": final_state.get("answer", ""),
                "citations": [
                    {
                        "marker": c.marker,
                        "chunk_id": c.chunk_id,
                        "source_uri": c.source_uri,
                        "quote": c.quote,
                    }
                    for c in citations_raw
                ],
                "trace_ref": trace_ref_id,
                "acl_applied": True,
                "attempts": step_attempt,
            },
        }
    else:
        # Fail-closed refusal
        refusal_msg = final_state.get("refusal") or "insufficient grounded evidence"
        yield {
            "event": "refusal",
            "data": {
                "reason_code": "no_grounded_answer",
                "message": refusal_msg,
                "trace_ref": trace_ref_id,
            },
        }
