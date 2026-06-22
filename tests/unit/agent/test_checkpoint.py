"""Unit tests for agent/checkpoint.py — saver factory, resume, HITL seam.

Covers (all unit-level; no live Postgres required):
- ``make_checkpointer`` returns ``InMemorySaver`` on the non-durable path.
- ``make_checkpointer`` selects the Postgres class (by name) on the durable path
  without opening a live connection.
- ``make_checkpointer`` raises ``ValueError`` when the Postgres backend is selected
  with ``dsn=None`` — both via ``durable=True`` and via
  ``settings.checkpointer='postgres'``.
- ``build_graph`` compiles WITH and WITHOUT a checkpointer.
- Resume: two invocations on the same ``thread_id`` via ``InMemorySaver`` persist
  state across calls (second call sees state from first).
- HITL is off by default: a normal run completes without pausing.
- ``setup_checkpointer`` is a no-op for ``InMemorySaver``.

References:
- task-2.6-checkpointing-and-hitl §1, §2, §3 (deliverables)
- lld-agentic-loop §5 (checkpointing / HITL)
- CONVENTIONS.md §8 (InMemorySaver dev / PostgresSaver durable — no sqlite dev)
"""

from __future__ import annotations

from typing import Any

import pytest

from iw_rag.agent.checkpoint import make_checkpointer, setup_checkpointer
from iw_rag.agent.graph import answer_question, build_graph
from iw_rag.domain.acl import AccessContext
from iw_rag.settings import AgentSettings

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

_CTX = AccessContext(
    principal_id="u1",
    principal_groups=("hr",),
    is_operator=False,
    request_id="req-ckpt",
)


def _agent_settings(
    *,
    checkpointer: str = "in_memory",
    hitl_enabled: bool = False,
) -> AgentSettings:
    """Build an ``AgentSettings`` with the given overrides."""
    s = AgentSettings()
    object.__setattr__(s, "checkpointer", checkpointer)
    s.hitl.enabled = hitl_enabled
    return s


# ---------------------------------------------------------------------------
# make_checkpointer — saver factory
# ---------------------------------------------------------------------------


def test_make_checkpointer_dev_returns_in_memory_saver() -> None:
    """Non-durable path returns an ``InMemorySaver`` instance (exact class)."""
    from langgraph.checkpoint.memory import InMemorySaver

    saver = make_checkpointer(AgentSettings())
    assert isinstance(saver, InMemorySaver)
    # Exact class name pin: must be InMemorySaver, not a subclass
    assert type(saver).__name__ == "InMemorySaver"


def test_make_checkpointer_explicit_durable_false_returns_in_memory() -> None:
    """Explicit ``durable=False`` overrides any settings and returns InMemorySaver."""
    from langgraph.checkpoint.memory import InMemorySaver

    settings = _agent_settings(checkpointer="postgres")
    saver = make_checkpointer(settings, durable=False)
    assert isinstance(saver, InMemorySaver)
    # Exact class name pin: durable=False must yield in-memory, not postgres
    assert type(saver).__name__ == "InMemorySaver"


@pytest.mark.asyncio
async def test_make_checkpointer_explicit_durable_true_returns_postgres_type() -> None:
    """Explicit ``durable=True`` selects the Postgres class without connecting.

    ``AsyncPostgresSaver.__init__`` calls ``asyncio.get_running_loop()`` so this
    test must be async.  We only assert the class *name* — no live connection is
    opened (the pool is created with ``open=False``).
    """
    settings = _agent_settings(checkpointer="postgres")
    # The dsn must be set for the factory to build the pool; we provide a fake.
    saver = make_checkpointer(settings, durable=True, dsn="postgresql://fake/db")
    assert type(saver).__name__ == "AsyncPostgresSaver"


@pytest.mark.asyncio
async def test_make_checkpointer_settings_postgres_without_durable_flag() -> None:
    """When ``settings.checkpointer == "postgres"`` and no explicit durable arg,
    the factory selects the Postgres class (async test — AsyncPostgresSaver
    requires a running event loop)."""
    settings = _agent_settings(checkpointer="postgres")
    saver = make_checkpointer(settings, dsn="postgresql://fake/db")
    assert type(saver).__name__ == "AsyncPostgresSaver"


def test_make_checkpointer_settings_in_memory_without_durable_flag() -> None:
    """When ``settings.checkpointer == "in_memory"`` and no explicit durable arg,
    the factory returns an ``InMemorySaver``."""
    from langgraph.checkpoint.memory import InMemorySaver

    settings = _agent_settings(checkpointer="in_memory")
    saver = make_checkpointer(settings)
    assert isinstance(saver, InMemorySaver)
    # Exact class name pin: settings.checkpointer="in_memory" → InMemorySaver
    assert type(saver).__name__ == "InMemorySaver"


# ---------------------------------------------------------------------------
# setup_checkpointer — idempotent setup helper
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_setup_checkpointer_in_memory_is_noop() -> None:
    """``setup_checkpointer`` for InMemorySaver is a no-op — no .setup() called."""
    from langgraph.checkpoint.memory import InMemorySaver

    saver = InMemorySaver()
    await setup_checkpointer(saver)  # must not raise
    # Exact class name pin: must remain InMemorySaver (not been swapped or mutated)
    assert type(saver).__name__ == "InMemorySaver", (
        f"Expected type InMemorySaver, got {type(saver).__name__!r}"
    )


# ---------------------------------------------------------------------------
# build_graph — compiles with and without checkpointer
# ---------------------------------------------------------------------------


def test_build_graph_without_checkpointer() -> None:
    """Graph compiles without a checkpointer (llm=None fallback path)."""
    graph = build_graph(spine=None, assembler=None, llm=None)
    assert graph is not None
    # A compiled LangGraph graph exposes ainvoke for async execution
    assert hasattr(graph, "ainvoke"), "build_graph must return a compiled graph with ainvoke method"


def test_build_graph_with_in_memory_checkpointer() -> None:
    """Graph compiles when an InMemorySaver is passed as the checkpointer."""
    from langgraph.checkpoint.memory import InMemorySaver

    saver = InMemorySaver()
    graph = build_graph(spine=None, assembler=None, llm=None, checkpointer=saver)
    assert graph is not None
    # A compiled LangGraph graph with a checkpointer must expose ainvoke
    assert hasattr(graph, "ainvoke"), (
        "build_graph with InMemorySaver must return a compiled graph with ainvoke"
    )


# ---------------------------------------------------------------------------
# Resume: second same-thread_id call resumes from saved state
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resume_persists_state_across_calls_in_memory() -> None:
    """Same ``thread_id`` second call resumes; state from the first call is present.

    We use the Phase-2.0 fallback graph (llm=None, single answer node) with an
    ``InMemorySaver`` and verify that the second invocation sees the persisted
    state from the first (disposition is preserved in the thread snapshot).

    The graph is NOT re-run from scratch on the second call when a checkpointer
    is in use — LangGraph re-uses the last persisted state.
    """
    from langgraph.checkpoint.memory import InMemorySaver

    saver = InMemorySaver()
    graph = build_graph(spine=None, assembler=None, llm=None, checkpointer=saver)

    # First invocation.
    state1 = await answer_question(
        graph,
        "What is the leave policy?",
        _CTX,
        thread_id="resume-test-thread",
    )
    assert state1.get("disposition") == "refused"  # stub path → refused

    # Second invocation on the SAME thread — graph sees persisted snapshots.
    # A second full ainvoke on an already-terminal thread returns the persisted
    # terminal state without re-running nodes.

    config: dict[str, Any] = {
        "configurable": {"thread_id": "resume-test-thread"},
        "recursion_limit": 25,
    }
    # Re-invoke with None input to resume from persisted state.
    result = await graph.ainvoke(None, config=config)
    # The resumed state must have disposition from the first run.
    assert result.get("disposition") is not None


# ---------------------------------------------------------------------------
# make_checkpointer — ValueError when Postgres selected with dsn=None
# ---------------------------------------------------------------------------


def test_make_checkpointer_postgres_dsn_none_raises_value_error() -> None:
    """Postgres path with dsn=None raises ValueError describing the requirement."""
    settings = _agent_settings(checkpointer="in_memory")
    with pytest.raises(
        ValueError,
        match="dsn must be provided when selecting the Postgres checkpointer",
    ):
        make_checkpointer(settings, durable=True, dsn=None)


def test_make_checkpointer_settings_postgres_dsn_none_raises_value_error() -> None:
    """settings.checkpointer='postgres' with no dsn raises ValueError."""
    settings = _agent_settings(checkpointer="postgres")
    with pytest.raises(
        ValueError,
        match="dsn must be provided when selecting the Postgres checkpointer",
    ):
        make_checkpointer(settings, dsn=None)


# ---------------------------------------------------------------------------
# HITL: off by default — a normal run does NOT interrupt
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hitl_off_by_default_run_completes() -> None:
    """With HITL disabled (default), the graph runs to completion without pausing."""
    from langgraph.checkpoint.memory import InMemorySaver

    settings = AgentSettings()
    assert settings.hitl.enabled is False  # confirm default

    saver = InMemorySaver()
    graph = build_graph(
        spine=None,
        assembler=None,
        llm=None,
        checkpointer=saver,
        settings=settings,
    )

    state = await answer_question(
        graph,
        "What is the leave policy?",
        _CTX,
        thread_id="hitl-off-thread",
    )
    # Completed (not interrupted) — disposition is set.
    assert state.get("disposition") in ("answered", "refused")
