"""FastAPI app factory.

``create_app`` wires the security keystone (authenticator), the Phase-2
``AgentQueryService`` (spine + assembler + compiled LangGraph graph),
request-id tracing, and domain-error → HTTP-status handlers (error bodies
carry no content/PII, only a code + trace id).

Collaborators are injectable so tests substitute fakes and never build live
clients; production builds them from ``Settings``.  Tests that inject a fake
``query_service`` remain valid — the injectable seam is unchanged.

Phase 2.6 adds checkpointer injection: ``_build_query_service`` builds a
checkpointer via :func:`~iw_rag.agent.checkpoint.make_checkpointer` (driven by
``settings.agent.checkpointer``: ``"in_memory"`` for dev/demo, ``"postgres"``
for durable) and passes it to ``build_graph``.  This enables fault-tolerant
resume, time-travel, and optional HITL — see
:mod:`iw_rag.agent.checkpoint` and lld-agentic-loop §5.

F-00094 (Benchmark Studio) adds the model-swap catalog: a
``SwapControllerClient`` is built from
``settings.serving.swap_controller_base_url`` and placed on
``app.state.swap_controller_client``.  The main API holds NO in-process
``SwapController`` — the swap-controller sidecar (``:8090``) is the sole
swap-state authority; this client is the only seam between the public
API and that authority.  The activation group is read from
``settings.serving.activation_group`` (default ``"eval-operator"``).
"""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from iw_rag.api.auth import Authenticator, DemoUserAuthenticator, StaticTokenAuthenticator
from iw_rag.api.routes import router
from iw_rag.domain.errors import (
    AuthenticationError,
    AuthorizationError,
    GenerationError,
    IwRagError,
    NotFoundError,
    RetrievalError,
    ValidationError,
)
from iw_rag.serving.llama_swap_client import LlamaSwapUnavailable
from iw_rag.settings import Settings, get_settings

_ERROR_STATUS: list[tuple[type[IwRagError], int]] = [
    (AuthenticationError, 401),
    (AuthorizationError, 403),
    (NotFoundError, 404),
    (ValidationError, 400),
    (RetrievalError, 503),
    (GenerationError, 503),
    (LlamaSwapUnavailable, 503),
]


def _build_query_service(settings: Settings) -> tuple[Any, Any, Any]:
    """Build the production AgentQueryService + database + llm from settings.

    Constructs the retrieval spine, answer assembler, and LangGraph graph, then
    wires them into an :class:`~iw_rag.api.service.AgentQueryService`.  The
    graph is compiled once here and reused for all requests.

    Args:
        settings: The validated ``Settings`` instance.

    Returns:
        ``(agent_query_service, database, llm)``.
    """
    from iw_rag.agent.graph import build_graph
    from iw_rag.api.service import AgentQueryService
    from iw_rag.generation.answer import AnswerAssembler
    from iw_rag.generation.llm import GenerationParams, VllmClient
    from iw_rag.generation.prompts import CitationPrompt
    from iw_rag.retrieval.dense import DenseRetriever
    from iw_rag.retrieval.encode import InfinityQueryEncoder
    from iw_rag.retrieval.rerank import InfinityReranker
    from iw_rag.retrieval.sparse import SparseRetriever
    from iw_rag.retrieval.spine import RetrievalConfig, RetrievalSpine
    from iw_rag.storage.db import Database

    embedder = settings.models.get(settings.retrieval.embed_model)
    reranker_cfg = settings.models.get(settings.retrieval.rerank_model)
    generator = settings.models.get(settings.generation.model)

    database = Database(settings.database)
    rcfg = settings.retrieval
    spine = RetrievalSpine(
        db=database,
        encoder=InfinityQueryEncoder(embedder.endpoint, embedder.served_name),
        dense=DenseRetriever(),
        sparse=SparseRetriever(),
        reranker=InfinityReranker(
            reranker_cfg.endpoint,
            reranker_cfg.served_name,
            max_doc_chars=rcfg.rerank_max_doc_chars,
        ),
        cfg=RetrievalConfig(
            candidate_k=rcfg.candidate_k,
            fused_k=rcfg.fused_k,
            top_k=rcfg.top_k,
            rrf_k=rcfg.rrf_k,
            rrf_weights=rcfg.rrf_weights,
            expansion_mode=rcfg.expansion_mode,
            parent_expansion_max_chars=rcfg.parent_expansion_max_chars,
            rerank_on_failure=rcfg.rerank_on_failure,
        ),
    )
    llm = VllmClient(generator.endpoint, enable_thinking=settings.generation.enable_thinking)
    # Token budget shared by every prompt assembled from retrieved chunks so the
    # generator window (max_model_len) can never overflow (I-CTXOVERFLOW).  The
    # citation prompt reserves the answer's max_tokens for output; the grader
    # nodes build their own budgets (reserving their smaller output ceilings).
    from iw_rag.generation.context_budget import ContextBudget

    generation_budget = ContextBudget(
        max_model_len=settings.generation.max_model_len,
        reserved_output_tokens=settings.generation.max_tokens,
        headroom_tokens=settings.generation.context_headroom_tokens,
    )
    assembler = AnswerAssembler(
        llm=llm,
        prompt=CitationPrompt(budget=generation_budget),
        params=GenerationParams(
            model=generator.served_name,
            temperature=settings.generation.temperature,
            max_tokens=settings.generation.max_tokens,
        ),
        cite_required=settings.generation.cite_required,
    )
    # Build the checkpointer (Phase 2.6): InMemorySaver (dev/demo) or
    # AsyncPostgresSaver (durable) — driven by settings.agent.checkpointer.
    # CONVENTIONS §8: dev default = InMemorySaver; durable = PostgresSaver on
    # the SAME Postgres instance as the data model.  SqliteSaver is NOT used.
    from iw_rag.agent.checkpoint import make_checkpointer

    checkpointer = make_checkpointer(
        settings.agent,
        # Pass the database DSN so the Postgres saver can build its pool if
        # settings.agent.checkpointer == "postgres".
        dsn=settings.database.dsn,
    )

    # Resolve the grader's LOGICAL role name (e.g. "generator") to the actual
    # served model name (e.g. "Qwen/Qwen3-14B-AWQ") before building the graph.
    # The router / grader / rewrite nodes pass ``settings.agent.grader_model``
    # straight to vLLM as the ``model`` field; vLLM 404s on a logical role name.
    # The registry is only available here at the composition root, so resolve
    # here (the faked unit tests never exercise this — model-agnostic clients
    # ignore the name).  If the value is not a registry key it is assumed to be
    # an explicit served name already and used verbatim.
    agent_settings = settings.agent
    try:
        grader_served = settings.models.get(settings.agent.grader_model).served_name
        agent_settings = settings.agent.model_copy(update={"grader_model": grader_served})
    except (KeyError, AttributeError):
        pass

    # Compile the graph once at startup (non-trivial; singleton per process).
    # Phase 2.1: pass the llm client + agent settings so the router is wired in.
    # Phase 2.6: pass the checkpointer for persistent state + HITL support.
    # I-00131: thread the parent-expansion char budget from RetrievalSettings
    # into the grader node so the assembled prompt is bounded defensively
    # (mirror of the spine-level gate).
    graph = build_graph(
        spine,
        assembler,
        llm,
        settings=agent_settings,
        checkpointer=checkpointer,
        parent_expansion_max_chars=rcfg.parent_expansion_max_chars,
        generation_max_model_len=settings.generation.max_model_len,
        context_headroom_tokens=settings.generation.context_headroom_tokens,
    )
    return AgentQueryService(spine=spine, assembler=assembler, graph=graph), database, llm


def _build_document_service(settings: Settings, database: Any) -> Any:
    """Build the DocumentService from settings + an existing Database handle.

    Uses the shared Infinity embed endpoint (pooled per-chunk BGE-M3) — the API
    container has no torch/Docling, so the online upload path embeds via HTTP.
    """
    from iw_rag.api.documents import DocumentService, make_online_chunker
    from iw_rag.ingestion.caption import make_caption_engine
    from iw_rag.ingestion.http_embed import InfinityDocumentEmbedder

    embed_entry = settings.models.get(settings.retrieval.embed_model)
    embedder = InfinityDocumentEmbedder(embed_entry.endpoint, embed_entry.served_name)
    return DocumentService(
        database,
        embedder,
        make_online_chunker(settings),
        upload_dir=settings.api.upload_dir,
        caption_engine=make_caption_engine(settings),
        max_upload_bytes=settings.api.max_upload_mb * 1024 * 1024,
        ephemeral_ttl_minutes=settings.api.ephemeral_ttl_minutes,
    )


def _build_swap_controller_client(settings: Settings) -> Any:
    """Build the swap-controller HTTP client from settings (F-00094).

    The main API holds NO in-process :class:`SwapController` — it proxies to
    the swap-controller sidecar (``:8090``) over HTTP.  This is the **only**
    seam between the public API and the swap authority.

    Args:
        settings: The validated ``Settings`` instance.

    Returns:
        A configured :class:`~iw_rag.serving.swap_controller_client.SwapControllerClient`
        pointed at ``settings.serving.swap_controller_base_url``.
    """
    from iw_rag.serving.swap_controller_client import SwapControllerClient

    return SwapControllerClient(
        base_url=settings.serving.swap_controller_base_url,
        timeout_s=float(settings.serving.health_timeout_s),
    )


def create_app(
    settings: Settings | None = None,
    *,
    authenticator: Authenticator | None = None,
    query_service: Any | None = None,
    database: Any | None = None,
    llm: Any | None = None,
    document_service: Any | None = None,
    conversation_service: Any | None = None,
    swap_controller_client: Any | None = None,
    benchmark_runner: Any | None = None,
    benchmark_service: Any | None = None,
    config_service: Any | None = None,
    scoreboard_service: Any | None = None,
) -> FastAPI:
    """Build the FastAPI app.

    Args:
        settings: Settings (loaded from env when needed and not provided).
        authenticator: Override the authenticator (tests).
        query_service: Override the query service (tests; skips live wiring).
        database: Optional database handle for deep health.
        llm: Optional llm handle for deep health.
        document_service: Override the document service (tests; skips live wiring).
        swap_controller_client: Override the swap-controller client
            (F-00094; tests inject a fake to avoid the live sidecar).
        benchmark_runner: Override the S01 :class:`BenchmarkRunner`
            (F-00095; tests inject a fake to avoid the live swap +
            retrieval+generation pipeline).
        benchmark_service: Override the F-00093 report service
            (F-00095; tests inject a fake to avoid the live DB + stats
            math).  When ``None`` the composition root leaves the
            benchmark routes in a state where the S02 routes raise
            503 — the public app requires an explicit binding once
            F-00093 / F-00094 land.
        config_service: Override the S03
            :class:`~iw_rag.api.config_service.ConfigService` (F-00096
            S07; tests inject a fake to avoid the live DB +
            ``Settings`` validation).  When ``None`` the
            ``config_service`` state attribute is left ``None`` and
            the S07 routes raise :class:`ValidationError` → 400 (the
            existing ``_ERROR_STATUS`` table maps it).
        scoreboard_service: Override the S05
            :class:`~iw_rag.eval.benchmark.scoreboard.ScoreboardService`
            (F-00096 S08; tests inject a fake to avoid the live DB +
            ranking).  When ``None`` the state attribute is left
            ``None`` and the S08 ``GET /benchmark/scoreboard`` route
            raises :class:`ValidationError` → 400 (the existing
            ``_ERROR_STATUS`` table maps it).

    Returns:
        A configured ``FastAPI`` instance.
    """
    if query_service is None or authenticator is None or swap_controller_client is None:
        settings = settings or get_settings()
    if query_service is None:
        assert settings is not None
        query_service, database, llm = _build_query_service(settings)
        if document_service is None and database is not None:
            document_service = _build_document_service(settings, database)
    if authenticator is None:
        assert settings is not None
        # Prefer the richer login-capable demo auth when configured (stable
        # per-user principal_id for document ownership); fall back to the legacy
        # static token→groups map otherwise.
        if settings.api.demo_users:
            authenticator = DemoUserAuthenticator.from_demo_users(dict(settings.api.demo_users))
        else:
            authenticator = StaticTokenAuthenticator.from_config(
                settings.api.static_tokens, settings.api.operator_tokens
            )
    if swap_controller_client is None:
        assert settings is not None
        swap_controller_client = _build_swap_controller_client(settings)

    # Configure OTel/Phoenix export (idempotent; no-ops when the endpoint is
    # unset or the observability deps are absent). Without this call the loop's
    # span() instrumentation never reaches Phoenix.
    if settings is not None:
        from iw_rag.observability.tracing import configure_tracing

        configure_tracing(
            endpoint=settings.observability.phoenix_endpoint, service_name="iw-rag-api"
        )

    app = FastAPI(title="iw-rag", version="0.0.0")
    app.state.authenticator = authenticator
    app.state.query_service = query_service
    app.state.database = database
    app.state.llm = llm
    app.state.document_service = document_service
    # Chat memory (multi-turn): built from the shared database when not injected,
    # so the /query + /conversations routes persist + replay conversation turns.
    # Kept injectable so tests wire a fake without a live DB.
    if conversation_service is None and database is not None:
        from iw_rag.api.conversation import ConversationService

        conversation_service = ConversationService(database)
    app.state.conversation_service = conversation_service
    app.state.swap_controller_client = swap_controller_client
    # F-00095 / S02 — Benchmark Studio composition root.  Both
    # ``benchmark_runner`` (S01) and ``benchmark_service`` (F-00093) remain
    # **injectable** so tests can wire fakes.  The live ``benchmark_service``
    # (the read path: list/detail/compare) is built here from the shared
    # database when none is injected — it composes the append-only F-00093
    # repository + stats compare module (see
    # ``iw_rag.eval.benchmark.service.BenchmarkReportService``).  The
    # ``benchmark_runner`` (POST /benchmark/run) still needs the F-00093
    # metrics engine + F-00094 swap-controller client and stays injected;
    # until it is wired the run route refuses with a typed error.
    #
    # The live build reaches into the query service's Phase-0 retrieval seam
    # (``_spine`` + ``_assembler``).  The real ``AgentQueryService`` always
    # exposes both, so production wires the runner; a minimal/stub query
    # service (e.g. route tests that only exercise document/ACL paths) lacks
    # the seam — in that case we leave ``benchmark_runner`` as ``None`` and
    # the run route degrades to its typed refusal, rather than crashing app
    # construction.
    if (
        benchmark_runner is None
        and settings is not None
        and database is not None
        and query_service is not None
        and hasattr(query_service, "_spine")
        and hasattr(query_service, "_assembler")
    ):
        from iw_rag.api.benchmark_composition import build_benchmark_runner

        benchmark_runner = build_benchmark_runner(settings, database, query_service)
    app.state.benchmark_runner = benchmark_runner
    # Singleton live-run holder (F-00096) — keeps the current/most-recent
    # benchmark run server-side so the console's "Live progress" panel
    # survives a navigate-away-and-back (see benchmark_run_state).
    from iw_rag.api.benchmark_run_state import BenchmarkRunManager

    app.state.benchmark_run_manager = BenchmarkRunManager()
    if benchmark_service is None and database is not None:
        from iw_rag.eval.benchmark.service import BenchmarkReportService

        benchmark_service = BenchmarkReportService(database)
    app.state.benchmark_service = benchmark_service
    # F-00096 S07 — Operator Console Config API composition root.
    # The S03 ``ConfigService`` is the **sole** owner of the validation,
    # locked-set, bounds, and audit-write logic; the S07 route is a
    # thin HTTP facade that delegates to it.  Kept **injectable** so
    # tests can wire a fake and avoid the live DB + Settings
    # validation.  When ``None`` the live service is built here from
    # the shared ``database`` + ``settings`` (mirrors the
    # ``benchmark_service`` block above) so the e2e / production
    # composition root wires the S07 routes end-to-end and they stop
    # surfacing ``ValidationError`` → 400 on every call.  The
    # ``Settings`` ctor consults the in-process overlay source the S03
    # service registers at import time, so the constructed service
    # picks up overlay edits on the next ``get_settings()`` call.
    if config_service is None and database is not None and settings is not None:
        from iw_rag.api.config_service import ConfigService

        config_service = ConfigService(database)
    app.state.config_service = config_service
    # F-00096 S08 — Operator Console Scoreboard composition root.
    # The S05 ``ScoreboardService`` is the **sole** authority on the
    # per-row ordering (AC6: descending faithfulness for the
    # ``faithfulness`` board); the S08 route is a thin HTTP facade
    # that delegates to it.  Kept **injectable** so tests can wire a
    # fake and avoid the live DB + ranking.  When ``None`` the live
    # service is built here from the shared ``database`` (mirrors the
    # ``benchmark_service`` block above) so the e2e / production
    # composition root wires the S08 route end-to-end and it stops
    # surfacing ``ValidationError`` → 400 on every call.
    if scoreboard_service is None and database is not None:
        from iw_rag.eval.benchmark.scoreboard import ScoreboardService

        scoreboard_service = ScoreboardService(database)
    app.state.scoreboard_service = scoreboard_service
    # Settings is also exposed on app.state so per-route dependencies can
    # read sub-settings (e.g. settings.serving.activation_group) without
    # re-importing the settings module.
    app.state.settings = settings

    @app.middleware("http")
    async def _trace_id(request: Request, call_next: Callable[[Request], Awaitable[Any]]) -> Any:
        request.state.trace_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        response = await call_next(request)
        response.headers["x-request-id"] = request.state.trace_id
        return response

    def _handler(status: int) -> Callable[[Request, Exception], JSONResponse]:
        async def handle(request: Request, exc: Exception) -> JSONResponse:
            return JSONResponse(
                status_code=status,
                content={
                    "error": type(exc).__name__,
                    "trace_id": getattr(request.state, "trace_id", ""),
                },
            )

        return handle  # type: ignore[return-value]

    for exc_type, status in _ERROR_STATUS:
        app.add_exception_handler(exc_type, _handler(status))
    app.add_exception_handler(IwRagError, _handler(500))

    app.include_router(router)

    from iw_rag.api.benchmark_routes import benchmark_router
    from iw_rag.api.config_routes import config_router
    from iw_rag.api.console_routes import console_router
    from iw_rag.api.documents_routes import documents_router
    from iw_rag.api.models_routes import models_router
    from iw_rag.api.scoreboard_routes import scoreboard_router

    app.include_router(documents_router)
    app.include_router(models_router)
    app.include_router(benchmark_router)
    app.include_router(config_router)
    app.include_router(scoreboard_router)
    app.include_router(console_router)
    return app
