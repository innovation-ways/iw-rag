"""HTTP routes — ``/query`` (cited answer), ``/query/stream`` (SSE), ``/health``, ``/ingest``.

``get_access_context`` is the security keystone: groups come only from verified
auth. ``grounded=False`` returns HTTP 200 (honest refusal) — never a 404 that
would leak the existence of restricted documents.

``/query`` accepts ``?loop=off`` to bypass the agent graph and call
spine+assembler directly (Phase-0 path; enables loop-on-vs-off experiment #4).
``/query/stream`` returns ``text/event-stream`` with SSE events following the
canonical Q-SSE-SCHEMA taxonomy.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from iw_rag.api import download_tokens
from iw_rag.api.auth import Authenticator, bearer_token
from iw_rag.api.schemas import (
    CitationOut,
    ConversationCreateResponse,
    ConversationHistoryResponse,
    ConversationMessageOut,
    HealthResponse,
    LoginRequest,
    LoginResponse,
    QueryRequest,
    QueryResponse,
)
from iw_rag.api.system_power import maybe_warm_generator
from iw_rag.domain.acl import AccessContext
from iw_rag.domain.errors import AuthenticationError, AuthorizationError
from iw_rag.domain.models import Citation

router = APIRouter()
logger = logging.getLogger(__name__)


_SESSION_GROUP_PREFIX = "session:"


def _download_token_secret(request: Request) -> str:
    """Return the configured ``download_token_secret`` (``""`` if unset).

    The secret is sourced from :class:`Settings.api.download_token_secret`
    (env-only, ``IW_RAG__API__DOWNLOAD_TOKEN_SECRET``).  An empty/blank
    value is the **fail-safe** — the S04 token path must reject every
    token in that case (the feature is off); the bearer path is
    unaffected.

    Defined here (not in ``documents_routes``) so the S05
    :func:`build_citation_out` helper and the S04 download route can
    share a single source of truth.
    """
    settings = getattr(request.app.state, "settings", None)
    if settings is None:
        return ""
    api = getattr(settings, "api", None)
    if api is None:
        return ""
    return str(getattr(api, "download_token_secret", "") or "")


def _download_token_ttl_seconds(request: Request) -> int:
    """Return the configured ``download_token_ttl_seconds`` (``300`` when unset).

    The TTL is read from :class:`Settings.api.download_token_ttl_seconds`
    (env override ``IW_RAG__API__DOWNLOAD_TOKEN_TTL_SECONDS``;
    ``config/default.yaml`` defaults to 300 s).  A non-positive value
    falls back to 300 s so a misconfigured deployment never produces
    a zero/negative-TTL token.
    """
    settings = getattr(request.app.state, "settings", None)
    if settings is None:
        return 300
    api = getattr(settings, "api", None)
    if api is None:
        return 300
    try:
        ttl = int(getattr(api, "download_token_ttl_seconds", 300) or 300)
    except (TypeError, ValueError):
        return 300
    return ttl if ttl > 0 else 300


def _extract_session_id(ctx: AccessContext) -> str | None:
    """Return the validated ``session_id`` carried in ``ctx``'s groups, or ``None``.

    The :func:`_scope_to_session` helper widens ``ctx`` with a synthetic
    ``session:<id>`` group iff the request's ``session_id`` was
    re-validated server-side.  This helper recovers that id so the S05
    download token can be bound to it — the S04 path then re-validates
    session ownership again before serving bytes (defense in depth,
    CR-00098 design §"Attachments").

    Args:
        ctx: The :class:`AccessContext` after ``_scope_to_session`` ran.

    Returns:
        The session id when ``ctx.principal_groups`` contains a
        ``session:<id>`` entry, else ``None``.
    """
    for group in ctx.principal_groups:
        if group.startswith(_SESSION_GROUP_PREFIX):
            return group[len(_SESSION_GROUP_PREFIX) :]
    return None


def _citation_to_out(
    citation: Citation, *, ctx: AccessContext, secret: str, ttl_seconds: int
) -> CitationOut:
    """Map a domain :class:`Citation` to a :class:`CitationOut` (CR-00098 S05).

    Threads the chunk's owning ``document_id`` (S01) onto the response
    model, and mints a short-lived signed download token so the
    frontend does not need to mint anything client-side.  Token payload
    is ``{document_id, principal_id=ctx.principal_id, exp=now+ttl,
    session_id?}`` — the ``session_id`` is included iff the request
    was widened by :func:`_scope_to_session` (an attachment
    scenario) so the S04 path can re-validate session ownership before
    serving bytes.

    Fail-soft: a missing/blank secret, an empty ``document_id``, or any
    minting exception yields ``download_url=""`` (the response still
    carries ``document_id``).  Citation enrichment MUST NEVER 500 the
    query — the keystone is that ``principal_groups`` come from auth,
    not from the link.

    Args:
        citation: The resolved :class:`~iw_rag.domain.models.Citation`
            to serialize.
        ctx: The :class:`AccessContext` resolved from auth (or, for
            the token path, re-resolved server-side).  ``ctx.principal_id``
            is the only identity surfaced into the token.
        secret: The configured ``download_token_secret`` (``""`` when
            unset — fail-safe, no token will be minted).
        ttl_seconds: The token lifetime in seconds.

    Returns:
        A fully populated :class:`CitationOut` ready to be serialised
        into the ``/query`` response or the SSE ``final`` event.
    """
    document_id = citation.document_id or ""
    download_url = ""
    if document_id and secret:
        try:
            session_id = _extract_session_id(ctx)
            token = download_tokens.mint(
                document_id=document_id,
                principal_id=ctx.principal_id,
                secret=secret,
                ttl_seconds=ttl_seconds,
                session_id=session_id,
            )
            download_url = f"/documents/{document_id}/raw?token={token}"
        except ValueError:
            # ``mint`` raises only on programmer-error inputs (empty
            # secret / ids / non-positive TTL) — all of which are
            # guarded above, but the catch is here so a future caller
            # cannot accidentally 500 the query because of the link.
            download_url = ""
    return CitationOut(
        marker=citation.marker,
        chunk_id=citation.chunk_id,
        source_uri=citation.source_uri,
        quote=citation.quote,
        document_id=document_id,
        download_url=download_url,
    )


async def get_access_context(request: Request) -> AccessContext:
    """Resolve the caller's AccessContext from the bearer token (keystone).

    Raises:
        AuthenticationError: If the token is missing or unknown (→ 401).
    """
    authenticator: Authenticator = request.app.state.authenticator
    trace_id = getattr(request.state, "trace_id", "")
    ctx = authenticator.authenticate(bearer_token(request.headers.get("authorization")), trace_id)
    if ctx is None:
        raise AuthenticationError("invalid or missing bearer token")
    return ctx


AccessCtx = Annotated[AccessContext, Depends(get_access_context)]


async def get_access_context_optional(request: Request) -> AccessContext | None:
    """Resolve the caller's AccessContext from the bearer token, or ``None``.

    Same as :func:`get_access_context` but does **not** raise on a
    missing or unknown bearer — returns ``None`` instead.  Used by the
    S04 (CR-00098) ``/documents/{id}/raw`` route, which can be reached
    *without* an ``Authorization`` header iff a valid signed
    ``?token=`` query parameter is present.  The route handler is the
    single point that decides whether the request is authorized (either
    path: a valid bearer OR a valid token + ACL overlap); the dependency
    is intentionally a thin pass-through so the bearer keystone stays
    uniform across the surface.

    Security: the route handler never trusts request input.  The
    ``principal_groups`` it eventually threads through to
    ``DocumentService.get_raw`` come from auth (``augment_groups``) or
    from re-resolving the token's ``principal_id`` server-side
    (``Authenticator.resolve_principal``) — never from the request
    body, query string, or token payload.
    """
    authenticator: Authenticator = request.app.state.authenticator
    trace_id = getattr(request.state, "trace_id", "")
    return authenticator.authenticate(bearer_token(request.headers.get("authorization")), trace_id)


OptionalAccessCtx = Annotated[AccessContext | None, Depends(get_access_context_optional)]


async def _scope_to_session(
    request: Request, ctx: AccessContext, session_id: str | None
) -> AccessContext:
    """Widen ``ctx`` with the ``session:<id>`` group iff the caller owns the session.

    The synthetic session group is granted ONLY here, server-side, after the
    document service re-validates that the session exists, is owned by the
    authenticated principal, and is unexpired.  An unknown/foreign/expired session
    leaves ``ctx`` unchanged (the keystone is preserved — groups never come from
    unvalidated request input).
    """
    if not session_id:
        return ctx
    service = getattr(request.app.state, "document_service", None)
    if service is None:
        return ctx
    group = await service.session_group_for(ctx, session_id)
    if group is None:
        return ctx
    return AccessContext(
        principal_id=ctx.principal_id,
        principal_groups=(*ctx.principal_groups, group),
        is_operator=ctx.is_operator,
        request_id=ctx.request_id,
    )


async def require_operator(ctx: AccessCtx) -> AccessContext:
    """Require operator privilege (for ``/ingest``).

    Raises:
        AuthorizationError: If the caller is not an operator (→ 403).
    """
    if not ctx.is_operator:
        raise AuthorizationError("operator privilege required")
    return ctx


OperatorCtx = Annotated[AccessContext, Depends(require_operator)]


@router.post("/auth/login", response_model=LoginResponse)
async def login(req: LoginRequest, request: Request) -> LoginResponse:
    """Exchange demo username/password for a bearer token + resolved identity.

    Only available when the configured authenticator supports login (the
    document-management ``DemoUserAuthenticator``).  The returned ``groups`` are
    for UI display; they never flow back as an ACL assertion — every request
    re-resolves groups server-side from the token (the keystone is unchanged).

    Raises:
        AuthenticationError: On bad credentials or when login is unsupported (→ 401).
    """
    authenticator = request.app.state.authenticator
    login_fn = getattr(authenticator, "login", None)
    trace_id = getattr(request.state, "trace_id", "")
    if login_fn is None:
        raise AuthenticationError("login is not enabled for this deployment")
    token = login_fn(req.username, req.password)
    if token is None:
        raise AuthenticationError("invalid username or password")
    ctx = authenticator.authenticate(token, trace_id)
    if ctx is None:  # pragma: no cover — a freshly issued token must resolve
        raise AuthenticationError("token resolution failed after login")
    return LoginResponse(
        token=token,
        principal_id=ctx.principal_id,
        groups=list(ctx.principal_groups),
        is_operator=ctx.is_operator,
    )


@router.post("/query", response_model=QueryResponse)
async def query(
    req: QueryRequest,
    request: Request,
    ctx: AccessCtx,
    loop: bool = True,
) -> QueryResponse:
    """Answer a query, grounded + cited, under the caller's ACL context.

    Args:
        req: The query request body.
        request: The raw FastAPI request (for app state + trace id).
        ctx: The authenticated ``AccessContext`` (from auth only).
        loop: When ``False`` (``?loop=off``) the agent graph is bypassed and the
            Phase-0 spine+assembler path is used directly (experiment #4).
    """
    # Block-and-warm: if the generator was auto-unloaded for VRAM, power it
    # back on and wait before serving (no-op when power control is off /
    # benchmark mode / generator already up).  Also resets the idle clock.
    await maybe_warm_generator(request)
    service = request.app.state.query_service
    ctx = await _scope_to_session(request, ctx, req.session_id)

    # Chat memory (conversation-as-source): resolve the conversation this turn
    # belongs to, then replay its prior turns as citable, principal-scoped
    # evidence so a follow-up ("what is my name?") can be answered + cited from
    # earlier turns. Best-effort — memory must NEVER break a query, and the
    # service is absent in legacy/test wirings.
    conversation_service = getattr(request.app.state, "conversation_service", None)
    conversation_id = ""
    history_chunks: list[Any] = []
    if conversation_service is not None:
        try:
            conversation_id = await conversation_service.resolve(ctx, req.conversation_id)
            history_chunks = await conversation_service.history_chunks(ctx, conversation_id)
        except Exception:
            logger.warning("conversation memory load failed; continuing without it", exc_info=True)
            conversation_id = ""
            history_chunks = []

    # Support loop= kwarg only on AgentQueryService (Phase-2); fall back to
    # plain answer() for legacy QueryService injected by tests.
    import inspect

    sig = inspect.signature(service.answer)
    kwargs: dict[str, Any] = {}
    if "loop" in sig.parameters:
        kwargs["loop"] = loop
    if "allow_ungrounded" in sig.parameters:
        kwargs["allow_ungrounded"] = req.allow_ungrounded
    if history_chunks and "history_chunks" in sig.parameters:
        kwargs["history_chunks"] = history_chunks
    outcome = await service.answer(req.query, ctx, **kwargs)

    ans = outcome.answer
    secret = _download_token_secret(request)
    ttl_seconds = _download_token_ttl_seconds(request)
    citations_out = [
        _citation_to_out(c, ctx=ctx, secret=secret, ttl_seconds=ttl_seconds) for c in ans.citations
    ]

    # Persist the completed turn so the next turn can see it (best-effort).
    if conversation_service is not None and conversation_id:
        diag = outcome.diagnostics if isinstance(outcome.diagnostics, dict) else {}
        disposition = diag.get("disposition")
        if not isinstance(disposition, str):
            disposition = "answered" if ans.grounded else "refused"
        await conversation_service.record_turn(
            conversation_id,
            user_text=req.query,
            answer_text=ans.answer,
            citations=[c.model_dump() for c in citations_out],
            disposition=disposition,
        )

    return QueryResponse(
        answer=ans.answer,
        grounded=ans.grounded,
        citations=citations_out,
        used_chunk_ids=ans.used_chunk_ids,
        retriever=outcome.retriever,
        trace_id=getattr(request.state, "trace_id", ""),
        diagnostics=outcome.diagnostics or None,
        conversation_id=conversation_id,
    )


async def _sse_generator(
    service: Any,
    query: str,
    ctx: AccessContext,
    trace_id: str,
    request: Request,
) -> AsyncIterator[str]:
    """Yield SSE-formatted strings from the query service stream.

    Delegates to ``service.answer_question_stream`` (on ``AgentQueryService``)
    when available, otherwise falls back to a single synthetic final/refusal
    event built from ``service.answer``.

    Each SSE message is formatted as::

        event: <type>\\n
        data: <json>\\n
        \\n

    A ``: ping`` heartbeat comment may precede the first event.

    Args:
        service: The query service (AgentQueryService or any duck-type).
        query: The user question.
        ctx: The authenticated ``AccessContext`` (post ``_scope_to_session``).
        trace_id: The request trace id.
        request: The raw FastAPI request (for settings / app state when
            serialising the synthetic ``final`` event with download links).

    Yields:
        Raw SSE-formatted byte strings.
    """
    import uuid

    thread_id = str(uuid.uuid4())

    # Ping heartbeat so the client knows the stream started.
    yield ": ping\n\n"

    secret = _download_token_secret(request)
    ttl_seconds = _download_token_ttl_seconds(request)

    # Any failure mid-stream ends with the canonical Q-SSE-SCHEMA ``error`` event
    # (``{message, trace_ref}``) rather than a silently-dropped connection. The
    # message is generic — never the exception text, retrieved content, or PII
    # (lld-api §error mapping); the trace_id correlates to Phoenix for diagnosis.
    try:
        # Prefer a native stream if the service exposes one.
        if hasattr(service, "answer_question_stream"):
            async for event in service.answer_question_stream(query, ctx, thread_id=thread_id):
                event_type = event.get("event", "data")
                event_data = event.get("data", {})
                # The terminal ``final`` event is the only place citations
                # are surfaced in the Phase-2 stream — enrich the dict
                # shape with ``document_id`` + ``download_url`` so the
                # frontend can render clickable download anchors without
                # ever minting anything client-side (CR-00098 S05).
                if event_type == "final" and isinstance(event_data, dict):
                    citations_raw = event_data.get("citations") or []
                    if citations_raw and "marker" in citations_raw[0]:
                        event_data = {
                            **event_data,
                            "citations": [
                                _citation_to_out(
                                    Citation(
                                        marker=int(cit.get("marker", 0)),
                                        chunk_id=str(cit.get("chunk_id", "")),
                                        document_id=str(cit.get("document_id", "")),
                                        source_uri=str(cit.get("source_uri", "")),
                                        quote=str(cit.get("quote", "")),
                                    ),
                                    ctx=ctx,
                                    secret=secret,
                                    ttl_seconds=ttl_seconds,
                                ).model_dump()
                                for cit in citations_raw
                            ],
                        }
                data_str = json.dumps(event_data)
                yield f"event: {event_type}\ndata: {data_str}\n\n"
            return

        # Fallback: call answer() and synthesise a single final/refusal event.
        import inspect

        sig = inspect.signature(service.answer)
        if "loop" in sig.parameters:
            outcome = await service.answer(query, ctx, loop=True)
        else:
            outcome = await service.answer(query, ctx)

        ans = outcome.answer
        if ans.grounded:
            data = {
                "answer": ans.answer,
                "citations": [
                    _citation_to_out(
                        c, ctx=ctx, secret=secret, ttl_seconds=ttl_seconds
                    ).model_dump()
                    for c in ans.citations
                ],
                "trace_ref": trace_id,
                "acl_applied": outcome.acl_applied,
                "attempts": 1,
            }
            yield f"event: final\ndata: {json.dumps(data)}\n\n"
        else:
            data_ref = {
                "reason_code": "no_grounded_answer",
                "message": ans.answer,
                "trace_ref": trace_id,
            }
            yield f"event: refusal\ndata: {json.dumps(data_ref)}\n\n"
    except Exception:
        logger.exception("SSE stream failed (trace_id=%s)", trace_id)
        err = {"message": "stream failed; see trace for details", "trace_ref": trace_id}
        yield f"event: error\ndata: {json.dumps(err)}\n\n"


@router.post("/query/stream")
async def query_stream(
    req: QueryRequest,
    request: Request,
    ctx: AccessCtx,
) -> StreamingResponse:
    """Stream a grounded answer as SSE events (Q-SSE-SCHEMA canonical taxonomy).

    Events follow the ``Q-SSE-SCHEMA`` canonical set (OPEN-QUESTIONS.md):
    ``step`` · ``source`` · ``verdict`` · ``final`` · ``refusal`` · ``: ping``.

    Returns:
        A ``StreamingResponse`` with ``media_type="text/event-stream"``.
    """
    # Block-and-warm before opening the stream (see /query).  Best-effort; a
    # power-on failure surfaces as the canonical SSE error event downstream.
    await maybe_warm_generator(request)
    service = request.app.state.query_service
    ctx = await _scope_to_session(request, ctx, req.session_id)
    trace_id = getattr(request.state, "trace_id", "")
    return StreamingResponse(
        _sse_generator(service, req.query, ctx, trace_id, request),
        media_type="text/event-stream",
    )


@router.post("/conversations", response_model=ConversationCreateResponse)
async def create_conversation(request: Request, ctx: AccessCtx) -> ConversationCreateResponse:
    """Start a new chat thread owned by the caller (chat memory).

    Returns an empty ``conversation_id`` when chat memory is not wired (no
    conversation service) — the client then runs memory-less, gracefully.
    """
    service = getattr(request.app.state, "conversation_service", None)
    if service is None:
        return ConversationCreateResponse(conversation_id="")
    conversation_id = await service.create(ctx)
    return ConversationCreateResponse(conversation_id=conversation_id)


@router.get("/conversations/{conversation_id}", response_model=ConversationHistoryResponse)
async def get_conversation(
    conversation_id: str, request: Request, ctx: AccessCtx
) -> ConversationHistoryResponse:
    """Return a conversation's turns (oldest-first) for the owning caller.

    A foreign / unknown id yields an empty list — never a 404 that would leak
    the existence of another principal's conversation (mirrors the ``grounded``
    HTTP-200 honest-refusal posture).
    """
    service = getattr(request.app.state, "conversation_service", None)
    if service is None:
        return ConversationHistoryResponse(conversation_id=conversation_id, messages=[])
    messages = await service.get_history(ctx, conversation_id)
    return ConversationHistoryResponse(
        conversation_id=conversation_id,
        messages=[
            ConversationMessageOut(
                role=m.role,
                content=m.content,
                citations=[CitationOut.model_validate(c) for c in m.citations],
                disposition=m.disposition,
                created_at=m.created_at,
            )
            for m in messages
        ],
    )


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: str, request: Request, ctx: AccessCtx
) -> dict[str, bool]:
    """Clear a conversation the caller owns (the "Clear" button) — cascades turns.

    Idempotent: deleting an unknown / not-owned conversation returns
    ``{"cleared": false}`` rather than erroring or leaking existence.
    """
    service = getattr(request.app.state, "conversation_service", None)
    if service is None:
        return {"cleared": False}
    cleared = await service.clear(ctx, conversation_id)
    return {"cleared": cleared}


@router.get("/health", response_model=HealthResponse)
async def health(request: Request, deep: bool = False) -> HealthResponse:
    """Liveness + (when ``deep=true``) dependency readiness."""
    checks: dict[str, Any] = {"api": {"ok": True}}
    if deep:
        database = getattr(request.app.state, "database", None)
        if database is not None:
            status = await database.health()
            checks["database"] = {"ok": status.ok, "detail": status.detail}
        llm = getattr(request.app.state, "llm", None)
        if llm is not None:
            checks["vllm"] = {"ok": await llm.health()}
    ok = all(c.get("ok") for c in checks.values())
    return HealthResponse(status="ok" if ok else "degraded", checks=checks)


@router.post("/ingest", status_code=202)
async def ingest(_: OperatorCtx) -> dict[str, str]:
    """Operator-only. Ingestion runs offline (``make ingest``); this acks the seam."""
    return {"status": "accepted", "detail": "ingestion runs offline via `make ingest` in Phase 0"}
