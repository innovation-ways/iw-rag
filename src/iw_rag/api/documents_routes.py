"""HTTP routes for document management — ``/documents`` (list/upload/delete/view).

All routes resolve the caller's :class:`AccessContext` via the same security
keystone as ``/query`` (groups from the bearer token only).  Listing returns the
caller's own + ``common`` documents; upload stamps the owner from auth; delete is
owner-only and purges all derived artifacts; raw view is gated by visibility.

The :class:`~iw_rag.api.documents.DocumentService` lives on ``app.state`` (built in
``create_app`` for live wiring, or injected by tests).
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, File, Form, Query, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

from iw_rag.api.auth import Authenticator
from iw_rag.api.documents import DocumentInfo, DocumentService
from iw_rag.api.documents_page import documents_page_html
from iw_rag.api.download_tokens import verify as verify_download_token
from iw_rag.api.routes import (
    AccessCtx,
    OptionalAccessCtx,
    _download_token_secret,
    _scope_to_session,
)
from iw_rag.api.schemas import AttachmentOut, DocumentOut
from iw_rag.domain.acl import AccessContext
from iw_rag.domain.errors import AuthenticationError, NotFoundError, ValidationError

documents_router = APIRouter()


@documents_router.get("/documents/ui", response_class=HTMLResponse, include_in_schema=False)
async def documents_ui() -> HTMLResponse:
    """Serve the standalone document-management page (handles its own login)."""
    return HTMLResponse(content=documents_page_html())


def _service(request: Request) -> DocumentService:
    """Resolve the document service from app state (503 if not wired)."""
    service: DocumentService | None = getattr(request.app.state, "document_service", None)
    if service is None:
        raise ValidationError("document management is not enabled for this deployment")
    return service


def _to_out(info: DocumentInfo) -> DocumentOut:
    """Map the service DTO to the API response model."""
    return DocumentOut(
        id=info.id,
        filename=info.filename,
        title=info.title,
        mime=info.mime,
        visibility=info.visibility,
        owner_id=info.owner_id,
        size_bytes=info.size_bytes,
        chunk_count=info.chunk_count,
        created_at=info.created_at,
        is_owner=info.is_owner,
        kind=info.kind,
    )


async def _resolve_ctx_from_token(
    request: Request,
    document_id: str,
    token: str,
) -> AccessContext:
    """Re-resolve the principal's current ``AccessContext`` from a signed token.

    The token must verify, the path ``{document_id}`` must match the
    token's bound ``document_id``, the auth source must know the
    ``principal_id``, and (if the token carries a session) the session
    must still be live and owned by the principal (re-validated
    server-side by :func:`iw_rag.api.routes._scope_to_session`).

    Every failure path raises :class:`NotFoundError` — a failed token is
    indistinguishable from a missing document, so the API never leaks
    which one.  ``principal_groups`` ALWAYS come from auth (either the
    bearer path's :func:`augment_groups` or
    :meth:`Authenticator.resolve_principal` here) — never from the
    token payload (the keystone is preserved).

    Args:
        request: The raw FastAPI request (for app state + settings +
            authenticator + document service).
        document_id: The path ``{document_id}`` the token is being
            used against.  Must equal the token's bound ``document_id``.
        token: The signed token from the ``?token=`` query parameter
            (attacker-controlled; treated as opaque until ``verify``
            passes).

    Returns:
        A :class:`AccessContext` whose ``principal_groups`` were
        re-resolved from the auth source and (when the token carries a
        ``session_id``) widened with the validated session group.

    Raises:
        NotFoundError: On any verification failure, document-id
            mismatch, unknown principal, or session-widening refusal
            (existence never leaked).
    """
    secret = _download_token_secret(request)
    verified = verify_download_token(token, secret=secret)
    if verified is None:
        raise NotFoundError("document not found")
    if verified.document_id != document_id:
        # Mismatched path — the token is a capability for ONE document;
        # a request against another id must not serve bytes.
        raise NotFoundError("document not found")

    authenticator: Authenticator = request.app.state.authenticator
    ctx = authenticator.resolve_principal(verified.principal_id)
    if ctx is None:
        # The principal is unknown to the auth source (revoked, renamed,
        # or the auth mode is non-reversible like static_token).  Never
        # serve bytes; never leak which one.
        raise NotFoundError("document not found")

    # Session-widen ONLY when the token actually carries a session id
    # and the principal still owns the live session — same server-side
    # re-validation as the bearer path's _scope_to_session (so a leaked
    # token cannot reach a session the bearer principal already lost
    # ownership of).
    if verified.session_id is not None:
        ctx = await _scope_to_session(request, ctx, verified.session_id)

    return ctx


@documents_router.get("/documents", response_model=list[DocumentOut])
async def list_documents(request: Request, ctx: AccessCtx) -> list[DocumentOut]:
    """List the caller's own + common documents (newest first)."""
    infos = await _service(request).list_documents(ctx)
    return [_to_out(i) for i in infos]


@documents_router.post("/documents", response_model=DocumentOut, status_code=201)
async def upload_document(
    request: Request,
    ctx: AccessCtx,
    file: Annotated[UploadFile, File(...)],
    visibility: Annotated[str, Form()] = "specific",
) -> DocumentOut:
    """Upload a document; ingest it under the caller's ACL.

    ``visibility`` is ``specific`` (owner-only) or ``common`` (all users).  The
    file is parsed (txt/md/pdf/image-OCR), chunked, embedded, and persisted; the
    owner is taken from auth (never the request body).
    """
    data = await file.read()
    info = await _service(request).upload(
        ctx,
        filename=file.filename or "upload",
        data=data,
        visibility=visibility,
        declared_mime=file.content_type,
    )
    return _to_out(info)


@documents_router.delete("/documents/{document_id}", status_code=204)
async def delete_document(request: Request, ctx: AccessCtx, document_id: str) -> None:
    """Delete a document the caller owns and purge its embeddings."""
    await _service(request).delete(ctx, document_id)


@documents_router.get("/documents/{document_id}/raw")
async def get_document_raw(
    request: Request,
    bearer_ctx: OptionalAccessCtx,
    document_id: str,
    token: Annotated[str | None, Query()] = None,
) -> Any:
    """Stream a document's original bytes (gated by ACL overlap).

    Authorization (CR-00098 / S04):

    1. **Bearer path** — when an ``Authorization`` header is present
       and authenticates, the resolved :class:`AccessContext` is used
       and ``DocumentService.get_raw`` re-runs the ACL overlap check.
       This is the original /raw behavior, unchanged.
    2. **Token path** — when **no** ``Authorization`` header is present
       (or it did not authenticate) but a ``?token=`` query parameter
       is provided, the signed token is verified
       (:func:`iw_rag.api.download_tokens.verify`), the principal's
       **current** groups are re-resolved from the auth source
       (:meth:`Authenticator.resolve_principal`), the session (if any)
       is re-validated server-side via
       :func:`iw_rag.api.routes._scope_to_session`, and the result is
       handed to ``get_raw`` (whose ACL overlap is the final
       authority — defense in depth; mint-gating alone is not
       trusted).
    3. **Both** — a successful bearer is preferred (the token is
       ignored).  A failed bearer is silently ignored and the token
       path is tried.  An empty secret makes the token path fail
       closed; the bearer path is unaffected.
    4. **Neither** — :class:`AuthenticationError` (→ 401).

    Every failed token is mapped to :class:`NotFoundError` (→ 404) so
    the API never leaks why the download was refused — a missing
    document and a refused token are indistinguishable.

    The response is a :class:`FileResponse` with
    ``Content-Disposition: attachment`` (the default when ``filename``
    is set); a cross-origin browser ``<a download>`` click ignores
    the HTML ``download`` attribute, so the server's ``attachment``
    disposition is what actually forces a download rather than a
    navigation.  The header is preserved on both the bearer and the
    token paths (S07 + S15 assert it).

    Args:
        request: The raw FastAPI request (for app state).
        bearer_ctx: The optional bearer-resolved context (``None`` when
            the header is absent or invalid).
        document_id: The path ``{document_id}`` the download is for.
        token: The optional ``?token=`` signed download token.

    Returns:
        A :class:`FileResponse` for the document's original bytes.

    Raises:
        AuthenticationError: When neither a valid bearer nor a valid
            token is present (→ 401).
        NotFoundError: When the token fails to verify, the path
            ``document_id`` does not match the token's bound id, the
            principal is unknown to the auth source, or the ACL
            overlap (re-checked by :meth:`DocumentService.get_raw`) is
            empty — existence is never leaked.
    """
    # Bearer preferred when it authenticates (richer, has a header, no
    # token-needed fallback).  Falls through to the token path when
    # the header is absent or did not resolve.
    if bearer_ctx is not None:
        ctx = bearer_ctx
    elif token:
        ctx = await _resolve_ctx_from_token(request, document_id, token)
    else:
        raise AuthenticationError("invalid or missing bearer token")

    raw = await _service(request).get_raw(ctx, document_id)
    return FileResponse(path=raw.path, media_type=raw.mime, filename=raw.filename)


@documents_router.post("/attachments", response_model=AttachmentOut, status_code=201)
async def attach_file(
    request: Request,
    ctx: AccessCtx,
    file: Annotated[UploadFile, File(...)],
    session_id: Annotated[str, Form()],
) -> AttachmentOut:
    """Attach an ephemeral file to a chat session (queryable only within that chat).

    The attachment is ingested with a server-issued ``session:<id>`` ACL tag and
    leaves no trace once the session is purged (on chat end or by TTL sweep).
    """
    data = await file.read()
    info = await _service(request).attach(
        ctx,
        session_id=session_id,
        filename=file.filename or "attachment",
        data=data,
        declared_mime=file.content_type,
    )
    return AttachmentOut(
        document_id=info.id,
        session_id=session_id,
        filename=info.filename,
        chunk_count=info.chunk_count,
    )


@documents_router.delete("/attachments/{session_id}", status_code=204)
async def purge_attachments(request: Request, ctx: AccessCtx, session_id: str) -> None:
    """Purge all of a chat session's ephemeral attachments (on chat end)."""
    await _service(request).purge_session(ctx, session_id)
