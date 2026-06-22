"""Chainlit frontend — surfaces the trust story over the iw-rag API.

Login establishes a bearer token (→ AccessContext server-side). Each message
calls ``/query`` and renders: the answer, a per-answer **trust badge** (grounded
vs general-knowledge vs refusal), the cited **sources**, and a clickable
**🔎 How grounded is this?** side panel with the full provenance (groundedness
verdict, retrieval route, guard checks, sources, Phoenix trace).

Assist mode (the "general knowledge" toggle in ⚙️ chat settings, default ON) is
sent to the API as ``allow_ungrounded``: when on, the server may answer from
general knowledge — clearly marked — instead of fail-closing. Every safeguard
(input/injection/safety guards, PII redaction) stays enforced regardless. This
module is a pure API client.
"""

from __future__ import annotations

import asyncio
import html
import os
import uuid
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import urlparse

import chainlit as cl
import httpx
from auth import token_for, token_for_username
from chainlit.auth import get_current_user
from chainlit.input_widget import Switch
from chainlit.server import app as cl_app
from client import attach_file, purge_session, query
from fastapi import Depends

PHOENIX_BASE = os.environ.get("IW_RAG_PHOENIX_BASE_URL", "http://localhost:6006").rstrip("/")
# Phoenix is published on the same host as the app, on this port. We keep the
# port from the configured URL but swap in the BROWSER's hostname at runtime so
# the "View trace" link works when the app is opened remotely (not just on the
# server's own localhost).
_PHOENIX_PORT = urlparse(PHOENIX_BASE).port or 6006


def _public_api_base() -> str:
    """Browser-reachable origin for the API's relative ``download_url``.

    In the docker deploy the Chainlit UI and the FastAPI live on different
    origins, so the API's ``/documents/{id}/raw?token=...`` href (relative
    by design — the API doesn't know its public hostname) must be prefixed
    with an origin the user's browser can actually navigate. Operators
    override via ``IW_RAG_PUBLIC_API_BASE`` (e.g.
    ``https://iw.example.com/api`` behind a reverse proxy); we fall back to
    the API base the frontend already talks to (``IW_RAG_API_BASE``) so
    same-origin dev setups work without extra config. Re-read on every call
    so the env var can be swapped at runtime (and so unit tests can
    monkeypatch it cheaply). The trailing slash is stripped so the relative
    ``download_url`` (which always starts with ``/``) concatenates cleanly.
    """
    return os.environ.get(
        "IW_RAG_PUBLIC_API_BASE", os.environ.get("IW_RAG_API_BASE", "http://localhost:8000")
    ).rstrip("/")


def _phoenix_base_for_session() -> str:
    """Build the Phoenix base URL from the *browser's* host, not the server's.

    Reads the connection headers (engineio exposes them WSGI-style on
    ``session.environ``) to recover the host the user actually typed, so the
    trace link points at e.g. ``http://iw-dev-01:6006`` rather than
    ``http://localhost:6006``. Falls back to the configured env value.
    """
    try:
        environ = getattr(cl.context.session, "environ", None) or {}
    except Exception:
        environ = {}
    return _phoenix_base_from_environ(environ)


def _phoenix_base_from_environ(environ: dict[str, Any]) -> str:
    """Pure host-derivation: browser host from request headers → Phoenix base URL."""
    scheme = environ.get("HTTP_X_FORWARDED_PROTO") or "http"
    hostname: str | None = None
    origin = environ.get("HTTP_ORIGIN")
    if origin:
        parsed = urlparse(origin)
        hostname = parsed.hostname or hostname
        if parsed.scheme:
            scheme = parsed.scheme
    if not hostname:
        host = environ.get("HTTP_X_FORWARDED_HOST") or environ.get("HTTP_HOST")
        if host:
            hostname = host.split(",")[0].strip().split(":")[0]
    if not hostname:
        return PHOENIX_BASE
    return f"{scheme}://{hostname}:{_PHOENIX_PORT}"


WELCOME = (
    "👋 **Welcome!** I answer questions about your documents — grounded, with "
    "citations, and I show you exactly *how grounded* each answer is.\n\n"
    "**Try asking:**\n"
    "- A specific question about your documents (an invoice total, a policy detail, a date)\n"
    "- *What can you do?* or *What kind of things can I ask?*\n\n"
    "📁 **[Open the Document Manager](/api/documents/ui)** to browse, upload, or "
    "remove the documents I can see — or just **📎 attach a file to this chat** "
    "(using the paperclip) to ask about it once, without saving it.\n\n"
    "Every answer carries a badge — 🟢 from your documents, ⚪ from general "
    "knowledge — and a clickable **🔎 How grounded is this?** panel with the "
    "sources, scores, and guard checks behind it.\n\n"
    "_Tip: the ⚙️ settings let you switch between **strict** (documents-only, "
    "fail-closed) and **assist** (general knowledge allowed). Assist is on by default._"
)


@cl_app.get("/iw/session-token")
async def iw_session_token(
    user: Annotated[Any, Depends(get_current_user)],
) -> dict[str, str | None]:
    """SSO for the Document Manager — return the logged-in user's API token.

    Authorized by the Chainlit auth cookie (``Depends(get_current_user)``).  The
    document-manager page is same-origin, so it fetches this with credentials and
    reuses the chat session instead of prompting for a second login.  Returns the
    token mapped from the authenticated username (the password is never needed).
    """
    username = getattr(user, "identifier", None)
    return {
        "token": token_for_username(username) if username else None,
        "username": username,
    }


def _prioritize_iw_routes() -> None:
    """Move our /iw/* routes ahead of Chainlit's catch-all SPA mount.

    Chainlit registers a catch-all that serves the single-page app for any
    unmatched path; because our route is added when this module is imported
    (after that catch-all exists), Starlette would match the SPA first and our
    endpoint would return HTML.  Re-ordering our routes to the front of the
    router list makes them match before the catch-all.
    """
    routes = cl_app.router.routes
    iw_routes = [r for r in routes if getattr(r, "path", "").startswith("/iw/")]
    for r in iw_routes:
        routes.remove(r)
    for r in reversed(iw_routes):
        routes.insert(0, r)


_prioritize_iw_routes()


@cl.password_auth_callback
def auth_callback(username: str, password: str) -> cl.User | None:
    """Validate demo credentials; stash the bearer token in the user's metadata."""
    token = token_for(username, password)
    if token is None:
        return None
    return cl.User(identifier=username, metadata={"token": token})


@cl.on_chat_start
async def on_start() -> None:
    """Greet the user, remember their token, and expose the assist-mode toggle."""
    user = cl.user_session.get("user")
    token = user.metadata.get("token") if user else None
    cl.user_session.set("token", token)
    # A per-chat session id scopes ephemeral attachments: files attached here are
    # visible to the agent only within this chat and purged when it ends.
    cl.user_session.set("session_id", uuid.uuid4().hex)
    cl.user_session.set("has_attachments", False)
    # Assist mode defaults ON (usable out of the box; every answer is labeled).
    cl.user_session.set("allow_ungrounded", True)
    # Resolve the Phoenix base against the browser's host once per session.
    cl.user_session.set("phoenix_base", _phoenix_base_for_session())

    await cl.ChatSettings(
        [
            Switch(
                id="allow_ungrounded",
                label="Answer from general knowledge when my documents don't cover it",
                initial=True,
                description=(
                    "ON (assist): I'll fall back to general knowledge — clearly marked "
                    "as NOT from your documents — instead of refusing. Document-grounded "
                    "answers are always preferred and cited. OFF (strict): documents-only, "
                    "fail-closed. Safety guards stay on either way."
                ),
            ),
        ]
    ).send()

    await cl.Message(content=WELCOME).send()


@cl.on_settings_update
async def on_settings(settings: dict[str, Any]) -> None:
    """Persist the assist-mode toggle for this session."""
    cl.user_session.set("allow_ungrounded", bool(settings.get("allow_ungrounded", True)))


def _render_sources(citations: list[dict[str, Any]]) -> str:
    """Compact inline source list under the answer (full detail is in the popup).

    Each citation becomes a clickable download anchor (the API returns a
    signed, short-lived ``download_url`` per citation, prefixed with the
    browser-reachable public API base so the link works across the docker
    frontend/api origin split). When ``download_url`` is empty (the API's
    fail-soft path when no token can be minted), the citation falls back to
    the original plain `` `source_uri` `` code text — no link, no surprise.
    """
    if not citations:
        return ""
    e = html.escape
    lines = ['\n\n<div class="iw-sources-title">Sources</div>\n']
    for c in citations:
        marker, uri = c.get("marker"), c.get("source_uri")
        quote = str(c.get("quote") or "").strip()
        snippet = f" — _{quote[:160]}_" if quote else ""
        download_url = str(c.get("download_url") or "").strip()
        if download_url:
            href = e(f"{_public_api_base()}{download_url}")
            uri_disp = e(str(uri))
            lines.append(
                f"- [{marker}] "
                f'<a class="iw-source-link" href="{href}" download>{uri_disp}</a>'
                f"{snippet}"
            )
        else:
            lines.append(f"- [{marker}] `{e(str(uri))}`{snippet}")
    return "\n".join(lines)


def _badge(grounded: bool, disposition: Any, general_knowledge: bool) -> str:
    """At-a-glance trust badge for the answer — a branded chip (styled in style.css)."""

    def chip(variant: str, label: str) -> str:
        return (
            f'<span class="iw-badge iw-badge--{variant}">'
            f'<span class="iw-badge__dot"></span>{label}</span>'
        )

    if disposition in ("refused", "rejected"):
        return chip("refused", "Fail-closed refusal · not enough grounded evidence")
    if grounded:
        return chip("grounded", "Grounded · answered from your documents")
    if general_knowledge:
        return chip("general", "General knowledge · not from your documents")
    return chip("unverified", "Unverified · not fully supported by your documents")


def _trace_id(result: dict[str, Any]) -> str:
    """Prefer the real OTel trace id (diagnostics.trace_ref) over the request id."""
    diagnostics = result.get("diagnostics") or {}
    trace_ref = diagnostics.get("trace_ref") or {}
    return str(trace_ref.get("trace_id") or result.get("trace_id") or "")


def _render_provenance(result: dict[str, Any], phoenix_base: str) -> str:
    """Build a collapsed ``<details>`` disclosure with the answer's provenance.

    Rendered inline at the end of the message, **collapsed by default** — the
    user clicks "🔎 How grounded is this?" to expand/collapse it. Plain HTML
    (requires ``unsafe_allow_html``) so it renders deterministically and never
    auto-opens. Dynamic values are HTML-escaped. The "Sources used" lines
    link to the citation's signed download URL (prefixed with the public API
    base so the browser can reach it); empty ``download_url`` falls back to
    the plain ``<code>source_uri</code>`` text.
    """
    diag = result.get("diagnostics") or {}
    grounded = bool(result.get("grounded"))
    citations = list(result.get("citations") or [])
    e = html.escape
    rule = '<span class="iw-prov__rule"></span>'

    rows: list[str] = []
    verdict = e(
        str(diag.get("output_guard_gen_verdict") or ("grounded" if grounded else "unsupported"))
    )
    if grounded:
        rows.append(
            f"<b>Groundedness</b> · ✅ grounded in your documents — verdict <code>{verdict}</code>"
        )
    else:
        rows.append(
            f"<b>Groundedness</b> · ⚪ not grounded (general knowledge / unsupported) — "
            f"verdict <code>{verdict}</code>"
        )

    if diag.get("output_guard_assist_mode"):
        rows.append(
            "<b>Mode</b> · assist — general knowledge allowed; groundedness is informational"
        )
    else:
        rows.append("<b>Mode</b> · strict — documents-only, fail-closed")

    rows.append(rule)
    rows.append(f"<b>Retrieval route</b> · <code>{e(str(result.get('retriever', '?')))}</code>")

    trail = diag.get("guard_trail") or []
    if trail:
        rows.append("<b>Guard checks</b>")
        for g in trail:
            kind, val = e(str(g.get("kind"))), e(str(g.get("value")))
            score = g.get("score")
            reason = g.get("reason")
            score_s = f" ({score:.2f})" if isinstance(score, (int, float)) else ""
            reason_s = f" — {e(str(reason))}" if reason else ""
            rows.append(
                f'<span class="iw-prov__src">• <code>{kind}</code>: '
                f"<b>{val}</b>{score_s}{reason_s}</span>"
            )

    if diag.get("output_guard_pii_redacted"):
        rows.append("🔒 PII was detected and redacted from this answer.")

    rows.append(rule)
    if citations:
        rows.append("<b>Sources used</b>")
        for c in citations:
            quote = str(c.get("quote") or "").strip().replace("\n", " ")
            quote = (quote[:240] + "…") if len(quote) > 240 else quote
            marker = e(str(c.get("marker")))
            uri = e(str(c.get("source_uri")))
            download_url = str(c.get("download_url") or "").strip()
            if download_url:
                href = e(f"{_public_api_base()}{download_url}")
                rows.append(
                    f'<span class="iw-prov__src">• [{marker}] '
                    f'<a class="iw-source-link" href="{href}" download>{uri}</a></span>'
                )
            else:
                rows.append(f'<span class="iw-prov__src">• [{marker}] <code>{uri}</code></span>')
            if quote:
                rows.append(f'<span class="iw-prov__quote">{e(quote)}</span>')
    else:
        rows.append("<b>Sources used</b> · none — this answer was not drawn from your documents.")

    trace_id = _trace_id(result)
    if trace_id:
        rows.append(rule)
        href = e(f"{phoenix_base}/projects")
        rows.append(
            f'<a href="{href}" target="_blank" rel="noopener">View full trace in Phoenix</a> '
            f"· <code>trace_id={e(trace_id)}</code>"
        )

    # Section rules are block elements; everything else is line-joined with <br>.
    inner = "".join(r if r.startswith('<span class="iw-prov__') else f"{r}<br>" for r in rows)
    return (
        '<details class="iw-prov">'
        "<summary>🔎 How grounded is this?</summary>"
        f'<div class="iw-prov__body">{inner}</div>'
        "</details>"
    )


def _fmt_duration(ms: float) -> str:
    """Human-friendly duration: ``820 ms`` under a second, else ``4.2 s``."""
    try:
        ms = float(ms)
    except (TypeError, ValueError):
        return "—"
    if ms < 1000:
        return f"{int(round(ms))} ms"
    return f"{ms / 1000:.1f} s"


def _render_timing(result: dict[str, Any]) -> str:
    """Build the per-answer timing badge + click-to-open breakdown popup.

    Renders an elegant ``⏱ <total>`` pill (top-right of the message) that opens
    an expandable card listing every pipeline step, in order, with its duration
    (as a proportional bar) and a plain-language description of what it did
    functionally and technically. Data comes from ``diagnostics.timeline`` +
    ``diagnostics.total_ms`` (emitted server-side by the agent loop). Returns an
    empty string when no timing is available (e.g. an error turn).
    """
    diag = result.get("diagnostics") or {}
    total_ms = diag.get("total_ms")
    timeline = diag.get("timeline") or []
    if total_ms is None and not timeline:
        return ""
    e = html.escape
    total_label = _fmt_duration(total_ms) if total_ms is not None else "—"

    # No per-step detail (e.g. loop=off path): a plain, non-interactive pill.
    if not timeline:
        return f'<span class="iw-timing-pill iw-timing-pill--flat">⏱ {e(total_label)}</span>'

    max_dur = max((float(s.get("duration_ms") or 0) for s in timeline), default=1.0) or 1.0
    rows: list[str] = []
    for step in timeline:
        label = e(str(step.get("label") or step.get("key") or "step"))
        dur = float(step.get("duration_ms") or 0)
        pct = max(2.0, min(100.0, dur / max_dur * 100.0))
        what = e(str(step.get("what") or ""))
        how = e(str(step.get("how") or ""))
        attrs = step.get("attributes") or {}
        chips = "".join(
            f'<span class="iw-tl__chip">{e(str(k))}={e(str(v))}</span>' for k, v in attrs.items()
        )
        rows.append(
            '<div class="iw-tl__step">'
            '<div class="iw-tl__row">'
            f'<span class="iw-tl__label">{label}</span>'
            f'<span class="iw-tl__dur">{e(_fmt_duration(dur))}</span>'
            "</div>"
            f'<div class="iw-tl__bar"><span style="width:{pct:.1f}%"></span></div>'
            f'<div class="iw-tl__what">{what}</div>'
            f'<div class="iw-tl__how">{how}</div>'
            + (f'<div class="iw-tl__chips">{chips}</div>' if chips else "")
            + "</div>"
        )
    n_steps = len(timeline)
    body = (
        '<div class="iw-tl">'
        f'<div class="iw-tl__head">Answer timeline · <b>{e(total_label)}</b> end-to-end · '
        f"{n_steps} step{'s' if n_steps != 1 else ''}</div>"
        f'<div class="iw-tl__steps">{"".join(rows)}</div>'
        '<div class="iw-tl__foot">Each step runs fully on-device (single GPU). '
        "Times are server-side wall-clock; repeated steps are corrective-retry passes.</div>"
        "</div>"
    )
    return (
        '<details class="iw-timing">'
        f'<summary class="iw-timing-pill">⏱ {e(total_label)}</summary>'
        f"{body}"
        "</details>"
    )


def _http_error_message(status: int) -> str:
    """A friendly, actionable message for an HTTP error status."""
    if status in (401, 403):
        return "🔒 Your session isn't authorized anymore. Please sign in again."
    if status >= 500:
        return (
            "⚠️ The answer engine hit an error (the generator may be over its context "
            "limit). Try a shorter or more specific question, or ask again."
        )
    return f"⚠️ Request failed ({status}). Please try again."


async def _render_answer(result: dict[str, Any]) -> None:
    """Render the badge + answer + sources, with the provenance popup attached."""
    answer = str(result.get("answer") or "")
    grounded = bool(result.get("grounded"))
    citations = list(result.get("citations") or [])
    diag = result.get("diagnostics") or {}
    disposition = diag.get("disposition")
    general_knowledge = bool(diag.get("general_knowledge")) or bool(diag.get("answer_ungrounded"))

    if not answer.strip():
        # Refusal or empty response — give the user something actionable.
        answer = (
            "I couldn't find an answer to that in your documents. Try rephrasing, be "
            "more specific, or turn on **general knowledge** mode in ⚙️ settings to let "
            "me answer beyond your documents."
        )

    badge = _badge(grounded, disposition, general_knowledge)

    # Per-answer timing: an ⏱ pill (top-right) that opens a step-by-step
    # breakdown of how long each pipeline stage took and what it did.
    timing = _render_timing(result)

    # Provenance is an inline, collapsed-by-default <details> disclosure appended
    # to the answer — it stays closed until the user clicks "🔎 How grounded is
    # this?", so it never auto-distracts (unlike a Chainlit side element, which
    # auto-opens in this version).
    phoenix_base = cl.user_session.get("phoenix_base") or PHOENIX_BASE
    # The timing pill is wrapped in a right-aligned bar so it sits at the
    # top-right of the message bubble, next to the copy/actions toolbar.
    header = f'<div class="iw-msg-head">{timing}</div>' if timing else ""
    body = (
        f"{header}{badge}\n\n{answer}{_render_sources(citations)}\n\n"
        f"{_render_provenance(result, phoenix_base)}"
    )
    await cl.Message(content=body).send()


def _element_bytes(element: Any) -> bytes:
    """Read an uploaded element's bytes (prefers in-memory content, else its path).

    Blocking — call via ``asyncio.to_thread`` so the event loop is never blocked.
    """
    content = getattr(element, "content", None)
    if isinstance(content, bytes):
        return content
    path = getattr(element, "path", None)
    if path:
        return Path(path).read_bytes()
    return b""


async def _handle_attachments(message: cl.Message, token: str, session_id: str) -> int:
    """Ingest any files attached to this message as ephemeral, in-chat attachments.

    Returns the number of files successfully attached.  Each becomes queryable for
    the rest of this chat only, then is purged when the chat ends.
    """
    elements = list(getattr(message, "elements", None) or [])
    files = [e for e in elements if getattr(e, "path", None) or getattr(e, "content", None)]
    if not files:
        return 0
    attached = 0
    for el in files:
        name = getattr(el, "name", None) or "attachment"
        mime = getattr(el, "mime", None)
        try:
            data = await asyncio.to_thread(_element_bytes, el)
            if not data:
                continue
            info = await attach_file(session_id, name, data, token, content_type=mime)
            attached += 1
            chunks = info.get("chunk_count", 0)
            note = (
                f"📎 Attached **{html.escape(name)}** to this chat "
                f"({chunks} section{'s' if chunks != 1 else ''} indexed). "
                "Ask me about it — it's used only here and removed when the chat ends."
            )
            if chunks == 0:
                note = (
                    f"📎 Attached **{html.escape(name)}**, but I couldn't extract any "
                    "text from it (an unlabelled image needs visual mode). "
                    "Try a text/PDF file or a scan with readable text."
                )
            await cl.Message(content=note).send()
        except httpx.HTTPStatusError as exc:
            await cl.Message(
                content=f"⚠️ Couldn't attach **{html.escape(name)}** "
                f"(error {exc.response.status_code})."
            ).send()
        except httpx.HTTPError:
            await cl.Message(
                content=f"⚠️ Couldn't reach the server to attach **{html.escape(name)}**."
            ).send()
    if attached:
        cl.user_session.set("has_attachments", True)
    return attached


@cl.on_chat_end
async def on_end() -> None:
    """Purge this chat's ephemeral attachments so nothing persists after the chat."""
    if not cl.user_session.get("has_attachments"):
        return
    token = cl.user_session.get("token")
    session_id = cl.user_session.get("session_id")
    if token and session_id:
        await purge_session(session_id, token)


@cl.on_message
async def on_message(message: cl.Message) -> None:
    """Query the API and render the answer with its trust badge + provenance."""
    token = cl.user_session.get("token")
    if not token:
        await cl.Message(content="Please sign in to ask questions.").send()
        return

    session_id = cl.user_session.get("session_id")
    attached = await _handle_attachments(message, token, session_id)

    # A bare attachment (no question) just confirms the upload — nothing to answer.
    if attached and not message.content.strip():
        return

    allow_ungrounded = bool(cl.user_session.get("allow_ungrounded", True))

    result: dict[str, Any] | None = None
    async with cl.Step(name="retrieve + generate") as step:
        try:
            result = await query(
                message.content,
                token,
                allow_ungrounded=allow_ungrounded,
                session_id=session_id,
            )
            step.output = f"retriever={result.get('retriever')} · grounded={result.get('grounded')}"
        except httpx.TimeoutException:
            step.output = "timeout"
            await cl.Message(
                content="⏳ That took too long to answer. Try a more specific question, "
                "or ask again."
            ).send()
            return
        except httpx.HTTPStatusError as exc:
            step.output = f"error {exc.response.status_code}"
            await cl.Message(content=_http_error_message(exc.response.status_code)).send()
            return
        except httpx.HTTPError as exc:
            # Connection reset / read error / etc. — never leave a blank screen.
            step.output = f"error {type(exc).__name__}"
            await cl.Message(
                content="⚠️ I couldn't reach the answer engine. Please try again in a moment."
            ).send()
            return

    await _render_answer(result)
