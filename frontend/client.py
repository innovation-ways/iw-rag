"""Thin async client for the iw-rag API (the frontend is a pure API client).

It never talks to Postgres/vLLM/infinity/Phoenix directly — only the FastAPI
``/query`` endpoint, carrying the user's bearer token (which establishes the
``AccessContext`` server-side; the frontend never sends groups).
"""

from __future__ import annotations

import os
from typing import Any

import httpx

API_BASE = os.environ.get("IW_RAG_API_BASE", "http://localhost:8000").rstrip("/")


async def query(
    question: str,
    token: str,
    *,
    allow_ungrounded: bool = True,
    session_id: str | None = None,
    timeout_s: float = 120.0,
) -> dict[str, Any]:
    """POST a question to ``/query`` and return the parsed response.

    Args:
        question: The user's question.
        token: The caller's bearer token (maps to groups server-side).
        allow_ungrounded: Assist mode. When True the server may answer from
            general knowledge (clearly marked) instead of fail-closing when the
            documents don't cover the question. The frontend toggle controls it.
        session_id: This chat's session id. When set, files attached to this chat
            (via ``/attachments``) become retrievable for this question only.
        timeout_s: Per-request timeout.

    Returns:
        The decoded ``QueryResponse`` JSON. An empty 2xx body (which previously
        crashed the caller on ``resp.json()``) is normalised into a structured
        refusal dict so the UI always has something to render.

    Raises:
        httpx.HTTPStatusError: On a non-2xx response.
        httpx.TimeoutException: When the request exceeds ``timeout_s``.
    """
    body: dict[str, Any] = {"query": question, "allow_ungrounded": allow_ungrounded}
    if session_id:
        body["session_id"] = session_id
    async with httpx.AsyncClient(timeout=timeout_s) as client:
        resp = await client.post(
            f"{API_BASE}/query",
            json=body,
            headers={"authorization": f"Bearer {token}"},
        )
        resp.raise_for_status()
        if not resp.content:
            # 2xx with an empty body — never crash on json(); return a refusal.
            return {
                "answer": "",
                "grounded": False,
                "diagnostics": {"disposition": "refused", "reason": "empty_response"},
            }
        try:
            data: dict[str, Any] = resp.json()
        except ValueError:
            return {
                "answer": "",
                "grounded": False,
                "diagnostics": {"disposition": "refused", "reason": "invalid_json"},
            }
        return data


async def attach_file(
    session_id: str,
    filename: str,
    content: bytes,
    token: str,
    *,
    content_type: str | None = None,
    timeout_s: float = 120.0,
) -> dict[str, Any]:
    """Attach a file to this chat session (queryable in-chat; purged on chat end).

    Args:
        session_id: This chat's session id (scopes the attachment's visibility).
        filename: The original filename.
        content: The raw file bytes.
        token: The caller's bearer token.
        content_type: The file's MIME type, if known.
        timeout_s: Per-request timeout.

    Returns:
        The decoded ``AttachmentOut`` JSON (``document_id``, ``chunk_count``, …).

    Raises:
        httpx.HTTPStatusError: On a non-2xx response.
    """
    files = {"file": (filename, content, content_type or "application/octet-stream")}
    async with httpx.AsyncClient(timeout=timeout_s) as client:
        resp = await client.post(
            f"{API_BASE}/attachments",
            data={"session_id": session_id},
            files=files,
            headers={"authorization": f"Bearer {token}"},
        )
        resp.raise_for_status()
        data: dict[str, Any] = resp.json()
        return data


async def purge_session(session_id: str, token: str, *, timeout_s: float = 30.0) -> None:
    """Delete all of this chat's ephemeral attachments (best-effort, on chat end)."""
    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            await client.delete(
                f"{API_BASE}/attachments/{session_id}",
                headers={"authorization": f"Bearer {token}"},
            )
    except httpx.HTTPError:
        # Chat teardown must never raise; the TTL sweep is the backstop.
        return


def parse_sse_lines(lines: list[str]) -> list[dict[str, Any]]:
    """Parse raw SSE text lines into a list of ``{"event", "data"}`` dicts.

    Comment lines (starting with ``:`` — e.g. ``": ping"`` heartbeats) are
    skipped. An event is flushed on a blank line. ``data:`` payloads are
    JSON-decoded.

    Args:
        lines: Raw SSE lines (without trailing newlines).

    Returns:
        The decoded events in order.
    """
    import json

    events: list[dict[str, Any]] = []
    event_type: str | None = None
    data_buf: list[str] = []

    def flush() -> None:
        nonlocal event_type, data_buf
        if event_type is not None:
            raw = "\n".join(data_buf)
            try:
                data = json.loads(raw) if raw else {}
            except ValueError:
                data = {"raw": raw}
            events.append({"event": event_type, "data": data})
        event_type = None
        data_buf = []

    for line in lines:
        if line.startswith(":"):
            continue  # comment/heartbeat
        if line == "":
            flush()
            continue
        if line.startswith("event:"):
            event_type = line[len("event:") :].strip()
        elif line.startswith("data:"):
            data_buf.append(line[len("data:") :].strip())
    flush()  # flush any trailing event without a terminating blank line
    return events


async def stream_query(question: str, token: str, *, timeout_s: float = 180.0) -> Any:
    """Async-generate decoded SSE events from ``/query/stream``.

    Args:
        question: The user's question.
        token: The caller's bearer token (groups resolved server-side).
        timeout_s: Per-request timeout.

    Yields:
        Decoded ``{"event", "data"}`` dicts as they arrive.
    """
    async with (
        httpx.AsyncClient(timeout=timeout_s) as client,
        client.stream(
            "POST",
            f"{API_BASE}/query/stream",
            json={"query": question},
            headers={"authorization": f"Bearer {token}"},
        ) as resp,
    ):
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            for event in parse_sse_lines([line, ""]):
                yield event
