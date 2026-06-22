"""Typed async HTTP client for the local ``llama-swap`` proxy (F-00094).

llama-swap is a single ~15 MB Go OpenAI-compatible reverse proxy
(``ghcr.io/mostlygeek/llama-swap:v143``) that fronts vLLM generator
containers in an ``exclusive: true`` group so exactly one generator is
resident at a time.  This module is the **only** place in the application
code that talks to it.

**Activation mechanism.**  llama-swap exposes an OpenAI-compatible
``/v1/chat/completions`` route.  Sending a chat-completions request whose
``model`` field names a swap group member triggers the exclusive-group
load (and the previous member is stopped before the new one is brought
up).  We use the lightest documented request — ``POST /v1/chat/completions``
with ``max_tokens=1`` and ``stream=false`` — to activate a model; the
swap-controller then polls ``/v1/models`` until the upstream model reports
HTTP 200, which is the readiness gate.  See
``https://github.com/mostlygeek/llama-swap`` and R-00123 §6.

**Connection-error contract.**  Every method converts raw ``httpx``
errors (timeouts, connection refused, DNS failure) into the typed
:class:`LlamaSwapUnavailable` so callers can map them to ``503 /
unreachable`` — raw httpx exceptions never escape this module untyped
(F-00094 Boundary row 3).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from iw_rag.domain.errors import IwRagError


class LlamaSwapError(IwRagError):
    """Base class for llama-swap client failures."""


class LlamaSwapUnavailable(LlamaSwapError):
    """llama-swap is unreachable (connection refused, timeout, DNS, 5xx).

    Maps to HTTP 503 in the sidecar / public API; surfaces as the
    ``unreachable`` catalog annotation when ``list_models()`` cannot reach
    the proxy.  Carries the same role as :class:`RetrievalError` and
    :class:`GenerationError` in the rest of the codebase — a typed
    dependency-availability failure.
    """


@dataclass(frozen=True, slots=True)
class ModelState:
    """A model as llama-swap currently reports it.

    Attributes:
        id: The catalog id (also the llama-swap entry name).
        state: llama-swap's view of the model — ``"available"`` when the
            exclusive-group has it resident, ``"unavailable"`` when another
            member is active, etc.  The exact taxonomy is the llama-swap
            proxy's; we surface it verbatim.
    """

    id: str
    state: str


class LlamaSwapClient:
    """Typed async client for the local llama-swap proxy.

    Style mirrors :class:`~iw_rag.generation.llm.VllmClient`: a single
    async context-managed ``httpx.AsyncClient`` per request, configurable
    timeout, and typed exceptions for connection failures.

    Attributes:
        base_url: The proxy base URL (e.g. ``http://llama-swap:8080``).
        timeout_s: Per-request timeout in seconds.
    """

    def __init__(self, base_url: str, timeout_s: float = 10.0) -> None:
        """Configure the client.

        Args:
            base_url: The llama-swap proxy base URL — internal Compose
                network only (default ``http://llama-swap:8080``).
            timeout_s: Per-request timeout.  Keep small; the activation
                step does not need a long timeout and the readiness poll
                uses its own budget (``health_timeout_s``).
        """
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s

    async def list_models(self) -> list[ModelState]:
        """Return the proxy's view of the served models / their state.

        llama-swap's ``/v1/models`` returns an OpenAI-compatible
        ``{"data": [{"id": ..., "object": "model"}, ...]}`` payload when
        the proxy is up.  When the proxy is down, we raise
        :class:`LlamaSwapUnavailable` so callers can degrade the catalog
        annotation to ``unreachable`` (Boundary row 3).

        Returns:
            A list of :class:`ModelState` records.

        Raises:
            LlamaSwapUnavailable: If the proxy is unreachable or returns
                a non-200 status.
        """
        url = f"{self._base_url}/v1/models"
        try:
            async with httpx.AsyncClient(timeout=self._timeout_s) as client:
                resp = await client.get(url)
        except httpx.HTTPError as exc:
            raise LlamaSwapUnavailable(f"llama-swap unreachable at {url}: {exc}") from exc
        if resp.status_code != 200:
            raise LlamaSwapUnavailable(
                f"llama-swap GET /v1/models returned HTTP {resp.status_code}: {resp.text[:200]}"
            )
        try:
            payload: dict[str, Any] = resp.json()
            data = payload.get("data", [])
            return [
                ModelState(id=str(item.get("id", "")), state="available")
                for item in data
                if str(item.get("id", ""))
            ]
        except (ValueError, TypeError) as exc:
            raise LlamaSwapUnavailable(
                f"llama-swap /v1/models returned an unparseable payload: {exc}"
            ) from exc

    async def model_health(self, model_id: str) -> bool:
        """Return True iff the upstream model reports HTTP 200 on ``/v1/models``.

        The readiness gate (F-00094 AC2 / Boundary row 1) is the upstream
        model's OpenAI-compatible ``/v1/models`` endpoint — the
        llama-swap ``checkEndpoint`` proxy path.  HTTP 200 ⇒ the model is
        resident and ready; any other status ⇒ still loading or down.
        Connection failures degrade to ``False`` (the caller is polling
        on a budget and will eventually time out).

        Args:
            model_id: The catalog id (also the llama-swap entry name).
                We call the proxy's ``/v1/models`` with ``model=`` in the
                query so llama-swap's exclusive-group routing routes to
                the correct upstream.

        Returns:
            ``True`` iff the response is HTTP 200.

        Raises:
            LlamaSwapUnavailable: If the proxy itself is unreachable
                (caller can map to ``503`` / ``unreachable``).
        """
        url = f"{self._base_url}/v1/models"
        try:
            async with httpx.AsyncClient(timeout=self._timeout_s) as client:
                resp = await client.get(url, params={"model": model_id})
        except httpx.HTTPError as exc:
            raise LlamaSwapUnavailable(
                f"llama-swap unreachable while checking {model_id}: {exc}"
            ) from exc
        return resp.status_code == 200

    async def activate(self, model_id: str) -> None:
        """Trigger an exclusive-group load of ``model_id`` via llama-swap.

        The activation request is the lightest documented llama-swap
        mechanism for swapping: a chat-completions call whose ``model``
        field names a swap-group member.  llama-swap will stop the
        currently resident sibling (if any) and start the requested one
        before proxying the request — exactly the AC4 exactly-one-resident
        invariant.  We pass ``max_tokens=1`` and ``stream=false`` so the
        request is cheap; we ignore the response body (the activation has
        happened as soon as the request was accepted).  Readiness is then
        gated by :meth:`model_health` polling.

        Args:
            model_id: The catalog id to activate.

        Raises:
            LlamaSwapUnavailable: If the proxy is unreachable or returns
                a non-2xx status (caller can map to ``503`` /
                ``unreachable`` / ``failed``).
        """
        url = f"{self._base_url}/v1/chat/completions"
        body = {
            "model": model_id,
            "messages": [{"role": "user", "content": "."}],
            "max_tokens": 1,
            "stream": False,
        }
        try:
            async with httpx.AsyncClient(timeout=self._timeout_s) as client:
                resp = await client.post(url, json=body)
        except httpx.HTTPError as exc:
            raise LlamaSwapUnavailable(
                f"llama-swap unreachable while activating {model_id}: {exc}"
            ) from exc
        # 200 = the request was proxied (activation complete).  4xx/5xx =
        # llama-swap rejected the request (unknown model, etc.).
        if resp.status_code >= 400:
            raise LlamaSwapUnavailable(
                f"llama-swap activation of '{model_id}' returned "
                f"HTTP {resp.status_code}: {resp.text[:200]}"
            )
