"""Typed async HTTP client the public API uses to proxy to the swap-controller sidecar (F-00094).

The main API process holds NO in-process :class:`~.swap_controller.SwapController` —
it proxies to the sidecar (``:8090``) over HTTP via this client.  This is the
**only seam** between the public API and the swap authority; the sidecar is
internal-only and unauthenticated (the public API enforces the
``eval-operator`` ACL gate *before* proxying).

**Connection-error contract.**  Every method converts raw ``httpx``
errors (timeouts, connection refused, DNS failure) and non-2xx HTTP
responses into the same typed :class:`~.llama_swap_client.LlamaSwapUnavailable`
the llama-swap client raises — so S03 can map an unreachable sidecar to
``503 / unreachable`` with one error class.  Raw httpx exceptions never
escape this module untyped.
"""

from __future__ import annotations

from typing import Any

import httpx

from .llama_swap_client import LlamaSwapUnavailable


class SwapControllerClient:
    """Typed async client for the swap-controller sidecar (``:8090``).

    Style mirrors :class:`~.llama_swap_client.LlamaSwapClient`: a single
    async context-managed ``httpx.AsyncClient`` per request, configurable
    timeout, and :class:`LlamaSwapUnavailable` for every connection
    failure.

    Attributes:
        base_url: The sidecar base URL (e.g. ``http://swap-controller:8090``).
        timeout_s: Per-request timeout in seconds.
    """

    def __init__(self, base_url: str, timeout_s: float = 30.0) -> None:
        """Configure the client.

        Args:
            base_url: The swap-controller sidecar base URL — internal
                Compose network only (default ``http://swap-controller:8090``).
            timeout_s: Per-request timeout.  Larger than the llama-swap
                client because the sidecar may be mid-activation (which
                can take up to ``health_timeout_s`` to fail).
        """
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s

    async def get_catalog(self) -> dict[str, Any]:
        """Return the catalog annotated with state (proxy for ``GET /catalog``).

        Returns:
            A JSON-serialisable dict mirroring the sidecar's response
            (``{"models": [...]}``).

        Raises:
            LlamaSwapUnavailable: If the sidecar is unreachable or
                returns a non-2xx status.
        """
        url = f"{self._base_url}/catalog"
        try:
            async with httpx.AsyncClient(timeout=self._timeout_s) as client:
                resp = await client.get(url)
        except httpx.HTTPError as exc:
            raise LlamaSwapUnavailable(f"swap-controller unreachable at {url}: {exc}") from exc
        if resp.status_code >= 400:
            raise LlamaSwapUnavailable(
                f"swap-controller GET /catalog returned HTTP {resp.status_code}: {resp.text[:200]}"
            )
        return resp.json()  # type: ignore[no-any-return]

    async def activate(self, model_id: str) -> dict[str, Any]:
        """Trigger ``POST /catalog/activate`` with ``{"model_id": ...}``.

        Args:
            model_id: The catalog id to activate.

        Returns:
            The sidecar's status payload (post-activation state).

        Raises:
            LlamaSwapUnavailable: If the sidecar is unreachable or
                returns a non-2xx status (caller maps to ``503``).
        """
        url = f"{self._base_url}/catalog/activate"
        try:
            async with httpx.AsyncClient(timeout=self._timeout_s) as client:
                resp = await client.post(url, json={"model_id": model_id})
        except httpx.HTTPError as exc:
            raise LlamaSwapUnavailable(f"swap-controller unreachable at {url}: {exc}") from exc
        if resp.status_code >= 400:
            raise LlamaSwapUnavailable(
                f"swap-controller POST /catalog/activate returned HTTP "
                f"{resp.status_code}: {resp.text[:200]}"
            )
        return resp.json()  # type: ignore[no-any-return]

    async def get_status(self) -> dict[str, Any]:
        """Return the swap-controller's current status (``GET /catalog/status``).

        Returns:
            The sidecar's status payload.

        Raises:
            LlamaSwapUnavailable: If the sidecar is unreachable.
        """
        url = f"{self._base_url}/catalog/status"
        try:
            async with httpx.AsyncClient(timeout=self._timeout_s) as client:
                resp = await client.get(url)
        except httpx.HTTPError as exc:
            raise LlamaSwapUnavailable(f"swap-controller unreachable at {url}: {exc}") from exc
        if resp.status_code >= 400:
            raise LlamaSwapUnavailable(
                f"swap-controller GET /catalog/status returned HTTP "
                f"{resp.status_code}: {resp.text[:200]}"
            )
        return resp.json()  # type: ignore[no-any-return]
