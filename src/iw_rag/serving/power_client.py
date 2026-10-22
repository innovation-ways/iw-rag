"""Typed async client the main API uses to proxy to the system-control sidecar.

The main API holds NO :class:`~iw_rag.serving.power_controller.PowerController`
and never touches the Docker socket — it proxies to the sidecar (``:8091``)
over HTTP via this client.  This is the **only seam** between the unprivileged
public API and the privileged power authority.

**Connection-error contract.**  Every method converts raw ``httpx`` errors
(timeouts, connection refused, DNS failure) and non-2xx responses into the
typed :class:`PowerControlUnavailable` so the API maps an unreachable sidecar
to a single error class.  Mirrors
:class:`~iw_rag.serving.swap_controller_client.SwapControllerClient`.
"""

from __future__ import annotations

from typing import Any

import httpx


class PowerControlUnavailable(Exception):
    """Raised when the system-control sidecar is unreachable or errors.

    The API maps this to a 503 (status routes) or degrades the System view to
    "power control unavailable" — it never 5xx-crashes the console on a
    transient sidecar outage.
    """


class PowerClient:
    """Typed async client for the system-control sidecar (``:8091``).

    Attributes:
        base_url: The sidecar base URL (internal Compose network only).
        timeout_s: Per-request timeout (large: a power-on blocks for the full
            cold-start of the generator).
    """

    def __init__(self, base_url: str, timeout_s: float = 600.0) -> None:
        """Configure the client.

        Args:
            base_url: The sidecar base URL (e.g. ``http://system-control:8091``).
            timeout_s: Per-request timeout.  Defaults high because
                ``POST /power/on`` is held open for the full readiness wait
                (a real cold vLLM load can take minutes).
        """
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s

    async def get_status(self) -> dict[str, Any]:
        """Return the full power snapshot (proxy for ``GET /power/status``)."""
        return await self._get("/power/status")

    async def get_vram(self) -> dict[str, Any]:
        """Return the VRAM snapshot (proxy for ``GET /power/vram``)."""
        return await self._get("/power/vram")

    async def power_on(self, components: list[str] | None = None) -> dict[str, Any]:
        """Trigger ``POST /power/on`` (block until ready).

        Args:
            components: Component keys to start, or ``None`` for all.

        Returns:
            The post-action status payload.

        Raises:
            PowerControlUnavailable: If the sidecar is unreachable or errors.
        """
        return await self._post("/power/on", {"components": components})

    async def power_off(self, components: list[str] | None = None) -> dict[str, Any]:
        """Trigger ``POST /power/off`` (frees VRAM).

        Args:
            components: Component keys to stop, or ``None`` for all.

        Returns:
            The post-action status payload.

        Raises:
            PowerControlUnavailable: If the sidecar is unreachable or errors.
        """
        return await self._post("/power/off", {"components": components})

    async def orchestrate(self, *, stop: list[str], start: list[str]) -> dict[str, Any]:
        """Trigger ``POST /power/orchestrate`` (container half of a mode switch).

        Args:
            stop: Container names to stop first.
            start: Container names to start + wait to ready.

        Returns:
            The post-switch status payload.

        Raises:
            PowerControlUnavailable: If the sidecar is unreachable or errors.
        """
        return await self._post("/power/orchestrate", {"stop": stop, "start": start})

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _get(self, path: str) -> dict[str, Any]:
        url = f"{self._base_url}{path}"
        try:
            async with httpx.AsyncClient(timeout=self._timeout_s) as client:
                resp = await client.get(url)
        except httpx.HTTPError as exc:
            raise PowerControlUnavailable(f"system-control unreachable at {url}: {exc}") from exc
        if resp.status_code >= 400:
            raise PowerControlUnavailable(
                f"system-control GET {path} returned HTTP {resp.status_code}: {resp.text[:200]}"
            )
        return resp.json()  # type: ignore[no-any-return]

    async def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        url = f"{self._base_url}{path}"
        try:
            async with httpx.AsyncClient(timeout=self._timeout_s) as client:
                resp = await client.post(url, json=body)
        except httpx.HTTPError as exc:
            raise PowerControlUnavailable(f"system-control unreachable at {url}: {exc}") from exc
        if resp.status_code >= 400:
            raise PowerControlUnavailable(
                f"system-control POST {path} returned HTTP {resp.status_code}: {resp.text[:200]}"
            )
        return resp.json()  # type: ignore[no-any-return]
