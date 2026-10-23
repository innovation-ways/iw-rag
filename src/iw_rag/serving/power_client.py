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


class PowerCapacityError(Exception):
    """Raised when the sidecar refuses an action because it would not fit in VRAM.

    The sidecar answers HTTP 409 with a structured ``insufficient_vram`` body
    (needed / free / projected / margin / shortfall in MiB); the API maps this
    to its own 409 and the System view shows the shortfall.

    Attributes:
        body: The parsed ``insufficient_vram`` response body.
    """

    def __init__(self, body: dict[str, Any]) -> None:
        self.body = body
        super().__init__(
            f"insufficient VRAM: short by {body.get('shortfall_mb', '?')} MiB "
            f"(need {body.get('needed_mb', '?')} MiB, free {body.get('free_mb', '?')} MiB)"
        )


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

    async def get_history(self, window_s: float | None = None) -> dict[str, Any]:
        """Return the VRAM time series (proxy for ``GET /power/history``).

        Args:
            window_s: Look-back window in seconds, or ``None`` for everything kept.

        Returns:
            ``{"samples": [...], "interval_s": float}``.

        Raises:
            PowerControlUnavailable: If the sidecar is unreachable or errors.
        """
        query = "" if window_s is None else f"?window_s={window_s}"
        return await self._get(f"/power/history{query}")

    async def get_capacity(self, components: list[str] | None = None) -> dict[str, Any]:
        """Dry-run the capacity pre-check (proxy for ``GET /power/capacity``).

        Args:
            components: Component keys that would be started, or ``None`` for all.

        Returns:
            The capacity verdict as a dict (``ok``, ``needed_mb``, ``shortfall_mb`` ...).

        Raises:
            PowerControlUnavailable: If the sidecar is unreachable or errors.
        """
        query = "" if components is None else "?components=" + ",".join(components)
        return await self._get(f"/power/capacity{query}")

    async def apply_profile(self, name: str, *, force: bool = False) -> dict[str, Any]:
        """Apply an operating profile (``POST /power/profile``).

        Args:
            name: Profile key.
            force: Operator override — bypass the capacity pre-check.

        Returns:
            The post-action status payload.

        Raises:
            PowerCapacityError: If the profile would not fit in free VRAM.
            PowerControlUnavailable: If the sidecar is unreachable or errors.
        """
        body: dict[str, Any] = {"name": name}
        if force:
            body["force"] = True
        return await self._post("/power/profile", body)

    async def power_on(
        self, components: list[str] | None = None, *, force: bool = False
    ) -> dict[str, Any]:
        """Trigger ``POST /power/on`` (block until ready).

        Args:
            components: Component keys to start, or ``None`` for all.
            force: Operator override — bypass the capacity pre-check.

        Returns:
            The post-action status payload.

        Raises:
            PowerCapacityError: If the start would not fit in free VRAM.
            PowerControlUnavailable: If the sidecar is unreachable or errors.
        """
        body: dict[str, Any] = {"components": components}
        if force:
            body["force"] = True
        return await self._post("/power/on", body)

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

    async def orchestrate(
        self, *, stop: list[str], start: list[str], force: bool = False
    ) -> dict[str, Any]:
        """Trigger ``POST /power/orchestrate`` (container half of a mode switch).

        Args:
            stop: Container names to stop first.
            start: Container names to start + wait to ready.
            force: Operator override — bypass the capacity pre-check.

        Returns:
            The post-switch status payload.

        Raises:
            PowerCapacityError: If the incoming stack would not fit in free VRAM.
            PowerControlUnavailable: If the sidecar is unreachable or errors.
        """
        body: dict[str, Any] = {"stop": stop, "start": start}
        if force:
            body["force"] = True
        return await self._post("/power/orchestrate", body)

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
        if resp.status_code == 409:
            raise PowerCapacityError(resp.json())
        if resp.status_code >= 400:
            raise PowerControlUnavailable(
                f"system-control POST {path} returned HTTP {resp.status_code}: {resp.text[:200]}"
            )
        return resp.json()  # type: ignore[no-any-return]
