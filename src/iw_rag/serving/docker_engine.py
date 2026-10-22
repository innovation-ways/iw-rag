"""Minimal async Docker Engine API client over the Unix socket (power control).

The ``system-control`` sidecar (the only process that mounts the host Docker
socket) uses this client to stop/start the VRAM-holding containers and read
their state.  It speaks the Docker Engine HTTP API directly over the Unix
socket via ``httpx`` — no ``docker`` CLI in the image, no ``docker`` SDK
dependency.  Three calls cover the whole surface:

* ``GET  /containers/{name}/json``  → :meth:`HttpDockerEngineClient.inspect`
* ``POST /containers/{name}/start`` → :meth:`HttpDockerEngineClient.start`
* ``POST /containers/{name}/stop``  → :meth:`HttpDockerEngineClient.stop`

**Error contract.**  Connection failures (socket missing, daemon down) raise
:class:`DockerUnavailable`.  A ``404`` (no such container) is NOT an error: it
maps to an ``absent`` :class:`ContainerState` for ``inspect`` and is a no-op
for ``start``/``stop`` — so the same component config works whether or not the
optional guard containers are present.

**Privilege note.**  Holding the Docker socket is host-root-equivalent.  This
client lives ONLY in the internal-only sidecar (never published to the host);
the main API never imports it.  Mirrors the llama-swap container's socket
mount (config/llama-swap.yaml) — the established privileged-sidecar pattern.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import httpx


class DockerUnavailable(Exception):
    """Raised when the Docker daemon cannot be reached over the socket.

    Connection refused, socket missing, timeout, or a 5xx from the daemon —
    every transport-level failure converges here so callers map an
    unreachable daemon to one error class (the sidecar maps it to HTTP 503).
    A ``404`` (no such container) is NOT this error — it is a normal "absent"
    result.
    """


@dataclass(frozen=True, slots=True)
class ContainerState:
    """A snapshot of one container's lifecycle state.

    Attributes:
        name: The container name queried.
        exists: ``False`` when the daemon returned 404 (no such container).
        running: ``True`` when ``State.Running`` is true.
        status: The raw ``State.Status`` string (``"running"``, ``"exited"``,
            ``"created"``, …); ``"absent"`` when the container does not exist.
        health: ``State.Health.Status`` (``"healthy"``, ``"unhealthy"``,
            ``"starting"``) or ``None`` when the container has no healthcheck
            (most do not) or does not exist.
    """

    name: str
    exists: bool
    running: bool
    status: str
    health: str | None


class DockerEngineClient(Protocol):
    """Structural type for the Docker client the power controller depends on.

    Defined so tests can pass a fake that quacks like
    :class:`HttpDockerEngineClient` without a real daemon.
    """

    async def inspect(self, name: str) -> ContainerState: ...
    async def start(self, name: str) -> bool: ...
    async def stop(self, name: str, *, timeout_s: int = 10) -> bool: ...
    async def list_matching(self, prefix: str) -> list[str]: ...


class HttpDockerEngineClient:
    """Async Docker Engine API client bound to a Unix socket.

    Each call opens a short-lived ``httpx.AsyncClient`` over an
    ``AsyncHTTPTransport(uds=...)`` (mirrors the per-request client style of
    :class:`~iw_rag.serving.llama_swap_client.LlamaSwapClient`).  Paths are
    unversioned — the daemon routes them to its current API version.

    Attributes:
        socket_path: Path to the Docker Unix socket (mounted into the sidecar).
        timeout_s: Per-request timeout in seconds.
    """

    # Dummy host — the request is routed to the Unix socket by the transport,
    # so the authority part of the URL is irrelevant but must be well-formed.
    _BASE_URL = "http://docker"

    def __init__(self, socket_path: str = "/var/run/docker.sock", timeout_s: float = 60.0) -> None:
        """Configure the client.

        Args:
            socket_path: Path to the Docker Unix socket.
            timeout_s: Per-request timeout (a ``stop`` waits up to the
                container's own stop-grace before SIGKILL, so keep this
                comfortably above ``timeout_s`` passed to :meth:`stop`).
        """
        self._socket_path = socket_path
        self._timeout_s = timeout_s

    def _client(self) -> httpx.AsyncClient:
        """Build a fresh AsyncClient bound to the Docker Unix socket."""
        transport = httpx.AsyncHTTPTransport(uds=self._socket_path)
        return httpx.AsyncClient(
            transport=transport, base_url=self._BASE_URL, timeout=self._timeout_s
        )

    async def inspect(self, name: str) -> ContainerState:
        """Return the container's lifecycle state (``absent`` on 404).

        Args:
            name: The container name (e.g. ``"iw-rag-vllm-1"``).

        Returns:
            A :class:`ContainerState`.  ``exists=False`` when the daemon
            returns 404.

        Raises:
            DockerUnavailable: On any transport failure or a 5xx response.
        """
        url = f"/containers/{name}/json"
        try:
            async with self._client() as client:
                resp = await client.get(url)
        except httpx.HTTPError as exc:
            raise DockerUnavailable(
                f"docker daemon unreachable at {self._socket_path}: {exc}"
            ) from exc
        if resp.status_code == 404:
            return ContainerState(
                name=name, exists=False, running=False, status="absent", health=None
            )
        if resp.status_code >= 500:
            raise DockerUnavailable(
                f"docker GET {url} returned HTTP {resp.status_code}: {resp.text[:200]}"
            )
        return _parse_inspect(name, resp.json())

    async def start(self, name: str) -> bool:
        """Start the container.  204 → started; 304 → already running.

        Args:
            name: The container name.

        Returns:
            ``True`` if a start was issued (204) or the container was already
            running (304); ``False`` if the container does not exist (404).

        Raises:
            DockerUnavailable: On any transport failure or a 5xx response.
        """
        return await self._post_lifecycle(f"/containers/{name}/start")

    async def stop(self, name: str, *, timeout_s: int = 10) -> bool:
        """Stop the container (graceful, ``timeout_s`` before SIGKILL).

        Args:
            name: The container name.
            timeout_s: Seconds the daemon waits after SIGTERM before SIGKILL.

        Returns:
            ``True`` if a stop was issued (204) or the container was already
            stopped (304); ``False`` if the container does not exist (404).

        Raises:
            DockerUnavailable: On any transport failure or a 5xx response.
        """
        return await self._post_lifecycle(f"/containers/{name}/stop", params={"t": timeout_s})

    async def list_matching(self, prefix: str) -> list[str]:
        """Return the names of the **running** containers whose name starts with ``prefix``.

        Used to resolve a dynamic container set — e.g. the benchmark-mode
        generator, which llama-swap spawns as ``vllm-<catalog-id>`` (a name the
        power config can't know ahead of time).  Running-only: a powered-off
        match holds no VRAM, so it is irrelevant to the gauge + the power
        controls (and there is nothing for the operator to stop).

        Args:
            prefix: The container-name prefix to match (e.g. ``"vllm-"``).

        Returns:
            Matching running container names, sorted; ``[]`` when none match.

        Raises:
            DockerUnavailable: On any transport failure or a 5xx response.
        """
        import json as _json

        url = "/containers/json"
        # No ``all`` flag → the daemon returns running containers only.
        params = {"filters": _json.dumps({"name": [prefix]})}
        try:
            async with self._client() as client:
                resp = await client.get(url, params=params)
        except httpx.HTTPError as exc:
            raise DockerUnavailable(
                f"docker daemon unreachable at {self._socket_path}: {exc}"
            ) from exc
        if resp.status_code >= 400:
            raise DockerUnavailable(
                f"docker GET {url} returned HTTP {resp.status_code}: {resp.text[:200]}"
            )
        out: list[str] = []
        for entry in resp.json():
            for raw in entry.get("Names", []) if isinstance(entry, dict) else []:
                name = raw.lstrip("/")
                # Docker's name filter is a substring match; keep only true
                # prefix matches so "vllm-" does not pull in "my-vllm-x".
                if name.startswith(prefix):
                    out.append(name)
        return sorted(set(out))

    async def _post_lifecycle(self, url: str, params: dict[str, Any] | None = None) -> bool:
        """POST a start/stop lifecycle call; map 204/304/404 → bool."""
        try:
            async with self._client() as client:
                resp = await client.post(url, params=params or {})
        except httpx.HTTPError as exc:
            raise DockerUnavailable(
                f"docker daemon unreachable at {self._socket_path}: {exc}"
            ) from exc
        if resp.status_code in (204, 304):
            return True
        if resp.status_code == 404:
            return False
        raise DockerUnavailable(
            f"docker POST {url} returned HTTP {resp.status_code}: {resp.text[:200]}"
        )


def _parse_inspect(name: str, payload: dict[str, Any]) -> ContainerState:
    """Map a Docker inspect JSON body to a :class:`ContainerState`."""
    state = payload.get("State", {}) if isinstance(payload, dict) else {}
    running = bool(state.get("Running", False))
    status = str(state.get("Status", "unknown"))
    health_obj = state.get("Health") if isinstance(state, dict) else None
    health = None
    if isinstance(health_obj, dict):
        health = str(health_obj.get("Status")) if health_obj.get("Status") is not None else None
    return ContainerState(name=name, exists=True, running=running, status=status, health=health)
