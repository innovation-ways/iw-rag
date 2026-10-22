"""Unit tests for the power-control Docker Engine client (Unix-socket HTTP).

All ``httpx`` calls are MOCKED — no real Docker daemon, no socket.  The client
speaks the Docker Engine API over the Unix socket; these tests verify the
status-code → result mapping and the connection-error → :class:`DockerUnavailable`
contract.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from iw_rag.serving.docker_engine import (
    ContainerState,
    DockerUnavailable,
    HttpDockerEngineClient,
    _parse_inspect,
)


def _client() -> HttpDockerEngineClient:
    return HttpDockerEngineClient(socket_path="/var/run/docker.sock", timeout_s=1.0)


def _mock_response(status_code: int, payload: Any = None, text: str = "") -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text
    resp.json = MagicMock(return_value=payload if payload is not None else {})
    return resp


def _mock_http(resp: MagicMock, *, method: str = "get") -> MagicMock:
    mock = MagicMock()
    mock.__aenter__ = AsyncMock(return_value=mock)
    mock.__aexit__ = AsyncMock(return_value=None)
    setattr(mock, method, AsyncMock(return_value=resp))
    return mock


# ---------------------------------------------------------------------------
# inspect()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_inspect_running_healthy_parses_state() -> None:
    """A 200 inspect with a healthy running container maps to a running state."""
    payload = {"State": {"Running": True, "Status": "running", "Health": {"Status": "healthy"}}}
    http = _mock_http(_mock_response(200, payload), method="get")
    with patch("httpx.AsyncClient", return_value=http):
        st = await _client().inspect("iw-rag-vllm-1")
    assert st.exists is True
    assert st.running is True
    assert st.status == "running"
    assert st.health == "healthy"
    assert st.name == "iw-rag-vllm-1"


@pytest.mark.asyncio
async def test_inspect_stopped_no_healthcheck() -> None:
    """An exited container with no healthcheck reports running=False, health=None."""
    payload = {"State": {"Running": False, "Status": "exited"}}
    http = _mock_http(_mock_response(200, payload), method="get")
    with patch("httpx.AsyncClient", return_value=http):
        st = await _client().inspect("iw-rag-infinity-1")
    assert st.running is False
    assert st.status == "exited"
    assert st.health is None
    assert st.exists is True


@pytest.mark.asyncio
async def test_inspect_404_is_absent_not_error() -> None:
    """A 404 (no such container) maps to an ``absent`` state, NOT an exception."""
    http = _mock_http(_mock_response(404, text="no such container"), method="get")
    with patch("httpx.AsyncClient", return_value=http):
        st = await _client().inspect("injguard")
    assert st.exists is False
    assert st.running is False
    assert st.status == "absent"
    assert st.health is None


@pytest.mark.asyncio
async def test_inspect_500_raises_docker_unavailable() -> None:
    """A 5xx from the daemon raises ``DockerUnavailable``."""
    http = _mock_http(_mock_response(500, text="boom"), method="get")
    with (
        patch("httpx.AsyncClient", return_value=http),
        pytest.raises(DockerUnavailable, match="HTTP 500"),
    ):
        await _client().inspect("iw-rag-vllm-1")


@pytest.mark.asyncio
async def test_inspect_connection_error_raises_docker_unavailable() -> None:
    """A transport error (socket missing) raises ``DockerUnavailable``."""
    http = MagicMock()
    http.__aenter__ = AsyncMock(return_value=http)
    http.__aexit__ = AsyncMock(return_value=None)
    http.get = AsyncMock(side_effect=httpx.ConnectError("no socket"))
    with (
        patch("httpx.AsyncClient", return_value=http),
        pytest.raises(DockerUnavailable, match="unreachable"),
    ):
        await _client().inspect("iw-rag-vllm-1")


# ---------------------------------------------------------------------------
# start() / stop()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("code", [204, 304])
async def test_start_204_or_304_returns_true(code: int) -> None:
    """204 (started) and 304 (already running) both report success."""
    http = _mock_http(_mock_response(code), method="post")
    with patch("httpx.AsyncClient", return_value=http):
        assert await _client().start("iw-rag-vllm-1") is True


@pytest.mark.asyncio
async def test_start_404_returns_false_not_error() -> None:
    """Starting a non-existent (optional) container returns False, not an error."""
    http = _mock_http(_mock_response(404), method="post")
    with patch("httpx.AsyncClient", return_value=http):
        assert await _client().start("injguard") is False


@pytest.mark.asyncio
async def test_stop_passes_timeout_param_and_returns_true() -> None:
    """``stop`` issues the daemon stop with the grace-timeout query param."""
    resp = _mock_response(204)
    http = _mock_http(resp, method="post")
    with patch("httpx.AsyncClient", return_value=http):
        ok = await _client().stop("iw-rag-vllm-1", timeout_s=7)
    assert ok is True
    # The POST carried the t=7 graceful-stop param.
    _args, kwargs = http.post.call_args
    assert kwargs["params"] == {"t": 7}


@pytest.mark.asyncio
async def test_stop_5xx_raises_docker_unavailable() -> None:
    """A 5xx on stop raises ``DockerUnavailable`` (transient daemon failure)."""
    http = _mock_http(_mock_response(500, text="daemon error"), method="post")
    with (
        patch("httpx.AsyncClient", return_value=http),
        pytest.raises(DockerUnavailable, match="HTTP 500"),
    ):
        await _client().stop("iw-rag-vllm-1")


# ---------------------------------------------------------------------------
# _parse_inspect — pure
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_matching_returns_prefix_matches_only() -> None:
    """list_matching keeps only true prefix matches (filter is substring on the daemon)."""
    payload = [
        {"Names": ["/vllm-qwen3-14b-awq"]},
        {"Names": ["/my-vllm-other"]},  # substring match the daemon may return — must be dropped
        {"Names": ["/vllm-qwen3-32b-awq"]},
    ]
    http = _mock_http(_mock_response(200, payload), method="get")
    with patch("httpx.AsyncClient", return_value=http):
        names = await _client().list_matching("vllm-")
    assert names == ["vllm-qwen3-14b-awq", "vllm-qwen3-32b-awq"]


@pytest.mark.asyncio
async def test_list_matching_empty_when_none() -> None:
    """No matches → empty list (not an error)."""
    http = _mock_http(_mock_response(200, []), method="get")
    with patch("httpx.AsyncClient", return_value=http):
        assert await _client().list_matching("vllm-") == []


def test_parse_inspect_handles_missing_state_block() -> None:
    """A malformed payload (no State) degrades to a non-running unknown state."""
    st = _parse_inspect("x", {})
    assert st == ContainerState(name="x", exists=True, running=False, status="unknown", health=None)
