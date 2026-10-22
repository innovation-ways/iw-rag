"""Unit tests for the :class:`PowerClient` proxy (API → system-control sidecar).

All ``httpx`` calls are MOCKED.  Verifies happy-path JSON parsing and the
connection-error / non-2xx → :class:`PowerControlUnavailable` contract.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from iw_rag.serving.power_client import PowerClient, PowerControlUnavailable


def _client() -> PowerClient:
    return PowerClient(base_url="http://system-control:8091/", timeout_s=1.0)


def _resp(status_code: int, payload: Any = None) -> MagicMock:
    r = MagicMock()
    r.status_code = status_code
    r.text = "" if payload is None else str(payload)
    r.json = MagicMock(return_value=payload if payload is not None else {})
    return r


def _http(resp: MagicMock, *, method: str) -> MagicMock:
    m = MagicMock()
    m.__aenter__ = AsyncMock(return_value=m)
    m.__aexit__ = AsyncMock(return_value=None)
    setattr(m, method, AsyncMock(return_value=resp))
    return m


def test_base_url_trailing_slash_stripped() -> None:
    """A trailing slash on base_url is stripped for stable URL concatenation."""
    assert PowerClient(base_url="http://x:8091/")._base_url == "http://x:8091"


@pytest.mark.asyncio
async def test_get_status_parses_payload() -> None:
    """get_status returns the parsed JSON status body."""
    payload = {"state": "on", "components": [], "vram": {"available": True}}
    with patch("httpx.AsyncClient", return_value=_http(_resp(200, payload), method="get")):
        out = await _client().get_status()
    assert out["state"] == "on"


@pytest.mark.asyncio
async def test_power_on_posts_components_body() -> None:
    """power_on POSTs the components list and returns the status payload."""
    http = _http(_resp(200, {"state": "powering_on"}), method="post")
    with patch("httpx.AsyncClient", return_value=http):
        out = await _client().power_on(["generator"])
    assert out["state"] == "powering_on"
    _args, kwargs = http.post.call_args
    assert kwargs["json"] == {"components": ["generator"]}


@pytest.mark.asyncio
async def test_power_off_master_sends_null_components() -> None:
    """power_off(None) (master) sends components=null in the body."""
    http = _http(_resp(200, {"state": "off"}), method="post")
    with patch("httpx.AsyncClient", return_value=http):
        await _client().power_off(None)
    _args, kwargs = http.post.call_args
    assert kwargs["json"] == {"components": None}


@pytest.mark.asyncio
async def test_orchestrate_posts_stop_and_start() -> None:
    """orchestrate POSTs the stop+start lists to /power/orchestrate."""
    http = _http(_resp(200, {"state": "partial"}), method="post")
    with patch("httpx.AsyncClient", return_value=http):
        out = await _client().orchestrate(stop=["a"], start=["b", "c"])
    assert out["state"] == "partial"
    _args, kwargs = http.post.call_args
    assert kwargs["json"] == {"stop": ["a"], "start": ["b", "c"]}


@pytest.mark.asyncio
async def test_non_2xx_raises_power_control_unavailable() -> None:
    """A non-2xx response maps to PowerControlUnavailable with the path."""
    with (
        patch("httpx.AsyncClient", return_value=_http(_resp(503, "down"), method="get")),
        pytest.raises(PowerControlUnavailable, match="HTTP 503"),
    ):
        await _client().get_status()


@pytest.mark.asyncio
async def test_connection_error_raises_power_control_unavailable() -> None:
    """A transport error maps to PowerControlUnavailable (unreachable sidecar)."""
    http = MagicMock()
    http.__aenter__ = AsyncMock(return_value=http)
    http.__aexit__ = AsyncMock(return_value=None)
    http.get = AsyncMock(side_effect=httpx.ConnectError("refused"))
    with (
        patch("httpx.AsyncClient", return_value=http),
        pytest.raises(PowerControlUnavailable, match="unreachable"),
    ):
        await _client().get_vram()
