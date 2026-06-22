"""Unit tests for the F-00094 ``SwapControllerClient`` (serving coverage step).

The :class:`SwapControllerClient` is the typed async HTTP client the public
API uses to proxy model-activation requests to the internal swap-controller
sidecar (port 8090).  All ``httpx`` calls are **MOCKED** — no live sidecar,
no live GPU, no network.  Boundary rows covered:

* ``activate()`` happy path returns the parsed JSON payload.
* ``activate()`` non-2xx → :class:`LlamaSwapUnavailable` (with ``match=``).
* ``activate()`` connection error → :class:`LlamaSwapUnavailable`.
* ``get_status()`` happy path parses the sidecar's status payload.
* ``get_status()`` non-2xx → :class:`LlamaSwapUnavailable`.
* ``get_catalog()`` happy path returns the parsed catalog dict.
* ``get_catalog()`` non-2xx → :class:`LlamaSwapUnavailable`.
* ``get_catalog()`` connection error → :class:`LlamaSwapUnavailable`.
* Base URL trailing slash is stripped (stable URL concatenation).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from iw_rag.serving.llama_swap_client import LlamaSwapUnavailable
from iw_rag.serving.swap_controller_client import SwapControllerClient


def _client() -> SwapControllerClient:
    """Build a small SwapControllerClient for the mocked-call tests."""
    return SwapControllerClient(base_url="http://swap-controller:8090", timeout_s=1.0)


def _mock_response(status_code: int, payload: Any = None) -> MagicMock:
    """Build a mock httpx Response.

    Args:
        status_code: The HTTP status code to return.
        payload: The JSON payload to return from ``.json()``, or ``None``.

    Returns:
        A ``MagicMock`` shaped like an ``httpx.Response``.
    """
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = "" if payload is None else str(payload)
    resp.json = MagicMock(return_value=payload if payload is not None else {})
    return resp


def _mock_http(resp: MagicMock, *, method: str = "get") -> MagicMock:
    """Build a mock ``httpx.AsyncClient`` context-manager returning ``resp``.

    Args:
        resp: The mock response the client method will return.
        method: HTTP verb to wire (``"get"`` or ``"post"``).

    Returns:
        A ``MagicMock`` that behaves as an async context manager and exposes
        the requested HTTP method as an ``AsyncMock``.
    """
    mock = MagicMock()
    mock.__aenter__ = AsyncMock(return_value=mock)
    mock.__aexit__ = AsyncMock(return_value=None)
    setattr(mock, method, AsyncMock(return_value=resp))
    return mock


# ---------------------------------------------------------------------------
# activate() — happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_activate_happy_path_returns_parsed_json() -> None:
    """``activate()`` returns the sidecar's JSON payload on HTTP 200."""
    client = _client()
    payload = {"status": "activating", "model_id": "qwen3-14b-awq"}
    mock_resp = _mock_response(200, payload)
    mock_http = _mock_http(mock_resp, method="post")
    with patch("httpx.AsyncClient", return_value=mock_http):
        result = await client.activate("qwen3-14b-awq")
    assert result == payload


@pytest.mark.asyncio
async def test_activate_posts_to_catalog_activate_endpoint() -> None:
    """``activate()`` posts ``{"model_id": ...}`` to ``/catalog/activate``."""
    client = _client()
    payload = {"status": "ok"}
    mock_resp = _mock_response(200, payload)
    mock_http = _mock_http(mock_resp, method="post")
    with patch("httpx.AsyncClient", return_value=mock_http):
        await client.activate("my-model")
    post_call = mock_http.post.call_args
    assert post_call is not None
    url = post_call.args[0]
    assert url.endswith("/catalog/activate"), f"unexpected url: {url}"
    body = post_call.kwargs["json"]
    assert body == {"model_id": "my-model"}


# ---------------------------------------------------------------------------
# activate() — error mapping
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_activate_non_2xx_raises_llama_swap_unavailable() -> None:
    """``activate()`` raises ``LlamaSwapUnavailable`` on a non-2xx status."""
    client = _client()
    mock_resp = _mock_response(503, "service unavailable")
    mock_http = _mock_http(mock_resp, method="post")
    with (
        patch("httpx.AsyncClient", return_value=mock_http),
        pytest.raises(LlamaSwapUnavailable, match="503"),
    ):
        await client.activate("qwen3-14b-awq")


@pytest.mark.asyncio
async def test_activate_4xx_raises_llama_swap_unavailable() -> None:
    """``activate()`` raises ``LlamaSwapUnavailable`` on a 4xx (e.g. unknown model)."""
    client = _client()
    mock_resp = _mock_response(404, "unknown model")
    mock_http = _mock_http(mock_resp, method="post")
    with (
        patch("httpx.AsyncClient", return_value=mock_http),
        pytest.raises(LlamaSwapUnavailable, match="404"),
    ):
        await client.activate("no-such-model")


@pytest.mark.asyncio
async def test_activate_connection_error_raises_llama_swap_unavailable() -> None:
    """``activate()`` raises ``LlamaSwapUnavailable`` on an httpx connection error."""
    client = _client()

    async def _boom(*_args: Any, **_kwargs: Any) -> Any:
        raise httpx.ConnectError("connection refused")

    with (
        patch("httpx.AsyncClient.post", new=_boom),
        pytest.raises(LlamaSwapUnavailable, match="unreachable"),
    ):
        await client.activate("qwen3-14b-awq")


@pytest.mark.asyncio
async def test_activate_timeout_raises_llama_swap_unavailable() -> None:
    """``activate()`` raises ``LlamaSwapUnavailable`` on an httpx read timeout."""
    client = _client()

    async def _boom(*_args: Any, **_kwargs: Any) -> Any:
        raise httpx.ReadTimeout("timed out")

    with patch("httpx.AsyncClient.post", new=_boom), pytest.raises(LlamaSwapUnavailable):
        await client.activate("qwen3-14b-awq")


# ---------------------------------------------------------------------------
# get_status() — happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_status_happy_path_returns_parsed_json() -> None:
    """``get_status()`` returns the sidecar's status dict on HTTP 200."""
    client = _client()
    payload = {"active_model": "qwen3-14b-awq", "state": "ready"}
    mock_resp = _mock_response(200, payload)
    mock_http = _mock_http(mock_resp, method="get")
    with patch("httpx.AsyncClient", return_value=mock_http):
        result = await client.get_status()
    assert result == payload


@pytest.mark.asyncio
async def test_get_status_calls_catalog_status_endpoint() -> None:
    """``get_status()`` calls ``GET /catalog/status`` on the sidecar."""
    client = _client()
    payload = {"state": "idle"}
    mock_resp = _mock_response(200, payload)
    mock_http = _mock_http(mock_resp, method="get")
    with patch("httpx.AsyncClient", return_value=mock_http):
        await client.get_status()
    get_call = mock_http.get.call_args
    assert get_call is not None
    url = get_call.args[0]
    assert url.endswith("/catalog/status"), f"unexpected url: {url}"


# ---------------------------------------------------------------------------
# get_status() — error mapping
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_status_non_2xx_raises_llama_swap_unavailable() -> None:
    """``get_status()`` raises ``LlamaSwapUnavailable`` on a non-2xx status."""
    client = _client()
    mock_resp = _mock_response(500, "internal server error")
    mock_http = _mock_http(mock_resp, method="get")
    with (
        patch("httpx.AsyncClient", return_value=mock_http),
        pytest.raises(LlamaSwapUnavailable, match="500"),
    ):
        await client.get_status()


@pytest.mark.asyncio
async def test_get_status_connection_error_raises_llama_swap_unavailable() -> None:
    """``get_status()`` raises ``LlamaSwapUnavailable`` on an httpx connection error."""
    client = _client()

    async def _boom(*_args: Any, **_kwargs: Any) -> Any:
        raise httpx.ConnectError("connection refused")

    with (
        patch("httpx.AsyncClient.get", new=_boom),
        pytest.raises(LlamaSwapUnavailable, match="unreachable"),
    ):
        await client.get_status()


# ---------------------------------------------------------------------------
# get_catalog() — happy path (error path is pre-existing; add the success path)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_catalog_happy_path_returns_parsed_catalog() -> None:
    """``get_catalog()`` returns the annotated catalog dict on HTTP 200."""
    client = _client()
    payload = {
        "models": [
            {"id": "qwen3-7b-awq", "state": "available"},
            {"id": "qwen3-14b-awq", "state": "unavailable"},
        ]
    }
    mock_resp = _mock_response(200, payload)
    mock_http = _mock_http(mock_resp, method="get")
    with patch("httpx.AsyncClient", return_value=mock_http):
        result = await client.get_catalog()
    assert result == payload
    assert len(result["models"]) == 2
    assert result["models"][0]["id"] == "qwen3-7b-awq"


@pytest.mark.asyncio
async def test_get_catalog_calls_catalog_endpoint() -> None:
    """``get_catalog()`` calls ``GET /catalog`` on the sidecar."""
    client = _client()
    payload = {"models": []}
    mock_resp = _mock_response(200, payload)
    mock_http = _mock_http(mock_resp, method="get")
    with patch("httpx.AsyncClient", return_value=mock_http):
        await client.get_catalog()
    get_call = mock_http.get.call_args
    assert get_call is not None
    url = get_call.args[0]
    assert url.endswith("/catalog"), f"unexpected url: {url}"


# ---------------------------------------------------------------------------
# get_catalog() — error mapping
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_catalog_non_2xx_raises_llama_swap_unavailable() -> None:
    """``get_catalog()`` raises ``LlamaSwapUnavailable`` on a non-2xx status."""
    client = _client()
    mock_resp = _mock_response(400, "bad request")
    mock_http = _mock_http(mock_resp, method="get")
    with (
        patch("httpx.AsyncClient", return_value=mock_http),
        pytest.raises(LlamaSwapUnavailable, match="400"),
    ):
        await client.get_catalog()


@pytest.mark.asyncio
async def test_get_catalog_connection_error_raises_llama_swap_unavailable() -> None:
    """``get_catalog()`` raises ``LlamaSwapUnavailable`` on an httpx connection error."""
    client = _client()

    async def _boom(*_args: Any, **_kwargs: Any) -> Any:
        raise httpx.ConnectError("connection refused")

    with (
        patch("httpx.AsyncClient.get", new=_boom),
        pytest.raises(LlamaSwapUnavailable, match="unreachable"),
    ):
        await client.get_catalog()


# ---------------------------------------------------------------------------
# SwapControllerClient construction contract
# ---------------------------------------------------------------------------


def test_swap_controller_client_strips_trailing_slash() -> None:
    """The base URL trailing slash is stripped so URL concatenation is stable."""
    c1 = SwapControllerClient(base_url="http://swap-controller:8090/", timeout_s=1.0)
    c2 = SwapControllerClient(base_url="http://swap-controller:8090", timeout_s=1.0)
    assert c1._base_url == c2._base_url == "http://swap-controller:8090"
