"""Unit tests for the F-00094 ``LlamaSwapClient`` (S05 coverage step).

The :class:`LlamaSwapClient` is the typed async HTTP client the
swap-controller uses to talk to the local ``llama-swap`` proxy.  All
``httpx`` calls are **MOCKED** — no live llama-swap, no live GPU, no
network.  Boundary rows covered:

* ``list_models`` parses ``{"data": [...]}`` into ``ModelState`` records.
* Connection errors map to the typed ``LlamaSwapUnavailable``.
* Non-2xx responses map to the typed ``LlamaSwapUnavailable``.
* ``model_health`` returns ``True`` only on HTTP 200; any other status
  (including connection errors surfaced through the client) is ``False``
  or the typed error.
* ``activate`` posts a chat-completions request whose ``model`` field
  names the requested id (the llama-swap trigger documented at
  https://github.com/mostlygeek/llama-swap — R-00123 §6).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from iw_rag.serving.llama_swap_client import (
    LlamaSwapClient,
    LlamaSwapUnavailable,
    ModelState,
)


def _client() -> LlamaSwapClient:
    """Build a small client for the mocked-call tests."""
    return LlamaSwapClient(base_url="http://llama-swap:8080", timeout_s=1.0)


def _ok_models_payload(ids: list[str]) -> dict[str, Any]:
    """Build an OpenAI-compatible ``/v1/models`` payload."""
    return {"data": [{"id": i, "object": "model"} for i in ids]}


def _mock_response(status_code: int, payload: Any = None) -> MagicMock:
    """Build a mock httpx Response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = "" if payload is None else str(payload)
    resp.json = MagicMock(return_value=payload if payload is not None else {})
    return resp


# ---------------------------------------------------------------------------
# list_models — happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_models_parses_openai_compatible_payload() -> None:
    """``list_models`` parses an OpenAI ``{"data": [...]}`` payload."""
    client = _client()
    payload = _ok_models_payload(["qwen3-7b-awq", "qwen3-14b-awq"])
    mock_resp = _mock_response(200, payload)
    mock_http = MagicMock()
    mock_http.__aenter__ = AsyncMock(return_value=mock_http)
    mock_http.__aexit__ = AsyncMock(return_value=None)
    mock_http.get = AsyncMock(return_value=mock_resp)
    with patch("httpx.AsyncClient", return_value=mock_http):
        out = await client.list_models()
    assert out == [
        ModelState(id="qwen3-7b-awq", state="available"),
        ModelState(id="qwen3-14b-awq", state="available"),
    ]


@pytest.mark.asyncio
async def test_list_models_skips_blank_ids() -> None:
    """``list_models`` ignores entries with no ``id``."""
    client = _client()
    payload = {
        "data": [
            {"id": "qwen3-7b-awq", "object": "model"},
            {"id": "", "object": "model"},
            {"object": "model"},  # no id at all
        ]
    }
    mock_resp = _mock_response(200, payload)
    mock_http = MagicMock()
    mock_http.__aenter__ = AsyncMock(return_value=mock_http)
    mock_http.__aexit__ = AsyncMock(return_value=None)
    mock_http.get = AsyncMock(return_value=mock_resp)
    with patch("httpx.AsyncClient", return_value=mock_http):
        out = await client.list_models()
    assert [m.id for m in out] == ["qwen3-7b-awq"]


@pytest.mark.asyncio
async def test_list_models_empty_payload_returns_empty_list() -> None:
    """``list_models`` with ``{"data": []}`` returns ``[]`` (no error)."""
    client = _client()
    mock_resp = _mock_response(200, {"data": []})
    mock_http = MagicMock()
    mock_http.__aenter__ = AsyncMock(return_value=mock_http)
    mock_http.__aexit__ = AsyncMock(return_value=None)
    mock_http.get = AsyncMock(return_value=mock_resp)
    with patch("httpx.AsyncClient", return_value=mock_http):
        out = await client.list_models()
    assert out == []


# ---------------------------------------------------------------------------
# list_models — error mapping
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_maps_connection_error_to_typed_exception() -> None:
    """``list_models`` raises ``LlamaSwapUnavailable`` on httpx connection error."""
    client = _client()

    async def _boom(*_args: Any, **_kwargs: Any) -> Any:
        raise httpx.ConnectError("connection refused")

    with patch("httpx.AsyncClient.get", new=_boom), pytest.raises(LlamaSwapUnavailable):
        await client.list_models()


@pytest.mark.asyncio
async def test_list_maps_timeout_to_typed_exception() -> None:
    """``list_models`` raises ``LlamaSwapUnavailable`` on httpx timeout."""
    client = _client()

    async def _boom(*_args: Any, **_kwargs: Any) -> Any:
        raise httpx.ReadTimeout("timed out")

    with patch("httpx.AsyncClient.get", new=_boom), pytest.raises(LlamaSwapUnavailable):
        await client.list_models()


@pytest.mark.asyncio
async def test_list_maps_non_2xx_to_typed_exception() -> None:
    """``list_models`` raises ``LlamaSwapUnavailable`` on a non-200 status."""
    client = _client()
    mock_resp = _mock_response(500, "internal error")
    mock_http = MagicMock()
    mock_http.__aenter__ = AsyncMock(return_value=mock_http)
    mock_http.__aexit__ = AsyncMock(return_value=None)
    mock_http.get = AsyncMock(return_value=mock_resp)
    with patch("httpx.AsyncClient", return_value=mock_http), pytest.raises(LlamaSwapUnavailable):
        await client.list_models()


@pytest.mark.asyncio
async def test_list_maps_unparseable_payload_to_typed_exception() -> None:
    """``list_models`` raises ``LlamaSwapUnavailable`` on a non-JSON payload."""
    client = _client()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json = MagicMock(side_effect=ValueError("not json"))
    mock_http = MagicMock()
    mock_http.__aenter__ = AsyncMock(return_value=mock_http)
    mock_http.__aexit__ = AsyncMock(return_value=None)
    mock_http.get = AsyncMock(return_value=mock_resp)
    with patch("httpx.AsyncClient", return_value=mock_http), pytest.raises(LlamaSwapUnavailable):
        await client.list_models()


# ---------------------------------------------------------------------------
# model_health — readiness gate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_model_health_returns_true_on_200() -> None:
    """AC2: ``model_health`` returns True on HTTP 200."""
    client = _client()
    mock_resp = _mock_response(200, _ok_models_payload(["qwen3-7b-awq"]))
    mock_http = MagicMock()
    mock_http.__aenter__ = AsyncMock(return_value=mock_http)
    mock_http.__aexit__ = AsyncMock(return_value=None)
    mock_http.get = AsyncMock(return_value=mock_resp)
    with patch("httpx.AsyncClient", return_value=mock_http):
        assert await client.model_health("qwen3-7b-awq") is True


@pytest.mark.asyncio
async def test_model_health_returns_false_on_non_2xx() -> None:
    """``model_health`` returns False on any non-2xx status (not 503)."""
    client = _client()
    mock_resp = _mock_response(404, "not found")
    mock_http = MagicMock()
    mock_http.__aenter__ = AsyncMock(return_value=mock_http)
    mock_http.__aexit__ = AsyncMock(return_value=None)
    mock_http.get = AsyncMock(return_value=mock_resp)
    with patch("httpx.AsyncClient", return_value=mock_http):
        assert await client.model_health("qwen3-7b-awq") is False


@pytest.mark.asyncio
async def test_model_health_passes_model_id_query_param() -> None:
    """``model_health`` calls ``/v1/models?model=<id>`` for the readiness probe."""
    client = _client()
    mock_resp = _mock_response(200, _ok_models_payload(["qwen3-7b-awq"]))
    mock_http = MagicMock()
    mock_http.__aenter__ = AsyncMock(return_value=mock_http)
    mock_http.__aexit__ = AsyncMock(return_value=None)
    mock_http.get = AsyncMock(return_value=mock_resp)
    with patch("httpx.AsyncClient", return_value=mock_http) as client_factory:
        await client.model_health("qwen3-7b-awq")
        # Inspect the per-call args passed to the factory.
        factory_call = client_factory.call_args
        # And the per-request call to .get(...)
        get_call = mock_http.get.call_args
        assert get_call is not None
        url, params = get_call.args[0], get_call.kwargs.get("params")
        assert url.endswith("/v1/models")
        assert params == {"model": "qwen3-7b-awq"}
        _ = factory_call  # silence unused-var lint


@pytest.mark.asyncio
async def test_model_health_propagates_unavailable_on_connection_error() -> None:
    """``model_health`` raises ``LlamaSwapUnavailable`` on connection failure.

    The controller treats this as "not ready yet" (keeps polling on the
    budget), but the client itself surfaces a typed error so the
    controller can choose what to do.
    """
    client = _client()

    async def _boom(*_args: Any, **_kwargs: Any) -> Any:
        raise httpx.ConnectError("connection refused")

    with patch("httpx.AsyncClient.get", new=_boom), pytest.raises(LlamaSwapUnavailable):
        await client.model_health("qwen3-7b-awq")


# ---------------------------------------------------------------------------
# activate — chat-completions trigger
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_activate_posts_chat_completions_with_model_id() -> None:
    """``activate`` posts a chat-completions request naming ``model_id``.

    This is the documented llama-swap activation mechanism — the exclusive
    group stops the currently resident sibling and starts the requested
    one.  We assert the request shape and the 200 path.
    """
    client = _client()
    mock_resp = _mock_response(200, {"choices": []})
    mock_http = MagicMock()
    mock_http.__aenter__ = AsyncMock(return_value=mock_http)
    mock_http.__aexit__ = AsyncMock(return_value=None)
    mock_http.post = AsyncMock(return_value=mock_resp)
    with patch("httpx.AsyncClient", return_value=mock_http):
        await client.activate("qwen3-7b-awq")
    post_call = mock_http.post.call_args
    assert post_call is not None
    url = post_call.args[0]
    assert url.endswith("/v1/chat/completions")
    body = post_call.kwargs["json"]
    assert body["model"] == "qwen3-7b-awq"
    assert body["max_tokens"] == 1
    assert body["stream"] is False
    # And the messages are minimal (1 char content).
    assert body["messages"] == [{"role": "user", "content": "."}]


@pytest.mark.asyncio
async def test_activate_maps_non_2xx_to_typed_exception() -> None:
    """``activate`` raises ``LlamaSwapUnavailable`` on a non-2xx status."""
    client = _client()
    mock_resp = _mock_response(500, "kaboom")
    mock_http = MagicMock()
    mock_http.__aenter__ = AsyncMock(return_value=mock_http)
    mock_http.__aexit__ = AsyncMock(return_value=None)
    mock_http.post = AsyncMock(return_value=mock_resp)
    with patch("httpx.AsyncClient", return_value=mock_http), pytest.raises(LlamaSwapUnavailable):
        await client.activate("qwen3-7b-awq")


@pytest.mark.asyncio
async def test_activate_maps_4xx_to_typed_exception() -> None:
    """``activate`` raises ``LlamaSwapUnavailable`` on a 4xx (e.g. unknown model)."""
    client = _client()
    mock_resp = _mock_response(404, "unknown model")
    mock_http = MagicMock()
    mock_http.__aenter__ = AsyncMock(return_value=mock_http)
    mock_http.__aexit__ = AsyncMock(return_value=None)
    mock_http.post = AsyncMock(return_value=mock_resp)
    with patch("httpx.AsyncClient", return_value=mock_http), pytest.raises(LlamaSwapUnavailable):
        await client.activate("nope")


@pytest.mark.asyncio
async def test_activate_maps_connection_error_to_typed_exception() -> None:
    """``activate`` raises ``LlamaSwapUnavailable`` on httpx connection error."""
    client = _client()

    async def _boom(*_args: Any, **_kwargs: Any) -> Any:
        raise httpx.ConnectError("connection refused")

    with patch("httpx.AsyncClient.post", new=_boom), pytest.raises(LlamaSwapUnavailable):
        await client.activate("qwen3-7b-awq")


# ---------------------------------------------------------------------------
# Typed error: LlamaSwapUnavailable contract
# ---------------------------------------------------------------------------


def test_llama_swap_unavailable_is_a_subclass_of_iw_rag_error() -> None:
    """``LlamaSwapUnavailable`` is a typed ``IwRagError`` — error-handler chain catches it."""
    from iw_rag.domain.errors import IwRagError

    assert issubclass(LlamaSwapUnavailable, IwRagError)


def test_llama_swap_client_base_url_trailing_slash_stripped() -> None:
    """The base URL trailing slash is stripped so URL concatenation is stable."""
    c1 = LlamaSwapClient(base_url="http://llama-swap:8080/", timeout_s=1.0)
    c2 = LlamaSwapClient(base_url="http://llama-swap:8080", timeout_s=1.0)
    # Internal attribute; documented as `_base_url`.
    assert c1._base_url == c2._base_url == "http://llama-swap:8080"
