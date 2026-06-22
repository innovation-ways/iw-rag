"""Unit tests for VllmClient error paths, _parse branches, and health().

Covers the branches not exercised by test_vllm_client.py:

- ``_post`` transport / status errors → ``GenerationError`` wrap with chained cause.
- ``_parse``: tool_calls branch (non-empty tool_calls present in the response),
  empty-content branch (content is None → text ``""``), usage, finish_reason fields.
- ``health()``: True on HTTP 200; False on httpx transport error.
- ``enable_thinking`` plumbing: False injects ``chat_template_kwargs``; True omits it.

All httpx.AsyncClient calls are patched — no live network.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from iw_rag.domain.errors import GenerationError
from iw_rag.generation.llm import GenerationParams, LlmResponse, ToolSpec, VllmClient

# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------

_PARAMS = GenerationParams(model="test-model", temperature=0.0, max_tokens=64, seed=None)
_MSGS: list[dict[str, Any]] = [{"role": "user", "content": "ping"}]


def _client(*, enable_thinking: bool = True) -> VllmClient:
    """Build a VllmClient pointed at a fake endpoint.

    Args:
        enable_thinking: Passed directly to VllmClient.__init__.

    Returns:
        A VllmClient instance configured for testing.
    """
    return VllmClient("http://vllm:8001/v1", enable_thinking=enable_thinking)


def _mock_http_client(
    *,
    method: str = "post",
    response_payload: dict[str, Any] | None = None,
    status_code: int = 200,
    side_effect: Exception | None = None,
) -> MagicMock:
    """Build a mock httpx.AsyncClient for use in ``async with`` context.

    Args:
        method: The HTTP method attribute to mock (``"post"`` or ``"get"``).
        response_payload: JSON payload the mock response returns.
        status_code: HTTP status code for the mock response.
        side_effect: If provided, the mocked method raises this instead of returning.

    Returns:
        A MagicMock compatible with ``async with httpx.AsyncClient() as client:``.
    """
    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    if side_effect is not None:
        setattr(mock_client, method, AsyncMock(side_effect=side_effect))
        return mock_client

    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    if response_payload is not None:
        resp.json = MagicMock(return_value=response_payload)
    if status_code >= 400:
        resp.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError(
                f"HTTP {status_code}",
                request=MagicMock(),
                response=resp,
            )
        )
    else:
        resp.raise_for_status = MagicMock(return_value=None)
    setattr(mock_client, method, AsyncMock(return_value=resp))
    return mock_client


def _chat_payload(
    content: str | None = "hello",
    tool_calls: list[dict[str, Any]] | None = None,
    finish_reason: str = "stop",
    usage: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Build a minimal OpenAI-compatible chat completion payload.

    Args:
        content: The assistant message content (None simulates a tool-call-only response).
        tool_calls: Optional list of tool call dicts.
        finish_reason: The finish_reason for the choice.
        usage: Optional token usage dict.

    Returns:
        A dict matching the vLLM /chat/completions response schema.
    """
    msg: dict[str, Any] = {}
    if content is not None:
        msg["content"] = content
    if tool_calls is not None:
        msg["tool_calls"] = tool_calls
    payload: dict[str, Any] = {
        "choices": [{"message": msg, "finish_reason": finish_reason}],
    }
    if usage is not None:
        payload["usage"] = usage
    return payload


# ---------------------------------------------------------------------------
# 1. _post — transport errors wrap to GenerationError
# ---------------------------------------------------------------------------


async def test_post_connect_error_raises_generation_error() -> None:
    """_post wraps httpx.ConnectError in GenerationError with the original as __cause__."""
    original = httpx.ConnectError("connection refused")
    mock_client = _mock_http_client(side_effect=original)
    client = _client()

    with (
        patch("httpx.AsyncClient", return_value=mock_client),
        pytest.raises(GenerationError, match=r"vLLM request failed") as exc_info,
    ):
        await client._post({"model": "x"})

    assert type(exc_info.value.__cause__) is httpx.ConnectError


async def test_post_read_timeout_raises_generation_error() -> None:
    """_post wraps httpx.ReadTimeout in GenerationError with the original as __cause__."""
    original = httpx.ReadTimeout("timed out")
    mock_client = _mock_http_client(side_effect=original)
    client = _client()

    with (
        patch("httpx.AsyncClient", return_value=mock_client),
        pytest.raises(GenerationError, match=r"vLLM request failed") as exc_info,
    ):
        await client._post({"model": "x"})

    assert type(exc_info.value.__cause__) is httpx.ReadTimeout


async def test_post_http_status_500_raises_generation_error() -> None:
    """_post wraps a 500 HTTPStatusError (via raise_for_status) in GenerationError."""
    mock_client = _mock_http_client(status_code=500, response_payload={})
    client = _client()

    with (
        patch("httpx.AsyncClient", return_value=mock_client),
        pytest.raises(GenerationError, match=r"vLLM request failed") as exc_info,
    ):
        await client._post({"model": "x"})

    assert type(exc_info.value.__cause__) is httpx.HTTPStatusError


async def test_post_http_status_429_raises_generation_error() -> None:
    """_post wraps a 429 rate-limit HTTPStatusError in GenerationError."""
    mock_client = _mock_http_client(status_code=429, response_payload={})
    client = _client()

    with (
        patch("httpx.AsyncClient", return_value=mock_client),
        pytest.raises(GenerationError, match=r"vLLM request failed") as exc_info,
    ):
        await client._post({"model": "x"})

    assert type(exc_info.value.__cause__) is httpx.HTTPStatusError


async def test_post_value_error_from_json_raises_generation_error() -> None:
    """_post wraps a ValueError (malformed JSON from resp.json()) in GenerationError."""
    resp = MagicMock(spec=httpx.Response)
    resp.raise_for_status = MagicMock(return_value=None)
    resp.json = MagicMock(side_effect=ValueError("not valid JSON"))
    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.post = AsyncMock(return_value=resp)
    client = _client()

    with (
        patch("httpx.AsyncClient", return_value=mock_client),
        pytest.raises(GenerationError, match=r"vLLM request failed") as exc_info,
    ):
        await client._post({"model": "x"})

    assert type(exc_info.value.__cause__) is ValueError


async def test_post_success_returns_dict() -> None:
    """_post returns the parsed JSON dict on a successful 200 response."""
    payload = _chat_payload("ok")
    mock_client = _mock_http_client(response_payload=payload)
    client = _client()

    with patch("httpx.AsyncClient", return_value=mock_client):
        result = await client._post({"model": "x"})

    assert result == payload


# ---------------------------------------------------------------------------
# 2. _parse — tool_calls branch
# ---------------------------------------------------------------------------


def test_parse_tool_calls_returned_verbatim() -> None:
    """_parse returns the tool_calls list exactly as present in the response."""
    tool_calls = [
        {
            "id": "call_abc",
            "type": "function",
            "function": {"name": "search", "arguments": '{"q":"test"}'},
        }
    ]
    payload = _chat_payload(content=None, tool_calls=tool_calls, finish_reason="tool_calls")
    result = VllmClient._parse(payload)

    assert result.tool_calls == tool_calls
    assert result.finish_reason == "tool_calls"


def test_parse_tool_calls_multiple_entries() -> None:
    """_parse preserves all tool calls when the model emits more than one."""
    tool_calls = [
        {"id": "c1", "type": "function", "function": {"name": "search", "arguments": "{}"}},
        {"id": "c2", "type": "function", "function": {"name": "lookup", "arguments": "{}"}},
    ]
    payload = _chat_payload(content=None, tool_calls=tool_calls, finish_reason="tool_calls")
    result = VllmClient._parse(payload)

    assert len(result.tool_calls) == 2
    assert result.tool_calls[0]["id"] == "c1"
    assert result.tool_calls[1]["id"] == "c2"


# ---------------------------------------------------------------------------
# 3. _parse — empty-content branch
# ---------------------------------------------------------------------------


def test_parse_none_content_yields_empty_text() -> None:
    """_parse converts None message content to empty string (tool-call-only response)."""
    payload = _chat_payload(content=None, tool_calls=[], finish_reason="tool_calls")
    result = VllmClient._parse(payload)

    assert result.text == ""


def test_parse_empty_string_content_yields_empty_text() -> None:
    """_parse treats an explicit empty string content as empty text (falsy or-branch)."""
    payload = _chat_payload(content="", finish_reason="stop")
    result = VllmClient._parse(payload)

    assert result.text == ""


def test_parse_normal_content_returned_verbatim() -> None:
    """_parse returns non-empty content string as-is."""
    payload = _chat_payload(content="The answer is 42.", finish_reason="stop")
    result = VllmClient._parse(payload)

    assert result.text == "The answer is 42."


# ---------------------------------------------------------------------------
# 4. _parse — usage and finish_reason fields
# ---------------------------------------------------------------------------


def test_parse_usage_dict_propagated() -> None:
    """_parse copies the usage dict from the response payload into LlmResponse.usage."""
    usage = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
    payload = _chat_payload(usage=usage)
    result = VllmClient._parse(payload)

    assert result.usage == {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}


def test_parse_missing_usage_defaults_to_empty_dict() -> None:
    """_parse defaults usage to {} when the response omits the usage field."""
    payload = _chat_payload()  # usage not set
    result = VllmClient._parse(payload)

    assert result.usage == {}


def test_parse_finish_reason_length_limit() -> None:
    """_parse captures the finish_reason string exactly from the choice."""
    payload = _chat_payload(finish_reason="length")
    result = VllmClient._parse(payload)

    assert result.finish_reason == "length"


def test_parse_no_tool_calls_in_response_returns_empty_list() -> None:
    """_parse returns an empty list for tool_calls when the field is absent from message."""
    payload = _chat_payload(content="hi")  # no tool_calls key
    result = VllmClient._parse(payload)

    assert result.tool_calls == []


def test_parse_returns_llm_response_type() -> None:
    """_parse always returns an LlmResponse instance."""
    payload = _chat_payload(content="x")
    result = VllmClient._parse(payload)

    assert type(result) is LlmResponse


# ---------------------------------------------------------------------------
# 5. health() — True on 200, False on transport error
# ---------------------------------------------------------------------------


async def test_health_returns_true_on_200() -> None:
    """health() returns True when /models returns HTTP 200."""
    mock_client = _mock_http_client(method="get", status_code=200)
    client = _client()

    with patch("httpx.AsyncClient", return_value=mock_client):
        result = await client.health()

    assert result is True


async def test_health_returns_false_on_connect_error() -> None:
    """health() returns False when httpx raises ConnectError (endpoint unreachable)."""
    mock_client = _mock_http_client(method="get", side_effect=httpx.ConnectError("refused"))
    client = _client()

    with patch("httpx.AsyncClient", return_value=mock_client):
        result = await client.health()

    assert result is False


async def test_health_returns_false_on_read_timeout() -> None:
    """health() returns False when httpx raises ReadTimeout."""
    mock_client = _mock_http_client(method="get", side_effect=httpx.ReadTimeout("timeout"))
    client = _client()

    with patch("httpx.AsyncClient", return_value=mock_client):
        result = await client.health()

    assert result is False


async def test_health_returns_false_on_non_200_status() -> None:
    """health() returns False when /models returns a non-200 status code."""
    # For health() we need a non-raising response with status 503 — override raise_for_status
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 503
    mock_resp.raise_for_status = MagicMock(return_value=None)  # health doesn't call this
    inner = MagicMock()
    inner.__aenter__ = AsyncMock(return_value=inner)
    inner.__aexit__ = AsyncMock(return_value=None)
    inner.get = AsyncMock(return_value=mock_resp)
    client = _client()

    with patch("httpx.AsyncClient", return_value=inner):
        result = await client.health()

    assert result is False


# ---------------------------------------------------------------------------
# 6. enable_thinking flag plumbing
# ---------------------------------------------------------------------------


def test_enable_thinking_false_injects_chat_template_kwargs() -> None:
    """_base injects chat_template_kwargs when enable_thinking is False."""
    client = _client(enable_thinking=False)
    body = client._base(_MSGS, _PARAMS)

    assert body["chat_template_kwargs"] == {"enable_thinking": False}


def test_enable_thinking_true_omits_chat_template_kwargs() -> None:
    """_base omits chat_template_kwargs entirely when enable_thinking is True (default)."""
    client = _client(enable_thinking=True)
    body = client._base(_MSGS, _PARAMS)

    assert "chat_template_kwargs" not in body


def test_seed_none_omits_seed_field() -> None:
    """_base omits the seed field from the body when params.seed is None."""
    params_no_seed = GenerationParams(model="m", seed=None)
    client = _client()
    body = client._base(_MSGS, params_no_seed)

    assert "seed" not in body


def test_seed_set_includes_seed_field() -> None:
    """_base includes seed in the body when params.seed is not None."""
    params_with_seed = GenerationParams(model="m", seed=42)
    client = _client()
    body = client._base(_MSGS, params_with_seed)

    assert body["seed"] == 42


# ---------------------------------------------------------------------------
# 7. complete() / complete_tools() / complete_structured() integration
# ---------------------------------------------------------------------------


async def test_complete_returns_parsed_llm_response() -> None:
    """complete() issues a POST and returns the parsed LlmResponse."""
    payload = _chat_payload(content="answer", finish_reason="stop")
    mock_client = _mock_http_client(response_payload=payload)
    client = _client()

    with patch("httpx.AsyncClient", return_value=mock_client):
        result = await client.complete(_MSGS, _PARAMS)

    assert result.text == "answer"
    assert result.finish_reason == "stop"


async def test_complete_tools_includes_tools_in_body() -> None:
    """complete_tools() adds tools and tool_choice fields to the request body."""
    tool = ToolSpec(
        name="search",
        description="search the corpus",
        parameters={"type": "object", "properties": {"q": {"type": "string"}}},
    )
    tool_calls = [{"id": "c1", "function": {"name": "search", "arguments": "{}"}}]
    payload = _chat_payload(content=None, tool_calls=tool_calls, finish_reason="tool_calls")
    mock_client = _mock_http_client(response_payload=payload)
    client = _client()

    with patch("httpx.AsyncClient", return_value=mock_client):
        result = await client.complete_tools(_MSGS, [tool], _PARAMS)

    # Check the body sent to the server
    call_kwargs = mock_client.post.call_args.kwargs
    body = call_kwargs.get("json", {})
    assert body["tool_choice"] == "auto"
    assert body["tools"][0]["function"]["name"] == "search"
    # Check parsed result
    assert result.tool_calls == tool_calls


async def test_complete_structured_includes_guided_json_in_body() -> None:
    """complete_structured() adds the guided_json schema to the request body."""
    schema = {"type": "object", "properties": {"score": {"type": "number"}}}
    payload = _chat_payload(content='{"score": 0.9}', finish_reason="stop")
    mock_client = _mock_http_client(response_payload=payload)
    client = _client()

    with patch("httpx.AsyncClient", return_value=mock_client):
        result = await client.complete_structured(_MSGS, schema, _PARAMS)

    call_kwargs = mock_client.post.call_args.kwargs
    body = call_kwargs.get("json", {})
    assert body["guided_json"] == schema
    assert result.text == '{"score": 0.9}'


async def test_complete_propagates_generation_error_on_transport_failure() -> None:
    """complete() lets GenerationError propagate when _post fails."""
    mock_client = _mock_http_client(side_effect=httpx.ConnectError("down"))
    client = _client()

    with (
        patch("httpx.AsyncClient", return_value=mock_client),
        pytest.raises(GenerationError, match=r"vLLM request failed"),
    ):
        await client.complete(_MSGS, _PARAMS)
