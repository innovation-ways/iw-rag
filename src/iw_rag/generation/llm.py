"""The LLM client — the only place that talks to the generator (vLLM).

OpenAI-compatible (chat + Hermes tool-calling + ``guided_json`` structured output).
``complete_tools``/``complete_structured`` exist from Phase 0 as the Phase-2
enabler (the agent drives retrieval-as-a-tool and JSON-schema graders) but
``/query`` uses only ``complete``. Endpoint + served model come from config —
no hardcoded model string (model-agnostic).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import httpx

from iw_rag.domain.errors import GenerationError

Message = dict[str, Any]


@dataclass(frozen=True, slots=True)
class GenerationParams:
    """Sampling/parameters for a generation call.

    Attributes:
        model: Served model name (from the registry; never hardcoded).
        temperature: Sampling temperature (0.1 default for grounded RAG).
        max_tokens: Max output tokens.
        seed: Deterministic seed for reproducible eval.
    """

    model: str
    temperature: float = 0.1
    max_tokens: int = 1024
    seed: int | None = 0


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """An OpenAI-style function tool the model may call."""

    name: str
    description: str
    parameters: dict[str, Any]

    def as_openai(self) -> dict[str, Any]:
        """Render as an OpenAI ``tools[]`` entry."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass(frozen=True, slots=True)
class LlmResponse:
    """A generation result.

    Attributes:
        text: The assistant message content (may be empty for a pure tool call).
        usage: Token usage (``prompt_tokens``/``completion_tokens``/``total_tokens``).
        tool_calls: Parsed tool calls (Hermes), if any.
        finish_reason: The model's finish reason.
    """

    text: str
    usage: dict[str, int] = field(default_factory=dict)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    finish_reason: str = ""


@runtime_checkable
class LlmClient(Protocol):
    """The generator client contract consumed by generation + (Phase 2) the agent."""

    async def complete(self, messages: Sequence[Message], params: GenerationParams) -> LlmResponse:
        """Plain chat completion."""
        ...

    async def complete_tools(
        self, messages: Sequence[Message], tools: Sequence[ToolSpec], params: GenerationParams
    ) -> LlmResponse:
        """Tool-calling completion (Hermes parser)."""
        ...

    async def complete_structured(
        self, messages: Sequence[Message], json_schema: dict[str, Any], params: GenerationParams
    ) -> LlmResponse:
        """Schema-constrained completion (vLLM ``guided_json``)."""
        ...

    async def health(self) -> bool:
        """Whether the generator endpoint is reachable."""
        ...


class VllmClient:
    """OpenAI-compatible vLLM client (httpx)."""

    def __init__(
        self, endpoint: str, timeout_s: float = 120.0, *, enable_thinking: bool = True
    ) -> None:
        """Configure the vLLM base URL (``…/v1``).

        Args:
            endpoint: Base URL including ``/v1`` (e.g. ``http://localhost:8001/v1``).
            timeout_s: Per-request timeout.
            enable_thinking: Whether the served reasoning model (Qwen3) may emit
                a ``<think>…</think>`` reasoning scratchpad.  When ``False`` we
                pass ``chat_template_kwargs={"enable_thinking": false}`` so the
                model SKIPS reasoning entirely — instead of generating reasoning
                tokens we immediately strip and throw away.  Disabling it is a
                large generation-latency win (the answer is short; the discarded
                reasoning was most of the decoded tokens) and honours the
                ``generation.enable_thinking=false`` config default, which was
                previously never plumbed to the request.  Defaults ``True`` so
                existing callers/tests are unchanged.
        """
        self._endpoint = endpoint.rstrip("/")
        self._timeout_s = timeout_s
        self._enable_thinking = enable_thinking

    async def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=self._timeout_s) as client:
                resp = await client.post(f"{self._endpoint}/chat/completions", json=body)
                resp.raise_for_status()
                data: dict[str, Any] = resp.json()
                return data
        except (httpx.HTTPError, ValueError) as exc:
            raise GenerationError(f"vLLM request failed: {exc}") from exc

    @staticmethod
    def _parse(data: dict[str, Any]) -> LlmResponse:
        choice = data["choices"][0]
        msg = choice.get("message", {})
        return LlmResponse(
            text=msg.get("content") or "",
            usage=data.get("usage", {}),
            tool_calls=msg.get("tool_calls") or [],
            finish_reason=choice.get("finish_reason", ""),
        )

    def _base(self, messages: Sequence[Message], params: GenerationParams) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": params.model,
            "messages": list(messages),
            "temperature": params.temperature,
            "max_tokens": params.max_tokens,
        }
        if params.seed is not None:
            body["seed"] = params.seed
        if not self._enable_thinking:
            # Qwen3's chat template gates its <think> reasoning on this kwarg;
            # turning it off makes the model answer directly instead of decoding
            # (then discarding) a long reasoning scratchpad — a major latency win.
            body["chat_template_kwargs"] = {"enable_thinking": False}
        return body

    async def complete(self, messages: Sequence[Message], params: GenerationParams) -> LlmResponse:
        """Plain chat completion."""
        return self._parse(await self._post(self._base(messages, params)))

    async def complete_tools(
        self, messages: Sequence[Message], tools: Sequence[ToolSpec], params: GenerationParams
    ) -> LlmResponse:
        """Tool-calling completion (Hermes parser)."""
        body = self._base(messages, params)
        body["tools"] = [t.as_openai() for t in tools]
        body["tool_choice"] = "auto"
        return self._parse(await self._post(body))

    async def complete_structured(
        self, messages: Sequence[Message], json_schema: dict[str, Any], params: GenerationParams
    ) -> LlmResponse:
        """Schema-constrained completion via vLLM ``guided_json``."""
        body = self._base(messages, params)
        body["guided_json"] = json_schema
        return self._parse(await self._post(body))

    async def health(self) -> bool:
        """Ping ``/models``."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self._endpoint}/models")
                return resp.status_code == 200
        except httpx.HTTPError:
            return False
