"""Unit tests for the VllmClient request-body builder.

Focused on the ``enable_thinking`` plumbing (a perf fix): the
``generation.enable_thinking=false`` config must reach vLLM as
``chat_template_kwargs={"enable_thinking": false}`` so Qwen3 answers directly
instead of decoding (then discarding) a long ``<think>`` scratchpad.  Pure logic
— the private ``_base`` builder is exercised directly; no network.
"""

from __future__ import annotations

from iw_rag.generation.llm import GenerationParams, VllmClient

_PARAMS = GenerationParams(model="Qwen/Qwen3-14B-AWQ", temperature=0.1, max_tokens=256, seed=0)
_MSGS = [{"role": "user", "content": "hello"}]


def test_thinking_disabled_sends_chat_template_kwargs() -> None:
    client = VllmClient("http://vllm:8001/v1", enable_thinking=False)
    body = client._base(_MSGS, _PARAMS)
    assert body["chat_template_kwargs"] == {"enable_thinking": False}


def test_thinking_enabled_by_default_omits_kwarg() -> None:
    client = VllmClient("http://vllm:8001/v1")  # default enable_thinking=True
    body = client._base(_MSGS, _PARAMS)
    assert "chat_template_kwargs" not in body


def test_base_body_carries_core_fields_and_seed() -> None:
    client = VllmClient("http://vllm:8001/v1", enable_thinking=False)
    body = client._base(_MSGS, _PARAMS)
    assert body["model"] == "Qwen/Qwen3-14B-AWQ"
    assert body["max_tokens"] == 256
    assert body["temperature"] == 0.1
    assert body["seed"] == 0
