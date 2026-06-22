"""Unit tests for :class:`ConclusionWriter` and edge branches in the conclusion module.

Covers the async ``write`` method via a fake chat client implementing the
``_ChatClientLike`` protocol, the NaN/invalid-input edge in ``_f``, and the
zero-denominator edge in ``_pct`` (lines 41-42 and 77 of conclusion.py).

No DB, no network, no real LLM.  The asyncio_mode = "auto" project setting
(pyproject.toml) drives async test execution without any per-test decorator.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

from iw_rag.eval.benchmark.conclusion import (
    ConclusionWriter,
    build_basis,
    summarize_report,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _row(gen: str, faith: float, vram: float, e2e: float) -> dict[str, object]:
    """Build a minimal benchmark_reports row for summarize_report.

    Args:
        gen: Generator model name (also used as report_label).
        faith: Faithfulness mean score.
        vram: VRAM peak in MB.
        e2e: End-to-end p99 latency in seconds.

    Returns:
        A dict that summarize_report accepts.
    """
    return {
        "report_label": gen,
        "model_config": {"generator": gen},
        "faithfulness_mean": faith,
        "ndcg_at_5": 0.5,
        "latency": {"e2e_p99_s": e2e, "per_stage_ms": {"retrieve_generate_ms": 1000.0}},
        "resource": {"gpu": {"vram_peak_mb": vram, "vram_incremental_mb": vram - 1000}},
    }


class _FakeResponse:
    """Stub whose ``text`` attribute the writer reads via getattr."""

    def __init__(self, text: str) -> None:
        self.text = text


class _FakeChatClient:
    """Minimal ``_ChatClientLike`` implementation backed by an AsyncMock.

    Returns a ``_FakeResponse`` carrying the configured text so that
    ``ConclusionWriter.write`` can harvest ``resp.text``.

    Args:
        text: The conclusion text the fake should return.
    """

    def __init__(self, text: str) -> None:
        self._response = _FakeResponse(text)
        self.complete = AsyncMock(return_value=self._response)


class _ErrorChatClient:
    """Fake client whose ``complete`` always raises to exercise the degrade path."""

    async def complete(self, messages: Any, params: Any) -> Any:
        raise RuntimeError("vllm unavailable")


# ---------------------------------------------------------------------------
# ConclusionWriter.write — happy path
# ---------------------------------------------------------------------------


async def test_writer_write_returns_llm_content_stripped() -> None:
    """Happy-path: write returns the fake client's text, stripped of whitespace."""
    client = _FakeChatClient("  This config trades VRAM for faithfulness.  ")
    writer = ConclusionWriter(chat_client=client, model="selene-2b", max_tokens=200)
    this = summarize_report(_row("qwen3-14b", 0.82, 24000, 2.1))
    comparables = [summarize_report(_row("qwen3-8b", 0.71, 16000, 1.7))]
    result = await writer.write(this, comparables)
    assert result == "This config trades VRAM for faithfulness."


async def test_writer_write_calls_client_with_system_and_user_messages() -> None:
    """The client receives exactly two messages: system role, then user role."""
    client = _FakeChatClient("The conclusion text.")
    writer = ConclusionWriter(chat_client=client, model="selene-2b")
    this = summarize_report(_row("qwen3-14b", 0.82, 24000, 2.1))
    await writer.write(this, [])
    call_args = client.complete.call_args
    messages = call_args[0][0]
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"


async def test_writer_write_passes_configured_model_and_max_tokens() -> None:
    """GenerationParams forwarded to complete carry the writer's model + max_tokens."""
    client = _FakeChatClient("conclusion")
    writer = ConclusionWriter(chat_client=client, model="judge-selene-7b", max_tokens=512)
    this = summarize_report(_row("qwen3-14b", 0.80, 20000, 1.8))
    await writer.write(this, [])
    _, params = client.complete.call_args[0]
    assert params.model == "judge-selene-7b"
    assert params.max_tokens == 512


# ---------------------------------------------------------------------------
# ConclusionWriter.write — degrade-to-None paths
# ---------------------------------------------------------------------------


async def test_writer_write_degrades_to_none_when_client_raises() -> None:
    """Any exception from complete must produce None (degrade; never fail the run)."""
    writer = ConclusionWriter(chat_client=_ErrorChatClient(), model="selene-2b")
    this = summarize_report(_row("qwen3-14b", 0.80, 20000, 2.0))
    result = await writer.write(this, [])
    assert result is None


async def test_writer_write_degrades_to_none_when_text_is_empty_string() -> None:
    """A response whose text is an empty string causes write to return None."""
    client = _FakeChatClient("")
    writer = ConclusionWriter(chat_client=client, model="selene-2b")
    this = summarize_report(_row("qwen3-14b", 0.80, 20000, 2.0))
    result = await writer.write(this, [])
    assert result is None


async def test_writer_write_degrades_to_none_when_text_is_whitespace_only() -> None:
    """A response whose text is all whitespace strips to empty, returning None."""
    client = _FakeChatClient("   \n\t  ")
    writer = ConclusionWriter(chat_client=client, model="selene-2b")
    this = summarize_report(_row("qwen3-14b", 0.80, 20000, 2.0))
    result = await writer.write(this, [])
    assert result is None


# ---------------------------------------------------------------------------
# ConclusionWriter constructor
# ---------------------------------------------------------------------------


def test_writer_stores_model_and_max_tokens() -> None:
    """Constructor correctly binds model, client, and max_tokens attributes."""
    client = _FakeChatClient("x")
    writer = ConclusionWriter(chat_client=client, model="judge-selene", max_tokens=128)
    assert writer._model == "judge-selene"
    assert writer._max_tokens == 128
    assert writer._client is client


# ---------------------------------------------------------------------------
# _f edge branches: NaN and unconvertable input (lines 41-42)
# ---------------------------------------------------------------------------


def test_summarize_report_treats_nan_faithfulness_as_none() -> None:
    """NaN faithfulness (via float('nan')) is coerced to None, not propagated."""
    row = _row("qwen3-14b", 0.80, 20000, 2.0)
    row["faithfulness_mean"] = float("nan")
    summary = summarize_report(row)
    assert summary["faithfulness"] is None


def test_summarize_report_treats_string_faithfulness_as_none() -> None:
    """Non-numeric string faithfulness degrades to None (TypeError path in _f)."""
    row = _row("qwen3-14b", 0.80, 20000, 2.0)
    row["faithfulness_mean"] = "not-a-number"
    summary = summarize_report(row)
    assert summary["faithfulness"] is None


def test_summarize_report_treats_none_faithfulness_as_none() -> None:
    """Explicit None faithfulness is preserved as None in the summary."""
    row = _row("qwen3-14b", 0.80, 20000, 2.0)
    row["faithfulness_mean"] = None
    summary = summarize_report(row)
    assert summary["faithfulness"] is None


# ---------------------------------------------------------------------------
# _pct edge branch: zero denominator (line 77)
# ---------------------------------------------------------------------------


def test_build_basis_delta_is_none_when_baseline_vram_is_zero() -> None:
    """When baseline vram_peak_mb is 0, vram_pct must be None (zero-division guard)."""
    this = summarize_report(_row("qwen3-14b", 0.80, 24000, 2.0))
    base_row = _row("qwen3-8b", 0.70, 0.0, 1.5)
    # Force vram_peak_mb to exactly 0 so the denominator is 0.
    base_row["resource"] = {"gpu": {"vram_peak_mb": 0.0, "vram_incremental_mb": 0.0}}
    base_summary = summarize_report(base_row)
    basis = build_basis(this, [base_summary])
    assert basis["deltas"]["vram_pct"] is None


def test_build_basis_delta_is_none_when_baseline_e2e_is_zero() -> None:
    """When baseline e2e_p99_s is 0, e2e_pct must be None (zero-division guard)."""
    this = summarize_report(_row("qwen3-14b", 0.80, 24000, 2.0))
    base_row = _row("qwen3-8b", 0.70, 16000, 0.0)
    base_row["latency"] = {"e2e_p99_s": 0.0, "per_stage_ms": {"retrieve_generate_ms": 1000.0}}
    base_summary = summarize_report(base_row)
    basis = build_basis(this, [base_summary])
    assert basis["deltas"]["e2e_pct"] is None


def test_build_basis_delta_is_none_when_this_metric_is_none() -> None:
    """When the current run's metric is None, the corresponding delta is None."""
    this = summarize_report(_row("qwen3-14b", 0.80, 24000, 2.0))
    this["faithfulness"] = None  # clear after summarize
    comparables = [summarize_report(_row("qwen3-8b", 0.70, 16000, 1.6))]
    basis = build_basis(this, comparables)
    assert basis["deltas"]["faithfulness_pct"] is None
