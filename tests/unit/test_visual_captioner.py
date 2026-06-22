"""Unit tests for modes/visual/captioner.py — TDD RED-first.

Covers: VlmCaptioner.caption returns text from a mocked VLM; captioner failure
on an asset → skip-and-log (no raise, returns None/empty + logs); table vs
figure asset → correct content_type on produced chunk; config-driven endpoint
(model-agnostic).  All use faked httpx clients — no live VLM.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from iw_rag.modes.visual.captioner import CaptionResult, VlmCaptioner, VlmCaptionerConfig
from iw_rag.modes.visual.types import VisualAsset, VisualAssetType

# ---------------------------------------------------------------------------
# Helpers / fakes
# ---------------------------------------------------------------------------


def _make_fake_client(response_text: str) -> MagicMock:
    """Build a mock VLM client returning a fixed chat completion response.

    Args:
        response_text: The caption text the fake VLM should return.

    Returns:
        A mock whose ``complete`` coroutine returns a fixed LlmResponse-like object.
    """
    from iw_rag.generation.llm import LlmResponse

    client = MagicMock()
    client.complete = AsyncMock(return_value=LlmResponse(text=response_text, usage={}))
    return client


def _make_erroring_client() -> MagicMock:
    """Build a mock VLM client that always raises a network error.

    Returns:
        A mock whose ``complete`` coroutine raises GenerationError.
    """
    from iw_rag.domain.errors import GenerationError

    client = MagicMock()
    client.complete = AsyncMock(side_effect=GenerationError("VLM endpoint unreachable"))
    return client


def _table_asset() -> VisualAsset:
    """Return a minimal table HTML asset for captioning.

    Returns:
        A VisualAsset of type TABLE.
    """
    return VisualAsset(
        asset_type=VisualAssetType.TABLE,
        content="<table><tr><td>Revenue</td><td>$10M</td></tr></table>",
        source_uri="corpus://finance/invoice_q3.md",
        page=1,
        acl_tags=["finance"],
    )


def _figure_asset() -> VisualAsset:
    """Return a minimal figure crop asset for captioning.

    Returns:
        A VisualAsset of type FIGURE.
    """
    return VisualAsset(
        asset_type=VisualAssetType.FIGURE,
        content="<img data-src='figure-1.png' alt='Revenue chart'/>",
        source_uri="corpus://finance/invoice_q3.md",
        page=2,
        acl_tags=["finance"],
    )


def _make_captioner(client: MagicMock) -> VlmCaptioner:
    """Wire a VlmCaptioner with a fake VLM client and test config.

    Args:
        client: The mock VLM client to inject.

    Returns:
        A configured VlmCaptioner.
    """
    cfg = VlmCaptionerConfig(
        model_role="visual_caption",
        served_name="Qwen/Qwen2.5-VL-7B-Instruct",
        max_tokens=256,
    )
    return VlmCaptioner(vlm_client=client, cfg=cfg)


# ---------------------------------------------------------------------------
# 1. Returns caption text from a mocked VLM
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_caption_returns_vlm_text_for_table() -> None:
    """VlmCaptioner.caption returns the VLM's text for a table asset."""
    expected = "A table showing Q3 revenue of $10M."
    client = _make_fake_client(expected)
    captioner = _make_captioner(client)

    result = await captioner.caption(_table_asset())

    assert result == expected


@pytest.mark.asyncio
async def test_caption_returns_vlm_text_for_figure() -> None:
    """VlmCaptioner.caption returns the VLM's text for a figure asset."""
    expected = "A bar chart displaying quarterly revenue growth."
    client = _make_fake_client(expected)
    captioner = _make_captioner(client)

    result = await captioner.caption(_figure_asset())

    assert result == expected


# ---------------------------------------------------------------------------
# 2. Captioner failure → skip-and-log (no raise, returns None/empty)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_caption_failure_returns_none_and_does_not_raise() -> None:
    """Captioner failure on an asset returns None and does not propagate the exception."""
    client = _make_erroring_client()
    captioner = _make_captioner(client)

    result = await captioner.caption(_table_asset())

    # Must not raise; returns None on failure (caller skips the asset)
    assert result is None


@pytest.mark.asyncio
async def test_caption_failure_logs_warning(caplog: pytest.LogCaptureFixture) -> None:
    """Captioner failure logs a warning with the asset source_uri."""
    import logging

    client = _make_erroring_client()
    captioner = _make_captioner(client)

    with caplog.at_level(logging.WARNING):
        await captioner.caption(_table_asset())

    # A warning mentioning the failed asset should have been logged
    assert any("corpus://finance/invoice_q3.md" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# 3. Correct content_type on produced chunk from asset type
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_caption_to_chunk_table_has_table_content_type() -> None:
    """Table asset produces a CaptionResult with content_type='table'."""
    caption_text = "Revenue table for Q3."
    client = _make_fake_client(caption_text)
    captioner = _make_captioner(client)

    result = await captioner.caption_to_chunk(_table_asset(), ordinal=0)

    assert result is not None
    assert isinstance(result, CaptionResult)
    assert result.content_type == "table"
    assert result.content == caption_text


@pytest.mark.asyncio
async def test_caption_to_chunk_figure_has_figure_caption_content_type() -> None:
    """Figure asset produces a CaptionResult with content_type='figure_caption'."""
    caption_text = "A bar chart of quarterly revenue."
    client = _make_fake_client(caption_text)
    captioner = _make_captioner(client)

    result = await captioner.caption_to_chunk(_figure_asset(), ordinal=1)

    assert result is not None
    assert isinstance(result, CaptionResult)
    assert result.content_type == "figure_caption"
    assert result.content == caption_text


@pytest.mark.asyncio
async def test_caption_to_chunk_inherits_acl_tags() -> None:
    """caption_to_chunk propagates the asset's acl_tags onto the CaptionResult."""
    client = _make_fake_client("Some caption.")
    captioner = _make_captioner(client)
    asset = _table_asset()  # acl_tags=["finance"]

    result = await captioner.caption_to_chunk(asset, ordinal=0)

    assert result is not None
    assert result.acl_tags == ["finance"]


@pytest.mark.asyncio
async def test_caption_to_chunk_failure_returns_none() -> None:
    """caption_to_chunk returns None when the VLM call fails (skip-and-log)."""
    client = _make_erroring_client()
    captioner = _make_captioner(client)

    result = await captioner.caption_to_chunk(_table_asset(), ordinal=0)

    assert result is None


# ---------------------------------------------------------------------------
# 4. Config-driven endpoint (model-agnostic)
# ---------------------------------------------------------------------------


def test_captioner_uses_config_served_name() -> None:
    """VlmCaptioner stores the served model name from config (model-agnostic)."""
    cfg = VlmCaptionerConfig(
        model_role="visual_caption",
        served_name="Qwen/Qwen2.5-VL-7B-Instruct",
        max_tokens=512,
    )
    captioner = VlmCaptioner(vlm_client=MagicMock(), cfg=cfg)

    assert captioner.served_name == "Qwen/Qwen2.5-VL-7B-Instruct"


def test_captioner_config_stores_model_role() -> None:
    """VlmCaptionerConfig preserves model_role for model-agnostic dispatch."""
    cfg = VlmCaptionerConfig(
        model_role="visual_caption",
        served_name="Qwen/Qwen2.5-VL-7B-Instruct",
        max_tokens=256,
    )
    assert cfg.model_role == "visual_caption"


@pytest.mark.asyncio
async def test_caption_sends_served_name_to_client() -> None:
    """VlmCaptioner passes its served model name in the GenerationParams."""
    from iw_rag.generation.llm import GenerationParams, LlmResponse

    captured_params: list[GenerationParams] = []
    client = MagicMock()

    async def _capture(messages: Any, params: GenerationParams) -> LlmResponse:
        captured_params.append(params)
        return LlmResponse(text="Caption text.", usage={})

    client.complete = AsyncMock(side_effect=_capture)
    cfg = VlmCaptionerConfig(
        model_role="visual_caption",
        served_name="Qwen/Qwen2.5-VL-7B-Instruct",
        max_tokens=128,
    )
    captioner = VlmCaptioner(vlm_client=client, cfg=cfg)

    await captioner.caption(_table_asset())

    assert len(captured_params) == 1
    assert captured_params[0].model == "Qwen/Qwen2.5-VL-7B-Instruct"
