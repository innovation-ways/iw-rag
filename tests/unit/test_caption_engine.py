"""Unit tests for the upload-path VLM caption engine (ingestion/caption.py).

Covers the OFFLINE-degradation contract:
- ``make_caption_engine`` returns ``None`` when visual mode is disabled.
- ``make_caption_engine`` returns ``None`` (with a warning) when the model role is
  not in the registry.
- ``VlmCaptionEngine.available()`` returns ``False`` on transport / HTTP errors.
- ``VlmCaptionEngine.available()`` returns ``True`` when the VLM endpoint responds.
- ``caption_image`` happy path: canned VLM response, base64 encoding, hint/mime pass-through.

``caption_image`` lazy-imports ``VllmClient``, ``VlmCaptioner``, ``VlmCaptionerConfig``,
``VisualAsset``, and ``VisualAssetType`` inside the method body (two-clocks / offline guard
rule).  Because they are resolved at call time via ``importlib`` / ``sys.modules``, the
tests inject fakes through ``sys.modules`` rather than ``unittest.mock.patch``.
All ``httpx.AsyncClient`` calls are patched — no live VLM, no network.
"""

from __future__ import annotations

import base64
import logging
import sys
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from iw_rag.ingestion.caption import VlmCaptionEngine, make_caption_engine
from iw_rag.settings import (
    ModelEntry,
    ModelRegistry,
    ModesSettings,
    VisualCaptionSettings,
    VisualModeSettings,
)

# ---------------------------------------------------------------------------
# Helpers — Settings mock assembly
# ---------------------------------------------------------------------------


def _make_settings(
    *,
    visual_enabled: bool = True,
    model_role: str = "visual_caption",
    registry_has_role: bool = True,
    max_tokens: int = 512,
) -> MagicMock:
    """Build a minimal Settings-like mock for ``make_caption_engine``.

    Args:
        visual_enabled: Whether ``modes.visual.enabled`` is ``True``.
        model_role: The role name stored in ``visual.caption.model_role``.
        registry_has_role: Whether the registry contains the role.
        max_tokens: Caption token budget.

    Returns:
        A MagicMock shaped like the ``Settings`` object.
    """
    from iw_rag.domain.errors import ConfigError

    settings = MagicMock()
    settings.modes = MagicMock(spec=ModesSettings)
    settings.modes.visual = MagicMock(spec=VisualModeSettings)
    settings.modes.visual.enabled = visual_enabled

    cap_cfg = MagicMock(spec=VisualCaptionSettings)
    cap_cfg.model_role = model_role
    cap_cfg.max_tokens = max_tokens
    settings.modes.visual.caption = cap_cfg

    settings.models = MagicMock(spec=ModelRegistry)
    if registry_has_role:
        entry = ModelEntry(
            served_name="Qwen/Qwen2.5-VL-7B-Instruct",
            role="visual_caption",
            engine="vllm",
            endpoint="http://vlm:8003/v1",
        )
        settings.models.get = MagicMock(return_value=entry)
    else:
        settings.models.get = MagicMock(side_effect=ConfigError("missing role"))

    return settings


def _mock_httpx_get(
    *,
    status_code: int = 200,
    raise_error: type[httpx.HTTPError] | None = None,
) -> MagicMock:
    """Build a mock ``httpx.AsyncClient`` for GET requests.

    Args:
        status_code: HTTP status code for the response.
        raise_error: If set, the GET call raises this exception type instead.

    Returns:
        A MagicMock matching the ``async with httpx.AsyncClient() as client:`` pattern.
    """
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
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

    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    if raise_error is not None:
        mock_client.get = AsyncMock(side_effect=raise_error("connection refused"))
    else:
        mock_client.get = AsyncMock(return_value=resp)

    return mock_client


@contextmanager
def _inject_caption_deps(
    caption_result: str | None = "A caption.",
) -> Generator[dict[str, Any], None, None]:
    """Inject fake lazy-imported modules so ``caption_image`` never hits a real VLM.

    ``caption_image`` does three lazy imports inside its body:
    ``from iw_rag.generation.llm import VllmClient``,
    ``from iw_rag.modes.visual.captioner import VlmCaptioner, VlmCaptionerConfig``,
    ``from iw_rag.modes.visual.types import VisualAsset, VisualAssetType``.

    We install fake module objects in ``sys.modules`` so those imports resolve to
    our fakes without loading the real modules.

    Args:
        caption_result: The value the fake captioner returns from ``caption()``.

    Yields:
        A dict of captured call-kwargs for the fake ``VisualAsset`` constructor,
        so tests can inspect what arguments the production code passed.
    """
    captured: dict[str, Any] = {}

    # Fake captioner
    fake_captioner = MagicMock()
    fake_captioner.caption = AsyncMock(return_value=caption_result)

    # Fake VlmCaptioner class — returns the fake_captioner instance
    fake_captioner_cls = MagicMock(return_value=fake_captioner)
    fake_captioner_config_cls = MagicMock()

    # Fake VllmClient class
    fake_vllm_client_cls = MagicMock()

    # Fake VisualAsset — capture kwargs so tests can inspect them
    def _capture_asset(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return MagicMock()

    fake_asset_cls = MagicMock(side_effect=_capture_asset)

    # Fake VisualAssetType enum-like
    fake_vat = MagicMock()
    fake_vat.FIGURE = "FIGURE"

    # Build fake modules
    fake_llm_mod = MagicMock()
    fake_llm_mod.VllmClient = fake_vllm_client_cls

    fake_captioner_mod = MagicMock()
    fake_captioner_mod.VlmCaptioner = fake_captioner_cls
    fake_captioner_mod.VlmCaptionerConfig = fake_captioner_config_cls

    fake_types_mod = MagicMock()
    fake_types_mod.VisualAsset = fake_asset_cls
    fake_types_mod.VisualAssetType = fake_vat

    # Save originals (these may not be loaded yet)
    originals = {
        "iw_rag.generation.llm": sys.modules.get("iw_rag.generation.llm"),
        "iw_rag.modes.visual.captioner": sys.modules.get("iw_rag.modes.visual.captioner"),
        "iw_rag.modes.visual.types": sys.modules.get("iw_rag.modes.visual.types"),
    }
    sys.modules["iw_rag.generation.llm"] = fake_llm_mod
    sys.modules["iw_rag.modes.visual.captioner"] = fake_captioner_mod
    sys.modules["iw_rag.modes.visual.types"] = fake_types_mod

    try:
        yield captured
    finally:
        for key, val in originals.items():
            if val is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = val


# ---------------------------------------------------------------------------
# 1. make_caption_engine — returns None when visual disabled
# ---------------------------------------------------------------------------


def test_make_caption_engine_returns_none_when_visual_disabled() -> None:
    """make_caption_engine returns None when modes.visual.enabled is False."""
    settings = _make_settings(visual_enabled=False)
    result = make_caption_engine(settings)
    assert result is None


# ---------------------------------------------------------------------------
# 2. make_caption_engine — returns None when model role is missing
# ---------------------------------------------------------------------------


def test_make_caption_engine_returns_none_when_role_missing() -> None:
    """make_caption_engine returns None when the visual_caption role is absent."""
    settings = _make_settings(visual_enabled=True, registry_has_role=False)
    result = make_caption_engine(settings)
    assert result is None


def test_make_caption_engine_logs_warning_when_role_missing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """make_caption_engine logs a warning naming the missing model role."""
    settings = _make_settings(
        visual_enabled=True, model_role="visual_caption", registry_has_role=False
    )
    with caplog.at_level(logging.WARNING, logger="iw_rag.ingestion.caption"):
        make_caption_engine(settings)

    messages = [r.message for r in caplog.records]
    assert any("visual_caption" in m for m in messages), (
        f"Expected warning mentioning 'visual_caption'; got: {messages}"
    )


# ---------------------------------------------------------------------------
# 3. make_caption_engine — returns VlmCaptionEngine when enabled + role present
# ---------------------------------------------------------------------------


def test_make_caption_engine_returns_vlm_engine_when_enabled() -> None:
    """make_caption_engine returns a VlmCaptionEngine (not None) wired to the vlm endpoint."""
    settings = _make_settings(visual_enabled=True, registry_has_role=True)
    result = make_caption_engine(settings)
    assert isinstance(result, VlmCaptionEngine)
    assert result._endpoint == "http://vlm:8003/v1"


def test_make_caption_engine_uses_entry_endpoint() -> None:
    """make_caption_engine wires the engine with the registry ModelEntry endpoint."""
    settings = _make_settings(visual_enabled=True, registry_has_role=True)
    result = make_caption_engine(settings)
    assert isinstance(result, VlmCaptionEngine)
    assert result._endpoint == "http://vlm:8003/v1"


def test_make_caption_engine_uses_entry_served_name() -> None:
    """make_caption_engine wires the engine with the registry served_name."""
    settings = _make_settings(visual_enabled=True, registry_has_role=True)
    result = make_caption_engine(settings)
    assert isinstance(result, VlmCaptionEngine)
    assert result._served_name == "Qwen/Qwen2.5-VL-7B-Instruct"


def test_make_caption_engine_passes_max_tokens() -> None:
    """make_caption_engine propagates max_tokens from the visual caption config."""
    settings = _make_settings(visual_enabled=True, registry_has_role=True, max_tokens=256)
    result = make_caption_engine(settings)
    assert isinstance(result, VlmCaptionEngine)
    assert result._max_tokens == 256


# ---------------------------------------------------------------------------
# 4. VlmCaptionEngine.available() — transport error → False
# ---------------------------------------------------------------------------


async def test_available_returns_false_on_connect_error() -> None:
    """VlmCaptionEngine.available() returns False when the endpoint is unreachable."""
    engine = VlmCaptionEngine("http://vlm:8003/v1", "Qwen/Qwen2.5-VL-7B-Instruct")
    mock_client = _mock_httpx_get(raise_error=httpx.ConnectError)

    with patch("httpx.AsyncClient", return_value=mock_client):
        result = await engine.available()

    assert result is False


async def test_available_returns_false_on_http_status_error() -> None:
    """VlmCaptionEngine.available() returns False when /models returns non-2xx."""
    engine = VlmCaptionEngine("http://vlm:8003/v1", "Qwen/Qwen2.5-VL-7B-Instruct")
    mock_client = _mock_httpx_get(status_code=503)

    with patch("httpx.AsyncClient", return_value=mock_client):
        result = await engine.available()

    assert result is False


async def test_available_returns_false_on_timeout_error() -> None:
    """VlmCaptionEngine.available() returns False on connection timeout."""
    engine = VlmCaptionEngine("http://vlm:8003/v1", "Qwen/Qwen2.5-VL-7B-Instruct")
    mock_client = _mock_httpx_get(raise_error=httpx.TimeoutException)

    with patch("httpx.AsyncClient", return_value=mock_client):
        result = await engine.available()

    assert result is False


async def test_available_returns_true_on_200() -> None:
    """VlmCaptionEngine.available() returns True when /models responds HTTP 200."""
    engine = VlmCaptionEngine("http://vlm:8003/v1", "Qwen/Qwen2.5-VL-7B-Instruct")
    mock_client = _mock_httpx_get(status_code=200)

    with patch("httpx.AsyncClient", return_value=mock_client):
        result = await engine.available()

    assert result is True


async def test_available_checks_models_endpoint() -> None:
    """VlmCaptionEngine.available() GETs the /models path on the configured endpoint."""
    engine = VlmCaptionEngine("http://vlm:8003/v1", "some-model")
    mock_client = _mock_httpx_get(status_code=200)

    with patch("httpx.AsyncClient", return_value=mock_client):
        await engine.available()

    call_args = mock_client.get.call_args
    url = call_args.args[0] if call_args.args else call_args.kwargs.get("url", "")
    assert url == "http://vlm:8003/v1/models"


# ---------------------------------------------------------------------------
# 5. caption_image — happy path with canned VLM response
# ---------------------------------------------------------------------------


async def test_caption_image_returns_text_from_vlm() -> None:
    """caption_image returns the VLM captioner's text on success."""
    engine = VlmCaptionEngine(
        "http://vlm:8003/v1",
        "Qwen/Qwen2.5-VL-7B-Instruct",
        max_tokens=256,
    )

    with _inject_caption_deps("A revenue bar chart for Q3."):
        result = await engine.caption_image(b"PNG fake data", mime="image/png", hint="chart")

    assert result == "A revenue bar chart for Q3."


async def test_caption_image_passes_hint_to_asset() -> None:
    """caption_image propagates the hint argument to VisualAsset as its content."""
    engine = VlmCaptionEngine("http://vlm:8003/v1", "Qwen/Qwen2.5-VL-7B-Instruct")

    with _inject_caption_deps("A whiteboard.") as captured:
        await engine.caption_image(b"data", mime="image/jpeg", hint="whiteboard-photo")

    assert captured.get("content") == "whiteboard-photo"


async def test_caption_image_encodes_data_as_base64() -> None:
    """caption_image base64-encodes the raw image bytes into the VisualAsset."""
    engine = VlmCaptionEngine("http://vlm:8003/v1", "Qwen/Qwen2.5-VL-7B-Instruct")

    raw_bytes = b"\x89PNG\r\n\x1a\n"
    expected_b64 = base64.b64encode(raw_bytes).decode("ascii")

    with _inject_caption_deps("An image.") as captured:
        await engine.caption_image(raw_bytes, mime="image/png", hint="")

    assert captured.get("image_b64") == expected_b64


async def test_caption_image_passes_mime_to_asset() -> None:
    """caption_image passes the mime argument to VisualAsset.image_mime."""
    engine = VlmCaptionEngine("http://vlm:8003/v1", "Qwen/Qwen2.5-VL-7B-Instruct")

    with _inject_caption_deps("Caption.") as captured:
        await engine.caption_image(b"data", mime="image/webp", hint="")

    assert captured.get("image_mime") == "image/webp"


async def test_caption_image_returns_none_on_captioner_failure() -> None:
    """caption_image returns None when the VLM captioner returns None."""
    engine = VlmCaptionEngine("http://vlm:8003/v1", "Qwen/Qwen2.5-VL-7B-Instruct")

    with _inject_caption_deps(None):
        result = await engine.caption_image(b"data", mime="image/png", hint="")

    assert result is None


# ---------------------------------------------------------------------------
# 6. VlmCaptionEngine construction
# ---------------------------------------------------------------------------


def test_vlm_caption_engine_strips_trailing_slash() -> None:
    """VlmCaptionEngine strips a trailing slash from the endpoint URL."""
    engine = VlmCaptionEngine("http://vlm:8003/v1/", "some-model")
    assert engine._endpoint == "http://vlm:8003/v1"


def test_vlm_caption_engine_stores_max_tokens() -> None:
    """VlmCaptionEngine stores the max_tokens budget from the constructor."""
    engine = VlmCaptionEngine("http://vlm:8003/v1", "some-model", max_tokens=128)
    assert engine._max_tokens == 128


def test_vlm_caption_engine_stores_served_name() -> None:
    """VlmCaptionEngine stores the served model name from the constructor."""
    engine = VlmCaptionEngine("http://vlm:8003/v1", "Qwen/Qwen2.5-VL-7B-Instruct")
    assert engine._served_name == "Qwen/Qwen2.5-VL-7B-Instruct"
