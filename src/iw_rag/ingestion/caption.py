"""VLM image-caption engine for the upload path (visual mode, gated).

Bridges the document-upload path to the offline VLM captioner so an uploaded
image whose text OCR cannot read (a chart, a diagram, a photo of a whiteboard)
still becomes queryable: the VLM describes the pixels and that caption is embedded
as the document's text.

Two-clocks / offline-guard discipline: the captioner (``modes.visual.captioner``)
must never be imported by a hot-path module (api/agent/retrieval/generation).
This module lives in ``ingestion/`` (not a guarded dir) and *lazy-imports* the
captioner inside its methods, so importing it costs nothing until a caption is
actually requested.  It is gated by ``modes.visual.enabled`` — when off,
:func:`make_caption_engine` returns ``None`` and the upload path is unchanged.

The VLM endpoint + served name come from ``config/models.yaml`` (the
``visual_caption`` role) — never hardcoded.  When the VLM endpoint is unreachable
(the common case: the VLM is not co-resident with the serving stack), the engine
reports ``available() == False`` and the caller degrades gracefully.
"""

from __future__ import annotations

import base64
import logging
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import httpx

if TYPE_CHECKING:
    from iw_rag.settings import Settings

logger = logging.getLogger(__name__)


@runtime_checkable
class CaptionEngine(Protocol):
    """Describes a raster image with a local VLM (offline/batch-time model)."""

    async def caption_image(self, data: bytes, *, mime: str, hint: str = "") -> str | None:
        """Return a plain-text description of the image, or None on failure."""
        ...

    async def available(self) -> bool:
        """Whether the VLM endpoint is reachable right now."""
        ...


class VlmCaptionEngine:
    """Caption images via the configured visual-caption VLM (Qwen2.5-VL by default)."""

    def __init__(self, endpoint: str, served_name: str, *, max_tokens: int = 512) -> None:
        """Configure the VLM endpoint + served model.

        Args:
            endpoint: OpenAI-compatible base URL (``…/v1``) of the VLM server.
            served_name: The served model id (from the registry).
            max_tokens: Caption length budget.
        """
        self._endpoint = endpoint.rstrip("/")
        self._served_name = served_name
        self._max_tokens = max_tokens

    async def caption_image(self, data: bytes, *, mime: str, hint: str = "") -> str | None:
        """Caption an image's pixels with the VLM (multimodal); None on failure.

        Args:
            data: The raw image bytes.
            mime: The image MIME type (e.g. ``image/png``).
            hint: Optional textual context (filename, surrounding text).

        Returns:
            The caption text, or ``None`` if the VLM call failed.
        """
        # Lazy imports keep the offline captioner out of any hot-path import graph
        # (two-clocks rule) and cost nothing unless a caption is requested.
        from iw_rag.generation.llm import VllmClient  # noqa: PLC0415
        from iw_rag.modes.visual.captioner import (  # noqa: PLC0415
            VlmCaptioner,
            VlmCaptionerConfig,
        )
        from iw_rag.modes.visual.types import VisualAsset, VisualAssetType  # noqa: PLC0415

        asset = VisualAsset(
            asset_type=VisualAssetType.FIGURE,
            content=hint,
            source_uri="upload://image",
            image_b64=base64.b64encode(data).decode("ascii"),
            image_mime=mime,
        )
        captioner = VlmCaptioner(
            VllmClient(self._endpoint),
            VlmCaptionerConfig(
                model_role="visual_caption",
                served_name=self._served_name,
                max_tokens=self._max_tokens,
            ),
        )
        return await captioner.caption(asset)

    async def available(self) -> bool:
        """Whether the VLM endpoint answers ``/models`` quickly."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self._endpoint}/models")
                resp.raise_for_status()
        except httpx.HTTPError:
            return False
        return True


def make_caption_engine(settings: Settings) -> CaptionEngine | None:
    """Build the VLM caption engine when visual mode is enabled, else None.

    Returns ``None`` (no captioning) when ``modes.visual.enabled`` is False or the
    ``visual_caption`` model role is not in the registry — the upload path then
    relies on OCR alone and is otherwise unchanged.

    Args:
        settings: The validated settings.

    Returns:
        A :class:`CaptionEngine`, or ``None`` when visual mode is off/unconfigured.
    """
    if not settings.modes.visual.enabled:
        return None
    role = settings.modes.visual.caption.model_role
    try:
        entry = settings.models.get(role)
    except Exception:  # noqa: BLE001 - missing role → no captioning, not a crash
        logger.warning("visual mode enabled but model role '%s' is not in the registry", role)
        return None
    return VlmCaptionEngine(
        entry.endpoint,
        entry.served_name,
        max_tokens=settings.modes.visual.caption.max_tokens,
    )
