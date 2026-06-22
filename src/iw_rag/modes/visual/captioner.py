"""VlmCaptioner — offline VLM client for visual asset captioning (WI-030 / Task 4b-fallback).

OFFLINE / batch only.  This module MUST NOT be imported by any hot-path module
(api/, agent/, retrieval/, generation/).  The two-clocks rule (CLAUDE.md, lld-
multimodal §3) places all VLM captioning in the offline/batch path — the
captioner is called once at ingest, its output is embedded by BGE-M3, and the
model is then unloaded (no runtime VRAM contention with Qwen3-14B).

The client is injected (LlmClient protocol) so tests supply a fake without any
network.  Endpoint and served model name come from config/models.yaml via
Settings — never hardcoded (model-agnostic by design).

Captioner failure on an asset → skip-and-log; the document continues indexing
via its other chunks (text, other tables/figures).  A failed caption never
raises to the caller — it returns None so the caller can skip gracefully.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from iw_rag.generation.llm import GenerationParams, LlmClient
from iw_rag.modes.visual.types import VisualAsset, VisualAssetType

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_TABLE_SYSTEM = (
    "You are an expert technical writer. Given an HTML table, produce a concise, "
    "information-dense plain-text description of the table's content, structure, "
    "and key values. Do not reproduce the HTML. Focus on facts."
)

_TABLE_USER = "Describe the following HTML table in plain text:\n\n{content}"

_FIGURE_SYSTEM = (
    "You are an expert technical writer. Given a reference to a figure, chart, or "
    "image, produce a concise, information-dense plain-text caption describing what "
    "the figure likely depicts based on any available context (alt text, surrounding "
    "markup). Be specific about visual elements, trends, and key values when inferable."
)

_FIGURE_USER = "Describe the following figure reference in plain text:\n\n{content}"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VlmCaptionerConfig:
    """Configuration for a VlmCaptioner instance.

    Attributes:
        model_role: Logical model role in ``config/models.yaml``
            (e.g. ``"visual_caption"``).  Used for audit and routing only;
            the actual served model name is passed in ``served_name``.
        served_name: The model identifier passed to the VLM endpoint
            (e.g. ``"Qwen/Qwen2.5-VL-7B-Instruct"``).  Comes from
            models.yaml — never hardcoded.
        max_tokens: Maximum tokens in the generated caption.
        temperature: Sampling temperature (0.0 for deterministic captions).
    """

    model_role: str
    served_name: str
    max_tokens: int = 512
    temperature: float = 0.0


# ---------------------------------------------------------------------------
# Caption result (pre-embed, no document_id needed yet)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CaptionResult:
    """The output of a successful captioning call before embedding and persistence.

    Carries the fields needed to construct a ``NewChunk`` once the caller knows
    the ``document_id`` and has obtained the dense embedding from the BGE-M3 path.

    Attributes:
        content: The VLM-generated caption text.
        content_type: ``"table"`` or ``"figure_caption"`` (determined by asset type).
        ordinal: Position among the asset's sibling chunks.
        section_path: Heading breadcrumb from the source (optional).
        acl_tags: Entitlement tags copied from the source asset manifest.
        metadata: Free-form metadata (source_page, asset_type, …).
    """

    content: str
    content_type: str
    ordinal: int
    section_path: str | None = None
    acl_tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# VlmCaptioner
# ---------------------------------------------------------------------------


class VlmCaptioner:
    """Qwen2.5-VL-7B captioner for table HTML and figure crops (OFFLINE/ingest-time).

    Turns visual assets into descriptive text that flows through the existing
    BGE-M3 hybrid spine, inheriting full ACL pre-filter, metadata, and rerank
    parity (lld-multimodal §1, fallback path).

    The VLM client is injected so tests supply a fake.  Endpoints come from
    ``config/models.yaml`` via Settings — model-agnostic.

    Attributes:
        served_name: The served model name used in GenerationParams.
    """

    def __init__(self, vlm_client: LlmClient, cfg: VlmCaptionerConfig) -> None:
        """Wire the captioner with its injected VLM client and config.

        Args:
            vlm_client: An ``LlmClient`` pointing at the batch vLLM endpoint
                for the visual caption model.  Tests inject a fake here.
            cfg: Captioner configuration (served name, max_tokens, etc.).
        """
        self._client = vlm_client
        self._cfg = cfg
        self.served_name = cfg.served_name

    async def caption(self, asset: VisualAsset) -> str | None:
        """Caption a visual asset by calling the VLM.

        Sends the asset content to the VLM using role-specific prompts and
        returns the raw caption text.  On any VLM error the failure is
        logged at WARNING level and ``None`` is returned — the caller should
        skip the asset and continue indexing the document.

        Args:
            asset: The visual element (table HTML or figure ref) to caption.

        Returns:
            The VLM-generated caption text, or ``None`` on failure.
        """
        messages = _build_messages(asset)
        params = GenerationParams(
            model=self._cfg.served_name,
            temperature=self._cfg.temperature,
            max_tokens=self._cfg.max_tokens,
        )
        try:
            response = await self._client.complete(messages, params)
            return response.text
        except Exception as exc:  # noqa: BLE001
            # skip-and-log: the page still indexes via its other chunks
            _log.warning(
                "VlmCaptioner: failed to caption asset from '%s' (page=%s): %s — skipping",
                asset.source_uri,
                asset.page,
                exc,
            )
            return None

    async def caption_to_chunk(
        self,
        asset: VisualAsset,
        ordinal: int,
        *,
        section_path: str | None = None,
    ) -> CaptionResult | None:
        """Caption an asset and return a pre-persistence CaptionResult.

        Calls :meth:`caption`; on failure (``None`` result) returns ``None``
        so the caller can skip the asset cleanly.

        The ``content_type`` is set based on the asset type:
        - ``VisualAssetType.TABLE`` → ``"table"``
        - ``VisualAssetType.FIGURE`` → ``"figure_caption"``

        ACL tags are copied directly from the asset (set by the ACL resolver,
        never from model output).

        The returned :class:`CaptionResult` does **not** carry a ``document_id``
        or ``dense_embedding`` — the caller obtains those from the ingestion
        pipeline and constructs the final ``NewChunk``.

        Args:
            asset: The visual element to caption.
            ordinal: The ordinal position of this chunk among the asset's peers.
            section_path: The heading breadcrumb for the chunk (optional).

        Returns:
            A :class:`CaptionResult` with content, content_type, and ACL tags,
            or ``None`` on VLM failure.
        """
        text = await self.caption(asset)
        if text is None:
            return None

        content_type = _asset_type_to_content_type(asset.asset_type)
        return CaptionResult(
            content=text,
            content_type=content_type,
            ordinal=ordinal,
            section_path=section_path,
            acl_tags=list(asset.acl_tags),
            metadata={"source_page": asset.page, "asset_type": asset.asset_type.value},
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_FIGURE_IMAGE_USER = (
    "Describe this figure/chart/image in concise, information-dense plain text. "
    "Transcribe any text, numbers, axis labels, legends, or table values you can "
    "read, and state the key trend or finding. Context (if any):\n\n{content}"
)


def _build_messages(asset: VisualAsset) -> list[dict[str, Any]]:
    """Construct the VLM chat messages for a given asset type.

    When the asset carries an actual image (``image_b64``), the user turn is an
    OpenAI-vision multimodal message (text + ``image_url`` data URI) so the VLM
    reads the pixels.  Otherwise it falls back to a text-only prompt describing the
    figure reference.

    Args:
        asset: The visual asset to caption.

    Returns:
        A list of chat message dicts (system + user) ready for the VLM.
    """
    if asset.image_b64:
        data_uri = f"data:{asset.image_mime};base64,{asset.image_b64}"
        return [
            {"role": "system", "content": _FIGURE_SYSTEM},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _FIGURE_IMAGE_USER.format(content=asset.content)},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            },
        ]
    if asset.asset_type == VisualAssetType.TABLE:
        return [
            {"role": "system", "content": _TABLE_SYSTEM},
            {"role": "user", "content": _TABLE_USER.format(content=asset.content)},
        ]
    # FIGURE (and any future types) — text-only reference prompt.
    return [
        {"role": "system", "content": _FIGURE_SYSTEM},
        {"role": "user", "content": _FIGURE_USER.format(content=asset.content)},
    ]


def _asset_type_to_content_type(asset_type: VisualAssetType) -> str:
    """Map a VisualAssetType to the canonical chunk content_type string.

    Args:
        asset_type: The visual asset type enum value.

    Returns:
        ``"table"`` for TABLE assets; ``"figure_caption"`` for FIGURE assets.
    """
    if asset_type == VisualAssetType.TABLE:
        return "table"
    return "figure_caption"
