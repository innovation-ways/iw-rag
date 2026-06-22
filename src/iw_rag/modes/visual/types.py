"""Visual mode types — fallback and late-interaction paths (WI-030/WI-031).

Contains:
- WI-030 (fallback): VisualAsset, VisualAssetType, VisualCaptionConfig —
  OFFLINE/batch only, consumed by VlmCaptioner.
- WI-031 (late-interaction): VisualBackend, VisualRetrieverConfig,
  VisualIndexConfig, VisualPageRecord — the late-interaction store/retriever
  configuration types.

Two-clocks rule: VisualAsset / VisualAssetType / VisualCaptionConfig are
offline-only. VisualRetrieverConfig is used by the hot-path retriever.
VisualIndexConfig is offline-only (used by VisualIndexer).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal


class VisualAssetType(StrEnum):
    """The kind of visual element to be captioned.

    ``TABLE`` — rendered table HTML (from the ingestion table-extraction step).
    ``FIGURE`` — a cropped figure image or image-reference (figure crop).
    """

    TABLE = "table"
    FIGURE = "figure"


@dataclass(frozen=True, slots=True)
class VisualAsset:
    """A single visual element extracted during ingestion that needs a caption.

    Passed to :class:`~iw_rag.modes.visual.captioner.VlmCaptioner.caption` at
    batch / offline time.  Never constructed on the hot path.

    Attributes:
        asset_type: Whether the element is a table or figure.
        content: The raw content — HTML string for tables, image ref/data-URI
            or alt text for figures.  Passed verbatim to the VLM prompt.
        source_uri: The document URI this asset was extracted from.  Used to
            propagate ``acl_tags`` from the source-system manifest; NEVER
            derived from the content or an LLM response.
        page: The 1-based page number within the source document (when known).
        acl_tags: Entitlement tags copied from the source document manifest.
            Set by the ACL resolver, never by model output.
        metadata: Optional free-form metadata (bbox, caption-hint, etc.).
    """

    asset_type: VisualAssetType
    content: str
    source_uri: str
    page: int | None = None
    acl_tags: list[str] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)
    # WI-DOCMGMT/visual: the figure's actual image, base64-encoded, so the VLM
    # reads the *pixels* (charts, scanned invoices, diagrams) rather than just an
    # alt-text reference.  When set, the captioner sends an OpenAI-vision
    # multimodal message; when None it falls back to text-only prompting.
    image_b64: str | None = None
    image_mime: str = "image/png"


@dataclass(frozen=True)
class VisualCaptionConfig:
    """Configuration for the visual caption mode (modes.visual).

    Loaded from ``config/default.yaml`` under the ``modes.visual.caption`` block
    and mapped to Settings via :class:`~iw_rag.settings.VisualModeSettings`.

    Attributes:
        model_role: Logical model role name in ``config/models.yaml``
            (default ``"visual_caption"`` → Qwen2.5-VL-7B-Instruct).
        max_assets_per_doc: Maximum visual assets captioned per document;
            ``0`` means unlimited.
        max_tokens: Maximum tokens in the generated caption.
    """

    model_role: str = "visual_caption"
    max_assets_per_doc: int = 0
    max_tokens: int = 512


# ---------------------------------------------------------------------------
# WI-031 — Late-interaction types
# ---------------------------------------------------------------------------

#: Backend enum for the multi-vector store (T1 trigger, ADR-007/ADR-010).
VisualBackend = Literal["postgres_candidates", "qdrant"]


@dataclass(frozen=True)
class VisualRetrieverConfig:
    """Configuration for VisualRetriever — the hot-path two-stage retriever.

    Loaded from ``config/default.yaml`` under ``modes.visual`` and used by
    :class:`~iw_rag.modes.visual.retriever.VisualRetriever`.

    Attributes:
        top_k: Final number of pages returned (after MaxSim rerank).
        prefetch_n: Stage-1 candidate budget (pooled/binary prefetch).
        pooled_vectors_per_page: Number of pooled vectors stored per page
            (canonical default ≈38, per R-00107/F3).
        backend: The multi-vector store backend.  ``"postgres_candidates"``
            (default) uses a mean-pooled single-vector HNSW proxy for stage-1
            and in-process MaxSim for stage-2.  ``"qdrant"`` is the headline
            trigger (ADR-007/T1); fails fast at startup if unreachable.
        model_role: Logical model role for the ColVision encoder in models.yaml.
    """

    top_k: int = 8
    prefetch_n: int = 200
    pooled_vectors_per_page: int = 38
    backend: VisualBackend = "postgres_candidates"
    model_role: str = "visual_retrieval"


@dataclass(frozen=True)
class VisualIndexConfig:
    """Configuration for VisualIndexer — the OFFLINE batch indexer.

    OFFLINE/batch-only. Used by :class:`~iw_rag.modes.visual.indexer.VisualIndexer`
    in ``scripts/index_visual.py``. MUST NOT be imported by hot-path modules.

    Attributes:
        pooled_vectors_per_page: Target number of pooled vectors per page
            (canonical ≈38, per R-00107/F3).
        model_role: Logical model role for the ColVision encoder in models.yaml.
        backend: Which multi-vector store to upsert into.
    """

    pooled_vectors_per_page: int = 38
    model_role: str = "visual_retrieval"
    backend: VisualBackend = "postgres_candidates"


@dataclass(frozen=True, slots=True)
class VisualPageRecord:
    """A visual page ready for indexing or retrieval.

    Produced by the ingestion pipeline (offline) and consumed by
    :class:`~iw_rag.modes.visual.indexer.VisualIndexer`.  NEVER constructed
    on the hot path.

    Attributes:
        source_chunk_id: The ``chunks.id`` this page was extracted from
            (back-ref for ACL resolution and citation grounding).
        source_uri: The document URI.
        page: 1-based page number within the document.
        image: The rendered page image (bytes or path reference, passed to
            the encoder).  May be raw bytes or a data-URI string.
        text: The extracted text of this page (R-00102 extraction, used as
            ``content`` in the ``RetrievedChunk``).
        acl_tags: Entitlement tags copied from the source document manifest.
            Set by the ACL resolver, never from model output.
        bbox: Optional bounding box for the full page in the source PDF
            coordinate system.
        metadata: Free-form per-page metadata.
    """

    source_chunk_id: str
    source_uri: str
    page: int
    image: bytes | str  # raw bytes or path/data-URI
    text: str = ""
    acl_tags: list[str] = field(default_factory=list)
    bbox: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)
