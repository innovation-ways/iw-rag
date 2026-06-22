"""iw_rag.modes.visual — visual RAG modes (WI-030 fallback + WI-031 late-interaction).

WI-030 (fallback): offline VLM captioner that turns table HTML / figure crops
into descriptive text, ingested through the BGE-M3 hybrid spine.

WI-031 (late-interaction): ColVision encoder → pooled+binary multi-vector store
→ two-stage MaxSim retrieve → RetrievalResult.

Two-clocks rule (CLAUDE.md):
- OFFLINE/batch only: VlmCaptioner, VisualIndexer, ColVisionEncoder.encode_pages.
- HOT-PATH allowed: VisualRetriever, ColVisionEncoder.encode_query.
- This __init__ exports the hot-path retriever and offline types separately.
  Hot-path imports: VisualRetriever, ColVisionEncoder, errors.
  Offline imports: VlmCaptioner, VisualIndexer — MUST NOT appear in agent/api/retrieval.
"""

# ---------------------------------------------------------------------------
# Offline-only exports (WI-030 fallback path)
# ---------------------------------------------------------------------------
from iw_rag.modes.visual.captioner import VlmCaptioner, VlmCaptionerConfig

# ---------------------------------------------------------------------------
# WI-031 late-interaction: hot-path retriever + errors (allowed on /query)
# ---------------------------------------------------------------------------
from iw_rag.modes.visual.errors import (
    VisualEncodeUnavailable,
    VisualModeError,
    VisualRouteTimeout,
    VisualStoreUnavailable,
)
from iw_rag.modes.visual.retriever import VisualRetriever
from iw_rag.modes.visual.types import (
    VisualAsset,
    VisualAssetType,
    VisualCaptionConfig,
    VisualIndexConfig,
    VisualPageRecord,
    VisualRetrieverConfig,
)

__all__ = [
    # WI-030 fallback (OFFLINE-only)
    "VlmCaptioner",
    "VlmCaptionerConfig",
    "VisualAsset",
    "VisualAssetType",
    "VisualCaptionConfig",
    # WI-031 late-interaction hot-path
    "VisualRetriever",
    "VisualRetrieverConfig",
    # WI-031 late-interaction offline
    "VisualIndexConfig",
    "VisualPageRecord",
    # WI-031 errors
    "VisualEncodeUnavailable",
    "VisualModeError",
    "VisualRouteTimeout",
    "VisualStoreUnavailable",
]
