"""ColVisionEncoder — transient ColQwen2.5/ColNomic-3B encoder (WI-031 / Task 4b).

HOT-PATH for encode_query (per-query, within VisualRetriever).
OFFLINE for encode_pages (ingest-only, within VisualIndexer).

Two-clocks rule (CLAUDE.md):
- encode_pages is called ONLY by VisualIndexer (OFFLINE/batch).
- encode_query is called by VisualRetriever (hot path per-query).
- The encoder is TRANSIENT: loaded, used, and released — never co-resident
  with Qwen3-14B (R-00103/R-00107/F5).

The client is injected (LlmClient protocol) so tests supply a fake without
any network or GPU.  Endpoints come from config/models.yaml via Settings.

Model-agnostic (CONVENTIONS.md): swapping ColQwen2.5 → ColNomic-3B is a
config change in models.yaml, never a code change.

References:
- task-4b-visual-rag-route §3 (encoder design)
- lld-multimodal §4 (ColVision encoder + transient loading)
- ADR-010 (ColQwen2.5/ColNomic-3B, retrieval-only, local-only)
- R-00107 F5 (VRAM rule, transient encoder)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from iw_rag.modes.visual.errors import VisualEncodeUnavailable

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ColVisionEncoderConfig:
    """Configuration for the ColVisionEncoder.

    Attributes:
        model_role: Logical model role in models.yaml (e.g. ``"visual_retrieval"``).
        served_name: The model identifier on the serving engine
            (e.g. ``"vidore/colqwen2.5-v0.2"``).  Comes from models.yaml.
        endpoint: The HTTP endpoint for the encoder service.
        dims: Embedding dimensionality.
    """

    model_role: str
    served_name: str
    endpoint: str
    dims: int = 128


class ColVisionEncoder:
    """Transient ColQwen2.5/ColNomic-3B encoder for visual late-interaction retrieval.

    Encodes page images (ingest) and query text (per-query).  The encoder is
    loaded for the operation, then released — never co-resident with Qwen3-14B.

    The underlying client is injected so tests supply a fake without GPU/network.
    In production, the client issues HTTP requests to the transformers-served
    ColVision endpoint.

    Attributes:
        served_name: The served model name (from config/models.yaml).
    """

    def __init__(self, client: Any, cfg: ColVisionEncoderConfig) -> None:
        """Wire the encoder with its injected client and config.

        Args:
            client: An HTTP client (or any duck-type with async methods)
                for the ColVision endpoint.  Tests inject a fake here.
            cfg: Encoder configuration (served name, endpoint, dims).
        """
        self._client = client
        self._cfg = cfg
        self.served_name = cfg.served_name

    async def encode_pages(self, images: list[bytes | str]) -> list[list[list[float]]]:
        """Encode a batch of page images into multi-vectors (OFFLINE/ingest only).

        Each image produces a list of patch vectors (one per visual token/patch),
        forming the raw multi-vector representation before pooling and quantization.

        This method is called ONLY by VisualIndexer at ingest time.  It MUST NOT
        be called from the hot-path retrieve flow (two-clocks rule).

        Args:
            images: A list of page images, each as raw bytes or a data-URI string.

        Returns:
            A list of per-image patch-vector matrices.  Each element is a
            ``list[list[float]]`` of shape ``[n_patches, dims]``.

        Raises:
            VisualEncodeUnavailable: If the encoder endpoint is unreachable,
                OOM, or returns an unexpected response.
        """
        try:
            # In production, this calls the transformers-served endpoint.
            # The client is injected; tests supply a fake.
            return await self._client.encode_pages(images)  # type: ignore[no-any-return]
        except VisualEncodeUnavailable:
            raise
        except Exception as exc:
            _log.warning(
                "ColVisionEncoder.encode_pages: encoder call failed for %d images: %s",
                len(images),
                exc,
            )
            raise VisualEncodeUnavailable(f"encode_pages failed: {exc}") from exc

    async def encode_query(self, query: str) -> list[list[float]]:
        """Encode a query string into query-token vectors (hot-path, per-query).

        Each call loads the encoder transiently; the caller (VisualRetriever)
        is responsible for not co-residing with the generator.

        Args:
            query: The user query string.

        Returns:
            A list of query-token vectors — ``list[list[float]]`` of shape
            ``[n_query_tokens, dims]``.

        Raises:
            VisualEncodeUnavailable: If the encoder is unavailable (OOM, GPU
                busy, timeout, or any unexpected error).
        """
        try:
            return await self._client.encode_query(query)  # type: ignore[no-any-return]
        except VisualEncodeUnavailable:
            raise
        except Exception as exc:
            _log.warning(
                "ColVisionEncoder.encode_query: encoder call failed for query %r: %s",
                query[:80],
                exc,
            )
            raise VisualEncodeUnavailable(f"encode_query failed: {exc}") from exc
