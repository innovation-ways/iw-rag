"""Embedding stage — BGE-M3 Late Chunking (whole-doc encode → pool per leaf).

True Late Chunking (lld-ingestion §2.4): the whole document is encoded in one
forward pass so every token attends to full-document context, then per-leaf
token spans are mean-pooled into a normalized 1024-d vector — zero extra LLM
calls. Documents longer than the model window are split into overlapping windows
(overlap ≥ max leaf size) and each leaf is pooled from the single window that
fully contains it (so every leaf is embedded exactly once).

On Blackwell the embedder is a **local BGE-M3** loaded via transformers (the
``ingest`` dep group), because the served infinity/TEI endpoints expose only a
pooled vector, not the token-level states Late Chunking needs (Q-TEI). The pure
windowing/pooling orchestration (:class:`LateChunkingEmbedder`) is unit-tested
with a fake :class:`TokenEmbedder`; the heavy model is lazy-loaded.

``sparse_weights`` are left empty in Phase 0 — ParadeDB BM25 is the default
sparse arm; BGE-M3 learned-sparse is the M-SPARSE experiment.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from iw_rag.domain.errors import EmbedError
from iw_rag.ingestion.chunk import ChunkNode, TokenSpan

EMBEDDING_DIMS = 1024


@dataclass(frozen=True, slots=True)
class ChunkEmbedding:
    """A leaf's embedding outputs.

    Attributes:
        local_id: The leaf's within-document id.
        dense: Normalized 1024-d BGE-M3 dense vector.
        sparse_weights: BGE-M3 learned-sparse term→weight map (empty in Phase 0).
    """

    local_id: str
    dense: list[float]
    sparse_weights: dict[str, float]


@dataclass(frozen=True, slots=True)
class EmbedConfig:
    """Late-Chunking embedding config.

    Attributes:
        window_tokens: Model context window (BGE-M3 = 8192).
        overlap_tokens: Window overlap; MUST be ≥ the max leaf size so every leaf
            is fully contained in some window.
        normalize: L2-normalize pooled vectors (cosine-ready).
    """

    window_tokens: int = 8192
    overlap_tokens: int = 512
    normalize: bool = True


@runtime_checkable
class TokenEmbedder(Protocol):
    """Encodes one window of token ids → per-token hidden vectors."""

    async def embed_window(self, token_ids: list[int]) -> list[list[float]]:
        """Return one vector per input token (len == len(token_ids))."""
        ...

    async def health(self) -> bool:
        """Whether the embedder is ready."""
        ...


@runtime_checkable
class EmbeddingClient(Protocol):
    """Produces one :class:`ChunkEmbedding` per leaf via Late Chunking."""

    async def embed_document(
        self, doc_token_ids: list[int], leaves: Sequence[ChunkNode]
    ) -> list[ChunkEmbedding]:
        """Embed all leaves with whole-document context."""
        ...

    async def health(self) -> bool:
        """Whether the underlying embedder is ready."""
        ...


def _mean(vectors: list[list[float]]) -> list[float]:
    """Element-wise mean of equal-length vectors."""
    n = len(vectors)
    dims = len(vectors[0])
    acc = [0.0] * dims
    for vec in vectors:
        for i, value in enumerate(vec):
            acc[i] += value
    return [value / n for value in acc]


def _l2_normalize(vec: list[float]) -> list[float]:
    """L2-normalize a vector (no-op for the zero vector)."""
    norm = math.sqrt(sum(value * value for value in vec))
    if norm == 0.0:
        return vec
    return [value / norm for value in vec]


def plan_windows(n_tokens: int, window: int, overlap: int) -> list[tuple[int, int]]:
    """Plan overlapping ``[start, end)`` windows covering ``n_tokens``.

    Args:
        n_tokens: Total document tokens.
        window: Window size (tokens).
        overlap: Overlap between consecutive windows (must be < window).

    Returns:
        Window ``[start, end)`` ranges; a single window when the doc fits.
    """
    if n_tokens <= window:
        return [(0, n_tokens)]
    step = max(1, window - overlap)
    windows: list[tuple[int, int]] = []
    start = 0
    while start < n_tokens:
        end = min(start + window, n_tokens)
        windows.append((start, end))
        if end == n_tokens:
            break
        start += step
    return windows


def _window_for(span: TokenSpan, windows: list[tuple[int, int]]) -> int:
    """Index of the first window fully containing ``span``.

    Raises:
        EmbedError: If no window fully contains the span (overlap too small).
    """
    for idx, (start, end) in enumerate(windows):
        if start <= span.start and span.end <= end:
            return idx
    raise EmbedError(
        f"leaf span {span} not fully contained in any window "
        f"(increase overlap_tokens ≥ max leaf size)"
    )


class LateChunkingEmbedder:
    """Pure Late-Chunking orchestration over an injected :class:`TokenEmbedder`."""

    def __init__(self, token_embedder: TokenEmbedder, cfg: EmbedConfig | None = None) -> None:
        """Bind the per-token embedder and config.

        Args:
            token_embedder: Encodes a window of token ids → per-token vectors.
            cfg: Windowing/normalization config.
        """
        self._embedder = token_embedder
        self._cfg = cfg or EmbedConfig()

    async def embed_document(
        self, doc_token_ids: list[int], leaves: Sequence[ChunkNode]
    ) -> list[ChunkEmbedding]:
        """Embed every leaf by pooling its span from its containing window.

        Args:
            doc_token_ids: The whole-document token stream (from the chunker).
            leaves: Leaf nodes, each carrying a ``token_span`` into the stream.

        Returns:
            One :class:`ChunkEmbedding` per leaf, in leaf order.

        Raises:
            EmbedError: If a leaf has no span or its span isn't window-contained.
        """
        windows = plan_windows(
            len(doc_token_ids), self._cfg.window_tokens, self._cfg.overlap_tokens
        )
        # Group leaf indices by the window that will embed them (embed each once).
        by_window: dict[int, list[int]] = {}
        for i, leaf in enumerate(leaves):
            if leaf.token_span is None:
                raise EmbedError(f"leaf {leaf.local_id} has no token_span")
            by_window.setdefault(_window_for(leaf.token_span, windows), []).append(i)

        results: list[ChunkEmbedding | None] = [None] * len(leaves)
        for wi, leaf_idxs in by_window.items():
            w_start, w_end = windows[wi]
            token_vecs = await self._embedder.embed_window(doc_token_ids[w_start:w_end])
            if len(token_vecs) != w_end - w_start:
                raise EmbedError(
                    f"token embedder returned {len(token_vecs)} vecs for {w_end - w_start} tokens"
                )
            for i in leaf_idxs:
                span = leaves[i].token_span
                assert span is not None  # guarded above
                local = token_vecs[span.start - w_start : span.end - w_start]
                if not local:
                    raise EmbedError(f"empty span for leaf {leaves[i].local_id}")
                dense = _mean(local)
                if self._cfg.normalize:
                    dense = _l2_normalize(dense)
                results[i] = ChunkEmbedding(leaves[i].local_id, dense, {})
        return [r for r in results if r is not None]

    async def health(self) -> bool:
        """Delegate readiness to the token embedder."""
        return await self._embedder.health()


class BgeM3TokenEmbedder:
    """Local BGE-M3 per-token embedder via transformers (lazy-loaded, offline)."""

    def __init__(self, model_name: str = "BAAI/bge-m3", device: str | None = None) -> None:
        """Configure the model name/device; the model loads on first use.

        Args:
            model_name: HF repo id of the embedding model.
            device: Torch device (``cuda``/``cpu``); auto-detected when None.
        """
        self._model_name = model_name
        self._device = device
        self._model: Any = None
        self._tokenizer: Any = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import torch  # noqa: PLC0415
            from transformers import AutoModel, AutoTokenizer  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - needs the ingest group
            raise EmbedError(
                "BgeM3TokenEmbedder requires the 'ingest' dependency group "
                "(uv sync --group ingest)."
            ) from exc
        self._device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        self._model = AutoModel.from_pretrained(self._model_name).to(self._device).eval()

    async def embed_window(self, token_ids: list[int]) -> list[list[float]]:
        """Forward a window of token ids → per-token last-hidden-state vectors.

        Args:
            token_ids: The window's token ids (no special tokens).

        Returns:
            One vector per input token (special tokens stripped).
        """
        self._ensure_loaded()
        import torch  # noqa: PLC0415

        assert self._model is not None
        ids = torch.tensor([token_ids], device=self._device)
        with torch.no_grad():
            out = self._model(input_ids=ids, attention_mask=torch.ones_like(ids))
        hidden = out.last_hidden_state[0]  # [seq, dim]
        return [row.tolist() for row in hidden.cpu()]

    async def health(self) -> bool:
        """Attempt to load the model; True if it loads."""
        try:
            self._ensure_loaded()
        except EmbedError:
            return False
        return True


class BgeM3Tokenizer:
    """BGE-M3 tokenizer adapter (lazy transformers) implementing ``chunk.Tokenizer``.

    The SAME tokenization the chunker uses to build ``doc_token_ids``/spans is
    what the embedder forwards through the model, so leaf spans align exactly.
    Special tokens are omitted (content-token pooling).
    """

    def __init__(self, model_name: str = "BAAI/bge-m3") -> None:
        """Configure the model name; the tokenizer loads on first use."""
        self._model_name = model_name
        self._tok: Any = None

    def _ensure(self) -> None:
        if self._tok is not None:
            return
        try:
            from transformers import AutoTokenizer  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - needs the ingest group
            raise EmbedError(
                "BgeM3Tokenizer requires the 'ingest' dependency group (uv sync --group ingest)."
            ) from exc
        self._tok = AutoTokenizer.from_pretrained(self._model_name)

    def encode(self, text: str) -> list[int]:
        """Encode text → token ids (no special tokens)."""
        self._ensure()
        ids: list[int] = self._tok(text, add_special_tokens=False)["input_ids"]
        return ids

    def decode(self, ids: list[int]) -> str:
        """Decode token ids → text."""
        self._ensure()
        return str(self._tok.decode(ids))


def make_bge_m3_embedder(cfg: EmbedConfig | None = None) -> LateChunkingEmbedder:
    """Build the default local BGE-M3 Late-Chunking embedder."""
    return LateChunkingEmbedder(BgeM3TokenEmbedder(), cfg)
