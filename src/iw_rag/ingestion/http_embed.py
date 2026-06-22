"""HTTP document embedder — pooled BGE-M3 vectors via the served Infinity endpoint.

The offline pipeline (:mod:`iw_rag.ingestion.embed`) does *true Late Chunking*
with a local transformers BGE-M3 (the heavy ``ingest`` dep group, GPU-resident).
The online document-management upload path runs inside the **API container**,
which has neither torch nor the model loaded — only the shared Infinity embed
endpoint (the same one the query encoder uses).  So uploaded documents are
embedded with a *pooled* per-chunk vector from that endpoint: one OpenAI-style
``/embeddings`` call per leaf-batch, returning the same 1024-d cosine-normalized
BGE-M3 space the corpus lives in.  Pooled-per-chunk is the standard RAG embedding;
Late Chunking is a quality refinement kept for the offline batch path.

This class implements the :class:`~iw_rag.ingestion.embed.EmbeddingClient`
protocol (``embed_document``) so the existing :class:`IngestionPipeline` drives it
unchanged — it simply ignores ``doc_token_ids`` (no whole-doc forward pass) and
embeds each leaf's text directly.
"""

from __future__ import annotations

from collections.abc import Sequence

import httpx

from iw_rag.domain.errors import EmbedError
from iw_rag.ingestion.chunk import ChunkNode
from iw_rag.ingestion.embed import EMBEDDING_DIMS, ChunkEmbedding, _l2_normalize


class InfinityDocumentEmbedder:
    """Pooled BGE-M3 document embedder over the Infinity ``/embeddings`` endpoint."""

    def __init__(
        self,
        endpoint: str,
        model: str = "BAAI/bge-m3",
        *,
        batch_size: int = 32,
        timeout_s: float = 60.0,
        normalize: bool = True,
    ) -> None:
        """Configure the endpoint + model.

        Args:
            endpoint: Base URL of the Infinity server (e.g. ``http://infinity:7997``).
            model: Served embedding model id.
            batch_size: Max leaf texts per ``/embeddings`` request.
            timeout_s: Per-request timeout.
            normalize: L2-normalize the returned vectors (cosine-ready).
        """
        self._endpoint = endpoint.rstrip("/")
        self._model = model
        self._batch_size = max(1, batch_size)
        self._timeout_s = timeout_s
        self._normalize = normalize

    async def embed_document(
        self, doc_token_ids: list[int], leaves: Sequence[ChunkNode]
    ) -> list[ChunkEmbedding]:
        """Embed every leaf's text via pooled Infinity embeddings.

        ``doc_token_ids`` is accepted for protocol parity with the Late-Chunking
        embedder but unused (this path has no whole-document forward pass).

        Args:
            doc_token_ids: Ignored (protocol parity).
            leaves: Leaf nodes whose ``content`` is embedded.

        Returns:
            One :class:`ChunkEmbedding` per leaf, in leaf order.

        Raises:
            EmbedError: On transport/decoding failure or a dimension mismatch.
        """
        if not leaves:
            return []
        texts = [leaf.content for leaf in leaves]
        vectors: list[list[float]] = []
        try:
            async with httpx.AsyncClient(timeout=self._timeout_s) as client:
                for start in range(0, len(texts), self._batch_size):
                    batch = texts[start : start + self._batch_size]
                    resp = await client.post(
                        f"{self._endpoint}/embeddings",
                        json={"model": self._model, "input": batch},
                    )
                    resp.raise_for_status()
                    payload = resp.json()
                    data = payload["data"]
                    if len(data) != len(batch):
                        raise EmbedError(
                            f"embedder returned {len(data)} vectors for {len(batch)} inputs"
                        )
                    for item in data:
                        vectors.append(list(item["embedding"]))
        except (httpx.HTTPError, KeyError, IndexError, ValueError, TypeError) as exc:
            raise EmbedError(f"document embedding failed: {exc}") from exc

        results: list[ChunkEmbedding] = []
        for leaf, vec in zip(leaves, vectors, strict=True):
            if len(vec) != EMBEDDING_DIMS:
                raise EmbedError(f"embedder returned dim {len(vec)} (expected {EMBEDDING_DIMS})")
            dense = _l2_normalize(vec) if self._normalize else vec
            results.append(ChunkEmbedding(leaf.local_id, dense, {}))
        return results

    async def health(self) -> bool:
        """Whether the Infinity embedder endpoint is reachable."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self._endpoint}/models")
                resp.raise_for_status()
        except httpx.HTTPError:
            return False
        return True
