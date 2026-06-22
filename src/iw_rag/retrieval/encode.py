"""Query embedding — a single pooled query vector for dense retrieval.

Unlike document ingestion (which needs token-level Late Chunking), a *query* is
a single short string, so a pooled embedding from the served infinity endpoint is
exactly right (and reuses the running embedder). Returns the 1024-d query vector.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import httpx

from iw_rag.domain.errors import RetrievalError


@runtime_checkable
class QueryEncoder(Protocol):
    """Encodes a query string into a dense embedding."""

    async def encode(self, query: str) -> list[float]:
        """Return the query's dense embedding."""
        ...


class InfinityQueryEncoder:
    """BGE-M3 query encoder via the infinity ``/embeddings`` endpoint."""

    def __init__(self, endpoint: str, model: str, timeout_s: float = 15.0) -> None:
        """Configure the embedder endpoint + model.

        Args:
            endpoint: Base URL of the infinity server.
            model: Served embedding model id (e.g. ``BAAI/bge-m3``).
            timeout_s: Per-request timeout.
        """
        self._endpoint = endpoint.rstrip("/")
        self._model = model
        self._timeout_s = timeout_s

    async def encode(self, query: str) -> list[float]:
        """Embed the query via infinity.

        Args:
            query: The user query string.

        Returns:
            The 1024-d query embedding.

        Raises:
            RetrievalError: On transport/decoding failure (embedding down is a
                hard error — unlike a reranker outage, which only degrades).
        """
        try:
            async with httpx.AsyncClient(timeout=self._timeout_s) as client:
                resp = await client.post(
                    f"{self._endpoint}/embeddings",
                    json={"model": self._model, "input": query},
                )
                resp.raise_for_status()
                payload = resp.json()
            embedding: list[float] = payload["data"][0]["embedding"]
            return embedding
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
            raise RetrievalError(f"query embedding failed: {exc}") from exc
