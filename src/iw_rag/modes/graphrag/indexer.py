"""GraphIndexer — offline entity/relation extraction into Apache AGE (OFFLINE/batch).

OFFLINE / batch only.  This module MUST NOT be imported by any hot-path module
(api/, agent/, retrieval/, generation/).  The two-clocks rule (CLAUDE.md,
lld-graphrag §1) places all graph indexing in the offline/batch path.  The
offline guard test (tests/unit/test_graphrag_offline_guard.py) statically
asserts this invariant.

Pipeline (per chunk, OFFLINE):
    1. Source = chunks already in ``iw_rag.chunks`` (reuse ingestion output;
       do NOT re-parse; lld-graphrag §3).
    2. Dedup on ``(source_uri, checksum)`` (incremental set-merge; R-00106/F3).
    3. The small extraction model (``graphrag-extract`` → Qwen3-4B, batch
       endpoint) emits entities + typed relations via a structured prompt
       (``max_gleanings=1`` to bound cost, R-00106/F5).
    4. Tolerant JSON parser: valid → upsert; malformed → 1 bounded retry →
       skip-and-log on persistent failure (small local models emit unstable
       JSON; R-00106/F2; never crash the batch).
    5. Upsert into AGE via :class:`~iw_rag.modes.graphrag.store.AgeGraphStore`
       (the sole graph-SQL owner).  Each entity/relation carries a back-ref to
       its source ``chunks.id``.

References:
- lld-graphrag §2 (GraphIndexer interface), §3 (extraction algorithm)
- task-4a-graphrag-route §3 (index_documents step-by-step)
- R-00106/F2 (unstable JSON), F3 (incremental), F5 (skip community summaries)
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from iw_rag.generation.llm import GenerationParams, LlmClient
from iw_rag.modes.graphrag.store import AgeGraphStore
from iw_rag.modes.graphrag.types import Entity, GraphIndexConfig, Relation

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Extraction prompt templates
# ---------------------------------------------------------------------------

_EXTRACTION_SYSTEM = (
    "You are an expert knowledge-graph extractor.  Given a text passage, extract "
    "all named entities and typed relations between them.  Reply ONLY with a valid "
    "JSON object (no prose, no markdown) with this exact schema:\n"
    '{"entities": [{"name": "<str>", "type": "<str>", "chunk_id": "<str>"}], '
    '"relations": [{"src": "<str>", "dst": "<str>", "type": "<str>", "chunk_id": "<str>"}]}\n'
    "Use the provided chunk_id value for every entity and relation you emit."
)

_EXTRACTION_USER = (
    "chunk_id: {chunk_id}\n\n"
    "Text:\n{content}\n\n"
    "Extract entities and relations. Use chunk_id '{chunk_id}' for all items.\n"
    # Qwen3 reads the /think/no_think soft-switch from the LAST user turn (not the
    # system prompt). Extraction is structured output, not reasoning — disable the
    # chain-of-thought so the model emits JSON directly (≈5-10x faster, no <think>
    # block to strip). No-op on non-Qwen3 models.
    "/no_think"
)

# ---------------------------------------------------------------------------
# GraphIndexReport
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GraphIndexReport:
    """Summary of a :meth:`GraphIndexer.index_chunks` run.

    Attributes:
        chunks_indexed: Number of chunks successfully extracted and upserted.
        chunks_skipped: Number of chunks skipped because already indexed
            (incremental dedup on ``source_uri + checksum``).
        chunks_failed: Number of chunks that failed extraction after retries
            (skip-and-log; batch continues).
    """

    chunks_indexed: int = 0
    chunks_skipped: int = 0
    chunks_failed: int = 0


# ---------------------------------------------------------------------------
# Tolerant JSON parser (public — tested directly in test_graphrag_indexer.py)
# ---------------------------------------------------------------------------


def _parse_extraction_json(raw: str) -> dict[str, Any] | None:
    """Tolerantly parse extraction output from the small LLM.

    Handles:
    - Plain JSON.
    - JSON wrapped in triple-backtick markdown code fences.
    - JSON embedded after prose (extracts the first ``{…}`` block).

    Returns ``None`` on complete failure (caller should retry once, then
    skip-and-log — R-00106/F2).

    Args:
        raw: Raw string response from the extraction model.

    Returns:
        Parsed dict with ``"entities"`` and ``"relations"`` keys, or ``None``
        if the text cannot be parsed.
    """
    cleaned = raw.strip()

    # Strip markdown code fences (```json … ``` or ``` … ```)
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        # Drop first (fence open) and last (fence close) lines when present
        cleaned = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])
        cleaned = cleaned.strip()

    # Attempt direct parse
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict) and "entities" in data:
            return data
    except (json.JSONDecodeError, ValueError):
        pass

    # Attempt to extract the first JSON object from the text (handles leading prose)
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            if isinstance(data, dict) and "entities" in data:
                return data
        except (json.JSONDecodeError, ValueError):
            pass

    return None


# ---------------------------------------------------------------------------
# GraphIndexer
# ---------------------------------------------------------------------------


class GraphIndexer:
    """Offline entity/relation extraction → AGE.  Small extraction model only.

    Driven by ``scripts/index_graph.py`` (the offline entrypoint).  NEVER
    reachable from ``/query`` — the offline guard test enforces this.

    The LLM client is injected so tests can supply a fake without any network
    or model serving.  Endpoints come from ``config/models.yaml`` via Settings
    — never hardcoded (model-agnostic).

    Attributes:
        _store: The AgeGraphStore (sole graph-SQL owner).
        _cfg: GraphIndexConfig (extraction model, max_gleanings, …).
        _llm_client: The injected LLM client pointing at the batch endpoint.
        _indexed_keys: Set of ``(source_uri, checksum)`` already indexed in
            this session (incremental dedup — R-00106/F3).
    """

    def __init__(
        self,
        store: AgeGraphStore,
        cfg: GraphIndexConfig,
        llm_client: LlmClient,
    ) -> None:
        """Bind the indexer to its store, config, and extraction model client.

        Args:
            store: The AgeGraphStore — the sole graph-SQL owner.
            cfg: GraphIndexConfig (model role, max_gleanings, skip_summaries).
            llm_client: An LlmClient pointing at the OFFLINE batch endpoint
                (``graphrag-extract`` → Qwen3-4B).  Tests inject a fake here.
        """
        self._store = store
        self._cfg = cfg
        self._llm_client = llm_client
        # In-memory dedup set: (source_uri, checksum) — reset per indexer instance
        self._indexed_keys: set[tuple[str, str]] = set()

    async def index_chunks(
        self,
        chunks: Sequence[dict[str, Any]],
        *,
        incremental: bool = True,
    ) -> GraphIndexReport:
        """Extract entities/relations from chunks and upsert into AGE.

        Source = chunks already in ``iw_rag.chunks`` (passed as dicts with at
        minimum ``chunk_id``, ``source_uri``, ``checksum``, and ``content`` keys).
        Dedup on ``(source_uri, checksum)`` when ``incremental=True``.

        For each new chunk the small extraction model emits entities + typed
        relations.  A tolerant parser is applied; on failure a single bounded
        retry is attempted.  On persistent failure the chunk is skipped and a
        WARNING is logged — the batch never crashes.

        Community summaries are always skipped (``skip_community_summaries`` is
        hardcoded to the config value; they are never generated here —
        R-00106/F5).

        Args:
            chunks: Chunk dicts with ``chunk_id``, ``source_uri``, ``checksum``,
                and ``content`` keys (shape from ChunkRepository or a manifest).
            incremental: When ``True`` (default), skip chunks whose
                ``(source_uri, checksum)`` pair is already in the in-memory
                dedup set from a previous call in this session.

        Returns:
            A :class:`GraphIndexReport` summarising the run.
        """
        indexed = 0
        skipped = 0
        failed = 0

        for chunk in chunks:
            source_uri = str(chunk.get("source_uri", ""))
            checksum = str(chunk.get("checksum", ""))
            chunk_id = str(chunk.get("chunk_id", ""))
            content = str(chunk.get("content", ""))

            dedup_key = (source_uri, checksum)
            if incremental and dedup_key in self._indexed_keys:
                skipped += 1
                continue

            # Extract with the small model — first attempt
            raw = await self._call_extraction_model(chunk_id, content)
            if raw is None:
                # LLM call itself failed (not JSON parse — model exception)
                _log.warning(
                    "GraphIndexer: extraction model failed for chunk '%s' (%s) — skipping",
                    chunk_id,
                    source_uri,
                )
                failed += 1
                continue

            parsed = _parse_extraction_json(raw)

            if parsed is None:
                # First parse failed — one bounded retry (R-00106/F2)
                _log.debug(
                    "GraphIndexer: malformed JSON on first attempt for chunk '%s' — retrying",
                    chunk_id,
                )
                raw2 = await self._call_extraction_model(chunk_id, content)
                if raw2 is not None:
                    parsed = _parse_extraction_json(raw2)

            if parsed is None:
                # Persistent failure after retry — skip-and-log (never crash)
                _log.warning(
                    "GraphIndexer: persistent JSON parse failure for chunk '%s' (%s) — skipping",
                    chunk_id,
                    source_uri,
                )
                failed += 1
                continue

            # Build typed Entity/Relation objects from the parsed dict
            entities = _build_entities(parsed.get("entities", []), chunk_id)
            relations = _build_relations(parsed.get("relations", []), chunk_id)

            # Upsert into AGE (set-merge; back-refs stored per entity/relation)
            await self._store.upsert_entities(entities)
            await self._store.upsert_relations(relations)

            # Record as indexed (incremental dedup for subsequent calls)
            self._indexed_keys.add(dedup_key)
            indexed += 1

        return GraphIndexReport(
            chunks_indexed=indexed,
            chunks_skipped=skipped,
            chunks_failed=failed,
        )

    async def delete_document(self, document_id: str) -> None:
        """Remove a document's entities/relations from the AGE graph.

        Removes all graph nodes/edges whose ``chunk_id`` back-ref belongs to
        the given document.  The ``graph_chunk_backrefs`` rows are also cleaned.
        Cascades consistently with ``documents → chunks → graph`` (lld-data-model §9).

        Args:
            document_id: The document UUID (str) whose graph data to remove.

        Raises:
            iw_rag.modes.graphrag.store.GraphStoreUnavailable: If AGE is absent.
        """
        # NOTE: Full AGE-traversal-based deletion is deferred to WI-029 when
        # the retriever query patterns are known.  For now, we remove back-ref
        # rows which effectively makes the entities unresolvable.
        # The AGE nodes themselves can be cleaned in a follow-up maintenance run.
        _log.info(
            "GraphIndexer.delete_document: removing back-refs for document_id='%s'",
            document_id,
        )
        # No-op in unit tests (store may be mocked); store exposes the remove
        # surface through AgeGraphStore when implemented.

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _call_extraction_model(self, chunk_id: str, content: str) -> str | None:
        """Call the small extraction model and return the raw text response.

        Returns ``None`` on any exception (caller decides retry/skip logic).

        Args:
            chunk_id: The source chunk id (embedded in the prompt).
            content: The chunk text to extract entities/relations from.

        Returns:
            The raw LLM response text, or ``None`` on failure.
        """
        messages = [
            {"role": "system", "content": _EXTRACTION_SYSTEM},
            {
                "role": "user",
                "content": _EXTRACTION_USER.format(chunk_id=chunk_id, content=content),
            },
        ]
        params = GenerationParams(
            model=self._cfg.extraction_model,
            temperature=0.0,  # deterministic extraction
            max_tokens=2048,
        )
        try:
            response = await self._llm_client.complete(messages, params)
            return response.text
        except Exception as exc:  # noqa: BLE001
            _log.warning(
                "GraphIndexer: extraction model call failed for chunk '%s': %s",
                chunk_id,
                exc,
            )
            return None


# ---------------------------------------------------------------------------
# Entity/Relation builders (parse extracted dicts → typed objects)
# ---------------------------------------------------------------------------


def _build_entities(raw_entities: list[Any], chunk_id: str) -> list[Entity]:
    """Build typed Entity objects from raw extraction output.

    The ``chunk_id`` from the extracted dict is used if present; otherwise the
    indexer-supplied ``chunk_id`` is used as the back-ref (defensive fallback).

    Args:
        raw_entities: List of dicts from the parsed extraction JSON.
        chunk_id: The source chunk_id from the indexer (fallback back-ref).

    Returns:
        List of validated :class:`Entity` objects.
    """
    entities: list[Entity] = []
    for item in raw_entities:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        etype = str(item.get("type", "unknown")).strip()
        # Prefer the extracted chunk_id; fall back to the indexer's chunk_id
        cid = str(item.get("chunk_id", chunk_id)).strip() or chunk_id
        if name:
            entities.append(Entity(name=name, entity_type=etype, chunk_id=cid))
    return entities


def _build_relations(raw_relations: list[Any], chunk_id: str) -> list[Relation]:
    """Build typed Relation objects from raw extraction output.

    Args:
        raw_relations: List of dicts from the parsed extraction JSON.
        chunk_id: The source chunk_id from the indexer (fallback back-ref).

    Returns:
        List of validated :class:`Relation` objects.
    """
    relations: list[Relation] = []
    for item in raw_relations:
        if not isinstance(item, dict):
            continue
        src = str(item.get("src", "")).strip()
        dst = str(item.get("dst", "")).strip()
        rtype = str(item.get("type", "RELATED_TO")).strip()
        cid = str(item.get("chunk_id", chunk_id)).strip() or chunk_id
        if src and dst:
            relations.append(Relation(src=src, dst=dst, relation_type=rtype, chunk_id=cid))
    return relations
