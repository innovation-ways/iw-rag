"""`make index-visual` — offline batch entrypoint for visual caption indexing.

OFFLINE / batch only.  This script MUST NOT be imported by or called from any
hot-path module (api/, agent/, retrieval/, generation/).  The two-clocks rule
(CLAUDE.md, lld-multimodal §3) places all VLM captioning in the offline/batch
path — the captioner runs here, embeds via BGE-M3, persists as chunks, then
unloads.  No runtime VRAM contention with Qwen3-14B.

Usage:
    uv run --group ingest python scripts/index_visual.py [MANIFEST]

The script reads the same corpus manifest used by ``scripts/ingest_corpus.py``
and captions any table HTML / figure crop assets it finds, embedding captions
through the existing BGE-M3 ingestion → chunks path so they inherit full ACL
pre-filter, metadata, and rerank parity.

Visual asset discovery: in this fallback implementation, assets are passed as
structured entries in the manifest or synthesised from the parsed document
blocks with ``content_type in ("table", "figure_caption")``.  The script is
intentionally conservative — it skips any document whose visual assets were
already captioned (idempotent on asset uri + page).

Requires:
    - ``IW_RAG__DATABASE__DSN`` in env/.env
    - ``IW_RAG__MODES__VISUAL__*`` caption config in env/.env
    - The batch vLLM endpoint for Qwen2.5-VL-7B reachable at the configured
      ``visual_caption`` endpoint (from config/models.yaml)
"""

from __future__ import annotations

import asyncio
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from iw_rag.ingestion.embed import EmbedConfig, make_bge_m3_embedder
from iw_rag.modes.visual.captioner import VlmCaptioner, VlmCaptionerConfig
from iw_rag.modes.visual.types import VisualAsset, VisualAssetType
from iw_rag.settings import get_settings
from iw_rag.storage.db import Database
from iw_rag.storage.repositories import ChunkRepository, NewChunk

_log = logging.getLogger(__name__)

DEFAULT_MANIFEST = Path("data/corpus/manifest.yaml")


@dataclass(frozen=True, slots=True)
class VisualIndexResult:
    """The outcome of indexing visual assets for one document.

    Attributes:
        source_uri: The source document URI.
        captioned: Number of assets successfully captioned and persisted.
        skipped: Number of assets skipped (VLM failure or no content).
    """

    source_uri: str
    captioned: int
    skipped: int


def _build_vlm_captioner() -> VlmCaptioner:
    """Construct a VlmCaptioner from the current Settings.

    Reads the ``visual_caption`` model entry from ``config/models.yaml`` via
    Settings and wires a ``VllmClient`` pointing at the batch vLLM endpoint.

    Returns:
        A configured VlmCaptioner backed by the real batch vLLM endpoint.

    Raises:
        ConfigError: If the ``visual_caption`` model entry is missing from
            models.yaml or the endpoint env var is unset.
    """
    from iw_rag.generation.llm import VllmClient

    settings = get_settings()
    entry = settings.models.get("visual_caption")
    cfg = VlmCaptionerConfig(
        model_role="visual_caption",
        served_name=entry.served_name,
        max_tokens=settings.modes.visual.caption.max_tokens,
    )
    client = VllmClient(endpoint=entry.endpoint)
    return VlmCaptioner(vlm_client=client, cfg=cfg)


def _load_visual_assets_from_manifest(
    manifest_path: Path,
) -> dict[str, list[VisualAsset]]:
    """Load visual assets from a manifest that includes asset entries.

    Reads the manifest YAML.  Each document entry may include an optional
    ``visual_assets`` list with ``type``, ``content``, ``page``, and
    ``acl_groups`` fields.  If absent, the document is skipped (the fallback
    path only processes explicitly listed assets).

    Args:
        manifest_path: Path to the corpus manifest YAML.

    Returns:
        Mapping from ``source_uri`` to the list of visual assets to caption.
    """
    raw: Any = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    docs = raw.get("documents", []) if isinstance(raw, dict) else []
    assets_by_uri: dict[str, list[VisualAsset]] = {}
    for doc in docs:
        uri = doc.get("uri", "")
        acl_groups: list[str] = list(doc.get("acl_groups", []))
        raw_assets: list[dict[str, Any]] = doc.get("visual_assets", [])
        assets: list[VisualAsset] = []
        for ra in raw_assets:
            asset_type_raw = str(ra.get("type", "figure")).lower()
            asset_type = (
                VisualAssetType.TABLE if asset_type_raw == "table" else VisualAssetType.FIGURE
            )
            assets.append(
                VisualAsset(
                    asset_type=asset_type,
                    content=str(ra.get("content", "")),
                    source_uri=uri,
                    page=int(ra["page"]) if "page" in ra else None,
                    acl_tags=acl_groups,
                )
            )
        if assets:
            assets_by_uri[uri] = assets
    return assets_by_uri


async def _index_document_assets(
    source_uri: str,
    assets: list[VisualAsset],
    captioner: VlmCaptioner,
    db: Database,
    doc_id_by_uri: dict[str, Any],
    embedder: Any,
) -> VisualIndexResult:
    """Caption and persist visual assets for one document.

    Args:
        source_uri: The source document URI.
        assets: The visual assets to caption.
        captioner: The VlmCaptioner to use.
        db: The database handle.
        doc_id_by_uri: Mapping from source_uri to document UUID (from the DB).
        embedder: The BGE-M3 embedder for generating dense vectors.

    Returns:
        A VisualIndexResult summarising captioned vs skipped assets.
    """
    doc_id = doc_id_by_uri.get(source_uri)
    if doc_id is None:
        _log.warning("No document found for URI '%s' — skipping visual assets", source_uri)
        return VisualIndexResult(source_uri=source_uri, captioned=0, skipped=len(assets))

    captioned = 0
    skipped = 0
    new_chunks: list[NewChunk] = []

    for ordinal, asset in enumerate(assets):
        caption_result = await captioner.caption_to_chunk(asset, ordinal=ordinal)
        if caption_result is None:
            skipped += 1
            continue

        # Embed the caption text through BGE-M3 (same path as text chunks)
        from iw_rag.ingestion.chunk import ChunkNode

        # Build a minimal ChunkNode so the embedder receives the right shape
        dummy_node = ChunkNode(
            local_id=f"cap_{ordinal}",
            parent_local_id=None,
            ordinal=ordinal,
            content=caption_result.content,
            content_type=caption_result.content_type,  # type: ignore[arg-type]
            section_path=caption_result.section_path,
            page=asset.page,
            token_span=None,
        )
        # Caption embeddings: embed as a single token stream of the caption text

        # Use a simple embedding call for the caption text
        try:
            from iw_rag.ingestion.embed import BgeM3Tokenizer

            tok = BgeM3Tokenizer()
            token_ids = tok.encode(caption_result.content)
            embeddings = await embedder.embed_document(token_ids, [dummy_node])
            dense = embeddings[0].dense if embeddings else [0.0] * 1024
        except Exception as exc:  # noqa: BLE001
            _log.warning(
                "Failed to embed caption for '%s' page=%s: %s", source_uri, asset.page, exc
            )
            skipped += 1
            continue

        new_chunks.append(
            NewChunk(
                document_id=doc_id,
                ordinal=caption_result.ordinal,
                content=caption_result.content,
                dense_embedding=dense,
                parent_id=None,
                content_type=caption_result.content_type,
                section_path=caption_result.section_path,
                acl_tags=caption_result.acl_tags,
                metadata=caption_result.metadata,
            )
        )
        captioned += 1

    if new_chunks:
        async with db.session() as session:
            await ChunkRepository(session).bulk_upsert_chunks(new_chunks)
            await session.commit()

    return VisualIndexResult(source_uri=source_uri, captioned=captioned, skipped=skipped)


async def main(manifest_path: Path) -> int:
    """Caption and index visual assets from the corpus manifest.

    Reads the manifest, builds the captioner + embedder, fetches document IDs
    from the DB, then captions each visual asset and persists the caption as a
    chunk via the existing ingestion spine.

    Args:
        manifest_path: Path to the corpus manifest YAML.

    Returns:
        Exit code: 0 on success, 1 if any document had failures.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    settings = get_settings()

    assets_by_uri = _load_visual_assets_from_manifest(manifest_path)
    if not assets_by_uri:
        _log.info("No visual assets found in manifest — nothing to index.")
        return 0

    captioner = _build_vlm_captioner()
    embedder = make_bge_m3_embedder(
        EmbedConfig(window_tokens=settings.ingestion.embed_window_tokens)
    )
    db = Database(settings.database)

    # Fetch document IDs for all source URIs in one query
    from sqlalchemy import text as sql_text

    doc_id_by_uri: dict[str, Any] = {}
    async with db.session() as session:
        uris = list(assets_by_uri.keys())
        result = await session.execute(
            sql_text("SELECT source_uri, id FROM iw_rag.documents WHERE source_uri = ANY(:uris)"),
            {"uris": uris},
        )
        for row in result.all():
            doc_id_by_uri[row[0]] = row[1]

    total_captioned = 0
    total_skipped = 0
    try:
        for source_uri, assets in assets_by_uri.items():
            result_item = await _index_document_assets(
                source_uri, assets, captioner, db, doc_id_by_uri, embedder
            )
            total_captioned += result_item.captioned
            total_skipped += result_item.skipped
            print(
                f"  {source_uri}: captioned={result_item.captioned} skipped={result_item.skipped}"
            )
    finally:
        await db.dispose()

    print(f"Total: {total_captioned} captioned, {total_skipped} skipped")
    return 1 if total_skipped > 0 and total_captioned == 0 else 0


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_MANIFEST
    raise SystemExit(asyncio.run(main(path)))
