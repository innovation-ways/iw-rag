"""Parsing stage — source files → layout-aware ``ParsedDocument`` blocks.

Docling is the canonical default parser (lld-ingestion §2.1); MinerU is the
escalation parser for hard docs. Because the Phase-0 demo corpus is born-digital
markdown/text (public/synthetic), a light :class:`TextParser` handles ``text/*``
and ``markdown`` without the heavy Docling dependency, while ``DoclingParser``
(lazy-imported, in the ``ingest`` dep group) handles PDF/DOCX and ``MinerUParser``
remains the stubbed escalation path. ``select_parser`` routes by mime + hard flag.

DTOs here are ingestion-local (transient); persisted types live in ``domain``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

from iw_rag.domain.errors import ParseError

BlockType = Literal["text", "table", "figure_caption", "heading"]


@dataclass(frozen=True, slots=True)
class ParseHints:
    """Operator hints influencing parser selection.

    Attributes:
        hard_doc: Force the escalation parser when True (scanned/formula/CJK);
            None means "decide by heuristic".
    """

    hard_doc: bool | None = None


@dataclass(frozen=True, slots=True)
class SourceRef:
    """A reference to one source document to ingest.

    Attributes:
        uri: Logical, stable identifier (e.g. ``file:///corpus/hr/policy.md``).
        mime: MIME type used for parser routing.
        hints: Operator parse hints.
        path: Filesystem path to read (defaults to the ``file://`` uri's path).
    """

    uri: str
    mime: str
    hints: ParseHints = field(default_factory=ParseHints)
    path: Path | None = None

    def local_path(self) -> Path:
        """Resolve the filesystem path to read.

        Returns:
            ``path`` if set, else the path component of a ``file://`` uri.

        Raises:
            ParseError: If no readable path can be derived from the source.
        """
        if self.path is not None:
            return self.path
        if self.uri.startswith("file://"):
            return Path(self.uri[len("file://") :])
        raise ParseError(f"SourceRef has no readable path: {self.uri}")


@dataclass(frozen=True, slots=True)
class Block:
    """One layout-aware content block from a parsed document.

    Attributes:
        text: The block's text (table blocks carry a text rendering too).
        block_type: ``text|table|figure_caption|heading``.
        section_path: Breadcrumb of enclosing headings (e.g. ``"Policies > Leave"``).
        page: 1-based page number when known.
        payload: Preserved structured payload (table ``html``, figure ``image_ref``,
            ``bbox``) — kept for the Phase-4 visual mode.
    """

    text: str
    block_type: BlockType = "text"
    section_path: str | None = None
    page: int | None = None
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    """The parser's output: ordered blocks + the concatenated full text.

    Attributes:
        source_uri: The originating source uri.
        blocks: Ordered content blocks in reading order.
        full_text: The whole-document text (used for Late Chunking encode).
        title: Document title when derivable (e.g. first heading).
    """

    source_uri: str
    blocks: list[Block]
    full_text: str
    title: str | None = None


@dataclass(frozen=True, slots=True)
class ParseConfig:
    """Parser-selection configuration (subset of IngestionSettings)."""

    parser_default: str = "docling"
    parser_hard: str = "mineru"
    force_parser: str | None = None
    parse_timeout_s: int = 120


@runtime_checkable
class Parser(Protocol):
    """A document parser producing a :class:`ParsedDocument`."""

    async def parse(self, source: SourceRef) -> ParsedDocument:
        """Parse a source into layout-aware blocks."""
        ...


def _split_markdown_blocks(text: str) -> list[Block]:
    """Split markdown/plain text into heading/table/text blocks (best-effort).

    Args:
        text: Raw markdown or plain text.

    Returns:
        Ordered blocks; markdown ``#`` headings become ``heading`` blocks and
        update the running ``section_path``; ``|``-delimited runs become ``table``
        blocks; blank-line-separated runs become ``text`` blocks.
    """
    blocks: list[Block] = []
    section: list[str] = []
    para: list[str] = []
    table: list[str] = []

    def flush_para() -> None:
        if para:
            blocks.append(
                Block(
                    text="\n".join(para).strip(),
                    block_type="text",
                    section_path=" > ".join(section) or None,
                )
            )
            para.clear()

    def flush_table() -> None:
        if table:
            blocks.append(
                Block(
                    text="\n".join(table).strip(),
                    block_type="table",
                    section_path=" > ".join(section) or None,
                    payload={"markdown": "\n".join(table)},
                )
            )
            table.clear()

    for raw in text.splitlines():
        line = raw.rstrip()
        if line.startswith("#"):
            flush_para()
            flush_table()
            level = len(line) - len(line.lstrip("#"))
            heading = line.lstrip("#").strip()
            section = section[: max(0, level - 1)] + [heading]
            blocks.append(
                Block(text=heading, block_type="heading", section_path=" > ".join(section))
            )
        elif line.strip().startswith("|") and line.strip().endswith("|"):
            flush_para()
            table.append(line)
        elif not line.strip():
            flush_para()
            flush_table()
        else:
            flush_table()
            para.append(line)
    flush_para()
    flush_table()
    return blocks


class TextParser:
    """Parser for born-digital ``text/*`` and ``markdown`` sources (no heavy deps)."""

    def __init__(self, cfg: ParseConfig) -> None:
        """Store config (unused for plain text, kept for interface parity)."""
        self._cfg = cfg

    async def parse(self, source: SourceRef) -> ParsedDocument:
        """Read and structure a markdown/text file into a ParsedDocument.

        Args:
            source: The source to read.

        Returns:
            A ParsedDocument with heading/table/text blocks.

        Raises:
            ParseError: If the file cannot be read or decoded.
        """
        try:
            text = source.local_path().read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise ParseError(f"failed to read {source.uri}: {exc}") from exc
        return parse_markdown_text(text, source.uri)


def parse_markdown_text(text: str, source_uri: str) -> ParsedDocument:
    """Parse markdown/plain text already in memory into a :class:`ParsedDocument`.

    This is the same logic :class:`TextParser` applies to a file, exposed so callers
    that hold the text (the Learn section's chunking lab) need no temporary file.

    Args:
        text: Raw markdown or plain text.
        source_uri: Logical identifier recorded on the result.

    Returns:
        A ParsedDocument with heading/table/text blocks.
    """
    blocks = _split_markdown_blocks(text)
    title = next((b.text for b in blocks if b.block_type == "heading"), None)
    return ParsedDocument(source_uri=source_uri, blocks=blocks, full_text=text, title=title)


class DoclingParser:
    """Default layout-aware parser for PDF/DOCX (lazy-imports ``docling``)."""

    def __init__(self, cfg: ParseConfig) -> None:
        """Store config; the Docling converter is created lazily on first parse."""
        self._cfg = cfg

    async def parse(self, source: SourceRef) -> ParsedDocument:
        """Parse a rich document via Docling into blocks.

        Args:
            source: The source to parse.

        Returns:
            A ParsedDocument; headings/tables/text mapped from Docling items.

        Raises:
            ParseError: If Docling is unavailable or conversion fails.
        """
        try:
            from docling.document_converter import DocumentConverter  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - exercised only with the ingest group
            raise ParseError(
                "DoclingParser requires the 'ingest' dependency group (uv sync --group ingest)."
            ) from exc
        try:
            result = DocumentConverter().convert(str(source.local_path()))
            markdown = result.document.export_to_markdown()
        except Exception as exc:  # noqa: BLE001 - map any Docling failure
            raise ParseError(f"docling failed on {source.uri}: {exc}") from exc
        blocks = _split_markdown_blocks(markdown)
        title = next((b.text for b in blocks if b.block_type == "heading"), None)
        return ParsedDocument(source_uri=source.uri, blocks=blocks, full_text=markdown, title=title)


class MinerUParser:
    """Escalation parser for hard docs — stubbed in Phase 0 (Q-MINERU)."""

    def __init__(self, cfg: ParseConfig) -> None:
        """Store config; the real MinerU integration is a later task."""
        self._cfg = cfg

    async def parse(self, source: SourceRef) -> ParsedDocument:
        """Not implemented in Phase 0.

        Raises:
            ParseError: Always — MinerU escalation is stubbed (re-run with
                ``force_parser`` once implemented).
        """
        raise ParseError(
            f"MinerUParser is stubbed in Phase 0 (hard doc: {source.uri}); "
            "ship Docling-only or implement MinerU."
        )


_TEXT_MIMES = ("text/markdown", "text/plain", "text/x-markdown")


def select_parser(source: SourceRef, cfg: ParseConfig) -> Parser:
    """Route a source to its parser.

    Order: explicit ``force_parser`` override → hard-doc escalation (MinerU) →
    born-digital text (TextParser) → Docling default.

    Args:
        source: The source to route.
        cfg: Parser-selection config.

    Returns:
        The selected parser instance.
    """
    if cfg.force_parser:
        return _by_name(cfg.force_parser, cfg)
    if source.hints.hard_doc:
        return MinerUParser(cfg)
    if source.mime in _TEXT_MIMES:
        return TextParser(cfg)
    return DoclingParser(cfg)


def _by_name(name: str, cfg: ParseConfig) -> Parser:
    """Resolve a parser by config name (``docling|mineru|text``)."""
    match name:
        case "docling":
            return DoclingParser(cfg)
        case "mineru":
            return MinerUParser(cfg)
        case "text":
            return TextParser(cfg)
        case _:
            raise ParseError(f"unknown parser '{name}'")
