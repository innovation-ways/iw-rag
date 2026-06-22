"""Unit tests for parser selection and the born-digital TextParser."""

from __future__ import annotations

from pathlib import Path

import pytest

from iw_rag.domain.errors import ParseError
from iw_rag.ingestion.parse import (
    ParseConfig,
    ParseHints,
    SourceRef,
    TextParser,
    select_parser,
)

_CFG = ParseConfig()


def _src(mime: str, *, hard: bool | None = None, path: Path | None = None) -> SourceRef:
    return SourceRef(uri="file:///x", mime=mime, hints=ParseHints(hard_doc=hard), path=path)


def test_text_mime_routes_to_text_parser() -> None:
    """text/markdown and text/plain both route to TextParser, not Docling or MinerU."""
    md_parser = select_parser(_src("text/markdown"), _CFG)
    plain_parser = select_parser(_src("text/plain"), _CFG)
    assert type(md_parser).__name__ == "TextParser", (
        f"text/markdown must route to TextParser, got {type(md_parser).__name__}"
    )
    assert type(plain_parser).__name__ == "TextParser", (
        f"text/plain must route to TextParser, got {type(plain_parser).__name__}"
    )


def test_pdf_default_routes_to_docling() -> None:
    """application/pdf without hard_doc hint routes to DoclingParser."""
    parser = select_parser(_src("application/pdf"), _CFG)
    assert type(parser).__name__ == "DoclingParser", (
        f"application/pdf must route to DoclingParser by default, got {type(parser).__name__}"
    )


def test_hard_doc_escalates_to_mineru() -> None:
    """application/pdf with hard_doc=True escalates to MinerUParser."""
    parser = select_parser(_src("application/pdf", hard=True), _CFG)
    assert type(parser).__name__ == "MinerUParser", (
        f"hard_doc=True must escalate to MinerUParser, got {type(parser).__name__}"
    )


def test_force_parser_overrides() -> None:
    """force_parser='mineru' overrides mime-based routing for any mime type."""
    cfg = ParseConfig(force_parser="mineru")
    # Even a text mime is overridden by force_parser.
    parser = select_parser(_src("text/markdown"), cfg)
    assert type(parser).__name__ == "MinerUParser", (
        f"force_parser='mineru' must select MinerUParser regardless of mime, "
        f"got {type(parser).__name__}"
    )


def test_unknown_force_parser_raises() -> None:
    """force_parser with an unknown name raises ParseError naming the bad value."""
    cfg = ParseConfig(force_parser="bogus")
    with pytest.raises(ParseError, match="bogus"):
        select_parser(_src("text/markdown"), cfg)


async def test_text_parser_structures_markdown(tmp_path: Path) -> None:
    md = tmp_path / "doc.md"
    md.write_text(
        "# Leave Policy\n\nStaff accrue leave monthly.\n\n"
        "## Rates\n\n| band | days |\n| --- | --- |\n| A | 20 |\n",
        encoding="utf-8",
    )
    src = SourceRef(uri="file:///doc.md", mime="text/markdown", path=md)
    parsed = await TextParser(_CFG).parse(src)

    types = [b.block_type for b in parsed.blocks]
    assert "heading" in types
    assert "table" in types
    assert parsed.title == "Leave Policy"
    table = next(b for b in parsed.blocks if b.block_type == "table")
    assert table.section_path == "Leave Policy > Rates"


async def test_text_parser_missing_file_raises() -> None:
    """TextParser raises ParseError naming the URI when the file does not exist."""
    src = SourceRef(uri="file:///nope.md", mime="text/markdown", path=Path("/no/such.md"))
    with pytest.raises(ParseError, match="nope.md"):
        await TextParser(_CFG).parse(src)
