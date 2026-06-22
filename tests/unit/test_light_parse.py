"""Unit tests for the online-upload light parsers (ingestion/light_parse.py).

Covers:
- ``select_light_parser`` MIME routing to TextParser, PdfTextParser, OcrImageParser,
  plus the unsupported-type ParseError.
- ``PdfTextParser.parse``: born-digital extraction, OCR fallback for text-free pages,
  per-page exception swallow, and no-text raise.
- ``PdfTextParser._ocr_page``: pdf2image absent, convert_from_path failure, empty
  image list, and OCR ParseError swallowed.
- ``OcrImageParser.parse``: OCR happy path, caption fallback, and no-text raise.
- ``TesseractOcrEngine``: OCR-failure-to-ParseError mapping and ``available()``
  happy/failure paths via injected fake modules.
- ``CharTokenizer``: encode/decode round-trip identity.

All tests inject fakes — no live models, no Tesseract binary, no network.
``pypdf``, ``pdf2image``, ``pytesseract``, and ``PIL`` are injected via
``sys.modules`` because the production code lazy-imports them inside try/except
blocks to keep heavy deps optional on the API container.
"""

from __future__ import annotations

import sys
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from iw_rag.domain.errors import ParseError
from iw_rag.ingestion.light_parse import (
    CharTokenizer,
    OcrImageParser,
    PdfTextParser,
    TesseractOcrEngine,
    select_light_parser,
)
from iw_rag.ingestion.parse import ParseConfig, ParseHints, SourceRef, TextParser

_CFG = ParseConfig()


# ---------------------------------------------------------------------------
# Fake helpers
# ---------------------------------------------------------------------------


class FakeOcrEngine:
    """Fake OCR engine that returns a fixed string and reports itself available.

    Args:
        text: The text to return from ``image_to_text``.
        is_available: Whether to report as available.
    """

    def __init__(self, text: str = "", *, is_available: bool = True) -> None:
        """Store canned return values."""
        self._text = text
        self._is_available = is_available

    def image_to_text(self, path: Path) -> str:
        """Return the canned text regardless of path."""
        return self._text

    def available(self) -> bool:
        """Return the preconfigured availability flag."""
        return self._is_available


class FakeCaptionEngine:
    """Fake VLM caption engine returning a fixed caption.

    Args:
        caption: The caption to return; ``None`` simulates a VLM failure.
        is_available: Whether to report as available.
    """

    def __init__(self, caption: str | None = None, *, is_available: bool = True) -> None:
        """Store canned values."""
        self._caption = caption
        self._is_available = is_available

    async def caption_image(self, data: bytes, *, mime: str, hint: str = "") -> str | None:
        """Return the canned caption."""
        return self._caption

    async def available(self) -> bool:
        """Return the preconfigured availability flag."""
        return self._is_available


def _src(mime: str, path: Path | None = None) -> SourceRef:
    """Build a minimal SourceRef for routing/parsing tests.

    Args:
        mime: The MIME type used for parser selection.
        path: Optional filesystem path to attach.

    Returns:
        A SourceRef with a fixed synthetic URI.
    """
    return SourceRef(uri="file:///test", mime=mime, hints=ParseHints(), path=path)


def _make_fake_pdf_reader(pages: list[str]) -> Any:
    """Build a mock PdfReader whose pages yield the given per-page text strings.

    Args:
        pages: Per-page text; an empty string simulates a scanned (image-only) page.

    Returns:
        A MagicMock with a ``.pages`` list whose items implement ``extract_text()``.
    """
    reader = MagicMock()
    page_mocks = []
    for text in pages:
        p = MagicMock()
        p.extract_text.return_value = text
        page_mocks.append(p)
    reader.pages = page_mocks
    return reader


@contextmanager
def _inject_pypdf(reader: Any) -> Generator[None, None, None]:
    """Inject a fake pypdf module so PdfTextParser.parse() uses it.

    ``PdfTextParser.parse`` does ``from pypdf import PdfReader`` inside a
    try/except block, so the import is resolved at call time via sys.modules.

    Args:
        reader: The fake PdfReader instance to return when PdfReader() is called.

    Yields:
        Nothing; restores the original module on exit.
    """
    fake_mod = MagicMock()
    fake_mod.PdfReader = MagicMock(return_value=reader)
    original = sys.modules.get("pypdf")
    sys.modules["pypdf"] = fake_mod
    try:
        yield
    finally:
        if original is None:
            sys.modules.pop("pypdf", None)
        else:
            sys.modules["pypdf"] = original


@contextmanager
def _inject_pdf2image(images: list[Any]) -> Generator[None, None, None]:
    """Inject a fake pdf2image module so _ocr_page() uses it.

    Args:
        images: List of image objects to return from ``convert_from_path``.

    Yields:
        Nothing; restores the original module on exit.
    """
    fake_mod = MagicMock()
    fake_mod.convert_from_path = MagicMock(return_value=images)
    original = sys.modules.get("pdf2image")
    sys.modules["pdf2image"] = fake_mod
    try:
        yield
    finally:
        if original is None:
            sys.modules.pop("pdf2image", None)
        else:
            sys.modules["pdf2image"] = original


@contextmanager
def _inject_pdf2image_raising(exc: Exception) -> Generator[None, None, None]:
    """Inject a fake pdf2image module whose convert_from_path raises ``exc``.

    Args:
        exc: The exception that convert_from_path will raise.

    Yields:
        Nothing; restores the original module on exit.
    """
    fake_mod = MagicMock()
    fake_mod.convert_from_path = MagicMock(side_effect=exc)
    original = sys.modules.get("pdf2image")
    sys.modules["pdf2image"] = fake_mod
    try:
        yield
    finally:
        if original is None:
            sys.modules.pop("pdf2image", None)
        else:
            sys.modules["pdf2image"] = original


@contextmanager
def _inject_pytesseract_and_pil(
    *,
    ocr_text: str = "ocr text",
    ocr_raises: Exception | None = None,
    version_raises: Exception | None = None,
) -> Generator[None, None, None]:
    """Inject fake ``pytesseract`` and ``PIL.Image`` modules into sys.modules.

    Both modules are lazy-imported by ``TesseractOcrEngine`` inside try/except
    blocks, so we must inject them at the sys.modules level.

    Args:
        ocr_text: Text that ``pytesseract.image_to_string`` returns.
        ocr_raises: If set, ``image_to_string`` raises this instead.
        version_raises: If set, ``get_tesseract_version`` raises this.

    Yields:
        Nothing; restores originals on exit.
    """
    fake_pytesseract = MagicMock()
    if ocr_raises is not None:
        fake_pytesseract.image_to_string = MagicMock(side_effect=ocr_raises)
    else:
        fake_pytesseract.image_to_string = MagicMock(return_value=ocr_text)
    if version_raises is not None:
        fake_pytesseract.get_tesseract_version = MagicMock(side_effect=version_raises)
    else:
        fake_pytesseract.get_tesseract_version = MagicMock(return_value="5.0")

    # PIL.Image: provide a context-manager-compatible open()
    fake_pil_image = MagicMock()
    fake_pil_image_obj = MagicMock()
    fake_pil_image.open = MagicMock(return_value=fake_pil_image_obj)
    fake_pil_image_obj.__enter__ = MagicMock(return_value=fake_pil_image_obj)
    fake_pil_image_obj.__exit__ = MagicMock(return_value=False)

    fake_pil_mod = MagicMock()
    fake_pil_mod.Image = fake_pil_image

    orig_pytesseract = sys.modules.get("pytesseract")
    orig_pil = sys.modules.get("PIL")
    orig_pil_image = sys.modules.get("PIL.Image")

    sys.modules["pytesseract"] = fake_pytesseract
    sys.modules["PIL"] = fake_pil_mod
    sys.modules["PIL.Image"] = fake_pil_image
    try:
        yield
    finally:
        if orig_pytesseract is None:
            sys.modules.pop("pytesseract", None)
        else:
            sys.modules["pytesseract"] = orig_pytesseract
        if orig_pil is None:
            sys.modules.pop("PIL", None)
        else:
            sys.modules["PIL"] = orig_pil
        if orig_pil_image is None:
            sys.modules.pop("PIL.Image", None)
        else:
            sys.modules["PIL.Image"] = orig_pil_image


# ---------------------------------------------------------------------------
# 1. select_light_parser — MIME routing
# ---------------------------------------------------------------------------


def test_text_markdown_routes_to_text_parser() -> None:
    """select_light_parser maps text/markdown to TextParser, not to a PDF or OCR parser."""
    parser = select_light_parser(_src("text/markdown"), _CFG)
    assert type(parser) is TextParser


def test_text_plain_routes_to_text_parser() -> None:
    """select_light_parser maps text/plain to TextParser, not to a PDF or OCR parser."""
    parser = select_light_parser(_src("text/plain"), _CFG)
    assert type(parser) is TextParser


def test_text_x_markdown_routes_to_text_parser() -> None:
    """select_light_parser maps text/x-markdown to TextParser, not to a PDF or OCR parser."""
    parser = select_light_parser(_src("text/x-markdown"), _CFG)
    assert type(parser) is TextParser


def test_pdf_routes_to_pdf_text_parser() -> None:
    """select_light_parser maps application/pdf to PdfTextParser, not TextParser or OCR."""
    parser = select_light_parser(_src("application/pdf"), _CFG)
    assert type(parser) is PdfTextParser


def test_image_png_routes_to_ocr_image_parser() -> None:
    """select_light_parser maps image/png to OcrImageParser, not TextParser or PDF."""
    parser = select_light_parser(_src("image/png"), _CFG)
    assert type(parser) is OcrImageParser


def test_image_jpeg_routes_to_ocr_image_parser() -> None:
    """select_light_parser maps image/jpeg to OcrImageParser, not TextParser or PDF."""
    parser = select_light_parser(_src("image/jpeg"), _CFG)
    assert type(parser) is OcrImageParser


def test_unknown_text_subtype_falls_back_to_text_parser() -> None:
    """select_light_parser falls back to TextParser for unrecognised text/* MIME, not OCR."""
    parser = select_light_parser(_src("text/rtf"), _CFG)
    assert type(parser) is TextParser


def test_unsupported_mime_raises_parse_error() -> None:
    """select_light_parser raises ParseError for non-text, non-pdf, non-image MIME."""
    with pytest.raises(ParseError, match="unsupported upload type 'application/zip'"):
        select_light_parser(_src("application/zip"), _CFG)


def test_unsupported_mime_message_includes_uri() -> None:
    """ParseError from unsupported MIME includes the source URI."""
    with pytest.raises(ParseError, match="file:///test"):
        select_light_parser(_src("video/mp4"), _CFG)


def test_injected_ocr_engine_reaches_pdf_parser() -> None:
    """select_light_parser injects the supplied ocr engine into PdfTextParser."""
    fake_ocr = FakeOcrEngine("hello")
    parser = select_light_parser(_src("application/pdf"), _CFG, ocr=fake_ocr)
    assert isinstance(parser, PdfTextParser)
    assert parser._ocr is fake_ocr  # type: ignore[attr-defined]


def test_injected_ocr_and_caption_reach_image_parser() -> None:
    """select_light_parser injects ocr + caption_engine into OcrImageParser."""
    fake_ocr = FakeOcrEngine("text")
    fake_cap = FakeCaptionEngine("a caption")
    parser = select_light_parser(_src("image/png"), _CFG, ocr=fake_ocr, caption_engine=fake_cap)
    assert isinstance(parser, OcrImageParser)
    assert parser._ocr is fake_ocr  # type: ignore[attr-defined]
    assert parser._caption is fake_cap  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# 2. PdfTextParser.parse — born-digital, OCR fallback, no-text raise
# ---------------------------------------------------------------------------


async def test_pdf_text_parser_born_digital_returns_blocks(tmp_path: Path) -> None:
    """PdfTextParser.parse extracts text from born-digital pages into blocks."""
    pdf_file = tmp_path / "doc.pdf"
    pdf_file.write_bytes(b"%PDF fake")
    src = SourceRef(uri="file:///doc.pdf", mime="application/pdf", path=pdf_file)

    fake_reader = _make_fake_pdf_reader(["Page one text.", "Page two text."])

    with _inject_pypdf(fake_reader):
        parser = PdfTextParser(_CFG, ocr=None)
        result = await parser.parse(src)

    assert len(result.blocks) == 2
    assert result.blocks[0].text == "Page one text."
    assert result.blocks[0].page == 1
    assert result.blocks[1].text == "Page two text."
    assert result.blocks[1].page == 2
    assert result.source_uri == "file:///doc.pdf"
    assert result.title == "doc"


async def test_pdf_text_parser_full_text_joins_pages(tmp_path: Path) -> None:
    """PdfTextParser.parse joins page texts with double newlines into full_text."""
    pdf_file = tmp_path / "multi.pdf"
    pdf_file.write_bytes(b"%PDF fake")
    src = SourceRef(uri="file:///multi.pdf", mime="application/pdf", path=pdf_file)

    fake_reader = _make_fake_pdf_reader(["First page.", "Second page."])

    with _inject_pypdf(fake_reader):
        parser = PdfTextParser(_CFG, ocr=None)
        result = await parser.parse(src)

    assert result.full_text == "First page.\n\nSecond page."


async def test_pdf_text_parser_ocr_fallback_for_empty_page(tmp_path: Path) -> None:
    """PdfTextParser.parse falls back to OCR when a page has no embedded text."""
    pdf_file = tmp_path / "scanned.pdf"
    pdf_file.write_bytes(b"%PDF fake")
    src = SourceRef(uri="file:///scanned.pdf", mime="application/pdf", path=pdf_file)

    # Page 1 has text; page 2 is blank (simulates a scanned page)
    fake_reader = _make_fake_pdf_reader(["Born-digital text.", ""])
    ocr = FakeOcrEngine("OCR text from page 2.", is_available=True)
    fake_image = MagicMock()
    # save() must not raise — it's called by _ocr_page before OCR
    fake_image.save = MagicMock(return_value=None)

    with _inject_pypdf(fake_reader), _inject_pdf2image([fake_image]):
        parser = PdfTextParser(_CFG, ocr=ocr)
        result = await parser.parse(src)

    texts = [b.text for b in result.blocks]
    assert "Born-digital text." in texts
    assert any("OCR" in t for t in texts)


async def test_pdf_text_parser_no_text_raises_parse_error(tmp_path: Path) -> None:
    """PdfTextParser.parse raises ParseError when no page yields any text."""
    pdf_file = tmp_path / "empty.pdf"
    pdf_file.write_bytes(b"%PDF fake")
    src = SourceRef(uri="file:///empty.pdf", mime="application/pdf", path=pdf_file)

    # All pages are blank; OCR is not available
    fake_reader = _make_fake_pdf_reader(["", ""])
    ocr = FakeOcrEngine("", is_available=False)

    with _inject_pypdf(fake_reader):
        parser = PdfTextParser(_CFG, ocr=ocr)
        with pytest.raises(
            ParseError,
            match="no extractable text in file:///empty.pdf",
        ):
            await parser.parse(src)


async def test_pdf_text_parser_pypdf_failure_raises_parse_error(tmp_path: Path) -> None:
    """PdfTextParser.parse wraps a pypdf read error in ParseError."""
    pdf_file = tmp_path / "bad.pdf"
    pdf_file.write_bytes(b"not a pdf")
    src = SourceRef(uri="file:///bad.pdf", mime="application/pdf", path=pdf_file)

    # Make PdfReader() itself raise
    fake_mod = MagicMock()
    fake_mod.PdfReader = MagicMock(side_effect=Exception("corrupted PDF"))
    original = sys.modules.get("pypdf")
    sys.modules["pypdf"] = fake_mod
    try:
        parser = PdfTextParser(_CFG, ocr=None)
        with pytest.raises(ParseError, match="pypdf failed on file:///bad.pdf"):
            await parser.parse(src)
    finally:
        if original is None:
            sys.modules.pop("pypdf", None)
        else:
            sys.modules["pypdf"] = original


# ---------------------------------------------------------------------------
# 3. OcrImageParser.parse — OCR happy path, caption fallback, no-text raise
# ---------------------------------------------------------------------------


async def test_ocr_image_parser_ocr_text_returned(tmp_path: Path) -> None:
    """OcrImageParser.parse returns a block from OCR text when OCR is available."""
    img_file = tmp_path / "invoice.png"
    img_file.write_bytes(b"PNG fake data")
    src = SourceRef(uri="file:///invoice.png", mime="image/png", path=img_file)

    ocr = FakeOcrEngine("Invoice total: $500", is_available=True)
    parser = OcrImageParser(_CFG, ocr=ocr, caption_engine=None)

    result = await parser.parse(src)

    assert len(result.blocks) == 1
    assert result.blocks[0].text == "Invoice total: $500"
    assert result.blocks[0].block_type == "text"
    assert result.blocks[0].page == 1
    assert result.full_text == "Invoice total: $500"
    assert result.title == "invoice"
    assert result.source_uri == "file:///invoice.png"


async def test_ocr_image_parser_ocr_unavailable_caption_fallback(tmp_path: Path) -> None:
    """OcrImageParser.parse uses the VLM caption when OCR is unavailable."""
    img_file = tmp_path / "chart.png"
    img_file.write_bytes(b"PNG fake data")
    src = SourceRef(uri="file:///chart.png", mime="image/png", path=img_file)

    ocr = FakeOcrEngine("", is_available=False)
    caption_engine = FakeCaptionEngine("A bar chart showing revenue growth.", is_available=True)
    parser = OcrImageParser(_CFG, ocr=ocr, caption_engine=caption_engine)

    result = await parser.parse(src)

    assert len(result.blocks) == 1
    assert result.blocks[0].text == "A bar chart showing revenue growth."
    assert result.blocks[0].block_type == "figure_caption"


async def test_ocr_image_parser_ocr_empty_uses_caption(tmp_path: Path) -> None:
    """OcrImageParser.parse uses caption when OCR returns an empty string."""
    img_file = tmp_path / "diagram.png"
    img_file.write_bytes(b"PNG fake data")
    src = SourceRef(uri="file:///diagram.png", mime="image/png", path=img_file)

    ocr = FakeOcrEngine("", is_available=True)
    caption_engine = FakeCaptionEngine("System architecture diagram.", is_available=True)
    parser = OcrImageParser(_CFG, ocr=ocr, caption_engine=caption_engine)

    result = await parser.parse(src)

    assert result.blocks[0].text == "System architecture diagram."
    assert result.blocks[0].block_type == "figure_caption"


async def test_ocr_image_parser_no_text_no_caption_raises(tmp_path: Path) -> None:
    """OcrImageParser.parse raises ParseError when OCR and caption both yield nothing."""
    img_file = tmp_path / "blank.png"
    img_file.write_bytes(b"PNG fake data")
    src = SourceRef(uri="file:///blank.png", mime="image/png", path=img_file)

    ocr = FakeOcrEngine("", is_available=True)
    parser = OcrImageParser(_CFG, ocr=ocr, caption_engine=None)

    with pytest.raises(
        ParseError,
        match="could not extract text from image file:///blank.png",
    ):
        await parser.parse(src)


async def test_ocr_image_parser_caption_unavailable_raises(tmp_path: Path) -> None:
    """OcrImageParser.parse raises ParseError when OCR is empty and caption unavailable."""
    img_file = tmp_path / "blank2.png"
    img_file.write_bytes(b"PNG fake data")
    src = SourceRef(uri="file:///blank2.png", mime="image/png", path=img_file)

    ocr = FakeOcrEngine("", is_available=False)
    caption_engine = FakeCaptionEngine(None, is_available=False)
    parser = OcrImageParser(_CFG, ocr=ocr, caption_engine=caption_engine)

    with pytest.raises(ParseError, match="could not extract text from image"):
        await parser.parse(src)


async def test_ocr_image_parser_caption_returns_none_raises(tmp_path: Path) -> None:
    """OcrImageParser.parse raises ParseError when caption engine returns None."""
    img_file = tmp_path / "blank3.png"
    img_file.write_bytes(b"PNG fake data")
    src = SourceRef(uri="file:///blank3.png", mime="image/png", path=img_file)

    ocr = FakeOcrEngine("", is_available=True)
    caption_engine = FakeCaptionEngine(None, is_available=True)
    parser = OcrImageParser(_CFG, ocr=ocr, caption_engine=caption_engine)

    with pytest.raises(ParseError, match="could not extract text from image"):
        await parser.parse(src)


# ---------------------------------------------------------------------------
# 4. CharTokenizer — encode/decode round-trip identity
# ---------------------------------------------------------------------------


def test_char_tokenizer_encode_is_unicode_codepoints() -> None:
    """CharTokenizer.encode returns the Unicode code point of each character."""
    tokenizer = CharTokenizer()
    assert tokenizer.encode("ABC") == [65, 66, 67]


def test_char_tokenizer_decode_is_inverse_of_encode() -> None:
    """CharTokenizer.decode is the exact inverse of encode (lossless round-trip)."""
    tokenizer = CharTokenizer()
    original = "Hello, world! 🌍"
    ids = tokenizer.encode(original)
    recovered = tokenizer.decode(ids)
    assert recovered == original


def test_char_tokenizer_roundtrip_empty_string() -> None:
    """CharTokenizer encode/decode preserves an empty string."""
    tokenizer = CharTokenizer()
    assert tokenizer.decode(tokenizer.encode("")) == ""


def test_char_tokenizer_roundtrip_unicode_multibyte() -> None:
    """CharTokenizer preserves multi-byte Unicode characters exactly."""
    tokenizer = CharTokenizer()
    text = "日本語テスト"
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_char_tokenizer_roundtrip_newlines_and_spaces() -> None:
    """CharTokenizer preserves whitespace and newline characters."""
    tokenizer = CharTokenizer()
    text = "line one\nline two\ttabbed"
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_char_tokenizer_encode_length_equals_char_count() -> None:
    """CharTokenizer.encode produces one id per Unicode code point, not UTF-8 byte."""
    tokenizer = CharTokenizer()
    # "é" (U+00E9) is one code point but two UTF-8 bytes
    text = "café"
    ids = tokenizer.encode(text)
    assert len(ids) == len(text) == 4


def test_char_tokenizer_decode_from_raw_ids() -> None:
    """CharTokenizer.decode converts code-point integers to text correctly."""
    tokenizer = CharTokenizer()
    ids = [72, 101, 108, 108, 111]  # "Hello"
    assert tokenizer.decode(ids) == "Hello"


# ---------------------------------------------------------------------------
# 5. PdfTextParser — per-page exception swallow (line ~148-149)
# ---------------------------------------------------------------------------


async def test_pdf_text_parser_bad_page_extract_text_swallowed(tmp_path: Path) -> None:
    """PdfTextParser.parse swallows per-page extract_text() exceptions and continues.

    A page whose ``extract_text()`` raises must not abort the whole document.
    The bad page is silently skipped; good pages are still returned.
    """
    pdf_file = tmp_path / "mixed.pdf"
    pdf_file.write_bytes(b"%PDF fake")
    src = SourceRef(uri="file:///mixed.pdf", mime="application/pdf", path=pdf_file)

    reader = MagicMock()
    good_page = MagicMock()
    good_page.extract_text.return_value = "Good page text."
    bad_page = MagicMock()
    bad_page.extract_text.side_effect = RuntimeError("corrupt xref table")
    reader.pages = [bad_page, good_page]

    with _inject_pypdf(reader):
        parser = PdfTextParser(_CFG, ocr=None)
        result = await parser.parse(src)

    assert len(result.blocks) == 1
    assert result.blocks[0].text == "Good page text."
    assert result.blocks[0].page == 2


async def test_pdf_text_parser_all_pages_raise_then_no_text_parse_error(
    tmp_path: Path,
) -> None:
    """PdfTextParser.parse raises ParseError when every page's extract_text() raises."""
    pdf_file = tmp_path / "allbad.pdf"
    pdf_file.write_bytes(b"%PDF fake")
    src = SourceRef(uri="file:///allbad.pdf", mime="application/pdf", path=pdf_file)

    reader = MagicMock()
    bad_page = MagicMock()
    bad_page.extract_text.side_effect = RuntimeError("decoding error")
    reader.pages = [bad_page, bad_page]

    with _inject_pypdf(reader):
        parser = PdfTextParser(_CFG, ocr=None)
        with pytest.raises(ParseError, match="no extractable text in file:///allbad.pdf"):
            await parser.parse(src)


# ---------------------------------------------------------------------------
# 6. PdfTextParser._ocr_page — error branches
# ---------------------------------------------------------------------------


async def test_ocr_page_returns_empty_when_pdf2image_absent(tmp_path: Path) -> None:
    """PdfTextParser._ocr_page returns empty string when pdf2image is not installed.

    A missing pdf2image module (ImportError) must not raise — the method degrades
    gracefully and the surrounding parse() loop skips the page.
    """
    pdf_file = tmp_path / "scan.pdf"
    pdf_file.write_bytes(b"%PDF fake")
    src = SourceRef(uri="file:///scan.pdf", mime="application/pdf", path=pdf_file)

    # Page yields no text (scanned), OCR is available, but pdf2image is absent.
    fake_reader = _make_fake_pdf_reader(["", ""])
    ocr = FakeOcrEngine("some text", is_available=True)

    # Remove pdf2image from sys.modules to simulate it being absent.
    original = sys.modules.pop("pdf2image", None)
    try:
        with _inject_pypdf(fake_reader):
            parser = PdfTextParser(_CFG, ocr=ocr)
            with pytest.raises(ParseError, match="no extractable text in file:///scan.pdf"):
                await parser.parse(src)
    finally:
        if original is not None:
            sys.modules["pdf2image"] = original


async def test_ocr_page_returns_empty_when_convert_from_path_raises(tmp_path: Path) -> None:
    """PdfTextParser._ocr_page returns empty string when convert_from_path fails.

    A poppler error (or any exception from convert_from_path) must be swallowed;
    the page is skipped rather than aborting the whole document.
    """
    pdf_file = tmp_path / "bad_render.pdf"
    pdf_file.write_bytes(b"%PDF fake")
    src = SourceRef(uri="file:///bad_render.pdf", mime="application/pdf", path=pdf_file)

    fake_reader = _make_fake_pdf_reader([""])
    ocr = FakeOcrEngine("text", is_available=True)

    with _inject_pypdf(fake_reader), _inject_pdf2image_raising(RuntimeError("poppler not found")):
        parser = PdfTextParser(_CFG, ocr=ocr)
        # convert_from_path raises → _ocr_page returns "" → no text → ParseError
        with pytest.raises(ParseError, match="no extractable text in file:///bad_render.pdf"):
            await parser.parse(src)


async def test_ocr_page_returns_empty_when_convert_returns_no_images(tmp_path: Path) -> None:
    """PdfTextParser._ocr_page returns empty string when convert_from_path yields no images."""
    pdf_file = tmp_path / "no_images.pdf"
    pdf_file.write_bytes(b"%PDF fake")
    src = SourceRef(uri="file:///no_images.pdf", mime="application/pdf", path=pdf_file)

    fake_reader = _make_fake_pdf_reader([""])
    ocr = FakeOcrEngine("text", is_available=True)

    # convert_from_path returns an empty list → _ocr_page returns ""
    with _inject_pypdf(fake_reader), _inject_pdf2image([]):
        parser = PdfTextParser(_CFG, ocr=ocr)
        with pytest.raises(ParseError, match="no extractable text in file:///no_images.pdf"):
            await parser.parse(src)


async def test_ocr_page_swallows_parse_error_from_ocr_engine(tmp_path: Path) -> None:
    """PdfTextParser._ocr_page swallows ParseError from the OCR engine and returns empty.

    If OCR itself raises ParseError (e.g. the image file is corrupt), _ocr_page
    must catch it and return "", so the page is silently skipped.
    """
    pdf_file = tmp_path / "ocr_fail.pdf"
    pdf_file.write_bytes(b"%PDF fake")
    src = SourceRef(uri="file:///ocr_fail.pdf", mime="application/pdf", path=pdf_file)

    fake_reader = _make_fake_pdf_reader([""])

    class _RaisingOcrEngine:
        """OCR engine that always raises ParseError."""

        def image_to_text(self, path: Path) -> str:
            """Raise ParseError unconditionally."""
            raise ParseError(f"OCR failed on {path}: simulated failure")

        def available(self) -> bool:
            """Always report as available."""
            return True

    fake_image = MagicMock()
    fake_image.save = MagicMock(return_value=None)

    with _inject_pypdf(fake_reader), _inject_pdf2image([fake_image]):
        parser = PdfTextParser(_CFG, ocr=_RaisingOcrEngine())
        # _ocr_page catches ParseError → returns "" → page skipped → no blocks
        with pytest.raises(ParseError, match="no extractable text in file:///ocr_fail.pdf"):
            await parser.parse(src)


# ---------------------------------------------------------------------------
# 7. TesseractOcrEngine — OCR failure and available() via injected modules
# ---------------------------------------------------------------------------


def test_tesseract_image_to_text_wraps_ocr_failure_as_parse_error(
    tmp_path: Path,
) -> None:
    """TesseractOcrEngine.image_to_text raises ParseError when OCR raises any exception."""
    img_file = tmp_path / "bad_img.png"
    img_file.write_bytes(b"PNG data")

    with _inject_pytesseract_and_pil(ocr_raises=OSError("tesseract not found")):
        engine = TesseractOcrEngine()
        with pytest.raises(ParseError, match=r"OCR failed on .*bad_img\.png"):
            engine.image_to_text(img_file)


def test_tesseract_image_to_text_returns_string_on_success(tmp_path: Path) -> None:
    """TesseractOcrEngine.image_to_text returns the OCR string on success."""
    img_file = tmp_path / "receipt.png"
    img_file.write_bytes(b"PNG data")

    with _inject_pytesseract_and_pil(ocr_text="Total: $42.00"):
        engine = TesseractOcrEngine()
        result = engine.image_to_text(img_file)

    assert result == "Total: $42.00"


def test_tesseract_available_returns_true_when_backend_reachable() -> None:
    """TesseractOcrEngine.available() returns True when pytesseract and binary are present."""
    with _inject_pytesseract_and_pil():
        engine = TesseractOcrEngine()
        assert engine.available() is True


def test_tesseract_available_returns_false_when_version_check_fails() -> None:
    """TesseractOcrEngine.available() returns False when get_tesseract_version raises."""
    with _inject_pytesseract_and_pil(version_raises=OSError("tesseract binary not found")):
        engine = TesseractOcrEngine()
        assert engine.available() is False
