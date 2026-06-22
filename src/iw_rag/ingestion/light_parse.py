"""Light-weight, dependency-optional parsers for the online upload path.

The offline pipeline parses PDF/DOCX with Docling (the heavy ``ingest`` dep
group).  The document-management *upload* endpoint runs inside the API container,
which does not carry Docling/torch — so it uses these light parsers instead:

- ``text/*`` / markdown  → :class:`~iw_rag.ingestion.parse.TextParser` (no deps).
- ``application/pdf``     → :class:`PdfTextParser` (born-digital text via ``pypdf``;
  falls back to per-page OCR when a page has no embedded text and an OCR engine
  is available — this is how scanned/image invoices become queryable).
- ``image/*``             → :class:`OcrImageParser` (text *inside* the image via
  an injected :class:`OcrEngine`, default Tesseract — CPU-only, fully local).

OCR is wired through the :class:`OcrEngine` protocol so tests inject a fake and
never require the ``tesseract`` binary; the default :class:`TesseractOcrEngine`
lazy-imports ``pytesseract``/``Pillow`` and degrades to a clear ``ParseError``
when they are absent.

A :class:`CharTokenizer` (lossless char-level encode/decode) feeds the chunker on
this path: leaf token spans are unused (the pooled HTTP embedder ignores them),
so a real model tokenizer is unnecessary and the heavy transformers dep is avoided.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from iw_rag.domain.errors import ParseError
from iw_rag.ingestion.parse import (
    Block,
    ParseConfig,
    ParsedDocument,
    Parser,
    SourceRef,
    TextParser,
    _split_markdown_blocks,
)

if TYPE_CHECKING:
    from iw_rag.ingestion.caption import CaptionEngine

# Char-budget per leaf for the online path (~4 chars/token ⇒ ~400 tokens), well
# under the BGE-M3 8192-token window so each chunk embeds in one pooled call.
ONLINE_LEAF_CHARS = 1600

_TEXT_MIMES = ("text/markdown", "text/plain", "text/x-markdown")
_IMAGE_MIME_PREFIX = "image/"


class CharTokenizer:
    """Lossless character-level tokenizer (Unicode code points).

    Implements the :class:`~iw_rag.ingestion.chunk.Tokenizer` protocol.  Used only
    to size/split chunks on the online path; spans are never pooled, so fidelity
    of chunk *text* (exact round-trip) is what matters, not model-token alignment.
    """

    def encode(self, text: str) -> list[int]:
        """Encode text to a list of Unicode code points."""
        return [ord(ch) for ch in text]

    def decode(self, ids: list[int]) -> str:
        """Decode code points back to the exact original text."""
        return "".join(chr(i) for i in ids)


@runtime_checkable
class OcrEngine(Protocol):
    """Extracts text from a raster image file (CPU-local OCR)."""

    def image_to_text(self, path: Path) -> str:
        """Return the recognised text in the image at ``path`` (may be empty)."""
        ...

    def available(self) -> bool:
        """Whether the OCR backend is usable in this environment."""
        ...


class TesseractOcrEngine:
    """Default OCR engine — Tesseract via ``pytesseract`` (lazy-imported)."""

    def image_to_text(self, path: Path) -> str:
        """OCR the image with Tesseract.

        Raises:
            ParseError: If ``pytesseract``/``Pillow`` or the tesseract binary are
                unavailable, or OCR fails.
        """
        try:
            import pytesseract  # noqa: PLC0415
            from PIL import Image  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - needs the docman group
            raise ParseError(
                "OCR requires the 'docman' dependency group (pytesseract + Pillow) "
                "and the tesseract binary."
            ) from exc
        try:
            with Image.open(path) as img:
                return str(pytesseract.image_to_string(img))
        except Exception as exc:  # noqa: BLE001 - map any OCR failure
            raise ParseError(f"OCR failed on {path}: {exc}") from exc

    def available(self) -> bool:
        """True when pytesseract + the tesseract binary are importable/runnable."""
        try:
            import pytesseract  # noqa: PLC0415
            from PIL import Image  # noqa: F401, PLC0415

            pytesseract.get_tesseract_version()
        except Exception:  # noqa: BLE001 - any import/binary error → unavailable
            return False
        return True


class PdfTextParser:
    """Born-digital PDF text extraction via ``pypdf``, with per-page OCR fallback."""

    def __init__(self, cfg: ParseConfig, ocr: OcrEngine | None = None) -> None:
        """Store config + optional OCR engine (for image-only/scanned pages)."""
        self._cfg = cfg
        self._ocr = ocr

    async def parse(self, source: SourceRef) -> ParsedDocument:
        """Extract text per page; OCR pages with no embedded text when possible.

        Raises:
            ParseError: If ``pypdf`` is unavailable or the file cannot be read.
        """
        try:
            from pypdf import PdfReader  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - needs the docman group
            raise ParseError(
                "PdfTextParser requires the 'docman' dependency group (pypdf)."
            ) from exc
        path = source.local_path()
        try:
            reader = PdfReader(str(path))
        except Exception as exc:  # noqa: BLE001 - map any pypdf failure
            raise ParseError(f"pypdf failed on {source.uri}: {exc}") from exc

        blocks: list[Block] = []
        page_texts: list[str] = []
        for idx, page in enumerate(reader.pages):
            try:
                text = (page.extract_text() or "").strip()
            except Exception:  # noqa: BLE001 - a bad page must not abort the doc
                text = ""
            if not text and self._ocr is not None and self._ocr.available():
                text = self._ocr_page(path, idx).strip()
            if not text:
                continue
            page_texts.append(text)
            blocks.append(Block(text=text, block_type="text", page=idx + 1))

        if not blocks:
            raise ParseError(
                f"no extractable text in {source.uri} (scanned PDF without OCR support?)"
            )
        full_text = "\n\n".join(page_texts)
        title = Path(path).stem
        return ParsedDocument(
            source_uri=source.uri, blocks=blocks, full_text=full_text, title=title
        )

    def _ocr_page(self, pdf_path: Path, page_index: int) -> str:
        """Render one PDF page to an image and OCR it (best-effort, optional dep).

        Returns an empty string when page rendering (``pdf2image``/poppler) is not
        available — born-digital extraction already covers the common case.
        """
        try:
            from pdf2image import convert_from_path  # noqa: PLC0415
        except ImportError:
            return ""
        try:
            images = convert_from_path(
                str(pdf_path), first_page=page_index + 1, last_page=page_index + 1
            )
        except Exception:  # noqa: BLE001 - poppler missing / render error
            return ""
        if not images or self._ocr is None:
            return ""
        tmp = pdf_path.with_suffix(f".p{page_index + 1}.png")
        try:
            images[0].save(tmp)
            return self._ocr.image_to_text(tmp)
        except ParseError:
            return ""
        finally:
            tmp.unlink(missing_ok=True)


class OcrImageParser:
    """Parser for raster image files — OCR for in-image text, VLM caption fallback.

    Strategy: read the *text* inside the image with OCR first (invoices, scans,
    screenshots).  When OCR finds nothing AND a visual caption engine is available,
    fall back to a VLM that *describes the pixels* (charts, diagrams, photos) so the
    image is still queryable.  Either way the extracted/described text flows through
    the normal embed → chunk → ACL pipeline.
    """

    def __init__(
        self, cfg: ParseConfig, ocr: OcrEngine, caption_engine: CaptionEngine | None = None
    ) -> None:
        """Store config + the OCR engine + an optional VLM caption fallback."""
        self._cfg = cfg
        self._ocr = ocr
        self._caption = caption_engine

    async def parse(self, source: SourceRef) -> ParsedDocument:
        """Extract text (OCR) or, failing that, a VLM caption of the image.

        Raises:
            ParseError: If neither OCR nor a caption engine yields any text.
        """
        path = source.local_path()
        text = ""
        if self._ocr.available():
            text = self._ocr.image_to_text(path).strip()
        block_type: str = "text"
        if not text and self._caption is not None and await self._caption.available():
            caption = await self._caption.caption_image(
                _read_bytes(path), mime=source.mime, hint=Path(path).stem
            )
            if caption:
                text = caption.strip()
                block_type = "figure_caption"
        if not text:
            raise ParseError(
                f"could not extract text from image {source.uri}: OCR found nothing and "
                "no visual-caption VLM is available (install the 'docman' group + tesseract, "
                "or enable + serve visual mode)."
            )
        title = Path(path).stem
        block = Block(text=text, block_type=block_type, page=1)  # type: ignore[arg-type]
        return ParsedDocument(source_uri=source.uri, blocks=[block], full_text=text, title=title)


def _read_bytes(path: Path) -> bytes:
    """Read a file's bytes (blocking helper; kept out of async bodies for lint)."""
    return path.read_bytes()


def select_light_parser(
    source: SourceRef,
    cfg: ParseConfig,
    ocr: OcrEngine | None = None,
    caption_engine: CaptionEngine | None = None,
) -> Parser:
    """Route an uploaded source to a dependency-light parser (no Docling/torch).

    Args:
        source: The source to route (``mime`` drives selection).
        cfg: Parser-selection config.
        ocr: OCR engine for image/scanned inputs (default Tesseract when None).
        caption_engine: Optional VLM caption fallback for images OCR can't read.

    Returns:
        The selected parser instance.

    Raises:
        ParseError: For an unsupported MIME type.
    """
    engine = ocr or TesseractOcrEngine()
    mime = source.mime
    if mime in _TEXT_MIMES:
        return TextParser(cfg)
    if mime == "application/pdf":
        return PdfTextParser(cfg, engine)
    if mime.startswith(_IMAGE_MIME_PREFIX):
        return OcrImageParser(cfg, engine, caption_engine)
    # Fall back to text for unknown-but-textual types (best-effort).
    if mime.startswith("text/"):
        return TextParser(cfg)
    raise ParseError(f"unsupported upload type '{mime}' for {source.uri}")


# A small re-export of the markdown splitter for callers that already hold text.
__all__ = [
    "CharTokenizer",
    "OcrEngine",
    "OcrImageParser",
    "PdfTextParser",
    "TesseractOcrEngine",
    "ONLINE_LEAF_CHARS",
    "select_light_parser",
    "_split_markdown_blocks",
]
