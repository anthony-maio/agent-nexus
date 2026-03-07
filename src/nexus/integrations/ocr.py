"""Document text extraction for Discord attachments.

Provides a two-tier extraction pipeline:

1. **pymupdf** (fast, lightweight) — handles text-based PDFs in milliseconds.
2. **docling** (OCR fallback) — handles scanned/image PDFs via ML models.

Both libraries run synchronously so all extraction is wrapped in
:func:`asyncio.to_thread` to avoid blocking the Discord event loop.

Usage::

    from nexus.integrations.ocr import DocumentExtractor

    extractor = DocumentExtractor()
    text = await extractor.extract_from_url(url, "paper.pdf")
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
from pathlib import Path

import aiohttp

log = logging.getLogger(__name__)

# Probe for optional OCR backend.
try:
    from docling.document_converter import DocumentConverter  # noqa: F401

    _HAS_DOCLING = True
except Exception:  # ImportError or missing model deps
    _HAS_DOCLING = False

# Probe for pymupdf (primary extractor).
try:
    import pymupdf  # noqa: F401

    _HAS_PYMUPDF = True
except ImportError:
    _HAS_PYMUPDF = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_DOWNLOAD_BYTES: int = 25 * 1024 * 1024  # 25 MB (Discord limit)
_MAX_TEXT_CHARS: int = 15_000  # Cap injected text to fit model context

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {".pdf", ".docx", ".pptx", ".png", ".jpg", ".jpeg"}
)


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------


class DocumentExtractor:
    """Extract text from documents using pymupdf with docling OCR fallback.

    The extractor downloads a file from a URL to a temporary location,
    determines the extraction strategy based on file extension, and returns
    the extracted text.  Temporary files are cleaned up after extraction.
    """

    async def extract_from_url(self, url: str, filename: str) -> str | None:
        """Download a file from *url* and extract text.

        Args:
            url: Direct download URL (e.g. ``discord.Attachment.url``).
            filename: Original filename for extension detection.

        Returns:
            Extracted text (capped at 15 000 chars), or ``None`` on failure.
        """
        ext = Path(filename).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            return None

        if not _HAS_PYMUPDF and not _HAS_DOCLING:
            log.warning("No document extraction library available. Install pymupdf or docling.")
            return None

        # Download to a temporary file.
        tmp_path: Path | None = None
        try:
            tmp_path = await self._download(url, ext)
            if tmp_path is None:
                return None

            if ext == ".pdf":
                text = await self._extract_pdf(tmp_path)
            elif ext in {".png", ".jpg", ".jpeg"}:
                text = await self._extract_image(tmp_path)
            else:
                # .docx, .pptx — docling handles these natively
                text = await self._extract_office(tmp_path)

            if text:
                return text[:_MAX_TEXT_CHARS]
            return None

        except Exception:
            log.exception("Document extraction failed for %s", filename)
            return None
        finally:
            if tmp_path is not None:
                try:
                    tmp_path.unlink(missing_ok=True)
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    async def _download(self, url: str, ext: str) -> Path | None:
        """Download a file to a named temp file and return its path."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as resp:
                    if resp.status != 200:
                        log.warning("Download failed: HTTP %d", resp.status)
                        return None

                    size = int(resp.headers.get("Content-Length", 0))
                    if size > _MAX_DOWNLOAD_BYTES:
                        log.warning(
                            "Attachment too large (%d bytes, max %d).",
                            size,
                            _MAX_DOWNLOAD_BYTES,
                        )
                        return None

                    tmp = tempfile.NamedTemporaryFile(
                        suffix=ext,
                        delete=False,
                    )
                    tmp_path = Path(tmp.name)
                    data = await resp.read()
                    tmp.write(data)
                    tmp.close()
                    return tmp_path

        except Exception:
            log.exception("Failed to download attachment.")
            return None

    # ------------------------------------------------------------------
    # PDF extraction (two-tier)
    # ------------------------------------------------------------------

    async def _extract_pdf(self, path: Path) -> str:
        """Extract text from a PDF — pymupdf first, docling fallback."""
        # Tier 1: pymupdf (fast, text-based PDFs)
        if _HAS_PYMUPDF:
            text = await asyncio.to_thread(self._pymupdf_extract, path)
            if text and len(text.strip()) > 10:
                log.debug("pymupdf extracted %d chars from %s", len(text), path.name)
                return text

        # Tier 2: docling OCR (scanned/image PDFs)
        if _HAS_DOCLING:
            log.info("pymupdf yielded no text — trying docling OCR for %s", path.name)
            text = await asyncio.to_thread(self._docling_extract, path)
            if text:
                log.debug("docling extracted %d chars from %s", len(text), path.name)
                return text

        return ""

    # ------------------------------------------------------------------
    # Image extraction (docling only)
    # ------------------------------------------------------------------

    async def _extract_image(self, path: Path) -> str:
        """Extract text from an image via docling OCR."""
        if not _HAS_DOCLING:
            log.warning("Image OCR requires docling. Install with: pip install docling")
            return ""
        return await asyncio.to_thread(self._docling_extract, path)

    # ------------------------------------------------------------------
    # Office document extraction
    # ------------------------------------------------------------------

    async def _extract_office(self, path: Path) -> str:
        """Extract text from office documents (.docx, .pptx) via docling."""
        if _HAS_DOCLING:
            return await asyncio.to_thread(self._docling_extract, path)
        log.warning(
            "Office document extraction requires docling. Install with: pip install docling"
        )
        return ""

    # ------------------------------------------------------------------
    # Synchronous backends (run in thread pool)
    # ------------------------------------------------------------------

    @staticmethod
    def _pymupdf_extract(path: Path) -> str:
        """Extract text from a PDF using pymupdf."""
        import pymupdf

        doc = pymupdf.open(str(path))
        pages: list[str] = []
        for page in doc:
            text = page.get_text("text")
            if text.strip():
                pages.append(text)
        doc.close()
        return "\n\n".join(pages)

    @staticmethod
    def _docling_extract(path: Path) -> str:
        """Extract text from a document using docling."""
        from docling.document_converter import DocumentConverter

        converter = DocumentConverter()
        result = converter.convert(str(path))
        return result.document.export_to_markdown()
