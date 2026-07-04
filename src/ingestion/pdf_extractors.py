"""PDF text extraction backends.

The service keeps a small pypdf fallback for deployability, while allowing a
layout-aware Docling path when the optional dependency is installed.
"""

from __future__ import annotations

import io
import re
import tempfile
from dataclasses import dataclass, field
from typing import Literal

PdfExtractorName = Literal["auto", "pypdf", "docling"]

# Hard cap on pages parsed by lightweight extractors, independent of the
# character budget, so a huge or malformed PDF cannot blow up parse time.
MAX_PDF_PAGES = 30


@dataclass
class PdfExtractionResult:
    text: str
    truncated: bool
    extractor: str
    page_count: int | None = None
    markdown: str | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass
class PdfExtractionError(RuntimeError):
    code: str
    message: str
    extractor: str | None = None

    def __post_init__(self) -> None:
        super().__init__(self.message)


def extract_pdf(
    data: bytes, *, char_budget: int, extractor: PdfExtractorName = "auto"
) -> PdfExtractionResult:
    """Extract bounded text from PDF bytes using the requested backend."""
    if char_budget <= 0:
        raise ValueError("char_budget must be positive")
    if extractor not in ("auto", "pypdf", "docling"):
        raise ValueError("extractor must be one of: auto, pypdf, docling")

    if extractor == "pypdf":
        return _extract_with_pypdf(data, char_budget=char_budget)
    if extractor == "docling":
        return _extract_with_docling(data, char_budget=char_budget)

    try:
        return _extract_with_docling(data, char_budget=char_budget)
    except PdfExtractionError as exc:
        if exc.code != "extractor_unavailable":
            fallback = _extract_with_pypdf(data, char_budget=char_budget)
            fallback.warnings.append(
                f"Docling extraction failed; fell back to pypdf ({exc.message})"
            )
            return fallback
        fallback = _extract_with_pypdf(data, char_budget=char_budget)
        fallback.warnings.append("Docling is not installed; used pypdf fallback.")
        return fallback


def _extract_with_pypdf(data: bytes, *, char_budget: int) -> PdfExtractionResult:
    from pypdf import PdfReader

    try:
        reader = PdfReader(io.BytesIO(data))
        parts: list[str] = []
        total = 0
        pages = reader.pages[:MAX_PDF_PAGES]
        for page in pages:
            chunk = page.extract_text() or ""
            parts.append(chunk)
            total += len(chunk)
            if total > char_budget:
                break
        text = "\n".join(parts)
        return _result(
            text,
            char_budget=char_budget,
            extractor="pypdf",
            page_count=len(reader.pages),
        )
    except Exception as exc:
        raise PdfExtractionError(
            "parse_error",
            f"Could not parse PDF with pypdf: {exc}",
            extractor="pypdf",
        ) from exc


def _extract_with_docling(data: bytes, *, char_budget: int) -> PdfExtractionResult:
    try:
        from docling.document_converter import DocumentConverter
    except ImportError as exc:
        raise PdfExtractionError(
            "extractor_unavailable",
            "Docling is not installed. Install the pdf extra or set PDF_EXTRACTOR=pypdf.",
            extractor="docling",
        ) from exc

    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf") as handle:
            handle.write(data)
            handle.flush()
            conversion = DocumentConverter().convert(handle.name)
        document = conversion.document
        markdown = _export_docling_markdown(document)
        text = markdown or _export_docling_text(document)
        return _result(
            text,
            char_budget=char_budget,
            extractor="docling",
            markdown=markdown or None,
            page_count=_docling_page_count(document),
        )
    except PdfExtractionError:
        raise
    except Exception as exc:
        raise PdfExtractionError(
            "parse_error",
            f"Could not parse PDF with Docling: {exc}",
            extractor="docling",
        ) from exc


def _export_docling_markdown(document) -> str:
    export = getattr(document, "export_to_markdown", None)
    if callable(export):
        return str(export() or "")
    return ""


def _export_docling_text(document) -> str:
    export = getattr(document, "export_to_text", None)
    if callable(export):
        return str(export() or "")
    return ""


def _docling_page_count(document) -> int | None:
    pages = getattr(document, "pages", None)
    if pages is None:
        return None
    try:
        return len(pages)
    except TypeError:
        return None


def _result(
    text: str,
    *,
    char_budget: int,
    extractor: str,
    page_count: int | None = None,
    markdown: str | None = None,
) -> PdfExtractionResult:
    normalized = _normalize_text(text)
    truncated = len(normalized) > char_budget
    bounded = normalized[:char_budget].strip()
    bounded_markdown = markdown[:char_budget].strip() if markdown else None
    return PdfExtractionResult(
        text=bounded,
        truncated=truncated,
        extractor=extractor,
        page_count=page_count,
        markdown=bounded_markdown,
    )


def _normalize_text(text: str) -> str:
    lines = [line.rstrip() for line in str(text).splitlines()]
    compacted = "\n".join(lines)
    compacted = re.sub(r"\n{3,}", "\n\n", compacted)
    return compacted.strip()
