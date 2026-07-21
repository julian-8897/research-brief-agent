"""Fetch and extract the full body text of an arXiv paper from its PDF.

Deployable by default: downloads the PDF over HTTP and extracts text with the
configured local extractor. ``pypdf`` is the lightweight fallback; Docling can be
enabled as an optional layout-aware backend.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.ingestion.pdf_extractors import (
    PdfExtractionError,
    PdfExtractorName,
    extract_pdf,
)


@dataclass
class FullTextFetchError(RuntimeError):
    """Classified full-text failure safe to report through tool payloads."""

    code: str
    message: str
    url: str | None = None
    status_code: int | None = None

    def __post_init__(self) -> None:
        super().__init__(self.message)


def fetch_arxiv_fulltext(
    pdf_url: str,
    *,
    timeout: float = 20.0,
    char_budget: int = 12000,
    extractor: PdfExtractorName = "auto",
) -> tuple[str, bool]:
    """Return ``(text, truncated)`` for a paper PDF, capped at ``char_budget``."""
    if not pdf_url:
        raise ValueError("pdf_url must be a non-empty URL")
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    if char_budget <= 0:
        raise ValueError("char_budget must be positive")

    data = _download(pdf_url, timeout)
    result = _extract_text(data, char_budget, extractor=extractor)
    if not result.text:
        raise FullTextFetchError(
            "empty_text",
            f"PDF text extraction returned no text for {pdf_url}",
            url=pdf_url,
        )
    return result.text, result.truncated


def _download(url: str, timeout: float) -> bytes:
    import httpx

    try:
        response = httpx.get(url, timeout=timeout, follow_redirects=True)
        response.raise_for_status()
    except httpx.TimeoutException as exc:
        raise FullTextFetchError(
            "timeout",
            f"Timed out fetching PDF from {url}",
            url=url,
        ) from exc
    except httpx.HTTPStatusError as exc:
        status_code = exc.response.status_code
        raise FullTextFetchError(
            "http_error",
            f"PDF request failed with HTTP {status_code} for {url}",
            url=url,
            status_code=status_code,
        ) from exc
    except httpx.RequestError as exc:
        raise FullTextFetchError(
            "network_error",
            f"Network error fetching PDF from {url}: {exc}",
            url=url,
        ) from exc
    return response.content


def _extract_text(
    data: bytes, char_budget: int, *, extractor: PdfExtractorName = "auto"
):
    try:
        return extract_pdf(data, char_budget=char_budget, extractor=extractor)
    except PdfExtractionError as exc:
        code = (
            "extractor_unavailable"
            if exc.code == "extractor_unavailable"
            else "parse_error"
        )
        raise FullTextFetchError(
            code,
            exc.message,
        ) from exc
