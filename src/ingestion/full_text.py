"""Fetch and extract the full body text of an arXiv paper from its PDF.

Deployable and self-contained: downloads the PDF over HTTP and extracts text
with pypdf, so it runs inside the service (no external MCP or hosted API).
Page count and character budget are bounded to keep latency and tokens in check.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

# Hard cap on pages parsed, independent of the character budget, so a huge or
# malformed PDF cannot blow up parse time.
_MAX_PAGES = 30


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
    pdf_url: str, *, timeout: float = 20.0, char_budget: int = 12000
) -> tuple[str, bool]:
    """Return ``(text, truncated)`` for a paper PDF, capped at ``char_budget``."""
    if not pdf_url:
        raise ValueError("pdf_url must be a non-empty URL")
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    if char_budget <= 0:
        raise ValueError("char_budget must be positive")

    data = _download(pdf_url, timeout)
    text = _extract_text(data, char_budget)
    truncated = len(text) > char_budget
    text = text[:char_budget].strip()
    if not text:
        raise FullTextFetchError(
            "empty_text",
            f"PDF text extraction returned no text for {pdf_url}",
            url=pdf_url,
        )
    return text, truncated


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


def _extract_text(data: bytes, char_budget: int) -> str:
    from pypdf import PdfReader

    try:
        reader = PdfReader(io.BytesIO(data))
        parts: list[str] = []
        total = 0
        for page in reader.pages[:_MAX_PAGES]:
            chunk = page.extract_text() or ""
            parts.append(chunk)
            total += len(chunk)
            if total > char_budget:
                break
        return "\n".join(parts)
    except Exception as exc:
        raise FullTextFetchError(
            "parse_error",
            f"Could not parse PDF text: {exc}",
        ) from exc
