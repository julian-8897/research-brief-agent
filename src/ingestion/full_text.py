"""Fetch and extract the full body text of an arXiv paper from its PDF.

Deployable and self-contained: downloads the PDF over HTTP and extracts text
with pypdf, so it runs inside the service (no external MCP or hosted API).
Page count and character budget are bounded to keep latency and tokens in check.
"""

from __future__ import annotations

import io

# Hard cap on pages parsed, independent of the character budget, so a huge or
# malformed PDF cannot blow up parse time.
_MAX_PAGES = 30


def fetch_arxiv_fulltext(
    pdf_url: str, *, timeout: float = 20.0, char_budget: int = 12000
) -> tuple[str, bool]:
    """Return ``(text, truncated)`` for a paper PDF, capped at ``char_budget``."""
    data = _download(pdf_url, timeout)
    text = _extract_text(data, char_budget)
    truncated = len(text) > char_budget
    return text[:char_budget].strip(), truncated


def _download(url: str, timeout: float) -> bytes:
    import httpx

    response = httpx.get(url, timeout=timeout, follow_redirects=True)
    response.raise_for_status()
    return response.content


def _extract_text(data: bytes, char_budget: int) -> str:
    from pypdf import PdfReader

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
