from __future__ import annotations

import re
from datetime import UTC, date, datetime

from src.models import BriefRequest

_RECENCY_PATTERNS = (
    re.compile(
        r"\b(?:latest|newest|recent|recently|currently|up[- ]to[- ]date)\b", re.I
    ),
    re.compile(
        r"\bcurrent\s+(?:best|leading|top|frontier|landscape|state|models?|"
        r"products?|releases?|versions?|leaderboards?|benchmarks?|pricing)\b",
        re.I,
    ),
    re.compile(r"\bstate[- ]of[- ]the[- ]art\b", re.I),
    re.compile(
        r"\bas\s+of\s+(?:today|now|(?:[a-z]+\s+)?\d{4}|"
        r"\d{4}-\d{1,2}(?:-\d{1,2})?)\b",
        re.I,
    ),
)
_EXPLICIT_AS_OF_PATTERN = re.compile(
    r"\bas\s+of\s+((?:[a-z]+\s+)?\d{4}(?:-\d{1,2}(?:-\d{1,2})?)?)\b",
    re.I,
)
_YEAR_PATTERN = re.compile(r"\b(?:19|20)\d{2}\b")

ARXIV_RECENCY_CAVEAT = (
    "Freshness is limited to papers discoverable on arXiv. This is not a complete "
    "catalogue of current proprietary products, releases, prices, or live leaderboard "
    "results."
)

WEB_RECENCY_CAVEAT = (
    "Current-product evidence came from a bounded allow-list of web sources and "
    "arXiv. It is a point-in-time snapshot, not a complete catalogue of releases, "
    "prices, or live leaderboard results."
)


def has_recency_intent(request: BriefRequest) -> bool:
    text = " ".join(
        [
            request.research_question,
            request.domain or "",
            *request.constraints,
        ]
    )
    return any(pattern.search(text) for pattern in _RECENCY_PATTERNS)


def recency_reference_date(
    request: BriefRequest,
    *,
    today: date | None = None,
) -> str:
    """Return the user-supplied evidence cutoff, or today's UTC date."""
    if request.date_range and request.date_range.end:
        return request.date_range.end
    match = _EXPLICIT_AS_OF_PATTERN.search(request.research_question)
    if match:
        return match.group(1)
    return (today or datetime.now(UTC).date()).isoformat()


def build_current_web_query(
    request: BriefRequest,
    model_query: str,
    *,
    today: date | None = None,
) -> str:
    """Anchor a model-selected web query to the request and effective date."""
    anchor = recency_reference_date(request, today=today)
    focus = " ".join(model_query.split())
    if not _EXPLICIT_AS_OF_PATTERN.search(request.research_question):
        anchor_year = _YEAR_PATTERN.search(anchor)
        if anchor_year:
            focus = _YEAR_PATTERN.sub(anchor_year.group(0), focus)
    question = " ".join(request.research_question.split())
    return (
        f"Evidence current as of {anchor}. Decision question: {question}. "
        f"Search focus: {focus}. Prefer direct official releases, model cards, "
        "and live comparative benchmark results."
    )
