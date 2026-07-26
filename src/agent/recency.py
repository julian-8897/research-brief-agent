from __future__ import annotations

import re

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
