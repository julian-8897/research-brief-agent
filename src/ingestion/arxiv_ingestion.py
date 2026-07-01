import re
from datetime import datetime

from src.arxiv_client import ArxivClient
from src.models import DateRange, PaperRecord

# arXiv field prefixes (https://info.arxiv.org/help/api/user-manual.html).
# A query that already names a field is treated as power-user syntax and passed
# through untouched; anything else is a plain query we wrap in ``all:``.
_FIELD_PREFIX = re.compile(r"\b(?:ti|au|abs|co|jr|cat|rn|id|all):", re.IGNORECASE)


def _date_token(value: str) -> str:
    if len(value) == 8 and value.isdigit():
        return value
    return datetime.fromisoformat(value).strftime("%Y%m%d")


def _normalize_query(query: str) -> str:
    """Wrap a plain query in ``all:`` so general phrases work without arXiv syntax.

    ``"long context transformers"`` becomes ``"all:long context transformers"``,
    while field-qualified queries (``cat:cs.LG AND all:rag``) are left as-is.
    """
    if _FIELD_PREFIX.search(query):
        return query
    return f"all:{query}"


def build_arxiv_query(query: str, date_range: DateRange | None = None) -> str:
    query = query.strip()
    if not query:
        raise ValueError("query cannot be empty")
    query = _normalize_query(query)
    if not date_range or not (date_range.start or date_range.end):
        return query
    start = _date_token(date_range.start) if date_range.start else "19910101"
    end = _date_token(date_range.end) if date_range.end else "29991231"
    return f"({query}) AND submittedDate:[{start}* TO {end}*]"


def fetch_arxiv_papers(
    client: ArxivClient,
    query: str,
    max_papers: int,
    date_range: DateRange | None = None,
) -> list[PaperRecord]:
    full_query = build_arxiv_query(query, date_range)
    return [
        PaperRecord(**paper) for paper in client.search_papers(full_query, max_papers)
    ]
