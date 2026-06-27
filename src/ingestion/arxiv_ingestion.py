from datetime import datetime

from src.arxiv_client import ArxivClient
from src.models import DateRange, PaperRecord


def _date_token(value: str) -> str:
    if len(value) == 8 and value.isdigit():
        return value
    return datetime.fromisoformat(value).strftime("%Y%m%d")


def build_arxiv_query(query: str, date_range: DateRange | None = None) -> str:
    query = query.strip()
    if not query:
        raise ValueError("query cannot be empty")
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
