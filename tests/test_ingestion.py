import arxiv
import pytest

from src.arxiv_client import resolve_sort_criterion
from src.ingestion import build_arxiv_query, fetch_arxiv_papers
from src.models import DateRange


def test_build_arxiv_query_without_dates():
    assert build_arxiv_query("cat:cs.LG") == "cat:cs.LG"


def test_build_arxiv_query_wraps_plain_query():
    assert (
        build_arxiv_query("long context transformers")
        == "all:long context transformers"
    )


def test_build_arxiv_query_preserves_field_syntax():
    assert build_arxiv_query("cat:cs.LG AND all:rag") == "cat:cs.LG AND all:rag"


def test_build_arxiv_query_wraps_plain_query_with_dates():
    query = build_arxiv_query(
        "retrieval augmented generation",
        DateRange(start="2025-01-01", end="2025-01-31"),
    )
    assert query == (
        "(all:retrieval augmented generation) "
        "AND submittedDate:[20250101* TO 20250131*]"
    )


def test_build_arxiv_query_with_iso_dates():
    query = build_arxiv_query(
        "cat:cs.LG", DateRange(start="2025-01-01", end="2025-01-31")
    )
    assert query == "(cat:cs.LG) AND submittedDate:[20250101* TO 20250131*]"


def test_build_arxiv_query_rejects_empty_query():
    try:
        build_arxiv_query("  ")
    except ValueError as exc:
        assert "query cannot be empty" in str(exc)
    else:
        raise AssertionError("empty query should fail")


def test_resolve_sort_criterion_defaults_to_relevance():
    assert resolve_sort_criterion(None) == arxiv.SortCriterion.Relevance
    assert resolve_sort_criterion("relevance") == arxiv.SortCriterion.Relevance
    assert resolve_sort_criterion("submitted_date") == arxiv.SortCriterion.SubmittedDate
    assert resolve_sort_criterion("LAST_UPDATED") == arxiv.SortCriterion.LastUpdatedDate


def test_resolve_sort_criterion_rejects_unknown():
    with pytest.raises(ValueError, match="unknown arxiv sort"):
        resolve_sort_criterion("popularity")


class _RecordingArxiv:
    def __init__(self):
        self.calls: list[dict] = []

    def search_papers(self, query, max_results, sort_by=None):
        self.calls.append(
            {"query": query, "max_results": max_results, "sort_by": sort_by}
        )
        return []


def test_fetch_arxiv_papers_forwards_sort_criterion():
    client = _RecordingArxiv()
    fetch_arxiv_papers(client, query="rag", max_papers=5, sort="submitted_date")
    assert client.calls[0]["sort_by"] == arxiv.SortCriterion.SubmittedDate


def test_fetch_arxiv_papers_defaults_to_relevance():
    client = _RecordingArxiv()
    fetch_arxiv_papers(client, query="rag", max_papers=5)
    assert client.calls[0]["sort_by"] == arxiv.SortCriterion.Relevance
