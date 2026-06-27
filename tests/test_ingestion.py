from src.ingestion import build_arxiv_query
from src.models import DateRange


def test_build_arxiv_query_without_dates():
    assert build_arxiv_query("cat:cs.LG") == "cat:cs.LG"


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
