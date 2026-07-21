"""
arxiv_client.py

Provides ArxivClient for searching and retrieving papers from arXiv.
"""

from datetime import datetime, timedelta
from typing import Any

import arxiv
import pandas as pd

_SORT_CRITERIA = {
    "relevance": arxiv.SortCriterion.Relevance,
    "submitted_date": arxiv.SortCriterion.SubmittedDate,
    "last_updated": arxiv.SortCriterion.LastUpdatedDate,
}


def resolve_sort_criterion(name: str | None) -> arxiv.SortCriterion:
    """Map a config string to an arXiv sort criterion, defaulting to relevance.

    Relevance is the sensible default for research briefs: sorting by submission
    date silently biases the corpus toward the newest papers and can drop the
    most relevant (often older, seminal) work.
    """
    if not name:
        return arxiv.SortCriterion.Relevance
    try:
        return _SORT_CRITERIA[name.strip().lower()]
    except KeyError as exc:
        raise ValueError(
            f"unknown arxiv sort '{name}'; expected one of {sorted(_SORT_CRITERIA)}"
        ) from exc


class ArxivClient:
    """
    Client for fetching and processing arXiv papers.
    """

    def __init__(self):
        """Initializes the arXiv API client."""
        self.client = arxiv.Client()

    @staticmethod
    def normalize_result(result: Any) -> dict:
        """Convert an arxiv.Result into the service paper payload."""
        paper_id = result.entry_id.split("/")[-1]
        return {
            "id": paper_id,
            "title": result.title.strip(),
            "summary": result.summary.strip(),
            "authors": [author.name for author in result.authors],
            "published": result.published,
            "updated": result.updated,
            "categories": list(result.categories or []),
            "primary_category": result.primary_category,
            "pdf_url": result.pdf_url,
            "arxiv_url": f"https://arxiv.org/abs/{paper_id}",
            "links": [link.href for link in result.links],
        }

    def search_papers(
        self,
        query: str = "cat:astro-ph.GA",
        max_results: int = 100,
        sort_by: arxiv.SortCriterion = arxiv.SortCriterion.Relevance,
    ) -> list[dict]:
        """
        Search arXiv for papers matching a query.

        Args:
            query (str): arXiv search query string (e.g., "cat:cs.AI").
            max_results (int): Maximum number of papers to fetch.
            sort_by (arxiv.SortCriterion): Sort criterion (e.g., by date).

        Returns:
            List[Dict]: List of paper metadata dictionaries.
        """
        search = arxiv.Search(query=query, max_results=max_results, sort_by=sort_by)

        return [self.normalize_result(result) for result in self.client.results(search)]

    def get_recent_papers(self, category: str = "cs.AI", days: int = 7) -> list[dict]:
        """Get papers submitted to an arXiv category within the recent window."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        query = (
            f"cat:{category} AND "
            f"submittedDate:[{start_date.strftime('%Y%m%d')}* TO "
            f"{end_date.strftime('%Y%m%d')}*]"
        )
        return self.search_papers(
            query=query,
            max_results=50,
            sort_by=arxiv.SortCriterion.SubmittedDate,
        )

    @staticmethod
    def papers_to_dataframe(papers: list[dict]) -> pd.DataFrame:
        """Convert normalized paper records to a pandas DataFrame."""
        return pd.DataFrame(papers)
