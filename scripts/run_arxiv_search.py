"""Minimal end-to-end example: fetch arXiv papers, embed, and semantic-search.

Legacy demo (see AGENTS.md). Infra config (embedding model/adapters, arXiv sort
order) is sourced from the central `Settings`; only the demo-specific inputs
(fetch query, search phrase, counts) are exposed as CLI arguments so nothing is
hardcoded in the body.
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.arxiv_client import ArxivClient, resolve_sort_criterion
from src.embeddings import TextEmbedder
from src.settings import get_settings
from src.vector_store import VectorStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--query",
        default="cat:cs.AI",
        help="arXiv fetch query (e.g. a category or Boolean expression).",
    )
    parser.add_argument(
        "--search",
        default="diffusion model",
        help="Free-text phrase to semantically search the fetched corpus.",
    )
    parser.add_argument(
        "--max-results", type=int, default=100, help="Papers to fetch and index."
    )
    parser.add_argument(
        "--top-k", type=int, default=5, help="Nearest neighbours to return."
    )
    args = parser.parse_args()

    settings = get_settings()
    client = ArxivClient()
    embedder = TextEmbedder(
        settings.embedding_model,
        document_adapter=settings.embedding_document_adapter,
        query_adapter=settings.embedding_query_adapter,
    )

    papers = client.search_papers(
        args.query,
        max_results=args.max_results,
        sort_by=resolve_sort_criterion(settings.arxiv_sort),
    )

    title_embeddings = embedder.encode_documents([paper["title"] for paper in papers])

    store = VectorStore(title_embeddings.shape[1])
    store.add_papers(papers, title_embeddings)

    query_embedding = embedder.encode_queries([args.search])
    results = store.search(query_embedding[0], k=args.top_k)

    print(len(results))


if __name__ == "__main__":
    main()
