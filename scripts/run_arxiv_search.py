import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.arxiv_client import ArxivClient
from src.embeddings import TextEmbedder
from src.vector_store import VectorStore


def main():
    client = ArxivClient()
    embedder = TextEmbedder()

    papers = client.search_papers("cat:cs.AI", max_results=100)

    title_embeddings = embedder.encode_documents([paper["title"] for paper in papers])

    store = VectorStore(title_embeddings.shape[1])
    store.add_papers(papers, title_embeddings)

    query_embedding = embedder.encode_queries(["diffusion model"])
    results = store.search(query_embedding[0], k=5)

    print(len(results))


if __name__ == "__main__":
    main()
