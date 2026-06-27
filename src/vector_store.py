from src.retrieval.store import InMemoryVectorStore


class VectorStore(InMemoryVectorStore):
    """Backward-compatible alias for older scripts."""

    def __init__(self, embedding_dimension: int):
        super().__init__(embedding_dimension=embedding_dimension)

    def add_papers(self, papers, embeddings):
        return self.upsert(papers, embeddings)
