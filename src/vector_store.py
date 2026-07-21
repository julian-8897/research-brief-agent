from src.retrieval.store import InMemoryVectorStore


class VectorStore(InMemoryVectorStore):
    """Backward-compatible alias for older scripts.

    New code should import :class:`src.retrieval.InMemoryVectorStore`.
    """

    def __init__(self, embedding_dimension: int):
        super().__init__(embedding_dimension=embedding_dimension)

    def add_papers(self, papers, embeddings):
        """Forward the legacy method name to the canonical store API."""
        return self.upsert(papers, embeddings)
