import json

import numpy as np

from src.agent import ResearchTools
from src.agent.toolset import ResearchToolset, _build_search_embedding_text
from src.llm import TurnResult
from src.models import BriefRequest, PaperRecord
from src.retrieval import InMemoryVectorStore
from src.settings import Settings


class RecordingEmbedder:
    def __init__(self, vector=None):
        self.texts: list[str] = []
        self.document_texts: list[str] = []
        self.query_texts: list[str] = []
        self._vector = vector or [1.0, 0.0]

    def _encode(self, texts):
        self.texts.extend(texts)
        return np.array([self._vector for _ in texts])

    def encode_documents(self, texts, batch_size=32):
        self.document_texts.extend(texts)
        return self._encode(texts)

    def encode_queries(self, texts, batch_size=32):
        self.query_texts.extend(texts)
        return self._encode(texts)


class RecordingArxivClient:
    def __init__(self, paper_id: str = "2501.00001"):
        self.calls: list[tuple[str, int]] = []
        self.paper_id = paper_id

    def search_papers(self, query, max_results, sort_by=None):
        self.calls.append((query, max_results))
        return [
            {
                "id": self.paper_id,
                "title": "Neural Operator Backfill",
                "summary": "Fourier neural operators and DeepONet for PDEs.",
                "authors": ["A. Researcher"],
                "categories": ["cs.LG"],
                "primary_category": "cs.LG",
                "arxiv_url": f"https://arxiv.org/abs/{self.paper_id}",
            }
        ][:max_results]


class FakeProvider:
    name = "fake"
    model = "fake"

    def __init__(self, text: str):
        self.text = text

    def run_turn(self, system, messages, tools, *, tool_choice="auto"):
        return TurnResult(
            text=self.text,
            tool_calls=[],
            input_tokens=1,
            output_tokens=1,
            model=self.model,
            stop_reason="end",
        )


def _paper(paper_id: str, title: str) -> PaperRecord:
    return PaperRecord(id=paper_id, title=title, summary="abstract")


def test_build_search_embedding_text_includes_descriptive_request_context():
    text = _build_search_embedding_text(
        "Which neural operators solve PDEs by learning maps between function spaces?",
        "neural operators",
        ["prefer methods with public baselines"],
    )

    assert "Which neural operators solve PDEs" in text
    assert "learning maps between function spaces" in text
    assert "Search focus: neural operators" in text
    assert "prefer methods with public baselines" in text


def test_search_papers_embeds_expanded_text_but_echoes_model_query():
    settings = Settings(vector_store_backend="memory", embedding_dimension=2)
    store = InMemoryVectorStore(embedding_dimension=2)
    store.upsert([_paper("1", "Neural Operators")], np.array([[1.0, 0.0]]))
    embedder = RecordingEmbedder()
    tools = ResearchTools(settings, object(), embedder, store)
    toolset = ResearchToolset(
        tools,
        BriefRequest(
            research_question=(
                "Which neural operators solve PDEs by learning maps between "
                "function spaces?"
            ),
            constraints=["include operator-learning baselines"],
            max_papers=1,
        ),
    )

    content, _meta = toolset._search_papers("neural operators", 1)
    payload = json.loads(content)

    assert payload["query"] == "neural operators"
    assert embedder.texts == [
        "Research question: Which neural operators solve PDEs by learning maps "
        "between function spaces?\nSearch focus: neural operators\n"
        "Constraints: include operator-learning baselines"
    ]


def test_vector_retrieve_applies_configured_relevance_floor():
    settings = Settings(
        vector_store_backend="memory",
        embedding_dimension=2,
        retrieval_min_score=0.8,
    )
    store = InMemoryVectorStore(embedding_dimension=2)
    store.upsert(
        [_paper("strong", "Strong Match"), _paper("weak", "Weak Match")],
        np.array([[1.0, 0.0], [0.6, 0.8]]),
    )
    tools = ResearchTools(settings, object(), RecordingEmbedder(), store)

    result = tools.vector_retrieve("operator learning", 2)

    assert [item.paper.id for item in result.items] == ["strong"]
    assert result.diagnostics.returned == 1
    assert result.diagnostics.min_score == result.diagnostics.max_score == 1.0


def test_backfill_skips_embedding_papers_already_indexed():
    settings = Settings(
        vector_store_backend="memory",
        embedding_dimension=2,
        query_expansion_enabled=False,
        search_auto_backfill=True,
        search_backfill_query_expansion=False,
    )
    store = InMemoryVectorStore(embedding_dimension=2)
    embedder = RecordingEmbedder()
    arxiv_client = RecordingArxivClient()
    tools = ResearchTools(settings, arxiv_client, embedder, store)

    first = tools.semantic_search("neural operators", 1)
    first_document_encodes = len(embedder.document_texts)
    second = tools.semantic_search("neural operators", 1)

    assert arxiv_client.calls == [
        ("all:neural operators", 25),
        ("all:neural operators", 25),
    ]
    assert first.backfilled == 1
    assert second.backfilled == 0
    assert first_document_encodes == 1
    assert len(embedder.document_texts) == first_document_encodes
    assert [item.paper.id for item in second.results] == [
        item.paper.id for item in first.results
    ]


def test_backfill_uses_expanded_arxiv_query_when_provider_available():
    settings = Settings(
        vector_store_backend="memory",
        embedding_dimension=2,
        query_expansion_enabled=False,
        search_auto_backfill=True,
        search_backfill_query_expansion=True,
    )
    store = InMemoryVectorStore(embedding_dimension=2)
    arxiv_client = RecordingArxivClient()
    provider = FakeProvider("neural operators OR DeepONet OR Fourier Neural Operator")
    tools = ResearchTools(
        settings, arxiv_client, RecordingEmbedder(), store, llm=provider
    )

    response = tools.semantic_search("neural operators", 1)

    assert response.backfilled == 1
    assert arxiv_client.calls[0][0] == (
        "all:neural operators OR DeepONet OR Fourier Neural Operator"
    )


def test_backfill_query_expansion_falls_back_without_provider():
    settings = Settings(
        vector_store_backend="memory",
        embedding_dimension=2,
        query_expansion_enabled=False,
        search_auto_backfill=True,
        search_backfill_query_expansion=True,
    )
    store = InMemoryVectorStore(embedding_dimension=2)
    arxiv_client = RecordingArxivClient()
    tools = ResearchTools(settings, arxiv_client, RecordingEmbedder(), store)

    tools.semantic_search("neural operators", 1)

    assert arxiv_client.calls[0][0] == "all:neural operators"


class ReverseReranker:
    def rerank(self, query, items, top_k):
        return list(reversed(items))[:top_k]


def test_vector_retrieve_can_rerank_candidate_pool():
    settings = Settings(
        vector_store_backend="memory",
        embedding_dimension=2,
        max_retrieval_results=3,
        rerank_candidate_k=3,
    )
    store = InMemoryVectorStore(embedding_dimension=2)
    store.upsert(
        [_paper("p1", "First"), _paper("p2", "Second"), _paper("p3", "Third")],
        np.array([[1.0, 0.0], [0.9, 0.1], [0.8, 0.2]]),
    )
    tools = ResearchTools(
        settings, object(), RecordingEmbedder(), store, reranker=ReverseReranker()
    )

    result = tools.vector_retrieve("operator learning", 2)

    assert [item.paper.id for item in result.items] == ["p3", "p2"]


def test_search_papers_returns_fetch_hint_when_floor_filters_all_results():
    settings = Settings(
        vector_store_backend="memory",
        embedding_dimension=2,
        retrieval_min_score=0.8,
    )
    store = InMemoryVectorStore(embedding_dimension=2)
    store.upsert([_paper("weak", "Weak Match")], np.array([[0.0, 1.0]]))
    tools = ResearchTools(settings, object(), RecordingEmbedder(), store)
    toolset = ResearchToolset(
        tools,
        BriefRequest(
            research_question="Which neural operators solve PDEs?",
            max_papers=1,
        ),
    )

    content, meta = toolset._search_papers("neural operators", 1)
    payload = json.loads(content)

    assert meta == {"returned": 0}
    assert payload["returned"] == 0
    assert payload["papers"] == []
    assert "fetch_arxiv" in payload["hint"]
