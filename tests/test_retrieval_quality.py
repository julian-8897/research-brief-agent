import json

import numpy as np

from src.agent import ResearchTools
from src.agent.toolset import ResearchToolset, _build_search_embedding_text
from src.models import BriefRequest, PaperRecord
from src.retrieval import InMemoryVectorStore
from src.settings import Settings


class RecordingEmbedder:
    def __init__(self, vector=None):
        self.texts: list[str] = []
        self._vector = vector or [1.0, 0.0]

    def _encode(self, texts):
        self.texts.extend(texts)
        return np.array([self._vector for _ in texts])

    def encode_documents(self, texts, batch_size=32):
        return self._encode(texts)

    def encode_queries(self, texts, batch_size=32):
        return self._encode(texts)


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
