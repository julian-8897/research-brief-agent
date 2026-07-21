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


class AsymmetricEmbedder:
    """Embeds documents and queries into orthogonal vectors.

    A backfilled document therefore never matches the query, so a second search
    for the same topic would re-trigger the score-based backfill unless the
    per-query dedup guard suppresses it.
    """

    def __init__(self):
        self.document_texts: list[str] = []
        self.query_texts: list[str] = []

    def encode_documents(self, texts, batch_size=32):
        self.document_texts.extend(texts)
        return np.array([[0.0, 1.0] for _ in texts])

    def encode_queries(self, texts, batch_size=32):
        self.query_texts.extend(texts)
        return np.array([[1.0, 0.0] for _ in texts])


def _backfill_toolset(tools: ResearchTools, question: str) -> ResearchToolset:
    return ResearchToolset(
        tools, BriefRequest(research_question=question, max_papers=5)
    )


def test_search_papers_auto_backfills_when_corpus_is_empty():
    settings = Settings(
        vector_store_backend="memory",
        embedding_dimension=2,
        query_expansion_enabled=False,
        search_backfill_query_expansion=False,
        agent_search_auto_backfill=True,
    )
    store = InMemoryVectorStore(embedding_dimension=2)
    arxiv_client = RecordingArxivClient(paper_id="2501.09999")
    tools = ResearchTools(settings, arxiv_client, RecordingEmbedder(), store)
    toolset = _backfill_toolset(tools, "Which quantization method preserves accuracy?")

    content, meta = toolset._search_papers("quantization methods", 5)
    payload = json.loads(content)

    assert arxiv_client.calls == [("all:quantization methods", 25)]
    assert payload["backfilled"] == 1
    assert meta["backfilled"] == 1
    assert [p["id"] for p in payload["papers"]] == ["2501.09999"]


def test_search_papers_skips_backfill_when_local_hit_is_strong():
    settings = Settings(
        vector_store_backend="memory",
        embedding_dimension=2,
        agent_search_auto_backfill=True,
        agent_search_backfill_min_score=0.75,
    )
    store = InMemoryVectorStore(embedding_dimension=2)
    store.upsert([_paper("local", "Local Strong")], np.array([[1.0, 0.0]]))
    arxiv_client = RecordingArxivClient()
    tools = ResearchTools(settings, arxiv_client, RecordingEmbedder(), store)
    toolset = _backfill_toolset(tools, "A question about operator learning methods")

    content, _meta = toolset._search_papers("operators", 1)
    payload = json.loads(content)

    assert arxiv_client.calls == []
    assert "backfilled" not in payload
    assert [p["id"] for p in payload["papers"]] == ["local"]


def test_search_papers_backfills_on_off_topic_local_hit():
    settings = Settings(
        vector_store_backend="memory",
        embedding_dimension=2,
        query_expansion_enabled=False,
        search_backfill_query_expansion=False,
        agent_search_auto_backfill=True,
        agent_search_backfill_min_score=0.75,
    )
    store = InMemoryVectorStore(embedding_dimension=2)
    # Off-topic paper: cosine with the [1,0] query is 0.5, below the 0.75 floor.
    store.upsert([_paper("offtopic", "Off Topic")], np.array([[0.5, 0.8660254]]))
    arxiv_client = RecordingArxivClient(paper_id="2501.07777")
    tools = ResearchTools(settings, arxiv_client, RecordingEmbedder(), store)
    toolset = _backfill_toolset(tools, "Quantization accuracy for on-device serving")

    content, _meta = toolset._search_papers("quantization", 5)
    payload = json.loads(content)

    assert arxiv_client.calls == [("all:quantization", 25)]
    assert payload["backfilled"] == 1
    assert "2501.07777" in [p["id"] for p in payload["papers"]]


def test_search_papers_respects_auto_backfill_toggle():
    settings = Settings(
        vector_store_backend="memory",
        embedding_dimension=2,
        agent_search_auto_backfill=False,
    )
    store = InMemoryVectorStore(embedding_dimension=2)
    arxiv_client = RecordingArxivClient()
    tools = ResearchTools(settings, arxiv_client, RecordingEmbedder(), store)
    toolset = _backfill_toolset(tools, "A question about anything at all here")

    content, meta = toolset._search_papers("anything", 1)
    payload = json.loads(content)

    assert arxiv_client.calls == []
    assert meta == {"returned": 0}
    assert payload["returned"] == 0
    assert "backfilled" not in payload
    assert "fetch_arxiv" in payload["hint"]


def test_search_papers_backfills_each_query_once_per_run():
    settings = Settings(
        vector_store_backend="memory",
        embedding_dimension=2,
        query_expansion_enabled=False,
        search_backfill_query_expansion=False,
        agent_search_auto_backfill=True,
    )
    store = InMemoryVectorStore(embedding_dimension=2)
    arxiv_client = RecordingArxivClient(paper_id="2501.05555")
    tools = ResearchTools(settings, arxiv_client, AsymmetricEmbedder(), store)
    toolset = _backfill_toolset(tools, "Quantization tradeoffs for edge inference")

    toolset._search_papers("quantization", 5)
    # The backfilled paper is orthogonal to the query, so the corpus still lacks
    # a strong hit; only the per-query dedup guard prevents a second fetch.
    toolset._search_papers("quantization", 5)

    assert arxiv_client.calls == [("all:quantization", 25)]


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
