import arxiv
import numpy as np
import requests
from fastapi.testclient import TestClient

from src.api.main import Services, create_app
from src.models import PaperRecord
from src.observability import Tracer
from src.retrieval import InMemoryVectorStore
from src.settings import Settings


class FakeEmbedder:
    def encode_texts(self, texts, batch_size=32, show_progress_bar=False):
        return np.array([[1.0, 0.0] for _ in texts])


class FakeArxivClient:
    def search_papers(self, query, max_results):
        return [
            {
                "id": "2501.00001",
                "title": "Scientific Retrieval",
                "summary": "Retrieval for scientific briefs.",
                "authors": ["A. Researcher"],
                "categories": ["cs.IR"],
                "primary_category": "cs.IR",
                "arxiv_url": "https://arxiv.org/abs/2501.00001",
            }
        ][:max_results]


class FailingArxivClient:
    def search_papers(self, query, max_results):
        raise arxiv.HTTPError("https://export.arxiv.org/api/query", 0, 500)


class NetworkFailingArxivClient:
    def search_papers(self, query, max_results):
        raise requests.ConnectionError("connection aborted")


def _client():
    settings = Settings(vector_store_backend="memory", embedding_dimension=2)
    store = InMemoryVectorStore(embedding_dimension=2)
    store.upsert(
        [
            PaperRecord(
                id="2501.00001",
                title="Scientific Retrieval",
                summary="Retrieval for scientific briefs.",
                arxiv_url="https://arxiv.org/abs/2501.00001",
            )
        ],
        np.array([[1.0, 0.0]]),
    )
    app = create_app(
        Services(
            settings=settings,
            arxiv_client=FakeArxivClient(),
            embedder=FakeEmbedder(),
            vector_store=store,
            tracer=Tracer(settings),
        )
    )

    return TestClient(app)


def _client_with_arxiv_client(arxiv_client):
    settings = Settings(vector_store_backend="memory", embedding_dimension=2)
    store = InMemoryVectorStore(embedding_dimension=2)
    app = create_app(
        Services(
            settings=settings,
            arxiv_client=arxiv_client,
            embedder=FakeEmbedder(),
            vector_store=store,
            tracer=Tracer(settings),
        )
    )

    return TestClient(app)


def test_health_endpoint():
    with _client() as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["retrieval_backend"] == "memory"


def test_search_endpoint():
    with _client() as client:
        response = client.get("/papers/search", params={"query": "retrieval", "k": 1})

    assert response.status_code == 200
    assert response.json()["results"][0]["paper"]["id"] == "2501.00001"


def test_ingest_endpoint():
    with _client() as client:
        response = client.post("/ingest", json={"query": "cat:cs.IR", "max_papers": 1})

    assert response.status_code == 200
    assert response.json()["ingested"] == 1
    assert response.json()["retrieval_backend"] == "memory"


def test_ingest_endpoint_returns_json_for_arxiv_failure():
    with _client_with_arxiv_client(FailingArxivClient()) as client:
        response = client.post("/ingest", json={"query": "cat:cs.IR", "max_papers": 1})

    assert response.status_code == 502
    assert "arXiv API request failed" in response.json()["detail"]


def test_ingest_endpoint_returns_json_for_network_failure():
    with _client_with_arxiv_client(NetworkFailingArxivClient()) as client:
        response = client.post("/ingest", json={"query": "cat:cs.IR", "max_papers": 1})

    assert response.status_code == 502
    assert "Network request to arXiv failed" in response.json()["detail"]


def test_stream_brief_endpoint_returns_final_event():
    with _client() as client:
        response = client.post(
            "/briefs/stream",
            json={
                "research_question": "How should retrieval support scientific briefs?",
                "max_papers": 1,
            },
        )

    assert response.status_code == 200
    assert "data:" in response.text
    assert '"event": "final"' in response.text
