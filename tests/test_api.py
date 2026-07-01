import json

import arxiv
import numpy as np
import pytest
import requests
from fastapi.testclient import TestClient

from src.api.main import Services, VectorStoreStatus, create_app, create_services
from src.api.sse import SSE_EVENT_TYPES, format_sse_event, validate_sse_event
from src.models import PaperRecord
from src.observability import Tracer
from src.retrieval import InMemoryVectorStore
from src.settings import Settings


class FakeEmbedder:
    def encode_documents(self, texts, batch_size=32):
        return np.array([[1.0, 0.0] for _ in texts])

    def encode_queries(self, texts, batch_size=32):
        return np.array([[1.0, 0.0] for _ in texts])


class FakeArxivClient:
    def search_papers(self, query, max_results, sort_by=None):
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
    def search_papers(self, query, max_results, sort_by=None):
        raise arxiv.HTTPError("https://export.arxiv.org/api/query", 0, 500)


class NetworkFailingArxivClient:
    def search_papers(self, query, max_results, sort_by=None):
        raise requests.ConnectionError("connection aborted")


def _parse_sse_events(body: str):
    events = []
    for chunk in body.split("\n\n"):
        if not chunk.strip():
            continue
        data_lines = [
            line.removeprefix("data: ")
            for line in chunk.splitlines()
            if line.startswith("data: ")
        ]
        assert data_lines, f"SSE chunk without data line: {chunk}"
        events.append(json.loads("\n".join(data_lines)))
    return events


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


def _client_with_settings(settings):
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

    assert response.status_code == 503
    body = response.json()
    assert body["ready"] is False
    assert body["retrieval_backend"] == "memory"
    assert body["vector_store"]["status"] == "ok"
    assert body["vector_store"]["backend"] == "memory"
    assert body["llm_provider"]["status"] == "missing_key"


def test_health_endpoint_reports_ready_when_dependencies_configured():
    settings = Settings(
        vector_store_backend="memory",
        embedding_dimension=2,
        anthropic_api_key="test-key",
    )
    with _client_with_settings(settings) as client:
        response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["ready"] is True
    assert body["vector_store"]["status"] == "ok"
    assert body["llm_provider"]["key_present"] is True


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


def test_sse_event_contract_accepts_stable_event_shapes():
    examples = {
        "started": {"event": "started", "message": "run started"},
        "retrieval_complete": {
            "event": "retrieval_complete",
            "returned": 1,
            "latency_ms": 1.2,
        },
        "llm_turn": {"event": "llm_turn", "turn": 1, "tools_requested": []},
        "tool_call": {"event": "tool_call", "name": "search_papers", "arguments": {}},
        "tool_result": {"event": "tool_result", "name": "search_papers"},
        "discovery_budget_reached": {
            "event": "discovery_budget_reached",
            "reason": "search_budget_reached",
            "message": "read papers",
        },
        "evidence_required": {
            "event": "evidence_required",
            "reason": "full_text_missing",
            "required_full_text_papers": 1,
            "full_text_fetched": 0,
            "candidate_ids": ["2501.00001"],
            "message": "read full text",
        },
        "warning": {
            "event": "warning",
            "code": "thin_evidence",
            "message": "evidence is thin",
        },
        "degraded": {
            "event": "degraded",
            "reason": "live_agent_failure",
            "message": "fallback",
        },
        "error": {
            "event": "error",
            "stage": "llm_turn",
            "message": "provider failed",
            "type": "RuntimeError",
        },
        "synthesis_complete": {"event": "synthesis_complete", "llm_calls": 1},
        "final": {"event": "final", "data": {"final_brief": "# Decision Memo"}},
    }

    assert set(examples) == set(SSE_EVENT_TYPES)
    for event in examples.values():
        assert validate_sse_event(event) == event
        assert format_sse_event(event).startswith("data: ")


def test_sse_event_contract_rejects_unknown_event():
    with pytest.raises(ValueError, match="Unknown SSE event type"):
        validate_sse_event({"event": "surprise"})


def test_sse_event_contract_rejects_missing_required_fields():
    with pytest.raises(ValueError, match="missing required fields: message"):
        validate_sse_event({"event": "started"})


def test_stream_brief_endpoint_events_match_sse_contract():
    with _client() as client:
        response = client.post(
            "/briefs/stream",
            json={
                "research_question": "How should retrieval support scientific briefs?",
                "max_papers": 1,
            },
        )

    assert response.status_code == 200
    events = _parse_sse_events(response.text)
    assert [event["event"] for event in events] == [
        "started",
        "retrieval_complete",
        "synthesis_complete",
        "warning",
        "final",
    ]
    for event in events:
        validate_sse_event(event)
    assert events[-1]["data"]["final_brief"].startswith("# Decision Memo")


def test_api_key_auth_protects_non_health_endpoints():
    settings = Settings(
        vector_store_backend="memory",
        embedding_dimension=2,
        api_keys=("secret",),
        rate_limit_requests=10,
    )
    with _client_with_settings(settings) as client:
        health_response = client.get("/health")
        missing_key_response = client.get(
            "/papers/search", params={"query": "retrieval", "k": 1}
        )
        wrong_key_response = client.get(
            "/papers/search",
            params={"query": "retrieval", "k": 1},
            headers={"X-API-Key": "wrong"},
        )
        ok_response = client.get(
            "/papers/search",
            params={"query": "retrieval", "k": 1},
            headers={"X-API-Key": "secret"},
        )

    assert health_response.status_code == 503
    assert missing_key_response.status_code == 401
    assert wrong_key_response.status_code == 401
    assert ok_response.status_code == 200


def test_per_ip_rate_limit_blocks_excess_authenticated_requests():
    settings = Settings(
        vector_store_backend="memory",
        embedding_dimension=2,
        api_keys=("secret",),
        rate_limit_requests=2,
        rate_limit_window_seconds=60,
    )
    headers = {"X-API-Key": "secret", "fly-client-ip": "203.0.113.9"}
    with _client_with_settings(settings) as client:
        first = client.get(
            "/papers/search",
            params={"query": "retrieval", "k": 1},
            headers=headers,
        )
        second = client.get(
            "/papers/search",
            params={"query": "retrieval", "k": 1},
            headers=headers,
        )
        third = client.get(
            "/papers/search",
            params={"query": "retrieval", "k": 1},
            headers=headers,
        )

    assert first.status_code == 200
    assert second.status_code == 200
    assert third.status_code == 429


def test_create_services_reports_local_vector_store_fallback(monkeypatch):
    def fail_build(_settings):
        raise RuntimeError("qdrant unavailable")

    monkeypatch.setattr("src.api.main.build_vector_store", fail_build)

    services = create_services(
        Settings(
            environment="local",
            vector_store_backend="qdrant",
            embedding_dimension=2,
        )
    )

    assert services.vector_store.backend_name == "memory"
    assert services.vector_store_status == VectorStoreStatus(
        backend="memory",
        fallback=True,
        error="qdrant unavailable",
    )


def test_create_services_raises_vector_store_errors_outside_local(monkeypatch):
    def fail_build(_settings):
        raise RuntimeError("qdrant unavailable")

    monkeypatch.setattr("src.api.main.build_vector_store", fail_build)

    with pytest.raises(RuntimeError, match="qdrant unavailable"):
        create_services(
            Settings(
                environment="production",
                api_keys=("secret",),
                vector_store_backend="qdrant",
                embedding_dimension=2,
            )
        )


def test_create_services_requires_api_keys_outside_local():
    with pytest.raises(RuntimeError, match="API_KEYS"):
        create_services(
            Settings(
                environment="production",
                vector_store_backend="memory",
                embedding_dimension=2,
            )
        )
