import json

import numpy as np
import pytest

from src.agent import ResearchTools
from src.agent.toolset import ResearchToolset
from src.ingestion import full_text
from src.ingestion.pdf_extractors import PdfExtractionResult
from src.llm import ToolCall
from src.models import BriefRequest, PaperRecord
from src.settings import Settings


class FakeEmbedder:
    def encode_documents(self, texts, batch_size=32):
        return np.array([[1.0, 0.0] for _ in texts])

    def encode_queries(self, texts, batch_size=32):
        return np.array([[1.0, 0.0] for _ in texts])


def test_fetch_arxiv_fulltext_truncates(monkeypatch):
    monkeypatch.setattr(full_text, "_download", lambda url, timeout: b"pdf-bytes")
    monkeypatch.setattr(
        full_text,
        "_extract_text",
        lambda data, budget, **kwargs: PdfExtractionResult(
            text="A" * 1000,
            truncated=True,
            extractor="test",
        ),
    )

    text, truncated = full_text.fetch_arxiv_fulltext("http://x/pdf", char_budget=1000)

    assert len(text) == 1000
    assert truncated is True


def test_fetch_arxiv_fulltext_classifies_empty_extraction(monkeypatch):
    monkeypatch.setattr(full_text, "_download", lambda url, timeout: b"pdf-bytes")
    monkeypatch.setattr(
        full_text,
        "_extract_text",
        lambda data, budget, **kwargs: PdfExtractionResult(
            text="",
            truncated=False,
            extractor="test",
        ),
    )

    with pytest.raises(full_text.FullTextFetchError) as exc_info:
        full_text.fetch_arxiv_fulltext("https://arxiv.org/pdf/2401.00001")

    assert exc_info.value.code == "empty_text"
    assert "returned no text" in exc_info.value.message


def test_fetch_arxiv_fulltext_classifies_timeout(monkeypatch):
    import httpx

    def timeout(*args, **kwargs):
        raise httpx.TimeoutException("slow")

    monkeypatch.setattr(httpx, "get", timeout)

    with pytest.raises(full_text.FullTextFetchError) as exc_info:
        full_text.fetch_arxiv_fulltext("https://arxiv.org/pdf/2401.00001", timeout=0.1)

    assert exc_info.value.code == "timeout"
    assert "Timed out" in exc_info.value.message


def _toolset_with_paper():
    from src.retrieval import InMemoryVectorStore

    settings = Settings(
        vector_store_backend="memory",
        embedding_dimension=2,
        full_text_max_papers=2,
        full_text_char_budget=500,
        pdf_extractor="pypdf",
    )
    store = InMemoryVectorStore(embedding_dimension=2)
    store.upsert(
        [
            PaperRecord(
                id="2401.00001",
                title="Retrieval Grounding",
                summary="abstract text",
                pdf_url="https://arxiv.org/pdf/2401.00001",
            )
        ],
        np.array([[1.0, 0.0]]),
    )
    tools = ResearchTools(settings, object(), FakeEmbedder(), store)
    toolset = ResearchToolset(
        tools, BriefRequest(research_question="grounding test", max_papers=1)
    )
    # Prime the retrieved set so the toolset knows the paper + its pdf_url.
    toolset.call(
        ToolCall(id="s", name="search_papers", arguments={"query": "grounding", "k": 1})
    )
    return toolset


def _toolset_with_versioned_paper():
    from src.retrieval import InMemoryVectorStore

    settings = Settings(vector_store_backend="memory", embedding_dimension=2)
    store = InMemoryVectorStore(embedding_dimension=2)
    store.upsert(
        [
            PaperRecord(
                id="2401.00001v1",
                title="Versioned Retrieval Grounding",
                summary="abstract text",
                pdf_url="https://arxiv.org/pdf/2401.00001v1",
            )
        ],
        np.array([[1.0, 0.0]]),
    )
    tools = ResearchTools(settings, object(), FakeEmbedder(), store)
    toolset = ResearchToolset(
        tools, BriefRequest(research_question="grounding test", max_papers=1)
    )
    toolset.call(
        ToolCall(
            id="s",
            name="search_papers",
            arguments={"query": "grounding", "k": 1},
        )
    )
    return toolset


def test_get_full_text_tool_returns_body_and_caches(monkeypatch):
    calls = {"n": 0, "extractor": None}

    def fake_fetch(pdf_url, *, timeout, char_budget, **kwargs):
        calls["n"] += 1
        calls["extractor"] = kwargs.get("extractor")
        return "FULL BODY TEXT", False

    monkeypatch.setattr("src.agent.toolset.fetch_arxiv_fulltext", fake_fetch)
    toolset = _toolset_with_paper()

    content, meta = toolset.call(
        ToolCall(id="f1", name="get_full_text", arguments={"paper_ids": ["2401.00001"]})
    )
    payload = json.loads(content)
    assert meta["fetched"] == 1
    assert toolset.fulltext_success_count == 1
    assert toolset.fulltext_attempt_count == 1
    assert toolset.fulltext_error_count == 0
    assert payload["papers"][0]["full_text"] == "FULL BODY TEXT"
    assert payload["papers"][0]["title"] == "Retrieval Grounding"
    assert calls["extractor"] == "pypdf"

    # Second call for the same id must hit the cache, not refetch.
    toolset.call(
        ToolCall(id="f2", name="get_full_text", arguments={"paper_ids": ["2401.00001"]})
    )
    assert calls["n"] == 1


def test_get_full_text_respects_total_paper_budget(monkeypatch):
    monkeypatch.setattr(
        "src.agent.toolset.fetch_arxiv_fulltext",
        lambda pdf_url, *, timeout, char_budget, **kwargs: ("FULL BODY TEXT", False),
    )
    toolset = _toolset_with_paper()
    # Simulate a stricter run-level budget than the default.
    toolset._tools.settings = Settings(
        vector_store_backend="memory",
        embedding_dimension=2,
        full_text_total_paper_budget=1,
    )

    first_content, first_meta = toolset.call(
        ToolCall(id="f1", name="get_full_text", arguments={"paper_ids": ["2401.00001"]})
    )
    second_content, second_meta = toolset.call(
        ToolCall(id="f2", name="get_full_text", arguments={"paper_ids": ["2401.00001"]})
    )

    assert first_meta == {"fetched": 1}
    assert json.loads(first_content)["papers"][0]["full_text"] == "FULL BODY TEXT"
    assert second_meta == {"fetched": 0, "budget_exhausted": True}
    assert "budget is already met" in json.loads(second_content)["hint"]


def test_get_full_text_reports_error_without_failing(monkeypatch):
    def boom(pdf_url, *, timeout, char_budget, **kwargs):
        raise RuntimeError("network down")

    monkeypatch.setattr("src.agent.toolset.fetch_arxiv_fulltext", boom)
    toolset = _toolset_with_paper()

    content, meta = toolset.call(
        ToolCall(id="f", name="get_full_text", arguments={"paper_ids": ["2401.00001"]})
    )
    payload = json.loads(content)
    assert meta["fetched"] == 0
    assert toolset.fulltext_success_count == 0
    assert toolset.fulltext_attempt_count == 1
    assert toolset.fulltext_error_count == 1
    assert payload["papers"][0]["error_code"] == "unknown"
    assert payload["error_counts"] == {"unknown": 1}
    assert meta["error_counts"] == {"unknown": 1}


def test_get_full_text_reports_bad_id_error_category():
    toolset = _toolset_with_paper()

    content, meta = toolset.call(
        ToolCall(id="f", name="get_full_text", arguments={"paper_ids": ["missing-id"]})
    )
    payload = json.loads(content)

    assert meta == {"fetched": 0, "error_counts": {"bad_id": 1}}
    assert payload["missing"] == ["missing-id"]
    assert payload["errors"][0]["error_code"] == "bad_id"
    assert toolset.fulltext_error_count == 1
    assert toolset.fulltext_error_counts == {"bad_id": 1}
    diagnostics = toolset.fulltext_diagnostics()
    assert diagnostics.failed == 1
    assert diagnostics.error_counts == {"bad_id": 1}
    assert diagnostics.missing_ids == ["missing-id"]


def test_toolset_resolves_unversioned_arxiv_ids(monkeypatch):
    monkeypatch.setattr(
        "src.agent.toolset.fetch_arxiv_fulltext",
        lambda pdf_url, *, timeout, char_budget, **kwargs: ("FULL BODY TEXT", False),
    )
    toolset = _toolset_with_versioned_paper()

    details_content, details_meta = toolset.call(
        ToolCall(
            id="d",
            name="get_paper_details",
            arguments={"paper_ids": ["2401.00001"]},
        )
    )
    details = json.loads(details_content)
    assert details_meta == {"count": 1}
    assert details["evidence"][0]["id"] == "2401.00001v1"

    full_text_content, full_text_meta = toolset.call(
        ToolCall(
            id="f",
            name="get_full_text",
            arguments={"paper_ids": ["2401.00001"]},
        )
    )
    full_text = json.loads(full_text_content)
    assert full_text_meta == {"fetched": 1}
    assert full_text["papers"][0]["id"] == "2401.00001v1"
    assert full_text["papers"][0]["full_text"] == "FULL BODY TEXT"


def test_fetch_arxiv_reports_only_new_papers_and_stop_hint(monkeypatch):
    from src.retrieval import InMemoryVectorStore

    settings = Settings(vector_store_backend="memory", embedding_dimension=2)
    tools = ResearchTools(
        settings,
        object(),
        FakeEmbedder(),
        InMemoryVectorStore(embedding_dimension=2),
    )
    paper = PaperRecord(
        id="2401.00001",
        title="Retrieval Grounding",
        summary="abstract text",
    )

    def fake_fetch_and_ingest(query, max_results, date_range=None):
        return 1, [paper]

    monkeypatch.setattr(tools, "fetch_and_ingest", fake_fetch_and_ingest)
    toolset = ResearchToolset(
        tools, BriefRequest(research_question="grounding test", max_papers=1)
    )
    call = ToolCall(
        id="a",
        name="fetch_arxiv",
        arguments={"query": "all:grounding", "max_results": 1},
    )

    first_content, first_meta = toolset.call(call)
    second_content, second_meta = toolset.call(call)

    assert first_meta == {"new": 1}
    assert json.loads(first_content)["new"] == 1
    assert second_meta == {"new": 0}
    second_payload = json.loads(second_content)
    assert second_payload["new"] == 0
    assert second_payload["already_known"] == 1
    assert "Stop fetching" in second_payload["hint"]


def test_toolset_rejects_malformed_tool_arguments():
    toolset = _toolset_with_paper()

    content, meta = toolset.call(
        ToolCall(
            id="bad-search",
            name="search_papers",
            arguments={"query": None, "k": "many"},
        )
    )
    payload = json.loads(content)
    assert meta == {"error": "invalid_tool_arguments"}
    assert payload["error"] == "invalid_tool_arguments"
    assert "query" in payload["message"]

    content, meta = toolset.call(
        ToolCall(
            id="bad-full-text",
            name="get_full_text",
            arguments={"paper_ids": "2401.00001"},
        )
    )
    payload = json.loads(content)
    assert meta == {"error": "invalid_tool_arguments"}
    assert payload["error"] == "invalid_tool_arguments"
    assert "paper_ids" in payload["message"]


def test_toolset_clamps_numeric_tool_arguments(monkeypatch):
    toolset = _toolset_with_paper()

    toolset.call(
        ToolCall(
            id="large-search",
            name="search_papers",
            arguments={"query": "grounding", "k": 10_000},
        )
    )
    assert toolset.diagnostics().requested_k == 20

    captured = {}

    def fake_fetch_and_ingest(query, max_results, date_range=None):
        captured["max_results"] = max_results
        return 0, []

    monkeypatch.setattr(toolset._tools, "fetch_and_ingest", fake_fetch_and_ingest)

    toolset.call(
        ToolCall(
            id="large-fetch",
            name="fetch_arxiv",
            arguments={"query": "all:grounding", "max_results": 10_000},
        )
    )
    assert captured["max_results"] == 50
