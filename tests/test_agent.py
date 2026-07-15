import numpy as np

from src.agent import ResearchBriefAgent, ResearchTools
from src.agent.toolset import ResearchToolset
from src.llm import ToolCall, ToolResultsMessage, TurnResult
from src.models import BriefRequest, PaperRecord
from src.observability import Tracer
from src.retrieval import InMemoryVectorStore
from src.settings import Settings


class FakeEmbedder:
    def _encode(self, texts):
        vectors = []
        for text in texts:
            vectors.append([1.0, 0.0] if "retrieval" in text.lower() else [0.0, 1.0])
        return np.array(vectors)

    def encode_documents(self, texts, batch_size=32):
        return self._encode(texts)

    def encode_queries(self, texts, batch_size=32):
        return self._encode(texts)


class FakeArxivClient:
    pass


class ScriptedProvider:
    """A provider that replays a fixed sequence of turns for deterministic tests."""

    name = "fake"
    model = "fake-model"

    def __init__(self, steps):
        self._steps = list(steps)
        self.turns = 0
        self.offered_tool_names = []
        self.tool_choices = []
        self.messages_seen = []

    def run_turn(self, system, messages, tools, *, tool_choice="auto"):
        self.turns += 1
        self.offered_tool_names.append([tool.name for tool in tools])
        self.tool_choices.append(tool_choice)
        self.messages_seen.append(messages)
        step = self._steps.pop(0)
        tool_calls = step.get("tool_calls", [])
        return TurnResult(
            text=step.get("text"),
            tool_calls=tool_calls,
            input_tokens=step.get("in", 0),
            output_tokens=step.get("out", 0),
            model=self.model,
            stop_reason="tool_calls" if tool_calls else "end",
        )


class FailingProvider:
    name = "fake"
    model = "fake-model"

    def run_turn(self, system, messages, tools, *, tool_choice="auto"):
        raise RuntimeError("provider unavailable")


def _store_with_paper():
    store = InMemoryVectorStore(embedding_dimension=2)
    store.upsert(
        [
            PaperRecord(
                id="2401.00001",
                title="Retrieval Grounding for Science",
                summary="A retrieval system with citations and latency measurements.",
                arxiv_url="https://arxiv.org/abs/2401.00001",
            )
        ],
        np.array([[1.0, 0.0]]),
    )
    return store


def test_agent_fallback_returns_cited_brief():
    # No provider key -> deterministic offline memo, no LLM calls.
    settings = Settings(
        vector_store_backend="memory",
        embedding_dimension=2,
        anthropic_api_key=None,
        openai_api_key=None,
    )
    tools = ResearchTools(
        settings, FakeArxivClient(), FakeEmbedder(), _store_with_paper()
    )
    agent = ResearchBriefAgent(settings, tools, Tracer(settings))

    response = agent.run(
        BriefRequest(
            research_question="How should retrieval systems support research briefs?",
            max_papers=1,
        )
    )

    assert "Decision Memo" in response.final_brief
    assert response.cited_papers[0].id == "2401.00001"
    assert response.token_cost_estimate.llm_call_count == 0


def test_agent_runs_tool_loop_and_reports_measured_usage(monkeypatch):
    monkeypatch.setattr(
        "src.agent.toolset.fetch_arxiv_fulltext",
        lambda pdf_url, *, timeout, char_budget, **kwargs: ("FULL BODY TEXT", False),
    )
    settings = Settings(
        vector_store_backend="memory",
        embedding_dimension=2,
        anthropic_api_key=None,
        openai_api_key=None,
    )
    tools = ResearchTools(
        settings, FakeArxivClient(), FakeEmbedder(), _store_with_paper()
    )
    provider = ScriptedProvider(
        [
            {
                "tool_calls": [
                    ToolCall(
                        id="t1",
                        name="search_papers",
                        arguments={"query": "retrieval grounding", "k": 1},
                    )
                ],
                "in": 120,
                "out": 30,
            },
            {
                "tool_calls": [
                    ToolCall(
                        id="t2",
                        name="get_full_text",
                        arguments={"paper_ids": ["2401.00001"]},
                    )
                ],
                "in": 150,
                "out": 40,
            },
            {
                "text": "Now I have enough information to write the memo.\n\n# Decision Memo\n\nAdopt the approach in [2401.00001].",
                "in": 180,
                "out": 60,
            },
        ]
    )
    agent = ResearchBriefAgent(settings, tools, Tracer(settings), llm=provider)

    response = agent.run(
        BriefRequest(
            research_question="How should retrieval systems support research briefs?",
            max_papers=1,
        )
    )

    usage = response.token_cost_estimate
    assert provider.turns == 3
    assert usage.llm_call_count == 3
    assert usage.tool_call_count == 2
    # Token counts are summed straight from the provider turns, not estimated.
    assert usage.input_tokens == 450
    assert usage.output_tokens == 130
    assert usage.estimated_cost_usd > 0
    assert response.final_brief.startswith("# Decision Memo")
    assert "Now I have enough information" not in response.final_brief
    assert any(p.id == "2401.00001" for p in response.cited_papers)
    assert response.full_text_diagnostics.attempted == 1
    assert response.full_text_diagnostics.succeeded == 1
    assert response.full_text_diagnostics.failed == 0


def test_agent_stops_at_iteration_budget():
    # A provider that always asks for a tool must be forced to a final answer.
    settings = Settings(
        vector_store_backend="memory",
        embedding_dimension=2,
        anthropic_api_key=None,
        openai_api_key=None,
        agent_max_iterations=2,
    )
    tools = ResearchTools(
        settings, FakeArxivClient(), FakeEmbedder(), _store_with_paper()
    )
    forever_search = {
        "tool_calls": [
            ToolCall(
                id="t", name="search_papers", arguments={"query": "retrieval", "k": 1}
            )
        ],
        "in": 10,
        "out": 5,
    }
    final_step = {
        "text": "# Decision Memo\n\nForced conclusion [2401.00001].",
        "in": 10,
        "out": 5,
    }
    provider = ScriptedProvider([forever_search, forever_search, final_step])
    agent = ResearchBriefAgent(settings, tools, Tracer(settings), llm=provider)

    response = agent.run(
        BriefRequest(research_question="Loop budget enforcement test?", max_papers=1)
    )

    assert any("max agent iterations" in w for w in response.warnings)
    assert "Decision Memo" in response.final_brief
    assert provider.tool_choices[-1] == "none"
    assert any("Full-text evidence requirement" in w for w in response.warnings)


def test_forced_final_discards_tool_markup_and_uses_fallback():
    settings = Settings(
        vector_store_backend="memory",
        embedding_dimension=2,
        anthropic_api_key=None,
        openai_api_key=None,
        agent_max_tool_calls=1,
    )
    tools = ResearchTools(
        settings, FakeArxivClient(), FakeEmbedder(), _store_with_paper()
    )
    provider = ScriptedProvider(
        [
            {
                "tool_calls": [
                    ToolCall(
                        id="t",
                        name="search_papers",
                        arguments={"query": "retrieval", "k": 1},
                    )
                ],
                "in": 10,
                "out": 5,
            },
            {
                "text": '<｜｜DSML｜｜tool_calls><invoke name="search_papers">',
                "in": 10,
                "out": 5,
            },
        ]
    )
    agent = ResearchBriefAgent(settings, tools, Tracer(settings), llm=provider)

    response = agent.run(
        BriefRequest(research_question="Loop budget enforcement test?", max_papers=1)
    )

    assert "Decision Memo" in response.final_brief
    assert "DSML" not in response.final_brief
    assert "tool_calls" not in response.final_brief
    assert "deterministic fallback was used" in response.final_brief
    assert provider.tool_choices[-1] == "none"


def test_discovery_tools_withdrawn_after_search_budget(monkeypatch):
    monkeypatch.setattr(
        "src.agent.toolset.fetch_arxiv_fulltext",
        lambda pdf_url, *, timeout, char_budget, **kwargs: ("FULL BODY TEXT", False),
    )
    settings = Settings(
        vector_store_backend="memory",
        embedding_dimension=2,
        anthropic_api_key=None,
        openai_api_key=None,
        agent_max_search_calls=1,
    )
    tools = ResearchTools(
        settings, FakeArxivClient(), FakeEmbedder(), _store_with_paper()
    )
    provider = ScriptedProvider(
        [
            {
                "tool_calls": [
                    ToolCall(
                        id="t",
                        name="search_papers",
                        arguments={"query": "retrieval", "k": 1},
                    )
                ],
                "in": 10,
                "out": 5,
            },
            {
                "text": "# Decision Memo\n\nPremature conclusion [2401.00001].",
                "in": 20,
                "out": 8,
            },
            {
                "tool_calls": [
                    ToolCall(
                        id="f",
                        name="get_full_text",
                        arguments={"paper_ids": ["2401.00001"]},
                    )
                ],
                "in": 30,
                "out": 9,
            },
            {
                "text": "# Decision Memo\n\nRead-only conclusion [2401.00001].",
                "in": 40,
                "out": 8,
            },
        ]
    )
    agent = ResearchBriefAgent(settings, tools, Tracer(settings), llm=provider)

    response = agent.run(
        BriefRequest(research_question="Budget withdrawal test?", max_papers=1)
    )

    assert "Decision Memo" in response.final_brief
    assert provider.offered_tool_names[0] == [
        "search_papers",
        "fetch_arxiv",
        "get_paper_details",
        "get_full_text",
    ]
    assert provider.offered_tool_names[1] == ["get_full_text"]
    assert provider.offered_tool_names[2] == ["get_full_text"]
    assert provider.offered_tool_names[3] == ["get_paper_details", "get_full_text"]


def test_agent_blocks_final_until_full_text_read(monkeypatch):
    monkeypatch.setattr(
        "src.agent.toolset.fetch_arxiv_fulltext",
        lambda pdf_url, *, timeout, char_budget, **kwargs: ("FULL BODY TEXT", False),
    )
    settings = Settings(
        vector_store_backend="memory",
        embedding_dimension=2,
        anthropic_api_key=None,
        openai_api_key=None,
        agent_max_search_calls=1,
    )
    tools = ResearchTools(
        settings, FakeArxivClient(), FakeEmbedder(), _store_with_paper()
    )
    provider = ScriptedProvider(
        [
            {
                "tool_calls": [
                    ToolCall(
                        id="s",
                        name="search_papers",
                        arguments={"query": "retrieval", "k": 1},
                    )
                ],
                "in": 10,
                "out": 5,
            },
            {"text": "# Decision Memo\n\nToo early [2401.00001].", "in": 10, "out": 5},
            {
                "tool_calls": [
                    ToolCall(
                        id="f",
                        name="get_full_text",
                        arguments={"paper_ids": ["2401.00001"]},
                    )
                ],
                "in": 10,
                "out": 5,
            },
            {
                "text": "# Decision Memo\n\nGrounded now [2401.00001].",
                "in": 10,
                "out": 5,
            },
        ]
    )
    agent = ResearchBriefAgent(settings, tools, Tracer(settings), llm=provider)

    events = list(
        agent._iterate(
            BriefRequest(research_question="Full text gate enforcement?", max_papers=1)
        )
    )
    final = events[-1]["data"]

    assert any(event["event"] == "evidence_required" for event in events)
    assert final["final_brief"].startswith("# Decision Memo")
    assert "Grounded now" in final["final_brief"]
    assert final["token_cost_estimate"]["tool_call_count"] == 2


def test_provider_failure_emits_error_and_degraded_before_fallback():
    settings = Settings(
        vector_store_backend="memory",
        embedding_dimension=2,
        anthropic_api_key=None,
        openai_api_key=None,
    )
    tools = ResearchTools(
        settings, FakeArxivClient(), FakeEmbedder(), _store_with_paper()
    )
    agent = ResearchBriefAgent(settings, tools, Tracer(settings), llm=FailingProvider())

    events = list(
        agent._iterate(
            BriefRequest(
                research_question="How should retrieval systems support research briefs?",
                max_papers=1,
            )
        )
    )

    assert any(
        event["event"] == "error" and event["stage"] == "llm_turn" for event in events
    )
    assert any(
        event["event"] == "degraded" and event["reason"] == "live_agent_failure"
        for event in events
    )
    final = events[-1]["data"]
    assert "deterministic fallback was used" in final["final_brief"]
    assert any("deterministic fallback" in warning for warning in final["warnings"])


def test_agent_compacts_older_full_text_tool_results(monkeypatch):
    body = "FULL BODY TEXT " * 500
    monkeypatch.setattr(
        "src.agent.toolset.fetch_arxiv_fulltext",
        lambda pdf_url, *, timeout, char_budget, **kwargs: (body, False),
    )
    settings = Settings(
        vector_store_backend="memory",
        embedding_dimension=2,
        anthropic_api_key=None,
        openai_api_key=None,
        transcript_keep_recent_tool_results=1,
        transcript_full_text_excerpt_chars=40,
    )
    tools = ResearchTools(
        settings, FakeArxivClient(), FakeEmbedder(), _store_with_paper()
    )
    provider = ScriptedProvider(
        [
            {
                "tool_calls": [
                    ToolCall(
                        id="s",
                        name="search_papers",
                        arguments={"query": "retrieval", "k": 1},
                    )
                ],
            },
            {
                "tool_calls": [
                    ToolCall(
                        id="f",
                        name="get_full_text",
                        arguments={"paper_ids": ["2401.00001"]},
                    )
                ],
            },
            {
                "tool_calls": [
                    ToolCall(
                        id="d",
                        name="get_paper_details",
                        arguments={"paper_ids": ["2401.00001"]},
                    )
                ],
            },
            {"text": "# Decision Memo\n\nGrounded [2401.00001]."},
        ]
    )
    agent = ResearchBriefAgent(settings, tools, Tracer(settings), llm=provider)

    response = agent.run(
        BriefRequest(research_question="Full text compaction test?", max_papers=1)
    )

    assert "Decision Memo" in response.final_brief
    immediate_full_text = _tool_result_content(provider.messages_seen[2], "f")
    later_full_text = _tool_result_content(provider.messages_seen[3], "f")
    assert '"full_text":' in immediate_full_text
    assert body[:80] in immediate_full_text
    assert '"full_text":' not in later_full_text
    assert '"full_text_excerpt":' in later_full_text
    assert len(later_full_text) < len(immediate_full_text)


def test_agent_strips_ungrounded_citations(monkeypatch):
    # A capable model may cite famous papers from memory that were never
    # retrieved. The agent must remove those and keep only grounded citations.
    monkeypatch.setattr(
        "src.agent.toolset.fetch_arxiv_fulltext",
        lambda pdf_url, *, timeout, char_budget, **kwargs: ("FULL BODY TEXT", False),
    )
    settings = Settings(
        vector_store_backend="memory",
        embedding_dimension=2,
        anthropic_api_key=None,
        openai_api_key=None,
    )
    tools = ResearchTools(
        settings, FakeArxivClient(), FakeEmbedder(), _store_with_paper()
    )
    provider = ScriptedProvider(
        [
            {
                "tool_calls": [
                    ToolCall(
                        id="s",
                        name="search_papers",
                        arguments={"query": "retrieval grounding", "k": 1},
                    )
                ],
            },
            {
                "tool_calls": [
                    ToolCall(
                        id="f",
                        name="get_full_text",
                        arguments={"paper_ids": ["2401.00001"]},
                    )
                ],
            },
            {
                "text": (
                    "# Decision Memo\n\n"
                    "Adopt the retrieved approach [2401.00001]. This builds on the "
                    "Transformer [1706.03762] and AdamW [1711.05101] papers."
                ),
            },
        ]
    )
    agent = ResearchBriefAgent(settings, tools, Tracer(settings), llm=provider)

    events = list(
        agent._iterate(
            BriefRequest(
                research_question="How should retrieval systems support briefs?",
                max_papers=1,
            )
        )
    )
    final = events[-1]["data"]

    # Grounded citation survives; the two memory citations are removed.
    assert "[2401.00001]" in final["final_brief"]
    assert "[1706.03762]" not in final["final_brief"]
    assert "[1711.05101]" not in final["final_brief"]
    # cited_papers never contains the ungrounded ids.
    assert {p["id"] for p in final["cited_papers"]} == {"2401.00001"}
    # The removal is surfaced as a visible warning event and in the response.
    warning_event = next(
        event
        for event in events
        if event["event"] == "warning"
        and event["code"] == "ungrounded_citations_removed"
    )
    assert set(warning_event["ungrounded_ids"]) == {"1706.03762", "1711.05101"}
    assert any("not retrieved this run" in warning for warning in final["warnings"])


def test_filter_ungrounded_citations_keeps_grounded_and_version_variants():
    settings = Settings(
        vector_store_backend="memory",
        embedding_dimension=2,
        anthropic_api_key=None,
        openai_api_key=None,
    )
    store = InMemoryVectorStore(embedding_dimension=2)
    store.upsert(
        [
            PaperRecord(
                id="1705.08292v2",
                title="Marginal Value of Adaptive Gradient Methods",
                summary="Retrieval-style analysis of Adam vs SGD generalization.",
                arxiv_url="https://arxiv.org/abs/1705.08292",
            )
        ],
        np.array([[1.0, 0.0]]),
    )
    tools = ResearchTools(settings, FakeArxivClient(), FakeEmbedder(), store)
    toolset = ResearchToolset(
        tools, BriefRequest(research_question="Adam vs SGD?", max_papers=1)
    )
    toolset.call(
        ToolCall(id="s", name="search_papers", arguments={"query": "retrieval", "k": 1})
    )

    brief = (
        "SGD generalizes better [1705.08292] than adaptive methods. "
        "See also LAMB [1904.00962] and Table [2]. Not a cite [foo]."
    )
    cleaned, ungrounded = toolset.filter_ungrounded_citations(brief)

    # Grounded id kept even though it was cited without the version suffix.
    assert "[1705.08292]" in cleaned
    # Unretrieved id removed; non-citation brackets left untouched.
    assert "[1904.00962]" not in cleaned
    assert "[2]" in cleaned
    assert "[foo]" in cleaned
    assert ungrounded == ["1904.00962"]


def _tool_result_content(messages, tool_call_id: str) -> str:
    for message in messages:
        if isinstance(message, ToolResultsMessage):
            for result in message.results:
                if result.tool_call_id == tool_call_id:
                    return result.content
    raise AssertionError(f"missing tool result {tool_call_id}")
