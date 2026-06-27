import numpy as np

from src.agent import ResearchBriefAgent, ResearchTools
from src.llm import ToolCall, TurnResult
from src.models import BriefRequest, PaperRecord
from src.observability import Tracer
from src.retrieval import InMemoryVectorStore
from src.settings import Settings


class FakeEmbedder:
    def encode_texts(self, texts, batch_size=32, show_progress_bar=False):
        vectors = []
        for text in texts:
            vectors.append([1.0, 0.0] if "retrieval" in text.lower() else [0.0, 1.0])
        return np.array(vectors)


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

    def run_turn(self, system, messages, tools, *, tool_choice="auto"):
        self.turns += 1
        self.offered_tool_names.append([tool.name for tool in tools])
        self.tool_choices.append(tool_choice)
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
    tools = ResearchTools(settings, FakeArxivClient(), FakeEmbedder(), _store_with_paper())
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
        lambda pdf_url, *, timeout, char_budget: ("FULL BODY TEXT", False),
    )
    settings = Settings(
        vector_store_backend="memory",
        embedding_dimension=2,
        anthropic_api_key=None,
        openai_api_key=None,
    )
    tools = ResearchTools(settings, FakeArxivClient(), FakeEmbedder(), _store_with_paper())
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
                "text": "# Decision Memo\n\nAdopt the approach in [2401.00001].",
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
    assert "Decision Memo" in response.final_brief
    assert any(p.id == "2401.00001" for p in response.cited_papers)


def test_agent_stops_at_iteration_budget():
    # A provider that always asks for a tool must be forced to a final answer.
    settings = Settings(
        vector_store_backend="memory",
        embedding_dimension=2,
        anthropic_api_key=None,
        openai_api_key=None,
        agent_max_iterations=2,
    )
    tools = ResearchTools(settings, FakeArxivClient(), FakeEmbedder(), _store_with_paper())
    forever_search = {
        "tool_calls": [
            ToolCall(id="t", name="search_papers", arguments={"query": "retrieval", "k": 1})
        ],
        "in": 10,
        "out": 5,
    }
    final_step = {"text": "# Decision Memo\n\nForced conclusion [2401.00001].", "in": 10, "out": 5}
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
    tools = ResearchTools(settings, FakeArxivClient(), FakeEmbedder(), _store_with_paper())
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
            {"text": '<｜｜DSML｜｜tool_calls><invoke name="search_papers">', "in": 10, "out": 5},
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
        lambda pdf_url, *, timeout, char_budget: ("FULL BODY TEXT", False),
    )
    settings = Settings(
        vector_store_backend="memory",
        embedding_dimension=2,
        anthropic_api_key=None,
        openai_api_key=None,
        agent_max_search_calls=1,
    )
    tools = ResearchTools(settings, FakeArxivClient(), FakeEmbedder(), _store_with_paper())
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
            {"text": "# Decision Memo\n\nPremature conclusion [2401.00001].", "in": 20, "out": 8},
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
            {"text": "# Decision Memo\n\nRead-only conclusion [2401.00001].", "in": 40, "out": 8},
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
        lambda pdf_url, *, timeout, char_budget: ("FULL BODY TEXT", False),
    )
    settings = Settings(
        vector_store_backend="memory",
        embedding_dimension=2,
        anthropic_api_key=None,
        openai_api_key=None,
        agent_max_search_calls=1,
    )
    tools = ResearchTools(settings, FakeArxivClient(), FakeEmbedder(), _store_with_paper())
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
            {"text": "# Decision Memo\n\nGrounded now [2401.00001].", "in": 10, "out": 5},
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
    tools = ResearchTools(settings, FakeArxivClient(), FakeEmbedder(), _store_with_paper())
    agent = ResearchBriefAgent(settings, tools, Tracer(settings), llm=FailingProvider())

    events = list(
        agent._iterate(
            BriefRequest(
                research_question="How should retrieval systems support research briefs?",
                max_papers=1,
            )
        )
    )

    assert any(event["event"] == "error" and event["stage"] == "llm_turn" for event in events)
    assert any(
        event["event"] == "degraded" and event["reason"] == "live_agent_failure"
        for event in events
    )
    final = events[-1]["data"]
    assert "deterministic fallback was used" in final["final_brief"]
    assert any("deterministic fallback" in warning for warning in final["warnings"])
