from src.agent.query_expansion import expand_arxiv_query, expand_query
from src.llm import TurnResult


class FakeProvider:
    name = "fake"
    model = "fake"

    def __init__(self, text="Neural operators learn mappings between function spaces."):
        self._text = text
        self.calls = 0

    def run_turn(self, system, messages, tools, *, tool_choice="auto"):
        self.calls += 1
        return TurnResult(
            text=self._text,
            tool_calls=[],
            input_tokens=5,
            output_tokens=9,
            model=self.model,
            stop_reason="end",
        )


class ExplodingProvider:
    name = "boom"
    model = "boom"

    def run_turn(self, *args, **kwargs):
        raise RuntimeError("provider down")


def test_expand_query_appends_hypothetical_abstract_for_short_query():
    provider = FakeProvider()
    text, expanded = expand_query("neural operators", provider)
    assert expanded is True
    assert text.startswith("neural operators.")
    assert "function spaces" in text
    assert provider.calls == 1


def test_expand_query_skips_when_disabled():
    provider = FakeProvider()
    text, expanded = expand_query("neural operators", provider, enabled=False)
    assert (text, expanded) == ("neural operators", False)
    assert provider.calls == 0


def test_expand_query_skips_when_no_provider():
    assert expand_query("neural operators", None) == ("neural operators", False)


def test_expand_query_passes_through_descriptive_query():
    provider = FakeProvider()
    long_query = (
        "which neural operators learn mappings between function spaces to solve "
        "partial differential equations efficiently"
    )
    text, expanded = expand_query(long_query, provider, max_words=12)
    assert (text, expanded) == (long_query, False)
    assert provider.calls == 0


def test_expand_query_falls_back_on_provider_error():
    text, expanded = expand_query("neural operators", ExplodingProvider())
    assert (text, expanded) == ("neural operators", False)


def test_expand_query_ignores_empty_expansion():
    provider = FakeProvider(text="   ")
    assert expand_query("neural operators", provider) == ("neural operators", False)


def test_expand_arxiv_query_returns_keyword_boolean_query():
    provider = FakeProvider(
        text="Query: neural operators OR DeepONet OR Fourier Neural Operator"
    )

    text, expanded = expand_arxiv_query("neural operators", provider)

    assert expanded is True
    assert text == "neural operators OR DeepONet OR Fourier Neural Operator"
    assert provider.calls == 1


def test_expand_arxiv_query_reports_usage_to_observer():
    provider = FakeProvider(text="neural operators OR DeepONet")
    observed = []

    expand_arxiv_query(
        "neural operators",
        provider,
        on_turn=lambda name, system, messages, turn, latency_ms: observed.append(
            (name, system, messages, turn, latency_ms)
        ),
    )

    assert len(observed) == 1
    name, _system, _messages, turn, latency_ms = observed[0]
    assert name == "arxiv_query_expansion"
    assert turn.input_tokens == 5
    assert turn.output_tokens == 9
    assert latency_ms >= 0


def test_expand_arxiv_query_falls_back_without_provider():
    assert expand_arxiv_query("neural operators", None) == (
        "neural operators",
        False,
    )
