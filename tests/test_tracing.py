import sys
from types import ModuleType

from src.observability.tracing import Tracer
from src.settings import Settings


class FakeSpan:
    def __init__(self, name, metadata):
        self.name = name
        self.metadata = metadata
        self.ended_with: dict | None = None

    def end(self, metadata=None):
        self.ended_with = metadata


class FakeTrace:
    def __init__(self, name, input):
        self.name = name
        self.input = input
        self.spans: list[FakeSpan] = []

    def get_trace_url(self):
        return "https://langfuse.example/trace/abc"

    def span(self, name, metadata=None):
        span = FakeSpan(name, metadata)
        self.spans.append(span)
        return span


class FakeLangfuse:
    instances: list["FakeLangfuse"] = []

    def __init__(self, public_key, secret_key, host=None):
        self.public_key = public_key
        self.secret_key = secret_key
        self.host = host
        self.traces: list[FakeTrace] = []
        FakeLangfuse.instances.append(self)

    def trace(self, name, input=None):
        trace = FakeTrace(name, input)
        self.traces.append(trace)
        return trace


class FakeV4Span:
    def __init__(
        self,
        name,
        input=None,
        metadata=None,
        trace_id="trace-v4",
        as_type="span",
        model=None,
        model_parameters=None,
    ):
        self.name = name
        self.input = input
        self.metadata = metadata
        self.trace_id = trace_id
        self.as_type = as_type
        self.model = model
        self.model_parameters = model_parameters
        self.children: list[FakeV4Span] = []
        self.output = None
        self.usage_details = None
        self.cost_details = None
        self.end_count = 0

    def start_observation(
        self,
        name,
        as_type="span",
        input=None,
        metadata=None,
        model=None,
        model_parameters=None,
    ):
        child = FakeV4Span(
            name,
            input=input,
            metadata=metadata,
            trace_id=self.trace_id,
            as_type=as_type,
            model=model,
            model_parameters=model_parameters,
        )
        self.children.append(child)
        return child

    def update(
        self,
        metadata=None,
        output=None,
        usage_details=None,
        cost_details=None,
        model=None,
    ):
        if metadata is not None:
            self.metadata = metadata
        if output is not None:
            self.output = output
        if usage_details is not None:
            self.usage_details = usage_details
        if cost_details is not None:
            self.cost_details = cost_details
        if model is not None:
            self.model = model

    def end(self):
        self.end_count += 1


class FakeLangfuseV4:
    instances: list["FakeLangfuseV4"] = []

    def __init__(self, public_key, secret_key, base_url=None):
        self.public_key = public_key
        self.secret_key = secret_key
        self.base_url = base_url
        self.roots: list[FakeV4Span] = []
        self.flush_count = 0
        FakeLangfuseV4.instances.append(self)

    def start_observation(self, name, as_type="span", input=None):
        root = FakeV4Span(name, input=input)
        self.roots.append(root)
        return root

    def get_trace_url(self, trace_id=None):
        return f"https://langfuse.example/trace/{trace_id}"

    def flush(self):
        self.flush_count += 1


def _install_fake_langfuse(monkeypatch):
    module = ModuleType("langfuse")
    module.Langfuse = FakeLangfuse
    monkeypatch.setitem(sys.modules, "langfuse", module)
    FakeLangfuse.instances.clear()
    return FakeLangfuse


def _install_fake_langfuse_v4(monkeypatch):
    module = ModuleType("langfuse")
    module.Langfuse = FakeLangfuseV4
    monkeypatch.setitem(sys.modules, "langfuse", module)
    FakeLangfuseV4.instances.clear()
    return FakeLangfuseV4


def test_tracer_noops_without_keys():
    tracer = Tracer(Settings(langfuse_public_key=None, langfuse_secret_key=None))
    assert tracer._client is None

    context = tracer.start("run", {"q": "x"})
    assert context.trace_url is None

    with tracer.span(context, "turn", turn=1):
        pass
    assert context.spans[-1]["name"] == "turn"
    assert context.spans[-1]["metadata"]["turn"] == 1
    assert context.spans[-1]["metadata"]["latency_ms"] >= 0
    assert context.spans[-1]["input"] is None
    assert context.spans[-1]["output"] is None
    assert context.spans[-1]["latency_ms"] >= 0


def test_tracer_creates_trace_and_records_spans(monkeypatch):
    fake = _install_fake_langfuse(monkeypatch)
    tracer = Tracer(
        Settings(
            langfuse_public_key="pk",
            langfuse_secret_key="sk",
            langfuse_host="https://langfuse.example",
        )
    )

    assert tracer._client is fake.instances[0]
    assert tracer._client.public_key == "pk"
    assert tracer._client.host == "https://langfuse.example"

    context = tracer.start("run", {"q": "x"})
    assert context.trace_url == "https://langfuse.example/trace/abc"

    with tracer.span(context, "tool", tool="search_papers"):
        pass

    trace = fake.instances[0].traces[0]
    assert trace.name == "run"
    assert trace.input == {"q": "x"}
    assert len(trace.spans) == 1
    assert trace.spans[0].name == "tool"
    assert trace.spans[0].ended_with is not None
    assert trace.spans[0].ended_with["tool"] == "search_papers"
    assert "latency_ms" in trace.spans[0].ended_with


def test_tracer_uses_v4_observations_finishes_and_flushes(monkeypatch):
    fake = _install_fake_langfuse_v4(monkeypatch)
    tracer = Tracer(
        Settings(
            langfuse_public_key="pk",
            langfuse_secret_key="sk",
            langfuse_host="https://langfuse.example",
        )
    )

    client = fake.instances[0]
    assert client.base_url == "https://langfuse.example"

    context = tracer.start("run", {"q": "x"})
    assert context.trace_url == "https://langfuse.example/trace/trace-v4"

    with tracer.span(
        context,
        "tool:web_search",
        input_payload={"arguments": {"query": "latest coding models"}},
        tool="web_search",
    ) as span:
        span.update(
            output={"sources": [{"id": "web-1"}]},
            returned=1,
            estimated_cost_usd=0.007,
        )

    root = client.roots[0]
    child = root.children[0]
    assert child.name == "tool:web_search"
    assert child.metadata["tool"] == "web_search"
    assert child.metadata["returned"] == 1
    assert child.metadata["estimated_cost_usd"] == 0.007
    assert child.metadata["latency_ms"] >= 0
    assert child.input == {"arguments": {"query": "latest coding models"}}
    assert child.output == {"sources": [{"id": "web-1"}]}
    assert child.end_count == 1

    tracer.finish(context, {"status": "completed"})
    tracer.finish(context, {"status": "duplicate"})
    assert root.output == {"status": "completed"}
    assert root.end_count == 1

    tracer.flush()
    assert client.flush_count == 1


def test_tracer_records_generation_payload_usage_and_cost(monkeypatch):
    fake = _install_fake_langfuse_v4(monkeypatch)
    tracer = Tracer(Settings(langfuse_public_key="pk", langfuse_secret_key="sk"))
    context = tracer.start("run", {"q": "x"})

    with tracer.generation(
        context,
        "llm_turn",
        input_payload={"messages": [{"role": "user", "content": "question"}]},
        model="deepseek-v4-flash",
        model_parameters={"temperature": 0.2},
        turn=1,
    ) as generation:
        generation.update(
            output={"text": "answer", "reasoning": "analysis"},
            usage_details={"input": 100, "output": 20, "total": 120},
            cost_details={"total": 0.0000196},
            pricing_source="deepseek_official_2026-07-26",
        )

    child = fake.instances[0].roots[0].children[0]
    assert child.as_type == "generation"
    assert child.model == "deepseek-v4-flash"
    assert child.input["messages"][0]["content"] == "question"
    assert child.output == {"text": "answer", "reasoning": "analysis"}
    assert child.usage_details == {"input": 100, "output": 20, "total": 120}
    assert child.cost_details == {"total": 0.0000196}
    assert child.metadata["turn"] == 1
    assert child.metadata["pricing_source"] == "deepseek_official_2026-07-26"
    assert child.end_count == 1


def test_tracer_survives_client_construction_failure(monkeypatch):
    module = ModuleType("langfuse")

    def broken(**kwargs):
        raise RuntimeError("boom")

    module.Langfuse = broken
    monkeypatch.setitem(sys.modules, "langfuse", module)

    tracer = Tracer(Settings(langfuse_public_key="pk", langfuse_secret_key="sk"))
    assert tracer._client is None

    context = tracer.start("run", {})
    with tracer.span(context, "turn"):
        pass
    assert context.spans[-1]["name"] == "turn"


def test_tracer_survives_trace_and_span_failures(monkeypatch):
    _install_fake_langfuse(monkeypatch)
    tracer = Tracer(Settings(langfuse_public_key="pk", langfuse_secret_key="sk"))

    def broken_trace(name, input=None):
        raise RuntimeError("trace failed")

    tracer._client.trace = broken_trace
    context = tracer.start("run", {})
    assert context.trace_url is None

    # A trace whose span() raises still yields a timed local span record.
    class BrokenSpanTrace:
        def get_trace_url(self):
            return "url"

        def span(self, name, metadata=None):
            raise RuntimeError("span failed")

    context.spans.append({"_langfuse_trace": BrokenSpanTrace()})
    with tracer.span(context, "turn", turn=2):
        pass
    assert context.spans[-1]["name"] == "turn"
    assert context.spans[-1]["metadata"]["turn"] == 2
    assert context.spans[-1]["metadata"]["latency_ms"] >= 0
