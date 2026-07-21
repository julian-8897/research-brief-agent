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


def _install_fake_langfuse(monkeypatch):
    module = ModuleType("langfuse")
    module.Langfuse = FakeLangfuse
    monkeypatch.setitem(sys.modules, "langfuse", module)
    FakeLangfuse.instances.clear()
    return FakeLangfuse


def test_tracer_noops_without_keys():
    tracer = Tracer(Settings(langfuse_public_key=None, langfuse_secret_key=None))
    assert tracer._client is None

    context = tracer.start("run", {"q": "x"})
    assert context.trace_url is None

    with tracer.span(context, "turn", turn=1):
        pass
    assert context.spans[-1]["name"] == "turn"
    assert context.spans[-1]["metadata"] == {"turn": 1}
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
    assert context.spans[-1]["metadata"] == {"turn": 2}
