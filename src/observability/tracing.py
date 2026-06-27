import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from src.settings import Settings


@dataclass
class TraceContext:
    name: str
    trace_url: str | None = None
    spans: list[dict[str, Any]] = field(default_factory=list)


class Tracer:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._client = None
        if settings.langfuse_public_key and settings.langfuse_secret_key:
            try:
                from langfuse import Langfuse

                self._client = Langfuse(
                    public_key=settings.langfuse_public_key,
                    secret_key=settings.langfuse_secret_key,
                    host=settings.langfuse_host,
                )
            except Exception:
                self._client = None

    def start(self, name: str, input_payload: dict[str, Any]) -> TraceContext:
        context = TraceContext(name=name)
        if self._client is None:
            return context
        try:
            trace = self._client.trace(name=name, input=input_payload)
            context.trace_url = trace.get_trace_url()
            context.spans.append({"_langfuse_trace": trace})
        except Exception:
            pass
        return context

    @contextmanager
    def span(self, context: TraceContext, name: str, **metadata: Any) -> Iterator[None]:
        started = time.perf_counter()
        span = None
        trace = next(
            (
                item.get("_langfuse_trace")
                for item in context.spans
                if "_langfuse_trace" in item
            ),
            None,
        )
        if trace is not None:
            try:
                span = trace.span(name=name, metadata=metadata)
            except Exception:
                span = None
        try:
            yield
        finally:
            latency_ms = (time.perf_counter() - started) * 1000
            context.spans.append(
                {"name": name, "latency_ms": latency_ms, "metadata": metadata}
            )
            if span is not None:
                try:
                    span.end(metadata={**metadata, "latency_ms": latency_ms})
                except Exception:
                    pass
