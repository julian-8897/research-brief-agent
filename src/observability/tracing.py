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
    root: Any | None = field(default=None, repr=False)
    finished: bool = False


class Tracer:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._client = None
        if settings.langfuse_public_key and settings.langfuse_secret_key:
            try:
                from langfuse import Langfuse

                client_kwargs = {
                    "public_key": settings.langfuse_public_key,
                    "secret_key": settings.langfuse_secret_key,
                }
                if settings.langfuse_host:
                    endpoint_argument = (
                        "base_url" if hasattr(Langfuse, "start_observation") else "host"
                    )
                    client_kwargs[endpoint_argument] = settings.langfuse_host
                self._client = Langfuse(**client_kwargs)
            except Exception:
                self._client = None

    def start(self, name: str, input_payload: dict[str, Any]) -> TraceContext:
        context = TraceContext(name=name)
        if self._client is None:
            return context
        try:
            if hasattr(self._client, "start_observation"):
                root = self._client.start_observation(
                    name=name,
                    as_type="agent",
                    input=input_payload,
                )
                context.trace_url = self._client.get_trace_url(trace_id=root.trace_id)
            else:
                root = self._client.trace(name=name, input=input_payload)
                context.trace_url = root.get_trace_url()
            context.root = root
        except Exception:
            pass
        return context

    @contextmanager
    def span(self, context: TraceContext, name: str, **metadata: Any) -> Iterator[None]:
        started = time.perf_counter()
        span = None
        if context.root is not None:
            try:
                if hasattr(context.root, "start_observation"):
                    span = context.root.start_observation(
                        name=name,
                        as_type="span",
                        metadata=metadata,
                    )
                else:
                    span = context.root.span(name=name, metadata=metadata)
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
                    if hasattr(span, "update"):
                        span.update(metadata={**metadata, "latency_ms": latency_ms})
                        span.end()
                    else:
                        span.end(metadata={**metadata, "latency_ms": latency_ms})
                except Exception:
                    pass

    def finish(
        self, context: TraceContext, output_payload: dict[str, Any] | None = None
    ) -> None:
        if context.finished:
            return
        context.finished = True
        if context.root is None:
            return
        try:
            if output_payload is not None and hasattr(context.root, "update"):
                context.root.update(output=output_payload)
            if hasattr(context.root, "end"):
                context.root.end()
        except Exception:
            pass

    def flush(self) -> None:
        if self._client is None or not hasattr(self._client, "flush"):
            return
        try:
            self._client.flush()
        except Exception:
            pass
