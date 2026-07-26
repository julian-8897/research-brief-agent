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


@dataclass
class GenerationRecord:
    output: dict[str, Any] | str | None = None
    usage_details: dict[str, int] | None = None
    cost_details: dict[str, float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def update(
        self,
        *,
        output: dict[str, Any] | str | None = None,
        usage_details: dict[str, int] | None = None,
        cost_details: dict[str, float] | None = None,
        **metadata: Any,
    ) -> None:
        self.output = output
        self.usage_details = usage_details
        self.cost_details = cost_details
        self.metadata.update(metadata)


@dataclass
class SpanRecord:
    output: dict[str, Any] | list[Any] | str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def update(
        self,
        *,
        output: dict[str, Any] | list[Any] | str | None = None,
        **metadata: Any,
    ) -> None:
        self.output = output
        self.metadata.update(metadata)


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
    def span(
        self,
        context: TraceContext,
        name: str,
        *,
        input_payload: dict[str, Any] | None = None,
        **metadata: Any,
    ) -> Iterator[SpanRecord]:
        started = time.perf_counter()
        span = None
        record = SpanRecord()
        if context.root is not None:
            try:
                if hasattr(context.root, "start_observation"):
                    span = context.root.start_observation(
                        name=name,
                        as_type="span",
                        input=input_payload,
                        metadata=metadata,
                    )
                else:
                    try:
                        span = context.root.span(
                            name=name,
                            input=input_payload,
                            metadata=metadata,
                        )
                    except TypeError:
                        span = context.root.span(
                            name=name,
                            metadata={**metadata, "input": input_payload},
                        )
            except Exception:
                span = None
        try:
            yield record
        except Exception as exc:
            record.update(
                error_type=type(exc).__name__,
                error=str(exc),
            )
            raise
        finally:
            latency_ms = (time.perf_counter() - started) * 1000
            final_metadata = {
                **metadata,
                **record.metadata,
                "latency_ms": latency_ms,
            }
            context.spans.append(
                {
                    "name": name,
                    "latency_ms": latency_ms,
                    "metadata": final_metadata,
                    "input": input_payload,
                    "output": record.output,
                }
            )
            if span is not None:
                try:
                    if hasattr(span, "update"):
                        span.update(
                            output=record.output,
                            metadata=final_metadata,
                        )
                        span.end()
                    else:
                        try:
                            span.end(
                                output=record.output,
                                metadata=final_metadata,
                            )
                        except TypeError:
                            span.end(metadata=final_metadata)
                except Exception:
                    pass

    @contextmanager
    def generation(
        self,
        context: TraceContext,
        name: str,
        *,
        input_payload: dict[str, Any],
        model: str,
        model_parameters: dict[str, Any] | None = None,
        **metadata: Any,
    ) -> Iterator[GenerationRecord]:
        """Record an LLM call as a Langfuse generation, including its payload."""
        started = time.perf_counter()
        observation = None
        record = GenerationRecord()
        if context.root is not None:
            try:
                if hasattr(context.root, "start_observation"):
                    observation = context.root.start_observation(
                        name=name,
                        as_type="generation",
                        input=input_payload,
                        model=model,
                        model_parameters=model_parameters,
                        metadata=metadata,
                    )
                elif hasattr(context.root, "generation"):
                    observation = context.root.generation(
                        name=name,
                        input=input_payload,
                        model=model,
                        model_parameters=model_parameters,
                        metadata=metadata,
                    )
                else:
                    observation = context.root.span(name=name, metadata=metadata)
            except Exception:
                observation = None
        try:
            yield record
        finally:
            latency_ms = (time.perf_counter() - started) * 1000
            final_metadata = {
                **metadata,
                **record.metadata,
                "latency_ms": latency_ms,
            }
            context.spans.append(
                {
                    "name": name,
                    "type": "generation",
                    "latency_ms": latency_ms,
                    "metadata": final_metadata,
                    "input": input_payload,
                    "output": record.output,
                    "usage_details": record.usage_details,
                    "cost_details": record.cost_details,
                    "model": model,
                }
            )
            if observation is not None:
                try:
                    if hasattr(observation, "update"):
                        observation.update(
                            output=record.output,
                            metadata=final_metadata,
                            usage_details=record.usage_details,
                            cost_details=record.cost_details,
                            model=model,
                        )
                        observation.end()
                    else:
                        observation.end(metadata=final_metadata)
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
