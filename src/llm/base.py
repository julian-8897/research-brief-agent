from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

# --- Canonical tool-use types -------------------------------------------------
# These are provider-neutral. Each provider translates them to and from its own
# wire format (Anthropic tool_use blocks, OpenAI tool_calls) so the agent loop
# never sees provider-specific shapes.


@dataclass(frozen=True)
class ToolSpec:
    """A tool the model may call, described with a JSON Schema for its input."""

    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(frozen=True)
class ToolCall:
    """A model's request to invoke a tool. ``id`` correlates the later result."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ToolResult:
    """The output of a tool, fed back to the model on the next turn."""

    tool_call_id: str
    content: str


@dataclass(frozen=True)
class UserMessage:
    text: str


@dataclass(frozen=True)
class AssistantMessage:
    text: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)


@dataclass(frozen=True)
class ToolResultsMessage:
    results: list[ToolResult]


# A transcript entry the providers know how to serialize.
Message = UserMessage | AssistantMessage | ToolResultsMessage


@dataclass(frozen=True)
class TurnResult:
    """One model turn: free text and/or tool-call requests, plus measured usage.

    ``stop_reason`` is normalized to ``"tool_calls"`` when the model wants tools
    run, or ``"end"`` when it produced a final answer. Token counts come from
    the provider response, not an estimate.
    """

    text: str | None
    tool_calls: list[ToolCall]
    input_tokens: int
    output_tokens: int
    model: str
    stop_reason: str


@runtime_checkable
class LLMProvider(Protocol):
    """Tool-using synthesis backend.

    ``run_turn`` sends the system prompt, the running transcript, and the tool
    catalogue, and returns a single normalized :class:`TurnResult`. The agent
    loop drives multiple turns until the model stops requesting tools.
    """

    name: str
    model: str

    def run_turn(
        self,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
        *,
        tool_choice: str = "auto",
    ) -> TurnResult: ...
