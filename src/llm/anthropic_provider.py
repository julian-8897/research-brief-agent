from __future__ import annotations

from typing import Any

from src.llm.base import (
    AssistantMessage,
    Message,
    ToolCall,
    ToolResultsMessage,
    ToolSpec,
    TurnResult,
    UserMessage,
)


class AnthropicProvider:
    """Tool-using synthesis backend on Anthropic's native Messages API.

    Anthropic represents tool use as ``tool_use``/``tool_result`` content blocks
    and signals ``stop_reason == "tool_use"`` when it wants tools run.
    """

    name = "anthropic"

    def __init__(
        self, *, api_key: str, model: str, max_tokens: int, temperature: float
    ):
        self.model = model
        self._api_key = api_key
        self._max_tokens = max_tokens
        self._temperature = temperature

    def run_turn(
        self,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
        *,
        tool_choice: str = "auto",
    ) -> TurnResult:
        if tool_choice not in {"auto", "none"}:
            raise ValueError("tool_choice must be 'auto' or 'none'")

        from anthropic import Anthropic

        client = Anthropic(api_key=self._api_key)
        kwargs: dict[str, Any] = dict(
            model=self.model,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            system=system,
            messages=[self._to_message(m) for m in messages],
        )
        if tools:
            kwargs["tools"] = [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.input_schema,
                }
                for t in tools
            ]
            if tool_choice == "none":
                kwargs["tool_choice"] = {"type": "none"}

        response = client.messages.create(**kwargs)

        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "thinking":
                reasoning = getattr(block, "thinking", None)
                if reasoning:
                    reasoning_parts.append(reasoning)
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(id=block.id, name=block.name, arguments=dict(block.input))
                )

        return TurnResult(
            text="\n".join(text_parts) or None,
            tool_calls=tool_calls,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            model=self.model,
            stop_reason="tool_calls" if response.stop_reason == "tool_use" else "end",
            reasoning="\n".join(reasoning_parts) or None,
        )

    @staticmethod
    def _to_message(message: Message) -> dict[str, Any]:
        if isinstance(message, UserMessage):
            return {"role": "user", "content": message.text}
        if isinstance(message, AssistantMessage):
            content: list[dict[str, Any]] = []
            if message.text:
                content.append({"type": "text", "text": message.text})
            for call in message.tool_calls:
                content.append(
                    {
                        "type": "tool_use",
                        "id": call.id,
                        "name": call.name,
                        "input": call.arguments,
                    }
                )
            return {"role": "assistant", "content": content}
        if isinstance(message, ToolResultsMessage):
            return {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": result.tool_call_id,
                        "content": result.content,
                    }
                    for result in message.results
                ],
            }
        raise TypeError(f"Unsupported message type: {type(message)!r}")
