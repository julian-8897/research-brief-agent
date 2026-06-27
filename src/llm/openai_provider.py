from __future__ import annotations

import json
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


class OpenAICompatibleProvider:
    """Tool-using backend for any OpenAI Chat Completions-compatible endpoint.

    Covers OpenAI plus self-hosted or proxied backends (local models,
    OpenRouter, codex/opencode-style gateways) via ``base_url``. OpenAI
    represents tool use as ``tool_calls`` on the assistant message and replies
    with ``role: "tool"`` messages, which this class translates to and from the
    canonical types.
    """

    name = "openai"

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        max_tokens: int,
        temperature: float,
        base_url: str | None = None,
    ):
        self.model = model
        self._api_key = api_key
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._base_url = base_url

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

        from openai import OpenAI

        client = OpenAI(api_key=self._api_key, base_url=self._base_url)
        payload: list[dict[str, Any]] = [{"role": "system", "content": system}]
        for message in messages:
            payload.extend(self._to_messages(message))

        kwargs: dict[str, Any] = dict(
            model=self.model,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            messages=payload,
        )
        if tools:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.input_schema,
                    },
                }
                for t in tools
            ]
            kwargs["tool_choice"] = tool_choice

        completion = client.chat.completions.create(**kwargs)
        choice = completion.choices[0]
        msg = choice.message

        tool_calls: list[ToolCall] = []
        for call in msg.tool_calls or []:
            try:
                arguments = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {}
            tool_calls.append(
                ToolCall(id=call.id, name=call.function.name, arguments=arguments)
            )

        usage = completion.usage
        return TurnResult(
            text=msg.content or None,
            tool_calls=tool_calls,
            input_tokens=getattr(usage, "prompt_tokens", 0),
            output_tokens=getattr(usage, "completion_tokens", 0),
            model=self.model,
            stop_reason="tool_calls" if tool_calls else "end",
        )

    @staticmethod
    def _to_messages(message: Message) -> list[dict[str, Any]]:
        if isinstance(message, UserMessage):
            return [{"role": "user", "content": message.text}]
        if isinstance(message, AssistantMessage):
            entry: dict[str, Any] = {"role": "assistant", "content": message.text}
            if message.tool_calls:
                entry["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": json.dumps(call.arguments),
                        },
                    }
                    for call in message.tool_calls
                ]
            return [entry]
        if isinstance(message, ToolResultsMessage):
            return [
                {
                    "role": "tool",
                    "tool_call_id": result.tool_call_id,
                    "content": result.content,
                }
                for result in message.results
            ]
        raise TypeError(f"Unsupported message type: {type(message)!r}")
