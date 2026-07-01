"""Verify the canonical tool-use translation for both providers without network.

These tests fake the Anthropic and OpenAI SDK modules so we can assert the exact
wire payloads the providers build and how they parse tool-call responses.
"""

from __future__ import annotations

import sys
import types

from src.llm import (
    AnthropicProvider,
    AssistantMessage,
    OpenAICompatibleProvider,
    ToolCall,
    ToolResult,
    ToolResultsMessage,
    ToolSpec,
    UserMessage,
)

_TOOLS = [
    ToolSpec(
        name="search_papers", description="search", input_schema={"type": "object"}
    )
]
_TRANSCRIPT = [
    UserMessage("research question"),
    AssistantMessage(
        text="let me search",
        tool_calls=[ToolCall(id="c1", name="search_papers", arguments={"query": "x"})],
    ),
    ToolResultsMessage([ToolResult(tool_call_id="c1", content='{"returned": 1}')]),
]


def _obj(**kw):
    o = types.SimpleNamespace()
    o.__dict__.update(kw)
    return o


def test_anthropic_translation_and_parsing(monkeypatch):
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        return _obj(
            content=[
                _obj(type="text", text="thinking"),
                _obj(
                    type="tool_use",
                    id="tu1",
                    name="search_papers",
                    input={"query": "y"},
                ),
            ],
            stop_reason="tool_use",
            usage=_obj(input_tokens=11, output_tokens=7),
        )

    fake = types.ModuleType("anthropic")
    fake.Anthropic = lambda api_key=None: _obj(messages=_obj(create=create))
    monkeypatch.setitem(sys.modules, "anthropic", fake)

    provider = AnthropicProvider(
        api_key="k", model="claude-x", max_tokens=100, temperature=0.1
    )
    result = provider.run_turn("system prompt", _TRANSCRIPT, _TOOLS)

    # Request: system is a top-level param; tool_use/tool_result are content blocks.
    assert captured["system"] == "system prompt"
    assert captured["tools"][0]["name"] == "search_papers"
    msgs = captured["messages"]
    assert msgs[0] == {"role": "user", "content": "research question"}
    assert msgs[1]["role"] == "assistant"
    assert msgs[1]["content"][1]["type"] == "tool_use"
    assert msgs[1]["content"][1]["id"] == "c1"
    assert msgs[2]["content"][0]["type"] == "tool_result"
    assert msgs[2]["content"][0]["tool_use_id"] == "c1"

    # Response: tool_use parsed, stop_reason normalized, usage measured.
    assert result.stop_reason == "tool_calls"
    assert result.tool_calls[0].id == "tu1"
    assert result.tool_calls[0].arguments == {"query": "y"}
    assert (result.input_tokens, result.output_tokens) == (11, 7)


def test_anthropic_tool_choice_none_keeps_tools_visible(monkeypatch):
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        return _obj(
            content=[_obj(type="text", text="final")],
            stop_reason="end_turn",
            usage=_obj(input_tokens=5, output_tokens=2),
        )

    fake = types.ModuleType("anthropic")
    fake.Anthropic = lambda api_key=None: _obj(messages=_obj(create=create))
    monkeypatch.setitem(sys.modules, "anthropic", fake)

    provider = AnthropicProvider(
        api_key="k", model="claude-x", max_tokens=100, temperature=0.1
    )
    result = provider.run_turn("system prompt", _TRANSCRIPT, _TOOLS, tool_choice="none")

    assert captured["tools"][0]["name"] == "search_papers"
    assert captured["tool_choice"] == {"type": "none"}
    assert result.text == "final"


def test_openai_translation_and_parsing(monkeypatch):
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        message = _obj(
            content=None,
            tool_calls=[
                _obj(
                    id="tc1",
                    function=_obj(name="search_papers", arguments='{"query": "z"}'),
                )
            ],
        )
        return _obj(
            choices=[_obj(message=message)],
            usage=_obj(prompt_tokens=13, completion_tokens=4),
        )

    fake = types.ModuleType("openai")
    fake.OpenAI = lambda api_key=None, base_url=None: _obj(
        chat=_obj(completions=_obj(create=create))
    )
    monkeypatch.setitem(sys.modules, "openai", fake)

    provider = OpenAICompatibleProvider(
        api_key="k", model="gpt-x", max_tokens=100, temperature=0.1
    )
    result = provider.run_turn("system prompt", _TRANSCRIPT, _TOOLS)

    # Request: system is the first message; tools wrapped under "function";
    # tool results become role="tool" messages.
    payload = captured["messages"]
    assert payload[0] == {"role": "system", "content": "system prompt"}
    assert payload[2]["role"] == "assistant"
    assert payload[2]["tool_calls"][0]["function"]["name"] == "search_papers"
    assert payload[3]["role"] == "tool"
    assert payload[3]["tool_call_id"] == "c1"
    assert captured["tools"][0]["type"] == "function"
    assert captured["tool_choice"] == "auto"
    assert "extra_body" not in captured

    # Response: JSON arguments parsed to a dict, usage measured.
    assert result.stop_reason == "tool_calls"
    assert result.tool_calls[0].id == "tc1"
    assert result.tool_calls[0].arguments == {"query": "z"}
    assert (result.input_tokens, result.output_tokens) == (13, 4)


def test_openai_tool_choice_none_keeps_tools_visible(monkeypatch):
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        message = _obj(content="final", tool_calls=[])
        return _obj(
            choices=[_obj(message=message)],
            usage=_obj(prompt_tokens=5, completion_tokens=2),
        )

    fake = types.ModuleType("openai")
    fake.OpenAI = lambda api_key=None, base_url=None: _obj(
        chat=_obj(completions=_obj(create=create))
    )
    monkeypatch.setitem(sys.modules, "openai", fake)

    provider = OpenAICompatibleProvider(
        api_key="k", model="gpt-x", max_tokens=100, temperature=0.1
    )
    result = provider.run_turn("system prompt", _TRANSCRIPT, _TOOLS, tool_choice="none")

    assert captured["tools"][0]["type"] == "function"
    assert captured["tool_choice"] == "none"
    assert result.text == "final"


def test_openai_deepseek_v4_disables_thinking(monkeypatch):
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        message = _obj(content="READY - DeepSeek", tool_calls=[])
        return _obj(
            choices=[_obj(message=message)],
            usage=_obj(prompt_tokens=19, completion_tokens=6),
        )

    fake = types.ModuleType("openai")
    fake.OpenAI = lambda api_key=None, base_url=None: _obj(
        chat=_obj(completions=_obj(create=create))
    )
    monkeypatch.setitem(sys.modules, "openai", fake)

    provider = OpenAICompatibleProvider(
        api_key="k",
        model="deepseek-v4-flash",
        max_tokens=100,
        temperature=0.1,
        base_url="https://api.deepseek.com",
    )
    result = provider.run_turn("system prompt", [UserMessage("ping")], [])

    assert captured["extra_body"] == {"thinking": {"type": "disabled"}}
    assert result.text == "READY - DeepSeek"
