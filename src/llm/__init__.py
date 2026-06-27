from __future__ import annotations

from src.llm.anthropic_provider import AnthropicProvider
from src.llm.base import (
    AssistantMessage,
    LLMProvider,
    Message,
    ToolCall,
    ToolResult,
    ToolResultsMessage,
    ToolSpec,
    TurnResult,
    UserMessage,
)
from src.llm.openai_provider import OpenAICompatibleProvider
from src.settings import Settings


def build_llm_provider(settings: Settings) -> LLMProvider | None:
    """Construct the configured synthesis backend.

    Returns ``None`` when the selected provider has no API key, which lets the
    service fall back to the deterministic offline memo so local runs and CI
    work without credentials.
    """

    provider = settings.llm_provider.lower()
    if provider == "anthropic":
        if not settings.anthropic_api_key:
            return None
        return AnthropicProvider(
            api_key=settings.anthropic_api_key,
            model=settings.anthropic_model,
            max_tokens=settings.llm_max_tokens,
            temperature=settings.llm_temperature,
        )
    if provider == "openai":
        if not settings.openai_api_key:
            return None
        return OpenAICompatibleProvider(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            max_tokens=settings.llm_max_tokens,
            temperature=settings.llm_temperature,
            base_url=settings.openai_base_url,
        )
    raise ValueError(
        f"Unknown LLM_PROVIDER '{settings.llm_provider}'. Use 'anthropic' or 'openai'."
    )


__all__ = [
    "AnthropicProvider",
    "AssistantMessage",
    "LLMProvider",
    "Message",
    "OpenAICompatibleProvider",
    "ToolCall",
    "ToolResult",
    "ToolResultsMessage",
    "ToolSpec",
    "TurnResult",
    "UserMessage",
    "build_llm_provider",
]
