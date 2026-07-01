import pytest

from src.llm import AnthropicProvider, OpenAICompatibleProvider, build_llm_provider
from src.settings import Settings


def test_build_provider_returns_none_without_key():
    assert (
        build_llm_provider(Settings(llm_provider="anthropic", anthropic_api_key=None))
        is None
    )
    assert (
        build_llm_provider(Settings(llm_provider="openai", openai_api_key=None)) is None
    )


def test_build_anthropic_provider():
    provider = build_llm_provider(
        Settings(
            llm_provider="anthropic",
            anthropic_api_key="sk-test",
            anthropic_model="claude-sonnet-4-6",
        )
    )
    assert isinstance(provider, AnthropicProvider)
    assert provider.name == "anthropic"
    assert provider.model == "claude-sonnet-4-6"


def test_build_openai_compatible_provider():
    provider = build_llm_provider(
        Settings(
            llm_provider="openai",
            openai_api_key="sk-test",
            openai_model="gpt-4o-mini",
            openai_base_url="http://localhost:11434/v1",
        )
    )
    assert isinstance(provider, OpenAICompatibleProvider)
    assert provider.name == "openai"
    assert provider.model == "gpt-4o-mini"


def test_unknown_provider_raises():
    with pytest.raises(ValueError):
        build_llm_provider(Settings(llm_provider="mystery", anthropic_api_key="x"))
