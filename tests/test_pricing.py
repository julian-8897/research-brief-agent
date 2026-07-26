import pytest

from src.llm.pricing import estimate_token_cost, resolve_token_pricing
from src.settings import Settings


def _deepseek_settings(**overrides):
    return Settings(
        llm_provider="openai",
        openai_base_url="https://api.deepseek.com",
        openai_model="deepseek-v4-flash",
        **overrides,
    )


def test_deepseek_v4_flash_uses_official_cache_aware_tariff():
    pricing = resolve_token_pricing(_deepseek_settings(), "deepseek-v4-flash")

    assert pricing.input_per_1k == pytest.approx(0.00014)
    assert pricing.cached_input_per_1k == pytest.approx(0.0000028)
    assert pricing.output_per_1k == pytest.approx(0.00028)
    assert pricing.source == "deepseek_official_2026-07-26"
    assert estimate_token_cost(
        pricing,
        input_tokens=1_000_000,
        output_tokens=100_000,
        cache_hit_input_tokens=900_000,
        cache_miss_input_tokens=100_000,
    ) == pytest.approx(0.04452)


def test_manual_input_override_also_applies_to_cached_input_by_default():
    pricing = resolve_token_pricing(
        _deepseek_settings(estimated_input_token_cost_per_1k=0.5),
        "deepseek-v4-flash",
    )

    assert pricing.input_per_1k == 0.5
    assert pricing.cached_input_per_1k == 0.5
    assert pricing.output_per_1k == pytest.approx(0.00028)
    assert pricing.source == "environment_override"


def test_unknown_endpoint_uses_generic_fallback():
    settings = Settings(
        llm_provider="openai",
        openai_base_url="https://gateway.example/v1",
        openai_model="custom",
    )

    pricing = resolve_token_pricing(settings, "custom")

    assert pricing.source == "generic_fallback"
    assert estimate_token_cost(
        pricing,
        input_tokens=1_000,
        output_tokens=1_000,
    ) == pytest.approx(0.018)
