from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse

from src.settings import Settings


@dataclass(frozen=True)
class TokenPricing:
    input_per_1k: float
    cached_input_per_1k: float
    output_per_1k: float
    source: str


_GENERIC_PRICING = TokenPricing(
    input_per_1k=0.003,
    cached_input_per_1k=0.003,
    output_per_1k=0.015,
    source="generic_fallback",
)

_DEEPSEEK_PRICING = {
    "deepseek-v4-flash": TokenPricing(
        input_per_1k=0.00014,
        cached_input_per_1k=0.0000028,
        output_per_1k=0.00028,
        source="deepseek_official_2026-07-26",
    ),
    "deepseek-v4-pro": TokenPricing(
        input_per_1k=0.000435,
        cached_input_per_1k=0.000003625,
        output_per_1k=0.00087,
        source="deepseek_official_2026-07-26",
    ),
}


def resolve_token_pricing(settings: Settings, model: str | None) -> TokenPricing:
    pricing = _provider_pricing(settings, model) or _GENERIC_PRICING
    input_rate = settings.estimated_input_token_cost_per_1k
    cached_rate = settings.estimated_cached_input_token_cost_per_1k
    output_rate = settings.estimated_output_token_cost_per_1k
    if input_rate is None and cached_rate is None and output_rate is None:
        return pricing
    return TokenPricing(
        input_per_1k=pricing.input_per_1k if input_rate is None else input_rate,
        cached_input_per_1k=(
            pricing.cached_input_per_1k
            if cached_rate is None and input_rate is None
            else input_rate
            if cached_rate is None
            else cached_rate
        ),
        output_per_1k=pricing.output_per_1k if output_rate is None else output_rate,
        source="environment_override",
    )


def estimate_token_cost(
    pricing: TokenPricing,
    *,
    input_tokens: int,
    output_tokens: int,
    cache_hit_input_tokens: int = 0,
    cache_miss_input_tokens: int = 0,
) -> float:
    cache_hit = max(0, min(cache_hit_input_tokens, input_tokens))
    cache_miss = max(0, cache_miss_input_tokens)
    accounted = cache_hit + cache_miss
    if accounted < input_tokens:
        cache_miss += input_tokens - accounted
    elif accounted > input_tokens:
        cache_miss = max(0, input_tokens - cache_hit)
    return (
        cache_hit / 1000 * pricing.cached_input_per_1k
        + cache_miss / 1000 * pricing.input_per_1k
        + max(0, output_tokens) / 1000 * pricing.output_per_1k
    )


def _provider_pricing(settings: Settings, model: str | None) -> TokenPricing | None:
    if settings.llm_provider.casefold() != "openai" or not settings.openai_base_url:
        return None
    hostname = urlparse(settings.openai_base_url).hostname or ""
    if not hostname.endswith("deepseek.com"):
        return None
    return _DEEPSEEK_PRICING.get((model or settings.openai_model).casefold())
