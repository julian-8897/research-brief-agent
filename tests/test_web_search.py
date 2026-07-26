import json

import httpx
import pytest

from src.settings import Settings
from src.web_search import (
    ExaWebSearchProvider,
    WebSearchError,
    build_web_search_provider,
)


def test_exa_search_is_bounded_filtered_and_returns_highlights():
    seen_request = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen_request["headers"] = request.headers
        seen_request["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "requestId": "exa-request",
                "costDollars": {"total": 0.004},
                "results": [
                    {
                        "title": "Official model release",
                        "url": "https://openai.com/index/model-release",
                        "publishedDate": "2026-07-20",
                        "author": "OpenAI",
                        "highlights": ["The model improves coding performance."],
                    },
                    {
                        "title": "Untrusted mirror",
                        "url": "https://example.com/copied-release",
                        "highlights": ["Copied text."],
                    },
                    {
                        "title": "Unsafe transport",
                        "url": "http://openai.com/plaintext",
                        "highlights": ["Not HTTPS."],
                    },
                    {
                        "title": "No usable evidence",
                        "url": "https://openai.com/index/no-highlight",
                        "highlights": [],
                    },
                ],
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    settings = Settings(
        exa_api_key="exa-secret",
        web_search_allowed_domains=("openai.com",),
        web_search_max_results=5,
        web_search_highlight_chars=600,
    )
    provider = ExaWebSearchProvider(settings, client=client)

    response = provider.search(
        "latest coding model release",
        max_results=99,
        start_published_date="2026-01-01",
        end_published_date="20260725",
    )

    assert seen_request["headers"]["x-api-key"] == "exa-secret"
    assert seen_request["body"] == {
        "query": "latest coding model release",
        "type": "auto",
        "numResults": 5,
        "contents": {
            "highlights": {
                "query": "latest coding model release",
                "maxCharacters": 600,
            }
        },
        "includeDomains": ["openai.com"],
        "startPublishedDate": "2026-01-01",
        "endPublishedDate": "2026-07-25",
    }
    assert [hit.title for hit in response.results] == ["Official model release"]
    assert response.results[0].snippet == "The model improves coding performance."
    assert response.request_id == "exa-request"
    assert response.estimated_cost_usd == 0.004


def test_exa_search_wraps_http_errors_as_recoverable():
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(429, text="rate limited")
        )
    )
    provider = ExaWebSearchProvider(
        Settings(exa_api_key="exa-secret"),
        client=client,
    )

    with pytest.raises(WebSearchError, match="Exa search failed"):
        provider.search("latest model", max_results=5)


def test_web_search_provider_requires_feature_and_key():
    assert (
        build_web_search_provider(
            Settings(web_search_enabled=False, exa_api_key="exa-secret")
        )
        is None
    )
    assert build_web_search_provider(Settings(exa_api_key=None)) is None
    assert isinstance(
        build_web_search_provider(Settings(exa_api_key="exa-secret")),
        ExaWebSearchProvider,
    )
