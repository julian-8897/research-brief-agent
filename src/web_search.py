from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import urlparse

import httpx

from src.settings import Settings

_EXA_SEARCH_URL = "https://api.exa.ai/search"
_MAX_HIGHLIGHT_CHARS = 2000
_MAX_QUERY_CHARS = 1000
_MAX_TIMEOUT_S = 30.0


class WebSearchError(RuntimeError):
    """A recoverable failure from the optional web-search provider."""


@dataclass(frozen=True)
class WebSearchHit:
    title: str
    url: str
    published_date: str | None
    author: str | None
    snippet: str


@dataclass(frozen=True)
class WebSearchResponse:
    results: list[WebSearchHit]
    request_id: str | None = None
    estimated_cost_usd: float = 0.0


class WebSearchProvider(Protocol):
    name: str

    def search(
        self,
        query: str,
        *,
        max_results: int,
        start_published_date: str | None = None,
        end_published_date: str | None = None,
    ) -> WebSearchResponse: ...


class ExaWebSearchProvider:
    """Bounded Exa search returning query-focused page highlights."""

    name = "exa"

    def __init__(
        self,
        settings: Settings,
        *,
        client: httpx.Client | None = None,
    ):
        if not settings.exa_api_key:
            raise ValueError("EXA_API_KEY is required")
        self._api_key = settings.exa_api_key
        self._timeout = max(1.0, min(settings.web_search_timeout_s, _MAX_TIMEOUT_S))
        self._max_results = max(1, min(settings.web_search_max_results, 5))
        self._highlight_chars = max(
            100,
            min(settings.web_search_highlight_chars, _MAX_HIGHLIGHT_CHARS),
        )
        self._allowed_domains = tuple(
            _normalise_domain(domain) for domain in settings.web_search_allowed_domains
        )
        self._client = client

    def search(
        self,
        query: str,
        *,
        max_results: int,
        start_published_date: str | None = None,
        end_published_date: str | None = None,
    ) -> WebSearchResponse:
        query = query.strip()[:_MAX_QUERY_CHARS]
        if not query:
            raise ValueError("web search query cannot be empty")
        result_limit = max(1, min(max_results, self._max_results, 5))
        payload: dict[str, Any] = {
            "query": query,
            "type": "auto",
            "numResults": result_limit,
            "contents": {
                "highlights": {
                    "query": query,
                    "maxCharacters": self._highlight_chars,
                }
            },
        }
        if self._allowed_domains:
            payload["includeDomains"] = list(self._allowed_domains)
        if start_published_date:
            payload["startPublishedDate"] = _normalise_published_date(
                start_published_date
            )
        if end_published_date:
            payload["endPublishedDate"] = _normalise_published_date(end_published_date)

        try:
            response = self._post(payload)
            response.raise_for_status()
            body = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise WebSearchError(f"Exa search failed: {exc}") from exc
        if not isinstance(body, dict) or not isinstance(body.get("results"), list):
            raise WebSearchError("Exa search returned an invalid response")

        results: list[WebSearchHit] = []
        seen_urls: set[str] = set()
        for raw in body["results"]:
            hit = self._parse_hit(raw)
            if hit is None or hit.url in seen_urls:
                continue
            seen_urls.add(hit.url)
            results.append(hit)
            if len(results) >= result_limit:
                break
        return WebSearchResponse(
            results=results,
            request_id=_optional_string(body.get("requestId")),
            estimated_cost_usd=_cost_total(body.get("costDollars")),
        )

    def _post(self, payload: dict[str, Any]) -> httpx.Response:
        kwargs = {
            "headers": {
                "x-api-key": self._api_key,
                "Content-Type": "application/json",
            },
            "json": payload,
            "timeout": self._timeout,
        }
        if self._client is not None:
            return self._client.post(_EXA_SEARCH_URL, **kwargs)
        return httpx.post(_EXA_SEARCH_URL, **kwargs)

    def _parse_hit(self, raw: Any) -> WebSearchHit | None:
        if not isinstance(raw, dict):
            return None
        url = _optional_string(raw.get("url"))
        if not url or not _allowed_https_url(url, self._allowed_domains):
            return None
        title = _optional_string(raw.get("title")) or urlparse(url).netloc
        highlights = raw.get("highlights")
        snippets = (
            [
                item.strip()
                for item in highlights
                if isinstance(item, str) and item.strip()
            ]
            if isinstance(highlights, list)
            else []
        )
        snippet = " ".join(snippets)[: self._highlight_chars].strip()
        if not snippet:
            return None
        return WebSearchHit(
            title=title[:300],
            url=url,
            published_date=_optional_string(raw.get("publishedDate")),
            author=_optional_string(raw.get("author")),
            snippet=snippet,
        )


def build_web_search_provider(settings: Settings) -> WebSearchProvider | None:
    if not settings.web_search_enabled or not settings.exa_api_key:
        return None
    return ExaWebSearchProvider(settings)


def _normalise_domain(domain: str) -> str:
    value = domain.strip().casefold()
    if "://" in value:
        value = urlparse(value).netloc
    return value.strip(".")


def _allowed_https_url(url: str, allowed_domains: tuple[str, ...]) -> bool:
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username:
        return False
    hostname = parsed.hostname.casefold().strip(".")
    return not allowed_domains or any(
        hostname == domain or hostname.endswith(f".{domain}")
        for domain in allowed_domains
    )


def _optional_string(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _normalise_published_date(value: str) -> str:
    compact = value.strip()
    if re.fullmatch(r"\d{8}", compact):
        return f"{compact[:4]}-{compact[4:6]}-{compact[6:]}"
    return compact


def _cost_total(value: Any) -> float:
    if not isinstance(value, dict):
        return 0.0
    total = value.get("total")
    if isinstance(total, bool) or not isinstance(total, int | float):
        return 0.0
    return max(0.0, float(total))
