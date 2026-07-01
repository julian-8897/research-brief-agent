"""LLM query expansion for semantic search.

SPECTER2's adhoc-query adapter ranks best on descriptive, abstract-like text.
A bare keyword query like ``"neural operators"`` is out of distribution and
tends to match on surface tokens (e.g. any paper with the word "operator")
rather than the concept. This module expands a short query into a hypothetical
abstract (a HyDE-style pseudo-document) so the query vector lands nearer the
relevant papers. It degrades to the raw query when no provider is configured or
the query is already descriptive.
"""

from __future__ import annotations

from src.llm import LLMProvider, UserMessage

_SYSTEM = (
    "You expand short academic search queries into a single dense, abstract-like "
    "sentence describing the research topic, its methods, and its problem "
    "setting. Output only that sentence, no preamble, no quotes."
)

_PROMPT = (
    "Search query: {query}\n\n"
    "Write one concise sentence (max 40 words) describing what an ideal paper "
    "answering this query would be about, as if it were the paper's abstract."
)


def expand_query(
    query: str,
    provider: LLMProvider | None,
    *,
    enabled: bool = True,
    max_words: int = 12,
) -> tuple[str, bool]:
    """Return ``(embed_text, expanded)`` for a raw search query.

    Expansion runs only when enabled, a provider is available, and the query is
    short enough to be keyword-like (``<= max_words`` words); already-descriptive
    queries pass through untouched. The generated sentence is appended to the
    original query so surface terms are preserved alongside the semantic
    expansion. Any provider error falls back to the raw query.
    """
    stripped = query.strip()
    if not enabled or provider is None or not stripped:
        return stripped, False
    if len(stripped.split()) > max_words:
        return stripped, False
    try:
        turn = provider.run_turn(
            _SYSTEM,
            [UserMessage(_PROMPT.format(query=stripped))],
            [],
            tool_choice="none",
        )
    except Exception:
        return stripped, False
    expansion = (turn.text or "").strip()
    if not expansion:
        return stripped, False
    return f"{stripped}. {expansion}", True
