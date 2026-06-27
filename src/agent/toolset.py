from __future__ import annotations

import json
from typing import Any

from src.agent.tools import ResearchTools
from src.ingestion import fetch_arxiv_fulltext
from src.llm import ToolCall, ToolSpec
from src.models import (
    BriefRequest,
    CitedPaper,
    RetrievalDiagnostics,
    SearchResponseItem,
)

# Snippet length for abstracts returned inside search results, to keep tool
# payloads (and therefore input tokens) bounded.
_ABSTRACT_SNIPPET = 600
_MIN_SEARCH_K = 1
_MAX_SEARCH_K = 20
_MIN_FETCH_RESULTS = 1
_MAX_FETCH_RESULTS = 50


class ResearchToolset:
    """Adapts :class:`ResearchTools` into model-callable tools for one brief run.

    It owns per-run state: every paper surfaced by a tool call is remembered so
    the agent can report citations and retrieval diagnostics after the loop,
    regardless of the order in which the model chose to call things.
    """

    def __init__(self, tools: ResearchTools, request: BriefRequest):
        self._tools = tools
        self._request = request
        self._retrieved: dict[str, SearchResponseItem] = {}
        self._ingested_ids: set[str] = set()
        self._search_latency_ms = 0.0
        self._max_requested_k = 0
        # Full-text bodies fetched this run, cached by id so a repeated request
        # never refetches the PDF.
        self._fulltext_cache: dict[str, tuple[str, bool]] = {}
        self._fulltext_success_ids: set[str] = set()
        self._fulltext_attempted_ids: set[str] = set()
        self._fulltext_error_count = 0

    # -- Tool catalogue -------------------------------------------------------

    @property
    def specs(self) -> list[ToolSpec]:
        return [*self.discovery_specs, *self.read_only_specs]

    @property
    def discovery_specs(self) -> list[ToolSpec]:
        return [
            self._search_papers_spec(),
            self._fetch_arxiv_spec(),
        ]

    @property
    def read_only_specs(self) -> list[ToolSpec]:
        return [
            self._get_paper_details_spec(),
            self._get_full_text_spec(),
        ]

    @staticmethod
    def _search_papers_spec() -> ToolSpec:
        return ToolSpec(
            name="search_papers",
            description=(
                "Semantic search over the indexed arXiv corpus. Returns the "
                "most relevant papers with id, title, similarity score, and an "
                "abstract snippet. Call this first to gather evidence."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural-language search query.",
                    },
                    "k": {
                        "type": "integer",
                        "description": "Number of papers to return (1-20).",
                        "default": 8,
                    },
                },
                "required": ["query"],
            },
        )

    @staticmethod
    def _fetch_arxiv_spec() -> ToolSpec:
        return ToolSpec(
            name="fetch_arxiv",
            description=(
                "Fetch fresh paper metadata directly from arXiv for a query "
                "and add it to the corpus. Use this only when search_papers "
                "returns too few results, then search once again."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "arXiv query, e.g. 'all:retrieval augmented generation'.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Max papers to fetch (1-50).",
                        "default": 15,
                    },
                },
                "required": ["query"],
            },
        )

    @staticmethod
    def _get_paper_details_spec() -> ToolSpec:
        return ToolSpec(
            name="get_paper_details",
            description=(
                "Return abstract-level evidence (abstract, authors, category, "
                "date) for specific paper ids previously surfaced by "
                "search_papers. Cheap; use to triage candidates."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "paper_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "arXiv ids to expand.",
                    }
                },
                "required": ["paper_ids"],
            },
        )

    @staticmethod
    def _get_full_text_spec() -> ToolSpec:
        return ToolSpec(
            name="get_full_text",
            description=(
                "Fetch and read the full body text (methods, experiments, "
                "results) of specific papers by arXiv id, not just the "
                "abstract. Use on the few most promising papers before "
                "writing the memo to ground claims in actual evidence."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "paper_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "arXiv ids to read in full (a few at most).",
                    }
                },
                "required": ["paper_ids"],
            },
        )

    # -- Dispatch -------------------------------------------------------------

    def call(self, tool_call: ToolCall) -> tuple[str, dict[str, Any]]:
        """Execute a tool call, returning (json_content, event_metadata)."""
        args = tool_call.arguments or {}
        if not isinstance(args, dict):
            return self._invalid_args("tool arguments must be a JSON object")
        if tool_call.name == "search_papers":
            query = self._string_arg(args, "query")
            if query is None:
                return self._invalid_args("query must be a non-empty string")
            k = self._bounded_int_arg(
                args,
                "k",
                default=8,
                minimum=_MIN_SEARCH_K,
                maximum=min(_MAX_SEARCH_K, self._tools.settings.max_retrieval_results),
            )
            if k is None:
                return self._invalid_args("k must be an integer")
            return self._search_papers(query, k)
        if tool_call.name == "fetch_arxiv":
            query = self._string_arg(args, "query")
            if query is None:
                return self._invalid_args("query must be a non-empty string")
            max_results = self._bounded_int_arg(
                args,
                "max_results",
                default=15,
                minimum=_MIN_FETCH_RESULTS,
                maximum=min(_MAX_FETCH_RESULTS, self._tools.settings.max_ingest_results),
            )
            if max_results is None:
                return self._invalid_args("max_results must be an integer")
            return self._fetch_arxiv(query, max_results)
        if tool_call.name == "get_paper_details":
            paper_ids = self._paper_ids_arg(args)
            if paper_ids is None:
                return self._invalid_args("paper_ids must be an array of strings")
            return self._get_paper_details(paper_ids)
        if tool_call.name == "get_full_text":
            paper_ids = self._paper_ids_arg(args)
            if paper_ids is None:
                return self._invalid_args("paper_ids must be an array of strings")
            return self._get_full_text(paper_ids)
        return json.dumps({"error": f"unknown tool '{tool_call.name}'"}), {}

    @staticmethod
    def _invalid_args(message: str) -> tuple[str, dict[str, Any]]:
        return (
            json.dumps({"error": "invalid_tool_arguments", "message": message}),
            {"error": "invalid_tool_arguments"},
        )

    @staticmethod
    def _string_arg(args: dict[str, Any], name: str) -> str | None:
        value = args.get(name)
        if not isinstance(value, str):
            return None
        value = value.strip()
        return value or None

    @staticmethod
    def _bounded_int_arg(
        args: dict[str, Any],
        name: str,
        *,
        default: int,
        minimum: int,
        maximum: int,
    ) -> int | None:
        value = args.get(name, default)
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            parsed = value
        elif isinstance(value, str):
            try:
                parsed = int(value.strip())
            except ValueError:
                return None
        else:
            return None
        return max(minimum, min(parsed, maximum))

    @staticmethod
    def _paper_ids_arg(args: dict[str, Any]) -> list[str] | None:
        value = args.get("paper_ids")
        if not isinstance(value, list):
            return None
        ids: list[str] = []
        for item in value:
            if not isinstance(item, str):
                return None
            item = item.strip()
            if item:
                ids.append(item)
        return ids

    def _search_papers(self, query: str, k: int) -> tuple[str, dict[str, Any]]:
        if not query:
            return json.dumps({"error": "query is required"}), {"returned": 0}
        result = self._tools.vector_retrieve(query, k)
        self._search_latency_ms += result.diagnostics.retrieval_latency_ms
        self._max_requested_k = max(self._max_requested_k, k)
        for item in result.items:
            self._retrieved[item.paper.id] = item
        papers = [
            {
                "id": item.paper.id,
                "title": item.paper.title,
                "score": round(item.score, 4),
                "abstract": item.paper.summary[:_ABSTRACT_SNIPPET].strip(),
            }
            for item in result.items
        ]
        content = json.dumps(
            {"query": query, "returned": len(papers), "papers": papers}
        )
        return content, {"returned": len(papers)}

    def _fetch_arxiv(self, query: str, max_results: int) -> tuple[str, dict[str, Any]]:
        if not query:
            return json.dumps({"error": "query is required"}), {"new": 0}
        _ingested, papers = self._tools.fetch_and_ingest(
            query, max_results, date_range=self._request.date_range
        )
        new = [paper for paper in papers if paper.id not in self._ingested_ids]
        self._ingested_ids.update(paper.id for paper in papers)
        already_known = len(papers) - len(new)
        payload: dict[str, Any] = {
            "new": len(new),
            "already_known": already_known,
            "titles": [p.title for p in papers[:20]],
        }
        if not new:
            payload["hint"] = (
                "No new papers found. Stop fetching; read full text of the papers "
                "you already have and write the memo."
            )
        return json.dumps(payload), {"new": len(new)}

    def _get_paper_details(self, paper_ids: Any) -> tuple[str, dict[str, Any]]:
        ids = [str(pid) for pid in paper_ids] if isinstance(paper_ids, list) else []
        items = [self._retrieved[pid] for pid in ids if pid in self._retrieved]
        evidence = self._tools.extract_evidence(items)
        return json.dumps({"evidence": evidence}), {"count": len(evidence)}

    def _get_full_text(self, paper_ids: Any) -> tuple[str, dict[str, Any]]:
        settings = self._tools.settings
        ids = [str(pid) for pid in paper_ids] if isinstance(paper_ids, list) else []
        ids = ids[: settings.full_text_max_papers]
        papers: list[dict[str, Any]] = []
        fetched = 0
        for pid in ids:
            self._fulltext_attempted_ids.add(pid)
            item = self._retrieved.get(pid)
            title = item.paper.title if item else None
            if pid not in self._fulltext_cache:
                pdf_url = (
                    item.paper.pdf_url
                    if item and item.paper.pdf_url
                    else f"https://arxiv.org/pdf/{pid}"
                )
                try:
                    self._fulltext_cache[pid] = fetch_arxiv_fulltext(
                        pdf_url,
                        timeout=settings.full_text_timeout_s,
                        char_budget=settings.full_text_char_budget,
                    )
                except Exception as exc:
                    self._fulltext_error_count += 1
                    papers.append({"id": pid, "error": f"could not fetch full text ({exc})"})
                    continue
            text, truncated = self._fulltext_cache[pid]
            if not text.strip():
                self._fulltext_error_count += 1
                papers.append({"id": pid, "error": "full text extraction returned no text"})
                continue
            self._fulltext_success_ids.add(pid)
            fetched += 1
            papers.append(
                {
                    "id": pid,
                    "title": title,
                    "chars": len(text),
                    "truncated": truncated,
                    "full_text": text,
                }
            )
        return json.dumps({"papers": papers}), {"fetched": fetched}

    # -- Post-run reporting ---------------------------------------------------

    @property
    def retrieved_count(self) -> int:
        return len(self._retrieved)

    @property
    def fulltext_success_count(self) -> int:
        return len(self._fulltext_success_ids)

    @property
    def fulltext_attempt_count(self) -> int:
        return len(self._fulltext_attempted_ids)

    @property
    def fulltext_error_count(self) -> int:
        return self._fulltext_error_count

    @property
    def retrieved_items(self) -> list[SearchResponseItem]:
        items = list(self._retrieved.values())
        items.sort(key=lambda item: item.score, reverse=True)
        return items

    def cited_papers(self, brief_text: str) -> list[CitedPaper]:
        """Papers whose id appears in the final brief, falling back to all retrieved."""
        mentioned = [
            item for item in self._retrieved.values() if item.paper.id in brief_text
        ]
        chosen = mentioned or list(self._retrieved.values())
        chosen.sort(key=lambda item: item.score, reverse=True)
        return [
            CitedPaper(
                id=item.paper.id,
                title=item.paper.title,
                arxiv_url=item.paper.arxiv_url,
                score=item.score,
            )
            for item in chosen
        ]

    def diagnostics(self) -> RetrievalDiagnostics:
        scores = [item.score for item in self._retrieved.values()]
        return RetrievalDiagnostics(
            query=self._request.research_question,
            requested_k=self._max_requested_k or self._request.max_papers,
            returned=len(self._retrieved),
            retrieval_latency_ms=self._search_latency_ms,
            min_score=min(scores) if scores else None,
            max_score=max(scores) if scores else None,
        )
