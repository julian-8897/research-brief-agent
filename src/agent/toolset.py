from __future__ import annotations

import json
import re
from typing import Any

from src.agent.tools import ResearchTools
from src.ingestion import FullTextFetchError, fetch_arxiv_fulltext
from src.llm import ToolCall, ToolSpec
from src.models import (
    BriefRequest,
    CitedPaper,
    FullTextDiagnostics,
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
_ARXIV_VERSION_SUFFIX = re.compile(r"v\d+$")


def _build_search_embedding_text(
    research_question: str, query: str, constraints: list[str] | None = None
) -> str:
    parts = [
        ("Research question", research_question),
        ("Search focus", query),
    ]
    if constraints:
        clean_constraints = [item.strip() for item in constraints if item.strip()]
        if clean_constraints:
            parts.append(("Constraints", "; ".join(clean_constraints)))

    seen: set[str] = set()
    lines: list[str] = []
    for label, value in parts:
        value = value.strip()
        if not value:
            continue
        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"{label}: {value}")
    return "\n".join(lines)


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
        self._fulltext_missing_ids: set[str] = set()
        self._fulltext_error_count = 0
        self._fulltext_error_counts: dict[str, int] = {}

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

    @property
    def full_text_specs(self) -> list[ToolSpec]:
        return [self._get_full_text_spec()]

    @property
    def details_specs(self) -> list[ToolSpec]:
        return [self._get_paper_details_spec()]

    @staticmethod
    def _search_papers_spec() -> ToolSpec:
        return ToolSpec(
            name="search_papers",
            description=(
                "Semantic search over the indexed arXiv corpus. Returns the "
                "most relevant papers with id, title, similarity score, and an "
                "abstract snippet. Call this first to gather evidence. SPECTER "
                "ranks best when the query is a descriptive, abstract-like "
                "sentence, not bare keywords."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Descriptive natural-language sentence about the "
                            "target methods, domain, and evidence needed; avoid "
                            "keyword-only fragments."
                        ),
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
                maximum=min(
                    _MAX_FETCH_RESULTS, self._tools.settings.max_ingest_results
                ),
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
        embed_text = _build_search_embedding_text(
            self._request.research_question, query, self._request.constraints
        )
        result = self._tools.vector_retrieve(query, k, embed_text=embed_text)
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
        if not papers:
            content = json.dumps(
                {
                    "query": query,
                    "returned": 0,
                    "papers": [],
                    "hint": (
                        "No indexed papers cleared the relevance threshold. "
                        "Call fetch_arxiv with a descriptive arXiv query, then "
                        "run search_papers once again."
                    ),
                }
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
        ids, missing = self._resolve_paper_ids(paper_ids)
        items = [self._retrieved[pid] for pid in ids if pid in self._retrieved]
        evidence = self._tools.extract_evidence(items)
        payload: dict[str, Any] = {"evidence": evidence}
        if missing:
            payload["missing"] = missing
            payload["available_ids"] = self.candidate_ids()
            payload["hint"] = (
                "Use the exact paper ids returned by search_papers. If you need "
                "paper body evidence, call get_full_text with available_ids."
            )
        return json.dumps(payload), {"count": len(evidence)}

    def _get_full_text(self, paper_ids: Any) -> tuple[str, dict[str, Any]]:
        settings = self._tools.settings
        ids, missing = self._resolve_paper_ids(paper_ids)
        for missing_id in missing:
            self._record_fulltext_error("bad_id")
            self._fulltext_missing_ids.add(missing_id)
        remaining_budget = max(
            0, settings.full_text_total_paper_budget - self.fulltext_success_count
        )
        if remaining_budget <= 0:
            return (
                json.dumps(
                    {
                        "papers": [],
                        "hint": (
                            "Full-text paper budget is already met. Stop calling "
                            "get_full_text and write the memo from the evidence read."
                        ),
                    }
                ),
                {"fetched": 0, "budget_exhausted": True},
            )
        ids = ids[: min(settings.full_text_max_papers, remaining_budget)]
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
                        extractor=settings.pdf_extractor,
                    )
                except FullTextFetchError as exc:
                    self._record_fulltext_error(exc.code)
                    papers.append(
                        {
                            "id": pid,
                            "title": title,
                            "error_code": exc.code,
                            "error": exc.message,
                            "status_code": exc.status_code,
                        }
                    )
                    continue
                except ValueError:
                    raise
                except Exception as exc:
                    self._record_fulltext_error("unknown")
                    papers.append(
                        {
                            "id": pid,
                            "title": title,
                            "error_code": "unknown",
                            "error": f"could not fetch full text ({exc})",
                        }
                    )
                    continue
            text, truncated = self._fulltext_cache[pid]
            if not text.strip():
                self._record_fulltext_error("empty_text")
                papers.append(
                    {
                        "id": pid,
                        "title": title,
                        "error_code": "empty_text",
                        "error": "full text extraction returned no text",
                    }
                )
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
        payload: dict[str, Any] = {"papers": papers}
        if missing:
            payload["missing"] = missing
            payload["available_ids"] = self.candidate_ids()
            payload["errors"] = [
                {
                    "id": missing_id,
                    "error_code": "bad_id",
                    "error": "paper id was not found in retrieved candidates",
                }
                for missing_id in missing
            ]
        if self._fulltext_error_counts:
            payload["error_counts"] = dict(sorted(self._fulltext_error_counts.items()))
        meta: dict[str, Any] = {"fetched": fetched}
        if self._fulltext_error_counts:
            meta["error_counts"] = dict(sorted(self._fulltext_error_counts.items()))
        return json.dumps(payload), meta

    def _record_fulltext_error(self, code: str) -> None:
        self._fulltext_error_count += 1
        self._fulltext_error_counts[code] = self._fulltext_error_counts.get(code, 0) + 1

    def _resolve_paper_ids(self, paper_ids: Any) -> tuple[list[str], list[str]]:
        ids = (
            [str(pid).strip() for pid in paper_ids]
            if isinstance(paper_ids, list)
            else []
        )
        resolved: list[str] = []
        missing: list[str] = []
        for pid in ids:
            if not pid:
                continue
            resolved_id = self._resolve_paper_id(pid)
            if resolved_id is None:
                missing.append(pid)
            else:
                resolved.append(resolved_id)
        return resolved, missing

    def _resolve_paper_id(self, paper_id: str) -> str | None:
        if paper_id in self._retrieved:
            return paper_id
        requested_base = _ARXIV_VERSION_SUFFIX.sub("", paper_id)
        candidates = [
            item
            for item in self._retrieved.values()
            if _ARXIV_VERSION_SUFFIX.sub("", item.paper.id) == requested_base
        ]
        if not candidates:
            return None
        candidates.sort(key=lambda item: item.score, reverse=True)
        return candidates[0].paper.id

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
    def fulltext_error_counts(self) -> dict[str, int]:
        return dict(sorted(self._fulltext_error_counts.items()))

    @property
    def fulltext_budget_reached(self) -> bool:
        return (
            self.fulltext_success_count
            >= self._tools.settings.full_text_total_paper_budget
        )

    @property
    def retrieved_items(self) -> list[SearchResponseItem]:
        items = list(self._retrieved.values())
        items.sort(key=lambda item: item.score, reverse=True)
        return items

    def candidate_ids(self, limit: int = 5) -> list[str]:
        return [item.paper.id for item in self.retrieved_items[:limit]]

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

    def fulltext_diagnostics(self) -> FullTextDiagnostics:
        return FullTextDiagnostics(
            attempted=self.fulltext_attempt_count,
            succeeded=self.fulltext_success_count,
            failed=self.fulltext_error_count,
            error_counts=self.fulltext_error_counts,
            missing_ids=sorted(self._fulltext_missing_ids),
            succeeded_ids=sorted(self._fulltext_success_ids),
        )
