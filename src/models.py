from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, HttpUrl


class DateRange(BaseModel):
    start: str | None = Field(default=None, description="YYYY-MM-DD or YYYYMMDD")
    end: str | None = Field(default=None, description="YYYY-MM-DD or YYYYMMDD")


class PaperRecord(BaseModel):
    id: str
    title: str
    summary: str
    authors: list[str] = Field(default_factory=list)
    published: datetime | None = None
    updated: datetime | None = None
    categories: list[str] = Field(default_factory=list)
    primary_category: str | None = None
    pdf_url: str | None = None
    arxiv_url: str | None = None
    links: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestRequest(BaseModel):
    query: str = Field(..., examples=["cat:cs.LG"])
    max_papers: int = Field(default=50, ge=1, le=500)
    date_range: DateRange | None = None
    refresh_existing: bool = Field(
        default=False,
        description="Re-embed and update papers already present in the corpus.",
    )


class IngestResponse(BaseModel):
    ingested: int
    collection: str
    retrieval_backend: str
    latency_ms: float


class BriefRequest(BaseModel):
    research_question: str = Field(..., min_length=8)
    domain: str | None = Field(default=None, examples=["applied machine learning"])
    constraints: list[str] = Field(default_factory=list)
    max_papers: int = Field(default=12, ge=1, le=50)
    date_range: DateRange | None = None
    brief_type: Literal["decision_memo", "technical_brief", "literature_scan"] = (
        "decision_memo"
    )
    research_depth: Literal["quick", "balanced", "deep"] = "balanced"


class SearchResponseItem(BaseModel):
    paper: PaperRecord
    score: float


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResponseItem]
    retrieval_latency_ms: float
    # Search-time coverage/expansion transparency.
    expanded_query: bool = False
    backfilled: int = 0


class RetrievalDiagnostics(BaseModel):
    query: str
    requested_k: int
    returned: int
    retrieval_latency_ms: float
    min_score: float | None = None
    max_score: float | None = None
    backfilled: int = 0
    corpus_size: int | None = None
    recency_sensitive: bool = False
    recency_backfill_attempted: bool = False
    recent_candidates: int = 0
    freshness_source: str | None = None


class FullTextDiagnostics(BaseModel):
    attempted: int = 0
    succeeded: int = 0
    failed: int = 0
    error_counts: dict[str, int] = Field(default_factory=dict)
    missing_ids: list[str] = Field(default_factory=list)
    succeeded_ids: list[str] = Field(default_factory=list)


class UsageEstimate(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_hit_input_tokens: int = 0
    cache_miss_input_tokens: int = 0
    reasoning_tokens: int = 0
    estimated_cost_usd: float = 0.0
    model: str | None = None
    pricing_source: str | None = None
    llm_call_count: int = 0
    tool_call_count: int = 0


class CitedPaper(BaseModel):
    id: str
    title: str
    arxiv_url: str | None = None
    score: float | None = None


class CitedWebSource(BaseModel):
    id: str
    title: str
    url: HttpUrl
    published_date: str | None = None
    author: str | None = None
    retrieved_at: datetime


class WebSearchDiagnostics(BaseModel):
    available: bool = False
    attempted: bool = False
    calls: int = 0
    returned: int = 0
    failed: int = 0
    provider: str | None = None
    estimated_cost_usd: float = 0.0


class BriefResponse(BaseModel):
    final_brief: str
    cited_papers: list[CitedPaper]
    cited_web_sources: list[CitedWebSource] = Field(default_factory=list)
    retrieval_diagnostics: RetrievalDiagnostics
    full_text_diagnostics: FullTextDiagnostics = Field(
        default_factory=FullTextDiagnostics
    )
    web_search_diagnostics: WebSearchDiagnostics = Field(
        default_factory=WebSearchDiagnostics
    )
    latency_ms: float
    token_cost_estimate: UsageEstimate
    langfuse_trace_url: HttpUrl | None = None
    warnings: list[str] = Field(default_factory=list)
