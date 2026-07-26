import os
from dataclasses import dataclass
from functools import lru_cache

from src.embeddings import (
    DEFAULT_DOCUMENT_ADAPTER,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_QUERY_ADAPTER,
)

# Load a local .env so secrets/config live in a gitignored file rather than the
# shell. Tests set DISABLE_DOTENV=1 (in tests/conftest.py) to stay hermetic and
# never pick up real provider credentials. Existing env vars always win.
if os.getenv("DISABLE_DOTENV") != "1":
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a float") from exc


def _csv_env(name: str) -> tuple[str, ...]:
    value = os.getenv(name)
    if not value:
        return ()
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def _optional_env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip()
    return value or None


@dataclass(frozen=True)
class Settings:
    app_name: str = os.getenv("APP_NAME", "research-brief-agent")
    environment: str = os.getenv("ENVIRONMENT", "local")

    api_keys: tuple[str, ...] = _csv_env("API_KEYS")
    api_key_header_name: str = os.getenv("API_KEY_HEADER_NAME", "X-API-Key")
    rate_limit_requests: int = _int_env("RATE_LIMIT_REQUESTS", 20)
    rate_limit_window_seconds: int = _int_env("RATE_LIMIT_WINDOW_SECONDS", 60)

    # Synthesis backend: "anthropic" (native Claude) or "openai"
    # (OpenAI Chat Completions-compatible: OpenAI, local models, OpenRouter,
    # codex/opencode-style gateways via OPENAI_BASE_URL).
    llm_provider: str = os.getenv("LLM_PROVIDER", "anthropic")
    llm_max_tokens: int = _int_env(
        "LLM_MAX_TOKENS", _int_env("CLAUDE_MAX_TOKENS", 1800)
    )
    llm_temperature: float = _float_env("LLM_TEMPERATURE", 0.2)

    # Agent loop guardrails: bound how many model turns and tool calls a single
    # brief may consume so latency and cost stay predictable.
    agent_max_iterations: int = _int_env("AGENT_MAX_ITERATIONS", 8)
    agent_max_tool_calls: int = _int_env("AGENT_MAX_TOOL_CALLS", 12)
    agent_max_search_calls: int = _int_env("AGENT_MAX_SEARCH_CALLS", 3)

    # Agent search auto-backfill: when the model's `search_papers` call finds no
    # local paper at or above the relevance floor, transparently fetch fresh
    # arXiv papers for that query, index them, and re-run the search once. This
    # makes cold-start questions work without the model having to notice thin
    # results and choose `fetch_arxiv` itself. Deduped per distinct query per run
    # and bounded by SEARCH_BACKFILL_MAX_PAPERS. Adds one arXiv round-trip on the
    # first search of an uncovered topic; set AGENT_SEARCH_AUTO_BACKFILL=false to
    # rely solely on the indexed corpus. The floor is SPECTER2 cosine similarity.
    # Calibrated 2026-07-21 against the seeded benchmark corpus (see
    # scripts/calibrate_backfill_floor.py): best-hit cosines for covered topics
    # measured 0.7318-0.8162, while an uncovered topic's best off-topic hits
    # measured ~0.71-0.73. 0.72 sits below the whole covered band with a small
    # margin; the bands are close because SPECTER2 cosines are compressed, so
    # treat the floor as a weak instrument and re-run the calibration if the
    # embedding model or corpus mix changes.
    agent_search_auto_backfill: bool = _bool_env("AGENT_SEARCH_AUTO_BACKFILL", True)
    agent_search_backfill_min_score: float = _float_env(
        "AGENT_SEARCH_BACKFILL_MIN_SCORE", 0.72
    )

    # Transcript compaction: each provider turn must include the prior
    # tool-call/tool-result pairs, but older bulky payloads can be replaced with
    # bounded summaries once the model has seen them.
    transcript_keep_recent_tool_results: int = _int_env(
        "TRANSCRIPT_KEEP_RECENT_TOOL_RESULTS", 1
    )
    transcript_full_text_excerpt_chars: int = _int_env(
        "TRANSCRIPT_FULL_TEXT_EXCERPT_CHARS", 2500
    )
    transcript_abstract_excerpt_chars: int = _int_env(
        "TRANSCRIPT_ABSTRACT_EXCERPT_CHARS", 500
    )

    # Full-text evidence tool: how much paper body the agent may pull per paper,
    # how many papers per call and per run, plus the PDF fetch timeout.
    full_text_char_budget: int = _int_env("FULL_TEXT_CHAR_BUDGET", 12000)
    full_text_max_papers: int = _int_env("FULL_TEXT_MAX_PAPERS", 3)
    full_text_total_paper_budget: int = _int_env("FULL_TEXT_TOTAL_PAPER_BUDGET", 3)
    full_text_timeout_s: float = _float_env("FULL_TEXT_TIMEOUT_S", 20.0)
    pdf_extractor: str = os.getenv("PDF_EXTRACTOR", "auto")

    anthropic_api_key: str | None = os.getenv("ANTHROPIC_API_KEY")
    anthropic_model: str = os.getenv(
        "ANTHROPIC_MODEL", os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")
    )

    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    openai_model: str = os.getenv("OPENAI_MODEL", "deepseek-v4-flash")
    openai_base_url: str | None = os.getenv("OPENAI_BASE_URL")

    embedding_model: str = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    embedding_query_adapter: str = os.getenv(
        "EMBEDDING_QUERY_ADAPTER", DEFAULT_QUERY_ADAPTER
    )
    embedding_document_adapter: str = os.getenv(
        "EMBEDDING_DOCUMENT_ADAPTER", DEFAULT_DOCUMENT_ADAPTER
    )
    embedding_batch_size: int = _int_env("EMBEDDING_BATCH_SIZE", 16)
    embedding_dimension: int = _int_env("EMBEDDING_DIMENSION", 768)

    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key: str | None = os.getenv("QDRANT_API_KEY")
    # Embedded local mode: when set, qdrant-client persists to this folder
    # in-process (no server / Docker). Takes precedence over qdrant_url.
    qdrant_path: str | None = os.getenv("QDRANT_PATH")
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "arxiv_papers")
    vector_store_backend: str = os.getenv("VECTOR_STORE_BACKEND", "qdrant")

    langfuse_public_key: str | None = os.getenv("LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str | None = os.getenv("LANGFUSE_SECRET_KEY")
    langfuse_host: str | None = _optional_env(
        "LANGFUSE_BASE_URL", _optional_env("LANGFUSE_HOST")
    )
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    structured_logs: bool = _bool_env("STRUCTURED_LOGS", True)
    run_records_dir: str | None = _optional_env("RUN_RECORDS_DIR", ".local/run-records")
    # When true, run-record persistence is treated as critical: the directory
    # is probed at startup and write failures propagate. Default is best-effort
    # so observability IO faults never break the agent stream.
    run_records_required: bool = _bool_env("RUN_RECORDS_REQUIRED", False)

    default_max_papers: int = _int_env("DEFAULT_MAX_PAPERS", 12)
    max_ingest_results: int = _int_env("MAX_INGEST_RESULTS", 200)
    # How arXiv fetches are ordered: "relevance" (default), "submitted_date",
    # or "last_updated". Relevance avoids biasing the corpus toward the newest
    # submissions and missing seminal older work; use a date sort for recency.
    arxiv_sort: str = os.getenv("ARXIV_SORT", "relevance")
    max_retrieval_results: int = _int_env("MAX_RETRIEVAL_RESULTS", 20)
    retrieval_min_score: float = _float_env("RETRIEVAL_MIN_SCORE", 0.0)

    # Search-time query handling. SPECTER2's query adapter ranks best on
    # descriptive, abstract-like text, so short keyword queries are expanded via
    # the LLM (HyDE-style) before embedding. Optional backfill fetches fresh
    # arXiv papers for the query when explicitly enabled for slower coverage.
    query_expansion_enabled: bool = _bool_env("QUERY_EXPANSION_ENABLED", True)
    query_expansion_max_words: int = _int_env("QUERY_EXPANSION_MAX_WORDS", 12)
    search_auto_backfill: bool = _bool_env("SEARCH_AUTO_BACKFILL", False)
    search_backfill_max_papers: int = _int_env("SEARCH_BACKFILL_MAX_PAPERS", 25)
    search_backfill_query_expansion: bool = _bool_env(
        "SEARCH_BACKFILL_QUERY_EXPANSION", True
    )
    rerank_enabled: bool = _bool_env("RERANK_ENABLED", False)
    rerank_model: str = os.getenv(
        "RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    rerank_candidate_k: int = _int_env("RERANK_CANDIDATE_K", 40)
    estimated_input_token_cost_per_1k: float = _float_env(
        "ESTIMATED_INPUT_TOKEN_COST_PER_1K", 0.003
    )
    estimated_output_token_cost_per_1k: float = _float_env(
        "ESTIMATED_OUTPUT_TOKEN_COST_PER_1K", 0.015
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
