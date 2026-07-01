import os
from dataclasses import dataclass
from functools import lru_cache

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

    anthropic_api_key: str | None = os.getenv("ANTHROPIC_API_KEY")
    anthropic_model: str = os.getenv(
        "ANTHROPIC_MODEL", os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")
    )

    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    openai_model: str = os.getenv("OPENAI_MODEL", "deepseek-v4-flash")
    openai_base_url: str | None = os.getenv("OPENAI_BASE_URL")

    embedding_model: str = os.getenv("EMBEDDING_MODEL", "allenai/specter2_base")
    embedding_query_adapter: str = os.getenv(
        "EMBEDDING_QUERY_ADAPTER", "allenai/specter2_adhoc_query"
    )
    embedding_document_adapter: str = os.getenv(
        "EMBEDDING_DOCUMENT_ADAPTER", "allenai/specter2"
    )
    embedding_batch_size: int = _int_env("EMBEDDING_BATCH_SIZE", 16)
    embedding_dimension: int = _int_env("EMBEDDING_DIMENSION", 768)

    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key: str | None = os.getenv("QDRANT_API_KEY")
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "arxiv_papers")
    vector_store_backend: str = os.getenv("VECTOR_STORE_BACKEND", "qdrant")

    langfuse_public_key: str | None = os.getenv("LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str | None = os.getenv("LANGFUSE_SECRET_KEY")
    langfuse_host: str | None = os.getenv("LANGFUSE_HOST")

    default_max_papers: int = _int_env("DEFAULT_MAX_PAPERS", 12)
    max_ingest_results: int = _int_env("MAX_INGEST_RESULTS", 200)
    max_retrieval_results: int = _int_env("MAX_RETRIEVAL_RESULTS", 20)
    retrieval_min_score: float = _float_env("RETRIEVAL_MIN_SCORE", 0.0)
    estimated_input_token_cost_per_1k: float = _float_env(
        "ESTIMATED_INPUT_TOKEN_COST_PER_1K", 0.003
    )
    estimated_output_token_cost_per_1k: float = _float_env(
        "ESTIMATED_OUTPUT_TOKEN_COST_PER_1K", 0.015
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
