# AGENTS.md

Single source of truth for agents working in this repo. `CLAUDE.md` defers here. Current status and deployment gaps live in [PROGRESS.md](PROGRESS.md) — keep it updated as work lands.

## What this is

A streaming FastAPI service that turns a research question into a cited decision memo over arXiv literature. A tool-using LLM agent drives its own retrieval (semantic search over a persistent Qdrant corpus, with on-demand arXiv backfill), then writes the memo. Token cost, latency, and a Langfuse trace are returned as first-class output.

## Commands

Python is managed with `uv`. Run everything through it.

```bash
uv sync --dev                      # install deps + dev group
uv run pytest -q                   # full test suite
uv run pytest tests/test_agent.py::test_agent_runs_tool_loop_and_reports_measured_usage  # single test
uv run ruff check .                # lint
uv run ruff format .               # format
uv run uvicorn src.api.main:app --reload   # run API locally (in-memory store if Qdrant absent)
uv run python evals/run_eval.py    # latency/cost benchmark -> evals/reports/latest.{jsonl,md}
docker compose up --build          # full stack: API + Qdrant
```

Tests run without any API keys or network: with no provider key the agent uses a deterministic fallback memo, and provider/translation tests fake the SDK modules. Keep it that way.

## Architecture

Two phases. **Ingest** (`POST /ingest`): fetch arXiv metadata, embed title+abstract with SPECTER2's document/proximity adapter, upsert into Qdrant. **Brief** (`POST /briefs/stream`): the agent loop answers a question against that corpus using SPECTER2's adhoc-query adapter for search text.

The brief request is the core flow and spans several modules:

- `src/api/main.py` — FastAPI app, dependency-injected `Services`, the streaming `/briefs/stream` endpoint (SSE), plus `/ingest`, `/papers/search`, `/health`.
- `src/agent/brief_agent.py` — **the agent loop**. Builds a system prompt + tool catalogue and calls `provider.run_turn()` repeatedly: model requests tools → loop dispatches them → results fed back → repeat until the model writes the memo. Accumulates measured token usage across turns. `AGENT_MAX_ITERATIONS` / `AGENT_MAX_TOOL_CALLS` bound the loop; `AGENT_MAX_SEARCH_CALLS` withdraws discovery tools after enough search/fetch calls so the model must read or write. On budget exhaustion it forces a final synthesis with `tool_choice="none"` and falls back deterministically if the provider leaks tool-call markup. `stream()` (async, yields SSE events) and `run()` (sync) both wrap one sync generator `_iterate()`, so loop logic lives in one place.
- `src/agent/toolset.py` — `ResearchToolset` adapts `ResearchTools` into the four model-callable tools (`search_papers`, `fetch_arxiv`, `get_paper_details`, `get_full_text`) and holds per-run state (retrieved papers, fetched arXiv ids, a full-text cache) so citations and diagnostics can be reconstructed regardless of call order. Repeated `fetch_arxiv` calls report `new: 0` plus a stop hint instead of pretending to add evidence.
- `src/agent/tools.py` — `ResearchTools`: the underlying stateless operations (vector retrieve, arXiv fetch+ingest, evidence extraction).
- `src/ingestion/full_text.py` — `fetch_arxiv_fulltext`: downloads a paper PDF and extracts body text with pypdf, bounded by page count and `FULL_TEXT_CHAR_BUDGET`. Deployable (runs in-process), backing the `get_full_text` tool so the agent reads methods/results, not just abstracts.
- `src/llm/` — **the pluggable backend**. `base.py` defines a canonical, provider-neutral tool-use layer (`ToolSpec`, `ToolCall`, `ToolResult`, message types, `TurnResult`, `LLMProvider.run_turn`). `anthropic_provider.py` and `openai_provider.py` each translate that canonical form to/from their own wire format (Anthropic `tool_use`/`tool_result` blocks vs OpenAI `tool_calls` + `role:"tool"`). `build_llm_provider(settings)` selects by `LLM_PROVIDER` and returns `None` when no key is set.
- `src/retrieval/` — `build_vector_store(settings)` returns Qdrant or an in-memory store (fallback when Qdrant is unreachable). This is the canonical store layer.
- `src/observability/tracing.py` — `Tracer` wraps Langfuse; every agent turn and tool call is a span. No-ops cleanly when Langfuse keys are absent.
- `src/settings.py` — frozen `Settings` dataclass, all config from env. `get_settings()` is the cached entrypoint.
- `src/models.py` — Pydantic request/response models shared across layers.

## Conventions and gotchas

- **The LLM is genuinely agentic.** It chooses its own retrieval strategy via tool calls; do not reintroduce a hardcoded retrieve→synthesize pipeline. Tools are dispatched by the loop in `brief_agent.py`, not chosen by Python.
- **Token usage is measured, not estimated.** Always read counts from the provider `TurnResult`. The `_estimate_tokens` char heuristic exists only for the offline fallback memo, never the live path.
- **Provider parity is the fragile part.** Any change to the tool-use layer must be mirrored in both providers and covered by `tests/test_llm_providers.py` (which fakes the SDKs).
- **Both providers and the no-key fallback must keep working.** `LLM_PROVIDER=anthropic|openai`; `openai` also covers DeepSeek/local/OpenRouter/codex/opencode via `OPENAI_BASE_URL`. The default OpenAI-compatible model is `deepseek-v4-flash`; override `OPENAI_MODEL` for other endpoints.
- **Legacy, do not build on:** `src/vector_store.py` is a backward-compat alias for old scripts; `scripts/run_arxiv_search.py` and `config/categories.yaml` are leftovers from the prior Streamlit demo. New code uses `src/retrieval/`.
- Config is env-driven; see `.env.example`. A local `.env` is auto-loaded (python-dotenv in `src/settings.py`), so secrets live there (gitignored), not in the shell. Defaults favor a runnable local setup (in-memory store, deterministic fallback) when services/keys are missing. Embeddings default to `EMBEDDING_MODEL=allenai/specter2_base`, `EMBEDDING_DOCUMENT_ADAPTER=allenai/specter2`, and `EMBEDDING_QUERY_ADAPTER=allenai/specter2_adhoc_query`.
- Retrieval quality is guarded in the agent layer: `search_papers` embeds descriptive text built from the brief request, tool query, and constraints, and `RETRIEVAL_MIN_SCORE` can drop weak vector matches before they reach the model.
- **Tests are hermetic by contract:** `tests/conftest.py` sets `DISABLE_DOTENV=1` and clears provider keys before `src.settings` imports, so a real key in `.env` never causes the suite to hit a live LLM. `Settings` reads env at class-definition (import) time — overrides must be passed explicitly to `Settings(...)`, not set after import.
