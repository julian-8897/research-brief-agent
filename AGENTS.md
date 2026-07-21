# AGENTS.md

Single source of truth for agents working in this repo. `CLAUDE.md` defers here. Current status and deployment gaps live in [PROGRESS.md](PROGRESS.md) — keep it updated as work lands.

## What this is

A streaming FastAPI service that turns a research/engineering decision question into a cited decision memo for AI/ML and scientific-ML engineers and researchers (method selection, architecture tradeoffs, technique adoption, deployment and uncertainty risk). The evidence backend is arXiv literature: a tool-using LLM agent drives its own retrieval (semantic search over a persistent Qdrant corpus, with on-demand arXiv backfill), reads promising full texts, then writes the memo. Token cost, latency, and a Langfuse trace are returned as first-class output.

## Commands

Python is managed with `uv`. Run everything through it.

```bash
uv sync --dev                      # install deps + dev group
uv run pytest -q                   # full test suite
uv run pytest tests/test_agent.py::test_agent_runs_tool_loop_and_reports_measured_usage  # single test
uv run ruff check .                # lint
uv run ruff format .               # format
uv run ruff format --check .       # format check (CI enforces this)
uv run uvicorn src.api.main:app --reload   # run API locally (in-memory store if Qdrant absent)
uv run python evals/run_eval.py    # latency/cost + automated quality metrics -> evals/reports/latest.{jsonl,md} (default output paths overwrite the release evidence; pass --jsonl/--markdown to redirect)
uv run python evals/run_eval.py --offline-fixture   # hermetic smoke run (no keys/network)
uv run python evals/run_eval.py --fixture-corpus --judge  # add LLM faithfulness/answer-relevance grading
docker compose up --build          # full stack: API + Qdrant
```

Tests run without any API keys or network: with no provider key the agent uses a deterministic fallback memo, and provider/translation tests fake the SDK modules. Keep it that way.

## Architecture

Two phases. **Ingest** (`POST /ingest`): fetch arXiv metadata, embed title+abstract with SPECTER2's document/proximity adapter, upsert into Qdrant. **Brief** (`POST /briefs/stream`): the agent loop answers a technical decision question against that corpus using SPECTER2's adhoc-query adapter for search text.

The brief request is the core flow and spans several modules:

- `src/api/main.py` — FastAPI app, dependency-injected `Services`, the streaming `/briefs/stream` endpoint (SSE), plus `/ingest`, `/papers/search`, `/health`.
- `src/agent/brief_agent.py` — **the agent loop**. Builds a system prompt + tool catalogue and calls `provider.run_turn()` repeatedly: model requests tools → loop dispatches them → results fed back → repeat until the model writes the memo. Accumulates measured token usage across turns. `AGENT_MAX_ITERATIONS` / `AGENT_MAX_TOOL_CALLS` bound the loop; `AGENT_MAX_SEARCH_CALLS` withdraws discovery tools after enough search/fetch calls so the model must read or write. On budget exhaustion it forces a final synthesis with `tool_choice="none"` and falls back deterministically if the provider leaks tool-call markup. `stream()` (async, yields SSE events) and `run()` (sync) both wrap one sync generator `_iterate()`, so loop logic lives in one place.
- `src/agent/toolset.py` — `ResearchToolset` adapts `ResearchTools` into the four model-callable tools (`search_papers`, `fetch_arxiv`, `get_paper_details`, `get_full_text`) and holds per-run state (retrieved papers, fetched arXiv ids, a full-text cache) so citations and diagnostics can be reconstructed regardless of call order. Repeated `fetch_arxiv` calls report `new: 0` plus a stop hint instead of pretending to add evidence.
- `src/agent/tools.py` — `ResearchTools`: the underlying stateless operations (vector retrieve, arXiv fetch+ingest, evidence extraction).
- `src/ingestion/full_text.py` / `src/ingestion/pdf_extractors.py` — `fetch_arxiv_fulltext`: downloads a paper PDF and extracts body text with the configured local extractor, bounded by page count and `FULL_TEXT_CHAR_BUDGET`. `PDF_EXTRACTOR=auto` uses optional Docling when installed and falls back to pypdf; `pypdf` remains the lightweight always-installed path. This backs the `get_full_text` tool so the agent reads methods/results, not just abstracts.
- `src/llm/` — **the pluggable backend**. `base.py` defines a canonical, provider-neutral tool-use layer (`ToolSpec`, `ToolCall`, `ToolResult`, message types, `TurnResult`, `LLMProvider.run_turn`). `anthropic_provider.py` and `openai_provider.py` each translate that canonical form to/from their own wire format (Anthropic `tool_use`/`tool_result` blocks vs OpenAI `tool_calls` + `role:"tool"`). `build_llm_provider(settings)` selects by `LLM_PROVIDER` and returns `None` when no key is set.
- `src/retrieval/` — `build_vector_store(settings)` returns Qdrant or an in-memory store (fallback when Qdrant is unreachable). This is the canonical store layer.
- `src/observability/` — `Tracer` wraps Langfuse; every agent turn and tool call is a span. `RunRecordStore` writes append-only JSONL records for `/briefs/stream` runs under `RUN_RECORDS_DIR` (default `.local/run-records`); persistence is best-effort so IO faults (unwritable dir, full disk) are logged once but never break the stream, unless `RUN_RECORDS_REQUIRED=true`, which probes the directory at startup and propagates write failures, and structured logs include request/run ids, HTTP status, agent events, tool/turn counts, token usage, cost, fallback/degraded reasons, and provider/model metadata.
- `src/settings.py` — frozen `Settings` dataclass, all config from env. `get_settings()` is the cached entrypoint.
- `src/models.py` — Pydantic request/response models shared across layers.

## Conventions and gotchas

- **The LLM is genuinely agentic.** It chooses its own retrieval strategy via tool calls; do not reintroduce a hardcoded retrieve→synthesize pipeline. Tools are dispatched by the loop in `brief_agent.py`, not chosen by Python.
- **Token usage is measured, not estimated.** Always read counts from the provider `TurnResult`. The `_estimate_tokens` char heuristic exists only for the offline fallback memo, never the live path.
- **Provider parity is the fragile part.** Any change to the tool-use layer must be mirrored in both providers and covered by `tests/test_llm_providers.py` (which fakes the SDKs).
- **Both providers and the no-key fallback must keep working.** `LLM_PROVIDER=anthropic|openai`; `openai` also covers DeepSeek/local/OpenRouter/codex/opencode via `OPENAI_BASE_URL`. The default OpenAI-compatible model is `deepseek-v4-flash`; override `OPENAI_MODEL` for other endpoints.
- Config is env-driven; see `.env.example`. A local `.env` is auto-loaded (python-dotenv in `src/settings.py`), so secrets live there (gitignored), not in the shell. Defaults favor a runnable local setup (in-memory store, deterministic fallback) when services/keys are missing. Embeddings default to `EMBEDDING_MODEL=allenai/specter2_base`, `EMBEDDING_DOCUMENT_ADAPTER=allenai/specter2`, and `EMBEDDING_QUERY_ADAPTER=allenai/specter2_adhoc_query`; `TextEmbedder` construction rejects the known SPECTER1-base/canonical-SPECTER2-adapter mismatch while leaving renamed repositories and local adapter paths available for custom deployments.
- Retrieval quality is guarded in the agent layer: `search_papers` embeds descriptive text built from the brief request, tool query, and constraints, and `RETRIEVAL_MIN_SCORE` can drop weak vector matches before they reach the model.
- **Cold-start coverage — agent search auto-backfill.** When `search_papers` finds no local paper at/above `AGENT_SEARCH_BACKFILL_MIN_SCORE`, `ResearchToolset` transparently fetches fresh arXiv papers for that query, indexes them, and re-runs the search once (deduped per query per run, `AGENT_SEARCH_AUTO_BACKFILL=true` by default). This is why a question against a thin corpus no longer returns an honest-but-useless "no evidence" memo: the tool tops up its own evidence instead of relying on the model to notice and call `fetch_arxiv`. The floor is SPECTER2 cosine and interacts with the adapter-activation state — see the caveat below.
- **Warm standing corpus.** `scripts/seed_corpus.py` fetches one topical arXiv query per benchmark case into an embedded on-disk Qdrant store (`.local/qdrant-corpus`, ~320 papers), so retrieval is meaningful from the first request. The seeder pins `VECTOR_STORE_BACKEND=qdrant` and calls `store.close()` so writes actually flush (embedded local Qdrant persists on close; interpreter-exit finalization does not reliably run it — writers must close). Point the app at it with `QDRANT_PATH=.local/qdrant-corpus VECTOR_STORE_BACKEND=qdrant`. Embedded mode is single-process: stop the app before re-seeding.
- **Known caveat (open):** the SPECTER2 document/query adapters log "adapters available but none are activated"/"Could not identify valid prediction head(s)" and may not be applying, so absolute similarity scores are compressed (on- and off-topic both ~0.71) and only ranking is reliable. This means the `AGENT_SEARCH_BACKFILL_MIN_SCORE=0.75` default currently behaves as "always backfill"; retune it once the adapter activation is fixed.
- arXiv fetches default to relevance ordering (`ARXIV_SORT=relevance`, via `resolve_sort_criterion` in `src/arxiv_client.py`), not newest-first, so the ingested corpus is not biased toward the latest submissions; set `submitted_date`/`last_updated` for recency-focused runs.
- `/papers/search` (via `ResearchTools.semantic_search`) adds two optional search-quality behaviors, both toggleable per-request (`expand`/`backfill` query params) or by settings: query expansion (`src/agent/query_expansion.py`, HyDE-style — short keyword queries are rewritten into an abstract-like sentence before embedding, since SPECTER2 ranks descriptive text better) and arXiv backfill (fetch+index fresh papers for the query so the endpoint has coverage beyond the existing corpus). Backfill defaults off because it adds arXiv + embedding latency; when enabled, already-indexed arXiv ids are skipped before embedding. Backfill uses a separate compact keyword/Boolean arXiv-query expansion when a provider is available. Optional cross-encoder reranking (`RERANK_ENABLED=false` by default) reorders an over-retrieved vector candidate pool for precision.
- Brief quality is measured, not just eyeballed: `evals/metrics.py` computes deterministic citation-grounding (hallucinated ids, fraction read in full), evidence-utilization, and uncertainty-signaling scores on every eval run. The optional `--judge` flag adds two LLM graders: whole-brief faithfulness/answer-relevance, and per-citation grounding (`citation_grounding_judge`) that checks each inline `[id]` claim is semantically supported by that specific paper, catching real-but-misused citations the deterministic check cannot. `FullTextDiagnostics.succeeded_ids` records which cited papers were actually read.
- **Tests are hermetic by contract:** `tests/conftest.py` sets `DISABLE_DOTENV=1` and clears provider keys before `src.settings` imports, so a real key in `.env` never causes the suite to hit a live LLM. `Settings` reads env at class-definition (import) time — overrides must be passed explicitly to `Settings(...)`, not set after import.
- **Supported Python runtime: 3.11–3.13.** `requires-python = ">=3.11,<3.14"` in `pyproject.toml`; CI tests all three versions in a matrix and the Docker image pins 3.12. The upper bound is deliberate: an open-ended `>=3.10` previously let `uv` select the system's Python 3.14 and break test collection. Keep the pyproject bound, the CI matrix (`.github/workflows/ci.yml`), and the Docker base image in sync. After changing `requires-python`, run `uv lock` so `uv sync --frozen` stays valid.
