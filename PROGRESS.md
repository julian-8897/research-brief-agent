# Project Progress

Living status tracker for Research Brief Agent. Update this as work lands.

_Last updated: 2026-07-03_

## Status at a glance

The core product works end-to-end and is **runnable locally today**. It is scoped to a cited recommendation memo for AI/ML and scientific-ML engineering decisions; the evidence backend is arXiv/SPECTER/Qdrant only (no plan to add other source backends). Direct `uvicorn` smoke testing works without Docker: `/` serves the static UI, `/briefs/stream` emits contracted SSE events and a final `BriefResponse`, and `/papers/search` returns results. Persistent local Qdrant has now been smoke-tested with the official macOS arm64 Qdrant binary, repo-local storage under `.local/qdrant-storage`, a 3-paper arXiv ingest, semantic search, fallback brief generation, and a Qdrant+uvicorn restart that preserved `papers_indexed=3` with `retrieval_backend=qdrant` and `vector_store.fallback=false`. Live DeepSeek `deepseek-v4-flash` has been validated with the fixture corpus: 3/3 eval cases returned `ok`, no deterministic fallback, no warnings, no tool-call markup, and full-text success for all attempted papers. The Fly.io deployment manifest and image path are present, but the app has not been deployed from this checkout. The public deployment path runs the real agent loop with an LLM key, gated by `X-API-Key` and a tight per-IP limiter. Normal live synthesis is gated on successful full-text evidence when retrieved papers exist; if the run cannot meet that contract before budget/failure, the stream emits explicit degraded/error events instead of silently looking successful.

## Scope (2026-07-03)

- **Audience:** AI/ML and scientific-ML engineers and researchers making evidence-backed engineering decisions.
- **Product promise:** ask a research/engineering decision question (method selection, architecture tradeoffs, technique adoption, deployment/uncertainty risk) and receive a cited memo with recommendation, evidence, tradeoffs, risks, uncertainty, and next actions.
- **Source backend:** arXiv papers with SPECTER retrieval, arXiv backfill, and full-text PDF reading. This is deliberately the only evidence backend; the project stays a niche so brief quality can be measured, rather than generalizing into a multi-connector platform.
- **PDF extraction:** full-text reading goes through `src/ingestion/pdf_extractors.py`; `PDF_EXTRACTOR=auto` uses optional Docling when installed and falls back to pypdf, while `PDF_EXTRACTOR=docling` can require the layout-aware backend explicitly.

## Search quality follow-ups (2026-07-01)

- **Backfill dedup:** vector stores now expose `existing_ids()`, and `ResearchTools.ingest_papers` skips already-indexed arXiv ids before embedding/upsert. Repeated identical backfill searches fetch metadata but embed 0 duplicate papers.
- **Fast search default:** `SEARCH_AUTO_BACKFILL` now defaults to `false`; `/papers/search` stays local unless settings or `?backfill=true` enables arXiv coverage. Backfill can still be toggled per request.
- **Backfill query expansion:** arXiv backfill uses a separate compact keyword/Boolean query generator (`expand_arxiv_query`) when a provider is available. It falls back to the raw query without a provider or on provider errors; semantic HyDE expansion remains separate for vector embedding.
- **Optional reranking:** `src/rerank.py` adds a lazy Transformers cross-encoder reranker (`RERANK_ENABLED=false`, `RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2`, `RERANK_CANDIDATE_K=40`). `vector_retrieve` over-retrieves up to the configured candidate pool, then reranks the raw user query against title+abstract text.
- **Retrieval metrics:** fixture eval cases now carry relevant ids, and `evals/metrics.py` reports recall@k and nDCG@k. Fixture validation with rerank off: retrieval recall@k 100% for all cases, nDCG@k 100%, 63%, 63%. With rerank on: recall@k 100% for all cases, nDCG@k 100%, 100%, 100%. Citation grounding remained 100%, hallucination 0%, and cited-papers-read-in-full 100% in both runs.
- **Regression coverage:** full suite is now 95 passing with one Starlette/httpx deprecation warning, ruff clean.

## Operability pass (2026-07-01)

- **Persisted run records:** `/briefs/stream` now assigns `X-Request-ID` and `X-Run-ID`, enriches every SSE event with both ids, and writes append-only JSONL records under `RUN_RECORDS_DIR` (default `.local/run-records`) plus a `runs.jsonl` index. Records include request metadata, provider/model, vector backend, every streamed agent event, final diagnostics, token usage, cost, warnings, and completion status.
- **Structured logging:** FastAPI startup configures JSON logs by default (`STRUCTURED_LOGS=true`, `LOG_LEVEL=INFO`). HTTP requests, auth/rate-limit failures, brief events, errors, and run summaries are logged with request/run ids, provider/model, timings, tool/turn counts, token usage, and degraded/fallback context.
- **Deployment smoke status:** Fly deployment smoke is still blocked in this environment because neither `fly` nor `flyctl` is installed. Local API smoke remains runnable with `uv run uvicorn src.api.main:app --reload`.
- **Regression coverage:** full suite covers persisted stream records and remains hermetic by disabling `RUN_RECORDS_DIR` in tests unless a test opts into a temp directory.

## Frontend redesign and ask-first UX (2026-06-30)

- **Retrieval-quality fixes.** `search_papers` now embeds descriptive query text built from the brief's decision question, model query, and constraints while preserving the model's original displayed query; `RETRIEVAL_MIN_SCORE` adds an optional relevance floor with an empty-result hint that steers the model to `fetch_arxiv`.
- **SPECTER2 asymmetric embeddings.** Paper indexing now uses `allenai/specter2_base` with the `allenai/specter2` proximity adapter, while retrieval queries use the `allenai/specter2_adhoc_query` adapter. Adapter switching and forward passes are locked so concurrent FastAPI requests do not race shared model state.
- **Console redesign.** `static/index.html` was reworked from a generic gray dashboard into a single-page console: a deep-ink instrument surface with a warm-paper memo, a streaming reasoning-spine timeline, a telemetry gauge cluster, and masthead health readouts (Space Grotesk / Fraunces / IBM Plex Mono from Google Fonts). The agent/SSE wiring is unchanged; only markup and CSS were rewritten.
- **Ask-first UI.** The page is now a single "Ask the agent" flow. The manual corpus tools (ingest form, search preview) were removed from the UI because the agent backfills arXiv itself via `fetch_arxiv`; the orphaned "Search Results" tab and its now-dead JS were removed. The `/ingest` and `/papers/search` endpoints remain available programmatically.
- **Plain-language queries.** `build_arxiv_query` (`src/ingestion/arxiv_ingestion.py`) now wraps a bare query in `all:` so general phrases work without arXiv field syntax, while field-qualified queries (`cat:`, `ti:`, `all:` ...) pass through unchanged. This covers both the `/ingest` path and the agent's own `fetch_arxiv` queries. Three new tests in `tests/test_ingestion.py`.
- **Tests:** 59 passing, ruff clean.

## Tier 1 deploy-safety bundle

- **Fly manifest added:** `fly.toml` provisions an API machine, mounts a `qdrant_data` volume at `/qdrant/storage`, runs Qdrant locally on the Fly machine, and points the API at `http://127.0.0.1:6333`.
- **Cold-start fix:** `Dockerfile` now downloads the configured SPECTER embedding model during image build so the first request does not pay the model download cost.
- **Public endpoint protection:** non-`/health` routes are protected by env-driven API-key auth (`API_KEYS`, `API_KEY_HEADER_NAME`) and an in-process per-IP limiter (`RATE_LIMIT_REQUESTS`, `RATE_LIMIT_WINDOW_SECONDS`). Non-local startup requires `API_KEYS`.
- **Environment-aware fallback:** vector-store construction only falls back to memory in `ENVIRONMENT=local`; non-local environments raise instead of silently losing persistence.
- **Readiness health:** `/health` is unauthenticated and returns structured `vector_store` and `llm_provider` status. It returns 503 when not ready, including local Qdrant fallback or missing LLM key.

## Reliability guard rails fixed this pass

- **Full-text-before-final gate:** `ResearchToolset` tracks successful full-text reads, attempts, and errors. `ResearchBriefAgent` blocks normal final synthesis after retrieval until at least one `get_full_text` call succeeds, emits `evidence_required`, and only allows budget-forced synthesis with an explicit degraded warning.
- **Visible live-loop failure path:** provider/tool boundary failures now emit `error` and `degraded` SSE events before deterministic fallback. The broad whole-loop `except Exception` fallback was removed; known live-boundary failures are wrapped as `AgentLiveRunError`, while programmer errors are allowed to surface during development.
- **SSE contract:** `/briefs/stream` now formats events through `src/api/sse.py`, which defines stable event types and required fields. Unknown events and missing required fields fail fast at the API boundary. `warning` is now a first-class stream event while remaining duplicated in the final response for compatibility.
- **UI visibility:** the static workflow console now labels `evidence_required`, `warning`, `degraded`, and `error` events in the timeline/state strip and surfaces streaming warnings before the final response.
- **Regression coverage:** full suite is now 54 passing with one Starlette/httpx deprecation warning, ruff clean. New tests cover the full-text gate, provider failure events, full-text success/error counters, arXiv id normalization, plain-query `all:` wrapping, full-text budget enforcement, transcript compaction, classified full-text failures, DeepSeek V4 thinking-mode compatibility, the SSE event contract, API-key auth, rate limiting, environment-aware vector-store fallback, and readiness health.

## Reliability review: remaining brittle points

These are the highest-priority guard-rail gaps still open.

1. **Live eval report now has a clean DeepSeek fixture run, but needs a commit decision.** `evals/reports/deepseek-v4-flash-fixture.{jsonl,md}` records the latest live fixture-corpus run. It should either be committed as a qualified provider-cost report or regenerated as `latest.*` before release.
2. **Static UI is useful but not yet a robust operator console.** It visualizes the run, but it does not yet expose config budgets, model/provider identity, retry controls, or persisted run history.
3. **Deployment still needs a live smoke.** Fly config exists, but this checkout has not been deployed and validated against a public URL; local Fly CLI tooling is missing in this environment.

## DeepSeek V4 agent-loop policy fixes

A live DeepSeek V4 run exposed policy failures: repeated discovery calls with no convergence to `get_full_text`, forced-final synthesis leaking raw tool-call markup when tools were omitted, and repeated `fetch_arxiv` calls reporting the same papers as newly ingested. Implemented fixes:

- The system prompt now caps discovery at two rounds and directs the model to read 2-3 full texts before writing.
- `AGENT_MAX_SEARCH_CALLS` defaults to 3; after that, the agent offers only tools appropriate to the remaining evidence requirement, emits `discovery_budget_reached`, and blocks hallucinated/offered-out tools.
- Forced-final synthesis now passes the real tool catalogue with `tool_choice="none"` for both OpenAI-compatible and Anthropic providers; leaked tool markup or empty output is replaced by the deterministic fallback memo with an operational note.
- `fetch_arxiv` tracks per-run ids and reports `new`, `already_known`, titles, and a stop hint when a repeated fetch adds nothing.
- Normal final synthesis is blocked until full-text evidence has been successfully read after retrieval; premature final text triggers `evidence_required`.
- Older tool results are compacted before later LLM turns so full-text bodies are not resent indefinitely after the model has had a turn to use them.
- DeepSeek V4 flash thinking mode is disabled automatically for the official DeepSeek API so the OpenAI-compatible adapter receives normal assistant `content` and tool calls instead of reasoning-only output.
- Final memo text is normalized to start at the `# Decision Memo` heading when the model prepends process commentary.
- Live DeepSeek V4 validation passed with `AGENT_MAX_SEARCH_CALLS=3`, `AGENT_MAX_ITERATIONS=8`, and a 3-paper full-text budget: the stream reached `get_full_text`, fetched 3 full texts, produced a real cited memo, and returned no fallback/tool-markup warnings.
- Live fixture eval on `deepseek-v4-flash` passed 3/3 cases with status `ok`, no warnings, no fallback, no tool markup, and full-text success of 2/2, 2/2, and 3/3. Case latencies were ~22.8-25.9s and estimated costs were ~$0.068-$0.077 per case under the configured cost assumptions.
- The OpenAI-compatible default model and `.env.example` now use `deepseek-v4-flash` for forward compatibility.

## Local backend verification (Ollama / Qwen3 4B)

Confirmed the OpenAI-compatible backend works against a local model: the agent loop genuinely chose tools (`search_papers` → `fetch_arxiv` → synthesize), with measured token usage from Ollama responses. Caveats found with the 4B model that need a stronger model or config tuning:

- It did **not** call the deeper `get_full_text` tool even when available — small models are less proactive at multi-step tool use. A more directive prompt or a stronger model is needed to exercise full-text evidence.
- One run produced an empty final brief ("No brief was produced") despite generating tokens — likely Qwen3 thinking-mode consuming the `LLM_MAX_TOKENS=1800` budget before emitting the answer. Fixes: raise `LLM_MAX_TOKENS` (e.g. 4000+) and/or disable Qwen3 thinking; or use a non-reasoning small model / cloud model.
- Local cost figures are placeholders (configured per-1k rates); set `ESTIMATED_*_COST_PER_1K=0` for true local $0.

## Done

- Rescope from Streamlit semantic-search demo to a streaming FastAPI research-brief service.
- Endpoints: `POST /briefs/stream` (SSE), `POST /ingest`, `GET /papers/search`, `GET /health`.
- Persistent retrieval via Qdrant with automatic in-memory fallback only for `ENVIRONMENT=local`; non-local startup fails loudly when Qdrant is unavailable.
- SPECTER2 asymmetric embeddings for paper indexing/search (lazy-loaded on first encode): documents use the proximity adapter and queries use the adhoc-query adapter in the same 768-dim space.
- **Tool-using agent loop** (`src/agent/brief_agent.py`): model drives `search_papers` / `fetch_arxiv` / `get_paper_details` / `get_full_text`, with `AGENT_MAX_ITERATIONS` / `AGENT_MAX_TOOL_CALLS` / `AGENT_MAX_SEARCH_CALLS` budgets, discovery-tool withdrawal, and robust forced final synthesis.
- **Full-text evidence tool** (`src/ingestion/full_text.py`, `get_full_text`): downloads paper PDFs and extracts body text in-process (pypdf), bounded by `FULL_TEXT_CHAR_BUDGET`. Deployable, no external paper service. Real arXiv fetch verified; 3 unit tests. (Live: small local model doesn't yet choose to call it — see caveats above.)
- **Pluggable LLM backend** with full tool-use parity: Anthropic and OpenAI-compatible (`OPENAI_BASE_URL` covers OpenAI/local/OpenRouter/codex/opencode), via one canonical tool-use layer.
- Measured token usage/cost from provider responses (not estimated) on the live path.
- Langfuse tracing spans for every turn and tool call (no-ops without keys).
- Deterministic offline fallback memo so local runs and CI work without keys/network.
- Tests: 59 passing with one Starlette/httpx deprecation warning, ruff clean. Includes faked-SDK tests for both providers' wire translation and `tool_choice`, plus DeepSeek V4 thinking-mode compatibility, full-text evidence-gate, SSE contract, plain-query normalization, asymmetric embedding adapter activation, and degraded-fallback regressions.
- Container setup: `Dockerfile` (uv, frozen lock, baked SPECTER cache, Fly Qdrant sidecar binary) and `docker-compose.yml` (API + Qdrant v1.9.2 with a volume).
- Fly deployment manifest: `fly.toml` with a Qdrant volume mount, readiness check, API-key/rate-limit env, and documented secret provisioning commands.
- Docs: `README.md`, `AGENTS.md` (source of truth), `CLAUDE.md` (defers to AGENTS.md), `.env.example`.

## In progress / partial

- Eval harness exists (`evals/run_eval.py`, benchmark fixtures) and the latest named live DeepSeek fixture report is in `evals/reports/deepseek-v4-flash-fixture.{jsonl,md}`. Decide whether to commit that named report, regenerate `latest.*`, or keep reports untracked before release.
- `evals/run_eval.py --fixture-corpus` runs the configured live LLM against deterministic fixture papers and synthetic full text, so provider cost can be measured without external retrieval dependencies.
- Direct uvicorn smoke testing works in local fallback mode and persistent Qdrant mode. The persistent test used a 3-paper corpus to stay light on an 8GB MacBook, verified ingest/search/brief, then restarted Qdrant and uvicorn to confirm the collection persisted.
- Static UI (`static/index.html`) provides an ask-first console: a single technical decision-question form, masthead health readouts, a streaming reasoning-spine timeline, telemetry gauges, evidence/diagnostics tabs, and the final memo. Manual ingest and search preview were removed from the page (the agent backfills arXiv itself); both endpoints stay available via the API. It is still a single-file prototype and should be treated as an observability surface, not a production UI.
- arXiv date-filter failures are surfaced as JSON at the API boundary. The UI-side retry-without-date-range fallback was removed along with the ingest form, so programmatic `/ingest` callers handle this themselves. The underlying arXiv API remains flaky for some filtered queries, especially date ranges close to the current date.
- DeepSeek V4 live behaviour is improved and revalidated; token use is now measured in the fixture report, but real-corpus costs still need a production budget.

## Not started / gaps before "deployable"

- **Docker image not yet built/run** this session — compose is defined but intentionally deferred because the next local smoke target is direct uvicorn + separately started Qdrant.
- Real agent briefs require `ANTHROPIC_API_KEY` or `OPENAI_API_KEY`; otherwise only the fallback memo is produced.

## How to test it

Quickest, no Docker (in-memory store if Qdrant is absent; first call downloads SPECTER2 unless cached):

```bash
uv sync --dev
uv run uvicorn src.api.main:app --port 8000 &
curl -s http://localhost:8000/health
curl -N -X POST http://localhost:8000/briefs/stream \
  -H 'content-type: application/json' \
  -d '{"research_question":"What methods help source-grounded technical decision briefs?","max_papers":3}'
```

Persistent local Qdrant path:

```bash
# Start Qdrant separately so http://localhost:6333 is reachable first.
uv run uvicorn src.api.main:app --port 8000 &
curl -s http://localhost:8000/health  # expect retrieval_backend=qdrant, fallback=false
```

Real agent loop (set a key first), with a curated corpus:

```bash
cp .env.example .env   # add ANTHROPIC_API_KEY (or set LLM_PROVIDER=openai + OPENAI_API_KEY)
uv run uvicorn src.api.main:app --port 8000 &
curl -X POST http://localhost:8000/ingest \
  -H 'content-type: application/json' -d '{"query":"cat:cs.LG","max_papers":25}'
curl -N -X POST http://localhost:8000/briefs/stream \
  -H 'content-type: application/json' \
  -d '{"research_question":"Should an AI team use agentic RAG for technical decision briefs?","domain":"cs.LG","max_papers":6}'
```

Full persistent Docker stack, when needed: `cp .env.example .env && docker compose up --build`, then open `http://localhost:8000`.

## Suggested next steps

1. Commit or explicitly qualify the live DeepSeek fixture eval report, and decide whether `latest.*` should be regenerated from the named report before release.
2. Persist run transcripts and diagnostics so brittle model behaviour can be replayed, evaluated, and compared across model/provider changes.
3. Install Fly CLI and run the production smoke: deploy, hit `/health`, ingest a tiny corpus, stream one brief, restart the machine, and confirm Qdrant persistence plus run-record/log output.
4. Deploy the Fly app with real secrets, ingest a seed corpus, and run an end-to-end public URL smoke test.
