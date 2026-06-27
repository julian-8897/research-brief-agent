# Project Progress

Living status tracker for the arXiv Research Brief Agent. Update this as work lands.

_Last updated: 2026-06-27_

## Status at a glance

The core product works end-to-end and is **runnable locally today**. It is **not yet deployed** to the stated Fly.io target, and the real agent loop needs an LLM API key (without one it returns a deterministic fallback memo). Verified this session: app boots, `/health` 200, `/ingest` returns JSON errors for arXiv failures, the static workflow UI streams `/briefs/stream`, and a full interactive run can produce a cited memo. A DeepSeek V4 policy-fix pass is implemented and covered by hermetic tests. Normal live synthesis is now gated on successful full-text evidence when retrieved papers exist; if the run cannot meet that contract before budget/failure, the stream emits explicit degraded/error events instead of silently looking successful.

## Reliability guard rails fixed this pass

- **Full-text-before-final gate:** `ResearchToolset` tracks successful full-text reads, attempts, and errors. `ResearchBriefAgent` blocks normal final synthesis after retrieval until at least one `get_full_text` call succeeds, emits `evidence_required`, and only allows budget-forced synthesis with an explicit degraded warning.
- **Visible live-loop failure path:** provider/tool boundary failures now emit `error` and `degraded` SSE events before deterministic fallback. The broad whole-loop `except Exception` fallback was removed; known live-boundary failures are wrapped as `AgentLiveRunError`, while programmer errors are allowed to surface during development.
- **UI visibility:** the static workflow console now labels `evidence_required`, `degraded`, and `error` events in the timeline/state strip.
- **Regression coverage:** full suite is now 32 passing with one Starlette/httpx deprecation warning, ruff clean. New tests cover the full-text gate, provider failure events, and full-text success/error counters.

## Reliability review: remaining brittle points

These are the highest-priority guard-rail gaps still open.

1. **Tool argument validation is too permissive.** `ResearchToolset.call()` coerces model args with `str(...)` and `int(...)`; bad values can crash the loop or become odd queries like `"None"`. `k` and `max_results` are not clamped at the tool boundary. Next fix: validate each tool schema server-side, return structured tool errors for invalid args, and add tests for malformed model tool calls.
2. **Full-text fetch errors are still coarse.** `_get_full_text()` no longer treats empty extraction as success, and counters are tracked, but exceptions are still reported as generic string errors inside the tool payload. Next fix: catch known `httpx`/`pypdf` failures narrowly, classify timeout/HTTP/parse/empty-text/bad-id errors, and expose counts in diagnostics/UI.
3. **Qdrant startup failure silently switches to memory.** `create_services()` catches all exceptions when building the vector store and falls back to `InMemoryVectorStore`. This is convenient locally but dangerous in production because persistence can disappear without a hard failure. Next fix: only allow automatic memory fallback in `ENVIRONMENT=local`, or include a prominent `/health` warning.
4. **SSE has useful guard events but no formal contract.** `/briefs/stream` now emits `evidence_required`, `error`, and `degraded`, but the event schema is still implicit. Next fix: document and test stable event types so the UI can distinguish success, fallback, budget-forced final, arXiv outage, and partial evidence.
5. **Static UI is useful but not yet a robust operator console.** It visualizes the run, but it does not yet expose config budgets, model/provider identity, full-text coverage totals, retry controls, or persisted run history. It should become the place where brittle behaviour is visible, not hidden.
6. **Live DeepSeek revalidation is still pending.** The hermetic regressions cover the policy fixes, but a fresh cloud run should confirm that DeepSeek V4 now reaches `get_full_text` and avoids the old forced-final markup leak.

## DeepSeek V4 agent-loop policy fixes

A live DeepSeek V4 run exposed policy failures: repeated discovery calls with no convergence to `get_full_text`, forced-final synthesis leaking raw tool-call markup when tools were omitted, and repeated `fetch_arxiv` calls reporting the same papers as newly ingested. Implemented fixes:

- The system prompt now caps discovery at two rounds and directs the model to read 2-3 full texts before writing.
- `AGENT_MAX_SEARCH_CALLS` defaults to 4; after that, the agent offers only `get_paper_details` and `get_full_text`, emits `discovery_budget_reached`, and blocks hallucinated discovery calls.
- Forced-final synthesis now passes the real tool catalogue with `tool_choice="none"` for both OpenAI-compatible and Anthropic providers; leaked tool markup or empty output is replaced by the deterministic fallback memo with an operational note.
- `fetch_arxiv` tracks per-run ids and reports `new`, `already_known`, titles, and a stop hint when a repeated fetch adds nothing.
- Normal final synthesis is blocked until full-text evidence has been successfully read after retrieval; premature final text triggers `evidence_required`.
- The OpenAI-compatible default model and `.env.example` now use `deepseek-v4-flash` for forward compatibility.

## Local backend verification (Ollama / Qwen3 4B)

Confirmed the OpenAI-compatible backend works against a local model: the agent loop genuinely chose tools (`search_papers` → `fetch_arxiv` → synthesize), with measured token usage from Ollama responses. Caveats found with the 4B model that need a stronger model or config tuning:

- It did **not** call the deeper `get_full_text` tool even when available — small models are less proactive at multi-step tool use. A more directive prompt or a stronger model is needed to exercise full-text evidence.
- One run produced an empty final brief ("No brief was produced") despite generating tokens — likely Qwen3 thinking-mode consuming the `LLM_MAX_TOKENS=1800` budget before emitting the answer. Fixes: raise `LLM_MAX_TOKENS` (e.g. 4000+) and/or disable Qwen3 thinking; or use a non-reasoning small model / cloud model.
- Local cost figures are placeholders (configured per-1k rates); set `ESTIMATED_*_COST_PER_1K=0` for true local $0.

## Done

- Rescope from Streamlit semantic-search demo to a streaming FastAPI research-brief service.
- Endpoints: `POST /briefs/stream` (SSE), `POST /ingest`, `GET /papers/search`, `GET /health`.
- Persistent retrieval via Qdrant with automatic in-memory fallback when Qdrant is unreachable (verified).
- SPECTER embeddings for paper indexing/search (lazy-loaded on first encode).
- **Tool-using agent loop** (`src/agent/brief_agent.py`): model drives `search_papers` / `fetch_arxiv` / `get_paper_details` / `get_full_text`, with `AGENT_MAX_ITERATIONS` / `AGENT_MAX_TOOL_CALLS` / `AGENT_MAX_SEARCH_CALLS` budgets, discovery-tool withdrawal, and robust forced final synthesis.
- **Full-text evidence tool** (`src/ingestion/full_text.py`, `get_full_text`): downloads paper PDFs and extracts body text in-process (pypdf), bounded by `FULL_TEXT_CHAR_BUDGET`. Deployable, no external paper service. Real arXiv fetch verified; 3 unit tests. (Live: small local model doesn't yet choose to call it — see caveats above.)
- **Pluggable LLM backend** with full tool-use parity: Anthropic and OpenAI-compatible (`OPENAI_BASE_URL` covers OpenAI/local/OpenRouter/codex/opencode), via one canonical tool-use layer.
- Measured token usage/cost from provider responses (not estimated) on the live path.
- Langfuse tracing spans for every turn and tool call (no-ops without keys).
- Deterministic offline fallback memo so local runs and CI work without keys/network.
- Tests: 32 passing with one Starlette/httpx deprecation warning, ruff clean. Includes faked-SDK tests for both providers' wire translation and `tool_choice`, plus full-text evidence-gate and degraded-fallback regressions.
- Container setup: `Dockerfile` (uv, frozen lock) and `docker-compose.yml` (API + Qdrant v1.9.2 with a volume).
- Docs: `README.md`, `AGENTS.md` (source of truth), `CLAUDE.md` (defers to AGENTS.md), `.env.example`.

## In progress / partial

- Eval harness exists (`evals/run_eval.py`, benchmark fixtures) but no committed run report yet; `evals/reports/` is created on first run.
- Static UI (`static/index.html`) now provides a local workflow console for health, ingest, search preview, streaming agent runs, event timeline, evidence, diagnostics, and final memo. It is still a single-file prototype and should be treated as an observability surface, not a production UI.
- arXiv date-filter failures are handled as JSON at the API boundary and the UI can retry ingest without the date range. The underlying arXiv API remains flaky for some filtered queries, especially date ranges close to the current date.
- DeepSeek/local-model behaviour is improved but still needs live revalidation with the new full-text gate and degraded/error event path.

## Not started / gaps before "deployable"

- **No persisted run records** for replay/debugging. Once an SSE run finishes, the event stream is only in the browser.
- **No Fly.io manifest** (`fly.toml`) despite Fly being the stated target. No deployment performed.
- **No CI workflow** (`.github/workflows/`) to run pytest/ruff on push.
- **Docker image not yet built/run** this session — compose is defined but unverified end-to-end.
- **`.env` must be created** (`cp .env.example .env`); `docker-compose.yml` requires it via `env_file`.
- **Cold start is slow**: SPECTER (~440MB) downloads on the first ingest/search/brief (observed ~90s). For production, bake the model into the image or mount a cache volume.
- Real agent briefs require `ANTHROPIC_API_KEY` or `OPENAI_API_KEY`; otherwise only the fallback memo is produced.

## How to test it

Quickest, no key, no Docker (in-memory store + fallback memo; first call downloads SPECTER):

```bash
uv sync --dev
uv run uvicorn src.api.main:app --port 8000 &
curl -s http://localhost:8000/health
curl -N -X POST http://localhost:8000/briefs/stream \
  -H 'content-type: application/json' \
  -d '{"research_question":"How can retrieval improve scientific literature review?","max_papers":3}'
```

Real agent loop (set a key first), with a curated corpus:

```bash
cp .env.example .env   # add ANTHROPIC_API_KEY (or set LLM_PROVIDER=openai + OPENAI_API_KEY)
uv run uvicorn src.api.main:app --port 8000 &
curl -X POST http://localhost:8000/ingest \
  -H 'content-type: application/json' -d '{"query":"cat:cs.LG","max_papers":25}'
curl -N -X POST http://localhost:8000/briefs/stream \
  -H 'content-type: application/json' \
  -d '{"research_question":"What methods are promising for retrieval-augmented scientific review?","domain":"cs.LG","max_papers":6}'
```

Full persistent stack (API + Qdrant): `cp .env.example .env && docker compose up --build`, then open `http://localhost:8000`.

## Suggested next steps

1. Harden tool-call validation: parse each tool's arguments with typed validation, clamp numeric bounds, return structured tool errors, and add regression tests for malformed model calls.
2. Classify full-text failures: distinguish timeout, HTTP status, PDF parse failure, empty extraction, and unknown ids; show those counts in diagnostics/UI.
3. Standardize SSE event semantics (`started`, `tool_call`, `tool_result`, `evidence_required`, `warning`, `degraded`, `error`, `final`) and surface full-text coverage, fallback state, and budget-forced synthesis in the UI.
4. Make vector-store fallback environment-aware: local can fall back to memory; non-local should fail clearly or expose a health warning that persistence is unavailable.
5. Persist run transcripts and diagnostics so brittle model behaviour can be replayed, evaluated, and compared across model/provider changes.
6. Add CI running `uv run pytest -q` and `uv run ruff check .`.
7. Build and smoke-test the Docker Compose stack against live Qdrant.
8. Add `fly.toml` + a documented deploy path; bake or cache the SPECTER model to fix cold start.
