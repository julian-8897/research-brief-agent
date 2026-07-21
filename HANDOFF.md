# Handoff — Research Brief Agent

_Handoff snapshot, 2026-07-21. For the live source of truth read `AGENTS.md`
(conventions) and `PROGRESS.md` (status). This file is a cold-start summary._

## What this project is (1 paragraph)

A streaming FastAPI service that turns an AI/ML or scientific-ML **engineering
decision question** (e.g. "GPTQ vs AWQ vs QLoRA for on-device serving?") into a
**cited decision memo** (recommendation, evidence, tradeoffs, risks, uncertainty,
next actions) with measured token cost, latency, and a Langfuse trace. A
tool-using LLM agent drives its own retrieval over an arXiv corpus (SPECTER2
embeddings in Qdrant), reads full-text PDFs, then writes the memo. The niche/moat
is **enforced citation grounding**: the agent is blocked from writing until it has
read full text, and a deterministic post-filter strips any citation to a paper it
did not retrieve — so it cannot pad the memo with papers recalled from training.
Scope is deliberately narrow (arXiv only, engineering decisions only) so brief
quality is measurable.

## Architecture (where the logic lives)

- `src/api/main.py` — FastAPI, SSE `/briefs/stream`, `/ingest`, `/papers/search`, `/health`.
- `src/agent/brief_agent.py` — the agent loop (turn/tool budgets, evidence gate, forced synthesis, memo finalization).
- `src/agent/toolset.py` — the 4 model-callable tools (`search_papers`, `fetch_arxiv`, `get_paper_details`, `get_full_text`), per-run state, citation filtering + linkification, **auto-backfill**.
- `src/agent/tools.py` — stateless ops (vector retrieve, arXiv fetch+ingest).
- `src/llm/` — provider-neutral tool-use layer + Anthropic and OpenAI-compatible backends.
- `src/retrieval/store.py` — Qdrant / in-memory vector stores.
- `src/embeddings.py` — SPECTER2 asymmetric embedder (doc vs adhoc-query adapter).
- `src/settings.py` — frozen env-driven `Settings`.

## What was done this session (all committed on branch `retrieval-usefulness`)

Live testing exposed that against a thin/empty corpus the agent returned honest
but useless "no evidence" memos: `search_papers` only queries the local store and
the model never fell back to `fetch_arxiv`, so on-topic papers were never
retrieved. Fixes landed:

1. **Fix A — agent search auto-backfill** (`toolset.py`, `settings.py`). When a
   `search_papers` call finds no local paper at/above `AGENT_SEARCH_BACKFILL_MIN_SCORE`,
   the toolset fetches fresh arXiv papers for that query, indexes them, and
   re-runs the search once. Deduped per query per run. Toggle
   `AGENT_SEARCH_AUTO_BACKFILL` (default true). Best-effort: arXiv failures never
   surface as a tool error. 5 new tests.
2. **Fix B — warm standing corpus** (`scripts/seed_corpus.py`, `store.py`). Seeded
   ~320 papers across 16 benchmark topics into embedded on-disk Qdrant at
   `.local/qdrant-corpus`. Fixed a real persistence bug: the seeder set
   `QDRANT_PATH` but not `VECTOR_STORE_BACKEND`, so a local `.env` with
   `VECTOR_STORE_BACKEND=memory` sent the seed into an in-memory store discarded
   at exit. Added `PaperVectorStore.close()` (embedded Qdrant flushes on close,
   not on interpreter exit) and pinned the backend in the seeder.
3. **Clickable citations** (`toolset.py`, `brief_agent.py`). After the grounding
   filter, grounded inline `[id]` citations are rewritten to `[id](arxiv_url)` on
   both the live and fallback memo paths. The console renderer already turns
   markdown links into `<a>`. 2 new tests.
4. **Domain UX** (`static/index.html`). The console's Domain field was a
   misleading `cs.LG` default that read as a required arXiv category; now an
   optional field with a plain-English placeholder. (`domain` is only injected
   into the prompt as free text in the live path.)

**Verification:** `uv run pytest -q` → 147 passed; `ruff check` + `ruff format --check` clean.

## Git state

- Branch **`retrieval-usefulness`** pushed to `origin`. Two commits:
  `14c64ad` (prior 2026-07-19 review) and `2403b66` (today's work). Not yet merged
  to `main`; PR not opened.
- **Uncommitted/local-only (gitignored):** `.env` (secrets) and `.local/`
  (the seeded corpus + run records). The corpus is NOT in git — re-seed on a new
  machine with `uv run python scripts/seed_corpus.py`.

## Open items (priority order)

1. **[P0, blocking quality] SPECTER2 adapter activation bug.** Logs show
   "adapters available but none are activated" (document side) and "Could not
   identify valid prediction head(s) from setup 'Stack[adhoc_query]'" (query
   side) on every embed. The proximity/query adapters may not be applying, so
   similarity scores are compressed — on-topic and off-topic both ~0.71, only
   ranking is reliable. Consequence: the `AGENT_SEARCH_BACKFILL_MIN_SCORE=0.75`
   floor currently behaves as "always backfill." Fixing the adapter is the
   prerequisite for a discriminative floor; retune the floor afterward. Start in
   `src/embeddings.py` (adapter load/activation for `allenai/specter2_base` +
   `allenai/specter2` / `allenai/specter2_adhoc_query`).
2. **[P1] Live end-to-end re-test** of the quantization question on the warm
   corpus to confirm a genuinely grounded memo. Retrieval layer is proven (5/6
   top hits are on-topic quantization papers); the paid live LLM run was not
   repeated. Costs ~$0.30 DeepSeek per run.
3. **[P2] Prompt/budget tuning** so the model reaches for `fetch_arxiv` early and
   stops spamming redundant `search_papers` (band-aid now that A is in).
4. **[P2] Merge/PR** `retrieval-usefulness` → `main` once the above are settled.
5. Longer tail (see `PROGRESS.md`): operator-console depth, optional Fly demo,
   Qdrant 1.9→1.18 data-migration note.

## How to run / test

```bash
uv sync --dev
uv run pytest -q                       # 147 tests, hermetic (no keys/network)
uv run ruff check . && uv run ruff format --check .

# Run the app on the warm corpus (real agent needs a key in .env):
QDRANT_PATH=.local/qdrant-corpus VECTOR_STORE_BACKEND=qdrant \
  uv run uvicorn src.api.main:app --port 8000
# open http://localhost:8000 ; POST /briefs/stream is the core flow

# Re-seed the corpus (stop the app first — embedded Qdrant is single-process):
uv run python scripts/seed_corpus.py
```

Current `.env`: `LLM_PROVIDER=openai`, DeepSeek base URL, `deepseek-v4-flash`,
`OPENAI_API_KEY` set, `ANTHROPIC_API_KEY` empty.

## Critical gotchas

- **Tests are hermetic by contract.** `tests/conftest.py` sets `DISABLE_DOTENV=1`
  and clears provider keys before `src.settings` imports; `Settings` reads env at
  import time, so pass overrides to `Settings(...)`, not via env after import.
  Every test injects a non-network arXiv client, so auto-backfill never hits the
  network in tests.
- **Embedded Qdrant (`QDRANT_PATH`) is single-process** and persists only on
  `close()`. Don't run the seeder while the app holds the path.
- **The LLM is genuinely agentic** — don't reintroduce a hardcoded
  retrieve→synthesize pipeline. Tools are dispatched by the loop, chosen by the model.
- **Token usage is measured** from provider `TurnResult`, never estimated on the live path.
- **Provider parity is fragile** — any tool-use-layer change must be mirrored in
  both `anthropic_provider.py` and `openai_provider.py` and covered by
  `tests/test_llm_providers.py`.
- **Do not clobber `evals/reports/latest.*`** — it is the release evidence of
  record. Redirect eval output with `--jsonl`/`--markdown`.
- Supported Python 3.11–3.13; `uv lock` after any `requires-python` change.
