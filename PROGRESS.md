# Project Progress

Living status tracker for Research Brief Agent. Update this as work lands.

_Last updated: 2026-07-21_

## Live product review (2026-07-21) — retrieval usefulness gap

First real hands-on browser test of the end-to-end product on live
`deepseek-v4-flash` with the in-memory store. **The plumbing is robust; retrieval
relevance on a cold corpus is not, and that is what makes the product feel
un-useful today.** Findings, with run evidence:

- **The pipeline works end-to-end.** Question "What quantization method (GPTQ, AWQ,
  QLoRA) best preserves accuracy for on-device LLM serving?" (max_papers 20, run
  `ae848de7b12141a095c988b48c2b295e`) completed cleanly: 9 LLM turns, 14 tool calls,
  3/3 full-text PDFs read, 5 citations, `status=completed`, ~39s, ~$0.33 estimated.
  A well-structured memo was produced and returned in `final.data.final_brief`.
- **But the memo was an honest refusal, not an answer.** It opened with "No
  recommendation can be made from the retrieved evidence" — because all five cited
  papers were about Retrieval-Augmented Generation, not quantization. The
  strict-grounding contract behaved correctly (it refused rather than bluffing from
  parametric memory); the problem is upstream in what was retrieved.
- **Root cause: cold, topically-wrong corpus + wrong-tool bias.**
  1. The in-memory store held 25 papers, all RAG-related, left over from an earlier
     query in the same process; it contained **zero** quantization papers.
  2. `search_papers` only queries the local store — it does **not** backfill from
     arXiv. Only the separate `fetch_arxiv` tool does.
  3. The model called `search_papers` 12+ times (many redundant) and **never once
     called `fetch_arxiv`**, so it never pulled quantization papers from arXiv, then
     exhausted the tool-call budget.
  4. The off-topic RAG papers scored ~0.71–0.73, clearing `RETRIEVAL_MIN_SCORE`, so
     nothing flagged them as irrelevant.
  - A second run (`f5126853764f4fedb517ffdbac088823`) shows the same failure mode
    from the empty side: `retrieved=0`, `thin_evidence`, `iteration_budget_reached`,
    ~102s, `cited_papers=0`.
- **Suspected embedding bug (unverified).** Every query embed logs
  `Could not identify valid prediction head(s) from setup 'Stack[adhoc_query]'`. This
  may mean the SPECTER2 adhoc-query adapter is **not actually activating** on the
  query side, which would degrade all retrieval relevance and could be contributing
  to the mediocre ~0.71 scores. Needs direct verification (confirm the adapter is
  applied and the query embedding differs from the base-adapter embedding).
- **Minor UX fix landed (uncommitted).** The console's "Domain" field defaulted to
  `cs.LG`, which read as a required arXiv-category picker; in the live agent path
  `domain` is only injected into the prompt as free text (the category-filter code
  path, `ResearchTools.arxiv_metadata_search`, has no live callers). Changed
  `static/index.html` to an optional field with a plain-English placeholder. Markup
  only; backend untouched.

### P0 retrieval usefulness — A + B landed 2026-07-21

- [x] **A. Auto-backfill in the agent's `search_papers`.** `ResearchToolset` now
  backfills from arXiv when a `search_papers` call finds no local paper at/above
  `AGENT_SEARCH_BACKFILL_MIN_SCORE` (empty or weak result), indexes the fetched
  papers, and re-runs the search once. Deduped per query per run; master toggle
  `AGENT_SEARCH_AUTO_BACKFILL` (default true) + the score floor
  (`AGENT_SEARCH_BACKFILL_MIN_SCORE`, default 0.75) in `Settings`. Five focused
  tests in `tests/test_retrieval_quality.py` cover empty-corpus backfill, the
  strong-hit skip, the off-topic-hit trigger, the toggle, and per-query dedup.
  Hermetic: every test/eval path injects a non-network arXiv client, and the
  backfill fetch is wrapped so arXiv flakiness never surfaces as a tool error.
- [x] **B. Warm standing corpus seeded and wired.** Ran `scripts/seed_corpus.py`
  into embedded on-disk Qdrant: **320 papers across all 16 benchmark topics**
  (incl. 20 quantization papers), 3.1 MB at `.local/qdrant-corpus`. Verified the
  quantization query now returns 5/6 on-topic quantization papers at the top
  (previously zero). **Fixed a real persistence bug found while doing this:** the
  seeder set `QDRANT_PATH` but not `VECTOR_STORE_BACKEND`, so a local `.env` with
  `VECTOR_STORE_BACKEND=memory` was loaded and the 320 papers went into an
  in-memory store discarded at exit (disk stayed empty). The seeder now pins the
  backend and calls a new `PaperVectorStore.close()` so embedded Qdrant actually
  flushes to disk (it persists on close, not on interpreter-exit finalization).
  Point the app at the corpus with `QDRANT_PATH=.local/qdrant-corpus
  VECTOR_STORE_BACKEND=qdrant`; the live dashboard is currently running against it
  (`/health`: backend=qdrant, papers_indexed=320, fallback=false).

**Verification:** `uv run pytest -q` → 145 passed (was 140; +5 backfill tests);
`ruff check` and `ruff format --check` clean.

### Still open — retrieval

- [ ] **C. Prompt/budget tuning** so the model reaches for `fetch_arxiv` early and
  stops spamming redundant `search_papers`. Band-aid on top of A; lower priority now
  that A+B are in.
- [ ] **Verify/fix the SPECTER2 adapter activation** — the "adapters available but
  none are activated" / "Could not identify valid prediction head(s)" warnings fire
  on both document and query embeds, so scores are compressed (on- and off-topic
  both ~0.71) and only ranking is reliable. This is why the 0.75 backfill floor
  currently behaves as "always backfill". Fixing the adapter is the prerequisite for
  making the score floor discriminative; retune the floor afterward.
- [ ] **Live end-to-end re-test** of the quantization question on the warm corpus to
  confirm a genuinely grounded memo (backend proven; the paid live run was not
  repeated this session).

## Current review (2026-07-19)

- **End-to-end review + hygiene pass.** Re-verified the suite (140 passing on
  Python 3.11, 3.12, 3.13), found and fixed a set of small but real issues.
- **Local `.env` config drift fixed.** `.env` had drifted to the known-bad
  under-provisioned config (`deepseek-chat`, `AGENT_MAX_ITERATIONS=6`) plus
  undocumented drift: `EMBEDDING_MODEL=sentence-transformers/allenai-specter`
  (SPECTER1 base with SPECTER2 adapters) and `LLM_MAX_TOKENS=3000`. All four now
  match the validated `.env.example` defaults.
- **Embedding config validation added and review-hardened.** `TextEmbedder`
  rejects the known SPECTER1-base/canonical-SPECTER2-adapter mismatch that
  caused the local drift. Validation is deliberately conservative for unknown
  repository names and local paths, avoiding false compatibility claims and
  preserving valid renamed/fine-tuned adapters. Six focused tests cover the
  accepted and rejected cases.
- **Dependency truth.** `requests` and `pydantic` are now declared direct
  dependencies (previously resolved only transitively). `pandas` remains direct
  because the public `papers_to_dataframe` compatibility helper is retained.
  Lock regenerated.
- **Format gate in CI.** `ruff format --check` was failing on three files and CI
  didn't run it; files reformatted and the check is now a CI step. CI also runs
  pytest with `--cov=src` (informational, no threshold yet).
- **Dead surface removed without breaking compatibility.** Deleted
  `config/categories.yaml` (unreferenced, but previously copied into the Docker
  image) and legacy `scripts/run_arxiv_search.py`. The root `app.py` ASGI shim,
  `src.vector_store.VectorStore` alias/export, and public `ArxivClient` helpers
  remain for existing consumers, with regression tests. The stale
  Streamlit-era `.devcontainer` now runs uvicorn instead. Ruff `target-version`
  is `py311`, matching the supported floor.
- **Superseded eval report removed.** `evals/reports/deepseek-v4-flash-fixture.*`
  (the 3-case run of 2026-07-13) is deleted; `latest.*` (7-case, 2026-07-15)
  remains the release evidence of record. This resolves the open
  commit-or-regenerate decision.
- **New coverage.** Direct tests for `CrossEncoderReranker` (faked
  tokenizer/model), `Tracer` (faked `langfuse` module: trace/span creation,
  failure resilience, no-key no-op), `Settings` env-parsing helpers, embedding
  compatibility, and retained legacy imports/helpers. Suite grew 114 → 140.
- **Review findings closed.** Reconciled `NEXT_PHASE.md` and the HTML project
  brief with the seven-case `evals/reports/latest.*` evidence and repaired the
  live-eval link.
- **Still open:** the clean-checkout release rebuild is blocked in this
  environment (no Docker daemon); the P2 operator-console items, the optional
  hosted demo, and the Qdrant 1.9→1.18 data-migration note are unchanged.
- **Verification:** `uv run pytest -q` 140 passed on 3.11/3.12/3.13;
  `ruff check` and `ruff format --check` clean; offline core quality gate
  passing (run with `--jsonl`/`--markdown` redirected so `latest.*` is not
  clobbered — the eval runner's default output paths overwrite the release
  evidence).

## Current review (2026-07-15)

- **Deployment posture reframed.** The Fly smoke was demoted from a P0 release gate
  to an optional P2 portfolio affordance. The Docker Compose smoke already proves
  every architecturally risky surface; a Fly deploy reruns the same container and
  validates nothing new, while a public LLM agent endpoint is a standing cost/abuse
  liability. The release artifact of record is the local live-provider evidence below.
- **Live release-evidence capture surfaced a real source-grounding gap — now fixed.**
  Running the fixture-corpus eval on live DeepSeek at the full 7-core-case scale (not
  the earlier 3-case sample) exposed that a capable model cites famous papers from
  parametric memory (AdamW, GPTQ, QLoRA, even the Transformer paper) that were never
  retrieved or read. Grounding dropped to 74% with a 26% "hallucination" rate. These
  were real papers, not fabricated ids, but citing unretrieved sources violates the
  product's core contract, so it blocked the evidence freeze. Note: the earlier
  "3/3 ok, 100% grounding" claim did not generalize past 3 cases.
- **Strict-grounding enforcement landed** (two layers): the system prompt now forbids
  citing any paper the tools did not return, naming the memory-citation failure mode
  explicitly; and `ResearchToolset.filter_ungrounded_citations` deterministically
  strips any inline `[id]` that does not resolve to a retrieved paper, emitting an
  `ungrounded_citations_removed` warning. New tests cover both the end-to-end strip +
  warning and the filter's version-tolerance / non-citation-bracket handling.
- **Verified on the live path:** re-running the 7-core-case fixture eval on
  `deepseek-v4-flash` (documented default config: `AGENT_MAX_ITERATIONS=8`,
  `AGENT_MAX_SEARCH_CALLS=3`, 3-paper full-text budget) now returns **7/7 `ok`, 0%
  hallucination, 100% citation grounding, 100% full-text success, no fallbacks or
  warnings**. The prompt change alone eliminated memory citations this run; the filter
  is the tested safety net. Report saved to `evals/reports/latest.{jsonl,md}`.
  Measured tokens are the hard evidence; the per-case dollar figure is a configured
  estimate at default rates, not DeepSeek's real (much lower) pricing.
- **Config drift found:** the local `.env` had drifted to `deepseek-chat` with
  `AGENT_MAX_ITERATIONS=6`, which under-provisions the loop and produced 4/7 canned
  fallbacks. The documented default in `.env.example` (`deepseek-v4-flash`, iters 8)
  is the validated config; the eval above uses it via explicit env overrides.
- **Runtime support contract closed.** `requires-python` is now bounded to
  `>=3.11,<3.14` (the open-ended `>=3.10` let `uv` pick the system Python 3.14 and
  break test collection); the lock was regenerated. CI runs a 3.11/3.12/3.13 matrix,
  the classifiers and AGENTS.md document the same range, and all three were verified
  locally at 114 passing tests. Docker stays pinned to 3.12.
- **Verification:** `uv run pytest -q` passes with **114 tests** on 3.11, 3.12, and
  3.13; `uv run ruff check .` is clean.

## Current review (2026-07-13)

- **Verification:** `uv run pytest -q` passes with **112 tests** and one
  Starlette/httpx deprecation warning on Python 3.12; `uv run ruff check .` is clean.
- **Release posture:** the service is now a packaged local alpha rather than only a
  source-level demo. The production-mode Docker path passed auth, readiness, ingest,
  multi-tool brief, run-record/log, and restart-persistence checks. A clean-checkout
  build and public Fly smoke are still required before a production claim.
- **Eval quality gate restored:** the offline corpus now contains one explicit
  synthetic relevant paper per benchmark case and uses deterministic lexical
  feature-hash embeddings rather than whole-text hash vectors. Both the seven-case
  core run and all 16 cases pass with a relevant hit at rank one, 100% citation-ID
  grounding, no warnings, and no hallucinated ids. CI runs the core gate. The 33%
  recall is expected because the small fixture represents one of each case's three
  live-corpus relevance ids; full-text remains a live-agent metric.
- **Reporting drift:** `README.md`, older sections of this file, `NEXT_PHASE.md`, and
  `evals/reports/latest.*` report historical test totals or superseded eval state. The
  named DeepSeek report is still the strongest live-model evidence; the release-facing
  status and `latest.*` report need one deliberate refresh.
- **Container compatibility:** `qdrant-client` 1.18 failed against the previously
  pinned Qdrant 1.9.2 query API. The image and both Compose files now pin Qdrant
  1.18.0. A direct 1.9.2-volume-to-1.18.0 startup also failed, so existing 1.9 data
  requires a supported snapshot/export migration rather than an in-place version
  jump.
- **Runtime metadata:** Docker targets Python 3.12 and CI targets 3.13, while package
  metadata declares an open-ended `>=3.10`. Local verification initially selected
  Python 3.14 and could not collect tests until the environment was synchronized on
  3.12. Pin the contributor runtime or add a tested-version matrix/upper bound.
- **Review artifact:** [docs/project-brief.html](docs/project-brief.html) is a
  self-contained snapshot of product maturity, evidence, risks, and recommended next
  steps from this review.

## Current implementation checklist

This is the active, ordered checklist. Update it in the same change that completes an
item. **Active next: capture the local live-provider release evidence and freeze it as
the artifact of record.** A hosted (Fly) demo is now an optional portfolio affordance,
not a release gate — see the rationale in "Deployment posture" below.

### P0 — Evaluation integrity

- [x] Give every benchmark case at least one relevant paper in the offline fixture.
- [x] Replace random whole-text fixture vectors with deterministic lexical embeddings.
- [x] Add a CLI quality gate for warnings, citation validity, uncertainty, and
  zero-hit retrieval.
- [x] Add focused regression tests for fixture coverage, lexical similarity, and
  actionable gate failures.
- [x] Run the seven-case core gate in CI without keys, network, or model downloads.

### P0 — Packaged release smoke

- [x] Build the current locked working tree as a production image and record its size.
- [x] Start API + Qdrant and wait for readiness.
- [x] Ingest a tiny corpus and stream one authenticated brief.
- [x] Confirm JSONL run records and structured summary logs for the same run.
- [x] Restart the stack and verify Qdrant corpus persistence.
- [ ] Rebuild the committed release candidate from a clean checkout or CI runner.
  (Blocked 2026-07-19: no Docker daemon in this environment.)

### Deployment posture (2026-07-15)

The Docker Compose smoke already proves everything architecturally risky: the
multi-tool agent loop, API-key auth, per-IP limiting, Qdrant persistence across
restart, run-record IO, and the CPU-only image booting. A Fly deployment reruns the
same container against a different scheduler and validates nothing new about the
system, so it is **no longer a P0 release gate**. Its only distinct value is a
click-through demo URL, which for this app is an ongoing cost/abuse liability (a public
LLM agent spends real tokens per request). It is therefore demoted to an optional P2
portfolio affordance; the release artifact of record is the local live-provider
evidence below, which is more informative and free of a standing public endpoint.

### Packaged smoke evidence (2026-07-13)

- Built `research-brief-agent:release-smoke` from the frozen lock. Selecting the
  explicit Linux CPU PyTorch index reduced the image from **11.27 GB** to **2.06 GB**
  (about **82% smaller**) while keeping the SPECTER2 base model, document/query
  adapters, and Qdrant binary baked into the artifact. Final local digest:
  `sha256:dbfe3bfb7fa7afb57841315caf45503a1d8ed97bd95d567c67921fb1dab07521`.
- Started `docker-compose.smoke.yml` in `ENVIRONMENT=production` with Qdrant 1.18.0,
  required API-key auth, required JSONL records, structured logs, and a deterministic
  OpenAI-compatible tool-use provider. `/health` returned ready with the persistent
  Qdrant backend and two indexed papers; an unauthenticated brief returned 401.
- Final-source run `e9915f9e9f6844bf94d671953220300f` completed
  `search_papers` → `get_full_text` → synthesis, retrieved two papers, read one full
  PDF, cited that paper, used three provider turns and two tool calls, and returned no
  warnings. End-to-end latency was **10.52 s** on the first post-rebuild request
  (including **8.12 s** retrieval and lazy embedding-model initialization); measured mock-provider
  usage was 420 input / 228 output tokens with a configured **$0.00468** estimate.
- The matching `run_started`/`run_finished` JSONL entries and structured
  `brief_run_finished` log reported `status=completed`. Restarting API and Qdrant
  preserved both `papers_indexed=2` and the completed run record.
- Verification after the smoke: **112 tests passed**, Ruff clean, and the seven-case
  offline core quality gate passed with rank-one relevant hits, 100% citation-ID
  grounding, no hallucinated ids, and no warnings.

### P1 — Release evidence of record (active next)

- [x] Capture one live-provider run against the fixture corpus (DeepSeek) with measured
  tokens, estimated cost, latency, and quality metrics; save it as the release-facing
  `latest.{jsonl,md}` artifact. Done 2026-07-15: 7/7 `ok`, 0% hallucination, 100%
  grounding on `deepseek-v4-flash` after the strict-grounding fix.
- [x] Refresh `README.md`, `NEXT_PHASE.md`, and stale report prose from that one
  verified snapshot. Done 2026-07-15: README now states 114 tests on Python
  3.11–3.13, documents strict citation grounding, and reframes Fly as an optional
  demo; NEXT_PHASE.md carries a superseded banner pointing here.
- [ ] (Optional) Add Langfuse keys and capture a trace + run-record ids for the same
  run. Deferred: no Langfuse project is configured, so the tracer no-ops; measured
  usage/cost/latency and the JSONL run record already stand as evidence without it.

### P1 — Runtime support contract

- [x] Bound `requires-python` to `>=3.11,<3.14` (the open-ended `>=3.10` let `uv`
  select the system Python 3.14 and break test collection); relocked. Done 2026-07-15.
- [x] Test the supported Python versions (3.11, 3.12, 3.13) in a CI matrix and document
  the matrix in AGENTS.md. Verified locally: 114 tests pass on all three. Done
  2026-07-15.

### P2 — Operator experience

- [ ] Expose provider/model and active agent budgets in the console.
- [ ] Add retry controls and a persisted run-history view.

### P2 — Release engineering

- [ ] Move the Dockerfile's `README.md` copy after the dependency/model bake so a
  documentation-only change does not invalidate the expensive SPECTER2 cache layer.

### P2 — Optional hosted demo (was P0 Fly smoke)

- [ ] If a click-through demo URL is wanted, deploy the existing `fly.toml` (or a
  scale-to-zero host), keep the API-key + per-IP limiter on, and add a provider spend
  cap. Treat it as a demo affordance, not a reliability gate.

## Status at a glance

The core product works end-to-end and is **runnable locally today**. It is scoped to a cited recommendation memo for AI/ML and scientific-ML engineering decisions; the evidence backend is arXiv/SPECTER/Qdrant only (no plan to add other source backends). Direct `uvicorn` smoke testing works without Docker, and the production-mode Docker Compose path now has a recorded two-paper ingest, authenticated multi-tool brief, full-text read, structured logs, required run records, and API/Qdrant restart with persistence. The final local image is 2.06 GB and uses CPU-only PyTorch. Live DeepSeek `deepseek-v4-flash` has been validated with the fixture corpus at the full 7-core-case scale: 7/7 `ok`, 0% hallucination, 100% citation grounding, no fallbacks or warnings. The Fly.io deployment manifest and image path are present, but the app has not been deployed from this checkout. The public deployment path runs the real agent loop with an LLM key, gated by `X-API-Key` and a tight per-IP limiter. Normal live synthesis is gated on successful full-text evidence when retrieved papers exist; if the run cannot meet that contract before budget/failure, the stream emits explicit degraded/error events instead of silently looking successful.

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
- **Deployment smoke status:** the production-mode local Compose smoke passed on
  2026-07-13. Fly remains unverified because neither `fly` nor `flyctl` is installed
  in this environment.
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

1. ~~Live eval report commit decision~~ — resolved 2026-07-19: the superseded 3-case `deepseek-v4-flash-fixture.*` report was deleted; `evals/reports/latest.{jsonl,md}` (7-case, 2026-07-15) is the release evidence of record.
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
- Container setup: `Dockerfile` (uv, frozen lock, explicit CPU-only Linux PyTorch,
  baked SPECTER cache, Fly Qdrant 1.18.0 sidecar binary), `docker-compose.yml` (API +
  Qdrant 1.18.0 with a volume), and a deterministic production-mode smoke stack.
- Packaged local release smoke: authenticated search → full-text → cited synthesis,
  required run records, structured completion log, and restart persistence all passed.
- Fly deployment manifest: `fly.toml` with a Qdrant volume mount, readiness check, API-key/rate-limit env, and documented secret provisioning commands.
- Docs: `README.md`, `AGENTS.md` (source of truth), `CLAUDE.md` (defers to AGENTS.md), `.env.example`.

## In progress / partial

- Eval harness exists (`evals/run_eval.py`, benchmark fixtures); the release evidence of record is `evals/reports/latest.{jsonl,md}` (7-case live DeepSeek fixture run, 2026-07-15). Note the runner's default output paths overwrite `latest.*` — pass `--jsonl`/`--markdown` for any run that should not replace the release evidence.
- `evals/run_eval.py --fixture-corpus` runs the configured live LLM against deterministic fixture papers and synthetic full text, so provider cost can be measured without external retrieval dependencies.
- Direct uvicorn smoke testing works in local fallback mode and persistent Qdrant
  mode. The packaged production-mode path is also verified locally; only the public
  Fly path and clean-checkout reproducibility remain open.
- Static UI (`static/index.html`) provides an ask-first console: a single technical decision-question form, masthead health readouts, a streaming reasoning-spine timeline, telemetry gauges, evidence/diagnostics tabs, and the final memo. Manual ingest and search preview were removed from the page (the agent backfills arXiv itself); both endpoints stay available via the API. It is still a single-file prototype and should be treated as an observability surface, not a production UI.
- arXiv date-filter failures are surfaced as JSON at the API boundary. The UI-side retry-without-date-range fallback was removed along with the ingest form, so programmatic `/ingest` callers handle this themselves. The underlying arXiv API remains flaky for some filtered queries, especially date ranges close to the current date.
- DeepSeek V4 live behaviour is improved and revalidated; token use is now measured in the fixture report, but real-corpus costs still need a production budget.

## Not started / gaps before "deployable"

- The public Fly path has not been deployed or smoked from this checkout.
- The local packaged smoke used a deterministic OpenAI-compatible provider; a release
  smoke still needs one real provider plus Langfuse trace evidence.
- Existing Qdrant 1.9.x data cannot be mounted directly into 1.18.0; define and test a
  supported snapshot/export migration before upgrading an installation with data.

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

Full persistent Docker stack: `cp .env.example .env && docker compose up --build`,
then open `http://localhost:8000`. The deterministic release-smoke stack is documented
in `README.md` and runs on `http://127.0.0.1:8767`.

## Suggested next steps

1. Commit the release candidate, rebuild it from a clean checkout or CI runner, and
   record the image digest. (Blocked in the current local environment: no Docker
   daemon; attempted 2026-07-19.)
2. Install Fly CLI and run the production smoke: deploy, hit `/health`, ingest a tiny
   corpus, stream one brief, restart the machine, and confirm Qdrant persistence plus
   run-record/log output.
3. Capture a real-provider Langfuse trace, measured usage/cost, JSONL record, and
   structured completion log for the same public run.
