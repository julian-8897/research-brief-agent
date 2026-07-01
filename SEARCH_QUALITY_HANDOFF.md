# Handoff: search quality follow-ups (B + C + reranker)

Context: `/papers/search` now has query expansion (`src/agent/query_expansion.py`)
and arXiv coverage backfill (`ResearchTools.semantic_search` in `src/agent/tools.py`).
Live testing showed three remaining gaps. Do these three tasks in order. They are
independent enough to land as separate commits.

Constraints (all tasks):
- Match existing style. Keep model loads lazy; no network/model downloads in tests.
- `uv run pytest` and `uv run ruff check .` must pass. Add/adjust tests for each change.
- Do not commit; leave changes in the working tree for review.
- Do not revert unrelated pre-existing working-tree changes.

---

## Task B1 — make backfill cheap on repeat searches (id dedup)

Problem: `semantic_search` backfill calls `fetch_and_ingest` → `ingest_papers`
(`src/agent/tools.py`), which re-embeds and re-upserts **every** fetched paper on
every search. SPECTER2 CPU embedding of ~25 docs dominates latency (~35s observed),
even when those papers are already indexed.

Change:
- Add `existing_ids(ids: list[str]) -> set[str]` to the `PaperVectorStore` interface
  (`src/retrieval/store.py`). Implement for `InMemoryVectorStore` (check `_papers`)
  and `QdrantPaperVectorStore` (use `client.retrieve(..., with_payload=False)` or a
  filtered scroll by point id).
- In `ingest_papers` (or a new `ingest_new_papers`), skip papers whose id is already
  present before embedding, so only genuinely new papers get embedded/upserted.
  The backfill path in `semantic_search` should use this; keep the model-facing
  `fetch_arxiv` tool behavior unchanged unless trivially shared.

Acceptance: a second identical search re-embeds 0 papers (assert via a counting
embedder in a unit test); results are unchanged.

## Task B2 — flip backfill default to off

Problem: `SEARCH_AUTO_BACKFILL=true` makes every raw search hit arXiv. Even with B1,
the first search for a topic is slow; a search box should be fast by default.

Change:
- Default `search_auto_backfill` to `False` in `src/settings.py` and `.env.example`.
- Keep the `?backfill=true` query param and settings override working (already wired
  in `src/api/main.py` / `semantic_search`).
- Update AGENTS.md wording (currently says backfill is the default coverage behavior).

Acceptance: default search does not call arXiv (assert arxiv client call count 0);
`?backfill=true` still fetches. Existing `test_search_backfills_*` tests in
`tests/test_api.py` may need their client built with `search_auto_backfill=True`.

## Task C — better arXiv backfill query

Problem: backfill fetches `all:<raw user query>` from arXiv. arXiv `all:` is
keyword/boolean, so bare "neural operators" returns token-match junk (Koopman
operator theory, sectorial operators). A keyword-rich query yields the canonical
papers (DeepONet, Fourier Neural Operator).

Change:
- Add an arXiv-query generator (extend `src/agent/query_expansion.py` or a sibling):
  given the user query + optional LLM provider, produce a keyword-expanded arXiv
  query string (related terms/methods, no long prose — arXiv is boolean, not
  semantic). Fall back to the raw query when no provider or on error.
- Use it for the backfill fetch in `semantic_search` (this is separate from the
  HyDE sentence used for the *embedding* query — do not reuse the prose sentence as
  the arXiv query).
- Gate with a setting (e.g. `SEARCH_BACKFILL_QUERY_EXPANSION`, default true).

Acceptance: unit test with a fake provider asserts the arXiv fetch receives the
expanded query, not the raw one; graceful fallback when no provider.

## Task D — cross-encoder reranker (the real precision lever)

Problem: SPECTER2 is a bi-encoder over abstracts; its cosine scores are compressed
(~0.76–0.85 for both relevant and irrelevant), so token-match false positives
(Koopman) rank near real hits. Expansion only nudges this. A cross-encoder that
jointly scores (query, abstract) reorders the candidate set precisely.

Change:
- New `src/rerank.py` with a `CrossEncoderReranker` (lazy load, like `TextEmbedder`).
  Suggested model: `cross-encoder/ms-marco-MiniLM-L-6-v2` (or a scientific-domain
  cross-encoder if one is preferred). Method `rerank(query, items, top_k) -> items`.
- In `vector_retrieve` (`src/agent/tools.py`): over-retrieve a larger candidate pool
  (e.g. `rerank_candidate_k`, default ~40, bounded by `max_retrieval_results`), then
  rerank down to the requested `k`. The reranker sees the query text (use the raw or
  expanded query; pick one and document it) against each paper's title+abstract.
- Settings: `RERANK_ENABLED` (default false until validated), `RERANK_MODEL`,
  `RERANK_CANDIDATE_K`. Add `sentence-transformers`/`torch` are already deps; confirm
  the cross-encoder loads via the installed stack.
- Wire a test double (fake reranker) into the existing test embedders/fakes so the
  suite stays hermetic.

Acceptance: with a fake reranker that reverses order, `vector_retrieve` returns the
reranked order; real-model path is exercised only in a manual/opt-in eval, not CI.

Validation (do this, report numbers): run
`uv run python evals/run_eval.py --fixture-corpus` before/after enabling rerank and
report the citation-grounding / retrieval metrics delta. The retrieval-eval metrics
live in `evals/metrics.py`. If a labeled retrieval set does not exist yet, add a
small one (a few queries with known-relevant arXiv ids) and recall@k / nDCG@k
functions — this is the measurement layer that tells you if rerank actually helps.

---

## Report back
For each task: files changed, the setting defaults, test results (exact `uv run pytest`
/ `uv run ruff check` output), and for Task D the before/after eval numbers. Flag any
deviations and anything you could not verify (e.g. real cross-encoder download).
