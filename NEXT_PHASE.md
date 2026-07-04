# Next Phase: Research Brief Agent Demo

Planned work for the next phase, framed around a single goal: make this project read as
a production-grade research briefing system for AI/ML and scientific-ML engineering
decisions. A live, verifiable URL plus the
engineering that makes a public deploy responsible is worth more than any additional
feature.

Status when written: core works locally, tests pass (51), CI runs ruff + pytest, **not yet
deployed**. See [PROGRESS.md](PROGRESS.md) for full current state.

Current status: Tier 1 implementation is present in this checkout but not deployed from
it yet. Direct uvicorn smoke testing works in local fallback mode and persistent Qdrant mode;
a 3-paper Qdrant ingest/search/brief survived a Qdrant+uvicorn restart. The live DeepSeek
`deepseek-v4-flash` fixture eval now passes 3/3 cases with real memo output, no warnings,
no deterministic fallback, and full-text success for every attempted paper.

Organizing principle: a recruiter can verify a live URL in 15 seconds; an interviewer will
drill into how it's deployed, secured, and operated. Optimize for that.

---

## Tier 1 — Ship a live demo, safely (CV-gating bundle)

These ship together. A public URL is not responsible without all four, and each is a real
production decision, not theater. This unblocks the strongest CV edit (replacing the old
Streamlit project with a deployed Research Brief Agent).

1. **Deploy to Fly (`fly.toml` + cold-start fix).** The headline. Bake SPECTER (~440MB) into
   the Docker image so first request doesn't eat a ~90s cold download; provision a Fly volume
   for Qdrant. Talking point: moving a large model artifact into an image layer to cut cold
   start (build caching, startup latency).

2. **Protect the endpoint: API-key auth + rate limiting.** Not optional once public. The
   service calls a paid LLM; an unauthenticated public endpoint is a key-draining liability.
   Add an API-key header check and per-IP rate limit. Strong production-judgment signal that
   most portfolio projects skip.

3. **Environment-aware vector-store fallback.** Implemented: Qdrant failure only falls back
   to in-memory in `ENVIRONMENT=local`; non-local startup fails loud. `/health` reports the
   fallback state so persistence loss is visible during local development.

4. **Make `/health` a real readiness check.** Report dependency state (vector store reachable,
   provider configured). Ties into #3; what a load balancer or operator actually relies on.

**Outcome:** a live, defensible URL. This is the item gating the strongest CV edit.

**Deploy-mode decision (resolve before starting):** run the public instance as the real agent
(needs an LLM key on Fly — hence auth + rate limiting come first) or in keyless fallback mode
(zero-cost, unabusable demo). Recommendation: real agent behind an API key with a tight rate
limit, since the agent loop is the whole point.

---

## Tier 2 — Maturity an interviewer will probe

5. **Persistent local Qdrant smoke test.** Done without Docker using the official macOS arm64
   Qdrant binary, repo-local storage, a 3-paper ingest, search, fallback brief generation, and
   a restart persistence check.

6. **Cost/transcript trimming, with a qualified eval report.** Done for the fixture path:
   transcript compaction is implemented, DeepSeek V4 flash thinking mode is disabled for
   normal assistant content/tool calls, and `evals/reports/deepseek-v4-flash-fixture.*`
   records a 3-case live fixture run. Remaining work is to decide whether to commit the
   named report or regenerate `latest.*` for release.

7. **Formalize the SSE event contract.** Done: `src/api/sse.py` defines stable event
   types and required fields, `/briefs/stream` validates events before writing them,
   the README documents the contract, and tests parse real stream output against it.

8. **Structured logging.** Request IDs, tool-call timing, per-run token counts as JSON to stdout.
   Operability is half of what "production" means in 2026; small change.

---

## Tier 3 — Depth and narrative

9. **Persist run transcripts** so runs can be replayed and compared across models/providers.

10. **Classify full-text failures.** Implemented: timeout / HTTP / network / parse /
   empty-text / bad-id / unknown categories are surfaced in tool payloads, degraded
   events, final diagnostics, and the static UI.

11. **Rewrite the README to tell the production story.** Architecture, failure modes and how
    they're handled, cost per run, the live URL, the deploy path. Capstone: the engineering
    judgment is invisible unless the front page says it out loud. This is what a reviewer reads.

---

## Recommended sequence

Ship Tier 1 as one deploy push and get the live URL before touching Tier 2 — the job goal points
to "have a verifiable live system." Tier 2 makes it survive a technical deep-dive. Tier 3 is the
finish, with the README rewrite as the last step once everything it describes is true.
