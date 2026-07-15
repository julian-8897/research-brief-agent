# Next Phase: Research Brief Agent Demo

> **Superseded (2026-07-15) — kept as a historical planning record.**
> Most of this plan has shipped. Two premises here are now out of date, so read
> [PROGRESS.md](PROGRESS.md) as the live source of truth, not this file:
> 1. **The Fly deploy is no longer the headline / CV-gating item.** The Docker
>    Compose smoke proves every architecturally risky path, so a hosted deploy was
>    reframed from a P0 gate to an optional demo affordance. The release artifact of
>    record is the local live-provider eval, not a public URL. See the "Deployment
>    posture" note in PROGRESS.md.
> 2. **Status numbers below are stale** (they read "51 tests" / "3/3 fixture cases").
>    Current: 114 tests pass on Python 3.11–3.13; the live fixture eval runs 7 core
>    cases at 0% hallucination and 100% citation grounding after strict-grounding
>    enforcement landed.
>
> The tier structure below records the reasoning at the time and which items were
> delivered; it is no longer the forward plan.

Planned work for the next phase, framed around a single goal: make this project read as
a production-grade research briefing system for AI/ML and scientific-ML engineering
decisions.

Original organizing principle: a recruiter can verify a live URL in 15 seconds; an
interviewer will drill into how it's deployed, secured, and operated. That framing has
since been tempered — a verifiable, well-tested system with measured evidence carries the
same signal without a standing public LLM endpoint to babysit.

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
