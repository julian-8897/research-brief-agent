# Next Phase: Production-Grade Hardening

Planned work for the next phase, framed around a single goal: make this project read as
production-grade in a technical screen. A live, verifiable URL plus the engineering that
makes a public deploy responsible is worth more than any additional feature.

Status when written: core works locally, tests pass (36), CI runs ruff + pytest, **not yet
deployed**. See [PROGRESS.md](PROGRESS.md) for full current state. Implementation deferred.

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

3. **Environment-aware vector-store fallback.** Today Qdrant failure silently downgrades to
   in-memory, so persistence can vanish with no signal. Gate auto-fallback to
   `ENVIRONMENT=local`; in prod, fail loud or expose it as a `/health` degradation.
   The canonical "thought past the happy path" answer; protects the deployed instance.

4. **Make `/health` a real readiness check.** Report dependency state (vector store reachable,
   provider configured). Ties into #3; what a load balancer or operator actually relies on.

**Outcome:** a live, defensible URL. This is the item gating the strongest CV edit.

**Deploy-mode decision (resolve before starting):** run the public instance as the real agent
(needs an LLM key on Fly — hence auth + rate limiting come first) or in keyless fallback mode
(zero-cost, unabusable demo). Recommendation: real agent behind an API key with a tight rate
limit, since the agent loop is the whole point.

---

## Tier 2 — Maturity an interviewer will probe

5. **Cost/transcript trimming, with a committed eval report.** The ~70k-token synthesis is real
   production economics. Summarize older tool results, stop re-sending full-text bodies once the
   evidence gate is met, then run `evals/run_eval.py` and commit the report (harness exists, no
   report committed yet). Converts "I care about cost" into a number for the CV and interview.

6. **Formalize the SSE event contract.** `evidence_required` / `degraded` / `error` are already
   emitted; document and test the stable event set. API-design maturity, cheap.

7. **Structured logging.** Request IDs, tool-call timing, per-run token counts as JSON to stdout.
   Operability is half of what "production" means in 2026; small change.

---

## Tier 3 — Depth and narrative

8. **Persist run transcripts** so runs can be replayed and compared across models/providers.

9. **Classify full-text failures** (timeout / HTTP / parse / empty / bad-id) with counts surfaced
   in diagnostics.

10. **Rewrite the README to tell the production story.** Architecture, failure modes and how
    they're handled, cost per run, the live URL, the deploy path. Capstone: the engineering
    judgment is invisible unless the front page says it out loud. This is what a reviewer reads.

---

## Recommended sequence

Ship Tier 1 as one deploy push and get the live URL before touching Tier 2 — the job goal points
to "have a verifiable live system." Tier 2 makes it survive a technical deep-dive. Tier 3 is the
finish, with the README rewrite as the last step once everything it describes is true.
