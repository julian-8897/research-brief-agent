# Plan: Agent-loop policy fixes (convergence, forced-final, fetch dedup)

Implementation handoff. Self-contained: everything needed is below. Read
`AGENTS.md` first for architecture and commands.

## Context

The brief agent (`src/agent/brief_agent.py`) is a tool-using loop: the model
calls research tools (`search_papers`, `fetch_arxiv`, `get_paper_details`,
`get_full_text` in `src/agent/toolset.py`) until it writes a cited decision memo.
Mechanics (tool dispatch, parallel tool calls, streaming SSE, measured token
usage, both Anthropic and OpenAI-compatible backends) work. The **policy** is
broken. A live run against DeepSeek V4 (`deepseek-v4-flash`, OpenAI-compatible)
exposed three failures:

1. **Breadth loop, no convergence.** The model called `search_papers` /
   `fetch_arxiv` repeatedly (7 turns, parallel calls), exhausted the 12-call
   tool budget, and **never** called `get_full_text` or `get_paper_details`. It
   gathered papers endlessly instead of reading any.
2. **Forced-final synthesis breaks.** When the tool budget is hit, the loop
   calls `_force_final`, which runs a turn with **no tools** to make the model
   write the memo. DeepSeek instead tried to call more tools and, with no
   structured tool channel, leaked its raw internal tool-call markup
   (`<｜｜DSML｜｜tool_calls> ... invoke name="search_papers" ...`) as plain text
   into `final_brief`. Result: the "brief" is garbage, not a memo.
3. **`fetch_arxiv` redundantly re-ingests.** Each `fetch_arxiv` reported
   "ingested 30" even when re-fetching the same papers. The model couldn't tell
   it was making no progress, which fueled the loop. It also blew up cost:
   ~60,686 input tokens for one brief, because every turn re-sends the whole
   growing transcript.

Goal: make the agent converge to reading + writing, guarantee a valid memo even
when the model misbehaves, and stop the redundant fetch loop. Keep full
Anthropic + OpenAI-compatible parity. Keep the no-key deterministic fallback and
all existing tests green.

## Fix 1 — Convergence steering (highest leverage)

Stop the model searching forever; force it to read full text and write.

**1a. Strengthen the system prompt** in `ResearchBriefAgent._system_prompt`
(`src/agent/brief_agent.py`). Make the limits explicit and imperative:
- At most TWO rounds of discovery (`search_papers` / `fetch_arxiv`).
- Do not repeat a search you already ran.
- Once you have a handful of relevant papers, STOP searching, call
  `get_full_text` on the 2–3 most promising, then write the memo.
- Only `fetch_arxiv` if `search_papers` returned too few results.

**1b. Hard enforcement via tool withdrawal.** Prompt text alone is insufficient
(DeepSeek ignored it). Add a discovery budget and, once spent, stop offering the
discovery tools so the model physically cannot search again:
- New setting in `src/settings.py`: `agent_max_search_calls: int =
  _int_env("AGENT_MAX_SEARCH_CALLS", 4)`.
- In `ResearchToolset` (`src/agent/toolset.py`) add a property
  `read_only_specs` returning only `[get_paper_details, get_full_text]` specs
  (refactor `specs` so the two lists share the same `ToolSpec` definitions, no
  duplication).
- In `ResearchBriefAgent._iterate`, track a local `discovery_calls` counter;
  increment it whenever a dispatched tool is `search_papers` or `fetch_arxiv`.
  When `discovery_calls >= settings.agent_max_search_calls`, pass
  `toolset.read_only_specs` to subsequent `run_turn` calls instead of
  `toolset.specs`, and emit a one-time SSE event
  `{"event": "discovery_budget_reached"}` plus a transcript nudge
  (a `UserMessage` like: "You have enough papers. Do not search or fetch again.
  Read full text and write the memo.").

Acceptance: a model that keeps trying to search is, after
`agent_max_search_calls`, only offered the read tools, so it must either read or
write. `get_full_text` becomes reachable in practice.

## Fix 2 — Robust forced-final synthesis

When forced to finish, the model must produce prose, and we must never return
tool-call markup as a brief.

**2a. Add a `tool_choice` mode to the provider interface.** Extend
`LLMProvider.run_turn` (`src/llm/base.py`) signature to:
```python
def run_turn(self, system, messages, tools, *, tool_choice: str = "auto") -> TurnResult: ...
```
`tool_choice` is `"auto"` (model may call tools) or `"none"` (tools visible but
the model must answer in text).
- **OpenAI** (`src/llm/openai_provider.py`): when `tool_choice == "none"`, still
  send the `tools` list but set `kwargs["tool_choice"] = "none"`. Do NOT just
  omit tools — omitting them is what made DeepSeek improvise raw markup. When
  `"auto"`, keep current behavior (`tool_choice="auto"` with tools).
- **Anthropic** (`src/llm/anthropic_provider.py`): when `tool_choice == "none"`,
  send `tools` plus `kwargs["tool_choice"] = {"type": "none"}`. When `"auto"`,
  current behavior.

**2b. Use it in `_force_final`** (`src/agent/brief_agent.py`): call
`self.llm.run_turn(directive, messages, toolset.specs, tool_choice="none")`
(pass the real tools list, not `[]`).

**2c. Guard against a still-bad final answer.** After the forced turn, validate
the text. If it is empty, or looks like leaked tool-call markup, discard it and
return the deterministic `_fallback_brief` over the retrieved items, appending an
operational note. Detection helper (treat as invalid if any apply):
- text is empty/whitespace, or
- contains `DSML`, or `tool_calls`, or the substring `invoke name=`, or
- starts with `<` and contains `name="`.
`_force_final` needs access to the toolset's retrieved items to build the
fallback — pass the `toolset` in, and reuse
`toolset.cited_papers(...)` / `_fallback_brief`.

Acceptance: a budget-exhausted run always yields a real memo (either a clean
model memo or the deterministic fallback), never raw tool markup.

## Fix 3 — `fetch_arxiv` dedup + progress signal

Tell the model when a fetch added nothing, so it stops re-fetching, and cut cost.

In `ResearchToolset` (`src/agent/toolset.py`):
- Track ingested ids across the run: add `self._ingested_ids: set[str]` in
  `__init__`.
- In `_fetch_arxiv`, after calling `self._tools.fetch_and_ingest(...)`, compute
  `new = [p for p in papers if p.id not in self._ingested_ids]`, update the set,
  and return a payload reporting both counts, e.g.
  `{"new": len(new), "already_known": len(papers) - len(new), "titles": [...]}`.
  When `new == 0`, include a `"hint": "No new papers found. Stop fetching; read
  full text of the papers you already have and write the memo."`
- Event metadata: change the `tool_result` meta from `{"ingested": n}` to
  `{"new": len(new)}`.

Acceptance: repeated identical `fetch_arxiv` calls report `new: 0` with a stop
hint.

## Optional stretch — transcript cost control

Not required, but addresses the ~60k-token blowup. In `_iterate`, before each
`run_turn`, trim older `ToolResultsMessage` payloads so only the most recent
search results keep full abstracts; older ones keep ids/titles/scores only. Keep
behind a setting (`AGENT_TRIM_TRANSCRIPT`, default off) to avoid changing default
behavior. Skip if it complicates the core fixes.

## Test plan

All tests must stay hermetic (no live API; `tests/conftest.py` enforces this).
Run `uv run pytest -q` and `uv run ruff check .`.

- **Provider `tool_choice`** (`tests/test_llm_providers.py`): extend the faked
  SDK tests to assert that with `tool_choice="none"` the OpenAI payload sets
  `tool_choice == "none"` and still includes `tools`; same for Anthropic
  (`{"type": "none"}`). Update the fakes' `run_turn` to accept the new kwarg.
- **Forced-final fallback** (`tests/test_agent.py`): a `ScriptedProvider` whose
  forced-final turn returns markup-like text (e.g. `'<｜｜DSML｜｜tool_calls>...'`)
  must yield a deterministic memo (contains "Decision Memo"), not the markup.
- **Discovery withdrawal** (`tests/test_agent.py`): with
  `agent_max_search_calls=1`, a provider that always requests `search_papers`
  must, after the budget, be offered only read tools (assert via a provider that
  records the `tools` arg names it receives), and the run still completes.
- **Fetch dedup** (`tests/test_full_text.py` or a new `tests/test_toolset.py`):
  calling `fetch_arxiv` twice with a monkeypatched `fetch_and_ingest` returning
  the same papers reports `new: 0` and includes the stop hint on the second call.
- Update existing `ScriptedProvider` / fake providers anywhere to accept the new
  `tool_choice` keyword so they remain valid `LLMProvider`s.

## Validation (manual, after implementation)

Re-run against DeepSeek V4 (`.env` already configured; `LLM_PROVIDER=openai`,
`OPENAI_MODEL=deepseek-v4-flash`):
```bash
uv run uvicorn src.api.main:app --port 8000
curl -X POST localhost:8000/ingest -H 'content-type: application/json' \
  -d '{"query":"cat:cs.LG AND all:retrieval augmented generation","max_papers":25}'
curl -N -X POST localhost:8000/briefs/stream -H 'content-type: application/json' \
  -d '{"research_question":"What RAG methods improve factual grounding, and their tradeoffs?","domain":"cs.LG","max_papers":6}'
```
Success looks like: a `get_full_text` tool call appears in the event stream, the
run does not hit the tool budget, and `final_brief` is a real cited decision memo
(no `DSML`/tool markup). Note the new input-token count vs the ~60k baseline.

## Files to touch

- `src/llm/base.py` — `run_turn` signature (`tool_choice` kwarg) on the Protocol.
- `src/llm/openai_provider.py`, `src/llm/anthropic_provider.py` — honor `tool_choice`.
- `src/agent/brief_agent.py` — system prompt, discovery counter + tool
  withdrawal + nudge, `_force_final` (tool_choice="none" + invalid-output guard).
- `src/agent/toolset.py` — `read_only_specs`, `_ingested_ids` dedup in `_fetch_arxiv`.
- `src/settings.py` — `agent_max_search_calls` (+ optional `agent_trim_transcript`).
- `tests/` — as above.
- Update `PROGRESS.md` (record the DeepSeek findings + these fixes) and the tool
  list in `AGENTS.md` if behavior notes change.
