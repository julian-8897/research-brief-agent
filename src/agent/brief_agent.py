import json
import time
from collections.abc import AsyncIterator, Iterator
from typing import Any

from src.agent.tools import ResearchTools
from src.agent.toolset import ResearchToolset
from src.llm import (
    AssistantMessage,
    LLMProvider,
    Message,
    ToolCall,
    ToolResult,
    ToolResultsMessage,
    UserMessage,
    build_llm_provider,
)
from src.models import (
    BriefRequest,
    BriefResponse,
    CitedPaper,
    SearchResponseItem,
    UsageEstimate,
)
from src.observability import Tracer
from src.settings import Settings

_DISCOVERY_TOOLS = {"search_papers", "fetch_arxiv"}
_DISCOVERY_BUDGET_NUDGE = (
    "You have enough papers. Do not search or fetch again. Read full text and "
    "write the memo."
)
_MIN_FULL_TEXT_PAPERS = 1
_FULL_TEXT_REQUIRED_NUDGE = (
    "Final synthesis is blocked because you have retrieved papers but have not "
    "successfully read full text yet. Call get_full_text on the most relevant "
    "retrieved paper ids, then write the memo."
)
_PROGRAMMER_ERRORS = (AssertionError, IndexError, KeyError, TypeError, ValueError)


class AgentLiveRunError(RuntimeError):
    """A known live-loop boundary failure that can fall back explicitly."""

    def __init__(self, stage: str, message: str):
        super().__init__(message)
        self.stage = stage


def _estimate_tokens(text: str) -> int:
    """Rough token estimate used only for the offline fallback memo.

    The live agent loop reports measured token counts from each provider turn;
    see :meth:`ResearchBriefAgent._run_agent_loop`.
    """
    return max(1, len(text) // 4)


def _estimate_cost(settings: Settings, input_tokens: int, output_tokens: int) -> float:
    return round(
        (input_tokens / 1000 * settings.estimated_input_token_cost_per_1k)
        + (output_tokens / 1000 * settings.estimated_output_token_cost_per_1k),
        6,
    )


class ResearchBriefAgent:
    """A tool-using agent that researches a question and writes a cited brief.

    The model drives a loop over callable research tools (semantic search,
    arXiv backfill, evidence expansion) and decides when it has enough evidence
    to write the final decision memo. Turn and tool-call budgets bound cost.
    Without a configured LLM provider it falls back to a deterministic memo so
    local runs and CI work without credentials.
    """

    def __init__(
        self,
        settings: Settings,
        tools: ResearchTools,
        tracer: Tracer,
        llm: LLMProvider | None = None,
    ):
        self.settings = settings
        self.tools = tools
        self.tracer = tracer
        self.llm = llm if llm is not None else build_llm_provider(settings)

    async def stream(self, request: BriefRequest) -> AsyncIterator[dict[str, Any]]:
        for event in self._iterate(request):
            yield event

    def run(self, request: BriefRequest) -> BriefResponse:
        final: dict[str, Any] | None = None
        for event in self._iterate(request):
            if event.get("event") == "final":
                final = event["data"]
        assert final is not None, "agent did not emit a final event"
        return BriefResponse.model_validate(final)

    # -- Orchestration --------------------------------------------------------

    def _iterate(self, request: BriefRequest) -> Iterator[dict[str, Any]]:
        started = time.perf_counter()
        trace = self.tracer.start("research_brief", request.model_dump(mode="json"))
        yield {"event": "started", "message": "Research brief run started"}

        toolset = ResearchToolset(self.tools, request)

        if self.llm is None:
            yield from self._iterate_fallback(request, trace, started)
            return

        usage = UsageEstimate()
        warnings: list[str] = []
        final_text: str | None = None
        discovery_calls = 0
        discovery_budget_reached = self.settings.agent_max_search_calls <= 0
        discovery_budget_event_emitted = False
        system = self._system_prompt(request)
        messages: list[Message] = [UserMessage(self._user_prompt(request))]

        try:
            for turn_index in range(1, self.settings.agent_max_iterations + 1):
                if discovery_budget_reached and not discovery_budget_event_emitted:
                    messages.append(UserMessage(_DISCOVERY_BUDGET_NUDGE))
                    yield {"event": "discovery_budget_reached"}
                    discovery_budget_event_emitted = True
                offered_tools = (
                    toolset.read_only_specs
                    if discovery_budget_reached
                    else toolset.specs
                )
                with self.tracer.span(trace, "llm_turn", turn=turn_index):
                    turn = self._run_llm_turn(
                        system, messages, offered_tools, stage="llm_turn"
                    )
                self._accumulate(usage, turn)
                yield {
                    "event": "llm_turn",
                    "turn": turn_index,
                    "tools_requested": [call.name for call in turn.tool_calls],
                }
                messages.append(AssistantMessage(turn.text, turn.tool_calls))

                if not turn.tool_calls:
                    if self._needs_full_text(toolset):
                        discovery_budget_reached = True
                        messages.append(UserMessage(_FULL_TEXT_REQUIRED_NUDGE))
                        yield self._evidence_required_event(toolset)
                        continue
                    final_text = turn.text or ""
                    break

                results: list[ToolResult] = []
                for call in turn.tool_calls:
                    is_discovery_call = call.name in _DISCOVERY_TOOLS
                    if is_discovery_call:
                        discovery_calls += 1
                    yield {
                        "event": "tool_call",
                        "name": call.name,
                        "arguments": call.arguments,
                    }
                    if discovery_budget_reached and is_discovery_call:
                        content = json.dumps(
                            {
                                "error": "discovery budget reached",
                                "hint": _DISCOVERY_BUDGET_NUDGE,
                            }
                        )
                        meta = {"blocked": True}
                    else:
                        with self.tracer.span(trace, f"tool:{call.name}"):
                            content, meta = self._call_tool(toolset, call)
                    usage.tool_call_count += 1
                    results.append(ToolResult(call.id, content))
                    yield {"event": "tool_result", "name": call.name, **meta}
                messages.append(ToolResultsMessage(results))

                if discovery_calls >= self.settings.agent_max_search_calls:
                    discovery_budget_reached = True
                    if not discovery_budget_event_emitted:
                        messages.append(UserMessage(_DISCOVERY_BUDGET_NUDGE))
                        yield {"event": "discovery_budget_reached"}
                        discovery_budget_event_emitted = True

                if usage.tool_call_count >= self.settings.agent_max_tool_calls:
                    warnings.append(
                        "Tool-call budget reached; forcing final synthesis."
                    )
                    if self._needs_full_text(toolset):
                        warning = self._full_text_degraded_warning(toolset)
                        warnings.append(warning)
                        yield self._degraded_event("full_text_missing", warning, toolset)
                    final_text = self._force_final(
                        trace, request, system, messages, usage, toolset
                    )
                    break
            else:
                warnings.append(
                    "Reached max agent iterations; forcing final synthesis."
                )
                if self._needs_full_text(toolset):
                    warning = self._full_text_degraded_warning(toolset)
                    warnings.append(warning)
                    yield self._degraded_event("full_text_missing", warning, toolset)
                final_text = self._force_final(
                    trace, request, system, messages, usage, toolset
                )
        except AgentLiveRunError as exc:
            yield {
                "event": "error",
                "stage": exc.stage,
                "message": str(exc),
                "type": type(exc.__cause__).__name__ if exc.__cause__ else None,
            }
            yield {
                "event": "degraded",
                "reason": "live_agent_failure",
                "message": "Live agent failed; deterministic fallback memo will be returned.",
            }
            yield from self._iterate_fallback(
                request, trace, started, error=str(exc), toolset=toolset
            )
            return

        yield {
            "event": "synthesis_complete",
            "llm_calls": usage.llm_call_count,
            "full_text_fetched": toolset.fulltext_success_count,
        }

        brief = final_text or "No brief was produced."
        if toolset.retrieved_count < 2:
            warnings.append(
                "Retrieved evidence is thin; the brief should use explicit uncertainty."
            )
        usage.estimated_cost_usd = _estimate_cost(
            self.settings, usage.input_tokens, usage.output_tokens
        )
        response = BriefResponse(
            final_brief=brief,
            cited_papers=toolset.cited_papers(brief),
            retrieval_diagnostics=toolset.diagnostics(),
            latency_ms=(time.perf_counter() - started) * 1000,
            token_cost_estimate=usage,
            langfuse_trace_url=trace.trace_url,
            warnings=warnings,
        )
        yield {"event": "final", "data": response.model_dump(mode="json")}

    def _run_llm_turn(
        self,
        system: str,
        messages: list[Message],
        offered_tools,
        *,
        stage: str,
        tool_choice: str = "auto",
    ):
        try:
            return self.llm.run_turn(
                system, messages, offered_tools, tool_choice=tool_choice
            )
        except _PROGRAMMER_ERRORS:
            raise
        except Exception as exc:
            raise AgentLiveRunError(
                stage,
                f"LLM provider turn failed. Check provider credentials, model, "
                f"base URL, and request limits. Original error: {exc}",
            ) from exc

    @staticmethod
    def _call_tool(
        toolset: ResearchToolset, call: ToolCall
    ) -> tuple[str, dict[str, Any]]:
        try:
            return toolset.call(call)
        except _PROGRAMMER_ERRORS:
            raise
        except Exception as exc:
            raise AgentLiveRunError(
                f"tool:{call.name}",
                f"Tool '{call.name}' failed. Check the corpus, arXiv availability, "
                f"embedding service, and tool arguments. Original error: {exc}",
            ) from exc

    @staticmethod
    def _needs_full_text(toolset: ResearchToolset) -> bool:
        required = min(_MIN_FULL_TEXT_PAPERS, toolset.retrieved_count)
        return required > 0 and toolset.fulltext_success_count < required

    @staticmethod
    def _evidence_required_event(toolset: ResearchToolset) -> dict[str, Any]:
        candidates = [item.paper.id for item in toolset.retrieved_items[:3]]
        return {
            "event": "evidence_required",
            "reason": "full_text_missing",
            "required_full_text_papers": min(
                _MIN_FULL_TEXT_PAPERS, toolset.retrieved_count
            ),
            "full_text_fetched": toolset.fulltext_success_count,
            "candidate_ids": candidates,
            "message": _FULL_TEXT_REQUIRED_NUDGE,
        }

    @staticmethod
    def _full_text_degraded_warning(toolset: ResearchToolset) -> str:
        if toolset.fulltext_attempt_count:
            return (
                "Full-text evidence requirement was not met after attempted "
                "full-text fetches; forced synthesis is degraded."
            )
        return (
            "Full-text evidence requirement was not met before budget exhaustion; "
            "forced synthesis is degraded."
        )

    @staticmethod
    def _degraded_event(
        reason: str, message: str, toolset: ResearchToolset
    ) -> dict[str, Any]:
        return {
            "event": "degraded",
            "reason": reason,
            "message": message,
            "full_text_fetched": toolset.fulltext_success_count,
            "full_text_attempts": toolset.fulltext_attempt_count,
            "full_text_errors": toolset.fulltext_error_count,
        }

    def _force_final(
        self,
        trace,
        request: BriefRequest,
        system: str,
        messages: list[Message],
        usage: UsageEstimate,
        toolset: ResearchToolset,
    ) -> str:
        """Ask the model for a final brief with tools disabled."""
        directive = (
            system
            + "\n\nYou have reached your tool budget. Write the final decision "
            "memo now using the evidence already gathered. Do not call any tools."
        )
        with self.tracer.span(trace, "forced_synthesis"):
            turn = self._run_llm_turn(
                directive,
                messages,
                toolset.specs,
                stage="forced_synthesis",
                tool_choice="none",
            )
        self._accumulate(usage, turn)
        messages.append(AssistantMessage(turn.text, turn.tool_calls))
        if not self._is_invalid_final_text(turn.text):
            return turn.text or ""

        brief, _fallback_usage = self._fallback_brief(request, toolset.retrieved_items)
        return (
            brief
            + "\n\nOperational note: forced synthesis returned tool-call markup, "
            "so this deterministic fallback was used."
        )

    @staticmethod
    def _is_invalid_final_text(text: str | None) -> bool:
        if not text or not text.strip():
            return True
        stripped = text.lstrip()
        return (
            "DSML" in text
            or "tool_calls" in text
            or "invoke name=" in text
            or (stripped.startswith("<") and 'name="' in stripped)
        )

    @staticmethod
    def _accumulate(usage: UsageEstimate, turn) -> None:
        usage.llm_call_count += 1
        usage.input_tokens += turn.input_tokens
        usage.output_tokens += turn.output_tokens

    # -- Prompts --------------------------------------------------------------

    def _system_prompt(self, request: BriefRequest) -> str:
        return (
            "You are a scientific research brief agent for applied science teams.\n"
            "Your job: research the user's question using the provided tools, then "
            "write a cited decision memo grounded only in retrieved arXiv evidence.\n\n"
            "Workflow:\n"
            "1. Run at most TWO discovery rounds total using search_papers and "
            "fetch_arxiv. Do not repeat a search query you already ran.\n"
            "2. Call fetch_arxiv only if search_papers returned too few relevant "
            "results, then search once more.\n"
            "3. Once you have a handful of relevant papers, STOP searching. "
            "Triage candidates with get_paper_details (abstract-level), then call "
            "get_full_text on the few most promising papers to read their methods "
            "and results, not just the abstract.\n"
            "4. After reading full text for 2-3 promising papers, write the memo. "
            "Do not keep broadening the paper set.\n\n"
            f"The memo must be a {request.brief_type} containing: a recommendation, "
            "relevant methods, baselines to compare, implementation risks, explicit "
            "uncertainty (refuse to over-claim when evidence is weak), and concrete "
            "next steps. Cite papers inline by arXiv id, e.g. [2401.00001]. Only cite "
            "papers returned by the tools."
        )

    def _user_prompt(self, request: BriefRequest) -> str:
        return (
            f"Research question: {request.research_question}\n"
            f"Domain: {request.domain or 'unspecified'}\n"
            f"Constraints: {request.constraints or []}\n"
            f"Target papers to consider: up to {request.max_papers}."
        )

    # -- Offline fallback (no provider configured, or provider error) ---------

    def _iterate_fallback(
        self,
        request: BriefRequest,
        trace,
        started: float,
        *,
        error: str | None = None,
        toolset: ResearchToolset | None = None,
    ) -> Iterator[dict[str, Any]]:
        with self.tracer.span(trace, "fallback_retrieval"):
            retrieval = self.tools.vector_retrieve(
                request.research_question, request.max_papers
            )
            if retrieval.diagnostics.returned == 0:
                ingested, papers = self.tools.fetch_and_ingest(
                    f"all:{request.research_question}",
                    request.max_papers,
                    date_range=request.date_range,
                )
                if ingested:
                    retrieval = self.tools.vector_retrieve(
                        request.research_question, request.max_papers
                    )
        yield {
            "event": "retrieval_complete",
            "returned": retrieval.diagnostics.returned,
            "latency_ms": retrieval.diagnostics.retrieval_latency_ms,
        }

        brief, usage = self._fallback_brief(request, retrieval.items)
        if error:
            brief += (
                f"\n\nOperational note: live synthesis failed, so this deterministic "
                f"fallback was used ({error})."
            )
        yield {"event": "synthesis_complete", "llm_calls": usage.llm_call_count}

        warnings: list[str] = []
        if error:
            warnings.append("LLM provider error; deterministic fallback memo returned.")
        if retrieval.diagnostics.returned < max(2, min(4, request.max_papers)):
            warnings.append(
                "Retrieved evidence is thin; the brief uses explicit uncertainty."
            )
        usage.estimated_cost_usd = _estimate_cost(
            self.settings, usage.input_tokens, usage.output_tokens
        )
        response = BriefResponse(
            final_brief=brief,
            cited_papers=[
                CitedPaper(
                    id=item.paper.id,
                    title=item.paper.title,
                    arxiv_url=item.paper.arxiv_url,
                    score=item.score,
                )
                for item in retrieval.items
            ],
            retrieval_diagnostics=retrieval.diagnostics,
            latency_ms=(time.perf_counter() - started) * 1000,
            token_cost_estimate=usage,
            langfuse_trace_url=trace.trace_url,
            warnings=warnings,
        )
        yield {"event": "final", "data": response.model_dump(mode="json")}

    def _fallback_brief(
        self, request: BriefRequest, items: list[SearchResponseItem]
    ) -> tuple[str, UsageEstimate]:
        if not items:
            brief = (
                "# Decision Memo\n\n"
                "I do not have enough retrieved arXiv evidence to answer this question "
                "reliably. Ingest a relevant corpus first, then rerun the brief.\n"
            )
            return brief, UsageEstimate(
                input_tokens=_estimate_tokens(request.research_question),
                output_tokens=_estimate_tokens(brief),
                llm_call_count=0,
            )

        citations = ", ".join(f"[{item.paper.id}]" for item in items[:5])
        methods = "\n".join(
            f"- {item.paper.title} [{item.paper.id}]: {item.paper.summary[:280].strip()}..."
            for item in items[:5]
        )
        constraints = (
            "\n".join(f"- {constraint}" for constraint in request.constraints)
            if request.constraints
            else "- No explicit deployment constraints were provided."
        )
        uncertainty = (
            "Evidence is limited because fewer than four papers were retrieved."
            if len(items) < 4
            else "Evidence is based on title, abstract, and metadata; full-paper claims need follow-up reading."
        )
        brief = f"""# Decision Memo

## Recommendation
Use the retrieved literature as a scoping signal, not as final technical proof. The current evidence suggests the question is worth pursuing if the team can validate the strongest methods against its own data and operational constraints. Key supporting papers: {citations}.

## Evidence And Methods
{methods}

## Baselines To Compare
- Reproduce a simple non-agent or non-neural baseline before adopting complex methods.
- Compare against the strongest recent method represented in the retrieved papers.
- Track latency, cost, and failure cases alongside headline accuracy or scientific metrics.

## Constraints
{constraints}

## Risks And Uncertainty
- {uncertainty}
- Abstract-level evidence can miss negative results, data leakage, and implementation details.
- If decisions affect safety, clinical, financial, or high-cost operations, treat this as a triage memo only.

## Next Steps
- Read the top cited papers in full and extract datasets, baselines, and evaluation protocols.
- Build a small benchmark matching the deployment domain.
- Run an ablation that separates retrieval quality, model quality, and domain-specific preprocessing.
"""
        return brief, UsageEstimate(
            input_tokens=_estimate_tokens(request.research_question + methods),
            output_tokens=_estimate_tokens(brief),
            llm_call_count=0,
        )
