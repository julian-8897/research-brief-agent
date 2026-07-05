import argparse
import asyncio
import hashlib
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evals import metrics
from src.agent import ResearchBriefAgent, ResearchTools
from src.agent import toolset as toolset_module
from src.arxiv_client import ArxivClient
from src.embeddings import TextEmbedder
from src.llm import build_llm_provider
from src.models import BriefRequest, PaperRecord
from src.observability import Tracer
from src.retrieval import InMemoryVectorStore, build_vector_store
from src.settings import get_settings

# Fixture runs use a tiny deterministic embedding space so the offline/CI smoke
# path needs no model download; the app default (768) is irrelevant here.
FIXTURE_EMBEDDING_DIMENSION = 8


class DeterministicEmbedder:
    def __init__(self, dimension: int = FIXTURE_EMBEDDING_DIMENSION):
        self.dimension = dimension

    def _encode(self, texts):
        vectors = []
        for text in texts:
            digest = hashlib.sha256(text.lower().encode("utf-8")).digest()
            values = [digest[i] / 255 for i in range(self.dimension)]
            vectors.append(values)
        return np.array(vectors, dtype="float32")

    def encode_documents(self, texts, batch_size=32):
        return self._encode(texts)

    def encode_queries(self, texts, batch_size=32):
        return self._encode(texts)


class OfflineArxivClient:
    def search_papers(self, query, max_results, sort_by=None):
        return []


async def _collect(agent: ResearchBriefAgent, request: BriefRequest):
    final = None
    async for event in agent.stream(request):
        if event["event"] == "final":
            final = event["data"]
    if final is None:
        raise RuntimeError("agent stream ended without a final event")
    return final


def load_cases(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def render_markdown(rows: list[dict]) -> str:
    lines = [
        "# Research Brief Evaluation",
        "",
        "| Case | Status | Latency ms | Retrieval ms | LLM calls | Tool calls | Full text | Est. cost | Citations | Trace |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        final = row["final"]
        diagnostics = final["retrieval_diagnostics"]
        full_text = final.get("full_text_diagnostics", {})
        usage = final["token_cost_estimate"]
        warnings = final.get("warnings") or []
        fallback = "deterministic fallback" in final.get("final_brief", "")
        if fallback:
            status = "fallback"
        elif warnings:
            status = f"warnings:{len(warnings)}"
        else:
            status = "ok"
        lines.append(
            "| {case} | {status} | {latency:.0f} | {retrieval:.0f} | {calls} | {tool_calls} | {full_text_ok}/{full_text_attempted} | ${cost:.5f} | {cites} | {trace} |".format(
                case=row["id"],
                status=status,
                latency=final["latency_ms"],
                retrieval=diagnostics["retrieval_latency_ms"],
                calls=usage["llm_call_count"],
                tool_calls=usage["tool_call_count"],
                full_text_ok=full_text.get("succeeded", 0),
                full_text_attempted=full_text.get("attempted", 0),
                cost=usage["estimated_cost_usd"],
                cites=len(final["cited_papers"]),
                trace="yes" if final.get("langfuse_trace_url") else "no",
            )
        )

    metric_rows = [row["metrics"] for row in rows if "metrics" in row]
    if metric_rows:
        lines.extend(
            [
                "",
                "## Quality metrics (automated)",
                "",
                "| Case | Cited | Valid ids | Halluc. | Read-in-full | Uncertainty | Cite support (LLM) | Faithfulness | Answer rel. |",
                "|---|---:|---:|---:|---:|---|---:|---:|---:|",
            ]
        )
        for row in rows:
            metric = row.get("metrics")
            if not metric:
                continue
            grounding = metric["citation_grounding"]
            uncertainty = metric["uncertainty_signaling"]
            judge = row.get("judge") or {}
            faith = judge.get("faithfulness") or {}
            cite_judge = judge.get("citation_grounding") or {}
            lines.append(
                "| {case} | {cited} | {grounded:.0%} | {halluc} | {read:.0%} | {unc} | {support} | {faith} | {rel} |".format(
                    case=row["id"],
                    cited=len(grounding["cited"]),
                    grounded=grounding["grounding_rate"],
                    halluc=len(grounding["hallucinated"]),
                    read=grounding["full_text_rate"],
                    unc="ok" if uncertainty["appropriate"] else "MISSING",
                    support=_fmt_ratio(cite_judge.get("grounded_rate")),
                    faith=_fmt_ratio(faith.get("faithfulness")),
                    rel=_fmt_ratio(faith.get("answer_relevance")),
                )
            )

        retrieval_rows = [
            (row["id"], row["retrieval_eval"])
            for row in rows
            if row.get("retrieval_eval")
        ]
        if retrieval_rows:
            lines.extend(
                [
                    "",
                    "## Retrieval relevance",
                    "",
                    "| Case | k | Hits | Recall@k | nDCG@k |",
                    "|---|---:|---:|---:|---:|",
                ]
            )
            for case_id, retrieval in retrieval_rows:
                lines.append(
                    "| {case} | {k} | {hits} | {recall:.0%} | {ndcg:.0%} |".format(
                        case=case_id,
                        k=retrieval["k"],
                        hits=retrieval["hits"],
                        recall=retrieval["recall"],
                        ndcg=retrieval["ndcg"],
                    )
                )

        summary = metrics.aggregate(metric_rows)
        lines.extend(
            [
                "",
                "## Aggregate",
                "",
                f"- Cases: {summary['cases']}",
                f"- Mean citation grounding: {summary['mean_grounding_rate']:.0%}",
                f"- Mean hallucination rate: {summary['mean_hallucination_rate']:.0%} "
                f"({summary['cases_with_hallucinations']} case(s) with fabricated ids)",
                f"- Mean cited-papers-read-in-full: {summary['mean_full_text_rate']:.0%}",
                f"- Mean full-text fetch success: {summary['mean_full_text_success']:.0%}",
                f"- Uncertainty signaled appropriately: "
                f"{summary['uncertainty_appropriate_rate']:.0%} of cases",
            ]
        )
    return "\n".join(lines) + "\n"


def _fmt_ratio(value) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.0%}"
    except (TypeError, ValueError):
        return "-"


def build_fixture_agent(fixture_path: Path, *, live_llm: bool) -> ResearchBriefAgent:
    base_settings = get_settings()
    settings = replace(
        base_settings,
        vector_store_backend="memory",
        embedding_dimension=FIXTURE_EMBEDDING_DIMENSION,
        anthropic_api_key=base_settings.anthropic_api_key if live_llm else None,
        openai_api_key=base_settings.openai_api_key if live_llm else None,
    )
    embedder = DeterministicEmbedder(settings.embedding_dimension)
    vector_store = InMemoryVectorStore(settings.embedding_dimension)
    papers = []
    with fixture_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                papers.append(PaperRecord(**json.loads(line)))
    embeddings = embedder.encode_documents(
        [f"{paper.title}\n{paper.summary}" for paper in papers]
    )
    vector_store.upsert(papers, embeddings)
    paper_by_id = {paper.id: paper for paper in papers}

    def fixture_full_text(
        pdf_url: str, *, timeout: float, char_budget: int, **kwargs
    ):
        paper_id = pdf_url.rstrip("/").split("/")[-1]
        paper = paper_by_id.get(paper_id)
        if paper is None:
            return "", False
        body = (
            f"Title: {paper.title}\n"
            f"Abstract: {paper.summary}\n"
            "Methods: compare retrieval quality, citation grounding, baselines, "
            "latency, calibration, and operational failure modes.\n"
            "Results: use explicit uncertainty and cite only retrieved evidence. "
        )
        repeated = (body * max(1, (char_budget // max(len(body), 1)) + 1))[:char_budget]
        return repeated, len(repeated) >= char_budget

    toolset_module.fetch_arxiv_fulltext = fixture_full_text
    tools = ResearchTools(
        settings=settings,
        arxiv_client=OfflineArxivClient(),
        embedder=embedder,
        vector_store=vector_store,
    )
    agent = ResearchBriefAgent(settings=settings, tools=tools, tracer=Tracer(settings))
    return agent, paper_by_id


def main():
    parser = argparse.ArgumentParser(
        description="Run research brief latency/cost evals."
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path("evals/benchmarks/research_questions.jsonl"),
    )
    parser.add_argument(
        "--jsonl",
        type=Path,
        default=Path("evals/reports/latest.jsonl"),
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=Path("evals/reports/latest.md"),
    )
    parser.add_argument(
        "--core",
        action="store_true",
        help="Run only the core-tagged benchmark subset (fast tuning-iteration loop).",
    )
    parser.add_argument(
        "--offline-fixture",
        action="store_true",
        help="Run with fixture papers and deterministic embeddings for CI smoke tests.",
    )
    parser.add_argument(
        "--fixture-corpus",
        action="store_true",
        help=(
            "Run against fixture papers and deterministic embeddings while keeping "
            "the configured live LLM provider."
        ),
    )
    parser.add_argument(
        "--fixture-papers",
        type=Path,
        default=Path("evals/benchmarks/fixture_papers.jsonl"),
    )
    parser.add_argument(
        "--judge",
        action="store_true",
        help=(
            "Also run an LLM-graded faithfulness/answer-relevance judge "
            "(requires a configured provider key)."
        ),
    )
    args = parser.parse_args()

    if args.offline_fixture and args.fixture_corpus:
        raise SystemExit("Use only one of --offline-fixture or --fixture-corpus")

    if args.offline_fixture:
        agent, paper_by_id = build_fixture_agent(args.fixture_papers, live_llm=False)
        settings = agent.settings
    elif args.fixture_corpus:
        agent, paper_by_id = build_fixture_agent(args.fixture_papers, live_llm=True)
        settings = agent.settings
    else:
        settings = get_settings()
        vector_store = build_vector_store(settings)
        tools = ResearchTools(
            settings=settings,
            arxiv_client=ArxivClient(),
            embedder=TextEmbedder(
                settings.embedding_model,
                document_adapter=settings.embedding_document_adapter,
                query_adapter=settings.embedding_query_adapter,
            ),
            vector_store=vector_store,
        )
        agent = ResearchBriefAgent(
            settings=settings, tools=tools, tracer=Tracer(settings)
        )
        paper_by_id = {}

    judge_provider = build_llm_provider(settings) if args.judge else None
    if args.judge and judge_provider is None:
        print("warning: --judge requested but no provider configured; skipping judge")

    rows = []
    args.jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.jsonl.open("w", encoding="utf-8") as handle:
        for case in load_cases(args.cases):
            if args.core and not case.get("core"):
                continue
            request = BriefRequest(**case)
            started = time.perf_counter()
            final = asyncio.run(_collect(agent, request))
            # Legitimate corpus: the fixture ids when known, else the ids the
            # system itself recognized as citations (a weaker live-run signal).
            known_ids = (
                set(paper_by_id)
                if paper_by_id
                else {p["id"] for p in final.get("cited_papers", [])}
            )
            case_metrics = metrics.score_case(final, known_ids)
            retrieval_eval = _run_retrieval_eval(
                agent, request, set(case.get("relevant_ids", []))
            )
            row = {
                "id": case.get("id", request.research_question[:40]),
                "elapsed_ms": (time.perf_counter() - started) * 1000,
                "final": final,
                "metrics": case_metrics,
            }
            if retrieval_eval is not None:
                row["retrieval_eval"] = retrieval_eval
            if judge_provider is not None:
                row["judge"] = _run_judge(judge_provider, request, final, paper_by_id)
            rows.append(row)
            handle.write(json.dumps(row) + "\n")

    args.markdown.write_text(render_markdown(rows), encoding="utf-8")


def _run_retrieval_eval(agent, request, relevant_ids):
    if not relevant_ids:
        return None
    retrieval = agent.tools.vector_retrieve(
        request.research_question, request.max_papers
    )
    ranked_ids = [item.paper.id for item in retrieval.items]
    return metrics.retrieval_relevance(
        ranked_ids, relevant_ids, k=min(request.max_papers, len(ranked_ids) or 1)
    )


def _run_judge(provider, request, final, paper_by_id):
    """Grade the brief with LLM judges: whole-brief faithfulness and per-citation
    grounding."""
    brief = final.get("final_brief", "")
    evidence_by_id = {}
    for cid in metrics.extract_citations(brief):
        paper = paper_by_id.get(cid) or paper_by_id.get(cid.split("v")[0])
        if paper is not None:
            evidence_by_id[cid] = {"title": paper.title, "text": paper.summary}
    if not evidence_by_id:
        return {"error": "no citable evidence available for judging"}
    faith_evidence = [{"id": cid, **ev} for cid, ev in evidence_by_id.items()]
    return {
        "faithfulness": metrics.faithfulness_judge(
            provider, request.research_question, brief, faith_evidence
        ),
        "citation_grounding": metrics.citation_grounding_judge(
            provider, brief, evidence_by_id
        ),
    }


if __name__ == "__main__":
    main()
