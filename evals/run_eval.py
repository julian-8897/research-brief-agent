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

from src.agent import ResearchBriefAgent, ResearchTools
from src.agent import toolset as toolset_module
from src.arxiv_client import ArxivClient
from src.embeddings import TextEmbedder
from src.models import BriefRequest, PaperRecord
from src.observability import Tracer
from src.retrieval import InMemoryVectorStore, build_vector_store
from src.settings import get_settings


class DeterministicEmbedder:
    def __init__(self, dimension: int = 8):
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
    def search_papers(self, query, max_results):
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
    lines.extend(
        [
            "",
            "Judge checks to run on reviewed outputs:",
            "- Answer relevance to the research question.",
            "- Citation grounding against supplied titles and abstracts.",
            "- Unsupported-claim risk.",
            "- Useful uncertainty or refusal behavior when evidence is weak.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_fixture_agent(fixture_path: Path, *, live_llm: bool) -> ResearchBriefAgent:
    base_settings = get_settings()
    settings = replace(
        base_settings,
        vector_store_backend="memory",
        embedding_dimension=8,
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

    def fixture_full_text(pdf_url: str, *, timeout: float, char_budget: int):
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
    return ResearchBriefAgent(settings=settings, tools=tools, tracer=Tracer(settings))


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
    args = parser.parse_args()

    if args.offline_fixture and args.fixture_corpus:
        raise SystemExit("Use only one of --offline-fixture or --fixture-corpus")

    if args.offline_fixture:
        agent = build_fixture_agent(args.fixture_papers, live_llm=False)
    elif args.fixture_corpus:
        agent = build_fixture_agent(args.fixture_papers, live_llm=True)
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

    rows = []
    args.jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.jsonl.open("w", encoding="utf-8") as handle:
        for case in load_cases(args.cases):
            request = BriefRequest(**case)
            started = time.perf_counter()
            final = asyncio.run(_collect(agent, request))
            row = {
                "id": case.get("id", request.research_question[:40]),
                "elapsed_ms": (time.perf_counter() - started) * 1000,
                "final": final,
            }
            rows.append(row)
            handle.write(json.dumps(row) + "\n")

    args.markdown.write_text(render_markdown(rows), encoding="utf-8")


if __name__ == "__main__":
    main()
