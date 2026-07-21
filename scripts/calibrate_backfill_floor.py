"""Measure SPECTER2 adapter behavior and retrieval-score distributions.

Two diagnostics behind the AGENT_SEARCH_BACKFILL_MIN_SCORE retune:

1. Adapter activation check. Logs from the ``adapters`` library ("adapters
   available but none are activated", "Could not identify valid prediction
   head(s)") raised the suspicion that the SPECTER2 proximity/adhoc adapters
   might not be applying. This encodes fixed texts with and without each
   adapter and reports pairwise cosines, so activation is verified directly
   from embedding geometry rather than inferred from log noise.

2. Score-distribution calibration. Runs the benchmark research questions
   against the seeded warm corpus and records the cosine of every hit, labeled
   relevant/non-relevant via each case's ``relevant_ids`` (fixture-only ids
   excluded). The separation between the relevant and non-relevant bands is
   what makes an absolute backfill floor defensible; the suggested floor is
   derived from the measured bands, not guessed.

Writes a JSON report (default: .local/calibration/backfill-floor.json).

The embedded Qdrant store is single-process: stop the app before running this.

Usage:
    uv run python scripts/calibrate_backfill_floor.py
    uv run python scripts/calibrate_backfill_floor.py --k 12 --output /tmp/floor.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_VERSION_SUFFIX = re.compile(r"v\d+$")

# Representative texts for the adapter check: one abstract-style document, one
# decision-style query, matching how the app uses each adapter.
_PROBE_DOCUMENT = (
    "Title: GPTQ: Accurate Post-Training Quantization for Generative Pre-trained "
    "Transformers\nAbstract: We propose a new one-shot weight quantization method "
    "based on approximate second-order information, enabling 3-4 bit per-weight "
    "compression of large language models with negligible accuracy loss."
)
_PROBE_QUERY = (
    "Research question: Should we adopt 4-bit quantization for on-device LLM "
    "serving, and what accuracy-versus-latency tradeoffs does it carry?"
)
# Pairwise cosine below this flags the two encodings as meaningfully different.
# Adapter outputs for the same input should differ clearly; identical outputs
# (cosine ~1.0) mean the adapter is not being applied.
_ADAPTER_DIFF_COSINE = 0.999


def _base_id(paper_id: str) -> str:
    return _VERSION_SUFFIX.sub("", paper_id.strip())


def _cosine(a, b) -> float:
    import numpy as np

    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def _encode_with_active_adapter(embedder, texts, adapter_name):
    """Encode with an explicit active adapter, or None for the raw base model.

    Reaches into the embedder's private lock/forward so the measurement goes
    through exactly the same pooling path as production encodes.
    """
    import torch

    embedder._ensure_loaded()
    inputs = embedder._tokenizer(
        texts, truncation=True, max_length=512, padding=True, return_tensors="pt"
    )
    device = next(embedder._model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with embedder._adapter_lock, torch.inference_mode():
        embedder._model.set_active_adapters(adapter_name)
        output = embedder._model(**inputs)
    return output.last_hidden_state[:, 0, :].detach().cpu().numpy()


def run_adapter_check(embedder) -> dict:
    """Pairwise cosines between base/proximity/adhoc encodings of fixed texts."""
    doc_base = _encode_with_active_adapter(embedder, [_PROBE_DOCUMENT], None)[0]
    doc_prox = _encode_with_active_adapter(
        embedder, [_PROBE_DOCUMENT], embedder.document_adapter_name
    )[0]
    query_base = _encode_with_active_adapter(embedder, [_PROBE_QUERY], None)[0]
    query_adhoc = _encode_with_active_adapter(
        embedder, [_PROBE_QUERY], embedder.query_adapter_name
    )[0]

    pairs = {
        "document base vs proximity": _cosine(doc_base, doc_prox),
        "query base vs adhoc_query": _cosine(query_base, query_adhoc),
        "document proximity vs query adhoc_query": _cosine(doc_prox, query_adhoc),
        "document base vs query base": _cosine(doc_base, query_base),
    }
    applied = {
        "proximity_changes_document": pairs["document base vs proximity"]
        < _ADAPTER_DIFF_COSINE,
        "adhoc_changes_query": pairs["query base vs adhoc_query"]
        < _ADAPTER_DIFF_COSINE,
    }
    return {
        "cosines": {k: round(v, 6) for k, v in pairs.items()},
        "adapters_appear_applied": applied,
        "note": (
            "cosine ~1.0 between base and adapter encodings of the same text "
            "would mean the adapter is NOT changing the output"
        ),
    }


def run_score_calibration(tools, vector_store, cases, fixture_ids, k) -> dict:
    """Retrieve each benchmark question and label hit scores by relevance."""
    from src.agent.toolset import _build_search_embedding_text

    per_case = []
    relevant_scores: list[float] = []
    nonrelevant_scores: list[float] = []
    for case in cases:
        question = case["research_question"]
        relevant_bases = {
            _base_id(pid) for pid in case.get("relevant_ids", [])
        } - fixture_ids
        embed_text = _build_search_embedding_text(
            question, question, case.get("constraints")
        )
        query_vec = tools.embedder.encode_queries([embed_text])[0]
        items = vector_store.search(query_vec, k=k)
        hits = []
        for rank, item in enumerate(items, start=1):
            is_relevant = _base_id(item.paper.id) in relevant_bases
            hits.append(
                {
                    "rank": rank,
                    "id": item.paper.id,
                    "score": round(item.score, 6),
                    "relevant": is_relevant,
                }
            )
            (relevant_scores if is_relevant else nonrelevant_scores).append(item.score)
        per_case.append(
            {
                "id": case["id"],
                "best_score": round(items[0].score, 6) if items else None,
                "relevant_hits": [h for h in hits if h["relevant"]],
                "hits": hits,
            }
        )

    def band(scores: list[float]) -> dict:
        if not scores:
            return {"n": 0}
        import numpy as np

        arr = np.asarray(scores)
        return {
            "n": len(scores),
            "min": round(float(arr.min()), 6),
            "p5": round(float(np.percentile(arr, 5)), 6),
            "median": round(float(np.median(arr)), 6),
            "max": round(float(arr.max()), 6),
        }

    rel = band(relevant_scores)
    nonrel = band(nonrelevant_scores)
    # A floor is discriminative if the non-relevant bulk sits below the relevant
    # band's low end. Suggest the midpoint of the gap when one exists; otherwise
    # report the overlap honestly.
    suggested = None
    if rel.get("n") and nonrel.get("n"):
        gap_low, gap_high = nonrel["p5"], rel["min"]
        if gap_high > gap_low:
            suggested = round((gap_low + gap_high) / 2, 4)
    return {
        "relevant_band": rel,
        "nonrelevant_band": nonrel,
        "suggested_floor": suggested,
        "suggested_floor_basis": "midpoint of nonrelevant p5 .. relevant min"
        if suggested is not None
        else "bands overlap; no clean threshold",
        "per_case": per_case,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--qdrant-path",
        default=".local/qdrant-corpus",
        help="Embedded Qdrant corpus to calibrate against.",
    )
    parser.add_argument("--k", type=int, default=10, help="Hits per question.")
    parser.add_argument(
        "--output",
        default=".local/calibration/backfill-floor.json",
        help="Where to write the JSON report.",
    )
    parser.add_argument(
        "--skip-adapter-check",
        action="store_true",
        help="Only run the score calibration.",
    )
    args = parser.parse_args()

    # Pin the store before settings import, same as scripts/seed_corpus.py.
    os.environ["QDRANT_PATH"] = args.qdrant_path
    os.environ["VECTOR_STORE_BACKEND"] = "qdrant"

    from src.agent.tools import ResearchTools
    from src.embeddings import TextEmbedder
    from src.retrieval.store import build_vector_store
    from src.settings import get_settings

    settings = get_settings()
    vector_store = build_vector_store(settings)
    embedder = TextEmbedder(
        settings.embedding_model,
        document_adapter=settings.embedding_document_adapter,
        query_adapter=settings.embedding_query_adapter,
    )

    report: dict = {
        "generated_at": datetime.now(UTC).isoformat(),
        "corpus": args.qdrant_path,
        "corpus_size": vector_store.count(),
        "embedding_model": settings.embedding_model,
        "document_adapter": settings.embedding_document_adapter,
        "query_adapter": settings.embedding_query_adapter,
        "current_floor": settings.agent_search_backfill_min_score,
    }

    if not args.skip_adapter_check:
        print("Adapter activation check (fixed probe texts)...")
        report["adapter_check"] = run_adapter_check(embedder)
        for name, cos in report["adapter_check"]["cosines"].items():
            print(f"  {name}: {cos}")
        print(f"  applied: {report['adapter_check']['adapters_appear_applied']}")

    cases = [
        json.loads(line)
        for line in (ROOT / "evals/benchmarks/research_questions.jsonl")
        .read_text()
        .splitlines()
        if line.strip()
    ]
    fixture_ids = {
        _base_id(json.loads(line)["id"])
        for line in (ROOT / "evals/benchmarks/fixture_papers.jsonl")
        .read_text()
        .splitlines()
        if line.strip()
    }

    tools = ResearchTools(
        settings=settings,
        arxiv_client=None,  # retrieval only; no fetching in this script
        embedder=embedder,
        vector_store=vector_store,
    )
    print(f"\nScore calibration over {len(cases)} benchmark questions (k={args.k})...")
    report["score_calibration"] = run_score_calibration(
        tools, vector_store, cases, fixture_ids, args.k
    )
    sc = report["score_calibration"]
    print(f"  relevant band:     {sc['relevant_band']}")
    print(f"  non-relevant band: {sc['nonrelevant_band']}")
    print(
        f"  suggested floor:   {sc['suggested_floor']} ({sc['suggested_floor_basis']})"
    )

    close = getattr(vector_store, "close", None)
    if callable(close):
        close()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nReport written to {output}")


if __name__ == "__main__":
    main()
