"""Seed a persistent embedded-Qdrant corpus for the benchmark questions.

Builds the evidence corpus the eval benchmark retrieves against. Each entry maps
a benchmark question id (see evals/benchmarks/research_questions.jsonl) to a
topical arXiv query, so every question has real, on-topic papers in the corpus
while the other topics act as retrieval distractors. Ingestion dedupes by arXiv
id before embedding, so re-running is safe and cheap.

No LLM is called: this is arXiv fetch + SPECTER embedding only.

Usage:
    uv run python scripts/seed_corpus.py --dry-run           # print the manifest, fetch nothing
    uv run python scripts/seed_corpus.py                     # ingest into .local/qdrant-corpus
    uv run python scripts/seed_corpus.py --per-question 25   # override papers per query
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# One topical arXiv query per benchmark question id. Field prefixes: abs:/ti:/cat:.
# Kept deliberately broad (OR-ed synonyms) so a single flaky term does not empty a
# query; off-topic hits are fine because they serve as distractors for other questions.
QUERY_MANIFEST: dict[str, str] = {
    # --- AI/ML engineering ---
    "agentic_rag": '(abs:"retrieval-augmented generation" OR abs:RAG) AND (abs:agentic OR abs:agent OR abs:"tool use" OR abs:"iterative retrieval" OR abs:"multi-hop" OR abs:"active retrieval") AND cat:cs.CL',
    "rag_vs_finetune_vs_longcontext": '(abs:"retrieval-augmented generation" OR abs:"retrieval augmented") AND (abs:"fine-tuning" OR abs:"long context") AND cat:cs.CL',
    "lora_vs_full_finetune": '(abs:LoRA OR abs:"low-rank adaptation" OR abs:"parameter-efficient fine-tuning" OR abs:PEFT) AND (abs:"full fine-tuning" OR abs:"fine-tuning" OR abs:"instruction tuning") AND (cat:cs.LG OR cat:cs.CL)',
    "quantization_inference": 'abs:quantization AND (abs:"large language model" OR abs:LLM OR abs:inference) AND cat:cs.LG',
    "hybrid_vs_dense_retrieval": '(abs:"dense retrieval" OR abs:ColBERT OR abs:"late interaction" OR abs:"hybrid retrieval" OR abs:BM25) AND cat:cs.IR',
    # Comparison-question balance: pair uncertainty estimation with the decision/communication side.
    "uncertainty_communication": '((abs:calibration OR abs:"predictive uncertainty" OR abs:"uncertainty quantification" OR abs:"confidence estimation") AND (abs:"decision making" OR abs:"decision support" OR abs:communication OR abs:interpretability OR abs:"human-AI")) AND (cat:cs.LG OR cat:cs.HC OR cat:stat.ML)',
    # --- Fundamental / theoretical ML ---
    "adam_vs_sgd": '(abs:Adam OR abs:AdamW OR abs:"adaptive methods") AND (abs:SGD OR abs:"stochastic gradient") AND cat:cs.LG',
    "compute_optimal_scaling": 'abs:"scaling laws" AND (abs:"compute-optimal" OR abs:"compute optimal" OR abs:"training tokens" OR abs:"model size") AND cat:cs.LG',
    "flat_minima_sam": '(abs:"sharpness-aware" OR abs:"flat minima" OR abs:sharpness) AND abs:generalization AND cat:cs.LG',
    "double_descent_overparam": '(abs:"double descent" OR abs:overparameterization OR abs:"over-parameterized") AND abs:generalization AND cat:stat.ML',
    # Require BOTH paradigms named in the abstract so hits are comparisons, not
    # single-method application papers; cs.LG only (cs.CV floods with vision-MAE apps).
    "contrastive_vs_generative_ssl": 'abs:"self-supervised" AND abs:contrastive AND (abs:generative OR abs:"masked image modeling" OR abs:"masked autoencoders" OR abs:"masked prediction") AND cat:cs.LG',
    # --- Scientific ML ---
    # Comparison-question balance: force in classical-solver baselines and failure-mode papers.
    "pinns_vs_solvers": '(abs:"physics-informed neural networks" OR abs:PINNs) AND (abs:solver OR abs:"finite element" OR abs:"finite difference" OR abs:"spectral method" OR abs:"failure mode" OR abs:stiff OR abs:benchmark) AND (cat:physics.comp-ph OR cat:cs.LG OR cat:math.NA)',
    # Comparison-question balance: add the classical-surrogate side (POD, reduced-order, GP).
    "neural_operators_vs_surrogates": '(abs:"neural operator" OR abs:"Fourier neural operator" OR abs:DeepONet OR abs:"operator learning") AND (abs:surrogate OR abs:"reduced order" OR abs:POD OR abs:"Gaussian process" OR abs:simulation) AND (cat:cs.LG OR cat:math.NA OR cat:physics.comp-ph)',
    "anomaly_detection_sensor": 'abs:"anomaly detection" AND (abs:"time series" OR abs:sensor OR abs:"multivariate time series") AND (abs:classical OR abs:statistical OR abs:"deep learning" OR abs:autoencoder OR abs:transformer OR abs:benchmark) AND (cat:cs.LG OR cat:stat.ML OR cat:eess.SP)',
    "equation_discovery_sindy": '(abs:"symbolic regression" OR abs:"governing equations" OR abs:SINDy OR abs:"sparse identification") AND cat:cs.LG',
    "uq_ensembles_vs_gp": '(abs:"deep ensemble" OR abs:"deep ensembles" OR abs:"Gaussian process" OR abs:"Bayesian neural network") AND (abs:uncertainty OR abs:"uncertainty quantification" OR abs:calibration OR abs:surrogate) AND (cat:stat.ML OR cat:cs.LG)',
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seed the benchmark evidence corpus from arXiv."
    )
    parser.add_argument(
        "--qdrant-path",
        default=".local/qdrant-corpus",
        help="Embedded Qdrant storage directory (persistent, no server needed).",
    )
    parser.add_argument(
        "--per-question",
        type=int,
        default=25,
        help="Max papers to fetch per benchmark question query.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the query manifest and exit without fetching or embedding.",
    )
    args = parser.parse_args()

    if args.dry_run:
        print(
            f"Manifest: {len(QUERY_MANIFEST)} queries, {args.per_question} papers each"
        )
        print(f"Target store: {args.qdrant_path} (embedded Qdrant)\n")
        for qid, query in QUERY_MANIFEST.items():
            print(f"[{qid}]\n    {query}\n")
        return

    # Force embedded Qdrant before settings import (Settings reads env at import
    # time). Both the path and the backend must be pinned: a local .env with
    # VECTOR_STORE_BACKEND=memory would otherwise be loaded and silently send the
    # seed into an in-memory store that is discarded at process exit. These are
    # set before import and python-dotenv does not override existing env vars, so
    # the seed always writes to the persistent embedded corpus.
    os.environ["QDRANT_PATH"] = args.qdrant_path
    os.environ["VECTOR_STORE_BACKEND"] = "qdrant"

    from src.agent.tools import ResearchTools
    from src.arxiv_client import ArxivClient
    from src.embeddings import TextEmbedder
    from src.retrieval.store import build_vector_store
    from src.settings import get_settings

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

    print(f"Seeding {len(QUERY_MANIFEST)} queries into {args.qdrant_path}\n")
    total_new = 0
    for qid, query in QUERY_MANIFEST.items():
        try:
            new_count, papers = tools.fetch_and_ingest(
                query, max_papers=args.per_question
            )
        except Exception as exc:  # arXiv API is flaky; log and keep going.
            print(f"  [{qid}] ERROR: {exc}")
            continue
        total_new += new_count
        print(
            f"  [{qid}] fetched {len(papers):>3} | new {new_count:>3} | corpus {vector_store.count():>4}"
        )

    final_count = vector_store.count()
    # Embedded local Qdrant flushes to disk on close; without this the in-memory
    # writes are lost when the process exits (interpreter finalization does not
    # reliably run the client's destructor).
    close = getattr(vector_store, "close", None)
    if callable(close):
        close()

    print(
        f"\nDone. {total_new} new papers embedded; corpus now holds {final_count} papers."
    )


if __name__ == "__main__":
    main()
