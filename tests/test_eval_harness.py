import json
from pathlib import Path

import numpy as np

from evals.run_eval import (
    DeterministicEmbedder,
    fixture_coverage_failures,
    load_cases,
    quality_gate_failures,
)

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "evals/benchmarks/research_questions.jsonl"
FIXTURE_PAPERS = ROOT / "evals/benchmarks/fixture_papers.jsonl"


def _fixture_ids() -> set[str]:
    with FIXTURE_PAPERS.open(encoding="utf-8") as handle:
        return {json.loads(line)["id"] for line in handle if line.strip()}


def test_offline_fixture_covers_every_benchmark_case():
    fixture_ids = _fixture_ids()
    missing = {
        case["id"]: case["relevant_ids"]
        for case in load_cases(CASES)
        if fixture_ids.isdisjoint(case.get("relevant_ids", []))
    }

    assert missing == {}, (
        "Each benchmark case needs at least one relevant paper in the offline fixture; "
        f"missing coverage for: {missing}"
    )


def test_deterministic_embedder_preserves_lexical_similarity():
    embedder = DeterministicEmbedder()
    query = embedder.encode_queries(["four bit quantization for LLM inference"])[0]
    related, unrelated = embedder.encode_documents(
        [
            "Four-bit LLM quantization reduces inference memory.",
            "Physics-informed neural networks solve partial differential equations.",
        ]
    )

    assert np.dot(query, related) > np.dot(query, unrelated)


def test_fixture_coverage_failure_names_the_uncovered_case():
    cases = [{"id": "quantization", "relevant_ids": ["2403.01384v2"]}]

    assert fixture_coverage_failures(cases, {"different-paper"}) == [
        "quantization: expected one of ['2403.01384v2']"
    ]


def test_quality_gate_accepts_grounded_zero_warning_retrieval_hit():
    row = {
        "id": "case-a",
        "final": {"warnings": []},
        "metrics": {
            "citation_grounding": {
                "grounding_rate": 1.0,
                "hallucinated": [],
            },
            "uncertainty_signaling": {"appropriate": True},
        },
        "retrieval_eval": {"hits": 1, "k": 3},
    }

    assert quality_gate_failures([row]) == []


def test_quality_gate_reports_actionable_case_failures():
    row = {
        "id": "case-a",
        "final": {"warnings": ["thin evidence"]},
        "metrics": {
            "citation_grounding": {
                "grounding_rate": 0.5,
                "hallucinated": ["9999.99999"],
            },
            "uncertainty_signaling": {"appropriate": False},
        },
        "retrieval_eval": {"hits": 0, "k": 3},
    }

    failures = quality_gate_failures([row])

    assert len(failures) == 5
    assert all(failure.startswith("case-a:") for failure in failures)
    assert any("no relevant paper" in failure for failure in failures)


def test_quality_gate_rejects_empty_run():
    assert quality_gate_failures([]) == ["no evaluation cases ran"]
