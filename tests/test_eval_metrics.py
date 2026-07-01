from evals import metrics
from src.llm import TurnResult


class FakeJudgeProvider:
    """Returns a canned response so judge parsing/aggregation is deterministic."""

    name = "fake"
    model = "fake-judge"

    def __init__(self, text):
        self._text = text
        self.prompts: list[str] = []

    def run_turn(self, system, messages, tools, *, tool_choice="auto"):
        self.prompts.append(messages[-1].text)
        return TurnResult(
            text=self._text,
            tool_calls=[],
            input_tokens=10,
            output_tokens=5,
            model=self.model,
            stop_reason="end",
        )


def test_extract_citations_dedupes_and_preserves_order():
    brief = "See [2401.00001] and [hep-th/9901001], then [2401.00001] again."
    assert metrics.extract_citations(brief) == ["2401.00001", "hep-th/9901001"]


def test_citation_grounding_flags_hallucinated_and_unread():
    brief = "Method A [2401.00001] beats B [2401.99999]; also [2401.00002]."
    known = {"2401.00001", "2401.00002"}
    read = {"2401.00001"}
    grounding = metrics.citation_grounding(brief, known, read)

    assert grounding.hallucinated == ["2401.99999"]
    assert set(grounding.valid) == {"2401.00001", "2401.00002"}
    assert grounding.read_in_full == ["2401.00001"]
    assert grounding.grounding_rate == 2 / 3
    assert grounding.full_text_rate == 0.5


def test_citation_grounding_ignores_version_suffix():
    grounding = metrics.citation_grounding(
        "Result [2401.00001v2].", {"2401.00001"}, {"2401.00001"}
    )
    assert grounding.hallucinated == []
    assert grounding.read_in_full == ["2401.00001v2"]


def test_uncertainty_signaling_flags_missing_caveats_on_thin_evidence():
    confident = metrics.uncertainty_signaling(
        "We strongly recommend adopting this now.", [], retrieved=1
    )
    assert confident["thin_evidence"] is True
    assert confident["appropriate"] is False

    hedged = metrics.uncertainty_signaling(
        "Evidence is limited; validate before adopting.", [], retrieved=1
    )
    assert hedged["appropriate"] is True


def test_uncertainty_signaling_ok_when_evidence_is_sufficient():
    result = metrics.uncertainty_signaling("Adopt method X.", [], retrieved=6)
    assert result["thin_evidence"] is False
    assert result["appropriate"] is True


def test_evidence_utilization_reports_success_rate():
    result = metrics.evidence_utilization({"attempted": 4, "succeeded": 3})
    assert result["success_rate"] == 0.75
    assert result["read_any_full_text"] is True


def test_score_case_combines_metrics_from_response_payload():
    final = {
        "final_brief": "Use A [2401.00001]; evidence is limited so validate.",
        "full_text_diagnostics": {
            "attempted": 1,
            "succeeded": 1,
            "succeeded_ids": ["2401.00001"],
        },
        "retrieval_diagnostics": {"returned": 1},
        "warnings": [],
    }
    scored = metrics.score_case(final, {"2401.00001"})
    assert scored["citation_grounding"]["hallucinated"] == []
    assert scored["citation_grounding"]["read_in_full"] == ["2401.00001"]
    assert scored["uncertainty_signaling"]["appropriate"] is True


def test_extract_citation_claims_pairs_each_id_with_its_sentence():
    brief = (
        "Transformers scale well [2401.00001]. "
        "RAG improves grounding [2401.00002]. "
        "It also cuts hallucination [2401.00002]."
    )
    pairs = metrics.extract_citation_claims(brief)

    assert {"claim": "Transformers scale well.", "id": "2401.00001"} in pairs
    # Same id in two different claim sentences yields two distinct pairs.
    claims_for = [p["claim"] for p in pairs if p["id"] == "2401.00002"]
    assert len(claims_for) == 2
    # Citation markers are stripped from the claim text.
    assert all("[" not in p["claim"] for p in pairs)


def test_citation_grounding_judge_scores_support():
    brief = "A is true [2401.00001]. B is false [2401.00002]."
    evidence = {
        "2401.00001": {"title": "Paper A", "text": "A is true."},
        "2401.00002": {"title": "Paper B", "text": "Unrelated content."},
    }
    provider = FakeJudgeProvider(
        '[{"index": 0, "verdict": "supported", "reason": "matches"}, '
        '{"index": 1, "verdict": "unsupported", "reason": "no support"}]'
    )
    result = metrics.citation_grounding_judge(provider, brief, evidence)

    assert result["judged"] == 2
    assert result["supported"] == 1
    assert result["unsupported"] == 1
    assert result["grounded_rate"] == 0.5
    assert result["items"][1]["verdict"] == "unsupported"


def test_citation_grounding_judge_credits_partial_at_half():
    brief = "Claim one [2401.00001]. Claim two [2401.00002]."
    evidence = {
        "2401.00001": {"title": "A", "text": "x"},
        "2401.00002": {"title": "B", "text": "y"},
    }
    provider = FakeJudgeProvider(
        '[{"index": 0, "verdict": "supported"}, {"index": 1, "verdict": "partial"}]'
    )
    result = metrics.citation_grounding_judge(provider, brief, evidence)
    assert result["partial"] == 1
    assert result["grounded_rate"] == 0.75


def test_citation_grounding_judge_skips_ids_without_evidence():
    brief = "Real [2401.00001]. Fabricated [9999.99999]."
    evidence = {"2401.00001": {"title": "A", "text": "x"}}
    provider = FakeJudgeProvider('[{"index": 0, "verdict": "supported"}]')
    result = metrics.citation_grounding_judge(provider, brief, evidence)

    assert result["judged"] == 1
    assert result["skipped_ids"] == ["9999.99999"]


def test_citation_grounding_judge_reports_parse_failure():
    brief = "Claim [2401.00001]."
    evidence = {"2401.00001": {"title": "A", "text": "x"}}
    provider = FakeJudgeProvider("sorry, no json here")
    result = metrics.citation_grounding_judge(provider, brief, evidence)
    assert "error" in result


def test_citation_grounding_judge_no_citable_evidence():
    provider = FakeJudgeProvider("[]")
    result = metrics.citation_grounding_judge(provider, "No citations here.", {})
    assert result["judged"] == 0
    assert result["grounded_rate"] is None


def test_aggregate_counts_cases_with_hallucinations():
    rows = [
        metrics.score_case(
            {
                "final_brief": "A [2401.00001]",
                "full_text_diagnostics": {
                    "attempted": 1,
                    "succeeded": 1,
                    "succeeded_ids": ["2401.00001"],
                },
                "retrieval_diagnostics": {"returned": 5},
                "warnings": [],
            },
            {"2401.00001"},
        ),
        metrics.score_case(
            {
                "final_brief": "B [9999.99999]",
                "full_text_diagnostics": {"attempted": 0, "succeeded": 0},
                "retrieval_diagnostics": {"returned": 5},
                "warnings": [],
            },
            {"2401.00001"},
        ),
    ]
    summary = metrics.aggregate(rows)
    assert summary["cases"] == 2
    assert summary["cases_with_hallucinations"] == 1
