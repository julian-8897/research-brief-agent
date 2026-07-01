"""Automated quality metrics for research-brief evaluation.

The deterministic metrics here turn the previously manual "judge checks"
checklist into computed scores that run on every eval, with no API key or
network needed:

- citation grounding: are inline ``[arxiv_id]`` citations real (present in the
  known corpus) and were the cited papers actually read in full?
- evidence utilization: did the agent read full text rather than stopping at
  abstracts?
- uncertainty signaling: when evidence is thin, does the brief flag it?

The optional :func:`faithfulness_judge` adds an LLM-graded faithfulness and
answer-relevance score; it is only invoked when a provider is configured.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from src.llm import LLMProvider, UserMessage

# arXiv ids: modern (2401.00001, optional version) and legacy (hep-th/9901001).
_CITATION_RE = re.compile(
    r"\[(\d{4}\.\d{4,5}(?:v\d+)?|[a-z][a-z\-]+(?:\.[A-Z]{2})?/\d{7})\]"
)

_UNCERTAINTY_MARKERS = (
    "uncertain",
    "uncertainty",
    "limited",
    "thin evidence",
    "not enough",
    "insufficient",
    "cannot conclude",
    "inconclusive",
    "caveat",
    "further",
    "follow-up",
    "validate",
    "validation",
    "preliminary",
    "unclear",
    "triage",
    "scoping",
    "abstain",
)


def _base_id(arxiv_id: str) -> str:
    """Strip a trailing version suffix so 2401.1v2 and 2401.1 compare equal."""
    return re.sub(r"v\d+$", "", arxiv_id)


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def extract_citations(brief: str) -> list[str]:
    """Return inline ``[arxiv_id]`` citations in first-seen order, de-duplicated."""
    seen: set[str] = set()
    ordered: list[str] = []
    for match in _CITATION_RE.findall(brief or ""):
        if match not in seen:
            seen.add(match)
            ordered.append(match)
    return ordered


def extract_citation_claims(brief: str) -> list[dict[str, str]]:
    """Pair each cited id with the claim sentence it is attached to.

    Splits the brief into sentences, and for every sentence that carries one or
    more ``[arxiv_id]`` citations emits one ``{"claim", "id"}`` pair per cited id
    (citation markers stripped from the claim text). Pairs are de-duplicated.
    """
    pairs: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for line in (brief or "").split("\n"):
        line = line.strip()
        if not line:
            continue
        for sentence in _SENTENCE_SPLIT.split(line):
            ids = _CITATION_RE.findall(sentence)
            if not ids:
                continue
            claim = re.sub(r"\s+", " ", _CITATION_RE.sub("", sentence)).strip()
            claim = re.sub(r"\s+([.,;:!?])", r"\1", claim)
            if not claim:
                continue
            for cid in ids:
                key = (claim, cid)
                if key not in seen:
                    seen.add(key)
                    pairs.append({"claim": claim, "id": cid})
    return pairs


@dataclass
class CitationGrounding:
    cited: list[str] = field(default_factory=list)
    valid: list[str] = field(default_factory=list)
    hallucinated: list[str] = field(default_factory=list)
    read_in_full: list[str] = field(default_factory=list)

    @property
    def grounding_rate(self) -> float:
        return len(self.valid) / len(self.cited) if self.cited else 1.0

    @property
    def hallucination_rate(self) -> float:
        return len(self.hallucinated) / len(self.cited) if self.cited else 0.0

    @property
    def full_text_rate(self) -> float:
        """Fraction of validly cited papers whose full text was actually read."""
        return len(self.read_in_full) / len(self.valid) if self.valid else 0.0

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["grounding_rate"] = round(self.grounding_rate, 4)
        data["hallucination_rate"] = round(self.hallucination_rate, 4)
        data["full_text_rate"] = round(self.full_text_rate, 4)
        return data


def citation_grounding(
    brief: str,
    known_ids: set[str],
    full_text_ids: set[str] | None = None,
) -> CitationGrounding:
    """Grade inline citations against the corpus and the papers read in full.

    ``known_ids`` is the set of legitimate ids (the ingested/retrieved corpus);
    a cited id outside it is a fabrication. ``full_text_ids`` are the ids whose
    body text was successfully read. Comparison ignores version suffixes.
    """
    known_bases = {_base_id(i) for i in known_ids}
    read_bases = {_base_id(i) for i in (full_text_ids or set())}
    result = CitationGrounding(cited=extract_citations(brief))
    for cid in result.cited:
        base = _base_id(cid)
        if base in known_bases:
            result.valid.append(cid)
            if base in read_bases:
                result.read_in_full.append(cid)
        else:
            result.hallucinated.append(cid)
    return result


def evidence_utilization(full_text_diagnostics: dict[str, Any]) -> dict[str, Any]:
    diagnostics = full_text_diagnostics or {}
    attempted = int(diagnostics.get("attempted", 0))
    succeeded = int(diagnostics.get("succeeded", 0))
    return {
        "attempted": attempted,
        "succeeded": succeeded,
        "success_rate": round(succeeded / attempted, 4) if attempted else 0.0,
        "read_any_full_text": succeeded > 0,
    }


def uncertainty_signaling(
    brief: str,
    warnings: list[str],
    retrieved: int,
    thin_threshold: int = 4,
) -> dict[str, Any]:
    """Check the brief flags uncertainty when the evidence base is thin."""
    text = (brief or "").lower()
    has_language = any(marker in text for marker in _UNCERTAINTY_MARKERS)
    flagged_by_warning = any("thin" in w.lower() for w in (warnings or []))
    thin = retrieved < thin_threshold
    signaled = has_language or flagged_by_warning
    return {
        "thin_evidence": thin,
        "signaled_uncertainty": signaled,
        # Appropriate when thin evidence is acknowledged, or when it is not thin.
        "appropriate": signaled if thin else True,
    }


def score_case(final: dict[str, Any], known_ids: set[str]) -> dict[str, Any]:
    """Compute all deterministic metrics for one brief response payload."""
    brief = final.get("final_brief", "")
    full_text = final.get("full_text_diagnostics", {}) or {}
    diagnostics = final.get("retrieval_diagnostics", {}) or {}
    warnings = final.get("warnings", []) or []
    read_ids = set(full_text.get("succeeded_ids", []))
    grounding = citation_grounding(brief, known_ids, read_ids)
    return {
        "citation_grounding": grounding.as_dict(),
        "evidence_utilization": evidence_utilization(full_text),
        "uncertainty_signaling": uncertainty_signaling(
            brief, warnings, int(diagnostics.get("returned", 0))
        ),
    }


def aggregate(metric_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Roll per-case deterministic metrics up to corpus-level averages."""
    if not metric_rows:
        return {}
    n = len(metric_rows)

    def mean(path: list[str]) -> float:
        total = 0.0
        for row in metric_rows:
            node: Any = row
            for key in path:
                node = node.get(key, {}) if isinstance(node, dict) else 0
            total += float(node or 0)
        return round(total / n, 4)

    hallucinating = sum(
        1
        for row in metric_rows
        if row["citation_grounding"]["hallucinated"]
    )
    return {
        "cases": n,
        "mean_grounding_rate": mean(["citation_grounding", "grounding_rate"]),
        "mean_hallucination_rate": mean(["citation_grounding", "hallucination_rate"]),
        "cases_with_hallucinations": hallucinating,
        "mean_full_text_rate": mean(["citation_grounding", "full_text_rate"]),
        "mean_full_text_success": mean(["evidence_utilization", "success_rate"]),
        "uncertainty_appropriate_rate": mean(
            ["uncertainty_signaling", "appropriate"]
        ),
    }


# -- Optional LLM-graded faithfulness -------------------------------------------

_JUDGE_SYSTEM = (
    "You are a strict evaluator of AI-written research briefs. You are given a "
    "research question, a brief, and the evidence snippets the brief was allowed "
    "to use. Judge only against the supplied evidence; treat outside knowledge as "
    "unsupported. Respond with ONLY a JSON object, no prose."
)


def _judge_prompt(question: str, brief: str, evidence: list[dict[str, Any]]) -> str:
    evidence_block = "\n".join(
        f"[{item['id']}] {item.get('title', '')}\n{item.get('text', '')}"
        for item in evidence
    )
    return (
        f"Research question:\n{question}\n\n"
        f"Evidence snippets (the only allowed support):\n{evidence_block}\n\n"
        f"Brief to evaluate:\n{brief}\n\n"
        "Return JSON with exactly these keys:\n"
        '{"total_claims": int, "supported_claims": int, '
        '"faithfulness": float 0-1 (supported/total), '
        '"answer_relevance": float 0-1 (how well the brief answers the question), '
        '"unsupported_examples": [up to 3 short strings]}'
    )


def _extract_json(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _extract_json_array(text: str) -> list[Any] | None:
    if not text:
        return None
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, list) else None


def faithfulness_judge(
    provider: LLMProvider,
    research_question: str,
    brief: str,
    evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    """Grade faithfulness and answer-relevance with an LLM judge.

    ``evidence`` items are ``{"id", "title", "text"}``. Returns the parsed
    verdict plus measured token usage, or an ``{"error": ...}`` payload if the
    provider response could not be parsed.
    """
    turn = provider.run_turn(
        _JUDGE_SYSTEM,
        [UserMessage(_judge_prompt(research_question, brief, evidence))],
        [],
        tool_choice="none",
    )
    verdict = _extract_json(turn.text or "")
    usage = {
        "input_tokens": turn.input_tokens,
        "output_tokens": turn.output_tokens,
        "model": turn.model,
    }
    if verdict is None:
        return {"error": "could not parse judge response", "usage": usage}
    verdict["usage"] = usage
    return verdict


# -- LLM-graded per-citation grounding ------------------------------------------

_GROUNDING_JUDGE_SYSTEM = (
    "You verify whether each claim in a research brief is actually supported by "
    "the specific paper it cites. Judge ONLY against the evidence provided for "
    "that claim's cited paper; treat any outside knowledge as unsupported. "
    "Respond with ONLY a JSON array, no prose."
)

# Verdicts that count as the claim being grounded in its citation.
_GROUNDED_VERDICTS = {"supported"}
_PARTIAL_VERDICTS = {"partial"}


def _grounding_prompt(
    items: list[dict[str, str]], evidence_by_id: dict[str, dict[str, Any]]
) -> str:
    blocks = []
    for index, item in enumerate(items):
        evidence = evidence_by_id.get(item["id"], {})
        blocks.append(
            f"Item {index}:\n"
            f"Claim: {item['claim']}\n"
            f"Cited paper [{item['id']}] {evidence.get('title', '')}\n"
            f"Evidence: {evidence.get('text', '(no evidence available)')}"
        )
    return (
        "\n\n".join(blocks) + "\n\n"
        "For each item, decide whether the cited paper's evidence supports the "
        "claim. Return a JSON array with one object per item, each with keys:\n"
        '{"index": int, "verdict": "supported" | "partial" | "unsupported" | '
        '"no_info", "reason": short string}.'
    )


def citation_grounding_judge(
    provider: LLMProvider,
    brief: str,
    evidence_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Grade whether each inline citation actually supports its claim.

    Complements the deterministic :func:`citation_grounding` (which only checks
    that a cited id exists and was read) by asking an LLM whether the cited
    paper's evidence semantically supports the sentence it is attached to.
    ``evidence_by_id`` maps arXiv id to ``{"title", "text"}``. Ids without
    evidence (e.g. fabricated citations) are reported as skipped, not judged.
    """
    pairs = extract_citation_claims(brief)
    judged = [pair for pair in pairs if pair["id"] in evidence_by_id]
    skipped = [pair["id"] for pair in pairs if pair["id"] not in evidence_by_id]
    if not judged:
        return {
            "judged": 0,
            "skipped_ids": skipped,
            "items": [],
            "grounded_rate": None,
        }

    turn = provider.run_turn(
        _GROUNDING_JUDGE_SYSTEM,
        [UserMessage(_grounding_prompt(judged, evidence_by_id))],
        [],
        tool_choice="none",
    )
    usage = {
        "input_tokens": turn.input_tokens,
        "output_tokens": turn.output_tokens,
        "model": turn.model,
    }
    verdicts = _extract_json_array(turn.text or "")
    if verdicts is None:
        return {"error": "could not parse judge response", "usage": usage}

    by_index: dict[int, dict[str, Any]] = {}
    for entry in verdicts:
        if isinstance(entry, dict) and isinstance(entry.get("index"), int):
            by_index[entry["index"]] = entry

    items: list[dict[str, Any]] = []
    supported = partial = 0
    for index, pair in enumerate(judged):
        entry = by_index.get(index, {})
        verdict = str(entry.get("verdict", "no_info")).strip().lower()
        if verdict in _GROUNDED_VERDICTS:
            supported += 1
        elif verdict in _PARTIAL_VERDICTS:
            partial += 1
        items.append(
            {
                "id": pair["id"],
                "claim": pair["claim"],
                "verdict": verdict,
                "reason": entry.get("reason", ""),
            }
        )

    total = len(judged)
    return {
        "judged": total,
        "skipped_ids": skipped,
        "supported": supported,
        "partial": partial,
        "unsupported": total - supported - partial,
        # Strict: only full support counts. Partial credited at half.
        "grounded_rate": round((supported + 0.5 * partial) / total, 4),
        "items": items,
        "usage": usage,
    }
