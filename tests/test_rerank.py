import numpy as np
import pytest
import torch

from src.models import PaperRecord, SearchResponseItem
from src.rerank import CrossEncoderReranker


def _item(paper_id: str, title: str, score: float = 0.5) -> SearchResponseItem:
    paper = PaperRecord(
        id=paper_id,
        title=title,
        summary=f"Abstract for {title}",
        authors=["A"],
        categories=["cs.LG"],
        primary_category="cs.LG",
        arxiv_url=f"https://arxiv.org/abs/{paper_id}",
    )
    return SearchResponseItem(paper=paper, score=score)


class FakeTokenizer:
    """Encodes each (query, document) pair as a row carrying the doc index."""

    def __init__(self):
        self.calls: list[tuple[list[str], list[str]]] = []

    def __call__(
        self, queries, documents, *, truncation, padding, max_length, return_tensors
    ):
        assert truncation is True
        assert padding is True
        assert max_length == 512
        assert return_tensors == "pt"
        assert queries == [queries[0]] * len(documents)
        self.calls.append((list(queries), list(documents)))
        # Row i encodes index i so the fake model can score per document.
        input_ids = torch.arange(len(documents), dtype=torch.long).unsqueeze(1)
        return {"input_ids": input_ids}


class FakeCrossEncoder:
    """Scores row i with a fixed per-index score, as 2-class logits."""

    def __init__(self, scores: list[float]):
        self._scores = scores
        self._parameter = torch.nn.Parameter(torch.empty(0))

    def parameters(self):
        return iter([self._parameter])

    def eval(self):
        return self

    def __call__(self, **inputs):
        indices = inputs["input_ids"].squeeze(1).tolist()
        rows = [[0.0, self._scores[i]] for i in indices]
        from types import SimpleNamespace

        return SimpleNamespace(logits=torch.tensor(rows, dtype=torch.float32))


def _loaded_reranker(scores: list[float]) -> CrossEncoderReranker:
    reranker = CrossEncoderReranker("fake-cross-encoder")
    reranker._tokenizer = FakeTokenizer()
    reranker._model = FakeCrossEncoder(scores)
    return reranker


def test_rerank_reorders_by_cross_encoder_score_and_updates_item_scores():
    items = [_item("1", "Alpha"), _item("2", "Beta"), _item("3", "Gamma")]
    reranker = _loaded_reranker([0.1, 2.0, 1.0])

    ordered = reranker.rerank("query", items, top_k=3)

    assert [item.paper.id for item in ordered] == ["2", "3", "1"]
    assert [item.score for item in ordered] == pytest.approx([2.0, 1.0, 0.1])


def test_rerank_truncates_to_top_k():
    items = [_item("1", "Alpha"), _item("2", "Beta"), _item("3", "Gamma")]
    reranker = _loaded_reranker([0.1, 2.0, 1.0])

    ordered = reranker.rerank("query", items, top_k=1)

    assert len(ordered) == 1
    assert ordered[0].paper.id == "2"


def test_rerank_returns_empty_for_empty_items_or_nonpositive_top_k():
    reranker = _loaded_reranker([1.0])
    assert reranker.rerank("query", [], top_k=3) == []
    assert reranker.rerank("query", [_item("1", "Alpha")], top_k=0) == []


def test_rerank_flattens_single_logit_outputs():
    items = [_item("1", "Alpha"), _item("2", "Beta")]
    reranker = CrossEncoderReranker("fake-cross-encoder")
    reranker._tokenizer = FakeTokenizer()

    class SingleLogitModel(FakeCrossEncoder):
        def __call__(self, **inputs):
            indices = inputs["input_ids"].squeeze(1).tolist()
            from types import SimpleNamespace

            return SimpleNamespace(
                logits=torch.tensor(
                    [[self._scores[i]] for i in indices], dtype=torch.float32
                )
            )

    reranker._model = SingleLogitModel([3.0, 1.0])
    ordered = reranker.rerank("query", items, top_k=2)
    assert [item.paper.id for item in ordered] == ["1", "2"]


def test_score_builds_query_document_pairs_from_title_and_abstract():
    items = [_item("1", "Alpha", score=0.0)]
    reranker = _loaded_reranker([0.5])

    scores = reranker._score("my query", items)

    queries, documents = reranker._tokenizer.calls[0]
    assert queries == ["my query"]
    assert documents == ["Title: Alpha\nAbstract: Abstract for Alpha"]
    assert isinstance(scores, np.ndarray)
    assert scores.tolist() == [0.5]
