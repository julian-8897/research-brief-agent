from types import SimpleNamespace

import pytest
import torch

from src.embeddings import (
    TextEmbedder,
    build_paper_embedding_text,
    validate_adapter_compatibility,
)


def test_build_paper_embedding_text_uses_title_and_abstract():
    text = build_paper_embedding_text(
        {"title": "A Useful Method", "summary": "We evaluate the method."}
    )
    assert text == "Title: A Useful Method\nAbstract: We evaluate the method."


class FakeTokenizer:
    def __call__(self, texts, *, truncation, max_length, padding, return_tensors):
        assert truncation is True
        assert max_length == 512
        assert padding is True
        assert return_tensors == "pt"
        return {
            "input_ids": torch.ones((len(texts), 3), dtype=torch.long),
            "attention_mask": torch.ones((len(texts), 3), dtype=torch.long),
        }


class FakeAdapterModel:
    def __init__(self):
        self.activated: list[str] = []
        self._parameter = torch.nn.Parameter(torch.empty(0))

    def parameters(self):
        return iter([self._parameter])

    def set_active_adapters(self, adapter_name):
        self.activated.append(adapter_name)

    def __call__(self, **inputs):
        batch_size = inputs["input_ids"].shape[0]
        hidden_state = torch.zeros((batch_size, 3, 768), dtype=torch.float32)
        return SimpleNamespace(last_hidden_state=hidden_state)


def test_text_embedder_uses_distinct_query_and_document_adapters():
    embedder = TextEmbedder(
        "base",
        document_adapter="document-repo",
        query_adapter="query-repo",
        document_adapter_name="proximity",
        query_adapter_name="adhoc_query",
    )
    fake_model = FakeAdapterModel()
    embedder._model = fake_model
    embedder._tokenizer = FakeTokenizer()

    query_embeddings = embedder.encode_queries(["natural language query"])
    document_embeddings = embedder.encode_documents(["Title: Paper\nAbstract: Body"])

    assert fake_model.activated == ["adhoc_query", "proximity"]
    assert query_embeddings.shape == (1, 768)
    assert document_embeddings.shape == (1, 768)


def test_validate_adapter_compatibility_accepts_specter2_pairing():
    validate_adapter_compatibility(
        "allenai/specter2_base", "allenai/specter2", "allenai/specter2_adhoc_query"
    )


def test_validate_adapter_compatibility_accepts_non_specter2_custom_pairing():
    validate_adapter_compatibility("some/base-model", "org/doc-adapter", "org/query")


def test_validate_adapter_compatibility_accepts_renamed_specter2_adapters():
    validate_adapter_compatibility(
        "allenai/specter2_base", "local/document-adapter", "/models/query-adapter"
    )


def test_validate_adapter_compatibility_does_not_infer_from_unknown_names():
    validate_adapter_compatibility(
        "/models/renamed-base", "allenai/specter2", "allenai/specter2_adhoc_query"
    )


def test_validate_adapter_compatibility_rejects_specter2_adapters_on_other_base():
    with pytest.raises(ValueError, match="require a SPECTER2 base model"):
        validate_adapter_compatibility(
            "sentence-transformers/allenai-specter",
            "allenai/specter2",
            "allenai/specter2_adhoc_query",
        )


def test_text_embedder_rejects_mismatched_configuration_at_construction():
    with pytest.raises(ValueError, match="require a SPECTER2 base model"):
        TextEmbedder(
            "sentence-transformers/allenai-specter",
            document_adapter="allenai/specter2",
            query_adapter="allenai/specter2_adhoc_query",
        )
