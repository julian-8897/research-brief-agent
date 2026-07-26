from types import SimpleNamespace

import numpy as np
import pytest

from src import VectorStore
from src.models import PaperRecord
from src.retrieval.store import (
    InMemoryVectorStore,
    QdrantPaperVectorStore,
    _point_id,
)
from src.settings import Settings


def test_in_memory_vector_store_upsert_and_search():
    store = InMemoryVectorStore(embedding_dimension=3)
    papers = [
        PaperRecord(id="1", title="Graph search", summary="Graphs for retrieval"),
        PaperRecord(id="2", title="Vision", summary="Image classifiers"),
    ]
    store.upsert(papers, np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))

    results = store.search(np.array([1.0, 0.0, 0.0]), k=1)

    assert store.count() == 2
    assert results[0].paper.id == "1"
    assert results[0].score > 0.99


def test_in_memory_vector_store_replaces_duplicate_ids():
    store = InMemoryVectorStore(embedding_dimension=2)
    store.upsert(
        [PaperRecord(id="1", title="Old", summary="old")], np.array([[1.0, 0.0]])
    )
    store.upsert(
        [PaperRecord(id="1", title="New", summary="new")], np.array([[0.0, 1.0]])
    )

    results = store.search(np.array([0.0, 1.0]), k=1)

    assert store.count() == 1
    assert results[0].paper.title == "New"


def test_in_memory_vector_store_reports_existing_ids():
    store = InMemoryVectorStore(embedding_dimension=2)
    store.upsert(
        [PaperRecord(id="1", title="One", summary="one")], np.array([[1.0, 0.0]])
    )

    assert store.existing_ids(["1", "2"]) == {"1"}


def test_in_memory_vector_store_scores_requested_ids_outside_global_top_k():
    store = InMemoryVectorStore(embedding_dimension=2)
    store.upsert(
        [
            PaperRecord(id="leader", title="Leader", summary="leader"),
            PaperRecord(id="fresh", title="Fresh", summary="fresh"),
        ],
        np.array([[1.0, 0.0], [0.5, 0.8660254]]),
    )

    global_results = store.search(np.array([1.0, 0.0]), k=1)
    requested_results = store.search_ids(
        np.array([1.0, 0.0]), ["fresh", "missing"], k=1
    )

    assert [item.paper.id for item in global_results] == ["leader"]
    assert [item.paper.id for item in requested_results] == ["fresh"]
    assert requested_results[0].score == pytest.approx(0.5)


def test_qdrant_point_id_is_deterministic_uuid():
    assert _point_id("2401.00001") == _point_id("2401.00001")
    assert len(_point_id("2401.00001").split("-")) == 5


def test_qdrant_vector_store_scores_requested_ids():
    store = object.__new__(QdrantPaperVectorStore)
    store.settings = Settings(
        vector_store_backend="qdrant",
        embedding_dimension=2,
        qdrant_collection="arxiv_papers",
    )
    store._collection_validated = True
    store.client = SimpleNamespace(
        retrieve=lambda **_kwargs: [
            SimpleNamespace(
                payload=PaperRecord(
                    id="fresh", title="Fresh", summary="fresh"
                ).model_dump(mode="json"),
                vector=[0.5, 0.8660254],
            )
        ]
    )

    results = store.search_ids(np.array([1.0, 0.0]), ["fresh"], k=1)

    assert [item.paper.id for item in results] == ["fresh"]
    assert results[0].score == pytest.approx(0.5)


def test_qdrant_collection_schema_accepts_expected_vectors():
    store = object.__new__(QdrantPaperVectorStore)
    store.settings = Settings(
        vector_store_backend="qdrant",
        embedding_dimension=768,
        qdrant_collection="arxiv_papers",
    )

    store._validate_collection_vectors(
        SimpleNamespace(size=768, distance=SimpleNamespace(value="Cosine"))
    )


@pytest.mark.parametrize(
    ("vectors", "message"),
    [
        (
            SimpleNamespace(size=384, distance=SimpleNamespace(value="Cosine")),
            "size 384; expected 768",
        ),
        (
            SimpleNamespace(size=768, distance=SimpleNamespace(value="Dot")),
            "distance Dot; expected Cosine",
        ),
        ({"title": SimpleNamespace(size=768)}, "uses named vectors"),
    ],
)
def test_qdrant_collection_schema_rejects_incompatible_vectors(vectors, message):
    store = object.__new__(QdrantPaperVectorStore)
    store.settings = Settings(
        vector_store_backend="qdrant",
        embedding_dimension=768,
        qdrant_collection="arxiv_papers",
    )

    with pytest.raises(RuntimeError, match=message):
        store._validate_collection_vectors(vectors)


def test_legacy_vector_store_export_forwards_add_papers():
    store = VectorStore(embedding_dimension=2)
    store.add_papers(
        [PaperRecord(id="1", title="Legacy", summary="compatibility")],
        np.array([[1.0, 0.0]]),
    )

    assert store.count() == 1
