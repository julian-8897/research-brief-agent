import numpy as np

from src import VectorStore
from src.models import PaperRecord
from src.retrieval.store import InMemoryVectorStore, _point_id


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


def test_qdrant_point_id_is_deterministic_uuid():
    assert _point_id("2401.00001") == _point_id("2401.00001")
    assert len(_point_id("2401.00001").split("-")) == 5


def test_legacy_vector_store_export_forwards_add_papers():
    store = VectorStore(embedding_dimension=2)
    store.add_papers(
        [PaperRecord(id="1", title="Legacy", summary="compatibility")],
        np.array([[1.0, 0.0]]),
    )

    assert store.count() == 1
