import hashlib
import math
import uuid
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from src.models import PaperRecord, SearchResponseItem
from src.settings import Settings


def _as_payload(paper: PaperRecord | dict[str, Any]) -> dict[str, Any]:
    if isinstance(paper, PaperRecord):
        return paper.model_dump(mode="json")
    payload = dict(paper)
    for key in ("published", "updated"):
        value = payload.get(key)
        if hasattr(value, "isoformat"):
            payload[key] = value.isoformat()
    return payload


def _point_id(arxiv_id: str) -> str:
    digest = hashlib.sha1(arxiv_id.encode("utf-8")).hexdigest()
    return str(uuid.UUID(digest[:32]))


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    vectors = vectors.astype("float32")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return vectors / norms


class PaperVectorStore(ABC):
    backend_name: str

    @abstractmethod
    def ensure_collection(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def upsert(
        self, papers: list[PaperRecord | dict[str, Any]], embeddings: np.ndarray
    ) -> int:
        raise NotImplementedError

    @abstractmethod
    def search(
        self, query_embedding: np.ndarray, k: int = 10
    ) -> list[SearchResponseItem]:
        raise NotImplementedError

    @abstractmethod
    def existing_ids(self, ids: list[str]) -> set[str]:
        raise NotImplementedError

    @abstractmethod
    def count(self) -> int:
        raise NotImplementedError


class InMemoryVectorStore(PaperVectorStore):
    backend_name = "memory"

    def __init__(self, embedding_dimension: int = 768):
        self.embedding_dimension = embedding_dimension
        self._papers: list[PaperRecord] = []
        self._vectors: np.ndarray | None = None

    def ensure_collection(self) -> None:
        return None

    def upsert(
        self, papers: list[PaperRecord | dict[str, Any]], embeddings: np.ndarray
    ) -> int:
        if len(papers) != len(embeddings):
            raise ValueError("papers and embeddings must have the same length")
        normalized = _normalize_rows(np.asarray(embeddings))
        parsed = [
            paper if isinstance(paper, PaperRecord) else PaperRecord(**paper)
            for paper in papers
        ]
        existing = {paper.id: idx for idx, paper in enumerate(self._papers)}
        for paper, vector in zip(parsed, normalized, strict=True):
            if paper.id in existing:
                idx = existing[paper.id]
                self._papers[idx] = paper
                if self._vectors is not None:
                    self._vectors[idx] = vector
            else:
                self._papers.append(paper)
                if self._vectors is None:
                    self._vectors = vector.reshape(1, -1)
                else:
                    self._vectors = np.vstack([self._vectors, vector])
        return len(parsed)

    def search(
        self, query_embedding: np.ndarray, k: int = 10
    ) -> list[SearchResponseItem]:
        if self._vectors is None or not self._papers:
            return []
        query = np.asarray(query_embedding).reshape(1, -1).astype("float32")
        query = _normalize_rows(query)[0]
        scores = self._vectors @ query
        top_indices = np.argsort(scores)[::-1][:k]
        return [
            SearchResponseItem(paper=self._papers[idx], score=float(scores[idx]))
            for idx in top_indices
            if math.isfinite(float(scores[idx]))
        ]

    def existing_ids(self, ids: list[str]) -> set[str]:
        candidates = set(ids)
        return {paper.id for paper in self._papers if paper.id in candidates}

    def count(self) -> int:
        return len(self._papers)


class QdrantPaperVectorStore(PaperVectorStore):
    backend_name = "qdrant"

    def __init__(self, settings: Settings):
        self.settings = settings
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http.models import Distance, VectorParams
        except ImportError as exc:
            raise RuntimeError(
                "qdrant-client is not installed. Install dependencies or set "
                "VECTOR_STORE_BACKEND=memory."
            ) from exc

        self._models = {"Distance": Distance, "VectorParams": VectorParams}
        if settings.qdrant_path:
            # Embedded local mode: no server, persists on disk in-process.
            # url/api_key are mutually exclusive with path in qdrant-client.
            self.client = QdrantClient(path=settings.qdrant_path)
        else:
            self.client = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key,
                prefer_grpc=False,
            )

    def ensure_collection(self) -> None:
        collections = self.client.get_collections().collections
        if any(item.name == self.settings.qdrant_collection for item in collections):
            return
        self.client.create_collection(
            collection_name=self.settings.qdrant_collection,
            vectors_config=self._models["VectorParams"](
                size=self.settings.embedding_dimension,
                distance=self._models["Distance"].COSINE,
            ),
        )

    def upsert(
        self, papers: list[PaperRecord | dict[str, Any]], embeddings: np.ndarray
    ) -> int:
        if len(papers) != len(embeddings):
            raise ValueError("papers and embeddings must have the same length")
        self.ensure_collection()
        from qdrant_client.http.models import PointStruct

        vectors = _normalize_rows(np.asarray(embeddings))
        points = []
        for paper, vector in zip(papers, vectors, strict=True):
            payload = _as_payload(paper)
            points.append(
                PointStruct(
                    id=_point_id(str(payload["id"])),
                    vector=vector.tolist(),
                    payload=payload,
                )
            )
        self.client.upsert(
            collection_name=self.settings.qdrant_collection, points=points
        )
        return len(points)

    def search(
        self, query_embedding: np.ndarray, k: int = 10
    ) -> list[SearchResponseItem]:
        self.ensure_collection()
        query = _normalize_rows(np.asarray(query_embedding).reshape(1, -1))[0]
        if hasattr(self.client, "search"):
            hits = self.client.search(
                collection_name=self.settings.qdrant_collection,
                query_vector=query.tolist(),
                limit=k,
                with_payload=True,
            )
        else:
            hits = self.client.query_points(
                collection_name=self.settings.qdrant_collection,
                query=query.tolist(),
                limit=k,
                with_payload=True,
            ).points
        return [
            SearchResponseItem(paper=PaperRecord(**hit.payload), score=float(hit.score))
            for hit in hits
            if hit.payload
        ]

    def existing_ids(self, ids: list[str]) -> set[str]:
        self.ensure_collection()
        requested = {paper_id for paper_id in ids if paper_id}
        if not requested:
            return set()
        point_ids = [_point_id(paper_id) for paper_id in requested]
        points = self.client.retrieve(
            collection_name=self.settings.qdrant_collection,
            ids=point_ids,
            with_payload=True,
            with_vectors=False,
        )
        existing: set[str] = set()
        for point in points:
            payload = getattr(point, "payload", None) or {}
            paper_id = payload.get("id")
            if paper_id in requested:
                existing.add(paper_id)
        return existing

    def count(self) -> int:
        self.ensure_collection()
        return int(
            self.client.count(collection_name=self.settings.qdrant_collection).count
        )

    def close(self) -> None:
        """Flush and release the client.

        Embedded local mode (``qdrant_path``) keeps recent writes in an
        in-memory segment and persists them to disk when the client closes.
        Relying on interpreter-exit finalization is unreliable, so writers
        (e.g. the corpus seeder) must call this explicitly to guarantee the
        corpus survives the process.
        """
        close = getattr(self.client, "close", None)
        if callable(close):
            close()


def build_vector_store(settings: Settings) -> PaperVectorStore:
    if settings.vector_store_backend.lower() == "memory":
        return InMemoryVectorStore(settings.embedding_dimension)
    return QdrantPaperVectorStore(settings)
