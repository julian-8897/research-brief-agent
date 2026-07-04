from __future__ import annotations

from threading import Lock
from typing import Protocol

import numpy as np

from src.embeddings import build_paper_embedding_text
from src.models import SearchResponseItem


class Reranker(Protocol):
    def rerank(
        self, query: str, items: list[SearchResponseItem], top_k: int
    ) -> list[SearchResponseItem]: ...


class CrossEncoderReranker:
    """Lazy cross-encoder reranker over query and paper title/abstract pairs."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._tokenizer = None
        self._model = None
        self._lock = Lock()

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name
            )
            self._model.eval()

    def rerank(
        self, query: str, items: list[SearchResponseItem], top_k: int
    ) -> list[SearchResponseItem]:
        if not items or top_k <= 0:
            return []
        self._ensure_loaded()
        scores = self._score(query, items)
        ordered = sorted(
            zip(items, scores, strict=True), key=lambda pair: pair[1], reverse=True
        )
        return [
            item.model_copy(update={"score": float(score)})
            for item, score in ordered[:top_k]
        ]

    def _score(self, query: str, items: list[SearchResponseItem]) -> np.ndarray:
        import torch

        documents = [
            build_paper_embedding_text(item.paper.model_dump()) for item in items
        ]
        inputs = self._tokenizer(
            [query] * len(documents),
            documents,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt",
        )
        device = next(self._model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            output = self._model(**inputs)
        logits = output.logits.detach().cpu()
        if logits.ndim == 2 and logits.shape[1] > 1:
            return logits[:, -1].numpy()
        return logits.reshape(-1).numpy()
