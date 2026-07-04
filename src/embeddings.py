from threading import Lock

import numpy as np

# Canonical SPECTER2 default identities. Defined here (the module that owns the
# embedder) and referenced by Settings so the model/adapter names live in one
# place rather than being restated as literals in both.
DEFAULT_EMBEDDING_MODEL = "allenai/specter2_base"
DEFAULT_DOCUMENT_ADAPTER = "allenai/specter2"
DEFAULT_QUERY_ADAPTER = "allenai/specter2_adhoc_query"


def build_paper_embedding_text(paper: dict) -> str:
    """Build the SPECTER-style title/abstract input used for paper indexing."""
    title = str(paper.get("title", "")).strip()
    summary = str(paper.get("summary", "")).strip()
    return f"Title: {title}\nAbstract: {summary}".strip()


class TextEmbedder:
    """Generates asymmetric SPECTER2 embeddings for documents and queries."""

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        *,
        document_adapter: str = DEFAULT_DOCUMENT_ADAPTER,
        query_adapter: str = DEFAULT_QUERY_ADAPTER,
        document_adapter_name: str = "proximity",
        query_adapter_name: str = "adhoc_query",
    ):
        self.model_name = model_name
        self.document_adapter = document_adapter
        self.query_adapter = query_adapter
        self.document_adapter_name = document_adapter_name
        self.query_adapter_name = query_adapter_name
        self._model = None
        self._tokenizer = None
        self._adapter_lock = Lock()

    def _ensure_loaded(self) -> None:
        if self._model is None:
            with self._adapter_lock:
                if self._model is not None:
                    return
                from adapters import AutoAdapterModel
                from transformers import AutoTokenizer

                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self._model = AutoAdapterModel.from_pretrained(self.model_name)
                self._model.load_adapter(
                    self.document_adapter, load_as=self.document_adapter_name
                )
                self._model.load_adapter(
                    self.query_adapter, load_as=self.query_adapter_name
                )
                self._model.eval()

    def encode_documents(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Encode paper title/abstract texts with the SPECTER2 proximity adapter."""
        return self._encode(
            texts, batch_size=batch_size, adapter_name=self.document_adapter_name
        )

    def encode_queries(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Encode natural-language search queries with the SPECTER2 query adapter."""
        return self._encode(
            texts, batch_size=batch_size, adapter_name=self.query_adapter_name
        )

    def _encode(
        self, texts: list[str], *, batch_size: int, adapter_name: str
    ) -> np.ndarray:
        self._ensure_loaded()
        if not texts:
            return np.empty((0, 768), dtype="float32")

        import torch

        embeddings = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            inputs = self._tokenizer(
                batch,
                truncation=True,
                max_length=512,
                padding=True,
                return_tensors="pt",
            )
            device = next(self._model.parameters()).device
            inputs = {key: value.to(device) for key, value in inputs.items()}
            with self._adapter_lock, torch.inference_mode():
                self._model.set_active_adapters(adapter_name)
                output = self._model(**inputs)
            embeddings.append(output.last_hidden_state[:, 0, :].detach().cpu().numpy())

        return np.concatenate(embeddings, axis=0)

    def encode_papers(self, papers: list[dict], field: str = "title") -> np.ndarray:
        """
        Encodes papers into embeddings based on the specified field.

        Args:
            papers (List[Dict]): List of paper metadata dictionaries.
            field (str): Which field to use for encoding ('title', 'summary', or 'title_summary').

        Returns:
            np.ndarray: Array of embeddings for the selected paper field.

        Raises:
            ValueError: If the field is not one of 'title', 'summary', or 'title_summary'.
        """
        if field == "title":
            texts = [paper["title"] for paper in papers]
        elif field == "summary":
            texts = [paper["summary"] for paper in papers]
        elif field == "title_summary":
            texts = [build_paper_embedding_text(paper) for paper in papers]
        else:
            raise ValueError("field must be 'title', 'summary', or 'title_summary'")

        return self.encode_documents(texts)
