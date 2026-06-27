import numpy as np


def build_paper_embedding_text(paper: dict) -> str:
    """Build the SPECTER-style title/abstract input used for paper indexing."""
    title = str(paper.get("title", "")).strip()
    summary = str(paper.get("summary", "")).strip()
    return f"Title: {title}\nAbstract: {summary}".strip()


class TextEmbedder:
    """
    Generates text embeddings for semantic search using a SentenceTransformer model.

    Args:
        model_name (str): Name of the sentence transformer model to use.

    Attributes:
        model_name (str): The name of the transformer model.
        model (SentenceTransformer): The loaded sentence transformer model.
    """

    def __init__(self, model_name: str = "sentence-transformers/allenai-specter"):
        """
        Initializes the TextEmbedder with the specified model.

        Args:
            model_name (str): Name of the sentence transformer model to use.
        """
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode_texts(
        self, texts: list[str], batch_size: int = 32, show_progress_bar: bool = False
    ) -> np.ndarray:
        """
        Encodes a list of texts into embeddings.

        Args:
            texts (List[str]): List of text strings to encode.
            batch_size (int): Batch size for encoding.

        Returns:
            np.ndarray: Array of embeddings for the input texts.
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
        )

        return embeddings

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

        return self.encode_texts(texts)
