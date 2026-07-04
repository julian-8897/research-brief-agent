import pytest

from src.ingestion import pdf_extractors
from src.ingestion.pdf_extractors import PdfExtractionError, PdfExtractionResult


def test_auto_extractor_falls_back_to_pypdf_when_docling_is_unavailable(monkeypatch):
    def unavailable(data, *, char_budget):
        raise PdfExtractionError(
            "extractor_unavailable",
            "Docling is not installed.",
            extractor="docling",
        )

    monkeypatch.setattr(pdf_extractors, "_extract_with_docling", unavailable)
    monkeypatch.setattr(
        pdf_extractors,
        "_extract_with_pypdf",
        lambda data, *, char_budget: PdfExtractionResult(
            text="fallback text",
            truncated=False,
            extractor="pypdf",
        ),
    )

    result = pdf_extractors.extract_pdf(b"pdf", char_budget=100, extractor="auto")

    assert result.text == "fallback text"
    assert result.extractor == "pypdf"
    assert result.warnings == ["Docling is not installed; used pypdf fallback."]


def test_docling_extractor_is_required_when_explicitly_selected(monkeypatch):
    def unavailable(data, *, char_budget):
        raise PdfExtractionError(
            "extractor_unavailable",
            "Docling is not installed.",
            extractor="docling",
        )

    monkeypatch.setattr(pdf_extractors, "_extract_with_docling", unavailable)

    with pytest.raises(PdfExtractionError) as exc_info:
        pdf_extractors.extract_pdf(b"pdf", char_budget=100, extractor="docling")

    assert exc_info.value.code == "extractor_unavailable"


def test_extraction_result_bounds_and_normalizes_text():
    result = pdf_extractors._result(
        "Title\n\n\nBody text   ",
        char_budget=10,
        extractor="unit",
        page_count=2,
    )

    assert result.text == "Title\n\nBod"
    assert result.truncated is True
    assert result.page_count == 2
