from src.embeddings import build_paper_embedding_text


def test_build_paper_embedding_text_uses_title_and_abstract():
    text = build_paper_embedding_text(
        {"title": "A Useful Method", "summary": "We evaluate the method."}
    )
    assert text == "Title: A Useful Method\nAbstract: We evaluate the method."
