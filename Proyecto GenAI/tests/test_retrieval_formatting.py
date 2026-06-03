from rag_chatbot.retrieval.formatting import (
    cosine_distance_to_score,
    make_snippet,
    make_source_label,
)


def test_cosine_distance_to_score_is_clamped() -> None:
    assert cosine_distance_to_score(0.2) == 0.8
    assert cosine_distance_to_score(-0.2) == 1.0
    assert cosine_distance_to_score(1.5) == 0.0
    assert cosine_distance_to_score(None) == 0.0


def test_make_snippet_normalizes_whitespace_and_limits_length() -> None:
    text = "Uno\n\nDos   Tres " + ("palabra " * 100)

    snippet = make_snippet(text, max_chars=40)

    assert "\n" not in snippet
    assert len(snippet) <= 43
    assert snippet.endswith("...")


def test_make_source_label_includes_page_when_available() -> None:
    assert make_source_label("archivo.pdf", 4) == "archivo.pdf, pagina 4"
    assert make_source_label("archivo.txt", None) == "archivo.txt"
