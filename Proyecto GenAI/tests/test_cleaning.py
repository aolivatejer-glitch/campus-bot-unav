from rag_chatbot.chunking.cleaning import (
    clean_text,
    is_table_of_contents_candidate,
    normalize_dot_leaders,
    remove_table_of_contents_lines,
)


def test_cleaning_normalizes_repeated_spaces() -> None:
    assert clean_text("Texto    con\t espacios") == "Texto con espacios"


def test_cleaning_normalizes_newlines() -> None:
    assert clean_text("Linea 1\r\nLinea 2\rLinea 3") == "Linea 1\nLinea 2\nLinea 3"


def test_cleaning_removes_problematic_invisible_characters() -> None:
    assert clean_text("\ufeffTexto\u200b visible\x00") == "Texto visible"


def test_cleaning_returns_empty_for_non_useful_text() -> None:
    assert clean_text(" \n\t --- ") == ""


def test_normalize_dot_leaders_removes_long_sequences() -> None:
    text = "3. Evaluación de riesgos penales ........................................ 4"

    cleaned = normalize_dot_leaders(text)

    assert "....." not in cleaned
    assert "Evaluación de riesgos penales" in cleaned


def test_normalize_dot_leaders_keeps_normal_periods_decimals_and_urls() -> None:
    text = "Art. 3.1 indica 2.5 puntos. Ver https://example.com/doc."

    assert normalize_dot_leaders(text) == text


def test_remove_table_of_contents_lines_removes_dot_leader_entries() -> None:
    text = (
        "ÍNDICE\n"
        "3. Evaluación de riesgos penales ........................................ 4\n"
        "4. Formación ............................................................ 4\n"
        "Contenido explicativo posterior con una frase completa."
    )

    cleaned = remove_table_of_contents_lines(text)

    assert "Evaluación de riesgos penales" not in cleaned
    assert "Formación" not in cleaned
    assert "Contenido explicativo posterior" in cleaned


def test_remove_table_of_contents_lines_keeps_explanatory_text() -> None:
    text = "La formación se dirige a estudiantes y personal. El documento explica criterios."

    assert remove_table_of_contents_lines(text) == text


def test_detects_table_of_contents_candidate() -> None:
    text = (
        "ÍNDICE\n"
        "1. Introducción ........................................................ 2\n"
        "2. Compliance .......................................................... 3\n"
        "3. Formación ........................................................... 4\n"
    )

    assert is_table_of_contents_candidate(text) is True


def test_does_not_mark_normal_content_as_table_of_contents() -> None:
    text = (
        "La política de compliance general establece criterios de actuación y "
        "responsabilidades para distintos órganos de la Universidad de Navarra."
    )

    assert is_table_of_contents_candidate(text) is False
