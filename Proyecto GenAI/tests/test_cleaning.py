from rag_chatbot.chunking.cleaning import clean_text


def test_cleaning_normalizes_repeated_spaces() -> None:
    assert clean_text("Texto    con\t espacios") == "Texto con espacios"


def test_cleaning_normalizes_newlines() -> None:
    assert clean_text("Linea 1\r\nLinea 2\rLinea 3") == "Linea 1\nLinea 2\nLinea 3"


def test_cleaning_removes_problematic_invisible_characters() -> None:
    assert clean_text("\ufeffTexto\u200b visible\x00") == "Texto visible"


def test_cleaning_returns_empty_for_non_useful_text() -> None:
    assert clean_text(" \n\t --- ") == ""
