from rag_chatbot.ui.streamlit_app import _spinner_message


def test_spinner_message_for_extractive_mode() -> None:
    assert _spinner_message("extractive") == "Buscando en los documentos indexados..."


def test_spinner_message_for_llm_mode() -> None:
    assert (
        _spinner_message("llm")
        == "Consultando documentos y generando respuesta con Gemini..."
    )
