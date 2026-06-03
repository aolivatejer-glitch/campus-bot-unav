from rag_chatbot.rag.guardrails import evaluate_context_sufficiency
from rag_chatbot.schemas import RetrievalResult


def _result(score: float, text: str = "texto largo " * 80) -> RetrievalResult:
    return RetrievalResult(
        chunk_id="chunk_1",
        score=score,
        distance=1 - score,
        text=text,
        snippet=text[:80],
        document_id="doc_1",
        file_name="archivo.pdf",
        file_path="Documentos/archivo.pdf",
        file_type="pdf",
        page_number=1,
        chunk_index=0,
        char_count=len(text),
        source_label="archivo.pdf, pagina 1",
    )


def test_context_sufficient_when_score_and_length_pass() -> None:
    result = evaluate_context_sufficiency(
        [_result(0.8)],
        min_score=0.3,
        min_context_chars=100,
    )

    assert result.has_sufficient_context is True
    assert result.reason == "sufficient"


def test_context_insufficient_without_results() -> None:
    result = evaluate_context_sufficiency([], min_score=0.3, min_context_chars=100)

    assert result.has_sufficient_context is False
    assert result.reason == "no_results"
    assert result.warning is not None


def test_context_insufficient_with_low_score() -> None:
    result = evaluate_context_sufficiency(
        [_result(0.2)],
        min_score=0.3,
        min_context_chars=100,
    )

    assert result.has_sufficient_context is False
    assert result.reason == "low_score"


def test_context_insufficient_with_short_context() -> None:
    result = evaluate_context_sufficiency(
        [_result(0.8, text="corto")],
        min_score=0.3,
        min_context_chars=100,
    )

    assert result.has_sufficient_context is False
    assert result.reason == "too_little_context"
