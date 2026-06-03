from rag_chatbot.rag.answer_builder import build_extractive_answer
from rag_chatbot.schemas import ContextSufficiencyResult, RagSource, RetrievalResult


def _context(sufficient: bool) -> ContextSufficiencyResult:
    return ContextSufficiencyResult(
        has_sufficient_context=sufficient,
        warning=None if sufficient else "No encontré información suficiente.",
        reason="sufficient" if sufficient else "no_results",
        best_score=0.8 if sufficient else None,
        total_context_chars=1000 if sufficient else 0,
        result_count=1 if sufficient else 0,
    )


def _out_of_domain_context() -> ContextSufficiencyResult:
    return ContextSufficiencyResult(
        has_sufficient_context=False,
        warning="Pregunta fuera del alcance del corpus.",
        reason="out_of_domain",
        rejection_reason="out_of_domain",
        best_score=None,
        total_context_chars=0,
        result_count=0,
    )


def _result(chunk_id: str = "chunk_1", *, is_toc_candidate: bool = False) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        score=0.8,
        distance=0.2,
        text="Texto completo",
        snippet=(
            "ÍNDICE 1. Introducción ................ 2"
            if is_toc_candidate
            else "Fragmento recuperado"
        ),
        document_id="doc_1",
        file_name="archivo.pdf",
        file_path="Documentos/archivo.pdf",
        file_type="pdf",
        page_number=2,
        chunk_index=0,
        char_count=14,
        source_label="archivo.pdf, pagina 2",
        metadata={"is_toc_candidate": is_toc_candidate},
    )


def test_build_extractive_answer_with_context() -> None:
    answer = build_extractive_answer(
        results=[_result()],
        sources=[
            RagSource(
                file_name="archivo.pdf",
                page_number=2,
                chunk_id="chunk_1",
                source_label="archivo.pdf, pagina 2",
                snippet="Fragmento recuperado",
            )
        ],
        context=_context(True),
        max_context_chars=500,
    )

    assert "Con base en los documentos recuperados" in answer
    assert "Fragmento recuperado" in answer
    assert "Fuentes:" in answer
    assert "chunk_1" in answer


def test_build_extractive_answer_without_context_does_not_invent() -> None:
    answer = build_extractive_answer(
        results=[],
        sources=[],
        context=_context(False),
        max_context_chars=500,
    )

    assert "No encontré información suficiente" in answer
    assert "Fragmento recuperado" not in answer


def test_build_extractive_answer_out_of_domain_does_not_add_retrieved_fragments() -> None:
    answer = build_extractive_answer(
        results=[_result()],
        sources=[],
        context=_out_of_domain_context(),
        max_context_chars=500,
    )

    assert answer == "Pregunta fuera del alcance del corpus."
    assert "Se recuperaron algunos fragmentos" not in answer


def test_build_extractive_answer_skips_toc_candidate_when_useful_chunk_exists() -> None:
    answer = build_extractive_answer(
        results=[_result("toc", is_toc_candidate=True), _result("chunk_1")],
        sources=[
            RagSource(
                file_name="archivo.pdf",
                page_number=2,
                chunk_id="toc",
                source_label="archivo.pdf, pagina 2",
                snippet="ÍNDICE 1. Introducción ................ 2",
            ),
            RagSource(
                file_name="archivo.pdf",
                page_number=3,
                chunk_id="chunk_1",
                source_label="archivo.pdf, pagina 3",
                snippet="Fragmento recuperado",
            ),
        ],
        context=_context(True),
        max_context_chars=500,
    )

    assert "Fragmento recuperado" in answer
    assert "................" not in answer
    assert "toc" not in answer
