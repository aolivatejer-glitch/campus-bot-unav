from rag_chatbot.llm.prompt_builder import build_grounded_prompt
from rag_chatbot.schemas import RagSource, RetrievalResult


def _result() -> RetrievalResult:
    return RetrievalResult(
        chunk_id="chunk_1",
        score=0.9,
        distance=0.1,
        text="La política de compliance establece obligaciones documentadas.",
        snippet="La política de compliance establece obligaciones.",
        document_id="doc_1",
        file_name="politica-de-compliance-general.pdf",
        file_path="Documentos/politica-de-compliance-general.pdf",
        file_type="pdf",
        page_number=2,
        chunk_index=0,
        char_count=64,
        source_label="politica-de-compliance-general.pdf, pagina 2",
    )


def test_prompt_includes_question_context_and_sources() -> None:
    prompt = build_grounded_prompt(
        question="¿Qué dice compliance?",
        results=[_result()],
        sources=[
            RagSource(
                file_name="politica-de-compliance-general.pdf",
                page_number=2,
                chunk_id="chunk_1",
                source_label="politica-de-compliance-general.pdf, pagina 2",
                snippet="La política de compliance establece obligaciones.",
            )
        ],
        max_context_chars=500,
    )

    assert "responde exclusivamente" in prompt.casefold()
    assert "¿Qué dice compliance?" in prompt
    assert "politica-de-compliance-general.pdf" in prompt
    assert "chunk_1" in prompt
    assert "La política de compliance" in prompt


def test_prompt_skips_toc_candidate_chunks() -> None:
    result = _result()
    result.metadata["is_toc_candidate"] = True

    prompt = build_grounded_prompt(
        question="pregunta",
        results=[result],
        sources=[],
        max_context_chars=500,
    )

    assert "Sin contexto util" in prompt
    assert "La política de compliance" not in prompt
