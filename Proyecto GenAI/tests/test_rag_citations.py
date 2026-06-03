from rag_chatbot.rag.citations import build_source_label, deduplicate_sources
from rag_chatbot.schemas import RetrievalResult


def _result(chunk_id: str, file_name: str, page_number: int | None) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        score=0.8,
        distance=0.2,
        text="Texto",
        snippet="Snippet",
        document_id="doc_1",
        file_name=file_name,
        file_path=f"Documentos/{file_name}",
        file_type="pdf",
        page_number=page_number,
        chunk_index=0,
        char_count=5,
        source_label=build_source_label(file_name, page_number),
    )


def test_build_source_label_with_and_without_page() -> None:
    assert build_source_label("archivo.pdf", 3) == "archivo.pdf, pagina 3"
    assert build_source_label("archivo.txt", None) == "archivo.txt"


def test_deduplicate_sources_by_file_and_page() -> None:
    sources = deduplicate_sources(
        [
            _result("chunk_1", "archivo.pdf", 3),
            _result("chunk_2", "archivo.pdf", 3),
            _result("chunk_3", "otro.pdf", 1),
        ],
        max_sources=5,
    )

    assert [source.chunk_id for source in sources] == ["chunk_1", "chunk_3"]


def test_deduplicate_sources_applies_limit() -> None:
    sources = deduplicate_sources(
        [
            _result("chunk_1", "a.pdf", 1),
            _result("chunk_2", "b.pdf", 1),
            _result("chunk_3", "c.pdf", 1),
        ],
        max_sources=2,
    )

    assert [source.file_name for source in sources] == ["a.pdf", "b.pdf"]
