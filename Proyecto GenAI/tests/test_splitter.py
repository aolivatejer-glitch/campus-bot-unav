from datetime import datetime, timezone

from rag_chatbot.chunking.splitter import build_chunks_for_page, make_chunk_id, split_text
from rag_chatbot.schemas import DocumentPage, DocumentStatus, ProcessedDocument


def _document() -> ProcessedDocument:
    return ProcessedDocument(
        document_id="doc_123",
        file_name="sample.pdf",
        file_path="Documentos/sample.pdf",
        file_type="pdf",
        file_hash="hash",
        extraction_status=DocumentStatus.EXTRACTED,
        requires_ocr=False,
        page_count=1,
        char_count=0,
        processed_at=datetime.now(timezone.utc),
        pages=[],
        metadata={"source": "Documentos/sample.pdf"},
    )


def test_chunking_is_deterministic() -> None:
    text = "abcde" * 100

    first = split_text(text, chunk_size=100, chunk_overlap=20, min_chunk_size=30)
    second = split_text(text, chunk_size=100, chunk_overlap=20, min_chunk_size=30)

    assert first == second


def test_chunking_applies_overlap() -> None:
    text = "abcdefghijklmnopqrstuvwxyz" * 4

    chunks = split_text(text, chunk_size=30, chunk_overlap=5, min_chunk_size=10)

    assert len(chunks) > 1
    assert chunks[0][-5:] == chunks[1][:5]


def test_chunking_does_not_create_empty_chunks() -> None:
    chunks = split_text("   \n\t", chunk_size=30, chunk_overlap=5, min_chunk_size=10)

    assert chunks == []


def test_chunking_avoids_small_tail_for_long_text() -> None:
    text = "a" * 95

    chunks = split_text(text, chunk_size=40, chunk_overlap=5, min_chunk_size=20)

    assert all(len(chunk) >= 20 for chunk in chunks)


def test_chunking_allows_small_short_document() -> None:
    chunks = split_text("texto corto", chunk_size=100, chunk_overlap=10, min_chunk_size=50)

    assert chunks == ["texto corto"]


def test_chunk_metadata_is_preserved() -> None:
    document = _document()
    page = DocumentPage(page_number=3, text="Texto de la pagina " * 30, char_count=570)

    chunks = build_chunks_for_page(
        document,
        page,
        chunk_size=120,
        chunk_overlap=20,
        min_chunk_size=30,
    )

    assert chunks
    assert chunks[0].document_id == "doc_123"
    assert chunks[0].file_name == "sample.pdf"
    assert chunks[0].page_number == 3
    assert chunks[0].metadata["source"] == "Documentos/sample.pdf"


def test_chunk_id_is_stable() -> None:
    chunk_id = make_chunk_id(
        document_id="doc_123",
        page_number=2,
        chunk_index=0,
        text="Texto estable",
    )

    assert chunk_id == make_chunk_id(
        document_id="doc_123",
        page_number=2,
        chunk_index=0,
        text="Texto estable",
    )
    assert chunk_id.startswith("doc_123_p2_c0_")
