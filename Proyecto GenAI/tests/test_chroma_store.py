from rag_chatbot.schemas import Chunk
from rag_chatbot.vectorstores.chroma_store import build_chroma_metadata, build_chroma_where


def test_chroma_metadata_contains_expected_fields() -> None:
    chunk = Chunk(
        chunk_id="chunk_1",
        document_id="doc_1",
        file_name="sample.pdf",
        file_path="Documentos/sample.pdf",
        file_type="pdf",
        page_number=2,
        chunk_index=3,
        text="Texto del chunk",
        char_count=15,
        metadata={"source": "Documentos/sample.pdf"},
    )

    metadata = build_chroma_metadata(chunk)

    assert metadata == {
        "document_id": "doc_1",
        "file_name": "sample.pdf",
        "file_path": "Documentos/sample.pdf",
        "file_type": "pdf",
        "page_number": 2,
        "chunk_index": 3,
        "char_count": 15,
        "source": "Documentos/sample.pdf",
    }


def test_chroma_metadata_uses_minus_one_for_missing_page() -> None:
    chunk = Chunk(
        chunk_id="chunk_1",
        document_id="doc_1",
        file_name="sample.txt",
        file_path="Documentos/sample.txt",
        file_type="txt",
        page_number=None,
        chunk_index=0,
        text="Texto del chunk",
        char_count=15,
    )

    assert build_chroma_metadata(chunk)["page_number"] == -1


def test_chroma_where_builds_single_filter() -> None:
    assert build_chroma_where(document_id="doc_1") == {"document_id": "doc_1"}


def test_chroma_where_builds_combined_filter() -> None:
    assert build_chroma_where(document_id="doc_1", file_name="sample.pdf") == {
        "$and": [{"document_id": "doc_1"}, {"file_name": "sample.pdf"}]
    }
