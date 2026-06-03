import json
from datetime import datetime, timezone

from rag_chatbot.chunking.pipeline import build_chunks, find_chunk, read_chunks
from rag_chatbot.config import AppSettings
from rag_chatbot.schemas import DocumentPage, DocumentStatus, ProcessedDocument


def _settings(tmp_path) -> AppSettings:
    return AppSettings(
        documents_dir=tmp_path / "Documentos",
        data_dir=tmp_path / "data",
        processed_dir=tmp_path / "data" / "processed",
        eval_dir=tmp_path / "data" / "eval",
        chunks_dir=tmp_path / "data" / "chunks",
        chunks_file=tmp_path / "data" / "chunks" / "chunks.jsonl",
        storage_dir=tmp_path / "storage",
        chroma_dir=tmp_path / "storage" / "chroma",
        manifest_db_path=tmp_path / "storage" / "manifest.sqlite",
        log_dir=tmp_path / "logs",
        chunk_size=80,
        chunk_overlap=10,
        min_chunk_size=20,
        _env_file=None,
    )


def _write_processed_document(settings: AppSettings, document: ProcessedDocument) -> None:
    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    path = settings.processed_dir / f"{document.document_id}.json"
    path.write_text(
        json.dumps(document.model_dump(mode="json"), ensure_ascii=False),
        encoding="utf-8",
    )


def _processed_document(status: DocumentStatus = DocumentStatus.EXTRACTED) -> ProcessedDocument:
    text = "Esta es una pagina de prueba para chunking. " * 8
    page = DocumentPage(page_number=1, text=text, char_count=len(text))
    return ProcessedDocument(
        document_id="doc_abc",
        file_name="sample.pdf",
        file_path="Documentos/sample.pdf",
        file_type="pdf",
        file_hash="hash",
        extraction_status=status,
        requires_ocr=False,
        page_count=1,
        char_count=len(text) if status == DocumentStatus.EXTRACTED else 0,
        processed_at=datetime.now(timezone.utc),
        pages=[page] if status == DocumentStatus.EXTRACTED else [],
        metadata={"source": "Documentos/sample.pdf"},
    )


def test_chunking_pipeline_reads_processed_documents_and_writes_jsonl(tmp_path) -> None:
    settings = _settings(tmp_path)
    _write_processed_document(settings, _processed_document())

    result = build_chunks(settings)
    chunks = read_chunks(settings.chunks_file)

    assert result.documents_processed == 1
    assert result.documents_skipped == 0
    assert result.total_chunks == len(chunks)
    assert settings.chunks_file.exists()
    assert chunks[0].document_id == "doc_abc"
    assert chunks[0].file_name == "sample.pdf"
    assert chunks[0].page_number == 1


def test_chunking_pipeline_skips_failed_documents(tmp_path) -> None:
    settings = _settings(tmp_path)
    _write_processed_document(settings, _processed_document(status=DocumentStatus.FAILED))

    result = build_chunks(settings)

    assert result.documents_processed == 0
    assert result.documents_skipped == 1
    assert result.total_chunks == 0


def test_find_chunk_returns_chunk_by_id(tmp_path) -> None:
    settings = _settings(tmp_path)
    _write_processed_document(settings, _processed_document())
    build_chunks(settings)
    first_chunk = read_chunks(settings.chunks_file, limit=1)[0]

    found = find_chunk(settings.chunks_file, first_chunk.chunk_id)

    assert found is not None
    assert found.chunk_id == first_chunk.chunk_id
