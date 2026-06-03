import json
import logging
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from rag_chatbot.config import AppSettings, get_settings
from rag_chatbot.chunking.splitter import build_chunks_for_page
from rag_chatbot.logging_config import LOGGER_NAME
from rag_chatbot.schemas import (
    Chunk,
    ChunkingManifest,
    ChunkingResult,
    DocumentStatus,
    ProcessedDocument,
)


CHUNK_MANIFEST_FILE_NAME = "chunk_manifest.json"


def build_chunks(
    settings: AppSettings | None = None,
    *,
    logger: logging.Logger | None = None,
) -> ChunkingResult:
    settings = settings or get_settings()
    logger = logger or logging.getLogger(LOGGER_NAME)
    settings.ensure_directories()

    processed_files = sorted(settings.processed_dir.glob("*.json"))
    chunks: list[Chunk] = []
    documents_processed = 0
    documents_skipped = 0
    chunks_by_document: Counter[str] = Counter()
    chunks_by_page: Counter[str] = Counter()

    logger.info("Starting chunk build for %s processed documents", len(processed_files))

    for processed_file in processed_files:
        document = _read_processed_document(processed_file)

        if not _is_chunkable(document):
            documents_skipped += 1
            logger.info(
                "Skipping document for chunking: %s status=%s chars=%s",
                document.file_name,
                document.extraction_status.value,
                document.char_count,
            )
            continue

        document_chunks: list[Chunk] = []
        for page in document.pages:
            page_chunks = build_chunks_for_page(
                document,
                page,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
                min_chunk_size=settings.min_chunk_size,
                exclude_toc_chunks=settings.exclude_toc_chunks,
                clean_dot_leaders=settings.clean_dot_leaders,
            )
            document_chunks.extend(page_chunks)

            page_key = f"{document.document_id}:p{page.page_number}"
            chunks_by_page[page_key] += len(page_chunks)

        if document_chunks:
            documents_processed += 1
            chunks.extend(document_chunks)
            chunks_by_document[document.document_id] += len(document_chunks)
            logger.info(
                "Document chunked: %s chunks=%s",
                document.file_name,
                len(document_chunks),
            )
        else:
            documents_skipped += 1
            logger.info("Skipping document with no chunks: %s", document.file_name)

    _write_chunks_jsonl(chunks, settings.chunks_file)
    manifest_path = settings.chunks_dir / CHUNK_MANIFEST_FILE_NAME
    _write_chunk_manifest(
        manifest_path=manifest_path,
        chunks_file=settings.chunks_file,
        documents_processed=documents_processed,
        documents_skipped=documents_skipped,
        chunks=chunks,
        chunks_by_document=chunks_by_document,
        chunks_by_page=chunks_by_page,
        settings=settings,
    )

    logger.info(
        "Chunk build finished documents_processed=%s documents_skipped=%s total_chunks=%s",
        documents_processed,
        documents_skipped,
        len(chunks),
    )

    return ChunkingResult(
        documents_processed=documents_processed,
        documents_skipped=documents_skipped,
        total_chunks=len(chunks),
        chunks_file=str(settings.chunks_file),
        manifest_file=str(manifest_path),
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        min_chunk_size=settings.min_chunk_size,
        exclude_toc_chunks=settings.exclude_toc_chunks,
        clean_dot_leaders=settings.clean_dot_leaders,
    )


def read_chunks(chunks_file: Path, *, limit: int | None = None) -> list[Chunk]:
    if not chunks_file.exists():
        return []

    chunks: list[Chunk] = []
    with chunks_file.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            chunks.append(Chunk.model_validate_json(line))
            if limit is not None and len(chunks) >= limit:
                break

    return chunks


def find_chunk(chunks_file: Path, chunk_id: str) -> Chunk | None:
    for chunk in read_chunks(chunks_file):
        if chunk.chunk_id == chunk_id:
            return chunk
    return None


def read_chunk_manifest(manifest_file: Path) -> ChunkingManifest | None:
    if not manifest_file.exists():
        return None

    return ChunkingManifest.model_validate_json(manifest_file.read_text(encoding="utf-8"))


def chunk_manifest_path(settings: AppSettings) -> Path:
    return settings.chunks_dir / CHUNK_MANIFEST_FILE_NAME


def _read_processed_document(path: Path) -> ProcessedDocument:
    return ProcessedDocument.model_validate_json(path.read_text(encoding="utf-8"))


def _is_chunkable(document: ProcessedDocument) -> bool:
    if document.extraction_status != DocumentStatus.EXTRACTED:
        return False

    if document.char_count <= 0:
        return False

    return bool(document.pages)


def _write_chunks_jsonl(chunks: list[Chunk], chunks_file: Path) -> None:
    chunks_file.parent.mkdir(parents=True, exist_ok=True)

    with chunks_file.open("w", encoding="utf-8", newline="\n") as file:
        for chunk in chunks:
            payload = chunk.model_dump(mode="json")
            file.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _write_chunk_manifest(
    *,
    manifest_path: Path,
    chunks_file: Path,
    documents_processed: int,
    documents_skipped: int,
    chunks: list[Chunk],
    chunks_by_document: Counter[str],
    chunks_by_page: Counter[str],
    settings: AppSettings,
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = ChunkingManifest(
        generated_at=datetime.now(timezone.utc),
        documents_processed=documents_processed,
        documents_skipped=documents_skipped,
        total_chunks=len(chunks),
        chunks_by_document=dict(chunks_by_document),
        chunks_by_page=dict(chunks_by_page),
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        min_chunk_size=settings.min_chunk_size,
        exclude_toc_chunks=settings.exclude_toc_chunks,
        clean_dot_leaders=settings.clean_dot_leaders,
        chunks_file=str(chunks_file),
    )
    manifest_path.write_text(
        json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
