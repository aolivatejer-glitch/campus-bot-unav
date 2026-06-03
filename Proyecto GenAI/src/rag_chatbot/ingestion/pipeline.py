import json
import logging
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path

from rag_chatbot.config import PROJECT_ROOT, AppSettings, get_settings
from rag_chatbot.ingestion.discovery import discover_documents
from rag_chatbot.ingestion.hashing import calculate_file_sha256
from rag_chatbot.ingestion.loaders import load_document
from rag_chatbot.ingestion.manifest import DocumentManifest
from rag_chatbot.logging_config import LOGGER_NAME
from rag_chatbot.schemas import DocumentStatus, ManifestRecord, ProcessedDocument


@dataclass
class IngestionSummary:
    documents_found: int = 0
    processed: int = 0
    skipped: int = 0
    failed: int = 0
    requires_ocr: int = 0
    output_dir: Path | None = None
    manifest_path: Path | None = None
    processed_document_ids: list[str] = field(default_factory=list)


def ingest_documents(
    settings: AppSettings | None = None,
    *,
    recursive: bool = True,
    logger: logging.Logger | None = None,
) -> IngestionSummary:
    settings = settings or get_settings()
    logger = logger or logging.getLogger(LOGGER_NAME)
    settings.ensure_directories()

    manifest = DocumentManifest(settings.manifest_db_path)
    manifest.initialize()

    document_paths = discover_documents(settings.documents_dir, recursive=recursive)
    summary = IngestionSummary(
        documents_found=len(document_paths),
        output_dir=settings.processed_dir,
        manifest_path=settings.manifest_db_path,
    )

    logger.info("Starting ingestion for %s documents", summary.documents_found)

    for path in document_paths:
        try:
            resolved_path = path.resolve()
            file_hash = calculate_file_sha256(resolved_path)

            if manifest.has_same_hash(resolved_path, file_hash):
                summary.skipped += 1
                logger.info("Skipping unchanged document: %s", path.name)
                continue

            document_id = make_document_id(resolved_path)
            document = load_document(resolved_path, file_hash, document_id)
            _write_processed_document(document, settings.processed_dir)
            manifest.upsert(ManifestRecord.from_processed_document(document))

            summary.processed_document_ids.append(document.document_id)

            if document.extraction_status == DocumentStatus.FAILED:
                summary.failed += 1
                logger.warning("Document ingestion failed: %s", path.name)
            else:
                summary.processed += 1
                logger.info(
                    "Document ingested: %s pages=%s chars=%s status=%s",
                    path.name,
                    document.page_count,
                    document.char_count,
                    document.extraction_status.value,
                )

            if document.requires_ocr:
                summary.requires_ocr += 1
        except Exception as exc:
            summary.failed += 1
            logger.exception("Unexpected ingestion error for %s: %s", path.name, exc)

    logger.info(
        "Ingestion finished found=%s processed=%s skipped=%s failed=%s requires_ocr=%s",
        summary.documents_found,
        summary.processed,
        summary.skipped,
        summary.failed,
        summary.requires_ocr,
    )
    return summary


def make_document_id(path: Path) -> str:
    try:
        label = path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        label = str(path.resolve())

    return "doc_" + sha256(label.lower().encode("utf-8")).hexdigest()[:16]


def processed_document_path(document_id: str, processed_dir: Path) -> Path:
    return processed_dir / f"{document_id}.json"


def _write_processed_document(document: ProcessedDocument, processed_dir: Path) -> Path:
    processed_dir.mkdir(parents=True, exist_ok=True)
    output_path = processed_document_path(document.document_id, processed_dir)
    payload = document.model_dump(mode="json")
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path
