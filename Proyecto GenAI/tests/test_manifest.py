from datetime import datetime, timezone

from rag_chatbot.ingestion.manifest import DocumentManifest
from rag_chatbot.schemas import DocumentStatus, ManifestRecord


def _record(file_path: str, file_hash: str = "abc") -> ManifestRecord:
    return ManifestRecord(
        document_id="doc_1",
        file_name="sample.txt",
        file_path=file_path,
        file_type="txt",
        file_hash=file_hash,
        status=DocumentStatus.EXTRACTED,
        requires_ocr=False,
        page_count=1,
        char_count=12,
        processed_at=datetime.now(timezone.utc),
        error_message=None,
    )


def test_manifest_creates_table(tmp_path) -> None:
    db_path = tmp_path / "manifest.sqlite"
    manifest = DocumentManifest(db_path)

    manifest.initialize()

    assert db_path.exists()


def test_manifest_registers_documents(tmp_path) -> None:
    db_path = tmp_path / "manifest.sqlite"
    document_path = tmp_path / "sample.txt"
    document_path.write_text("sample", encoding="utf-8")
    manifest = DocumentManifest(db_path)
    manifest.initialize()

    manifest.upsert(_record(str(document_path.resolve())))

    records = manifest.list_documents()
    assert len(records) == 1
    assert records[0].file_name == "sample.txt"


def test_manifest_detects_unchanged_hash(tmp_path) -> None:
    db_path = tmp_path / "manifest.sqlite"
    document_path = tmp_path / "sample.txt"
    document_path.write_text("sample", encoding="utf-8")
    manifest = DocumentManifest(db_path)
    manifest.initialize()

    manifest.upsert(_record(str(document_path.resolve()), file_hash="same"))

    assert manifest.has_same_hash(document_path, "same") is True
    assert manifest.has_same_hash(document_path, "different") is False
