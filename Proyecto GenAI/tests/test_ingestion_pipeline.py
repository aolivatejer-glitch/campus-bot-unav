import json

from rag_chatbot.config import AppSettings
from rag_chatbot.ingestion.pipeline import ingest_documents


def _settings(tmp_path) -> AppSettings:
    return AppSettings(
        documents_dir=tmp_path / "Documentos",
        data_dir=tmp_path / "data",
        processed_dir=tmp_path / "data" / "processed",
        eval_dir=tmp_path / "data" / "eval",
        storage_dir=tmp_path / "storage",
        chroma_dir=tmp_path / "storage" / "chroma",
        manifest_db_path=tmp_path / "storage" / "manifest.sqlite",
        log_dir=tmp_path / "logs",
        _env_file=None,
    )


def test_pipeline_processes_document_and_writes_json(tmp_path) -> None:
    settings = _settings(tmp_path)
    settings.documents_dir.mkdir(parents=True)
    (settings.documents_dir / "sample.txt").write_text(
        "Texto para ingesta",
        encoding="utf-8",
    )

    summary = ingest_documents(settings)

    processed_files = list(settings.processed_dir.glob("*.json"))
    payload = json.loads(processed_files[0].read_text(encoding="utf-8"))

    assert summary.documents_found == 1
    assert summary.processed == 1
    assert summary.skipped == 0
    assert summary.failed == 0
    assert len(processed_files) == 1
    assert payload["file_name"] == "sample.txt"
    assert payload["pages"][0]["text"] == "Texto para ingesta"


def test_pipeline_skips_unchanged_documents(tmp_path) -> None:
    settings = _settings(tmp_path)
    settings.documents_dir.mkdir(parents=True)
    (settings.documents_dir / "sample.txt").write_text(
        "Texto para ingesta",
        encoding="utf-8",
    )

    first_summary = ingest_documents(settings)
    second_summary = ingest_documents(settings)

    assert first_summary.processed == 1
    assert second_summary.documents_found == 1
    assert second_summary.processed == 0
    assert second_summary.skipped == 1
