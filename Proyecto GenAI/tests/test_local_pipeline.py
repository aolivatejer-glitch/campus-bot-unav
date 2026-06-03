from pathlib import Path

from rag_chatbot.config import AppSettings
from rag_chatbot.ingestion.pipeline import IngestionSummary
from rag_chatbot.local_pipeline import run_full_local_pipeline
from rag_chatbot.schemas import ChunkingResult, IndexingResult


def _settings(tmp_path: Path) -> AppSettings:
    return AppSettings(
        documents_dir=tmp_path / "Documentos",
        data_dir=tmp_path / "data",
        processed_dir=tmp_path / "data" / "processed",
        eval_dir=tmp_path / "data" / "eval",
        eval_dataset_path=tmp_path / "data" / "eval" / "evaluation_questions.jsonl",
        eval_reports_dir=tmp_path / "data" / "eval" / "reports",
        chunks_dir=tmp_path / "data" / "chunks",
        chunks_file=tmp_path / "data" / "chunks" / "chunks.jsonl",
        storage_dir=tmp_path / "storage",
        chroma_dir=tmp_path / "storage" / "chroma",
        manifest_db_path=tmp_path / "storage" / "manifest.sqlite",
        log_dir=tmp_path / "logs",
        _env_file=None,
    )


def test_run_full_local_pipeline_uses_existing_stages(tmp_path, monkeypatch) -> None:
    from rag_chatbot import local_pipeline as module

    calls = []
    settings = _settings(tmp_path)

    def fake_ingest(settings, *, logger):
        calls.append("ingest")
        return IngestionSummary(documents_found=1, processed=1, skipped=0, failed=0)

    def fake_build_chunks(settings, *, logger):
        calls.append("build-chunks")
        return ChunkingResult(
            documents_processed=1,
            documents_skipped=0,
            total_chunks=2,
            chunks_file=str(settings.chunks_file),
            manifest_file=str(settings.chunks_dir / "chunk_manifest.json"),
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            min_chunk_size=settings.min_chunk_size,
        )

    def fake_build_index(settings, *, reset, limit, logger):
        calls.append(("build-index", reset, limit))
        return IndexingResult(
            chunks_read=2,
            chunks_indexed=2,
            chunks_skipped=0,
            collection_name=settings.chroma_collection_name,
            chroma_dir=str(settings.chroma_dir),
            embedding_model=settings.embedding_model,
            embedding_device=settings.embedding_device,
        )

    monkeypatch.setattr(module, "ingest_documents", fake_ingest)
    monkeypatch.setattr(module, "build_chunks", fake_build_chunks)
    monkeypatch.setattr(module, "build_index", fake_build_index)

    result = run_full_local_pipeline(
        settings,
        reset_index=True,
        limit=50,
        run_eval=False,
    )

    assert calls == ["ingest", "build-chunks", ("build-index", True, 50)]
    assert result.ingestion.processed == 1
    assert result.chunking.total_chunks == 2
    assert result.indexing.chunks_indexed == 2
    assert result.evaluation_json_path is None


def test_run_full_local_pipeline_runs_eval_only_when_requested(tmp_path, monkeypatch) -> None:
    from rag_chatbot import local_pipeline as module

    settings = _settings(tmp_path)
    eval_calls = []

    monkeypatch.setattr(
        module,
        "ingest_documents",
        lambda settings, *, logger: IngestionSummary(documents_found=0),
    )
    monkeypatch.setattr(
        module,
        "build_chunks",
        lambda settings, *, logger: ChunkingResult(
            documents_processed=0,
            documents_skipped=0,
            total_chunks=0,
            chunks_file=str(settings.chunks_file),
            manifest_file=str(settings.chunks_dir / "chunk_manifest.json"),
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            min_chunk_size=settings.min_chunk_size,
        ),
    )
    monkeypatch.setattr(
        module,
        "build_index",
        lambda settings, *, reset, limit, logger: IndexingResult(
            chunks_read=0,
            chunks_indexed=0,
            chunks_skipped=0,
            collection_name=settings.chroma_collection_name,
            chroma_dir=str(settings.chroma_dir),
            embedding_model=settings.embedding_model,
            embedding_device=settings.embedding_device,
        ),
    )

    def fake_run_evaluation(settings):
        eval_calls.append("eval")
        return None, tmp_path / "evaluation_report.json", tmp_path / "evaluation_report.csv"

    monkeypatch.setattr(module, "run_evaluation", fake_run_evaluation)

    result = run_full_local_pipeline(settings, run_eval=True)

    assert eval_calls == ["eval"]
    assert result.evaluation_json_path == tmp_path / "evaluation_report.json"
    assert result.evaluation_csv_path == tmp_path / "evaluation_report.csv"
