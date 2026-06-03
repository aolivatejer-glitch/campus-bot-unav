import logging
from dataclasses import dataclass
from pathlib import Path

from rag_chatbot.chunking.pipeline import build_chunks
from rag_chatbot.config import AppSettings, get_settings
from rag_chatbot.evaluation.evaluator import run_evaluation
from rag_chatbot.indexing.pipeline import build_index
from rag_chatbot.ingestion.pipeline import IngestionSummary, ingest_documents
from rag_chatbot.logging_config import LOGGER_NAME
from rag_chatbot.schemas import ChunkingResult, IndexingResult


@dataclass(frozen=True)
class LocalPipelineResult:
    ingestion: IngestionSummary
    chunking: ChunkingResult
    indexing: IndexingResult
    evaluation_json_path: Path | None = None
    evaluation_csv_path: Path | None = None


def run_full_local_pipeline(
    settings: AppSettings | None = None,
    *,
    reset_index: bool = False,
    limit: int | None = None,
    run_eval: bool = False,
    logger: logging.Logger | None = None,
) -> LocalPipelineResult:
    """Run the local MVP pipeline: ingest -> chunks -> vector index."""
    settings = settings or get_settings()
    logger = logger or logging.getLogger(LOGGER_NAME)

    logger.info("Starting full local pipeline")
    ingestion = ingest_documents(settings, logger=logger)
    chunking = build_chunks(settings, logger=logger)
    indexing = build_index(
        settings,
        reset=reset_index,
        limit=limit,
        logger=logger,
    )

    evaluation_json_path: Path | None = None
    evaluation_csv_path: Path | None = None
    if run_eval:
        _, evaluation_json_path, evaluation_csv_path = run_evaluation(settings)

    logger.info(
        "Full local pipeline finished processed=%s chunks=%s indexed=%s",
        ingestion.processed,
        chunking.total_chunks,
        indexing.chunks_indexed,
    )
    return LocalPipelineResult(
        ingestion=ingestion,
        chunking=chunking,
        indexing=indexing,
        evaluation_json_path=evaluation_json_path,
        evaluation_csv_path=evaluation_csv_path,
    )
