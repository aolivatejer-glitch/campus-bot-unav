import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException

from rag_chatbot import __version__
from rag_chatbot.api.dependencies import (
    get_api_settings,
    get_rag_pipeline,
    get_semantic_retriever,
)
from rag_chatbot.api.models import (
    BuildChunksResponse,
    BuildIndexRequest,
    BuildIndexResponse,
    ConfigResponse,
    EvalSummaryResponse,
    HealthResponse,
    IndexInfoResponse,
    IngestResponse,
    QueryChunkItem,
    QueryRequest,
    QueryResponse,
    QuerySourceItem,
    RetrieveRequest,
    RetrieveResponse,
    RetrieveResultItem,
)
from rag_chatbot.chunking.pipeline import build_chunks
from rag_chatbot.config import AppSettings
from rag_chatbot.evaluation.reporting import JSON_REPORT_NAME, load_evaluation_report
from rag_chatbot.indexing.pipeline import build_index, get_index_info
from rag_chatbot.ingestion.pipeline import ingest_documents
from rag_chatbot.logging_config import LOGGER_NAME
from rag_chatbot.rag.pipeline import LocalRagPipeline
from rag_chatbot.retrieval.retriever import SemanticRetriever


router = APIRouter()
logger = logging.getLogger(LOGGER_NAME)


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        service="rag-chatbot",
        version=__version__,
        environment="local",
    )


@router.get("/config", response_model=ConfigResponse)
def config(settings: AppSettings = Depends(get_api_settings)) -> ConfigResponse:
    return ConfigResponse(
        documents_dir=str(settings.documents_dir),
        chroma_collection=settings.chroma_collection_name,
        embedding_model=settings.embedding_model,
        top_k=settings.top_k,
        min_retrieval_score=settings.min_retrieval_score,
        allow_external_llm=settings.allow_external_llm,
    )


@router.post(
    "/retrieve",
    response_model=RetrieveResponse,
    response_model_exclude_none=True,
)
def retrieve(
    request: RetrieveRequest,
    retriever: SemanticRetriever = Depends(get_semantic_retriever),
) -> RetrieveResponse:
    question = _require_question(request.question)
    try:
        response = retriever.search(
            question,
            top_k=request.top_k,
            min_score=request.min_score,
            document_id=request.document_id,
            file_name=request.file_name,
        )
    except Exception as exc:
        raise _to_http_exception(exc) from exc

    return RetrieveResponse(
        question=response.query.question,
        top_k=response.query.top_k,
        result_count=len(response.results),
        results=[
            RetrieveResultItem(
                rank=index,
                chunk_id=result.chunk_id,
                score=result.score,
                distance=result.distance,
                file_name=result.file_name,
                page_number=result.page_number,
                source_label=result.source_label,
                snippet=result.snippet,
                metadata=result.metadata,
                text=result.text if request.include_text else None,
            )
            for index, result in enumerate(response.results, start=1)
        ],
    )


@router.post(
    "/query",
    response_model=QueryResponse,
    response_model_exclude_none=True,
)
def query(
    request: QueryRequest,
    pipeline: LocalRagPipeline = Depends(get_rag_pipeline),
) -> QueryResponse:
    question = _require_question(request.question)
    try:
        response = pipeline.ask(
            question,
            top_k=request.top_k,
            min_score=request.min_score,
            document_id=request.document_id,
            file_name=request.file_name,
        )
    except Exception as exc:
        raise _to_http_exception(exc) from exc

    return QueryResponse(
        question=response.question,
        answer=response.answer,
        has_sufficient_context=response.has_sufficient_context,
        warning=response.warning,
        sources=[
            QuerySourceItem(
                file_name=source.file_name,
                page_number=source.page_number,
                chunk_id=source.chunk_id,
                source_label=source.source_label,
                snippet=source.snippet,
            )
            for source in response.sources
        ],
        retrieved_chunks=[
            QueryChunkItem(
                chunk_id=chunk.chunk_id,
                score=chunk.score,
                file_name=chunk.file_name,
                page_number=chunk.page_number,
                snippet=chunk.snippet,
                text=chunk.text if request.show_chunks else None,
            )
            for chunk in response.retrieved_chunks
        ]
        if request.show_chunks
        else [],
    )


@router.post("/ingest", response_model=IngestResponse)
def ingest(settings: AppSettings = Depends(get_api_settings)) -> IngestResponse:
    try:
        summary = ingest_documents(settings, logger=logger)
    except Exception as exc:
        raise _to_http_exception(exc) from exc

    return IngestResponse(
        status="completed",
        documents_found=summary.documents_found,
        processed=summary.processed,
        skipped=summary.skipped,
        failed=summary.failed,
        requires_ocr=summary.requires_ocr,
    )


@router.post("/build-chunks", response_model=BuildChunksResponse)
def build_chunks_endpoint(
    settings: AppSettings = Depends(get_api_settings),
) -> BuildChunksResponse:
    try:
        result = build_chunks(settings, logger=logger)
    except Exception as exc:
        raise _to_http_exception(exc) from exc

    return BuildChunksResponse(
        status="completed",
        documents_processed=result.documents_processed,
        documents_skipped=result.documents_skipped,
        chunks_generated=result.total_chunks,
        chunks_file=result.chunks_file,
    )


@router.post("/build-index", response_model=BuildIndexResponse)
def build_index_endpoint(
    request: BuildIndexRequest,
    settings: AppSettings = Depends(get_api_settings),
) -> BuildIndexResponse:
    try:
        result = build_index(
            settings,
            reset=request.reset,
            limit=request.limit,
            logger=logger,
        )
    except Exception as exc:
        raise _to_http_exception(exc) from exc

    return BuildIndexResponse(
        status="completed",
        chunks_read=result.chunks_read,
        chunks_indexed=result.chunks_indexed,
        chunks_skipped=result.chunks_skipped,
        collection=result.collection_name,
        chroma_dir=result.chroma_dir,
    )


@router.get("/index-info", response_model=IndexInfoResponse)
def index_info(settings: AppSettings = Depends(get_api_settings)) -> IndexInfoResponse:
    info = get_index_info(settings)
    return IndexInfoResponse(
        collection=info.collection_name,
        vector_count=info.vector_count,
        chroma_dir=info.chroma_dir,
        embedding_model=info.embedding_model,
        embedding_device=info.embedding_device,
        chroma_available=info.chroma_available,
        error_message=info.error_message,
    )


@router.get("/eval-summary", response_model=EvalSummaryResponse)
def eval_summary(
    settings: AppSettings = Depends(get_api_settings),
) -> EvalSummaryResponse:
    report_path = settings.eval_reports_dir / JSON_REPORT_NAME
    try:
        report = load_evaluation_report(report_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    summary = report.summary
    return EvalSummaryResponse(
        total_questions=summary.total_questions,
        passed=summary.passed,
        failed=summary.failed,
        pass_rate=summary.pass_rate,
        avg_keyword_hit_rate=summary.avg_keyword_hit_rate,
        avg_top_score=summary.avg_top_score,
    )


def _require_question(question: str) -> str:
    normalized = question.strip()
    if not normalized:
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    return normalized


def _to_http_exception(exc: Exception) -> HTTPException:
    message = str(exc)
    lower_message = message.casefold()

    if isinstance(exc, FileNotFoundError):
        return HTTPException(status_code=404, detail=message)
    if isinstance(exc, ValueError) and "empty" in lower_message:
        return HTTPException(status_code=409, detail=message)
    if isinstance(exc, ValueError):
        return HTTPException(status_code=400, detail=message)
    if isinstance(exc, RuntimeError):
        return HTTPException(status_code=409, detail=message)

    logger.exception("Unhandled API error: %s", exc)
    return HTTPException(status_code=500, detail="Internal API error")
