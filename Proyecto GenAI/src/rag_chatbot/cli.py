import os
import subprocess
import sys

import typer

from rag_chatbot import __version__
from rag_chatbot.chunking.pipeline import (
    build_chunks,
    chunk_manifest_path,
    find_chunk,
    read_chunk_manifest,
    read_chunks,
)
from rag_chatbot.config import PROJECT_ROOT, get_settings
from rag_chatbot.diagnostics import format_doctor_report, run_doctor
from rag_chatbot.embeddings.sentence_transformers_provider import (
    SentenceTransformersEmbeddingProvider,
)
from rag_chatbot.evaluation.calibration import (
    CALIBRATION_REPORT_NAME,
    load_calibration_report,
)
from rag_chatbot.evaluation.evaluator import run_evaluation
from rag_chatbot.evaluation.grid import (
    parse_float_values,
    parse_int_values,
    run_evaluation_grid,
)
from rag_chatbot.evaluation.reporting import JSON_REPORT_NAME, load_evaluation_report
from rag_chatbot.ingestion.manifest import DocumentManifest
from rag_chatbot.ingestion.pipeline import ingest_documents, processed_document_path
from rag_chatbot.indexing.pipeline import build_index, get_index_info
from rag_chatbot.logging_config import configure_logging
from rag_chatbot.local_pipeline import run_full_local_pipeline
from rag_chatbot.rag.pipeline import LocalRagPipeline
from rag_chatbot.retrieval.retriever import SemanticRetriever
from rag_chatbot.schemas import HealthCheck


app = typer.Typer(
    add_completion=False,
    help="CLI minima para el chatbot documental RAG local-first.",
)


@app.command()
def health() -> None:
    """Muestra si la configuracion base del proyecto esta lista."""
    settings = get_settings()
    logger = configure_logging(settings)
    check = HealthCheck(
        status="ok",
        project_name=settings.project_name,
        version=settings.version,
        documents_dir_exists=settings.documents_dir.exists(),
        external_llm_enabled=settings.allow_external_llm,
    )
    logger.info("Health check executed")

    typer.echo(f"OK - {check.project_name} {check.version}")
    typer.echo(f"Documents dir: {settings.documents_dir}")
    typer.echo(f"Documents dir exists: {check.documents_dir_exists}")
    typer.echo(f"External LLM enabled: {check.external_llm_enabled}")


@app.command()
def doctor() -> None:
    """Revisa el estado local end-to-end del MVP."""
    settings = get_settings()
    logger = configure_logging(settings)
    report = run_doctor(settings)

    logger.info(
        "Doctor executed ok=%s warnings=%s errors=%s",
        report.ok_count,
        report.warning_count,
        report.error_count,
    )

    typer.echo("Doctor report:")
    for line in format_doctor_report(report):
        typer.echo(line)

    typer.echo("")
    typer.echo(
        "Summary: "
        f"ok={report.ok_count} warnings={report.warning_count} errors={report.error_count}"
    )

    if report.has_errors:
        raise typer.Exit(code=1)


@app.command("show-config")
def show_config() -> None:
    """Muestra la configuracion principal sin exponer secretos."""
    settings = get_settings()
    configure_logging(settings).info("Configuration displayed")

    for key, value in settings.public_dict().items():
        typer.echo(f"{key}: {value}")


@app.command("init-dirs")
def init_dirs() -> None:
    """Crea las carpetas necesarias para la ejecucion local."""
    settings = get_settings()
    logger = configure_logging(settings)
    directories = settings.ensure_directories()
    logger.info("Project directories initialized")

    typer.echo("Project directories are ready:")
    for directory in directories:
        typer.echo(f"- {directory}")


@app.command()
def ingest(
    recursive: bool = typer.Option(
        True,
        "--recursive/--no-recursive",
        help="Buscar documentos tambien dentro de subcarpetas.",
    ),
) -> None:
    """Ejecuta la ingesta local de PDF, TXT y DOCX."""
    settings = get_settings()
    logger = configure_logging(settings)
    summary = ingest_documents(settings, recursive=recursive, logger=logger)

    typer.echo("Ingestion finished:")
    typer.echo(f"- Documents found: {summary.documents_found}")
    typer.echo(f"- Processed: {summary.processed}")
    typer.echo(f"- Skipped unchanged: {summary.skipped}")
    typer.echo(f"- Failed: {summary.failed}")
    typer.echo(f"- Requires OCR: {summary.requires_ocr}")
    typer.echo(f"- Output dir: {summary.output_dir}")
    typer.echo(f"- Manifest: {summary.manifest_path}")


@app.command("list-documents")
def list_documents() -> None:
    """Lista documentos registrados en el manifest local."""
    settings = get_settings()
    configure_logging(settings).info("Listing manifest documents")
    manifest = DocumentManifest(settings.manifest_db_path)
    manifest.initialize()
    records = manifest.list_documents()

    if not records:
        typer.echo("No documents registered in the manifest.")
        return

    typer.echo("Documents:")
    for record in records:
        typer.echo(
            " | ".join(
                [
                    record.document_id,
                    record.file_name,
                    f"status={record.status.value}",
                    f"pages={record.page_count}",
                    f"chars={record.char_count}",
                    f"requires_ocr={record.requires_ocr}",
                    f"processed_at={record.processed_at.isoformat()}",
                ]
            )
        )


@app.command("inspect-document")
def inspect_document(document_id: str) -> None:
    """Muestra metadatos de un documento sin imprimir el texto extraido."""
    settings = get_settings()
    configure_logging(settings).info("Inspecting document metadata")
    manifest = DocumentManifest(settings.manifest_db_path)
    manifest.initialize()
    record = manifest.get(document_id)

    if record is None:
        typer.echo(f"Document not found: {document_id}")
        raise typer.Exit(code=1)

    typer.echo(f"Document ID: {record.document_id}")
    typer.echo(f"File name: {record.file_name}")
    typer.echo(f"File path: {record.file_path}")
    typer.echo(f"File type: {record.file_type}")
    typer.echo(f"Status: {record.status.value}")
    typer.echo(f"Requires OCR: {record.requires_ocr}")
    typer.echo(f"Pages: {record.page_count}")
    typer.echo(f"Characters: {record.char_count}")
    typer.echo(f"Processed at: {record.processed_at.isoformat()}")
    typer.echo(f"Processed JSON: {processed_document_path(record.document_id, settings.processed_dir)}")
    if record.error_message:
        typer.echo(f"Error: {record.error_message}")


@app.command("build-chunks")
def build_chunks_command() -> None:
    """Genera chunks trazables desde los documentos procesados."""
    settings = get_settings()
    logger = configure_logging(settings)
    result = build_chunks(settings, logger=logger)

    typer.echo("Chunk build finished:")
    typer.echo(f"- Documents processed: {result.documents_processed}")
    typer.echo(f"- Documents skipped: {result.documents_skipped}")
    typer.echo(f"- Chunks generated: {result.total_chunks}")
    typer.echo(f"- Chunks file: {result.chunks_file}")
    typer.echo(f"- Manifest file: {result.manifest_file}")
    typer.echo(f"- chunk_size: {result.chunk_size}")
    typer.echo(f"- chunk_overlap: {result.chunk_overlap}")
    typer.echo(f"- min_chunk_size: {result.min_chunk_size}")
    typer.echo(f"- exclude_toc_chunks: {result.exclude_toc_chunks}")
    typer.echo(f"- clean_dot_leaders: {result.clean_dot_leaders}")


@app.command("list-chunks")
def list_chunks(
    limit: int = typer.Option(10, "--limit", "-n", min=1, help="Numero de chunks a mostrar."),
) -> None:
    """Lista un resumen de los chunks generados sin imprimir texto completo."""
    settings = get_settings()
    configure_logging(settings).info("Listing chunks")
    chunks = read_chunks(settings.chunks_file, limit=limit)
    manifest = read_chunk_manifest(chunk_manifest_path(settings))

    if manifest is None:
        typer.echo("No chunk manifest found. Run build-chunks first.")
        return

    typer.echo(f"Total chunks: {manifest.total_chunks}")
    typer.echo(f"Documents: {len(manifest.chunks_by_document)}")
    typer.echo(f"Chunks file: {manifest.chunks_file}")

    if not chunks:
        typer.echo("No chunks available.")
        return

    typer.echo("Sample chunks:")
    for chunk in chunks:
        typer.echo(
            " | ".join(
                [
                    chunk.chunk_id,
                    chunk.file_name,
                    f"page={chunk.page_number}",
                    f"index={chunk.chunk_index}",
                    f"chars={chunk.char_count}",
                ]
            )
        )


@app.command("inspect-chunk")
def inspect_chunk(
    chunk_id: str,
    preview_chars: int = typer.Option(
        500,
        "--preview-chars",
        min=1,
        max=2000,
        help="Longitud maxima del fragmento mostrado.",
    ),
) -> None:
    """Muestra metadatos de un chunk y un fragmento corto del texto."""
    settings = get_settings()
    configure_logging(settings).info("Inspecting chunk metadata")
    chunk = find_chunk(settings.chunks_file, chunk_id)

    if chunk is None:
        typer.echo(f"Chunk not found: {chunk_id}")
        raise typer.Exit(code=1)

    preview = chunk.text[:preview_chars]
    if len(chunk.text) > preview_chars:
        preview = f"{preview}..."

    typer.echo(f"Chunk ID: {chunk.chunk_id}")
    typer.echo(f"Document ID: {chunk.document_id}")
    typer.echo(f"File name: {chunk.file_name}")
    typer.echo(f"File path: {chunk.file_path}")
    typer.echo(f"File type: {chunk.file_type}")
    typer.echo(f"Page: {chunk.page_number}")
    typer.echo(f"Chunk index: {chunk.chunk_index}")
    typer.echo(f"Characters: {chunk.char_count}")
    typer.echo(f"Source: {chunk.metadata.get('source')}")
    typer.echo("Preview:")
    typer.echo(preview)


@app.command("build-index")
def build_index_command(
    reset: bool = typer.Option(
        False,
        "--reset",
        help="Recrear la coleccion vectorial antes de indexar.",
    ),
    limit: int | None = typer.Option(
        None,
        "--limit",
        min=1,
        help="Indexar solo los primeros N chunks.",
    ),
) -> None:
    """Genera embeddings locales e indexa chunks en Chroma persistente."""
    settings = get_settings()
    logger = configure_logging(settings)

    try:
        result = build_index(settings, reset=reset, limit=limit, logger=logger)
    except Exception as exc:
        logger.error("Index build failed: %s", exc)
        typer.echo(f"Index build failed: {exc}")
        raise typer.Exit(code=1)

    typer.echo("Index build finished:")
    typer.echo(f"- Chunks read: {result.chunks_read}")
    typer.echo(f"- Chunks indexed: {result.chunks_indexed}")
    typer.echo(f"- Chunks skipped: {result.chunks_skipped}")
    typer.echo(f"- Collection: {result.collection_name}")
    typer.echo(f"- Chroma dir: {result.chroma_dir}")
    typer.echo(f"- Embedding model: {result.embedding_model}")
    typer.echo(f"- Embedding device: {result.embedding_device}")


@app.command("run-local-pipeline")
def run_local_pipeline_command(
    reset_index: bool = typer.Option(
        False,
        "--reset-index",
        help="Recrear la coleccion Chroma antes de indexar.",
    ),
    limit: int | None = typer.Option(
        None,
        "--limit",
        min=1,
        help="Indexar solo los primeros N chunks generados.",
    ),
    run_eval: bool = typer.Option(
        False,
        "--run-eval",
        help="Ejecutar evaluacion basica al terminar la indexacion.",
    ),
) -> None:
    """Ejecuta el flujo local completo: ingest -> build-chunks -> build-index."""
    settings = get_settings()
    logger = configure_logging(settings)

    try:
        result = run_full_local_pipeline(
            settings,
            reset_index=reset_index,
            limit=limit,
            run_eval=run_eval,
            logger=logger,
        )
    except Exception as exc:
        logger.error("Full local pipeline failed: %s", exc)
        typer.echo(f"Local pipeline failed: {exc}")
        raise typer.Exit(code=1)

    typer.echo("Local pipeline finished:")
    typer.echo("Ingestion:")
    typer.echo(f"- Documents found: {result.ingestion.documents_found}")
    typer.echo(f"- Processed: {result.ingestion.processed}")
    typer.echo(f"- Skipped unchanged: {result.ingestion.skipped}")
    typer.echo(f"- Failed: {result.ingestion.failed}")
    typer.echo(f"- Requires OCR: {result.ingestion.requires_ocr}")
    typer.echo("Chunking:")
    typer.echo(f"- Documents processed: {result.chunking.documents_processed}")
    typer.echo(f"- Documents skipped: {result.chunking.documents_skipped}")
    typer.echo(f"- Chunks generated: {result.chunking.total_chunks}")
    typer.echo(f"- Chunks file: {result.chunking.chunks_file}")
    typer.echo("Indexing:")
    typer.echo(f"- Chunks read: {result.indexing.chunks_read}")
    typer.echo(f"- Chunks indexed: {result.indexing.chunks_indexed}")
    typer.echo(f"- Chunks skipped: {result.indexing.chunks_skipped}")
    typer.echo(f"- Collection: {result.indexing.collection_name}")
    typer.echo(f"- Chroma dir: {result.indexing.chroma_dir}")
    if result.evaluation_json_path:
        typer.echo("Evaluation:")
        typer.echo(f"- JSON report: {result.evaluation_json_path}")
        typer.echo(f"- CSV report: {result.evaluation_csv_path}")


@app.command("index-info")
def index_info() -> None:
    """Muestra informacion del indice vectorial local."""
    settings = get_settings()
    configure_logging(settings).info("Index info requested")
    info = get_index_info(settings)

    typer.echo(f"Collection: {info.collection_name}")
    typer.echo(f"Vector count: {info.vector_count if info.vector_count is not None else 'unavailable'}")
    typer.echo(f"Chroma dir: {info.chroma_dir}")
    typer.echo(f"Embedding model: {info.embedding_model}")
    typer.echo(f"Embedding device: {info.embedding_device}")
    typer.echo(f"Chroma available: {info.chroma_available}")
    if info.error_message:
        typer.echo(f"Error: {info.error_message}")


@app.command("test-embedding")
def test_embedding(text: str) -> None:
    """Genera un embedding local de prueba sin imprimir el vector completo."""
    settings = get_settings()
    logger = configure_logging(settings)

    try:
        provider = SentenceTransformersEmbeddingProvider(
            model_name=settings.embedding_model,
            device=settings.embedding_device,
            batch_size=settings.embedding_batch_size,
        )
        embedding = provider.embed_query(text)
    except Exception as exc:
        logger.error("Embedding test failed: %s", exc)
        typer.echo(f"Embedding test failed: {exc}")
        raise typer.Exit(code=1)

    preview = [round(value, 6) for value in embedding[:5]]
    typer.echo(f"Dimension: {len(embedding)}")
    typer.echo(f"Model: {provider.model_name}")
    typer.echo(f"Device: {provider.device}")
    typer.echo(f"First values: {preview}")


@app.command()
def search(
    question: str,
    top_k: int | None = typer.Option(
        None,
        "--top-k",
        min=1,
        help="Numero maximo de chunks a recuperar.",
    ),
    min_score: float | None = typer.Option(
        None,
        "--min-score",
        min=0.0,
        max=1.0,
        help="Score minimo aproximado de similitud.",
    ),
    show_text: bool = typer.Option(
        False,
        "--show-text",
        help="Mostrar el texto completo del chunk recuperado.",
    ),
    document_id: str | None = typer.Option(
        None,
        "--document-id",
        help="Filtrar por document_id.",
    ),
    file_name: str | None = typer.Option(
        None,
        "--file-name",
        help="Filtrar por nombre de archivo.",
    ),
) -> None:
    """Busca chunks relevantes en el indice vectorial local."""
    settings = get_settings()
    logger = configure_logging(settings)

    try:
        response = SemanticRetriever(settings, logger=logger).search(
            question,
            top_k=top_k,
            min_score=min_score,
            document_id=document_id,
            file_name=file_name,
        )
    except Exception as exc:
        logger.error("Semantic search failed: %s", exc)
        typer.echo(f"Search failed: {exc}")
        raise typer.Exit(code=1)

    typer.echo(f"Question: {response.query.question}")
    typer.echo(f"top_k: {response.query.top_k}")
    typer.echo(f"min_score: {response.query.min_score}")
    typer.echo(f"Results found: {len(response.results)}")
    typer.echo("Score: approximate cosine similarity; higher is more relevant.")

    for index, result in enumerate(response.results, start=1):
        typer.echo("")
        typer.echo(f"[{index}] score={result.score:.4f} distance={_format_distance(result.distance)}")
        typer.echo(f"Source: {result.source_label}")
        typer.echo(f"Chunk ID: {result.chunk_id}")
        typer.echo(f"Document ID: {result.document_id}")
        typer.echo(f"File path: {result.file_path}")
        typer.echo("Text:" if show_text else "Snippet:")
        typer.echo(result.text if show_text else result.snippet)


@app.command("search-debug")
def search_debug(
    question: str,
    top_k: int | None = typer.Option(None, "--top-k", min=1),
    min_score: float | None = typer.Option(None, "--min-score", min=0.0, max=1.0),
    document_id: str | None = typer.Option(None, "--document-id"),
    file_name: str | None = typer.Option(None, "--file-name"),
) -> None:
    """Busca chunks y muestra metadatos tecnicos de trazabilidad."""
    settings = get_settings()
    logger = configure_logging(settings)

    try:
        retriever = SemanticRetriever(settings, logger=logger)
        response = retriever.search(
            question,
            top_k=top_k,
            min_score=min_score,
            document_id=document_id,
            file_name=file_name,
        )
        vector_count = retriever.vector_store.count()
    except Exception as exc:
        logger.error("Semantic debug search failed: %s", exc)
        typer.echo(f"Search failed: {exc}")
        raise typer.Exit(code=1)

    typer.echo(f"Embedding model: {response.embedding_model}")
    typer.echo(f"Embedding device: {response.embedding_device}")
    typer.echo(f"Collection: {response.collection_name}")
    typer.echo(f"Chroma dir: {settings.chroma_dir}")
    typer.echo(f"Vector count: {vector_count}")
    typer.echo(f"top_k: {response.query.top_k}")
    typer.echo(f"min_score: {response.query.min_score}")
    typer.echo(f"document_id: {response.query.document_id}")
    typer.echo(f"file_name: {response.query.file_name}")
    typer.echo(f"Results found: {len(response.results)}")
    typer.echo(response.score_description)

    for index, result in enumerate(response.results, start=1):
        typer.echo("")
        typer.echo(f"[{index}] {result.chunk_id}")
        typer.echo(f"score: {result.score:.6f}")
        typer.echo(f"distance: {_format_distance(result.distance)}")
        typer.echo(f"metadata: {result.metadata}")
        typer.echo(f"snippet: {result.snippet}")


@app.command()
def ask(
    question: str,
    top_k: int | None = typer.Option(
        None,
        "--top-k",
        min=1,
        help="Numero maximo de chunks a recuperar.",
    ),
    min_score: float | None = typer.Option(
        None,
        "--min-score",
        min=0.0,
        max=1.0,
        help="Score minimo aproximado de similitud.",
    ),
    show_chunks: bool = typer.Option(
        False,
        "--show-chunks",
        help="Mostrar chunks recuperados y sus snippets.",
    ),
    show_sources: bool = typer.Option(
        True,
        "--show-sources/--hide-sources",
        help="Mostrar u ocultar fuentes.",
    ),
    document_id: str | None = typer.Option(
        None,
        "--document-id",
        help="Filtrar por document_id.",
    ),
    file_name: str | None = typer.Option(
        None,
        "--file-name",
        help="Filtrar por nombre de archivo.",
    ),
) -> None:
    """Responde de forma extractiva usando solo chunks recuperados localmente."""
    settings = get_settings()
    logger = configure_logging(settings)

    try:
        response = LocalRagPipeline(settings, logger=logger).ask(
            question,
            top_k=top_k,
            min_score=min_score,
            document_id=document_id,
            file_name=file_name,
        )
    except Exception as exc:
        logger.error("Local RAG ask failed: %s", exc)
        typer.echo(f"Ask failed: {exc}")
        raise typer.Exit(code=1)

    typer.echo(f"Question: {response.question}")
    typer.echo("")
    typer.echo("Answer:")
    typer.echo(response.answer)

    if response.warning:
        typer.echo("")
        typer.echo(f"Warning: {response.warning}")

    typer.echo("")
    typer.echo(f"Context sufficient: {response.has_sufficient_context}")
    typer.echo(f"Context reason: {response.context.reason}")
    if response.rejection_reason:
        typer.echo(f"Rejection reason: {response.rejection_reason}")
    typer.echo(f"Retrieved chunk count: {len(response.retrieved_chunks)}")

    if show_sources and response.has_sufficient_context:
        typer.echo("")
        typer.echo("Sources:")
        if not response.sources:
            typer.echo("- No sources available.")
        for source in response.sources:
            typer.echo(f"- {source.source_label} | chunk_id={source.chunk_id}")

    if show_chunks:
        typer.echo("")
        typer.echo("Retrieved chunks:")
        for index, chunk in enumerate(response.retrieved_chunks, start=1):
            typer.echo(
                f"[{index}] score={chunk.score:.4f} "
                f"source={chunk.file_name} page={chunk.page_number} "
                f"chunk_id={chunk.chunk_id}"
            )
            typer.echo(chunk.snippet)


@app.command("eval-run")
def eval_run(
    limit: int | None = typer.Option(
        None,
        "--limit",
        min=1,
        help="Evaluar solo las primeras N preguntas.",
    ),
    output_dir: str | None = typer.Option(
        None,
        "--output-dir",
        help="Carpeta donde guardar reportes.",
    ),
) -> None:
    """Ejecuta evaluacion basica local del sistema RAG."""
    settings = get_settings()
    logger = configure_logging(settings)
    output_path = None if output_dir is None else settings.data_dir.parent / output_dir

    try:
        report, json_path, csv_path = run_evaluation(
            settings,
            output_dir=output_path,
            limit=limit,
        )
    except Exception as exc:
        logger.error("Evaluation failed: %s", exc)
        typer.echo(f"Evaluation failed: {exc}")
        raise typer.Exit(code=1)

    summary = report.summary
    typer.echo("Evaluation finished:")
    typer.echo(f"- Questions evaluated: {summary.total_questions}")
    typer.echo(f"- Passed: {summary.passed}")
    typer.echo(f"- Failed: {summary.failed}")
    typer.echo(f"- Pass rate: {summary.pass_rate:.2%}")
    typer.echo(f"- False positives: {summary.false_positive_count}")
    typer.echo(f"- False negatives: {summary.false_negative_count}")
    typer.echo(f"- Expected file hit rate: {summary.expected_file_hit_rate:.2%}")
    typer.echo(f"- Insufficient context: {summary.insufficient_context_count}")
    typer.echo(
        "- Out-of-domain correct rejections: "
        f"{summary.out_of_domain_correct_rejections}"
    )
    typer.echo(f"- JSON report: {json_path}")
    typer.echo(f"- CSV report: {csv_path}")


@app.command("eval-grid")
def eval_grid(
    limit: int | None = typer.Option(
        None,
        "--limit",
        min=1,
        help="Evaluar solo las primeras N preguntas.",
    ),
    top_k_values: str = typer.Option(
        "3,5,8",
        "--top-k-values",
        help="Valores de top_k separados por coma.",
    ),
    min_score_values: str = typer.Option(
        "0.2,0.3,0.4,0.5",
        "--min-score-values",
        help="Valores de min_score separados por coma.",
    ),
    output_dir: str | None = typer.Option(
        None,
        "--output-dir",
        help="Carpeta donde guardar reportes comparativos.",
    ),
) -> None:
    """Compara configuraciones simples de retrieval con el dataset local."""
    settings = get_settings()
    logger = configure_logging(settings)
    output_path = None if output_dir is None else settings.data_dir.parent / output_dir

    try:
        parsed_top_k = parse_int_values(top_k_values)
        parsed_min_scores = parse_float_values(min_score_values)
        report, json_path, csv_path = run_evaluation_grid(
            settings,
            output_dir=output_path,
            limit=limit,
            top_k_values=parsed_top_k,
            min_score_values=parsed_min_scores,
        )
    except Exception as exc:
        logger.error("Evaluation grid failed: %s", exc)
        typer.echo(f"Evaluation grid failed: {exc}")
        raise typer.Exit(code=1)

    typer.echo("Evaluation grid finished:")
    typer.echo(f"- Configurations evaluated: {len(report.results)}")
    typer.echo(f"- JSON report: {json_path}")
    typer.echo(f"- CSV report: {csv_path}")
    typer.echo(f"- Calibration report: {settings.eval_reports_dir / CALIBRATION_REPORT_NAME}")

    if report.recommendation is not None:
        recommendation = report.recommendation
        typer.echo("")
        typer.echo("Configuracion recomendada:")
        typer.echo(f"- top_k={recommendation.top_k}")
        typer.echo(f"- min_score={recommendation.min_score}")
        typer.echo(f"- pass_rate={recommendation.pass_rate:.2%}")
        typer.echo(f"- false_positives={recommendation.false_positive_count}")
        typer.echo(f"- expected_file_hit_rate={recommendation.expected_file_hit_rate:.2%}")
        typer.echo(f"- Motivo: {recommendation.reason}")


@app.command("eval-summary")
def eval_summary() -> None:
    """Muestra resumen del ultimo reporte de evaluacion."""
    settings = get_settings()
    configure_logging(settings).info("Evaluation summary requested")
    report_path = settings.eval_reports_dir / JSON_REPORT_NAME

    try:
        report = load_evaluation_report(report_path)
    except Exception as exc:
        typer.echo(f"Evaluation summary unavailable: {exc}")
        raise typer.Exit(code=1)

    summary = report.summary
    typer.echo(f"Report: {report_path}")
    typer.echo(f"Total questions: {summary.total_questions}")
    typer.echo(f"Passed: {summary.passed}")
    typer.echo(f"Failed: {summary.failed}")
    typer.echo(f"Pass rate: {summary.pass_rate:.2%}")
    typer.echo(f"False positives: {summary.false_positive_count}")
    typer.echo(f"False negatives: {summary.false_negative_count}")
    typer.echo(f"Expected file hit rate: {summary.expected_file_hit_rate:.2%}")
    typer.echo(f"Average keyword_hit_rate: {summary.avg_keyword_hit_rate:.3f}")
    typer.echo(f"Average source_count: {summary.avg_source_count:.2f}")
    typer.echo(
        "Average top_score: "
        f"{summary.avg_top_score:.3f}" if summary.avg_top_score is not None else "Average top_score: unavailable"
    )

    if summary.failure_reason_counts:
        typer.echo("")
        typer.echo("Failure reasons:")
        for reason, count in sorted(
            summary.failure_reason_counts.items(),
            key=lambda item: (-item[1], item[0]),
        ):
            typer.echo(f"- {reason}: {count}")

    failed = [result for result in report.results if not result.metrics.passed]
    if failed:
        typer.echo("")
        typer.echo("Top failed questions:")
        for result in failed[:5]:
            typer.echo(
                f"- {result.question_id}: behavior={result.metrics.expected_answer_behavior} "
                f"keywords={result.metrics.keyword_hit_rate:.2f} "
                f"file_hit={result.metrics.expected_file_hit} "
                f"reasons={','.join(result.failure_reasons) or 'unspecified'} "
                f"warning={result.warning or ''}"
            )

        typer.echo("")
        typer.echo(_evaluation_recommendation(summary))

    calibration = load_calibration_report(settings.eval_reports_dir)
    if calibration is not None:
        typer.echo("")
        typer.echo("Calibration recommendation:")
        final = calibration.get("recommendation_final") or {}
        typer.echo(f"- TOP_K: {final.get('TOP_K', 'unavailable')}")
        typer.echo(
            "- MIN_RETRIEVAL_SCORE: "
            f"{final.get('MIN_RETRIEVAL_SCORE', 'unavailable')}"
        )
        typer.echo(f"- MAX_SOURCES: {final.get('MAX_SOURCES', 'unavailable')}")
        typer.echo(
            "- MIN_CONTEXT_CHARS: "
            f"{final.get('MIN_CONTEXT_CHARS', 'unavailable')}"
        )
        metrics = calibration.get("metrics") or {}
        typer.echo(f"- pass_rate: {_format_percent(metrics.get('pass_rate'))}")
        typer.echo(f"- false_positives: {calibration.get('false_positive_count', 0)}")
        typer.echo(f"- false_negatives: {calibration.get('false_negative_count', 0)}")
        warnings = calibration.get("warnings") or []
        if warnings:
            typer.echo(f"- warnings: {', '.join(warnings)}")


@app.command()
def serve(
    host: str | None = typer.Option(
        None,
        "--host",
        help="Host para exponer la API local.",
    ),
    port: int | None = typer.Option(
        None,
        "--port",
        min=1,
        max=65535,
        help="Puerto para exponer la API local.",
    ),
    reload: bool = typer.Option(
        False,
        "--reload",
        help="Recargar automaticamente durante desarrollo.",
    ),
) -> None:
    """Levanta la API local con Uvicorn."""
    settings = get_settings()
    api_host = host or settings.api_host
    api_port = port or settings.api_port
    api_reload = reload or settings.api_reload

    try:
        import uvicorn
    except ImportError as exc:
        typer.echo(
            "uvicorn is not installed. Install/update dependencies with: "
            "python -m pip install -e ."
        )
        raise typer.Exit(code=1) from exc

    typer.echo(f"Starting API at http://{api_host}:{api_port}")
    uvicorn.run(
        "rag_chatbot.api.main:app",
        host=api_host,
        port=api_port,
        reload=api_reload,
    )


@app.command()
def ui(
    api_base_url: str | None = typer.Option(
        None,
        "--api-base-url",
        help="URL de la API local que consumira la interfaz.",
    ),
) -> None:
    """Levanta la interfaz local de chatbot con Streamlit."""
    try:
        import streamlit  # noqa: F401
    except ImportError as exc:
        typer.echo(
            "streamlit is not installed. Install/update dependencies with: "
            "python -m pip install -e ."
        )
        raise typer.Exit(code=1) from exc

    settings = get_settings()
    env = os.environ.copy()
    env["API_BASE_URL"] = api_base_url or settings.api_base_url
    app_path = PROJECT_ROOT / "src" / "rag_chatbot" / "ui" / "streamlit_app.py"

    typer.echo("Starting Streamlit UI")
    typer.echo(f"API base URL: {env['API_BASE_URL']}")
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(app_path)],
        env=env,
        check=False,
    )


@app.command()
def version() -> None:
    """Muestra la version del proyecto."""
    typer.echo(__version__)


def _evaluation_recommendation(summary) -> str:
    if summary.false_positive_count:
        return (
            "Recommendation: revisa preguntas fuera de dominio y prueba subir "
            "MIN_RETRIEVAL_SCORE o usar eval-grid para calibrar el umbral."
        )
    if summary.false_negative_count:
        return (
            "Recommendation: hay preguntas del dominio rechazadas; prueba bajar "
            "MIN_RETRIEVAL_SCORE, subir top_k o revisar chunking."
        )
    if summary.expected_file_hit_rate < 0.8:
        return (
            "Recommendation: bajo acierto de archivos esperados; revisa chunking, "
            "keywords del dataset y posibles mejoras futuras como reranking local."
        )
    if summary.avg_keyword_hit_rate < 0.6:
        return (
            "Recommendation: bajo keyword_hit_rate; amplia el dataset o revisa "
            "si los snippets recuperados conservan terminos clave."
        )
    return "Recommendation: resultados razonables; usa eval-grid para confirmar parametros."


def _format_percent(value) -> str:
    if value is None:
        return "unavailable"
    return f"{float(value):.2%}"


def main() -> None:
    app()


def _format_distance(distance: float | None) -> str:
    return "unavailable" if distance is None else f"{distance:.4f}"


if __name__ == "__main__":
    main()
