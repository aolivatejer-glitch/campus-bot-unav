from pathlib import Path

from typer.testing import CliRunner

from rag_chatbot import cli as cli_module
from rag_chatbot.cli import app
from rag_chatbot.config import get_settings
from rag_chatbot.diagnostics import DiagnosticCheck, DiagnosticStatus, DoctorReport
from rag_chatbot.ingestion.pipeline import IngestionSummary
from rag_chatbot.local_pipeline import LocalPipelineResult
from rag_chatbot.schemas import (
    ChunkingResult,
    ContextSufficiencyResult,
    EvaluationGridRecommendation,
    EvaluationGridReport,
    EvaluationGridResult,
    EvaluationMetrics,
    EvaluationReport,
    EvaluationResult,
    EvaluationSummary,
    IndexInfo,
    IndexingResult,
    RagAnswer,
    RagSource,
    RetrievedChunk,
    RetrievalQuery,
    RetrievalResponse,
    RetrievalResult,
)


runner = CliRunner()


def _set_project_env(monkeypatch, tmp_path: Path) -> dict[str, Path]:
    path_values = {
        "DOCUMENTS_DIR": tmp_path / "Documentos",
        "DATA_DIR": tmp_path / "data",
        "PROCESSED_DIR": tmp_path / "data" / "processed",
        "EVAL_DIR": tmp_path / "data" / "eval",
        "EVAL_DATASET_PATH": tmp_path / "data" / "eval" / "evaluation_questions.jsonl",
        "EVAL_REPORTS_DIR": tmp_path / "data" / "eval" / "reports",
        "CHUNKS_DIR": tmp_path / "data" / "chunks",
        "CHUNKS_FILE": tmp_path / "data" / "chunks" / "chunks.jsonl",
        "STORAGE_DIR": tmp_path / "storage",
        "CHROMA_DIR": tmp_path / "storage" / "chroma",
        "MANIFEST_DB_PATH": tmp_path / "storage" / "manifest.sqlite",
        "LOG_DIR": tmp_path / "logs",
    }

    for key, value in path_values.items():
        monkeypatch.setenv(key, str(value))

    get_settings.cache_clear()
    return path_values


def test_health_command_responds(monkeypatch) -> None:
    monkeypatch.setenv("ALLOW_EXTERNAL_LLM", "false")
    monkeypatch.setenv("LLM_PROVIDER", "none")
    get_settings.cache_clear()

    result = runner.invoke(app, ["health"])

    assert result.exit_code == 0
    assert "OK - rag-chatbot" in result.output
    assert "External LLM enabled: False" in result.output


def test_init_dirs_command_creates_expected_directories(tmp_path, monkeypatch) -> None:
    path_values = _set_project_env(monkeypatch, tmp_path)
    result = runner.invoke(app, ["init-dirs"])

    assert result.exit_code == 0
    assert "Project directories are ready:" in result.output
    for key, value in path_values.items():
        if key in {"MANIFEST_DB_PATH", "CHUNKS_FILE", "EVAL_DATASET_PATH"}:
            continue
        assert Path(value).exists()


def test_doctor_command_responds_with_mock(tmp_path, monkeypatch) -> None:
    _set_project_env(monkeypatch, tmp_path)

    def fake_run_doctor(settings):
        return DoctorReport(
            checks=[
                DiagnosticCheck(
                    DiagnosticStatus.OK,
                    "Configuracion",
                    "Configuracion cargada",
                )
            ]
        )

    monkeypatch.setattr(cli_module, "run_doctor", fake_run_doctor)

    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 0
    assert "Doctor report:" in result.output
    assert "[OK] Configuracion: Configuracion cargada" in result.output
    assert "errors=0" in result.output


def test_doctor_command_exits_when_errors(tmp_path, monkeypatch) -> None:
    _set_project_env(monkeypatch, tmp_path)

    def fake_run_doctor(settings):
        return DoctorReport(
            checks=[
                DiagnosticCheck(
                    DiagnosticStatus.ERROR,
                    "Carpeta",
                    "No existe",
                )
            ]
        )

    monkeypatch.setattr(cli_module, "run_doctor", fake_run_doctor)

    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 1
    assert "[ERROR] Carpeta: No existe" in result.output


def test_ingest_command_responds(tmp_path, monkeypatch) -> None:
    path_values = _set_project_env(monkeypatch, tmp_path)
    documents_dir = path_values["DOCUMENTS_DIR"]
    documents_dir.mkdir(parents=True)
    (documents_dir / "sample.txt").write_text("Contenido de prueba", encoding="utf-8")

    result = runner.invoke(app, ["ingest"])

    assert result.exit_code == 0
    assert "Ingestion finished:" in result.output
    assert "Documents found: 1" in result.output
    assert "Processed: 1" in result.output


def test_list_documents_command_responds(tmp_path, monkeypatch) -> None:
    path_values = _set_project_env(monkeypatch, tmp_path)
    documents_dir = path_values["DOCUMENTS_DIR"]
    documents_dir.mkdir(parents=True)
    (documents_dir / "sample.txt").write_text("Contenido de prueba", encoding="utf-8")

    ingest_result = runner.invoke(app, ["ingest"])
    list_result = runner.invoke(app, ["list-documents"])

    assert ingest_result.exit_code == 0
    assert list_result.exit_code == 0
    assert "Documents:" in list_result.output
    assert "sample.txt" in list_result.output


def test_build_chunks_command_responds(tmp_path, monkeypatch) -> None:
    path_values = _set_project_env(monkeypatch, tmp_path)
    documents_dir = path_values["DOCUMENTS_DIR"]
    documents_dir.mkdir(parents=True)
    (documents_dir / "sample.txt").write_text("Contenido de prueba " * 30, encoding="utf-8")

    ingest_result = runner.invoke(app, ["ingest"])
    chunk_result = runner.invoke(app, ["build-chunks"])

    assert ingest_result.exit_code == 0
    assert chunk_result.exit_code == 0
    assert "Chunk build finished:" in chunk_result.output
    assert "Chunks generated:" in chunk_result.output


def test_list_chunks_command_responds(tmp_path, monkeypatch) -> None:
    path_values = _set_project_env(monkeypatch, tmp_path)
    documents_dir = path_values["DOCUMENTS_DIR"]
    documents_dir.mkdir(parents=True)
    (documents_dir / "sample.txt").write_text("Contenido de prueba " * 30, encoding="utf-8")

    runner.invoke(app, ["ingest"])
    runner.invoke(app, ["build-chunks"])
    result = runner.invoke(app, ["list-chunks", "--limit", "1"])

    assert result.exit_code == 0
    assert "Total chunks:" in result.output
    assert "Sample chunks:" in result.output
    assert "sample.txt" in result.output


def test_inspect_chunk_command_responds(tmp_path, monkeypatch) -> None:
    import json

    path_values = _set_project_env(monkeypatch, tmp_path)
    documents_dir = path_values["DOCUMENTS_DIR"]
    chunks_file = path_values["CHUNKS_FILE"]
    documents_dir.mkdir(parents=True)
    (documents_dir / "sample.txt").write_text("Contenido de prueba " * 30, encoding="utf-8")

    runner.invoke(app, ["ingest"])
    runner.invoke(app, ["build-chunks"])
    first_chunk = json.loads(chunks_file.read_text(encoding="utf-8").splitlines()[0])
    result = runner.invoke(app, ["inspect-chunk", first_chunk["chunk_id"]])

    assert result.exit_code == 0
    assert "Chunk ID:" in result.output
    assert "Preview:" in result.output
    assert "sample.txt" in result.output


def test_index_info_command_responds(tmp_path, monkeypatch) -> None:
    _set_project_env(monkeypatch, tmp_path)

    def fake_get_index_info(settings):
        return IndexInfo(
            collection_name="documents",
            chroma_dir=str(settings.chroma_dir),
            embedding_model=settings.embedding_model,
            embedding_device=settings.embedding_device,
            chroma_available=False,
            vector_count=None,
        )

    monkeypatch.setattr(cli_module, "get_index_info", fake_get_index_info)
    result = runner.invoke(app, ["index-info"])

    assert result.exit_code == 0
    assert "Collection: documents" in result.output
    assert "Vector count: unavailable" in result.output
    assert "Chroma available: False" in result.output


def test_build_index_command_responds_with_limit(tmp_path, monkeypatch) -> None:
    _set_project_env(monkeypatch, tmp_path)
    calls = {}

    def fake_build_index(settings, *, reset, limit, logger):
        calls["reset"] = reset
        calls["limit"] = limit
        return IndexingResult(
            chunks_read=2,
            chunks_indexed=2,
            chunks_skipped=0,
            collection_name="documents",
            chroma_dir=str(settings.chroma_dir),
            embedding_model=settings.embedding_model,
            embedding_device=settings.embedding_device,
        )

    monkeypatch.setattr(cli_module, "build_index", fake_build_index)
    result = runner.invoke(app, ["build-index", "--limit", "20"])

    assert result.exit_code == 0
    assert calls == {"reset": False, "limit": 20}
    assert "Index build finished:" in result.output
    assert "Chunks indexed: 2" in result.output


def test_run_local_pipeline_command_responds_with_mock(tmp_path, monkeypatch) -> None:
    _set_project_env(monkeypatch, tmp_path)
    calls = {}

    def fake_run_full_local_pipeline(
        settings,
        *,
        reset_index,
        limit,
        run_eval,
        logger,
    ):
        calls["reset_index"] = reset_index
        calls["limit"] = limit
        calls["run_eval"] = run_eval
        return LocalPipelineResult(
            ingestion=IngestionSummary(
                documents_found=1,
                processed=1,
                skipped=0,
                failed=0,
                requires_ocr=0,
            ),
            chunking=ChunkingResult(
                documents_processed=1,
                documents_skipped=0,
                total_chunks=2,
                chunks_file=str(settings.chunks_file),
                manifest_file=str(settings.chunks_dir / "chunk_manifest.json"),
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
                min_chunk_size=settings.min_chunk_size,
            ),
            indexing=IndexingResult(
                chunks_read=2,
                chunks_indexed=2,
                chunks_skipped=0,
                collection_name=settings.chroma_collection_name,
                chroma_dir=str(settings.chroma_dir),
                embedding_model=settings.embedding_model,
                embedding_device=settings.embedding_device,
            ),
        )

    monkeypatch.setattr(cli_module, "run_full_local_pipeline", fake_run_full_local_pipeline)

    result = runner.invoke(
        app,
        ["run-local-pipeline", "--reset-index", "--limit", "50", "--run-eval"],
    )

    assert result.exit_code == 0
    assert calls == {"reset_index": True, "limit": 50, "run_eval": True}
    assert "Local pipeline finished:" in result.output
    assert "Chunks generated: 2" in result.output
    assert "Chunks indexed: 2" in result.output


class FakeCliRetriever:
    vector_store = type("Store", (), {"count": lambda self: 10})()

    def __init__(self, settings, *, logger):
        self.settings = settings

    def search(
        self,
        question,
        *,
        top_k=None,
        min_score=None,
        document_id=None,
        file_name=None,
        mode=None,
    ):
        text = "Texto completo que solo debe mostrarse si se usa show-text."
        return RetrievalResponse(
            query=RetrievalQuery(
                question=question,
                top_k=top_k or self.settings.top_k,
                min_score=min_score if min_score is not None else self.settings.min_retrieval_score,
                document_id=document_id,
                file_name=file_name,
            ),
            results=[
                RetrievalResult(
                    chunk_id="chunk_1",
                    score=0.82,
                    distance=0.18,
                    text=text,
                    snippet="Texto resumido.",
                    document_id="doc_1",
                    file_name="sample.pdf",
                    file_path="Documentos/sample.pdf",
                    file_type="pdf",
                    page_number=4,
                    chunk_index=0,
                    char_count=len(text),
                    source_label="sample.pdf, pagina 4",
                    metadata={"document_id": "doc_1", "page_number": 4},
                )
            ],
            score_description="score description",
            collection_name="documents",
            embedding_model=self.settings.embedding_model,
            embedding_device=self.settings.embedding_device,
        )


def test_search_command_responds_without_full_text(tmp_path, monkeypatch) -> None:
    _set_project_env(monkeypatch, tmp_path)
    monkeypatch.setattr(cli_module, "SemanticRetriever", FakeCliRetriever)

    result = runner.invoke(app, ["search", "pregunta", "--top-k", "3"])

    assert result.exit_code == 0
    assert "Question: pregunta" in result.output
    assert "Results found: 1" in result.output
    assert "Texto resumido." in result.output
    assert "Texto completo que solo debe" not in result.output


def test_search_command_show_text_prints_full_text(tmp_path, monkeypatch) -> None:
    _set_project_env(monkeypatch, tmp_path)
    monkeypatch.setattr(cli_module, "SemanticRetriever", FakeCliRetriever)

    result = runner.invoke(app, ["search", "pregunta", "--show-text"])

    assert result.exit_code == 0
    assert "Texto completo que solo debe mostrarse" in result.output


def test_search_debug_command_responds(tmp_path, monkeypatch) -> None:
    _set_project_env(monkeypatch, tmp_path)
    monkeypatch.setattr(cli_module, "SemanticRetriever", FakeCliRetriever)

    result = runner.invoke(app, ["search-debug", "pregunta"])

    assert result.exit_code == 0
    assert "Embedding model:" in result.output
    assert "Vector count: 10" in result.output
    assert "metadata:" in result.output


class FakeCliRagPipeline:
    def __init__(self, settings, *, logger):
        self.settings = settings

    def ask(
        self,
        question,
        *,
        top_k=None,
        min_score=None,
        document_id=None,
        file_name=None,
        mode=None,
    ):
        text = "Texto completo recuperado para la respuesta extractiva."
        return RagAnswer(
            question=question,
            answer="Con base en los documentos recuperados, se encontro lo siguiente.",
            has_sufficient_context=True,
            warning=None,
            sources=[
                RagSource(
                    file_name="sample.pdf",
                    page_number=4,
                    chunk_id="chunk_1",
                    source_label="sample.pdf, pagina 4",
                    snippet="Fragmento citado.",
                )
            ],
            retrieved_chunks=[
                RetrievedChunk(
                    chunk_id="chunk_1",
                    score=0.82,
                    file_name="sample.pdf",
                    page_number=4,
                    text=text,
                    snippet="Fragmento citado.",
                )
            ],
            context=ContextSufficiencyResult(
                has_sufficient_context=True,
                warning=None,
                reason="sufficient",
                best_score=0.82,
                total_context_chars=len(text),
                result_count=1,
            ),
        )


def test_ask_command_responds_without_chunks_by_default(tmp_path, monkeypatch) -> None:
    _set_project_env(monkeypatch, tmp_path)
    monkeypatch.setattr(cli_module, "LocalRagPipeline", FakeCliRagPipeline)

    result = runner.invoke(app, ["ask", "pregunta"])

    assert result.exit_code == 0
    assert "Question: pregunta" in result.output
    assert "Answer:" in result.output
    assert "Sources:" in result.output
    assert "Retrieved chunks:" not in result.output
    assert "Fragmento citado." not in result.output


def test_ask_command_show_chunks_prints_chunks(tmp_path, monkeypatch) -> None:
    _set_project_env(monkeypatch, tmp_path)
    monkeypatch.setattr(cli_module, "LocalRagPipeline", FakeCliRagPipeline)

    result = runner.invoke(app, ["ask", "pregunta", "--show-chunks"])

    assert result.exit_code == 0
    assert "Retrieved chunks:" in result.output
    assert "Fragmento citado." in result.output


def _fake_evaluation_report(tmp_path: Path) -> EvaluationReport:
    return EvaluationReport(
        generated_at="2026-06-02T00:00:00+00:00",
        dataset_path=str(tmp_path / "questions.jsonl"),
        results=[
            EvaluationResult(
                question_id="q1",
                question="pregunta",
                should_have_answer=True,
                expected_keywords=["clave"],
                expected_files=["sample.pdf"],
                answer="respuesta con clave",
                answer_preview="respuesta con clave",
                has_sufficient_context=True,
                warning=None,
                sources=[
                    RagSource(
                        file_name="sample.pdf",
                        page_number=1,
                        chunk_id="chunk_1",
                        source_label="sample.pdf, pagina 1",
                        snippet="clave",
                    )
                ],
                retrieved_chunks=[
                    RetrievedChunk(
                        chunk_id="chunk_1",
                        score=0.9,
                        file_name="sample.pdf",
                        page_number=1,
                        text="clave",
                        snippet="clave",
                    )
                ],
                retrieved_files=["sample.pdf"],
                found_keywords=["clave"],
                missing_keywords=[],
                expected_files_found=["sample.pdf"],
                expected_files_missing=[],
                metrics=EvaluationMetrics(
                    has_answer=True,
                    expected_answer_behavior=True,
                    keyword_hit_rate=1.0,
                    expected_file_hit=True,
                    source_count=1,
                    retrieved_chunk_count=1,
                    top_score=0.9,
                    avg_score=0.9,
                    passed=True,
                ),
            )
        ],
        summary=EvaluationSummary(
            total_questions=1,
            passed=1,
            failed=0,
            pass_rate=1.0,
            insufficient_context_count=0,
            out_of_domain_correct_rejections=0,
            avg_keyword_hit_rate=1.0,
            avg_top_score=0.9,
        ),
    )


def test_eval_run_command_responds(tmp_path, monkeypatch) -> None:
    _set_project_env(monkeypatch, tmp_path)

    def fake_run_evaluation(settings, *, output_dir, limit):
        report = _fake_evaluation_report(tmp_path)
        return report, tmp_path / "report.json", tmp_path / "report.csv"

    monkeypatch.setattr(cli_module, "run_evaluation", fake_run_evaluation)
    result = runner.invoke(app, ["eval-run", "--limit", "1"])

    assert result.exit_code == 0
    assert "Evaluation finished:" in result.output
    assert "Questions evaluated: 1" in result.output
    assert "Pass rate: 100.00%" in result.output


def test_eval_grid_command_responds(tmp_path, monkeypatch) -> None:
    _set_project_env(monkeypatch, tmp_path)
    calls = {}

    def fake_run_evaluation_grid(
        settings,
        *,
        output_dir,
        limit,
        top_k_values,
        min_score_values,
    ):
        calls["limit"] = limit
        calls["top_k_values"] = top_k_values
        calls["min_score_values"] = min_score_values
        report = EvaluationGridReport(
            generated_at="2026-06-02T00:00:00+00:00",
            dataset_path=str(settings.eval_dataset_path),
            results=[
                EvaluationGridResult(
                    top_k=3,
                    min_score=0.3,
                    total_questions=1,
                    passed=1,
                    failed=0,
                    pass_rate=1.0,
                    avg_keyword_hit_rate=1.0,
                    expected_file_hit_rate=1.0,
                    false_positive_count=0,
                    false_negative_count=0,
                    avg_top_score=0.9,
                    avg_source_count=1.0,
                )
            ],
            recommendation=EvaluationGridRecommendation(
                top_k=3,
                min_score=0.3,
                reason="mejor configuracion",
                pass_rate=1.0,
                false_positive_count=0,
                expected_file_hit_rate=1.0,
            ),
        )
        return report, tmp_path / "grid.json", tmp_path / "grid.csv"

    monkeypatch.setattr(cli_module, "run_evaluation_grid", fake_run_evaluation_grid)
    result = runner.invoke(
        app,
        [
            "eval-grid",
            "--limit",
            "2",
            "--top-k-values",
            "3,5",
            "--min-score-values",
            "0.2,0.3",
        ],
    )

    assert result.exit_code == 0
    assert calls == {
        "limit": 2,
        "top_k_values": [3, 5],
        "min_score_values": [0.2, 0.3],
    }
    assert "Evaluation grid finished:" in result.output
    assert "Configuracion recomendada:" in result.output
    assert "top_k=3" in result.output


def test_eval_summary_command_responds(tmp_path, monkeypatch) -> None:
    path_values = _set_project_env(monkeypatch, tmp_path)
    reports_dir = path_values["EVAL_REPORTS_DIR"]
    reports_dir.mkdir(parents=True)
    report = _fake_evaluation_report(tmp_path)
    (reports_dir / "evaluation_report.json").write_text(
        report.model_dump_json(),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["eval-summary"])

    assert result.exit_code == 0
    assert "Total questions: 1" in result.output
    assert "Passed: 1" in result.output
    assert "Average keyword_hit_rate: 1.000" in result.output


def test_eval_summary_command_shows_failure_reasons(tmp_path, monkeypatch) -> None:
    path_values = _set_project_env(monkeypatch, tmp_path)
    reports_dir = path_values["EVAL_REPORTS_DIR"]
    reports_dir.mkdir(parents=True)
    result = _fake_evaluation_report(tmp_path).results[0]
    result.metrics.passed = False
    result.failure_reasons = ["low_keyword_hit_rate", "expected_file_not_found"]
    report = EvaluationReport(
        generated_at="2026-06-02T00:00:00+00:00",
        dataset_path=str(tmp_path / "questions.jsonl"),
        results=[result],
        summary=EvaluationSummary(
            total_questions=1,
            passed=0,
            failed=1,
            pass_rate=0.0,
            insufficient_context_count=0,
            out_of_domain_correct_rejections=0,
            avg_keyword_hit_rate=0.0,
            avg_top_score=0.2,
            false_positive_count=0,
            false_negative_count=0,
            expected_file_hit_rate=0.0,
            avg_source_count=1.0,
            failure_reason_counts={
                "low_keyword_hit_rate": 1,
                "expected_file_not_found": 1,
            },
        ),
    )
    (reports_dir / "evaluation_report.json").write_text(
        report.model_dump_json(),
        encoding="utf-8",
    )

    cli_result = runner.invoke(app, ["eval-summary"])

    assert cli_result.exit_code == 0
    assert "Failure reasons:" in cli_result.output
    assert "low_keyword_hit_rate: 1" in cli_result.output
    assert "Top failed questions:" in cli_result.output


def test_eval_summary_command_shows_calibration_recommendation(tmp_path, monkeypatch) -> None:
    import json

    path_values = _set_project_env(monkeypatch, tmp_path)
    reports_dir = path_values["EVAL_REPORTS_DIR"]
    reports_dir.mkdir(parents=True)
    report = _fake_evaluation_report(tmp_path)
    (reports_dir / "evaluation_report.json").write_text(
        report.model_dump_json(),
        encoding="utf-8",
    )
    (reports_dir / "calibration_recommendation.json").write_text(
        json.dumps(
            {
                "recommendation_final": {
                    "TOP_K": 3,
                    "MIN_RETRIEVAL_SCORE": 0.84,
                    "MAX_SOURCES": 3,
                    "MIN_CONTEXT_CHARS": 500,
                },
                "metrics": {"pass_rate": 1.0},
                "false_positive_count": 0,
                "false_negative_count": 0,
                "warnings": [],
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["eval-summary"])

    assert result.exit_code == 0
    assert "Calibration recommendation:" in result.output
    assert "MIN_RETRIEVAL_SCORE: 0.84" in result.output


def test_serve_command_is_registered() -> None:
    result = runner.invoke(app, ["serve", "--help"])

    assert result.exit_code == 0
    assert "Levanta la API local" in result.output


def test_ui_command_is_registered() -> None:
    result = runner.invoke(app, ["ui", "--help"])

    assert result.exit_code == 0
    assert "Levanta la interfaz local" in result.output
