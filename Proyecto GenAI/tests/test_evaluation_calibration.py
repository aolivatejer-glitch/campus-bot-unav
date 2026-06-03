from datetime import datetime, timezone

from rag_chatbot.config import AppSettings
from rag_chatbot.evaluation.calibration import (
    CALIBRATION_REPORT_NAME,
    build_calibration_report,
    load_calibration_report,
    write_calibration_report,
)
from rag_chatbot.schemas import (
    EvaluationGridRecommendation,
    EvaluationGridReport,
    EvaluationGridResult,
)


def _settings(tmp_path) -> AppSettings:
    return AppSettings(
        documents_dir=tmp_path / "Documentos",
        data_dir=tmp_path / "data",
        processed_dir=tmp_path / "data" / "processed",
        eval_dir=tmp_path / "data" / "eval",
        eval_dataset_path=tmp_path / "data" / "eval" / "questions.jsonl",
        eval_reports_dir=tmp_path / "data" / "eval" / "reports",
        chunks_dir=tmp_path / "data" / "chunks",
        chunks_file=tmp_path / "data" / "chunks" / "chunks.jsonl",
        storage_dir=tmp_path / "storage",
        chroma_dir=tmp_path / "storage" / "chroma",
        manifest_db_path=tmp_path / "storage" / "manifest.sqlite",
        log_dir=tmp_path / "logs",
        min_context_chars=500,
        _env_file=None,
    )


def test_calibration_report_recommends_demo_parameters(tmp_path) -> None:
    settings = _settings(tmp_path)
    grid_report = EvaluationGridReport(
        generated_at=datetime.now(timezone.utc),
        dataset_path=str(settings.eval_dataset_path),
        results=[
            EvaluationGridResult(
                top_k=3,
                min_score=0.84,
                total_questions=36,
                passed=36,
                failed=0,
                pass_rate=1.0,
                avg_keyword_hit_rate=0.8,
                expected_file_hit_rate=1.0,
                false_positive_count=0,
                false_negative_count=0,
                avg_top_score=0.86,
                avg_source_count=2.5,
                score_margin=0.04,
            )
        ],
        recommendation=EvaluationGridRecommendation(
            top_k=3,
            min_score=0.84,
            reason="best",
            pass_rate=1.0,
            false_positive_count=0,
            expected_file_hit_rate=1.0,
        ),
    )

    report = build_calibration_report(grid_report, settings)
    output_path = write_calibration_report(report, settings.eval_reports_dir)
    loaded = load_calibration_report(settings.eval_reports_dir)

    assert output_path.name == CALIBRATION_REPORT_NAME
    assert loaded is not None
    assert loaded["recommendation_final"]["TOP_K"] == 3
    assert loaded["recommendation_final"]["MIN_RETRIEVAL_SCORE"] == 0.84
    assert loaded["recommendation_final"]["MAX_SOURCES"] == 3
    assert loaded["recommendation_final"]["MIN_CONTEXT_CHARS"] == 500


def test_calibration_report_warns_when_false_positives_remain(tmp_path) -> None:
    settings = _settings(tmp_path)
    grid_report = EvaluationGridReport(
        generated_at=datetime.now(timezone.utc),
        dataset_path=str(settings.eval_dataset_path),
        results=[
            EvaluationGridResult(
                top_k=3,
                min_score=0.84,
                total_questions=36,
                passed=34,
                failed=2,
                pass_rate=34 / 36,
                avg_keyword_hit_rate=0.8,
                expected_file_hit_rate=1.0,
                false_positive_count=2,
                false_negative_count=0,
                avg_top_score=0.86,
                avg_source_count=2.5,
                score_margin=0.01,
            )
        ],
        recommendation=EvaluationGridRecommendation(
            top_k=3,
            min_score=0.84,
            reason="best",
            pass_rate=34 / 36,
            false_positive_count=2,
            expected_file_hit_rate=1.0,
        ),
    )

    report = build_calibration_report(grid_report, settings)

    assert "false_positives_remain" in report["warnings"]
    assert "small_score_margin" in report["warnings"]
