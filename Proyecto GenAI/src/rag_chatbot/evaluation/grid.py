from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from rag_chatbot.config import AppSettings, get_settings
from rag_chatbot.evaluation.calibration import (
    build_calibration_report,
    write_calibration_report,
)
from rag_chatbot.evaluation.dataset import load_evaluation_questions
from rag_chatbot.evaluation.evaluator import RagPipelineProtocol, evaluate_question
from rag_chatbot.evaluation.reporting import build_summary, write_evaluation_grid_report
from rag_chatbot.rag.pipeline import LocalRagPipeline
from rag_chatbot.schemas import (
    EvaluationGridRecommendation,
    EvaluationGridReport,
    EvaluationGridResult,
    EvaluationResult,
)


DEFAULT_TOP_K_VALUES = (3, 5, 8)
DEFAULT_MIN_SCORE_VALUES = (0.2, 0.3, 0.4, 0.5)


def run_evaluation_grid(
    settings: AppSettings | None = None,
    *,
    dataset_path: Path | None = None,
    output_dir: Path | None = None,
    limit: int | None = None,
    top_k_values: Sequence[int] = DEFAULT_TOP_K_VALUES,
    min_score_values: Sequence[float] = DEFAULT_MIN_SCORE_VALUES,
    pipeline: RagPipelineProtocol | None = None,
) -> tuple[EvaluationGridReport, Path, Path]:
    settings = settings or get_settings()
    dataset_path = dataset_path or settings.eval_dataset_path
    output_dir = output_dir or settings.eval_reports_dir
    settings.ensure_directories()

    questions = load_evaluation_questions(dataset_path, limit=limit)
    rag_pipeline = pipeline or LocalRagPipeline(settings)
    rows: list[EvaluationGridResult] = []

    for top_k in top_k_values:
        for min_score in min_score_values:
            results = [
                evaluate_question(
                    question,
                    rag_pipeline,
                    answer_preview_chars=settings.eval_answer_preview_chars,
                    top_k=top_k,
                    min_score=min_score,
                )
                for question in questions
            ]
            rows.append(build_grid_result(top_k, min_score, results))

    recommendation = recommend_grid_config(rows)
    report = EvaluationGridReport(
        generated_at=datetime.now(timezone.utc),
        dataset_path=str(dataset_path),
        results=rows,
        recommendation=recommendation,
    )
    json_path, csv_path = write_evaluation_grid_report(report, output_dir)
    write_calibration_report(build_calibration_report(report, settings), output_dir)

    return report, json_path, csv_path


def build_grid_result(
    top_k: int,
    min_score: float,
    results: list[EvaluationResult],
) -> EvaluationGridResult:
    summary = build_summary(results)
    return EvaluationGridResult(
        top_k=top_k,
        min_score=min_score,
        total_questions=summary.total_questions,
        passed=summary.passed,
        failed=summary.failed,
        pass_rate=summary.pass_rate,
        avg_keyword_hit_rate=summary.avg_keyword_hit_rate,
        expected_file_hit_rate=summary.expected_file_hit_rate,
        false_positive_count=summary.false_positive_count,
        false_negative_count=summary.false_negative_count,
        avg_top_score=summary.avg_top_score,
        avg_source_count=summary.avg_source_count,
        score_margin=calculate_score_margin(results),
        failure_reason_counts=summary.failure_reason_counts,
    )


def recommend_grid_config(
    rows: Sequence[EvaluationGridResult],
) -> EvaluationGridRecommendation | None:
    if not rows:
        return None

    best = sorted(
        rows,
        key=lambda row: (
            -row.pass_rate,
            row.false_positive_count,
            -row.expected_file_hit_rate,
            row.top_k,
            row.min_score,
        ),
    )[0]

    reason = (
        "mejor pass_rate con bajo numero de falsos positivos; "
        "en empate prioriza archivos esperados y menor top_k."
    )
    return EvaluationGridRecommendation(
        top_k=best.top_k,
        min_score=best.min_score,
        reason=reason,
        pass_rate=best.pass_rate,
        false_positive_count=best.false_positive_count,
        expected_file_hit_rate=best.expected_file_hit_rate,
    )


def parse_int_values(raw_values: str) -> list[int]:
    values = []
    for raw_value in _split_csv_values(raw_values):
        try:
            value = int(raw_value)
        except ValueError as exc:
            raise ValueError(f"Invalid integer value: {raw_value}") from exc
        if value <= 0:
            raise ValueError("top_k values must be greater than 0")
        values.append(value)
    return values


def parse_float_values(raw_values: str) -> list[float]:
    values = []
    for raw_value in _split_csv_values(raw_values):
        try:
            value = float(raw_value)
        except ValueError as exc:
            raise ValueError(f"Invalid float value: {raw_value}") from exc
        if value < 0.0 or value > 1.0:
            raise ValueError("min_score values must be between 0.0 and 1.0")
        values.append(value)
    return values


def calculate_score_margin(results: list[EvaluationResult]) -> float | None:
    in_domain_scores = [
        result.metrics.top_score
        for result in results
        if result.should_have_answer and result.metrics.top_score is not None
    ]
    out_of_domain_scores = [
        result.metrics.top_score
        for result in results
        if not result.should_have_answer and result.metrics.top_score is not None
    ]

    if not in_domain_scores or not out_of_domain_scores:
        return None

    return min(in_domain_scores) - max(out_of_domain_scores)


def _split_csv_values(raw_values: str) -> list[str]:
    values = [value.strip() for value in raw_values.split(",") if value.strip()]
    if not values:
        raise ValueError("At least one value is required")
    return values
