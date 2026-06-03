import csv
import json
from collections import Counter
from pathlib import Path

from rag_chatbot.schemas import (
    EvaluationGridReport,
    EvaluationGridResult,
    EvaluationReport,
    EvaluationResult,
    EvaluationSummary,
)


JSON_REPORT_NAME = "evaluation_report.json"
CSV_REPORT_NAME = "evaluation_report.csv"
GRID_JSON_REPORT_NAME = "evaluation_grid_report.json"
GRID_CSV_REPORT_NAME = "evaluation_grid_report.csv"


def write_evaluation_report(
    report: EvaluationReport,
    output_dir: Path,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / JSON_REPORT_NAME
    csv_path = output_dir / CSV_REPORT_NAME

    json_path.write_text(
        json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_csv(report.results, csv_path)

    return json_path, csv_path


def load_evaluation_report(json_path: Path) -> EvaluationReport:
    if not json_path.exists():
        raise FileNotFoundError(f"Evaluation report not found: {json_path}")

    return EvaluationReport.model_validate_json(json_path.read_text(encoding="utf-8"))


def write_evaluation_grid_report(
    report: EvaluationGridReport,
    output_dir: Path,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / GRID_JSON_REPORT_NAME
    csv_path = output_dir / GRID_CSV_REPORT_NAME

    json_path.write_text(
        json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_grid_csv(report.results, csv_path)

    return json_path, csv_path


def build_summary(results: list[EvaluationResult]) -> EvaluationSummary:
    total = len(results)
    passed = sum(1 for result in results if result.metrics.passed)
    failed = total - passed
    insufficient = sum(1 for result in results if not result.has_sufficient_context)
    out_of_domain_rejections = sum(
        1
        for result in results
        if not result.should_have_answer and not result.has_sufficient_context
    )
    avg_keyword = (
        sum(result.metrics.keyword_hit_rate for result in results) / total
        if total
        else 0.0
    )
    top_scores = [
        result.metrics.top_score
        for result in results
        if result.metrics.top_score is not None
    ]
    avg_top_score = sum(top_scores) / len(top_scores) if top_scores else None
    false_positive_count = sum(
        1
        for result in results
        if not result.should_have_answer and result.has_sufficient_context
    )
    false_negative_count = sum(
        1
        for result in results
        if result.should_have_answer and not result.has_sufficient_context
    )
    expected_file_hit_rate = calculate_expected_file_hit_rate(results)
    avg_source_count = (
        sum(result.metrics.source_count for result in results) / total
        if total
        else 0.0
    )
    failure_reason_counts = count_failure_reasons(results)

    return EvaluationSummary(
        total_questions=total,
        passed=passed,
        failed=failed,
        pass_rate=(passed / total) if total else 0.0,
        insufficient_context_count=insufficient,
        out_of_domain_correct_rejections=out_of_domain_rejections,
        avg_keyword_hit_rate=avg_keyword,
        avg_top_score=avg_top_score,
        false_positive_count=false_positive_count,
        false_negative_count=false_negative_count,
        expected_file_hit_rate=expected_file_hit_rate,
        avg_source_count=avg_source_count,
        failure_reason_counts=failure_reason_counts,
    )


def calculate_expected_file_hit_rate(results: list[EvaluationResult]) -> float:
    applicable = [
        result
        for result in results
        if result.should_have_answer and result.expected_files
    ]
    if not applicable:
        return 0.0

    hits = sum(1 for result in applicable if result.metrics.expected_file_hit)
    return hits / len(applicable)


def count_failure_reasons(results: list[EvaluationResult]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for result in results:
        counter.update(result.failure_reasons)
    return dict(counter)


def _write_csv(results: list[EvaluationResult], csv_path: Path) -> None:
    fieldnames = [
        "question_id",
        "question",
        "should_have_answer",
        "has_sufficient_context",
        "passed",
        "keyword_hit_rate",
        "expected_file_hit",
        "source_count",
        "retrieved_chunk_count",
        "top_score",
        "avg_score",
        "warning",
        "retrieved_files",
        "expected_files",
        "failure_reasons",
        "answer_preview",
    ]

    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "question_id": result.question_id,
                    "question": result.question,
                    "should_have_answer": result.should_have_answer,
                    "has_sufficient_context": result.has_sufficient_context,
                    "passed": result.metrics.passed,
                    "keyword_hit_rate": result.metrics.keyword_hit_rate,
                    "expected_file_hit": result.metrics.expected_file_hit,
                    "source_count": result.metrics.source_count,
                    "retrieved_chunk_count": result.metrics.retrieved_chunk_count,
                    "top_score": result.metrics.top_score,
                    "avg_score": result.metrics.avg_score,
                    "warning": result.warning or "",
                    "retrieved_files": "; ".join(result.retrieved_files),
                    "expected_files": "; ".join(result.expected_files),
                    "failure_reasons": "; ".join(result.failure_reasons),
                    "answer_preview": result.answer_preview,
                }
            )


def _write_grid_csv(results: list[EvaluationGridResult], csv_path: Path) -> None:
    fieldnames = [
        "top_k",
        "min_score",
        "total_questions",
        "passed",
        "failed",
        "pass_rate",
        "avg_keyword_hit_rate",
        "expected_file_hit_rate",
        "false_positive_count",
        "false_negative_count",
        "avg_top_score",
        "avg_source_count",
        "score_margin",
        "failure_reason_counts",
    ]

    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "top_k": result.top_k,
                    "min_score": result.min_score,
                    "total_questions": result.total_questions,
                    "passed": result.passed,
                    "failed": result.failed,
                    "pass_rate": result.pass_rate,
                    "avg_keyword_hit_rate": result.avg_keyword_hit_rate,
                    "expected_file_hit_rate": result.expected_file_hit_rate,
                    "false_positive_count": result.false_positive_count,
                    "false_negative_count": result.false_negative_count,
                    "avg_top_score": result.avg_top_score,
                    "avg_source_count": result.avg_source_count,
                    "score_margin": result.score_margin,
                    "failure_reason_counts": json.dumps(
                        result.failure_reason_counts,
                        ensure_ascii=False,
                    ),
                }
            )
