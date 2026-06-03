import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rag_chatbot.config import AppSettings
from rag_chatbot.schemas import EvaluationGridReport, EvaluationGridResult


CALIBRATION_REPORT_NAME = "calibration_recommendation.json"
SMALL_SCORE_MARGIN = 0.03
DEMO_MAX_SOURCES = 3


def build_calibration_report(
    grid_report: EvaluationGridReport,
    settings: AppSettings,
) -> dict[str, Any]:
    best_result = _best_result(grid_report)
    warnings = _build_warnings(best_result)
    recommendation = grid_report.recommendation

    best_parameters = (
        {
            "top_k": recommendation.top_k,
            "min_score": recommendation.min_score,
        }
        if recommendation is not None
        else None
    )
    final_recommendation = (
        {
            "TOP_K": recommendation.top_k,
            "MIN_RETRIEVAL_SCORE": recommendation.min_score,
            "MAX_SOURCES": min(DEMO_MAX_SOURCES, recommendation.top_k),
            "MIN_CONTEXT_CHARS": settings.min_context_chars,
        }
        if recommendation is not None
        else None
    )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_path": grid_report.dataset_path,
        "question_count": best_result.total_questions if best_result else 0,
        "best_parameters": best_parameters,
        "metrics": _metrics(best_result),
        "false_positive_count": best_result.false_positive_count if best_result else 0,
        "false_negative_count": best_result.false_negative_count if best_result else 0,
        "recommendation_final": final_recommendation,
        "warnings": warnings,
    }


def write_calibration_report(report: dict[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / CALIBRATION_REPORT_NAME
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path


def load_calibration_report(output_dir: Path) -> dict[str, Any] | None:
    path = output_dir / CALIBRATION_REPORT_NAME
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _best_result(grid_report: EvaluationGridReport) -> EvaluationGridResult | None:
    recommendation = grid_report.recommendation
    if recommendation is None:
        return None

    for result in grid_report.results:
        if result.top_k == recommendation.top_k and result.min_score == recommendation.min_score:
            return result
    return None


def _metrics(result: EvaluationGridResult | None) -> dict[str, Any]:
    if result is None:
        return {}

    return {
        "total_questions": result.total_questions,
        "passed": result.passed,
        "failed": result.failed,
        "pass_rate": result.pass_rate,
        "expected_file_hit_rate": result.expected_file_hit_rate,
        "avg_keyword_hit_rate": result.avg_keyword_hit_rate,
        "avg_top_score": result.avg_top_score,
        "avg_source_count": result.avg_source_count,
        "score_margin": result.score_margin,
    }


def _build_warnings(result: EvaluationGridResult | None) -> list[str]:
    if result is None:
        return ["no_recommendation_available"]

    warnings: list[str] = []
    if result.false_positive_count > 0:
        warnings.append("false_positives_remain")
    if result.false_negative_count > 0:
        warnings.append("false_negatives_remain")
    if result.pass_rate < 1.0:
        warnings.append("pass_rate_below_100")
    if result.score_margin is not None and result.score_margin < SMALL_SCORE_MARGIN:
        warnings.append("small_score_margin")
    return warnings
