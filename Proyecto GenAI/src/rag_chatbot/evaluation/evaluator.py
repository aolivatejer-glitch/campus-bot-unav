from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from rag_chatbot.config import AppSettings, get_settings
from rag_chatbot.evaluation.dataset import load_evaluation_questions
from rag_chatbot.evaluation.metrics import (
    calculate_evaluation_metrics,
    calculate_expected_file_hits,
    calculate_failure_reasons,
    calculate_keyword_hits,
    calculate_scores,
)
from rag_chatbot.evaluation.reporting import build_summary, write_evaluation_report
from rag_chatbot.rag.pipeline import LocalRagPipeline
from rag_chatbot.schemas import EvaluationReport, EvaluationResult, RagAnswer


class RagPipelineProtocol(Protocol):
    def ask(
        self,
        question: str,
        *,
        top_k: int | None = None,
        min_score: float | None = None,
        document_id: str | None = None,
        file_name: str | None = None,
    ) -> RagAnswer:
        ...


def run_evaluation(
    settings: AppSettings | None = None,
    *,
    dataset_path: Path | None = None,
    output_dir: Path | None = None,
    limit: int | None = None,
    top_k: int | None = None,
    min_score: float | None = None,
    pipeline: RagPipelineProtocol | None = None,
) -> tuple[EvaluationReport, Path, Path]:
    settings = settings or get_settings()
    dataset_path = dataset_path or settings.eval_dataset_path
    output_dir = output_dir or settings.eval_reports_dir
    settings.ensure_directories()

    questions = load_evaluation_questions(dataset_path, limit=limit)
    rag_pipeline = pipeline or LocalRagPipeline(settings)
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
    report = EvaluationReport(
        generated_at=datetime.now(timezone.utc),
        dataset_path=str(dataset_path),
        results=results,
        summary=build_summary(results),
    )
    json_path, csv_path = write_evaluation_report(report, output_dir)

    return report, json_path, csv_path


def evaluate_question(
    question,
    pipeline: RagPipelineProtocol,
    *,
    answer_preview_chars: int,
    top_k: int | None = None,
    min_score: float | None = None,
) -> EvaluationResult:
    answer = pipeline.ask(question.question, top_k=top_k, min_score=min_score)
    retrieved_files = _unique_files([source.file_name for source in answer.sources])
    found_keywords, missing_keywords, keyword_hit_rate = calculate_keyword_hits(
        question.expected_keywords,
        answer=answer.answer,
        retrieved_chunks=answer.retrieved_chunks,
    )
    expected_files_found, expected_files_missing, expected_file_hit = (
        calculate_expected_file_hits(question.expected_files, retrieved_files)
    )
    top_score, avg_score = calculate_scores(answer.retrieved_chunks)
    metrics = calculate_evaluation_metrics(
        question,
        answer,
        keyword_hit_rate=keyword_hit_rate,
        expected_file_hit=expected_file_hit,
        top_score=top_score,
        avg_score=avg_score,
    )
    failure_reasons = (
        []
        if metrics.passed
        else calculate_failure_reasons(
            question,
            answer,
            keyword_hit_rate=keyword_hit_rate,
            expected_file_hit=expected_file_hit,
            top_score=top_score,
        )
    )

    return EvaluationResult(
        question_id=question.question_id,
        question=question.question,
        should_have_answer=question.should_have_answer,
        expected_keywords=question.expected_keywords,
        expected_files=question.expected_files,
        answer=answer.answer,
        answer_preview=_preview(answer.answer, answer_preview_chars),
        has_sufficient_context=answer.has_sufficient_context,
        warning=answer.warning,
        sources=answer.sources,
        retrieved_chunks=answer.retrieved_chunks,
        retrieved_files=retrieved_files,
        found_keywords=found_keywords,
        missing_keywords=missing_keywords,
        expected_files_found=expected_files_found,
        expected_files_missing=expected_files_missing,
        metrics=metrics,
        failure_reasons=failure_reasons,
        notes=question.notes,
    )


def _unique_files(file_names: list[str]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()

    for file_name in file_names:
        key = file_name.casefold()
        if key in seen:
            continue
        seen.add(key)
        unique.append(file_name)

    return unique


def _preview(text: str, max_chars: int) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[:max_chars].rstrip() + "..."
