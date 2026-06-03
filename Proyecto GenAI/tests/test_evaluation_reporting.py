from datetime import datetime, timezone

from rag_chatbot.evaluation.reporting import (
    CSV_REPORT_NAME,
    GRID_CSV_REPORT_NAME,
    GRID_JSON_REPORT_NAME,
    JSON_REPORT_NAME,
    build_summary,
    load_evaluation_report,
    write_evaluation_grid_report,
    write_evaluation_report,
)
from rag_chatbot.schemas import (
    EvaluationGridReport,
    EvaluationGridResult,
    EvaluationMetrics,
    EvaluationReport,
    EvaluationResult,
)


def _result(passed: bool = True) -> EvaluationResult:
    return EvaluationResult(
        question_id="q1",
        question="Pregunta",
        should_have_answer=True,
        expected_keywords=["clave"],
        expected_files=["archivo.pdf"],
        answer="respuesta",
        answer_preview="respuesta",
        has_sufficient_context=True,
        sources=[],
        retrieved_chunks=[],
        retrieved_files=["archivo.pdf"],
        found_keywords=["clave"],
        missing_keywords=[],
        expected_files_found=["archivo.pdf"],
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
            passed=passed,
        ),
        failure_reasons=[] if passed else ["expected_file_not_found"],
    )


def test_build_summary_counts_results() -> None:
    summary = build_summary([_result(True), _result(False)])

    assert summary.total_questions == 2
    assert summary.passed == 1
    assert summary.failed == 1
    assert summary.pass_rate == 0.5
    assert summary.expected_file_hit_rate == 1.0
    assert summary.failure_reason_counts == {"expected_file_not_found": 1}


def test_write_report_creates_json_and_csv(tmp_path) -> None:
    results = [_result()]
    report = EvaluationReport(
        generated_at=datetime.now(timezone.utc),
        dataset_path="questions.jsonl",
        results=results,
        summary=build_summary(results),
    )

    json_path, csv_path = write_evaluation_report(report, tmp_path)

    assert json_path.name == JSON_REPORT_NAME
    assert csv_path.name == CSV_REPORT_NAME
    assert json_path.exists()
    assert csv_path.exists()
    assert "question_id" in csv_path.read_text(encoding="utf-8-sig")
    assert load_evaluation_report(json_path).summary.total_questions == 1


def test_build_summary_counts_false_positives_and_false_negatives() -> None:
    false_positive = _result(False)
    false_positive.should_have_answer = False
    false_positive.has_sufficient_context = True
    false_positive.failure_reasons = ["answered_when_should_reject"]

    false_negative = _result(False)
    false_negative.should_have_answer = True
    false_negative.has_sufficient_context = False
    false_negative.failure_reasons = ["rejected_when_should_answer"]

    summary = build_summary([false_positive, false_negative])

    assert summary.false_positive_count == 1
    assert summary.false_negative_count == 1


def test_write_grid_report_creates_json_and_csv(tmp_path) -> None:
    report = EvaluationGridReport(
        generated_at=datetime.now(timezone.utc),
        dataset_path="questions.jsonl",
        results=[
            EvaluationGridResult(
                top_k=5,
                min_score=0.3,
                total_questions=2,
                passed=1,
                failed=1,
                pass_rate=0.5,
                avg_keyword_hit_rate=0.7,
                expected_file_hit_rate=0.8,
                false_positive_count=0,
                false_negative_count=1,
                avg_top_score=0.6,
                avg_source_count=2.0,
                failure_reason_counts={"rejected_when_should_answer": 1},
            )
        ],
        recommendation=None,
    )

    json_path, csv_path = write_evaluation_grid_report(report, tmp_path)

    assert json_path.name == GRID_JSON_REPORT_NAME
    assert csv_path.name == GRID_CSV_REPORT_NAME
    assert json_path.exists()
    assert csv_path.exists()
    csv_text = csv_path.read_text(encoding="utf-8-sig")
    assert "top_k" in csv_text
    assert "false_negative_count" in csv_text
    assert "score_margin" in csv_text
