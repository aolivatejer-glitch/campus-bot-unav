from pathlib import Path

import pytest

from rag_chatbot.config import AppSettings
from rag_chatbot.evaluation.grid import (
    parse_float_values,
    parse_int_values,
    recommend_grid_config,
    run_evaluation_grid,
)
from rag_chatbot.schemas import (
    ContextSufficiencyResult,
    EvaluationGridResult,
    RagAnswer,
    RagSource,
    RetrievedChunk,
)


class FakeGridPipeline:
    def ask(self, question, *, top_k=None, min_score=None, **kwargs):
        if "fuera" in question.casefold():
            has_context = min_score is not None and min_score < 0.3
            return RagAnswer(
                question=question,
                answer="Respuesta indebida." if has_context else "Sin contexto suficiente.",
                has_sufficient_context=has_context,
                warning=None if has_context else "Sin contexto suficiente.",
                sources=[
                    RagSource(
                        file_name="otro.pdf",
                        page_number=1,
                        chunk_id="chunk_out",
                        source_label="otro.pdf, pagina 1",
                        snippet="otro",
                    )
                ]
                if has_context
                else [],
                retrieved_chunks=[],
                context=ContextSufficiencyResult(
                    has_sufficient_context=has_context,
                    warning=None,
                    reason="sufficient" if has_context else "below_threshold",
                    best_score=0.25 if has_context else None,
                    total_context_chars=100 if has_context else 0,
                    result_count=1 if has_context else 0,
                ),
            )

        return RagAnswer(
            question=question,
            answer="Respuesta sobre convivencia, normas y Universidad de Navarra.",
            has_sufficient_context=True,
            warning=None,
            sources=[
                RagSource(
                    file_name="convivencia.pdf",
                    page_number=1,
                    chunk_id="chunk_1",
                    source_label="convivencia.pdf, pagina 1",
                    snippet="convivencia normas",
                )
            ],
            retrieved_chunks=[
                RetrievedChunk(
                    chunk_id="chunk_1",
                    score=0.8,
                    file_name="convivencia.pdf",
                    page_number=1,
                    text="Texto sobre convivencia, normas y Universidad de Navarra.",
                    snippet="convivencia normas",
                )
            ],
            context=ContextSufficiencyResult(
                has_sufficient_context=True,
                warning=None,
                reason="sufficient",
                best_score=0.8,
                total_context_chars=100,
                result_count=1,
            ),
        )


def _settings(tmp_path: Path) -> AppSettings:
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
        _env_file=None,
    )


def test_parse_int_values() -> None:
    assert parse_int_values("3,5, 8") == [3, 5, 8]


def test_parse_float_values() -> None:
    assert parse_float_values("0.2,0.3, 0.5") == [0.2, 0.3, 0.5]


def test_parse_values_reject_invalid_input() -> None:
    with pytest.raises(ValueError):
        parse_int_values("0")
    with pytest.raises(ValueError):
        parse_float_values("1.5")


def test_recommend_grid_config_prioritizes_pass_rate_and_false_positives() -> None:
    rows = [
        EvaluationGridResult(
            top_k=8,
            min_score=0.2,
            total_questions=10,
            passed=8,
            failed=2,
            pass_rate=0.8,
            avg_keyword_hit_rate=0.8,
            expected_file_hit_rate=0.9,
            false_positive_count=2,
            false_negative_count=0,
            avg_top_score=0.7,
            avg_source_count=3.0,
        ),
        EvaluationGridResult(
            top_k=5,
            min_score=0.3,
            total_questions=10,
            passed=8,
            failed=2,
            pass_rate=0.8,
            avg_keyword_hit_rate=0.8,
            expected_file_hit_rate=0.9,
            false_positive_count=0,
            false_negative_count=2,
            avg_top_score=0.7,
            avg_source_count=2.0,
        ),
    ]

    recommendation = recommend_grid_config(rows)

    assert recommendation is not None
    assert recommendation.top_k == 5
    assert recommendation.min_score == 0.3


def test_run_evaluation_grid_writes_reports(tmp_path) -> None:
    settings = _settings(tmp_path)
    settings.eval_dataset_path.parent.mkdir(parents=True, exist_ok=True)
    settings.eval_dataset_path.write_text(
        '{"question_id":"q1","question":"convivencia","expected_keywords":["convivencia"],'
        '"expected_files":["convivencia.pdf"],"should_have_answer":true}\n'
        '{"question_id":"q2","question":"fuera de dominio","should_have_answer":false}\n',
        encoding="utf-8",
    )

    report, json_path, csv_path = run_evaluation_grid(
        settings,
        top_k_values=[3],
        min_score_values=[0.2, 0.3],
        pipeline=FakeGridPipeline(),
    )

    assert len(report.results) == 2
    assert report.recommendation is not None
    assert json_path.exists()
    assert csv_path.exists()
    assert (settings.eval_reports_dir / "calibration_recommendation.json").exists()
