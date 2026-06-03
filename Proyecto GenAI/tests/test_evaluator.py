from rag_chatbot.config import AppSettings
from rag_chatbot.evaluation.evaluator import run_evaluation
from rag_chatbot.schemas import (
    ContextSufficiencyResult,
    RagAnswer,
    RagSource,
    RetrievedChunk,
)


class FakeEvaluationPipeline:
    def __init__(self):
        self.calls = []

    def ask(self, question, **kwargs):
        self.calls.append((question, kwargs))
        if "fuera" in question.casefold():
            return RagAnswer(
                question=question,
                answer="No encontre informacion suficiente.",
                has_sufficient_context=False,
                warning="No encontre informacion suficiente.",
                sources=[],
                retrieved_chunks=[],
                context=ContextSufficiencyResult(
                    has_sufficient_context=False,
                    warning="No encontre informacion suficiente.",
                    reason="no_results",
                    best_score=None,
                    total_context_chars=0,
                    result_count=0,
                ),
            )

        return RagAnswer(
            question=question,
            answer="Respuesta sobre convivencia.",
            has_sufficient_context=True,
            warning=None,
            sources=[
                RagSource(
                    file_name="convivencia.pdf",
                    page_number=1,
                    chunk_id="chunk_1",
                    source_label="convivencia.pdf, pagina 1",
                    snippet="convivencia",
                )
            ],
            retrieved_chunks=[
                RetrievedChunk(
                    chunk_id="chunk_1",
                    score=0.8,
                    file_name="convivencia.pdf",
                    page_number=1,
                    text="Texto sobre convivencia.",
                    snippet="convivencia",
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
        _env_file=None,
    )


def test_run_evaluation_writes_reports_and_scores_questions(tmp_path) -> None:
    settings = _settings(tmp_path)
    settings.eval_dataset_path.parent.mkdir(parents=True, exist_ok=True)
    settings.eval_dataset_path.write_text(
        '{"question_id":"q1","question":"convivencia","expected_keywords":["convivencia"],'
        '"expected_files":["convivencia.pdf"],"should_have_answer":true}\n'
        '{"question_id":"q2","question":"fuera de dominio","should_have_answer":false}\n',
        encoding="utf-8",
    )

    pipeline = FakeEvaluationPipeline()
    report, json_path, csv_path = run_evaluation(
        settings,
        pipeline=pipeline,
        top_k=3,
        min_score=0.4,
    )

    assert report.summary.total_questions == 2
    assert report.summary.passed == 2
    assert report.summary.out_of_domain_correct_rejections == 1
    assert report.results[0].failure_reasons == []
    assert pipeline.calls[0][1] == {"top_k": 3, "min_score": 0.4}
    assert json_path.exists()
    assert csv_path.exists()
