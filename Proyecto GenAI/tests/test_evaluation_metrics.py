from rag_chatbot.evaluation.metrics import (
    calculate_failure_reasons,
    calculate_evaluation_metrics,
    calculate_expected_file_hits,
    calculate_keyword_hits,
    calculate_scores,
)
from rag_chatbot.schemas import (
    ContextSufficiencyResult,
    EvaluationQuestion,
    RagAnswer,
    RagSource,
    RetrievedChunk,
)


def _answer(has_context: bool = True) -> RagAnswer:
    return RagAnswer(
        question="pregunta",
        answer="Respuesta sobre convivencia y normas.",
        has_sufficient_context=has_context,
        warning=None if has_context else "Sin contexto",
        sources=[
            RagSource(
                file_name="convivencia.pdf",
                page_number=1,
                chunk_id="chunk_1",
                source_label="convivencia.pdf, pagina 1",
                snippet="convivencia",
            )
        ]
        if has_context
        else [],
        retrieved_chunks=[
            RetrievedChunk(
                chunk_id="chunk_1",
                score=0.8,
                file_name="convivencia.pdf",
                page_number=1,
                text="Texto sobre convivencia y normas.",
                snippet="convivencia y normas",
            )
        ]
        if has_context
        else [],
        context=ContextSufficiencyResult(
            has_sufficient_context=has_context,
            warning=None,
            reason="sufficient" if has_context else "no_results",
            best_score=0.8 if has_context else None,
            total_context_chars=100 if has_context else 0,
            result_count=1 if has_context else 0,
        ),
    )


def test_keyword_hit_rate_uses_answer_and_chunks() -> None:
    found, missing, hit_rate = calculate_keyword_hits(
        ["convivencia", "normas", "ausente"],
        answer="Respuesta sobre convivencia.",
        retrieved_chunks=_answer().retrieved_chunks,
    )

    assert found == ["convivencia", "normas"]
    assert missing == ["ausente"]
    assert hit_rate == 2 / 3


def test_expected_file_hit_detects_files() -> None:
    found, missing, hit = calculate_expected_file_hits(
        ["convivencia.pdf", "otro.pdf"],
        ["Convivencia.pdf"],
    )

    assert found == ["convivencia.pdf"]
    assert missing == ["otro.pdf"]
    assert hit is True


def test_scores_are_calculated() -> None:
    top_score, avg_score = calculate_scores(_answer().retrieved_chunks)

    assert top_score == 0.8
    assert avg_score == 0.8


def test_passed_for_in_domain_question() -> None:
    question = EvaluationQuestion(
        question_id="q1",
        question="Pregunta",
        expected_keywords=["convivencia"],
        expected_files=["convivencia.pdf"],
        should_have_answer=True,
    )

    metrics = calculate_evaluation_metrics(
        question,
        _answer(),
        keyword_hit_rate=1.0,
        expected_file_hit=True,
        top_score=0.8,
        avg_score=0.8,
    )

    assert metrics.passed is True


def test_passed_for_out_of_domain_rejection() -> None:
    question = EvaluationQuestion(
        question_id="q2",
        question="Fuera de dominio",
        should_have_answer=False,
    )

    metrics = calculate_evaluation_metrics(
        question,
        _answer(has_context=False),
        keyword_hit_rate=1.0,
        expected_file_hit=True,
        top_score=None,
        avg_score=None,
    )

    assert metrics.passed is True


def test_failure_reasons_detect_false_positive() -> None:
    question = EvaluationQuestion(
        question_id="q3",
        question="Fuera de dominio",
        should_have_answer=False,
    )

    reasons = calculate_failure_reasons(
        question,
        _answer(has_context=True),
        keyword_hit_rate=1.0,
        expected_file_hit=True,
        top_score=0.8,
    )

    assert "answered_when_should_reject" in reasons
    assert "out_of_domain_not_rejected" in reasons


def test_failure_reasons_detect_false_negative() -> None:
    question = EvaluationQuestion(
        question_id="q4",
        question="Dominio",
        expected_keywords=["convivencia"],
        expected_files=["convivencia.pdf"],
        should_have_answer=True,
    )

    reasons = calculate_failure_reasons(
        question,
        _answer(has_context=False),
        keyword_hit_rate=0.0,
        expected_file_hit=False,
        top_score=None,
    )

    assert "rejected_when_should_answer" in reasons
    assert "expected_file_not_found" in reasons
    assert "low_keyword_hit_rate" in reasons
    assert "no_sources" in reasons
