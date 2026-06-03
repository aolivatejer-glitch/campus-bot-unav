import pytest
from pathlib import Path

from rag_chatbot.evaluation.dataset import EvaluationDatasetError, load_evaluation_questions


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_load_evaluation_questions_reads_jsonl_and_skips_empty_lines(tmp_path) -> None:
    dataset = tmp_path / "questions.jsonl"
    dataset.write_text(
        "\n"
        '{"question_id":"q1","question":"Pregunta","expected_keywords":["x"],'
        '"expected_files":["a.pdf"],"should_have_answer":true}\n',
        encoding="utf-8",
    )

    questions = load_evaluation_questions(dataset)

    assert len(questions) == 1
    assert questions[0].question_id == "q1"


def test_load_evaluation_questions_skips_invalid_lines_by_default(tmp_path) -> None:
    dataset = tmp_path / "questions.jsonl"
    dataset.write_text(
        "not-json\n"
        '{"question_id":"q1","question":"Pregunta","should_have_answer":false}\n',
        encoding="utf-8",
    )

    questions = load_evaluation_questions(dataset)

    assert len(questions) == 1
    assert questions[0].should_have_answer is False


def test_load_evaluation_questions_can_raise_on_invalid_line(tmp_path) -> None:
    dataset = tmp_path / "questions.jsonl"
    dataset.write_text("not-json\n", encoding="utf-8")

    with pytest.raises(EvaluationDatasetError):
        load_evaluation_questions(dataset, skip_invalid=False)


def test_load_evaluation_questions_applies_limit(tmp_path) -> None:
    dataset = tmp_path / "questions.jsonl"
    dataset.write_text(
        '{"question_id":"q1","question":"Uno","should_have_answer":true}\n'
        '{"question_id":"q2","question":"Dos","should_have_answer":true}\n',
        encoding="utf-8",
    )

    questions = load_evaluation_questions(dataset, limit=1)

    assert [question.question_id for question in questions] == ["q1"]


def test_project_evaluation_dataset_is_valid_and_expanded() -> None:
    dataset = PROJECT_ROOT / "data" / "eval" / "evaluation_questions.jsonl"

    questions = load_evaluation_questions(dataset, skip_invalid=False)
    question_ids = {question.question_id for question in questions}

    assert len(questions) >= 36
    assert "compliance_penal_001" in question_ids
    assert "fuera_deporte_001" in question_ids
    assert "fuera_becas_001" in question_ids
    assert "fuera_restaurantes_001" in question_ids
    assert any(not question.should_have_answer for question in questions)
