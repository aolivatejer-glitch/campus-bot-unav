import json
from pathlib import Path

from pydantic import ValidationError

from rag_chatbot.schemas import EvaluationQuestion


class EvaluationDatasetError(ValueError):
    pass


def load_evaluation_questions(
    dataset_path: Path,
    *,
    limit: int | None = None,
    skip_invalid: bool = True,
) -> list[EvaluationQuestion]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found: {dataset_path}")

    questions: list[EvaluationQuestion] = []

    with dataset_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            try:
                payload = json.loads(stripped)
                questions.append(EvaluationQuestion.model_validate(payload))
            except (json.JSONDecodeError, ValidationError) as exc:
                if skip_invalid:
                    continue
                raise EvaluationDatasetError(
                    f"Invalid evaluation question at line {line_number}: {exc}"
                ) from exc

            if limit is not None and len(questions) >= limit:
                break

    return questions
