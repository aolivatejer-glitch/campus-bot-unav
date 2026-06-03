import unicodedata

from rag_chatbot.schemas import (
    EvaluationMetrics,
    EvaluationQuestion,
    RagAnswer,
    RetrievedChunk,
)


LOW_KEYWORD_HIT_RATE_THRESHOLD = 0.5
LOW_TOP_SCORE_THRESHOLD = 0.3
MIN_ANSWER_CHARS = 80


def calculate_keyword_hits(
    expected_keywords: list[str],
    *,
    answer: str,
    retrieved_chunks: list[RetrievedChunk],
) -> tuple[list[str], list[str], float]:
    if not expected_keywords:
        return [], [], 1.0

    haystack = _normalize_text(
        " ".join(
            [answer]
            + [chunk.snippet for chunk in retrieved_chunks]
            + [chunk.text for chunk in retrieved_chunks]
        )
    )
    found: list[str] = []
    missing: list[str] = []

    for keyword in expected_keywords:
        if _normalize_text(keyword) in haystack:
            found.append(keyword)
        else:
            missing.append(keyword)

    return found, missing, len(found) / len(expected_keywords)


def calculate_expected_file_hits(
    expected_files: list[str],
    retrieved_files: list[str],
) -> tuple[list[str], list[str], bool]:
    if not expected_files:
        return [], [], True

    normalized_retrieved = {_normalize_file_name(file_name) for file_name in retrieved_files}
    found = [
        file_name
        for file_name in expected_files
        if _normalize_file_name(file_name) in normalized_retrieved
    ]
    missing = [file_name for file_name in expected_files if file_name not in found]

    return found, missing, bool(found)


def calculate_scores(chunks: list[RetrievedChunk]) -> tuple[float | None, float | None]:
    if not chunks:
        return None, None

    scores = [chunk.score for chunk in chunks]
    return max(scores), sum(scores) / len(scores)


def calculate_evaluation_metrics(
    question: EvaluationQuestion,
    answer: RagAnswer,
    *,
    keyword_hit_rate: float,
    expected_file_hit: bool,
    top_score: float | None,
    avg_score: float | None,
) -> EvaluationMetrics:
    has_answer = answer.has_sufficient_context
    expected_answer_behavior = has_answer == question.should_have_answer
    source_count = len(answer.sources)
    retrieved_chunk_count = len(answer.retrieved_chunks)

    if question.should_have_answer:
        passed = (
            expected_answer_behavior
            and keyword_hit_rate > 0.0
            and expected_file_hit
            and source_count > 0
        )
    else:
        passed = expected_answer_behavior

    return EvaluationMetrics(
        has_answer=has_answer,
        expected_answer_behavior=expected_answer_behavior,
        keyword_hit_rate=keyword_hit_rate,
        expected_file_hit=expected_file_hit,
        source_count=source_count,
        retrieved_chunk_count=retrieved_chunk_count,
        top_score=top_score,
        avg_score=avg_score,
        passed=passed,
    )


def calculate_failure_reasons(
    question: EvaluationQuestion,
    answer: RagAnswer,
    *,
    keyword_hit_rate: float,
    expected_file_hit: bool,
    top_score: float | None,
) -> list[str]:
    reasons: list[str] = []
    has_answer = answer.has_sufficient_context

    if not question.should_have_answer and has_answer:
        reasons.append("answered_when_should_reject")
        reasons.append("out_of_domain_not_rejected")

    if question.should_have_answer and not has_answer:
        reasons.append("rejected_when_should_answer")

    if question.should_have_answer and question.expected_files and not expected_file_hit:
        reasons.append("expected_file_not_found")

    if (
        question.should_have_answer
        and question.expected_keywords
        and keyword_hit_rate < LOW_KEYWORD_HIT_RATE_THRESHOLD
    ):
        reasons.append("low_keyword_hit_rate")

    if question.should_have_answer and not answer.sources:
        reasons.append("no_sources")

    if top_score is not None and top_score < LOW_TOP_SCORE_THRESHOLD:
        reasons.append("low_top_score")

    if question.should_have_answer and has_answer and len(_single_line(answer.answer)) < MIN_ANSWER_CHARS:
        reasons.append("answer_too_short")

    return reasons


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.casefold())
    return "".join(character for character in normalized if not unicodedata.combining(character))


def _normalize_file_name(file_name: str) -> str:
    return file_name.strip().casefold()


def _single_line(text: str) -> str:
    return " ".join(text.split())
