from rag_chatbot.schemas import ContextSufficiencyResult, RetrievalResult


INSUFFICIENT_CONTEXT_WARNING = (
    "No encontré información suficiente en los documentos indexados para responder "
    "con seguridad a esta pregunta."
)


def evaluate_context_sufficiency(
    results: list[RetrievalResult],
    *,
    min_score: float,
    min_context_chars: int,
) -> ContextSufficiencyResult:
    if not results:
        return ContextSufficiencyResult(
            has_sufficient_context=False,
            warning=INSUFFICIENT_CONTEXT_WARNING,
            reason="no_results",
            best_score=None,
            total_context_chars=0,
            result_count=0,
        )

    best_score = max(result.score for result in results)
    total_context_chars = sum(len(result.text.strip()) for result in results)

    if best_score < min_score:
        return ContextSufficiencyResult(
            has_sufficient_context=False,
            warning=INSUFFICIENT_CONTEXT_WARNING,
            reason="low_score",
            best_score=best_score,
            total_context_chars=total_context_chars,
            result_count=len(results),
        )

    if total_context_chars < min_context_chars:
        return ContextSufficiencyResult(
            has_sufficient_context=False,
            warning=INSUFFICIENT_CONTEXT_WARNING,
            reason="too_little_context",
            best_score=best_score,
            total_context_chars=total_context_chars,
            result_count=len(results),
        )

    return ContextSufficiencyResult(
        has_sufficient_context=True,
        warning=None,
        reason="sufficient",
        best_score=best_score,
        total_context_chars=total_context_chars,
        result_count=len(results),
    )
