from rag_chatbot.schemas import (
    ContextSufficiencyResult,
    DomainGuardrailResult,
    RetrievalResult,
)


INSUFFICIENT_CONTEXT_WARNING = (
    "No encontré información suficiente en los documentos indexados para responder "
    "con seguridad a esta pregunta."
)

OUT_OF_DOMAIN_WARNING = (
    "No puedo responder esa pregunta porque está fuera del alcance de los documentos "
    "indexados. Este chatbot está limitado a normativas y políticas de la Universidad "
    "de Navarra cargadas en el sistema."
)


def evaluate_context_sufficiency(
    results: list[RetrievalResult],
    *,
    min_score: float,
    min_context_chars: int,
    domain_result: DomainGuardrailResult | None = None,
    enable_domain_guardrails: bool = True,
) -> ContextSufficiencyResult:
    if (
        enable_domain_guardrails
        and domain_result is not None
        and domain_result.is_in_domain is False
    ):
        return ContextSufficiencyResult(
            has_sufficient_context=False,
            warning=OUT_OF_DOMAIN_WARNING,
            reason="out_of_domain",
            rejection_reason="out_of_domain",
            best_score=None,
            total_context_chars=0,
            result_count=0,
        )

    if not results:
        return ContextSufficiencyResult(
            has_sufficient_context=False,
            warning=INSUFFICIENT_CONTEXT_WARNING,
            reason="no_results",
            rejection_reason="insufficient_context",
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
            rejection_reason="low_score",
            best_score=best_score,
            total_context_chars=total_context_chars,
            result_count=len(results),
        )

    if total_context_chars < min_context_chars:
        return ContextSufficiencyResult(
            has_sufficient_context=False,
            warning=INSUFFICIENT_CONTEXT_WARNING,
            reason="too_little_context",
            rejection_reason="insufficient_context",
            best_score=best_score,
            total_context_chars=total_context_chars,
            result_count=len(results),
        )

    return ContextSufficiencyResult(
        has_sufficient_context=True,
        warning=None,
        reason="sufficient",
        rejection_reason=None,
        best_score=best_score,
        total_context_chars=total_context_chars,
        result_count=len(results),
    )
