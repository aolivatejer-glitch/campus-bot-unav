import re
import unicodedata

from rag_chatbot.schemas import DomainGuardrailResult


ALLOWED_DOMAIN_TERMS = (
    "universidad de navarra",
    "normativa",
    "normativas",
    "politica",
    "politicas",
    "convivencia",
    "acoso",
    "estudiantes",
    "compliance",
    "ia",
    "inteligencia artificial",
    "defensoria universitaria",
    "creditos",
    "grado",
    "master",
    "practicas academicas",
    "docencia",
    "colaboradores docentes",
    "ensenanzas propias",
    "examenes de certificacion",
    "rectorado",
    "alumnos",
)

BLOCKED_OR_OUT_OF_DOMAIN_TERMS = (
    "rey",
    "23f",
    "champions league",
    "futbol",
    "recetas medicas",
    "clima",
    "prediccion meteorologica",
    "empresa privada",
    "restaurantes",
    "hoteles",
    "turismo",
)


def classify_question_domain(question: str) -> DomainGuardrailResult:
    normalized_question = _normalize_text(question)
    blocked_terms = _matched_terms(normalized_question, BLOCKED_OR_OUT_OF_DOMAIN_TERMS)
    if blocked_terms:
        return DomainGuardrailResult(
            is_in_domain=False,
            reason="matched_blocked_terms",
            matched_terms=[],
            blocked_terms=blocked_terms,
        )

    matched_terms = _matched_terms(normalized_question, ALLOWED_DOMAIN_TERMS)
    if matched_terms:
        return DomainGuardrailResult(
            is_in_domain=True,
            reason="matched_allowed_terms",
            matched_terms=matched_terms,
            blocked_terms=[],
        )

    return DomainGuardrailResult(
        is_in_domain=None,
        reason="unknown_no_domain_terms",
        matched_terms=[],
        blocked_terms=[],
    )


def _matched_terms(normalized_question: str, terms: tuple[str, ...]) -> list[str]:
    matches: list[str] = []
    for term in terms:
        normalized_term = _normalize_text(term)
        if re.search(rf"\b{re.escape(normalized_term)}\b", normalized_question):
            matches.append(term)
    return matches


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.casefold())
    without_accents = "".join(
        character for character in normalized if not unicodedata.combining(character)
    )
    return re.sub(r"\s+", " ", without_accents).strip()
