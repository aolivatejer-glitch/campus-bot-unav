from rag_chatbot.rag.domain_guardrails import classify_question_domain


def test_classifies_rey_as_out_of_domain() -> None:
    result = classify_question_domain("¿Qué documentos mencionan al Rey?")

    assert result.is_in_domain is False
    assert result.reason == "matched_blocked_terms"
    assert "rey" in result.blocked_terms


def test_classifies_champions_league_as_out_of_domain() -> None:
    result = classify_question_domain("¿Qué equipo ganó la última Champions League?")

    assert result.is_in_domain is False
    assert "champions league" in result.blocked_terms


def test_classifies_compliance_as_in_domain() -> None:
    result = classify_question_domain("¿Qué documentos hablan sobre compliance?")

    assert result.is_in_domain is True
    assert "compliance" in result.matched_terms


def test_classifies_acoso_as_in_domain() -> None:
    result = classify_question_domain("¿Qué dice el protocolo sobre acoso entre estudiantes?")

    assert result.is_in_domain is True
    assert "acoso" in result.matched_terms
    assert "estudiantes" in result.matched_terms


def test_classifies_politica_ia_as_in_domain() -> None:
    result = classify_question_domain("¿Qué establece la política de IA?")

    assert result.is_in_domain is True
    assert "ia" in result.matched_terms


def test_classifies_ambiguous_question_as_unknown() -> None:
    result = classify_question_domain("¿Dónde encuentro el formulario?")

    assert result.is_in_domain is None
    assert result.reason == "unknown_no_domain_terms"
    assert result.matched_terms == []
    assert result.blocked_terms == []
