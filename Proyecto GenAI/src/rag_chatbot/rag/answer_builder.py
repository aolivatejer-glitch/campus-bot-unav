from rag_chatbot.schemas import ContextSufficiencyResult, RagSource, RetrievalResult


def build_extractive_answer(
    *,
    results: list[RetrievalResult],
    sources: list[RagSource],
    context: ContextSufficiencyResult,
    max_context_chars: int,
) -> str:
    if not context.has_sufficient_context:
        answer = context.warning or (
            "No encontré información suficiente en los documentos indexados para "
            "responder con seguridad a esta pregunta."
        )
        if results:
            answer += (
                "\n\nSe recuperaron algunos fragmentos, pero no superan los criterios "
                "minimos de confianza o contexto para formular una respuesta."
            )
        return answer

    lines = [
        "Con base en los documentos recuperados, se encontro lo siguiente:",
        "",
    ]
    used_chars = 0

    for result in results:
        if used_chars >= max_context_chars:
            break

        snippet = result.snippet.strip()
        if not snippet:
            continue

        remaining = max_context_chars - used_chars
        if len(snippet) > remaining:
            snippet = snippet[:remaining].rstrip() + "..."

        lines.append(f"- {result.source_label}: {snippet}")
        used_chars += len(snippet)

    if sources:
        lines.extend(["", "Fuentes:"])
        for index, source in enumerate(sources, start=1):
            lines.append(f"{index}. {source.source_label} ({source.chunk_id})")

    return "\n".join(lines)
