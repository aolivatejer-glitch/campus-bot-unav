from rag_chatbot.schemas import RagSource, RetrievalResult


SYSTEM_INSTRUCTIONS = """Eres un asistente RAG documental.
Responde exclusivamente con la informacion del contexto proporcionado.
No inventes datos, normas, fechas, procedimientos ni conclusiones.
Si el contexto no contiene evidencia suficiente, responde:
"No encontré información suficiente en los documentos indexados para responder con seguridad a esta pregunta."
Incluye citas breves con archivo y pagina cuando esten disponibles.
No menciones documentos o fuentes que no aparezcan en el contexto.
"""


def build_grounded_prompt(
    *,
    question: str,
    results: list[RetrievalResult],
    sources: list[RagSource],
    max_context_chars: int,
) -> str:
    context_blocks: list[str] = []
    used_chars = 0

    for index, result in enumerate(results, start=1):
        if result.metadata.get("is_toc_candidate"):
            continue

        text = result.text.strip()
        if not text:
            continue

        remaining = max_context_chars - used_chars
        if remaining <= 0:
            break
        if len(text) > remaining:
            text = text[:remaining].rstrip()

        page = "sin pagina" if result.page_number is None else f"pagina {result.page_number}"
        context_blocks.append(
            "\n".join(
                [
                    f"[{index}] Fuente: {result.file_name}, {page}",
                    f"chunk_id: {result.chunk_id}",
                    "Texto:",
                    text,
                ]
            )
        )
        used_chars += len(text)

    source_lines = [
        f"- {source.source_label} | chunk_id={source.chunk_id}"
        for source in sources
    ]

    return "\n\n".join(
        [
            SYSTEM_INSTRUCTIONS.strip(),
            f"Pregunta:\n{question.strip()}",
            "Fuentes recuperadas:\n" + ("\n".join(source_lines) if source_lines else "Ninguna"),
            "Contexto recuperado:\n" + ("\n\n".join(context_blocks) if context_blocks else "Sin contexto util."),
            "Respuesta:",
        ]
    )
