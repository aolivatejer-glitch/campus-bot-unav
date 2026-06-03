from rag_chatbot.retrieval.formatting import make_source_label
from rag_chatbot.schemas import RagSource, RetrievalResult


def build_source_label(file_name: str, page_number: int | None) -> str:
    return make_source_label(file_name, page_number)


def deduplicate_sources(
    results: list[RetrievalResult],
    *,
    max_sources: int,
) -> list[RagSource]:
    sources: list[RagSource] = []
    seen: set[tuple[str, int | None]] = set()

    for result in results:
        key = (result.file_name, result.page_number)
        if key in seen:
            continue

        seen.add(key)
        sources.append(
            RagSource(
                file_name=result.file_name,
                page_number=result.page_number,
                chunk_id=result.chunk_id,
                source_label=build_source_label(result.file_name, result.page_number),
                snippet=result.snippet,
            )
        )

        if len(sources) >= max_sources:
            break

    return sources
