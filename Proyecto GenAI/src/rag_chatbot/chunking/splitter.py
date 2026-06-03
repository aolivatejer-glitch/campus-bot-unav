from hashlib import sha256

from rag_chatbot.chunking.cleaning import clean_text, text_quality_metadata
from rag_chatbot.schemas import Chunk, DocumentPage, ProcessedDocument


def split_text(
    text: str,
    *,
    chunk_size: int,
    chunk_overlap: int,
    min_chunk_size: int,
    clean_dot_leaders: bool = True,
    remove_toc_lines: bool = True,
) -> list[str]:
    cleaned = clean_text(
        text,
        clean_dot_leaders=clean_dot_leaders,
        remove_toc_lines=remove_toc_lines,
    )
    if not cleaned:
        return []

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    if len(cleaned) <= chunk_size:
        return [cleaned]

    chunks: list[str] = []
    start = 0
    text_length = len(cleaned)

    while start < text_length:
        max_end = min(start + chunk_size, text_length)
        end = _find_split_end(cleaned, start, max_end, min_chunk_size)
        chunk = cleaned[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        next_start = max(end - chunk_overlap, start + 1)
        start = next_start

    return _merge_small_tail(chunks, min_chunk_size)


def build_chunks_for_page(
    document: ProcessedDocument,
    page: DocumentPage,
    *,
    chunk_size: int,
    chunk_overlap: int,
    min_chunk_size: int,
    exclude_toc_chunks: bool = True,
    clean_dot_leaders: bool = True,
) -> list[Chunk]:
    text_chunks = split_text(
        page.text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        min_chunk_size=min_chunk_size,
        clean_dot_leaders=clean_dot_leaders,
        remove_toc_lines=exclude_toc_chunks,
    )
    source = document.metadata.get("source", document.file_path)
    chunks: list[Chunk] = []

    for index, text in enumerate(text_chunks):
        quality_metadata = text_quality_metadata(text)
        if exclude_toc_chunks and quality_metadata["is_toc_candidate"]:
            continue

        page_label = _page_label(page.page_number)
        chunk_id = make_chunk_id(
            document_id=document.document_id,
            page_number=page.page_number,
            chunk_index=index,
            text=text,
        )
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                document_id=document.document_id,
                file_name=document.file_name,
                file_path=document.file_path,
                file_type=document.file_type,
                page_number=page.page_number,
                chunk_index=index,
                text=text,
                char_count=len(text),
                metadata={
                    "source": source,
                    "page_number": page.page_number,
                    "page_label": page_label,
                    **quality_metadata,
                },
            )
        )

    return chunks


def make_chunk_id(
    *,
    document_id: str,
    page_number: int | None,
    chunk_index: int,
    text: str,
) -> str:
    text_hash = sha256(text.encode("utf-8")).hexdigest()[:8]
    return f"{document_id}_p{_page_label(page_number)}_c{chunk_index}_{text_hash}"


def _find_split_end(
    text: str,
    start: int,
    max_end: int,
    min_chunk_size: int,
) -> int:
    if max_end >= len(text):
        return len(text)

    min_end = min(start + min_chunk_size, max_end)

    for index in range(max_end, min_end, -1):
        if text[index - 1].isspace():
            return index

    return max_end


def _merge_small_tail(chunks: list[str], min_chunk_size: int) -> list[str]:
    if len(chunks) <= 1:
        return chunks

    if len(chunks[-1]) < min_chunk_size:
        chunks[-2] = f"{chunks[-2]} {chunks[-1]}".strip()
        chunks.pop()

    return chunks


def _page_label(page_number: int | None) -> str:
    return "none" if page_number is None else str(page_number)
