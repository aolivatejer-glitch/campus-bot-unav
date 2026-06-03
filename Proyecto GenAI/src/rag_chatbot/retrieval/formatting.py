import re

from rag_chatbot.chunking.cleaning import normalize_dot_leaders


def cosine_distance_to_score(distance: float | None) -> float:
    if distance is None:
        return 0.0

    return max(0.0, min(1.0, 1.0 - distance))


def make_snippet(text: str, max_chars: int = 400) -> str:
    normalized = normalize_dot_leaders(text)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if len(normalized) <= max_chars:
        return normalized

    cutoff = max_chars
    last_space = normalized.rfind(" ", 0, max_chars)
    if last_space >= max_chars * 0.65:
        cutoff = last_space

    return normalized[:cutoff].rstrip() + "..."


def make_source_label(file_name: str, page_number: int | None) -> str:
    if page_number is None or page_number < 0:
        return file_name

    return f"{file_name}, pagina {page_number}"
