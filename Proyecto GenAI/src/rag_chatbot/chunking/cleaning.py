import re


INVISIBLE_CHARACTERS = {
    "\ufeff": "",
    "\u200b": "",
    "\u200c": "",
    "\u200d": "",
    "\u2060": "",
    "\x00": "",
}

DOT_LEADER_PATTERN = re.compile(r"(?:\s*\.\s*){5,}")
TOC_HEADING_PATTERN = re.compile(r"^\s*(índice|indice|contenido|sumario)\s*$", re.IGNORECASE)
TOC_ENTRY_PATTERN = re.compile(
    r"^\s*(?:\d+(?:\.\d+)*\.?\s+)?\S.{2,180}?(?:\s*\.\s*){5,}\s*\d{1,4}\s*$"
)
PAGE_NUMBER_LINE_PATTERN = re.compile(
    r"^\s*(?:\d+(?:\.\d+)*\.?\s+)?[A-ZÁÉÍÓÚÑa-záéíóúñ][^.!?]{2,140}\s+\d{1,4}\s*$"
)
NUMBERED_TITLE_PATTERN = re.compile(r"^\s*\d+(?:\.\d+)*\.?\s+\S.{0,120}$")


def normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def remove_problematic_characters(text: str) -> str:
    for character, replacement in INVISIBLE_CHARACTERS.items():
        text = text.replace(character, replacement)
    return text


def normalize_spaces(text: str) -> str:
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_dot_leaders(text: str) -> str:
    return DOT_LEADER_PATTERN.sub(" ", text)


def remove_table_of_contents_lines(text: str) -> str:
    lines = normalize_newlines(text).split("\n")
    cleaned_lines: list[str] = []
    near_toc_heading = False
    toc_window = 0

    for line in lines:
        stripped = line.strip()
        if not stripped:
            cleaned_lines.append(line)
            continue

        if TOC_HEADING_PATTERN.match(stripped):
            near_toc_heading = True
            toc_window = 12
            continue

        is_toc_entry = bool(TOC_ENTRY_PATTERN.match(stripped))
        is_page_number_entry = (
            near_toc_heading
            and bool(PAGE_NUMBER_LINE_PATTERN.match(stripped))
            and not _looks_like_sentence(stripped)
        )

        if is_toc_entry or is_page_number_entry:
            toc_window = max(toc_window - 1, 0)
            near_toc_heading = toc_window > 0
            continue

        cleaned_lines.append(line)
        toc_window = max(toc_window - 1, 0)
        near_toc_heading = toc_window > 0

    return "\n".join(cleaned_lines)


def count_dot_leaders(text: str) -> int:
    return len(DOT_LEADER_PATTERN.findall(text))


def is_table_of_contents_candidate(text: str) -> bool:
    normalized = normalize_newlines(text)
    lines = [line.strip() for line in normalized.split("\n") if line.strip()]
    if not lines:
        return False

    dot_leader_count = count_dot_leaders(normalized)
    toc_heading_count = sum(1 for line in lines if TOC_HEADING_PATTERN.match(line))
    toc_entry_count = sum(1 for line in lines if TOC_ENTRY_PATTERN.match(line))
    page_number_entry_count = sum(
        1
        for line in lines
        if PAGE_NUMBER_LINE_PATTERN.match(line) and not _looks_like_sentence(line)
    )
    numbered_title_count = sum(1 for line in lines if NUMBERED_TITLE_PATTERN.match(line))
    alnum_count = sum(character.isalnum() for character in normalized)
    punctuation_count = sum(character in ".·•-" for character in normalized)
    punctuation_ratio = punctuation_count / max(len(normalized), 1)

    signals = 0
    if dot_leader_count >= 2:
        signals += 2
    elif dot_leader_count == 1:
        signals += 1
    if toc_heading_count:
        signals += 2
    if toc_entry_count >= 2:
        signals += 2
    elif toc_entry_count == 1:
        signals += 1
    if page_number_entry_count >= max(3, len(lines) // 2):
        signals += 2
    if numbered_title_count >= 4:
        signals += 1
    if punctuation_ratio > 0.25 and alnum_count < 500:
        signals += 1

    return signals >= 3


def text_quality_metadata(text: str) -> dict[str, object]:
    is_toc_candidate = is_table_of_contents_candidate(text)
    dot_leader_count = count_dot_leaders(text)
    return {
        "is_toc_candidate": is_toc_candidate,
        "dot_leader_count": dot_leader_count,
        "text_quality": "low" if is_toc_candidate else "normal",
    }


def clean_text(
    text: str | None,
    *,
    clean_dot_leaders: bool = True,
    remove_toc_lines: bool = True,
) -> str:
    if text is None:
        return ""

    cleaned = normalize_newlines(text)
    cleaned = remove_problematic_characters(cleaned)
    if remove_toc_lines:
        cleaned = remove_table_of_contents_lines(cleaned)
    if clean_dot_leaders:
        cleaned = normalize_dot_leaders(cleaned)
    cleaned = normalize_spaces(cleaned)

    if not any(character.isalnum() for character in cleaned):
        return ""

    return cleaned


def _looks_like_sentence(line: str) -> bool:
    words = line.split()
    return len(words) >= 12 or any(mark in line[:-1] for mark in ".;:")
