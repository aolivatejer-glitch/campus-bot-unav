import re


INVISIBLE_CHARACTERS = {
    "\ufeff": "",
    "\u200b": "",
    "\u200c": "",
    "\u200d": "",
    "\u2060": "",
    "\x00": "",
}


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


def clean_text(text: str | None) -> str:
    if text is None:
        return ""

    cleaned = normalize_newlines(text)
    cleaned = remove_problematic_characters(cleaned)
    cleaned = normalize_spaces(cleaned)

    if not any(character.isalnum() for character in cleaned):
        return ""

    return cleaned
