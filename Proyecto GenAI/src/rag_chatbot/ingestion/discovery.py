from pathlib import Path


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx"}
TEMPORARY_PREFIXES = ("~$", ".~", "._")
TEMPORARY_SUFFIXES = (".tmp", ".temp", ".bak")


def is_supported_document(path: Path) -> bool:
    if not path.is_file():
        return False

    name = path.name
    lower_name = name.lower()

    if name.startswith(TEMPORARY_PREFIXES):
        return False

    if lower_name.endswith(TEMPORARY_SUFFIXES):
        return False

    return path.suffix.lower() in SUPPORTED_EXTENSIONS


def discover_documents(documents_dir: Path, recursive: bool = True) -> list[Path]:
    if not documents_dir.exists() or not documents_dir.is_dir():
        return []

    iterator = documents_dir.rglob("*") if recursive else documents_dir.glob("*")
    return sorted(
        (path for path in iterator if is_supported_document(path)),
        key=lambda path: str(path).lower(),
    )
