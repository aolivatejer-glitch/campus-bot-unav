from hashlib import sha256
from pathlib import Path


def calculate_file_sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(block_size), b""):
            digest.update(block)

    return digest.hexdigest()
