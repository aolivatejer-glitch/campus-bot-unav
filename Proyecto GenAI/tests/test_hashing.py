from rag_chatbot.ingestion.hashing import calculate_file_sha256


def test_sha256_is_stable(tmp_path) -> None:
    path = tmp_path / "document.txt"
    path.write_text("contenido estable", encoding="utf-8")

    first_hash = calculate_file_sha256(path)
    second_hash = calculate_file_sha256(path)

    assert first_hash == second_hash
    assert len(first_hash) == 64
