from rag_chatbot.ingestion.discovery import discover_documents


def test_discovery_finds_supported_files(tmp_path) -> None:
    documents_dir = tmp_path / "docs"
    nested_dir = documents_dir / "nested"
    nested_dir.mkdir(parents=True)

    (documents_dir / "a.pdf").write_bytes(b"%PDF")
    (documents_dir / "b.txt").write_text("txt", encoding="utf-8")
    (nested_dir / "c.docx").write_bytes(b"docx")

    discovered = discover_documents(documents_dir)

    assert [path.name for path in discovered] == ["a.pdf", "b.txt", "c.docx"]


def test_discovery_ignores_unsupported_and_temporary_files(tmp_path) -> None:
    documents_dir = tmp_path / "docs"
    documents_dir.mkdir()

    (documents_dir / "a.md").write_text("markdown", encoding="utf-8")
    (documents_dir / "~$temp.docx").write_text("temp", encoding="utf-8")
    (documents_dir / "notes.tmp").write_text("temp", encoding="utf-8")
    (documents_dir / "valid.txt").write_text("txt", encoding="utf-8")

    discovered = discover_documents(documents_dir)

    assert [path.name for path in discovered] == ["valid.txt"]
