from pathlib import Path

from rag_chatbot.config import AppSettings
from rag_chatbot.diagnostics import (
    ChromaDiagnosticInfo,
    DiagnosticStatus,
    format_doctor_report,
    run_doctor,
    validate_privacy,
)


def _settings(tmp_path: Path, *, create_dirs: bool = False, **overrides) -> AppSettings:
    values = {
        "documents_dir": tmp_path / "Documentos",
        "data_dir": tmp_path / "data",
        "processed_dir": tmp_path / "data" / "processed",
        "eval_dir": tmp_path / "data" / "eval",
        "eval_dataset_path": tmp_path / "data" / "eval" / "evaluation_questions.jsonl",
        "eval_reports_dir": tmp_path / "data" / "eval" / "reports",
        "chunks_dir": tmp_path / "data" / "chunks",
        "chunks_file": tmp_path / "data" / "chunks" / "chunks.jsonl",
        "storage_dir": tmp_path / "storage",
        "chroma_dir": tmp_path / "storage" / "chroma",
        "manifest_db_path": tmp_path / "storage" / "manifest.sqlite",
        "log_dir": tmp_path / "logs",
        "_env_file": None,
    }
    values.update(overrides)
    settings = AppSettings(**values)
    if create_dirs:
        settings.ensure_directories()
    return settings


def test_validate_privacy_accepts_local_first_defaults(tmp_path) -> None:
    source_root = tmp_path / "src"
    source_root.mkdir()
    settings = _settings(tmp_path)

    checks = validate_privacy(settings, environ={}, source_root=source_root)

    assert all(check.status == DiagnosticStatus.OK for check in checks)


def test_validate_privacy_detects_external_llm_settings(tmp_path) -> None:
    source_root = tmp_path / "src"
    source_root.mkdir()
    settings = _settings(
        tmp_path,
        allow_external_llm=True,
        llm_provider="external",
    )

    checks = validate_privacy(settings, environ={}, source_root=source_root)

    assert [check.status for check in checks].count(DiagnosticStatus.ERROR) == 2


def test_validate_privacy_warns_for_external_api_env_vars(tmp_path) -> None:
    source_root = tmp_path / "src"
    source_root.mkdir()
    settings = _settings(tmp_path)

    checks = validate_privacy(
        settings,
        environ={"OPENAI_API_KEY": "present"},
        source_root=source_root,
    )

    assert any(
        check.status == DiagnosticStatus.WARNING
        and check.label == "Privacidad variables externas"
        for check in checks
    )


def test_validate_privacy_detects_external_imports(tmp_path) -> None:
    source_root = tmp_path / "src"
    source_root.mkdir()
    (source_root / "bad.py").write_text("import openai\n", encoding="utf-8")
    settings = _settings(tmp_path)

    checks = validate_privacy(settings, environ={}, source_root=source_root)

    assert any(
        check.status == DiagnosticStatus.ERROR
        and check.label == "Privacidad imports externos"
        for check in checks
    )


def test_doctor_detects_missing_directories(tmp_path) -> None:
    settings = _settings(tmp_path, create_dirs=False)

    report = run_doctor(
        settings,
        chroma_inspector=lambda _: ChromaDiagnosticInfo(available=False),
    )

    assert report.has_errors
    assert any(
        check.status == DiagnosticStatus.ERROR and check.label == "Carpeta"
        for check in report.checks
    )


def test_doctor_detects_missing_chunks(tmp_path) -> None:
    settings = _settings(tmp_path, create_dirs=True)
    (settings.documents_dir / "sample.txt").write_text("texto", encoding="utf-8")

    report = run_doctor(
        settings,
        chroma_inspector=lambda _: ChromaDiagnosticInfo(available=False),
    )

    assert any(
        check.status == DiagnosticStatus.WARNING
        and check.label == "Chunks"
        and "build-chunks" in check.message
        for check in report.checks
    )


def test_doctor_detects_empty_index_with_mock(tmp_path) -> None:
    settings = _settings(tmp_path, create_dirs=True)
    settings.chunks_file.write_text('{"chunk_id":"chunk_1"}\n', encoding="utf-8")

    report = run_doctor(
        settings,
        chroma_inspector=lambda _: ChromaDiagnosticInfo(
            available=True,
            collection_exists=True,
            vector_count=0,
        ),
    )

    assert any(
        check.status == DiagnosticStatus.WARNING
        and check.label == "Indice Chroma"
        and "vacia" in check.message
        for check in report.checks
    )


def test_format_doctor_report_uses_status_prefixes(tmp_path) -> None:
    settings = _settings(tmp_path, create_dirs=True)
    report = run_doctor(
        settings,
        chroma_inspector=lambda _: ChromaDiagnosticInfo(available=False),
    )

    lines = format_doctor_report(report)

    assert lines
    assert lines[0].startswith("[OK]")


def test_phase_10c_does_not_add_reranking_module() -> None:
    project_root = Path(__file__).resolve().parents[1]

    assert not (project_root / "src" / "rag_chatbot" / "reranking").exists()
    assert not (project_root / "src" / "rag_chatbot" / "rerank.py").exists()
