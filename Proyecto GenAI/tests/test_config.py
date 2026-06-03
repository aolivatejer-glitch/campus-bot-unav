from pathlib import Path

from rag_chatbot.config import AppSettings


ENV_KEYS = [
    "DOCUMENTS_DIR",
    "DATA_DIR",
    "PROCESSED_DIR",
    "EVAL_DIR",
    "CHUNKS_DIR",
    "CHUNKS_FILE",
    "STORAGE_DIR",
    "CHROMA_DIR",
    "CHROMA_COLLECTION_NAME",
    "MANIFEST_DB_PATH",
    "LOG_DIR",
    "LOG_LEVEL",
    "EMBEDDING_MODEL",
    "EMBEDDING_DEVICE",
    "EMBEDDING_BATCH_SIZE",
    "CHUNK_SIZE",
    "CHUNK_OVERLAP",
    "MIN_CHUNK_SIZE",
    "EXCLUDE_TOC_CHUNKS",
    "CLEAN_DOT_LEADERS",
    "TOP_K",
    "MIN_RETRIEVAL_SCORE",
    "SNIPPET_MAX_CHARS",
    "MIN_CONTEXT_CHARS",
    "MAX_CONTEXT_CHARS",
    "MAX_SOURCES",
    "ENABLE_DOMAIN_GUARDRAILS",
    "DOMAIN_NAME",
    "EVAL_DATASET_PATH",
    "EVAL_REPORTS_DIR",
    "EVAL_ANSWER_PREVIEW_CHARS",
    "ALLOW_EXTERNAL_LLM",
    "LLM_PROVIDER",
    "GEMINI_API_KEY",
    "GEMINI_MODEL",
    "LLM_TEMPERATURE",
    "LLM_MAX_CONTEXT_CHARS",
    "LLM_MODE_DEFAULT",
    "API_HOST",
    "API_PORT",
    "API_RELOAD",
    "API_INCLUDE_DEBUG_ERRORS",
    "API_BASE_URL",
]


def test_default_settings_load(monkeypatch) -> None:
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)

    settings = AppSettings(_env_file=None)

    assert settings.project_name == "rag-chatbot"
    assert settings.documents_dir.name == "Documentos"
    assert settings.embedding_model == "intfloat/multilingual-e5-base"
    assert settings.embedding_batch_size == 16
    assert settings.chroma_collection_name == "documents"
    assert settings.allow_external_llm is False
    assert settings.llm_provider == "none"
    assert settings.gemini_api_key == ""
    assert settings.gemini_model == "gemini-2.5-flash"
    assert settings.llm_temperature == 0.2
    assert settings.llm_max_context_chars == 6000
    assert settings.llm_mode_default == "extractive"
    assert settings.chunk_size == 1200
    assert settings.chunk_overlap == 200
    assert settings.min_chunk_size == 200
    assert settings.exclude_toc_chunks is True
    assert settings.clean_dot_leaders is True
    assert settings.top_k == 3
    assert settings.min_retrieval_score == 0.84
    assert settings.snippet_max_chars == 400
    assert settings.min_context_chars == 500
    assert settings.max_context_chars == 6000
    assert settings.max_sources == 3
    assert settings.enable_domain_guardrails is True
    assert settings.domain_name == "normativas y políticas de la Universidad de Navarra"
    assert settings.eval_answer_preview_chars == 300
    assert settings.api_reload is False
    assert settings.api_include_debug_errors is False
    assert settings.api_base_url == "http://127.0.0.1:8000"


def test_public_dict_masks_gemini_api_key() -> None:
    settings = AppSettings(gemini_api_key="secret-key", _env_file=None)

    assert settings.public_dict()["gemini_api_key"] == "***"


def test_path_settings_are_path_instances() -> None:
    settings = AppSettings(_env_file=None)

    assert isinstance(settings.documents_dir, Path)
    assert isinstance(settings.data_dir, Path)
    assert isinstance(settings.processed_dir, Path)
    assert isinstance(settings.eval_dir, Path)
    assert isinstance(settings.eval_dataset_path, Path)
    assert isinstance(settings.eval_reports_dir, Path)
    assert isinstance(settings.chunks_dir, Path)
    assert isinstance(settings.chunks_file, Path)
    assert isinstance(settings.storage_dir, Path)
    assert isinstance(settings.chroma_dir, Path)
    assert isinstance(settings.manifest_db_path, Path)
    assert isinstance(settings.log_dir, Path)


def test_ensure_directories_creates_expected_paths(tmp_path) -> None:
    settings = AppSettings(
        documents_dir=tmp_path / "Documentos",
        data_dir=tmp_path / "data",
        processed_dir=tmp_path / "data" / "processed",
        eval_dir=tmp_path / "data" / "eval",
        eval_dataset_path=tmp_path / "data" / "eval" / "evaluation_questions.jsonl",
        eval_reports_dir=tmp_path / "data" / "eval" / "reports",
        chunks_dir=tmp_path / "data" / "chunks",
        chunks_file=tmp_path / "data" / "chunks" / "chunks.jsonl",
        storage_dir=tmp_path / "storage",
        chroma_dir=tmp_path / "storage" / "chroma",
        manifest_db_path=tmp_path / "storage" / "manifest.sqlite",
        log_dir=tmp_path / "logs",
        _env_file=None,
    )

    settings.ensure_directories()

    for directory in settings.required_directories:
        assert directory.exists()
        assert directory.is_dir()
