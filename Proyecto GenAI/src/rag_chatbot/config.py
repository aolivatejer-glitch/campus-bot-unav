from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class AppSettings(BaseSettings):
    """Application settings loaded from defaults, environment, and optional .env."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    project_name: str = "rag-chatbot"
    version: str = "0.1.0"

    documents_dir: Path = Path("Documentos")
    data_dir: Path = Path("data")
    processed_dir: Path = Path("data/processed")
    eval_dir: Path = Path("data/eval")
    chunks_dir: Path = Path("data/chunks")
    chunks_file: Path = Path("data/chunks/chunks.jsonl")
    storage_dir: Path = Path("storage")
    chroma_dir: Path = Path("storage/chroma")
    chroma_collection_name: str = "documents"
    manifest_db_path: Path = Path("storage/manifest.sqlite")
    log_dir: Path = Path("logs")
    log_level: str = "INFO"

    embedding_model: str = "intfloat/multilingual-e5-base"
    embedding_device: str = "cpu"
    embedding_batch_size: int = Field(default=16, gt=0)

    chunk_size: int = Field(default=1200, gt=0)
    chunk_overlap: int = Field(default=200, ge=0)
    min_chunk_size: int = Field(default=200, gt=0)
    exclude_toc_chunks: bool = True
    clean_dot_leaders: bool = True
    top_k: int = Field(default=3, gt=0)
    min_retrieval_score: float = Field(default=0.84, ge=0.0, le=1.0)
    snippet_max_chars: int = Field(default=400, gt=0)
    min_context_chars: int = Field(default=500, ge=0)
    max_context_chars: int = Field(default=6000, gt=0)
    max_sources: int = Field(default=3, gt=0)
    enable_domain_guardrails: bool = True
    domain_name: str = "normativas y políticas de la Universidad de Navarra"
    eval_dataset_path: Path = Path("data/eval/evaluation_questions.jsonl")
    eval_reports_dir: Path = Path("data/eval/reports")
    eval_answer_preview_chars: int = Field(default=300, gt=0)

    allow_external_llm: bool = False
    llm_provider: str = "none"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    llm_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    llm_max_context_chars: int = Field(default=6000, gt=0)
    llm_mode_default: str = "extractive"

    api_host: str = "127.0.0.1"
    api_port: int = Field(default=8000, gt=0, le=65535)
    api_reload: bool = False
    api_include_debug_errors: bool = False
    api_base_url: str = "http://127.0.0.1:8000"
    api_request_timeout_seconds: float = Field(default=120.0, gt=0)

    @field_validator(
        "documents_dir",
        "data_dir",
        "processed_dir",
        "eval_dir",
        "chunks_dir",
        "chunks_file",
        "eval_dataset_path",
        "eval_reports_dir",
        "storage_dir",
        "chroma_dir",
        "manifest_db_path",
        "log_dir",
        mode="after",
    )
    @classmethod
    def resolve_project_path(cls, value: Path) -> Path:
        if value.is_absolute():
            return value
        return PROJECT_ROOT / value

    @field_validator("log_level", mode="after")
    @classmethod
    def normalize_log_level(cls, value: str) -> str:
        return value.upper()

    @property
    def required_directories(self) -> tuple[Path, ...]:
        return (
            self.documents_dir,
            self.data_dir,
            self.processed_dir,
            self.eval_dir,
            self.eval_reports_dir,
            self.chunks_dir,
            self.storage_dir,
            self.chroma_dir,
            self.log_dir,
        )

    def ensure_directories(self) -> list[Path]:
        created_or_existing: list[Path] = []
        for directory in self.required_directories:
            directory.mkdir(parents=True, exist_ok=True)
            created_or_existing.append(directory)
        return created_or_existing

    def public_dict(self) -> dict[str, Any]:
        hidden_terms = ("key", "secret", "token", "password")
        values = self.model_dump()
        public: dict[str, Any] = {}

        for key, value in values.items():
            if any(term in key.lower() for term in hidden_terms):
                public[key] = "***"
            elif isinstance(value, Path):
                public[key] = str(value)
            else:
                public[key] = value

        return public


@lru_cache
def get_settings() -> AppSettings:
    return AppSettings()
