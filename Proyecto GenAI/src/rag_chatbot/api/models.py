from typing import Any

from pydantic import BaseModel, Field


class ApiErrorResponse(BaseModel):
    detail: str


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    environment: str


class ConfigResponse(BaseModel):
    documents_dir: str
    chroma_collection: str
    embedding_model: str
    top_k: int
    min_retrieval_score: float
    allow_external_llm: bool


class RetrieveRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int | None = Field(default=None, ge=1)
    min_score: float | None = Field(default=None, ge=0.0, le=1.0)
    document_id: str | None = None
    file_name: str | None = None
    include_text: bool = False


class RetrieveResultItem(BaseModel):
    rank: int
    chunk_id: str
    score: float
    distance: float | None = None
    file_name: str
    page_number: int | None
    source_label: str
    snippet: str
    metadata: dict[str, Any]
    text: str | None = None


class RetrieveResponse(BaseModel):
    question: str
    top_k: int
    result_count: int
    results: list[RetrieveResultItem]


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int | None = Field(default=None, ge=1)
    min_score: float | None = Field(default=None, ge=0.0, le=1.0)
    show_chunks: bool = False
    document_id: str | None = None
    file_name: str | None = None


class QueryChunkItem(BaseModel):
    chunk_id: str
    score: float
    file_name: str
    page_number: int | None
    snippet: str
    text: str | None = None


class QuerySourceItem(BaseModel):
    file_name: str
    page_number: int | None
    chunk_id: str
    source_label: str
    snippet: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    has_sufficient_context: bool
    warning: str | None = None
    sources: list[QuerySourceItem]
    retrieved_chunks: list[QueryChunkItem]


class IngestResponse(BaseModel):
    status: str
    documents_found: int
    processed: int
    skipped: int
    failed: int
    requires_ocr: int


class BuildChunksResponse(BaseModel):
    status: str
    documents_processed: int
    documents_skipped: int
    chunks_generated: int
    chunks_file: str


class BuildIndexRequest(BaseModel):
    reset: bool = False
    limit: int | None = Field(default=None, ge=1)


class BuildIndexResponse(BaseModel):
    status: str
    chunks_read: int
    chunks_indexed: int
    chunks_skipped: int
    collection: str
    chroma_dir: str


class IndexInfoResponse(BaseModel):
    collection: str
    vector_count: int | None = None
    chroma_dir: str
    embedding_model: str
    embedding_device: str
    chroma_available: bool
    error_message: str | None = None


class EvalSummaryResponse(BaseModel):
    total_questions: int
    passed: int
    failed: int
    pass_rate: float
    avg_keyword_hit_rate: float
    avg_top_score: float | None = None
