from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class DocumentStatus(StrEnum):
    PENDING = "pending"
    EXTRACTED = "extracted"
    EMPTY_TEXT = "empty_text"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"


ExtractionStatus = DocumentStatus


class DocumentPage(BaseModel):
    page_number: int | None
    text: str
    char_count: int


class ProcessedDocument(BaseModel):
    document_id: str
    file_name: str
    file_path: str
    file_type: str
    file_hash: str
    extraction_status: DocumentStatus
    requires_ocr: bool
    page_count: int
    char_count: int
    processed_at: datetime
    pages: list[DocumentPage]
    metadata: dict[str, Any] = Field(default_factory=dict)
    error_message: str | None = None


class ManifestRecord(BaseModel):
    document_id: str
    file_name: str
    file_path: str
    file_type: str
    file_hash: str
    status: DocumentStatus
    requires_ocr: bool
    page_count: int
    char_count: int
    processed_at: datetime
    error_message: str | None = None

    @classmethod
    def from_processed_document(cls, document: ProcessedDocument) -> "ManifestRecord":
        return cls(
            document_id=document.document_id,
            file_name=document.file_name,
            file_path=document.file_path,
            file_type=document.file_type,
            file_hash=document.file_hash,
            status=document.extraction_status,
            requires_ocr=document.requires_ocr,
            page_count=document.page_count,
            char_count=document.char_count,
            processed_at=document.processed_at,
            error_message=document.error_message,
        )


class Chunk(BaseModel):
    chunk_id: str
    document_id: str
    file_name: str
    file_path: str
    file_type: str
    page_number: int | None
    chunk_index: int
    text: str
    char_count: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChunkingResult(BaseModel):
    documents_processed: int
    documents_skipped: int
    total_chunks: int
    chunks_file: str
    manifest_file: str
    chunk_size: int
    chunk_overlap: int
    min_chunk_size: int


class ChunkingManifest(BaseModel):
    generated_at: datetime
    documents_processed: int
    documents_skipped: int
    total_chunks: int
    chunks_by_document: dict[str, int]
    chunks_by_page: dict[str, int]
    chunk_size: int
    chunk_overlap: int
    min_chunk_size: int
    chunks_file: str


class IndexingResult(BaseModel):
    chunks_read: int
    chunks_indexed: int
    chunks_skipped: int
    collection_name: str
    chroma_dir: str
    embedding_model: str
    embedding_device: str


class IndexInfo(BaseModel):
    collection_name: str
    chroma_dir: str
    embedding_model: str
    embedding_device: str
    chroma_available: bool
    vector_count: int | None = None
    error_message: str | None = None


class RetrievalQuery(BaseModel):
    question: str
    top_k: int
    min_score: float | None = None
    document_id: str | None = None
    file_name: str | None = None


class RetrievalResult(BaseModel):
    chunk_id: str
    score: float
    distance: float | None = None
    text: str
    snippet: str
    document_id: str
    file_name: str
    file_path: str
    file_type: str
    page_number: int | None
    chunk_index: int
    char_count: int
    source_label: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalResponse(BaseModel):
    query: RetrievalQuery
    results: list[RetrievalResult]
    score_description: str
    collection_name: str
    embedding_model: str
    embedding_device: str


class RagQuestion(BaseModel):
    question: str
    top_k: int
    min_score: float | None = None


class RagSource(BaseModel):
    file_name: str
    page_number: int | None
    chunk_id: str
    source_label: str
    snippet: str


class RetrievedChunk(BaseModel):
    chunk_id: str
    score: float
    file_name: str
    page_number: int | None
    text: str
    snippet: str


class ContextSufficiencyResult(BaseModel):
    has_sufficient_context: bool
    warning: str | None = None
    reason: str
    rejection_reason: str | None = None
    best_score: float | None = None
    total_context_chars: int
    result_count: int


class DomainGuardrailResult(BaseModel):
    is_in_domain: bool | None
    reason: str
    matched_terms: list[str] = Field(default_factory=list)
    blocked_terms: list[str] = Field(default_factory=list)


class RagAnswer(BaseModel):
    question: str
    answer: str
    has_sufficient_context: bool
    warning: str | None = None
    rejection_reason: str | None = None
    domain: DomainGuardrailResult | None = None
    sources: list[RagSource]
    retrieved_chunks: list[RetrievedChunk]
    context: ContextSufficiencyResult


class EvaluationQuestion(BaseModel):
    question_id: str
    question: str
    expected_keywords: list[str] = Field(default_factory=list)
    expected_files: list[str] = Field(default_factory=list)
    should_have_answer: bool
    notes: str | None = None


class EvaluationMetrics(BaseModel):
    has_answer: bool
    expected_answer_behavior: bool
    keyword_hit_rate: float
    expected_file_hit: bool
    source_count: int
    retrieved_chunk_count: int
    top_score: float | None = None
    avg_score: float | None = None
    passed: bool


class EvaluationResult(BaseModel):
    question_id: str
    question: str
    should_have_answer: bool
    expected_keywords: list[str]
    expected_files: list[str]
    answer: str
    answer_preview: str
    has_sufficient_context: bool
    warning: str | None = None
    sources: list[RagSource]
    retrieved_chunks: list[RetrievedChunk]
    retrieved_files: list[str]
    found_keywords: list[str]
    missing_keywords: list[str]
    expected_files_found: list[str]
    expected_files_missing: list[str]
    metrics: EvaluationMetrics
    failure_reasons: list[str] = Field(default_factory=list)
    notes: str | None = None


class EvaluationSummary(BaseModel):
    total_questions: int
    passed: int
    failed: int
    pass_rate: float
    insufficient_context_count: int
    out_of_domain_correct_rejections: int
    avg_keyword_hit_rate: float
    avg_top_score: float | None = None
    false_positive_count: int = 0
    false_negative_count: int = 0
    expected_file_hit_rate: float = 0.0
    avg_source_count: float = 0.0
    failure_reason_counts: dict[str, int] = Field(default_factory=dict)


class EvaluationGridResult(BaseModel):
    top_k: int
    min_score: float
    total_questions: int
    passed: int
    failed: int
    pass_rate: float
    avg_keyword_hit_rate: float
    expected_file_hit_rate: float
    false_positive_count: int
    false_negative_count: int
    avg_top_score: float | None = None
    avg_source_count: float
    score_margin: float | None = None
    failure_reason_counts: dict[str, int] = Field(default_factory=dict)


class EvaluationGridRecommendation(BaseModel):
    top_k: int
    min_score: float
    reason: str
    pass_rate: float
    false_positive_count: int
    expected_file_hit_rate: float


class EvaluationGridReport(BaseModel):
    generated_at: datetime
    dataset_path: str
    results: list[EvaluationGridResult]
    recommendation: EvaluationGridRecommendation | None = None


class EvaluationReport(BaseModel):
    generated_at: datetime
    dataset_path: str
    results: list[EvaluationResult]
    summary: EvaluationSummary


class HealthCheck(BaseModel):
    status: str
    project_name: str
    version: str
    documents_dir_exists: bool
    external_llm_enabled: bool
