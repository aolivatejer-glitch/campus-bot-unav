import logging
from typing import Any, Protocol

from rag_chatbot.config import AppSettings, get_settings
from rag_chatbot.embeddings.base import EmbeddingProvider
from rag_chatbot.embeddings.sentence_transformers_provider import (
    SentenceTransformersEmbeddingProvider,
)
from rag_chatbot.logging_config import LOGGER_NAME
from rag_chatbot.retrieval.formatting import (
    cosine_distance_to_score,
    make_snippet,
    make_source_label,
)
from rag_chatbot.schemas import RetrievalQuery, RetrievalResponse, RetrievalResult
from rag_chatbot.vectorstores.chroma_store import ChromaVectorStore


SCORE_DESCRIPTION = (
    "score is an approximate cosine similarity derived from Chroma cosine distance "
    "as max(0, min(1, 1 - distance)); higher is more relevant."
)


class QueryableVectorStore(Protocol):
    @property
    def collection_name(self) -> str:
        ...

    def count(self) -> int:
        ...

    def query(
        self,
        *,
        query_embedding: list[float],
        top_k: int,
        document_id: str | None = None,
        file_name: str | None = None,
    ) -> dict:
        ...


class SemanticRetriever:
    def __init__(
        self,
        settings: AppSettings | None = None,
        *,
        embedding_provider: EmbeddingProvider | None = None,
        vector_store: QueryableVectorStore | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.logger = logger or logging.getLogger(LOGGER_NAME)
        self.embedding_provider = embedding_provider or SentenceTransformersEmbeddingProvider(
            model_name=self.settings.embedding_model,
            device=self.settings.embedding_device,
            batch_size=self.settings.embedding_batch_size,
        )
        self.vector_store = vector_store or ChromaVectorStore(
            persist_dir=self.settings.chroma_dir,
            collection_name=self.settings.chroma_collection_name,
        )

    def search(
        self,
        question: str,
        *,
        top_k: int | None = None,
        min_score: float | None = None,
        document_id: str | None = None,
        file_name: str | None = None,
    ) -> RetrievalResponse:
        question = question.strip()
        if not question:
            raise ValueError("Question cannot be empty")

        top_k = top_k or self.settings.top_k
        min_score = self.settings.min_retrieval_score if min_score is None else min_score

        vector_count = self.vector_store.count()
        if vector_count == 0:
            raise ValueError(
                "Vector index is empty. Run build-index after ingest and build-chunks."
            )

        query_embedding = self.embedding_provider.embed_query(question)
        raw_results = self.vector_store.query(
            query_embedding=query_embedding,
            top_k=top_k,
            document_id=document_id,
            file_name=file_name,
        )
        results = self._parse_results(raw_results, min_score=min_score)

        query = RetrievalQuery(
            question=question,
            top_k=top_k,
            min_score=min_score,
            document_id=document_id,
            file_name=file_name,
        )

        self.logger.info(
            "Semantic search completed top_k=%s min_score=%s results=%s",
            top_k,
            min_score,
            len(results),
        )

        return RetrievalResponse(
            query=query,
            results=results,
            score_description=SCORE_DESCRIPTION,
            collection_name=self.vector_store.collection_name,
            embedding_model=self.embedding_provider.model_name,
            embedding_device=self.embedding_provider.device,
        )

    def _parse_results(
        self,
        raw_results: dict[str, Any],
        *,
        min_score: float | None,
    ) -> list[RetrievalResult]:
        ids = _first_result_list(raw_results.get("ids"))
        documents = _first_result_list(raw_results.get("documents"))
        metadatas = _first_result_list(raw_results.get("metadatas"))
        distances = _first_result_list(raw_results.get("distances"))

        results: list[RetrievalResult] = []
        for index, chunk_id in enumerate(ids):
            text = documents[index] if index < len(documents) else ""
            metadata = metadatas[index] if index < len(metadatas) else {}
            distance = distances[index] if index < len(distances) else None
            score = cosine_distance_to_score(distance)

            if min_score is not None and score < min_score:
                continue

            page_number = _metadata_page_number(metadata.get("page_number"))
            file_name = str(metadata.get("file_name", "unknown"))
            result = RetrievalResult(
                chunk_id=str(chunk_id),
                score=score,
                distance=distance,
                text=text,
                snippet=make_snippet(text, self.settings.snippet_max_chars),
                document_id=str(metadata.get("document_id", "")),
                file_name=file_name,
                file_path=str(metadata.get("file_path", "")),
                file_type=str(metadata.get("file_type", "")),
                page_number=page_number,
                chunk_index=int(metadata.get("chunk_index", -1)),
                char_count=int(metadata.get("char_count", len(text))),
                source_label=make_source_label(file_name, page_number),
                metadata=dict(metadata),
            )
            results.append(result)

        return sorted(results, key=lambda result: result.score, reverse=True)


def _first_result_list(value: Any) -> list:
    if not value:
        return []

    if isinstance(value, list) and value and isinstance(value[0], list):
        return value[0]

    if isinstance(value, list):
        return value

    return []


def _metadata_page_number(value: Any) -> int | None:
    if value is None:
        return None

    page_number = int(value)
    return None if page_number < 0 else page_number
