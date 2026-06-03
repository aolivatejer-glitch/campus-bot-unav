import logging
from typing import Protocol

from rag_chatbot.config import AppSettings, get_settings
from rag_chatbot.logging_config import LOGGER_NAME
from rag_chatbot.rag.answer_builder import build_extractive_answer
from rag_chatbot.rag.citations import deduplicate_sources
from rag_chatbot.rag.guardrails import evaluate_context_sufficiency
from rag_chatbot.retrieval.retriever import SemanticRetriever
from rag_chatbot.schemas import RagAnswer, RetrievedChunk, RetrievalResponse


class RetrieverProtocol(Protocol):
    def search(
        self,
        question: str,
        *,
        top_k: int | None = None,
        min_score: float | None = None,
        document_id: str | None = None,
        file_name: str | None = None,
    ) -> RetrievalResponse:
        ...


class LocalRagPipeline:
    def __init__(
        self,
        settings: AppSettings | None = None,
        *,
        retriever: RetrieverProtocol | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.logger = logger or logging.getLogger(LOGGER_NAME)
        self.retriever = retriever or SemanticRetriever(self.settings, logger=self.logger)

    def ask(
        self,
        question: str,
        *,
        top_k: int | None = None,
        min_score: float | None = None,
        document_id: str | None = None,
        file_name: str | None = None,
    ) -> RagAnswer:
        min_score = self.settings.min_retrieval_score if min_score is None else min_score
        retrieval_response = self.retriever.search(
            question,
            top_k=top_k,
            min_score=min_score,
            document_id=document_id,
            file_name=file_name,
        )
        results = retrieval_response.results
        context = evaluate_context_sufficiency(
            results,
            min_score=min_score,
            min_context_chars=self.settings.min_context_chars,
        )
        sources = deduplicate_sources(results, max_sources=self.settings.max_sources)
        answer = build_extractive_answer(
            results=results,
            sources=sources,
            context=context,
            max_context_chars=self.settings.max_context_chars,
        )
        retrieved_chunks = [
            RetrievedChunk(
                chunk_id=result.chunk_id,
                score=result.score,
                file_name=result.file_name,
                page_number=result.page_number,
                text=result.text,
                snippet=result.snippet,
            )
            for result in results
        ]

        self.logger.info(
            "Local RAG answer built sufficient=%s results=%s sources=%s",
            context.has_sufficient_context,
            len(results),
            len(sources),
        )

        return RagAnswer(
            question=question,
            answer=answer,
            has_sufficient_context=context.has_sufficient_context,
            warning=context.warning,
            sources=sources,
            retrieved_chunks=retrieved_chunks,
            context=context,
        )
