import logging
from typing import Protocol

from rag_chatbot.config import AppSettings, get_settings
from rag_chatbot.llm.base import LLMProvider
from rag_chatbot.llm.gemini_provider import GeminiProvider
from rag_chatbot.llm.prompt_builder import build_grounded_prompt
from rag_chatbot.logging_config import LOGGER_NAME
from rag_chatbot.rag.answer_builder import build_extractive_answer
from rag_chatbot.rag.citations import deduplicate_sources
from rag_chatbot.rag.domain_guardrails import classify_question_domain
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
        llm_provider: LLMProvider | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.logger = logger or logging.getLogger(LOGGER_NAME)
        self.retriever = retriever or SemanticRetriever(self.settings, logger=self.logger)
        self.llm_provider = llm_provider

    def ask(
        self,
        question: str,
        *,
        top_k: int | None = None,
        min_score: float | None = None,
        document_id: str | None = None,
        file_name: str | None = None,
        mode: str | None = None,
    ) -> RagAnswer:
        mode = self._normalize_mode(mode)
        min_score = self.settings.min_retrieval_score if min_score is None else min_score
        domain_result = (
            classify_question_domain(question)
            if self.settings.enable_domain_guardrails
            else None
        )

        if domain_result is not None and domain_result.is_in_domain is False:
            context = evaluate_context_sufficiency(
                [],
                min_score=min_score,
                min_context_chars=self.settings.min_context_chars,
                domain_result=domain_result,
                enable_domain_guardrails=self.settings.enable_domain_guardrails,
            )
            answer = build_extractive_answer(
                results=[],
                sources=[],
                context=context,
                max_context_chars=self.settings.max_context_chars,
            )
            self.logger.info(
                "Local RAG rejected question by domain guardrail reason=%s",
                domain_result.reason,
            )
            return RagAnswer(
                question=question,
                answer=answer,
                has_sufficient_context=False,
                warning=context.warning,
                sources=[],
                retrieved_chunks=[],
                context=context,
                rejection_reason=context.rejection_reason,
                mode="extractive",
                llm_provider=None,
                llm_model=None,
                llm_used=False,
                domain=domain_result,
            )

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
            domain_result=domain_result,
            enable_domain_guardrails=self.settings.enable_domain_guardrails,
        )
        sources = deduplicate_sources(results, max_sources=self.settings.max_sources)
        answer = build_extractive_answer(
            results=results,
            sources=sources,
            context=context,
            max_context_chars=self.settings.max_context_chars,
        )
        llm_used = False
        llm_warning = None
        llm_provider_name = None
        llm_model = None

        if mode == "llm" and context.has_sufficient_context:
            generated_answer, llm_warning, llm_used, llm_provider_name, llm_model = (
                self._try_generate_answer(
                    question=question,
                    results=results,
                    sources=sources,
                )
            )
            if generated_answer:
                answer = generated_answer
        elif mode == "llm" and not context.has_sufficient_context:
            llm_warning = "Gemini no se llamó porque el contexto no fue suficiente."
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
            rejection_reason=context.rejection_reason,
            mode="llm" if llm_used else "extractive",
            llm_provider=llm_provider_name,
            llm_model=llm_model,
            llm_used=llm_used,
            llm_warning=llm_warning,
            domain=domain_result,
        )

    def _normalize_mode(self, mode: str | None) -> str:
        selected = (mode or self.settings.llm_mode_default).strip().casefold()
        if selected not in {"extractive", "llm"}:
            raise ValueError("mode must be either 'extractive' or 'llm'")
        return selected

    def _try_generate_answer(
        self,
        *,
        question: str,
        results,
        sources,
    ) -> tuple[str | None, str | None, bool, str | None, str | None]:
        if not self.settings.allow_external_llm:
            return (
                None,
                "Gemini no se llamó porque ALLOW_EXTERNAL_LLM=false.",
                False,
                None,
                None,
            )

        if self.settings.llm_provider.casefold() != "gemini":
            return (
                None,
                "Gemini no se llamó porque LLM_PROVIDER no es 'gemini'.",
                False,
                None,
                None,
            )

        provider = self.llm_provider or GeminiProvider(
            api_key=self.settings.gemini_api_key,
            model_name=self.settings.gemini_model,
            temperature=self.settings.llm_temperature,
        )
        prompt = build_grounded_prompt(
            question=question,
            results=results,
            sources=sources,
            max_context_chars=self.settings.llm_max_context_chars,
        )

        try:
            answer = provider.generate(prompt)
        except Exception as exc:
            self.logger.warning("LLM generation failed; falling back to extractive answer: %s", exc)
            return (
                None,
                "Gemini falló; se muestra respuesta extractiva local.",
                False,
                provider.provider_name,
                provider.model_name,
            )

        if not answer:
            return (
                None,
                "Gemini no devolvió contenido; se muestra respuesta extractiva local.",
                False,
                provider.provider_name,
                provider.model_name,
            )

        return answer, None, True, provider.provider_name, provider.model_name
