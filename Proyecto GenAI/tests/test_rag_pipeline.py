from rag_chatbot.config import AppSettings
from rag_chatbot.rag.pipeline import LocalRagPipeline
from rag_chatbot.schemas import RetrievalQuery, RetrievalResponse, RetrievalResult


class FakeRetriever:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def search(
        self,
        question,
        *,
        top_k=None,
        min_score=None,
        document_id=None,
        file_name=None,
    ):
        self.calls.append(
            {
                "question": question,
                "top_k": top_k,
                "min_score": min_score,
                "document_id": document_id,
                "file_name": file_name,
            }
        )
        return RetrievalResponse(
            query=RetrievalQuery(
                question=question,
                top_k=top_k or 5,
                min_score=min_score,
                document_id=document_id,
                file_name=file_name,
            ),
            results=self.results,
            score_description="score",
            collection_name="documents",
            embedding_model="fake",
            embedding_device="cpu",
        )


class FakeLLMProvider:
    provider_name = "gemini"
    model_name = "fake-gemini"

    def __init__(self, answer: str = "Respuesta generativa con fuentes.") -> None:
        self.answer = answer
        self.prompts = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.answer


class FailingLLMProvider(FakeLLMProvider):
    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        raise RuntimeError("boom")


def _settings(tmp_path) -> AppSettings:
    return AppSettings(
        documents_dir=tmp_path / "Documentos",
        data_dir=tmp_path / "data",
        processed_dir=tmp_path / "data" / "processed",
        eval_dir=tmp_path / "data" / "eval",
        chunks_dir=tmp_path / "data" / "chunks",
        chunks_file=tmp_path / "data" / "chunks" / "chunks.jsonl",
        storage_dir=tmp_path / "storage",
        chroma_dir=tmp_path / "storage" / "chroma",
        manifest_db_path=tmp_path / "storage" / "manifest.sqlite",
        log_dir=tmp_path / "logs",
        min_context_chars=100,
        max_sources=2,
        llm_provider="gemini",
        gemini_api_key="test-key",
        _env_file=None,
    )


def _result(
    chunk_id: str,
    file_name: str = "archivo.pdf",
    page_number: int = 1,
    score: float = 0.8,
):
    text = "Texto normativo recuperado. " * 20
    return RetrievalResult(
        chunk_id=chunk_id,
        score=score,
        distance=1 - score,
        text=text,
        snippet="Texto normativo recuperado.",
        document_id="doc_1",
        file_name=file_name,
        file_path=f"Documentos/{file_name}",
        file_type="pdf",
        page_number=page_number,
        chunk_index=0,
        char_count=len(text),
        source_label=f"{file_name}, pagina {page_number}",
    )


def test_local_rag_pipeline_builds_answer_with_context(tmp_path) -> None:
    retriever = FakeRetriever([_result("chunk_1")])
    pipeline = LocalRagPipeline(_settings(tmp_path), retriever=retriever)

    answer = pipeline.ask("pregunta", top_k=3, min_score=0.3)

    assert retriever.calls[0]["question"] == "pregunta"
    assert retriever.calls[0]["top_k"] == 3
    assert answer.has_sufficient_context is True
    assert answer.sources[0].file_name == "archivo.pdf"
    assert answer.sources[0].page_number == 1
    assert answer.sources[0].chunk_id == "chunk_1"
    assert answer.retrieved_chunks[0].snippet == "Texto normativo recuperado."


def test_local_rag_pipeline_rejects_without_context(tmp_path) -> None:
    pipeline = LocalRagPipeline(_settings(tmp_path), retriever=FakeRetriever([]))

    answer = pipeline.ask("pregunta")

    assert answer.has_sufficient_context is False
    assert "No encontré información suficiente" in answer.answer
    assert answer.rejection_reason == "insufficient_context"
    assert answer.sources == []


def test_local_rag_pipeline_rejects_when_top_score_is_below_threshold(tmp_path) -> None:
    pipeline = LocalRagPipeline(
        _settings(tmp_path),
        retriever=FakeRetriever([_result("chunk_1", score=0.8)]),
    )

    answer = pipeline.ask("pregunta", min_score=0.82)

    assert answer.has_sufficient_context is False
    assert answer.context.reason == "low_score"
    assert answer.rejection_reason == "low_score"
    assert "Con base en los documentos recuperados" not in answer.answer


def test_local_rag_pipeline_limits_sources(tmp_path) -> None:
    pipeline = LocalRagPipeline(
        _settings(tmp_path),
        retriever=FakeRetriever(
            [
                _result("chunk_1", "a.pdf", 1),
                _result("chunk_2", "b.pdf", 1),
                _result("chunk_3", "c.pdf", 1),
            ]
        ),
    )

    answer = pipeline.ask("pregunta")

    assert len(answer.sources) == 2
    assert [source.file_name for source in answer.sources] == ["a.pdf", "b.pdf"]


def test_local_rag_pipeline_rejects_blocked_domain_without_retrieval(tmp_path) -> None:
    retriever = FakeRetriever([_result("chunk_1", score=0.99)])
    pipeline = LocalRagPipeline(_settings(tmp_path), retriever=retriever)

    answer = pipeline.ask("¿Qué documentos mencionan al Rey?")

    assert retriever.calls == []
    assert answer.has_sufficient_context is False
    assert answer.context.reason == "out_of_domain"
    assert answer.rejection_reason == "out_of_domain"
    assert answer.domain is not None
    assert answer.domain.blocked_terms == ["rey"]
    assert "fuera del alcance" in answer.answer


def test_local_rag_pipeline_allows_ambiguous_question_when_retrieval_is_strong(
    tmp_path,
) -> None:
    retriever = FakeRetriever([_result("chunk_1", score=0.95)])
    pipeline = LocalRagPipeline(_settings(tmp_path), retriever=retriever)

    answer = pipeline.ask("¿Dónde aparece este criterio?", min_score=0.84)

    assert retriever.calls[0]["question"] == "¿Dónde aparece este criterio?"
    assert answer.has_sufficient_context is True
    assert answer.domain is not None
    assert answer.domain.is_in_domain is None


def test_local_rag_pipeline_does_not_call_llm_when_external_llm_disabled(tmp_path) -> None:
    settings = _settings(tmp_path)
    settings.allow_external_llm = False
    llm = FakeLLMProvider()
    pipeline = LocalRagPipeline(
        settings,
        retriever=FakeRetriever([_result("chunk_1", score=0.95)]),
        llm_provider=llm,
    )

    answer = pipeline.ask("pregunta", min_score=0.3, mode="llm")

    assert llm.prompts == []
    assert answer.llm_used is False
    assert answer.mode == "extractive"
    assert "ALLOW_EXTERNAL_LLM=false" in answer.llm_warning


def test_local_rag_pipeline_uses_llm_when_allowed_and_context_sufficient(tmp_path) -> None:
    settings = _settings(tmp_path)
    settings.allow_external_llm = True
    llm = FakeLLMProvider("Respuesta Gemini citando archivo.pdf, pagina 1.")
    pipeline = LocalRagPipeline(
        settings,
        retriever=FakeRetriever([_result("chunk_1", score=0.95)]),
        llm_provider=llm,
    )

    answer = pipeline.ask("¿Qué dice compliance?", min_score=0.3, mode="llm")

    assert answer.answer == "Respuesta Gemini citando archivo.pdf, pagina 1."
    assert answer.llm_used is True
    assert answer.mode == "llm"
    assert answer.llm_provider == "gemini"
    assert answer.llm_model == "fake-gemini"
    assert "archivo.pdf" in llm.prompts[0]
    assert "chunk_1" in llm.prompts[0]


def test_local_rag_pipeline_does_not_call_llm_when_guardrails_reject(tmp_path) -> None:
    settings = _settings(tmp_path)
    settings.allow_external_llm = True
    llm = FakeLLMProvider()
    retriever = FakeRetriever([_result("chunk_1", score=0.99)])
    pipeline = LocalRagPipeline(settings, retriever=retriever, llm_provider=llm)

    answer = pipeline.ask("¿Qué documentos mencionan al Rey?", mode="llm")

    assert retriever.calls == []
    assert llm.prompts == []
    assert answer.rejection_reason == "out_of_domain"
    assert answer.llm_used is False


def test_local_rag_pipeline_does_not_call_llm_when_context_insufficient(tmp_path) -> None:
    settings = _settings(tmp_path)
    settings.allow_external_llm = True
    llm = FakeLLMProvider()
    pipeline = LocalRagPipeline(
        settings,
        retriever=FakeRetriever([_result("chunk_1", score=0.2)]),
        llm_provider=llm,
    )

    answer = pipeline.ask("pregunta", min_score=0.84, mode="llm")

    assert llm.prompts == []
    assert answer.has_sufficient_context is False
    assert answer.llm_used is False
    assert "contexto no fue suficiente" in answer.llm_warning


def test_local_rag_pipeline_falls_back_to_extractive_when_llm_fails(tmp_path) -> None:
    settings = _settings(tmp_path)
    settings.allow_external_llm = True
    llm = FailingLLMProvider()
    pipeline = LocalRagPipeline(
        settings,
        retriever=FakeRetriever([_result("chunk_1", score=0.95)]),
        llm_provider=llm,
    )

    answer = pipeline.ask("pregunta", min_score=0.3, mode="llm")

    assert llm.prompts
    assert answer.llm_used is False
    assert answer.mode == "extractive"
    assert "Con base en los documentos recuperados" in answer.answer
    assert "Gemini falló" in answer.llm_warning


def test_local_rag_pipeline_does_not_log_api_key_when_llm_fails(tmp_path, caplog) -> None:
    settings = _settings(tmp_path)
    settings.allow_external_llm = True
    settings.gemini_api_key = "super-secret-key"
    pipeline = LocalRagPipeline(
        settings,
        retriever=FakeRetriever([_result("chunk_1", score=0.95)]),
        llm_provider=FailingLLMProvider(),
    )

    with caplog.at_level("WARNING"):
        pipeline.ask("pregunta", min_score=0.3, mode="llm")

    assert "super-secret-key" not in caplog.text
