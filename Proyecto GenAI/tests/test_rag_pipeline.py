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
