from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from rag_chatbot.api import dependencies
from rag_chatbot.api import routes as routes_module
from rag_chatbot.api.main import app
from rag_chatbot.config import AppSettings
from rag_chatbot.schemas import (
    ContextSufficiencyResult,
    IndexInfo,
    RagAnswer,
    RagSource,
    RetrievedChunk,
    RetrievalQuery,
    RetrievalResponse,
    RetrievalResult,
)


@pytest.fixture()
def client(tmp_path):
    settings = AppSettings(
        documents_dir=tmp_path / "Documentos",
        data_dir=tmp_path / "data",
        processed_dir=tmp_path / "data" / "processed",
        eval_dir=tmp_path / "data" / "eval",
        eval_dataset_path=tmp_path / "data" / "eval" / "questions.jsonl",
        eval_reports_dir=tmp_path / "data" / "eval" / "reports",
        chunks_dir=tmp_path / "data" / "chunks",
        chunks_file=tmp_path / "data" / "chunks" / "chunks.jsonl",
        storage_dir=tmp_path / "storage",
        chroma_dir=tmp_path / "storage" / "chroma",
        manifest_db_path=tmp_path / "storage" / "manifest.sqlite",
        log_dir=tmp_path / "logs",
        _env_file=None,
    )
    app.dependency_overrides[dependencies.get_api_settings] = lambda: settings
    yield TestClient(app)
    app.dependency_overrides.clear()


class FakeRetriever:
    def search(
        self,
        question,
        *,
        top_k=None,
        min_score=None,
        document_id=None,
        file_name=None,
        mode=None,
    ):
        return RetrievalResponse(
            query=RetrievalQuery(
                question=question,
                top_k=top_k or 5,
                min_score=min_score,
                document_id=document_id,
                file_name=file_name,
            ),
            results=[
                RetrievalResult(
                    chunk_id="chunk_1",
                    score=0.82,
                    distance=0.18,
                    text="Texto completo del chunk.",
                    snippet="Texto breve.",
                    document_id="doc_1",
                    file_name="sample.pdf",
                    file_path="Documentos/sample.pdf",
                    file_type="pdf",
                    page_number=4,
                    chunk_index=0,
                    char_count=24,
                    source_label="sample.pdf, pagina 4",
                    metadata={"document_id": "doc_1"},
                )
            ],
            score_description="score",
            collection_name="documents",
            embedding_model="fake",
            embedding_device="cpu",
        )


class EmptyIndexRetriever(FakeRetriever):
    def search(self, *args, **kwargs):
        raise ValueError("Vector index is empty. Run build-index after ingest and build-chunks.")


class FakeRagPipeline:
    def ask(
        self,
        question,
        *,
        top_k=None,
        min_score=None,
        document_id=None,
        file_name=None,
        mode=None,
    ):
        text = "Texto completo recuperado."
        return RagAnswer(
            question=question,
            answer="Respuesta extractiva local.",
            has_sufficient_context=True,
            warning=None,
            mode="llm" if mode == "llm" else "extractive",
            llm_provider="gemini" if mode == "llm" else None,
            llm_model="fake-gemini" if mode == "llm" else None,
            llm_used=mode == "llm",
            sources=[
                RagSource(
                    file_name="sample.pdf",
                    page_number=4,
                    chunk_id="chunk_1",
                    source_label="sample.pdf, pagina 4",
                    snippet="Texto breve.",
                )
            ],
            retrieved_chunks=[
                RetrievedChunk(
                    chunk_id="chunk_1",
                    score=0.82,
                    file_name="sample.pdf",
                    page_number=4,
                    text=text,
                    snippet="Texto breve.",
                )
            ],
            context=ContextSufficiencyResult(
                has_sufficient_context=True,
                warning=None,
                reason="sufficient",
                best_score=0.82,
                total_context_chars=len(text),
                result_count=1,
            ),
        )


class RejectedDomainRagPipeline:
    def ask(
        self,
        question,
        *,
        top_k=None,
        min_score=None,
        document_id=None,
        file_name=None,
        mode=None,
    ):
        return RagAnswer(
            question=question,
            answer=(
                "No puedo responder esa pregunta porque está fuera del alcance de "
                "los documentos indexados."
            ),
            has_sufficient_context=False,
            warning="Pregunta fuera del alcance del corpus.",
            rejection_reason="out_of_domain",
            sources=[],
            retrieved_chunks=[],
            context=ContextSufficiencyResult(
                has_sufficient_context=False,
                warning="Pregunta fuera del alcance del corpus.",
                reason="out_of_domain",
                rejection_reason="out_of_domain",
                best_score=None,
                total_context_chars=0,
                result_count=0,
            ),
        )


def test_health(client) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_config(client) -> None:
    response = client.get("/config")

    assert response.status_code == 200
    payload = response.json()
    assert payload["chroma_collection"] == "documents"
    assert payload["allow_external_llm"] is False


def test_retrieve_with_mock_does_not_return_text_by_default(client) -> None:
    app.dependency_overrides[dependencies.get_semantic_retriever] = lambda: FakeRetriever()

    response = client.post("/retrieve", json={"question": "pregunta", "top_k": 1})

    assert response.status_code == 200
    payload = response.json()
    assert payload["result_count"] == 1
    assert payload["results"][0]["chunk_id"] == "chunk_1"
    assert "text" not in payload["results"][0]


def test_retrieve_can_include_text_explicitly(client) -> None:
    app.dependency_overrides[dependencies.get_semantic_retriever] = lambda: FakeRetriever()

    response = client.post(
        "/retrieve",
        json={"question": "pregunta", "include_text": True},
    )

    assert response.status_code == 200
    assert response.json()["results"][0]["text"] == "Texto completo del chunk."


def test_query_with_mock_hides_chunks_by_default(client) -> None:
    app.dependency_overrides[dependencies.get_rag_pipeline] = lambda: FakeRagPipeline()

    response = client.post("/query", json={"question": "pregunta"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "Respuesta extractiva local."
    assert payload["retrieved_chunks"] == []
    assert payload["mode"] == "extractive"
    assert payload["llm_used"] is False


def test_query_can_show_chunks(client) -> None:
    app.dependency_overrides[dependencies.get_rag_pipeline] = lambda: FakeRagPipeline()

    response = client.post("/query", json={"question": "pregunta", "show_chunks": True})

    assert response.status_code == 200
    assert response.json()["retrieved_chunks"][0]["text"] == "Texto completo recuperado."


def test_query_accepts_llm_mode(client) -> None:
    app.dependency_overrides[dependencies.get_rag_pipeline] = lambda: FakeRagPipeline()

    response = client.post("/query", json={"question": "pregunta", "mode": "llm"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["mode"] == "llm"
    assert payload["llm_used"] is True
    assert payload["llm_provider"] == "gemini"


def test_query_exposes_domain_rejection_reason(client) -> None:
    app.dependency_overrides[dependencies.get_rag_pipeline] = (
        lambda: RejectedDomainRagPipeline()
    )

    response = client.post("/query", json={"question": "¿Qué documentos mencionan al Rey?"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["has_sufficient_context"] is False
    assert payload["rejection_reason"] == "out_of_domain"
    assert payload["sources"] == []
    assert payload["retrieved_chunks"] == []


def test_empty_question_returns_400(client) -> None:
    app.dependency_overrides[dependencies.get_semantic_retriever] = lambda: FakeRetriever()

    response = client.post("/retrieve", json={"question": "   "})

    assert response.status_code == 400


def test_empty_index_returns_409(client) -> None:
    app.dependency_overrides[dependencies.get_semantic_retriever] = lambda: EmptyIndexRetriever()

    response = client.post("/retrieve", json={"question": "pregunta"})

    assert response.status_code == 409
    assert "Vector index is empty" in response.json()["detail"]


def test_index_info(client, monkeypatch) -> None:
    def fake_index_info(settings):
        return IndexInfo(
            collection_name="documents",
            chroma_dir=str(settings.chroma_dir),
            embedding_model=settings.embedding_model,
            embedding_device=settings.embedding_device,
            chroma_available=False,
            vector_count=None,
        )

    monkeypatch.setattr(routes_module, "get_index_info", fake_index_info)
    response = client.get("/index-info")

    assert response.status_code == 200
    assert response.json()["collection"] == "documents"


def test_eval_summary(client, tmp_path) -> None:
    report_dir = tmp_path / "data" / "eval" / "reports"
    report_dir.mkdir(parents=True)
    (report_dir / "evaluation_report.json").write_text(
        """
        {
          "generated_at": "2026-06-02T00:00:00+00:00",
          "dataset_path": "questions.jsonl",
          "results": [],
          "summary": {
            "total_questions": 0,
            "passed": 0,
            "failed": 0,
            "pass_rate": 0,
            "insufficient_context_count": 0,
            "out_of_domain_correct_rejections": 0,
            "avg_keyword_hit_rate": 0,
            "avg_top_score": null
          }
        }
        """,
        encoding="utf-8",
    )

    response = client.get("/eval-summary")

    assert response.status_code == 200
    assert response.json()["total_questions"] == 0
