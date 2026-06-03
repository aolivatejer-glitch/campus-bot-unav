import json
from urllib.error import URLError

import pytest

from rag_chatbot.ui import api_client as api_client_module
from rag_chatbot.ui.api_client import ApiConnectionError, ApiTimeoutError, RagApiClient


class FakeResponse:
    def __init__(self, payload: dict):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def test_api_client_builds_query_request(monkeypatch) -> None:
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["method"] = request.method
        captured["body"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return FakeResponse(
            {
                "question": "pregunta",
                "answer": "respuesta",
                "has_sufficient_context": True,
                "sources": [],
                "retrieved_chunks": [],
            }
        )

    monkeypatch.setattr(api_client_module, "urlopen", fake_urlopen)
    client = RagApiClient("http://127.0.0.1:8000/", timeout=3)

    response = client.query(
        question="pregunta",
        mode="llm",
        top_k=5,
        min_score=0.3,
        show_chunks=False,
    )

    assert captured["url"] == "http://127.0.0.1:8000/query"
    assert captured["method"] == "POST"
    assert captured["body"] == {
        "question": "pregunta",
        "mode": "llm",
        "top_k": 5,
        "min_score": 0.3,
        "show_chunks": False,
    }
    assert captured["timeout"] == 3
    assert response["answer"] == "respuesta"


def test_api_client_uses_default_timeout_for_generativo_queries(monkeypatch) -> None:
    captured = {}

    def fake_urlopen(request, timeout):
        captured["timeout"] = timeout
        return FakeResponse(
            {
                "question": "pregunta",
                "answer": "respuesta",
                "has_sufficient_context": True,
                "sources": [],
                "retrieved_chunks": [],
            }
        )

    monkeypatch.setattr(api_client_module, "urlopen", fake_urlopen)
    client = RagApiClient("http://127.0.0.1:8000")

    client.query(
        question="pregunta",
        mode="llm",
        top_k=3,
        min_score=0.84,
        show_chunks=False,
    )

    assert captured["timeout"] == 120.0


def test_api_client_builds_extractive_query_request(monkeypatch) -> None:
    captured = {}

    def fake_urlopen(request, timeout):
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse(
            {
                "question": "pregunta",
                "answer": "respuesta",
                "has_sufficient_context": True,
                "sources": [],
                "retrieved_chunks": [],
            }
        )

    monkeypatch.setattr(api_client_module, "urlopen", fake_urlopen)
    client = RagApiClient("http://127.0.0.1:8000")

    client.query(
        question="pregunta",
        mode="extractive",
        top_k=3,
        min_score=0.84,
        show_chunks=False,
    )

    assert captured["body"]["mode"] == "extractive"


def test_api_client_handles_connection_errors(monkeypatch) -> None:
    def fake_urlopen(request, timeout):
        raise URLError("connection refused")

    monkeypatch.setattr(api_client_module, "urlopen", fake_urlopen)
    client = RagApiClient("http://127.0.0.1:8000")

    with pytest.raises(ApiConnectionError, match="No se pudo conectar"):
        client.health()


def test_api_client_handles_timeout_errors(monkeypatch) -> None:
    def fake_urlopen(request, timeout):
        raise TimeoutError("timed out")

    monkeypatch.setattr(api_client_module, "urlopen", fake_urlopen)
    client = RagApiClient("http://127.0.0.1:8000", timeout=1)

    with pytest.raises(ApiTimeoutError, match="API_REQUEST_TIMEOUT_SECONDS"):
        client.query(
            question="pregunta",
            mode="llm",
            top_k=3,
            min_score=0.84,
            show_chunks=False,
        )


def test_api_client_handles_insufficient_context_response(monkeypatch) -> None:
    def fake_urlopen(request, timeout):
        return FakeResponse(
            {
                "question": "fuera de dominio",
                "answer": "No encontre informacion suficiente.",
                "has_sufficient_context": False,
                "warning": "No encontre informacion suficiente.",
                "rejection_reason": "out_of_domain",
                "sources": [],
                "retrieved_chunks": [],
            }
        )

    monkeypatch.setattr(api_client_module, "urlopen", fake_urlopen)
    client = RagApiClient("http://127.0.0.1:8000")

    response = client.query(
        question="fuera de dominio",
        mode="extractive",
        top_k=5,
        min_score=0.3,
        show_chunks=False,
    )

    assert response["has_sufficient_context"] is False
    assert response["warning"] == "No encontre informacion suficiente."
    assert response["rejection_reason"] == "out_of_domain"
