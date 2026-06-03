import pytest
from pydantic import ValidationError

from rag_chatbot.api.models import BuildIndexRequest, QueryRequest, QueryResponse, RetrieveRequest


def test_retrieve_request_defaults() -> None:
    request = RetrieveRequest(question="pregunta")

    assert request.top_k is None
    assert request.include_text is False


def test_query_request_defaults() -> None:
    request = QueryRequest(question="pregunta")

    assert request.show_chunks is False


def test_query_response_accepts_rejection_reason() -> None:
    response = QueryResponse(
        question="pregunta",
        answer="rechazo",
        has_sufficient_context=False,
        rejection_reason="out_of_domain",
        sources=[],
        retrieved_chunks=[],
    )

    assert response.rejection_reason == "out_of_domain"


def test_build_index_request_validates_limit() -> None:
    with pytest.raises(ValidationError):
        BuildIndexRequest(limit=0)
