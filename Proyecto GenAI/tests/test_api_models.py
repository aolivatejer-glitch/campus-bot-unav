import pytest
from pydantic import ValidationError

from rag_chatbot.api.models import BuildIndexRequest, QueryRequest, RetrieveRequest


def test_retrieve_request_defaults() -> None:
    request = RetrieveRequest(question="pregunta")

    assert request.top_k is None
    assert request.include_text is False


def test_query_request_defaults() -> None:
    request = QueryRequest(question="pregunta")

    assert request.show_chunks is False


def test_build_index_request_validates_limit() -> None:
    with pytest.raises(ValidationError):
        BuildIndexRequest(limit=0)
