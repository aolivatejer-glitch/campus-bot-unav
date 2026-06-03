import pytest

from rag_chatbot.config import AppSettings
from rag_chatbot.embeddings.base import EmbeddingProvider
from rag_chatbot.retrieval.retriever import SemanticRetriever


class FakeEmbeddingProvider(EmbeddingProvider):
    def __init__(self) -> None:
        self.queries: list[str] = []

    @property
    def model_name(self) -> str:
        return "fake-model"

    @property
    def device(self) -> str:
        return "cpu"

    @property
    def dimension(self) -> int:
        return 3

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return [1.0, 0.0, 0.0]


class FakeQueryableStore:
    collection_name = "documents"

    def __init__(self, *, count: int = 3) -> None:
        self._count = count
        self.last_top_k = None
        self.last_document_id = None
        self.last_file_name = None

    def count(self) -> int:
        return self._count

    def query(
        self,
        *,
        query_embedding,
        top_k,
        document_id=None,
        file_name=None,
    ):
        self.last_top_k = top_k
        self.last_document_id = document_id
        self.last_file_name = file_name
        return {
            "ids": [["chunk_low", "chunk_high", "chunk_filtered"]],
            "documents": [["Texto bajo", "Texto alto", "Texto filtrado"]],
            "distances": [[0.7, 0.1, 0.95]],
            "metadatas": [
                [
                    _metadata("doc_1", "sample.pdf", 1, 0, 10),
                    _metadata("doc_1", "sample.pdf", 2, 1, 10),
                    _metadata("doc_2", "other.pdf", 3, 0, 13),
                ]
            ],
        }


def _metadata(document_id, file_name, page_number, chunk_index, char_count):
    return {
        "document_id": document_id,
        "file_name": file_name,
        "file_path": f"Documentos/{file_name}",
        "file_type": "pdf",
        "page_number": page_number,
        "chunk_index": chunk_index,
        "char_count": char_count,
        "source": f"Documentos/{file_name}",
    }


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
        top_k=5,
        min_retrieval_score=0.3,
        snippet_max_chars=20,
        _env_file=None,
    )


def test_retriever_applies_top_k_filters_and_preserves_metadata(tmp_path) -> None:
    provider = FakeEmbeddingProvider()
    store = FakeQueryableStore()
    retriever = SemanticRetriever(
        _settings(tmp_path),
        embedding_provider=provider,
        vector_store=store,
    )

    response = retriever.search(
        "pregunta",
        top_k=2,
        min_score=0.2,
        document_id="doc_1",
        file_name="sample.pdf",
    )

    assert provider.queries == ["pregunta"]
    assert store.last_top_k == 2
    assert store.last_document_id == "doc_1"
    assert store.last_file_name == "sample.pdf"
    assert [result.chunk_id for result in response.results] == ["chunk_high", "chunk_low"]
    assert response.results[0].score == 0.9
    assert response.results[0].document_id == "doc_1"
    assert response.results[0].file_name == "sample.pdf"
    assert response.results[0].page_number == 2
    assert response.results[0].source_label == "sample.pdf, pagina 2"


def test_retriever_applies_min_score(tmp_path) -> None:
    retriever = SemanticRetriever(
        _settings(tmp_path),
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=FakeQueryableStore(),
    )

    response = retriever.search("pregunta", min_score=0.8)

    assert [result.chunk_id for result in response.results] == ["chunk_high"]


def test_retriever_returns_empty_results_when_all_filtered(tmp_path) -> None:
    retriever = SemanticRetriever(
        _settings(tmp_path),
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=FakeQueryableStore(),
    )

    response = retriever.search("pregunta", min_score=0.99)

    assert response.results == []


def test_retriever_errors_when_index_is_empty(tmp_path) -> None:
    retriever = SemanticRetriever(
        _settings(tmp_path),
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=FakeQueryableStore(count=0),
    )

    with pytest.raises(ValueError, match="Vector index is empty"):
        retriever.search("pregunta")
