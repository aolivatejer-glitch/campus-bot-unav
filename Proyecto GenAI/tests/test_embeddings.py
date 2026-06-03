from rag_chatbot.embeddings.base import EmbeddingProvider
from rag_chatbot.embeddings.sentence_transformers_provider import (
    SentenceTransformersEmbeddingProvider,
)


class FakeEmbeddingProvider(EmbeddingProvider):
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
        return [[float(len(text)), 0.0, 1.0] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text)), 1.0, 0.0]


def test_fake_embedding_provider_has_consistent_dimension() -> None:
    provider = FakeEmbeddingProvider()

    embeddings = provider.embed_documents(["uno", "dos"])
    query_embedding = provider.embed_query("pregunta")

    assert provider.dimension == 3
    assert all(len(embedding) == 3 for embedding in embeddings)
    assert len(query_embedding) == 3


def test_e5_prefix_helpers() -> None:
    assert (
        SentenceTransformersEmbeddingProvider._passage_text("texto")
        == "passage: texto"
    )
    assert SentenceTransformersEmbeddingProvider._query_text("texto") == "query: texto"
